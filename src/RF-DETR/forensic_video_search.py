"""
forensic_video_search.py — Part 2: 자연어·이미지로 비디오 인물 검색

IRRA 인코더 + Milvus 벡터 인덱스를 이용하여
  - 텍스트(영어 자연어)
  - 이미지 파일
로 인물 크롭을 순위화합니다.

PostgreSQL : 메타데이터 (video, track, frame, path)
Milvus     : IRRA 벡터 인덱스 (512-dim, IP, IVF_FLAT nprobe=64)

Python API:
  from forensic_video_search import ForensicSearcher, encode_image_irra
  searcher = ForensicSearcher()
  results  = searcher.search("red jacket man", top_k=20)
  results  = searcher.search_by_image("query.jpg", top_k=20)

정확도 개선 (FAISS+SQLite 대비):
  - Milvus IVF_FLAT nprobe=64: 대규모 색인에서 안정적 recall
  - search_by_track(): 서버사이드 트랙 집계 → 더 많은 후보(top_k×20) 검색
  - search_with_negative(): neg_idx↔neg_sim 매핑 버그 수정
"""

from __future__ import annotations

import sys
import json
import argparse
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image as _PIL_Image

# ── 경로 설정 ─────────────────────────────────────────────────────────
_ROOT   = Path(__file__).resolve().parents[2]   # TransReID 루트
_RFDETR = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src" / "IRRA"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_RFDETR))               # 최우선 (루트 동명 파일 차단)

from model.build import build_model
from db_manager     import PostgresMetadataDB
from milvus_manager import MilvusManager

WEIGHT_PATH = "weights/IRRA/irra_cuhk_pedes_download"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════
# BPE Tokenizer
# ═══════════════════════════════════════════════════════════════════════

from utils.simple_tokenizer import SimpleTokenizer as _SimpleTokenizer

_bpe_tokenizer: _SimpleTokenizer | None = None


def _get_bpe_tokenizer() -> _SimpleTokenizer:
    global _bpe_tokenizer
    if _bpe_tokenizer is None:
        _bpe_tokenizer = _SimpleTokenizer()
    return _bpe_tokenizer


def clip_tokenize(texts: list[str], context_length: int = 77) -> torch.Tensor:
    tok = _get_bpe_tokenizer()
    sot = tok.encoder["<|startoftext|>"]
    eot = tok.encoder["<|endoftext|>"]
    result = torch.zeros(len(texts), context_length, dtype=torch.long)
    for i, text in enumerate(texts):
        tokens = [sot] + tok.encode(text) + [eot]
        if len(tokens) > context_length:
            tokens = tokens[:context_length]
            tokens[-1] = eot
        result[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return result


# ═══════════════════════════════════════════════════════════════════════
# 이미지 전처리
# ═══════════════════════════════════════════════════════════════════════

import torchvision.transforms as T

_IMG_TRANSFORM = T.Compose([
    T.Resize((384, 128)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std =[0.26862954, 0.26130258, 0.27577711],
    ),
])


# ═══════════════════════════════════════════════════════════════════════
# IRRA 모델 로드
# ═══════════════════════════════════════════════════════════════════════

def _make_irra_args() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        pretrain_choice             = "ViT-B/16",
        img_size                    = (384, 128),
        stride_size                 = 16,
        temperature                 = 0.02,
        loss_names                  = "sdm+id+mlm",
        cmt_depth                   = 4,
        masked_token_rate           = 0.8,
        masked_token_unchanged_rate = 0.1,
        lr_factor                   = 5.0,
        MLM                         = True,
        mlm_loss_weight             = 1.0,
        id_loss_weight              = 1.0,
        vocab_size                  = 49408,
        text_length                 = 77,
    )


def load_irra_model(weight_path: str):
    print(f"[*] IRRA 모델 로드: {weight_path}")
    args  = _make_irra_args()
    model = build_model(args, num_classes=11003)

    import zipfile, io
    if zipfile.is_zipfile(weight_path):
        print("[*] zip 아카이브 감지 — 내부 .pth 추출 중 ...")
        with zipfile.ZipFile(weight_path, "r") as z:
            pth_names = sorted(n for n in z.namelist() if n.endswith(".pth"))
            if not pth_names:
                raise RuntimeError(f"zip 안에 .pth 없음: {weight_path}")
            print(f"    추출: {pth_names[0]}")
            with z.open(pth_names[0]) as f:
                ckpt = torch.load(io.BytesIO(f.read()), map_location="cpu")
    else:
        ckpt = torch.load(weight_path, map_location="cpu")

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    else:
        state_dict = ckpt

    clean_sd = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd, strict=False)
    model.eval()
    model = model.to(DEVICE)
    print(f"[+] 모델 준비 완료 | device={DEVICE}")
    return model


# ═══════════════════════════════════════════════════════════════════════
# 인코딩 함수 (공용)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_text(model, query: str) -> np.ndarray:
    """텍스트 쿼리 → L2 정규화 임베딩 (1, 512)."""
    tokens = clip_tokenize([query]).to(DEVICE)
    feat   = model.encode_text(tokens)
    feat   = feat.float().cpu().numpy()
    norm   = np.linalg.norm(feat, axis=1, keepdims=True).clip(min=1e-8)
    return (feat / norm).astype("float32")


@torch.no_grad()
def encode_image_irra(model, img_input) -> np.ndarray:
    """이미지 파일 경로 또는 PIL Image → L2 정규화 임베딩 (1, 512)."""
    if isinstance(img_input, (str, Path)):
        pil = _PIL_Image.open(str(img_input)).convert("RGB")
    else:
        pil = img_input.convert("RGB")
    pixel = _IMG_TRANSFORM(pil).unsqueeze(0).to(DEVICE)
    feat  = model.encode_image(pixel)
    feat  = feat.float().cpu().numpy()
    norm  = np.linalg.norm(feat, axis=1, keepdims=True).clip(min=1e-8)
    return (feat / norm).astype("float32")


def avg_normalize(embs: list[np.ndarray]) -> np.ndarray:
    """임베딩 리스트를 평균내고 L2 정규화."""
    stacked = np.vstack(embs)
    avg     = stacked.mean(axis=0, keepdims=True)
    norm    = np.linalg.norm(avg, axis=1, keepdims=True).clip(min=1e-8)
    return (avg / norm).astype("float32")


# ═══════════════════════════════════════════════════════════════════════
# ForensicSearcher
# ═══════════════════════════════════════════════════════════════════════

class ForensicSearcher:
    """
    IRRA + Milvus + PostgreSQL 기반 인물 검색기.

    FAISS + SQLite 대비 개선점:
      - Milvus IVF_FLAT nprobe=64: 대규모 벡터에서 안정적 recall
      - search_by_track(): 서버사이드 트랙 집계 → 더 많은 후보(top_k×20) 검색
      - search_with_negative(): neg_idx 매핑 버그 수정으로 정확도 향상
    """

    def __init__(self, weight_path: str = WEIGHT_PATH):
        self.model  = load_irra_model(weight_path)
        self.milvus = MilvusManager()
        self.pg     = PostgresMetadataDB()

    # ── 공통 내부 검색 ─────────────────────────────────────────────────
    def _search_by_emb(
        self,
        q_emb:          np.ndarray,
        top_k:          int   = 10,
        by_track:       bool  = True,
        irra_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        사전 계산된 임베딩(1, 512)으로 Milvus 검색.

        by_track=True  → 서버사이드 트랙 집계 (top_k*20 후보 → 트랙별 최고 점수)
        by_track=False → 프레임 단위 결과 반환
        """
        if by_track:
            # ── 서버사이드 트랙 집계 (정확도 ↑) ─────────────────────
            track_hits = self.milvus.search_by_track(
                q_emb,
                top_k          = top_k,
                irra_threshold = irra_threshold,
            )
            results = []
            for item in track_hits:
                try:
                    frame = self.pg[item["milvus_id"]]
                except KeyError:
                    frame = {}
                results.append({
                    "rank":       item["rank"],
                    "video":      item["video"],
                    "track":      item["track"],
                    "source":     item["source"],
                    "similarity": item["similarity"],
                    "best_frame": frame.get("frame", ""),
                    "best_path":  frame.get("path", ""),
                    "n_frames":   1,
                })
            return results

        else:
            # ── 프레임 단위 검색 ──────────────────────────────────────
            search_k  = min(top_k, self.milvus.ntotal)
            sims, ids = self.milvus.search(q_emb, search_k)
            results   = []
            for rank, (idx, sim) in enumerate(zip(ids[0], sims[0]), start=1):
                if idx < 0:
                    continue
                try:
                    frame = self.pg[int(idx)]
                except KeyError:
                    continue
                results.append({
                    "rank":       rank,
                    "video":      frame.get("video", ""),
                    "track":      frame.get("track", ""),
                    "source":     frame.get("source", ""),
                    "frame":      frame.get("frame", ""),
                    "person":     frame.get("person", ""),
                    "path":       frame.get("path", ""),
                    "similarity": round(float(sim), 4),
                })
                if len(results) >= top_k:
                    break
            return results

    # ── 텍스트 검색 ────────────────────────────────────────────────────
    def search(
        self,
        query:    str,
        top_k:    int  = 10,
        by_track: bool = True,
    ) -> list[dict[str, Any]]:
        q_emb = encode_text(self.model, query)
        return self._search_by_emb(q_emb, top_k, by_track)

    # ── 이미지 검색 ────────────────────────────────────────────────────
    def search_by_image(
        self,
        img_path: str,
        top_k:    int  = 10,
        by_track: bool = True,
    ) -> list[dict[str, Any]]:
        """이미지 파일로 유사 인물 검색."""
        q_emb = encode_image_irra(self.model, img_path)
        return self._search_by_emb(q_emb, top_k, by_track)

    # ── 멀티 쿼리 텍스트 검색 ─────────────────────────────────────────
    def search_multi(
        self,
        queries:  list[str],
        top_k:    int  = 10,
        by_track: bool = True,
    ) -> list[dict]:
        """여러 텍스트 쿼리 임베딩을 평균내어 검색."""
        embs  = [encode_text(self.model, q) for q in queries]
        q_emb = avg_normalize(embs)
        return self._search_by_emb(q_emb, top_k, by_track)

    # ── Negative 쿼리 차감 ─────────────────────────────────────────────
    def search_with_negative(
        self,
        positive: str,
        negative: str,
        alpha:    float = 0.5,
        top_k:    int   = 10,
        by_track: bool  = True,
    ) -> list[dict]:
        """
        Positive 점수에서 alpha * Negative 점수를 차감하여 순위화.

        수정 (v2): neg_idx → neg_sim 매핑 오류 수정.
        이전 버전은 pos_idx[i] ↔ neg_sim[i] 로 잘못 매핑되어
        negative 제거 효과가 부정확했음. 이제 neg_idx[i] → neg_sim[i] 로 정확히 매핑.
        """
        pos_emb = encode_text(self.model, positive)
        neg_emb = encode_text(self.model, negative)

        search_k = min(top_k * 30, self.milvus.ntotal)

        pos_sims, pos_ids = self.milvus.search(pos_emb, search_k)
        neg_sims, neg_ids = self.milvus.search(neg_emb, search_k)

        # 정확한 neg 매핑: neg 쿼리의 결과 idx → neg 유사도
        idx_to_neg = {
            int(idx): float(ns)
            for idx, ns in zip(neg_ids[0], neg_sims[0])
            if idx >= 0
        }

        scored = []
        for idx, ps in zip(pos_ids[0], pos_sims[0]):
            if idx < 0:
                continue
            ns    = idx_to_neg.get(int(idx), 0.0)
            final = float(ps) - alpha * ns
            scored.append((int(idx), float(ps), ns, final))

        scored.sort(key=lambda x: x[3], reverse=True)

        if not by_track:
            results = []
            for rank, (idx, ps, ns, final) in enumerate(scored, start=1):
                try:
                    frame = self.pg[idx]
                except KeyError:
                    continue
                results.append({
                    "rank":       rank,
                    "video":      frame.get("video", ""),
                    "track":      frame.get("track", ""),
                    "source":     frame.get("source", ""),
                    "frame":      frame.get("frame", ""),
                    "path":       frame.get("path", ""),
                    "similarity": round(final, 4),
                    "pos_score":  round(ps, 4),
                    "neg_score":  round(ns, 4),
                })
                if len(results) >= top_k:
                    break
            return results

        track_best: dict[str, dict] = {}
        for idx, ps, ns, final in scored:
            try:
                frame = self.pg[idx]
            except KeyError:
                continue
            video = frame.get("video", "")
            track = frame.get("track", "")
            key   = f"{video}/{track}"
            if key not in track_best or final > track_best[key]["similarity"]:
                track_best[key] = {
                    "video":      video,
                    "track":      track,
                    "source":     frame.get("source", ""),
                    "similarity": round(final, 4),
                    "pos_score":  round(ps, 4),
                    "neg_score":  round(ns, 4),
                    "best_frame": frame.get("frame", ""),
                    "best_path":  frame.get("path", ""),
                    "n_frames":   1,
                }
            else:
                track_best[key]["n_frames"] += 1

        ranked = sorted(track_best.values(), key=lambda x: x["similarity"], reverse=True)
        return [{"rank": i + 1, **item} for i, item in enumerate(ranked[:top_k])]

    # ── 출력 ─────────────────────────────────────────────────────────
    def print_results(self, results: list[dict], query: str) -> None:
        print(f"\n{'═'*60}")
        print(f"  쿼리: {query}")
        print(f"  결과: {len(results)}건")
        print(f"{'═'*60}")
        for r in results:
            frames_info = f"  ({r.get('n_frames', 1)}프레임)" if "n_frames" in r else ""
            print(
                f"  {r['rank']:>2}위  [{r['similarity']:.4f}]  "
                f"{r.get('source', ''):5}  {r['video']}  {r['track']}{frames_info}"
            )
            best = r.get("best_path") or r.get("path", "")
            if best:
                print(f"       → {best}")
        print()

    # ── 정리 ─────────────────────────────────────────────────────────
    def close(self):
        self.milvus.close()
        self.pg.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="IRRA + Milvus 인물 검색")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--query",  "-q", help="텍스트 쿼리")
    grp.add_argument("--image",  "-i", help="이미지 파일 경로")
    ap.add_argument("--top-k",        "-k", type=int, default=10)
    ap.add_argument("--weight",             default=WEIGHT_PATH)
    ap.add_argument("--no-aggregate",       action="store_true")
    ap.add_argument("--json-out",           action="store_true")

    args     = ap.parse_args()
    searcher = ForensicSearcher(weight_path=args.weight)
    by_track = not args.no_aggregate

    if args.query:
        results = searcher.search(args.query, top_k=args.top_k, by_track=by_track)
        label   = args.query
    else:
        results = searcher.search_by_image(args.image, top_k=args.top_k, by_track=by_track)
        label   = args.image

    if args.json_out:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        searcher.print_results(results, label)
