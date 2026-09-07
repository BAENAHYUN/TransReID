"""
forensic_video_search.py — Part 2: 자연어 텍스트로 비디오 인물 검색

IRRA 텍스트 인코더 + FAISS 인덱스를 이용하여
한국어/영어 텍스트 설명으로 인물 크롭을 순위화합니다.

사용법:
  # 기본 검색 (상위 10개)
  python forensic_video_search.py --query "빨간 재킷을 입은 남성"

  # 상위 20개, 트랙별 집계
  python forensic_video_search.py --query "red jacket man" --top-k 20

  # 인덱스 경로 직접 지정
  python forensic_video_search.py \\
      --query "검은 가방을 멘 여성" \\
      --index data/irra_index/irra.faiss \\
      --meta  data/irra_index/metadata.db

Python API:
  from forensic_video_search import ForensicSearcher
  searcher = ForensicSearcher()
  results  = searcher.search("파란 모자를 쓴 남자", top_k=10)
  for r in results:
      print(r["rank"], r["video"], r["track"], r["similarity"])
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
import faiss

# ── IRRA 소스 경로 추가 ───────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src" / "IRRA"))
sys.path.insert(0, str(_ROOT))

from model.build import build_model
from metadata_db import MetadataDB

# ── 기본 경로 ─────────────────────────────────────────────────────────
DEFAULT_INDEX  = "data/irra_index/irra.faiss"
DEFAULT_META   = "data/irra_index/metadata.db"
WEIGHT_PATH    = "weights/IRRA/irra_cuhk_pedes_download"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════
# IRRA 내장 BPE 토크나이저 (openai-clip 패키지 불필요)
# ═══════════════════════════════════════════════════════════════════════

# IRRA가 src/IRRA/utils/simple_tokenizer.py 와
# src/IRRA/data/bpe_simple_vocab_16e6.txt.gz 를 자체 포함
from utils.simple_tokenizer import SimpleTokenizer as _SimpleTokenizer

_bpe_tokenizer: _SimpleTokenizer | None = None


def _get_bpe_tokenizer() -> _SimpleTokenizer:
    global _bpe_tokenizer
    if _bpe_tokenizer is None:
        _bpe_tokenizer = _SimpleTokenizer()
    return _bpe_tokenizer


def clip_tokenize(texts: list[str], context_length: int = 77) -> torch.Tensor:
    """
    텍스트 리스트 → CLIP BPE 토큰 텐서 (B, context_length) dtype=long.
    IRRA 내장 SimpleTokenizer 사용 (외부 패키지 불필요).
    """
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
# IRRA 모델 로드 (build_video_index.py 와 동일)
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


def _load_checkpoint(weight_path: str):
    import zipfile, io
    if zipfile.is_zipfile(weight_path):
        print("[*] zip 아카이브 감지 — 내부 .pth 추출 중 ...")
        with zipfile.ZipFile(weight_path, "r") as z:
            pth_names = sorted(n for n in z.namelist() if n.endswith(".pth"))
            if not pth_names:
                raise RuntimeError(f"zip 안에 .pth 없음: {weight_path}")
            print(f"    추출: {pth_names[0]}")
            with z.open(pth_names[0]) as f:
                return torch.load(io.BytesIO(f.read()), map_location="cpu")
    return torch.load(weight_path, map_location="cpu")


def load_irra_model(weight_path: str):
    print(f"[*] IRRA 모델 로드: {weight_path}")
    args = _make_irra_args()
    model = build_model(args, num_classes=11003)

    ckpt = _load_checkpoint(weight_path)
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
# 텍스트 임베딩
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_text(model, query: str) -> np.ndarray:
    """텍스트 쿼리 → L2 정규화 임베딩 (1, 512)."""
    tokens = clip_tokenize([query]).to(DEVICE)
    # fp16 모델 대응
    if next(model.parameters()).dtype == torch.float16:
        pass  # encode_text 내부에서 float() 호출됨

    feat = model.encode_text(tokens)           # (1, 512) float32
    feat = feat.float().cpu().numpy()
    norm = np.linalg.norm(feat, axis=1, keepdims=True).clip(min=1e-8)
    return (feat / norm).astype("float32")


# ═══════════════════════════════════════════════════════════════════════
# ForensicSearcher 클래스
# ═══════════════════════════════════════════════════════════════════════

class ForensicSearcher:
    """
    텍스트 쿼리 → 인물 크롭 검색 인터페이스.

    사용 예:
        searcher = ForensicSearcher()
        results  = searcher.search("빨간 옷을 입은 남성", top_k=10)
    """

    def __init__(
        self,
        index_path:  str = DEFAULT_INDEX,
        meta_path:   str = DEFAULT_META,
        weight_path: str = WEIGHT_PATH,
    ):
        self.model = load_irra_model(weight_path)

        print(f"[*] FAISS 인덱스 로드: {index_path}")
        self.index = faiss.read_index(index_path)
        print(f"[+] 인덱스 준비 완료: {self.index.ntotal:,} 벡터")

        print(f"[*] 메타데이터 로드: {meta_path}")
        self.meta = MetadataDB(meta_path)
        print(f"[+] 메타데이터 준비: {len(self.meta):,} 레코드")

    def search(
        self,
        query:   str,
        top_k:   int  = 10,
        by_track: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Parameters
        ----------
        query    : 한국어 또는 영어 인물 설명 텍스트
        top_k    : 반환할 상위 결과 수
        by_track : True 이면 트랙별로 집계 후 최고 유사도 기준 정렬
                   False 이면 개별 크롭 단위로 반환

        Returns
        -------
        [
          {
            "rank":       1,
            "video":      "Normal_Videos_003_x264",
            "track":      "track_0001",
            "source":     "video" | "scvd",
            "similarity": 0.842,
            "best_frame": "frame_00001770",  # by_track=True 일 때 최고 점수 프레임
            "best_path":  "data/.../frame_00001770_person_01.jpg",
            "n_frames":   2,                 # 해당 트랙의 매칭 프레임 수
          }, ...
        ]
        """
        # 텍스트 → 임베딩
        q_emb = encode_text(self.model, query)   # (1, 512)

        # FAISS 검색 (많이 뽑아서 트랙 집계에 쓸 여유 확보)
        search_k = top_k * 20 if by_track else top_k
        search_k = min(search_k, self.index.ntotal)

        similarities, indices = self.index.search(q_emb, search_k)
        sims = similarities[0]
        idxs = indices[0]

        if not by_track:
            # 크롭 단위 결과
            results = []
            for rank, (idx, sim) in enumerate(zip(idxs, sims), start=1):
                if idx < 0:
                    continue
                meta = self.meta[int(idx)]
                results.append({
                    "rank":       rank,
                    "video":      meta.get("video", ""),
                    "track":      meta.get("track", ""),
                    "source":     meta.get("source", ""),
                    "frame":      meta.get("frame", ""),
                    "person":     meta.get("person", ""),
                    "path":       meta.get("path", ""),
                    "similarity": round(float(sim), 4),
                })
                if len(results) >= top_k:
                    break
            return results

        # 트랙별 집계
        track_best: dict[str, dict] = {}
        for idx, sim in zip(idxs, sims):
            if idx < 0:
                continue
            meta   = self.meta[int(idx)]
            video  = meta.get("video", "")
            track  = meta.get("track", "")
            key    = f"{video}/{track}"
            score  = float(sim)

            if key not in track_best or score > track_best[key]["similarity"]:
                track_best[key] = {
                    "video":      video,
                    "track":      track,
                    "source":     meta.get("source", ""),
                    "similarity": round(score, 4),
                    "best_frame": meta.get("frame", ""),
                    "best_path":  meta.get("path", ""),
                    "n_frames":   1,
                }
            else:
                track_best[key]["n_frames"] += 1

        # 유사도 내림차순 정렬, top-k
        ranked = sorted(track_best.values(), key=lambda x: x["similarity"], reverse=True)
        return [
            {"rank": i + 1, **item}
            for i, item in enumerate(ranked[:top_k])
        ]

    def print_results(self, results: list[dict], query: str) -> None:
        """검색 결과를 콘솔에 출력."""
        print(f"\n{'═'*60}")
        print(f"  쿼리: {query}")
        print(f"  결과: {len(results)}건")
        print(f"{'═'*60}")
        for r in results:
            frames_info = f"  ({r.get('n_frames', 1)}프레임)" if "n_frames" in r else ""
            print(
                f"  {r['rank']:>2}위  [{r['similarity']:.4f}]  "
                f"{r['source']:5}  {r['video']}  {r['track']}{frames_info}"
            )
            best = r.get("best_path") or r.get("path", "")
            if best:
                print(f"       → {best}")
        print()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="IRRA 텍스트 임베딩으로 비디오 인물 검색"
    )
    ap.add_argument(
        "--query", "-q", required=True,
        help='검색 쿼리 (예: "빨간 재킷을 입은 남성")',
    )
    ap.add_argument(
        "--top-k", "-k", type=int, default=10,
        help="반환할 상위 결과 수 (기본: 10)",
    )
    ap.add_argument(
        "--index", default=DEFAULT_INDEX,
        help=f"FAISS 인덱스 경로 (기본: {DEFAULT_INDEX})",
    )
    ap.add_argument(
        "--meta", default=DEFAULT_META,
        help=f"SQLite 메타데이터 경로 (기본: {DEFAULT_META})",
    )
    ap.add_argument(
        "--weight", default=WEIGHT_PATH,
        help="IRRA 가중치 파일 경로",
    )
    ap.add_argument(
        "--no-aggregate", action="store_true",
        help="트랙 집계 없이 크롭 단위로 결과 반환",
    )
    ap.add_argument(
        "--json-out", action="store_true",
        help="결과를 JSON 형식으로 출력",
    )

    args = ap.parse_args()

    searcher = ForensicSearcher(
        index_path  = args.index,
        meta_path   = args.meta,
        weight_path = args.weight,
    )

    results = searcher.search(
        query    = args.query,
        top_k    = args.top_k,
        by_track = not args.no_aggregate,
    )

    if args.json_out:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        searcher.print_results(results, args.query)
