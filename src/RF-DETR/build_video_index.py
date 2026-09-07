"""
build_video_index.py — Part 1: IRRA 모델로 비디오 인물 트랙 임베딩 생성

소스 (이미 추출된 인물 크롭):
  - data/video_person_tracks_v4/   (data/videos/ 에서 추출)
  - data/scvd_person_tracks_v1/    (data/SCVD/   에서 추출)

출력:
  - data/irra_index/irra.faiss        FAISS IndexFlatIP (dim=512)
  - data/irra_index/metadata.db       SQLite 메타데이터
  - data/irra_index/metadata.json     JSON 백업
  - data/irra_index/build_state.json  체크포인트 (중단 후 재개 가능)

실행:
  python build_video_index.py
  python build_video_index.py --batch-size 256   # GPU 메모리에 맞춰 조절
  python build_video_index.py --no-resume        # 처음부터 다시 시작
"""

from __future__ import annotations

import sys
import json
import time
import argparse
import types
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import faiss

# ── IRRA 소스 경로 추가 ───────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src" / "IRRA"))
sys.path.insert(0, str(_ROOT))   # metadata_db.py 가 루트에 있음

from model.build import build_model
from metadata_db import build_from_json

# ── 기본 설정 ─────────────────────────────────────────────────────────
TRACK_DIRS = [
    "data/video_person_tracks_v4",
    "data/scvd_person_tracks_v1",
]
OUTPUT_DIR  = "data/irra_index"
WEIGHT_PATH = "weights/IRRA/irra_cuhk_pedes_download"
BATCH_SIZE  = 128
EMBED_DIM   = 512   # ViT-B/16 → 512차원
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ── IRRA inference 이미지 변환 (datasets/build.py, is_train=False) ────
_TRANSFORM = T.Compose([
    T.Resize((384, 128)),   # height=384, width=128 (인물 크롭 비율)
    T.ToTensor(),
    T.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std =[0.26862954, 0.26130258, 0.27577711],
    ),
])


# ═══════════════════════════════════════════════════════════════════════
# 모델 로드
# ═══════════════════════════════════════════════════════════════════════

def _make_irra_args() -> types.SimpleNamespace:
    """options.py 기본값으로 args 네임스페이스 생성 (학습 불필요, 추론 전용)."""
    return types.SimpleNamespace(
        pretrain_choice             = "ViT-B/16",
        img_size                    = (384, 128),
        stride_size                 = 16,
        temperature                 = 0.02,
        loss_names                  = "sdm+id+mlm",  # 구조 생성에만 필요
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
    """IRRA 체크포인트 로드 후 eval 모드 반환 → (model, dtype)."""
    print(f"[*] IRRA 모델 로드: {weight_path}")
    args = _make_irra_args()

    # CUHK-PEDES 기준 (분류기 구조용, 실제 검색엔 encoder 만 사용)
    model = build_model(args, num_classes=11003)

    ckpt = torch.load(weight_path, map_location="cpu")

    # 체크포인트 형식 자동 감지
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    else:
        state_dict = ckpt

    # DDP 저장 체크포인트의 "module." 접두어 제거
    clean_sd = {k.removeprefix("module."): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    if missing:
        print(f"[!] 누락된 키 {len(missing)}개 (정상 — 분류기 등 검색 무관 레이어)")

    model.eval()
    model = model.to(DEVICE)

    dtype = next(model.parameters()).dtype
    print(f"[+] 모델 준비 완료 | device={DEVICE} | dtype={dtype}")
    return model, dtype


# ═══════════════════════════════════════════════════════════════════════
# 이미지 배치 임베딩
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def embed_images(model, dtype, image_paths: list[str]) -> np.ndarray:
    """경로 리스트 → L2 정규화 임베딩 (B, 512) float32."""
    tensors = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(_TRANSFORM(img))
        except Exception as exc:
            print(f"\n[!] 이미지 오류 (zero 벡터로 대체): {p}  —  {exc}")
            tensors.append(torch.zeros(3, 384, 128))

    batch = torch.stack(tensors).to(DEVICE)
    if dtype == torch.float16:
        batch = batch.half()

    feats = model.encode_image(batch).float().cpu().numpy()   # (B, 512)

    # L2 정규화 → 내적 == 코사인 유사도
    norms = np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-8)
    return (feats / norms).astype("float32")


# ═══════════════════════════════════════════════════════════════════════
# 크롭 파일 목록 수집
# ═══════════════════════════════════════════════════════════════════════

def collect_crops(track_dirs: list[str]) -> list[dict]:
    """
    트랙 디렉토리를 재귀 순회하여 모든 .jpg 크롭 메타데이터 수집.

    디렉토리 구조:
      video_person_tracks_v4 / {video} / {track} / frame_XXXX_person_NN.jpg
      scvd_person_tracks_v1  / {video} / {track} / frame_XXXX_person_NN.jpg

    반환:
      [{"path", "source", "video", "track", "frame", "person"}, ...]
    """
    records: list[dict] = []

    for td in track_dirs:
        td_path = Path(td)
        if not td_path.exists():
            print(f"[!] 경로 없음 (건너뜀): {td}")
            continue

        source = "scvd" if "scvd" in td.lower() else "video"

        for video_dir in sorted(td_path.iterdir()):
            if not video_dir.is_dir():
                continue
            for track_dir in sorted(video_dir.iterdir()):
                if not track_dir.is_dir():
                    continue
                for jpg in sorted(track_dir.glob("*.jpg")):
                    stem   = jpg.stem          # "frame_00001770_person_01"
                    parts  = stem.rsplit("_", 2)
                    if len(parts) == 3:
                        frame_id  = "_".join(parts[:2])   # "frame_00001770"
                        person_id = parts[2]               # "01"
                    else:
                        frame_id  = stem
                        person_id = "00"

                    records.append({
                        "path":   str(jpg),
                        "source": source,
                        "video":  video_dir.name,
                        "track":  track_dir.name,
                        "frame":  frame_id,
                        "person": person_id,
                    })

    print(f"[+] 총 {len(records):,} 개 크롭 발견")
    return records


# ═══════════════════════════════════════════════════════════════════════
# 인덱스 빌드
# ═══════════════════════════════════════════════════════════════════════

def build_index(
    track_dirs:  list[str] = TRACK_DIRS,
    output_dir:  str       = OUTPUT_DIR,
    weight_path: str       = WEIGHT_PATH,
    batch_size:  int       = BATCH_SIZE,
    resume:      bool      = True,
    save_every:  int       = 5000,
) -> faiss.Index | None:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    faiss_path = out / "irra.faiss"
    meta_json  = out / "metadata.json"
    meta_db    = out / "metadata.db"
    state_path = out / "build_state.json"

    # ── 크롭 목록 수집 ──────────────────────────────────────────────────
    all_crops = collect_crops(track_dirs)
    total = len(all_crops)
    if total == 0:
        print("[!] 처리할 크롭이 없습니다.")
        return None

    # ── 체크포인트 확인 ─────────────────────────────────────────────────
    start_idx = 0
    all_meta: list[dict] = []
    index = faiss.IndexFlatIP(EMBED_DIM)

    if resume and state_path.exists() and faiss_path.exists() and meta_json.exists():
        try:
            state = json.loads(state_path.read_text())
            start_idx = int(state.get("processed", 0))
            if 0 < start_idx < total:
                index    = faiss.read_index(str(faiss_path))
                all_meta = json.loads(meta_json.read_text(encoding="utf-8"))
                print(f"[*] 체크포인트 재개: {start_idx:,}/{total:,} 완료")
            elif start_idx >= total:
                print("[+] 이미 모든 크롭 처리 완료.")
                if not meta_db.exists():
                    build_from_json(meta_json, meta_db)
                return faiss.read_index(str(faiss_path))
        except Exception as exc:
            print(f"[!] 체크포인트 오류, 처음부터 시작: {exc}")
            start_idx = 0
            all_meta  = []
            index     = faiss.IndexFlatIP(EMBED_DIM)

    # ── 모델 로드 ───────────────────────────────────────────────────────
    model, dtype = load_irra_model(weight_path)

    # ── 임베딩 루프 ─────────────────────────────────────────────────────
    remaining  = all_crops[start_idx:]
    t0         = time.time()
    saved_at   = start_idx

    print(f"[*] 임베딩 시작: {len(remaining):,} 크롭 남음  (배치={batch_size})")

    for chunk_start in range(0, len(remaining), batch_size):
        chunk = remaining[chunk_start : chunk_start + batch_size]
        embs  = embed_images(model, dtype, [c["path"] for c in chunk])
        index.add(embs)
        all_meta.extend(chunk)

        processed = start_idx + chunk_start + len(chunk)
        elapsed   = time.time() - t0
        speed     = (processed - start_idx) / max(elapsed, 1)
        eta_min   = (total - processed) / max(speed, 1) / 60

        print(
            f"  [{processed:>8,}/{total:,}]  {speed:5.0f} crops/s  ETA {eta_min:.1f}분  ",
            end="\r", flush=True,
        )

        # 주기적 저장
        if (processed - saved_at) >= save_every or processed == total:
            faiss.write_index(index, str(faiss_path))
            meta_json.write_text(
                json.dumps(all_meta, ensure_ascii=False), encoding="utf-8"
            )
            state_path.write_text(
                json.dumps({"processed": processed, "total": total})
            )
            saved_at = processed

    print()

    # ── SQLite 메타데이터 변환 ──────────────────────────────────────────
    print("[*] SQLite 메타데이터 변환 중 ...")
    build_from_json(meta_json, meta_db)

    elapsed_total = time.time() - t0
    print(f"\n[+] 인덱스 빌드 완료!")
    print(f"    FAISS   : {faiss_path}  ({index.ntotal:,} 벡터, {EMBED_DIM}차원)")
    print(f"    메타    : {meta_db}")
    print(f"    소요시간: {elapsed_total / 60:.1f}분")

    return index


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="IRRA 모델로 비디오 인물 트랙 임베딩 인덱스 생성"
    )
    ap.add_argument(
        "--track-dirs", nargs="+", default=TRACK_DIRS,
        help="인물 트랙 크롭 디렉토리 목록",
    )
    ap.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help="FAISS 인덱스 / 메타데이터 저장 디렉토리",
    )
    ap.add_argument(
        "--weight", default=WEIGHT_PATH,
        help="IRRA 가중치 파일 경로",
    )
    ap.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"배치 크기 (기본 {BATCH_SIZE}, GPU VRAM에 맞춰 조절)",
    )
    ap.add_argument(
        "--no-resume", action="store_true",
        help="체크포인트 무시하고 처음부터 시작",
    )
    ap.add_argument(
        "--save-every", type=int, default=5000,
        help="몇 개 처리마다 중간 저장 (기본 5000)",
    )

    a = ap.parse_args()
    build_index(
        track_dirs  = a.track_dirs,
        output_dir  = a.output_dir,
        weight_path = a.weight,
        batch_size  = a.batch_size,
        resume      = not a.no_resume,
        save_every  = a.save_every,
    )
