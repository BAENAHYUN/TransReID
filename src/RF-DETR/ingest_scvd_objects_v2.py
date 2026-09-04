"""
ingest_scvd_objects_v2.py — SCVD 객체 임베딩 (SigLIP2 + DINOv2) → Qdrant

RF-DETR로 검출된 객체 크롭을 SigLIP2(768) + DINOv2(768)으로 임베딩하여
forensic_object_local 컬렉션에 삽입합니다.

객체 크롭 경로 (RF-DETR 트래커 출력):
  data/scvd_object_tracks_v1/<video_stem>/<label>_<track_id>/frame_NNNN_<conf>.jpg

사용법:
  python ingest_scvd_objects_v2.py
  python ingest_scvd_objects_v2.py --batch-size 32 --max-crops 500
  python ingest_scvd_objects_v2.py --force          # 기존 재삽입
  python ingest_scvd_objects_v2.py --validate-only  # 통계만

환경변수:
  FORENSIC_QDRANT_PATH  기본: <ROOT>/data/qdrant_local
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# ── 경로 ──────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
ROOT_DIR = _HERE.parents[1]

sys.path.insert(0, str(_HERE))
from qdrant_object_manager import ObjectQdrantManager

# 객체 크롭 루트 (person_video_track_search_v4 에서 생성)
OBJECT_TRACK_ROOT = ROOT_DIR / "data" / "scvd_object_tracks_v1"
SCVD_ROOT         = ROOT_DIR / "data" / "scvd" / "SCVD_converted"

SIGLIP2_MODEL = "google/siglip2-base-patch16-224"

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# DINOv2 모델 크기: vits14 / vitb14 / vitl14 / vitg14
DINO_MODEL = "facebook/dinov2-base"   # vitb14 768-dim, 로컬 허브 캐시 사용


# ── ID 생성 ────────────────────────────────────────────────────────────
def crop_to_id(p: Path) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "object|scvd|" + str(p.resolve())))


# ── 유틸 ──────────────────────────────────────────────────────────────
def l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1
    return x / n


def all_scvd_video_stems() -> set[str]:
    return {p.stem for p in SCVD_ROOT.rglob("*") if p.is_file()}


def track_video_stem(p: Path) -> str:
    """객체 크롭 경로에서 video stem 추출.
    예: .../scvd_object_tracks_v1/<video_stem>/<label>_<track>/frame_001.jpg
    → parents[1].name
    """
    return p.parents[1].name


def parse_track_dir(track_dir: str) -> tuple[str, str]:
    """
    'knife_0003' → ('knife', '0003')
    'person_0001' → ('person', '0001')   # person 크롭도 섞여 있을 수 있음
    """
    parts = track_dir.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return track_dir, "0000"


def list_crops(stems: set[str]) -> list[Path]:
    return sorted(
        p for p in OBJECT_TRACK_ROOT.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTS
        and track_video_stem(p) in stems
    )


def build_video_info_index() -> dict[str, dict]:
    idx: dict[str, dict] = {}
    for p in SCVD_ROOT.rglob("*"):
        if not p.is_file():
            continue
        try:
            parts = p.relative_to(SCVD_ROOT).parts
            idx[p.stem] = {
                "split":    parts[0] if len(parts) > 0 else "",
                "category": parts[1] if len(parts) > 1 else "",
            }
        except Exception:
            pass
    return idx


def crop_meta(p: Path, info_idx: dict[str, dict]) -> dict:
    video_stem = track_video_stem(p)
    track_dir  = p.parent.name
    label, track_id = parse_track_dir(track_dir)
    info  = info_idx.get(video_stem, {})
    parts = p.stem.split("_")
    frame_str = parts[1] if len(parts) > 1 and parts[0] == "frame" else p.stem
    return {
        "video":      video_stem,
        "track":      track_dir,
        "label":      label,
        "source":     "SCVD",
        "split":      info.get("split",    ""),
        "category":   info.get("category", ""),
        "frame":      frame_str,
        "confidence": _parse_confidence(p.stem),
        "path":       str(p),
    }


def _parse_confidence(stem: str) -> float:
    """frame_0045_0.92 → 0.92"""
    parts = stem.rsplit("_", 1)
    try:
        return float(parts[-1])
    except ValueError:
        return 0.0


# ── SigLIP2 인코더 ────────────────────────────────────────────────────
class SigLIP2Encoder:
    def __init__(self, model_name: str = SIGLIP2_MODEL):
        from transformers import AutoProcessor, AutoModel
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] SigLIP2 로드 중: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(self.dev).eval()
        print(f"[+] SigLIP2 완료 ({self.dev})")

    @torch.inference_mode()
    def encode(self, paths: list[Path]) -> np.ndarray:
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.dev) for k, v in inputs.items()}
        out  = self.model.get_image_features(**inputs)
        feat = (out.pooler_output if hasattr(out, "pooler_output")
                else out.last_hidden_state[:, 0] if hasattr(out, "last_hidden_state")
                else out)
        return l2(feat.float().cpu().numpy())   # (N, 768)


# ── DINOv2 인코더 ────────────────────────────────────────────────────
class DINOv2Encoder:
    def __init__(self, model_name: str = DINO_MODEL):
        from transformers import AutoImageProcessor, AutoModel
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] DINOv2 로드 중: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(self.dev).eval()
        print(f"[+] DINOv2 완료 ({self.dev})")

    @torch.inference_mode()
    def encode(self, paths: list[Path]) -> np.ndarray:
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dev) for k, v in inputs.items()}
        out = self.model(**inputs)
        # CLS 토큰 사용 (last_hidden_state[:, 0])
        feat = out.last_hidden_state[:, 0]
        return l2(feat.float().cpu().numpy())   # (N, 768)


# ── 통계 ─────────────────────────────────────────────────────────────
def print_stats(qdrant: ObjectQdrantManager, total_crops: int):
    print(f"\n[통계]")
    print(f"  객체 크롭 총계:         {total_crops:,}개")
    print(f"  Qdrant (forensic_object_local): {qdrant.ntotal:,} 포인트")
    missing = total_crops - qdrant.ntotal
    if missing <= 0:
        print(f"  ✓ 인제스트 완료")
    else:
        print(f"  ✗ 미완료: {missing:,}개 남음")


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="SCVD 객체 임베딩 → Qdrant")
    ap.add_argument("--batch-size",    type=int, default=32)
    ap.add_argument("--max-crops",     type=int, help="디버그용 최대 처리 수")
    ap.add_argument("--force",         action="store_true", help="기존 레코드도 재삽입")
    ap.add_argument("--validate-only", action="store_true", help="통계만 출력")
    args = ap.parse_args()

    print("=" * 72)
    print("SCVD OBJECT EMBED v2  (SigLIP2 768 + DINOv2 768 → Qdrant)")
    print("=" * 72)

    if not OBJECT_TRACK_ROOT.exists():
        print(f"[!] 객체 크롭 루트 없음: {OBJECT_TRACK_ROOT}")
        print("    RF-DETR 객체 추출 먼저 실행:")
        print("    python src/RF-DETR/scvd_object_track_v2.py --split all")
        return

    stems     = all_scvd_video_stems()
    info_idx  = build_video_info_index()
    all_crops = list_crops(stems)
    print(f"객체 크롭 총계: {len(all_crops):,}개  (루트: {OBJECT_TRACK_ROOT})")

    qdrant = ObjectQdrantManager()

    if args.validate_only:
        print_stats(qdrant, len(all_crops))
        qdrant.close()
        return

    # ── 이어하기 ─────────────────────────────────────────────────
    if not args.force and qdrant.ntotal > 0:
        all_ids  = [crop_to_id(p) for p in all_crops]
        existing = qdrant.existing_ids(all_ids)
        pending  = [p for p, pid in zip(all_crops, all_ids) if pid not in existing]
        print(f"이미 완료 건너뜀: {len(all_crops) - len(pending):,}  남음: {len(pending):,}")
    else:
        pending = all_crops
        reason  = "--force" if args.force else "Qdrant 비어 있음(최초 실행)"
        print(f"{reason}: {len(pending):,}개 전체 처리")

    if args.max_crops:
        pending = pending[:args.max_crops]

    if not pending:
        print("처리할 크롭 없음. 종료.")
        qdrant.close()
        return

    total = len(pending)
    print(f"처리 대상: {total:,}개\n")

    siglip = SigLIP2Encoder()
    dino   = DINOv2Encoder()

    done = 0
    for start in range(0, total, args.batch_size):
        batch = pending[start : start + args.batch_size]
        ids   = [crop_to_id(p) for p in batch]
        metas = [crop_meta(p, info_idx) for p in batch]

        siglip_vecs = siglip.encode(batch)   # (B, 768)
        dino_vecs   = dino.encode(batch)     # (B, 768)

        payloads = [{
            k: m[k] for k in
            ("video", "track", "label", "source", "split",
             "category", "frame", "confidence", "path")
        } for m in metas]

        qdrant.insert(
            point_ids   = ids,
            siglip_vecs = siglip_vecs,
            dino_vecs   = dino_vecs,
            payloads    = payloads,
        )

        done += len(batch)
        pct = done / total * 100
        print(f"  [{done:>6,}/{total:,}] {pct:5.1f}%  "
              f"siglip={siglip_vecs.shape}  dino={dino_vecs.shape}", end="\r")

    qdrant.close()

    print("\n\n" + "=" * 72)
    print("SCVD OBJECT EMBED COMPLETE")
    print(f"  SigLIP2 768-dim + DINOv2 768-dim 삽입: {done:,}개")
    print(f"  Qdrant forensic_object_local → 검색 준비 완료")
    print("=" * 72)
    print("\n다음 단계: python ingest_scvd_objects_v2.py --validate-only")


if __name__ == "__main__":
    main()
