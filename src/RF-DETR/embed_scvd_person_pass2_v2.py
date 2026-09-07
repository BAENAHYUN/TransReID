"""
embed_scvd_person_pass2_v2.py — SCVD 인물 SOLIDER 임베딩 업데이트 → Qdrant

pass1 에서 solider 벡터를 zeros placeholder 로 삽입했습니다.
이 스크립트는 SOLIDER(1024-dim) 을 인코딩하여 Qdrant 에 업데이트합니다.

사용법:
  python embed_scvd_person_pass2_v2.py
  python embed_scvd_person_pass2_v2.py --batch-size 32 --max-crops 500
  python embed_scvd_person_pass2_v2.py --validate-only   # 통계만 출력

환경변수:
  FORENSIC_QDRANT_PATH  기본: <ROOT>/data/qdrant_local
  FORENSIC_PG_DSN       기본: postgresql://forensic:forensic_secret@localhost:5432/forensic_db
"""

from __future__ import annotations

import argparse
import random
import sys
import uuid
from pathlib import Path

import numpy as np
import psycopg2
import torch
import torchvision.transforms as T
from PIL import Image

# ── 경로 ──────────────────────────────────────────────────────────────
_HERE    = Path(__file__).parent
ROOT_DIR = _HERE.parents[1]

sys.path.insert(0, str(_HERE))
from db_manager            import _DEFAULT_DSN
from qdrant_person_manager import PersonQdrantManager

CROP_ROOT = ROOT_DIR / "data" / "scvd_person_tracks_v1"
SCVD_ROOT = ROOT_DIR / "data" / "scvd" / "SCVD_converted"

SOLIDER_ROOT   = ROOT_DIR / "src" / "SOLIDER-REID"
SOLIDER_WEIGHT = ROOT_DIR / "weights" / "SOLIDER" / "solider_market_swin_base.pth"

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── ID 생성 (pass1 과 동일) ────────────────────────────────────────────
def crop_to_id(p: Path) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "person|scvd|" + str(p.resolve())))


# ── 유틸 ──────────────────────────────────────────────────────────────
def l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1
    return x / n


def all_scvd_video_stems() -> set[str]:
    """Train + Test 전체 영상 stem."""
    return {p.stem for p in SCVD_ROOT.rglob("*") if p.is_file()}


def track_video_stem(p: Path) -> str:
    return p.parents[1].name


def list_crops(stems: set[str]) -> list[Path]:
    return sorted(
        p for p in CROP_ROOT.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTS
        and track_video_stem(p) in stems
    )


# ── SOLIDER 인코더 ────────────────────────────────────────────────────
class SOLIDEREncoder:
    def __init__(self, weight_path: Path = SOLIDER_WEIGHT):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

        sys.path.insert(0, str(SOLIDER_ROOT))
        from config import cfg
        from model  import make_model

        print(f"[*] SOLIDER 로드 중: {weight_path}")
        cfg.merge_from_file(str(SOLIDER_ROOT / "configs" / "market" / "swin_base.yml"))
        cfg.TEST.WEIGHT = str(weight_path)
        cfg.MODEL.PRETRAIN_CHOICE = "self"
        cfg.SOLVER.IMS_PER_BATCH  = 64
        cfg.freeze()

        state      = torch.load(str(weight_path), map_location="cpu")
        state_dict = state.get("model", state)
        clf_key    = next((k for k in state_dict if "classifier" in k and "weight" in k), None)
        num_class  = state_dict[clf_key].shape[0] if clf_key else 751
        print(f"[*] SOLIDER num_class: {num_class}")

        self.model = make_model(cfg, num_class=num_class, camera_num=6, view_num=1,
                                semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.dev).eval()

        self.tf = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        print(f"[+] SOLIDER 로드 완료 ({self.dev})")

    @torch.inference_mode()
    def encode(self, paths: list[Path]) -> np.ndarray:
        b = torch.stack([
            self.tf(Image.open(p).convert("RGB")) for p in paths
        ]).to(self.dev)
        out = self.model(b)
        if isinstance(out, (tuple, list)):
            out = out[0]
        feat = out[:, :1024]   # (B, 1024) global feature
        return l2(feat.float().cpu().numpy())


# ── 검증 통계 ─────────────────────────────────────────────────────────
def print_stats(pg_dsn: str, qdrant: PersonQdrantManager, expected: int,
                sample: int, all_crops: list[Path]):
    pg  = psycopg2.connect(pg_dsn)
    cur = pg.cursor()
    cur.execute("SELECT COUNT(*) FROM frames WHERE source='SCVD'")
    pg_frames = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM tracks WHERE source='SCVD'")
    pg_tracks = cur.fetchone()[0]
    cur.execute(
        "SELECT video, COUNT(*) as n FROM frames WHERE source='SCVD' "
        "GROUP BY video ORDER BY n DESC LIMIT 5"
    )
    top_videos = cur.fetchall()
    pg.close()

    print(f"\n[PostgreSQL]")
    print(f"  frames (SCVD): {pg_frames:,}  /  tracks (SCVD): {pg_tracks:,}")
    for vn, cnt in top_videos:
        print(f"    {vn}: {cnt:,}")

    print(f"\n[Qdrant forensic_person_local]")
    print(f"  총 포인트 수: {qdrant.ntotal:,}")

    missing = expected - pg_frames
    print(f"\n[진단]")
    if missing <= 0:
        print(f"  ✓ pass1 완료 ({pg_frames:,}/{expected:,})")
    else:
        print(f"  ✗ pass1 미완료: {missing:,}개  → embed_scvd_person_pass1_v2.py 먼저 실행")

    # 샘플 벡터 norm 확인
    if sample > 0 and all_crops:
        sample_paths = random.sample(all_crops, min(sample, len(all_crops)))
        sample_ids   = [crop_to_id(p) for p in sample_paths]
        hits = qdrant._client.retrieve(
            collection_name = qdrant._col,
            ids             = sample_ids,
            with_vectors    = True,
            with_payload    = False,
        )
        print(f"\n[샘플 벡터 norm ({len(hits)}개)]")
        for h in hits:
            v = h.vector or {}
            sv  = np.array(v.get("siglip",  []), dtype=np.float32)
            iv  = np.array(v.get("irra",    []), dtype=np.float32)
            sol = np.array(v.get("solider", []), dtype=np.float32)
            solnorm = np.linalg.norm(sol)
            print(f"  id={str(h.id)[:8]}...  "
                  f"siglip={np.linalg.norm(sv):.3f}  "
                  f"irra={np.linalg.norm(iv):.3f}  "
                  f"solider={solnorm:.3f}"
                  f"{'  ← zero(미완료)' if solnorm < 0.01 else ''}")


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="SCVD 인물 SOLIDER 임베딩 업데이트 → Qdrant"
    )
    ap.add_argument("--batch-size",    type=int, default=32)
    ap.add_argument("--max-crops",     type=int, help="디버그용 최대 처리 수")
    ap.add_argument("--pg-dsn",        default=_DEFAULT_DSN)
    ap.add_argument("--validate-only", action="store_true",
                    help="SOLIDER 인코딩 없이 통계만 출력")
    ap.add_argument("--sample",        type=int, default=5,
                    help="벡터 norm 확인 샘플 수")
    args = ap.parse_args()

    print("=" * 72)
    print("SCVD PERSON EMBED v2 PASS2  (SOLIDER 1024 → Qdrant)")
    print("=" * 72)

    stems     = all_scvd_video_stems()
    all_crops = list_crops(stems)
    expected  = len(all_crops)
    print(f"\n인물 크롭 총계: {expected:,}개")

    qdrant = PersonQdrantManager()

    if args.validate_only:
        print_stats(args.pg_dsn, qdrant, expected, args.sample, all_crops)
        qdrant.close()
        return

    # ── pass1 완료 여부 확인 ─────────────────────────────────────
    pg  = psycopg2.connect(args.pg_dsn)
    cur = pg.cursor()
    cur.execute("SELECT COUNT(*) FROM frames WHERE source='SCVD'")
    pg_frames = cur.fetchone()[0]
    pg.close()

    if pg_frames == 0:
        print("[!] pass1 미완료 (frames 테이블이 비어 있음)")
        print("    → python embed_scvd_person_pass1_v2.py 를 먼저 실행하세요")
        qdrant.close()
        return

    print(f"  pass1 완료 프레임: {pg_frames:,}")

    pending = all_crops
    if args.max_crops:
        pending = pending[:args.max_crops]

    total = len(pending)
    print(f"처리 대상: {total:,}개 (SOLIDER 업데이트)\n")

    solider = SOLIDEREncoder()

    done = 0
    for start in range(0, total, args.batch_size):
        batch = pending[start : start + args.batch_size]
        ids   = [crop_to_id(p) for p in batch]

        solider_vecs = solider.encode(batch)   # (B, 1024)

        qdrant.update_solider(
            point_ids    = ids,
            solider_vecs = solider_vecs,
        )

        done += len(batch)
        pct = done / total * 100
        print(f"  [{done:>6,}/{total:,}] {pct:5.1f}%  solider={solider_vecs.shape}", end="\r")

    qdrant.close()

    print("\n\n" + "=" * 72)
    print("SCVD PERSON EMBED PASS2 COMPLETE")
    print(f"  SOLIDER 1024-dim 업데이트: {done:,}개")
    print(f"  Qdrant forensic_person_local → 3-vector 검색 준비 완료")
    print("=" * 72)
    print("\n다음 단계: python embed_scvd_person_pass2_v2.py --validate-only  (확인)")


if __name__ == "__main__":
    main()
