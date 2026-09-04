"""
embed_scvd_person_pass1_v2.py — SCVD 인물 (SigLIP2 + IRRA) 임베딩 → Qdrant + PostgreSQL

pass1: SigLIP2(768) + IRRA(512) → forensic_person_local  (solider = zeros placeholder)
       + PostgreSQL frames 테이블 upsert
pass2: SOLIDER(1024) → update_solider() 로 업데이트

사용법:
  python embed_scvd_person_pass1_v2.py
  python embed_scvd_person_pass1_v2.py --batch-size 32 --max-crops 500
  python embed_scvd_person_pass1_v2.py --force   # 기존 레코드도 재삽입

환경변수:
  FORENSIC_QDRANT_PATH  기본: <ROOT>/data/qdrant_local
  FORENSIC_PG_DSN       기본: postgresql://forensic:forensic_secret@localhost:5432/forensic_db
"""

from __future__ import annotations

import argparse
import sys
import uuid
import zipfile
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoProcessor, AutoModel

# ── 경로 ──────────────────────────────────────────────────────────────
_HERE    = Path(__file__).parent
ROOT_DIR = _HERE.parents[1]

sys.path.insert(0, str(_HERE))
from db_manager              import _DEFAULT_DSN
from qdrant_person_manager   import PersonQdrantManager

CROP_ROOT = ROOT_DIR / "data" / "scvd_person_tracks_v1"
SCVD_ROOT = ROOT_DIR / "data" / "scvd" / "SCVD_converted"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

IRRA_REPO    = ROOT_DIR / "src" / "IRRA"
IRRA_ARCHIVE = ROOT_DIR / "weights" / "IRRA" / "irra_cuhk_pedes_download"
IRRA_DIR     = ROOT_DIR / "weights" / "IRRA" / "cuhk_pedes"

SIGLIP2_MODEL = "google/siglip2-base-patch16-224"

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── ID 생성 ────────────────────────────────────────────────────────────
def crop_to_id(p: Path) -> str:
    """경로 기반 결정론적 UUID5 (Qdrant point ID)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "person|scvd|" + str(p.resolve())))


def crop_to_pg_id(p: Path) -> int:
    """PostgreSQL frames 테이블용 int64 ID."""
    u = uuid.uuid5(uuid.NAMESPACE_URL, "person|scvd|" + str(p.resolve()))
    return u.int >> 65


# ── 유틸 ──────────────────────────────────────────────────────────────
def l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1
    return x / n


def extract_irra() -> tuple[Path, Path]:
    IRRA_DIR.mkdir(parents=True, exist_ok=True)
    best = IRRA_DIR / "best.pth"
    cfg  = IRRA_DIR / "configs.yaml"
    if not (best.exists() and cfg.exists()):
        with zipfile.ZipFile(IRRA_ARCHIVE, "r") as z:
            z.extractall(IRRA_DIR)
    return best, cfg


# ── 크롭 목록 ─────────────────────────────────────────────────────────
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


def build_video_file_index() -> dict[str, Path]:
    """stem → 영상 파일 경로 (Train + Test)."""
    return {p.stem: p for p in SCVD_ROOT.rglob("*") if p.is_file()}


def build_video_info_index() -> dict[str, dict]:
    """stem → {split: 'Train'|'Test', category: 'Normal'|'Violence'|'Weaponized'}."""
    idx: dict[str, dict] = {}
    for p in SCVD_ROOT.rglob("*"):
        if not p.is_file():
            continue
        try:
            parts = p.relative_to(SCVD_ROOT).parts  # ('Train', 'Normal', 'video.mp4')
            idx[p.stem] = {
                "split":    parts[0] if len(parts) > 0 else "",
                "category": parts[1] if len(parts) > 1 else "",
            }
        except Exception:
            pass
    return idx


def crop_meta(p: Path, file_idx: dict[str, Path], info_idx: dict[str, dict]) -> dict:
    video_stem = track_video_stem(p)
    track_dir  = p.parent.name
    vp         = file_idx.get(video_stem)
    video_name = vp.name if vp is not None else video_stem
    info       = info_idx.get(video_stem, {})
    parts      = p.stem.split("_")
    frame_str  = parts[1] if len(parts) > 1 and parts[0] == "frame" else p.stem
    return {
        "video":    video_name,
        "track":    track_dir,
        "source":   "SCVD",
        "split":    info.get("split", ""),
        "category": info.get("category", ""),
        "frame":    frame_str,
        "person":   "",
        "path":     str(p),
    }


# ── SigLIP2 인코더 ────────────────────────────────────────────────────
class SigLIP2Encoder:
    def __init__(self, model_name: str = SIGLIP2_MODEL):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] SigLIP2 로드 중: {model_name}")
        _dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.dev_dtype = _dtype
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=_dtype,
            low_cpu_mem_usage=True,
        ).to(self.dev).eval()
        print(f"[+] SigLIP2 로드 완료 ({self.dev}, dtype={_dtype})")

    @torch.inference_mode()
    def encode(self, paths: list[Path]) -> np.ndarray:
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: (v.to(self.dev_dtype) if v.is_floating_point() else v).to(self.dev)
                  for k, v in inputs.items()}
        out = self.model.get_image_features(**inputs)
        if hasattr(out, "pooler_output"):
            feat = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            feat = out.last_hidden_state[:, 0]
        else:
            feat = out
        return l2(feat.float().cpu().numpy())   # (B, 768)


# ── IRRA 인코더 ──────────────────────────────────────────────────────
class IRRAEncoder:
    def __init__(self, best: Path, cfgp: Path):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        sys.path.insert(0, str(IRRA_REPO))
        from utils.iotools    import load_train_configs
        from model            import build_model
        from utils.checkpoint import Checkpointer

        a = load_train_configs(str(cfgp))
        a.training = False
        self.model = build_model(a, num_classes=11003)
        Checkpointer(self.model).load(f=str(best))
        self.model = self.model.to(self.dev).eval()

        h, w = a.img_size
        self.tf = T.Compose([
            T.Resize((h, w)),
            T.ToTensor(),
            T.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        print(f"[+] IRRA 로드 완료 ({self.dev})")

    @torch.inference_mode()
    def encode(self, paths: list[Path]) -> np.ndarray:
        b = torch.stack([
            self.tf(Image.open(p).convert("RGB")) for p in paths
        ]).to(self.dev)
        return l2(self.model.encode_image(b).float().cpu().numpy())  # (B, 512)


# ── PostgreSQL 헬퍼 ───────────────────────────────────────────────────
def pg_upsert_frames(conn, rows: list[tuple]) -> None:
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO frames
                (milvus_id, video, track, source, frame, person, path)
            VALUES %s
            ON CONFLICT (milvus_id) DO UPDATE SET
                video  = EXCLUDED.video,
                track  = EXCLUDED.track,
                source = EXCLUDED.source,
                frame  = EXCLUDED.frame,
                person = EXCLUDED.person,
                path   = EXCLUDED.path
            """,
            rows,
            page_size=1000,
        )
    conn.commit()


def pg_rebuild_tracks(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO tracks (video, track, source, n_frames, best_path)
            SELECT video, track, source, COUNT(*), MIN(path)
            FROM frames
            GROUP BY video, track, source
            ON CONFLICT (video, track) DO UPDATE SET
                n_frames  = EXCLUDED.n_frames,
                best_path = EXCLUDED.best_path
        """)
        n = cur.rowcount
    conn.commit()
    return n


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="SCVD 인물 SigLIP2+IRRA 임베딩 → Qdrant + PostgreSQL"
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-crops",  type=int, help="디버그용 최대 처리 수")
    ap.add_argument("--force",      action="store_true", help="기존 레코드도 재삽입")
    ap.add_argument("--pg-dsn",     default=_DEFAULT_DSN)
    args = ap.parse_args()

    print("=" * 72)
    print("SCVD PERSON EMBED v2 PASS1  (SigLIP2 768 + IRRA 512 → Qdrant + PG)")
    print("=" * 72)

    stems     = all_scvd_video_stems()
    file_idx  = build_video_file_index()
    info_idx  = build_video_info_index()
    all_crops = list_crops(stems)
    print(f"SCVD 영상(Train+Test): {len(stems):,}개")
    print(f"인물 크롭 총계        : {len(all_crops):,}개")

    pg_conn = psycopg2.connect(args.pg_dsn)
    qdrant  = PersonQdrantManager()

    # ── 이어하기: Qdrant 에 없는 것만 처리 ─────────────────────────
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
        pg_conn.close()
        qdrant.close()
        return

    total = len(pending)
    print(f"처리 대상: {total:,}개\n")

    siglip    = SigLIP2Encoder()
    best, cfg = extract_irra()
    irra      = IRRAEncoder(best, cfg)

    done = 0
    for start in range(0, total, args.batch_size):
        batch = pending[start : start + args.batch_size]
        ids   = [crop_to_id(p)    for p in batch]
        pgids = [crop_to_pg_id(p) for p in batch]
        metas = [crop_meta(p, file_idx, info_idx) for p in batch]

        siglip_vecs = siglip.encode(batch)   # (B, 768)
        irra_vecs   = irra.encode(batch)     # (B, 512)

        # Qdrant 삽입 (solider = zeros placeholder)
        qdrant.insert(
            point_ids   = ids,
            siglip_vecs = siglip_vecs,
            irra_vecs   = irra_vecs,
            payloads    = [{k: m[k] for k in ("video","track","source","split","category")} for m in metas],
        )

        # PostgreSQL frames upsert
        pg_upsert_frames(pg_conn, [
            (pgid, m["video"], m["track"], m["source"],
             m["frame"], m["person"], m["path"])
            for pgid, m in zip(pgids, metas)
        ])

        done += len(batch)
        pct = done / total * 100
        print(f"  [{done:>6,}/{total:,}] {pct:5.1f}%  "
              f"siglip={siglip_vecs.shape}  irra={irra_vecs.shape}", end="\r")

    print(f"\n\ntracks 테이블 재구성 중 ...")
    n_tracks = pg_rebuild_tracks(pg_conn)
    print(f"[+] tracks: {n_tracks:,}개 트랙")

    pg_conn.close()
    qdrant.close()

    print("\n" + "=" * 72)
    print("SCVD PERSON EMBED PASS1 COMPLETE")
    print(f"  Qdrant (forensic_person_local)")
    print(f"    siglip  : {done:,} × 768-dim  (SigLIP2)")
    print(f"    irra    : {done:,} × 512-dim  (IRRA)")
    print(f"    solider : {done:,} × 1024-dim (zeros → pass2에서 채움)")
    print(f"  PostgreSQL")
    print(f"    frames : {done:,}  /  tracks : {n_tracks:,}")
    print("=" * 72)
    print("\n다음 단계: python embed_scvd_person_pass2_v2.py  (SOLIDER 업데이트)")


if __name__ == "__main__":
    main()
