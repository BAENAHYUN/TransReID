"""
ingest_coco_persons.py — COCO 데이터셋 인물 임베딩 → Qdrant + SQLite

COCO annotations/instances_*.json 에서 person(category_id=1) BBox를 읽어
crop 후 SigLIP2(768) + IRRA(512) + SOLIDER(1024) 임베딩 → forensic_person_local

사용법:
  python ingest_coco_persons.py --img-dir data/coco/train2017 \
                                 --ann    data/coco/annotations/instances_train2017.json

  python ingest_coco_persons.py --img-dir data/coco/val2017 \
                                 --ann    data/coco/annotations/instances_val2017.json \
                                 --batch-size 64

  python ingest_coco_persons.py --img-dir ...  --ann ...  --validate-only

옵션:
  --batch-size   임베딩 배치 크기 (기본 32, GPU 메모리에 따라 조정)
  --min-area     최소 bbox 넓이 px² (기본 1024 = 32×32, 너무 작은 crop 제외)
  --min-score    COCO annotation 없음 (bbox confidence 무관, 0.0 고정)
  --force        이미 처리된 crop도 재삽입
  --validate-only  통계만 출력
  --max-images   디버그용 이미지 수 제한
  --split        Qdrant payload split 필드값 (기본: 어노테이션 파일명 추출)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True   # 손상된 이미지 건너뜀

# ── 경로 ──────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
ROOT_DIR = _HERE.parents[1]

sys.path.insert(0, str(_HERE))
from qdrant_person_manager import PersonQdrantManager

IRRA_REPO    = ROOT_DIR / "src" / "IRRA"
IRRA_ARCHIVE = ROOT_DIR / "weights" / "IRRA" / "irra_cuhk_pedes_download"
IRRA_DIR     = ROOT_DIR / "weights" / "IRRA" / "cuhk_pedes"
SOLIDER_ROOT   = ROOT_DIR / "src" / "SOLIDER-REID"
SOLIDER_WEIGHT = ROOT_DIR / "weights" / "SOLIDER" / "solider_market_swin_base.pth"
SIGLIP2_MODEL  = "google/siglip2-base-patch16-224"
DB_PATH        = ROOT_DIR / "data" / "forensic.db"

COCO_PERSON_ID = 1          # COCO category_id for "person"
USE_FP16       = torch.cuda.is_available()
IMG_EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SOURCE = "COCO"


# ── UUID ──────────────────────────────────────────────────────────────
def crop_id(img_id: int, ann_id: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL,
                          f"coco|person|img{img_id}|ann{ann_id}"))

def track_id(img_id: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL,
                          f"coco|track|img{img_id}"))


# ── L2 정규화 ──────────────────────────────────────────────────────────
def l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1
    return x / n


# ── COCO 어노테이션 파싱 ───────────────────────────────────────────────
def load_coco_persons(ann_path: Path, min_area: float) -> dict[int, list[dict]]:
    """
    {image_id: [{ann_id, bbox:[x,y,w,h], area}, ...]} — person만.
    min_area 이하 bbox 제외.
    """
    print(f"[*] 어노테이션 로드: {ann_path}")
    with open(ann_path, encoding="utf-8") as f:
        data = json.load(f)

    # image_id → filename 매핑
    id2file: dict[int, str] = {img["id"]: img["file_name"]
                                for img in data["images"]}

    groups: dict[int, list[dict]] = defaultdict(list)
    n_skip = 0
    for ann in data["annotations"]:
        if ann["category_id"] != COCO_PERSON_ID:
            continue
        area = ann.get("area", 0)
        if area < min_area:
            n_skip += 1
            continue
        groups[ann["image_id"]].append({
            "ann_id": ann["id"],
            "bbox":   ann["bbox"],   # [x, y, w, h]
            "area":   area,
        })

    print(f"  이미지: {len(id2file):,}장  |  person 어노테이션: "
          f"{sum(len(v) for v in groups.values()):,}개  "
          f"(min_area<{min_area:.0f} 제외: {n_skip:,}개)")
    return groups, id2file


# ── 이미지에서 crop 잘라내기 ──────────────────────────────────────────
def crop_person(img: Image.Image, bbox: list) -> Image.Image | None:
    """COCO bbox [x, y, w, h] → PIL crop (최소 16×16 보장)."""
    x, y, w, h = bbox
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    return img.crop((x1, y1, x2, y2))


# ── 모델 인코더 ───────────────────────────────────────────────────────
class SigLIP2Encoder:
    def __init__(self):
        from transformers import AutoProcessor, AutoModel
        self.dev   = "cuda" if torch.cuda.is_available() else "cpu"
        dtype      = torch.float16 if USE_FP16 else torch.float32
        print(f"[*] SigLIP2 로드 ({self.dev}, {dtype})...")
        self.proc  = AutoProcessor.from_pretrained(SIGLIP2_MODEL)
        self.model = AutoModel.from_pretrained(
            SIGLIP2_MODEL, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(self.dev).eval()
        print("[+] SigLIP2 완료")

    @torch.inference_mode()
    def encode(self, imgs: list[Image.Image]) -> np.ndarray:
        inp = self.proc(images=imgs, return_tensors="pt", padding=True)
        inp = {k: (v.half() if USE_FP16 and v.is_floating_point() else v).to(self.dev)
               for k, v in inp.items()}
        out  = self.model.get_image_features(**inp)
        feat = (out.pooler_output if hasattr(out, "pooler_output")
                else out.last_hidden_state[:, 0])
        return l2(feat.float().cpu().numpy())


class IRRAEncoder:
    def __init__(self):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        sys.path.insert(0, str(IRRA_REPO))
        from utils.iotools    import load_train_configs
        from model            import build_model
        from utils.checkpoint import Checkpointer

        IRRA_DIR.mkdir(parents=True, exist_ok=True)
        best = IRRA_DIR / "best.pth"
        cfg_p = IRRA_DIR / "configs.yaml"
        if not (best.exists() and cfg_p.exists()):
            with zipfile.ZipFile(IRRA_ARCHIVE, "r") as z:
                z.extractall(IRRA_DIR)

        print(f"[*] IRRA 로드 ({self.dev}, fp32 고정)...")
        a = load_train_configs(str(cfg_p))
        a.training = False
        self.model = build_model(a, num_classes=11003)
        Checkpointer(self.model).load(f=str(best))
        self.model = self.model.to(self.dev).eval()
        h, w = a.img_size
        self.tf = T.Compose([
            T.Resize((h, w)), T.ToTensor(),
            T.Normalize([0.48145466, 0.4578275, 0.40821073],
                        [0.26862954, 0.26130258, 0.27577711]),
        ])
        print("[+] IRRA 완료")

    @torch.inference_mode()
    def encode(self, imgs: list[Image.Image]) -> np.ndarray:
        b = torch.stack([self.tf(img) for img in imgs]).to(self.dev)
        return l2(self.model.encode_image(b).float().cpu().numpy())


class SOLIDEREncoder:
    def __init__(self):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        sys.path.insert(0, str(SOLIDER_ROOT))
        from config import cfg
        from model  import make_model

        print(f"[*] SOLIDER 로드 ({self.dev}, fp16={USE_FP16})...")
        cfg.merge_from_file(str(SOLIDER_ROOT / "configs" / "market" / "swin_base.yml"))
        cfg.TEST.WEIGHT = str(SOLIDER_WEIGHT)
        cfg.MODEL.PRETRAIN_CHOICE = "self"
        cfg.SOLVER.IMS_PER_BATCH  = 64
        cfg.freeze()
        state      = torch.load(str(SOLIDER_WEIGHT), map_location="cpu")
        state_dict = state.get("model", state)
        clf_key    = next((k for k in state_dict if "classifier" in k and "weight" in k), None)
        num_class  = state_dict[clf_key].shape[0] if clf_key else 751
        self.model = make_model(cfg, num_class=num_class, camera_num=6, view_num=1,
                                semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = (self.model.half() if USE_FP16 else self.model).to(self.dev).eval()
        self.tf = T.Compose([
            T.Resize((256, 128)), T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        print("[+] SOLIDER 완료")

    @torch.inference_mode()
    def encode(self, imgs: list[Image.Image]) -> np.ndarray:
        b = torch.stack([self.tf(img) for img in imgs])
        b = (b.half() if USE_FP16 else b).to(self.dev)
        out = self.model(b)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return l2(out[:, :1024].float().cpu().numpy())


# ── SQLite 삽입 ────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    id TEXT PRIMARY KEY, video TEXT, track TEXT, source TEXT,
    split TEXT, category TEXT, path TEXT,
    frame_num INTEGER, confidence REAL, timestamp_sec REAL
);
CREATE TABLE IF NOT EXISTS tracks (
    id TEXT PRIMARY KEY, video TEXT, track TEXT, source TEXT,
    split TEXT, category TEXT, best_path TEXT,
    n_frames INTEGER, start_frame INTEGER, end_frame INTEGER,
    start_sec REAL, end_sec REAL
);
CREATE INDEX IF NOT EXISTS idx_frames_video_track ON frames(video, track);
CREATE INDEX IF NOT EXISTS idx_tracks_video       ON tracks(video);
CREATE INDEX IF NOT EXISTS idx_tracks_source      ON tracks(source);
"""

def open_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH))
    db.executescript(SCHEMA)
    return db


def upsert_frame(cur: sqlite3.Cursor, fid: str, img_id: int,
                 ann_id: int, split: str, bbox_path: str):
    cur.execute("""
        INSERT OR REPLACE INTO frames
        (id, video, track, source, split, category,
         path, frame_num, confidence, timestamp_sec)
        VALUES (?,?,?,?,?,?, ?,?,?,?)
    """, (fid,
          f"coco_img_{img_id}",   # video 필드 = 이미지 파일 식별자
          f"ann_{ann_id}",
          SOURCE, split, "",
          bbox_path, 0, 1.0, 0.0))


def upsert_track(cur: sqlite3.Cursor, tid: str, img_id: int,
                 anns: list[dict], split: str, best_path: str):
    cur.execute("""
        INSERT OR REPLACE INTO tracks
        (id, video, track, source, split, category,
         best_path, n_frames, start_frame, end_frame, start_sec, end_sec)
        VALUES (?,?,?,?,?,?, ?,?,?,?,?,?)
    """, (tid,
          f"coco_img_{img_id}",
          f"img_{img_id}_persons",
          SOURCE, split, "",
          best_path, len(anns), 0, 0, 0.0, 0.0))


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="COCO 인물 임베딩 → Qdrant + SQLite")
    ap.add_argument("--img-dir",     required=True, type=Path, help="이미지 폴더 (train2017 등)")
    ap.add_argument("--ann",         required=True, type=Path, help="COCO instances_*.json 경로")
    ap.add_argument("--batch-size",  type=int, default=32)
    ap.add_argument("--min-area",    type=float, default=1024.0, help="최소 bbox 넓이 px² (기본 1024)")
    ap.add_argument("--force",       action="store_true", help="기존 레코드 재삽입")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--max-images",  type=int, default=None, help="디버그용 이미지 수 제한")
    ap.add_argument("--split",       type=str, default=None,
                    help="Qdrant payload split 값 (미지정 시 파일명에서 추출: train/val)")
    args = ap.parse_args()

    # split 자동 추출: instances_train2017.json → "train"
    if args.split:
        split = args.split
    else:
        name = args.ann.stem.lower()   # instances_train2017
        split = "train" if "train" in name else "val" if "val" in name else "coco"

    print("=" * 72)
    print("COCO PERSON INGEST")
    print(f"  이미지 폴더: {args.img_dir}")
    print(f"  어노테이션 : {args.ann}")
    print(f"  split      : {split}")
    print(f"  배치 크기  : {args.batch_size}  |  min_area: {args.min_area:.0f}px²")
    print("=" * 72)

    # ── 어노테이션 로드 ────────────────────────────────────────────
    groups, id2file = load_coco_persons(args.ann, args.min_area)

    img_ids = sorted(groups.keys())
    if args.max_images:
        img_ids = img_ids[:args.max_images]
    total_imgs  = len(img_ids)
    total_crops = sum(len(groups[i]) for i in img_ids)
    print(f"\n처리 대상: 이미지 {total_imgs:,}장  |  person crop 최대 {total_crops:,}개")

    if args.validate_only:
        qdrant = PersonQdrantManager()
        db     = open_db()
        cur    = db.cursor()
        fn = cur.execute("SELECT COUNT(*) FROM frames WHERE source='COCO'").fetchone()[0]
        tn = cur.execute("SELECT COUNT(*) FROM tracks WHERE source='COCO'").fetchone()[0]
        print(f"\n[SQLite] frames(COCO)={fn:,}  tracks(COCO)={tn:,}")
        print(f"[Qdrant] forensic_person_local = {qdrant.ntotal:,} 포인트 (전체)")
        db.close(); qdrant.close()
        return

    # ── 이어하기: 이미 처리된 ID 확인 ────────────────────────────
    qdrant = PersonQdrantManager()
    if not args.force and qdrant.ntotal > 0:
        all_fids = []
        for iid in img_ids:
            for ann in groups[iid]:
                all_fids.append(crop_id(iid, ann["ann_id"]))
        existing = qdrant.existing_ids(all_fids)
        print(f"  이미 완료: {len(existing):,}  |  남은 crop: {total_crops - len(existing):,}")
    else:
        existing = set()

    db  = open_db()
    cur = db.cursor()

    # ── 모델 로드 ─────────────────────────────────────────────────
    siglip  = SigLIP2Encoder()
    irra    = IRRAEncoder()
    solider = SOLIDEREncoder()

    # ── 배치 루프 ─────────────────────────────────────────────────
    batch_imgs:    list[Image.Image] = []
    batch_meta:    list[dict]        = []
    done_crops     = 0
    done_imgs      = 0
    err_count      = 0

    def flush_batch():
        nonlocal done_crops
        if not batch_imgs:
            return
        try:
            sv = siglip.encode(batch_imgs)
            iv = irra.encode(batch_imgs)
            so = solider.encode(batch_imgs)

            ids      = [m["fid"]   for m in batch_meta]
            payloads = [{
                "video":    m["video"],
                "track":    m["track"],
                "source":   SOURCE,
                "split":    m["split"],
                "category": "",
                "path":     m["path"],
            } for m in batch_meta]

            qdrant.insert(
                point_ids    = ids,
                siglip_vecs  = sv,
                irra_vecs    = iv,
                solider_vecs = so,
                payloads     = payloads,
            )

            for m, bp in zip(batch_meta, [m["path"] for m in batch_meta]):
                upsert_frame(cur, m["fid"], m["img_id"], m["ann_id"], split, bp)

            db.commit()
            done_crops += len(batch_imgs)
        except Exception as e:
            print(f"\n  [!] 배치 오류: {e}")
        finally:
            batch_imgs.clear()
            batch_meta.clear()

    print(f"\n[임베딩 시작]\n")
    for img_idx, img_id in enumerate(img_ids):
        fname = id2file.get(img_id, "")
        img_path = args.img_dir / fname
        if not img_path.exists():
            # 확장자 없으면 탐색
            candidates = [p for p in args.img_dir.glob(f"{Path(fname).stem}.*")
                          if p.suffix.lower() in IMG_EXTS]
            if not candidates:
                err_count += 1
                continue
            img_path = candidates[0]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            err_count += 1
            continue

        anns       = groups[img_id]
        best_area  = -1
        best_path  = ""

        for ann in anns:
            fid = crop_id(img_id, ann["ann_id"])
            if fid in existing:
                continue

            crop = crop_person(img, ann["bbox"])
            if crop is None:
                continue

            if ann["area"] > best_area:
                best_area = ann["area"]
                best_path = str(img_path)

            batch_imgs.append(crop)
            batch_meta.append({
                "fid":    fid,
                "img_id": img_id,
                "ann_id": ann["ann_id"],
                "video":  f"coco_img_{img_id}",
                "track":  f"ann_{ann['ann_id']}",
                "split":  split,
                "path":   str(img_path),
            })

        # 트랙(이미지 단위) 기록
        if best_path:
            tid = track_id(img_id)
            upsert_track(cur, tid, img_id, anns, split, best_path)

        if len(batch_imgs) >= args.batch_size:
            flush_batch()

        done_imgs += 1
        pct = done_imgs / total_imgs * 100
        print(f"  [{done_imgs:>6,}/{total_imgs:,}] {pct:5.1f}%  "
              f"crops={done_crops:,}  err={err_count}", end="\r")

    flush_batch()   # 나머지 처리
    db.commit()

    print(f"\n\n{'=' * 72}")
    print("COCO PERSON INGEST COMPLETE")
    print(f"  처리 이미지 : {done_imgs:,}장")
    print(f"  삽입 crops  : {done_crops:,}개")
    print(f"  오류 건너뜀 : {err_count:,}개")
    print(f"  Qdrant      : forensic_person_local  source=COCO")
    print(f"  SQLite      : {DB_PATH}")
    print(f"{'=' * 72}")
    print("\n다음: python build_track_embeddings.py  (트랙 집계 임베딩 갱신)")

    qdrant.close()
    db.close()


if __name__ == "__main__":
    main()
