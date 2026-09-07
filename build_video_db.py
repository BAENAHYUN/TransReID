from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from dataclasses import replace
from pathlib import Path

from config import PipelineConfig
from registry import EmbedderRegistry
from router import Detection, Router
from qdrant_store import QdrantStore


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "pipeline.yaml"

SCVD_ROOT = ROOT / "data" / "scvd" / "SCVD_converted"
PERSON_ROOT = ROOT / "data" / "scvd_person_tracks_v1"
OBJECT_ROOT = ROOT / "data" / "scvd_object_tracks_v1"

STATE_DIR = ROOT / "data" / "video_embedding_checkpoint"
STATE_PATH = STATE_DIR / "state.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def make_collection_configs(cfg: PipelineConfig):
    person_retrievers = {
        name: spec
        for name, spec in cfg.retrievers.items()
        if spec.accepts_person()
    }
    object_retrievers = {
        name: spec
        for name, spec in cfg.retrievers.items()
        if spec.accepts_object()
    }

    person_cfg = replace(
        cfg,
        collection="forensic_person",
        retrievers=person_retrievers,
    )
    object_cfg = replace(
        cfg,
        collection="forensic_object",
        retrievers=object_retrievers,
    )
    return person_cfg, object_cfg


def build_video_index():
    file_idx = {}
    info_idx = {}

    if not SCVD_ROOT.exists():
        return file_idx, info_idx

    for p in SCVD_ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue

        file_idx[p.stem] = p

        try:
            parts = p.relative_to(SCVD_ROOT).parts
            info_idx[p.stem] = {
                "split": parts[0] if len(parts) > 0 else "",
                "category": parts[1] if len(parts) > 1 else "",
            }
        except Exception:
            info_idx[p.stem] = {"split": "", "category": ""}

    return file_idx, info_idx


def trailing_int(text: str):
    m = re.search(r"(\d+)$", text)
    return int(m.group(1)) if m else None


def frame_idx_from_name(stem: str) -> int:
    # frame_0045_0.92 -> 45
    m = re.search(r"(?:^|_)frame[_-]?(\d+)", stem, flags=re.I)
    if not m:
        m = re.search(r"^frame[_-]?(\d+)", stem, flags=re.I)
    if not m:
        m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else 0


def confidence_from_name(stem: str, default: float = 1.0) -> float:
    # frame_0045_0.92 -> 0.92
    m = re.search(r"_([01](?:\.\d+)?)$", stem)
    if not m:
        return default
    try:
        return float(m.group(1))
    except ValueError:
        return default


def object_label_and_track(track_dir: str):
    parts = track_dir.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return track_dir, trailing_int(track_dir)


def stable_detection_id(kind: str, crop_path: Path) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"video|scvd|{kind}|{crop_path.resolve()}",
        )
    )


def video_meta(video_stem: str, file_idx, info_idx):
    vp = file_idx.get(video_stem)
    info = info_idx.get(video_stem, {})
    return {
        "video": vp.name if vp is not None else video_stem,
        "video_path": str(vp) if vp is not None else "",
        "split": info.get("split", ""),
        "category": info.get("category", ""),
    }


def person_detection(p: Path, file_idx, info_idx) -> Detection:
    # .../scvd_person_tracks_v1/<video_stem>/<track_dir>/<crop>.jpg
    video_stem = p.parents[1].name
    track_dir = p.parent.name
    track_id = trailing_int(track_dir)
    frame_idx = frame_idx_from_name(p.stem)
    vm = video_meta(video_stem, file_idx, info_idx)
    det_id = stable_detection_id("person", p)

    extra = {
        "media_type": "video",
        "detection_id": det_id,
        "crop_id": det_id,
        "crop_path": str(p),
        "bbox_space": "frame",
        "video": vm["video"],
        "video_path": vm["video_path"],
        "track_key": f"{video_stem}/{track_dir}",
        "source": "SCVD",
        "split": vm["split"],
        "category": vm["category"],
    }

    return Detection(
        crop=str(p),
        label="person",
        score=confidence_from_name(p.stem, 1.0),
        bbox=(0, 0, 0, 0),
        image_id=vm["video_path"] or vm["video"],
        frame_idx=frame_idx,
        track_id=track_id,
        extra=extra,
    )


def object_detection(p: Path, file_idx, info_idx) -> Detection:
    # .../scvd_object_tracks_v1/<video_stem>/<label>_<track_id>/<crop>.jpg
    video_stem = p.parents[1].name
    track_dir = p.parent.name
    label, track_id = object_label_and_track(track_dir)
    frame_idx = frame_idx_from_name(p.stem)
    vm = video_meta(video_stem, file_idx, info_idx)
    det_id = stable_detection_id("object", p)

    extra = {
        "media_type": "video",
        "detection_id": det_id,
        "crop_id": det_id,
        "crop_path": str(p),
        "bbox_space": "frame",
        "video": vm["video"],
        "video_path": vm["video_path"],
        "track_key": f"{video_stem}/{track_dir}",
        "source": "SCVD",
        "split": vm["split"],
        "category": vm["category"],
    }

    return Detection(
        crop=str(p),
        label=label,
        score=confidence_from_name(p.stem, 1.0),
        bbox=(0, 0, 0, 0),
        image_id=vm["video_path"] or vm["video"],
        frame_idx=frame_idx,
        track_id=track_id,
        extra=extra,
    )


def list_person_crops():
    if not PERSON_ROOT.exists():
        return []
    return sorted(
        p for p in PERSON_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def list_object_crops():
    if not OBJECT_ROOT.exists():
        return []
    return sorted(
        p for p in OBJECT_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def save_state(next_index: int, total: int):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {"next_index": next_index, "total": total},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tmp.replace(STATE_PATH)


def load_state(total: int, fresh: bool):
    if fresh or not STATE_PATH.exists():
        return 0

    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return 0

    if int(state.get("total", -1)) != total:
        print("[!] crop 총수가 이전 실행과 달라 checkpoint를 0부터 시작합니다.")
        return 0

    return max(0, int(state.get("next_index", 0)))


def validate_vectors(detections, vectors, cfg):
    person_labels = {str(x).lower() for x in cfg.person_labels}

    for i, (det, vector_map) in enumerate(zip(detections, vectors)):
        is_person = str(det.label).lower() in person_labels

        expected = {
            name
            for name, spec in cfg.retrievers.items()
            if (
                spec.scope == "all"
                or (spec.scope == "person" and is_person)
                or (spec.scope == "object" and not is_person)
            )
        }

        actual = set(vector_map)
        if actual != expected:
            raise RuntimeError(
                f"Router routing mismatch index={i}, label={det.label}, "
                f"expected={sorted(expected)}, actual={sorted(actual)}"
            )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Existing SCVD video person/object crops -> unified "
            "forensic_person / forensic_object Qdrant collections"
        )
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-crops", type=int, default=None)
    ap.add_argument(
        "--type",
        choices=["all", "person", "object"],
        default="all",
        help="which video crops to ingest",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="ignore video checkpoint and start from index 0 (does NOT recreate Qdrant)",
    )
    args = ap.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")

    cfg = PipelineConfig.load(CONFIG_PATH)
    person_cfg, object_cfg = make_collection_configs(cfg)

    print("=" * 76)
    print("UNIFIED VIDEO DB BUILDER")
    print("=" * 76)
    print("person collection :", person_cfg.collection)
    print("person vectors    :", sorted(person_cfg.retrievers))
    print("object collection :", object_cfg.collection)
    print("object vectors    :", sorted(object_cfg.retrievers))
    print("IMPORTANT         : existing image points are NOT deleted")

    # Existing image collections must be reused. Never recreate here.
    person_store = QdrantStore(person_cfg)
    object_store = QdrantStore(object_cfg)
    person_store.ensure_collection(recreate=False)
    object_store.ensure_collection(recreate=False)

    file_idx, info_idx = build_video_index()

    records = []
    if args.type in ("all", "person"):
        people = list_person_crops()
        records.extend(("person", p) for p in people)
        print(f"person video crops: {len(people):,}")

    if args.type in ("all", "object"):
        objects = list_object_crops()
        records.extend(("object", p) for p in objects)
        print(f"object video crops: {len(objects):,}")

    if args.max_crops is not None:
        records = records[: args.max_crops]

    total = len(records)
    if total == 0:
        print("No video crops found.")
        print("person root:", PERSON_ROOT)
        print("object root:", OBJECT_ROOT)
        return

    start_index = load_state(total, args.fresh)
    if start_index > 0:
        # deterministic IDs make a one-batch replay safe
        start_index = max(0, start_index - args.batch_size)
        print(f"resume with safety replay: {start_index:,}/{total:,}")

    # Use the SAME registry/router as the image DB.
    # This keeps SigLIP2/DINOv2 model IDs and preprocessing identical to pipeline.yaml.
    registry = EmbedderRegistry(cfg)
    router = Router(cfg, registry, input_format="rgb")

    run_start = time.time()
    processed = 0
    uploaded_person_total = 0
    uploaded_object_total = 0

    try:
        for start in range(start_index, total, args.batch_size):
            end = min(start + args.batch_size, total)
            batch_records = records[start:end]

            detections = []
            for kind, p in batch_records:
                if kind == "person":
                    detections.append(person_detection(p, file_idx, info_idx))
                else:
                    detections.append(object_detection(p, file_idx, info_idx))

            vectors = router.embed(detections)
            validate_vectors(detections, vectors, cfg)

            person_dets = []
            person_vecs = []
            object_dets = []
            object_vecs = []

            for det, vec_map in zip(detections, vectors):
                if str(det.label).lower() in {str(x).lower() for x in cfg.person_labels}:
                    person_dets.append(det)
                    person_vecs.append({
                        k: v for k, v in vec_map.items()
                        if k in person_cfg.retrievers
                    })
                else:
                    object_dets.append(det)
                    object_vecs.append({
                        k: v for k, v in vec_map.items()
                        if k in object_cfg.retrievers
                    })

            uploaded_person = (
                person_store.upsert(
                    person_dets,
                    person_vecs,
                    batch_size=args.batch_size,
                )
                if person_dets else 0
            )
            uploaded_object = (
                object_store.upsert(
                    object_dets,
                    object_vecs,
                    batch_size=args.batch_size,
                )
                if object_dets else 0
            )

            uploaded_person_total += uploaded_person
            uploaded_object_total += uploaded_object

            save_state(end, total)

            processed += end - start
            elapsed = time.time() - run_start
            rate = processed / elapsed if elapsed > 0 else 0.0
            remaining = total - end
            eta = remaining / rate if rate > 0 else 0.0

            h, rem = divmod(int(eta), 3600)
            m, s = divmod(rem, 60)

            print(
                f"[{end:,}/{total:,}] {end / total * 100:6.2f}% | "
                f"{rate:6.2f} crop/s | ETA {h:02d}:{m:02d}:{s:02d} | "
                f"Qdrant person={uploaded_person} object={uploaded_object}"
            )

    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoint was saved after the last successful batch.")
        return

    print("\n" + "=" * 76)
    print("VIDEO DB BUILD COMPLETE")
    print(f"processed      : {total:,}")
    print(f"person upserts : {uploaded_person_total:,}")
    print(f"object upserts : {uploaded_object_total:,}")
    print("image points   : preserved")
    print("=" * 76)


if __name__ == "__main__":
    main()
