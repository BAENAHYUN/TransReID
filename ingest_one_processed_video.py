from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from config import PipelineConfig
from registry import EmbedderRegistry
from router import Detection, Router
from qdrant_store import QdrantStore

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "pipeline.yaml"
VIDEO_ROOT = ROOT / "data" / "videos"
PROCESSED_ROOT = ROOT / "outputs" / "processed_videos"
INGEST_ROOT = ROOT / "outputs" / "single_video_ingest"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def make_collection_configs(cfg: PipelineConfig):
    person_retrievers = {name: spec for name, spec in cfg.retrievers.items() if spec.accepts_person()}
    object_retrievers = {name: spec for name, spec in cfg.retrievers.items() if spec.accepts_object()}
    return (
        replace(cfg, collection="forensic_person", retrievers=person_retrievers),
        replace(cfg, collection="forensic_object", retrievers=object_retrievers),
    )


def find_video(video_stem: str) -> Path:
    direct = [p for p in VIDEO_ROOT.glob(video_stem + ".*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    if direct:
        return sorted(direct)[0]
    recursive = [p for p in VIDEO_ROOT.rglob(video_stem + ".*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    if recursive:
        return sorted(recursive)[0]
    raise FileNotFoundError(f"video not found for stem={video_stem!r} under {VIDEO_ROOT}")


def load_final_tracks(video_stem: str) -> list[dict]:
    path = PROCESSED_ROOT / video_stem / "final_routed_tracks.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON array")
    return [dict(r) for r in data if str(r.get("final_db_route", "")).lower() in {"person", "object"}]


def stable_detection_id(video_stem: str, row: dict) -> str:
    bbox = row.get("bbox") or [0, 0, 0, 0]
    key = (
        f"processed-video|{video_stem}|{row.get('final_db_route')}|"
        f"{row.get('long_track_id')}|{row.get('frame_idx')}|"
        + ",".join(f"{float(x):.4f}" for x in bbox)
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def clamp_bbox(bbox, width: int, height: int):
    x1, y1, x2, y2 = map(float, bbox)
    x1 = max(0, min(width, int(np.floor(x1))))
    y1 = max(0, min(height, int(np.floor(y1))))
    x2 = max(0, min(width, int(np.ceil(x2))))
    y2 = max(0, min(height, int(np.ceil(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def create_crops(video_path: Path, video_stem: str, rows: list[dict]):
    out_root = INGEST_ROOT / video_stem / "crops"
    out_root.mkdir(parents=True, exist_ok=True)
    by_frame: dict[int, list[dict]] = {}
    for row in rows:
        by_frame.setdefault(int(row["frame_idx"]), []).append(row)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    wanted = sorted(by_frame)
    created, dropped = [], []
    for n, frame_idx in enumerate(wanted, 1):
        if frame_count > 0 and not (0 <= frame_idx < frame_count):
            for row in by_frame[frame_idx]:
                dropped.append({"reason": "frame_out_of_range", "row": row})
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            for row in by_frame[frame_idx]:
                dropped.append({"reason": "frame_read_failed", "row": row})
            continue

        h, w = frame.shape[:2]
        for row in by_frame[frame_idx]:
            bb = clamp_bbox(row["bbox"], w, h)
            if bb is None:
                dropped.append({"reason": "invalid_bbox", "row": row})
                continue
            x1, y1, x2, y2 = bb
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                dropped.append({"reason": "empty_crop", "row": row})
                continue

            route = str(row["final_db_route"]).lower()
            long_id = int(row.get("long_track_id", -1))
            label = "person" if route == "person" else str(row.get("class_name", "object"))
            track_dir = out_root / route / f"{label}_{long_id:04d}"
            track_dir.mkdir(parents=True, exist_ok=True)

            det_id = stable_detection_id(video_stem, row)
            crop_path = track_dir / f"frame_{frame_idx:06d}_{det_id[:8]}.jpg"
            if not cv2.imwrite(str(crop_path), crop):
                dropped.append({"reason": "crop_write_failed", "row": row})
                continue

            enriched = dict(row)
            enriched["crop_path"] = str(crop_path.resolve())
            enriched["detection_id"] = det_id
            enriched["bbox_clamped"] = [x1, y1, x2, y2]
            created.append(enriched)

        if n == 1 or n % 100 == 0 or n == len(wanted):
            print(f"[CROP] frames {n:,}/{len(wanted):,} | crops={len(created):,}")

    cap.release()
    return created, dropped


def row_to_detection(video_path: Path, video_stem: str, row: dict) -> Detection:
    route = str(row["final_db_route"]).lower()
    label = "person" if route == "person" else str(row.get("class_name", "object"))
    long_id = int(row.get("long_track_id", -1))
    short_id = int(row.get("short_track_id", row.get("track_id", -1)))
    extra = {
        "media_type": "video",
        "source": "processed_video",
        "video": video_path.name,
        "video_name": video_path.name,
        "video_path": str(video_path.resolve()),
        "video_stem": video_stem,
        "frame_idx": int(row["frame_idx"]),
        "timestamp_sec": float(row.get("timestamp_sec") or 0.0),
        "detection_id": row["detection_id"],
        "crop_id": row["detection_id"],
        "crop_path": row["crop_path"],
        "bbox_space": "frame",
        "bbox": [float(x) for x in row.get("bbox_clamped", row["bbox"])],
        "track_key": f"{video_stem}/{route}_{long_id}",
        "track_id": long_id,
        "original_track_id": short_id,
        "short_track_id": short_id,
        "long_track_id": long_id,
        "stitched_id": long_id,
        "identity_id": long_id,
        "stitch_method": row.get("stitch_method"),
        "final_db_route": route,
        "final_validation_decision": row.get("final_validation_decision"),
        "final_person_score": row.get("final_person_score"),
        "validation_person_score": row.get("validation_person_score"),
        "validation_siglip_score": row.get("validation_siglip_score"),
        "validation_detector_score": row.get("validation_detector_score"),
        "route_resolution": row.get("route_resolution"),
    }
    return Detection(
        crop=row["crop_path"],
        label=label,
        score=float(row.get("confidence") or 0.0),
        bbox=tuple(float(x) for x in row.get("bbox_clamped", row["bbox"])),
        image_id=str(video_path.resolve()),
        frame_idx=int(row["frame_idx"]),
        track_id=long_id,
        extra=extra,
    )


def expected_vector_names(cfg, label: str):
    person_labels = {str(x).lower() for x in cfg.person_labels}
    is_person = str(label).lower() in person_labels
    return {
        name for name, spec in cfg.retrievers.items()
        if spec.scope == "all" or (spec.scope == "person" and is_person) or (spec.scope == "object" and not is_person)
    }


def main():
    ap = argparse.ArgumentParser(description="One processed video -> crops -> embeddings -> Qdrant")
    ap.add_argument("--video-stem", default="Normal_Videos_015_x264")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--write-db", action="store_true", help="Actually upsert into forensic_person / forensic_object")
    ap.add_argument("--max-records", type=int, default=None)
    args = ap.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    started = time.time()
    video_stem = args.video_stem
    video_path = find_video(video_stem)
    rows = load_final_tracks(video_stem)
    if args.max_records is not None:
        rows = rows[: args.max_records]

    print("=" * 90)
    print("SINGLE PROCESSED VIDEO INGEST")
    print("=" * 90)
    print("video       :", video_path)
    print("route file  :", PROCESSED_ROOT / video_stem / "final_routed_tracks.json")
    print("records     :", len(rows))
    print("person rows :", sum(str(r["final_db_route"]).lower() == "person" for r in rows))
    print("object rows :", sum(str(r["final_db_route"]).lower() == "object" for r in rows))
    print("write db    :", args.write_db)
    print("=" * 90)

    enriched, dropped = create_crops(video_path, video_stem, rows)
    if not enriched:
        raise RuntimeError("No crops were created")

    cfg = PipelineConfig.load(CONFIG_PATH)
    person_cfg, object_cfg = make_collection_configs(cfg)
    person_store = QdrantStore(person_cfg)
    object_store = QdrantStore(object_cfg)
    if args.write_db:
        person_store.ensure_collection(recreate=False)
        object_store.ensure_collection(recreate=False)

    registry = EmbedderRegistry(cfg)
    router = Router(cfg, registry, input_format="rgb")
    person_dets, person_vecs, object_dets, object_vecs = [], [], [], []
    vector_dims = {}

    try:
        person_label_set = {str(x).lower() for x in cfg.person_labels}
        for start in range(0, len(enriched), args.batch_size):
            batch_rows = enriched[start:start + args.batch_size]
            dets = [row_to_detection(video_path, video_stem, r) for r in batch_rows]
            vec_maps = router.embed(dets)
            if len(vec_maps) != len(dets):
                raise RuntimeError(f"Router count mismatch: {len(dets)} -> {len(vec_maps)}")

            for det, vec_map in zip(dets, vec_maps):
                expected = expected_vector_names(cfg, det.label)
                actual = set(vec_map)
                if actual != expected:
                    raise RuntimeError(
                        f"Router routing mismatch label={det.label}: expected={sorted(expected)}, actual={sorted(actual)}"
                    )
                for name, vec in vec_map.items():
                    vector_dims[name] = int(np.asarray(vec, dtype=np.float32).reshape(-1).size)

                if str(det.label).lower() in person_label_set:
                    person_dets.append(det)
                    person_vecs.append({k: v for k, v in vec_map.items() if k in person_cfg.retrievers})
                else:
                    object_dets.append(det)
                    object_vecs.append({k: v for k, v in vec_map.items() if k in object_cfg.retrievers})

            done = min(start + args.batch_size, len(enriched))
            print(f"[EMBED] {done:,}/{len(enriched):,} | dims={vector_dims}")

        person_upserts = object_upserts = 0
        if args.write_db:
            if person_dets:
                person_upserts = person_store.upsert(person_dets, person_vecs, batch_size=args.batch_size)
            if object_dets:
                object_upserts = object_store.upsert(object_dets, object_vecs, batch_size=args.batch_size)

        summary = {
            "video_stem": video_stem,
            "video_path": str(video_path.resolve()),
            "input_records": len(rows),
            "crops_created": len(enriched),
            "dropped": len(dropped),
            "person_records": len(person_dets),
            "object_records": len(object_dets),
            "person_tracks": len({int(r["long_track_id"]) for r in enriched if str(r["final_db_route"]).lower() == "person"}),
            "object_tracks": len({int(r["long_track_id"]) for r in enriched if str(r["final_db_route"]).lower() == "object"}),
            "vector_dims": vector_dims,
            "write_db": args.write_db,
            "person_collection": person_cfg.collection,
            "object_collection": object_cfg.collection,
            "person_upserts": int(person_upserts),
            "object_upserts": int(object_upserts),
            "elapsed_sec": round(time.time() - started, 3),
        }

        out_dir = INGEST_ROOT / video_stem
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ingest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if dropped:
            (out_dir / "dropped_records.json").write_text(json.dumps(dropped, ensure_ascii=False, indent=2), encoding="utf-8")

        print()
        print("=" * 90)
        print("COMPLETE")
        print("=" * 90)
        for k, v in summary.items():
            print(f"{k:20s}: {v}")
        print("=" * 90)
    finally:
        try:
            registry.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()
