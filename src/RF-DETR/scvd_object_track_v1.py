from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

ROOT = Path(__file__).resolve().parents[2]
SCVD_ROOT = ROOT / "data" / "scvd" / "SCVD_converted"
OUT_ROOT = ROOT / "data" / "scvd_object_tracks_v1"
STATE_ROOT = ROOT / "data" / "scvd_object_track_state_v1"
STATE_FILE = STATE_ROOT / "completed.json"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def safe_label(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", name.strip().replace(" ", "_"))


def load_state() -> set[str]:
    if not STATE_FILE.exists():
        return set()
    try:
        return set(json.loads(STATE_FILE.read_text(encoding="utf-8")).get("completed", []))
    except Exception:
        return set()


def save_state(done: set[str]) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps({"completed": sorted(done)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_videos() -> list[Path]:
    return sorted(
        p for p in SCVD_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def load_model():
    print("[*] Loading RF-DETR Medium...")
    model = RFDETRMedium()
    try:
        model.inference(compile=False, dtype="float16")
    except Exception:
        try:
            model.inference(compile=False)
        except Exception:
            pass
    print("[+] RF-DETR ready")
    return model


def detect_objects(model, frame: np.ndarray, threshold: float):
    dets = model.predict(frame, threshold=threshold)
    out = []
    h, w = frame.shape[:2]

    for i in range(len(dets)):
        class_id = int(dets.class_id[i])
        class_name = str(COCO_CLASSES[class_id])
        if class_name.lower() == "person":
            continue

        conf = float(dets.confidence[i])
        x1, y1, x2, y2 = map(int, dets.xyxy[i])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        out.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": conf,
            "bbox": (x1, y1, x2, y2),
        })
    return out


def process_video(model, video_path: Path, threshold: float, interval_sec: float,
                  min_width: int, min_height: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    frame_step = max(1, int(round(fps * interval_sec)))

    trackers = {}
    sampled = 0
    saved = 0
    frame_idx = 0
    video_out = OUT_ROOT / video_path.stem

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        sampled += 1
        detected = detect_objects(model, frame, threshold)
        by_class = {}
        for d in detected:
            by_class.setdefault(d["class_id"], []).append(d)

        for class_id, class_dets in by_class.items():
            if class_id not in trackers:
                trackers[class_id] = sv.ByteTrack()

            xyxy = np.asarray([d["bbox"] for d in class_dets], dtype=np.float32)
            confs = np.asarray([d["confidence"] for d in class_dets], dtype=np.float32)
            cids = np.asarray([d["class_id"] for d in class_dets], dtype=int)

            tracked = trackers[class_id].update_with_detections(
                sv.Detections(xyxy=xyxy, confidence=confs, class_id=cids)
            )

            label = safe_label(str(COCO_CLASSES[class_id]))

            for i in range(len(tracked)):
                tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else i
                x1, y1, x2, y2 = map(int, tracked.xyxy[i])
                crop = frame[y1:y2, x1:x2]
                if crop is None or crop.size == 0:
                    continue

                ch, cw = crop.shape[:2]
                if cw < min_width or ch < min_height:
                    continue

                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
                track_dir = video_out / f"{label}_{tid:04d}"
                track_dir.mkdir(parents=True, exist_ok=True)

                out_path = track_dir / f"frame_{frame_idx:08d}_{conf:.4f}.jpg"
                if cv2.imwrite(str(out_path), crop):
                    saved += 1

        if sampled % 50 == 0:
            print(f"    sampled={sampled:,} saved={saved:,}", end="\r")

        frame_idx += 1

    cap.release()
    print()
    return sampled, saved


def main():
    ap = argparse.ArgumentParser(description="SCVD RF-DETR object track/crop builder")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--interval-sec", type=float, default=1.0)
    ap.add_argument("--min-width", type=int, default=24)
    ap.add_argument("--min-height", type=int, default=24)
    ap.add_argument("--max-videos", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    videos = list_videos()
    if args.max_videos:
        videos = videos[:args.max_videos]

    print("=" * 76)
    print("SCVD OBJECT TRACK/CROP BUILDER")
    print("=" * 76)
    print("videos       :", len(videos))
    print("output root  :", OUT_ROOT)
    print("classes      : all COCO classes except person")
    print("threshold    :", args.threshold)
    print("interval sec :", args.interval_sec)

    if not videos:
        raise RuntimeError(f"No videos under {SCVD_ROOT}")

    done = set() if args.force else load_state()
    pending = [p for p in videos if str(p.resolve()) not in done]
    print("pending      :", len(pending))

    if not pending:
        print("Nothing to do.")
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    model = load_model()
    total_saved = 0
    started = time.time()

    for idx, video in enumerate(pending, 1):
        print(f"\n[{idx}/{len(pending)}] {video.name}")
        sampled, saved = process_video(
            model, video, args.threshold, args.interval_sec,
            args.min_width, args.min_height,
        )
        total_saved += saved
        done.add(str(video.resolve()))
        save_state(done)
        print(
            f"  sampled={sampled:,} crops={saved:,} "
            f"total={total_saved:,} elapsed={(time.time()-started)/60:.1f}m"
        )

    print("\nOBJECT CROP BUILD COMPLETE")
    print("new crops:", total_saved)
    print("output   :", OUT_ROOT)


if __name__ == "__main__":
    main()
