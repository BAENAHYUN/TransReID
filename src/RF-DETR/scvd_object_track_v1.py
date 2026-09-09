from __future__ import annotations

import argparse
import json
import re
import shutil
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


# ============================================================
# Adaptive / content-aware frame sampling
# ============================================================

def _motion_thumb(frame: np.ndarray) -> np.ndarray:
    """
    Very cheap visual-change representation.
    RF-DETR is NOT run here. This is only used to decide whether
    the current frame should be sent to RF-DETR.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)


def _motion_score(prev_thumb: np.ndarray | None, curr_thumb: np.ndarray) -> float:
    if prev_thumb is None:
        return 1.0
    diff = cv2.absdiff(prev_thumb, curr_thumb)
    return float(diff.mean() / 255.0)


def adaptive_frames(
    cap: cv2.VideoCapture,
    fps: float,
    *,
    low_motion_fps: float = 1.0,
    medium_motion_fps: float = 3.0,
    high_motion_fps: float = 10.0,
    probe_fps: float = 10.0,
    low_threshold: float = 0.025,
    high_threshold: float = 0.080,
    scene_cut_threshold: float = 0.180,
    ema_alpha: float = 0.60,
):
    """
    Yield only frames that should be processed by RF-DETR.

    Quiet scene       -> ~1 FPS
    Moderate change   -> ~3 FPS
    Fast/large change -> ~10 FPS
    Scene cut         -> immediate sample

    We still decode the video sequentially, but expensive RF-DETR inference
    is performed only on selected frames.
    """
    probe_step = max(1, int(round(fps / max(probe_fps, 0.1))))

    frame_idx = 0
    prev_thumb = None
    ema_change = 0.0
    last_sample_time = -1e9

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Change estimation itself is limited to probe_fps.
        if frame_idx % probe_step != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps
        thumb = _motion_thumb(frame)
        raw_change = _motion_score(prev_thumb, thumb)

        if prev_thumb is None:
            ema_change = raw_change
        else:
            ema_change = (
                ema_alpha * raw_change
                + (1.0 - ema_alpha) * ema_change
            )

        prev_thumb = thumb

        scene_cut = raw_change >= scene_cut_threshold

        if ema_change < low_threshold:
            target_fps = low_motion_fps
            motion_level = "LOW"
        elif ema_change < high_threshold:
            target_fps = medium_motion_fps
            motion_level = "MID"
        else:
            target_fps = high_motion_fps
            motion_level = "HIGH"

        min_gap = 1.0 / max(target_fps, 0.1)
        should_sample = (
            last_sample_time < 0
            or scene_cut
            or (timestamp - last_sample_time) >= (min_gap - 1e-9)
        )

        if should_sample:
            last_sample_time = timestamp
            yield {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "frame": frame,
                "change": raw_change,
                "ema_change": ema_change,
                "motion_level": motion_level,
                "target_fps": target_fps,
                "scene_cut": scene_cut,
            }

        frame_idx += 1


def safe_label(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", name.strip().replace(" ", "_"))


def load_state() -> set[str]:
    if not STATE_FILE.exists():
        return set()
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return set(data.get("completed", []))
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

        # Person is handled by scvd_person_track_v2.py
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


def _empty_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=int),
    )


def process_video(
    model,
    video_path: Path,
    threshold: float,
    min_width: int,
    min_height: int,
    *,
    low_motion_fps: float,
    medium_motion_fps: float,
    high_motion_fps: float,
    probe_fps: float,
    low_threshold: float,
    high_threshold: float,
    scene_cut_threshold: float,
    clean_output: bool,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    video_out = OUT_ROOT / video_path.stem
    if clean_output and video_out.exists():
        shutil.rmtree(video_out)
    video_out.mkdir(parents=True, exist_ok=True)

    trackers: dict[int, sv.ByteTrack] = {}

    sampled = 0
    saved = 0
    level_counts = {"LOW": 0, "MID": 0, "HIGH": 0}

    print(f"  fps={fps:.2f} frames={total_frames:,} duration={duration:.1f}s")

    for sample in adaptive_frames(
        cap,
        fps,
        low_motion_fps=low_motion_fps,
        medium_motion_fps=medium_motion_fps,
        high_motion_fps=high_motion_fps,
        probe_fps=probe_fps,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        scene_cut_threshold=scene_cut_threshold,
    ):
        sampled += 1
        frame_idx = int(sample["frame_idx"])
        frame = sample["frame"]
        level = str(sample["motion_level"])
        level_counts[level] += 1

        detected = detect_objects(model, frame, threshold)
        by_class: dict[int, list[dict]] = {}

        for d in detected:
            by_class.setdefault(int(d["class_id"]), []).append(d)

        # Update classes that were detected in this sample.
        updated_classes = set()

        for class_id, class_dets in by_class.items():
            if class_id not in trackers:
                trackers[class_id] = sv.ByteTrack()

            xyxy = np.asarray([d["bbox"] for d in class_dets], dtype=np.float32)
            confs = np.asarray([d["confidence"] for d in class_dets], dtype=np.float32)
            cids = np.asarray([d["class_id"] for d in class_dets], dtype=int)

            tracked = trackers[class_id].update_with_detections(
                sv.Detections(
                    xyxy=xyxy,
                    confidence=confs,
                    class_id=cids,
                )
            )
            updated_classes.add(class_id)

            label = safe_label(str(COCO_CLASSES[class_id]))

            for i in range(len(tracked)):
                tid = (
                    int(tracked.tracker_id[i])
                    if tracked.tracker_id is not None
                    else i
                )
                x1, y1, x2, y2 = map(int, tracked.xyxy[i])

                x1 = max(0, min(frame.shape[1], x1))
                y1 = max(0, min(frame.shape[0], y1))
                x2 = max(0, min(frame.shape[1], x2))
                y2 = max(0, min(frame.shape[0], y2))

                crop = frame[y1:y2, x1:x2]
                if crop is None or crop.size == 0:
                    continue

                ch, cw = crop.shape[:2]
                if cw < min_width or ch < min_height:
                    continue

                conf = (
                    float(tracked.confidence[i])
                    if tracked.confidence is not None
                    else 1.0
                )

                track_dir = video_out / f"{label}_{tid:04d}"
                track_dir.mkdir(parents=True, exist_ok=True)

                out_path = track_dir / f"frame_{frame_idx:08d}_{conf:.4f}.jpg"
                if cv2.imwrite(str(out_path), crop):
                    saved += 1

        # Age trackers for classes missing in the current sampled frame.
        for class_id, tracker in trackers.items():
            if class_id not in updated_classes:
                tracker.update_with_detections(_empty_detections())

        if sampled % 50 == 0:
            print(
                f"    sampled={sampled:,} saved={saved:,} "
                f"motion={level} change={sample['ema_change']:.4f} "
                f"target={sample['target_fps']:.1f}fps",
                end="\r",
            )

    cap.release()
    print()

    return sampled, saved, level_counts


def main():
    ap = argparse.ArgumentParser(
        description="SCVD RF-DETR object track/crop builder with adaptive frame sampling"
    )
    ap.add_argument("--threshold", type=float, default=0.35)

    # Kept only so old commands do not fail.
    # Adaptive sampling uses the 1/3/10 FPS options below instead.
    ap.add_argument(
        "--interval-sec",
        type=float,
        default=None,
        help="Legacy option. Ignored when adaptive sampling is enabled.",
    )

    ap.add_argument("--min-width", type=int, default=24)
    ap.add_argument("--min-height", type=int, default=24)
    ap.add_argument("--max-videos", type=int)
    ap.add_argument("--force", action="store_true")

    ap.add_argument("--low-motion-fps", type=float, default=1.0)
    ap.add_argument("--medium-motion-fps", type=float, default=3.0)
    ap.add_argument("--high-motion-fps", type=float, default=10.0)
    ap.add_argument("--probe-fps", type=float, default=10.0)

    ap.add_argument("--low-change", type=float, default=0.025)
    ap.add_argument("--high-change", type=float, default=0.080)
    ap.add_argument("--scene-cut", type=float, default=0.180)

    args = ap.parse_args()

    videos = list_videos()
    done = set() if args.force else load_state()
    pending = [p for p in videos if str(p.resolve()) not in done]

    if args.max_videos:
        pending = pending[:args.max_videos]

    print("=" * 76)
    print("SCVD OBJECT TRACK V1 - RF-DETR + ADAPTIVE SAMPLING")
    print("=" * 76)
    print("videos       :", len(videos))
    print("pending      :", len(pending))
    print("output root  :", OUT_ROOT)
    print("detector     : RF-DETR Medium")
    print("classes      : all COCO classes except person")
    print(
        "sampling     : "
        f"LOW={args.low_motion_fps} / "
        f"MID={args.medium_motion_fps} / "
        f"HIGH={args.high_motion_fps} FPS"
    )

    if not pending:
        print("Nothing to do.")
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    model = load_model()

    total_saved = 0
    started = time.time()

    for idx, video in enumerate(pending, 1):
        print(f"\n[{idx}/{len(pending)}] {video.name}")

        sampled, saved, levels = process_video(
            model,
            video,
            args.threshold,
            args.min_width,
            args.min_height,
            low_motion_fps=args.low_motion_fps,
            medium_motion_fps=args.medium_motion_fps,
            high_motion_fps=args.high_motion_fps,
            probe_fps=args.probe_fps,
            low_threshold=args.low_change,
            high_threshold=args.high_change,
            scene_cut_threshold=args.scene_cut,
            clean_output=args.force,
        )

        total_saved += saved
        done.add(str(video.resolve()))
        save_state(done)

        print(
            f"  sampled={sampled:,} crops={saved:,} "
            f"levels={levels} "
            f"total={total_saved:,} "
            f"elapsed={(time.time()-started)/60:.1f}m"
        )

    print("\nOBJECT CROP BUILD COMPLETE")
    print("new crops:", total_saved)
    print("output   :", OUT_ROOT)


if __name__ == "__main__":
    main()
