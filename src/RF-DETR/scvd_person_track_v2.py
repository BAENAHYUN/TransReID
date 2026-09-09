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

ROOT_DIR = Path(__file__).resolve().parents[2]
SCVD_ROOT = ROOT_DIR / "data" / "scvd" / "SCVD_converted"
TRAIN_ROOT = SCVD_ROOT / "Train"
TEST_ROOT = SCVD_ROOT / "Test"

# Keep the existing downstream-compatible paths.
OUT_ROOT = ROOT_DIR / "data" / "scvd_person_tracks_v1"
STATE_ROOT = ROOT_DIR / "data" / "scvd_ingest_state_v2"
MANIFEST = STATE_ROOT / "person_track_manifest.jsonl"

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


# ============================================================
# Dataset / state
# ============================================================

def get_videos(split: str) -> list[Path]:
    roots: list[Path] = []

    if split in ("train", "all") and TRAIN_ROOT.exists():
        roots.append(TRAIN_ROOT)
    if split in ("test", "all") and TEST_ROOT.exists():
        roots.append(TEST_ROOT)

    videos: list[Path] = []
    for root in roots:
        videos.extend(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )
    return sorted(videos)


def rel(vp: Path) -> str:
    return vp.relative_to(SCVD_ROOT).as_posix()


def detect_split(vp: Path) -> str:
    try:
        parts = vp.relative_to(SCVD_ROOT).parts
        return parts[0] if parts else "Unknown"
    except Exception:
        return "Unknown"


def done_set() -> set[str]:
    done: set[str] = set()
    if not MANIFEST.exists():
        return done

    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            if rec.get("status") == "ok" and rec.get("video"):
                done.add(str(rec["video"]))
        except Exception:
            pass
    return done


def append_manifest(vp: Path, *, sampled: int, crops: int) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    rec = {
        "video": rel(vp),
        "status": "ok",
        "source_dataset": "SCVD",
        "split": detect_split(vp),
        "sampling": "adaptive_1_3_10fps",
        "sampled_frames": int(sampled),
        "person_crops": int(crops),
    }
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# RF-DETR only
# ============================================================

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


def detect_persons(model, frame: np.ndarray, threshold: float) -> list[dict]:
    dets = model.predict(frame, threshold=threshold)
    h, w = frame.shape[:2]
    out: list[dict] = []

    for i in range(len(dets)):
        class_id = int(dets.class_id[i])
        class_name = str(COCO_CLASSES[class_id])

        if class_name.lower() != "person":
            continue

        conf = float(dets.confidence[i])
        x1, y1, x2, y2 = map(int, dets.xyxy[i])

        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        out.append({
            "class_id": class_id,
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
    *,
    threshold: float,
    min_width: int,
    min_height: int,
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

    # Detection/Tracking only. No CLIP-ReID, no InsightFace, no SigLIP.
    tracker = sv.ByteTrack()

    sampled = 0
    saved = 0
    detected_persons = 0
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

        persons = detect_persons(model, frame, threshold)
        detected_persons += len(persons)

        if persons:
            xyxy = np.asarray([d["bbox"] for d in persons], dtype=np.float32)
            confs = np.asarray([d["confidence"] for d in persons], dtype=np.float32)
            cids = np.asarray([d["class_id"] for d in persons], dtype=int)

            tracked = tracker.update_with_detections(
                sv.Detections(
                    xyxy=xyxy,
                    confidence=confs,
                    class_id=cids,
                )
            )
        else:
            tracked = tracker.update_with_detections(_empty_detections())

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

            track_dir = video_out / f"track_{tid:04d}"
            track_dir.mkdir(parents=True, exist_ok=True)

            # Keep frame number + person token compatible with existing DB parser.
            out_path = track_dir / f"frame_{frame_idx:08d}_person_{i+1:02d}.jpg"
            if cv2.imwrite(str(out_path), crop):
                saved += 1

        if sampled % 50 == 0:
            print(
                f"    sampled={sampled:,} crops={saved:,} "
                f"motion={level} change={sample['ema_change']:.4f} "
                f"target={sample['target_fps']:.1f}fps",
                end="\r",
            )

    cap.release()
    print()

    return {
        "sampled": sampled,
        "saved": saved,
        "detected_persons": detected_persons,
        "level_counts": level_counts,
    }


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="SCVD RF-DETR person track/crop builder with adaptive frame sampling"
    )
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--split", choices=["train", "test", "all"], default="all")
    ap.add_argument("--force", action="store_true")

    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--min-width", type=int, default=20)
    ap.add_argument("--min-height", type=int, default=40)

    ap.add_argument("--low-motion-fps", type=float, default=1.0)
    ap.add_argument("--medium-motion-fps", type=float, default=3.0)
    ap.add_argument("--high-motion-fps", type=float, default=10.0)
    ap.add_argument("--probe-fps", type=float, default=10.0)

    ap.add_argument("--low-change", type=float, default=0.025)
    ap.add_argument("--high-change", type=float, default=0.080)
    ap.add_argument("--scene-cut", type=float, default=0.180)

    args = ap.parse_args()

    videos = get_videos(args.split)
    done = done_set()

    if not args.force:
        videos = [p for p in videos if rel(p) not in done]

    if args.max_videos:
        videos = videos[:args.max_videos]

    print("=" * 76)
    print("SCVD PERSON TRACK V2 - RF-DETR + ADAPTIVE SAMPLING")
    print("=" * 76)
    print("videos       :", len(videos))
    print("output root  :", OUT_ROOT)
    print("detector     : RF-DETR Medium")
    print("tracking     : ByteTrack")
    print("CLIP-ReID    : NOT USED")
    print("InsightFace  : NOT USED")
    print("SigLIP       : NOT USED HERE")
    print(
        "sampling     : "
        f"LOW={args.low_motion_fps} / "
        f"MID={args.medium_motion_fps} / "
        f"HIGH={args.high_motion_fps} FPS"
    )

    if not videos:
        print("Nothing to do.")
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    model = load_model()

    started = time.time()
    total_saved = 0

    for idx, video in enumerate(videos, 1):
        print(f"\n[{idx}/{len(videos)}] {rel(video)}")

        result = process_video(
            model,
            video,
            threshold=args.threshold,
            min_width=args.min_width,
            min_height=args.min_height,
            low_motion_fps=args.low_motion_fps,
            medium_motion_fps=args.medium_motion_fps,
            high_motion_fps=args.high_motion_fps,
            probe_fps=args.probe_fps,
            low_threshold=args.low_change,
            high_threshold=args.high_change,
            scene_cut_threshold=args.scene_cut,
            clean_output=args.force,
        )

        total_saved += int(result["saved"])
        append_manifest(
            video,
            sampled=int(result["sampled"]),
            crops=int(result["saved"]),
        )

        print(
            f"  sampled={result['sampled']:,} "
            f"crops={result['saved']:,} "
            f"levels={result['level_counts']} "
            f"total_crops={total_saved:,} "
            f"elapsed={(time.time()-started)/60:.1f}m"
        )

    print("\nSCVD PERSON TRACK V2 COMPLETE")
    print("output   :", OUT_ROOT)
    print("manifest :", MANIFEST)


if __name__ == "__main__":
    main()
