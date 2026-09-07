from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 이미지 파이프라인의 RF-DETR 설정/로더를 그대로 재사용
from detect_rf import (
    load_detect_model,
    _class_name,
    MIN_PERSON_CROP_WIDTH,
    MIN_PERSON_CROP_HEIGHT,
)


VIDEO_ROOT = ROOT / "data" / "videos"
OUT_ROOT = ROOT / "data" / "video_tracks" / "person"
TEMP_ROOT = ROOT / "data" / "video_person_track_temp"

STATE_ROOT = ROOT / "data" / "video_person_track_state_fixed"
STATE_FILE = STATE_ROOT / "completed.json"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def resolve_person_min_size(
    frame_width: int,
    frame_height: int,
    override_width: int | None = None,
    override_height: int | None = None,
) -> tuple[int, int]:
    """
    Keep detect_rf.py's person-size filter as the upper/default policy, but
    relax it for very low-resolution video.

    320x240 -> roughly 16x48
    1920x1080 -> original detect_rf.py minimum (currently imported constants)

    This is a validation default, not a calibrated SOLIDER/Re-ID threshold.
    """
    if override_width is not None:
        min_w = int(override_width)
    else:
        min_w = min(
            int(MIN_PERSON_CROP_WIDTH),
            max(16, int(round(frame_width * 0.05))),
        )

    if override_height is not None:
        min_h = int(override_height)
    else:
        min_h = min(
            int(MIN_PERSON_CROP_HEIGHT),
            max(48, int(round(frame_height * 0.20))),
        )

    if min_w <= 0 or min_h <= 0:
        raise ValueError("person minimum crop size must be >= 1")

    return min_w, min_h


def rel_video(path: Path) -> str:
    return path.relative_to(VIDEO_ROOT).as_posix()


def list_videos() -> list[Path]:
    if not VIDEO_ROOT.exists():
        return []
    return sorted(
        p for p in VIDEO_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


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


def probe_video(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    opened = cap.isOpened()

    fps = float(cap.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0

    frame_ok = False
    if opened:
        ok, frame = cap.read()
        frame_ok = bool(ok and frame is not None and frame.size > 0)

    cap.release()

    return {
        "opened": opened,
        "frame_ok": frame_ok,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": frame_count / fps if fps > 0 else 0.0,
    }


def empty_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=int),
    )


def detect_frame_like_image_pipeline(
    model,
    frame_bgr: np.ndarray,
    temp_frame_path: Path,
    threshold: float,
    min_person_width: int,
    min_person_height: int,
):
    """
    이미지 detect_rf.py와 동일하게 RF-DETR에 '이미지 파일 경로'를 전달한다.

    raw_records:
      RF-DETR 원시 검출 전체 기록.

    person_detections:
      class_name == person 이면서
      현재 영상 해상도에 맞춘 person 최소 crop 크기를 통과한 것만 반환.
    """
    temp_frame_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(temp_frame_path), frame_bgr):
        raise RuntimeError(f"Failed to save temp frame: {temp_frame_path}")

    detections = model.predict(
        str(temp_frame_path),
        threshold=threshold,
    )

    h, w = frame_bgr.shape[:2]

    raw_records = []
    person_boxes = []
    person_confs = []
    person_class_ids = []

    raw_person = 0
    accepted_person = 0
    filtered_small_person = 0

    for i in range(len(detections)):
        class_id = int(detections.class_id[i])
        class_name = _class_name(class_id)
        confidence = float(detections.confidence[i])

        x1, y1, x2, y2 = map(int, detections.xyxy[i])

        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        crop_w = x2 - x1
        crop_h = y2 - y1

        rec = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2],
            "width": crop_w,
            "height": crop_h,
        }

        raw_records.append(rec)

        if class_name.lower() != "person":
            continue

        raw_person += 1

        # 이미지 detect_rf.py와 동일한 person 최소 크기 필터
        if (
            crop_w < min_person_width
            or crop_h < min_person_height
        ):
            filtered_small_person += 1
            rec["person_filter"] = "too_small"
            continue

        accepted_person += 1
        rec["person_filter"] = "accepted"

        person_boxes.append([x1, y1, x2, y2])
        person_confs.append(confidence)
        person_class_ids.append(class_id)

    if not person_boxes:
        person_detections = empty_detections()
    else:
        person_detections = sv.Detections(
            xyxy=np.asarray(person_boxes, dtype=np.float32),
            confidence=np.asarray(person_confs, dtype=np.float32),
            class_id=np.asarray(person_class_ids, dtype=int),
        )

    stats = {
        "raw_person": raw_person,
        "accepted_person": accepted_person,
        "filtered_small_person": filtered_small_person,
    }

    return person_detections, raw_records, stats


def make_tracker(interval_sec: float):
    sample_fps = max(1, int(round(1.0 / interval_sec)))
    try:
        return sv.ByteTrack(frame_rate=sample_fps)
    except TypeError:
        return sv.ByteTrack()


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_video(
    model,
    video_path: Path,
    threshold: float,
    interval_sec: float,
    preview_every: int,
    force: bool,
    min_person_width_override: int | None = None,
    min_person_height_override: int | None = None,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    min_person_width, min_person_height = resolve_person_min_size(
        frame_width=frame_width,
        frame_height=frame_height,
        override_width=min_person_width_override,
        override_height=min_person_height_override,
    )

    frame_step = max(1, int(round(fps * interval_sec)))

    video_out = OUT_ROOT / video_path.stem
    if force and video_out.exists():
        shutil.rmtree(video_out)

    video_out.mkdir(parents=True, exist_ok=True)

    preview_dir = video_out / "preview"
    if preview_every > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    temp_frame_path = TEMP_ROOT / video_path.stem / "current_frame.jpg"
    metadata_path = video_out / "tracks.jsonl"
    summary_path = video_out / "raw_detection_summary.json"

    tracker = make_tracker(interval_sec)

    sampled_frames = 0
    frame_idx = 0

    raw_detection_total = 0
    raw_person_total = 0
    accepted_person_total = 0
    filtered_small_person_total = 0

    saved_person_crops = 0
    track_ids = set()

    raw_class_counts = Counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        sampled_frames += 1

        person_detections, raw_records, stats = detect_frame_like_image_pipeline(
            model=model,
            frame_bgr=frame,
            temp_frame_path=temp_frame_path,
            threshold=threshold,
            min_person_width=min_person_width,
            min_person_height=min_person_height,
        )

        raw_detection_total += len(raw_records)
        raw_person_total += stats["raw_person"]
        accepted_person_total += stats["accepted_person"]
        filtered_small_person_total += stats["filtered_small_person"]

        for rec in raw_records:
            raw_class_counts[rec["class_name"]] += 1

        # 크기 필터를 통과한 person만 ByteTrack으로 전달
        tracked = tracker.update_with_detections(person_detections)

        preview = frame.copy() if preview_every > 0 else None

        if preview is not None:
            # RF-DETR raw class 전체 표시
            for rec in raw_records:
                x1, y1, x2, y2 = rec["bbox"]
                class_name = rec["class_name"]
                confidence = rec["confidence"]

                if class_name.lower() == "person":
                    if rec.get("person_filter") == "too_small":
                        color = (0, 0, 255)
                        label = (
                            f"RAW person FILTERED "
                            f"{confidence:.2f} "
                            f"{rec['width']}x{rec['height']}"
                        )
                    else:
                        color = (0, 255, 255)
                        label = (
                            f"RAW person ACCEPTED "
                            f"{confidence:.2f} "
                            f"{rec['width']}x{rec['height']}"
                        )
                else:
                    color = (255, 180, 0)
                    label = f"RAW {class_name} {confidence:.2f}"

                cv2.rectangle(
                    preview,
                    (x1, y1),
                    (x2, y2),
                    color,
                    1,
                )
                cv2.putText(
                    preview,
                    label,
                    (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        for i in range(len(tracked)):
            if tracked.tracker_id is None:
                continue

            track_id = int(tracked.tracker_id[i])
            x1, y1, x2, y2 = map(int, tracked.xyxy[i])

            h, w = frame.shape[:2]
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue

            crop_h, crop_w = crop.shape[:2]

            # 저장 직전에도 동일 조건 재확인
            if (
                crop_w < min_person_width
                or crop_h < min_person_height
            ):
                continue

            confidence = (
                float(tracked.confidence[i])
                if tracked.confidence is not None
                else 1.0
            )

            track_dir = video_out / f"track_{track_id:04d}"
            track_dir.mkdir(parents=True, exist_ok=True)

            crop_path = (
                track_dir
                / f"frame_{frame_idx:08d}_{confidence:.4f}.jpg"
            )

            if not cv2.imwrite(str(crop_path), crop):
                continue

            record = {
                "video": rel_video(video_path),
                "video_path": str(video_path.resolve()),
                "video_id": video_path.stem,
                "media_type": "video",
                "scope": "person",
                "class_name": "person",
                "track_id": track_id,
                "frame_idx": frame_idx,
                "timestamp_sec": frame_idx / fps,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
                "bbox_space": "frame",
                "crop_path": str(crop_path.resolve()),
                "width": crop_w,
                "height": crop_h,
            }

            append_jsonl(metadata_path, record)

            saved_person_crops += 1
            track_ids.add(track_id)

            if preview is not None:
                cv2.rectangle(
                    preview,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    preview,
                    f"TRACK person T{track_id} {confidence:.2f}",
                    (x1, min(h - 5, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        if (
            preview is not None
            and preview_every > 0
            and sampled_frames % preview_every == 0
        ):
            cv2.imwrite(
                str(preview_dir / f"frame_{frame_idx:08d}.jpg"),
                preview,
            )

        if sampled_frames % 50 == 0:
            print(
                f"    sampled={sampled_frames:,} "
                f"raw={raw_detection_total:,} "
                f"raw_person={raw_person_total:,} "
                f"accepted_person={accepted_person_total:,} "
                f"small_filtered={filtered_small_person_total:,} "
                f"crops={saved_person_crops:,}",
                end="\r",
            )

        frame_idx += 1

    cap.release()
    print()

    metadata_rows = 0
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata_rows = sum(1 for line in f if line.strip())

    summary = {
        "video": rel_video(video_path),
        "sampled_frames": sampled_frames,
        "raw_detection_total": raw_detection_total,
        "raw_person_detections": raw_person_total,
        "accepted_person_detections": accepted_person_total,
        "filtered_small_person_detections": filtered_small_person_total,
        "person_min_width": min_person_width,
        "person_min_height": min_person_height,
        "person_min_width_base": int(MIN_PERSON_CROP_WIDTH),
        "person_min_height_base": int(MIN_PERSON_CROP_HEIGHT),
        "raw_class_counts": dict(raw_class_counts.most_common()),
        "saved_person_crops": saved_person_crops,
        "unique_person_tracks": len(track_ids),
        "metadata_rows": metadata_rows,
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return summary, summary_path, metadata_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Video person tracker using detect_rf.py RF-DETR "
            "+ same person crop size filter"
        )
    )

    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument(
        "--video",
        default=None,
        help="Process one exact video stem or filename, e.g. Normal_Videos_010_x264",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-every", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--interval-sec", type=float, default=0.2)
    parser.add_argument(
        "--min-person-width",
        type=int,
        default=None,
        help="Manual person crop minimum width. Default: adaptive low-res policy.",
    )
    parser.add_argument(
        "--min-person-height",
        type=int,
        default=None,
        help="Manual person crop minimum height. Default: adaptive low-res policy.",
    )

    args = parser.parse_args()

    if args.interval_sec <= 0:
        raise ValueError("--interval-sec must be > 0")

    if args.preview_every < 0:
        raise ValueError("--preview-every must be >= 0")

    if args.max_videos is not None and args.max_videos <= 0:
        raise ValueError("--max-videos must be >= 1")

    if args.min_person_width is not None and args.min_person_width <= 0:
        raise ValueError("--min-person-width must be >= 1")

    if args.min_person_height is not None and args.min_person_height <= 0:
        raise ValueError("--min-person-height must be >= 1")

    videos = list_videos()

    if args.video:
        wanted = str(args.video).strip().lower()
        videos = [
            p for p in videos
            if p.stem.lower() == wanted or p.name.lower() == wanted
        ]
        if not videos:
            raise RuntimeError(f"Video not found under {VIDEO_ROOT}: {args.video}")

    if args.max_videos is not None:
        videos = videos[:args.max_videos]

    print("=" * 76)
    print("VIDEO PERSON TRACK FIXED")
    print("=" * 76)
    print("input root      :", VIDEO_ROOT)
    print("output root     :", OUT_ROOT)
    print("videos          :", len(videos))
    print("threshold       :", args.threshold)
    print("interval sec    :", args.interval_sec)
    print("preview every   :", args.preview_every)
    print(
        "base person min :",
        f"{MIN_PERSON_CROP_WIDTH}x{MIN_PERSON_CROP_HEIGHT}",
    )
    print(
        "size policy     :",
        "manual override" if (
            args.min_person_width is not None
            or args.min_person_height is not None
        ) else "adaptive for low-resolution video",
    )

    if not videos:
        raise RuntimeError(f"No videos under {VIDEO_ROOT}")

    print("\n[CHECK 1] VIDEO DECODE")

    for video_path in videos:
        info = probe_video(video_path)

        status = (
            "PASS"
            if info["opened"] and info["frame_ok"]
            else "FAIL"
        )

        print(
            f"  [{status}] {rel_video(video_path)} | "
            f"{info['width']}x{info['height']} "
            f"fps={info['fps']:.3f} "
            f"frames={info['frame_count']} "
            f"duration={info['duration_sec']:.2f}s"
        )

        if status == "FAIL":
            raise RuntimeError(f"Decode failed: {video_path}")

    if args.dry_run:
        print("\nDRY RUN PASS")
        return

    done = set() if args.force else load_state()

    pending = [
        p for p in videos
        if rel_video(p) not in done
    ]

    print("\npending         :", len(pending))

    if not pending:
        print("Nothing to do.")
        return

    # 이미지 파이프라인과 같은 RF-DETR loader
    model = load_detect_model()

    started = time.time()

    print(
        "\n[CHECK 2] RF-DETR RAW -> PERSON SIZE FILTER -> BYTETRACK"
    )

    for index, video_path in enumerate(pending, start=1):
        print(f"\n[{index}/{len(pending)}] {video_path.name}")

        info = probe_video(video_path)
        effective_min_w, effective_min_h = resolve_person_min_size(
            frame_width=info["width"],
            frame_height=info["height"],
            override_width=args.min_person_width,
            override_height=args.min_person_height,
        )
        print(
            "  effective person min :",
            f"{effective_min_w}x{effective_min_h}",
        )

        summary, summary_path, metadata_path = process_video(
            model=model,
            video_path=video_path,
            threshold=args.threshold,
            interval_sec=args.interval_sec,
            preview_every=args.preview_every,
            force=args.force,
            min_person_width_override=args.min_person_width,
            min_person_height_override=args.min_person_height,
        )

        print("  sampled frames         :", summary["sampled_frames"])
        print("  raw detections         :", summary["raw_detection_total"])
        print("  raw person detections  :", summary["raw_person_detections"])
        print("  accepted person        :", summary["accepted_person_detections"])
        print("  small person filtered  :", summary["filtered_small_person_detections"])
        print("  raw classes            :", summary["raw_class_counts"])
        print("  person crops           :", summary["saved_person_crops"])
        print("  unique person tracks   :", summary["unique_person_tracks"])
        print("  metadata rows          :", summary["metadata_rows"])
        print("  raw summary            :", summary_path)
        print("  metadata               :", metadata_path)

        if summary["metadata_rows"] != summary["saved_person_crops"]:
            raise RuntimeError("person crop count != metadata rows")

        if summary["accepted_person_detections"] == 0:
            if summary["saved_person_crops"] != 0:
                raise RuntimeError(
                    "accepted person=0 but person crop was created"
                )
            print(
                "  [PASS] accepted person=0 -> person crop=0"
            )
        else:
            print(
                "  [CHECK] accepted person detection exists"
            )

        done.add(rel_video(video_path))
        save_state(done)

        print(
            "  elapsed                :",
            f"{(time.time() - started) / 60:.1f} min",
        )


if __name__ == "__main__":
    main()