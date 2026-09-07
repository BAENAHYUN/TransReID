from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 이미지 파이프라인의 RF-DETR 로더 / COCO class mapping 재사용
from detect_rf import load_detect_model, _class_name


VIDEO_ROOT = ROOT / "data" / "videos"
OUT_ROOT = ROOT / "data" / "video_tracks" / "object"
TEMP_ROOT = ROOT / "data" / "video_object_track_temp"

STATE_ROOT = ROOT / "data" / "video_object_track_state_v2"
STATE_FILE = STATE_ROOT / "completed.json"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


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
        json.dumps(
            {"completed": sorted(done)},
            ensure_ascii=False,
            indent=2,
        ),
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


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return inter / union


def deduplicate_class_agnostic(
    records: list[dict],
    iou_threshold: float,
) -> tuple[list[dict], list[dict]]:
    """
    RF-DETR이 같은 물체/같은 bbox에 dog + cow처럼 여러 class를 동시에
    예측한 경우 class와 무관하게 겹침 bbox를 하나로 정리한다.

    confidence가 높은 detection을 우선 유지한다.
    suppressed record에는 duplicate_of / object_filter를 기록한다.
    """
    if not records:
        return [], []

    ordered = sorted(
        records,
        key=lambda r: float(r["confidence"]),
        reverse=True,
    )

    kept: list[dict] = []
    suppressed: list[dict] = []

    for rec in ordered:
        duplicate_target = None

        for kept_rec in kept:
            if iou_xyxy(rec["bbox"], kept_rec["bbox"]) >= iou_threshold:
                duplicate_target = kept_rec
                break

        if duplicate_target is None:
            rec["object_filter"] = "accepted"
            kept.append(rec)
            continue

        rec["object_filter"] = "duplicate_suppressed"
        rec["duplicate_of"] = {
            "class_name": duplicate_target["class_name"],
            "confidence": duplicate_target["confidence"],
            "bbox": duplicate_target["bbox"],
        }
        suppressed.append(rec)

    return kept, suppressed


def make_detections(records: list[dict]) -> sv.Detections:
    if not records:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=int),
        )

    return sv.Detections(
        xyxy=np.asarray(
            [r["bbox"] for r in records],
            dtype=np.float32,
        ),
        confidence=np.asarray(
            [r["confidence"] for r in records],
            dtype=np.float32,
        ),
        class_id=np.asarray(
            [r["class_id"] for r in records],
            dtype=int,
        ),
    )


def detect_frame_like_image_pipeline(
    model,
    frame_bgr: np.ndarray,
    temp_frame_path: Path,
    threshold: float,
    min_width: int,
    min_height: int,
    duplicate_iou: float,
):
    """
    1) 이미지 detect_rf.py와 동일하게 RF-DETR에 이미지 파일 경로 전달
    2) person 제외
    3) 너무 작은 non-person object 제외
    4) class-agnostic IoU dedup
       - 동일 bbox의 dog/cow/horse 중 confidence 높은 1개만 tracking에 전달
    """
    temp_frame_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(temp_frame_path), frame_bgr):
        raise RuntimeError(f"Failed to save temp frame: {temp_frame_path}")

    detections = model.predict(
        str(temp_frame_path),
        threshold=threshold,
    )

    h, w = frame_bgr.shape[:2]

    raw_records: list[dict] = []
    candidate_objects: list[dict] = []

    raw_object_count = 0
    filtered_small_object_count = 0
    excluded_person_count = 0

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

        # person은 object DB / object tracking에서 완전히 제외
        if class_name.lower() == "person":
            excluded_person_count += 1
            rec["object_filter"] = "person_excluded"
            continue

        raw_object_count += 1

        if crop_w < min_width or crop_h < min_height:
            filtered_small_object_count += 1
            rec["object_filter"] = "too_small"
            continue

        candidate_objects.append(rec)

    accepted_objects, suppressed_duplicates = deduplicate_class_agnostic(
        candidate_objects,
        iou_threshold=duplicate_iou,
    )

    stats = {
        "raw_object": raw_object_count,
        "candidate_object": len(candidate_objects),
        "accepted_object": len(accepted_objects),
        "filtered_small_object": filtered_small_object_count,
        "duplicate_suppressed": len(suppressed_duplicates),
        "excluded_person": excluded_person_count,
    }

    return raw_records, accepted_objects, stats


def make_tracker(interval_sec: float):
    sample_fps = max(1, int(round(1.0 / interval_sec)))

    try:
        return sv.ByteTrack(frame_rate=sample_fps)
    except TypeError:
        return sv.ByteTrack()


def choose_track_class(
    class_counts: Counter,
    class_conf_sums: dict[str, float],
) -> str | None:
    """
    최종 class 결정:
      1순위 confidence 누적합
      2순위 관측 횟수
      3순위 class 이름(결정론적 tie-break)
    """
    if not class_counts:
        return None

    labels = list(class_counts.keys())

    return max(
        labels,
        key=lambda name: (
            float(class_conf_sums.get(name, 0.0)),
            int(class_counts[name]),
            name,
        ),
    )


def process_video(
    model,
    video_path: Path,
    threshold: float,
    interval_sec: float,
    min_width: int,
    min_height: int,
    duplicate_iou: float,
    preview_every: int,
    force: bool,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

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
    track_summary_path = video_out / "track_class_summary.json"
    raw_summary_path = video_out / "raw_detection_summary.json"

    tracker = make_tracker(interval_sec)

    sampled_frames = 0
    frame_idx = 0

    raw_detection_total = 0
    raw_object_total = 0
    candidate_object_total = 0
    accepted_object_total = 0
    filtered_small_object_total = 0
    duplicate_suppressed_total = 0
    excluded_person_total = 0

    saved_object_crops = 0
    track_ids = set()

    raw_class_counts = Counter()
    accepted_class_counts = Counter()

    # track별 class voting 정보
    track_class_counts: dict[int, Counter] = defaultdict(Counter)
    track_class_conf_sums: dict[int, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    # 최종 class를 나중에 붙이기 위해 metadata는 메모리에 모았다가 마지막에 기록
    metadata_records: list[dict] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        sampled_frames += 1

        raw_records, accepted_objects, stats = detect_frame_like_image_pipeline(
            model=model,
            frame_bgr=frame,
            temp_frame_path=temp_frame_path,
            threshold=threshold,
            min_width=min_width,
            min_height=min_height,
            duplicate_iou=duplicate_iou,
        )

        raw_detection_total += len(raw_records)
        raw_object_total += stats["raw_object"]
        candidate_object_total += stats["candidate_object"]
        accepted_object_total += stats["accepted_object"]
        filtered_small_object_total += stats["filtered_small_object"]
        duplicate_suppressed_total += stats["duplicate_suppressed"]
        excluded_person_total += stats["excluded_person"]

        for rec in raw_records:
            raw_class_counts[rec["class_name"]] += 1

        for rec in accepted_objects:
            accepted_class_counts[rec["class_name"]] += 1

        # 중요: class별 tracker가 아니라 모든 non-person object를 단일 ByteTrack에 전달
        detections = make_detections(accepted_objects)
        tracked = tracker.update_with_detections(detections)

        preview = frame.copy() if preview_every > 0 else None

        if preview is not None:
            for rec in raw_records:
                x1, y1, x2, y2 = rec["bbox"]
                class_name = rec["class_name"]
                confidence = rec["confidence"]
                state = rec.get("object_filter")

                if state == "person_excluded":
                    color = (0, 0, 255)
                    label = f"RAW person EXCLUDED {confidence:.2f}"

                elif state == "too_small":
                    color = (0, 165, 255)
                    label = (
                        f"RAW {class_name} SMALL "
                        f"{confidence:.2f} "
                        f"{rec['width']}x{rec['height']}"
                    )

                elif state == "duplicate_suppressed":
                    color = (255, 0, 255)
                    label = (
                        f"RAW {class_name} DUP "
                        f"{confidence:.2f}"
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
                    0.40,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        for i in range(len(tracked)):
            if tracked.tracker_id is None:
                continue

            track_id = int(tracked.tracker_id[i])

            x1, y1, x2, y2 = map(
                int,
                tracked.xyxy[i],
            )

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

            if crop_w < min_width or crop_h < min_height:
                continue

            confidence = (
                float(tracked.confidence[i])
                if tracked.confidence is not None
                else 1.0
            )

            raw_class_id = (
                int(tracked.class_id[i])
                if tracked.class_id is not None
                else None
            )

            raw_class_name = (
                _class_name(raw_class_id)
                if raw_class_id is not None
                else "unknown"
            )

            # 동일 track 안에서 RF-DETR class 관측을 계속 누적
            track_class_counts[track_id][raw_class_name] += 1
            track_class_conf_sums[track_id][raw_class_name] += confidence

            # class가 아니라 track_id 기준 폴더로 저장
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
                "scope": "object",
                "track_id": track_id,
                "track_key": f"object:{track_id}",
                "raw_class_name": raw_class_name,
                "raw_class_id": raw_class_id,
                "frame_idx": frame_idx,
                "timestamp_sec": frame_idx / fps,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
                "bbox_space": "frame",
                "crop_path": str(crop_path.resolve()),
                "width": crop_w,
                "height": crop_h,
            }

            metadata_records.append(record)

            saved_object_crops += 1
            track_ids.add(track_id)

            if preview is not None:
                current_vote = choose_track_class(
                    track_class_counts[track_id],
                    track_class_conf_sums[track_id],
                )

                cv2.rectangle(
                    preview,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    preview,
                    (
                        f"TRACK T{track_id} "
                        f"raw={raw_class_name} "
                        f"vote={current_vote}"
                    ),
                    (
                        x1,
                        min(h - 5, y2 + 18),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
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
                str(
                    preview_dir
                    / f"frame_{frame_idx:08d}.jpg"
                ),
                preview,
            )

        if sampled_frames % 50 == 0:
            print(
                f"    sampled={sampled_frames:,} "
                f"raw={raw_detection_total:,} "
                f"accepted={accepted_object_total:,} "
                f"dup_suppressed={duplicate_suppressed_total:,} "
                f"crops={saved_object_crops:,} "
                f"tracks={len(track_ids):,}",
                end="\r",
            )

        frame_idx += 1

    cap.release()
    print()

    # ---------------------------------------------------------
    # track 단위 최종 class 결정
    # ---------------------------------------------------------
    final_class_by_track: dict[int, str] = {}

    for track_id in sorted(track_ids):
        final_class = choose_track_class(
            track_class_counts[track_id],
            track_class_conf_sums[track_id],
        )

        if final_class is not None:
            final_class_by_track[track_id] = final_class

    # 모든 metadata row에 최종 class를 붙여 한번에 저장
    with metadata_path.open("w", encoding="utf-8") as f:
        for rec in metadata_records:
            track_id = int(rec["track_id"])
            final_class = final_class_by_track.get(
                track_id,
                rec["raw_class_name"],
            )

            rec["class_name"] = final_class
            rec["final_class_name"] = final_class

            f.write(
                json.dumps(
                    rec,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # track별 voting 결과
    track_class_summary = {}

    for track_id in sorted(track_ids):
        counts = track_class_counts[track_id]
        conf_sums = track_class_conf_sums[track_id]

        final_class = final_class_by_track.get(track_id)

        track_class_summary[str(track_id)] = {
            "final_class_name": final_class,
            "class_counts": dict(counts.most_common()),
            "class_confidence_sums": {
                name: round(
                    float(conf_sums.get(name, 0.0)),
                    6,
                )
                for name in sorted(conf_sums)
            },
            "observations": int(sum(counts.values())),
        }

    track_summary_path.write_text(
        json.dumps(
            track_class_summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    final_class_counts = Counter(
        final_class_by_track.values()
    )

    summary = {
        "video": rel_video(video_path),
        "sampled_frames": sampled_frames,
        "raw_detection_total": raw_detection_total,
        "raw_object_detections": raw_object_total,
        "candidate_object_detections": candidate_object_total,
        "accepted_object_detections": accepted_object_total,
        "filtered_small_object_detections": filtered_small_object_total,
        "duplicate_suppressed_detections": duplicate_suppressed_total,
        "excluded_person_detections": excluded_person_total,
        "object_min_width": min_width,
        "object_min_height": min_height,
        "duplicate_iou_threshold": duplicate_iou,
        "raw_class_counts": dict(raw_class_counts.most_common()),
        "accepted_raw_class_counts": dict(
            accepted_class_counts.most_common()
        ),
        "saved_object_crops": saved_object_crops,
        "unique_object_tracks": len(track_ids),
        "final_tracks_by_class": dict(
            final_class_counts.most_common()
        ),
        "metadata_rows": len(metadata_records),
    }

    raw_summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        summary,
        raw_summary_path,
        track_summary_path,
        metadata_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Class-agnostic video object tracker: "
            "RF-DETR -> person exclude -> IoU dedup -> "
            "single ByteTrack -> track-level class voting"
        )
    )

    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-every", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--interval-sec", type=float, default=0.2)
    parser.add_argument("--min-width", type=int, default=24)
    parser.add_argument("--min-height", type=int, default=24)

    parser.add_argument(
        "--duplicate-iou",
        type=float,
        default=0.85,
        help=(
            "같은 frame에서 class가 다르더라도 IoU가 이 값 이상이면 "
            "동일 object 중복 검출로 보고 confidence 높은 것만 유지"
        ),
    )

    args = parser.parse_args()

    if args.interval_sec <= 0:
        raise ValueError("--interval-sec must be > 0")

    if args.preview_every < 0:
        raise ValueError("--preview-every must be >= 0")

    if args.min_width <= 0 or args.min_height <= 0:
        raise ValueError("--min-width/--min-height must be > 0")

    if not (0.0 < args.duplicate_iou <= 1.0):
        raise ValueError("--duplicate-iou must be in (0, 1]")

    if (
        args.max_videos is not None
        and args.max_videos <= 0
    ):
        raise ValueError("--max-videos must be >= 1")

    videos = list_videos()

    if args.max_videos is not None:
        videos = videos[:args.max_videos]

    print("=" * 76)
    print("VIDEO OBJECT TRACK V2 - CLASS AGNOSTIC + TRACK CLASS VOTING")
    print("=" * 76)

    print("input root      :", VIDEO_ROOT)
    print("output root     :", OUT_ROOT)
    print("videos          :", len(videos))
    print("threshold       :", args.threshold)
    print("interval sec    :", args.interval_sec)
    print("preview every   :", args.preview_every)
    print(
        "object min size :",
        f"{args.min_width}x{args.min_height}",
    )
    print("person          : EXCLUDED")
    print("tracking        : CLASS-AGNOSTIC")
    print("class decision  : TRACK-LEVEL CONFIDENCE VOTING")
    print("duplicate IoU   :", args.duplicate_iou)

    if not videos:
        raise RuntimeError(
            f"No videos under {VIDEO_ROOT}"
        )

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
            raise RuntimeError(
                f"Decode failed: {video_path}"
            )

    if args.dry_run:
        print("\nDRY RUN PASS")
        return

    done = (
        set()
        if args.force
        else load_state()
    )

    pending = [
        p
        for p in videos
        if rel_video(p) not in done
    ]

    print("\npending         :", len(pending))

    if not pending:
        print("Nothing to do.")
        return

    # 이미지 파이프라인과 동일한 RF-DETR loader
    model = load_detect_model()

    started = time.time()

    print(
        "\n[CHECK 2] RF-DETR -> PERSON EXCLUDE -> "
        "DEDUP -> SINGLE BYTETRACK -> CLASS VOTING"
    )

    for index, video_path in enumerate(
        pending,
        start=1,
    ):
        print(
            f"\n[{index}/{len(pending)}] "
            f"{video_path.name}"
        )

        (
            summary,
            raw_summary_path,
            track_summary_path,
            metadata_path,
        ) = process_video(
            model=model,
            video_path=video_path,
            threshold=args.threshold,
            interval_sec=args.interval_sec,
            min_width=args.min_width,
            min_height=args.min_height,
            duplicate_iou=args.duplicate_iou,
            preview_every=args.preview_every,
            force=args.force,
        )

        print(
            "  sampled frames          :",
            summary["sampled_frames"],
        )
        print(
            "  raw detections          :",
            summary["raw_detection_total"],
        )
        print(
            "  raw object detections   :",
            summary["raw_object_detections"],
        )
        print(
            "  candidate object        :",
            summary["candidate_object_detections"],
        )
        print(
            "  accepted after dedup    :",
            summary["accepted_object_detections"],
        )
        print(
            "  small object filtered   :",
            summary["filtered_small_object_detections"],
        )
        print(
            "  duplicate suppressed    :",
            summary["duplicate_suppressed_detections"],
        )
        print(
            "  person excluded         :",
            summary["excluded_person_detections"],
        )
        print(
            "  raw classes             :",
            summary["raw_class_counts"],
        )
        print(
            "  accepted raw classes    :",
            summary["accepted_raw_class_counts"],
        )
        print(
            "  object crops            :",
            summary["saved_object_crops"],
        )
        print(
            "  unique object tracks    :",
            summary["unique_object_tracks"],
        )
        print(
            "  FINAL tracks by class   :",
            summary["final_tracks_by_class"],
        )
        print(
            "  metadata rows           :",
            summary["metadata_rows"],
        )
        print(
            "  raw summary             :",
            raw_summary_path,
        )
        print(
            "  track class summary     :",
            track_summary_path,
        )
        print(
            "  metadata                :",
            metadata_path,
        )

        if (
            summary["metadata_rows"]
            != summary["saved_object_crops"]
        ):
            raise RuntimeError(
                "object crop count != metadata rows"
            )

        if summary["saved_object_crops"] > 0:
            print(
                "  [PASS] class-agnostic object "
                "track/crop/metadata 생성"
            )
        else:
            print(
                "  [WARN] 저장된 object track crop 없음"
            )

        done.add(
            rel_video(video_path)
        )
        save_state(done)

        print(
            "  elapsed                 :",
            f"{(time.time() - started) / 60:.1f} min",
        )


if __name__ == "__main__":
    main()