from __future__ import annotations

import argparse
import hashlib
import queue
import threading
import time
from pathlib import Path

import cv2
import yaml

from detect.io_jsonl import write_jsonl
from detect.loader import load_component


class FrameReader:
    """Decode frames in a background thread so CPU video decode overlaps GPU inference."""

    def __init__(self, video: str, max_frames: int | None, prefetch: int = 64):
        self.video = video
        self.max_frames = max_frames
        self.q: queue.Queue = queue.Queue(maxsize=max(4, prefetch))
        self.exc: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.video)
            if not cap.isOpened():
                raise RuntimeError(f"video open failed: {self.video}")

            frame_idx = 0
            while True:
                if self.max_frames is not None and frame_idx >= self.max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                self.q.put((frame_idx, frame))
                frame_idx += 1
        except BaseException as e:
            self.exc = e
        finally:
            if cap is not None:
                cap.release()
            self.q.put(None)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                if self.exc is not None:
                    raise self.exc
                break
            yield item


def _video_id(video_path: str) -> str:
    p = Path(video_path).resolve()
    return hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:16]


def _normalize_row(row: dict, *, video: str, video_id: str, width: int, height: int) -> dict:
    row = dict(row)
    row["video"] = str(Path(video).resolve())
    row["video_id"] = video_id

    bbox = row.get("bbox")
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = map(float, bbox)
        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        row["bbox"] = [x1, y1, x2, y2]
        row["bbox_valid"] = bool(x2 > x1 and y2 > y1)

    return row


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("video")
    ap.add_argument("--config", default="pipeline.yaml")
    ap.add_argument("--output", default="outputs/video_tracks/tracks.jsonl")
    ap.add_argument("--no-stitch", action="store_true")

    # Legacy option. 0 means read/process all frames.
    ap.add_argument("--frame-skip", type=int, default=0)

    # Fast experiment mode:
    # detector runs only every Nth frame, but tracker.update is still called
    # on every frame so its age/motion clock remains continuous.
    ap.add_argument(
        "--det-stride",
        type=int,
        default=1,
        help="Detector interval. 1=every frame, 2=every 2nd frame. "
             "Tracker still advances on every frame.",
    )
    ap.add_argument(
        "--prefetch",
        type=int,
        default=64,
        help="Number of decoded frames buffered by background reader.",
    )
    ap.add_argument("--max-frames", type=int, default=None)

    args = ap.parse_args()

    if args.frame_skip < 0:
        raise ValueError("--frame-skip must be >= 0")
    if args.det_stride < 1:
        raise ValueError("--det-stride must be >= 1")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "detector" not in cfg:
        raise KeyError("pipeline.yaml에 'detector' 설정이 없습니다.")
    if "tracker" not in cfg:
        raise KeyError("pipeline.yaml에 'tracker' 설정이 없습니다.")

    detector = load_component(cfg["detector"])
    tracker = load_component(cfg["tracker"])

    stitcher = None
    if not args.no_stitch:
        if "stitcher" not in cfg:
            raise KeyError("pipeline.yaml에 'stitcher' 설정이 없습니다.")
        stitcher = load_component(cfg["stitcher"])

    # Probe metadata once. Actual decode happens in background reader.
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"video open failed: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    legacy_stride = args.frame_skip + 1
    effective_det_stride = max(legacy_stride, args.det_stride)

    processed_frames = 0
    detector_frames = 0
    all_tracks = []
    started = time.perf_counter()

    reader = FrameReader(
        args.video,
        max_frames=args.max_frames,
        prefetch=args.prefetch,
    ).start()

    for frame_idx, frame in reader:
        timestamp_sec = frame_idx / fps if fps > 0 else None
        run_detector = (frame_idx % effective_det_stride == 0)

        if run_detector:
            detections = detector.detect(
                frame,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
            )
            detector_frames += 1
        else:
            detections = []

        # Always advance tracker in real frame time.
        tracks = tracker.update(
            frame,
            detections,
            frame_idx=frame_idx,
        )

        # Save only actual detector-observation frames.
        # Predicted-only boxes from skipped frames are not persisted.
        if run_detector:
            all_tracks.extend(tracks)

        processed_frames += 1

        if processed_frames == 1 or processed_frames % 100 == 0:
            elapsed = time.perf_counter() - started
            progress = (
                f"{frame_idx + 1:,}/{total_frames:,}"
                if total_frames > 0
                else f"{frame_idx + 1:,}"
            )
            print(
                f"[frame {frame_idx:,}] progress={progress} "
                f"detect={'Y' if run_detector else 'N'} "
                f"detections={len(detections)} tracks={len(tracks)} "
                f"saved_records={len(all_tracks):,} "
                f"tracker_frames={processed_frames:,} "
                f"detector_frames={detector_frames:,} "
                f"elapsed={elapsed:.1f}s"
            )

    vid = _video_id(args.video)

    if stitcher is None:
        rows = (
            _normalize_row(
                track.to_dict(),
                video=args.video,
                video_id=vid,
                width=width,
                height=height,
            )
            for track in all_tracks
        )
        write_jsonl(args.output, rows)

        elapsed = time.perf_counter() - started
        print(
            f"saved: {args.output} | "
            f"tracker_frames={processed_frames:,} | "
            f"detector_frames={detector_frames:,} | "
            f"det_stride={effective_det_stride} | "
            f"track_records={len(all_tracks):,} | "
            f"elapsed={elapsed:.1f}s"
        )
        return

    long_tracks = stitcher.stitch(all_tracks)
    rows = []

    for long_track in long_tracks:
        for record in long_track.records:
            row = _normalize_row(
                record.to_dict(),
                video=args.video,
                video_id=vid,
                width=width,
                height=height,
            )
            row["long_track_id"] = long_track.long_track_id
            row["member_track_ids"] = long_track.member_track_ids
            rows.append(row)

    write_jsonl(args.output, rows)

    elapsed = time.perf_counter() - started
    print(
        f"saved: {args.output} | "
        f"long_tracks={len(long_tracks):,} | records={len(rows):,} | "
        f"det_stride={effective_det_stride} | elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
