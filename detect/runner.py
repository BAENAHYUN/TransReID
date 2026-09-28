from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import yaml

from detect.io_jsonl import write_jsonl
from detect.loader import load_component


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("video")
    ap.add_argument("--config", default="pipeline.yaml")
    ap.add_argument(
        "--output",
        default="outputs/video_tracks/tracks.jsonl",
    )
    ap.add_argument("--no-stitch", action="store_true")

    # N이면 N+1 프레임마다 1장 처리
    # 0 = 모든 프레임
    ap.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="N이면 N+1 프레임마다 1개 프레임만 처리 (0=모든 프레임)",
    )

    # 실제 읽은 원본 frame_idx 기준 상한
    ap.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="최대 원본 프레임 수. 예: 300이면 frame 0~299까지만 읽음",
    )

    args = ap.parse_args()

    if args.frame_skip < 0:
        raise ValueError("--frame-skip must be >= 0")

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

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError(f"video open failed: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_idx = 0
    processed_frames = 0
    all_tracks = []
    started = time.perf_counter()

    stride = args.frame_skip + 1

    while True:
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        # frame-skip: 원본 frame_idx는 유지
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        timestamp_sec = frame_idx / fps if fps > 0 else None

        detections = detector.detect(
            frame,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
        )

        tracks = tracker.update(
            frame,
            detections,
            frame_idx=frame_idx,
        )

        all_tracks.extend(tracks)
        processed_frames += 1

        if processed_frames == 1 or processed_frames % 100 == 0:
            elapsed = time.perf_counter() - started

            if total_frames > 0:
                progress = f"{frame_idx + 1:,}/{total_frames:,}"
            else:
                progress = f"{frame_idx + 1:,}"

            print(
                f"[frame {frame_idx:,}] "
                f"progress={progress} "
                f"detections={len(detections)} "
                f"tracks={len(tracks)} "
                f"total_tracks={len(all_tracks):,} "
                f"processed={processed_frames:,} "
                f"elapsed={elapsed:.1f}s"
            )

        frame_idx += 1

    cap.release()

    if stitcher is None:
        write_jsonl(
            args.output,
            (track.to_dict() for track in all_tracks),
        )

        elapsed = time.perf_counter() - started
        print(
            f"saved: {args.output} | "
            f"processed_frames={processed_frames:,} | "
            f"track_records={len(all_tracks):,} | "
            f"elapsed={elapsed:.1f}s"
        )
        return

    long_tracks = stitcher.stitch(all_tracks)

    rows = []

    for long_track in long_tracks:
        for record in long_track.records:
            row = record.to_dict()
            row["long_track_id"] = long_track.long_track_id
            row["member_track_ids"] = long_track.member_track_ids
            rows.append(row)

    write_jsonl(args.output, rows)

    elapsed = time.perf_counter() - started
    print(
        f"saved: {args.output} | "
        f"long_tracks={len(long_tracks):,} | "
        f"records={len(rows):,} | "
        f"elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
