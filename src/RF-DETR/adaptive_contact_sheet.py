from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from adaptive_frame_sampler import (
    AdaptiveSamplingConfig,
    adaptive_extract_frames,
    make_contact_sheet,
)


def fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    minute = int(sec // 60)
    second = sec - minute * 60
    return f"{minute:02d}:{second:05.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create 4x4 review contact sheets from adaptive video samples."
    )
    parser.add_argument("video")
    parser.add_argument("--out", default="adaptive_contact_sheets")
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--cols", type=int, default=4)
    args = parser.parse_args()

    video = Path(args.video)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = AdaptiveSamplingConfig()
    group_frames = []
    group_meta = []
    manifests = []
    sheet_index = 1

    def flush():
        nonlocal group_frames, group_meta, sheet_index
        if not group_frames:
            return
        labels = [
            f"#{m['frame']}  {fmt_time(m['time_sec'])}"
            for m in group_meta
        ]
        rows = max(1, (len(group_frames) + args.cols - 1) // args.cols)
        sheet = make_contact_sheet(
            group_frames,
            columns=args.cols,
            rows=rows,
            labels=labels,
        )
        first_t = group_meta[0]['time_sec']
        last_t = group_meta[-1]['time_sec']
        name = (
            f"sheet_{sheet_index:04d}_"
            f"{int(first_t):06d}s_{int(last_t):06d}s.jpg"
        )
        path = out_dir / name
        cv2.imwrite(str(path), sheet)
        manifests.append({
            "sheet": str(path),
            "video": str(video),
            "start_sec": first_t,
            "end_sec": last_t,
            "cells": list(group_meta),
        })
        print(f"[SHEET] {path} ({len(group_frames)} frames)")
        group_frames = []
        group_meta = []
        sheet_index += 1

    for frame_no, timestamp, frame in adaptive_extract_frames(video, cfg):
        group_frames.append(frame.copy())
        group_meta.append({"frame": int(frame_no), "time_sec": float(timestamp)})
        if len(group_frames) >= max(1, args.group_size):
            flush()

    flush()

    manifest_path = out_dir / "contact_sheet_manifest.json"
    manifest_path.write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
