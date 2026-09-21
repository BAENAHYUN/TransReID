#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
batch_preprocess_videos.py

data/videos 아래 모든 동영상을 순차 처리하여
DB/임베딩 직전 단계인 final_routed_tracks.json까지 생성한다.

Pipeline
-------
Raw video
  -> RF-DETR + BoT-SORT       (python -m detect.runner <video> --no-stitch)
  -> tracks.jsonl
  -> sushi_adapter.py
  -> sushi_inference.py       (SUSHI MOT17 Private)
  -> stitched_tracks.json
  -> auto_track_validator.py
  -> validated_tracks.json
  -> finalize_track_routes.py
  -> final_routed_tracks.json

중요:
- 임베딩/Qdrant 저장은 이 스크립트에서 하지 않는다.
- 모든 영상의 final_routed_tracks.json을 만든 뒤 별도 배치 임베딩 단계로 넘어간다.
- detect.runner가 공용 outputs/video_tracks/tracks.jsonl을 쓰므로
  각 영상 처리 직후 per-video 폴더로 복사해 보존한다.

예:
python .\batch_preprocess_videos.py
python .\batch_preprocess_videos.py --resume
python .\batch_preprocess_videos.py --limit 10
python .\batch_preprocess_videos.py --pattern "*Normal*"
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


VIDEO_EXTS = {
    ".mp4", ".avi", ".mov", ".mkv", ".m4v",
    ".wmv", ".webm", ".mpeg", ".mpg"
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos-root", default="./data/videos")
    p.add_argument("--processed-root", default="./outputs/processed_videos")
    p.add_argument("--sushi-input-root", default="./outputs/sushi_input")
    p.add_argument("--sushi-root", default="./third_party/SUSHI")
    p.add_argument(
        "--checkpoint",
        default="./third_party/SUSHI/pretrained_models/mot17private.pth"
    )
    p.add_argument(
        "--shared-tracks",
        default="./outputs/video_tracks/tracks.jsonl",
        help="detect.runner가 생성하는 공용 tracks.jsonl"
    )
    p.add_argument("--pattern", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def discover_videos(root: Path, pattern: str | None):
    videos = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        if pattern and pattern.lower() not in p.name.lower():
            continue
        videos.append(p.resolve())

    videos.sort(key=lambda x: str(x).lower())
    return videos


def safe_video_key(video: Path, videos_root: Path):
    """
    Avoid collisions if two folders contain files with same stem.
    Root-relative path parts are joined using '__'.
    """
    rel = video.relative_to(videos_root)
    parts = list(rel.with_suffix("").parts)
    key = "__".join(parts)
    bad = '<>:"/\\|?*'
    for ch in bad:
        key = key.replace(ch, "_")
    return key


def run_cmd(cmd, cwd: Path, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    printable = subprocess.list2cmdline([str(x) for x in cmd])

    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write(f"[{datetime.now().isoformat(timespec='seconds')}] {printable}\n")
        log.write("=" * 100 + "\n")
        log.flush()

        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert proc.stdout is not None

        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        return proc.wait()


def ensure_file(path: Path, stage: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{stage}: expected output not found: {path}"
        )
    if path.stat().st_size == 0:
        raise RuntimeError(
            f"{stage}: output is empty: {path}"
        )


def write_status(path: Path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def process_one(
    idx: int,
    total: int,
    video: Path,
    key: str,
    args,
    project_root: Path,
):
    per_root = Path(args.processed_root).resolve() / key
    per_root.mkdir(parents=True, exist_ok=True)

    log_path = per_root / "pipeline.log"
    status_path = per_root / "status.json"

    tracks_out = per_root / "tracks.jsonl"
    stitched_out = per_root / "stitched_tracks.json"
    validated_out = per_root / "validated_tracks.json"
    final_out = per_root / "final_routed_tracks.json"

    sushi_input = Path(args.sushi_input_root).resolve() / video.stem

    if args.resume and final_out.exists() and not args.force:
        print(
            f"\n[{idx}/{total}] SKIP {video.name} "
            f"(final_routed_tracks.json exists)"
        )
        return {
            "video": str(video),
            "key": key,
            "status": "skipped",
            "output": str(final_out),
        }

    state = {
        "video": str(video),
        "video_key": key,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "stage": None,
    }
    write_status(status_path, state)

    t0 = time.time()

    try:
        print("\n" + "#" * 100)
        print(f"[{idx}/{total}] {video}")
        print("#" * 100)

        # ------------------------------------------------------------------
        # 1. RF-DETR + BoT-SORT
        # ------------------------------------------------------------------
        state["stage"] = "rfdetr_botsort"
        write_status(status_path, state)

        shared_tracks = Path(args.shared_tracks).resolve()

        if shared_tracks.exists():
            shared_tracks.unlink()

        cmd = [
            args.python,
            "-m",
            "detect.runner",
            str(video),
            "--no-stitch",
        ]

        rc = run_cmd(cmd, project_root, log_path)
        if rc != 0:
            raise RuntimeError(
                f"RF-DETR/BoT-SORT failed with exit code {rc}"
            )

        ensure_file(shared_tracks, "RF-DETR/BoT-SORT")

        shutil.copy2(shared_tracks, tracks_out)
        ensure_file(tracks_out, "copy tracks.jsonl")

        # ------------------------------------------------------------------
        # 2. SUSHI adapter
        # ------------------------------------------------------------------
        state["stage"] = "sushi_adapter"
        write_status(status_path, state)

        cmd = [
            args.python,
            str(project_root / "sushi_adapter.py"),
            "--video",
            str(video),
            "--tracks",
            str(tracks_out),
            "--sushi-root",
            str(Path(args.sushi_root).resolve()),
            "--output-root",
            str(Path(args.sushi_input_root).resolve()),
        ]

        rc = run_cmd(cmd, project_root, log_path)
        if rc != 0:
            raise RuntimeError(
                f"sushi_adapter failed with exit code {rc}"
            )

        manifest = sushi_input / "manifest.json"
        ensure_file(manifest, "SUSHI adapter")

        # ------------------------------------------------------------------
        # 3. SUSHI inference
        # ------------------------------------------------------------------
        state["stage"] = "sushi_inference"
        write_status(status_path, state)

        cmd = [
            args.python,
            str(project_root / "sushi_inference.py"),
            "--input-root",
            str(sushi_input),
            "--sushi-root",
            str(Path(args.sushi_root).resolve()),
            "--checkpoint",
            str(Path(args.checkpoint).resolve()),
            "--tracks",
            str(tracks_out),
            "--output",
            str(stitched_out),
        ]

        rc = run_cmd(cmd, project_root, log_path)
        if rc != 0:
            raise RuntimeError(
                f"sushi_inference failed with exit code {rc}"
            )

        ensure_file(stitched_out, "SUSHI inference")

        # ------------------------------------------------------------------
        # 4. Automatic first-pass track validation
        # ------------------------------------------------------------------
        state["stage"] = "auto_track_validator"
        write_status(status_path, state)

        first_review_root = per_root / "track_validation_auto"

        cmd = [
            args.python,
            str(project_root / "auto_track_validator.py"),
            "--video",
            str(video),
            "--tracks",
            str(stitched_out),
            "--output",
            str(validated_out),
            "--review-dir",
            str(first_review_root),
        ]

        rc = run_cmd(cmd, project_root, log_path)
        if rc != 0:
            raise RuntimeError(
                f"auto_track_validator failed with exit code {rc}"
            )

        ensure_file(validated_out, "auto_track_validator")

        # ------------------------------------------------------------------
        # 5. Automatic second-pass route finalization
        # ------------------------------------------------------------------
        state["stage"] = "finalize_track_routes"
        write_status(status_path, state)

        report_root = per_root / "final_route_report"

        cmd = [
            args.python,
            str(project_root / "finalize_track_routes.py"),
            "--video",
            str(video),
            "--tracks",
            str(validated_out),
            "--output",
            str(final_out),
            "--report-dir",
            str(report_root),
        ]

        rc = run_cmd(cmd, project_root, log_path)
        if rc != 0:
            raise RuntimeError(
                f"finalize_track_routes failed with exit code {rc}"
            )

        ensure_file(final_out, "finalize_track_routes")

        elapsed = time.time() - t0

        state.update({
            "status": "complete",
            "stage": "complete",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": round(elapsed, 3),
            "outputs": {
                "tracks": str(tracks_out),
                "stitched": str(stitched_out),
                "validated": str(validated_out),
                "final_routed": str(final_out),
            }
        })
        write_status(status_path, state)

        print(
            f"\n[DONE] {video.name} "
            f"-> {final_out} "
            f"({elapsed/60:.1f} min)"
        )

        return {
            "video": str(video),
            "key": key,
            "status": "complete",
            "elapsed_sec": round(elapsed, 3),
            "output": str(final_out),
        }

    except Exception as e:
        elapsed = time.time() - t0

        state.update({
            "status": "failed",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": round(elapsed, 3),
            "error": repr(e),
        })
        write_status(status_path, state)

        print(f"\n[FAILED] {video.name}: {e}")

        return {
            "video": str(video),
            "key": key,
            "status": "failed",
            "elapsed_sec": round(elapsed, 3),
            "error": repr(e),
        }


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    videos_root = Path(args.videos_root).resolve()

    if not videos_root.exists():
        raise FileNotFoundError(
            f"videos root not found: {videos_root}"
        )

    required = [
        project_root / "sushi_adapter.py",
        project_root / "sushi_inference.py",
        project_root / "auto_track_validator.py",
        project_root / "finalize_track_routes.py",
        Path(args.checkpoint).resolve(),
    ]

    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError(
            "required files missing:\n  " + "\n  ".join(missing)
        )

    videos = discover_videos(videos_root, args.pattern)

    if args.limit is not None:
        videos = videos[:args.limit]

    if not videos:
        raise RuntimeError(
            f"no videos found under {videos_root}"
        )

    print("=" * 100)
    print("FULL VIDEO PREPROCESS BATCH")
    print("=" * 100)
    print(f"videos root       : {videos_root}")
    print(f"videos            : {len(videos)}")
    print(f"processed root    : {Path(args.processed_root).resolve()}")
    print(f"SUSHI checkpoint  : {Path(args.checkpoint).resolve()}")
    print(f"embedding/Qdrant  : NOT RUN IN THIS STAGE")
    print("=" * 100)

    results = []

    for i, video in enumerate(videos, 1):
        key = safe_video_key(video, videos_root)

        result = process_one(
            i,
            len(videos),
            video,
            key,
            args,
            project_root,
        )

        results.append(result)

        if result["status"] == "failed" and args.stop_on_error:
            break

    processed_root = Path(args.processed_root).resolve()
    processed_root.mkdir(parents=True, exist_ok=True)

    batch_summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "videos_root": str(videos_root),
        "total_selected": len(videos),
        "complete": sum(r["status"] == "complete" for r in results),
        "skipped": sum(r["status"] == "skipped" for r in results),
        "failed": sum(r["status"] == "failed" for r in results),
        "results": results,
    }

    summary_path = processed_root / "batch_summary.json"
    write_status(summary_path, batch_summary)

    print()
    print("=" * 100)
    print("BATCH COMPLETE")
    print("=" * 100)
    print(f"selected : {len(videos)}")
    print(f"complete : {batch_summary['complete']}")
    print(f"skipped  : {batch_summary['skipped']}")
    print(f"failed   : {batch_summary['failed']}")
    print(f"summary  : {summary_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
