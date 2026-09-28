#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
batch_preprocess_videos_parallel.py

Single-GPU 병렬 실험용 전처리 오케스트레이터.

핵심:
- 각 영상은 독립 worker directory에서 실행한다.
- detect.runner가 상대경로 ./outputs/video_tracks/tracks.jsonl 을 써도
  worker별 cwd가 다르므로 파일 충돌을 피한다.
- PYTHONPATH에 프로젝트 루트를 넣어 worker cwd에서도
  `python -m detect.runner`가 import되도록 한다.
- SUSHI input / validation / report도 영상별 디렉터리에 격리한다.
- 기본 workers=2. RTX 5090 32GB에서 먼저 2개로 검증 후 3개를 시험 권장.
- GPU OOM 발생 시 --workers 1 또는 2로 낮춘다.

주의:
detect.runner가 출력 경로를 "현재 작업 디렉터리 기준"이 아니라
소스 파일 기준 절대경로로 강제하고 있다면 worker isolation이 먹지 않는다.
이 경우 로그에서 expected tracks file not found가 뜨며 즉시 확인 가능하다.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    p.add_argument("--work-root", default="./outputs/parallel_work")
    p.add_argument("--sushi-root", default="./third_party/SUSHI")
    p.add_argument(
        "--checkpoint",
        default="./third_party/SUSHI/pretrained_models/mot17private.pth",
    )
    p.add_argument("--workers", type=int, default=2)
    p.add_argument(
        "--tracking-config",
        default="./pipeline.yaml",
        help="detect.runner 에 줄 yaml (detector/tracker/stitcher 블록). "
             "검출기 교체는 이 파일만 바꾼다: 예 pipeline_tracking_yolo26.yaml",
    )
    p.add_argument("--pattern", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--keep-workdir", action="store_true")
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def discover_videos(root: Path, pattern: str | None):
    out = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        if pattern and pattern.lower() not in p.name.lower():
            continue
        out.append(p.resolve())
    return sorted(out, key=lambda x: str(x).lower())


def safe_key(video: Path, root: Path):
    rel = video.relative_to(root).with_suffix("")
    key = "__".join(rel.parts)
    for ch in '<>:"/\\|?*':
        key = key.replace(ch, "_")
    return key


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def ensure_nonempty(path: Path, stage: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{stage}: expected output not found: {path}"
        )
    if path.stat().st_size == 0:
        raise RuntimeError(f"{stage}: empty output: {path}")


def run_cmd(cmd, cwd: Path, log_path: Path, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = subprocess.list2cmdline([str(x) for x in cmd])

    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"CWD={cwd}\n{printable}\n"
        )
        log.write("=" * 100 + "\n")
        log.flush()

        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            # Prefix console output with worker name for readability.
            print(f"[{cwd.name}] {line}", end="")
            log.write(line)
            log.flush()

        return proc.wait()


def process_video(
    video: Path,
    key: str,
    args,
    project_root: Path,
    videos_root: Path,
):
    t0 = time.time()

    processed_root = Path(args.processed_root).resolve()
    per_root = processed_root / key
    per_root.mkdir(parents=True, exist_ok=True)

    final_out = per_root / "final_routed_tracks.json"
    status_path = per_root / "status.json"
    log_path = per_root / "pipeline.log"

    if args.resume and final_out.exists() and not args.force:
        return {
            "video": str(video),
            "key": key,
            "status": "skipped",
            "output": str(final_out),
        }

    # Independent current working directory for this worker.
    work_dir = Path(args.work_root).resolve() / key
    work_dir.mkdir(parents=True, exist_ok=True)

    local_shared_tracks = work_dir / "outputs" / "video_tracks" / "tracks.jsonl"
    local_shared_tracks.parent.mkdir(parents=True, exist_ok=True)

    tracks_out = per_root / "tracks.jsonl"
    stitched_out = per_root / "stitched_tracks.json"
    validated_out = per_root / "validated_tracks.json"

    # Keep SUSHI adapter output isolated as well.
    sushi_out_root = per_root / "sushi_input"
    sushi_input = sushi_out_root / video.stem

    state = {
        "video": str(video),
        "video_key": key,
        "status": "running",
        "stage": None,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "worker_dir": str(work_dir),
    }
    write_json(status_path, state)

    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(project_root)
        if not old_pythonpath
        else str(project_root) + os.pathsep + old_pythonpath
    )

    try:
        # --------------------------------------------------------------
        # 1) RF-DETR + BoT-SORT
        # --------------------------------------------------------------
        state["stage"] = "rfdetr_botsort"
        write_json(status_path, state)

        if local_shared_tracks.exists():
            local_shared_tracks.unlink()

        cmd = [
            args.python,
            "-m",
            "detect.runner",
            str(video),
            "--config",
            str(Path(args.tracking_config).resolve()),
            "--no-stitch",
        ]

        rc = run_cmd(cmd, work_dir, log_path, env=env)
        if rc != 0:
            raise RuntimeError(
                f"RF-DETR/BoT-SORT failed, exit={rc}"
            )

        ensure_nonempty(
            local_shared_tracks,
            "RF-DETR/BoT-SORT worker-isolated output",
        )
        shutil.copy2(local_shared_tracks, tracks_out)

        # --------------------------------------------------------------
        # 2) SUSHI adapter
        # --------------------------------------------------------------
        state["stage"] = "sushi_adapter"
        write_json(status_path, state)

        cmd = [
            args.python,
            str(project_root / "video" / "sushi_adapter.py"),
            "--video", str(video),
            "--tracks", str(tracks_out),
            "--sushi-root", str(Path(args.sushi_root).resolve()),
            "--output-root", str(sushi_out_root),
        ]

        rc = run_cmd(cmd, project_root, log_path, env=env)
        if rc != 0:
            raise RuntimeError(f"sushi_adapter failed, exit={rc}")

        ensure_nonempty(
            sushi_input / "manifest.json",
            "SUSHI adapter",
        )

        # --------------------------------------------------------------
        # 3) SUSHI
        # --------------------------------------------------------------
        state["stage"] = "sushi_inference"
        write_json(status_path, state)

        cmd = [
            args.python,
            str(project_root / "video" / "sushi_inference.py"),
            "--input-root", str(sushi_input),
            "--sushi-root", str(Path(args.sushi_root).resolve()),
            "--checkpoint", str(Path(args.checkpoint).resolve()),
            "--tracks", str(tracks_out),
            "--output", str(stitched_out),
        ]

        rc = run_cmd(cmd, project_root, log_path, env=env)
        if rc != 0:
            raise RuntimeError(f"SUSHI failed, exit={rc}")

        ensure_nonempty(stitched_out, "SUSHI")

        # --------------------------------------------------------------
        # 4) Auto validation
        # --------------------------------------------------------------
        state["stage"] = "auto_track_validator"
        write_json(status_path, state)

        cmd = [
            args.python,
            str(project_root / "video" / "auto_track_validator.py"),
            "--video", str(video),
            "--tracks", str(stitched_out),
            "--output", str(validated_out),
            "--review-dir", str(per_root / "track_validation_auto"),
        ]

        rc = run_cmd(cmd, project_root, log_path, env=env)
        if rc != 0:
            raise RuntimeError(
                f"auto_track_validator failed, exit={rc}"
            )

        ensure_nonempty(validated_out, "auto validation")

        # --------------------------------------------------------------
        # 5) Final route
        # --------------------------------------------------------------
        state["stage"] = "finalize_track_routes"
        write_json(status_path, state)

        cmd = [
            args.python,
            str(project_root / "video" / "finalize_track_routes.py"),
            "--video", str(video),
            "--tracks", str(validated_out),
            "--output", str(final_out),
            "--report-dir", str(per_root / "final_route_report"),
        ]

        rc = run_cmd(cmd, project_root, log_path, env=env)
        if rc != 0:
            raise RuntimeError(
                f"finalize_track_routes failed, exit={rc}"
            )

        ensure_nonempty(final_out, "final route")

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
            },
        })
        write_json(status_path, state)

        if not args.keep_workdir:
            shutil.rmtree(work_dir, ignore_errors=True)

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
        write_json(status_path, state)

        return {
            "video": str(video),
            "key": key,
            "status": "failed",
            "elapsed_sec": round(elapsed, 3),
            "error": repr(e),
            "log": str(log_path),
        }


def main():
    args = parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    project_root = Path(__file__).resolve().parents[1]
    videos_root = Path(args.videos_root).resolve()

    required = [
        project_root / "video" / "sushi_adapter.py",
        project_root / "video" / "sushi_inference.py",
        project_root / "video" / "auto_track_validator.py",
        project_root / "video" / "finalize_track_routes.py",
        project_root / "detect" / "runner.py",
        Path(args.checkpoint).resolve(),
    ]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n  " + "\n  ".join(missing)
        )

    videos = discover_videos(videos_root, args.pattern)
    if args.limit is not None:
        videos = videos[:args.limit]

    if not videos:
        raise RuntimeError(f"No videos found: {videos_root}")

    jobs = [
        (video, safe_key(video, videos_root))
        for video in videos
    ]

    print("=" * 100)
    print("PARALLEL VIDEO PREPROCESS")
    print("=" * 100)
    print(f"videos        : {len(jobs)}")
    print(f"workers       : {args.workers}")
    print(f"GPU           : shared by all workers")
    print(f"work root     : {Path(args.work_root).resolve()}")
    print(f"processed root: {Path(args.processed_root).resolve()}")
    print("=" * 100)

    started = time.time()
    results = []
    stop_requested = False

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                process_video,
                video,
                key,
                args,
                project_root,
                videos_root,
            ): (video, key)
            for video, key in jobs
        }

        for fut in as_completed(futures):
            video, key = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                result = {
                    "video": str(video),
                    "key": key,
                    "status": "failed",
                    "error": repr(e),
                }

            results.append(result)

            if result["status"] == "complete":
                print(
                    f"\n[DONE] {video.name} "
                    f"({result.get('elapsed_sec', 0)/60:.1f} min)"
                )
            elif result["status"] == "skipped":
                print(f"\n[SKIP] {video.name}")
            else:
                print(
                    f"\n[FAILED] {video.name}: "
                    f"{result.get('error')}"
                )
                if args.stop_on_error:
                    stop_requested = True
                    break

        if stop_requested:
            for f in futures:
                f.cancel()

    elapsed = time.time() - started

    processed_root = Path(args.processed_root).resolve()
    summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "workers": args.workers,
        "elapsed_sec": round(elapsed, 3),
        "selected": len(jobs),
        "complete": sum(x["status"] == "complete" for x in results),
        "skipped": sum(x["status"] == "skipped" for x in results),
        "failed": sum(x["status"] == "failed" for x in results),
        "results": results,
    }
    summary_path = processed_root / "parallel_batch_summary.json"
    write_json(summary_path, summary)

    print()
    print("=" * 100)
    print("PARALLEL BATCH COMPLETE")
    print("=" * 100)
    print(f"selected : {len(jobs)}")
    print(f"complete : {summary['complete']}")
    print(f"skipped  : {summary['skipped']}")
    print(f"failed   : {summary['failed']}")
    print(f"wall time: {elapsed/60:.1f} min")
    print(f"summary  : {summary_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
