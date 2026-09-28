#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = ROOT / "data" / "videos"
PROCESSED_ROOT = ROOT / "outputs" / "processed_videos"


def parse_args():
    p = argparse.ArgumentParser(
        description="Build canonical DB candidates and ingest every video found in a directory"
    )
    p.add_argument(
        "--video-dir",
        default=str(DEFAULT_VIDEO_DIR),
        help="Directory containing videos. Recursively scans supported video files.",
    )
    p.add_argument(
        "--extensions",
        nargs="*",
        default=[".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"],
    )
    p.add_argument("--person-per-track", type=int, default=5)
    p.add_argument("--object-per-track", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--config", default="pipeline.yaml", help="임베더 구성 yaml (ingest_final_candidates_qdrant.py --config 로 전달)")
    p.add_argument("--person-collection", default="forensic_person")
    p.add_argument("--object-collection", default="forensic_object")
    p.add_argument(
        "--skip-missing-processed",
        action="store_true",
        help="Skip videos without outputs/processed_videos/<stem>/final_routed_tracks_canonical.json",
    )
    p.add_argument(
        "--audit-each",
        action="store_true",
        help="Run audit_qdrant_video_payloads.py after each successful ingest.",
    )
    return p.parse_args()


def find_videos(video_dir: Path, extensions):
    extset = {e.lower() if e.startswith(".") else "." + e.lower() for e in extensions}
    videos = [
        p for p in video_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extset
    ]
    return sorted(videos, key=lambda p: str(p).lower())


def run(cmd):
    print("\n$", " ".join(map(str, cmd)))
    t0 = time.time()
    cp = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    if cp.returncode != 0:
        raise RuntimeError(f"command failed rc={cp.returncode}")
    return elapsed


def main():
    args = parse_args()

    video_dir = Path(args.video_dir).resolve()
    if not video_dir.is_dir():
        raise NotADirectoryError(video_dir)

    build_script = ROOT / "ingest" / "build_final_db_candidates_canonical.py"
    ingest_script = ROOT / "ingest" / "ingest_final_candidates_qdrant.py"
    audit_script = ROOT / "ingest" / "audit_qdrant_video_payloads.py"

    required = [build_script, ingest_script]
    if args.audit_each:
        required.append(audit_script)

    missing = [str(x) for x in required if not x.is_file()]
    if missing:
        raise FileNotFoundError("Missing required scripts:\n  " + "\n  ".join(missing))

    videos = find_videos(video_dir, args.extensions)
    if not videos:
        raise RuntimeError(f"No supported videos found under: {video_dir}")

    print("=" * 100)
    print("BATCH VIDEO -> FINAL QDRANT")
    print("=" * 100)
    print("video_dir          :", video_dir)
    print("videos found       :", len(videos))
    print("person collection  :", args.person_collection)
    print("object collection  :", args.object_collection)
    print("person per track   :", args.person_per_track)
    print("object per track   :", args.object_per_track)
    print("batch size         :", args.batch_size)
    print("config (embedders) :", args.config)
    print("audit each         :", args.audit_each)
    print("=" * 100)

    completed = []
    skipped = []
    failed = []
    timings = {}

    for i, video in enumerate(videos, 1):
        stem = video.stem
        t0 = time.time()

        print("\n" + "#" * 100)
        print(f"[{i}/{len(videos)}] {video.name}")
        print("#" * 100)

        processed_dir = PROCESSED_ROOT / stem
        canonical = processed_dir / "final_routed_tracks_canonical.json"
        fallback = processed_dir / "final_routed_tracks.json"

        if not canonical.is_file() and not fallback.is_file():
            msg = f"processed route file missing: {processed_dir}"
            if args.skip_missing_processed:
                print("[SKIP]", msg)
                skipped.append({"video_stem": stem, "reason": msg})
                timings[stem] = round(time.time() - t0, 3)
                continue
            else:
                print("[FAILED]", msg)
                failed.append({"video_stem": stem, "error": msg})
                timings[stem] = round(time.time() - t0, 3)
                continue

        try:
            build_cmd = [
                sys.executable,
                str(build_script),
                "--video-stem", stem,
                "--person-per-track", str(args.person_per_track),
                "--object-per-track", str(args.object_per_track),
            ]
            build_elapsed = run(build_cmd)

            ingest_cmd = [
                sys.executable,
                str(ingest_script),
                "--video-stem", stem,
                "--batch-size", str(args.batch_size),
                "--config", str(args.config),
                "--person-collection", args.person_collection,
                "--object-collection", args.object_collection,
            ]
            ingest_elapsed = run(ingest_cmd)

            audit_elapsed = 0.0
            if args.audit_each:
                audit_cmd = [
                    sys.executable,
                    str(audit_script),
                    "--video-stem", stem,
                    "--person-collection", args.person_collection,
                    "--object-collection", args.object_collection,
                ]
                audit_elapsed = run(audit_cmd)

            elapsed = time.time() - t0
            timings[stem] = round(elapsed, 3)
            completed.append(stem)

            print(
                f"[COMPLETE] {stem} | total={elapsed/60:.2f} min "
                f"(build={build_elapsed:.1f}s ingest={ingest_elapsed:.1f}s audit={audit_elapsed:.1f}s)"
            )

        except Exception as exc:
            elapsed = time.time() - t0
            timings[stem] = round(elapsed, 3)
            failed.append({"video_stem": stem, "error": repr(exc)})
            print(f"[FAILED] {stem}: {exc}")

    summary = {
        "video_dir": str(video_dir),
        "videos_found": len(videos),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "person_collection": args.person_collection,
        "object_collection": args.object_collection,
        "person_per_track": args.person_per_track,
        "object_per_track": args.object_per_track,
        "batch_size": args.batch_size,
        "timings_sec": timings,
        "elapsed_sec": round(sum(timings.values()), 3),
    }

    out = ROOT / "outputs" / "final_db_candidates" / "batch_all_videos_ingest_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print("BATCH VIDEO -> FINAL QDRANT COMPLETE")
    print("=" * 100)
    print("completed :", len(completed))
    print("skipped   :", len(skipped))
    print("failed    :", len(failed))
    print("elapsed   :", f"{sum(timings.values())/60:.2f} min")
    print("summary   :", out)
    print("=" * 100)


if __name__ == "__main__":
    main()
