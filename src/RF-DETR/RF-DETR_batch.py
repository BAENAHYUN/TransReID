"""
RF-DETR batch detection script.

detect_rf.py contains the detection functions.
This script runs detection over multiple images and saves crop/filter statistics.

Safety features for large runs (~100k images):
  - checkpoint every CHECKPOINT_EVERY images
  - automatic resume (skips images already processed)
  - failed images logged, run keeps going
  - periodic progress / ETA output

Files written under <output_dir>/checkpoint/:
  state.json      run config + counters + file offsets
  done.txt        one image filename per completed image (append-only)
  crops.jsonl     accepted crop records   (append-only, one JSON per line)
  filtered.jsonl  filtered crop records   (append-only, one JSON per line)
  errors.jsonl    failed images           (append-only)

<output_dir>/filter_stats.json is rebuilt from the .jsonl files at the end of
each run, in the original {"crops": [...], "filtered": [...]} schema.
"""

import argparse
import glob
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path


# ------------------------------------------------------------
# Project root
# RF-DETR_batch.py:
#   TransReID/src/RF-DETR/RF-DETR_batch.py
#
# Project root:
#   TransReID/
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

# Allow imports from the project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# detect_rf.py is located in the project root
from detect_rf import (
    load_detect_model,
    detect_and_crop,
    FORENSIC_TARGET_CLASSES,
)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
# COCO train2017 dataset
SAMPLE_DIR = Path(r"C:\datasets\coco\train2017")

OUTPUT_DIR = ROOT / "data" / "crops"


# ------------------------------------------------------------
# Detection mode
# ------------------------------------------------------------
# None = detect ALL COCO classes
#
# If you want to use only the forensic classes later:
# DEFAULT_TARGET_CLASSES = FORENSIC_TARGET_CLASSES
DEFAULT_TARGET_CLASSES = None


# ------------------------------------------------------------
# Safety settings
# ------------------------------------------------------------
# Save a checkpoint every N images. Worst-case loss on a crash is
# the images processed since the last checkpoint.
CHECKPOINT_EVERY = 500

# Print a progress line every N images.
PROGRESS_EVERY = 100

STATE_VERSION = 1

# Abort the run after this many consecutive failed crop writes.
# cv2.imwrite failures are NOT exceptions - detect_rf.py reports them as
# filtered records with reason "save_failed". Without this guard a full disk
# would let the whole run finish while silently writing no crops at all.
MAX_SAVE_FAILURES = 20

# An image that keeps failing is skipped for good after this many attempts,
# so a handful of corrupt files cannot block every future run.
MAX_IMAGE_RETRIES = 3

# Append-only data files, in the order they are flushed.
DATA_KEYS = ("crops", "filtered", "errors", "done")


class _AbortRun(RuntimeError):
    """Raised to stop the run early while still saving a checkpoint."""


# ============================================================
# Checkpoint helpers
# ============================================================
def _run_paths(output_dir):
    ckpt_dir = output_dir / "checkpoint"

    return {
        "ckpt_dir": ckpt_dir,
        "state": ckpt_dir / "state.json",
        "done": ckpt_dir / "done.txt",
        "crops": ckpt_dir / "crops.jsonl",
        "filtered": ckpt_dir / "filtered.jsonl",
        "errors": ckpt_dir / "errors.jsonl",
        "stats": output_dir / "filter_stats.json",
    }


def _build_config(sample_dir, target_classes):
    """Config that must match for a resume to be valid."""
    return {
        "sample_dir": str(sample_dir),
        "target_classes": (
            sorted(target_classes) if target_classes else None
        ),
    }


def _new_state(config):
    return {
        "version": STATE_VERSION,
        "config": config,
        # byte size of each append-only file at the last checkpoint
        "offsets": {key: 0 for key in DATA_KEYS},
        "counters": {
            "images_done": 0,
            "failed_attempts": 0,
            "accepted": 0,
            "filtered": 0,
        },
        # class_name -> int
        "crops_by_class": {},
        # class_name -> [count, confidence_sum]
        "filtered_by_class": {},
        # image filename -> number of failed attempts
        "failed_counts": {},
        "last_updated": None,
    }


def _atomic_write_json(path, obj):
    """Write JSON via tmp file + os.replace so a kill never corrupts it."""
    tmp = path.with_name(path.name + ".tmp")

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def _reset_run_files(paths):
    for key in DATA_KEYS:
        if paths[key].exists():
            paths[key].unlink()

    if paths["state"].exists():
        paths["state"].unlink()


def _load_state(paths, config):
    """Load state.json, validating version and config."""
    if not paths["state"].exists():
        return None

    try:
        with open(paths["state"], "r", encoding="utf-8") as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(
            f"Checkpoint is unreadable: {paths['state']}\n"
            f"  {type(exc).__name__}: {exc}\n"
            f"Re-run with --fresh to start over."
        )

    if state.get("version") != STATE_VERSION:
        raise SystemExit(
            f"Checkpoint version mismatch "
            f"(found {state.get('version')}, expected {STATE_VERSION}).\n"
            f"Re-run with --fresh to start over."
        )

    if state.get("config") != config:
        raise SystemExit(
            "Checkpoint config does not match this run.\n"
            f"  checkpoint: {state.get('config')}\n"
            f"  current   : {config}\n"
            "Resuming would mix results from different settings.\n"
            "Re-run with --fresh, or point --output-dir somewhere else."
        )

    # Tolerate checkpoints written before these keys existed.
    state.setdefault("failed_counts", {})

    return state


def _record_failure(state, handles, image_path, error, message, tb=None):
    """Log a failed image and return how many times it has now failed."""
    name = Path(image_path).name

    counts = state["failed_counts"]
    counts[name] = counts.get(name, 0) + 1

    state["counters"]["failed_attempts"] += 1

    record = {
        "image": image_path,
        "error": error,
        "message": message,
        "attempts": counts[name],
        "time": datetime.now().isoformat(timespec="seconds"),
    }

    if tb:
        record["traceback"] = tb

    handles["errors"].write(json.dumps(record, ensure_ascii=False) + "\n")

    return counts[name]


def _rollback(paths, state):
    """
    Truncate append-only files back to the last checkpoint.

    Anything written after the last checkpoint is discarded, so state.json
    and the data files are always exactly consistent. Those images are
    simply reprocessed. This is what prevents duplicate records.
    """
    for key in DATA_KEYS:
        path = paths[key]
        want = state["offsets"].get(key, 0)

        if not path.exists():
            if want:
                raise SystemExit(
                    f"Checkpoint expects {want} bytes in {path.name}, "
                    f"but the file is missing.\n"
                    f"Re-run with --fresh to start over."
                )
            continue

        current = path.stat().st_size

        if current < want:
            raise SystemExit(
                f"{path.name} is shorter than the checkpoint expects "
                f"({current} < {want} bytes).\n"
                f"Re-run with --fresh to start over."
            )

        if current > want:
            with open(path, "r+b") as f:
                f.truncate(want)

            print(
                f"  rollback: {path.name} "
                f"{current} -> {want} bytes"
            )


def _load_done(paths):
    """Image filenames already fully processed."""
    if not paths["done"].exists():
        return set()

    with open(paths["done"], "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _flush(paths, state, handles):
    """fsync data files, record their sizes, then write state.json."""
    for key in DATA_KEYS:
        handle = handles[key]
        handle.flush()
        os.fsync(handle.fileno())

        # Use the real file size, not tell(), so Windows newline
        # translation cannot desync the recorded offset.
        state["offsets"][key] = paths[key].stat().st_size

    state["last_updated"] = datetime.now().isoformat(timespec="seconds")

    _atomic_write_json(paths["state"], state)


def _write_stats_mirror(paths):
    """
    Rebuild filter_stats.json from the .jsonl files in the original schema.

    Streams line by line, so the full record set is never held in memory.
    """
    stats_path = paths["stats"]
    tmp = stats_path.with_name(stats_path.name + ".tmp")

    with open(tmp, "w", encoding="utf-8") as out:
        out.write("{\n")

        for key in ("crops", "filtered"):
            out.write(f'  "{key}": [\n')

            first = True
            src = paths[key]

            if src.exists():
                with open(src, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()

                        if not line:
                            continue

                        if not first:
                            out.write(",\n")

                        out.write("    " + line)
                        first = False

            out.write("\n  ]")
            out.write(",\n" if key == "crops" else "\n")

        out.write("}\n")

        out.flush()
        os.fsync(out.fileno())

    os.replace(tmp, stats_path)


def _fmt_secs(seconds):
    return str(timedelta(seconds=int(max(0, seconds))))


# ============================================================
# Batch run
# ============================================================
def run_batch(
    sample_dir=SAMPLE_DIR,
    output_dir=OUTPUT_DIR,
    target_classes=DEFAULT_TARGET_CLASSES,
    limit=None,
    fresh=False,
):
    sample_dir = Path(sample_dir)
    output_dir = Path(output_dir)

    paths = _run_paths(output_dir)

    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    paths["ckpt_dir"].mkdir(parents=True, exist_ok=True)

    config = _build_config(sample_dir, target_classes)

    # --------------------------------------------------------
    # Checkpoint / resume
    # --------------------------------------------------------
    if fresh:
        print("--fresh: clearing existing checkpoint.")
        _reset_run_files(paths)

    state = _load_state(paths, config)

    if state is None:
        state = _new_state(config)
        done = set()

        print("No checkpoint found. Starting a new run.")
    else:
        print(f"Resuming from checkpoint ({state['last_updated']}).")

        _rollback(paths, state)

        done = _load_done(paths)

        print(f"  Already processed: {len(done)} images")

    # --------------------------------------------------------
    # Find input images
    # --------------------------------------------------------
    image_paths = sorted(
        glob.glob(str(sample_dir / "*.jpg"))
        + glob.glob(str(sample_dir / "*.jpeg"))
        + glob.glob(str(sample_dir / "*.png"))
    )

    total_found = len(image_paths)

    failed_counts = state["failed_counts"]

    # Skip images already processed, and images that have failed too often
    pending = []
    give_up = 0

    for path in image_paths:
        name = Path(path).name

        if name in done:
            continue

        if failed_counts.get(name, 0) >= MAX_IMAGE_RETRIES:
            give_up += 1
            continue

        pending.append(path)

    remaining_total = len(pending)

    # Limit applies to PENDING images, so --limit 100 always processes
    # 100 new images. Use --fresh to repeat the same first 100.
    if limit is not None:
        pending = pending[:limit]

    print(f"\nTotal images found : {total_found}")
    print(f"Already done       : {len(done)}")

    if give_up:
        print(
            f"Permanently failed : {give_up} "
            f"(>= {MAX_IMAGE_RETRIES} attempts, see errors.jsonl)"
        )

    print(f"To process now     : {len(pending)}")

    print(
        f"Target classes     : "
        f"{target_classes if target_classes else 'ALL classes'}"
    )

    if total_found == 0:
        print(f"No images found in: {sample_dir}")
        return

    if not pending:
        print("\nNothing left to process.")

        _write_stats_mirror(paths)
        _print_summary(state, paths)

        return

    # --------------------------------------------------------
    # Load RF-DETR model once
    # --------------------------------------------------------
    print("\nLoading RF-DETR model...")

    model = load_detect_model()

    print("RF-DETR model loaded.")

    # --------------------------------------------------------
    # Batch processing
    # --------------------------------------------------------
    handles = {
        key: open(paths[key], "a", encoding="utf-8")
        for key in DATA_KEYS
    }

    counters = state["counters"]
    crops_by_class = state["crops_by_class"]
    filtered_by_class = state["filtered_by_class"]

    run_start = time.time()
    processed_this_run = 0
    consecutive_save_failures = 0
    interrupted = False
    aborted = None

    print(
        f"\nProcessing {len(pending)} images "
        f"(checkpoint every {CHECKPOINT_EVERY})\n"
        + "=" * 60
    )

    try:
        for i, image_path in enumerate(pending, 1):

            try:
                crop_results, filtered_log = detect_and_crop(
                    model,
                    image_path,
                    output_dir=str(output_dir),

                    # None = ALL COCO classes
                    target_classes=target_classes,
                )

            except Exception as exc:
                # One bad image must not kill an 80k-image run.
                # Not marked done, so it is retried on the next run
                # until MAX_IMAGE_RETRIES is reached.
                attempts = _record_failure(
                    state,
                    handles,
                    image_path,
                    type(exc).__name__,
                    str(exc),
                    traceback.format_exc(limit=3),
                )

                print(
                    f"  [FAIL {attempts}/{MAX_IMAGE_RETRIES}] "
                    f"{Path(image_path).name} :: "
                    f"{type(exc).__name__}: {exc}"
                )

            else:
                save_failed = sum(
                    1
                    for record in filtered_log
                    if record.get("reason") == "save_failed"
                )

                if save_failed:
                    # The crops were detected but never written to disk.
                    # Treat the whole image as failed instead of recording
                    # it as "filtered", so nothing is silently lost.
                    consecutive_save_failures += save_failed

                    attempts = _record_failure(
                        state,
                        handles,
                        image_path,
                        "save_failed",
                        f"{save_failed} crop(s) could not be written",
                    )

                    print(
                        f"  [SAVE FAIL {attempts}/{MAX_IMAGE_RETRIES}] "
                        f"{Path(image_path).name}: "
                        f"{save_failed} crop(s) not written"
                    )

                    if consecutive_save_failures >= MAX_SAVE_FAILURES:
                        raise _AbortRun(
                            f"{consecutive_save_failures} consecutive crop "
                            f"writes failed. Disk full, or output directory "
                            f"not writable?"
                        )

                else:
                    if crop_results:
                        consecutive_save_failures = 0

                    for record in crop_results:
                        handles["crops"].write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )

                        cls = record.get("class_name", "unknown")
                        crops_by_class[cls] = crops_by_class.get(cls, 0) + 1

                    for record in filtered_log:
                        handles["filtered"].write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )

                        cls = record.get("class_name", "unknown")
                        entry = filtered_by_class.setdefault(cls, [0, 0.0])
                        entry[0] += 1
                        entry[1] += float(record.get("confidence", 0.0) or 0.0)

                    counters["accepted"] += len(crop_results)
                    counters["filtered"] += len(filtered_log)

                    # Mark done only after the records are written.
                    handles["done"].write(Path(image_path).name + "\n")

                    counters["images_done"] += 1

            processed_this_run = i

            # ----------------------------------------------------
            # Progress
            # ----------------------------------------------------
            if i % PROGRESS_EVERY == 0 or i == len(pending):
                elapsed = time.time() - run_start
                rate = i / elapsed if elapsed > 0 else 0
                eta_secs = (len(pending) - i) / rate if rate > 0 else 0

                print(
                    f"[{i}/{len(pending)}] "
                    f"{i / len(pending) * 100:5.1f}%  |  "
                    f"{rate:5.1f} img/s  |  "
                    f"elapsed {_fmt_secs(elapsed)}  |  "
                    f"ETA {_fmt_secs(eta_secs)}  |  "
                    f"accepted {counters['accepted']}  "
                    f"filtered {counters['filtered']}  "
                    f"failed {counters['failed_attempts']}"
                )

            # ----------------------------------------------------
            # Checkpoint
            # ----------------------------------------------------
            if i % CHECKPOINT_EVERY == 0:
                _flush(paths, state, handles)

                print(
                    f"  checkpoint saved "
                    f"({counters['images_done']} images done overall)"
                )

    except KeyboardInterrupt:
        interrupted = True

        print("\nInterrupt received (Ctrl+C). Saving checkpoint...")

    except _AbortRun as exc:
        aborted = str(exc)

        print(f"\nABORTING: {exc}")
        print("Saving checkpoint...")

    finally:
        _flush(paths, state, handles)

        for handle in handles.values():
            handle.close()

        print(
            f"Checkpoint saved: {processed_this_run} images this run, "
            f"{counters['images_done']} done overall."
        )

    # --------------------------------------------------------
    # Rebuild filter_stats.json from the .jsonl files
    # --------------------------------------------------------
    _write_stats_mirror(paths)

    _print_summary(state, paths)

    if aborted:
        print("\n" + "!" * 60)
        print(f"RUN ABORTED: {aborted}")
        print(
            "Free up disk space (or fix the output directory), then re-run "
            "the same command. The affected images were NOT marked done, so "
            "they will be retried."
        )
        print("!" * 60)

        raise SystemExit(1)

    if interrupted:
        print(
            "\nRun was interrupted. Re-run the same command to continue "
            "from the last checkpoint."
        )
        return

    still_left = remaining_total - processed_this_run

    if still_left > 0:
        print(
            f"\n{still_left} images still unprocessed "
            f"(--limit reached). Re-run to continue."
        )


# ============================================================
# Summary / statistics
# ============================================================
def _print_summary(state, paths):
    counters = state["counters"]
    crops_by_class = state["crops_by_class"]
    filtered_by_class = state["filtered_by_class"]

    accepted_total = counters["accepted"]
    filtered_total = counters["filtered"]
    total_detected = accepted_total + filtered_total

    print("\n" + "=" * 60)

    give_up = sum(
        1
        for count in state["failed_counts"].values()
        if count >= MAX_IMAGE_RETRIES
    )

    print(f"Images processed : {counters['images_done']}")
    print(f"Failed attempts  : {counters['failed_attempts']}")

    if give_up:
        print(
            f"Given up on      : {give_up} images "
            f"(>= {MAX_IMAGE_RETRIES} attempts)"
        )
    print(f"Accepted crops   : {accepted_total}")
    print(f"Filtered crops   : {filtered_total}")

    if total_detected > 0:
        filter_rate = filtered_total / total_detected * 100

        print(f"Overall filter rate: {filter_rate:.1f}%")

    # --------------------------------------------------------
    # Statistics by class
    # --------------------------------------------------------
    all_classes = set(crops_by_class.keys()) | set(filtered_by_class.keys())

    print(f"\nDetected class types: {len(all_classes)}")

    print("--- Filtering statistics by class ---")

    for cls in sorted(
        all_classes,
        key=lambda c: -crops_by_class.get(c, 0),
    ):
        accepted = crops_by_class.get(cls, 0)

        skipped_count, skipped_conf_sum = filtered_by_class.get(cls, [0, 0.0])

        total = accepted + skipped_count

        rate = skipped_count / total * 100 if total > 0 else 0

        avg_conf = (
            skipped_conf_sum / skipped_count if skipped_count else 0
        )

        print(
            f"  {cls:15s}: "
            f"accepted {accepted:6d} / "
            f"filtered {skipped_count:6d} "
            f"({rate:5.1f}%), "
            f"avg filtered conf: {avg_conf:.2f}"
        )

    print(f"\nDetailed log saved to: {paths['stats']}")
    print(f"Checkpoint directory : {paths['ckpt_dir']}")

    if paths["errors"].exists() and paths["errors"].stat().st_size > 0:
        print(f"Failed images logged : {paths['errors']}")


# ============================================================
# Run
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="RF-DETR batch detection with checkpoint/resume."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help=(
            "Number of UNPROCESSED images to handle this run "
            "(default: 100, smoke test)."
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every remaining image (ignores --limit).",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the existing checkpoint and start from scratch.",
    )

    parser.add_argument(
        "--sample-dir",
        default=str(SAMPLE_DIR),
        help="Input image directory.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Output directory for crops, checkpoint and stats.",
    )

    parser.add_argument(
        "--forensic",
        action="store_true",
        help="Use FORENSIC_TARGET_CLASSES instead of all COCO classes.",
    )

    args = parser.parse_args()

    run_batch(
        sample_dir=args.sample_dir,
        output_dir=args.output_dir,
        target_classes=(
            FORENSIC_TARGET_CLASSES if args.forensic
            else DEFAULT_TARGET_CLASSES
        ),
        limit=None if args.all else args.limit,
        fresh=args.fresh,
    )


if __name__ == "__main__":

    # --------------------------------------------------------
    # Step 1: 100-image smoke test (default)
    #   python RF-DETR_batch.py
    #
    # Step 2: next 500 images
    #   python RF-DETR_batch.py --limit 500
    #
    # Step 3: full COCO train2017
    #   python RF-DETR_batch.py --all
    #
    # Interrupted? Just run the same command again.
    # Want to start over?
    #   python RF-DETR_batch.py --all --fresh
    # --------------------------------------------------------
    main()