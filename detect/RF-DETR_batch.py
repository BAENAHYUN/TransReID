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
each run, in the original {"crops": [...], "filtered": [...]} schema. The
rebuild also applies the duplicate_detection_id safety net (see
_write_stats_mirror), and can be run on its own with --rebuild-stats.

Identity
--------
dataset_id is the logical label for the input directory. Every record carries
image_id = "<dataset_id>/<filename>", and detect_rf.py derives both the crop
filename and detection_id from that logical id rather than from an absolute
path. That is what makes the resulting Qdrant point IDs reproducible on a
different machine, drive letter or project location.

Because dataset_id feeds image_id -> detection_id -> Qdrant point ID, it is
part of the checkpoint fingerprint. Changing it invalidates a resume.

source is the provenance label written into the existing integrated-DB payload
field (for example "COCO" for this default image ingest path). It does not
participate in identity generation, but it is still part of the checkpoint
fingerprint so a resumed run cannot mix different provenance labels.

Detection settings vs DB settings
---------------------------------
Minimum crop size lives here, not in pipeline.yaml. pipeline.yaml is the DB
build/search contract, read by build_db.py and the search scripts; crop size
decides which detections become files on disk in the first place. Keeping it
on the CLI means the checkpoint records the exact rule a crop directory was
produced under, which a yaml file edited later cannot do.
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
#   TransReID/detect/RF-DETR_batch.py
#
# Project root:
#   TransReID/
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

# Allow imports from the project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# detect_rf.py lives next to this script in detect/.
# The minimum crop sizes are imported rather than duplicated so the defaults
# have exactly one definition site.
from detect.detect_rf import (
    DEFAULT_DETECTOR_SPEC,
    load_detector,
    detect_and_crop,
    validate_conf_threshold,
    FORENSIC_TARGET_CLASSES,
    MIN_PERSON_CROP_WIDTH,
    MIN_PERSON_CROP_HEIGHT,
    MIN_OBJECT_CROP_WIDTH,
    MIN_OBJECT_CROP_HEIGHT,
)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
# COCO train2017 dataset
SAMPLE_DIR = Path(r"C:\datasets\coco\train2017")

OUTPUT_DIR = ROOT / "data" / "crops"


# ------------------------------------------------------------
# Dataset identity
# ------------------------------------------------------------
# Stable logical name for the input directory. Deliberately not derived from
# sample_dir: renaming or moving the dataset folder must not change any
# image_id, and therefore must not change any Qdrant point ID.
DEFAULT_DATASET_ID = "coco_train2017"

# Provenance label written to the existing integrated-DB "source" field.
# This is a default, not a detector assumption: callers can override it with
# --source without changing detect_rf.py.
DEFAULT_SOURCE = "COCO"


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

# conf_threshold 키가 없는 구형 checkpoint 는 이 값으로 만들어졌다 (당시 detect_rf.CONF_THRESHOLD).
# detect_rf.CONF_THRESHOLD 를 참조하지 않는다 — 그 상수가 바뀌어도 구형 상태의 해석은 고정이어야 한다.
LEGACY_CONF_THRESHOLD = 0.5

# detector 키가 없는 구형 checkpoint 는 RF-DETR Medium(전체 class; 필터는 target_classes) 으로 만들어졌다.
LEGACY_DETECTOR = {
    "module": "detect.detectors.rfdetr_detector",
    "class": "RFDETRDetector",
    "params": {"filter_forensic": False},
}

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


def _normalize_dataset_id(dataset_id):
    """
    Canonical form of the logical dataset label.

    " coco_train2017 ", "/coco_train2017/" and "coco_train2017\\" all describe
    the same dataset, but would produce different image_id strings and a
    different checkpoint fingerprint. Normalize before anything reads it.
    """
    normalized = (
        str(dataset_id)
        .strip()
        .replace("\\", "/")
        .strip("/")
    )

    if not normalized:
        raise ValueError(
            "dataset_id must not be empty."
        )

    return normalized


def _normalize_source(source):
    """
    Canonical form of the provenance label stored in the existing payload.

    source is metadata rather than part of detection identity, so only trim
    accidental surrounding whitespace. Preserve the caller's spelling/case.
    """
    if source is None:
        raise ValueError(
            "source must not be empty."
        )

    normalized = str(source).strip()

    if not normalized:
        raise ValueError(
            "source must not be empty."
        )

    return normalized


def _build_min_crop_size(
    min_person_width,
    min_person_height,
    min_object_width,
    min_object_height,
):
    """
    Normalize the crop size rule into a JSON-round-trippable structure.

    The values are stored as lists, not tuples: state.json turns tuples into
    lists on read, and the config comparison in _load_state is an equality
    check. A tuple here would make every resume fail.
    """
    sizes = {
        "person": [
            int(min_person_width),
            int(min_person_height),
        ],
        "object": [
            int(min_object_width),
            int(min_object_height),
        ],
    }

    for kind, (width, height) in sizes.items():
        if width < 0 or height < 0:
            raise ValueError(
                f"min crop size for '{kind}' must not be negative: "
                f"{width}x{height}"
            )

    return sizes


def _detector_summary(spec):
    """checkpoint config 에 넣는 검출기 식별자 (module / class / params).

    conf_threshold 는 별도 키라 뺀다. filter_forensic 은 이미지 경로에서 항상 False 로 강제되므로
    (클래스 필터는 target_classes) 그 값으로 고정해 둔다 — yaml 에 true 라 적혀 있어도 결과는 같다.
    """
    params = {
        k: v for k, v in (spec.get("params") or {}).items()
        if k != "conf_threshold"
    }
    params["filter_forensic"] = False
    return {
        "module": str(spec["module"]),
        "class": str(spec["class"]),
        "params": params,
    }


def _build_config(
    sample_dir,
    target_classes,
    dataset_id,
    source,
    min_crop_size,
    conf_threshold=LEGACY_CONF_THRESHOLD,
    detector=None,
):
    """
    Config that must match for a resume to be valid.

    dataset_id is included because it flows into image_id, detection_id and
    ultimately the Qdrant point ID.

    source does not affect point identity, but it affects the metadata written
    for every record. A resume with a different source would silently mix
    provenance labels in the same crops.jsonl/filter_stats.json.

    min_crop_size decides which detections become crops at all, so resuming
    across a change would leave one output directory holding crops produced
    under two different acceptance rules.

    save_annotated is deliberately excluded: it only controls a debug
    visualization and does not affect which crops or records are produced.
    """
    return {
        "sample_dir": str(sample_dir),
        "dataset_id": dataset_id,
        "source": source,
        "min_crop_size": min_crop_size,
        "target_classes": (
            sorted(target_classes) if target_classes else None
        ),
        # 임계값이 다르면 같은 이미지에서 다른 crop 집합이 나오므로 resume 호환 조건이다.
        "conf_threshold": float(conf_threshold),
        # 검출기(RF-DETR / YOLO26 …)가 다르면 검출 자체가 달라진다. 구형 checkpoint 는 RF-DETR 로 간주.
        "detector": detector if detector is not None else LEGACY_DETECTOR,
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

    saved_config = state.get("config")
    if isinstance(saved_config, dict) and "conf_threshold" not in saved_config:
        # 구형 checkpoint(키 없음)는 당시 고정값으로 간주해 메모리에서만 채운다.
        # 키는 있는데 null/문자열 같은 이상한 값이면 그대로 비교해 불일치로 드러낸다.
        saved_config = dict(saved_config, conf_threshold=LEGACY_CONF_THRESHOLD)
    if isinstance(saved_config, dict) and "detector" not in saved_config:
        saved_config = dict(saved_config, detector=LEGACY_DETECTOR)

    if saved_config != config:
        raise SystemExit(
            "Checkpoint config does not match this run.\n"
            f"  checkpoint: {saved_config}\n"
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


def _begin_image_write(paths, handles):
    """이미지 하나의 기록을 시작하기 직전 세 데이터 파일의 크기를 기억한다."""
    sizes = {}
    for key in ("crops", "filtered", "done"):
        handles[key].flush()
        sizes[key] = paths[key].stat().st_size
    return sizes


def _rollback_partial(paths, handles, sizes):
    """
    기록 도중 인터럽트된 이미지의 부분 기록을 지운다.

    crops 만 쓰이고 done 이 없는 상태가 checkpoint 로 확정되면, 다음 실행이 그 이미지를
    다시 처리해 원시 로그와 통계가 중복된다. 세 파일을 기록 직전 크기로 되돌린다.
    append 모드 핸들은 다음 write 때 O_APPEND 로 새 끝에 붙으므로 그대로 써도 된다.
    """
    for key, want in sizes.items():
        handles[key].flush()
        current = paths[key].stat().st_size
        if current > want:
            with open(paths[key], "r+b") as f:
                f.truncate(want)
            print(f"  rollback(partial image): {paths[key].name} {current} -> {want} bytes")


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

    Streams line by line. The only thing held in memory is one small entry
    per accepted crop: detection_id -> (confidence, line_no, crop_path).

    duplicate_detection_id safety net
    ---------------------------------
    detect_rf.py drops same-id predictions BEFORE a crop is written, so new
    crops.jsonl lines never collide. Checkpoints written before that fix can
    still hold a few collisions: RF-DETR has no NMS, so two queries may
    predict the same box, and after integer clipping they collapse to one
    detection_id. Those crops are pixel-identical (same source, same box).

    This rebuild keeps the highest-confidence line per detection_id (ties:
    the first line) and moves the others into "filtered" with
    reason = "duplicate_detection_id", kept_confidence and kept_crop_path.
    The dropped record keeps its own crop_path so the now-orphaned file can
    be located; nothing is deleted here. build_db.py's global uniqueness
    check is unchanged and stays strict — this just stops it from tripping
    on data the detector should never have emitted twice.

    Lines without a detection_id (legacy) are never collapsed.

    Returns {"crops": kept, "filtered": written, "dropped_duplicates": n}.
    """
    stats_path = paths["stats"]
    tmp = stats_path.with_name(stats_path.name + ".tmp")

    crops_src = paths["crops"]
    filtered_src = paths["filtered"]

    # ---- pass 1: winner per detection_id ----
    winners = {}

    if crops_src.exists():
        with open(crops_src, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)
                did = record.get("detection_id")

                if not did:
                    continue

                confidence = float(record.get("confidence", 0.0) or 0.0)
                current = winners.get(did)

                if current is None or confidence > current[0]:
                    winners[did] = (
                        confidence,
                        line_no,
                        record.get("crop_path"),
                    )

    dropped = []
    kept = 0
    filtered_written = 0

    with open(tmp, "w", encoding="utf-8") as out:
        out.write("{\n")

        # ---- crops: winners only ----
        out.write('  "crops": [\n')
        first = True

        if crops_src.exists():
            with open(crops_src, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f):
                    line = line.strip()

                    if not line:
                        continue

                    record = json.loads(line)
                    did = record.get("detection_id")

                    if did and winners[did][1] != line_no:
                        loser = dict(record)
                        loser["reason"] = "duplicate_detection_id"
                        loser["kept_confidence"] = winners[did][0]
                        loser["kept_crop_path"] = winners[did][2]
                        dropped.append(loser)
                        continue

                    if not first:
                        out.write(",\n")

                    out.write("    " + line)
                    first = False
                    kept += 1

        out.write("\n  ],\n")

        # ---- filtered: original lines, then dropped duplicates ----
        out.write('  "filtered": [\n')
        first = True

        if filtered_src.exists():
            with open(filtered_src, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    # 잘린/깨진 줄이 filter_stats.json 에 그대로 복사되면 미러 자체가
                    # 잘못된 JSON 이 된다. 여기서 실패시킨다 (rollback 뒤 재실행으로 해결).
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise SystemExit(
                            f"{filtered_src.name} 에 깨진 JSON 줄이 있습니다: {exc}. "
                            "마지막 checkpoint 이후 강제 종료의 잔여물이면 같은 명령을 "
                            "다시 실행(rollback)하거나 --fresh 로 시작하세요."
                        ) from exc

                    if not first:
                        out.write(",\n")

                    out.write("    " + line)
                    first = False
                    filtered_written += 1

        for loser in dropped:
            if not first:
                out.write(",\n")

            out.write("    " + json.dumps(loser, ensure_ascii=False))
            first = False
            filtered_written += 1

        out.write("\n  ]\n")
        out.write("}\n")

        out.flush()
        os.fsync(out.fileno())

    os.replace(tmp, stats_path)

    if dropped:
        print(
            f"  duplicate_detection_id: {len(dropped)} crop record(s) moved "
            f"to filtered (pixel-identical duplicates; higher confidence kept)."
        )
        for loser in dropped[:5]:
            print(f"    {loser.get('detection_id')}")
            print(f"      kept   : {loser.get('kept_crop_path')}")
            print(f"      dropped: {loser.get('crop_path')}")

    return {
        "crops": kept,
        "filtered": filtered_written,
        "dropped_duplicates": len(dropped),
    }


def _fmt_secs(seconds):
    return str(timedelta(seconds=int(max(0, seconds))))


# ============================================================
# Batch run
# ============================================================
def run_batch(
    sample_dir=SAMPLE_DIR,
    output_dir=OUTPUT_DIR,
    dataset_id=DEFAULT_DATASET_ID,
    source=DEFAULT_SOURCE,
    target_classes=DEFAULT_TARGET_CLASSES,
    min_person_width=MIN_PERSON_CROP_WIDTH,
    min_person_height=MIN_PERSON_CROP_HEIGHT,
    min_object_width=MIN_OBJECT_CROP_WIDTH,
    min_object_height=MIN_OBJECT_CROP_HEIGHT,
    save_annotated=False,
    limit=None,
    fresh=False,
    conf_threshold=None,
    detector_spec=None,
):
    sample_dir = Path(sample_dir)
    output_dir = Path(output_dir)

    # Normalize before any of these reach checkpoint metadata or records.
    dataset_id = _normalize_dataset_id(dataset_id)
    source = _normalize_source(source)
    # None 이면 구형과 같은 0.5. 값은 여기서 한 번 검증되고 checkpoint config 에 들어간다.
    conf_threshold = validate_conf_threshold(
        LEGACY_CONF_THRESHOLD if conf_threshold is None else conf_threshold
    )
    # None 이면 내장 RF-DETR Medium. 식별자는 checkpoint config 에 들어가 검출기 교체 후 resume 을 막는다.
    detector_spec = dict(detector_spec or DEFAULT_DETECTOR_SPEC)
    detector_summary = _detector_summary(detector_spec)

    min_crop_size = _build_min_crop_size(
        min_person_width,
        min_person_height,
        min_object_width,
        min_object_height,
    )

    paths = _run_paths(output_dir)

    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    paths["ckpt_dir"].mkdir(parents=True, exist_ok=True)

    config = _build_config(
        sample_dir,
        target_classes,
        dataset_id,
        source,
        min_crop_size,
        conf_threshold,
        detector_summary,
    )

    # --------------------------------------------------------
    # Checkpoint / resume
    # --------------------------------------------------------
    if fresh:
        print("--fresh: clearing existing checkpoint.")
        _reset_run_files(paths)

    state = _load_state(paths, config)

    if state is None:
        # state.json 은 없는데 데이터 파일이 남아 있으면 첫 checkpoint 전에 강제 종료된
        # 잔여물이다. 그 위에 append 하면 원시 로그와 통계가 어긋나고 잘린 마지막 줄이
        # JSONL 파싱을 깨뜨린다. 이어 쓰지 않고 멈춘다.
        leftovers = [
            paths[key].name
            for key in DATA_KEYS
            if paths[key].exists() and paths[key].stat().st_size > 0
        ]
        if leftovers:
            raise SystemExit(
                "checkpoint(state.json) 가 없는데 기록 파일이 남아 있습니다: "
                f"{', '.join(leftovers)} ({paths['ckpt_dir']}).\n"
                "첫 checkpoint 전에 중단된 실행의 잔여물입니다. --fresh 로 지우고 "
                "시작하거나, 다른 --output-dir 을 쓰세요."
            )

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

    # dataset_id feeds image_id -> detection_id -> Qdrant point ID.
    print(f"Dataset ID         : {dataset_id}")
    print(f"Source             : {source}")
    print(f"Conf threshold     : {conf_threshold}")
    print(f"Detector           : {detector_spec['module']}.{detector_spec['class']}")

    print(
        f"Min crop size      : person "
        f"{min_crop_size['person'][0]}x{min_crop_size['person'][1]}, "
        f"object "
        f"{min_crop_size['object'][0]}x{min_crop_size['object'][1]}"
    )

    print(f"Save annotated     : {save_annotated}")

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
    # Load the detector plugin once
    # --------------------------------------------------------
    print(f"\nLoading detector {detector_spec['module']}.{detector_spec['class']} ...")

    # filter_forensic=False: 클래스 필터는 detect_and_crop 의 target_classes 가 한다 (yaml 값과 무관).
    detector = load_detector(
        detector_spec,
        conf_threshold=conf_threshold,
        filter_forensic=False,
    )

    print("Detector loaded.")

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
    # 기록 중인 이미지의 기록 직전 파일 크기. None 이면 진행 중인 기록이 없다.
    pending_write = None

    print(
        f"\nProcessing {len(pending)} images "
        f"(checkpoint every {CHECKPOINT_EVERY})\n"
        + "=" * 60
    )

    try:
        for i, image_path in enumerate(pending, 1):

            # Logical identity of the source image. Independent of drive
            # letter, project location and OS path separators.
            image_id = (
                f"{dataset_id}/"
                f"{Path(image_path).name}"
            )

            try:
                crop_results, filtered_log = detect_and_crop(
                    detector,
                    image_path,
                    output_dir=str(output_dir),

                    # None = ALL COCO classes
                    target_classes=target_classes,

                    image_id=image_id,
                    source=source,

                    min_person_width=min_crop_size["person"][0],
                    min_person_height=min_crop_size["person"][1],
                    min_object_width=min_crop_size["object"][0],
                    min_object_height=min_crop_size["object"][1],

                    save_annotated=save_annotated,
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

                    # 이미지 하나의 crops / filtered / done 기록은 한 트랜잭션이다.
                    # 도중에 인터럽트되면 finally 의 _rollback_partial 이 세 파일을 기록
                    # 직전 크기로 되돌려, 절반만 쓰인 이미지가 checkpoint 로 확정되지 않게
                    # 한다. (done 이 없으니 그 이미지는 다음 실행에서 통째로 재처리된다.)
                    crops_text = "".join(
                        json.dumps(record, ensure_ascii=False) + "\n"
                        for record in crop_results
                    )
                    filtered_text = "".join(
                        json.dumps(record, ensure_ascii=False) + "\n"
                        for record in filtered_log
                    )

                    pending_write = _begin_image_write(paths, handles)
                    handles["crops"].write(crops_text)
                    handles["filtered"].write(filtered_text)
                    # Mark done only after the records are written.
                    handles["done"].write(Path(image_path).name + "\n")
                    pending_write = None

                    # 통계는 기록이 끝난 뒤에만 갱신한다 (부분 기록과 어긋나지 않게).
                    for record in crop_results:
                        cls = record.get("class_name", "unknown")
                        crops_by_class[cls] = crops_by_class.get(cls, 0) + 1

                    for record in filtered_log:
                        cls = record.get("class_name", "unknown")
                        entry = filtered_by_class.setdefault(cls, [0, 0.0])
                        entry[0] += 1
                        entry[1] += float(record.get("confidence", 0.0) or 0.0)

                    counters["accepted"] += len(crop_results)
                    counters["filtered"] += len(filtered_log)
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
        if pending_write is not None:
            # 기록 도중 중단됨 — 부분 기록을 되돌린 뒤 checkpoint 를 확정한다.
            _rollback_partial(paths, handles, pending_write)
            pending_write = None

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
def _resolve_conf_threshold(cli_value, config_path):
    """
    유효 검출 임계값과 출처. 우선순위: --conf-threshold > yaml detector.params.conf_threshold
    (--config 가 주어졌을 때) > LEGACY_CONF_THRESHOLD(0.5).
    yaml 이 없거나 깨졌으면 즉시 오류 (조용한 기본값 fallback 은 하지 않는다).
    """
    if cli_value is not None:
        return validate_conf_threshold(cli_value), "cli"

    if config_path:
        import yaml  # 지연 import: --config 를 안 쓰는 실행에는 필요 없다

        path = Path(config_path)
        if not path.is_file():
            raise SystemExit(f"--config 파일이 없습니다: {path}")
        with open(path, "r", encoding="utf-8-sig") as f:
            raw = yaml.safe_load(f) or {}
        value = (
            ((raw.get("detector") or {}).get("params") or {}).get("conf_threshold")
            if isinstance(raw, dict) else None
        )
        if value is not None:
            return validate_conf_threshold(value), f"yaml:{path.name}"
        return LEGACY_CONF_THRESHOLD, f"default(yaml {path.name} has no detector.params.conf_threshold)"

    return LEGACY_CONF_THRESHOLD, "default"


def _resolve_detector_spec(detector_config, config_path):
    """
    검출기 플러그인 spec 과 출처. 우선순위: --detector-config 의 detector: 블록 > --config 의 detector: 블록
    > 내장 RF-DETR Medium. --detector-config 를 줬는데 블록이 없으면 오류 (조용한 fallback 없음).
    """
    for path_str, label in ((detector_config, "detector-config"), (config_path, "config")):
        if not path_str:
            continue

        import yaml  # 지연 import

        path = Path(path_str)
        if not path.is_file():
            raise SystemExit(f"--{label} 파일이 없습니다: {path}")
        with open(path, "r", encoding="utf-8-sig") as f:
            raw = yaml.safe_load(f) or {}
        block = raw.get("detector") if isinstance(raw, dict) else None
        if isinstance(block, dict) and block.get("module") and block.get("class"):
            return (
                {
                    "module": str(block["module"]),
                    "class": str(block["class"]),
                    "params": dict(block.get("params") or {}),
                },
                f"yaml:{path.name}",
            )
        if label == "detector-config":
            raise SystemExit(
                f"--detector-config 에 detector: module/class 블록이 없습니다: {path}"
            )

    return dict(DEFAULT_DETECTOR_SPEC), "default(RF-DETR Medium)"


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
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=(
            "Stable logical dataset identifier used in image IDs "
            f"(default: {DEFAULT_DATASET_ID}). Changing it invalidates "
            "an existing checkpoint."
        ),
    )

    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=(
            "Provenance label written to the existing integrated-DB "
            f"'source' field (default: {DEFAULT_SOURCE}). Changing it "
            "invalidates an existing checkpoint."
        ),
    )

    parser.add_argument(
        "--min-person-width",
        type=int,
        default=MIN_PERSON_CROP_WIDTH,
        help=(
            "Minimum width for a person crop "
            f"(default: {MIN_PERSON_CROP_WIDTH}). Changing it invalidates "
            "an existing checkpoint."
        ),
    )

    parser.add_argument(
        "--min-person-height",
        type=int,
        default=MIN_PERSON_CROP_HEIGHT,
        help=(
            "Minimum height for a person crop "
            f"(default: {MIN_PERSON_CROP_HEIGHT}). Changing it invalidates "
            "an existing checkpoint."
        ),
    )

    parser.add_argument(
        "--min-object-width",
        type=int,
        default=MIN_OBJECT_CROP_WIDTH,
        help=(
            "Minimum width for a non-person crop "
            f"(default: {MIN_OBJECT_CROP_WIDTH}). Small objects are "
            "legitimately small, so this must stay well below the person "
            "threshold."
        ),
    )

    parser.add_argument(
        "--min-object-height",
        type=int,
        default=MIN_OBJECT_CROP_HEIGHT,
        help=(
            "Minimum height for a non-person crop "
            f"(default: {MIN_OBJECT_CROP_HEIGHT})."
        ),
    )

    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help=(
            "Write a bbox visualization per image under "
            "<output-dir>/detected_full/. Off by default: at 80k images "
            "this doubles the file count and the write time."
        ),
    )

    parser.add_argument(
        "--forensic",
        action="store_true",
        help="Use FORENSIC_TARGET_CLASSES instead of all COCO classes.",
    )

    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help=(
            "RF-DETR detection confidence threshold (0..1). Priority: this flag "
            "> detector.params.conf_threshold of --config > 0.5. Changing it "
            "invalidates an existing checkpoint."
        ),
    )

    parser.add_argument(
        "--config",
        default=None,
        help=(
            "pipeline yaml whose detector.params.conf_threshold is used when "
            "--conf-threshold is not given (e.g. pipeline.yaml)."
        ),
    )

    parser.add_argument(
        "--detector-config",
        default=None,
        help=(
            "yaml whose detector: {module, class, params} block selects the "
            "detector plugin (e.g. pipeline_tracking_yolo26.yaml for YOLO26). "
            "Default: the detector: block of --config, else the built-in "
            "RF-DETR Medium. Its params.conf_threshold is used when "
            "--conf-threshold is not given. Changing the detector invalidates "
            "an existing checkpoint."
        ),
    )

    parser.add_argument(
        "--rebuild-stats",
        action="store_true",
        help=(
            "Only rebuild <output-dir>/filter_stats.json from the checkpoint "
            ".jsonl files, applying the duplicate_detection_id safety net. "
            "No detection, no model load, checkpoint files untouched."
        ),
    )

    args = parser.parse_args()

    if args.rebuild_stats:
        paths = _run_paths(Path(args.output_dir))

        if not paths["crops"].exists():
            raise SystemExit(
                f"Nothing to rebuild from: {paths['crops']} does not exist."
            )

        # 마지막 checkpoint 이후의 미확정 로그(강제 종료 잔여물)는 통계에 넣지 않는다.
        if paths["state"].exists():
            with open(paths["state"], "r", encoding="utf-8") as f:
                saved_state = json.load(f)
            if isinstance(saved_state, dict) and isinstance(saved_state.get("offsets"), dict):
                _rollback(paths, saved_state)

        result = _write_stats_mirror(paths)

        print(
            f"filter_stats.json rebuilt: crops={result['crops']} "
            f"filtered={result['filtered']} "
            f"dropped_duplicates={result['dropped_duplicates']}"
        )
        print(f"  -> {paths['stats']}")
        return

    detector_spec, detector_origin = _resolve_detector_spec(
        args.detector_config,
        args.config,
    )

    conf_threshold, conf_origin = _resolve_conf_threshold(
        args.conf_threshold,
        args.detector_config or args.config,
    )

    # 유효 설정과 출처. COCO 기본값을 그대로 쓰고 있으면 눈에 띄게 표시한다 —
    # 다른 데이터셋을 처리하면서 dataset_id/source 기본값을 놓치면 출처가 오기록된다.
    def _origin(value, default):
        return "cli" if str(value) != str(default) else "default(COCO)"

    print("Effective settings :")
    print(f"  sample_dir     = {args.sample_dir}  [{_origin(args.sample_dir, SAMPLE_DIR)}]")
    print(f"  output_dir     = {args.output_dir}  [{_origin(args.output_dir, OUTPUT_DIR)}]")
    print(f"  dataset_id     = {args.dataset_id}  [{_origin(args.dataset_id, DEFAULT_DATASET_ID)}]")
    print(f"  source         = {args.source}  [{_origin(args.source, DEFAULT_SOURCE)}]")
    print(f"  conf_threshold = {conf_threshold}  [{conf_origin}]")
    print(f"  detector       = {detector_spec['module']}.{detector_spec['class']}  [{detector_origin}]")

    run_batch(
        sample_dir=args.sample_dir,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        source=args.source,
        target_classes=(
            FORENSIC_TARGET_CLASSES if args.forensic
            else DEFAULT_TARGET_CLASSES
        ),
        min_person_width=args.min_person_width,
        min_person_height=args.min_person_height,
        min_object_width=args.min_object_width,
        min_object_height=args.min_object_height,
        save_annotated=args.save_annotated,
        limit=None if args.all else args.limit,
        fresh=args.fresh,
        conf_threshold=conf_threshold,
        detector_spec=detector_spec,
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