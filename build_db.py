from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

from config import PipelineConfig
from registry import EmbedderRegistry
from router import Router
from rfdetr_adapter import from_rfdetr
from qdrant_store import QdrantStore


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

CONFIG_PATH = ROOT / "pipeline.yaml"
STATS_PATH = ROOT / "data" / "crops" / "filter_stats.json"

CHECKPOINT_DIR = ROOT / "data" / "embedding_checkpoint"
STATE_PATH = CHECKPOINT_DIR / "state.json"


# ============================================================
# Helpers
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

        f.flush()
        os.fsync(f.fileno())

    os.replace(
        tmp,
        path,
    )


def format_seconds(seconds: float) -> str:
    seconds = max(
        0,
        int(seconds),
    )

    h, rem = divmod(
        seconds,
        3600,
    )

    m, s = divmod(
        rem,
        60,
    )

    return f"{h:02d}:{m:02d}:{s:02d}"


def is_person(det, cfg) -> bool:
    labels = {
        str(x).lower()
        for x in cfg.person_labels
    }

    return (
        str(det.label).lower()
        in labels
    )


def make_collection_configs(cfg: PipelineConfig):
    """
    하나의 전체 pipeline 설정에서 Qdrant 컬렉션별 schema를 분리한다.

    forensic_person:
      - siglip2
      - irra
      - solider

    forensic_object:
      - siglip2
      - dinov2
    """
    person_retrievers = {
        name: spec
        for name, spec in cfg.retrievers.items()
        if spec.accepts_person()
    }

    object_retrievers = {
        name: spec
        for name, spec in cfg.retrievers.items()
        if spec.accepts_object()
    }

    person_cfg = replace(
        cfg,
        collection="forensic_person",
        retrievers=person_retrievers,
    )

    object_cfg = replace(
        cfg,
        collection="forensic_object",
        retrievers=object_retrievers,
    )

    return person_cfg, object_cfg


def expected_vector_names(det, cfg):
    """
    pipeline.yaml ??scope 기�??�로
    ??detection??반드???�어???�는 vector ?�름.
    """

    person = is_person(
        det,
        cfg,
    )

    expected = set()

    for name, spec in cfg.retrievers.items():

        scope = str(
            spec.scope
        ).lower()

        if scope == "all":
            expected.add(name)

        elif (
            scope == "person"
            and person
        ):
            expected.add(name)

        elif (
            scope == "object"
            and not person
        ):
            expected.add(name)

    return expected


def validate_router_result(
    detections,
    vectors,
    cfg,
):
    """
    Router가 ?�못 분기???�태�?
    45�?건을 ?�재?�는 것을 방�??�다.
    """

    if not isinstance(
        vectors,
        (list, tuple),
    ):
        raise TypeError(
            "router.embed() 반환값이 "
            "list/tuple???�닙?�다: "
            f"{type(vectors)}"
        )

    if (
        len(detections)
        != len(vectors)
    ):
        raise RuntimeError(
            "Detection / vector 개수 불일�? "
            f"{len(detections)} "
            f"!= {len(vectors)}"
        )

    for i, (
        det,
        vector_map,
    ) in enumerate(
        zip(
            detections,
            vectors,
        )
    ):

        if not isinstance(
            vector_map,
            dict,
        ):
            raise TypeError(
                f"vectors[{i}]가 dict가 ?�닙?�다: "
                f"{type(vector_map)}"
            )

        actual = set(
            vector_map.keys()
        )

        expected = (
            expected_vector_names(
                det,
                cfg,
            )
        )

        if actual != expected:
            raise RuntimeError(
                "\nRouter routing ?�류\n"
                f"index    : {i}\n"
                f"label    : {det.label}\n"
                f"expected : {sorted(expected)}\n"
                f"actual   : {sorted(actual)}"
            )


# ============================================================
# State
# ============================================================

def make_run_info(
    batch_size: int,
    total: int,
):
    stats_stat = STATS_PATH.stat()

    return {
        "pipeline_sha256":
            sha256_file(
                CONFIG_PATH
            ),

        "stats_size":
            stats_stat.st_size,

        "stats_mtime_ns":
            stats_stat.st_mtime_ns,

        "total":
            total,

        "batch_size":
            batch_size,
    }


def load_checkpoint(
    run_info: dict,
):
    if not STATE_PATH.exists():
        return {
            "run_info": run_info,
            "next_index": 0,
        }

    with open(
        STATE_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        state = json.load(f)

    old = state.get(
        "run_info"
    )

    if old != run_info:
        raise RuntimeError(
            "기존 embedding checkpoint가 "
            "?�재 pipeline/data ?�정�??�릅?�다.\n"
            "처음부???�시 ?�려�?--fresh �??�용?�세??"
        )

    return state


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Full embedding + Qdrant build"
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Router outer batch size (default: 128)",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Embedding checkpoint�???��?�고 0부???�시 처리",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Qdrant collection????�� ???�로 ?�성",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size??1 ?�상?�어???�니??"
        )

    # --------------------------------------------------------
    # 1. Config
    # --------------------------------------------------------

    print("=" * 70)
    print("LOAD CONFIG")
    print("=" * 70)

    cfg = PipelineConfig.load(
        str(CONFIG_PATH)
    )

    person_cfg, object_cfg = make_collection_configs(cfg)

    print(
        "collections:",
        {
            "person": person_cfg.collection,
            "object": object_cfg.collection,
        },
    )

    print(
        "person vectors:",
        list(person_cfg.retrievers.keys()),
    )

    print(
        "object vectors:",
        list(object_cfg.retrievers.keys()),
    )

    # --------------------------------------------------------
    # 2. RF-DETR crop metadata
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOAD CROP METADATA")
    print("=" * 70)

    if not STATS_PATH.is_file():
        raise FileNotFoundError(
            f"?�음: {STATS_PATH}"
        )

    with open(
        STATS_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    crops = data["crops"]

    total = len(crops)

    if total == 0:
        raise RuntimeError(
            "crop ?�이?��? 0건입?�다."
        )

    print(
        "total crops:",
        total,
    )

    # --------------------------------------------------------
    # 3. Checkpoint
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.fresh:
        if STATE_PATH.exists():
            STATE_PATH.unlink()

        print(
            "embedding checkpoint reset"
        )

    if (
        args.recreate
        and STATE_PATH.exists()
        and not args.fresh
    ):
        raise RuntimeError(
            "--recreate??기존 collection????��?�니??\n"
            "checkpoint가 존재?��?�?"
            "--recreate --fresh�??�께 ?�용?�야 ?�니??"
        )

    run_info = make_run_info(
        args.batch_size,
        total,
    )

    state = load_checkpoint(
        run_info
    )

    saved_index = int(
        state.get(
            "next_index",
            0,
        )
    )

    # 마�?�?batch????�???처리?�다.
    # Point ID가 deterministic?��?�?중복???�니??overwrite ?�다.
    if saved_index > 0:
        start_index = max(
            0,
            saved_index
            - args.batch_size,
        )

        print(
            f"resume: checkpoint={saved_index:,}"
        )

        print(
            f"safety replay from={start_index:,}"
        )

    else:
        start_index = 0

    # --------------------------------------------------------
    # 4. Determine input format
    # --------------------------------------------------------

    _, fmt = from_rfdetr(
        crops[:1],
        load_mode="path",
    )

    print(
        "input_format:",
        fmt,
    )

    # --------------------------------------------------------
    # 5. Qdrant
    #
    # 먼�? Qdrant ?�결/collection???�인?�다.
    # GPU 모델??모두 ?�린 ??Qdrant ?�류가 ?�는 것을 방�?.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("QDRANT INITIALIZE")
    print("=" * 70)

    person_store = QdrantStore(
        person_cfg
    )

    object_store = QdrantStore(
        object_cfg
    )

    person_store.ensure_collection(
        recreate=args.recreate
    )

    object_store.ensure_collection(
        recreate=args.recreate
    )

    print(
        "Qdrant ready:",
        person_cfg.collection,
        list(person_cfg.retrievers.keys()),
    )

    print(
        "Qdrant ready:",
        object_cfg.collection,
        list(object_cfg.retrievers.keys()),
    )

    # --------------------------------------------------------
    # 6. Embedders + Router
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOAD EMBEDDERS")
    print("=" * 70)

    registry = EmbedderRegistry(
        cfg
    )

    router = Router(
        cfg,
        registry,
        input_format=fmt,
    )

    print(
        "Router ready"
    )

    # --------------------------------------------------------
    # 7. Full embedding
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FULL EMBEDDING START")
    print("=" * 70)

    print(
        f"start : {start_index:,}"
    )

    print(
        f"total : {total:,}"
    )

    run_start = time.time()

    processed_this_run = 0

    try:

        for start in range(
            start_index,
            total,
            args.batch_size,
        ):

            end = min(
                start
                + args.batch_size,
                total,
            )

            batch_start_time = (
                time.time()
            )

            # --------------------------------------------
            # RF-DETR metadata -> Detection
            # --------------------------------------------

            batch_records = crops[
                start:end
            ]

            detections, batch_fmt = (
                from_rfdetr(
                    batch_records,
                    load_mode="path",
                )
            )

            if batch_fmt != fmt:
                raise RuntimeError(
                    "input_format 변�?감�?: "
                    f"{fmt} -> {batch_fmt}"
                )

            if (
                len(detections)
                != len(batch_records)
            ):
                raise RuntimeError(
                    "from_rfdetr 변??개수 불일�? "
                    f"{len(batch_records)} -> "
                    f"{len(detections)}"
                )

            # --------------------------------------------
            # ?�심
            #
            # ?�체   -> SigLIP2
            # person -> IRRA + SOLIDER
            # object -> DINOv2
            # --------------------------------------------

            vectors = router.embed(
                detections
            )

            # Router routing sanity check
            validate_router_result(
                detections,
                vectors,
                cfg,
            )

            # --------------------------------------------
            # Count vectors for logging
            # --------------------------------------------

            vector_counts = {}

            for vector_map in vectors:
                for name in vector_map:
                    vector_counts[name] = (
                        vector_counts.get(
                            name,
                            0,
                        )
                        + 1
                    )

            # --------------------------------------------
            # Qdrant
            # --------------------------------------------

            person_detections = []
            person_vectors = []

            object_detections = []
            object_vectors = []

            for det, vector_map in zip(
                detections,
                vectors,
            ):
                if is_person(det, cfg):
                    # person collection에는 siglip2 / irra / solider만 저장
                    filtered = {
                        name: vector
                        for name, vector in vector_map.items()
                        if name in person_cfg.retrievers
                    }
                    person_detections.append(det)
                    person_vectors.append(filtered)

                else:
                    # object collection에는 siglip2 / dinov2만 저장
                    filtered = {
                        name: vector
                        for name, vector in vector_map.items()
                        if name in object_cfg.retrievers
                    }
                    object_detections.append(det)
                    object_vectors.append(filtered)

            uploaded_person = 0
            uploaded_object = 0

            if person_detections:
                uploaded_person = person_store.upsert(
                    person_detections,
                    person_vectors,
                    batch_size=args.batch_size,
                )

            if object_detections:
                uploaded_object = object_store.upsert(
                    object_detections,
                    object_vectors,
                    batch_size=args.batch_size,
                )

            uploaded = uploaded_person + uploaded_object

            # --------------------------------------------
            # checkpoint
            # Qdrant ?�출???�공???�에�?기록
            # --------------------------------------------

            state = {
                "run_info":
                    run_info,

                "next_index":
                    end,
            }

            atomic_write_json(
                STATE_PATH,
                state,
            )

            processed_this_run += (
                end - start
            )

            # --------------------------------------------
            # Progress
            # --------------------------------------------

            batch_elapsed = (
                time.time()
                - batch_start_time
            )

            elapsed = (
                time.time()
                - run_start
            )

            rate = (
                processed_this_run
                / elapsed
                if elapsed > 0
                else 0
            )

            remaining = (
                total - end
            )

            eta = (
                remaining / rate
                if rate > 0
                else 0
            )

            counts_text = ", ".join(
                f"{name}={count}"
                for name, count
                in sorted(
                    vector_counts.items()
                )
            )

            print(
                f"[{end:,}/{total:,}] "
                f"{end / total * 100:6.2f}% | "
                f"batch {batch_elapsed:6.1f}s | "
                f"{rate:6.2f} crop/s | "
                f"ETA {format_seconds(eta)} | "
                f"Qdrant={uploaded} "
                f"(person={uploaded_person}, object={uploaded_object}) | "
                f"{counts_text}"
            )

            # batch references ?�제
            del vectors
            del detections

    except KeyboardInterrupt:

        print(
            "\nCtrl+C 감�?."
        )

        print(
            "?�재 checkpoint까�? ?�?�되???�습?�다."
        )

        print(
            "같�? 명령???�시 ?�행?�면 ?�어??진행?�니??"
        )

        return

    # --------------------------------------------------------
    # 8. Finished
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BUILD COMPLETED")
    print("=" * 70)

    print(
        "total:",
        f"{total:,}",
    )

    print(
        "collection:",
        cfg.collection,
    )

    print(
        "checkpoint:",
        STATE_PATH,
    )


if __name__ == "__main__":
    main()
