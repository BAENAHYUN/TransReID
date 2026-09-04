from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
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


def expected_vector_names(det, cfg):
    """
    pipeline.yaml 의 scope 기준으로
    이 detection에 반드시 있어야 하는 vector 이름.
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
    Router가 잘못 분기한 상태로
    45만 건을 적재하는 것을 방지한다.
    """

    if not isinstance(
        vectors,
        (list, tuple),
    ):
        raise TypeError(
            "router.embed() 반환값이 "
            "list/tuple이 아닙니다: "
            f"{type(vectors)}"
        )

    if (
        len(detections)
        != len(vectors)
    ):
        raise RuntimeError(
            "Detection / vector 개수 불일치: "
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
                f"vectors[{i}]가 dict가 아닙니다: "
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
                "\nRouter routing 오류\n"
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
            "현재 pipeline/data 설정과 다릅니다.\n"
            "처음부터 다시 하려면 --fresh 를 사용하세요."
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
        help="Embedding checkpoint를 삭제하고 0부터 다시 처리",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Qdrant collection을 삭제 후 새로 생성",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size는 1 이상이어야 합니다."
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

    print(
        "collection:",
        cfg.collection,
    )

    print(
        "retrievers:",
        list(
            cfg.retrievers.keys()
        ),
    )

    # --------------------------------------------------------
    # 2. RF-DETR crop metadata
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOAD CROP METADATA")
    print("=" * 70)

    if not STATS_PATH.is_file():
        raise FileNotFoundError(
            f"없음: {STATS_PATH}"
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
            "crop 데이터가 0건입니다."
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
            "--recreate는 기존 collection을 삭제합니다.\n"
            "checkpoint가 존재하므로 "
            "--recreate --fresh를 함께 사용해야 합니다."
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

    # 마지막 batch는 한 번 더 처리한다.
    # Point ID가 deterministic이므로 중복이 아니라 overwrite 된다.
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
    # 먼저 Qdrant 연결/collection을 확인한다.
    # GPU 모델을 모두 올린 뒤 Qdrant 오류가 나는 것을 방지.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("QDRANT INITIALIZE")
    print("=" * 70)

    store = QdrantStore(
        cfg
    )

    store.ensure_collection(
        recreate=args.recreate
    )

    print(
        "Qdrant ready:",
        cfg.collection,
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
                    "input_format 변경 감지: "
                    f"{fmt} -> {batch_fmt}"
                )

            if (
                len(detections)
                != len(batch_records)
            ):
                raise RuntimeError(
                    "from_rfdetr 변환 개수 불일치: "
                    f"{len(batch_records)} -> "
                    f"{len(detections)}"
                )

            # --------------------------------------------
            # 핵심
            #
            # 전체   -> SigLIP2
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

            uploaded = store.upsert(
                detections,
                vectors,
                batch_size=args.batch_size,
            )

            # --------------------------------------------
            # checkpoint
            # Qdrant 호출이 성공한 뒤에만 기록
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
                f"Qdrant={uploaded} | "
                f"{counts_text}"
            )

            # batch references 해제
            del vectors
            del detections

    except KeyboardInterrupt:

        print(
            "\nCtrl+C 감지."
        )

        print(
            "현재 checkpoint까지 저장되어 있습니다."
        )

        print(
            "같은 명령을 다시 실행하면 이어서 진행합니다."
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