from __future__ import annotations

import json
from pathlib import Path

from qdrant_client import models

from config import PipelineConfig
from qdrant_store import QdrantStore


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "pipeline.yaml"
STATS_PATH = ROOT / "data" / "crops" / "filter_stats.json"


def make_person_filter(value: bool):
    return models.Filter(
        must=[
            models.FieldCondition(
                key="is_person",
                match=models.MatchValue(value=value),
            )
        ]
    )


def exact_count(client, collection, count_filter=None):
    return int(
        client.count(
            collection_name=collection,
            count_filter=count_filter,
            exact=True,
        ).count
    )


def check_sample(point, expected_vectors, expected_dims, person_labels):
    errors = []

    payload = point.payload or {}
    vectors = point.vector or {}

    label = str(payload.get("label", "")).lower()
    is_person = payload.get("is_person")

    # payload 확인
    for key in ("label", "is_person", "image_id", "bbox"):
        if key not in payload:
            errors.append(f"payload 누락: {key}")

    # label / is_person 일치 확인
    expected_person = label in person_labels

    if is_person != expected_person:
        errors.append(
            f"label/is_person 불일치: "
            f"label={label}, is_person={is_person}"
        )

    # vector 이름 확인
    actual_vectors = set(vectors.keys())

    if actual_vectors != expected_vectors:
        errors.append(
            f"vector 구성 오류: "
            f"expected={sorted(expected_vectors)}, "
            f"actual={sorted(actual_vectors)}"
        )

    # vector 차원 확인
    for name in expected_vectors:
        if name not in vectors:
            continue

        actual_dim = len(vectors[name])
        expected_dim = expected_dims[name]

        if actual_dim != expected_dim:
            errors.append(
                f"{name} dim 오류: "
                f"{actual_dim} != {expected_dim}"
            )

    return errors


def main():

    print("=" * 65)
    print("QUICK QDRANT DATABASE VALIDATION")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Config + 원본 crop 개수
    # --------------------------------------------------------

    cfg = PipelineConfig.load(str(CONFIG_PATH))

    with open(STATS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    crops = data["crops"]

    person_labels = {
        str(x).lower()
        for x in cfg.person_labels
    }

    expected_total = len(crops)

    expected_person = sum(
        1
        for x in crops
        if str(x.get("class_name", "")).lower()
        in person_labels
    )

    expected_object = expected_total - expected_person

    del crops
    del data

    print("\n[1] SOURCE")
    print(f"total  : {expected_total:,}")
    print(f"person : {expected_person:,}")
    print(f"object : {expected_object:,}")

    # --------------------------------------------------------
    # 2. Qdrant
    # --------------------------------------------------------

    store = QdrantStore(cfg)
    client = store.client
    collection = cfg.collection

    if not client.collection_exists(collection):
        raise RuntimeError(
            f"Qdrant collection 없음: {collection}"
        )

    print("\n[2] QDRANT CONNECTION : OK")

    # --------------------------------------------------------
    # 3. Schema
    # --------------------------------------------------------

    print("\n[3] VECTOR SCHEMA")

    info = client.get_collection(collection)

    vectors_config = info.config.params.vectors

    expected_dims = {
        name: int(spec.dim)
        for name, spec in cfg.retrievers.items()
    }

    schema_ok = True

    for name, expected_dim in expected_dims.items():

        if name not in vectors_config:
            print(f"{name:<10}: MISSING")
            schema_ok = False
            continue

        actual_dim = int(vectors_config[name].size)

        ok = actual_dim == expected_dim

        print(
            f"{name:<10}: "
            f"{actual_dim} "
            f"{'OK' if ok else 'FAIL'}"
        )

        if not ok:
            schema_ok = False

    # --------------------------------------------------------
    # 4. 전체 개수 확인
    # --------------------------------------------------------

    print("\n[4] POINT COUNTS")

    actual_total = exact_count(
        client,
        collection,
    )

    actual_person = exact_count(
        client,
        collection,
        make_person_filter(True),
    )

    actual_object = exact_count(
        client,
        collection,
        make_person_filter(False),
    )

    count_results = [
        ("total", expected_total, actual_total),
        ("person", expected_person, actual_person),
        ("object", expected_object, actual_object),
    ]

    counts_ok = True

    for name, expected, actual in count_results:

        ok = expected == actual

        print(
            f"{name:<8}: "
            f"{actual:>10,} / "
            f"{expected:>10,} "
            f"{'OK' if ok else 'FAIL'}"
        )

        if not ok:
            counts_ok = False

    # --------------------------------------------------------
    # 5. Person 샘플 5개
    # --------------------------------------------------------

    print("\n[5] PERSON SAMPLE")

    person_points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=make_person_filter(True),
        limit=5,
        with_payload=True,
        with_vectors=True,
    )

    person_vectors = {
        "siglip2",
        "irra",
        "solider",
    }

    samples_ok = True

    for i, point in enumerate(person_points, 1):

        errors = check_sample(
            point,
            person_vectors,
            expected_dims,
            person_labels,
        )

        if errors:
            samples_ok = False
            print(f"person #{i}: FAIL")

            for error in errors:
                print("   -", error)

        else:
            print(
                f"person #{i}: OK "
                f"{sorted(point.vector.keys())}"
            )

    # --------------------------------------------------------
    # 6. Object 샘플 5개
    # --------------------------------------------------------

    print("\n[6] OBJECT SAMPLE")

    object_points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=make_person_filter(False),
        limit=5,
        with_payload=True,
        with_vectors=True,
    )

    object_vectors = {
        "siglip2",
        "dinov2",
    }

    for i, point in enumerate(object_points, 1):

        errors = check_sample(
            point,
            object_vectors,
            expected_dims,
            person_labels,
        )

        if errors:
            samples_ok = False
            print(f"object #{i}: FAIL")

            for error in errors:
                print("   -", error)

        else:
            print(
                f"object #{i}: OK "
                f"{sorted(point.vector.keys())}"
            )

    # --------------------------------------------------------
    # 7. 최종 결과
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    if schema_ok and counts_ok and samples_ok:
        print("DATABASE QUICK VALIDATION: PASS")
        print("DB 구축 정상 완료로 판단할 수 있습니다.")
    else:
        print("DATABASE QUICK VALIDATION: FAIL")
        print("위 FAIL 항목을 확인하세요.")

    print("=" * 65)


if __name__ == "__main__":
    main()