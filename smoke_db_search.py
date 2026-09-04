from __future__ import annotations

from pathlib import Path

from config import PipelineConfig
from qdrant_store import QdrantStore

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "pipeline.yaml"


def pick_one(store, is_person: bool):
    from qdrant_client import models

    filt = models.Filter(
        must=[
            models.FieldCondition(
                key="is_person",
                match=models.MatchValue(value=is_person),
            )
        ]
    )

    points, _ = store.client.scroll(
        collection_name=store.collection,
        scroll_filter=filt,
        limit=1,
        with_payload=True,
        with_vectors=True,
    )

    if not points:
        raise RuntimeError(
            f"is_person={is_person} point를 찾지 못했습니다."
        )

    return points[0]


def run_single(store, point, vector_name: str, person_only):
    vectors = point.vector or {}

    if vector_name not in vectors:
        raise RuntimeError(
            f"선택한 point {point.id}에 '{vector_name}' vector가 없습니다. "
            f"actual={sorted(vectors)}"
        )

    results = store.search_single(
        vector_name,
        vectors[vector_name],
        limit=5,
        person_only=person_only,
    )

    if not results:
        raise RuntimeError(f"{vector_name}: 검색 결과가 없습니다.")

    top = results[0]
    ok = str(top.id) == str(point.id)

    print(
        f"{'[PASS]' if ok else '[WARN]'} "
        f"{vector_name:<8} "
        f"query_id={point.id} "
        f"top1={top.id} "
        f"score={getattr(top, 'score', None)}"
    )

    return ok


def run_fusion(store, point, names, person_only):
    vectors = point.vector or {}
    query_vectors = {
        name: vectors[name]
        for name in names
        if name in vectors
    }

    results = store.fused_search(
        query_vectors,
        person_only=person_only,
        limit=5,
    )

    if not results:
        raise RuntimeError(
            f"fusion {sorted(query_vectors)}: 검색 결과가 없습니다."
        )

    top = results[0]
    ok = str(top.id) == str(point.id)

    print(
        f"{'[PASS]' if ok else '[WARN]'} "
        f"fusion {sorted(query_vectors)} "
        f"query_id={point.id} "
        f"top1={top.id} "
        f"score={getattr(top, 'score', None)}"
    )

    return ok


def main():
    cfg = PipelineConfig.load(CONFIG)
    store = QdrantStore(cfg)

    print("=" * 72)
    print("QDRANT SEARCH SMOKE TEST (READ ONLY)")
    print("=" * 72)
    print("collection:", store.collection)

    person = pick_one(store, True)
    obj = pick_one(store, False)

    print("\n=== PERSON POINT ===")
    print("id   :", person.id)
    print("label:", (person.payload or {}).get("label"))
    print("vecs :", sorted((person.vector or {}).keys()))

    person_results = []
    person_results.append(run_single(store, person, "siglip2", True))
    person_results.append(run_single(store, person, "irra", True))
    person_results.append(run_single(store, person, "solider", True))
    person_results.append(
        run_fusion(
            store,
            person,
            ["siglip2", "irra", "solider"],
            True,
        )
    )

    print("\n=== OBJECT POINT ===")
    print("id   :", obj.id)
    print("label:", (obj.payload or {}).get("label"))
    print("vecs :", sorted((obj.vector or {}).keys()))

    object_results = []
    object_results.append(run_single(store, obj, "siglip2", False))
    object_results.append(run_single(store, obj, "dinov2", False))
    object_results.append(
        run_fusion(
            store,
            obj,
            ["siglip2", "dinov2"],
            False,
        )
    )

    print("\n" + "=" * 72)
    all_ok = all(person_results + object_results)

    if all_ok:
        print("RESULT: PASS")
        print("named-vector 단일 검색과 fusion 검색이 모두 정상입니다.")
    else:
        print("RESULT: WARN")
        print(
            "검색 자체는 실행됐지만 일부 self-query가 top1 자기 자신이 아닙니다. "
            "해당 출력 확인이 필요합니다."
        )

    print("=" * 72)


if __name__ == "__main__":
    main()
