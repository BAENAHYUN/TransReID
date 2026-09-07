from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Iterable, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from config import PipelineConfig


EXPECTED_HARDCODED = {
    "person": "forensic_person",
    "object": "forensic_object",
}


def media_filter(media_type: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="media_type",
                match=MatchValue(value=media_type),
            )
        ]
    )


def get_vector_schema(info: Any) -> Dict[str, int]:
    """
    Return {named_vector: dim}.
    This project uses named vectors, so a non-dict vector schema is a failure.
    """
    vectors = info.config.params.vectors

    if not isinstance(vectors, dict):
        return {"__UNNAMED_VECTOR_SCHEMA__": int(getattr(vectors, "size", -1))}

    out: Dict[str, int] = {}
    for name, params in vectors.items():
        out[str(name)] = int(getattr(params, "size", -1))
    return out


def expected_for_kind(cfg: PipelineConfig, kind: str) -> Dict[str, int]:
    if kind == "person":
        names = {
            name: spec
            for name, spec in cfg.retrievers.items()
            if spec.accepts_person()
        }
    elif kind == "object":
        names = {
            name: spec
            for name, spec in cfg.retrievers.items()
            if spec.accepts_object()
        }
    else:
        raise ValueError(kind)

    return {
        name: int(spec.dim)
        for name, spec in names.items()
    }


def sample_point(
    client: QdrantClient,
    collection: str,
    media_type: str,
):
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=media_filter(media_type),
        limit=1,
        with_payload=True,
        with_vectors=True,
    )
    return points[0] if points else None


def vector_lengths(point: Any) -> Dict[str, int]:
    if point is None:
        return {}

    vectors = point.vector
    if not isinstance(vectors, dict):
        try:
            return {"__UNNAMED__": len(vectors)}
        except Exception:
            return {}

    out = {}
    for name, vec in vectors.items():
        try:
            out[str(name)] = len(vec)
        except Exception:
            out[str(name)] = -1
    return out


def print_point_summary(collection: str, media: str, point: Any, expected: Dict[str, int]):
    if point is None:
        print(f"  [WARN] {collection} / {media}: sample point 없음")
        return 1, 0

    payload = point.payload or {}
    lengths = vector_lengths(point)

    print(f"  [{media.upper()} SAMPLE]")
    print(f"    point_id    : {point.id}")
    print(f"    label       : {payload.get('label')}")
    print(f"    crop_path   : {payload.get('crop_path')}")
    print(f"    vector dims : {lengths}")

    warns = 0
    fails = 0

    for name, dim in expected.items():
        actual = lengths.get(name)
        if actual is None:
            print(f"    [FAIL] vector missing: {name}")
            fails += 1
        elif actual != dim:
            print(f"    [FAIL] vector dim: {name} expected={dim}, actual={actual}")
            fails += 1
        else:
            print(f"    [PASS] vector {name}: {actual}")

    if payload.get("media_type") != media:
        print(
            f"    [FAIL] media_type expected={media!r}, "
            f"actual={payload.get('media_type')!r}"
        )
        fails += 1

    if not payload.get("crop_path"):
        print("    [WARN] crop_path 없음")
        warns += 1

    if media == "video":
        required = ["video", "track_key", "stitched_id", "original_track_id"]
        for key in required:
            if payload.get(key) is None:
                print(f"    [WARN] video payload missing: {key}")
                warns += 1

        if collection.endswith("_person") and payload.get("person_id") is None:
            print("    [WARN] person_id 없음")
            warns += 1

        if collection.endswith("_object") and payload.get("object_id") is None:
            print("    [WARN] object_id 없음")
            warns += 1

    return warns, fails


def smoke_query(
    client: QdrantClient,
    collection: str,
    image_point: Any,
    expected: Dict[str, int],
) -> Tuple[int, int]:
    """
    Query the same unified collection using an IMAGE vector with no media filter.
    Success means the named-vector schema is query-compatible across the unified DB.
    It does NOT claim semantic accuracy.
    """
    warns = 0
    fails = 0

    if image_point is None or not isinstance(image_point.vector, dict):
        print("  [WARN] mixed-media query smoke test 생략: image vector 없음")
        return 1, 0

    print("  [MIXED-MEDIA QUERY SMOKE TEST]")

    for vector_name in expected:
        vec = image_point.vector.get(vector_name)
        if vec is None:
            print(f"    [FAIL] {vector_name}: image sample vector 없음")
            fails += 1
            continue

        try:
            result = client.query_points(
                collection_name=collection,
                using=vector_name,
                query=list(vec),
                limit=10,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            print(f"    [FAIL] {vector_name}: query 실패: {type(exc).__name__}: {exc}")
            fails += 1
            continue

        media_types = sorted(
            {
                str((p.payload or {}).get("media_type"))
                for p in result.points
            }
        )
        print(
            f"    [PASS] {vector_name}: query 성공, "
            f"top10 media={media_types}"
        )

    return warns, fails


def main():
    ap = argparse.ArgumentParser(
        description="기존 이미지 DB와 동영상 DB의 Qdrant 호환성 검사"
    )
    ap.add_argument(
        "--config",
        default="pipeline.yaml",
        help="pipeline.yaml path (default: pipeline.yaml)",
    )
    ap.add_argument(
        "--url",
        default=None,
        help="Qdrant URL override. Default: pipeline.yaml qdrant.url",
    )
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)
    url = args.url or cfg.qdrant.url

    cfg_collections = {
        "person": cfg.person_collection(),
        "object": cfg.object_collection(),
    }

    expected = {
        "person": expected_for_kind(cfg, "person"),
        "object": expected_for_kind(cfg, "object"),
    }

    print("=" * 88)
    print("IMAGE ↔ VIDEO UNIFIED QDRANT COMPATIBILITY CHECK")
    print("=" * 88)
    print("Qdrant URL:", url)
    print("config person collection:", cfg_collections["person"])
    print("config object collection:", cfg_collections["object"])
    print("expected person vectors :", expected["person"])
    print("expected object vectors :", expected["object"])

    warnings = 0
    failures = 0

    # Critical because current build_video_db.py targets these collection names.
    for kind in ("person", "object"):
        configured = cfg_collections[kind]
        hardcoded = EXPECTED_HARDCODED[kind]
        if configured != hardcoded:
            print(
                f"[FAIL] collection-name mismatch ({kind}): "
                f"pipeline.yaml={configured!r}, build_video_db expected={hardcoded!r}"
            )
            failures += 1
        else:
            print(f"[PASS] collection-name match ({kind}): {configured}")

    try:
        client = QdrantClient(url=url)
        cols = {c.name for c in client.get_collections().collections}
    except Exception as exc:
        print(f"\n[FATAL] Qdrant 연결 실패: {type(exc).__name__}: {exc}")
        sys.exit(2)

    for kind in ("person", "object"):
        collection = EXPECTED_HARDCODED[kind]
        exp = expected[kind]

        print("\n" + "-" * 88)
        print(f"{kind.upper()} COLLECTION: {collection}")
        print("-" * 88)

        if collection not in cols:
            print(f"[FAIL] collection 없음: {collection}")
            failures += 1
            continue

        info = client.get_collection(collection)
        actual_schema = get_vector_schema(info)

        print("expected schema:", exp)
        print("actual schema  :", actual_schema)

        if actual_schema != exp:
            missing = sorted(set(exp) - set(actual_schema))
            extra = sorted(set(actual_schema) - set(exp))
            wrong_dim = {
                k: (exp[k], actual_schema.get(k))
                for k in exp
                if k in actual_schema and exp[k] != actual_schema[k]
            }
            print(
                f"[FAIL] Qdrant vector schema mismatch "
                f"missing={missing}, extra={extra}, wrong_dim={wrong_dim}"
            )
            failures += 1
        else:
            print("[PASS] Qdrant named-vector schema 동일")

        try:
            image_count = client.count(
                collection_name=collection,
                count_filter=media_filter("image"),
                exact=True,
            ).count
            video_count = client.count(
                collection_name=collection,
                count_filter=media_filter("video"),
                exact=True,
            ).count
        except Exception as exc:
            print(f"[FAIL] media_type count 실패: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        print(f"image points : {image_count:,}")
        print(f"video points : {video_count:,}")

        if image_count == 0:
            print("[WARN] image point가 0개 — 기존 이미지 DB 보존 여부 확인 필요")
            warnings += 1
        else:
            print("[PASS] 기존 image points 존재")

        if video_count == 0:
            print("[WARN] video point가 0개 — 아직 동영상 DB 미적재 상태일 수 있음")
            warnings += 1
        else:
            print("[PASS] video points 존재")

        image_point = sample_point(client, collection, "image")
        video_point = sample_point(client, collection, "video")

        w, f = print_point_summary(collection, "image", image_point, exp)
        warnings += w
        failures += f

        w, f = print_point_summary(collection, "video", video_point, exp)
        warnings += w
        failures += f

        # Directly compare IMAGE and VIDEO sample vector layouts.
        if image_point is not None and video_point is not None:
            image_layout = vector_lengths(image_point)
            video_layout = vector_lengths(video_point)

            if image_layout == video_layout == exp:
                print("[PASS] image/video sample의 vector name+dim 완전 동일")
            else:
                print(
                    "[FAIL] image/video sample vector layout 불일치\n"
                    f"       image={image_layout}\n"
                    f"       video={video_layout}\n"
                    f"       expected={exp}"
                )
                failures += 1

        w, f = smoke_query(client, collection, image_point, exp)
        warnings += w
        failures += f

    print("\n" + "=" * 88)
    print("FINAL RESULT")
    print("=" * 88)
    print("failures :", failures)
    print("warnings :", warnings)

    if failures == 0:
        print("\n[PASS] 이미지 DB와 동영상 DB는 동일 Qdrant collection/named-vector schema로 호환됩니다.")
        print("       이후 단계에서는 tracking/stitching 품질 검증을 별도로 진행하면 됩니다.")
        sys.exit(0)

    print("\n[FAIL] DB 호환성 문제를 먼저 해결해야 합니다.")
    sys.exit(2)


if __name__ == "__main__":
    main()
