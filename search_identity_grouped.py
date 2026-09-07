from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from config import PipelineConfig
from registry import EmbedderRegistry
from router import Router

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "pipeline.yaml"


def normalize(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v if n <= 0 else v / n


def media_filter(media: str) -> Filter | None:
    if media == "all":
        return None
    return Filter(
        must=[FieldCondition(key="media_type", match=MatchValue(value=media))]
    )


def and_filter(base: Filter | None, *conditions: FieldCondition) -> Filter:
    must = []
    if base is not None and base.must:
        must.extend(base.must)
    must.extend(conditions)
    return Filter(must=must)


def fmt_time(seconds: Any) -> str:
    try:
        sec = float(seconds)
    except (TypeError, ValueError):
        return "N/A"
    if sec < 0:
        return "N/A"
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


def scroll_all_payloads(
    client: QdrantClient,
    collection: str,
    filt: Filter,
    page_size: int = 256,
) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=filt,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        payloads.extend((p.payload or {}) for p in points)
        if next_offset is None:
            break
        offset = next_offset

    return payloads


def group_summary(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not payloads:
        return {
            "count": 0,
            "original_tracks": [],
            "start_sec": None,
            "end_sec": None,
            "videos": [],
        }

    original_tracks = sorted(
        {
            int(p["original_track_id"])
            for p in payloads
            if p.get("original_track_id") is not None
        }
    )

    times = []
    for p in payloads:
        v = p.get("timestamp_sec")
        if v is None:
            continue
        try:
            times.append(float(v))
        except (TypeError, ValueError):
            pass

    videos = sorted(
        {
            str(p.get("video") or p.get("video_name") or "")
            for p in payloads
            if (p.get("video") or p.get("video_name"))
        }
    )

    return {
        "count": len(payloads),
        "original_tracks": original_tracks,
        "start_sec": min(times) if times else None,
        "end_sec": max(times) if times else None,
        "videos": videos,
    }


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Image query -> Qdrant grouped video identity search. "
            "Groups by track_key (video_stem/stitched_id) so repeated frame points "
            "do not flood Top-K."
        )
    )
    ap.add_argument("--image", required=True)
    ap.add_argument(
        "--collection",
        required=True,
        choices=["forensic_person", "forensic_object"],
    )
    ap.add_argument(
        "--vector",
        required=True,
        choices=["siglip2", "irra", "solider", "dinov2"],
    )
    ap.add_argument("--top-k", type=int, default=10, help="number of identity groups")
    ap.add_argument(
        "--group-size",
        type=int,
        default=3,
        help="representative points Qdrant returns per identity group",
    )
    ap.add_argument(
        "--media",
        choices=["video"],
        default="video",
        help="grouped identity search currently targets stitched video points",
    )
    args = ap.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be >= 1")
    if args.group_size <= 0:
        raise ValueError("--group-size must be >= 1")

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    cfg = PipelineConfig.load(CONFIG_PATH)
    reg = EmbedderRegistry(cfg)
    router = Router(cfg, reg, input_format="rgb")

    if args.collection == "forensic_person":
        scope = "person"
        allowed = {"siglip2", "irra", "solider"}
        identity_key = "person_id"
        identity_name = "person"
    else:
        scope = "object"
        allowed = {"siglip2", "dinov2"}
        identity_key = "object_id"
        identity_name = "object"

    if args.vector not in allowed:
        raise ValueError(f"{args.collection} supports {sorted(allowed)}")

    vectors = router.embed_query_image(str(image_path), scope=scope)
    qv = normalize(vectors[args.vector])

    client = QdrantClient(url=cfg.qdrant.url)
    base_filter = media_filter(args.media)

    if not hasattr(client, "query_points_groups"):
        raise RuntimeError(
            "Installed qdrant-client does not provide query_points_groups(). "
            "Upgrade qdrant-client before using grouped identity search."
        )

    result = client.query_points_groups(
        collection_name=args.collection,
        query=qv.tolist(),
        using=args.vector,
        query_filter=base_filter,
        group_by="track_key",
        limit=args.top_k,
        group_size=args.group_size,
        with_payload=True,
        with_vectors=False,
    )

    groups = list(getattr(result, "groups", []) or [])

    print("\n" + "=" * 100)
    print(
        f"GROUPED IDENTITY RESULTS - {args.vector.upper()} | "
        f"collection={args.collection} | group_by=track_key"
    )
    print("=" * 100)

    if not groups:
        print("No grouped video identity results found.")
        return

    for rank, group in enumerate(groups, 1):
        group_id = getattr(group, "id", None)
        hits = list(getattr(group, "hits", []) or [])
        if not hits:
            continue

        best = hits[0]
        best_payload = best.payload or {}
        best_score = float(best.score)

        # Full group metadata: query result returns only group_size representatives,
        # so scroll the exact track_key to recover all timestamps/original track ids.
        exact_filter = and_filter(
            base_filter,
            FieldCondition(key="track_key", match=MatchValue(value=group_id)),
        )
        payloads = scroll_all_payloads(client, args.collection, exact_filter)
        summary = group_summary(payloads)

        identity_id = best_payload.get(identity_key)
        stitched_id = best_payload.get("stitched_id", "")
        video = best_payload.get("video") or best_payload.get("video_name") or ""
        crop = best_payload.get("crop_path", "")
        frame_idx = best_payload.get("frame_idx", "")
        best_time = best_payload.get("timestamp_sec")

        print(
            f"[{rank:02d}] score={best_score:.6f} | "
            f"{identity_name}_id={identity_id} | stitched_id={stitched_id}"
        )
        print(f"     track_key={group_id}")
        print(f"     video={video}")
        print(
            f"     points={summary['count']} | "
            f"original_tracks={summary['original_tracks']} | "
            f"range={fmt_time(summary['start_sec'])} ~ {fmt_time(summary['end_sec'])}"
        )
        print(
            f"     representative frame={frame_idx} | "
            f"time={fmt_time(best_time)} | crop={crop}"
        )


if __name__ == "__main__":
    main()
