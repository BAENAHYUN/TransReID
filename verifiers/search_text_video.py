from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from config import PipelineConfig
from registry import EmbedderRegistry
from router import Router


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "pipeline.yaml"

RRF_K = 60


def normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v if n <= 0 else v / n


def video_filter() -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="media_type",
                match=MatchValue(value="video"),
            )
        ]
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
        value = p.get("timestamp_sec")
        if value is None:
            continue

        try:
            times.append(float(value))
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


@dataclass
class GroupCandidate:
    group_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    vector_scores: Dict[str, float] = field(default_factory=dict)
    vector_ranks: Dict[str, int] = field(default_factory=dict)
    rrf_score: float = 0.0


def query_grouped(
    client: QdrantClient,
    collection: str,
    vector_name: str,
    query_vector: np.ndarray,
    limit: int,
    group_size: int,
) -> list:
    result = client.query_points_groups(
        collection_name=collection,
        query=query_vector.tolist(),
        using=vector_name,
        query_filter=video_filter(),
        group_by="track_key",
        limit=limit,
        group_size=group_size,
        with_payload=True,
        with_vectors=False,
    )

    return list(getattr(result, "groups", []) or [])


def merge_group_results(
    grouped_by_vector: Dict[str, list],
    scope: str,
) -> List[GroupCandidate]:
    merged: Dict[str, GroupCandidate] = {}

    for vector_name, groups in grouped_by_vector.items():
        for rank, group in enumerate(groups, start=1):
            group_id = getattr(group, "id", None)
            hits = list(getattr(group, "hits", []) or [])

            if group_id is None or not hits:
                continue

            group_id = str(group_id)
            best = hits[0]
            payload = best.payload or {}
            score = float(best.score)

            if group_id not in merged:
                merged[group_id] = GroupCandidate(
                    group_id=group_id,
                    payload=payload,
                )

            item = merged[group_id]
            item.vector_scores[vector_name] = score
            item.vector_ranks[vector_name] = rank

            # RRF는 서로 다른 임베딩 공간의 cosine score scale 차이를 피하기 위해
            # rank 기반으로 합친다.
            item.rrf_score += 1.0 / (RRF_K + rank)

    rows = list(merged.values())

    if scope == "person":
        rows.sort(
            key=lambda x: (
                -x.rrf_score,
                min(x.vector_ranks.values()) if x.vector_ranks else 10**9,
            )
        )
    else:
        # object는 자연어 검색 가능한 vector가 SigLIP2 하나뿐이므로
        # 실제 cosine score 순서를 그대로 사용한다.
        rows.sort(
            key=lambda x: -x.vector_scores.get("siglip2", -1e9)
        )

    return rows


def print_result(
    rank: int,
    item: GroupCandidate,
    summary: Dict[str, Any],
    scope: str,
) -> None:
    payload = item.payload

    identity_key = "person_id" if scope == "person" else "object_id"
    identity_name = "person" if scope == "person" else "object"

    identity_id = payload.get(identity_key)
    stitched_id = payload.get("stitched_id", "")
    video = payload.get("video") or payload.get("video_name") or ""
    crop = payload.get("crop_path", "")
    frame_idx = payload.get("frame_idx", "")
    best_time = payload.get("timestamp_sec")
    label = payload.get("label", "")

    print("\n" + "-" * 100)

    if scope == "person":
        sig_score = item.vector_scores.get("siglip2")
        irra_score = item.vector_scores.get("irra")
        sig_rank = item.vector_ranks.get("siglip2")
        irra_rank = item.vector_ranks.get("irra")

        print(
            f"[{rank:02d}] RRF={item.rrf_score:.8f} | "
            f"SigLIP2={sig_score if sig_score is not None else 'N/A'} "
            f"(rank={sig_rank if sig_rank is not None else 'N/A'}) | "
            f"IRRA={irra_score if irra_score is not None else 'N/A'} "
            f"(rank={irra_rank if irra_rank is not None else 'N/A'})"
        )
    else:
        sig_score = item.vector_scores.get("siglip2")
        print(
            f"[{rank:02d}] SigLIP2={sig_score:.6f}"
            if sig_score is not None
            else f"[{rank:02d}] SigLIP2=N/A"
        )

    print(
        f"     {identity_name}_id={identity_id} | "
        f"stitched_id={stitched_id} | label={label}"
    )
    print(f"     track_key={item.group_id}")
    print(f"     video={video}")
    print(
        f"     points={summary['count']} | "
        f"original_tracks={summary['original_tracks']} | "
        f"range={fmt_time(summary['start_sec'])} ~ "
        f"{fmt_time(summary['end_sec'])}"
    )
    print(
        f"     representative frame={frame_idx} | "
        f"time={fmt_time(best_time)}"
    )
    print(f"     crop={crop}")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Natural language -> Qdrant stitched VIDEO search. "
            "Person: SigLIP2 + IRRA RRF fusion. "
            "Object: SigLIP2."
        )
    )

    ap.add_argument(
        "--text",
        required=True,
        help='natural-language query, e.g. "검은 상의를 입은 사람"',
    )
    ap.add_argument(
        "--scope",
        required=True,
        choices=["person", "object"],
        help="search people or objects",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="final number of stitched video identity groups",
    )
    ap.add_argument(
        "--candidate-k",
        type=int,
        default=100,
        help="candidate groups retrieved per text-capable vector",
    )
    ap.add_argument(
        "--group-size",
        type=int,
        default=3,
        help="representative Qdrant points returned per track_key group",
    )

    args = ap.parse_args()

    if not args.text.strip():
        raise ValueError("--text must not be empty")
    if args.top_k <= 0:
        raise ValueError("--top-k must be >= 1")
    if args.candidate_k <= 0:
        raise ValueError("--candidate-k must be >= 1")
    if args.group_size <= 0:
        raise ValueError("--group-size must be >= 1")

    cfg = PipelineConfig.load(CONFIG_PATH)
    reg = EmbedderRegistry(cfg)
    router = Router(cfg, reg, input_format="rgb")
    client = QdrantClient(url=cfg.qdrant.url)

    if not hasattr(client, "query_points_groups"):
        raise RuntimeError(
            "Installed qdrant-client does not provide query_points_groups(). "
            "Upgrade qdrant-client before using grouped video search."
        )

    if args.scope == "person":
        collection = "forensic_person"
        text_vectors = ["siglip2", "irra"]
    else:
        collection = "forensic_object"
        text_vectors = ["siglip2"]

    print("=" * 100)
    print("NATURAL LANGUAGE -> VIDEO SEARCH")
    print("=" * 100)
    print(f"text       : {args.text}")
    print(f"scope      : {args.scope}")
    print(f"collection : {collection}")
    print(f"vectors    : {' + '.join(text_vectors)}")
    print("media      : video only")
    print("group_by   : track_key")

    print("\n[1/3] Text embedding...")
    vectors = router.embed_query_text(
        args.text,
        names=text_vectors,
    )

    query_vectors: Dict[str, np.ndarray] = {}
    for name in text_vectors:
        if name not in vectors:
            raise RuntimeError(
                f"Router did not return text vector '{name}'. "
                f"returned={list(vectors.keys())}"
            )
        query_vectors[name] = normalize(vectors[name])
        print(f"  {name}: dim={query_vectors[name].shape[0]}")

    print("\n[2/3] Qdrant grouped video retrieval...")
    grouped_by_vector: Dict[str, list] = {}

    for name in text_vectors:
        groups = query_grouped(
            client=client,
            collection=collection,
            vector_name=name,
            query_vector=query_vectors[name],
            limit=args.candidate_k,
            group_size=args.group_size,
        )
        grouped_by_vector[name] = groups
        print(f"  {name}: {len(groups):,} groups")

    merged = merge_group_results(
        grouped_by_vector,
        scope=args.scope,
    )

    final_rows = merged[: args.top_k]

    print("\n[3/3] Final results")
    print("=" * 100)

    if not final_rows:
        print("No grouped video results found.")
        return

    base_filter = video_filter()

    for rank, item in enumerate(final_rows, start=1):
        exact_filter = and_filter(
            base_filter,
            FieldCondition(
                key="track_key",
                match=MatchValue(value=item.group_id),
            ),
        )

        payloads = scroll_all_payloads(
            client,
            collection,
            exact_filter,
        )
        summary = group_summary(payloads)

        print_result(
            rank,
            item,
            summary,
            scope=args.scope,
        )

    print("\n" + "=" * 100)
    print("SEARCH COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
