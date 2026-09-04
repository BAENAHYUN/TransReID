from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from config import PipelineConfig
from registry import EmbedderRegistry
from router import Router


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "pipeline.yaml"

PERSON_COLLECTION = "forensic_person"
OBJECT_COLLECTION = "forensic_object"


def normalize(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v if n <= 0 else v / n


def canonical_path(path: str) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve()).lower()
    except Exception:
        return str(path).lower()


def make_filter(media_type: str, label: str | None = None):
    must = []

    if media_type != "all":
        must.append(
            FieldCondition(
                key="media_type",
                match=MatchValue(value=media_type),
            )
        )

    if label:
        must.append(
            FieldCondition(
                key="label",
                match=MatchValue(value=label),
            )
        )

    return Filter(must=must) if must else None


def get_timestamp(frame_idx: int, video_path: str) -> str:
    if not video_path:
        return ""

    p = Path(video_path)
    if not p.exists():
        return ""

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return ""

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if fps <= 0:
        return ""

    sec = frame_idx / fps
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


def run_single_vector_search(
    client,
    collection: str,
    vector_name: str,
    query_vector,
    limit: int,
    media_type: str,
    label: str | None,
):
    response = client.query_points(
        collection_name=collection,
        using=vector_name,
        query=normalize(query_vector).tolist(),
        query_filter=make_filter(media_type, label),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return response.points


def remove_self_match(hits, query_image: str | None):
    if not query_image:
        return hits

    q = canonical_path(query_image)
    filtered = []

    for hit in hits:
        p = hit.payload or {}
        crop_path = canonical_path(p.get("crop_path", ""))

        if crop_path and crop_path == q:
            continue

        filtered.append(hit)

    return filtered


def build_rank_maps(per_model_hits: dict[str, list]):
    rank_map = defaultdict(dict)
    score_map = defaultdict(dict)
    point_map = {}

    for model_name, hits in per_model_hits.items():
        for rank, hit in enumerate(hits, start=1):
            pid = str(hit.id)
            rank_map[pid][model_name] = rank
            score_map[pid][model_name] = float(hit.score)
            point_map[pid] = hit

    return rank_map, score_map, point_map


def weighted_rrf(
    per_model_hits: dict[str, list],
    weights: dict[str, float],
    rrf_k: int = 60,
):
    fused = defaultdict(float)

    for model_name, hits in per_model_hits.items():
        w = float(weights.get(model_name, 1.0))
        for rank, hit in enumerate(hits, start=1):
            fused[str(hit.id)] += w / (rrf_k + rank)

    return dict(fused)


def dedupe_video_tracks(rows: list[dict], top_k: int):
    out = []
    seen_tracks = set()

    for row in rows:
        p = row["payload"]

        if p.get("media_type") == "video":
            key = (
                p.get("track_key")
                or f"{p.get('video')}|{p.get('track_id')}|{p.get('label')}"
            )
            if key in seen_tracks:
                continue
            seen_tracks.add(key)

        out.append(row)

        if len(out) >= top_k:
            break

    return out


def search_and_explain(
    client,
    collection: str,
    query_vectors: dict[str, np.ndarray],
    cfg: PipelineConfig,
    candidate_k: int,
    top_k: int,
    media_type: str,
    label: str | None,
    query_image: str | None,
):
    info = client.get_collection(collection)
    available_vectors = set(info.config.params.vectors.keys())

    per_model_hits = {}
    weights = {}

    for model_name, vector in query_vectors.items():
        if model_name not in available_vectors:
            continue

        raw_hits = run_single_vector_search(
            client=client,
            collection=collection,
            vector_name=model_name,
            query_vector=vector,
            limit=candidate_k + (5 if query_image else 0),
            media_type=media_type,
            label=label,
        )

        hits = remove_self_match(raw_hits, query_image)
        hits = hits[:candidate_k]

        per_model_hits[model_name] = hits
        weights[model_name] = float(cfg.retrievers[model_name].weight)

    if not per_model_hits:
        return [], {}

    rank_map, score_map, point_map = build_rank_maps(per_model_hits)
    fused_scores = weighted_rrf(per_model_hits, weights)

    all_ids = set()
    for hits in per_model_hits.values():
        for h in hits:
            all_ids.add(str(h.id))

    rows = []
    for pid in all_ids:
        hit = point_map[pid]
        payload = hit.payload or {}

        rows.append({
            "point_id": pid,
            "fusion_score": float(fused_scores.get(pid, 0.0)),
            "payload": payload,
            "model_ranks": rank_map.get(pid, {}),
            "model_scores": score_map.get(pid, {}),
        })

    rows.sort(key=lambda x: x["fusion_score"], reverse=True)
    rows = dedupe_video_tracks(rows, top_k)

    return rows, per_model_hits


def print_single_model_summary(per_model_hits: dict[str, list], top_n: int = 10):
    print("\n" + "#" * 100)
    print("SINGLE MODEL TOP RESULTS")
    print("#" * 100)

    for model_name, hits in per_model_hits.items():
        print(f"\n[{model_name.upper()}]")

        for rank, hit in enumerate(hits[:top_n], start=1):
            p = hit.payload or {}
            media = p.get("media_type", "")
            crop = p.get("crop_path", "")

            print(
                f"  {rank:02d}. score={float(hit.score):.6f} "
                f"| {media} | {p.get('label', '')}"
            )
            print(f"      crop={crop}")


def print_fusion_results(rows: list[dict], model_order: list[str]):
    print("\n" + "=" * 100)
    print("FUSION RESULTS WITH MODEL-BY-MODEL EXPLANATION")
    print("=" * 100)

    for final_rank, row in enumerate(rows, start=1):
        p = row["payload"]
        media = p.get("media_type", "")
        frame_idx = int(p.get("frame_idx", 0) or 0)

        print(
            f"\n[{final_rank:02d}] "
            f"fusion={row['fusion_score']:.6f} "
            f"| {media} | {p.get('label', '')}"
        )

        for model_name in model_order:
            rank = row["model_ranks"].get(model_name)
            score = row["model_scores"].get(model_name)

            if rank is None:
                print(f"     {model_name:<8} : not in candidate top-K")
            else:
                print(
                    f"     {model_name:<8} : "
                    f"rank={rank:<3} raw_score={score:.6f}"
                )

        if media == "video":
            video_path = p.get("video_path", "") or p.get("image_id", "")
            print(
                f"     video={p.get('video', '')} "
                f"| frame={frame_idx} "
                f"| time={get_timestamp(frame_idx, video_path) or 'N/A'} "
                f"| track={p.get('track_id')}"
            )
            print(f"     video_path={video_path}")
        else:
            print(f"     image_id={p.get('image_id', '')}")

        print(f"     crop={p.get('crop_path', '')}")
        print(
            f"     source={p.get('source', '')} "
            f"| split={p.get('split', '')} "
            f"| category={p.get('category', '')}"
        )


def serialize_rows(rows: list[dict]):
    out = []

    for i, row in enumerate(rows, start=1):
        p = row["payload"]

        out.append({
            "final_rank": i,
            "fusion_score": row["fusion_score"],
            "point_id": row["point_id"],
            "model_ranks": row["model_ranks"],
            "model_scores": row["model_scores"],
            "payload": p,
        })

    return out


def main():
    ap = argparse.ArgumentParser(
        description="Validation search: raw model ranks/scores + RRF fusion"
    )

    q = ap.add_mutually_exclusive_group(required=True)
    q.add_argument("--image", type=str)
    q.add_argument("--text", type=str)

    ap.add_argument(
        "--scope",
        choices=["person", "object"],
        required=True,
        help="검증에서는 person/object를 명시적으로 분리",
    )
    ap.add_argument(
        "--media",
        choices=["all", "image", "video"],
        default="all",
    )
    ap.add_argument("--label", default=None)
    ap.add_argument("--candidate-k", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument(
        "--single-top",
        type=int,
        default=10,
        help="각 단일 모델별 출력 개수",
    )
    ap.add_argument(
        "--no-self-filter",
        action="store_true",
        help="이미지 query 자기 자신 제거 비활성화",
    )
    ap.add_argument("--json-out", default=None)

    args = ap.parse_args()

    cfg = PipelineConfig.load(CONFIG_PATH)
    registry = EmbedderRegistry(cfg)
    router = Router(cfg, registry, input_format="rgb")
    client = QdrantClient(url=cfg.qdrant.url)

    query_image = None

    if args.image:
        query_path = Path(args.image)
        if not query_path.exists():
            raise FileNotFoundError(query_path)

        query_image = None if args.no_self_filter else str(query_path)

        vectors = router.embed_query_image(
            str(query_path),
            scope=args.scope,
        )

    else:
        if args.scope == "person":
            names = ["siglip2", "irra"]
        else:
            names = ["siglip2"]

        vectors = router.embed_query_text(
            args.text,
            names=names,
        )

    if args.scope == "person":
        collection = PERSON_COLLECTION
        expected_order = ["solider", "irra", "siglip2"]
        label = args.label or "person"
    else:
        collection = OBJECT_COLLECTION
        expected_order = ["dinov2", "siglip2"]
        label = args.label

    vectors = {
        name: vec
        for name, vec in vectors.items()
        if name in expected_order
    }

    print("=" * 100)
    print("SEARCH VALIDATION")
    print("=" * 100)
    print("scope       :", args.scope)
    print("collection  :", collection)
    print("media       :", args.media)
    print("models      :", list(vectors.keys()))
    print("candidate_k :", args.candidate_k)
    print("top_k       :", args.top_k)
    print("self-filter :", bool(query_image))
    print()

    rows, per_model_hits = search_and_explain(
        client=client,
        collection=collection,
        query_vectors=vectors,
        cfg=cfg,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        media_type=args.media,
        label=label,
        query_image=query_image,
    )

    print_single_model_summary(
        per_model_hits,
        top_n=args.single_top,
    )

    print_fusion_results(
        rows,
        model_order=expected_order,
    )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "scope": args.scope,
            "media": args.media,
            "query_image": args.image,
            "query_text": args.text,
            "candidate_k": args.candidate_k,
            "top_k": args.top_k,
            "results": serialize_rows(rows),
        }

        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"\n[+] JSON saved: {out_path}")


if __name__ == "__main__":
    main()
