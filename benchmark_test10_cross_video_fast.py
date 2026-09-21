#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import cv2
import unified_search_4mode as us
from qdrant_client import QdrantClient
from config import PipelineConfig
from registry import EmbedderRegistry
from router import Router


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "pipeline.yaml"
VIDEOS_ROOT = ROOT / "data" / "videos"
PROCESSED_ROOT = ROOT / "outputs" / "processed_videos"
CAND_ROOT = ROOT / "outputs" / "final_db_candidates"

DEFAULT_STEMS = [
    "Normal_Videos_003_x264",
    "Normal_Videos_006_x264",
    "Normal_Videos_010_x264",
    "Normal_Videos_014_x264",
    "Normal_Videos_015_x264",
    "Normal_Videos_018_x264",
    "Normal_Videos_019_x264",
    "Normal_Videos_024_x264",
    "Normal_Videos_025_x264",
    "Normal_Videos_027_x264",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-video withheld retrieval benchmark for TEST10 collections"
    )
    p.add_argument("--stems", nargs="*", default=DEFAULT_STEMS)
    p.add_argument("--queries-per-track", type=int, default=2)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--group-size", type=int, default=3)
    p.add_argument("--person-collection", default="forensic_person_test10")
    p.add_argument("--object-collection", default="forensic_object_test10")
    return p.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_video(stem):
    hits = [
        p for p in VIDEOS_ROOT.rglob(stem + ".*")
        if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
    ]
    if not hits:
        raise FileNotFoundError(stem)
    return sorted(hits)[0]


def clamp_bbox(b, w, h):
    x1, y1, x2, y2 = map(float, b)
    x1 = max(0, min(w, int(math.floor(x1))))
    y1 = max(0, min(h, int(math.floor(y1))))
    x2 = max(0, min(w, int(math.ceil(x2))))
    y2 = max(0, min(h, int(math.ceil(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def sharpness(img):
    if img.size == 0:
        return 0.0
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def candidate_maps(data):
    person_map, object_map, db_frames = {}, {}, defaultdict(set)

    for r in data.get("candidates", []):
        scope = str(r.get("candidate_scope", "")).lower()
        canonical = int(r.get("selected_track_id", r.get("long_track_id", -1)))
        db_frames[(scope, canonical)].add(int(r["frame_idx"]))

        if scope == "person":
            person_map[int(r.get("long_track_id", canonical))] = canonical
        elif scope == "object":
            for src in (r.get("merged_from_track_ids") or [r.get("long_track_id", canonical)]):
                object_map[int(src)] = canonical

    return person_map, object_map, db_frames


def source_rows(stem, person_map, object_map):
    root = PROCESSED_ROOT / stem
    final = load_json(root / "final_routed_tracks.json")
    obj = load_json(root / "object_validated_tracks_margin.json")

    oi = {}
    for r in obj:
        oi[(int(r.get("long_track_id", r.get("track_id", -1))), int(r["frame_idx"]))] = r

    out = []
    for r in final:
        route = str(r.get("final_db_route", "")).lower()
        src = int(r.get("long_track_id", r.get("track_id", -1)))

        if route == "person" and src in person_map:
            x = dict(r)
            x["_scope"] = "person"
            x["_canonical"] = person_map[src]
            out.append(x)

        elif route == "object" and src in object_map:
            sem = oi.get((src, int(r["frame_idx"])))
            if not sem or str(sem.get("final_db_route_v2", "")).lower() != "object":
                continue
            x = dict(r)
            x["_scope"] = "object"
            x["_canonical"] = object_map[src]
            out.append(x)

    return out


def choose_queries(stem, video, rows, db_frames, n, root):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["_scope"], int(r["_canonical"]))].append(r)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(video)

    queries = []

    for (scope, canonical), grows in sorted(groups.items()):
        pool = []

        for r in grows:
            fi = int(r["frame_idx"])
            if fi in db_frames[(scope, canonical)]:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            h, w = frame.shape[:2]
            bb = clamp_bbox(r["bbox"], w, h)
            if not bb:
                continue

            x1, y1, x2, y2 = bb
            crop = frame[y1:y2, x1:x2]
            cw, ch = x2 - x1, y2 - y1

            if crop.size == 0:
                continue
            if scope == "person" and (cw < 14 or ch < 25 or cw * ch < 500):
                continue
            if scope == "object" and (cw < 10 or ch < 16 or cw * ch < 220):
                continue

            score = float(r.get("confidence") or 0.0) + min(sharpness(crop) / 200.0, 1.0)
            pool.append((score, fi, r, crop))

        pool.sort(key=lambda x: x[0], reverse=True)
        chosen = pool[:n]

        for qi, (_, fi, r, crop) in enumerate(chosen, 1):
            dst = root / "queries" / stem / scope / f"track_{canonical:04d}" / f"q{qi:02d}_frame_{fi:06d}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), crop)

            queries.append({
                "video_stem": stem,
                "scope": scope,
                "expected_track": canonical,
                "expected_track_key": f"{stem}/{scope}_{canonical}",
                "frame_idx": fi,
                "query_path": str(dst.resolve()),
            })

    cap.release()
    return queries


def result_track_key_from_group(group):
    gid = getattr(group, "id", None)
    return "" if gid is None else str(gid)


def evaluate_scope_queries(scope, queries, args, cfg):
    """
    Load the image embedder once for this scope and reuse it for every query.
    person -> SOLIDER
    object -> DINOv2
    """
    if not queries:
        return []

    vector_name = "solider" if scope == "person" else "dinov2"
    collection = (
        args.person_collection
        if scope == "person"
        else args.object_collection
    )

    registry = EmbedderRegistry(cfg)
    router = Router(cfg, registry, input_format="rgb")
    client = QdrantClient(url=cfg.qdrant.url, timeout=120)

    evaluated = []

    try:
        print()
        print("-" * 96)
        print(
            f"[LOAD ONCE] scope={scope} | "
            f"vector={vector_name} | collection={collection}"
        )
        print("-" * 96)

        for i, q in enumerate(queries, 1):
            vectors = router.embed_query_image(
                q["query_path"],
                scope=scope,
                names=[vector_name],
            )

            if vector_name not in vectors:
                raise RuntimeError(
                    f"query embedder did not return {vector_name}"
                )

            groups = us.grouped_query(
                client,
                collection=collection,
                vector_name=vector_name,
                query_vector=vectors[vector_name],
                limit=args.top_k,
                group_size=args.group_size,
            )

            keys = [
                result_track_key_from_group(g)
                for g in groups
            ]

            expected = q["expected_track_key"]
            rank = None
            for idx, key in enumerate(keys, 1):
                if key == expected:
                    rank = idx
                    break

            evaluated.append({
                **q,
                "vector": vector_name,
                "rank": rank,
                "top1": rank == 1,
                "top3": rank is not None and rank <= 3,
                "rr": 0.0 if rank is None else 1.0 / rank,
                "returned_track_keys": keys,
            })

            if (
                i == 1
                or i % 10 == 0
                or i == len(queries)
            ):
                print(
                    f"[{scope.upper()}] "
                    f"{i}/{len(queries)} complete"
                )

    finally:
        try:
            registry.release()
        except Exception:
            pass

    return evaluated


def agg(rows):
    if not rows:
        return {"n": 0, "top1": None, "top3": None, "mrr": None}

    return {
        "n": len(rows),
        "top1": mean(1.0 if r["rank"] == 1 else 0.0 for r in rows),
        "top3": mean(1.0 if r["rank"] is not None and r["rank"] <= 3 else 0.0 for r in rows),
        "mrr": mean(0.0 if r["rank"] is None else 1.0 / r["rank"] for r in rows),
    }


def main():
    args = parse_args()

    def test_collection(cfg, scope, *extra, **kwargs):
        return args.person_collection if scope == "person" else args.object_collection

    us.collection_for_scope = test_collection

    out_root = CAND_ROOT / "test10_cross_video_benchmark"
    out_root.mkdir(parents=True, exist_ok=True)

    queries = []
    usable_stems = []

    for stem in args.stems:
        cand = CAND_ROOT / stem / "final_db_candidates.json"
        if not cand.is_file():
            print("[SKIP] missing candidates:", stem)
            continue

        data = load_json(cand)
        pm, om, dbf = candidate_maps(data)
        rows = source_rows(stem, pm, om)
        video = find_video(stem)
        queries.extend(
            choose_queries(stem, video, rows, dbf, args.queries_per_track, out_root)
        )
        usable_stems.append(stem)

    if not queries:
        raise RuntimeError("no benchmark queries")

    print("=" * 96)
    print("CROSS-VIDEO WITHHELD BENCHMARK")
    print("=" * 96)
    print("videos  :", len(usable_stems))
    print("queries :", len(queries))
    print("=" * 96)

    cfg = PipelineConfig.load(CONFIG_PATH)

    person_queries = [
        q for q in queries
        if q["scope"] == "person"
    ]
    object_queries = [
        q for q in queries
        if q["scope"] == "object"
    ]

    print("person queries :", len(person_queries))
    print("object queries :", len(object_queries))

    # Critical optimization:
    # load SOLIDER once, evaluate all person queries, release;
    # then load DINOv2 once, evaluate all object queries.
    person = evaluate_scope_queries(
        "person",
        person_queries,
        args,
        cfg,
    )
    obj = evaluate_scope_queries(
        "object",
        object_queries,
        args,
        cfg,
    )
    evaluated = person + obj

    summary = {
        "overall": agg(evaluated),
        "person": agg(person),
        "object": agg(obj),
    }

    payload = {
        "videos": usable_stems,
        "person_collection": args.person_collection,
        "object_collection": args.object_collection,
        "summary": summary,
        "queries": evaluated,
    }

    out = out_root / "cross_video_benchmark.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 96)
    print("CROSS-VIDEO BENCHMARK RESULT")
    print("=" * 96)
    for name in ("overall", "person", "object"):
        s = summary[name]
        if not s["n"]:
            print(f"{name:8s}: n=0")
        else:
            print(
                f"{name:8s}: n={s['n']:3d} | "
                f"Top-1={s['top1']*100:6.2f}% | "
                f"Top-3={s['top3']*100:6.2f}% | "
                f"MRR={s['mrr']:.4f}"
            )

    fails = [r for r in evaluated if r["rank"] != 1]
    print("\nTop-1 failures:", len(fails))
    for r in fails[:30]:
        print(
            f" - {r['scope']} expected={r['expected_track_key']} "
            f"rank={r['rank']} returned={r['returned_track_keys'][:5]}"
        )

    print("json :", out)
    print("=" * 96)


if __name__ == "__main__":
    main()
