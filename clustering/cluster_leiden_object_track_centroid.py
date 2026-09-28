#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cluster_leiden_object_track_centroid.py

Track-centroid + mutual-kNN Leiden for forensic_object.

Problem with crop-level Leiden at low threshold (e.g. 0.85):
  - DINOv2 encodes category-level similarity (all cars similar, all bikes similar)
  - → Leiden produces category clusters, not instance clusters
  - We want: same vehicle/object across frames = one cluster

Fix (identical logic to person version):
  1. Scroll forensic_object, fetch DINOv2 vectors + payload
  2. Group by (video_stem, track_id) → one group per object track per video
  3. Per-group: L2-normalize each crop vector → average → re-normalize → centroid
  4. Build mutual kNN: edge (i,j) only if j∈kNN(i) AND i∈kNN(j) AND sim ≥ threshold
  5. Leiden on track graph (N_tracks << N_crops)
  6. Propagate cluster_id back to all crop point_ids in each track
  7. Write Qdrant payload + save JSONL

Note on threshold:
  DINOv2 is more category-sensitive than SOLIDER.
  Start sweep at 0.95+; expect the right instance-level threshold to be 0.96-0.98.

Usage:
    python cluster_leiden_object_track_centroid.py --dry-run
    python cluster_leiden_object_track_centroid.py --threshold 0.96
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

COLLECTION = "forensic_object"
VECTOR_NAME = "dinov2"
SOURCE_VALUE = "final_db_candidates"


# ---------------------------------------------------------------------------
# Qdrant HTTP helper
# ---------------------------------------------------------------------------

class Qdrant:
    def __init__(self, url: str, api_key: Optional[str], timeout: int):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"Content-Type": "application/json"})
        if api_key:
            self.s.headers.update({"api-key": api_key})

    def get(self, path: str) -> Dict[str, Any]:
        r = self.s.get(self.url + path, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f"GET {path} → {r.status_code}\n{r.text[:2000]}")
        return r.json()

    def post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        r = self.s.post(self.url + path, json=body, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f"POST {path} → {r.status_code}\n{r.text[:2000]}")
        return r.json()

    def scroll_with_vectors(
        self,
        collection: str,
        vector_name: str,
        source: Optional[str],
        batch: int,
        max_points: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        offset = None
        while True:
            body: Dict[str, Any] = {
                "limit": batch,
                "with_payload": True,
                "with_vector": [vector_name],
            }
            if source:
                body["filter"] = {
                    "must": [{"key": "source", "match": {"value": source}}]
                }
            if offset is not None:
                body["offset"] = offset

            data = self.post(f"/collections/{collection}/points/scroll", body)
            result = data.get("result") or {}
            batch_pts = result.get("points") or []

            for pt in batch_pts:
                points.append(pt)
                if max_points is not None and len(points) >= max_points:
                    return points

            offset = result.get("next_page_offset")
            if not batch_pts or offset is None:
                break
        return points

    def set_payload(
        self,
        collection: str,
        ids: Sequence[Any],
        payload: Dict[str, Any],
    ):
        if not ids:
            return
        self.post(
            f"/collections/{collection}/points/payload?wait=true",
            {"payload": payload, "points": list(ids)},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chunks(seq: Sequence[Any], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def object_track_key(payload: Dict[str, Any]) -> str:
    """
    Unique key per object track per video.
    Objects have no canonical_person_id — use track_id directly.
    Falls back chain: long_track_id → track_id → selected_track_id → "0"
    """
    stem = str(payload.get("video_stem", payload.get("video", "unk")))
    tid = str(
        payload.get("long_track_id",
        payload.get("track_id",
        payload.get("selected_track_id", "0")))
    )
    return f"{stem}||track||{tid}"


def stable_cluster_id(member_keys: List[str]) -> str:
    raw = "|".join(sorted(member_keys))
    h = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return f"leiden:obj:{h}"


# ---------------------------------------------------------------------------
# Graph + Leiden
# ---------------------------------------------------------------------------

def build_mutual_knn_graph(
    centroids: np.ndarray,
    knn: int,
    threshold: float,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """
    Mutual kNN on L2-normalized centroids.
    Edge (i,j) iff j∈topK(i) AND i∈topK(j) AND cosine_sim ≥ threshold.
    For N ≲ 10000 the full matrix is tractable.
    """
    N = len(centroids)
    sim = centroids @ centroids.T   # cosine sim (vectors are L2-normalized)
    np.fill_diagonal(sim, -1.0)

    actual_k = min(knn, N - 1)
    neighbor_sets: List[set] = []
    for i in range(N):
        row = sim[i]
        top_k = np.argpartition(row, -actual_k)[-actual_k:]
        neighbors = {int(j) for j in top_k if float(row[j]) >= threshold}
        neighbor_sets.append(neighbors)

    edges: List[Tuple[int, int]] = []
    weights: List[float] = []
    for i in range(N):
        for j in neighbor_sets[i]:
            if j > i and i in neighbor_sets[j]:
                edges.append((i, j))
                weights.append(float(sim[i, j]))

    return edges, weights


def run_leiden(
    n_nodes: int,
    edges: List[Tuple[int, int]],
    weights: List[float],
    resolution: float,
    seed: int,
) -> List[int]:
    try:
        import igraph as ig
        import leidenalg as la
    except ImportError as e:
        raise RuntimeError("pip install python-igraph leidenalg") from e

    g = ig.Graph(n=n_nodes, edges=list(edges), directed=False)
    if weights:
        g.es["weight"] = list(weights)
    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight" if weights else None,
        resolution_parameter=resolution,
        seed=seed,
        n_iterations=-1,
    )
    return list(part.membership)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Track-centroid mutual-kNN Leiden for forensic_object (DINOv2)"
    )
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--api-key", default=None)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--source", default=SOURCE_VALUE)
    p.add_argument("--collection", default=COLLECTION)
    p.add_argument("--vector", default=VECTOR_NAME)
    p.add_argument("--knn", type=int, default=15,
                   help="Mutual kNN per track centroid (default 15)")
    p.add_argument("--threshold", type=float, default=0.95,
                   help="Min cosine sim for an edge. Start at 0.95 for DINOv2 (default 0.95)")
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--min-cluster-size", type=int, default=2,
                   help="Min tracks per cluster (default 2)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scroll-batch", type=int, default=128)
    p.add_argument("--output-dir",
                   default="outputs/clustering/leiden_object_track_centroid")
    p.add_argument("--dry-run", action="store_true",
                   help="Build graph + Leiden but do NOT write Qdrant payloads")
    args = p.parse_args()

    q = Qdrant(args.qdrant_url, args.api_key, args.timeout)
    q.get(f"/collections/{args.collection}")

    print("=" * 72)
    print("TRACK-CENTROID LEIDEN  (object / DINOv2)")
    print("=" * 72)
    print(f"collection : {args.collection}")
    print(f"vector     : {args.vector}")
    print(f"source     : {args.source}")
    print(f"knn        : {args.knn}  threshold : {args.threshold}")
    print(f"resolution : {args.resolution}  min_cluster : {args.min_cluster_size}")
    print()

    # ------------------------------------------------------------------
    # 1. Scroll
    # ------------------------------------------------------------------
    t0 = time.time()
    print("Step 1 / 5  Scrolling points + vectors ...")
    all_points = q.scroll_with_vectors(
        args.collection, args.vector, args.source, args.scroll_batch
    )
    print(f"  fetched {len(all_points):,} points  ({time.time()-t0:.1f}s)")

    if not all_points:
        print("No points. Exiting.")
        return

    # ------------------------------------------------------------------
    # 2. Group by object track key
    # ------------------------------------------------------------------
    print("Step 2 / 5  Grouping by object track ...")
    track_to_pids: Dict[str, List[Any]] = defaultdict(list)
    track_to_vecs: Dict[str, List[np.ndarray]] = defaultdict(list)
    skipped = 0

    for pt in all_points:
        pid = pt["id"]
        payload = pt.get("payload") or {}
        vec_dict = pt.get("vector") or {}
        raw_vec = vec_dict.get(args.vector)
        if raw_vec is None:
            skipped += 1
            continue
        key = object_track_key(payload)
        track_to_pids[key].append(pid)
        track_to_vecs[key].append(np.asarray(raw_vec, dtype=np.float32))

    track_keys_list = sorted(track_to_pids.keys())
    n_tracks = len(track_keys_list)
    print(f"  tracks : {n_tracks:,}  skipped (no vector) : {skipped}")

    # ------------------------------------------------------------------
    # 3. Per-track L2-normalized centroid
    # ------------------------------------------------------------------
    print("Step 3 / 5  Computing track centroids ...")
    vec_dim = len(track_to_vecs[track_keys_list[0]][0])
    centroids = np.zeros((n_tracks, vec_dim), dtype=np.float32)
    for i, key in enumerate(track_keys_list):
        vecs = np.stack(
            [l2_normalize(v) for v in track_to_vecs[key]],
            axis=0,
        )
        mean_v = np.mean(vecs, axis=0)
        centroids[i] = l2_normalize(mean_v)
    print(f"  centroid matrix : {centroids.shape}  ({centroids.nbytes/1e6:.1f} MB)")

    # ------------------------------------------------------------------
    # 4. Mutual kNN graph
    # ------------------------------------------------------------------
    print(f"Step 4 / 5  Mutual kNN graph (k={args.knn}, thr={args.threshold}) ...")
    t1 = time.time()
    edges, weights = build_mutual_knn_graph(centroids, args.knn, args.threshold)
    avg_deg = (2 * len(edges) / n_tracks) if n_tracks else 0.0
    print(f"  edges={len(edges):,}  avg_degree={avg_deg:.3f}  ({time.time()-t1:.2f}s)")

    # ------------------------------------------------------------------
    # 5. Leiden
    # ------------------------------------------------------------------
    print(f"Step 5 / 5  Leiden ...")
    t2 = time.time()
    membership = run_leiden(n_tracks, edges, weights, args.resolution, args.seed)

    sizes: Dict[int, int] = defaultdict(int)
    for m in membership:
        sizes[int(m)] += 1
    raw_communities = len(sizes)
    largest_tracks = max(sizes.values()) if sizes else 0
    print(f"  raw_communities={raw_communities:,}  largest(tracks)={largest_tracks}  "
          f"({time.time()-t2:.3f}s)")

    # ------------------------------------------------------------------
    # Normalize: min_cluster_size, stable IDs
    # ------------------------------------------------------------------
    community_to_keys: Dict[int, List[str]] = defaultdict(list)
    for i, key in enumerate(track_keys_list):
        community_to_keys[int(membership[i])].append(key)

    assignments: Dict[Any, Dict] = {}
    kept_clusters = 0
    clustered_tracks = 0
    noise_tracks = 0
    clustered_points = 0
    noise_points = 0

    for raw_id, member_keys in community_to_keys.items():
        n_member_tracks = len(member_keys)
        total_pts = sum(len(track_to_pids[k]) for k in member_keys)

        if n_member_tracks < args.min_cluster_size:
            for key in member_keys:
                for pid in track_to_pids[key]:
                    assignments[pid] = {
                        "cluster_id": None,
                        "cluster_size": len(track_to_pids[key]),
                        "cluster_tracks": 1,
                        "raw_leiden_id": raw_id,
                        "noise": True,
                    }
                    noise_points += 1
            noise_tracks += n_member_tracks
        else:
            cid = stable_cluster_id(member_keys)
            kept_clusters += 1
            clustered_tracks += n_member_tracks
            clustered_points += total_pts
            for key in member_keys:
                for pid in track_to_pids[key]:
                    assignments[pid] = {
                        "cluster_id": cid,
                        "cluster_size": total_pts,
                        "cluster_tracks": n_member_tracks,
                        "raw_leiden_id": raw_id,
                        "noise": False,
                    }

    print()
    print(f"  kept_clusters      : {kept_clusters:,}")
    print(f"  clustered_tracks   : {clustered_tracks:,}  / {n_tracks:,}")
    print(f"  noise_tracks       : {noise_tracks:,}")
    print(f"  clustered_points   : {clustered_points:,}")
    print(f"  noise_points       : {noise_points:,}")

    # ------------------------------------------------------------------
    # Write Qdrant payloads
    # ------------------------------------------------------------------
    if not args.dry_run:
        print("\nWriting Qdrant payloads ...")
        by_cluster: Dict[str, List[Any]] = defaultdict(list)
        noise_ids: List[Any] = []

        for pid, a in assignments.items():
            if a["cluster_id"] is None:
                noise_ids.append(pid)
            else:
                by_cluster[a["cluster_id"]].append(pid)

        base_payload = {
            "cluster_method": "track_centroid_mutual_knn_leiden",
            "cluster_vector": args.vector,
            "cluster_knn": args.knn,
            "cluster_score_threshold": args.threshold,
            "cluster_resolution": args.resolution,
        }

        for cid, ids in by_cluster.items():
            a0 = assignments[ids[0]]
            q.set_payload(args.collection, ids, {
                **base_payload,
                "cluster_leiden_id": cid,
                "cluster_leiden_size": a0["cluster_size"],
                "cluster_leiden_tracks": a0["cluster_tracks"],
                "cluster_leiden_noise": False,
            })

        for batch in chunks(noise_ids, 5000):
            q.set_payload(args.collection, batch, {
                **base_payload,
                "cluster_leiden_id": None,
                "cluster_leiden_size": 1,
                "cluster_leiden_tracks": 1,
                "cluster_leiden_noise": True,
            })
        print("  Done.")
    else:
        print("\n[DRY-RUN] Qdrant payload NOT written.")

    # ------------------------------------------------------------------
    # Save JSONL + report
    # ------------------------------------------------------------------
    out = Path(args.output_dir) / "object"
    out.mkdir(parents=True, exist_ok=True)

    jsonl_path = out / "object_leiden_assignments.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for pid, a in assignments.items():
            f.write(json.dumps({
                "point_id": pid,
                "cluster_id": a["cluster_id"],
                "cluster_size": a["cluster_size"],
                "cluster_tracks": a.get("cluster_tracks", 1),
                "raw_leiden_id": a["raw_leiden_id"],
                "noise": a["noise"],
            }, ensure_ascii=False) + "\n")

    stats = {
        "total_points": len(all_points),
        "total_tracks": n_tracks,
        "edges": len(edges),
        "avg_degree": round(avg_deg, 3),
        "raw_communities": raw_communities,
        "largest_community_tracks": largest_tracks,
        "kept_clusters": kept_clusters,
        "clustered_tracks": clustered_tracks,
        "noise_tracks": noise_tracks,
        "clustered_points": clustered_points,
        "noise_points": noise_points,
    }
    report = {
        "config": {
            "collection": args.collection,
            "vector": args.vector,
            "source": args.source,
            "knn": args.knn,
            "threshold": args.threshold,
            "resolution": args.resolution,
            "min_cluster_size": args.min_cluster_size,
            "method": "track_centroid_mutual_knn_leiden",
        },
        "stats": stats,
        "dry_run": args.dry_run,
    }
    (out / "object_leiden_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print("=" * 72)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print()
    print(f"JSONL    : {jsonl_path}")
    print(f"Report   : {out / 'object_leiden_report.json'}")


if __name__ == "__main__":
    main()
