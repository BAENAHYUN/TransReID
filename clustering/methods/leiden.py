"""Leiden 플러그인 — cluster_leiden_qdrant.py 의 정확 kNN(exact_topk) → 간선 축약(reduce_edges) → Leiden → 과대 군집 재분할.

기본값은 GUI 4단계(cluster_leiden_qdrant) 와 같다: knn 30, threshold 0.97(person), mutual kNN, resolution 1.0, seed 42.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence

import numpy as np

from clustering.base import BaseClusterer, ClusterResult


class LeidenClusterer(BaseClusterer):
    name = "leiden"

    def __init__(self, knn: int = 30, threshold: float = 0.97, mutual_knn: bool = True, resolution: float = 1.0,
                 seed: int = 42, max_cluster_size: int = 0, refine_rounds: int = 5, knn_device: str = "auto"):
        self.knn = int(knn)
        self.threshold = float(threshold)
        self.mutual_knn = bool(mutual_knn)
        self.resolution = float(resolution)
        self.seed = int(seed)
        self.max_cluster_size = int(max_cluster_size or 0)
        self.refine_rounds = int(refine_rounds)
        self.knn_device = str(knn_device)
        if self.knn < 1:
            raise ValueError("knn 은 1 이상")
        if not (-1.0 <= self.threshold <= 1.0):
            raise ValueError("threshold 는 -1~1 (cosine)")

    def params(self) -> Dict[str, Any]:
        return dict(knn=self.knn, score_threshold=self.threshold, mutual_knn=self.mutual_knn, resolution=self.resolution,
                    seed=self.seed, max_cluster_size=self.max_cluster_size or None, refine_rounds=self.refine_rounds,
                    knn_device=self.knn_device, knn_mode="exact-memory")

    def cluster(self, ids: Sequence[Any], primary: np.ndarray, vectors: Dict[str, np.ndarray], log=print) -> ClusterResult:
        from clustering.cluster_leiden_qdrant import exact_topk, leiden, reduce_edges, refine_oversized

        n = len(ids)
        started = time.time()
        valid = np.linalg.norm(primary, axis=1) > 0
        mask: Optional[np.ndarray] = None if bool(valid.all()) else valid
        nbr, sc, s_knn = exact_topk(primary, self.knn, self.threshold, self.knn_device, valid_mask=mask, log=log)
        edges, weights, s_graph = reduce_edges(nbr, sc, n, self.mutual_knn, started, {**s_knn, "missing_vectors": int((~valid).sum())})
        if edges:
            membership, s_leiden = leiden(ids, edges, weights, self.resolution, self.seed)
        else:
            membership, s_leiden = list(range(n)), {"raw_communities": n, "largest_community": 1 if n else 0, "leiden_sec": 0.0}
        s_ref: Dict[str, Any] = {}
        if self.max_cluster_size and edges:
            membership, s_ref = refine_oversized(n, edges, weights, membership, self.max_cluster_size,
                                                 self.resolution, self.seed, self.refine_rounds)
        return ClusterResult(labels=[int(m) for m in membership], stats={**s_graph, **s_leiden, **s_ref})
