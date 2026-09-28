"""DBSCAN v6 플러그인 — 사용자 dbscan_person_v6_cl_fixed.py 알고리즘 (cluster_dbscan_qdrant.py 와 같은 assign_clusters).

  1) primary(기본 solider) 정확 kNN: K 개, score ≥ score_threshold 인 후보
  2) 결합 벡터 L2(w1·L2(v1) ⊕ w2·L2(v2) ⊕ …) 의 cosine 거리 ≤ eps 로 후보 재필터 (hybrid)
  3) greedy 단일 패스 배정 (이웃 ≥ min_faces 이면 core; 먼저 배정된 이웃의 군집을 따름)
required_vectors = combined_vectors 이므로 driver 가 그 named vector 들을 함께 받아 넘긴다.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from clustering.base import BaseClusterer, ClusterResult


def combine_rows(parts: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """행 단위 L2(w1·L2(p1) ⊕ …). 0 행은 0 으로 남긴다 (원본 combined_vec 의 극소 norm 규칙과 동일한 효과)."""
    if len(parts) != len(weights):
        raise ValueError("combined_vectors 와 weights 길이가 다릅니다")
    cols = []
    for p, w in zip(parts, weights):
        p = np.asarray(p, dtype=np.float64)
        norm = np.linalg.norm(p, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        cols.append(float(w) * (p / norm))
    out = np.concatenate(cols, axis=1)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (out / norm).astype(np.float32)


class DBSCANv6Clusterer(BaseClusterer):
    name = "dbscan_v6"

    def __init__(self, knn: int = 25, score_threshold: float = 0.88, eps: float = 0.12, min_faces: int = 3,
                 combined_vectors: Sequence[str] = ("siglip2", "irra", "solider"),
                 weights: Sequence[float] = (0.1, 0.3, 0.6), knn_device: str = "auto"):
        self.knn = int(knn)
        self.score_threshold = float(score_threshold)
        self.eps = float(eps)
        self.min_faces = int(min_faces)
        self.combined_vectors = tuple(str(v) for v in combined_vectors)
        self.weights = tuple(float(w) for w in weights)
        self.knn_device = str(knn_device)
        if len(self.combined_vectors) != len(self.weights):
            raise ValueError("combined_vectors 와 weights 길이가 다릅니다")
        if self.knn < 1 or self.min_faces < 1 or not (0 <= self.eps <= 2):
            raise ValueError("knn·min_faces 는 1 이상, eps 는 0~2 (cosine 거리)")
        self.required_vectors = self.combined_vectors

    def params(self) -> Dict[str, Any]:
        return dict(knn=self.knn, score_threshold=self.score_threshold, eps=self.eps, min_faces=self.min_faces,
                    combined_vectors=list(self.combined_vectors), weights=list(self.weights), knn_device=self.knn_device,
                    method="greedy_dbscan_v6_cl", exact=True)

    def cluster(self, ids: Sequence[Any], primary: np.ndarray, vectors: Dict[str, np.ndarray], log=print) -> ClusterResult:
        from clustering.cluster_dbscan_qdrant import assign_clusters
        from clustering.cluster_leiden_qdrant import exact_topk

        n = len(ids)
        missing = [v for v in self.combined_vectors if v not in vectors]
        if missing:
            raise ValueError(f"결합 벡터가 없습니다: {missing} (driver 가 required_vectors 를 받아야 함)")
        combined = combine_rows([vectors[v] for v in self.combined_vectors], self.weights)
        valid = np.linalg.norm(primary, axis=1) > 0
        mask: Optional[np.ndarray] = None if bool(valid.all()) else valid
        # K 는 자기 자신 제외 후보 수 (원본: limit=K+1 후 자기 제외). exact_topk 는 자기 자신을 이미 뺀다.
        nbr, sc, s_knn = exact_topk(primary, self.knn, self.score_threshold, self.knn_device, valid_mask=mask, log=log)
        t0 = time.time()
        neighbors: Dict[int, List[int]] = {}
        pairs = 0
        for i in range(n):
            cand = nbr[i][nbr[i] >= 0]
            if cand.size:
                d = 1.0 - (combined[cand].astype(np.float64) @ combined[i].astype(np.float64))
                cand = cand[d <= self.eps]
            lst = cand.tolist()
            neighbors[i] = lst
            pairs += len(lst)
        cluster_map, st = assign_clusters(list(range(n)), neighbors, self.min_faces)
        sizes = Counter(v for v in cluster_map.values() if v != -1)
        labels: List[Optional[int]] = [None if cluster_map[i] == -1 else int(cluster_map[i]) for i in range(n)]
        stats = {**s_knn, "neighbor_pairs": pairs, "avg_degree": (pairs / n) if n else 0.0,
                 "raw_clusters": len(sizes), "singleton_clusters": sum(1 for s in sizes.values() if s == 1),
                 "largest_raw": max(sizes.values(), default=0), "combined_dim": int(combined.shape[1]),
                 "missing_vectors": int((~valid).sum()), **st, "assign_sec": round(time.time() - t0, 3)}
        return ClusterResult(labels=labels, stats=stats)
