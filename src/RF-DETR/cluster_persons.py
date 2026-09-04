"""
cluster_persons.py — Track-level SOLIDER 임베딩 클러스터링 비교

forensic_person_tracks 컬렉션의 SOLIDER 벡터로 동일 인물 후보 그룹 생성.
10가지 알고리즘을 Silhouette / CH / DB 지수로 비교합니다.

사용법:
  python cluster_persons.py                    # 자동 k 탐색 + 전체 비교
  python cluster_persons.py --k 30             # k 고정
  python cluster_persons.py --algo kmeans      # 단일 알고리즘
  python cluster_persons.py --export results/  # 결과 JSON 저장
  python cluster_persons.py --source SCVD      # 소스 필터
  python cluster_persons.py --split Train      # split 필터

결과:
  - 클러스터별 대표 이미지 경로 (best_path)
  - 알고리즘 비교 테이블 (Silhouette / CH / DB)
  - JSON 저장 (--export)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

_HERE    = Path(__file__).resolve().parent
ROOT_DIR = _HERE.parents[1]
sys.path.insert(0, str(_HERE))

from qdrant_person_manager import _QDRANT_PATH
from qdrant_client import QdrantClient

TRACK_COLLECTION = "forensic_person_tracks"
SCROLL_BATCH     = 500

warnings.filterwarnings("ignore")   # sklearn 경고 억제


# ── Qdrant 스캔 ───────────────────────────────────────────────────────
def load_track_vectors(
    client:       QdrantClient,
    source_filter: str | None = None,
    split_filter:  str | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    forensic_person_tracks 전체 스캔 → (N, 1024) SOLIDER 벡터 + 메타 리스트.
    """
    vecs  = []
    metas = []
    offset = None

    while True:
        result, next_offset = client.scroll(
            collection_name = TRACK_COLLECTION,
            offset          = offset,
            limit           = SCROLL_BATCH,
            with_payload    = True,
            with_vectors    = True,
        )
        for p in result:
            pay = p.payload or {}
            if source_filter and pay.get("source", "") != source_filter:
                continue
            if split_filter  and pay.get("split",  "") != split_filter:
                continue
            v = (p.vector or {}).get("solider", [])
            if not v:
                continue
            vecs.append(np.array(v, dtype=np.float32))
            metas.append({
                "id":        p.id,
                "video":     pay.get("video",     ""),
                "track":     pay.get("track",     ""),
                "source":    pay.get("source",    ""),
                "split":     pay.get("split",     ""),
                "category":  pay.get("category",  ""),
                "n_frames":  pay.get("n_frames",   0),
                "best_path": pay.get("best_path", ""),
            })
        if next_offset is None:
            break
        offset = next_offset

    print(f"[+] 트랙 로드: {len(metas):,}개")
    return np.stack(vecs) if vecs else np.empty((0, 1024)), metas


# ── 평가 지수 ─────────────────────────────────────────────────────────
def evaluate(X: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import (
        silhouette_score, calinski_harabasz_score, davies_bouldin_score
    )
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return {"n_clusters": n_clusters, "silhouette": -1.0, "ch": -1.0, "db": -1.0}
    mask = labels != -1
    Xm, Lm = X[mask], labels[mask]
    if len(set(Lm)) < 2:
        return {"n_clusters": n_clusters, "silhouette": -1.0, "ch": -1.0, "db": -1.0}
    return {
        "n_clusters":  n_clusters,
        "silhouette":  round(float(silhouette_score(Xm, Lm, sample_size=min(5000, len(Xm)))), 4),
        "ch":          round(float(calinski_harabasz_score(Xm, Lm)), 2),
        "db":          round(float(davies_bouldin_score(Xm, Lm)), 4),
    }


# ── 알고리즘 ─────────────────────────────────────────────────────────
def run_kmeans(X, k):
    from sklearn.cluster import KMeans
    m = KMeans(n_clusters=k, n_init=10, random_state=42)
    return m.fit_predict(X)

def run_minibatch_kmeans(X, k):
    from sklearn.cluster import MiniBatchKMeans
    m = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42, batch_size=1024)
    return m.fit_predict(X)

def run_dbscan(X, eps=0.3, min_samples=3):
    from sklearn.cluster import DBSCAN
    m = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
    return m.fit_predict(X)

def run_hdbscan(X, min_cluster_size=5):
    try:
        import hdbscan
        m = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    except ImportError:
        from sklearn.cluster import HDBSCAN
        m = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    return m.fit_predict(X)

def run_agglomerative(X, k):
    from sklearn.cluster import AgglomerativeClustering
    m = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    return m.fit_predict(X)

def run_gmm(X, k):
    from sklearn.mixture import GaussianMixture
    m = GaussianMixture(n_components=k, covariance_type="full",
                        n_init=3, random_state=42, max_iter=200)
    m.fit(X)
    return m.predict(X)

def run_spectral(X, k):
    from sklearn.cluster import SpectralClustering
    m = SpectralClustering(n_clusters=k, affinity="rbf",
                           n_init=5, random_state=42, n_jobs=-1)
    return m.fit_predict(X)

def run_birch(X, k):
    from sklearn.cluster import Birch
    m = Birch(n_clusters=k, threshold=0.5)
    return m.fit_predict(X)

def run_optics(X, min_samples=5):
    from sklearn.cluster import OPTICS
    m = OPTICS(min_samples=min_samples, metric="cosine", n_jobs=-1)
    return m.fit_predict(X)

def run_finch(X):
    """
    FINCH (First Integer Neighbor Clustering Hierarchy).
    finch-clust 패키지가 없으면 간이 구현 폴백.
    """
    try:
        from finch import FINCH
        labels, _, _ = FINCH(X, distance="cosine", verbose=False)
        return labels[:, 0]   # 첫 번째 파티션
    except ImportError:
        # 폴백: 가장 가까운 이웃 기반 greedy union-find
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2, metric="cosine", n_jobs=-1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        parent = list(range(len(X)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(X)):
            j = indices[i, 1]
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj

        root_map: dict[int, int] = {}
        labels = []
        for i in range(len(X)):
            r = find(i)
            if r not in root_map:
                root_map[r] = len(root_map)
            labels.append(root_map[r])
        return np.array(labels)


ALGO_MAP = {
    "kmeans":       run_kmeans,
    "minibatch":    run_minibatch_kmeans,
    "dbscan":       run_dbscan,
    "hdbscan":      run_hdbscan,
    "agglomerative":run_agglomerative,
    "gmm":          run_gmm,
    "spectral":     run_spectral,
    "birch":        run_birch,
    "optics":       run_optics,
    "finch":        run_finch,
}

K_BASED   = {"kmeans", "minibatch", "agglomerative", "gmm", "spectral", "birch"}
DENSITY   = {"dbscan", "hdbscan", "optics", "finch"}


# ── 자동 k 탐색 ────────────────────────────────────────────────────────
def find_best_k(X: np.ndarray, k_min: int = 5, k_max: int = 200) -> int:
    """Silhouette 기반 최적 k 탐색 (대략적, KMeans 사용)."""
    from sklearn.cluster import MiniBatchKMeans

    k_candidates = list(range(k_min, min(k_max + 1, len(X) // 5), max(1, (k_max - k_min) // 20)))
    best_k, best_sil = k_candidates[0], -1.0

    print(f"\n[k 탐색] {k_min}~{k_max} 범위, {len(k_candidates)}개 후보...")
    for k in k_candidates:
        m = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42, batch_size=1024)
        labels = m.fit_predict(X)
        ev = evaluate(X, labels)
        sil = ev["silhouette"]
        if sil > best_sil:
            best_sil, best_k = sil, k
        print(f"  k={k:4d}  sil={sil:.4f}", end="\r")

    print(f"\n[+] 최적 k={best_k}  (sil={best_sil:.4f})")
    return best_k


# ── 결과 집계 ─────────────────────────────────────────────────────────
def build_cluster_summary(labels: np.ndarray, metas: list[dict]) -> list[dict]:
    from collections import defaultdict
    groups: dict[int, list[dict]] = defaultdict(list)
    for label, meta in zip(labels.tolist(), metas):
        groups[label].append(meta)

    summary = []
    for label, members in sorted(groups.items()):
        summary.append({
            "cluster":    label,
            "n_tracks":   len(members),
            "best_path":  members[0]["best_path"],
            "videos":     list({m["video"] for m in members}),
            "tracks":     [m["track"] for m in members],
        })
    return sorted(summary, key=lambda x: x["n_tracks"], reverse=True)


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="인물 트랙 클러스터링")
    ap.add_argument("--k",        type=int,  default=None,
                    help="클러스터 수 (미지정 시 자동 탐색)")
    ap.add_argument("--k-min",    type=int,  default=5)
    ap.add_argument("--k-max",    type=int,  default=200)
    ap.add_argument("--algo",     default="all",
                    choices=["all"] + list(ALGO_MAP.keys()),
                    help="실행할 알고리즘 (기본: all)")
    ap.add_argument("--export",   type=Path, default=None,
                    help="결과 JSON 저장 디렉터리")
    ap.add_argument("--source",   type=str,  default=None, help="소스 필터 (예: SCVD)")
    ap.add_argument("--split",    type=str,  default=None, help="split 필터 (예: Train)")
    ap.add_argument("--top-n",    type=int,  default=5,    help="출력할 상위 클러스터 수")
    ap.add_argument("--dbscan-eps",      type=float, default=0.30)
    ap.add_argument("--hdbscan-min",     type=int,   default=5)
    ap.add_argument("--optics-min",      type=int,   default=5)
    args = ap.parse_args()

    print("=" * 72)
    print("PERSON CLUSTERING")
    print("=" * 72)

    client = QdrantClient(path=_QDRANT_PATH)

    # 트랙 컬렉션 존재 확인
    existing = [c.name for c in client.get_collections().collections]
    if TRACK_COLLECTION not in existing:
        print(f"[!] {TRACK_COLLECTION} 컬렉션 없음.")
        print("    → python build_track_embeddings.py 먼저 실행하세요.")
        client.close()
        return

    X, metas = load_track_vectors(client, args.source, args.split)
    client.close()

    if len(X) < 10:
        print(f"[!] 트랙 수 부족: {len(X)}개. 최소 10개 필요.")
        return

    print(f"임베딩 shape: {X.shape}")

    # k 결정
    k = args.k
    if k is None and not (args.algo in DENSITY or args.algo == "all"):
        k = find_best_k(X, args.k_min, args.k_max)
    elif k is None:
        k = max(10, len(X) // 20)   # density 알고리즘용 참고값

    # 실행할 알고리즘 목록
    algos = list(ALGO_MAP.keys()) if args.algo == "all" else [args.algo]

    results_table = []
    all_labels    = {}

    print(f"\n{'알고리즘':20}  {'클러스터':>8}  {'Silhouette':>10}  {'CH':>10}  {'DB':>8}  {'시간':>6}")
    print("-" * 72)

    for algo_name in algos:
        fn = ALGO_MAP[algo_name]
        t0 = time.time()
        try:
            if algo_name in K_BASED:
                labels = fn(X, k)
            elif algo_name == "dbscan":
                labels = fn(X, eps=args.dbscan_eps)
            elif algo_name == "hdbscan":
                labels = fn(X, min_cluster_size=args.hdbscan_min)
            elif algo_name == "optics":
                labels = fn(X, min_samples=args.optics_min)
            else:
                labels = fn(X)
        except Exception as e:
            print(f"  {algo_name:20}  실패: {e}")
            continue

        elapsed = time.time() - t0
        ev      = evaluate(X, labels)
        all_labels[algo_name] = labels

        row = {
            "algo":       algo_name,
            "elapsed_s":  round(elapsed, 2),
            **ev,
        }
        results_table.append(row)

        print(f"  {algo_name:20}  {ev['n_clusters']:>8,}  {ev['silhouette']:>10.4f}  "
              f"{ev['ch']:>10.1f}  {ev['db']:>8.4f}  {elapsed:>5.1f}s")

    # 최고 알고리즘 (Silhouette)
    best_row = max(results_table, key=lambda r: r["silhouette"]) if results_table else None

    if best_row:
        print(f"\n[최고 Silhouette] {best_row['algo']}  ({best_row['silhouette']:.4f})")
        best_labels = all_labels[best_row["algo"]]
        summary     = build_cluster_summary(np.array(best_labels), metas)

        print(f"\n[상위 {args.top_n}개 클러스터]")
        for c in summary[:args.top_n]:
            print(f"  cluster={c['cluster']:4d}  n_tracks={c['n_tracks']:4d}  "
                  f"best={c['best_path'][:60]}")

        # JSON 저장
        if args.export:
            args.export.mkdir(parents=True, exist_ok=True)
            out = {
                "comparison":    results_table,
                "best_algo":     best_row["algo"],
                "best_k":        best_row["n_clusters"],
                "best_silhouette": best_row["silhouette"],
                "clusters":      summary,
            }
            out_path = args.export / "cluster_results.json"
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n[+] 결과 저장: {out_path}")

    print("\n" + "=" * 72)
    print("CLUSTERING COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
