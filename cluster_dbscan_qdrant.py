#!/usr/bin/env python
"""DBSCAN(근사) 사람 클러스터링을 Leiden 파이프라인과 같은 방식으로 실행한다.

알고리즘은 `dbscan_person_v6_cl_fixed.py` 를 그대로 옮겼다 (수치·순서 동일):

  combined = L2( concat( w_s * L2(siglip2), w_i * L2(irra), w_o * L2(solider) ) )
  이웃(pid)  = solider kNN(K, score >= 1 - EPS) 중 pid 자신을 뺀 뒤
               cosine_distance(combined[pid], combined[hit]) <= EPS 인 것 (점수순 유지)
  1차 배정   = scroll 순서로 순회. 이미 배정된 이웃이 있으면 그 클러스터(이웃 순서상 첫 것),
               없고 이웃 수 >= MIN_FACES(core) 면 새 클러스터, 아니면 보류
  2차 배정   = 보류 점은 배정된 이웃이 있으면 합류, 없으면 노이즈(-1)

원본과 다른 점은 "실행 방식" 뿐이다 (cluster_leiden_qdrant.py 와 동일한 규약):
  * 로컬 모드(qdrant_local 폴더) 대신 pipeline.yaml 의 Qdrant 서버를 쓴다.
  * --sources 로 대상을 제한하고, 이웃 검색도 같은 source 안에서만 한다
    (원본은 컬렉션 전체를 대상으로 하므로, 부분집합에서는 서버 필터가 그 등가물이다).
  * 이웃 검색은 quantization 을 무시한 정확 점수를 쓴다 (원본 로컬 모드는 양자화가 없다).
  * --dry-run 이면 payload 를 쓰지 않는다. 기록 시 원본과 같은 정수 키(--payload-key)와
    문자열 키 cluster_dbscan_id 를 함께 쓴다.
  * 산출물: <output-dir>/<target>/<target>_dbscan_assignments.jsonl, <target>_dbscan_report.json
    (build_leiden_gallery.py / compare_cluster_results.py 가 그대로 읽는다).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from report_common import (DEFAULT_CONFIG_PATH, load_pipeline_settings, normalize_sources,
                           resolve, validate_threshold)

CONFIG_HELP = "생략 시 --config 의 pipeline.yaml 값"
DEFAULT_VECTORS = ("siglip2", "irra", "solider")
DEFAULT_WEIGHTS = (0.1, 0.3, 0.6)


# ---------------------------------------------------------------------------
# 순수 함수 (테스트 대상) — 원본 스크립트의 수식/루프를 그대로 옮긴 것
# ---------------------------------------------------------------------------
_TINY_NORM = 10 * np.finfo(np.float64).eps


def l2(v: np.ndarray) -> np.ndarray:
    """sklearn.preprocessing.normalize([v])[0] 와 동일.

    sklearn 은 norm < 10*eps(float64) 이면 분모를 1 로 둔다(_handle_zeros_in_scale) — 0 벡터와
    극소 벡터 모두 그대로 반환한다.
    """
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v if n < _TINY_NORM else v / n


def combined_vec(parts: Sequence[Sequence[float]], weights: Sequence[float]) -> np.ndarray:
    if len(parts) != len(weights):
        raise ValueError("parts/weights 길이가 다릅니다")
    return l2(np.concatenate([w * l2(p) for p, w in zip(parts, weights)]))


def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def filter_neighbors(pid: Any, hit_ids: Sequence[Any], index: Dict[Any, int], matrix: np.ndarray,
                     eps: float) -> List[Any]:
    """원본: [h.id for h in hits if h.id != pid and h.id in all_vecs and cos_dist(...) <= EPS]

    matrix 는 (N, D) combined 벡터 행렬(float32 저장), index 는 point_id -> 행 번호.
    거리는 float64 로 계산한다 (원본은 float64).
    """
    row = index.get(pid)
    if row is None:
        return []
    cands = [h for h in hit_ids if h != pid and h in index]
    if not cands:
        return []
    own = matrix[row].astype(np.float64)
    dots = matrix[[index[h] for h in cands]].astype(np.float64) @ own
    return [h for h, d in zip(cands, dots) if float(1.0 - d) <= eps]


def assign_clusters(ids: Sequence[Any], neighbors_map: Dict[Any, Sequence[Any]], min_faces: int):
    """원본 Step 3 + Step 4. 반환: (cluster_map pid->int(-1=노이즈), 통계 dict)."""
    cluster_map: Dict[Any, int] = {}
    deferred: List[Any] = []
    next_cid = 0
    core_points = 0

    for pid in ids:
        if pid in cluster_map:
            continue
        neighbors = neighbors_map.get(pid, ())
        is_core = len(neighbors) >= min_faces
        core_points += int(is_core)
        assigned = next((cluster_map[n] for n in neighbors if n in cluster_map), None)
        if assigned is not None:
            cluster_map[pid] = assigned
        elif is_core:
            cluster_map[pid] = next_cid
            next_cid += 1
        else:
            deferred.append(pid)

    deferred_joined = deferred_noise_propagated = 0
    for pid in deferred:
        if pid in cluster_map:
            continue
        # 원본 그대로: 첫 "배정된" 이웃의 값을 따른다. 그 값이 -1(앞서 노이즈가 된 보류점)이면
        # 노이즈가 전파된다 — 배정 동작은 유지하고 통계만 구분한다.
        assigned = next((cluster_map[n] for n in neighbors_map.get(pid, ()) if n in cluster_map), None)
        cluster_map[pid] = assigned if assigned is not None else -1
        if assigned is None:
            continue
        if assigned == -1:
            deferred_noise_propagated += 1
        else:
            deferred_joined += 1

    return cluster_map, dict(core_points=core_points, deferred=len(deferred),
                             deferred_joined=deferred_joined,
                             deferred_noise_propagated=deferred_noise_propagated, raw_clusters=next_cid)


def stable_cluster_id(target: str, method: str, ids: Sequence[Any]) -> str:
    raw = target + "|" + method + "|" + "|".join(sorted(map(str, ids)))
    return f"dbscan:{target}:{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def normalize_assignments(target: str, method: str, ids: Sequence[Any], cluster_map: Dict[Any, int],
                          degree: Dict[Any, int], min_faces: int):
    groups: Dict[int, List[Any]] = defaultdict(list)
    for pid in ids:
        groups[int(cluster_map[pid])].append(pid)

    assignments: Dict[Any, Dict[str, Any]] = {}
    clustered = noise = singleton = 0
    largest = 0
    for raw_id, members in groups.items():
        if raw_id == -1:
            for pid in members:
                assignments[pid] = dict(cluster_id=None, cluster_size=1, raw_dbscan_id=-1, noise=True,
                                        degree=degree.get(pid, 0), is_core=degree.get(pid, 0) >= min_faces)
            noise += len(members)
            continue
        cid = stable_cluster_id(target, method, members)
        clustered += len(members)
        singleton += int(len(members) == 1)
        largest = max(largest, len(members))
        for pid in members:
            assignments[pid] = dict(cluster_id=cid, cluster_size=len(members), raw_dbscan_id=raw_id,
                                    noise=False, degree=degree.get(pid, 0),
                                    is_core=degree.get(pid, 0) >= min_faces)
    clusters = len(groups) - (1 if -1 in groups else 0)
    return assignments, dict(kept_clusters=clusters, singleton_clusters=singleton,
                             largest_community=largest, clustered_points=clustered, noise_points=noise)


# ---------------------------------------------------------------------------
# Qdrant 접근 (qdrant-client; 원본과 같은 API 를 서버에 대해 쓴다)
# ---------------------------------------------------------------------------
@dataclass
class Config:
    target: str
    collection: str
    vector: str                      # kNN 에 쓰는 named vector (원본: solider)
    combined_vectors: List[str]
    weights: List[float]
    sources: Optional[List[str]]
    knn: int
    score_threshold: float
    eps: float
    min_faces: int
    query_batch_size: int
    scroll_batch_size: int
    max_points: Optional[int]
    payload_key: str
    exact: bool = True               # 원본(로컬 모드)은 전수 정확 검색. False 면 HNSW 근사
    method: str = "greedy_dbscan_v6_cl"
    # Leiden report 와 같은 키 (갤러리 표시용). DBSCAN 에는 해당 없음.
    resolution: Optional[float] = None
    mutual_knn: bool = False
    max_cluster_size: Optional[int] = None
    min_cluster_size: int = 1


def make_client(url: str, api_key: Optional[str], timeout: int, prefer_grpc: bool):
    from qdrant_client import QdrantClient
    return QdrantClient(url=url, api_key=api_key or None, timeout=timeout, prefer_grpc=prefer_grpc)


def source_filter(sources: Optional[List[str]]):
    if not sources:
        return None
    from qdrant_client import models
    if len(sources) == 1:
        return models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=sources[0]))])
    return models.Filter(must=[models.FieldCondition(key="source", match=models.MatchAny(any=list(sources)))])


def fetch_vectors(client, cfg: Config, log=print):
    """scroll 순서(=원본 순회 순서)대로 id 목록과 combined / kNN 벡터를 모은다.

    반환: ids, index(pid->행), combined 행렬 (N, D) float32, kNN 벡터 행렬 (N, d) float32, 통계.
    메모리: 43k point 기준 combined 2304-d 가 약 400MB (float64 dict 로 두면 1GB+ 라서 행렬로 둔다).
    """
    ids: List[Any] = []
    combined_rows: List[np.ndarray] = []
    knn_rows: List[np.ndarray] = []
    names = list(cfg.combined_vectors)
    if cfg.vector not in names:
        names.append(cfg.vector)
    flt = source_filter(cfg.sources)
    offset = None
    started = time.time()
    missing = 0
    while True:
        points, offset = client.scroll(
            collection_name=cfg.collection, limit=cfg.scroll_batch_size, offset=offset,
            with_vectors=names, with_payload=False, scroll_filter=flt,
        )
        if not points:
            break
        for p in points:
            vec = p.vector or {}
            if any(vec.get(n) is None for n in names):
                missing += 1
                continue
            ids.append(p.id)
            combined_rows.append(combined_vec([vec[n] for n in cfg.combined_vectors], cfg.weights).astype(np.float32))
            knn_rows.append(l2(vec[cfg.vector]).astype(np.float32))
            if cfg.max_points is not None and len(ids) >= cfg.max_points:
                offset = None
                break
        log(f"  벡터 수집 {len(ids):,} ({time.time() - started:.0f}s)")
        if offset is None:
            break
    index = {pid: i for i, pid in enumerate(ids)}
    if len(index) != len(ids):
        raise RuntimeError("scroll 결과에 중복 point id 가 있습니다")
    matrix = np.vstack(combined_rows) if combined_rows else np.zeros((0, 0), dtype=np.float32)
    knn_matrix = np.vstack(knn_rows) if knn_rows else np.zeros((0, 0), dtype=np.float32)
    return ids, index, matrix, knn_matrix, dict(fetch_sec=round(time.time() - started, 3),
                                                 missing_vectors=missing)


def find_neighbors(client, cfg: Config, ids: Sequence[Any], index, matrix, knn_matrix, log=print):
    from qdrant_client import models
    flt = source_filter(cfg.sources)
    # exact=True: HNSW 를 건너뛰고 전수 점수 계산 (원본 로컬 모드와 같은 top-K).
    params = models.SearchParams(exact=cfg.exact, quantization=models.QuantizationSearchParams(ignore=True))
    neighbors_map: Dict[Any, List[Any]] = {}
    hits_total = pairs_total = 0
    started = time.time()
    for i in range(0, len(ids), cfg.query_batch_size):
        batch = ids[i:i + cfg.query_batch_size]
        requests = [models.QueryRequest(query=knn_matrix[index[pid]].tolist(), using=cfg.vector,
                                        limit=cfg.knn, score_threshold=cfg.score_threshold, filter=flt,
                                        params=params, with_payload=False, with_vector=False)
                    for pid in batch]
        results = client.query_batch_points(collection_name=cfg.collection, requests=requests)
        if len(results) != len(batch):
            raise RuntimeError(f"batch result mismatch: {len(results)} != {len(batch)}")
        for pid, res in zip(batch, results):
            hit_ids = [h.id for h in res.points]
            hits_total += len(hit_ids)
            neighbors = filter_neighbors(pid, hit_ids, index, matrix, cfg.eps)
            pairs_total += len(neighbors)
            neighbors_map[pid] = neighbors
        if (i // cfg.query_batch_size) % 20 == 0:
            done = i + len(batch)
            rate = done / max(time.time() - started, 1e-9)
            log(f"  이웃 검색 {done:,}/{len(ids):,} ({rate:,.0f} p/s)")
    return neighbors_map, dict(knn_sec=round(time.time() - started, 3), knn_hits=hits_total,
                               neighbor_pairs=pairs_total,
                               avg_degree=(pairs_total / len(ids)) if ids else 0.0)


def write_payloads(client, cfg: Config, ids, cluster_map, assignments, chunk=1000):
    by_cluster: Dict[int, List[Any]] = defaultdict(list)
    for pid in ids:
        by_cluster[int(cluster_map[pid])].append(pid)
    for raw_id, members in by_cluster.items():
        for j in range(0, len(members), chunk):
            part = members[j:j + chunk]
            client.set_payload(collection_name=cfg.collection, wait=True,
                               payload={cfg.payload_key: raw_id,
                                        "cluster_dbscan_id": assignments[part[0]]["cluster_id"]},
                               points=list(part))


# ---------------------------------------------------------------------------
# 벡터 캐시 (npz): 같은 대상(컬렉션/sources/vectors/weights)에 대해 5분짜리 수집을 반복하지 않는다.
# 컬렉션의 현재 필터 point 수와 캐시 길이가 다르면 사용하지 않는다.
# ---------------------------------------------------------------------------
def cache_meta(cfg: Config) -> Dict[str, Any]:
    return dict(collection=cfg.collection, sources=cfg.sources, combined_vectors=list(cfg.combined_vectors),
                weights=[float(w) for w in cfg.weights], knn_vector=cfg.vector, max_points=cfg.max_points)


def save_vector_cache(path: Path, cfg: Config, ids, matrix, knn_matrix, missing: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, meta=np.array(json.dumps({**cache_meta(cfg), "missing_vectors": int(missing)})),
             ids=np.array([str(x) for x in ids]), matrix=matrix, knn_matrix=knn_matrix)


def load_vector_cache(path: Path, cfg: Config, expected_count: Optional[int]):
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    missing = int(meta.pop("missing_vectors", 0))
    if meta != cache_meta(cfg):
        raise SystemExit(f"vector cache 설정 불일치: {path}\n  cache={meta}\n  now={cache_meta(cfg)}")
    ids = [str(x) for x in data["ids"].tolist()]
    if expected_count is not None and expected_count != len(ids) + missing:
        raise SystemExit(f"vector cache 가 오래됨: 컬렉션 필터 point 수 {expected_count:,} != 캐시 {len(ids) + missing:,} "
                         f"({path}) — 파일을 지우고 다시 실행")
    matrix, knn_matrix = data["matrix"], data["knn_matrix"]
    if matrix.shape[0] != len(ids) or knn_matrix.shape[0] != len(ids):
        raise SystemExit(f"vector cache 손상: 행 수 불일치 ({path})")
    return ids, {pid: i for i, pid in enumerate(ids)}, matrix, knn_matrix, dict(
        fetch_sec=0.0, missing_vectors=missing, vector_cache=str(path))


def count_points(client, cfg: Config) -> Optional[int]:
    try:
        return int(client.count(collection_name=cfg.collection, count_filter=source_filter(cfg.sources), exact=True).count)
    except Exception as exc:  # count 실패는 치명적이지 않다 (캐시 검증만 약해진다)
        print(f"  [경고] point count 실패: {exc}")
        return None


def config_provenance(config_path: Optional[str]):
    if not config_path:
        return dict(config_path=None, config_sha256=None)
    path = Path(config_path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return dict(config_path=str(path), config_sha256=digest)


def parse_floats(text: str, name: str) -> List[float]:
    try:
        values = [float(x) for x in str(text).split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"{name}: 쉼표로 구분한 실수 목록이어야 합니다: {text!r}") from exc
    if not values:
        raise ValueError(f"{name}: 비어 있습니다")
    return values


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", choices=["person"], default="person",
                   help="현재 person 전용 (siglip2+irra+solider 결합)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=CONFIG_HELP)
    p.add_argument("--qdrant-url", default=None, help=CONFIG_HELP)
    p.add_argument("--api-key", default=None)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--no-grpc", action="store_true", help="gRPC(6334) 대신 REST 만 사용")
    p.add_argument("--collection", default=None, help=CONFIG_HELP + " (person 컬렉션)")
    p.add_argument("--sources", default=None,
                   help="쉼표 구분 source 목록. 'all' 이면 원본처럼 컬렉션 전체. " + CONFIG_HELP)
    p.add_argument("--vectors", default=",".join(DEFAULT_VECTORS),
                   help="결합할 named vector 순서 (기본 siglip2,irra,solider)")
    p.add_argument("--weights", default=",".join(map(str, DEFAULT_WEIGHTS)),
                   help="결합 가중치 (기본 0.1,0.3,0.6)")
    p.add_argument("--knn-vector", default="solider", help="이웃 후보 검색에 쓰는 vector (원본 solider)")
    p.add_argument("--knn", type=int, default=25, help="원본 K=25")
    p.add_argument("--eps", type=float, default=0.12, help="원본 EPS=0.12 (combined cosine distance)")
    p.add_argument("--score-threshold", type=float, default=None,
                   help="kNN score 하한. 생략 시 1 - eps (원본 SCORE_THRESH)")
    p.add_argument("--min-faces", type=int, default=3, help="원본 MIN_FACES=3 (core 판정)")
    p.add_argument("--query-batch-size", type=int, default=256)
    p.add_argument("--scroll-batch-size", type=int, default=1024)
    p.add_argument("--max-points", type=int, default=None)
    p.add_argument("--payload-key", default="dbscan_person_v6_cl", help="기록 시 정수 cluster id 키 (원본 VERSION)")
    p.add_argument("--output-dir", default="outputs/clustering/dbscan")
    p.add_argument("--dry-run", action="store_true", help="Qdrant payload 를 쓰지 않는다")
    p.add_argument("--no-exact", action="store_true",
                   help="이웃 검색을 HNSW 근사로 (기본은 exact=True 전수 검색 = 원본 로컬 모드와 동일)")
    p.add_argument("--vector-cache", default=None,
                   help="수집한 벡터를 저장/재사용할 .npz 경로 (같은 대상 재실행 시 5분 수집 생략)")
    return p


def parse_args(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    try:
        settings = load_pipeline_settings(args.config, require=not (args.qdrant_url and args.collection))
        args.qdrant_url = resolve(args.qdrant_url, settings.qdrant_url if settings else None, "qdrant-url")
        args.collection = resolve(args.collection, settings.collection_for("person") if settings else None,
                                  "collection")
        if args.sources is not None and args.sources.strip().lower() in ("all", "*"):
            args.sources = None
        else:
            args.sources = normalize_sources(resolve(
                args.sources, (settings.clustering["sources"] if settings else None) or ["prw_image"], "sources"))
            if not args.sources:
                raise ValueError("sources must not be empty ('all' 이면 컬렉션 전체)")
        args.vectors = [v.strip() for v in args.vectors.split(",") if v.strip()]
        args.weights = parse_floats(args.weights, "weights")
        if len(args.vectors) != len(args.weights):
            raise ValueError(f"vectors({len(args.vectors)}) 와 weights({len(args.weights)}) 개수가 다릅니다")
        if settings:
            retrievers = settings.retrievers
            for name in [*args.vectors, args.knn_vector]:
                if name not in retrievers:
                    raise ValueError(f"{name!r} 는 pipeline.yaml retrievers 에 없습니다: {sorted(retrievers)}")
        if not (0.0 < args.eps < 2.0):
            raise ValueError("eps 는 (0, 2) 범위")
        args.score_threshold = validate_threshold(
            1.0 - args.eps if args.score_threshold is None else args.score_threshold, "score-threshold")
        if args.knn < 1 or args.min_faces < 1 or args.query_batch_size < 1 or args.scroll_batch_size < 1:
            raise ValueError("knn / min-faces / batch 크기는 1 이상")
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        p.error(str(exc))
    args.pipeline_settings = settings
    return args


def run(args) -> Dict[str, Any]:
    cfg = Config(target=args.target, collection=args.collection, vector=args.knn_vector,
                 combined_vectors=list(args.vectors), weights=list(args.weights), sources=args.sources,
                 knn=args.knn, score_threshold=args.score_threshold, eps=args.eps, min_faces=args.min_faces,
                 query_batch_size=args.query_batch_size, scroll_batch_size=args.scroll_batch_size,
                 max_points=args.max_points, payload_key=args.payload_key, exact=not args.no_exact)
    client = make_client(args.qdrant_url, args.api_key, args.timeout, prefer_grpc=not args.no_grpc)

    info = client.get_collection(cfg.collection)
    schema = info.config.params.vectors
    schema_names = set(schema.keys()) if isinstance(schema, dict) else set()
    needed = set(cfg.combined_vectors) | {cfg.vector}
    if schema_names and not needed <= schema_names:
        raise SystemExit(f"컬렉션 {cfg.collection} 에 없는 vector: {sorted(needed - schema_names)} "
                         f"(있는 것: {sorted(schema_names)})")

    print(f"\n[{cfg.target.upper()}] collection={cfg.collection} knn_vector={cfg.vector} "
          f"combined={cfg.combined_vectors} weights={cfg.weights} sources={cfg.sources}")
    print(f"  K={cfg.knn} EPS={cfg.eps} SCORE_THRESH={cfg.score_threshold} MIN_FACES={cfg.min_faces} "
          f"exact={cfg.exact}")

    cache_path = Path(args.vector_cache).resolve() if args.vector_cache else None
    if cache_path and cache_path.is_file():
        expected = count_points(client, cfg)
        ids, index, matrix, knn_matrix, s_fetch = load_vector_cache(cache_path, cfg, expected)
        print(f"  벡터 캐시 사용: {cache_path} ({len(ids):,} points)")
    else:
        ids, index, matrix, knn_matrix, s_fetch = fetch_vectors(client, cfg)
        if cache_path and ids:
            save_vector_cache(cache_path, cfg, ids, matrix, knn_matrix, s_fetch["missing_vectors"])
            s_fetch["vector_cache"] = str(cache_path)
            print(f"  벡터 캐시 저장: {cache_path}")
    print(f"  ids={len(ids):,} (vector 누락 {s_fetch['missing_vectors']:,}) "
          f"combined_dim={matrix.shape[1] if matrix.ndim == 2 else 0}")
    if not ids:
        raise SystemExit("대상 point 가 없습니다 (sources / collection 확인)")

    neighbors_map, s_knn = find_neighbors(client, cfg, ids, index, matrix, knn_matrix)
    del knn_matrix, matrix

    started = time.time()
    cluster_map, s_assign = assign_clusters(ids, neighbors_map, cfg.min_faces)
    degree = {pid: len(neighbors_map[pid]) for pid in ids}
    assignments, s_norm = normalize_assignments(cfg.target, cfg.method, ids, cluster_map, degree, cfg.min_faces)
    s_assign["assign_sec"] = round(time.time() - started, 3)

    counts = Counter(v for v in cluster_map.values() if v != -1)
    print(f"\n=== [{cfg.method}] ===")
    print(f"클러스터 수: {len(counts):,}   노이즈: {s_norm['noise_points']:,} "
          f"({s_norm['noise_points'] / len(ids) * 100:.2f}%)   배정: {s_norm['clustered_points']:,}")
    for cid, cnt in counts.most_common(10):
        print(f"  cluster {cid:>6d}: {cnt:>6d}개")

    if args.dry_run:
        print("[DRY-RUN] Qdrant payload 미수정")
    else:
        write_payloads(client, cfg, ids, cluster_map, assignments)
        print(f"[{cfg.payload_key}] payload 저장 완료")

    out = Path(args.output_dir) / cfg.target
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{cfg.target}_dbscan_assignments.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for pid in ids:
            f.write(json.dumps({"point_id": pid, **assignments[pid]}, ensure_ascii=False) + "\n")

    report = {"config": {**asdict(cfg), **config_provenance(args.config), "prefer_grpc": not args.no_grpc},
              "stats": {"points": len(ids), **s_fetch, **s_knn, **s_assign, **s_norm},
              "dry_run": bool(args.dry_run)}
    (out / f"{cfg.target}_dbscan_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["stats"], ensure_ascii=False, indent=2))
    print(f"assignments : {out / f'{cfg.target}_dbscan_assignments.jsonl'}")
    print(f"report      : {out / f'{cfg.target}_dbscan_report.json'}")
    return report


def main(argv=None):
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
