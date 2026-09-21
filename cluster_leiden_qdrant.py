from __future__ import annotations

"""
Qdrant ANN kNN -> sparse graph -> Leiden -> Qdrant payload

PERSON : forensic_person / solider
OBJECT : forensic_object / dinov2

Current integration backend: python-igraph + leidenalg.

Over-merge (chaining) controls
------------------------------
--mutual-knn            edge (i,j) only if j in kNN(i) AND i in kNN(j). Hub points
                        (low-res / back-view crops that are "0.95 to everyone")
                        otherwise chain unrelated identities into one giant community.
--<target>-max-cluster-size N
                        after Leiden, communities larger than N are re-partitioned on
                        their induced subgraph with resolution doubled per round
                        (--refine-rounds). Communities that cannot be split further are
                        kept and counted in stats["oversized_remaining"]. 0 = off.
For ~10M points, use this as the graph contract/prototype; graph construction or
in-memory Leiden can exceed 3 hours / host RAM, so validate at 100K -> 1M first.
"""

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests

from report_common import (
    DEFAULT_CONFIG_PATH, CONFIG_HELP, load_pipeline_settings, normalize_sources,
    resolve, validate_vector, validate_threshold,
)


def resolve_secondary(args, settings):
    value = getattr(args, 'secondary_vector', None)
    if value in (None, ''):
        value = settings.clustering['secondary_vector'] if settings else None
    if settings and value is not None:
        validate_vector(settings.retrievers, value, 'person', 'secondary_vector')
    return value


def resolve_targets(args, settings):
    """Resolve and validate CLI overrides without IO or mutating arguments."""
    selected = ('person', 'object') if args.target == 'both' else (args.target,)
    result = {}
    for target in selected:
        values = {}
        for key, method in (('collection', 'collection_for'), ('vector', 'vector_for'),
                            ('threshold', 'threshold_for')):
            values[key] = resolve(getattr(args, f'{target}_{key}', None),
                                  getattr(settings, method)(target) if settings else None,
                                  f'{target}-{key}')
        values['threshold'] = validate_threshold(values['threshold'], f'{target}-threshold')
        if settings:
            validate_vector(settings.retrievers, values['vector'], target, f'{target}-vector')
        result[target] = values
    secondary = resolve_secondary(args, settings)
    if getattr(args, 'identity_safe', False):
        if args.target != 'person':
            raise ValueError('identity-safe requires --target person')
        if secondary is None:
            raise ValueError('identity-safe requires secondary-vector')
        if result['person']['vector'] == secondary:
            raise ValueError('secondary-vector must differ from person-vector')
    return result


def config_provenance(args, target):
    settings = args.pipeline_settings
    return dict(config_path=settings.config_path if settings else None,
                config_sha256=settings.config_sha256 if settings else None,
                settings_inferred=settings.inferred if settings else {},
                retriever_validation='validated' if settings else '확인 불가 (config 없음; 명시 CLI)',
                applied=dict(**args.resolved_targets[target], sources=args.sources,
                             qdrant_url=args.qdrant_url, secondary_vector=args.secondary_vector))


@dataclass(frozen=True)
class Config:
    target: str
    collection: str
    vector: str
    sources: List[str]
    knn: int
    score_threshold: float
    resolution: float
    min_cluster_size: int
    query_batch_size: int
    scroll_batch_size: int
    seed: int
    max_points: Optional[int]
    identity_safe: bool = False
    secondary_vector: str = "irra"
    secondary_threshold: float = 0.75
    mutual_knn: bool = False
    max_cluster_size: Optional[int] = None
    refine_rounds: int = 5


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
            raise RuntimeError(f"GET {path} -> {r.status_code}\n{r.text[:2000]}")
        return r.json()

    def post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        r = self.s.post(self.url + path, json=body, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f"POST {path} -> {r.status_code}\n{r.text[:2000]}")
        return r.json()

    def scroll_ids(self, collection: str, batch: int, max_points: Optional[int], sources: Optional[List[str]] = None) -> List[Any]:
        ids: List[Any] = []
        offset = None
        while True:
            body: Dict[str, Any] = {
                "limit": batch,
                "with_payload": False,
                "with_vector": False,
            }
            if sources and len(sources) == 1:
                body["filter"] = {"must": [{"key": "source", "match": {"value": sources[0]}}]}
            elif sources and len(sources) > 1:
                body["filter"] = {"should": [{"key": "source", "match": {"value": s}} for s in sources]}
            if offset is not None:
                body["offset"] = offset
            data = self.post(f"/collections/{collection}/points/scroll", body)
            result = data.get("result") or {}
            points = result.get("points") or []
            for p in points:
                ids.append(p["id"])
                if max_points is not None and len(ids) >= max_points:
                    return ids
            offset = result.get("next_page_offset")
            if not points or offset is None:
                break
        return ids

    def query_batch(
        self,
        collection: str,
        point_ids: Sequence[Any],
        vector: str,
        limit: int,
        threshold: float,
        allowed_ids: Optional[Sequence[Any]] = None,
        sources: Optional[List[str]] = None,
    ) -> List[List[Dict[str, Any]]]:
        searches = [
            {
                "query": pid,
                "using": vector,
                "limit": limit,
                "score_threshold": threshold,
                "with_payload": False,
                "with_vector": False,
                "params": {"quantization": {"ignore": True}},
            }
            for pid in point_ids
        ]
        if allowed_ids is not None:
            for search in searches:
                search["filter"] = {"must": [{"has_id": list(allowed_ids)}]}
        elif sources:
            # 같은 source 안에서만 이웃을 찾는다. 필터 없이 컬렉션 전체에서 top-k 를
            # 받아 다른 source 를 버리면, 다른 source 가 자리를 차지한 만큼 in-source
            # 이웃이 k 개에 못 미친다 (forensic_person 은 COCO 34% / PRW / 영상 혼재).
            # source 에는 payload index 가 있어 필터 검색이 빠르다.
            for search in searches:
                search["filter"] = {"must": [{"key": "source", "match": {"any": list(sources)}}]}
        data = self.post(
            f"/collections/{collection}/points/query/batch",
            {"searches": searches},
        )
        result = data.get("result") or []
        if len(result) != len(point_ids):
            raise RuntimeError(f"batch result mismatch: {len(result)} != {len(point_ids)}")
        return [list((x or {}).get("points") or []) for x in result]

    def set_payload(self, collection: str, point_ids: Sequence[Any], payload: Dict[str, Any]):
        if not point_ids:
            return
        self.post(
            f"/collections/{collection}/points/payload?wait=true",
            {"payload": payload, "points": list(point_ids)},
        )


def chunks(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def stable_cluster_id(target: str, vector: str, ids: Sequence[Any]) -> str:
    raw = target + "|" + vector + "|" + "|".join(sorted(map(str, ids)))
    return f"leiden:{target}:{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def build_graph(q: Qdrant, cfg: Config, point_ids: Sequence[Any]):
    """
    ANN kNN -> undirected edge list.

    Neighbours are collected per node into (N, k) arrays first, then reduced to
    undirected edges in one vectorised pass:
      directed (default) : edge if j in kNN(i) OR i in kNN(j)   (== previous behaviour)
      mutual             : edge if j in kNN(i) AND i in kNN(j)
    Edge weight = max score over the two directions.
    """
    n = len(point_ids)
    k = cfg.knn
    id_to_idx = {str(pid): i for i, pid in enumerate(point_ids)}

    nbr = np.full((n, k), -1, dtype=np.int64)
    sc = np.full((n, k), -1.0, dtype=np.float32)

    started = time.time()
    warned = False
    processed = 0

    print("\n=== BUILD ANN kNN GRAPH ===")
    print(f"target={cfg.target} points={n:,} k={k} threshold={cfg.score_threshold} "
          f"mutual={cfg.mutual_knn}")

    for batch in chunks(list(point_ids), cfg.query_batch_size):
        results = q.query_batch(
            cfg.collection,
            batch,
            cfg.vector,
            k + 1,
            cfg.score_threshold,
            None,          # allowed_ids: has_id 필터는 느려서 쓰지 않는다
            cfg.sources,   # source 필터로 in-source 이웃 k 개를 보장한다
        )
        for src_id, hits in zip(batch, results):
            src_key = str(src_id)
            src_idx = id_to_idx[src_key]
            slot = 0
            for hit in hits:
                dst_id = hit.get("id")
                if dst_id is None or str(dst_id) == src_key:
                    continue
                dst_idx = id_to_idx.get(str(dst_id))
                if dst_idx is None:
                    continue
                score = float(hit.get("score", float("-inf")))
                if not math.isfinite(score) or score < cfg.score_threshold:
                    continue
                nbr[src_idx, slot] = dst_idx
                sc[src_idx, slot] = score
                slot += 1
                if slot >= k:
                    break

        processed += len(batch)
        elapsed = time.time() - started
        rate = processed / elapsed if elapsed > 0 else 0.0
        eta = (n - processed) / rate if rate > 0 else float("inf")
        if not warned and processed >= min(1000, max(100, cfg.query_batch_size * 5)) and eta > 10800:
            warned = True
            print(f"\n[WARNING] 예상 남은 시간이 3시간 이상: {eta/3600:.2f} h "
                  f"(초기 cold 속도 기준 — 실제는 더 빠를 수 있음)")
        print(
            f"\r{processed:,}/{n:,} | {rate:,.1f} point/s | ETA={eta/60:,.1f} min",
            end="", flush=True,
        )
    print()

    # ---- directed hits -> undirected keys ----
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = nbr.reshape(-1)
    w = sc.reshape(-1)
    valid = dst >= 0
    src, dst, w = src[valid], dst[valid], w[valid]
    directed_hits = int(valid.sum())

    if directed_hits == 0:
        return [], [], {
            "points": n, "edges": 0, "avg_degree": 0.0, "directed_hits": 0,
            "mutual_knn": cfg.mutual_knn, "graph_build_sec": round(time.time() - started, 3),
        }

    a = np.minimum(src, dst)
    b = np.maximum(src, dst)
    key = a * n + b
    order = np.argsort(key, kind="stable")
    key, a, b, w = key[order], a[order], b[order], w[order]
    uniq, start, counts = np.unique(key, return_index=True, return_counts=True)
    wmax = np.maximum.reduceat(w, start)

    if cfg.mutual_knn:
        keep = counts >= 2          # 양방향 모두 존재
    else:
        keep = np.ones(len(uniq), dtype=bool)

    edges = list(zip(a[start][keep].tolist(), b[start][keep].tolist()))
    weights = [float(x) for x in wmax[keep]]

    stats = {
        "points": n,
        "edges": len(edges),
        "avg_degree": (2 * len(edges) / n) if n else 0.0,
        "directed_hits": directed_hits,
        "undirected_candidates": int(len(uniq)),
        "mutual_knn": cfg.mutual_knn,
        "graph_build_sec": round(time.time() - started, 3),
    }
    print(f"edges={len(edges):,} (directed hits {directed_hits:,}, "
          f"undirected candidates {len(uniq):,}, mutual={cfg.mutual_knn})")
    return edges, weights, stats


def leiden(point_ids, edges, weights, resolution: float, seed: int):
    try:
        import igraph as ig
        import leidenalg as la
    except ImportError as exc:
        raise RuntimeError("설치 필요: pip install python-igraph leidenalg requests") from exc

    print("\n=== LEIDEN ===")
    g = ig.Graph(n=len(point_ids), edges=list(edges), directed=False)
    if weights:
        g.es["weight"] = list(weights)
    started = time.time()
    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight" if weights else None,
        resolution_parameter=resolution,
        seed=seed,
        n_iterations=-1,
    )
    membership = list(part.membership)
    sizes = defaultdict(int)
    for x in membership:
        sizes[int(x)] += 1
    stats = {
        "raw_communities": len(sizes),
        "largest_community": max(sizes.values()) if sizes else 0,
        "leiden_sec": round(time.time() - started, 3),
    }
    print(f"communities={stats['raw_communities']:,} largest={stats['largest_community']:,}")
    return membership, stats


def refine_oversized(n_nodes, edges, weights, membership, max_size: int,
                     resolution: float, seed: int, rounds: int):
    """
    max_size 를 넘는 커뮤니티를 그 부분그래프 위에서 다시 Leiden 으로 쪼갠다.
    라운드마다 resolution 을 2배로 올린다 (더 작은 커뮤니티를 선호).
    더 이상 갈라지지 않는 커뮤니티(밀집 클리크)는 그대로 두고 oversized_remaining 에 센다.
    결정적: 같은 seed / 같은 입력이면 같은 결과.
    """
    try:
        import igraph as ig
        import leidenalg as la
    except ImportError as exc:
        raise RuntimeError("설치 필요: pip install python-igraph leidenalg") from exc

    g = ig.Graph(n=n_nodes, edges=list(edges), directed=False)
    if weights:
        g.es["weight"] = list(weights)

    membership = list(membership)
    next_id = (max(membership) + 1) if membership else 0
    stats = {"refine_rounds_used": 0, "split_communities": 0}
    started = time.time()

    for r in range(1, rounds + 1):
        groups = defaultdict(list)
        for i, m in enumerate(membership):
            groups[int(m)].append(i)
        big = [c for c, nodes in groups.items() if len(nodes) > max_size]
        if not big:
            break
        stats["refine_rounds_used"] = r
        res = resolution * (2 ** r)
        print(f"  refine round {r}: {len(big)} communities > {max_size} "
              f"(largest {max(len(groups[c]) for c in big):,}), resolution={res}")
        for c in big:
            nodes = groups[c]
            sub = g.induced_subgraph(nodes)
            part = la.find_partition(
                sub,
                la.RBConfigurationVertexPartition,
                weights="weight" if weights else None,
                resolution_parameter=res,
                seed=seed,
                n_iterations=-1,
            )
            local = list(part.membership)
            if len(set(local)) <= 1:
                continue            # 더 못 쪼갬
            stats["split_communities"] += 1
            for idx, sub_m in zip(nodes, local):
                membership[idx] = next_id + int(sub_m)
            next_id += max(local) + 1

    sizes = Counter(membership)
    stats["oversized_remaining"] = sum(1 for s in sizes.values() if s > max_size)
    stats["largest_after_refine"] = max(sizes.values()) if sizes else 0
    stats["communities_after_refine"] = len(sizes)
    stats["refine_sec"] = round(time.time() - started, 3)
    print(f"  after refine: communities={len(sizes):,} largest={stats['largest_after_refine']:,} "
          f"oversized_remaining={stats['oversized_remaining']}")
    return membership, stats


def normalize(cfg: Config, point_ids, membership):
    groups = defaultdict(list)
    for pid, cid in zip(point_ids, membership):
        groups[int(cid)].append(pid)

    assignments = {}
    kept = 0
    clustered = 0
    noise = 0
    for raw_id, members in groups.items():
        if len(members) < cfg.min_cluster_size:
            for pid in members:
                assignments[pid] = {
                    "cluster_id": None,
                    "cluster_size": 1,
                    "raw_leiden_id": raw_id,
                    "noise": True,
                }
                noise += 1
            continue
        cid = stable_cluster_id(cfg.target, cfg.vector, members)
        kept += 1
        clustered += len(members)
        for pid in members:
            assignments[pid] = {
                "cluster_id": cid,
                "cluster_size": len(members),
                "raw_leiden_id": raw_id,
                "noise": False,
            }
    return assignments, {
        "kept_clusters": kept,
        "clustered_points": clustered,
        "noise_points": noise,
    }


def _method_name(cfg: Config) -> str:
    if cfg.identity_safe:
        return "exact_mutual_leiden_complete_link"
    return "ann_mutual_knn_leiden" if cfg.mutual_knn else "ann_knn_leiden"


def write_payloads(q: Qdrant, cfg: Config, assignments):
    by_cluster = defaultdict(list)
    noise_ids = []
    for pid, a in assignments.items():
        if a["cluster_id"] is None:
            noise_ids.append(pid)
        else:
            by_cluster[a["cluster_id"]].append(pid)

    for cid, ids in by_cluster.items():
        q.set_payload(
            cfg.collection,
            ids,
            {
                "cluster_leiden_id": cid,
                "cluster_leiden_size": len(ids),
                "cluster_leiden_noise": False,
                "cluster_method": _method_name(cfg),
                "cluster_max_size": cfg.max_cluster_size,
                "cluster_secondary_vector": cfg.secondary_vector if cfg.identity_safe else None,
                "cluster_secondary_threshold": cfg.secondary_threshold if cfg.identity_safe else None,
                "cluster_vector": cfg.vector,
                "cluster_knn": cfg.knn,
                "cluster_score_threshold": cfg.score_threshold,
                "cluster_resolution": cfg.resolution,
            },
        )

    for batch in chunks(noise_ids, 10000):
        q.set_payload(
            cfg.collection,
            batch,
            {
                "cluster_leiden_id": None,
                "cluster_leiden_size": 1,
                "cluster_leiden_noise": True,
                "cluster_method": _method_name(cfg),
                "cluster_max_size": cfg.max_cluster_size,
                "cluster_secondary_vector": cfg.secondary_vector if cfg.identity_safe else None,
                "cluster_secondary_threshold": cfg.secondary_threshold if cfg.identity_safe else None,
                "cluster_vector": cfg.vector,
                "cluster_knn": cfg.knn,
                "cluster_score_threshold": cfg.score_threshold,
                "cluster_resolution": cfg.resolution,
            },
        )


def run_target(args, target: str):
    d = args.resolved_targets[target]
    collection, vector, threshold = d['collection'], d['vector'], d['threshold']

    cfg = Config(
        target=target,
        collection=collection,
        vector=vector,
        sources=args.sources,
        knn=args.knn,
        score_threshold=threshold,
        resolution=args.resolution,
        min_cluster_size=args.min_cluster_size,
        query_batch_size=args.query_batch_size,
        scroll_batch_size=args.scroll_batch_size,
        seed=args.seed,
        max_points=args.max_points,
        identity_safe=args.identity_safe,
        secondary_vector=args.secondary_vector,
        secondary_threshold=args.secondary_threshold,
        mutual_knn=args.mutual_knn,
        # 0 또는 음수는 "상한 없음" — GUI 가 int 인자를 항상 넘기므로 0 을 off 로 쓴다.
        max_cluster_size=(getattr(args, f"{target}_max_cluster_size") or None)
        if (getattr(args, f"{target}_max_cluster_size") or 0) > 0 else None,
        refine_rounds=args.refine_rounds,
    )

    q = Qdrant(args.qdrant_url, args.api_key, args.timeout)
    collection_info = q.get(f"/collections/{cfg.collection}")["result"]
    ids = q.scroll_ids(cfg.collection, cfg.scroll_batch_size, cfg.max_points, cfg.sources)
    print(f"\n[{target.upper()}] collection={cfg.collection} vector={cfg.vector} sources={cfg.sources} ids={len(ids):,}")
    if not ids:
        provenance = config_provenance(args, target)
        return dict(target=target, skipped=True, config_path=provenance['config_path'],
                    config_sha256=provenance['config_sha256'], sources=cfg.sources,
                    collection=cfg.collection, config={**asdict(cfg), **provenance})

    if len(ids) >= 10_000_000:
        print("[WARNING] 10M+ 감지: 이 igraph 백엔드는 3시간 이상/메모리 초과 가능성이 큼.")
        print("[WARNING] 100K -> 1M로 파라미터 검증 후 GPU/분산 Leiden 백엔드로 교체 권장.")

    if cfg.identity_safe:
        if len(ids) > 6000:
            raise ValueError("identity-safe uses quadratic memory; limit this review to <= 6000 points.")
        schema = collection_info["config"]["params"]["vectors"]
        for name in (cfg.vector, cfg.secondary_vector):
            if str(schema.get(name, {}).get("distance", "")).lower() != "cosine":
                raise ValueError(f"identity-safe requires a named cosine vector: {name}")
        from identity_cluster import (fetch_points, score_matrices, identity_constraints,
                                      mutual_edges, refine_membership)
        started = time.time()
        points = fetch_points(q, cfg.collection, ids, cfg.vector, cfg.secondary_vector)
        a, b = score_matrices(points, cfg.vector, cfg.secondary_vector)
        allowed = identity_constraints(points, a, b, cfg.score_threshold, cfg.secondary_threshold)
        edges, weights = mutual_edges(a, allowed, cfg.knn)
        s1 = {"points": len(ids), "edges": len(edges),
              "avg_degree": 2 * len(edges) / len(ids),
              "graph_build_sec": round(time.time() - started, 3)}
        membership, s2 = leiden(ids, edges, weights, cfg.resolution, cfg.seed)
        membership = refine_membership(membership, a, b, allowed)
        sizes = Counter(membership)
        s2["refined_communities"] = len(sizes)
        s2["largest_refined_community"] = max(sizes.values())
    else:
        edges, weights, s1 = build_graph(q, cfg, ids)
        membership, s2 = leiden(ids, edges, weights, cfg.resolution, cfg.seed)

    if cfg.max_cluster_size:
        print(f"\n=== REFINE OVERSIZED (max {cfg.max_cluster_size}) ===")
        membership, s_ref = refine_oversized(
            len(ids), edges, weights, membership,
            cfg.max_cluster_size, cfg.resolution, cfg.seed, cfg.refine_rounds,
        )
        s2 = {**s2, **s_ref}

    assignments, s3 = normalize(cfg, ids, membership)

    if not args.dry_run:
        write_payloads(q, cfg, assignments)
    else:
        print("[DRY-RUN] Qdrant payload 미수정")

    out = Path(args.output_dir) / target
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{target}_leiden_assignments.jsonl").open("w", encoding="utf-8") as f:
        for pid in ids:
            f.write(json.dumps({"point_id": pid, **assignments[pid]}, ensure_ascii=False) + "\n")

    report = {"config": asdict(cfg), "stats": {**s1, **s2, **s3}, "dry_run": args.dry_run}
    report['config'].update(config_provenance(args, target))
    (out / f"{target}_leiden_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["stats"], ensure_ascii=False, indent=2))
    return report


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["person", "object", "both"], default="both")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=CONFIG_HELP)
    p.add_argument("--qdrant-url", default=None, help=CONFIG_HELP)
    p.add_argument("--api-key", default=None)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--sources", default=None,
                   help=CONFIG_HELP + "; 없으면 final_db_candidates,forensic_image")

    p.add_argument("--person-collection", default=None, help=CONFIG_HELP)
    p.add_argument("--person-vector", default=None, help=CONFIG_HELP)
    p.add_argument("--person-threshold", type=float, default=None, help=CONFIG_HELP)

    p.add_argument("--object-collection", default=None, help=CONFIG_HELP)
    p.add_argument("--object-vector", default=None, help=CONFIG_HELP)
    p.add_argument("--object-threshold", type=float, default=None, help=CONFIG_HELP)

    p.add_argument("--knn", type=int, default=30)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--min-cluster-size", type=int, default=2)
    p.add_argument("--query-batch-size", type=int, default=32)
    p.add_argument("--scroll-batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-points", type=int, default=None)
    p.add_argument("--output-dir", default="outputs/clustering/leiden")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mutual-knn", action="store_true",
                   help="edge only if i and j are in each other's kNN. Blocks hub-driven chaining.")
    p.add_argument("--person-max-cluster-size", type=int, default=0,
                   help="re-partition person communities larger than this (0 = off)")
    p.add_argument("--object-max-cluster-size", type=int, default=0,
                   help="re-partition object communities larger than this (0 = off)")
    p.add_argument("--refine-rounds", type=int, default=5,
                   help="max re-partition rounds; resolution doubles each round")
    p.add_argument("--identity-safe", action="store_true",
                   help="Image-person review: exact cosine, IRRA gate, mutual kNN and complete-link refinement; <=6000 points")
    p.add_argument("--secondary-vector", default=None, help=CONFIG_HELP)
    p.add_argument("--secondary-threshold", type=float, default=0.75)
    return p


def parse_args(argv=None):
    p = build_parser()
    args = p.parse_args(argv)

    if args.knn < 1 or args.min_cluster_size < 1:
        p.error("knn/min-cluster-size must be >= 1")
    if args.resolution <= 0:
        p.error("resolution must be > 0")
    if args.identity_safe and args.target != "person":
        p.error("identity-safe requires --target person")
    if not -1 <= args.secondary_threshold <= 1:
        p.error("secondary-threshold must be in [-1, 1]")
    for target in ('person', 'object'):
        value = getattr(args, f'{target}_threshold')
        if value is not None:
            try:
                validate_threshold(value, f'{target}-threshold')
            except ValueError as exc:
                p.error(str(exc))
    if args.sources not in (None, '') and not normalize_sources(args.sources):
        p.error('--sources must not be empty')
    return args


def main(argv=None):
    args = parse_args(argv)
    selected = ('person', 'object') if args.target == 'both' else (args.target,)
    required = ['qdrant_url', 'secondary_vector'] + [
        f'{target}_{key}' for target in selected for key in ('collection', 'vector', 'threshold')]
    try:
        settings = load_pipeline_settings(args.config, require=not all(
            getattr(args, name) not in (None, '') for name in required))
        args.resolved_targets = resolve_targets(args, settings)
        args.secondary_vector = resolve_secondary(args, settings)
        args.qdrant_url = resolve(args.qdrant_url, settings.qdrant_url if settings else None, 'qdrant-url')
        args.sources = normalize_sources(resolve(args.sources,
            (settings.clustering['sources'] if settings else None) or ['final_db_candidates', 'forensic_image'], 'sources'))
        if not args.sources:
            raise ValueError('sources must not be empty')
    except (OSError, ValueError) as exc:
        build_parser().error(str(exc))
    args.pipeline_settings = settings

    targets = ["person", "object"] if args.target == "both" else [args.target]
    summary = [run_target(args, t) for t in targets]
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "leiden_pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
