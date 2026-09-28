"""클러스터링 플러그인 driver — 검출기 yaml 교체와 같은 방식으로 클러스터링 알고리즘을 바꾼다.

  --method leiden | dbscan_v6        내장 플러그인 (clustering/methods/, 기본 leiden)
  --method-config <yaml>             clusterer: {module, class, params} 블록
  --module/--class + --param k=v     임의 BaseClusterer 구현

벡터는 pipeline.yaml 의 clustering 설정(target 별 vector, 임계값) 대로 Qdrant 에서 받아(--vector-cache 로 재사용)
플러그인에 넘긴다. 플러그인이 required_vectors 를 선언하면 그 named vector 들도 같이 받는다 (DBSCAN v6 의 결합 벡터).
출력은 Leiden 스크립트와 같은 형식이라 갤러리(report/build_leiden_gallery.py)·비교(compare_cluster_results.py)·
GT 평가(eval/prw_cluster_gt_eval.py) 가 그대로 읽는다:
  <output-dir>/<target>/<target>_<method>_assignments.jsonl   point_id / cluster_id / cluster_size / raw_id / noise
  <output-dir>/<target>/<target>_<method>_report.json         config(플러그인·벡터·출처) + stats
Qdrant payload 는 --write-payload 를 줄 때만 cluster_<method>_id / _size / _noise 로 기록한다 (기본은 dry-run).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from report_common import CONFIG_HELP, DEFAULT_CONFIG_PATH, load_pipeline_settings, normalize_sources, resolve  # noqa: E402
from clustering.base import (BUILTIN_METHODS, BaseClusterer, load_clusterer, normalize_labels,  # noqa: E402
                             resolve_clusterer_spec)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=CONFIG_HELP)
    p.add_argument("--target", choices=["person", "object"], default="person")
    p.add_argument("--collection", default=None, help="비우면 pipeline.yaml 의 target 컬렉션")
    p.add_argument("--vector", default=None, help="primary named vector. 비우면 pipeline.yaml clustering.<target>.vector (person solider / object dinov2)")
    p.add_argument("--sources", default=None, help="payload source 필터 (쉼표). 비우면 yaml clustering.sources, 없으면 전체")
    p.add_argument("--qdrant-url", default=None, help=CONFIG_HELP)
    p.add_argument("--api-key", default=None)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--scroll-batch-size", type=int, default=2048)
    p.add_argument("--max-points", type=int, default=None, help="앞에서 N 개만 (빠른 확인용)")
    p.add_argument("--method", default=None, help=f"내장 플러그인 {sorted(BUILTIN_METHODS)} (기본 leiden)")
    p.add_argument("--method-config", default=None, help="clusterer: {module, class, params} yaml")
    p.add_argument("--module", default=None, help="플러그인 module (덮어쓰기)")
    p.add_argument("--class", dest="cls", default=None, help="플러그인 class (덮어쓰기)")
    p.add_argument("--param", action="append", default=[], help="플러그인 params 덮어쓰기 key=value (여러 번). 예: threshold=0.96")
    p.add_argument("--min-cluster-size", type=int, default=2, help="이보다 작은 군집은 노이즈")
    p.add_argument("--vector-cache", default=None, help="npz 접두어. <접두어>_<target>_<vector>.npz 로 벡터를 재사용")
    p.add_argument("--output-dir", default="outputs/clustering/plugin")
    p.add_argument("--write-payload", action="store_true", help="Qdrant payload 에 cluster_<method>_* 기록 (기본: 기록 안 함)")
    return p


def fetch_all_vectors(q, collection: str, sources: Optional[List[str]], scroll_batch: int, ids: Sequence[Any],
                      names: Sequence[str], cache_prefix: Optional[str], target: str, log=print,
                      config_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """벡터 캐시 파일명에 임베더 지문(앞 8자)을 넣는다 — pipeline.yaml 의 임베더/가중치가 바뀌면 옛 캐시를 조용히
    재사용하지 않는다 (bench.ledger.retriever_fingerprint_sha). 지문을 못 구하면 옛 이름 그대로."""
    from clustering.cluster_leiden_qdrant import fetch_vectors_for
    out: Dict[str, Dict[str, Any]] = {}
    for name in dict.fromkeys(names):
        cfg = SimpleNamespace(collection=collection, vector=name, sources=sources, scroll_batch_size=scroll_batch)
        tag = ""
        if cache_prefix and config_path:
            from bench.ledger import retriever_fingerprint_sha
            fp = retriever_fingerprint_sha(config_path, name, log)
            tag = f"_{fp[:8]}" if fp else ""
        cache = f"{cache_prefix}_{target}_{name}{tag}.npz" if cache_prefix else None
        matrix, stats = fetch_vectors_for(q, cfg, ids, cache, log=log)
        stats.pop("missing_rows", None)
        out[name] = dict(matrix=matrix, stats=stats)
    return out


def run_from_vectors(clusterer: BaseClusterer, spec: Dict[str, Any], target: str, ids: Sequence[Any], primary_name: str,
                     vectors: Dict[str, np.ndarray], min_cluster_size: int, output_dir: Path,
                     config_extra: Optional[Dict[str, Any]] = None, fetch_stats: Optional[Dict[str, Any]] = None,
                     write_payload: bool = False, q=None, collection: Optional[str] = None, log=print) -> Dict[str, Any]:
    """벡터 → 플러그인 → assignments/report 파일 (+ 선택적 payload). Qdrant 없이도 (write_payload=False) 동작한다."""
    name = str(getattr(clusterer, "name", spec["class"]))
    primary = vectors[primary_name]
    log(f"\n=== [{name}] target={target} points={len(ids):,} primary={primary_name} dim={primary.shape[1]} "
        f"required={list(getattr(clusterer, 'required_vectors', ()))} ===")
    t0 = time.time()
    result = clusterer.cluster(ids, primary, vectors, log=log)
    cluster_sec = round(time.time() - t0, 3)
    assignments, s_norm = normalize_labels(name, target, ids, result.labels, min_cluster_size)
    log(f"클러스터 {s_norm['kept_clusters']:,} · 배정 {s_norm['clustered_points']:,} · 노이즈 {s_norm['noise_points']:,} "
        f"({(s_norm['noise_ratio'] or 0) * 100:.2f}%) · 최대 {s_norm['largest_community']:,} · {cluster_sec}s")

    if write_payload:
        if q is None or not collection:
            raise ValueError("write_payload 에는 Qdrant 클라이언트와 collection 이 필요")
        from clustering.cluster_leiden_qdrant import chunks
        by_cluster: Dict[str, List[Any]] = {}
        noise_ids: List[Any] = []
        for pid, a in assignments.items():
            (noise_ids if a["cluster_id"] is None else by_cluster.setdefault(a["cluster_id"], [])).append(pid)
        base = {"cluster_method": name, "cluster_vector": primary_name}
        for cid, members in by_cluster.items():
            q.set_payload(collection, members, {f"cluster_{name}_id": cid, f"cluster_{name}_size": len(members),
                                                f"cluster_{name}_noise": False, **base})
        for batch in chunks(noise_ids, 10000):
            q.set_payload(collection, batch, {f"cluster_{name}_id": None, f"cluster_{name}_size": 1,
                                              f"cluster_{name}_noise": True, **base})
        log(f"[payload] cluster_{name}_* 기록: 군집 {len(by_cluster):,} / 노이즈 {len(noise_ids):,}")
    else:
        log("[DRY-RUN] Qdrant payload 미수정 (--write-payload 로 기록)")

    out = Path(output_dir) / target
    out.mkdir(parents=True, exist_ok=True)
    assignments_path = out / f"{target}_{name}_assignments.jsonl"
    with assignments_path.open("w", encoding="utf-8", newline="\n") as f:
        for pid in ids:
            f.write(json.dumps({"point_id": pid, **assignments[pid]}, ensure_ascii=False) + "\n")
    params = dict(clusterer.params()) if hasattr(clusterer, "params") else {}
    config = {"target": target, "collection": collection, "vector": primary_name, "method": name,
              "plugin": {"module": spec["module"], "class": spec["class"], "params": spec.get("params", {})},
              **params, "min_cluster_size": min_cluster_size, **(config_extra or {})}
    report = {"config": config, "stats": {**(fetch_stats or {}), **result.stats, **s_norm, "cluster_sec": cluster_sec},
              "dry_run": not write_payload}
    report_path = out / f"{target}_{name}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8", newline="\n")
    log(f"assignments : {assignments_path}")
    log(f"report      : {report_path}")
    return report


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    try:
        if args.min_cluster_size < 1:
            raise ValueError("--min-cluster-size 는 1 이상")
        spec = resolve_clusterer_spec(args.method, args.module, args.cls, args.param, args.method_config)
        settings = load_pipeline_settings(args.config, require=not (args.qdrant_url and args.collection and args.vector))
        url = resolve(args.qdrant_url, settings.qdrant_url if settings else None, "qdrant-url")
        collection = resolve(args.collection, settings.collection_for(args.target) if settings else None, "collection")
        vector = resolve(args.vector, settings.vector_for(args.target) if settings else None, "vector")
    except (OSError, ValueError, AttributeError, KeyError) as exc:
        p.error(str(exc))
    sources = normalize_sources(args.sources) if args.sources else ((settings.clustering.get("sources") if settings else None) or None)

    clusterer = load_clusterer(spec)
    from clustering.cluster_leiden_qdrant import Qdrant
    q = Qdrant(url, args.api_key, args.timeout)
    ids = q.scroll_ids(collection, args.scroll_batch_size, args.max_points, sources)
    print(f"[{args.target}] collection={collection} vector={vector} sources={sources} ids={len(ids):,} "
          f"plugin={spec['module']}.{spec['class']} params={json.dumps(spec['params'], ensure_ascii=False)}")
    if not ids:
        raise SystemExit("대상 point 가 없습니다 (sources / collection 확인)")
    names = [vector, *getattr(clusterer, "required_vectors", ())]
    fetched = fetch_all_vectors(q, collection, sources, args.scroll_batch_size, ids, names, args.vector_cache, args.target,
                                config_path=settings.config_path if settings else None)
    vectors = {k: v["matrix"] for k, v in fetched.items()}
    fetch_stats = {f"fetch_{k}": v["stats"] for k, v in fetched.items()}
    extra = dict(sources=sources, max_points=args.max_points, qdrant_url=url,
                 config_path=settings.config_path if settings else None,
                 config_sha256=settings.config_sha256 if settings else None)
    return run_from_vectors(clusterer, spec, args.target, ids, vector, vectors, args.min_cluster_size, Path(args.output_dir),
                            config_extra=extra, fetch_stats=fetch_stats, write_payload=args.write_payload, q=q,
                            collection=collection)


if __name__ == "__main__":
    main()
