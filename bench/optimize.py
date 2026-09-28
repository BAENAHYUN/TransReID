"""bench/optimize.py — P3: 하이퍼파라미터 탐색 (Optuna) + 검출 임계값 스윕. tune 분할로 찾고 holdout 으로 검증한다.

  python bench/optimize.py cluster --method leiden|dbscan_v6 [--space knn=int:10:60 --space threshold=float:0.90:0.99 …]
        --trials 60 [--pid-split bench/splits/prw_pids_seed42.json:tune] [--objective b3_f1]
        [--constraint pair_precision>=0.90 …] [--max-points 0] [--name leiden_tune] [--validate]
  python bench/optimize.py search --trials 40 [--pid-split …:tune] [--objective map] [--constraint pool_recall>=90] [--validate]
  python bench/optimize.py detect --detections eval/results/detect_prw/yolo26m_test/detections.jsonl
        [--objective f1] [--constraint recall>=0.85] [--step 0.01]           (Optuna 없이 결정적 스윕, 같은 산출물)

탐색은 서브프로세스 없이 안에서 돈다: 클러스터는 Qdrant 벡터를 한 번만 받아(캐시) trial 마다 플러그인만 다시 실행,
검색은 eval/results/cache/prw_gt_<model>.npz(GT crop 임베딩)로 순위 조합만 다시 계산, 검출은 detections.jsonl 을 임계값별로 채점.

산출물 bench/studies/<stage>_<name>_<시각>/ : trials.jsonl · best.json · recommended.yaml · pareto.json · report.md (· optuna.db)
원장: 스터디 요약 1행 (name = study:<name>, metrics = best trial, params = best params, extra.study) — --validate 면 holdout 행 추가.
목적함수·제약 권고(기준표 §R): 클러스터 = pair_precision ≥ 0.90 제약 아래 b3_f1 최대 · 검색 = pool_recall ≥ 90 제약 아래 mAP 최대 ·
검출 = recall ≥ 기준 제약 아래 f1 최대. 잡음 기준: 클러스터 seed 변동 ≤ 0.002 → 그보다 작은 차이는 결론에 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench import ledger  # noqa: E402

DEFAULT_STUDIES_DIR = PROJECT_ROOT / "bench" / "studies"
DEFAULT_PIPELINE = PROJECT_ROOT / "pipeline.yaml"
NOISE = {"cluster": 0.002, "search": 0.1, "detect": 0.005}   # 이보다 작은 차이는 잡음 (기준표 §R)

DEFAULT_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "leiden": {"knn": {"type": "int", "low": 10, "high": 60},
               "threshold": {"type": "float", "low": 0.90, "high": 0.99},
               "resolution": {"type": "float", "low": 0.5, "high": 2.0},
               "mutual_knn": {"type": "cat", "choices": [True, False]},
               "max_cluster_size": {"type": "cat", "choices": [0, 200, 500]}},
    "dbscan_v6": {"knn": {"type": "int", "low": 10, "high": 50},
                  "score_threshold": {"type": "float", "low": 0.80, "high": 0.95},
                  "eps": {"type": "float", "low": 0.05, "high": 0.30},
                  "min_faces": {"type": "int", "low": 2, "high": 6}},
    "search": {"stage1": {"type": "cat", "choices": ["siglip2+irra", "irra", "siglip2+irra+solider"]},
               "rerank": {"type": "cat", "choices": ["solider", "none"]},
               "w_siglip2": {"type": "float", "low": 0.25, "high": 2.0},
               "w_irra": {"type": "float", "low": 0.25, "high": 2.0},
               "w_solider": {"type": "float", "low": 0.25, "high": 2.0},
               "rrf_k": {"type": "float", "low": 1.0, "high": 100.0, "log": True},
               "prefetch": {"type": "int", "low": 100, "high": 1000, "step": 50},
               "pool": {"type": "int", "low": 100, "high": 1000, "step": 50}},
}
DEFAULT_OBJECTIVE = {"cluster": "b3_f1", "search": "map", "detect": "f1"}
DEFAULT_CONSTRAINTS = {"cluster": ["pair_precision>=0.90"], "search": ["pool_recall>=90"], "detect": []}
PARETO_SECOND = {"cluster": "pair_precision", "search": "pool_recall", "detect": "recall"}


# ---------------------------------------------------------------- 공간 · 제약
def parse_split_arg_safe(text: Optional[str]) -> Tuple[Optional[Path], Optional[str]]:
    """'파일:부분' → (Path, 부분); 비었으면 (None, None) = 분할 없이 전체."""
    if not text:
        return None, None
    from bench.splits import parse_split_arg
    return parse_split_arg(text)


def parse_space(items: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """'knn=int:10:60' · 'threshold=float:0.9:0.99[:log]' · 'mutual_knn=cat:true,false' · 'pool=int:100:1000:step50'."""
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--space 는 이름=type:… 형식: {item!r}")
        name, rest = item.split("=", 1)
        parts = rest.split(":")
        kind = parts[0].strip().lower()
        if kind == "int":
            d: Dict[str, Any] = {"type": "int", "low": int(parts[1]), "high": int(parts[2])}
            for extra in parts[3:]:
                if extra.startswith("step"):
                    d["step"] = int(extra[4:])
        elif kind == "float":
            d = {"type": "float", "low": float(parts[1]), "high": float(parts[2])}
            if "log" in parts[3:]:
                d["log"] = True
        elif kind == "cat":
            d = {"type": "cat", "choices": [_auto(v) for v in parts[1].split(",")]}
        else:
            raise ValueError(f"알 수 없는 type {kind!r} ({item})")
        out[name.strip()] = d
    return out


def _auto(text: str) -> Any:
    t = text.strip()
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    try:
        return json.loads(t)
    except ValueError:
        return t


def parse_constraints(items: Sequence[str]) -> List[Tuple[str, str, float]]:
    out = []
    for item in items:
        for op in (">=", "<=", ">", "<"):
            if op in item:
                k, v = item.split(op, 1)
                out.append((k.strip(), op, float(v)))
                break
        else:
            raise ValueError(f"--constraint 는 지표>=값 형식: {item!r}")
    return out


def violations(metrics: Dict[str, Any], constraints: Sequence[Tuple[str, str, float]]) -> Dict[str, float]:
    """제약 위반량 (0 이면 만족). 지표가 없으면 위반으로 본다."""
    out: Dict[str, float] = {}
    for key, op, target in constraints:
        v = metrics.get(key)
        if not isinstance(v, (int, float)):
            out[key] = float("inf")
            continue
        if op in (">=", ">"):
            gap = target - float(v)
        else:
            gap = float(v) - target
        out[key] = max(0.0, gap)
    return out


def sample(trial, space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, d in space.items():
        if d["type"] == "int":
            params[name] = trial.suggest_int(name, d["low"], d["high"], step=d.get("step", 1))
        elif d["type"] == "float":
            params[name] = trial.suggest_float(name, d["low"], d["high"], log=bool(d.get("log")))
        else:
            params[name] = trial.suggest_categorical(name, list(d["choices"]))
    return params


def pareto_front(rows: Sequence[Dict[str, Any]], a: str, b: str) -> List[Dict[str, Any]]:
    """두 지표 모두 클수록 좋은 비지배 집합."""
    pts = [r for r in rows if isinstance((r.get("metrics") or {}).get(a), (int, float)) and isinstance((r.get("metrics") or {}).get(b), (int, float))]
    front = []
    for r in pts:
        ra, rb = r["metrics"][a], r["metrics"][b]
        dominated = any((o["metrics"][a] >= ra and o["metrics"][b] >= rb) and (o["metrics"][a] > ra or o["metrics"][b] > rb) for o in pts)
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: -r["metrics"][a])


# ---------------------------------------------------------------- 문제 정의
class Problem:
    stage = ""
    name = ""
    baseline: Dict[str, Any] = {}

    def evaluate(self, params: Dict[str, Any], part: Optional[str] = None) -> Dict[str, Any]:  # part: None=tune, "holdout"
        raise NotImplementedError

    def recommended_yaml(self, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def component(self) -> Dict[str, Any]:
        return {}

    def gt(self) -> Dict[str, Any]:
        return {}


def flatten_cluster(res: Dict[str, Any]) -> Dict[str, Any]:
    pn = res.get("pairs_noise_as_singletons") or {}
    b3 = res.get("bcubed") or {}
    return {"pair_precision": pn.get("precision"), "pair_recall": pn.get("recall"), "pair_f1": pn.get("f1"),
            "b3_precision": b3.get("precision"), "b3_recall": b3.get("recall"), "b3_f1": b3.get("f1"),
            "purity": res.get("purity"), "noise_ratio": res.get("noise_ratio"), "mixed_clusters": res.get("mixed_clusters"),
            "pids_split": res.get("pids_split"), "labeled_points": res.get("labeled_points"), "noise_points": res.get("noise_points")}


class ClusterProblem(Problem):
    """벡터를 한 번 받아 두고 trial 마다 플러그인만 다시 실행 → GT(point→pid) 평가."""
    stage = "cluster"

    def __init__(self, method: Optional[str], method_config: Optional[str], base_params: Dict[str, Any], target: str,
                 sources: Optional[str], vector: Optional[str], max_points: int, min_cluster_size: int, config_path: str,
                 matches_cache: Path, split_file: Optional[Path], tune_part: Optional[str], vector_cache: str, log=print):
        from clustering.base import load_clusterer, resolve_clusterer_spec
        from clustering.cluster_leiden_qdrant import Qdrant
        from clustering.cluster_qdrant import fetch_all_vectors
        from report_common import load_pipeline_settings, normalize_sources
        self.log = log
        self.spec = resolve_clusterer_spec(method, None, None, [], method_config)
        self.spec["params"] = {**(self.spec.get("params") or {}), **base_params}
        self.name = method or Path(method_config or "custom").stem
        settings = load_pipeline_settings(config_path)
        self.collection = settings.collection_for(target)
        self.vector = vector or settings.vector_for(target)
        self.target, self.min_cluster_size = target, min_cluster_size
        self.sources = normalize_sources(sources) if sources else ((settings.clustering.get("sources") or None))
        probe = load_clusterer(self.spec)
        names = [self.vector, *getattr(probe, "required_vectors", ())]
        q = Qdrant(settings.qdrant_url, None, 120)
        self.ids = q.scroll_ids(self.collection, 2048, max_points or None, self.sources)
        log(f"[cluster] {self.collection} vector={self.vector} sources={self.sources} points={len(self.ids):,} plugin={self.spec['module']}.{self.spec['class']}")
        fetched = fetch_all_vectors(q, self.collection, self.sources, 2048, self.ids, names, vector_cache, target,
                                    log=log, config_path=settings.config_path)
        self.vectors = {k: v["matrix"] for k, v in fetched.items()}
        # GT
        from eval.prw_e2e_search_eval import load_matches
        _, matches = load_matches(matches_cache)
        pid_all = {pk: int(d["pid"]) for pk, d in matches.items() if d.get("status") == "labeled"}
        idset = set(map(str, self.ids))
        pid_all = {pk: pid for pk, pid in pid_all.items() if pk in idset}
        from collections import Counter
        sizes = Counter(pid_all.values())
        self.pid_of_all = {pk: pid for pk, pid in pid_all.items() if sizes[pid] >= 2}
        self.parts: Dict[str, set] = {}
        if split_file:
            from bench.splits import load_split
            d = load_split(split_file)
            self.parts = {k: set(int(x) for x in v) for k, v in d["parts"].items()}
        self.tune_part = tune_part
        # 운영값 = 생성자 기본값 + yaml/CLI 로 고정한 params (첫 trial 로 넣어 "지금 설정이 몇 등인지" 를 같은 표에 남긴다)
        import inspect
        defaults = {k: p.default for k, p in inspect.signature(type(probe).__init__).parameters.items()
                    if p.default is not inspect.Parameter.empty and k != "self"}
        self.baseline = {**defaults, **self.spec["params"]}
        self.matches_cache = str(matches_cache)
        log(f"[cluster] GT point {len(self.pid_of_all):,} · 분할 {list(self.parts) or '없음'}")

    def pid_of(self, part: Optional[str]) -> Dict[Any, int]:
        part = part or self.tune_part
        if not part or part not in self.parts:
            return self.pid_of_all
        keep = self.parts[part]
        return {pk: pid for pk, pid in self.pid_of_all.items() if pid in keep}

    def evaluate(self, params: Dict[str, Any], part: Optional[str] = None) -> Dict[str, Any]:
        from clustering.base import load_clusterer, normalize_labels
        from eval.prw_cluster_gt_eval import evaluate_method
        spec = {"module": self.spec["module"], "class": self.spec["class"], "params": {**self.spec["params"], **params}}
        clusterer = load_clusterer(spec)
        t0 = time.time()
        result = clusterer.cluster(self.ids, self.vectors[self.vector], self.vectors, log=lambda *_a, **_k: None)
        assignments, _ = normalize_labels(str(getattr(clusterer, "name", spec["class"])), self.target, self.ids, result.labels, self.min_cluster_size)
        labels = {str(pid): a["cluster_id"] for pid, a in assignments.items()}
        pid_of = self.pid_of(part)
        res = evaluate_method(labels, pid_of, sorted(pid_of, key=str))
        m = flatten_cluster(res)
        m["cluster_sec"] = round(time.time() - t0, 2)
        m["evaluated_points"] = len(pid_of)
        return m

    def recommended_yaml(self, params: Dict[str, Any]) -> str:
        import yaml
        block = {"clusterer": {"module": self.spec["module"], "class": self.spec["class"], "params": {**self.spec["params"], **params}}}
        return "# bench/optimize.py 추천 (clustering/cluster_qdrant.py --method-config 로 사용)\n" + yaml.safe_dump(block, allow_unicode=True, sort_keys=False)

    def component(self) -> Dict[str, Any]:
        return {"module": self.spec["module"], "class": self.spec["class"], "params": self.spec["params"]}

    def gt(self) -> Dict[str, Any]:
        return {"dataset": "PRW", "collection": self.collection, "sources": self.sources, "points": len(self.ids),
                "gt_points": len(self.pid_of_all), "matches_cache": self.matches_cache}


class SearchProblem(Problem):
    """GT crop 임베딩 캐시(npz)로 1차 조합·재정렬을 다시 계산 (prw_eval_unified 와 같은 채점)."""
    stage = "search"
    name = "unified"

    def __init__(self, config_path: str, cache_dir: Path, split_file: Optional[Path], tune_part: Optional[str],
                 models: Sequence[str] = ("siglip2", "irra", "solider"), log=print):
        from eval.prw_eval_unified import l2n
        self.log = log
        self.sims: Dict[str, np.ndarray] = {}
        meta = None
        for m in models:
            path = cache_dir / f"prw_gt_{m}.npz"
            if not path.is_file():
                raise SystemExit(f"임베딩 캐시 없음: {path} (eval/prw_eval_unified.py --only-embed {m} 로 생성)")
            d = np.load(path, allow_pickle=False)
            fp = ledger.retriever_fingerprint_sha(config_path, m, log)
            if "retriever_fp" in d.files and fp and str(d["retriever_fp"]) != fp:
                log(f"[search] 경고: {m} 캐시의 임베더 지문이 지금 yaml 과 다름 → 결과가 지금 임베더를 대표하지 않음")
            self.sims[m] = (l2n(d["q_vecs"]) @ l2n(d["g_vecs"]).T).astype(np.float32)
            if meta is None:
                meta = dict(g_pids=np.asarray(d["g_pids"]), g_frames=np.asarray(d["g_frames"]), q_pids=np.asarray(d["q_pids"]),
                            q_frames=np.asarray(d["q_frames"]))
        assert meta is not None
        self.meta = meta
        self.parts: Dict[str, set] = {}
        if split_file:
            from bench.splits import load_split
            dd = load_split(split_file)
            self.parts = {k: set(int(x) for x in v) for k, v in dd["parts"].items()}
        self.tune_part = tune_part
        from eval.prw_eval_unified import DEFAULT_WEIGHTS, QDRANT_RRF_K_DEFAULT
        self.baseline = {"stage1": "siglip2+irra", "rerank": "solider", "w_siglip2": DEFAULT_WEIGHTS["siglip2"], "w_irra": DEFAULT_WEIGHTS["irra"],
                         "w_solider": DEFAULT_WEIGHTS["solider"], "rrf_k": float(QDRANT_RRF_K_DEFAULT), "prefetch": 200, "pool": 200}
        self.config_path = str(config_path)
        log(f"[search] gallery {len(meta['g_pids']):,} · query {len(meta['q_pids']):,} · 분할 {list(self.parts) or '없음'}")

    def query_mask(self, part: Optional[str]) -> np.ndarray:
        part = part or self.tune_part
        q = self.meta["q_pids"]
        if not part or part not in self.parts:
            return np.ones(len(q), dtype=bool)
        keep = self.parts[part]
        return np.array([int(p) in keep for p in q])

    def evaluate(self, params: Dict[str, Any], part: Optional[str] = None) -> Dict[str, Any]:
        from eval.prw_eval_unified import evaluate_ranked, rerank_by, rrf_fuse, topk_indices
        names = [n for n in str(params.get("stage1", "siglip2+irra")).split("+") if n]
        rerank = params.get("rerank", "solider")
        rerank = None if rerank in (None, "none", "") else str(rerank)
        weights = {"siglip2": float(params.get("w_siglip2", 1.0)), "irra": float(params.get("w_irra", 1.5)), "solider": float(params.get("w_solider", 1.5))}
        k = float(params.get("rrf_k", 2.0))
        pool = int(params.get("pool", 200))
        prefetch = max(int(params.get("prefetch", 200)), pool)
        mask = self.query_mask(part)
        idx = np.nonzero(mask)[0]
        t0 = time.time()
        ranked = []
        for qi in idx:
            if len(names) == 1:
                fused = topk_indices(self.sims[names[0]][qi], prefetch)
            else:
                lists = {m: topk_indices(self.sims[m][qi], prefetch) for m in names}
                fused, _ = rrf_fuse(lists, weights, k)
            cand = fused[:pool]
            ranked.append(rerank_by(cand, self.sims[rerank][qi]) if rerank else cand)
        res = evaluate_ranked(ranked, self.meta["q_pids"][idx], self.meta["q_frames"][idx], self.meta["g_pids"], self.meta["g_frames"])
        return {"map": res["mAP"], "rank1": res["Rank-1"], "rank5": res["Rank-5"], "rank10": res["Rank-10"],
                "pool_recall": res["pool_recall(%)"], "all_positives_in_pool": res["queries_all_positives_in_pool(%)"],
                "valid_queries": res["valid_queries"], "sec": round(time.time() - t0, 2)}

    def recommended_yaml(self, params: Dict[str, Any]) -> str:
        import yaml
        names = [n for n in str(params.get("stage1", "siglip2+irra")).split("+") if n]
        block = {"search_recommendation": {
            "stage1": names, "rerank": None if params.get("rerank") in ("none", None) else params.get("rerank"),
            "weights": {n: float(params.get(f"w_{n}", 1.0)) for n in names}, "rrf_k": float(params.get("rrf_k", 2.0)),
            "prefetch": int(params.get("prefetch", 200)), "pool": int(params.get("pool", 200)),
            "apply": "pipeline.yaml retrievers.<name>.weight 와 GUI/CLI --stage1/--rerank, 검색 후보 수(prefetch/pool)에 반영"}}
        return "# bench/optimize.py 추천 (통합 검색 조합)\n" + yaml.safe_dump(block, allow_unicode=True, sort_keys=False)

    def component(self) -> Dict[str, Any]:
        return {"variant": "unified", "models": list(self.sims)}

    def gt(self) -> Dict[str, Any]:
        return {"dataset": "PRW", "protocol": "GT crop gallery (prw_eval_unified 캐시)", "gallery_size": int(len(self.meta["g_pids"])),
                "query_total": int(len(self.meta["q_pids"]))}


class DetectProblem(Problem):
    """detections.jsonl 을 임계값별로 채점 (재검출 없음). Optuna 대신 결정적 스윕."""
    stage = "detect"

    def __init__(self, detections: Path, data_root: Path, iou: float = 0.5, step: float = 0.01, log=print):
        from eval import detect_eval_prw as de
        self.log = log
        self.meta, self.dets, _ = de.read_detections(detections)
        self.frames = sorted(self.dets)
        self.gt_by_frame, self.gt_stats = de.load_gt(data_root / "annotations", self.frames)
        self.iou = iou
        self.step = step
        self.de = de
        self.name = detections.parent.name
        det = self.meta.get("detector") or {}
        self.baseline = {"conf_threshold": float(det.get("config_conf_threshold") or 0.5)}
        self.detections = str(detections)
        log(f"[detect] {self.name}: 프레임 {len(self.frames):,} · GT {self.gt_stats.get('boxes'):,} · 운영 임계값 {self.baseline['conf_threshold']}")

    def sweep(self) -> List[Dict[str, Any]]:
        thresholds = [round(t, 4) for t in np.arange(0.05, 0.95 + 1e-9, self.step)]
        res = self.de.evaluate_method(self.dets, self.gt_by_frame, self.frames, self.iou, self.baseline["conf_threshold"], thresholds)
        rows = []
        for o in res["operating_points"]:
            rows.append({"params": {"conf_threshold": o["threshold"]},
                         "metrics": {"precision": o["precision"], "recall": o["recall"], "f1": o["f1"], "fp_duplicate": o["fp_duplicate"],
                                     "fp_background": o["fp_background"], "fn": o["fn"], "dets_per_frame": o["dets_per_frame"],
                                     "ap50": res["ap50"], "max_recall": res["max_recall"]}})
        return rows

    def evaluate(self, params: Dict[str, Any], part: Optional[str] = None) -> Dict[str, Any]:
        thr = float(params["conf_threshold"])
        res = self.de.evaluate_method(self.dets, self.gt_by_frame, self.frames, self.iou, thr, [thr])
        o = res["operating"]
        return {"precision": o["precision"], "recall": o["recall"], "f1": o["f1"], "fp_duplicate": o["fp_duplicate"],
                "fp_background": o["fp_background"], "fn": o["fn"], "dets_per_frame": o["dets_per_frame"], "ap50": res["ap50"]}

    def recommended_yaml(self, params: Dict[str, Any]) -> str:
        det = self.meta.get("detector") or {}
        import yaml
        block = {"detector": {"module": det.get("module"), "class": det.get("class"),
                              "params": {**{k: v for k, v in (det.get("params") or {}).items() if k != "conf_threshold"},
                                         "conf_threshold": float(params["conf_threshold"])}}}
        return "# bench/optimize.py 추천 (pipeline_tracking*.yaml detector 블록)\n" + yaml.safe_dump(block, allow_unicode=True, sort_keys=False)

    def component(self) -> Dict[str, Any]:
        det = self.meta.get("detector") or {}
        return {"module": det.get("module"), "class": det.get("class"), "params": det.get("params")}

    def gt(self) -> Dict[str, Any]:
        return {"dataset": "PRW", "split": (self.meta.get("split") or {}).get("split"), "frames": len(self.frames), "boxes": self.gt_stats.get("boxes"),
                "detections": self.detections}


# ---------------------------------------------------------------- 스터디
def run_study(problem: Problem, space: Dict[str, Dict[str, Any]], trials: int, objective: str,
              constraints: Sequence[Tuple[str, str, float]], out_dir: Path, seed: int = 42, timeout_min: float = 0,
              log: Callable[[str], Any] = print, rows_hook: Optional[Callable[[Dict[str, Any]], Any]] = None) -> Dict[str, Any]:
    """Optuna TPE 로 objective 최대화. 제약 위반은 큰 벌점(-100·위반량)으로 밀어내고 feasible 만 추천 후보로 쓴다."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    trials_path = out_dir / "trials.jsonl"

    def objective_fn(trial):
        params = sample(trial, space)
        t0 = time.time()
        metrics = problem.evaluate(params)
        vio = violations(metrics, constraints)
        feasible = all(v == 0 for v in vio.values())
        value = float(metrics.get(objective) or 0.0)
        penal = value - 100.0 * sum(v for v in vio.values() if math.isfinite(v)) - (1e6 if any(not math.isfinite(v) for v in vio.values()) else 0)
        row = {"number": trial.number, "params": params, "metrics": metrics, "feasible": feasible, "violations": vio,
               "value": value, "penalized": penal, "sec": round(time.time() - t0, 2)}
        rows.append(row)
        with trials_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(ledger._dumps(row) + "\n")
        log(f"  trial {trial.number:>3} {objective}={value:.4f} {'ok ' if feasible else 'NG '} "
            + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()) + f"  ({row['sec']}s)")
        if rows_hook:
            rows_hook(row)
        return penal

    storage = optuna.storages.RDBStorage(url=f"sqlite:///{(out_dir / 'optuna.db').as_posix()}")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed), storage=storage,
                                study_name=out_dir.name, load_if_exists=True)
    if problem.baseline and all(k in space for k in problem.baseline if k in space):
        base = {k: v for k, v in problem.baseline.items() if k in space}
        if base:
            try:
                study.enqueue_trial(base)      # 첫 trial = 현재 운영값 (baseline 이 탐색 공간 안이면)
            except Exception:                  # noqa: BLE001
                pass
    try:
        study.optimize(objective_fn, n_trials=trials, timeout=timeout_min * 60 if timeout_min else None)
    finally:
        try:                                   # Windows: sqlite 핸들을 닫아야 폴더 삭제·이동이 된다
            storage.remove_session()
            storage.engine.dispose()
        except Exception:                      # noqa: BLE001
            pass
    feasible = [r for r in rows if r["feasible"]]
    best = max(feasible, key=lambda r: r["value"]) if feasible else (max(rows, key=lambda r: r["penalized"]) if rows else None)
    return {"rows": rows, "best": best, "feasible": len(feasible), "study_name": study.study_name}


def sweep_study(problem: DetectProblem, objective: str, constraints: Sequence[Tuple[str, str, float]], out_dir: Path,
                log: Callable[[str], Any] = print) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with (out_dir / "trials.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(problem.sweep()):
            vio = violations(r["metrics"], constraints)
            row = {"number": i, "params": r["params"], "metrics": r["metrics"], "feasible": all(v == 0 for v in vio.values()),
                   "violations": vio, "value": float(r["metrics"].get(objective) or 0.0), "penalized": None, "sec": 0}
            rows.append(row)
            f.write(ledger._dumps(row) + "\n")
    feasible = [r for r in rows if r["feasible"]]
    best = max(feasible, key=lambda r: r["value"]) if feasible else (max(rows, key=lambda r: r["value"]) if rows else None)
    log(f"[detect] 스윕 {len(rows)} 임계값 · feasible {len(feasible)} · best {objective}={best['value']:.4f} @ {best['params']}" if best else "[detect] 결과 없음")
    return {"rows": rows, "best": best, "feasible": len(feasible), "study_name": out_dir.name}


# ---------------------------------------------------------------- 보고 · 원장
def fmtv(v: Any) -> str:
    return ledger.fmt(v)


def write_report(problem: Problem, study: Dict[str, Any], objective: str, constraints: Sequence[Tuple[str, str, float]],
                 base_tune: Dict[str, Any], out_dir: Path, validation: Optional[Dict[str, Any]], pareto: List[Dict[str, Any]]) -> Path:
    stage = problem.stage
    best = study["best"]
    keys = [objective] + [c[0] for c in constraints if c[0] != objective]
    extra_keys = {"cluster": ["pair_recall", "b3_precision", "b3_recall", "noise_ratio", "mixed_clusters"],
                  "search": ["rank1", "rank10", "all_positives_in_pool"], "detect": ["precision", "recall", "fp_background", "fn"]}.get(stage, [])
    cols = list(dict.fromkeys(keys + extra_keys))
    lines = [f"# 탐색 보고 — {stage} · {problem.name} · {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"목적: **{objective} 최대화**, 제약: {', '.join(f'{k}{op}{v}' for k, op, v in constraints) or '없음'} · trial {len(study['rows'])} (feasible {study['feasible']}) · 잡음 기준 {NOISE.get(stage)}",
             ""]
    lines += ["## 기준(운영값) vs 추천 (tune 분할)", "", "| | " + " | ".join(cols) + " | params |", "|---|" + "---|" * (len(cols) + 1)]
    lines.append("| 운영값 | " + " | ".join(fmtv(base_tune.get(k)) for k in cols) + f" | `{json.dumps(problem.baseline, ensure_ascii=False)}` |")
    if best:
        lines.append("| 추천 | " + " | ".join(fmtv(best['metrics'].get(k)) for k in cols) + f" | `{json.dumps(best['params'], ensure_ascii=False)}` |")
        delta = (best["metrics"].get(objective) or 0) - (base_tune.get(objective) or 0)
        lines.append("")
        lines.append(f"{objective} 차이 = {delta:+.4f} → {'잡음 초과 (의미 있음)' if abs(delta) > NOISE.get(stage, 0) else '잡음 이내 (같다고 본다)'}")
    if validation:
        lines += ["", "## holdout 검증 (탐색에 쓰지 않은 인물)", "", "| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for label in ("baseline", "best"):
            m = validation.get(label) or {}
            lines.append(f"| {label} | " + " | ".join(fmtv(m.get(k)) for k in cols) + " |")
        d = validation.get("delta")
        if d is not None:
            lines.append("")
            lines.append(f"holdout {objective} 차이 = {d:+.4f} → {'잡음 초과' if abs(d) > NOISE.get(stage, 0) else '잡음 이내'}; "
                         f"tune 차이와 부호가 {'같음' if validation.get('consistent') else '다름 (과적합 의심)'}")
            if validation.get("constraints_hold") is False:
                lines.append(f"**경고: holdout 에서 제약 위반** {validation.get('violations_holdout')} → 추천값을 그대로 채택하지 말 것 "
                             f"(제약을 더 엄격히 두고 재탐색하거나, 제약 지표를 목적에 포함).")
    lines += ["", f"## Pareto ({objective} vs {PARETO_SECOND.get(stage)})", ""]
    for r in pareto[:12]:
        lines.append(f"- {objective}={fmtv(r['metrics'].get(objective))} {PARETO_SECOND.get(stage)}={fmtv(r['metrics'].get(PARETO_SECOND.get(stage)))} "
                     f"{'ok' if r['feasible'] else 'NG'} `{json.dumps(r['params'], ensure_ascii=False)}`")
    top = sorted(study["rows"], key=lambda r: -(r["value"] if r["feasible"] else r["value"] - 1e3))[:10]
    lines += ["", "## 상위 10 trial", "", "| # | feasible | " + " | ".join(cols) + " | params |", "|---|---|" + "---|" * (len(cols) + 1)]
    for r in top:
        lines.append(f"| {r['number']} | {'ok' if r['feasible'] else 'NG'} | " + " | ".join(fmtv(r['metrics'].get(k)) for k in cols) + f" | `{json.dumps(r['params'], ensure_ascii=False)}` |")
    lines += ["", f"산출물: `{out_dir}` (trials.jsonl · best.json · recommended.yaml · pareto.json)"]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def record_study(problem: Problem, study: Dict[str, Any], objective: str, constraints: Sequence[Tuple[str, str, float]],
                 base_tune: Dict[str, Any], out_dir: Path, validation: Optional[Dict[str, Any]], args: argparse.Namespace,
                 name: str) -> List[Dict[str, Any]]:
    best = study["best"]
    if not best:
        return []
    extra = {"study": {"dir": str(out_dir), "trials": len(study["rows"]), "feasible": study["feasible"], "objective": objective,
                       "constraints": [f"{k}{op}{v}" for k, op, v in constraints], "baseline_params": problem.baseline,
                       "baseline_metrics_tune": base_tune, "pid_split": getattr(args, "pid_split", None), "noise": NOISE.get(problem.stage)},
             "hardware": ledger.hardware_info()}
    entries = [ledger.make_entry(problem.stage, "bench.optimize", f"study:{name}", component=problem.component(), params=best["params"],
                                 gt={**problem.gt(), "pid_split": getattr(args, "pid_split", None)}, metrics=best["metrics"],
                                 versions=ledger.versions_info(), report=out_dir / "report.md", command=sys.argv[1:],
                                 seed=getattr(args, "seed", None), note=f"optuna best (tune); baseline {objective}={base_tune.get(objective)}",
                                 extra=extra)]
    if validation:
        for label in ("best", "baseline"):
            m = validation.get(label)
            if m:
                params = best["params"] if label == "best" else problem.baseline
                entries.append(ledger.make_entry(problem.stage, "bench.optimize", f"study:{name}:holdout_{label}", component=problem.component(),
                                                 params=params, gt={**problem.gt(), "pid_split": validation.get("holdout_split")}, metrics=m,
                                                 report=out_dir / "report.md", command=sys.argv[1:], note=f"holdout {label}",
                                                 extra={"study": {"dir": str(out_dir), "role": f"holdout_{label}"}}))
    return entries


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P3 하이퍼파라미터 탐색 (Optuna) / 검출 임계값 스윕")
    sub = p.add_subparsers(dest="stage", required=True)

    def common(s):
        s.add_argument("--name", default=None)
        s.add_argument("--objective", default=None)
        s.add_argument("--constraint", action="append", default=None, help="지표>=값 (여러 번). 기본: 기준표 §R 권고")
        s.add_argument("--space", action="append", default=[], help="이름=int:low:high | float:low:high[:log] | cat:a,b")
        s.add_argument("--trials", type=int, default=None)
        s.add_argument("--timeout-min", type=float, default=0)
        s.add_argument("--seed", type=int, default=42)
        s.add_argument("--pid-split", default=str(PROJECT_ROOT / "bench" / "splits" / "prw_pids_seed42.json") + ":tune")
        s.add_argument("--validate", action="store_true", help="best 와 운영값을 holdout 분할로 재평가")
        s.add_argument("--studies-dir", default=str(DEFAULT_STUDIES_DIR))
        s.add_argument("--ledger", default=None)
        s.add_argument("--no-ledger", action="store_true")
        s.add_argument("--config", default=str(DEFAULT_PIPELINE))
        s.add_argument("--data-root", default="./data/PRW")
        # 단계별 옵션도 모든 하위 명령이 받는다 (GUI 가 한 화면에서 같은 인자를 넘길 수 있게; 해당 없는 것은 무시)
        g = s.add_argument_group("cluster")
        g.add_argument("--method", default="leiden")
        g.add_argument("--method-config", default=None)
        g.add_argument("--param", action="extend", nargs="+", default=[], help="고정 params key=value (여러 개)")
        g.add_argument("--target", default="person")
        g.add_argument("--sources", default="prw_image")
        g.add_argument("--vector", default=None)
        g.add_argument("--max-points", type=int, default=0)
        g.add_argument("--min-cluster-size", type=int, default=2)
        g.add_argument("--matches-cache", default=str(PROJECT_ROOT / "eval" / "results" / "cache" / "prw_gt_matches.jsonl"))
        g.add_argument("--vector-cache", default="outputs/clustering/cache/bench")
        g = s.add_argument_group("search")
        g.add_argument("--cache-dir", default=str(PROJECT_ROOT / "eval" / "results" / "cache"))
        g = s.add_argument_group("detect")
        g.add_argument("--detections", default=None, help="detect: detections.jsonl (필수)")
        g.add_argument("--iou", type=float, default=0.5)
        g.add_argument("--step", type=float, default=0.01)

    for stage in ("cluster", "search", "detect"):
        common(sub.add_parser(stage))
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    stage = args.stage
    objective = args.objective or DEFAULT_OBJECTIVE[stage]
    constraints = parse_constraints(args.constraint if args.constraint is not None else DEFAULT_CONSTRAINTS[stage])
    split_file, tune_part = (None, None)
    if args.pid_split and stage != "detect":
        from bench.splits import parse_split_arg
        split_file, tune_part = parse_split_arg(args.pid_split)
    started = time.time()

    if stage == "cluster":
        from bench.run import parse_params
        problem: Problem = ClusterProblem(args.method, args.method_config, parse_params(args.param), args.target, args.sources, args.vector,
                                          args.max_points, args.min_cluster_size, args.config, Path(args.matches_cache), split_file, tune_part,
                                          args.vector_cache)
        space = parse_space(args.space) or DEFAULT_SPACES.get(args.method, {})
        trials = args.trials or 60
    elif stage == "search":
        problem = SearchProblem(args.config, Path(args.cache_dir), split_file, tune_part)
        space = parse_space(args.space) or DEFAULT_SPACES["search"]
        trials = args.trials or 40
    else:
        if not args.detections:
            raise SystemExit("detect 는 --detections <detections.jsonl> 이 필요합니다")
        problem = DetectProblem(Path(args.detections), Path(args.data_root).expanduser().resolve(), args.iou, args.step)
        space, trials = {}, 0
    if stage != "detect" and not space:
        raise SystemExit("탐색 공간이 비었습니다 (--space 로 지정)")

    from bench.run import slug
    name = args.name or f"{problem.name}"
    out_dir = Path(args.studies_dir).resolve() / f"{stage}_{slug(name)}_{time.strftime('%Y%m%dT%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[optimize] {stage} · {name} · objective {objective} · constraints {[f'{k}{op}{v}' for k, op, v in constraints]} · trials {trials} · out {out_dir}")
    base_tune = problem.evaluate(problem.baseline)
    print(f"[optimize] 운영값 (tune): " + " ".join(f"{k}={fmtv(v)}" for k, v in base_tune.items() if k in (objective, *[c[0] for c in constraints])))

    if stage == "detect":
        study = sweep_study(problem, objective, constraints, out_dir)          # type: ignore[arg-type]
    else:
        study = run_study(problem, space, trials, objective, constraints, out_dir, args.seed, args.timeout_min)
    best = study["best"]
    if not best:
        raise SystemExit("trial 결과가 없습니다")
    pareto = pareto_front(study["rows"], objective, PARETO_SECOND.get(stage, objective))
    (out_dir / "pareto.json").write_text(ledger._dumps(pareto, indent=1), encoding="utf-8", newline="\n")
    (out_dir / "best.json").write_text(ledger._dumps({"objective": objective, "constraints": [f"{k}{op}{v}" for k, op, v in constraints],
                                                      "best": best, "baseline": {"params": problem.baseline, "metrics_tune": base_tune},
                                                      "space": space, "trials": len(study["rows"]), "feasible": study["feasible"]}, indent=1),
                                       encoding="utf-8", newline="\n")
    (out_dir / "recommended.yaml").write_text(problem.recommended_yaml(best["params"]), encoding="utf-8", newline="\n")

    validation = None
    if args.validate and stage != "detect":
        holdout = "holdout" if "holdout" in getattr(problem, "parts", {}) else None
        if holdout is None:
            print("[optimize] holdout 분할이 없어 검증 생략")
        else:
            mb = problem.evaluate(best["params"], part=holdout)
            mo = problem.evaluate(problem.baseline, part=holdout)
            d_hold = (mb.get(objective) or 0) - (mo.get(objective) or 0)
            d_tune = (best["metrics"].get(objective) or 0) - (base_tune.get(objective) or 0)
            vio_hold = violations(mb, constraints)
            validation = {"holdout_split": f"{split_file}:{holdout}", "best": mb, "baseline": mo, "delta": round(d_hold, 6),
                          "delta_tune": round(d_tune, 6), "consistent": (d_hold > 0) == (d_tune > 0) or abs(d_hold) <= NOISE.get(stage, 0),
                          "constraints_hold": all(v == 0 for v in vio_hold.values()), "violations_holdout": vio_hold}
            (out_dir / "validation.json").write_text(ledger._dumps(validation, indent=1), encoding="utf-8", newline="\n")
            print(f"[optimize] holdout: best {objective}={fmtv(mb.get(objective))} vs 운영 {fmtv(mo.get(objective))} (Δ {d_hold:+.4f}; tune Δ {d_tune:+.4f})"
                  + ("" if validation["constraints_hold"] else f" · 경고: holdout 에서 제약 위반 {vio_hold} → 그대로 채택 금지"))
    report = write_report(problem, study, objective, constraints, base_tune, out_dir, validation, pareto)
    print(f"[optimize] best (tune): {objective}={best['value']:.4f} feasible={best['feasible']} params={json.dumps(best['params'], ensure_ascii=False)}")
    print(f"[optimize] report: {report}  ({time.time() - started:.0f}s)")
    entries = record_study(problem, study, objective, constraints, base_tune, out_dir, validation, args, name)
    ledger.record(lambda: entries, args.ledger, args.no_ledger)
    print(f"RESULT_SUMMARY: {out_dir / 'best.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
