"""bench/combos.py — P4: 구성 요소 조합 탐색 (임베더 × 재정렬 / 벡터 × 클러스터러) → 리더보드 → 유망 조합만 미세조정 → 채택 yaml.

  python bench/combos.py search  [--embedders siglip2,irra,solider] [--reranks none,solider,irra,siglip2]
                                 [--pools 200,1000] [--params-from bench/studies/<search study>/best.json]
                                 [--pid-split …:tune] [--validate] [--top 3] [--refine-trials 20] [--adopt]
  python bench/combos.py cluster [--vectors siglip2,irra,solider] [--methods leiden,dbscan_v6]
                                 [--params-from bench/studies/<cluster study>/best.json] [--max-points 0]
                                 [--pid-split …:tune] [--validate] [--top 2] [--refine-trials 0] [--adopt]
  python bench/combos.py report  [--stage search|cluster]           원장의 combo 행으로 리더보드 표

격자의 모든 조합을 같은 GT(tune 분할)로 재고, 운영 조합의 순위를 숫자로 남긴다. --validate 면 holdout 으로도 잰다(검색은 전부,
클러스터는 상위 --top 만). --refine-trials N 이면 상위 --top 조합마다 Optuna N trial 로 미세조정(bench/optimize.run_study 재사용).
산출물 bench/combos/<단계_이름_시각>/ : grid.jsonl · leaderboard.md · best.json · recommended.yaml · pipeline_best.yaml(검색: 가중치만 바꾼
pipeline.yaml 사본, 로더 검증) — --adopt 면 프로젝트 루트에 pipeline_best.yaml 복사 (GUI 드롭다운에 자동 등장).
원장: 조합마다 1행 (name = combo:<라벨>, extra.combo), 최종 추천 combo_best 행.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench import ledger  # noqa: E402
from bench import optimize as O  # noqa: E402

DEFAULT_COMBOS_DIR = PROJECT_ROOT / "bench" / "combos"
DEFAULT_PIPELINE = PROJECT_ROOT / "pipeline.yaml"
PROD = {"search": {"stage1": "siglip2+irra", "rerank": "solider"}, "cluster": {"method": "leiden", "vector": "solider"}}


# ---------------------------------------------------------------- 격자
def search_grid(embedders: Sequence[str], reranks: Sequence[str], pools: Sequence[int], base: Dict[str, Any]) -> List[Dict[str, Any]]:
    """비어 있지 않은 임베더 부분집합 × 재정렬 × 후보 수. 단일 임베더에 같은 벡터로 재정렬하는 조합은 순서가 같아 뺀다."""
    combos = []
    for r in range(1, len(embedders) + 1):
        for subset in itertools.combinations(embedders, r):
            for rerank in reranks:
                if rerank != "none" and len(subset) == 1 and rerank == subset[0]:
                    continue
                for pool in pools:
                    params = dict(base)
                    params.update(stage1="+".join(subset), rerank=rerank, pool=int(pool), prefetch=max(int(pool), int(base.get("prefetch", pool))))
                    combos.append({"label": f"{'+'.join(subset)}→{rerank}@{pool}", "params": params,
                                   "axes": {"stage1": "+".join(subset), "rerank": rerank, "pool": int(pool)}})
    return combos


def cluster_grid(vectors: Sequence[str], methods: Sequence[str]) -> List[Dict[str, Any]]:
    """벡터 × 알고리즘. DBSCAN v6 는 세 벡터를 결합하므로 primary 벡터 축이 의미 없어 하나만."""
    combos = []
    for method in methods:
        vs = list(vectors) if method != "dbscan_v6" else [vectors[0] if "solider" not in vectors else "solider"]
        for v in vs:
            combos.append({"label": f"{method}@{v}", "axes": {"method": method, "vector": v}})
    return combos


def load_params_from(path: Optional[str], stage: str) -> Dict[str, Any]:
    """optimize.py 의 best.json → 조합 격자의 기본 params (검색: 가중치·rrf_k·prefetch / 클러스터: 플러그인 params)."""
    if not path:
        return {}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    params = dict((d.get("best") or {}).get("params") or {})
    if stage == "search":
        for k in ("stage1", "rerank", "pool"):
            params.pop(k, None)
    return params


def rank_rows(rows: Sequence[Dict[str, Any]], objective: str, constraints) -> List[Dict[str, Any]]:
    """feasible 먼저, 목적 지표 내림차순. 각 행에 rank 를 붙인다."""
    def key(r):
        return (0 if r["feasible"] else 1, -(float(r["metrics"].get(objective) or 0.0)))
    out = sorted(rows, key=key)
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out


def leaderboard_md(stage: str, rows: Sequence[Dict[str, Any]], objective: str, constraints, prod_label: Optional[str],
                   validation: Dict[str, Dict[str, Any]], refined: Dict[str, Dict[str, Any]]) -> str:
    second = O.PARETO_SECOND.get(stage, objective)
    cols = list(dict.fromkeys([objective, second] + {"search": ["rank1", "rank10"], "cluster": ["pair_recall", "b3_precision", "noise_ratio", "mixed_clusters"]}.get(stage, [])))
    lines = [f"# 조합 리더보드 — {stage} · {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"목적 **{objective}** · 제약 {', '.join(f'{k}{op}{v}' for k, op, v in constraints) or '없음'} · 조합 {len(rows)} · 잡음 기준 {O.NOISE.get(stage)}"
             + (f" · 운영 조합 = `{prod_label}`" if prod_label else ""), "",
             "| 순위 | 조합 | feasible | " + " | ".join(cols) + " | holdout " + objective + " | 미세조정 " + objective + " |",
             "|---|---|---|" + "---|" * len(cols) + "---|---|"]
    for r in rows:
        v = validation.get(r["label"], {}).get(objective)
        rf = refined.get(r["label"], {}).get("value")
        tag = " **(운영)**" if r["label"] == prod_label else ""
        lines.append(f"| {r['rank']} | `{r['label']}`{tag} | {'ok' if r['feasible'] else 'NG'} | " + " | ".join(ledger.fmt(r['metrics'].get(c)) for c in cols)
                     + f" | {ledger.fmt(v)} | {ledger.fmt(rf)} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- pipeline_best.yaml (검색 가중치만 바꾼 사본, 주석 보존)
def write_pipeline_best(src: Path, dst: Path, weights: Dict[str, float]) -> Tuple[bool, str]:
    text = src.read_text(encoding="utf-8-sig")
    out_lines = []
    current = None
    in_retrievers = False
    changed = []
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        if re.match(r"^retrievers:\s*(#.*)?$", stripped):
            in_retrievers = True
        elif re.match(r"^[A-Za-z_]", stripped):
            in_retrievers = False
        m = re.match(r"^  ([A-Za-z0-9_]+):\s*(#.*)?$", stripped)
        if in_retrievers and m:
            current = m.group(1)
        m2 = re.match(r"^(\s+weight:\s*)([0-9.]+)(.*)$", stripped)
        if in_retrievers and current in weights and m2:
            line = f"{m2.group(1)}{weights[current]}{m2.group(3)}" + ("\r\n" if line.endswith("\r\n") else "\n")
            changed.append(current)
        out_lines.append(line)
    header = f"# bench/combos.py 가 만든 사본 ({time.strftime('%Y-%m-%d %H:%M')}): retrievers.*.weight 만 변경 {weights}. 원본 pipeline.yaml 은 그대로.\n"
    dst.write_text(header + "".join(out_lines), encoding="utf-8", newline="")
    try:
        from config import PipelineConfig
        PipelineConfig.load(dst)
        return True, f"weight 변경 {changed}; 로더 검증 OK"
    except Exception as exc:  # noqa: BLE001
        return False, f"weight 변경 {changed}; 로더 검증 실패: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- 실행
def run_search(args) -> int:
    split_file, tune_part = O.parse_split_arg_safe(args.pid_split)
    problem = O.SearchProblem(args.config, Path(args.cache_dir), split_file, tune_part)
    base = {**problem.baseline, **load_params_from(args.params_from, "search")}
    embedders = [e for e in args.embedders.split(",") if e]
    reranks = [r for r in args.reranks.split(",") if r]
    pools = [int(p) for p in args.pools.split(",") if p]
    objective = args.objective or O.DEFAULT_OBJECTIVE["search"]
    constraints = O.parse_constraints(args.constraint if args.constraint is not None else O.DEFAULT_CONSTRAINTS["search"])
    combos = search_grid(embedders, reranks, pools, base)
    out_dir = Path(args.combos_dir).resolve() / f"search_{args.name or 'grid'}_{time.strftime('%Y%m%dT%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[combos] search · 조합 {len(combos)} · 목적 {objective} · 제약 {[f'{k}{op}{v}' for k, op, v in constraints]} · 기본 params {json.dumps(base, ensure_ascii=False)}")
    rows = []
    with (out_dir / "grid.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for i, c in enumerate(combos, 1):
            t0 = time.time()
            m = problem.evaluate(c["params"])
            vio = O.violations(m, constraints)
            row = {"label": c["label"], "axes": c["axes"], "params": c["params"], "metrics": m, "feasible": all(v == 0 for v in vio.values()),
                   "violations": vio, "sec": round(time.time() - t0, 2)}
            rows.append(row)
            f.write(ledger._dumps(row) + "\n")
            print(f"  {i:>3}/{len(combos)} {c['label']:<32} {objective}={m.get(objective):.2f} pool_recall={m.get('pool_recall'):.1f} {'ok' if row['feasible'] else 'NG'} ({row['sec']}s)")
    rows = rank_rows(rows, objective, constraints)
    prod_label = next((r["label"] for r in rows if r["axes"]["stage1"] == PROD["search"]["stage1"] and r["axes"]["rerank"] == PROD["search"]["rerank"]
                       and r["axes"]["pool"] == 200), None)
    validation: Dict[str, Dict[str, Any]] = {}
    if args.validate and "holdout" in problem.parts:
        for r in rows:
            validation[r["label"]] = problem.evaluate(r["params"], part="holdout")
    refined: Dict[str, Dict[str, Any]] = {}
    if args.refine_trials > 0:
        for r in rows[:args.top]:
            space = {k: v for k, v in O.DEFAULT_SPACES["search"].items() if k not in ("stage1", "rerank")}
            fixed = {"stage1": r["axes"]["stage1"], "rerank": r["axes"]["rerank"]}
            sub = _FixedProblem(problem, fixed)
            study = O.run_study(sub, space, args.refine_trials, objective, constraints, out_dir / f"refine_{ledger.fmt(r['rank'])}_{_slug(r['label'])}", args.seed, log=lambda s: None)
            if study["best"]:
                refined[r["label"]] = {"value": study["best"]["value"], "params": {**fixed, **study["best"]["params"]}, "metrics": study["best"]["metrics"]}
                print(f"  [refine] {r['label']}: {objective} {r['metrics'].get(objective):.2f} → {study['best']['value']:.2f}")
    best = _pick_best(rows, refined, objective)
    md = leaderboard_md("search", rows, objective, constraints, prod_label, validation, refined)
    (out_dir / "leaderboard.md").write_text(md, encoding="utf-8", newline="\n")
    (out_dir / "best.json").write_text(ledger._dumps({"objective": objective, "best": best, "prod_label": prod_label, "validation": validation, "refined": refined}, indent=1),
                                       encoding="utf-8", newline="\n")
    (out_dir / "recommended.yaml").write_text(problem.recommended_yaml(best["params"]), encoding="utf-8", newline="\n")
    names = best["params"]["stage1"].split("+")
    weights = {n: float(best["params"].get(f"w_{n}", 1.0)) for n in names}
    ok, note = write_pipeline_best(Path(args.config), out_dir / "pipeline_best.yaml", weights)
    print(f"[combos] pipeline_best.yaml: {note}")
    if args.adopt and ok:
        shutil.copyfile(out_dir / "pipeline_best.yaml", PROJECT_ROOT / "pipeline_best.yaml")
        print(f"[combos] 채택 → {PROJECT_ROOT / 'pipeline_best.yaml'} (GUI 임베더 구성 드롭다운에 등장)")
    print(md)
    _record(args, "search", rows, best, prod_label, validation, out_dir, problem, objective, constraints)
    print(f"RESULT_SUMMARY: {out_dir / 'best.json'}")
    return 0


def run_cluster(args) -> int:
    split_file, tune_part = O.parse_split_arg_safe(args.pid_split)
    vectors = [v for v in args.vectors.split(",") if v]
    methods = [m for m in args.methods.split(",") if m]
    tuned = load_params_from(args.params_from, "cluster")
    objective = args.objective or O.DEFAULT_OBJECTIVE["cluster"]
    constraints = O.parse_constraints(args.constraint if args.constraint is not None else O.DEFAULT_CONSTRAINTS["cluster"])
    combos = cluster_grid(vectors, methods)
    out_dir = Path(args.combos_dir).resolve() / f"cluster_{args.name or 'grid'}_{time.strftime('%Y%m%dT%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[combos] cluster · 조합 {len(combos)} · 목적 {objective} · 제약 {[f'{k}{op}{v}' for k, op, v in constraints]} · 미세조정 params {json.dumps(tuned, ensure_ascii=False)}")
    rows = []
    problems: Dict[str, O.ClusterProblem] = {}
    tuned_method = _tuned_method(args.params_from)
    with (out_dir / "grid.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for i, c in enumerate(combos, 1):
            method, vector = c["axes"]["method"], c["axes"]["vector"]
            # 탐색된 params 는 그 방법(스터디 폴더 이름으로 판별)에만 적용; 판별 불가면 전부에 적용
            params = tuned if (tuned and tuned_method in (None, method)) else {}
            prob = O.ClusterProblem(method, None, {}, args.target, args.sources, vector, args.max_points, args.min_cluster_size, args.config,
                                    Path(args.matches_cache), split_file, tune_part, args.vector_cache, log=lambda s: None)
            problems[c["label"]] = prob
            t0 = time.time()
            m = prob.evaluate(params)
            vio = O.violations(m, constraints)
            row = {"label": c["label"], "axes": c["axes"], "params": {**prob.baseline, **params}, "metrics": m,
                   "feasible": all(v == 0 for v in vio.values()), "violations": vio, "sec": round(time.time() - t0, 2)}
            rows.append(row)
            f.write(ledger._dumps(row) + "\n")
            print(f"  {i:>2}/{len(combos)} {c['label']:<20} {objective}={m.get(objective):.4f} pair_precision={m.get('pair_precision'):.4f} {'ok' if row['feasible'] else 'NG'} ({row['sec']}s)")
    rows = rank_rows(rows, objective, constraints)
    prod_label = f"{PROD['cluster']['method']}@{PROD['cluster']['vector']}"
    validation: Dict[str, Dict[str, Any]] = {}
    if args.validate:
        for r in rows[:args.top]:
            prob = problems[r["label"]]
            if "holdout" in prob.parts:
                validation[r["label"]] = prob.evaluate({k: v for k, v in r["params"].items() if k in O.DEFAULT_SPACES.get(r["axes"]["method"], {})}, part="holdout")
    refined: Dict[str, Dict[str, Any]] = {}
    if args.refine_trials > 0:
        for r in rows[:args.top]:
            prob = problems[r["label"]]
            space = O.DEFAULT_SPACES.get(r["axes"]["method"], {})
            study = O.run_study(prob, space, args.refine_trials, objective, constraints, out_dir / f"refine_{r['rank']}_{_slug(r['label'])}", args.seed, log=lambda s: None)
            if study["best"]:
                refined[r["label"]] = {"value": study["best"]["value"], "params": study["best"]["params"], "metrics": study["best"]["metrics"]}
                print(f"  [refine] {r['label']}: {objective} {r['metrics'].get(objective):.4f} → {study['best']['value']:.4f}")
    best = _pick_best(rows, refined, objective)
    md = leaderboard_md("cluster", rows, objective, constraints, prod_label, validation, refined)
    (out_dir / "leaderboard.md").write_text(md, encoding="utf-8", newline="\n")
    (out_dir / "best.json").write_text(ledger._dumps({"objective": objective, "best": best, "prod_label": prod_label, "validation": validation, "refined": refined}, indent=1),
                                       encoding="utf-8", newline="\n")
    prob = problems[best["label"]]
    (out_dir / "recommended.yaml").write_text(prob.recommended_yaml({k: v for k, v in best["params"].items() if k in O.DEFAULT_SPACES.get(best["axes"]["method"], {})})
                                              + f"# primary vector: {best['axes']['vector']} (clustering/cluster_qdrant.py --vector {best['axes']['vector']} --method-config <이 파일>)\n",
                                              encoding="utf-8", newline="\n")
    if args.adopt:
        shutil.copyfile(out_dir / "recommended.yaml", PROJECT_ROOT / "clusterer_best.yaml")
        print(f"[combos] 채택 → {PROJECT_ROOT / 'clusterer_best.yaml'} (4b 클러스터링 플러그인 --method-config)")
    print(md)
    _record(args, "cluster", rows, best, prod_label, validation, out_dir, prob, objective, constraints)
    print(f"RESULT_SUMMARY: {out_dir / 'best.json'}")
    return 0


def run_report(args) -> int:
    entries = ledger.read_entries(ledger.resolve_ledger_path(args.ledger) or ledger.DEFAULT_LEDGER)
    rows = [e for e in entries if str(e.get("name", "")).startswith("combo:") and (not args.stage or e.get("stage") == args.stage)]
    if not rows:
        print("combo 행이 없습니다")
        return 1
    latest = ledger.latest_by_name(rows)
    stage = args.stage or next(iter(latest.values()))["stage"]
    keys = O.METRIC_KEYS_FOR.get(stage) if hasattr(O, "METRIC_KEYS_FOR") else ledger.METRIC_KEYS.get(stage, [])[:6]
    print(ledger.render_table(list(latest.values()), stage, keys))
    return 0


# ---------------------------------------------------------------- 도우미
class _FixedProblem(O.Problem):
    """일부 params 를 고정한 채 나머지만 탐색하는 래퍼 (검색 조합 미세조정용)."""

    def __init__(self, inner: O.Problem, fixed: Dict[str, Any]):
        self.inner, self.fixed = inner, fixed
        self.stage, self.name, self.baseline = inner.stage, inner.name, {k: v for k, v in inner.baseline.items() if k not in fixed}
        self.parts = getattr(inner, "parts", {})

    def evaluate(self, params, part=None):
        return self.inner.evaluate({**params, **self.fixed}, part=part)

    def recommended_yaml(self, params):
        return self.inner.recommended_yaml({**params, **self.fixed})


def _tuned_method(params_from: Optional[str]) -> Optional[str]:
    if not params_from:
        return None
    name = Path(params_from).parent.name        # cluster_<name>_<ts> — 이름에 leiden/dbscan 이 들어 있으면 그 방법
    for m in ("dbscan_v6", "leiden"):
        if m in name:
            return m
    return None


def _slug(text: str) -> str:
    from bench.run import slug
    return slug(text.replace("→", "_to_").replace("@", "_at_"))


def _pick_best(rows: Sequence[Dict[str, Any]], refined: Dict[str, Dict[str, Any]], objective: str) -> Dict[str, Any]:
    cands = []
    for r in rows:
        cands.append({"label": r["label"], "axes": r["axes"], "params": r["params"], "metrics": r["metrics"], "feasible": r["feasible"], "source": "grid"})
        if r["label"] in refined:
            rf = refined[r["label"]]
            cands.append({"label": r["label"], "axes": r["axes"], "params": rf["params"], "metrics": rf["metrics"], "feasible": True, "source": "refined"})
    feasible = [c for c in cands if c["feasible"]]
    return max(feasible or cands, key=lambda c: float(c["metrics"].get(objective) or 0.0))


def _record(args, stage: str, rows, best, prod_label, validation, out_dir: Path, problem, objective, constraints) -> None:
    entries = []
    for r in rows:
        entries.append(ledger.make_entry(stage, "bench.combos", f"combo:{r['label']}", component={"axes": r["axes"], **problem.component()},
                                         params=r["params"], gt={**problem.gt(), "pid_split": args.pid_split}, metrics=r["metrics"],
                                         report=out_dir / "leaderboard.md", command=sys.argv[1:], note=f"rank {r['rank']}" + (" (운영)" if r["label"] == prod_label else ""),
                                         extra={"combo": {"dir": str(out_dir), "rank": r["rank"], "feasible": r["feasible"], "objective": objective,
                                                          "holdout": validation.get(r["label"]), "is_prod": r["label"] == prod_label}}))
    entries.append(ledger.make_entry(stage, "bench.combos", "combo_best", component={"axes": best["axes"], **problem.component()}, params=best["params"],
                                     gt={**problem.gt(), "pid_split": args.pid_split}, metrics=best["metrics"], report=out_dir / "best.json",
                                     command=sys.argv[1:], note=f"best of {len(rows)} combos ({best['source']}): {best['label']}",
                                     extra={"combo": {"dir": str(out_dir), "label": best["label"], "source": best["source"], "prod_label": prod_label}}))
    ledger.record(lambda: entries, args.ledger, args.no_ledger)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P4 조합 탐색")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--name", default=None)
        s.add_argument("--objective", default=None)
        s.add_argument("--constraint", action="extend", nargs="+", default=None)
        s.add_argument("--pid-split", default=str(PROJECT_ROOT / "bench" / "splits" / "prw_pids_seed42.json") + ":tune")
        s.add_argument("--validate", action="store_true")
        s.add_argument("--top", type=int, default=3)
        s.add_argument("--refine-trials", type=int, default=0)
        s.add_argument("--seed", type=int, default=42)
        s.add_argument("--params-from", default=None, help="optimize.py best.json (탐색된 params 를 격자 기본값으로)")
        s.add_argument("--adopt", action="store_true", help="추천을 프로젝트 루트 yaml 로 복사")
        s.add_argument("--combos-dir", default=str(DEFAULT_COMBOS_DIR))
        s.add_argument("--ledger", default=None)
        s.add_argument("--no-ledger", action="store_true")
        s.add_argument("--config", default=str(DEFAULT_PIPELINE))
        # 단계별 옵션 (GUI 공통)
        s.add_argument("--embedders", default="siglip2,irra,solider")
        s.add_argument("--reranks", default="none,solider,irra,siglip2")
        s.add_argument("--pools", default="200,1000")
        s.add_argument("--cache-dir", default=str(PROJECT_ROOT / "eval" / "results" / "cache"))
        s.add_argument("--vectors", default="siglip2,irra,solider")
        s.add_argument("--methods", default="leiden,dbscan_v6")
        s.add_argument("--target", default="person")
        s.add_argument("--sources", default="prw_image")
        s.add_argument("--max-points", type=int, default=0)
        s.add_argument("--min-cluster-size", type=int, default=2)
        s.add_argument("--matches-cache", default=str(PROJECT_ROOT / "eval" / "results" / "cache" / "prw_gt_matches.jsonl"))
        s.add_argument("--vector-cache", default="outputs/clustering/cache/bench")

    for stage in ("search", "cluster"):
        common(sub.add_parser(stage))
    s = sub.add_parser("report")
    s.add_argument("--stage", choices=["search", "cluster"], default=None)
    s.add_argument("--ledger", default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "search":
        return run_search(args)
    if args.cmd == "cluster":
        return run_cluster(args)
    return run_report(args)


if __name__ == "__main__":
    sys.exit(main())
