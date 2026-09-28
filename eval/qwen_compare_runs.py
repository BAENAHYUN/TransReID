#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""두 Qwen 판정 실행(qwen_verify_eval 의 qwen/ 캐시 폴더)을 후보 단위로 비교한다 — 라벨 없이도 되는 진단.

용도: 같은 후보를 다른 설정(예: --batch-size 1 vs 10, 2B vs 4B, max_pixels)으로 판정했을 때
  - 판정(verified PASS/FAIL/UNKNOWN) 일치율, PASS↔FAIL 뒤집힘 수
  - attr_score 평균 절대차
  - 최종 순위(final_score 순) 의 Spearman 상관 / Top-K 겹침
  - 후보당 시간
을 쿼리별·전체로 낸다. "배치가 결과를 바꾸는가" 를 라벨 전에 수치로 보는 것이 목적이고,
어느 쪽이 맞는지는 라벨(eval/gt/qwen/labels.json → qwen_verify_eval.py eval)이 정한다.

  python eval/qwen_compare_runs.py --a eval/results/qwen_verify/qwen_flag_norerank/qwen \\
                                   --b eval/results/qwen_verify/qwen_flag_norerank_b10/qwen [--out report.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def verdict(row: Dict[str, Any]) -> str:
    v = row.get("verified")
    if v is True:
        return "PASS"
    if v is False:
        return "FAIL"
    if row.get("attr_score") is not None or row.get("attr_skipped") or row.get("qwen_observation"):
        return "UNKNOWN"
    return "NA"          # Qwen 이 보지 않은 꼬리 후보


def cand_key(row: Dict[str, Any]) -> str:
    return str(row.get("cand_id") or row.get("point_id") or row.get("crop_path") or "")


def load_rows(qwen_dir: Path, qid: str) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    p = qwen_dir / f"{qid}.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    crops = d.get("crops") or []
    rows = list((crops[0].get("results") or [])) if crops else []
    meta = {"qwen_elapsed_sec": d.get("qwen_elapsed_sec"), "scored": d.get("qwen_scored_candidates"), "batch_size": d.get("qwen_batch_size", 1),
            "model": d.get("qwen_model")}
    return rows, meta


def spearman(a: List[float], b: List[float]) -> Optional[float]:
    n = len(a)
    if n < 2:
        return None
    def ranks(xs):
        order = sorted(range(n), key=lambda i: xs[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else None


def compare_query(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]], top_k: int = 10) -> Dict[str, Any]:
    A = {cand_key(r): r for r in rows_a}
    B = {cand_key(r): r for r in rows_b}
    common = [k for k in A if k in B and verdict(A[k]) != "NA" and verdict(B[k]) != "NA"]
    agree = sum(1 for k in common if verdict(A[k]) == verdict(B[k]))
    flips = sum(1 for k in common if {verdict(A[k]), verdict(B[k])} == {"PASS", "FAIL"})
    to_unknown = sum(1 for k in common if (verdict(A[k]) == "UNKNOWN") != (verdict(B[k]) == "UNKNOWN"))
    diffs = [abs(float(A[k]["attr_score"]) - float(B[k]["attr_score"])) for k in common
             if A[k].get("attr_score") is not None and B[k].get("attr_score") is not None]
    fa = [float(A[k].get("final_score") if A[k].get("final_score") is not None else -1) for k in common]
    fb = [float(B[k].get("final_score") if B[k].get("final_score") is not None else -1) for k in common]
    order_a = [k for k, _ in sorted(((k, A[k].get("rank", 1e9)) for k in common), key=lambda t: t[1])][:top_k]
    order_b = [k for k, _ in sorted(((k, B[k].get("rank", 1e9)) for k in common), key=lambda t: t[1])][:top_k]
    overlap = len(set(order_a) & set(order_b)) / max(1, min(top_k, len(common))) if common else None
    return {"common": len(common), "agree": agree, "agree_rate": (agree / len(common)) if common else None, "pass_fail_flips": flips,
            "unknown_changes": to_unknown, "attr_score_mad": (sum(diffs) / len(diffs)) if diffs else None,
            "final_spearman": spearman(fa, fb), f"top{top_k}_overlap": overlap,
            "verdicts_a": _counts(A, common), "verdicts_b": _counts(B, common)}


def _counts(M: Dict[str, Dict[str, Any]], keys: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k in keys:
        v = verdict(M[k])
        out[v] = out.get(v, 0) + 1
    return out


def compare_dirs(dir_a: Path, dir_b: Path, top_k: int = 10) -> Dict[str, Any]:
    qids = sorted({p.stem for p in dir_a.glob("q*.json") if not p.name.endswith(".meta.json")}
                  & {p.stem for p in dir_b.glob("q*.json") if not p.name.endswith(".meta.json")})
    per: List[Dict[str, Any]] = []
    tot_common = tot_agree = tot_flips = tot_unknown = 0
    sec_a: List[float] = []
    sec_b: List[float] = []
    meta_a = meta_b = {}
    for qid in qids:
        la, lb = load_rows(dir_a, qid), load_rows(dir_b, qid)
        if la is None or lb is None:
            continue
        rows_a, meta_a = la
        rows_b, meta_b = lb
        c = compare_query(rows_a, rows_b, top_k)
        c["query_id"] = qid
        if meta_a.get("qwen_elapsed_sec") and meta_a.get("scored"):
            sec_a.append(float(meta_a["qwen_elapsed_sec"]) / max(1, int(meta_a["scored"])))
        if meta_b.get("qwen_elapsed_sec") and meta_b.get("scored"):
            sec_b.append(float(meta_b["qwen_elapsed_sec"]) / max(1, int(meta_b["scored"])))
        per.append(c)
        tot_common += c["common"]
        tot_agree += c["agree"]
        tot_flips += c["pass_fail_flips"]
        tot_unknown += c["unknown_changes"]
    sp = [c["final_spearman"] for c in per if c["final_spearman"] is not None]
    ov = [c[f"top{top_k}_overlap"] for c in per if c[f"top{top_k}_overlap"] is not None]
    return {"a": str(dir_a), "b": str(dir_b), "queries": len(per), "top_k": top_k,
            "config_a": {k: meta_a.get(k) for k in ("batch_size", "model")}, "config_b": {k: meta_b.get(k) for k in ("batch_size", "model")},
            "summary": {"common": tot_common, "agree_rate": (tot_agree / tot_common) if tot_common else None, "pass_fail_flips": tot_flips,
                        "unknown_changes": tot_unknown, "final_spearman_mean": (sum(sp) / len(sp)) if sp else None,
                        f"top{top_k}_overlap_mean": (sum(ov) / len(ov)) if ov else None,
                        "sec_per_candidate_a": (sum(sec_a) / len(sec_a)) if sec_a else None,
                        "sec_per_candidate_b": (sum(sec_b) / len(sec_b)) if sec_b else None},
            "per_query": per}


def fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def report_md(out: Dict[str, Any]) -> str:
    s = out["summary"]
    k = out["top_k"]
    lines = [f"# Qwen 판정 실행 비교", "",
             f"- A: `{out['a']}` (batch {out['config_a'].get('batch_size')}, {out['config_a'].get('model')})",
             f"- B: `{out['b']}` (batch {out['config_b'].get('batch_size')}, {out['config_b'].get('model')})",
             f"- 쿼리 {out['queries']} · 공통 판정 후보 {s['common']}", "",
             "| 지표 | 값 |", "|---|---|",
             f"| 판정 일치율 (PASS/FAIL/UNKNOWN) | {fmt(s['agree_rate'])} |",
             f"| PASS↔FAIL 뒤집힘 | {s['pass_fail_flips']} |",
             f"| UNKNOWN 이 되거나 풀린 수 | {s['unknown_changes']} |",
             f"| 최종 순위 Spearman (쿼리 평균) | {fmt(s['final_spearman_mean'])} |",
             f"| Top-{k} 겹침 (쿼리 평균) | {fmt(s[f'top{k}_overlap_mean'])} |",
             f"| 후보당 초 A / B | {fmt(s['sec_per_candidate_a'], 2)} / {fmt(s['sec_per_candidate_b'], 2)} |", "",
             "라벨 없이 '같은 결과인가' 만 본다. 어느 쪽이 맞는지는 `qwen_verify_eval.py eval` 이 labels.json 으로 정한다.", "",
             "## 쿼리별", "", "| query | 공통 | 일치율 | 뒤집힘 | UNKNOWN 변화 | attr MAD | Spearman | Top-K 겹침 |", "|---|---|---|---|---|---|---|---|"]
    for c in out["per_query"]:
        lines.append(f"| {c['query_id']} | {c['common']} | {fmt(c['agree_rate'])} | {c['pass_fail_flips']} | {c['unknown_changes']} | {fmt(c['attr_score_mad'])} | {fmt(c['final_spearman'])} | {fmt(c[f'top{k}_overlap'])} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="기준 실행의 qwen/ 폴더 (예: 단건)")
    p.add_argument("--b", required=True, help="비교 실행의 qwen/ 폴더 (예: 배치)")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out", default=None, help="markdown 보고서 경로 (기본 <b 의 상위>/compare_<a 이름>.md)")
    args = p.parse_args(argv)
    out = compare_dirs(Path(args.a), Path(args.b), args.top_k)
    md = report_md(out)
    target = Path(args.out) if args.out else Path(args.b).parent / f"compare_vs_{Path(args.a).parent.name}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(md, encoding="utf-8")
    target.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(md)
    print(f"[compare] {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
