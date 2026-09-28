"""eval/qwen_verify_eval.py — Qwen 후처리 판정 시트(sheet) + Precision@K 변화·오탈락률 평가(eval)  (P6, 기준표 §6)

정답: 자연어 쿼리 N개(기본 30, eval/gt/qwen_queries.json) × 검색 상위 K(기본 20) 후보를 사람이 "맞다 / 아니다 / 모름" 으로 판정 (600건).
sheet : GUI 자연어 검색과 같은 경로(TextGeneralSearcher, person scope, 이미지 DB)로 후보를 뽑아 <gt-dir>/candidates/<qid>.json 에
        GUI 가 Qwen 에 넘기는 것과 같은 형식으로 저장하고, 판정 시트 <gt-dir>/sheet.html 을 만든다. 라벨 → labels.json (같은 폴더).
eval  : 후보 파일마다 verifiers/qwen_stage.py 를 별도 프로세스로 실행(GUI 의 [Qwen 검증 실행] 과 같은 인자) → 결과 <output-dir>/<name>/qwen/<qid>.json.
        지표: P@5/10/20 검증 전(검색 순위) vs 후(Qwen 순위; filter 모드는 FAIL 제거), 오탈락률(FAIL 중 정답), UNKNOWN 비율, 후보당 초.
        "모름" 은 분자·분모에서 뺀다. 라벨이 없는 쿼리는 건너뛴다(--allow-unlabeled 면 시간·UNKNOWN 만). 원장 stage=qwen.
        Qwen 결과는 캐시된다 — 같은 이름으로 다시 실행하면 재사용, --rerun 이면 다시 돈다. alpha/threshold 만 바꿔 보려면 --rescore 로 Qwen 호출 없이 재채점.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import gt_sheet as S  # noqa: E402

PRODUCER = "qwen_verify_eval"
DEFAULT_GT_DIR = ROOT / "eval" / "gt" / "qwen"
DEFAULT_QUERIES = ROOT / "eval" / "gt" / "qwen_queries.json"
DEFAULT_OUT = ROOT / "eval" / "results" / "qwen_verify"
QWEN_SCRIPT = ROOT / "verifiers" / "qwen_stage.py"
KS = (5, 10, 20)


# ---------------------------------------------------------------- 쿼리
def load_queries(path: Path) -> List[Dict[str, str]]:
    data = S.load_json(path)
    items = data.get("queries") if isinstance(data, dict) else data
    out = []
    for i, q in enumerate(items or [], 1):
        if isinstance(q, str):
            out.append({"query_id": f"q{i:02d}", "text": q})
        elif isinstance(q, dict) and q.get("text"):
            out.append({"query_id": str(q.get("query_id") or f"q{i:02d}"), "text": str(q["text"])})
    if not out:
        raise SystemExit(f"쿼리가 없습니다: {path}")
    return out


# ---------------------------------------------------------------- sheet
def candidate_payload(query: Dict[str, str], res: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """GUI 가 qwen_stage 에 넘기는 형식과 같게 (search_gui._qwen_payload 참조)."""
    for i, row in enumerate(rows, 1):
        resolved = S.resolve_path(row.get("crop_path"))
        if resolved is not None:
            row["crop_path"] = str(resolved)
            if isinstance(row.get("payload"), dict):
                row["payload"]["crop_path"] = str(resolved)
        row.setdefault("pre_qwen_rank", int(row.get("rank") or i))
        base = row.get("score")
        if base is None:
            base = row.get("retrieval_score", 0.0)
        row.setdefault("pre_qwen_score", float(base or 0.0))
        row.setdefault("qdrant_score", float(row.get("retrieval_score", base) or 0.0))
    original = str(res.get("query") or query["text"]).strip()
    en = str(res.get("query_en") or original).strip()
    return {"search_type": "text", "query": original, "query_en": en, "qwen": False, "query_id": query["query_id"],
            "pipeline": res.get("pipeline"), "vectors": res.get("vectors"),
            "crops": [{"crop_index": 1, "kind": "text", "query_text_original": original, "query_text": en, "scope": res.get("scope"),
                       "collection": res.get("collection"), "results": rows}]}


INTRO = """
<b>Qwen 판정 시트</b> — 자연어 쿼리마다 검색이 낸 상위 후보를 보여줍니다. <b>후보가 쿼리 설명에 맞는 사람인지</b>만 판정합니다.
<ol>
<li><b>맞다</b>: 설명의 핵심 조건(색·옷·소지품·성별 등)이 모두 보인다. <b>아니다</b>: 하나라도 어긋난다. <b>모름</b>: 잘려서·작아서 판단 불가 (평가에서 제외).</li>
<li>검색 순위나 점수는 참고하지 말고 그림만 보고 판정하세요. 같은 사람이 여러 번 나와도 각 후보를 따로 판정합니다.</li>
<li>끝나면 <b>labels.json 내려받기</b> → 이 시트와 같은 폴더에 <code>labels.json</code>.</li>
</ol>
"""


def build_sheet(items: List[Dict[str, Any]], thumb_h: int) -> str:
    cards = []
    for it in items:
        qid = it["query_id"]
        cells = []
        for c in it["candidates"]:
            item = f"{qid}:{c['rank']}"
            uri = S.thumb_b64_from_file(c.get("crop_path"), height=thumb_h)
            radios = "".join(f'<label><input type="radio" name="{S.esc(item)}__relevant" data-item="{S.esc(item)}" data-field="relevant" value="{v}"> {lab}</label>'
                             for v, lab in (("yes", "맞다"), ("no", "아니다"), ("unsure", "모름")))
            cells.append(f"""<div class="cell" data-block="{S.esc(item)}">{S.img_tag(uri, c.get('crop_path') or '')}<div class="meta">#{c['rank']} · {c.get('score', 0):.3f}</div><div class="radios">{radios}</div></div>""")
        cards.append(f"""
<div class="card"><h2>{S.esc(qid)} · {S.esc(it['text'])} <span class="meta">→ {S.esc(it.get('query_en'))} · 후보 {len(it['candidates'])}</span></h2>
<div class="grid">{''.join(cells)}</div></div>""")
    intro = INTRO + f'<div class="meta">쿼리 {len(items)} · 후보 {sum(len(i["candidates"]) for i in items)}</div>'
    return S.html_document("Qwen 판정 — 자연어 검색 후보", intro, "".join(cards), kind="qwen_labels", store_key="qwen:sheet",
                           export_name="labels.json", meta={"n_queries": len(items)}, extra_js="window.doneRule = v => !!v.relevant;")


def cmd_sheet(args: argparse.Namespace) -> Dict[str, Any]:
    gt_dir = Path(args.gt_dir).resolve()
    cand_dir = gt_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    queries = load_queries(Path(args.queries))
    if args.max_queries:
        queries = queries[:args.max_queries]
    from search.unified_search_4mode import TextGeneralSearcher, hit_to_dict
    searcher = TextGeneralSearcher(str(args.config), translate_backend=args.translate_backend, translate_model_id=None, expand=False)
    items = []
    try:
        for q in queries:
            t0 = time.time()
            res = searcher.search(q["text"], scope=args.scope, limit=args.top_k, translate=not args.no_translate,
                                  vectors=[v for v in (args.vectors or []) if v] or None)
            rows = [hit_to_dict(h) for h in res.pop("hits")][:args.top_k]
            payload = candidate_payload(q, res, rows)
            S.write_json(cand_dir / f"{q['query_id']}.json", payload)
            cands = [{"rank": int(r["pre_qwen_rank"]), "point_id": r.get("point_id"), "crop_path": r.get("crop_path"), "score": float(r.get("score") or 0.0)} for r in rows]
            items.append({"query_id": q["query_id"], "text": q["text"], "query_en": payload["query_en"], "candidates": cands})
            print(f"[sheet] {q['query_id']} {q['text']} → {payload['query_en']} · 후보 {len(rows)} · {time.time() - t0:.1f}s")
    finally:
        searcher.release()
    S.write_text(gt_dir / "sheet.html", build_sheet(items, args.thumb_height))
    S.write_json(gt_dir / "proposals.json", {"generated_at": S.now_iso(), "config": str(args.config), "scope": args.scope, "top_k": args.top_k,
                                             "queries": items})
    n = sum(len(i["candidates"]) for i in items)
    print(f"[sheet] 쿼리 {len(items)} · 후보 {n} → {gt_dir / 'sheet.html'}")
    print(f"RESULT_SUMMARY: {gt_dir / 'sheet.html'}")
    return {"queries": len(items), "candidates": n}


# ---------------------------------------------------------------- eval
def qwen_command(python: str, inp: Path, out: Path, args: argparse.Namespace) -> List[str]:
    cmd = [python, "-u", str(QWEN_SCRIPT), "--in", str(inp), "--out", str(out), "--top-k", str(args.top_k), "--alpha", str(args.alpha),
           "--threshold", str(args.threshold), "--verify-mode", args.verify_mode, "--dtype", args.dtype, "--max-pixels", str(args.max_pixels),
           "--show", str(args.top_k)]
    if args.no_reranker:
        cmd.append("--no-reranker")
    if args.model_id:
        cmd += ["--model-id", args.model_id]
    if args.reranker_model_id:
        cmd += ["--reranker-model-id", args.reranker_model_id]
    if args.device:
        cmd += ["--device", args.device]
    return cmd


def run_qwen(inp: Path, out: Path, args: argparse.Namespace, log_path: Path) -> float:
    cmd = qwen_command(args.python or sys.executable, inp, out, args)
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(f"qwen_stage 실패 (exit {proc.returncode}) — 로그 {log_path}")
    return time.time() - t0


def rescore(qwen_json: Path, out: Path, args: argparse.Namespace, log_path: Path) -> None:
    """저장된 관찰로 alpha/threshold 만 다시 채점 (Qwen 호출 0회)."""
    cmd = [args.python or sys.executable, "-u", str(QWEN_SCRIPT), "--in", str(qwen_json), "--out", str(out), "--top-k", str(args.top_k),
           "--alpha", str(args.alpha), "--threshold", str(args.threshold), "--verify-mode", args.verify_mode, "--rescore-only", "--no-reranker",
           "--show", str(args.top_k)]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(f"qwen_stage --rescore-only 실패 (exit {proc.returncode}) — 로그 {log_path}")


def precision_at(order: Sequence[Any], rel: Dict[Any, Optional[bool]], k: int) -> Optional[float]:
    """상위 k 중 판정된 것(모름 제외)의 정답 비율. 판정된 것이 없으면 None."""
    judged = [rel.get(x) for x in list(order)[:k] if rel.get(x) is not None]
    if not judged:
        return None
    return sum(1 for j in judged if j) / len(judged)


def score_query(cands: Sequence[Dict[str, Any]], qwen_rows: Sequence[Dict[str, Any]], labels: Dict[str, Dict[str, Any]], qid: str,
                ks: Sequence[int] = KS) -> Dict[str, Any]:
    """검증 전 순위(cands: rank 순) vs 후 순위(qwen_rows: rank 순) 의 P@K, 오탈락, UNKNOWN."""
    rel: Dict[int, Optional[bool]] = {}
    for c in cands:
        v = str((labels.get(f"{qid}:{c['rank']}") or {}).get("relevant") or "").strip().lower()
        rel[int(c["rank"])] = True if v == "yes" else (False if v == "no" else None)
    before = [int(c["rank"]) for c in sorted(cands, key=lambda c: int(c["rank"]))]
    after_rows = sorted(qwen_rows, key=lambda r: int(r.get("rank") or 10 ** 9))
    after = [int(r.get("pre_qwen_rank") or 0) for r in after_rows]
    out: Dict[str, Any] = {"judged": sum(1 for v in rel.values() if v is not None), "relevant": sum(1 for v in rel.values() if v),
                           "unsure": sum(1 for v in rel.values() if v is None)}
    for k in ks:
        out[f"p{k}_before"] = precision_at(before, rel, k)
        out[f"p{k}_after"] = precision_at(after, rel, k)
    # Qwen 이 실제로 관찰한 행만 채점 대상 (attr_score 키가 있음; 값 None = UNKNOWN). apply_verdict 는 관찰하지 않은
    # 꼬리 행에도 verified=None 을 붙이므로 verified 키로 세면 UNKNOWN 이 부풀려진다.
    scored = [r for r in qwen_rows if "attr_score" in r]
    unknown = [r for r in scored if r.get("attr_score") is None]
    fails = [r for r in scored if r.get("verified") is False]
    rel_judged = [r for r in scored if r.get("verified") is not None and rel.get(int(r.get("pre_qwen_rank") or 0)) is True]
    false_drops = [r for r in fails if rel.get(int(r.get("pre_qwen_rank") or 0)) is True]
    out.update({"scored": len(scored), "unknown": len(unknown), "fail": len(fails), "pass": sum(1 for r in scored if r.get("verified") is True),
                "false_drops": len(false_drops), "relevant_judged_by_qwen": len(rel_judged),
                "retained": len(after_rows)})
    return out


def aggregate(per: Sequence[Dict[str, Any]], ks: Sequence[int] = KS) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    for k in ks:
        for side in ("before", "after"):
            vals = [q[f"p{k}_{side}"] for q in per if q.get(f"p{k}_{side}") is not None]
            m[f"p{k}_{side}"] = round(100 * sum(vals) / len(vals), 2) if vals else None
        if m[f"p{k}_before"] is not None and m[f"p{k}_after"] is not None:
            m[f"p{k}_gain_pp"] = round(m[f"p{k}_after"] - m[f"p{k}_before"], 2)
        else:
            m[f"p{k}_gain_pp"] = None
    scored = sum(q["scored"] for q in per)
    rj = sum(q["relevant_judged_by_qwen"] for q in per)
    m["false_drop_rate"] = round(sum(q["false_drops"] for q in per) / rj, 4) if rj else None
    m["unknown_ratio"] = round(sum(q["unknown"] for q in per) / scored, 4) if scored else None
    m["queries"] = len(per)
    m["candidates"] = sum(q["scored"] for q in per)
    m["judged"] = sum(q["judged"] for q in per)
    return m


def cmd_eval(args: argparse.Namespace) -> Dict[str, Any]:
    gt_dir = Path(args.gt_dir).resolve()
    proposals = S.load_json(gt_dir / "proposals.json", {}) or {}
    queries = proposals.get("queries") or []
    if not queries:
        raise SystemExit(f"proposals.json 이 없거나 비었습니다: {gt_dir} (sheet 먼저)")
    labels = S.read_labels(gt_dir / "labels.json")
    name = args.name or f"qwen_{args.verify_mode}{'_norerank' if args.no_reranker else ''}"
    out_dir = Path(args.output_dir).resolve() / name
    qwen_dir = Path(args.qwen_dir).resolve() if args.qwen_dir else out_dir / "qwen"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    per: List[Dict[str, Any]] = []
    total_qwen_sec = total_wall = 0.0
    total_scored = 0
    n_run = n_cached = 0
    reranker_used = None
    model_id = None
    started = time.time()
    for q in queries[: args.max_queries or None]:
        qid = q["query_id"]
        has_labels = any(k.startswith(f"{qid}:") for k in labels)
        if not has_labels and not args.allow_unlabeled:
            continue
        inp = gt_dir / "candidates" / f"{qid}.json"
        if not inp.is_file():
            print(f"[eval] {qid}: 후보 파일 없음 — 건너뜀")
            continue
        out_json = qwen_dir / f"{qid}.json"
        wall = 0.0
        if args.rescore and out_json.is_file():
            rescored = qwen_dir / f"{qid}.rescore.json"
            rescore(out_json, rescored, args, qwen_dir / f"{qid}.rescore.log")
            payload = S.load_json(rescored, {})
        else:
            if args.rerun or not out_json.is_file():
                wall = run_qwen(inp, out_json, args, qwen_dir / f"{qid}.log")
                n_run += 1
            else:
                n_cached += 1
            payload = S.load_json(out_json, {})
        crops = payload.get("crops") or [{}]
        rows = crops[0].get("results") or []
        sc = score_query(q["candidates"], rows, labels, qid)
        sc.update({"query_id": qid, "text": q.get("text"), "qwen_elapsed_sec": payload.get("qwen_elapsed_sec"), "wall_sec": round(wall, 1),
                   "qwen_error": crops[0].get("qwen_error"), "labeled": has_labels})
        per.append(sc)
        total_qwen_sec += float(payload.get("qwen_elapsed_sec") or 0.0)
        total_wall += wall
        total_scored += sc["scored"]
        reranker_used = bool(payload.get("reranker")) if reranker_used is None else reranker_used
        model_id = payload.get("qwen_model") or model_id
        print(f"[eval] {qid}: P@10 {sc['p10_before']} → {sc['p10_after']} · FAIL {sc['fail']} (오탈락 {sc['false_drops']}) · UNKNOWN {sc['unknown']}/{sc['scored']} · {payload.get('qwen_elapsed_sec')}s")
    if not per:
        raise SystemExit("평가할 쿼리가 없습니다 (labels.json 이 없으면 --allow-unlabeled)")
    labeled = [q for q in per if q["labeled"]]
    metrics = aggregate(labeled) if labeled else aggregate(per)
    metrics["sec_per_candidate"] = round(total_qwen_sec / total_scored, 2) if total_scored else None
    metrics["wall_sec_per_candidate"] = round(total_wall / total_scored, 2) if (total_scored and total_wall) else None
    metrics["elapsed_sec"] = round(time.time() - started, 1)
    out = {"producer": PRODUCER, "generated_at": S.now_iso(), "name": name, "labeled_queries": len(labeled), "unlabeled_only": not labeled,
           "config": {"gt_dir": str(gt_dir), "top_k": args.top_k, "alpha": args.alpha, "threshold": args.threshold, "verify_mode": args.verify_mode,
                      "no_reranker": args.no_reranker, "model_id": args.model_id or model_id, "reranker_used": reranker_used, "rescore": args.rescore,
                      "qwen_dir": str(qwen_dir), "qwen_runs": n_run, "qwen_cached": n_cached},
           "gt": {"queries": len(per), "labeled_queries": len(labeled), "judged": metrics.get("judged"), "candidates": metrics.get("candidates"),
                  "protocol": "자연어 쿼리 × 상위 K 후보 사람 판정(맞다/아니다/모름); 모름 제외; P@K = 판정된 상위 K 중 정답 비율"},
           "metrics": metrics, "per_query": per}
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = S.write_json(out_dir / "qwen_verify_eval.json", out)
    S.write_text(out_dir / "report.md", report_md(out))
    print(f"\n[eval] {name}: 쿼리 {len(per)} (라벨 {len(labeled)}) · 후보 {total_scored} · Qwen 실행 {n_run} / 캐시 {n_cached}")
    print(f"  P@10 {metrics.get('p10_before')} → {metrics.get('p10_after')} ({metrics.get('p10_gain_pp')} %p) · 오탈락률 {metrics.get('false_drop_rate')} · "
          f"UNKNOWN {metrics.get('unknown_ratio')} · {metrics.get('sec_per_candidate')} s/후보")
    if labeled:
        from bench import ledger
        ledger.record(lambda: [ledger.entry_from_qwen_result(out, report=json_path, command=args.command_line, versions=ledger.versions_info())],
                      args.ledger, args.no_ledger)
    else:
        print("  (라벨 없음 — 시간·UNKNOWN 만 측정, 원장 기록 안 함)")
    print(f"RESULT_SUMMARY: {json_path}")
    return out


def report_md(out: Dict[str, Any]) -> str:
    m = out["metrics"]
    c = out["config"]
    lines = [f"# Qwen 후처리 평가 — {out['name']} ({out['generated_at']})", "",
             f"- 모델 {c.get('model_id')} · 재랭커 {'사용' if c.get('reranker_used') else '미사용'} · mode {c['verify_mode']} · alpha {c['alpha']} · threshold {c['threshold']} · K {c['top_k']}",
             f"- 쿼리 {out['gt']['queries']} (라벨 {out['gt']['labeled_queries']}) · 후보 {out['gt'].get('candidates')} · 판정 {out['gt'].get('judged')}"
             + (" — **라벨 없음: 시간·UNKNOWN 만**" if out["unlabeled_only"] else ""), "",
             "| 지표 | 검증 전 | 검증 후 | 변화(%p) |", "|---|---|---|---|"]
    for k in KS:
        lines.append(f"| P@{k} | {m.get(f'p{k}_before')} | {m.get(f'p{k}_after')} | {m.get(f'p{k}_gain_pp')} |")
    lines += ["", f"- 오탈락률(FAIL 중 정답): {m.get('false_drop_rate')} · UNKNOWN 비율: {m.get('unknown_ratio')} · 후보당 {m.get('sec_per_candidate')} s (Qwen 자체) / {m.get('wall_sec_per_candidate')} s (프로세스 포함)", "",
              "## 쿼리별", "", "| 쿼리 | 정답/판정 | P@10 전 → 후 | FAIL(오탈락) | UNKNOWN | 초 |", "|---|---|---|---|---|---|"]
    for q in out["per_query"]:
        lines.append(f"| {q['query_id']} {q.get('text')} | {q['relevant']}/{q['judged']} | {q['p10_before']} → {q['p10_after']} | {q['fail']} ({q['false_drops']}) | {q['unknown']}/{q['scored']} | {q.get('qwen_elapsed_sec')} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen 판정 시트 + P@K 변화 평가 (P6)")
    p.add_argument("cmd", choices=["sheet", "eval"])
    p.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR))
    p.add_argument("--queries", default=str(DEFAULT_QUERIES), help="sheet: 쿼리 json ({queries:[{query_id,text}]} 또는 문자열 목록)")
    p.add_argument("--config", default=str(ROOT / "pipeline.yaml"))
    p.add_argument("--scope", default="person", choices=["person", "object"])
    p.add_argument("--vectors", nargs="*", default=None, help="sheet: 텍스트 벡터 부분집합 (기본 전부 RRF)")
    p.add_argument("--translate-backend", default="opus", choices=["opus", "nllb", "none"])
    p.add_argument("--no-translate", action="store_true")
    p.add_argument("--top-k", type=int, default=20, help="sheet: 후보 수 / eval: Qwen 이 볼 상위 후보 수")
    p.add_argument("--max-queries", type=int, default=0, help="0 = 전부")
    p.add_argument("--thumb-height", type=int, default=150)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--verify-mode", default="flag", choices=["flag", "filter"])
    p.add_argument("--no-reranker", action="store_true", help="Qwen3-VL-Reranker 단계 생략")
    p.add_argument("--model-id", default=None, help="Instruct 모델 (기본 qwen_stage 의 DEFAULT_MODEL)")
    p.add_argument("--reranker-model-id", default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default=None)
    p.add_argument("--max-pixels", type=int, default=768 * 768)
    p.add_argument("--python", default=None, help="qwen_stage 를 돌릴 python (기본 현재 인터프리터)")
    p.add_argument("--qwen-dir", default=None, help="Qwen 결과 캐시 폴더 (기본 <output-dir>/<name>/qwen)")
    p.add_argument("--rerun", action="store_true", help="캐시된 Qwen 결과가 있어도 다시 실행")
    p.add_argument("--rescore", action="store_true", help="캐시된 관찰로 alpha/threshold 만 재채점 (Qwen 호출 없음)")
    p.add_argument("--allow-unlabeled", action="store_true", help="라벨 없는 쿼리도 실행 (시간·UNKNOWN 만)")
    p.add_argument("--name", default=None)
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--ledger", default=None, help="실행 원장 JSONL (기본 bench/ledger.jsonl; '' 또는 none 이면 기록 안 함)")
    p.add_argument("--no-ledger", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    p = build_parser()
    args = p.parse_args(argv)
    args.command_line = list(argv) if argv is not None else sys.argv[1:]
    return cmd_sheet(args) if args.cmd == "sheet" else cmd_eval(args)


if __name__ == "__main__":
    main()
