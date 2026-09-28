"""eval/qwen_verify_eval.py — Qwen 후처리 판정 시트(sheet) + Precision@K 변화·오탈락률 평가(eval)  (P6, 기준표 §6)

정답: 자연어 쿼리 N개(기본 30, eval/gt/qwen_queries.json) × 검색 상위 K(기본 20) 후보를 사람이 "맞다 / 아니다 / 모름" 으로 판정 (600건).
sheet : GUI 자연어 검색과 같은 경로(TextGeneralSearcher, person scope, 이미지 DB)로 후보를 뽑아 <gt-dir>/candidates/<qid>.json 에
        GUI 가 Qwen 에 넘기는 것과 같은 형식으로 저장하고, 판정 시트 <gt-dir>/sheet.html 을 만든다. 라벨 → labels.json (같은 폴더).
        후보의 불변 id = point_id(없으면 crop_path). 라벨은 (qid, 검색 순위) 로 저장되지만 평가는 불변 id 로 연결한다(재랭커가 순위를 바꿔도 안전).
eval  : 후보 파일마다 verifiers/qwen_stage.py 를 별도 프로세스로 실행 — 항상 flag 모드(모든 행의 판정을 보존)로 돌리고 filter 모드는 평가기가 재현한다
        (FAIL 제거 뒤 순위). 결과는 <out>/<name>/qwen/<qid>.json 에 캐시되며 캐시 계약 = 후보 파일 해시 + 모델·재랭커·top_k·dtype·max_pixels(<qid>.meta.json);
        계약이 다르면 다시 돌리고, alpha/threshold 만 다르면 저장된 관찰로 재채점(--rescore-only)한다. 시간(sec_per_candidate)은 언제나 원래 관찰 실행의 값.
        지표: P@5/10/20 검증 전(검색 순위) vs 후(Qwen 순위) — 같은 쿼리에서 둘 다 정의된 경우만 짝지어 평균(paired), "모름" 은 분자·분모 제외;
        false_drop_rate = FAIL 판정 중 실제 정답 비율(기준표 정의), lost_relevant_rate = Qwen 이 PASS/FAIL 로 판정한 정답 중 FAIL 비율;
        unknown_ratio = 관찰한 후보 중 UNKNOWN(판정 불능·처리 실패 포함). 라벨 없는 쿼리는 건너뛴다(--allow-unlabeled 면 시간·UNKNOWN 만). 원장 stage=qwen.
"""
from __future__ import annotations

import argparse
import hashlib
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
KIND = "qwen_labels"
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


def cand_id(row: Dict[str, Any]) -> str:
    """후보의 불변 id: point_id, 없으면 crop_path."""
    return str(row.get("point_id") or row.get("crop_path") or "")


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
        row["orig_rank"] = int(row.get("rank") or i)
        row["cand_id"] = cand_id(row)
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
<li>검색 순위나 점수는 참고하지 말고 그림만 보고 판정하세요. 같은 사람이 여러 번 나와도 각 후보를 따로 판정합니다. 판정하면 자동으로 검토 표시됩니다.</li>
<li>끝나면 <b>labels.json 내려받기</b> → 이 시트와 같은 폴더에 <code>labels.json</code>.</li>
</ol>
"""


def build_sheet(items: List[Dict[str, Any]], thumb_h: int, manifest: str) -> str:
    cards = []
    for it in items:
        qid = it["query_id"]
        cells = []
        for c in it["candidates"]:
            item = f"{qid}:{c['rank']}"
            uri = S.thumb_b64_from_file(c.get("crop_path"), height=thumb_h)
            radios = "".join(f'<label><input type="radio" name="{S.esc(item)}__relevant" data-item="{S.esc(item)}" data-field="relevant" value="{v}"> {lab}</label>'
                             for v, lab in (("yes", "맞다"), ("no", "아니다"), ("unsure", "모름")))
            cells.append(f"""<div class="cell" data-block="{S.esc(item)}">{S.img_tag(uri, f"{qid} #{c['rank']}")}<div class="meta">#{c['rank']}</div><div class="radios">{radios}</div>"""
                         f"""<input type="checkbox" data-item="{S.esc(item)}" data-field="reviewed" style="display:none"></div>""")
        cards.append(f"""
<div class="card"><h2>{S.esc(qid)} · {S.esc(it['text'])} <span class="meta">→ {S.esc(it.get('query_en'))} · 후보 {len(it['candidates'])}</span></h2>
<div class="grid">{''.join(cells)}</div></div>""")
    intro = INTRO + f'<div class="meta">쿼리 {len(items)} · 후보 {sum(len(i["candidates"]) for i in items)} · manifest {S.esc(manifest)}</div>'
    return S.html_document("Qwen 판정 — 자연어 검색 후보", intro, "".join(cards), kind=KIND, store_key="qwen",
                           export_name="labels.json", meta={"n_queries": len(items), "manifest": manifest})


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
            cands = [{"rank": int(r["orig_rank"]), "cand_id": r["cand_id"], "point_id": r.get("point_id"), "crop_path": r.get("crop_path"),
                      "score": float(r.get("score") or 0.0)} for r in rows]
            items.append({"query_id": q["query_id"], "text": q["text"], "query_en": payload["query_en"], "candidates": cands})
            print(f"[sheet] {q['query_id']} {q['text']} → {payload['query_en']} · 후보 {len(rows)} · {time.time() - t0:.1f}s")
    finally:
        searcher.release()
    manifest = S.manifest_of(KIND, [f"{i['query_id']}:{c['rank']}" for i in items for c in i["candidates"]],
                             {"cand_ids": {i["query_id"]: [c["cand_id"] for c in i["candidates"]] for i in items}})
    S.write_text(gt_dir / "sheet.html", build_sheet(items, args.thumb_height, manifest))
    S.write_json(gt_dir / "proposals.json", {"kind": KIND, "manifest": manifest, "generated_at": S.now_iso(), "config": str(args.config), "scope": args.scope,
                                             "top_k": args.top_k, "queries": items})
    n = sum(len(i["candidates"]) for i in items)
    print(f"[sheet] 쿼리 {len(items)} · 후보 {n} · manifest {manifest} → {gt_dir / 'sheet.html'}")
    print(f"RESULT_SUMMARY: {gt_dir / 'sheet.html'}")
    return {"queries": len(items), "candidates": n, "manifest": manifest}


# ---------------------------------------------------------------- Qwen 실행 · 캐시 계약
def cache_key(inp: Path, args: argparse.Namespace) -> str:
    parts = {"candidates_sha1": S.file_sha1(inp), "model_id": args.model_id, "no_reranker": bool(args.no_reranker),
             "reranker_model_id": args.reranker_model_id, "top_k": int(args.top_k), "dtype": args.dtype, "max_pixels": int(args.max_pixels),
             "device": args.device}
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def qwen_command(python: str, inp: Path, out: Path, args: argparse.Namespace) -> List[str]:
    """qwen_stage 실행 명령. 항상 flag 모드 — filter 는 평가기가 재현한다(FAIL 행의 판정 이력을 보존하기 위해)."""
    cmd = [python, "-u", str(QWEN_SCRIPT), "--in", str(inp), "--out", str(out), "--top-k", str(args.top_k), "--alpha", str(args.alpha),
           "--threshold", str(args.threshold), "--verify-mode", "flag", "--dtype", args.dtype, "--max-pixels", str(args.max_pixels),
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


def rescore_command(python: str, cached: Path, out: Path, args: argparse.Namespace) -> List[str]:
    return [python, "-u", str(QWEN_SCRIPT), "--in", str(cached), "--out", str(out), "--top-k", str(args.top_k), "--alpha", str(args.alpha),
            "--threshold", str(args.threshold), "--verify-mode", "flag", "--rescore-only", "--no-reranker", "--show", str(args.top_k)]


def _run(cmd: List[str], log_path: Path) -> float:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"qwen_stage 실패 (exit {proc.returncode}) — 로그 {log_path}")
    return time.time() - t0


def obtain_result(qid: str, inp: Path, qwen_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """캐시 계약을 확인하고 (재)실행 또는 재채점해 결과 payload 와 원래 실행의 시간 정보를 돌려준다."""
    out_json = qwen_dir / f"{qid}.json"
    meta_json = qwen_dir / f"{qid}.meta.json"
    key = cache_key(inp, args)
    meta = S.load_json(meta_json, {}) or {}
    reason = None
    if args.rerun:
        reason = "--rerun"
    elif not out_json.is_file():
        reason = "캐시 없음"
    elif meta.get("cache_key") != key:
        reason = f"캐시 계약 불일치 ({meta.get('cache_key')} ≠ {key}: 후보/모델/재랭커/top_k/dtype/max_pixels 중 하나가 다름)"
    if reason and args.rescore:
        raise SystemExit(f"{qid}: --rescore 는 계약이 맞는 캐시가 있어야 합니다 — {reason}")
    if reason:
        for old in qwen_dir.glob(f"{qid}.rescore_*"):          # 원본이 바뀌면 옛 재채점 결과는 무효
            old.unlink()
        wall = _run(qwen_command(args.python or sys.executable, inp, out_json, args), qwen_dir / f"{qid}.log")
        payload = S.load_json(out_json, {}) or {}
        meta = {"cache_key": key, "alpha": args.alpha, "threshold": args.threshold, "model_id": args.model_id or payload.get("qwen_model"),
                "reranker_used": bool(payload.get("reranker")), "qwen_elapsed_sec": payload.get("qwen_elapsed_sec"),
                "reranker_elapsed_sec": payload.get("reranker_elapsed_sec"), "scored": payload.get("qwen_scored_candidates"),
                "wall_sec": round(wall, 1), "ran_at": S.now_iso(), "reason": reason}
        S.write_json(meta_json, meta)
        payload["_cache"] = {"status": "ran", **meta}
        return payload
    payload = S.load_json(out_json, {}) or {}
    same_score = (meta.get("alpha") == args.alpha and meta.get("threshold") == args.threshold)
    if same_score and not args.rescore:
        payload["_cache"] = {"status": "cached", **meta}
        return payload
    # 재채점 파일은 원본 실행(계약 키 + 원본 결과 해시)에 결속 — 원본이 갱신되면 이름이 달라지고 옛 파일은 지워진다
    tag = hashlib.sha1(f"{key}:{S.file_sha1(out_json)}:{args.alpha}:{args.threshold}".encode()).hexdigest()[:10]
    rescored = qwen_dir / f"{qid}.rescore_{tag}.json"
    if not rescored.is_file() or args.rerun:
        _run(rescore_command(args.python or sys.executable, out_json, rescored, args), qwen_dir / f"{qid}.rescore_{tag}.log")
    payload = S.load_json(rescored, {}) or {}
    payload["_cache"] = {"status": "rescored", **meta, "rescore_alpha": args.alpha, "rescore_threshold": args.threshold}
    return payload


# ---------------------------------------------------------------- 채점
def precision_at(order: Sequence[Any], rel: Dict[Any, Optional[bool]], k: int) -> Optional[float]:
    """상위 k 중 판정된 것(모름·미라벨 제외)의 정답 비율. 판정된 것이 없으면 None."""
    judged = [rel.get(x) for x in list(order)[:k] if rel.get(x) is not None]
    if not judged:
        return None
    return sum(1 for j in judged if j) / len(judged)


def score_query(cands: Sequence[Dict[str, Any]], qwen_rows: Sequence[Dict[str, Any]], labels: Dict[str, Dict[str, Any]], qid: str,
                verify_mode: str = "flag", ks: Sequence[int] = KS) -> Dict[str, Any]:
    """검증 전 순위(cands: 검색 순위) vs 후 순위(qwen_rows: rank 순; filter 면 FAIL 제거) 의 P@K, 오탈락, UNKNOWN. 라벨 연결은 불변 cand_id."""
    rank_by_cid: Dict[str, int] = {}
    rel_by_cid: Dict[str, Optional[bool]] = {}
    for c in cands:
        cid = str(c.get("cand_id") or cand_id(c))
        rank_by_cid[cid] = int(c["rank"])
        lab = labels.get(f"{qid}:{c['rank']}") or {}
        v = str(lab.get("relevant") or "").strip().lower() if S.is_reviewed(lab) else ""
        rel_by_cid[cid] = True if v == "yes" else (False if v == "no" else None)
    before = [str(c.get("cand_id") or cand_id(c)) for c in sorted(cands, key=lambda c: int(c["rank"]))]
    rows = [r for r in qwen_rows if cand_id(r) in rank_by_cid]
    unmatched = len(qwen_rows) - len(rows)
    ordered = sorted(rows, key=lambda r: int(r.get("rank") or 10 ** 9))
    if verify_mode == "filter":
        ordered = [r for r in ordered if r.get("verified") is not False]
    after = [cand_id(r) for r in ordered]
    out: Dict[str, Any] = {"judged": sum(1 for v in rel_by_cid.values() if v is not None), "relevant": sum(1 for v in rel_by_cid.values() if v),
                           "unsure_or_unlabeled": sum(1 for v in rel_by_cid.values() if v is None), "unmatched_rows": unmatched}
    for k in ks:
        out[f"p{k}_before"] = precision_at(before, rel_by_cid, k)
        out[f"p{k}_after"] = precision_at(after, rel_by_cid, k)
    scored = [r for r in rows if "attr_score" in r]                      # Qwen 이 실제로 관찰한 행 (attr_score None = UNKNOWN 또는 처리 실패)
    unknown = [r for r in scored if r.get("attr_score") is None]
    fails = [r for r in scored if r.get("verified") is False]
    passes = [r for r in scored if r.get("verified") is True]
    rel_judged = [r for r in scored if r.get("verified") is not None and rel_by_cid.get(cand_id(r)) is True]
    false_drops = [r for r in fails if rel_by_cid.get(cand_id(r)) is True]
    fails_labeled = [r for r in fails if rel_by_cid.get(cand_id(r)) is not None]
    out.update({"scored": len(scored), "unknown": len(unknown), "fail": len(fails), "pass": len(passes), "false_drops": len(false_drops),
                "fails_labeled": len(fails_labeled), "relevant_judged_by_qwen": len(rel_judged), "retained": len(after)})
    return out


def aggregate(per: Sequence[Dict[str, Any]], ks: Sequence[int] = KS) -> Dict[str, Any]:
    """쿼리별 점수를 모은다. P@K 전·후는 둘 다 정의된 쿼리에서만 짝지어 평균(paired)."""
    m: Dict[str, Any] = {}
    for k in ks:
        paired = [(q[f"p{k}_before"], q[f"p{k}_after"]) for q in per if q.get(f"p{k}_before") is not None and q.get(f"p{k}_after") is not None]
        m[f"p{k}_before"] = round(100 * sum(b for b, _ in paired) / len(paired), 2) if paired else None
        m[f"p{k}_after"] = round(100 * sum(a for _, a in paired) / len(paired), 2) if paired else None
        m[f"p{k}_gain_pp"] = round(100 * sum(a - b for b, a in paired) / len(paired), 2) if paired else None
        m[f"p{k}_queries"] = len(paired)
    scored = sum(q["scored"] for q in per)
    fails_labeled = sum(q["fails_labeled"] for q in per)
    rj = sum(q["relevant_judged_by_qwen"] for q in per)
    fd = sum(q["false_drops"] for q in per)
    m["false_drop_rate"] = round(fd / fails_labeled, 4) if fails_labeled else None        # FAIL 판정 중 실제 정답 (기준표 정의)
    m["lost_relevant_rate"] = round(fd / rj, 4) if rj else None                            # 판정된 정답 중 FAIL 로 잃은 비율
    m["unknown_ratio"] = round(sum(q["unknown"] for q in per) / scored, 4) if scored else None
    m["queries"] = len(per)
    m["candidates"] = scored
    m["judged"] = sum(q["judged"] for q in per)
    return m


def cmd_eval(args: argparse.Namespace) -> Dict[str, Any]:
    gt_dir = Path(args.gt_dir).resolve()
    proposals = S.load_json(gt_dir / "proposals.json", {}) or {}
    queries = proposals.get("queries") or []
    if not queries:
        raise SystemExit(f"proposals.json 이 없거나 비었습니다: {gt_dir} (sheet 먼저)")
    labels, lmeta = S.read_labels_meta(gt_dir / "labels.json")
    if labels:
        S.check_manifest(lmeta, proposals.get("manifest"), "qwen", args.ignore_manifest, kind=KIND)
    name = args.name or f"qwen_{args.verify_mode}{'_norerank' if args.no_reranker else ''}"
    out_dir = Path(args.output_dir).resolve() / name
    qwen_dir = Path(args.qwen_dir).resolve() if args.qwen_dir else out_dir / "qwen"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    per: List[Dict[str, Any]] = []
    total_qwen_sec = total_wall = 0.0
    total_scored = 0
    n_run = n_cached = n_rescored = 0
    reranker_used = None
    model_id = None
    inputs = []
    started = time.time()
    for q in queries[: args.max_queries or None]:
        qid = q["query_id"]
        reviewed = sum(1 for c in q["candidates"] if S.is_reviewed(labels.get(f"{qid}:{c['rank']}")))
        q_cov = reviewed / len(q["candidates"]) if q["candidates"] else 0.0
        has_labels = q_cov >= float(args.min_coverage) and reviewed > 0     # 쿼리 단위 검토율 기준 (부분 라벨 쿼리는 정식 평가에 안 들어감)
        if not has_labels:
            if reviewed > 0:
                print(f"[eval] {qid}: 검토 {reviewed}/{len(q['candidates'])} < 기준 {float(args.min_coverage):.0%} — 정식 평가에서 제외")
            if not args.allow_unlabeled:
                continue
        inp = gt_dir / "candidates" / f"{qid}.json"
        if not inp.is_file():
            print(f"[eval] {qid}: 후보 파일 없음 — 건너뜀")
            continue
        payload = obtain_result(qid, inp, qwen_dir, args)
        cache = payload.get("_cache") or {}
        n_run += cache.get("status") == "ran"
        n_cached += cache.get("status") == "cached"
        n_rescored += cache.get("status") == "rescored"
        crops = payload.get("crops") or [{}]
        rows = crops[0].get("results") or []
        sc = score_query(q["candidates"], rows, labels, qid, args.verify_mode)
        sc.update({"query_id": qid, "text": q.get("text"), "qwen_elapsed_sec": cache.get("qwen_elapsed_sec"), "reranker_elapsed_sec": cache.get("reranker_elapsed_sec"),
                   "wall_sec": cache.get("wall_sec"), "cache": cache.get("status"), "qwen_error": crops[0].get("qwen_error"), "labeled": has_labels,
                   "reviewed": reviewed, "coverage": round(q_cov, 4), "empty_after": sc["retained"] == 0})
        per.append(sc)
        total_qwen_sec += float(cache.get("qwen_elapsed_sec") or 0.0)
        total_wall += float(cache.get("wall_sec") or 0.0)
        total_scored += sc["scored"]
        reranker_used = bool(cache.get("reranker_used")) if reranker_used is None else reranker_used
        model_id = cache.get("model_id") or model_id
        inputs.append({"role": f"candidates:{qid}", "path": str(inp), "sha1": S.file_sha1(inp)})
        print(f"[eval] {qid}: P@10 {sc['p10_before']} → {sc['p10_after']} · FAIL {sc['fail']} (오탈락 {sc['false_drops']}) · UNKNOWN {sc['unknown']}/{sc['scored']} · "
              f"{cache.get('qwen_elapsed_sec')}s ({cache.get('status')})")
    if not per:
        raise SystemExit("평가할 쿼리가 없습니다 (labels.json 이 없으면 --allow-unlabeled)")
    labeled = [q for q in per if q["labeled"]]
    coverage = round(sum(q["coverage"] for q in labeled) / len(labeled), 4) if labeled else 0.0
    metrics = aggregate(labeled) if labeled else aggregate(per)
    metrics["queries_empty_after"] = sum(1 for q in (labeled or per) if q.get("empty_after"))
    metrics["sec_per_candidate"] = round(total_qwen_sec / total_scored, 2) if total_scored else None
    metrics["wall_sec_per_candidate"] = round(total_wall / total_scored, 2) if (total_scored and total_wall) else None
    metrics["coverage"] = coverage
    metrics["elapsed_sec"] = round(time.time() - started, 1)
    if (gt_dir / "labels.json").is_file():
        inputs.append({"role": "labels", "path": str(gt_dir / "labels.json"), "sha1": S.file_sha1(gt_dir / "labels.json")})
    out = {"producer": PRODUCER, "generated_at": S.now_iso(), "name": name, "labeled_queries": len(labeled), "unlabeled_only": not labeled,
           "config": {"gt_dir": str(gt_dir), "top_k": args.top_k, "alpha": args.alpha, "threshold": args.threshold, "verify_mode": args.verify_mode,
                      "no_reranker": args.no_reranker, "model_id": args.model_id or model_id, "reranker_model_id": args.reranker_model_id,
                      "reranker_used": reranker_used, "dtype": args.dtype, "max_pixels": args.max_pixels, "rescore": args.rescore,
                      "qwen_dir": str(qwen_dir), "qwen_runs": n_run, "qwen_cached": n_cached, "qwen_rescored": n_rescored},
           "gt": {"queries": len(per), "labeled_queries": len(labeled), "judged": metrics.get("judged"), "candidates": metrics.get("candidates"), "coverage": coverage,
                  "protocol": "자연어 쿼리 × 상위 K 후보 사람 판정(맞다/아니다/모름; 불변 후보 id 로 연결); 모름 제외; P@K 전·후는 paired 평균; filter 는 평가기가 재현"},
           "inputs": inputs, "metrics": metrics, "per_query": per}
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = S.write_json(out_dir / "qwen_verify_eval.json", out)
    S.write_text(out_dir / "report.md", report_md(out))
    print(f"\n[eval] {name}: 쿼리 {len(per)} (라벨 {len(labeled)}) · 후보 {total_scored} · Qwen 실행 {n_run} / 캐시 {n_cached} / 재채점 {n_rescored}")
    print(f"  P@10 {metrics.get('p10_before')} → {metrics.get('p10_after')} ({metrics.get('p10_gain_pp')} %p, 쿼리 {metrics.get('p10_queries')}) · 오탈락률(FAIL 중 정답) {metrics.get('false_drop_rate')} · "
          f"정답 손실률 {metrics.get('lost_relevant_rate')} · UNKNOWN {metrics.get('unknown_ratio')} · {metrics.get('sec_per_candidate')} s/후보")
    if labeled or args.record_pseudo:
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
             f"- 모델 {c.get('model_id')} · 재랭커 {'사용' if c.get('reranker_used') else '미사용'} · mode {c['verify_mode']} (평가기 재현) · alpha {c['alpha']} · threshold {c['threshold']} · K {c['top_k']}",
             f"- 쿼리 {out['gt']['queries']} (라벨 {out['gt']['labeled_queries']}) · 후보 {out['gt'].get('candidates')} · 판정 {out['gt'].get('judged')} · 검토율 {out['gt'].get('coverage')}"
             + (" — **라벨 없음: 시간·UNKNOWN 만**" if out["unlabeled_only"] else ""), "",
             "| 지표 | 검증 전 | 검증 후 | 변화(%p) | 쿼리 |", "|---|---|---|---|---|"]
    for k in KS:
        lines.append(f"| P@{k} | {m.get(f'p{k}_before')} | {m.get(f'p{k}_after')} | {m.get(f'p{k}_gain_pp')} | {m.get(f'p{k}_queries')} |")
    lines += ["", f"- 오탈락률(FAIL 판정 중 실제 정답): {m.get('false_drop_rate')} · 정답 손실률(판정된 정답 중 FAIL): {m.get('lost_relevant_rate')} · UNKNOWN 비율: {m.get('unknown_ratio')} · "
              f"후보당 {m.get('sec_per_candidate')} s (Qwen 관찰 실행) / {m.get('wall_sec_per_candidate')} s (프로세스 포함)", "",
              "## 쿼리별", "", "| 쿼리 | 정답/판정 | P@10 전 → 후 | FAIL(오탈락) | UNKNOWN | 초 | 캐시 |", "|---|---|---|---|---|---|---|"]
    for q in out["per_query"]:
        lines.append(f"| {q['query_id']} {q.get('text')} | {q['relevant']}/{q['judged']} | {q['p10_before']} → {q['p10_after']} | {q['fail']} ({q['false_drops']}) | "
                     f"{q['unknown']}/{q['scored']} | {q.get('qwen_elapsed_sec')} | {q.get('cache')} |")
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
    p.add_argument("--verify-mode", default="flag", choices=["flag", "filter"], help="평가기가 재현: filter 는 FAIL 행을 뺀 순위로 P@K 후 를 잰다")
    p.add_argument("--no-reranker", action="store_true", help="Qwen3-VL-Reranker 단계 생략")
    p.add_argument("--model-id", default=None, help="Instruct 모델 (기본 qwen_stage 의 DEFAULT_MODEL)")
    p.add_argument("--reranker-model-id", default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default=None)
    p.add_argument("--max-pixels", type=int, default=768 * 768)
    p.add_argument("--python", default=None, help="qwen_stage 를 돌릴 python (기본 현재 인터프리터)")
    p.add_argument("--qwen-dir", default=None, help="Qwen 결과 캐시 폴더 (기본 <output-dir>/<name>/qwen)")
    p.add_argument("--rerun", action="store_true", help="캐시된 Qwen 결과가 있어도 다시 실행")
    p.add_argument("--rescore", action="store_true", help="계약이 맞는 캐시로 alpha/threshold 만 재채점 (Qwen 호출 없음; 캐시가 없으면 오류)")
    p.add_argument("--allow-unlabeled", action="store_true", help="라벨 없는 쿼리도 실행 (시간·UNKNOWN 만)")
    p.add_argument("--min-coverage", type=float, default=1.0, help="쿼리의 검토율이 이 미만이면 정식 평가에서 제외")
    p.add_argument("--ignore-manifest", action="store_true")
    p.add_argument("--name", default=None)
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--ledger", default=None, help="실행 원장 JSONL (기본 bench/ledger.jsonl; '' 또는 none 이면 기록 안 함)")
    p.add_argument("--no-ledger", action="store_true")
    p.add_argument("--record-pseudo", action="store_true", help="라벨 없는(pseudo/unlabeled) 결과도 --ledger 에 기록 (러너의 조각 원장용; 기본 원장에는 들어가지 않음)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    p = build_parser()
    args = p.parse_args(argv)
    args.command_line = list(argv) if argv is not None else sys.argv[1:]
    return cmd_sheet(args) if args.cmd == "sheet" else cmd_eval(args)


if __name__ == "__main__":
    main()
