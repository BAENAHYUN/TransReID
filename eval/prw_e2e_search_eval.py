"""eval/prw_e2e_search_eval.py — 전체 파이프라인(검출 → crop → 임베딩 → Qdrant → 검색) 검색 정확도, PRW GT.

단독 평가(prw_eval.py / prw_eval_unified.py 는 GT crop 을 gallery 로 씀)와 달리, 운영 DB 에 실제로 들어간
검출 crop 을 대상으로 GUI/CLI 와 같은 검색 경로(CropGeneralSearcher: 1차 조합 → 2차 재정렬)를
PRW query_box(2,057장)로 돌려 채점한다. 검출이 놓친 사람은 그대로 손실로 잡힌다.

정답: point → pid 매칭 캐시 (eval/prw_cluster_gt_eval.py 가 만드는 eval/results/cache/prw_gt_matches.jsonl, IoU ≥ 0.5).
      없으면 같은 방식으로 만든다.
프로토콜(--gallery test, 기본): 순위 목록에서 test 프레임의 PRW point 만 남기고(다른 source·train 프레임은 제외 →
      distractor_ratio), 쿼리와 같은 프레임의 같은 인물은 junk 로 뺀다 (prw_eval 과 동일).
지표: map      = AP@K 평균, 분모 = 다른 test 프레임의 GT 박스 수  → 검출 손실 포함 (전체 파이프라인)
      map_db   = 같은 AP, 분모 = DB 에 들어간 그 인물 point 수     → 검색만 (실제 crop 위에서)
      rank1/5/10, recall_at_k (K = --limit), det_ceiling = Σ min(DB 양성, GT 양성) / Σ GT 양성 (검출 상한),
      distractor_ratio, sec_per_query.
산출물: <output-dir>/<name>/e2e_search.json (+ summary.csv 행), RESULT_SUMMARY 마커, 원장(bench/ledger.jsonl, stage=e2e).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from report_common import CONFIG_HELP, DEFAULT_CONFIG_PATH, load_pipeline_settings, write_json  # noqa: E402
from eval import prw_eval  # noqa: E402

PRODUCER = "prw_e2e_search_eval"
DEFAULT_MATCHES = PROJECT_ROOT / "eval" / "results" / "cache" / "prw_gt_matches.jsonl"


# ---------------------------------------------------------------- 입력
def load_matches(path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """prw_gt_matches.jsonl → (meta, {point_id: {frame, pid, status, iou}})."""
    meta: Dict[str, Any] = {}
    matches: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("_meta"):
                meta = d
                continue
            matches[str(d.pop("point_id"))] = d
    return meta, matches


def read_matches_header(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        first = f.readline().strip()
    try:
        d = json.loads(first) if first else {}
    except json.JSONDecodeError:
        return {}
    return d if isinstance(d, dict) and d.get("_meta") else {}


def matches_compatible(meta: Dict[str, Any], iou: float, sources: Sequence[str], collection: str) -> bool:
    """캐시 헤더의 iou / sources / collection 이 지금 설정과 같아야 재사용 (다른 DB·다른 IoU 의 매칭은 분모와 순위를 왜곡)."""
    return bool(meta) and float(meta.get("iou", -1)) == float(iou) and list(meta.get("sources") or []) == list(sources) \
        and meta.get("collection") == collection


def ensure_matches(path: Path, config_path: str, sources: Sequence[str], iou: float, ann_dir: Path, log=print) -> Path:
    """캐시가 없거나 설정(iou/sources/collection)이 다르면 prw_cluster_gt_eval 과 같은 방식으로 다시 만든다 (Qdrant scroll → GT IoU 매칭)."""
    settings = load_pipeline_settings(config_path)
    if path.is_file():
        if matches_compatible(read_matches_header(path), iou, sources, settings.person_collection):
            return path
        log(f"[gt] 매칭 캐시 설정 불일치 (iou/sources/collection) → 다시 생성: {path}")
    from eval import prw_cluster_gt_eval as G
    log(f"[gt] Qdrant {settings.person_collection} / source={list(sources)} 에서 point→pid 매칭 생성")
    points = G.scroll_points(settings.qdrant_url, None, settings.person_collection, list(sources))
    matches, stats = G.match_all(points, ann_dir, iou, log=log)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(dict(_meta=True, iou=iou, sources=list(sources), collection=settings.person_collection,
                                stats=stats), ensure_ascii=False) + "\n")
        for pk, d in matches.items():
            f.write(json.dumps({"point_id": pk, **d}, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def load_queries(data_root: Path) -> List[Dict[str, Any]]:
    """query_info.txt → [{pid, frame, path}] (파일이 있는 것만, 파일 순서 유지)."""
    qbox = data_root / "query_box"
    out = []
    with (data_root / "query_info.txt").open() as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            pid, frame = int(parts[0]), parts[5]
            p = qbox / f"{pid}_{frame}.jpg"
            if p.is_file():
                out.append({"pid": pid, "frame": frame, "path": str(p)})
    return out


def select_queries(queries: List[Dict[str, Any]], max_queries: int = 0, pids: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
    """pid 분할로 거르고, max_queries 면 전체에 고르게 퍼진 부분집합 (결정적)."""
    qs = [q for q in queries if pids is None or q["pid"] in pids]
    if max_queries and 0 < max_queries < len(qs):
        idx = np.unique(np.linspace(0, len(qs) - 1, max_queries).round().astype(int))
        qs = [qs[i] for i in idx]
    return qs


def gt_positives(ann_dir: Path, frames: Iterable[str]) -> Dict[int, Counter]:
    """프레임별 주석 → {pid: Counter(frame → GT 박스 수)} (pid < 0 = 미표기 제외)."""
    out: Dict[int, Counter] = defaultdict(Counter)
    for frame in frames:
        arr = prw_eval.load_annotation(ann_dir / f"{frame}.jpg.mat")
        for row in arr:
            pid = int(row[0])
            if pid >= 0:
                out[pid][frame] += 1
    return out


def db_positives(matches: Dict[str, Dict[str, Any]], frame_set: Optional[Set[str]]) -> Dict[int, Counter]:
    out: Dict[int, Counter] = defaultdict(Counter)
    for m in matches.values():
        if m.get("status") != "labeled":
            continue
        if frame_set is not None and m.get("frame") not in frame_set:
            continue
        out[int(m["pid"])][m["frame"]] += 1
    return out


def positives_excluding(counter: Counter, frame: str) -> int:
    return int(sum(n for f, n in counter.items() if f != frame))


# ---------------------------------------------------------------- 채점 (순수 함수)
def _ap(rel: Sequence[bool], n_pos: int) -> Tuple[float, Optional[int], int]:
    hits, ap_sum, first = 0, 0.0, None
    for r, ok in enumerate(rel, 1):
        if ok:
            hits += 1
            ap_sum += hits / r
            if first is None:
                first = r
    return (min(1.0, ap_sum / n_pos) if n_pos else 0.0), first, hits


def score_query(ranked_ids: Sequence[str], matches: Dict[str, Dict[str, Any]], qpid: int, qframe: str,
                frame_set: Optional[Set[str]], n_pos_gt: int, n_pos_db: int,
                gt_frames: Optional[Counter] = None) -> Dict[str, Any]:
    """순위 목록 하나를 채점.
    frame_set 이 있으면(gallery=test) 그 밖의 point 는 제외(distractor), 없으면 비관련으로 남긴다.
    gt_frames(그 인물의 프레임별 GT 박스 수)가 있으면 같은 GT 의 중복 검출은 첫 것만 TP — 그 뒤는 비관련(순위를 차지한 만큼 벌점).
    ap/recall(분모 GT 박스) 은 중복 상한을 적용, ap_db(분모 DB point) 는 point 마다 양성 (검색만의 성능)."""
    rel_gt: List[bool] = []      # GT 기준 (중복 상한)
    rel_db: List[bool] = []      # DB point 기준 (중복도 양성)
    excluded = duplicates = 0
    credit: Dict[str, int] = {}
    for pid_ in ranked_ids:
        m = matches.get(str(pid_))
        if m is None or (frame_set is not None and m.get("frame") not in frame_set):
            if frame_set is not None:
                excluded += 1
                continue
            rel_gt.append(False)
            rel_db.append(False)
            continue
        is_pos = m.get("status") == "labeled" and int(m["pid"]) == qpid
        if is_pos and m.get("frame") == qframe:      # junk: 쿼리 자신 (같은 프레임 같은 인물)
            continue
        rel_db.append(is_pos)
        if is_pos and gt_frames is not None:
            frame = str(m.get("frame"))
            if credit.get(frame, 0) >= int(gt_frames.get(frame, 0)):
                duplicates += 1
                rel_gt.append(False)
                continue
            credit[frame] = credit.get(frame, 0) + 1
        rel_gt.append(is_pos)
    ap, first, hits = _ap(rel_gt, n_pos_gt)
    ap_db, _, hits_db = _ap(rel_db, n_pos_db)
    recall = min(1.0, hits / n_pos_gt) if n_pos_gt else 0.0
    return dict(ap=ap, ap_db=ap_db, first_rank=first, hits=hits, hits_db=hits_db, duplicates=duplicates,
                considered=len(rel_gt), excluded=excluded, recall=recall)


def capped_positives(db_frames: Counter, gt_frames: Counter, qframe: str) -> int:
    """검출이 DB 에 넣은 GT 수: 프레임마다 min(DB point 수, GT 박스 수) — 중복 검출을 한 번만 센다."""
    return int(sum(min(int(db_frames.get(f, 0)), int(n)) for f, n in gt_frames.items() if f != qframe))


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"valid_queries": 0}

    def rank_at(k: int) -> float:
        return round(100.0 * sum(1 for r in rows if r["first_rank"] is not None and r["first_rank"] <= k) / n, 4)

    gt_total = sum(r["n_pos_gt"] for r in rows)
    db_capped = sum(r.get("n_pos_db_capped", min(r["n_pos_db"], r["n_pos_gt"])) for r in rows)
    seen = sum(r["considered"] + r["excluded"] for r in rows)
    with_db = [r for r in rows if r["n_pos_db"] > 0]      # map_db = 검출된 인물만의 검색 성능
    return {
        "map": round(100.0 * float(np.mean([r["ap"] for r in rows])), 4),
        "map_db": round(100.0 * float(np.mean([r["ap_db"] for r in with_db])), 4) if with_db else 0.0,
        "rank1": rank_at(1), "rank5": rank_at(5), "rank10": rank_at(10),
        "recall_at_k": round(100.0 * float(np.mean([r["recall"] for r in rows])), 4),
        "det_ceiling": round(db_capped / gt_total, 4) if gt_total else None,
        "distractor_ratio": round(sum(r["excluded"] for r in rows) / seen, 4) if seen else 0.0,
        "duplicate_hits": int(sum(r.get("duplicates", 0) for r in rows)),
        "queries_without_db_positive": int(len(rows) - len(with_db)),
        "valid_queries": n,
        "sec_per_query": round(float(np.mean([r["sec"] for r in rows])), 4) if all("sec" in r for r in rows) else None,
    }


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=CONFIG_HELP)
    p.add_argument("--data-root", default="./data/PRW")
    p.add_argument("--matches-cache", default=str(DEFAULT_MATCHES), help="point→pid 매칭 캐시 (없으면 생성)")
    p.add_argument("--sources", default="prw_image", help="캐시 생성 시 payload source 필터 (쉼표)")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--stage1", nargs="*", default=None, help="1차 retriever 이름들 (기본: 운영 조합 siglip2 irra)")
    p.add_argument("--rerank", default=None, help="2차 재정렬 벡터 (기본 solider, none 이면 없음)")
    p.add_argument("--limit", type=int, default=200, help="채점할 순위 길이 K")
    p.add_argument("--pool", type=int, default=200, help="재정렬 후보 수 (GUI SOLIDER_POOL_DEFAULT)")
    p.add_argument("--gallery", choices=["test", "all"], default="test", help="test = test 프레임 PRW point 만 (prw_eval 프로토콜)")
    p.add_argument("--max-queries", type=int, default=0, help="0 = 전부. N 이면 고르게 N 개 (빠른 확인)")
    p.add_argument("--pid-split", default=None, help="bench/splits 파일:부분 (예 bench/splits/prw_pids_seed42.json:tune) — 그 인물의 쿼리만")
    p.add_argument("--name", default=None, help="결과 이름 (기본 <stage1>__<rerank>)")
    p.add_argument("--output-dir", default="eval/results/e2e_search")
    p.add_argument("--ledger", default=None, help="실행 원장 JSONL (기본 bench/ledger.jsonl; '' 또는 none 이면 기록 안 함)")
    p.add_argument("--no-ledger", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    p = build_parser()
    args = p.parse_args(argv)
    args.command_line = list(argv) if argv is not None else sys.argv[1:]
    if args.limit < 1 or args.pool < 1:
        p.error("--limit / --pool 은 1 이상")
    started = time.time()
    data_root = Path(args.data_root).expanduser().resolve()
    ann_dir = data_root / "annotations"
    for need in (ann_dir, data_root / "query_box", data_root / "query_info.txt", data_root / "frame_test.mat"):
        if not need.exists():
            raise SystemExit(f"경로가 없습니다: {need}")
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    # 정답
    cache = ensure_matches(Path(args.matches_cache), args.config, sources, args.iou, ann_dir)
    meta, matches = load_matches(cache)
    matches_sha = hashlib.sha256(cache.read_bytes()).hexdigest()
    test_frames = set(prw_eval.load_frame_list(data_root / "frame_test.mat"))
    frame_set: Optional[Set[str]] = test_frames if args.gallery == "test" else None
    gt_frames = sorted(test_frames) if frame_set is not None else sorted({q.stem[:-4] for q in ann_dir.glob("*.jpg.mat")})
    gt_pos = gt_positives(ann_dir, gt_frames)
    db_pos = db_positives(matches, frame_set)
    print(f"[gt] DB point {len(matches):,} (라벨 {sum(1 for m in matches.values() if m.get('status') == 'labeled'):,}) · "
          f"GT 프레임 {len(gt_frames):,} · 인물 {len(gt_pos):,}")

    # 쿼리
    pid_filter: Optional[Set[int]] = None
    if args.pid_split:
        from bench.splits import load_split_arg
        pid_filter = load_split_arg(args.pid_split)
    queries = select_queries(load_queries(data_root), args.max_queries, pid_filter)
    if not queries:
        raise SystemExit("쿼리가 없습니다 (query_box / --pid-split 확인)")
    print(f"[query] {len(queries):,} 개" + (f" (pid 분할 {args.pid_split})" if args.pid_split else ""))

    # 검색기 (GUI/CLI 와 같은 경로)
    from search.unified_search_4mode import CropGeneralSearcher, pipeline_label, resolve_rerank, resolve_stage1
    searcher = CropGeneralSearcher(str(args.config))
    stage1 = resolve_stage1(searcher.cfg, "person", args.stage1)
    rerank = resolve_rerank(searcher.cfg, "person", args.rerank)
    name = args.name or f"{'+'.join(stage1)}__{rerank or 'none'}"
    print(f"[search] {pipeline_label(stage1, rerank)} · K={args.limit} pool={args.pool} gallery={args.gallery}")

    rows: List[Dict[str, Any]] = []
    skipped = 0
    collection = None
    timing_sum: Dict[str, float] = {"embed": 0.0, "search": 0.0, "rerank": 0.0}
    t_search = time.time()
    try:
        for i, q in enumerate(queries, 1):
            gt_frames = gt_pos.get(q["pid"], Counter())
            n_pos_gt = positives_excluding(gt_frames, q["frame"])
            if n_pos_gt == 0:
                skipped += 1
                continue
            db_frames = db_pos.get(q["pid"], Counter())
            n_pos_db = positives_excluding(db_frames, q["frame"])
            t0 = time.time()
            # rerank=None 은 검색기에서 "운영 기본(solider)" 이므로 '없음' 은 반드시 "none" 으로 넘긴다
            res = searcher.search(q["path"], scope="person", limit=args.limit, solider_pool=args.pool, stage1=stage1,
                                  rerank="none" if rerank is None else rerank)
            sec = time.time() - t0
            collection = res.get("collection")
            for k in timing_sum:
                timing_sum[k] += float((res.get("timing") or {}).get(k) or 0.0)
            s = score_query([str(h.point_id) for h in res["hits"]], matches, q["pid"], q["frame"], frame_set, n_pos_gt, n_pos_db,
                            gt_frames=gt_frames)
            s.update(pid=q["pid"], frame=q["frame"], n_pos_gt=n_pos_gt, n_pos_db=n_pos_db,
                     n_pos_db_capped=capped_positives(db_frames, gt_frames, q["frame"]), sec=round(sec, 4))
            rows.append(s)
            if i % 100 == 0 or i == len(queries):
                agg = aggregate(rows)
                print(f"  {i:,}/{len(queries):,}  mAP {agg['map']:.2f} · mAP(db) {agg['map_db']:.2f} · R1 {agg['rank1']:.2f} · "
                      f"{(time.time() - t_search) / i:.2f} s/q")
    finally:
        searcher.release()

    metrics = aggregate(rows)
    out = {
        "producer": PRODUCER, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "name": name,
        "config": {"config_path": str(Path(args.config).resolve()), "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                   "stage1": list(stage1), "rerank": rerank, "pipeline": pipeline_label(stage1, rerank), "limit": args.limit,
                   "pool": args.pool, "gallery": args.gallery, "scope": "person", "pid_split": args.pid_split,
                   "max_queries": args.max_queries or None, "iou": args.iou},
        "gt": {"collection": collection, "queries": len(queries), "valid_queries": len(rows), "skipped_no_gt": skipped,
               "gt_positives": int(sum(r["n_pos_gt"] for r in rows)), "db_positives": int(sum(r["n_pos_db"] for r in rows)),
               "db_points": len(matches), "matches_cache": str(cache), "matches_sha256": matches_sha,
               "matches_meta": {k: v for k, v in meta.items() if k != "_meta"}, "frames": len(gt_frames)},
        "metrics": metrics,
        "timing": {"elapsed_sec": round(time.time() - started, 2), "search_sec": round(time.time() - t_search, 2),
                   "sec_per_query": metrics.get("sec_per_query"),
                   **{f"{k}_sec_per_query": round(v / len(rows), 4) for k, v in timing_sum.items() if rows}},
        "per_query": rows,
    }
    out_dir = Path(args.output_dir).resolve() / name
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "e2e_search.json"
    write_json(json_path, out)
    csv_path = Path(args.output_dir).resolve() / "summary.csv"
    new = not csv_path.is_file()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        keys = ["map", "map_db", "rank1", "rank5", "rank10", "recall_at_k", "det_ceiling", "distractor_ratio", "valid_queries", "sec_per_query"]
        if new:
            w.writerow(["name", "generated_at", "gallery", "limit", *keys])
        w.writerow([name, out["generated_at"], args.gallery, args.limit, *[metrics.get(k) for k in keys]])
    print("\n=== 전체 파이프라인 검색 (PRW GT, 운영 DB) ===")
    print(f"  {name}: mAP {metrics['map']:.2f} · mAP(db) {metrics['map_db']:.2f} · Rank-1 {metrics['rank1']:.2f} · "
          f"Rank-5 {metrics['rank5']:.2f} · recall@{args.limit} {metrics['recall_at_k']:.2f} · 검출 상한 {metrics['det_ceiling']} · "
          f"distractor {metrics['distractor_ratio']} · {metrics['sec_per_query']} s/q · 유효 쿼리 {metrics['valid_queries']:,} (GT 없음 {skipped})")
    print(f"json: {json_path}")
    from bench import ledger
    ledger.record(lambda: [ledger.entry_from_e2e_result(out, report=json_path, command=args.command_line,
                                                        versions=ledger.versions_info())], args.ledger, args.no_ledger)
    print(f"RESULT_SUMMARY: {json_path}")
    return out


if __name__ == "__main__":
    main()
