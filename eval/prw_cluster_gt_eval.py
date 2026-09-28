"""
PRW GT 기반 클러스터링 정확도 평가
==================================
DB 의 PRW person 검출 point(payload image_id / bbox[frame 좌표]) 를 같은 프레임의 PRW GT 박스
(`data/PRW/annotations/<frame>.jpg.mat`, 행 = [pid, x, y, w, h]) 와 IoU 로 매칭해 인물 ID 를 붙이고,
클러스터링 결과(assignments.jsonl: point_id / cluster_id / noise) 가 그 ID 를 얼마나 잘 묶는지 잰다.

GT 규칙
  * IoU >= --iou (기본 0.5) 인 GT 박스 중 최대 IoU 를 취한다. pid == -2 는 "미표기 보행자" 로 평가 제외.
  * GT 와 매칭되지 않은 검출(배경 오검출·미주석)은 평가 제외. 제외 비율을 함께 보고한다.
  * 같은 GT 박스에 검출이 여러 개 붙으면 모두 같은 pid (중복 검출) — 개수를 보고한다.

지표 (GT 가 붙은 point 집합 L 위에서)
  * 쌍(pair) 정밀도/재현율/F1  — 같은 클러스터 쌍 vs 같은 pid 쌍.
      noise_as_singletons : 노이즈 point 는 단독 클러스터 (재현율에서 손실로 반영)
      clustered_only      : 노이즈 point 를 빼고 계산 (묶은 것만 평가)
  * B-cubed 정밀도/재현율/F1 (noise_as_singletons)
  * purity(클러스터 다수 pid 비율) / inverse purity(pid 다수 클러스터 비율), ARI / NMI (두 가지 노이즈 처리)
  * 클러스터 관점: 순수 클러스터 비율, 2개 이상 pid 가 섞인 클러스터 수와 point 수
  * 인물 관점: 2개 이상 클러스터로 갈라진 pid 수, pid 당 평균 클러스터 수, 전부 노이즈인 pid 수
  * 대상은 기본적으로 GT 검출이 2개 이상인 pid 만 (--min-pid-size; 1개인 pid 는 묶을 짝이 없다)

산출물: <output-dir>/<html-name>, cluster_gt_report.json(사이드카), gt_matches.jsonl(point→pid), method 별 cluster_table.csv
마지막에 RESULT_SUMMARY / RESULT_HTML 마커 출력.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from report_common import (CONFIG_HELP, DEFAULT_CONFIG_PATH, common_summary, esc, file_info, histogram_html,  # noqa: E402
                           load_pipeline_settings, resolve, result_markers, table, warning, write_json)
from clustering.compare_cluster_results import (CSS, PALETTE, fetch_payloads, load_labels, resolve_crop, short,  # noqa: E402
                                     thumb_data_uri)

PRODUCER = "prw_cluster_gt_eval"
UNLABELED_PID = -2


# ---------------------------------------------------------------------------
# GT 매칭 (순수 함수)
# ---------------------------------------------------------------------------
def iou_xyxy_vs_xywh(det: Sequence[float], gt: Sequence[float]) -> float:
    """det=[x1,y1,x2,y2], gt=[x,y,w,h]."""
    gx1, gy1, gx2, gy2 = gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]
    ix = max(0.0, min(det[2], gx2) - max(det[0], gx1))
    iy = max(0.0, min(det[3], gy2) - max(det[1], gy1))
    inter = ix * iy
    union = (det[2] - det[0]) * (det[3] - det[1]) + gt[2] * gt[3] - inter
    return inter / union if union > 0 else 0.0


def match_detection(det: Sequence[float], ann: np.ndarray, iou_thr: float) -> Tuple[Optional[int], float, int]:
    """반환 (pid 또는 None, best_iou, gt_row_index). ann 행 = [pid, x, y, w, h]."""
    best, best_i = 0.0, -1
    for i, row in enumerate(ann):
        v = float(iou_xyxy_vs_xywh([float(x) for x in det], [float(x) for x in row[1:5]]))
        if v > best:
            best, best_i = v, i
    if best_i < 0 or best < iou_thr:
        return None, float(best), -1
    return int(ann[best_i][0]), float(best), best_i


def frame_from_image_id(image_id: str) -> Optional[str]:
    """'PRW/c5s2_116999.jpg' -> 'c5s2_116999'."""
    name = str(image_id).replace("\\", "/").rsplit("/", 1)[-1]
    return name[:-4] if name.lower().endswith(".jpg") else None


# ---------------------------------------------------------------------------
# 지표 (순수 함수)
# ---------------------------------------------------------------------------
def _c2(n: int) -> int:
    return n * (n - 1) // 2


def pair_metrics(labels: Dict[Any, Optional[str]], pid_of: Dict[Any, int], ids: Sequence[Any], noise_as_singletons: bool) -> Dict[str, Any]:
    """같은 클러스터 쌍 vs 같은 pid 쌍. noise_as_singletons=False 면 노이즈 point 를 제외."""
    use = [p for p in ids if noise_as_singletons or labels[p] is not None]
    by_pid: Counter = Counter(pid_of[p] for p in use)
    by_cl: Counter = Counter(labels[p] for p in use if labels[p] is not None)
    both: Counter = Counter((pid_of[p], labels[p]) for p in use if labels[p] is not None)
    pairs_true = sum(_c2(n) for n in by_pid.values())
    pairs_pred = sum(_c2(n) for n in by_cl.values())
    pairs_both = sum(_c2(n) for n in both.values())
    precision = pairs_both / pairs_pred if pairs_pred else None
    recall = pairs_both / pairs_true if pairs_true else None
    f1 = None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(points=len(use), pairs_true=pairs_true, pairs_pred=pairs_pred, pairs_both=pairs_both,
                precision=precision, recall=recall, f1=f1)


def bcubed(labels: Dict[Any, Optional[str]], pid_of: Dict[Any, int], ids: Sequence[Any]) -> Dict[str, Any]:
    """B-cubed (노이즈 = 단독 클러스터)."""
    cl_of = {p: (labels[p] if labels[p] is not None else f"__noise__{p}") for p in ids}
    by_pid: Dict[int, List[Any]] = defaultdict(list)
    by_cl: Dict[str, List[Any]] = defaultdict(list)
    for p in ids:
        by_pid[pid_of[p]].append(p)
        by_cl[cl_of[p]].append(p)
    cell: Counter = Counter((pid_of[p], cl_of[p]) for p in ids)
    prec = sum(cell[(pid_of[p], cl_of[p])] / len(by_cl[cl_of[p]]) for p in ids) / len(ids) if ids else None
    rec = sum(cell[(pid_of[p], cl_of[p])] / len(by_pid[pid_of[p]]) for p in ids) / len(ids) if ids else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else (0.0 if prec is not None else None)
    return dict(precision=prec, recall=rec, f1=f1)


def purity_metrics(labels: Dict[Any, Optional[str]], pid_of: Dict[Any, int], ids: Sequence[Any]) -> Dict[str, Any]:
    clustered = [p for p in ids if labels[p] is not None]
    by_cl: Dict[str, Counter] = defaultdict(Counter)
    by_pid: Dict[int, Counter] = defaultdict(Counter)
    for p in clustered:
        by_cl[labels[p]][pid_of[p]] += 1
        by_pid[pid_of[p]][labels[p]] += 1
    n = len(clustered)
    purity = sum(max(c.values()) for c in by_cl.values()) / n if n else None
    inverse = sum(max(c.values()) for c in by_pid.values()) / n if n else None
    pure_clusters = sum(1 for c in by_cl.values() if len(c) == 1)
    mixed = {cl: c for cl, c in by_cl.items() if len(c) >= 2}
    mixed_points = sum(sum(c.values()) for c in mixed.values())
    all_pids = Counter(pid_of[p] for p in ids)
    split_pids = sum(1 for c in by_pid.values() if len(c) >= 2)
    noise_only_pids = sum(1 for pid in all_pids if pid not in by_pid)
    clusters_per_pid = (sum(len(c) for c in by_pid.values()) / len(by_pid)) if by_pid else None
    return dict(clustered_points=n, purity=purity, inverse_purity=inverse,
                clusters_with_labeled=len(by_cl), pure_clusters=pure_clusters,
                mixed_clusters=len(mixed), mixed_cluster_points=mixed_points,
                pids=len(all_pids), pids_clustered=len(by_pid), pids_split=split_pids,
                pids_all_noise=noise_only_pids, clusters_per_pid=clusters_per_pid,
                mixed_detail={cl: dict(pids=len(c), points=sum(c.values()), top=c.most_common(3)) for cl, c in mixed.items()},
                split_detail={str(pid): dict(clusters=len(c), points=sum(c.values())) for pid, c in by_pid.items() if len(c) >= 2})


def ari_nmi(labels: Dict[Any, Optional[str]], pid_of: Dict[Any, int], ids: Sequence[Any]) -> Dict[str, Any]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    def encode(subset, noise_singleton):
        codes: Dict[str, int] = {}
        out = []
        for i, p in enumerate(subset):
            l = labels[p]
            out.append((-1 - i) if l is None else codes.setdefault(l, len(codes)))
        return out
    truth = [pid_of[p] for p in ids]
    res: Dict[str, Any] = {}
    if ids:
        pred = encode(ids, True)
        res["ari_noise_as_singletons"] = float(adjusted_rand_score(truth, pred))
        res["nmi_noise_as_singletons"] = float(normalized_mutual_info_score(truth, pred))
    sub = [p for p in ids if labels[p] is not None]
    if sub:
        res["ari_clustered_only"] = float(adjusted_rand_score([pid_of[p] for p in sub], encode(sub, False)))
        res["nmi_clustered_only"] = float(normalized_mutual_info_score([pid_of[p] for p in sub], encode(sub, False)))
    return res


def evaluate_method(labels: Dict[Any, Optional[str]], pid_of: Dict[Any, int], ids: Sequence[Any]) -> Dict[str, Any]:
    ids = [p for p in ids if p in labels]
    noise = sum(1 for p in ids if labels[p] is None)
    out = dict(labeled_points=len(ids), noise_points=noise, noise_ratio=(noise / len(ids)) if ids else None,
               pairs_noise_as_singletons=pair_metrics(labels, pid_of, ids, True),
               pairs_clustered_only=pair_metrics(labels, pid_of, ids, False),
               bcubed=bcubed(labels, pid_of, ids), **ari_nmi(labels, pid_of, ids))
    out.update(purity_metrics(labels, pid_of, ids))
    return out


# ---------------------------------------------------------------------------
# Qdrant / 주석 로딩
# ---------------------------------------------------------------------------
def scroll_points(url: str, api_key: Optional[str], collection: str, sources: List[str], batch: int = 1024,
                  timeout: int = 120, log=print) -> List[Dict[str, Any]]:
    import requests
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    if api_key:
        s.headers.update({"api-key": api_key})
    flt = {"must": [{"key": "source", "match": {"any": sources}}]}
    out: List[Dict[str, Any]] = []
    offset = None
    while True:
        body = {"limit": batch, "with_payload": ["image_id", "bbox", "bbox_space", "detection_id", "crop_path"],
                "with_vector": False, "filter": flt}
        if offset is not None:
            body["offset"] = offset
        r = s.post(f"{url.rstrip('/')}/collections/{collection}/points/scroll", json=body, timeout=timeout)
        if not r.ok:
            raise RuntimeError(f"scroll -> {r.status_code}\n{r.text[:500]}")
        res = r.json().get("result") or {}
        pts = res.get("points") or []
        out.extend({"point_id": p["id"], **(p.get("payload") or {})} for p in pts)
        offset = res.get("next_page_offset")
        if not pts or offset is None:
            break
        if len(out) % (batch * 10) == 0:
            log(f"  scroll {len(out):,}")
    return out


def match_all(points: List[Dict[str, Any]], ann_dir: Path, iou_thr: float, log=print) -> Tuple[Dict[Any, Dict[str, Any]], Dict[str, Any]]:
    from eval import prw_eval
    ann_cache: Dict[str, np.ndarray] = {}
    result: Dict[Any, Dict[str, Any]] = {}
    stats = Counter()
    gt_hits: Counter = Counter()
    for p in points:
        frame = frame_from_image_id(p.get("image_id", ""))
        bbox = p.get("bbox")
        if frame is None or not isinstance(bbox, list) or len(bbox) != 4:
            stats["bad_payload"] += 1
            continue
        if str(p.get("bbox_space", "frame")) != "frame":
            stats["bbox_space_not_frame"] += 1
            continue
        if frame not in ann_cache:
            path = ann_dir / f"{frame}.jpg.mat"
            ann_cache[frame] = prw_eval.load_annotation(path) if path.is_file() else np.empty((0, 5), dtype=np.float32)
            if not path.is_file():
                stats["frames_without_annotation"] += 1
        ann = ann_cache[frame]
        pid, best_iou, row = match_detection([float(x) for x in bbox], ann, iou_thr)
        rec = dict(frame=frame, iou=round(best_iou, 4), pid=pid, status=None)
        if pid is None:
            rec["status"] = "unmatched"
            stats["unmatched"] += 1
        elif pid == UNLABELED_PID:
            rec["status"] = "unlabeled"
            stats["unlabeled"] += 1
        else:
            rec["status"] = "labeled"
            stats["labeled"] += 1
            gt_hits[(frame, row)] += 1
        result[p["point_id"]] = rec
    stats["points"] = len(points)
    stats["frames"] = len(ann_cache)
    stats["gt_boxes_matched"] = len(gt_hits)
    stats["gt_boxes_with_duplicate_detections"] = sum(1 for n in gt_hits.values() if n >= 2)
    return result, dict(stats)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, int):
        return f"{v:,}"
    return esc(v)


def summary_table(results: Dict[str, Dict[str, Any]]) -> str:
    names = list(results)
    rows = [
        ("GT 가 붙은 point", lambda r: r["labeled_points"]),
        ("그중 노이즈", lambda r: r["noise_points"]),
        ("노이즈 비율", lambda r: r["noise_ratio"]),
        ("쌍 정밀도 (노이즈=단독)", lambda r: r["pairs_noise_as_singletons"]["precision"]),
        ("쌍 재현율 (노이즈=단독)", lambda r: r["pairs_noise_as_singletons"]["recall"]),
        ("쌍 F1 (노이즈=단독)", lambda r: r["pairs_noise_as_singletons"]["f1"]),
        ("쌍 정밀도 (배정된 point 만)", lambda r: r["pairs_clustered_only"]["precision"]),
        ("쌍 재현율 (배정된 point 만)", lambda r: r["pairs_clustered_only"]["recall"]),
        ("쌍 F1 (배정된 point 만)", lambda r: r["pairs_clustered_only"]["f1"]),
        ("B-cubed 정밀도", lambda r: r["bcubed"]["precision"]),
        ("B-cubed 재현율", lambda r: r["bcubed"]["recall"]),
        ("B-cubed F1", lambda r: r["bcubed"]["f1"]),
        ("ARI (노이즈=단독)", lambda r: r.get("ari_noise_as_singletons")),
        ("NMI (노이즈=단독)", lambda r: r.get("nmi_noise_as_singletons")),
        ("ARI (배정된 point 만)", lambda r: r.get("ari_clustered_only")),
        ("NMI (배정된 point 만)", lambda r: r.get("nmi_clustered_only")),
        ("purity (클러스터 다수 pid 비율)", lambda r: r["purity"]),
        ("inverse purity (pid 다수 클러스터 비율)", lambda r: r["inverse_purity"]),
        ("GT point 를 포함한 클러스터 수", lambda r: r["clusters_with_labeled"]),
        ("순수 클러스터 (pid 1개)", lambda r: r["pure_clusters"]),
        ("혼합 클러스터 (pid 2개 이상)", lambda r: r["mixed_clusters"]),
        ("혼합 클러스터 안의 GT point", lambda r: r["mixed_cluster_points"]),
        ("평가 대상 pid 수", lambda r: r["pids"]),
        ("2개 이상 클러스터로 갈라진 pid", lambda r: r["pids_split"]),
        ("pid 당 평균 클러스터 수", lambda r: r["clusters_per_pid"]),
        ("전부 노이즈인 pid", lambda r: r["pids_all_noise"]),
    ]
    head = "".join(f"<th>{esc(n)}</th>" for n in names)
    body = "".join(f"<tr><th>{esc(label)}</th>" + "".join(f"<td>{fmt(fn(results[n]))}</td>" for n in names) + "</tr>"
                   for label, fn in rows)
    return f"<table><tr><th></th>{head}</tr>{body}</table>"


def examples_html(results, labels_by_method, pid_of, members_by_method, payloads, project_root, inline, thumb, quality,
                  per_example, k, rng, html_path, show_images) -> str:
    parts = []
    thumb_cache: Dict[Any, Optional[str]] = {}

    def render_group(title, groups):
        body = []
        for gi, (chip_label, chosen, total) in enumerate(groups):
            color = PALETTE[gi % len(PALETTE)]
            chip = f"<span class='chip' style='background:{color}'>{esc(chip_label)} ({total:,}장 중 {len(chosen)}장)</span>"
            imgs = []
            for pid in (chosen if show_images else ()):
                payload = payloads.get(pid) or {}
                src = resolve_crop(payload, project_root)
                t = esc(payload.get("image_id") or pid)
                if src is None:
                    imgs.append(f"<div class='ph' style='--c:{color}' title='{t}'>없음</div>")
                elif inline:
                    if pid not in thumb_cache:
                        thumb_cache[pid] = thumb_data_uri(src, thumb, quality)
                    uri = thumb_cache[pid]
                    imgs.append(f"<img style='--c:{color}' src='{uri}' title='{t}'>" if uri else f"<div class='ph' style='--c:{color}'>실패</div>")
                else:
                    from report_common import path_href
                    imgs.append(f"<img style='--c:{color}' src='{path_href(src, html_path)}' title='{t}' loading='lazy'>")
            body.append(f"<div class='grp' style='--c:{color}'>{chip}{''.join(imgs)}</div>")
        return f"<div class='ex'>{title}{''.join(body)}</div>"

    for name, res in results.items():
        labels = labels_by_method[name]
        members = members_by_method[name]
        # 1) pid 가 가장 많이 섞인 클러스터
        mixed = sorted(res["mixed_detail"].items(), key=lambda kv: (-kv[1]["pids"], -kv[1]["points"]))[:k]
        parts.append(f"<h2>{esc(name)}: 서로 다른 인물이 가장 많이 섞인 클러스터 (오병합)</h2>")
        if not mixed:
            parts.append("<p class='muted'>없음</p>")
        for cl, d in mixed:
            pts = [p for p in members[cl] if p in pid_of]
            by_pid: Dict[int, List[Any]] = defaultdict(list)
            for p in pts:
                by_pid[pid_of[p]].append(p)
            ordered = sorted(by_pid.items(), key=lambda kv: -len(kv[1]))
            quota = max(2, per_example // max(1, min(len(ordered), 6)))
            groups = [(f"GT pid {pid}", (m if len(m) <= quota else rng.sample(m, quota)), len(m)) for pid, m in ordered[:6]]
            title = (f"클러스터 <code>{esc(short(cl))}</code> · GT point {d['points']:,} · 인물 {d['pids']} 명 · "
                     f"전체 크기 {len(members[cl]):,} (GT 없는 point 포함)")
            parts.append(render_group(title, groups))
        # 2) 가장 많이 갈라진 pid
        split = sorted(res["split_detail"].items(), key=lambda kv: (-kv[1]["clusters"], -kv[1]["points"]))[:k]
        parts.append(f"<h2>{esc(name)}: 한 인물이 가장 많이 갈라진 경우 (과분할) · 노이즈 포함</h2>")
        if not split:
            parts.append("<p class='muted'>없음</p>")
        pid_members: Dict[int, List[Any]] = defaultdict(list)
        for p, pid in pid_of.items():
            if p in labels:
                pid_members[pid].append(p)
        for pid_s, d in split:
            pid = int(pid_s)
            pts = pid_members[pid]
            by_cl: Dict[Optional[str], List[Any]] = defaultdict(list)
            for p in pts:
                by_cl[labels[p]].append(p)
            ordered = sorted(by_cl.items(), key=lambda kv: (kv[0] is None, -len(kv[1])))
            quota = max(2, per_example // max(1, min(len(ordered), 6)))
            groups = [((f"클러스터 {short(cl)}" if cl else "노이즈"), (m if len(m) <= quota else rng.sample(m, quota)), len(m))
                      for cl, m in ordered[:6]]
            title = f"GT pid {pid} · 검출 {len(pts):,}장 → 클러스터 {d['clusters']} 개" + (f" + 노이즈 {len(by_cl.get(None, []))}" if None in by_cl else "")
            parts.append(render_group(title, groups))
    return "".join(parts)


# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", action="append", required=True,
                   help="이름=assignments.jsonl (여러 번). 예: Leiden=outputs/clustering/leiden_image_prw/person/person_leiden_assignments.jsonl")
    p.add_argument("--data-root", default="./data/PRW")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--min-pid-size", type=int, default=2, help="GT 검출이 이 수 미만인 pid 는 평가 제외 (짝이 없음)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=CONFIG_HELP)
    p.add_argument("--collection", default=None, help=CONFIG_HELP)
    p.add_argument("--qdrant-url", default=None, help=CONFIG_HELP)
    p.add_argument("--api-key", default=None)
    p.add_argument("--sources", default="prw_image")
    p.add_argument("--matches-cache", default="eval/results/cache/prw_gt_matches.jsonl",
                   help="point→pid 매칭 캐시 (있으면 Qdrant/주석 재조회 생략)")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--project-root", default=".")
    p.add_argument("--output-dir", default="eval/results/cluster_gt_prw")
    p.add_argument("--html-name", default="cluster_gt_prw.html")
    p.add_argument("--title", default=None)
    p.add_argument("--examples", type=int, default=5)
    p.add_argument("--images-per-example", type=int, default=24)
    p.add_argument("--inline-images", action="store_true")
    p.add_argument("--no-images", action="store_true")
    p.add_argument("--thumb-size", type=int, default=150)
    p.add_argument("--thumb-quality", type=int, default=70)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pid-split", default=None, help="bench/splits 파일:부분 (예 bench/splits/prw_pids_seed42.json:tune) — 그 인물의 point 만 평가")
    p.add_argument("--ledger", default=None, help="실행 원장 JSONL (기본 bench/ledger.jsonl; '' 또는 none 이면 기록 안 함)")
    p.add_argument("--no-ledger", action="store_true", help="원장(bench/ledger.jsonl)에 기록하지 않음")
    return p


def parse_methods(items: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"--method 는 이름=경로 형식: {it!r}")
        name, path = it.split("=", 1)
        out[name.strip()] = Path(path.strip())
    return out


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    args.command_line = list(argv) if argv is not None else sys.argv[1:]
    try:
        methods = parse_methods(args.method)
        settings = load_pipeline_settings(args.config, require=not (args.qdrant_url and args.collection))
        args.qdrant_url = resolve(args.qdrant_url, settings.qdrant_url if settings else None, "qdrant-url")
        args.collection = resolve(args.collection, settings.person_collection if settings else None, "collection")
        html_name = Path(args.html_name)
        if html_name.name != args.html_name or not args.html_name.lower().endswith((".html", ".htm")):
            raise ValueError("html-name 은 하위 경로 없는 .html 파일명")
    except (OSError, ValueError, AttributeError) as exc:
        p.error(str(exc))
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    started = time.time()
    warnings: List[Dict[str, Any]] = []
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / args.html_name
    sidecar = out_dir / "cluster_gt_report.json"
    ann_dir = Path(args.data_root).expanduser().resolve() / "annotations"
    if not ann_dir.is_dir():
        raise SystemExit(f"주석 폴더가 없습니다: {ann_dir}")

    # 1) point -> pid 매칭 (캐시)
    cache = Path(args.matches_cache)
    matches: Dict[Any, Dict[str, Any]] = {}
    match_stats: Dict[str, Any] = {}
    if cache.is_file() and not args.no_cache:
        with cache.open(encoding="utf-8") as f:
            header = json.loads(f.readline())
            if header.get("_meta") and header.get("iou") == args.iou and header.get("sources") == sources \
                    and header.get("collection") == args.collection:
                match_stats = header.get("stats", {})
                for ln in f:
                    d = json.loads(ln)
                    matches[d.pop("point_id")] = d
                print(f"[gt] 매칭 캐시 사용: {cache} ({len(matches):,} points)")
            else:
                print("[gt] 매칭 캐시 설정 불일치 → 다시 계산")
    if not matches:
        print(f"[gt] Qdrant 에서 {args.collection} / source={sources} point 조회...")
        points = scroll_points(args.qdrant_url, args.api_key, args.collection, sources)
        print(f"[gt] {len(points):,} points → GT 매칭 (IoU >= {args.iou})...")
        matches, match_stats = match_all(points, ann_dir, args.iou)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(cache.suffix + ".tmp")   # 부분 기록이 캐시로 읽히지 않게 임시 파일 → 교체
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(dict(_meta=True, iou=args.iou, sources=sources, collection=args.collection,
                                    stats=match_stats), ensure_ascii=False) + "\n")
            for pid_key, d in matches.items():
                f.write(json.dumps({"point_id": pid_key, **d}, ensure_ascii=False) + "\n")
        import os
        os.replace(tmp, cache)
    print(f"[gt] 매칭 통계: {json.dumps(match_stats, ensure_ascii=False)}")

    pid_of_all = {pk: d["pid"] for pk, d in matches.items() if d["status"] == "labeled"}
    pid_sizes = Counter(pid_of_all.values())
    pid_of = {pk: pid for pk, pid in pid_of_all.items() if pid_sizes[pid] >= args.min_pid_size}
    dropped_small = len(pid_of_all) - len(pid_of)
    if args.pid_split:
        from bench.splits import load_split_arg
        keep = load_split_arg(args.pid_split)
        before = len(pid_of)
        pid_of = {pk: pid for pk, pid in pid_of.items() if pid in keep}
        print(f"[gt] 인물 분할 {args.pid_split}: 평가 point {before:,} → {len(pid_of):,} (인물 {len(set(pid_of.values())):,})")
    print(f"[gt] GT 가 붙은 point {len(pid_of_all):,} / 평가 대상(pid 크기>={args.min_pid_size}) {len(pid_of):,} "
          f"(단독 pid 제외 {dropped_small:,}) / 인물 {len(set(pid_of.values())):,}")
    ids = sorted(pid_of, key=str)
    (out_dir / "gt_matches.jsonl").write_text(
        "".join(json.dumps({"point_id": pk, **d}, ensure_ascii=False) + "\n" for pk, d in matches.items()),
        encoding="utf-8", newline="\n")

    # 2) 방법별 평가
    results: Dict[str, Dict[str, Any]] = {}
    labels_by_method: Dict[str, Dict[Any, Optional[str]]] = {}
    members_by_method: Dict[str, Dict[str, List[Any]]] = {}
    inputs = []
    for name, path in methods.items():
        errors = dict(count=0, first_line=None)
        labels, dup = load_labels(path, errors)
        inputs.append(file_info(f"assignments:{name}", path))
        if errors["count"]:
            warnings.append(warning("PARSE_ERRORS", f"{name}: 잘못된 줄 {errors['count']}", "person"))
        missing = sum(1 for pk in ids if pk not in labels)
        if missing:
            warnings.append(warning("MISSING_POINTS", f"{name}: 평가 대상 point {missing:,} 개가 assignments 에 없음 (제외)", "person"))
        res = evaluate_method(labels, pid_of, ids)
        results[name] = res
        labels_by_method[name] = labels
        members: Dict[str, List[Any]] = defaultdict(list)
        for pk, cl in labels.items():
            if cl is not None:
                members[cl].append(pk)
        members_by_method[name] = members
        pn = res["pairs_noise_as_singletons"]
        pc = res["pairs_clustered_only"]
        print(f"  {name:<14} labeled={res['labeled_points']:,} noise={res['noise_points']:,} | pair P/R/F1(noise=single) "
              f"{fmt(pn['precision'])}/{fmt(pn['recall'])}/{fmt(pn['f1'])} | clustered-only {fmt(pc['precision'])}/{fmt(pc['recall'])}/{fmt(pc['f1'])} "
              f"| B3 F1 {fmt(res['bcubed']['f1'])} | purity {fmt(res['purity'])} | mixed clusters {res['mixed_clusters']:,} | split pids {res['pids_split']:,}")
        with (out_dir / f"cluster_table_{name}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["cluster_id", "size_total", "labeled_points", "distinct_pids", "top_pid", "top_pid_points"])
            for cl, mem in members.items():
                lab = [pid_of[pk] for pk in mem if pk in pid_of]
                if not lab:
                    continue
                c = Counter(lab)
                top, n = c.most_common(1)[0]
                w.writerow([cl, len(mem), len(lab), len(c), top, n])

    # 3) 예시 + HTML
    rng = random.Random(args.seed)
    payloads: Dict[Any, Dict[str, Any]] = {}
    need: set = set()
    if not args.no_images:
        for name, res in results.items():
            members = members_by_method[name]
            for cl, _ in sorted(res["mixed_detail"].items(), key=lambda kv: (-kv[1]["pids"], -kv[1]["points"]))[:args.examples]:
                need.update(pk for pk in members[cl] if pk in pid_of)
            for pid_s, _ in sorted(res["split_detail"].items(), key=lambda kv: (-kv[1]["clusters"], -kv[1]["points"]))[:args.examples]:
                need.update(pk for pk, pid in pid_of.items() if pid == int(pid_s))
        # 표본은 examples_html 안에서 뽑지만 조회는 상위집합(해당 클러스터/pid 전체) 으로 해서 항상 포함되게 한다
        try:
            payloads = fetch_payloads(args.qdrant_url, args.api_key, args.collection, sorted(need, key=str))
        except Exception as exc:
            warnings.append(warning("PAYLOAD_FETCH_FAILED", f"payload 조회 실패: {exc}", "person"))
    examples_block = examples_html(results, labels_by_method, pid_of, members_by_method, payloads,
                                   Path(args.project_root).resolve(), args.inline_images, args.thumb_size,
                                   args.thumb_quality, args.images_per_example, args.examples, rng, html_path,
                                   show_images=not args.no_images)

    title = args.title or "PRW GT 기반 클러스터링 정확도 비교 (person)"
    gt_table = table({"평가 point (pid 크기>=%d)" % args.min_pid_size: len(ids), "인물(pid) 수": len(set(pid_of.values())),
                      "GT 가 붙었지만 단독 pid 라 제외": dropped_small, **{k: v for k, v in match_stats.items()}})
    pid_hist = histogram_html(dict(Counter(("2–4" if n < 5 else "5–19" if n < 20 else "20–99" if n < 100 else "100+")
                                            for n in Counter(pid_of.values()).values())))
    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>{esc(title)}</title>
<style>{CSS % dict(h=args.thumb_size)}</style></head><body>
<h1>{esc(title)}</h1>
<p class="muted">GT = PRW annotations (IoU ≥ {args.iou}, pid −2 미표기 제외) · 생성 {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
<h2>1. GT 매칭</h2>{gt_table}<h3>인물(pid) 당 검출 수 분포</h3>{pid_hist}
<h2>2. 방법별 정확도</h2>{summary_table(results)}
<p class="muted">쌍 정밀도 = 같은 클러스터로 묶인 GT 쌍 중 실제 같은 인물 비율(오병합↓). 쌍 재현율 = 같은 인물 쌍 중 같은 클러스터에 들어간 비율(과분할·노이즈↓).
"노이즈=단독" 은 노이즈 point 를 각자 단독 클러스터로 보아 재현율 손실에 반영, "배정된 point 만" 은 노이즈를 빼고 묶은 것만 평가.</p>
{examples_block}
<h2>입력</h2>{table({i['role']: {'path': i['path'], 'sha256': (i.get('sha256') or '')[:16]} for i in inputs})}
<h2>경고</h2>{('<ul>' + ''.join(f'<li>{esc(w["message"])}</li>' for w in warnings) + '</ul>') if warnings else '<p class="muted">없음</p>'}
</body></html>"""
    html_path.write_text(html, encoding="utf-8", newline="\n")

    summary = common_summary(PRODUCER, html_path, inputs, warnings)
    slim = {name: {k: v for k, v in res.items() if k not in ("mixed_detail", "split_detail")} for name, res in results.items()}
    summary.update(config=dict(methods={k: str(v) for k, v in methods.items()}, iou=args.iou, min_pid_size=args.min_pid_size,
                               sources=sources, collection=args.collection, data_root=str(Path(args.data_root).resolve()),
                               pid_split=args.pid_split),
                   gt=dict(match_stats=match_stats, evaluated_points=len(ids), pids=len(set(pid_of.values())),
                           dropped_singleton_pid_points=dropped_small),
                   results=slim, elapsed_sec=round(time.time() - started, 3))
    write_json(sidecar, summary)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        keys = ["labeled_points", "noise_points", "noise_ratio", "purity", "inverse_purity", "pure_clusters", "mixed_clusters",
                "mixed_cluster_points", "pids", "pids_split", "pids_all_noise", "clusters_per_pid",
                "ari_noise_as_singletons", "nmi_noise_as_singletons", "ari_clustered_only", "nmi_clustered_only"]
        w.writerow(["method", "pair_P(noise=single)", "pair_R(noise=single)", "pair_F1(noise=single)",
                    "pair_P(clustered)", "pair_R(clustered)", "pair_F1(clustered)", "B3_P", "B3_R", "B3_F1"] + keys)
        for name, res in results.items():
            pn, pc, b3 = res["pairs_noise_as_singletons"], res["pairs_clustered_only"], res["bcubed"]
            w.writerow([name, pn["precision"], pn["recall"], pn["f1"], pc["precision"], pc["recall"], pc["f1"],
                        b3["precision"], b3["recall"], b3["f1"]] + [res.get(k) for k in keys])
    print(f"html    : {html_path}")
    print(f"sidecar : {sidecar}")
    from bench import ledger
    ledger.record(lambda: ledger.entries_from_cluster_summary(summary, report=sidecar, command=getattr(args, "command_line", None)),
                  getattr(args, "ledger", None), getattr(args, "no_ledger", False))
    result_markers(sidecar, html_path)
    return summary


if __name__ == "__main__":
    main()
