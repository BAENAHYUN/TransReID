"""
PRW GT 기반 검출(detection) 정확도 평가
=====================================
검출 단계 단독 평가. detect/ 플러그인(BaseDetector: module / class / params) 을 PRW 프레임에 돌려
GT 사람 박스(`data/PRW/annotations/<frame>.jpg.mat`, 행 = [pid, x, y, w, h]; pid −2 "미표기 보행자" 도
사람 박스이므로 검출 평가에는 포함) 와 IoU 로 매칭해 AP / 정밀도·재현율 / 크기별 재현율 / 속도를 잰다.
검출기를 바꿔도(RF-DETR → YOLO26 …) 같은 숫자로 비교하는 것이 목적이다.

--mode
  run            검출기 플러그인을 PRW 프레임에 실행 → <output-dir>/<name>/detections.jsonl + run_meta.json → 바로 채점
  import-qdrant  운영 DB 에 이미 들어간 PRW 검출(payload bbox / score, source=prw_image) 을 detections.jsonl 로 변환 → 채점.
                 DB 구축 때의 임계값(0.5)·crop 최소 크기 필터가 그대로 반영된 "운영 동작점" 이다.
  score          detections.jsonl 여러 개(--method 이름=경로)를 같은 GT 로 채점

채점 규칙
  * 매칭: confidence 내림차순 greedy. 아직 안 잡힌 GT 중 IoU 최대가 임계값(--iou, 기본 0.5) 이상이면 TP, 아니면 FP.
    FP 는 "중복"(어떤 GT 와는 겹치지만 그 GT 가 이미 잡힘) 과 "배경" 으로 나눠 센다. 남은 GT = FN.
  * AP@0.5 = all-point 보간(VOC2010+). AP@[.5:.95] = IoU 0.50 … 0.95 (0.05 간격) 의 AP 평균.
  * 동작점 표: conf ≥ t (t = --thresholds + 운영 임계값) 에서 P / R / F1, 프레임당 검출 수.
  * 크기별 재현율: GT 높이 <50 / 50–74 / 75–119 / 120–199 / 200+ px (crop 최소 높이 75·120 기준과 맞춤),
    운영 임계값에서와 임계값 없이(max) 각각.
  * 방법이 여러 개이고 프레임 집합이 다르면 교집합에서만 비교한다 (경고 기록).

산출물: <output-dir>/<html-name>, detect_eval_report.json(사이드카), summary.csv, examples/ (FN 이 많은 프레임 그림)
마지막에 RESULT_SUMMARY / RESULT_HTML 마커 출력.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from report_common import (CONFIG_HELP, DEFAULT_CONFIG_PATH, common_summary, esc, file_info, histogram_html,  # noqa: E402
                           load_pipeline_settings, path_href, resolve, result_markers, table, warning, write_json)

PRODUCER = "detect_eval_prw"
UNLABELED_PID = -2
SIZE_BUCKETS: Tuple[Tuple[str, float, float], ...] = (
    ("<50", 0.0, 50.0), ("50–74", 50.0, 75.0), ("75–119", 75.0, 120.0), ("120–199", 120.0, 200.0), ("200+", 200.0, math.inf))
DEFAULT_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
IOU_RANGE = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
DEFAULT_DETECTOR_CONFIG = PROJECT_ROOT / "pipeline_tracking.yaml"
PALETTE = ("#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2")


# ---------------------------------------------------------------------------
# 순수 함수: 기하 · 매칭 · 지표
# ---------------------------------------------------------------------------
def size_bucket(height: float) -> str:
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= height < hi:
            return name
    return SIZE_BUCKETS[-1][0]


def xywh_to_xyxy(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
    out = rows.copy()
    out[:, 2] = rows[:, 0] + rows[:, 2]
    out[:, 3] = rows[:, 1] + rows[:, 3]
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (D,4), b (G,4) xyxy → (D,G)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    ix = np.maximum(0.0, np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]))
    iy = np.maximum(0.0, np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]))
    inter = ix * iy
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def match_frame(dets: np.ndarray, gts: np.ndarray, iou_thr: float) -> Dict[str, np.ndarray]:
    """dets (D,5) [x1,y1,x2,y2,conf], gts (G,4) xyxy.
    반환(det 은 conf 내림차순으로 정렬됨): conf (D,), tp (D,) bool, fp_kind (D,) int 0=TP 1=중복 2=배경,
    gt_conf (G,) 매칭한 det 의 conf (미매칭 = nan), det_gt (D,) 매칭한 GT index (없으면 -1)."""
    dets = np.asarray(dets, dtype=np.float64).reshape(-1, 5)
    gts = np.asarray(gts, dtype=np.float64).reshape(-1, 4)
    order = np.argsort(-dets[:, 4], kind="stable")
    dets = dets[order]
    d, g = dets.shape[0], gts.shape[0]
    iou = iou_matrix(dets[:, :4], gts)
    taken = np.zeros(g, dtype=bool)
    tp = np.zeros(d, dtype=bool)
    fp_kind = np.full(d, 2, dtype=np.int64)
    det_gt = np.full(d, -1, dtype=np.int64)
    gt_conf = np.full(g, np.nan, dtype=np.float64)
    for i in range(d):
        if g == 0:
            break
        row = iou[i]
        cand = np.where(taken, -1.0, row)
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            tp[i] = True
            fp_kind[i] = 0
            taken[j] = True
            det_gt[i] = j
            gt_conf[j] = dets[i, 4]
        elif row.max() >= iou_thr:
            fp_kind[i] = 1
    return dict(conf=dets[:, 4], tp=tp, fp_kind=fp_kind, gt_conf=gt_conf, det_gt=det_gt)


def pr_curve(conf: np.ndarray, tp: np.ndarray, n_gt: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """conf 내림차순으로 정렬해 누적 P/R 과 all-point 보간 AP 를 돌려준다."""
    if n_gt <= 0:
        return np.zeros(0), np.zeros(0), 0.0
    if conf.shape[0] == 0:
        return np.zeros(0), np.zeros(0), 0.0
    order = np.argsort(-conf, kind="stable")
    tp_s = tp[order].astype(np.float64)
    cum_tp = np.cumsum(tp_s)
    cum_fp = np.cumsum(1.0 - tp_s)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    return recall, precision, average_precision(recall, precision)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.shape[0] == 0:
        return 0.0
    mrec = np.concatenate([[0.0], recall, [recall[-1]]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(mpre.shape[0] - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def operating_point(conf: np.ndarray, tp: np.ndarray, fp_kind: np.ndarray, n_gt: int, thr: float, n_frames: int) -> Dict[str, Any]:
    keep = conf >= thr
    n = int(keep.sum())
    t = int(tp[keep].sum())
    fp = n - t
    dup = int((fp_kind[keep] == 1).sum())
    precision = t / n if n else None
    recall = t / n_gt if n_gt else None
    f1 = None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(threshold=float(thr), dets=n, tp=t, fp=fp, fp_duplicate=dup, fp_background=fp - dup, fn=int(n_gt - t),
                precision=precision, recall=recall, f1=f1, dets_per_frame=(n / n_frames) if n_frames else None)


def recall_by_size(gt_heights: np.ndarray, gt_conf: np.ndarray, thr: float) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    buckets = np.array([size_bucket(float(h)) for h in gt_heights]) if gt_heights.shape[0] else np.zeros(0, dtype=str)
    for name, _, _ in SIZE_BUCKETS:
        sel = buckets == name
        n = int(sel.sum())
        matched_any = int(np.sum(~np.isnan(gt_conf[sel]))) if n else 0
        matched_thr = int(np.sum(gt_conf[sel] >= thr)) if n else 0
        out[name] = dict(gt=n, recalled=matched_thr, recall=(matched_thr / n) if n else None,
                         recalled_any=matched_any, recall_any=(matched_any / n) if n else None)
    return out


def subsample_curve(recall: np.ndarray, precision: np.ndarray, n: int = 200) -> List[Tuple[float, float]]:
    if recall.shape[0] == 0:
        return []
    if recall.shape[0] <= n:
        idx = np.arange(recall.shape[0])
    else:
        idx = np.unique(np.linspace(0, recall.shape[0] - 1, n).astype(int))
    return [(round(float(recall[i]), 4), round(float(precision[i]), 4)) for i in idx]


def evaluate_method(dets_by_frame: Dict[str, np.ndarray], gt_by_frame: Dict[str, np.ndarray], frames: Sequence[str],
                    iou_thr: float, operating_thr: float, thresholds: Sequence[float]) -> Dict[str, Any]:
    """gt_by_frame[frame] = (G,4) xywh. 반환: 지표 + 예시용 프레임별 FN/FP(운영 임계값)."""
    n_gt = int(sum(gt_by_frame.get(f, np.zeros((0, 4))).shape[0] for f in frames))
    per_iou: Dict[float, Dict[str, Any]] = {}
    per_frame_counts: Dict[str, Dict[str, int]] = {}
    for iou_t in sorted(set([iou_thr, *IOU_RANGE])):
        confs, tps, kinds, heights, gconfs = [], [], [], [], []
        for f in frames:
            gt = gt_by_frame.get(f, np.zeros((0, 4)))
            dets = dets_by_frame.get(f, np.zeros((0, 5)))
            m = match_frame(dets, xywh_to_xyxy(gt), iou_t)
            confs.append(m["conf"]); tps.append(m["tp"]); kinds.append(m["fp_kind"])
            heights.append(gt[:, 3] if gt.shape[0] else np.zeros(0)); gconfs.append(m["gt_conf"])
            if iou_t == iou_thr:
                keep = m["conf"] >= operating_thr
                fn = int(np.sum(~(m["gt_conf"] >= operating_thr))) if gt.shape[0] else 0
                per_frame_counts[f] = dict(fn=fn, fp=int(np.sum(keep & ~m["tp"])), tp=int(np.sum(keep & m["tp"])), gt=int(gt.shape[0]))
        conf = np.concatenate(confs) if confs else np.zeros(0)
        tp = np.concatenate(tps).astype(bool) if tps else np.zeros(0, dtype=bool)
        kind = np.concatenate(kinds) if kinds else np.zeros(0, dtype=np.int64)
        h = np.concatenate(heights) if heights else np.zeros(0)
        gc = np.concatenate(gconfs) if gconfs else np.zeros(0)
        recall, precision, ap = pr_curve(conf, tp, n_gt)
        per_iou[iou_t] = dict(conf=conf, tp=tp, kind=kind, heights=h, gt_conf=gc, recall=recall, precision=precision, ap=ap)
    main = per_iou[iou_thr]
    ops = [operating_point(main["conf"], main["tp"], main["kind"], n_gt, t, len(frames))
           for t in sorted(set([*thresholds, operating_thr]))]
    result = dict(
        frames=len(frames), gt_boxes=n_gt, detections=int(main["conf"].shape[0]),
        iou=iou_thr, operating_threshold=operating_thr,
        ap50=per_iou[0.5]["ap"] if 0.5 in per_iou else None,
        ap_at_iou=main["ap"],
        ap50_95=float(np.mean([per_iou[t]["ap"] for t in IOU_RANGE])),
        ap_per_iou={str(t): per_iou[t]["ap"] for t in IOU_RANGE},
        max_recall=(float(main["recall"][-1]) if main["recall"].shape[0] else 0.0),
        operating=next(o for o in ops if o["threshold"] == operating_thr),
        operating_points=ops,
        recall_by_size=recall_by_size(main["heights"], main["gt_conf"], operating_thr),
        curve=subsample_curve(main["recall"], main["precision"]),
        per_frame=per_frame_counts,
    )
    return result


# ---------------------------------------------------------------------------
# PRW 입출력
# ---------------------------------------------------------------------------
def load_frame_list(mat_path: Path) -> List[str]:
    import scipy.io
    d = scipy.io.loadmat(str(mat_path))
    key = [k for k in d if not k.startswith("_")][0]
    out = []
    for x in d[key].flatten():
        name = str(x[0]) if hasattr(x, "__len__") else str(x)
        out.append(name[:-4] if name.lower().endswith(".jpg") else name)
    return out


def load_annotation(ann_path: Path) -> np.ndarray:
    """(N,5) [pid, x, y, w, h]; 파일이 없거나 깨지면 (0,5)."""
    import scipy.io
    try:
        d = scipy.io.loadmat(str(ann_path))
        key = [k for k in d if not k.startswith("_")][0]
        arr = np.asarray(d[key], dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return arr.reshape(-1, 5) if arr.size else np.zeros((0, 5))
    except Exception:
        return np.zeros((0, 5))


def frames_for_split(data_root: Path, split: str, limit: int = 0, every: int = 1) -> List[str]:
    if split == "all":
        names = sorted(p.name[:-8] for p in (data_root / "annotations").glob("*.jpg.mat"))
    else:
        names = load_frame_list(data_root / f"frame_{split}.mat")
    if every > 1:
        names = names[::every]
    if limit and limit > 0:
        names = names[:limit]
    return names


def load_gt(ann_dir: Path, frames: Sequence[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """frame → (G,4) xywh (pid 무관, 전부 사람 박스). 통계에 미표기(pid −2) 수·주석 없는 프레임 수."""
    gt: Dict[str, np.ndarray] = {}
    stats = Counter()
    heights: List[float] = []
    for f in frames:
        path = ann_dir / f"{f}.jpg.mat"
        if not path.is_file():
            stats["frames_without_annotation"] += 1
            gt[f] = np.zeros((0, 4))
            continue
        ann = load_annotation(path)
        gt[f] = ann[:, 1:5].copy()
        stats["boxes"] += int(ann.shape[0])
        stats["boxes_unlabeled_pid"] += int(np.sum(ann[:, 0] == UNLABELED_PID))
        heights.extend(float(h) for h in ann[:, 4])
    stats["frames"] = len(frames)
    stats["height_histogram"] = {name: 0 for name, _, _ in SIZE_BUCKETS}
    for h in heights:
        stats["height_histogram"][size_bucket(h)] += 1
    return gt, dict(stats)


def frame_from_image_id(image_id: str) -> Optional[str]:
    name = str(image_id).replace("\\", "/").rsplit("/", 1)[-1]
    return name[:-4] if name.lower().endswith(".jpg") else None


def write_detections(path: Path, meta: Dict[str, Any], rows: Iterable[Dict[str, Any]], append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a" if append else "w", encoding="utf-8", newline="\n") as f:
        if not append:
            f.write(json.dumps({"_meta": True, **meta}, ensure_ascii=False) + "\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_detections(path: Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    """반환: meta, frame → (D,5) [x1,y1,x2,y2,conf], frame → 부가정보(width/height/latency_ms)."""
    meta: Dict[str, Any] = {}
    dets: Dict[str, np.ndarray] = {}
    extra: Dict[str, Dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            row = json.loads(ln)
            if row.get("_meta"):
                meta = row
                continue
            frame = row["frame"]
            arr = np.asarray([[float(v) for v in d[:5]] for d in row.get("dets", [])], dtype=np.float64).reshape(-1, 5)
            dets[frame] = arr
            extra[frame] = {k: row.get(k) for k in ("width", "height", "latency_ms")}
    return meta, dets, extra


def rewrite_meta(path: Path, meta: Dict[str, Any]) -> None:
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    body = [l for l in lines if l.strip() and not json.loads(l).get("_meta")]
    Path(path).write_text(json.dumps({"_meta": True, **meta}, ensure_ascii=False) + "\n" + "\n".join(body) + ("\n" if body else ""),
                          encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# 검출기 실행 / DB 가져오기
# ---------------------------------------------------------------------------
def parse_param(text: str) -> Tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"--param 은 key=value 형식: {text!r}")
    k, v = text.split("=", 1)
    try:
        return k.strip(), json.loads(v)
    except ValueError:
        return k.strip(), v


def resolve_detector_spec(config_path: Optional[str], module: Optional[str], cls: Optional[str],
                          params: Sequence[str], run_threshold: Optional[float]) -> Dict[str, Any]:
    spec: Dict[str, Any] = {}
    if config_path:
        import hashlib
        import yaml
        data = Path(config_path).read_bytes()
        raw = yaml.safe_load(data.decode("utf-8-sig")) or {}
        spec = dict(raw.get("detector") or {})
        spec["config_path"] = str(Path(config_path).resolve())     # 원장(bench/ledger) 에 yaml 식별 기록
        spec["config_sha256"] = hashlib.sha256(data).hexdigest()
    if module:
        spec["module"] = module
    if cls:
        spec["class"] = cls
    if not spec.get("module") or not spec.get("class"):
        raise ValueError("검출기 module/class 를 --detector-config(detector: 블록) 또는 --module/--class 로 지정")
    p = dict(spec.get("params") or {})
    for item in params:
        k, v = parse_param(item)
        p[k] = v
    spec["config_conf_threshold"] = p.get("conf_threshold")
    if run_threshold is not None:
        p["conf_threshold"] = float(run_threshold)
    spec["params"] = p
    return spec


def versions_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    import importlib.metadata as md
    for pkg in ("rfdetr", "ultralytics"):
        try:
            info[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return info


def run_detector(spec: Dict[str, Any], frames: Sequence[str], frames_dir: Path, class_name: str, out_path: Path,
                 resume: bool, log=print) -> Dict[str, Any]:
    import cv2
    from detect.loader import load_component

    done: set = set()
    if resume and out_path.is_file():
        _, existing, _ = read_detections(out_path)
        done = set(existing)
        log(f"[run] 이어서: 이미 {len(done):,} 프레임 완료")
    todo = [f for f in frames if f not in done]
    log(f"[run] 검출기 로드: {spec['module']}.{spec['class']} params={json.dumps(spec['params'], ensure_ascii=False)}")
    t0 = time.perf_counter()
    detector = load_component({k: spec[k] for k in ("module", "class", "params")})
    load_sec = time.perf_counter() - t0
    log(f"[run] 로드 {load_sec:.1f}s · 프레임 {len(todo):,} 개 처리 시작")

    def rows():
        started = time.perf_counter()
        lat: List[float] = []
        for i, f in enumerate(todo, 1):
            img = cv2.imread(str(frames_dir / f"{f}.jpg"))
            if img is None:
                log(f"  [경고] 프레임 없음: {f}")
                yield dict(frame=f, width=None, height=None, latency_ms=None, dets=[], missing=True)
                continue
            t1 = time.perf_counter()
            dets = detector.detect(img, frame_idx=i)
            ms = (time.perf_counter() - t1) * 1000.0
            if i > 1:
                lat.append(ms)
            out = [[round(float(d.bbox[0]), 2), round(float(d.bbox[1]), 2), round(float(d.bbox[2]), 2), round(float(d.bbox[3]), 2),
                    round(float(d.confidence), 5), int(d.class_id), str(d.class_name)]
                   for d in dets if str(d.class_name) == class_name]
            yield dict(frame=f, width=int(img.shape[1]), height=int(img.shape[0]), latency_ms=round(ms, 2), dets=out)
            if i % 200 == 0 or i == len(todo):
                el = time.perf_counter() - started
                log(f"  {i:,}/{len(todo):,}  {i / el:.1f} f/s  경과 {el:.0f}s")
        stats["latency_ms_mean"] = float(np.mean(lat)) if lat else None
        stats["latency_ms_p50"] = float(np.median(lat)) if lat else None
        stats["fps"] = (1000.0 / float(np.mean(lat))) if lat else None
        stats["elapsed_sec"] = time.perf_counter() - started

    stats: Dict[str, Any] = dict(load_sec=round(load_sec, 2))
    meta = dict(producer=PRODUCER, kind="run", created_at=datetime.now().astimezone().isoformat(), detector=spec,
                class_name=class_name, frames_requested=len(frames), versions=versions_info())
    n = write_detections(out_path, meta, rows(), append=resume and bool(done))
    stats.update(frames_done=n + len(done))
    meta.update(stats=stats)
    rewrite_meta(out_path, meta)
    log(f"[run] 완료: {out_path} · fps {stats.get('fps') and round(stats['fps'], 1)} · 평균 {stats.get('latency_ms_mean') and round(stats['latency_ms_mean'], 1)} ms")
    return meta


def import_from_qdrant(url: str, api_key: Optional[str], collection: str, sources: List[str], frames: Sequence[str],
                       out_path: Path, log=print) -> Dict[str, Any]:
    import requests
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    if api_key:
        s.headers.update({"api-key": api_key})
    flt = {"must": [{"key": "source", "match": {"any": sources}}]}
    by_frame: Dict[str, List[List[Any]]] = defaultdict(list)
    offset = None
    total = skipped = 0
    while True:
        body = {"limit": 1024, "with_payload": ["image_id", "bbox", "bbox_space", "score", "label"], "with_vector": False, "filter": flt}
        if offset is not None:
            body["offset"] = offset
        r = s.post(f"{url.rstrip('/')}/collections/{collection}/points/scroll", json=body, timeout=120)
        if not r.ok:
            raise RuntimeError(f"scroll -> {r.status_code}\n{r.text[:500]}")
        res = r.json().get("result") or {}
        pts = res.get("points") or []
        for p in pts:
            pl = p.get("payload") or {}
            total += 1
            frame = frame_from_image_id(pl.get("image_id", ""))
            bbox = pl.get("bbox")
            if frame is None or not isinstance(bbox, list) or len(bbox) != 4 or str(pl.get("bbox_space", "frame")) != "frame":
                skipped += 1
                continue
            by_frame[frame].append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),
                                    float(pl.get("score", 1.0)), -1, str(pl.get("label", "person"))])
        offset = res.get("next_page_offset")
        if total % 10240 == 0:
            log(f"  scroll {total:,}")
        if not pts or offset is None:
            break
    wanted = set(frames)
    frames_with = sum(1 for f in wanted if f in by_frame)
    log(f"[import] DB point {total:,} (건너뜀 {skipped:,}) · 대상 프레임 {len(wanted):,} 중 검출 있는 프레임 {frames_with:,}")
    meta = dict(producer=PRODUCER, kind="import-qdrant", created_at=datetime.now().astimezone().isoformat(),
                detector=dict(module="qdrant", **{"class": collection}, params=dict(sources=sources)),
                class_name="person", frames_requested=len(frames), qdrant_url=url, collection=collection,
                stats=dict(points=total, skipped=skipped, frames_with_detections=frames_with))
    write_detections(out_path, meta, (dict(frame=f, width=None, height=None, latency_ms=None, dets=by_frame.get(f, [])) for f in frames))
    return meta


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
CSS = """
body{font-family:'Malgun Gothic',system-ui,sans-serif;margin:24px;color:#1f2937;background:#fff}
h1{font-size:1.5em}h2{margin-top:1.6em;border-bottom:1px solid #e5e7eb;padding-bottom:4px}
table{border-collapse:collapse;margin:8px 0;font-size:.92em}th,td{border:1px solid #d1d5db;padding:4px 8px;text-align:right}
th{background:#f3f4f6;text-align:left}.muted{color:#6b7280}.bar{background:#93c5fd;height:12px}
.ex{margin:14px 0}.ex img{max-width:100%;border:1px solid #d1d5db}.legend span{display:inline-block;padding:1px 8px;margin-right:6px;border-radius:3px;color:#fff}
code{background:#f3f4f6;padding:1px 4px}
"""


def fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "예" if v else "아니오"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return esc(v)


def metrics_table(results: Dict[str, Dict[str, Any]], metas: Dict[str, Dict[str, Any]]) -> str:
    names = list(results)
    rows = [
        ("검출기", lambda n: f"{metas[n].get('detector', {}).get('module', '?')}.{metas[n].get('detector', {}).get('class', '?')}"),
        ("실행 임계값 (파일에 든 최소 conf)", lambda n: fmt(metas[n].get("detector", {}).get("params", {}).get("conf_threshold"))),
        ("운영 임계값 (동작점)", lambda n: fmt(results[n]["operating_threshold"], 2)),
        ("프레임 수", lambda n: fmt(results[n]["frames"])),
        ("GT 박스", lambda n: fmt(results[n]["gt_boxes"])),
        ("검출 수 (실행 임계값 이상)", lambda n: fmt(results[n]["detections"])),
        ("AP@0.5", lambda n: fmt(results[n]["ap50"])),
        ("AP@[.5:.95]", lambda n: fmt(results[n]["ap50_95"])),
        ("최대 재현율 (임계값 없이)", lambda n: fmt(results[n]["max_recall"])),
        ("정밀도 @운영", lambda n: fmt(results[n]["operating"]["precision"])),
        ("재현율 @운영", lambda n: fmt(results[n]["operating"]["recall"])),
        ("F1 @운영", lambda n: fmt(results[n]["operating"]["f1"])),
        ("FP @운영 (중복 / 배경)", lambda n: f"{fmt(results[n]['operating']['fp'])} ({fmt(results[n]['operating']['fp_duplicate'])} / {fmt(results[n]['operating']['fp_background'])})"),
        ("FN @운영", lambda n: fmt(results[n]["operating"]["fn"])),
        ("프레임당 검출 @운영", lambda n: fmt(results[n]["operating"]["dets_per_frame"], 2)),
        ("평균 지연 ms / FPS", lambda n: (lambda s: f"{fmt(s.get('latency_ms_mean'), 1)} / {fmt(s.get('fps'), 1)}")(metas[n].get("stats") or {})),
        ("GPU", lambda n: esc((metas[n].get("versions") or {}).get("gpu", "—"))),
    ]
    head = "".join(f"<th>{esc(n)}</th>" for n in names)
    body = "".join(f"<tr><th>{esc(label)}</th>" + "".join(f"<td>{fn(n)}</td>" for n in names) + "</tr>" for label, fn in rows)
    return f"<table><tr><th></th>{head}</tr>{body}</table>"


def operating_table(res: Dict[str, Any]) -> str:
    head = "<tr><th>conf ≥</th><th>검출</th><th>TP</th><th>FP(중복)</th><th>FP(배경)</th><th>FN</th><th>정밀도</th><th>재현율</th><th>F1</th><th>프레임당 검출</th></tr>"
    body = "".join(
        f"<tr><th>{o['threshold']:.2f}{' ★' if o['threshold'] == res['operating_threshold'] else ''}</th><td>{fmt(o['dets'])}</td><td>{fmt(o['tp'])}</td>"
        f"<td>{fmt(o['fp_duplicate'])}</td><td>{fmt(o['fp_background'])}</td><td>{fmt(o['fn'])}</td><td>{fmt(o['precision'])}</td>"
        f"<td>{fmt(o['recall'])}</td><td>{fmt(o['f1'])}</td><td>{fmt(o['dets_per_frame'], 2)}</td></tr>" for o in res["operating_points"])
    return f"<table>{head}{body}</table>"


def size_table(results: Dict[str, Dict[str, Any]]) -> str:
    names = list(results)
    head = "<tr><th>GT 높이(px)</th><th>GT 수</th>" + "".join(f"<th>{esc(n)} @운영</th><th>{esc(n)} max</th>" for n in names) + "</tr>"
    first = results[names[0]]["recall_by_size"]
    body = ""
    for bucket, _, _ in SIZE_BUCKETS:
        body += f"<tr><th>{esc(bucket)}</th><td>{fmt(first[bucket]['gt'])}</td>"
        for n in names:
            b = results[n]["recall_by_size"][bucket]
            body += f"<td>{fmt(b['recall'])}</td><td>{fmt(b['recall_any'])}</td>"
        body += "</tr>"
    return f"<table>{head}{body}</table>"


def pr_svg(results: Dict[str, Dict[str, Any]], width: int = 520, height: int = 400) -> str:
    m = 44
    w, h = width - 2 * m, height - 2 * m
    def x(r): return m + r * w
    def y(p): return m + (1 - p) * h
    grid = "".join(f"<line x1='{x(v):.1f}' y1='{y(0):.1f}' x2='{x(v):.1f}' y2='{y(1):.1f}' stroke='#e5e7eb'/>"
                   f"<line x1='{x(0):.1f}' y1='{y(v):.1f}' x2='{x(1):.1f}' y2='{y(v):.1f}' stroke='#e5e7eb'/>"
                   f"<text x='{x(v):.1f}' y='{y(0) + 16:.1f}' font-size='11' text-anchor='middle'>{v:.1f}</text>"
                   f"<text x='{x(0) - 6:.1f}' y='{y(v) + 4:.1f}' font-size='11' text-anchor='end'>{v:.1f}</text>"
                   for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    lines, legend = [], []
    for i, (name, res) in enumerate(results.items()):
        color = PALETTE[i % len(PALETTE)]
        pts = " ".join(f"{x(r):.1f},{y(p):.1f}" for r, p in res["curve"])
        if pts:
            lines.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='2'/>")
        op = res["operating"]
        if op["precision"] is not None and op["recall"] is not None:
            lines.append(f"<circle cx='{x(op['recall']):.1f}' cy='{y(op['precision']):.1f}' r='4' fill='{color}'/>")
        legend.append(f"<tspan x='{m + 8}' dy='{16 if i else 0}' fill='{color}'>■ {esc(name)} (AP@{res['iou']:.2f} {fmt(res['ap_at_iou'], 3)})</tspan>")
    return (f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' style='border:1px solid #d1d5db;background:#fff'>{grid}"
            f"<text x='{x(0.5):.1f}' y='{height - 6}' font-size='12' text-anchor='middle'>recall</text>"
            f"<text x='12' y='{y(0.5):.1f}' font-size='12' text-anchor='middle' transform='rotate(-90 12 {y(0.5):.1f})'>precision</text>"
            f"{''.join(lines)}<text x='{m + 8}' y='{m + 14}' font-size='12'>{''.join(legend)}</text></svg>")


def render_examples(name: str, frames_dir: Path, gt_by_frame: Dict[str, np.ndarray], dets_by_frame: Dict[str, np.ndarray],
                    res: Dict[str, Any], out_dir: Path, n: int, inline: bool, html_path: Path, log=print) -> str:
    """운영 임계값에서 FN 이 많은 프레임 n 개: GT 초록 · TP 파랑 · FP 빨강 · FN 주황."""
    if n <= 0:
        return ""
    from PIL import Image, ImageDraw
    ranked = sorted(res["per_frame"].items(), key=lambda kv: (-kv[1]["fn"], -kv[1]["fp"], kv[0]))[:n]
    ex_dir = out_dir / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    parts = ["<p class='legend'><span style='background:#16a34a'>GT</span><span style='background:#2563eb'>TP</span>"
             "<span style='background:#dc2626'>FP</span><span style='background:#ea580c'>FN (놓친 GT)</span></p>"]
    thr = res["operating_threshold"]
    for frame, c in ranked:
        img_path = frames_dir / f"{frame}.jpg"
        if not img_path.is_file():
            parts.append(f"<div class='ex'><b>{esc(frame)}</b> · 프레임 파일 없음</div>")
            continue
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        gt = xywh_to_xyxy(gt_by_frame.get(frame, np.zeros((0, 4))))
        dets = dets_by_frame.get(frame, np.zeros((0, 5)))
        dets = dets[dets[:, 4] >= thr] if dets.shape[0] else dets
        m = match_frame(dets, gt, res["iou"])
        for j, box in enumerate(gt):
            missed = not (m["gt_conf"][j] >= thr) if gt.shape[0] else True
            draw.rectangle([box[0], box[1], box[2], box[3]], outline="#ea580c" if missed else "#16a34a", width=4 if missed else 2)
        order = np.argsort(-dets[:, 4], kind="stable") if dets.shape[0] else np.zeros(0, dtype=int)
        for k, i in enumerate(order):
            box = dets[i]
            draw.rectangle([box[0], box[1], box[2], box[3]], outline="#2563eb" if m["tp"][k] else "#dc2626", width=2)
        scale = min(1.0, 1280 / img.width)
        if scale < 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        out = ex_dir / f"{name}_{frame}.jpg"
        img.save(out, "JPEG", quality=80)
        if inline:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=70)
            src = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        else:
            src = path_href(out, html_path)
        parts.append(f"<div class='ex'><b>{esc(frame)}</b> · GT {c['gt']} · TP {c['tp']} · FP {c['fp']} · FN {c['fn']}<br><img src='{src}' alt='{esc(frame)}'></div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["run", "import-qdrant", "score"], default="run")
    p.add_argument("--name", default=None, help="run/import 결과 이름 (<output-dir>/<name>/detections.jsonl). 기본: 검출기 class 이름")
    p.add_argument("--data-root", default="./data/PRW")
    p.add_argument("--split", choices=["test", "train", "all"], default="test", help="frame_test.mat / frame_train.mat / 주석 있는 전부")
    p.add_argument("--limit", type=int, default=0, help="프레임 수 제한 (0 = 전부)")
    p.add_argument("--every", type=int, default=1, help="프레임 간격 (2 면 절반)")
    p.add_argument("--output-dir", default="eval/results/detect_prw")
    p.add_argument("--html-name", default="detect_eval_prw.html")
    p.add_argument("--title", default=None)
    p.add_argument("--iou", type=float, default=0.5, help="TP 판정 IoU (AP@[.5:.95] 는 항상 계산)")
    p.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS), help="동작점 표의 conf 임계값들 (쉼표)")
    p.add_argument("--operating-threshold", type=float, default=None,
                   help="운영 동작점 conf. 기본: 검출기 yaml 의 conf_threshold, 없으면 0.5 (import-qdrant 는 0.5)")
    p.add_argument("--examples", type=int, default=6, help="FN 많은 프레임 그림 수 (0 = 없음)")
    p.add_argument("--inline-images", action="store_true")
    p.add_argument("--no-images", action="store_true")
    p.add_argument("--no-score", action="store_true", help="run/import 만 하고 채점 생략")
    # run
    p.add_argument("--detector-config", default=str(DEFAULT_DETECTOR_CONFIG), help="detector: {module, class, params} yaml")
    p.add_argument("--module", default=None, help="검출기 module (yaml 대신/덮어쓰기)")
    p.add_argument("--class", dest="cls", default=None, help="검출기 class")
    p.add_argument("--param", action="append", default=[], help="params 덮어쓰기 key=value (여러 번)")
    p.add_argument("--conf-threshold", type=float, default=0.05, help="실행 임계값 (낮게 두고 AP 를 스윕). params.conf_threshold 로 전달")
    p.add_argument("--class-name", default="person", help="채점할 클래스 이름 (PRW GT 는 사람만)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--compare", action="append", default=[], nargs="*", help="같이 채점할 이름=detections.jsonl (여러 개)")
    # import-qdrant
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=CONFIG_HELP)
    p.add_argument("--collection", default=None, help=CONFIG_HELP)
    p.add_argument("--qdrant-url", default=None, help=CONFIG_HELP)
    p.add_argument("--api-key", default=None)
    p.add_argument("--sources", default="prw_image")
    # score
    p.add_argument("--method", action="append", default=[], nargs="*", help="이름=detections.jsonl (score 모드)")
    p.add_argument("--ledger", default=None, help="실행 원장 JSONL (기본 bench/ledger.jsonl; '' 또는 none 이면 기록 안 함)")
    p.add_argument("--no-ledger", action="store_true", help="원장(bench/ledger.jsonl)에 기록하지 않음")
    return p


def parse_named(items: Sequence[Any]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    flat: List[str] = []
    for it in items:
        flat.extend(it if isinstance(it, (list, tuple)) else [it])
    for it in flat:
        if not str(it).strip():
            continue
        if "=" not in it:
            raise ValueError(f"이름=경로 형식이어야 합니다: {it!r}")
        name, path = it.split("=", 1)
        out[name.strip()] = Path(path.strip())
    return out


def score(methods: Dict[str, Path], args, warnings: List[Dict[str, Any]], log=print) -> Dict[str, Any]:
    started = time.time()
    data_root = Path(args.data_root).expanduser().resolve()
    ann_dir = data_root / "annotations"
    frames_dir = data_root / "frames"
    if not ann_dir.is_dir():
        raise SystemExit(f"주석 폴더가 없습니다: {ann_dir}")
    thresholds = [float(t) for t in str(args.thresholds).split(",") if t.strip()]
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / args.html_name
    sidecar = out_dir / "detect_eval_report.json"

    metas: Dict[str, Dict[str, Any]] = {}
    dets_all: Dict[str, Dict[str, np.ndarray]] = {}
    inputs = []
    for name, path in methods.items():
        if not path.is_file():
            raise SystemExit(f"{name}: 파일 없음 {path}")
        meta, dets, _ = read_detections(path)
        metas[name] = meta
        dets_all[name] = dets
        inputs.append(file_info(f"detections:{name}", path))
        log(f"[score] {name}: {len(dets):,} 프레임, {sum(d.shape[0] for d in dets.values()):,} 검출 ({path})")
    frame_sets = [set(d) for d in dets_all.values()]
    frames = sorted(set.intersection(*frame_sets)) if frame_sets else []
    if len(frame_sets) > 1 and any(len(s) != len(frames) for s in frame_sets):
        detail = ", ".join(f"{n}={len(s):,}" for n, s in zip(dets_all, frame_sets))
        warnings.append(warning("FRAME_SET_MISMATCH", f"프레임 집합이 달라 교집합 {len(frames):,} 프레임에서만 비교 ({detail})"))
        log(f"[score] 경고: {warnings[-1]['message']}")
    if not frames:
        raise SystemExit("채점할 공통 프레임이 없습니다")
    gt_by_frame, gt_stats = load_gt(ann_dir, frames)
    if gt_stats.get("frames_without_annotation"):
        warnings.append(warning("FRAMES_WITHOUT_ANNOTATION", f"주석 없는 프레임 {gt_stats['frames_without_annotation']:,} (GT 0 으로 처리)"))
    log(f"[score] GT: 프레임 {gt_stats['frames']:,} · 박스 {gt_stats['boxes']:,} (pid −2 {gt_stats['boxes_unlabeled_pid']:,}) · 높이 분포 {gt_stats['height_histogram']}")

    results: Dict[str, Dict[str, Any]] = {}
    for name in methods:
        meta = metas[name]
        cfg_thr = (meta.get("detector") or {}).get("config_conf_threshold")
        op_thr = args.operating_threshold if args.operating_threshold is not None else (
            0.5 if meta.get("kind") == "import-qdrant" else (float(cfg_thr) if cfg_thr is not None else 0.5))
        res = evaluate_method(dets_all[name], gt_by_frame, frames, args.iou, op_thr, thresholds)
        results[name] = res
        o = res["operating"]
        log(f"  {name:<18} AP@{args.iou:.2f} {fmt(res['ap_at_iou'])} · AP@[.5:.95] {fmt(res['ap50_95'])} · max R {fmt(res['max_recall'])} · "
            f"@{op_thr:.2f}: P {fmt(o['precision'])} R {fmt(o['recall'])} F1 {fmt(o['f1'])} (FP {o['fp']:,} = 중복 {o['fp_duplicate']:,} + 배경 {o['fp_background']:,}, FN {o['fn']:,})")

    examples_block = ""
    if not args.no_images and args.examples > 0:
        for name in methods:
            examples_block += f"<h3>{esc(name)}: 놓친 GT(FN) 가 많은 프레임</h3>"
            examples_block += render_examples(name, frames_dir, gt_by_frame, dets_all[name], results[name], out_dir,
                                              args.examples, args.inline_images, html_path, log)

    title = args.title or f"PRW GT 기반 검출 정확도 (person, IoU ≥ {args.iou})"
    per_method_ops = "".join(f"<h3>{esc(n)}</h3>{operating_table(r)}" for n, r in results.items())
    gt_table = table({"프레임": gt_stats["frames"], "GT 박스": gt_stats["boxes"], "그중 pid −2 (미표기 보행자)": gt_stats["boxes_unlabeled_pid"],
                      "주석 없는 프레임": gt_stats.get("frames_without_annotation", 0)})
    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>{esc(title)}</title><style>{CSS}</style></head><body>
<h1>{esc(title)}</h1>
<p class="muted">GT = PRW annotations (모든 행 = 사람 박스, pid −2 포함) · 매칭 = conf 내림차순 greedy, IoU ≥ {args.iou} · 생성 {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
<h2>1. GT</h2>{gt_table}<h3>GT 높이(px) 분포</h3>{histogram_html(gt_stats['height_histogram'])}
<h2>2. 검출기별 요약</h2>{metrics_table(results, metas)}
<p class="muted">AP 는 실행 임계값 이상의 검출 전부로 스윕한 값이라 실행 임계값이 높은 파일(예: 운영 DB 0.5)은 곡선이 잘려 낮게 나온다.
FP(중복) = 이미 잡힌 GT 와 겹치는 검출(NMS 문제), FP(배경) = 어떤 GT 와도 IoU 미달.</p>
<h2>3. PR 곡선 (IoU {args.iou}) · ● 운영 동작점</h2>{pr_svg(results)}
<h2>4. 동작점별 정밀도 / 재현율 (★ 운영 임계값)</h2>{per_method_ops}
<h2>5. GT 높이별 재현율 (@운영 임계값 / 임계값 없이)</h2>{size_table(results)}
<p class="muted">crop 최소 높이(PRW DB 75 px, COCO 120 px) 아래 구간은 DB 에 들어가지 않으므로 검색 재현율의 상한이 된다.</p>
<h2>6. 예시</h2>{examples_block or '<p class="muted">생략</p>'}
<h2>입력</h2>{table({i['role']: {'path': i['path'], 'sha256': (i.get('sha256') or '')[:16]} for i in inputs})}
<h2>경고</h2>{('<ul>' + ''.join(f'<li>{esc(w["message"])}</li>' for w in warnings) + '</ul>') if warnings else '<p class="muted">없음</p>'}
</body></html>"""
    html_path.write_text(html, encoding="utf-8", newline="\n")

    summary = common_summary(PRODUCER, html_path, inputs, warnings)
    slim = {n: {k: v for k, v in r.items() if k != "per_frame"} for n, r in results.items()}
    summary.update(config=dict(methods={k: str(v) for k, v in methods.items()}, iou=args.iou, thresholds=thresholds,
                               data_root=str(data_root), frames=len(frames)),
                   gt=gt_stats, metas={n: {k: v for k, v in m.items() if k != "_meta"} for n, m in metas.items()},
                   results=slim, elapsed_sec=round(time.time() - started, 3))
    write_json(sidecar, summary)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method", "frames", "gt_boxes", "detections", "AP50", "AP50_95", "max_recall", "operating_threshold",
                    "precision", "recall", "f1", "fp_duplicate", "fp_background", "fn", "dets_per_frame", "latency_ms_mean", "fps"]
                   + [f"recall_h{b}" for b, _, _ in SIZE_BUCKETS])
        for n, r in results.items():
            o, st = r["operating"], (metas[n].get("stats") or {})
            w.writerow([n, r["frames"], r["gt_boxes"], r["detections"], r["ap50"], r["ap50_95"], r["max_recall"], r["operating_threshold"],
                        o["precision"], o["recall"], o["f1"], o["fp_duplicate"], o["fp_background"], o["fn"], o["dets_per_frame"],
                        st.get("latency_ms_mean"), st.get("fps")] + [r["recall_by_size"][b]["recall"] for b, _, _ in SIZE_BUCKETS])
    log(f"html    : {html_path}")
    log(f"sidecar : {sidecar}")
    from bench import ledger
    ledger.record(lambda: ledger.entries_from_detect_summary(summary, report=sidecar, command=getattr(args, "command_line", None)),
                  getattr(args, "ledger", None), getattr(args, "no_ledger", False), log=log)
    result_markers(sidecar, html_path)
    return summary


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    args.command_line = list(argv) if argv is not None else sys.argv[1:]
    warnings: List[Dict[str, Any]] = []
    try:
        if args.limit < 0 or args.every < 1:
            raise ValueError("--limit 은 0 이상, --every 는 1 이상")
        if not (0 <= args.iou <= 1):
            raise ValueError("--iou 는 0~1")
        for t in str(args.thresholds).split(","):
            if t.strip() and not (0 <= float(t) <= 1):
                raise ValueError("--thresholds 는 0~1 값들")
        html_name = Path(args.html_name)
        if html_name.name != args.html_name or not args.html_name.lower().endswith((".html", ".htm")):
            raise ValueError("html-name 은 하위 경로 없는 .html 파일명")
        compare = parse_named(args.compare)
        methods = parse_named(args.method)
        if args.mode == "score" and not methods:
            raise ValueError("score 모드는 --method 이름=경로 가 필요")
        if args.mode == "run":
            spec = resolve_detector_spec(args.detector_config, args.module, args.cls, args.param, args.conf_threshold)
    except (OSError, ValueError, AttributeError) as exc:
        p.error(str(exc))

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.output_dir).resolve()
    if args.mode in ("run", "import-qdrant"):
        frames = frames_for_split(data_root, args.split, args.limit, args.every)
        if not frames:
            raise SystemExit("평가할 프레임이 없습니다")
        print(f"[frames] split={args.split} every={args.every} limit={args.limit or '전부'} → {len(frames):,} 프레임")
        if args.mode == "run":
            name = args.name or str(spec["class"]).lower().replace("detector", "") or "detector"
            out_path = out_dir / name / "detections.jsonl"
            meta = run_detector(spec, frames, data_root / "frames", args.class_name, out_path, args.resume)
            meta["split"] = dict(split=args.split, limit=args.limit, every=args.every)
            rewrite_meta(out_path, meta)
            write_json(out_dir / name / "run_meta.json", meta)
        else:
            settings = load_pipeline_settings(args.config, require=not (args.qdrant_url and args.collection))
            url = resolve(args.qdrant_url, settings.qdrant_url if settings else None, "qdrant-url")
            collection = resolve(args.collection, settings.person_collection if settings else None, "collection")
            sources = [s.strip() for s in args.sources.split(",") if s.strip()]
            name = args.name or "prod_db"
            out_path = out_dir / name / "detections.jsonl"
            meta = import_from_qdrant(url, args.api_key, collection, sources, frames, out_path)
            meta["split"] = dict(split=args.split, limit=args.limit, every=args.every)
            rewrite_meta(out_path, meta)
            write_json(out_dir / name / "run_meta.json", meta)
        print(f"[{args.mode}] detections: {out_path}")
        if args.no_score:
            return None
        methods = {name: out_path, **compare}
    return score(methods, args, warnings)


if __name__ == "__main__":
    main()
