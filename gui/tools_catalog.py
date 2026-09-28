#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gui/tools_catalog.py — '도구' 페이지의 데이터: 단계별 후보 도구 + 원장(bench/ledger.jsonl) 성적 + ★ 최고.

단계
  detector  검출기       후보 = pipeline*.yaml 의 detector 블록(사본 합침)      성적 = 원장 detect (ap50)
  embedder  임베더       후보 = pipeline.yaml retrievers (person)                성적 = 원장 embed (map)
  search    검색 조합    후보 = 원장 search 의 combo:… + 운영 기본                 성적 = 전체 파이프라인(e2e) map 우선, 없으면 검색 단계 map
  clusterer 클러스터러   후보 = gui_pipelines.json image_cluster tools           성적 = 원장 cluster (b3_f1), 방법(leiden/dbscan)으로 매칭
  qwen      AI 재확인    후보 = 단건 / 배치 10                                   성적 = 원장 qwen (라벨 후)
★ = 상태(pass > partial > incomplete > fail) 우선, 같은 상태면 주 지표가 큰 것. study:/__verify/smoke 이름은 제외.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
STATUS_ORDER = {"pass": 0, "partial": 1, "incomplete": 2, "fail": 3, "n-a": 4}
MODEL_LABELS = {"siglip2": "SigLIP2", "irra": "IRRA", "solider": "SOLIDER", "dinov2": "DINOv2", "solider_copy": "SOLIDER(사본)"}
PROD_SEARCH = {"stage1": ["siglip2", "irra"], "rerank": "solider", "pool": 200}
_CACHE: Dict[str, Any] = {}

STEPS = [
    {"key": "detector", "title": "검출기", "stage": "detect", "group": "image_pipeline", "stage_id": "image_detect",
     "desc": "사진·영상에서 사람과 물건을 찾는 모델. 사진 처리 1단계(검출)의 '검출기' 기본값이 된다 (영상은 추적 설정 yaml 로 고른다)."},
    {"key": "video_tracking", "title": "영상 검출·추적 (검출기 + 추적기 + 스티처)", "stage": "track", "group": "video_pipeline", "stage_id": "video_preprocess",
     "desc": "영상 처리 1단계의 조합. 영상 처리 1단계(검출·추적)의 '도구' 기본값이 된다. 추적 성적(IDF1·HOTA)은 정답 라벨(추적 시트) 이후 원장에 쌓인다."},
    {"key": "embedder", "title": "임베더 (사람 재식별)", "stage": "embed", "group": "image_pipeline", "stage_id": "image_build",
     "desc": "사람 crop 을 벡터로 바꾸는 모델. 영상 검색의 사진 검색 모델 기본값이 된다. 임베딩 단계는 yaml 의 임베더를 전부 만든다."},
    {"key": "search", "title": "검색 조합 (사진에서 찾기)", "stage": "search", "group": None, "stage_id": None,
     "desc": "1차 후보 모델(들) → 2차 재정렬 → 후보 수. 전체 파이프라인 평가(e2e)가 있으면 그 mAP 로, 없으면 검색 단계 mAP 로 순위를 매긴다."},
    {"key": "clusterer", "title": "클러스터러", "stage": "cluster", "group": "image_pipeline", "stage_id": "image_cluster",
     "desc": "같은 사람·물건을 묶는 알고리즘. 사진 처리 3단계(클러스터)의 '도구' 기본값이 된다."},
    {"key": "qwen", "title": "AI 재확인 (Qwen)", "stage": "qwen", "group": None, "stage_id": None,
     "desc": "검색 결과를 다시 보고 조건에 맞는지 판정하는 방식. 성적은 정답 라벨(Qwen 시트) 이후에 생긴다."},
]


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def status_of(entry: Dict[str, Any]) -> str:
    try:
        from bench import criteria as C

        return str(C.evaluate(entry).get("status") or "n-a")
    except Exception:  # noqa: BLE001
        return "n-a"


def primary_metric(stage: str) -> Optional[str]:
    try:
        from bench import criteria as C

        return C.CHART_AXES.get(stage, (None, None))[1]
    except Exception:  # noqa: BLE001
        return None


def read_latest(ledger_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    from bench import ledger as L

    p = Path(ledger_path) if ledger_path else ROOT / "bench" / "ledger.jsonl"
    if not p.is_file():
        return {}
    return L.latest_by_name(L.read_entries(p))


def _skip(name: str) -> bool:
    n = name.lower()
    return n.startswith("study:") or "__verify" in n or n.startswith("smoke") or "__sample" in n


def rank_key(entry: Dict[str, Any], metric: Optional[str]) -> tuple:
    v = (entry.get("metrics") or {}).get(metric) if metric else None
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = float("-inf")
    return (STATUS_ORDER.get(status_of(entry), 9), -v)


def best_entry(latest: Dict[str, Dict[str, Any]], stage: str, match: Callable[[Dict[str, Any]], bool]) -> Optional[Dict[str, Any]]:
    metric = primary_metric(stage)
    rows = [e for n, e in latest.items() if e.get("stage") == stage and not _skip(n) and match(e)]
    if not rows:
        return None
    return min(rows, key=lambda e: rank_key(e, metric))


def _score(entry: Optional[Dict[str, Any]], metric: Optional[str]) -> Dict[str, Any]:
    if entry is None:
        return {"entry_name": None, "metric": metric, "metric_value": None, "status": None}
    v = (entry.get("metrics") or {}).get(metric) if metric else None
    return {"entry_name": entry.get("name"), "metric": metric, "metric_value": v, "status": status_of(entry)}


# ---------------------------------------------------------------- 후보
def detector_candidates(root: Path = ROOT) -> List[Dict[str, Any]]:
    from gui import pipeline_page as pp

    spec = {"type": "choice", "default": "pipeline.yaml", "choices_glob": "pipeline*.yaml", "choices_yaml_key": "detector",
            "choices_unique_by": "*", "choices_label_key": "class"}
    out = []
    for value, label in pp._choice_values(spec, root):
        block = pp._yaml_block(root / value, "detector") or {}
        cls = str(block.get("class") or "")
        params = block.get("params") or {}
        blob = " ".join(str(v) for v in params.values()).lower()
        token = next((t for t in ("yolo26s", "yolo26m", "yolo26l", "yolo26n", "yolo26x", "rfdetr_small", "rfdetr_medium", "rfdetr_large") if t in blob), "")
        out.append({"key": value, "label": label, "value": value, "class": cls, "token": token,
                    "detail": f"{block.get('module', '')}.{cls}" + (f" · {params}" if params else "")})
    return out


def _detector_match(c: Dict[str, Any]) -> Callable[[Dict[str, Any]], bool]:
    cls = c["class"].lower()
    token = c["token"]

    def m(e: Dict[str, Any]) -> bool:
        comp = e.get("component") or {}
        if str(comp.get("class") or "").lower() != cls:
            return False
        return (token in str(e.get("name") or "").lower()) if token else True
    return m


def embedder_candidates(config: Optional[Path] = None) -> List[Dict[str, Any]]:
    import yaml

    cfg = Path(config) if config else ROOT / "pipeline.yaml"
    try:
        raw = yaml.safe_load(cfg.read_text(encoding="utf-8-sig")) or {}
    except Exception:  # noqa: BLE001
        return []
    out = []
    for name, block in (raw.get("retrievers") or {}).items():
        scope = str((block or {}).get("scope") or "all")
        if scope not in ("person", "all"):
            continue
        out.append({"key": name, "label": model_label(name), "value": name, "scope": scope,
                    "detail": f"scope {scope}" + (" · 자연어 가능" if (block or {}).get("supports_text") else "")})
    return out


COMBO_RE = re.compile(r"^combo:(?P<s1>[^→]+)→(?P<rr>[^@]+)@(?P<pool>\d+)$")


def parse_combo(name: str) -> Optional[Dict[str, Any]]:
    m = COMBO_RE.match(name)
    if not m:
        return None
    rr = m.group("rr")
    return {"stage1": m.group("s1").split("+"), "rerank": None if rr == "none" else rr, "pool": int(m.group("pool"))}


def combo_label(v: Dict[str, Any]) -> str:
    return " + ".join(model_label(x) for x in v["stage1"]) + " → " + (model_label(v["rerank"]) if v.get("rerank") else "재정렬 없음") + f" · 후보 {v.get('pool', '')}"


def _e2e_for(latest: Dict[str, Dict[str, Any]], v: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rows = []
    for n, e in latest.items():
        if e.get("stage") != "e2e" or _skip(n):
            continue
        comp = e.get("component") or {}
        if list(comp.get("stage1") or []) == list(v["stage1"]) and (comp.get("rerank") or None) == (v.get("rerank") or None):
            rows.append(e)
    return min(rows, key=lambda e: rank_key(e, "map")) if rows else None


def search_candidates(latest: Dict[str, Dict[str, Any]], top: int = 6) -> List[Dict[str, Any]]:
    combos: Dict[str, Dict[str, Any]] = {}
    for n, e in latest.items():
        if e.get("stage") != "search" or _skip(n):
            continue
        v = parse_combo(n)
        if v is None:
            continue
        key = f"{'+'.join(v['stage1'])}→{v['rerank'] or 'none'}@{v['pool']}"
        combos[key] = {"key": key, "label": combo_label(v), "value": v, "search_entry": e}
    prod_key = f"{'+'.join(PROD_SEARCH['stage1'])}→{PROD_SEARCH['rerank']}@{PROD_SEARCH['pool']}"
    if prod_key not in combos:
        combos[prod_key] = {"key": prod_key, "label": combo_label(PROD_SEARCH), "value": dict(PROD_SEARCH), "search_entry": None}
    rows = list(combos.values())
    for c in rows:
        c["e2e_entry"] = _e2e_for(latest, c["value"])
        c["prod"] = c["key"] == prod_key
    # e2e 가 있는 것 먼저(그 안에서 상태·map), 그다음 검색 단계 성적
    rows.sort(key=lambda c: (0, rank_key(c["e2e_entry"], "map")) if c["e2e_entry"] else (1, rank_key(c["search_entry"], "map") if c["search_entry"] else (9, 0.0)))
    keep = rows[:top]
    if not any(c["prod"] for c in keep):
        keep.append(next(c for c in rows if c["prod"]))
    out = []
    for c in keep:
        e2e, se = c["e2e_entry"], c["search_entry"]
        score = _score(e2e, "map") if e2e else _score(se, "map")
        score["metric"] = "e2e map" if e2e else "search map"
        out.append({"key": c["key"], "label": c["label"] + (" (운영 기본)" if c["prod"] else ""), "value": c["value"],
                    "detail": (f"검색 단계 mAP {float((se.get('metrics') or {}).get('map')):.2f}" if se and (se.get('metrics') or {}).get('map') is not None else "검색 단계 성적 없음")
                              + (f" · 전체 파이프라인 mAP {float((e2e.get('metrics') or {}).get('map')):.2f}" if e2e else ""),
                    **score})
    return out


def stage_tool_candidates(root: Path, group_id: str, stage_id: str, flag: str) -> List[Dict[str, Any]]:
    """gui_pipelines.json 의 stage.tools 를 후보로 (key = 라벨, value = set[flag])."""
    from gui import pipeline_page as pp

    try:
        groups = {g["id"]: g for g in pp.load_registry(root / "gui_pipelines.json")}
        stage = next(s for s in groups[group_id]["stages"] if s["id"] == stage_id)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for t in stage.get("tools") or []:
        label = str(t.get("label") or "")
        value = (t.get("set") or {}).get(flag)
        out.append({"key": label, "label": label, "value": value, "detail": (f"{flag} {value}" if value else "") + (f" · {t['description']}" if t.get("description") else "")})
    return out


def _tracking_match(c: Dict[str, Any]) -> Callable[[Dict[str, Any]], bool]:
    want = Path(str(c.get("value") or "")).name.lower()

    def m(e: Dict[str, Any]) -> bool:
        got = str((e.get("params") or {}).get("tracking_config") or (e.get("component") or {}).get("tracking_config") or "")
        return bool(want) and Path(got).name.lower() == want
    return m


def cluster_candidates(root: Path = ROOT) -> List[Dict[str, Any]]:
    from gui import pipeline_page as pp

    try:
        groups = {g["id"]: g for g in pp.load_registry(root / "gui_pipelines.json")}
        stage = next(s for s in groups["image_pipeline"]["stages"] if s["id"] == "image_cluster")
    except Exception:  # noqa: BLE001
        return []
    out = []
    for t in stage.get("tools") or []:
        label = str(t.get("label") or "")
        method = "dbscan" if "dbscan" in label.lower() else ("leiden" if "leiden" in label.lower() else label.lower())
        out.append({"key": label, "label": label, "value": label, "method": method, "detail": f"단계 {t.get('stage')}" + (f" · {t.get('set')}" if t.get("set") else "")})
    return out


def _cluster_match(c: Dict[str, Any]) -> Callable[[Dict[str, Any]], bool]:
    method = c["method"]

    def m(e: Dict[str, Any]) -> bool:
        comp = e.get("component") or {}
        blob = " ".join(str(x) for x in (comp.get("class"), comp.get("method"), (comp.get("axes") or {}).get("method"), e.get("name"))).lower()
        return method in blob
    return m


def qwen_candidates() -> List[Dict[str, Any]]:
    return [{"key": "single", "label": "단건 (한 장씩)", "value": {"batch_size": 1}, "detail": "평가 기준 · 2B 기준 후보당 약 22초"},
            {"key": "b10", "label": "배치 10", "value": {"batch_size": 10}, "detail": "후보당 2~4초 · 판정이 일부 달라짐 (라벨 후 비교)"}]


def _qwen_match(c: Dict[str, Any]) -> Callable[[Dict[str, Any]], bool]:
    bs = int(c["value"]["batch_size"])

    def m(e: Dict[str, Any]) -> bool:
        return int((e.get("component") or {}).get("batch_size") or 1) == bs
    return m


# ---------------------------------------------------------------- 카탈로그
def catalog(root: Path = ROOT, ledger_path: Optional[Path] = None, config: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    latest = read_latest(ledger_path)
    out: Dict[str, Dict[str, Any]] = {}
    for step in STEPS:
        key, stage = step["key"], step["stage"]
        metric = primary_metric(stage)
        if key == "detector":
            cands = detector_candidates(root)
            for c in cands:
                c.update(_score(best_entry(latest, stage, _detector_match(c)), metric))
        elif key == "embedder":
            cands = embedder_candidates(config or (root / "pipeline.yaml"))
            for c in cands:
                c.update(_score(best_entry(latest, stage, lambda e, n=c["key"]: str(e.get("name")) == n), metric))
        elif key == "search":
            cands = search_candidates(latest)
        elif key == "video_tracking":
            cands = stage_tool_candidates(root, "video_pipeline", "video_preprocess", "--tracking-config")
            for c in cands:
                c.update(_score(best_entry(latest, stage, _tracking_match(c)), metric))
        elif key == "clusterer":
            cands = cluster_candidates(root)
            for c in cands:
                c.update(_score(best_entry(latest, stage, _cluster_match(c)), metric))
        else:
            cands = qwen_candidates()
            for c in cands:
                c.update(_score(best_entry(latest, stage, _qwen_match(c)), metric))
        scored = [c for c in cands if c.get("metric_value") is not None]
        if key == "search":
            best = cands[0]["key"] if cands else None          # search_candidates 가 이미 e2e 우선 → 상태 → mAP 순으로 정렬했다 (지표 종류가 섞이므로 값 비교 금지)
        elif scored:
            best = min(scored, key=lambda c: (STATUS_ORDER.get(c.get("status") or "n-a", 9), -float(c["metric_value"])))["key"]
        else:
            best = cands[0]["key"] if cands else None
        out[key] = {**{k: step[k] for k in ("title", "desc", "stage", "group", "stage_id")}, "metric": metric, "candidates": cands, "best_key": best}
    return out


def cached_catalog(refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    if refresh or "catalog" not in _CACHE:
        _CACHE["catalog"] = catalog()
    return _CACHE["catalog"]


def fmt_metric(c: Dict[str, Any]) -> str:
    v = c.get("metric_value")
    if v is None:
        return "성적 없음"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{c.get('metric')} {v:.3f}" if abs(v) <= 1.5 else f"{c.get('metric')} {v:.2f}"
