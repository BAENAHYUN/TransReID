#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gui/tool_choice.py — '도구' 페이지에서 고른 도구를 저장하고, 파이프라인 단계·검색 화면의 기본값으로 넣는다.

파일: <root>/gui_tool_choice.json  {"detector": {"key": "...", "value": ...}, "search": {...}, ...}  (git 미추적, 사용자 상태)
고른 것이 없으면 도구 카탈로그(원장 성적)의 ★ 최고 도구가 기본이다 → effective(step).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "gui_tool_choice.json"

# 단계 → (파이프라인 stage id, 플래그): 도구 선택이 그 단계 폼의 기본값이 된다
SPEC_TARGETS = {
    "detector": ("image_detect", "--detector-config"),
    "video_tracking": ("video_preprocess", "--tracking-config"),
}
TOOL_TARGETS = {
    "clusterer": "image_cluster",          # 핵심 단계의 '도구' 콤보 기본 항목
    "video_tracking": "video_preprocess",
}


def _load(path: Path = PATH) -> Dict[str, Any]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def get(step: str, path: Path = PATH) -> Optional[Dict[str, Any]]:
    d = _load(path).get(step)
    return d if isinstance(d, dict) and d.get("key") is not None else None


def set_choice(step: str, key: str, value: Any = None, label: str = "", path: Path = PATH) -> None:
    d = _load(path)
    d[step] = {"key": key, "value": value, "label": label}
    path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def clear(step: str, path: Path = PATH) -> None:
    d = _load(path)
    if step in d:
        d.pop(step)
        path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def effective(step: str, path: Path = PATH) -> Optional[Dict[str, Any]]:
    """사용자가 고른 것, 없으면 카탈로그의 ★ 최고 (원장 성적). 카탈로그를 못 읽으면 None."""
    chosen = get(step, path)
    if chosen is not None:
        return chosen
    try:
        from gui import tools_catalog as TC

        cat = TC.cached_catalog()
    except Exception:  # noqa: BLE001 — 원장/yaml 문제로 기본값만 못 정할 뿐, GUI 는 떠야 한다
        return None
    sec = cat.get(step) or {}
    best = sec.get("best_key")
    for c in sec.get("candidates", []):
        if c["key"] == best:
            return {"key": c["key"], "value": c.get("value"), "label": c.get("label"), "auto": True}
    return None


def apply_spec(stage_id: str, spec: Dict[str, Any], path: Path = PATH) -> Dict[str, Any]:
    """파이프라인 단계 폼의 arg spec 에 도구 선택을 기본값으로 넣는다 (사본 반환)."""
    for step, (sid, flag) in SPEC_TARGETS.items():
        if sid == stage_id and spec.get("flag") == flag:
            eff = effective(step, path)
            if eff and eff.get("key"):
                out = dict(spec)
                # value 가 문자열이면 그것이 실제 인자값(예: yaml 경로), 아니면 key
                out["default"] = str(eff["value"]) if isinstance(eff.get("value"), str) and eff["value"] else str(eff["key"])
                return out
    return spec


def tool_label(stage_id: str, path: Path = PATH) -> Optional[str]:
    for step, sid in TOOL_TARGETS.items():
        if sid == stage_id:
            eff = effective(step, path)
            return str(eff.get("label") or eff.get("key")) if eff else None
    return None


def search_defaults(scope: str, path: Path = PATH) -> Optional[Dict[str, Any]]:
    """사진 검색 기본 조합 {"stage1": [...], "rerank": str|None, "pool": int} — 사람 scope 만 다룬다."""
    if scope != "person":
        return None
    eff = effective("search", path)
    v = (eff or {}).get("value")
    return dict(v) if isinstance(v, dict) and v.get("stage1") else None


def embedder_default(scope: str, path: Path = PATH) -> Optional[str]:
    if scope != "person":
        return None
    eff = effective("embedder", path)
    return str(eff["key"]) if eff and eff.get("key") else None


def qwen_batch_default(path: Path = PATH) -> int:
    eff = effective("qwen", path)
    v = (eff or {}).get("value")
    try:
        return max(1, int((v or {}).get("batch_size", 1))) if isinstance(v, dict) else 1
    except (TypeError, ValueError):
        return 1
