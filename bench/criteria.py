"""bench/criteria.py — 채택 기준(기준표 §1~§5 의 현재 운영값)과 "채택 → yaml" 생성기 (P5).

ADOPTION_RULES[stage] = [(지표, 연산자, 기준값, 설명)]. 기준값은 기준표(outputs/audit/eval_criteria_20260926.md) 의 운영값·권고 그대로:
  detect  AP@0.5 ≥ 기존−0.01, 최대 재현율 ≥ 0.90, 75–119 px 재현율 ≥ 기존, 120–199 px ≥ 기존−0.02, fps ≥ 10
  embed   mAP ≥ 기존(89.06)
  search  mAP ≥ 기존 최선(88.07) AND pool_recall ≥ 90
  cluster 쌍 정밀도 ≥ 0.90 AND B³F1 ≥ 기존(0.851) AND 혼합 클러스터 ≤ 기존(138)
  e2e     mAP ≥ 운영(58.3) AND 검출 상한 ≥ 0.90
evaluate(entry) → {"status": pass|partial|fail|n/a, "checks": [...]} — 리더보드 색과 "채택" 버튼 활성화에 쓴다.

adopt_yaml(entry, root, overwrite=False) → 만든 파일 경로(들). 검출기 → pipeline_tracking_<이름>.yaml (tracker/stitcher 는 pipeline_tracking.yaml 에서),
클러스터 플러그인 → clusterer_<이름>.yaml, 검색 조합/스터디 → pipeline_<이름>.yaml (가중치만 변경한 사본, combos.write_pipeline_best) + <이름>.search.json.
드롭다운(gui_pipelines.json choices_glob pipeline*.yaml)이 새 파일을 자동으로 잡는다. 원본 pipeline.yaml / pipeline_tracking.yaml 은 바꾸지 않는다.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ADOPTION_RULES: Dict[str, List[Tuple[str, str, float, str]]] = {
    "detect": [("ap50", ">=", 0.866, "AP@0.5 ≥ 기존(0.876) − 0.01"),
               ("max_recall", ">=", 0.90, "최대 재현율 ≥ 0.90"),
               ("recall_h75_119", ">=", 0.659, "75–119 px 재현율 ≥ 기존(0.6597)"),
               ("recall_h120_199", ">=", 0.844, "120–199 px 재현율 ≥ 기존(0.864) − 0.02"),
               ("fps", ">=", 10.0, "≥ 10 fps")],
    "embed": [("map", ">=", 89.06, "mAP ≥ 기존 SOLIDER 89.06")],
    "search": [("map", ">=", 88.07, "mAP ≥ 기존 최선 88.07"), ("pool_recall", ">=", 90.0, "pool recall ≥ 90 %")],
    "cluster": [("pair_precision", ">=", 0.90, "쌍 정밀도 ≥ 0.90"), ("b3_f1", ">=", 0.851, "B³F1 ≥ 기존 0.851"),
                ("mixed_clusters", "<=", 138, "혼합 클러스터 ≤ 기존 138")],
    "e2e": [("map", ">=", 58.3, "e2e mAP ≥ 운영 58.3"), ("det_ceiling", ">=", 0.90, "검출 상한 ≥ 0.90")],
}

# 리더보드 그래프의 (x = 제약 지표, y = 목적 지표)
CHART_AXES: Dict[str, Tuple[str, str]] = {
    "detect": ("max_recall", "ap50"), "embed": ("rank1", "map"), "search": ("pool_recall", "map"),
    "cluster": ("pair_precision", "b3_f1"), "e2e": ("det_ceiling", "map"),
}


def _cmp(v: float, op: str, target: float) -> bool:
    return {">=": v >= target, "<=": v <= target, ">": v > target, "<": v < target}[op]


def evaluate(entry: Dict[str, Any]) -> Dict[str, Any]:
    """엔트리의 metrics 를 그 단계의 채택 기준에 대조. 지표가 없으면 그 검사는 건너뛴다(n/a)."""
    stage = entry.get("stage")
    metrics = entry.get("metrics") or {}
    checks = []
    for metric, op, target, label in ADOPTION_RULES.get(stage, []):
        v = metrics.get(metric)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            checks.append({"metric": metric, "label": label, "value": None, "ok": None})
            continue
        checks.append({"metric": metric, "label": label, "value": v, "ok": _cmp(float(v), op, target), "target": target, "op": op})
    applicable = [c for c in checks if c["ok"] is not None]
    if not applicable:
        status = "n/a"
    elif all(c["ok"] for c in applicable):
        status = "pass"
    elif any(c["ok"] for c in applicable):
        status = "partial"
    else:
        status = "fail"
    return {"status": status, "checks": checks, "passed": sum(1 for c in applicable if c["ok"]), "applicable": len(applicable)}


STATUS_LABEL = {"pass": "✓ 채택 기준 통과", "partial": "△ 일부 통과", "fail": "✗ 미달", "n/a": "— 기준 없음"}


# ---------------------------------------------------------------- 채택 → yaml
def slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(text)).strip("_")
    return s or "adopted"


def _header(entry: Dict[str, Any]) -> str:
    return (f"# bench 채택 ({time.strftime('%Y-%m-%d %H:%M')}): 원장 run_id {entry.get('run_id')} · name {entry.get('name')} · "
            f"stage {entry.get('stage')}. 원본 설정 파일은 바꾸지 않았다.\n")


def adopt_yaml(entry: Dict[str, Any], root: Path = PROJECT_ROOT, overwrite: bool = False, name: Optional[str] = None,
               tracking_template: Optional[Path] = None, pipeline_path: Optional[Path] = None) -> Dict[str, Any]:
    """엔트리를 실행 가능한 yaml 로. 반환 {"files": [...], "note": str, "kind": str}. 못 만드는 단계는 files=[]."""
    import yaml
    stage = entry.get("stage")
    comp = entry.get("component") or {}
    params = entry.get("params") or {}
    base_name = slug(name or entry.get("name") or "adopted")
    files: List[Path] = []

    if stage == "detect":
        if not comp.get("module") or comp.get("module") == "qdrant":
            return {"files": [], "note": "운영 DB(import-qdrant) 행은 검출기 설정이 아니라 채택할 수 없습니다", "kind": "detect"}
        template = tracking_template or (root / "pipeline_tracking.yaml")
        raw = yaml.safe_load(template.read_text(encoding="utf-8-sig")) or {}
        det_params = dict(comp.get("params") or {})
        if isinstance(params.get("operating_threshold"), (int, float)):
            det_params["conf_threshold"] = float(params["operating_threshold"])     # 운영 임계값 = 채점에 쓴 동작점
        raw["detector"] = {"module": comp["module"], "class": comp["class"], "params": det_params}
        out = root / f"pipeline_tracking_{base_name}.yaml"
        if out.exists() and not overwrite:
            raise FileExistsError(f"이미 있음: {out} (덮어쓰기를 켜거나 이름을 바꾸세요)")
        out.write_text(_header(entry) + f"# detector 만 채택값, tracker/stitcher 는 {template.name} 그대로\n"
                       + yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
        files.append(out)
        note = (f"영상 1단계 / 이미지 1단계 검출기 드롭다운은 module/class 로 중복을 없애므로 같은 class 의 yaml 이 이미 있으면 "
                f"기본 파일이 표시됩니다 — 이 파일을 쓰려면 --tracking-config / --detector-config 경로로 직접 지정하거나 기본 yaml 의 params 를 이 값으로 바꾸세요.")
        return {"files": files, "note": note, "kind": "detect"}

    if stage == "cluster":
        if not comp.get("module"):
            return {"files": [], "note": "플러그인(module/class) 정보가 없는 클러스터링 행은 채택할 수 없습니다", "kind": "cluster"}
        block = {"clusterer": {"module": comp["module"], "class": comp["class"], "params": dict(comp.get("params") or {})}}
        # 스터디/조합 행은 탐색된 params 가 entry.params 에 있다 (생성자 인자만 골라 넣는다)
        try:
            import importlib
            import inspect
            cls = getattr(importlib.import_module(comp["module"]), comp["class"])
            accepted = {k for k in inspect.signature(cls.__init__).parameters if k != "self"}
            for k, v in params.items():
                if k in accepted and k not in block["clusterer"]["params"]:
                    block["clusterer"]["params"][k] = v
        except Exception:  # noqa: BLE001 — import 실패면 component.params 만
            pass
        out = root / f"clusterer_{base_name}.yaml"
        if out.exists() and not overwrite:
            raise FileExistsError(f"이미 있음: {out}")
        vector = params.get("vector") or ((comp.get("axes") or {}).get("vector") if isinstance(comp.get("axes"), dict) else None)
        out.write_text(_header(entry) + (f"# primary vector: {vector} (clustering/cluster_qdrant.py --vector {vector} --method-config <이 파일>)\n" if vector else "")
                       + yaml.safe_dump(block, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
        files.append(out)
        return {"files": files, "note": "이미지 4b 클러스터링 플러그인의 --method-config 에 이 파일을 지정", "kind": "cluster"}

    if stage in ("search", "e2e"):
        stage1 = params.get("stage1") or comp.get("stage1") or (comp.get("axes") or {}).get("stage1")
        if isinstance(stage1, str):
            stage1 = [s for s in stage1.split("+") if s]
        rerank = params.get("rerank", comp.get("rerank", (comp.get("axes") or {}).get("rerank")))
        weights = {n: float(params[f"w_{n}"]) for n in (stage1 or []) if isinstance(params.get(f"w_{n}"), (int, float))}
        if not weights and isinstance(params.get("weights"), dict):
            weights = {k: float(v) for k, v in params["weights"].items()}
        src = pipeline_path or (root / "pipeline.yaml")
        out_yaml = root / f"pipeline_{base_name}.yaml"
        if out_yaml.exists() and not overwrite:
            raise FileExistsError(f"이미 있음: {out_yaml}")
        if weights:
            from bench.combos import write_pipeline_best
            ok, note = write_pipeline_best(src, out_yaml, weights)
            if not ok:
                raise ValueError(note)
            files.append(out_yaml)
        side = root / f"pipeline_{base_name}.search.json"
        side.write_text(json.dumps({"run_id": entry.get("run_id"), "name": entry.get("name"), "stage1": stage1,
                                    "rerank": None if rerank in (None, "none", "") else rerank,
                                    "prefetch": params.get("prefetch"), "pool": params.get("pool"), "rrf_k": params.get("rrf_k"),
                                    "weights": weights, "note": "검색 탭 드롭다운(1차 조합/재정렬)·후보 수에 맞춰 선택; 가중치는 옆 yaml"},
                                   ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        files.append(side)
        return {"files": files, "kind": stage,
                "note": ("가중치를 바꾼 pipeline yaml 은 이미지 2단계 '임베더 구성' 드롭다운에 나타납니다. " if weights else "가중치 정보가 없어 yaml 은 만들지 않았습니다. ")
                        + f"1차 조합 {stage1} / 재정렬 {rerank} / 후보 {params.get('pool')} 는 검색 탭에서 고르세요 (sidecar json)."}

    return {"files": [], "note": f"{stage} 단계 행은 채택할 설정 파일이 없습니다 (임베더는 pipeline.yaml retrievers 에 이미 등록)", "kind": stage or "?"}


def reproduce_command(entry: Dict[str, Any]) -> str:
    from bench.run import _quote, args_from_entry, runner_command
    return " ".join(_quote(c) for c in runner_command(entry["stage"], args_from_entry(entry)))
