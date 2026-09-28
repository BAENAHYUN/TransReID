"""클러스터링 플러그인 계약 — 임베더(registry.py)·검출기(detect/loader.py)와 같은 module / class / params 방식.

BaseClusterer.cluster(ids, primary, vectors) 는 point 순서에 맞춘 raw 라벨(int, 노이즈는 None) 을 돌려주고,
driver(clustering/cluster_qdrant.py) 가 그것을 안정 cluster_id · assignments.jsonl · report.json · Qdrant payload 로 바꾼다.
알고리즘은 Qdrant 를 모른다 — 벡터 행렬만 받는다. 그래서 같은 벡터로 알고리즘만 바꿔 비교할 수 있다.

내장 플러그인 (clustering/methods/):
  leiden     정확 kNN(mutual) 그래프 → Leiden (cluster_leiden_qdrant.py 와 같은 함수 사용)
  dbscan_v6  사용자 DBSCAN v6 (solider kNN 후보 → 결합 벡터 거리 필터 → greedy 배정; cluster_dbscan_qdrant.py 와 같은 함수)
새 알고리즘 = BaseClusterer 구현 하나 + yaml 의 clusterer: {module, class, params}.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ClusterResult:
    labels: List[Optional[int]]                 # ids 와 같은 순서. None = 노이즈
    stats: Dict[str, Any] = field(default_factory=dict)


class BaseClusterer:
    name: str = "base"                          # 파일명·payload 키(cluster_<name>_id)·stable id 접두어
    required_vectors: Tuple[str, ...] = ()      # primary 외에 추가로 필요한 named vector (driver 가 받아서 vectors 로 넘김)

    def params(self) -> Dict[str, Any]:
        """report.json config 에 그대로 실리는 하이퍼파라미터."""
        return {}

    def cluster(self, ids: Sequence[Any], primary: np.ndarray, vectors: Dict[str, np.ndarray], log=print) -> ClusterResult:
        """primary: (N, D) float32, 행 L2 정규화 (벡터 없는 point 는 0 행). vectors: name → (N, D_name) 같은 규약."""
        raise NotImplementedError


BUILTIN_METHODS: Dict[str, Dict[str, Any]] = {
    "leiden": {"module": "clustering.methods.leiden", "class": "LeidenClusterer", "params": {}},
    "dbscan_v6": {"module": "clustering.methods.dbscan_v6", "class": "DBSCANv6Clusterer", "params": {}},
}


def parse_param(text: str) -> Tuple[str, Any]:
    """key=value. value 는 JSON 으로 읽히면 그 타입(0.9 → float, true → bool, [1,2] → list), 아니면 문자열."""
    if "=" not in text:
        raise ValueError(f"--param 은 key=value 형식: {text!r}")
    key, value = text.split("=", 1)
    try:
        return key.strip(), json.loads(value)
    except ValueError:
        return key.strip(), value


def resolve_clusterer_spec(method: Optional[str] = None, module: Optional[str] = None, cls: Optional[str] = None,
                           params: Sequence[str] = (), config_path: Optional[str] = None) -> Dict[str, Any]:
    """우선순위: --module/--class > yaml 의 clusterer: 블록 > --method 내장 이름 (기본 leiden). --param 은 항상 덮어쓴다."""
    spec: Optional[Dict[str, Any]] = None
    if config_path:
        import yaml
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8-sig")) or {}
        block = raw.get("clusterer") if isinstance(raw, dict) else None
        if not (isinstance(block, dict) and block.get("module") and block.get("class")):
            raise ValueError(f"{config_path}: clusterer: {{module, class, params}} 블록이 없습니다")
        spec = {"module": str(block["module"]), "class": str(block["class"]), "params": dict(block.get("params") or {})}
    if spec is None:
        key = (method or "leiden").strip().lower()
        if key not in BUILTIN_METHODS and not (module and cls):
            raise ValueError(f"알 수 없는 --method {method!r}; 내장: {sorted(BUILTIN_METHODS)} 또는 --module/--class")
        spec = json.loads(json.dumps(BUILTIN_METHODS.get(key, {"module": "", "class": "", "params": {}})))
    if module:
        spec["module"] = module
    if cls:
        spec["class"] = cls
    if not spec.get("module") or not spec.get("class"):
        raise ValueError("클러스터링 플러그인 module/class 를 --method, yaml clusterer:, 또는 --module/--class 로 지정")
    p = dict(spec.get("params") or {})
    for item in params:
        k, v = parse_param(item)
        p[k] = v
    spec["params"] = p
    return spec


def load_clusterer(spec: Dict[str, Any]) -> BaseClusterer:
    module = importlib.import_module(spec["module"])
    klass = getattr(module, spec["class"])
    obj = klass(**dict(spec.get("params") or {}))
    if not hasattr(obj, "cluster"):
        raise TypeError(f"{spec['module']}.{spec['class']} 는 cluster() 가 없습니다 (BaseClusterer 계약)")
    return obj


def stable_cluster_id(name: str, target: str, ids: Sequence[Any]) -> str:
    """구성원 집합이 같으면 실행마다 같은 id (Leiden/DBSCAN 스크립트와 같은 방식, 접두어만 method 이름)."""
    raw = target + "|" + name + "|" + "|".join(sorted(map(str, ids)))
    return f"{name}:{target}:{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def normalize_labels(name: str, target: str, ids: Sequence[Any], labels: Sequence[Optional[int]],
                     min_cluster_size: int) -> Tuple[Dict[Any, Dict[str, Any]], Dict[str, Any]]:
    """raw 라벨 → assignments (point_id → {cluster_id, cluster_size, raw_id, noise}). 크기 < min_cluster_size 는 노이즈."""
    if len(labels) != len(ids):
        raise ValueError(f"labels 길이 {len(labels)} != ids 길이 {len(ids)}")
    groups: Dict[int, List[Any]] = defaultdict(list)
    noise_ids: List[Any] = []
    for pid, lab in zip(ids, labels):
        if lab is None:
            noise_ids.append(pid)
        else:
            groups[int(lab)].append(pid)
    assignments: Dict[Any, Dict[str, Any]] = {}
    kept = clustered = 0
    largest = 0
    for pid in noise_ids:
        assignments[pid] = dict(cluster_id=None, cluster_size=1, raw_id=None, noise=True)
    for raw_id, members in groups.items():
        if len(members) < min_cluster_size:
            for pid in members:
                assignments[pid] = dict(cluster_id=None, cluster_size=1, raw_id=raw_id, noise=True)
            continue
        cid = stable_cluster_id(name, target, members)
        kept += 1
        clustered += len(members)
        largest = max(largest, len(members))
        for pid in members:
            assignments[pid] = dict(cluster_id=cid, cluster_size=len(members), raw_id=raw_id, noise=False)
    noise = len(ids) - clustered
    return assignments, dict(points=len(ids), kept_clusters=kept, clustered_points=clustered, noise_points=noise,
                             noise_ratio=(noise / len(ids)) if ids else None, largest_community=largest,
                             min_cluster_size=min_cluster_size)
