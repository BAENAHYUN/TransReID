from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from config import PipelineConfig
from registry import EmbedderRegistry
from router import Router
from rfdetr_adapter import from_rfdetr, make_detection_id
from qdrant_store import QdrantStore


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

CONFIG_PATH = ROOT / "pipeline.yaml"
DEFAULT_STATS_PATH = ROOT / "data" / "crops" / "filter_stats.json"

CHECKPOINT_DIR = ROOT / "data" / "embedding_checkpoint"
STATE_PATH = CHECKPOINT_DIR / "state.json"

# embedding_build_id 별 불변 manifest. point payload 에는 id 만 들어가고
# 모델/설정/pipeline SHA/stats SHA 상세는 여기 한 곳에만 기록된다.
MANIFEST_DIR = ROOT / "data" / "build_manifests"


# ============================================================
# Helpers
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

        f.flush()
        os.fsync(f.fileno())

    os.replace(
        tmp,
        path,
    )


def format_seconds(seconds: float) -> str:
    seconds = max(
        0,
        int(seconds),
    )

    h, rem = divmod(
        seconds,
        3600,
    )

    m, s = divmod(
        rem,
        60,
    )

    return f"{h:02d}:{m:02d}:{s:02d}"


def is_person(det, cfg) -> bool:
    labels = {
        str(x).lower()
        for x in cfg.person_labels
    }

    return (
        str(det.label).lower()
        in labels
    )


def make_collection_configs(cfg: PipelineConfig):
    """
    Split one full pipeline configuration into per-collection Qdrant schemas.

    The collection names are derived from pipeline.yaml via
    cfg.person_collection() / cfg.object_collection().

    {prefix}_person:
      - siglip2
      - irra
      - solider

    {prefix}_object:
      - siglip2
      - dinov2
    """
    person_retrievers = {
        name: spec
        for name, spec in cfg.retrievers.items()
        if spec.accepts_person()
    }

    object_retrievers = {
        name: spec
        for name, spec in cfg.retrievers.items()
        if spec.accepts_object()
    }

    person_cfg = replace(
        cfg,
        collection=cfg.person_collection(),
        retrievers=person_retrievers,
    )

    object_cfg = replace(
        cfg,
        collection=cfg.object_collection(),
        retrievers=object_retrievers,
    )

    return person_cfg, object_cfg


def expected_vector_names(det, cfg):
    """
    Return the named vectors that each detection must contain
    according to retriever scope in pipeline.yaml.
    """

    person = is_person(
        det,
        cfg,
    )

    expected = set()

    for name, spec in cfg.retrievers.items():

        scope = str(
            spec.scope
        ).lower()

        if scope == "all":
            expected.add(name)

        elif (
            scope == "person"
            and person
        ):
            expected.add(name)

        elif (
            scope == "object"
            and not person
        ):
            expected.add(name)

    return expected


def validate_router_result(
    detections,
    vectors,
    cfg,
):
    """
    Validate Router output before upload so an incorrect routing result
    cannot be written to Qdrant at scale.
    """

    if not isinstance(
        vectors,
        (list, tuple),
    ):
        raise TypeError(
            "router.embed() did not return a list/tuple: "
            f"{type(vectors)}"
        )

    if (
        len(detections)
        != len(vectors)
    ):
        raise RuntimeError(
            "Detection / vector count mismatch: "
            f"{len(detections)} "
            f"!= {len(vectors)}"
        )

    for i, (
        det,
        vector_map,
    ) in enumerate(
        zip(
            detections,
            vectors,
        )
    ):

        if not isinstance(
            vector_map,
            dict,
        ):
            raise TypeError(
                f"vectors[{i}] is not a dict: "
                f"{type(vector_map)}"
            )

        actual = set(
            vector_map.keys()
        )

        expected = (
            expected_vector_names(
                det,
                cfg,
            )
        )

        if actual != expected:
            raise RuntimeError(
                "\nRouter routing error\n"
                f"index    : {i}\n"
                f"label    : {det.label}\n"
                f"expected : {sorted(expected)}\n"
                f"actual   : {sorted(actual)}"
            )


# ============================================================
# Retriever provenance
#
# 임베딩 공간을 결정하는 설정을 retriever 공통 구조로 다룬다. 예전에는
# SigLIP2 만 max_num_patches 를 검증했는데, 같은 논리가 SOLIDER 에는 더
# 강하게 적용된다: semantic_weight / neck_feat / checkpoint 가 바뀌면 DIM 은
# 1024 그대로인 채 임베딩 공간만 달라지고, SOLIDER 는 rerank 와 Leiden
# 클러스터링 벡터를 겸하므로 구/신 벡터가 섞이면 둘이 동시에 무너진다.
#
# 두 층으로 나눈다.
#   declared : pipeline.yaml 에서 계산. 모델 로드 전에 얻을 수 있어
#              checkpoint 호환성 판정과 embedding_build_id 에 쓴다.
#              파일 경로는 그대로 넣지 않고 내용 sha256 으로 바꾼다.
#              (경로/mtime 으로 resume 이 막히지 않게 — load_checkpoint 참조)
#   runtime  : 실제 로드된 임베더의 속성. declared 와 대조해 yaml 과 다른
#              값으로 떠 있으면 실패시키고, state.json / manifest 에 기록해
#              resume 시 이전 실행과 대조한다. DINOv2 처럼 yaml 에 선언이
#              없는 설정(model_id, dtype, pooling)은 이 층에서만 잡힌다.
#
# 규칙에 없는 retriever 가 yaml 에 나타나면 실패한다. 다섯 번째 임베더가
# 검증 없이 조용히 들어오는 일을 막기 위해서다.
# ============================================================

# 임베딩 값에 영향이 없는 params. 지문에서 제외한다.
PERF_ONLY_PARAMS = frozenset({
    "batch_size",
    "device",
    "cache_dir",
    "local_files_only",
})

# retriever 별 규칙.
#   required_params   : yaml 에 반드시 명시돼야 하는 params
#   file_params       : 경로 대신 내용 sha256 으로 지문에 들어가는 params
#   audit_only_params : 코드 루트 등. 지문에서 빼고 audit 에 경로만 남긴다
#   runtime_attrs     : (런타임 속성 경로, 대조할 yaml param 또는 None)
FINGERPRINT_RULES = {
    "siglip2": {
        "required_params": ("model_id", "max_num_patches"),
        "file_params": (),
        "audit_only_params": (),
        "runtime_attrs": (
            ("model_id", "model_id"),
            ("max_num_patches", "max_num_patches"),
            ("is_naflex", None),
            ("DIM", None),
        ),
    },
    "irra": {
        "required_params": ("ckpt_path",),
        "file_params": ("ckpt_path", "config_file", "clip_pretrained"),
        "audit_only_params": ("irra_root",),
        "runtime_attrs": (
            ("use_amp", "amp"),
            ("cfg.img_size", None),
            ("cfg.stride_size", None),
            ("cfg.text_length", None),
            ("DIM", None),
        ),
    },
    "solider": {
        "required_params": (
            "ckpt_path",
            "backbone",
            "semantic_weight",
            "neck_feat",
        ),
        "file_params": ("ckpt_path",),
        "audit_only_params": ("solider_root",),
        "runtime_attrs": (
            ("backbone", "backbone"),
            ("semantic_weight", "semantic_weight"),
            ("neck_feat", "neck_feat"),
            ("img_size", None),
            ("DIM", None),
        ),
    },
    "dinov2": {
        "required_params": (),
        "file_params": (),
        "audit_only_params": (),
        "runtime_attrs": (
            ("model_id", None),
            ("feature", None),
            ("resize_mode", None),
            ("image_size", None),
            ("dtype", None),
            ("DIM", None),
        ),
    },
}


def _norm(value):
    """JSON 에 넣고 다시 읽어도 같은 값이 되도록 정규화한다. 비교에도 쓴다."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _norm(v) for k, v in value.items()}
    return str(value)


def _canon(value):
    """dict 를 JSON 왕복시켜 디스크에서 읽은 값과 같은 형태로 만든다."""
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    )


def _getattr_path(obj, path: str):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            raise AttributeError(part)
        cur = getattr(cur, part)
    return cur


def _resolve_param_path(value) -> Path:
    """
    지문에 넣을 파일 경로를 **임베더가 실제로 열게 될 경로** 로 해석한다.

    config._resolve_paths 는 './' 또는 '../' 로 시작하는 값만 yaml 위치 기준 절대경로로
    바꿔 임베더에 넘긴다. 그 외 상대경로('weights/x.pth')는 그대로 전달되어 임베더가
    실행 디렉터리(CWD) 기준으로 연다. 여기서 규칙이 다르면 A 파일을 해시하고 B 파일로
    임베딩하는 불일치가 생기므로 같은 규칙을 따른다.
    """
    raw = str(value)
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    if raw.startswith("./") or raw.startswith("../"):
        return (CONFIG_PATH.parent / p).resolve()
    return (Path.cwd() / p).resolve()


_PATH_LIKE_SUFFIXES = (".pth", ".pt", ".safetensors", ".bin", ".ckpt", ".yaml", ".yml", ".json")

_YAML_FINGERPRINT_CACHE: dict = {}


def _yaml_fingerprint_rules() -> dict:
    """
    pipeline yaml 최상위 선택 섹션 `fingerprint:` — 새 retriever 의 지문 규칙을 코드 수정 없이
    선언한다. retrievers.*.params 에 넣으면 생성자 인자로 흘러가므로 별도 섹션이다.

        fingerprint:
          newreid:
            required_params: [ckpt_path, backbone]
            file_params: [ckpt_path]
            audit_only_params: [code_root]
            runtime_attrs: [[backbone, backbone], [DIM, null]]
    """
    key = str(CONFIG_PATH)
    if key in _YAML_FINGERPRINT_CACHE:
        return _YAML_FINGERPRINT_CACHE[key]

    import yaml  # 지연 import

    rules: dict = {}
    if CONFIG_PATH.is_file():
        with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
            raw = yaml.safe_load(f) or {}
        section = raw.get("fingerprint") if isinstance(raw, dict) else None
        if section is not None:
            if not isinstance(section, dict):
                raise RuntimeError("pipeline yaml 의 fingerprint: 는 매핑이어야 합니다.")
            for name, spec in section.items():
                if not isinstance(spec, dict):
                    raise RuntimeError(f"fingerprint.{name} 은 매핑이어야 합니다.")
                unknown = set(spec) - {"required_params", "file_params", "audit_only_params", "runtime_attrs"}
                if unknown:
                    raise RuntimeError(f"fingerprint.{name} 에 알 수 없는 키 {sorted(unknown)}")
                def _strs(k):
                    v = spec.get(k) or []
                    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                        raise RuntimeError(f"fingerprint.{name}.{k} 는 문자열 리스트여야 합니다.")
                    return tuple(v)
                attrs = spec.get("runtime_attrs") or []
                if not isinstance(attrs, list) or not all(
                    isinstance(a, list) and len(a) == 2 and isinstance(a[0], str)
                    and (a[1] is None or isinstance(a[1], str)) for a in attrs
                ):
                    raise RuntimeError(f"fingerprint.{name}.runtime_attrs 는 [[속성, yaml키|null], ...] 형식이어야 합니다.")
                rules[str(name)] = {
                    "required_params": _strs("required_params"),
                    "file_params": _strs("file_params"),
                    "audit_only_params": _strs("audit_only_params"),
                    "runtime_attrs": tuple((a[0], a[1]) for a in attrs),
                    "source": "yaml",
                }
    _YAML_FINGERPRINT_CACHE[key] = rules
    return rules


def _generic_rule(name: str, params: dict) -> dict:
    """
    명시 규칙도 yaml 규칙도 없는 retriever 의 제한된 기본 규칙.
    - 존재하는 파일을 가리키는 str param  → file_params (내용 sha256)
    - 존재하는 디렉터리를 가리키는 str param → audit_only_params (코드 루트로 간주)
    - 경로처럼 보이는데 없는 값            → 경고 후 문자열로 지문에 남는다
    - PERF_ONLY_PARAMS 는 제외
    - runtime: DIM 은 필수, 그 외 declared 키와 같은 이름의 속성이 있으면 대조, 없으면 미검증 기록
    이 규칙은 명시 규칙보다 약하다. 새 모델은 FINGERPRINT_RULES 또는 yaml fingerprint: 에 명시하라.
    """
    file_params, audit_params, declared = [], [], []
    for key, value in (params or {}).items():
        if key in PERF_ONLY_PARAMS:
            continue
        if isinstance(value, str):
            looks_like_path = ("/" in value or "\\" in value or value.lower().endswith(_PATH_LIKE_SUFFIXES))
            if looks_like_path:
                p = _resolve_param_path(value)
                if p.is_file():
                    file_params.append(key)
                    continue
                if p.is_dir():
                    audit_params.append(key)
                    continue
                print(
                    f"WARNING: retriever '{name}' params.{key}={value!r} 는 경로처럼 보이지만 "
                    f"파일/디렉터리가 없어 내용 해시 대신 문자열로 지문에 넣습니다 ({p})."
                )
        declared.append(key)
    return {
        "required_params": (),
        "file_params": tuple(file_params),
        "audit_only_params": tuple(audit_params),
        "runtime_attrs": (("DIM", None),) + tuple((k, k) for k in declared),
        "generic": True,
        "source": "generic",
    }


_GENERIC_WARNED: set = set()


def _rule_for(name: str, params=None) -> dict:
    """
    우선순위: FINGERPRINT_RULES 명시 규칙 → yaml `fingerprint:` 선언 → generic fallback.
    기존 4개 모델(siglip2/irra/solider/dinov2)은 명시 규칙 그대로라 이전 지문과 동일하다.
    """
    rule = FINGERPRINT_RULES.get(name)
    if rule is not None:
        return rule
    rule = _yaml_fingerprint_rules().get(name)
    if rule is not None:
        return rule
    if name not in _GENERIC_WARNED:
        _GENERIC_WARNED.add(name)
        print(
            f"WARNING: retriever '{name}' 에 지문 규칙이 없어 generic 규칙(파일/디렉터리 자동 분류, "
            "DIM 필수, 없는 속성은 미검증)으로 진행합니다. 재현성을 위해 build_db.FINGERPRINT_RULES "
            "또는 pipeline yaml 의 fingerprint: 섹션에 명시 규칙을 추가하세요."
        )
    return _generic_rule(name, dict(params or {}))


def fingerprint_rule_sources(cfg: PipelineConfig) -> dict:
    """audit 용: retriever 별로 어떤 규칙(explicit / yaml / generic)이 쓰였는지."""
    out = {}
    for name, spec in cfg.retrievers.items():
        rule = _rule_for(name, dict(getattr(spec, "params", None) or {}))
        out[name] = rule.get("source", "explicit")
    return out


def retriever_source_hashes(cfg: PipelineConfig) -> dict:
    """
    audit 용: 임베더 구현 모듈 파일의 sha256. 전처리/정규화 코드가 바뀌었는데 yaml 이
    같으면 지문이 놓친다는 점을 보완한다 (compat 에는 넣지 않는다 — resume 을 깨지 않도록).
    모듈을 import 하지 않고 위치만 찾는다.
    """
    import importlib.util

    out = {}
    for name, spec in cfg.retrievers.items():
        module = str(getattr(spec, "module", ""))
        try:
            found = importlib.util.find_spec(module)
            origin = getattr(found, "origin", None) if found else None
        except (ImportError, ValueError, AttributeError):
            origin = None
        if origin and Path(origin).is_file():
            out[name] = {"module": module, "path": str(Path(origin).resolve()), "sha256": sha256_file(Path(origin))}
        else:
            out[name] = {"module": module, "path": None, "sha256": None}
    return out


def declared_retriever_fingerprint(cfg: PipelineConfig) -> dict:
    """
    pipeline.yaml 만으로 계산하는 retriever 지문. 모델 로드 전에 쓸 수 있다.

    파일 params 는 경로가 아니라 내용 sha256 으로 들어간다. 같은 checkpoint 를
    다른 드라이브에 두어도 지문이 같고, 같은 경로의 파일을 바꿔 끼우면 달라진다.
    """
    out = {}

    for name, spec in cfg.retrievers.items():
        params = dict(getattr(spec, "params", None) or {})
        rule = _rule_for(name, params)

        for key in rule["required_params"]:
            if key not in params:
                raise RuntimeError(
                    f"pipeline.yaml 의 retrievers.{name}.params.{key} 를 "
                    "명시해야 합니다."
                )

        files = {}
        for key in rule["file_params"]:
            if not params.get(key):
                continue
            p = _resolve_param_path(params[key])
            if not p.is_file():
                raise FileNotFoundError(
                    f"retrievers.{name}.params.{key} 가 가리키는 파일이 "
                    f"없습니다: {p}"
                )
            files[key] = {
                "sha256": sha256_file(p),
                "size": p.stat().st_size,
            }

        skip = (
            set(PERF_ONLY_PARAMS)
            | set(rule["file_params"])
            | set(rule["audit_only_params"])
        )

        out[name] = {
            "dim": int(spec.dim),
            "module": str(spec.module),
            "class": str(spec.class_name),
            "params": {
                k: _norm(v)
                for k, v in params.items()
                if k not in skip
            },
            "files": files,
        }

    return out


def declared_retriever_paths(cfg: PipelineConfig) -> dict:
    """audit 용. 지문에서 뺀 경로들을 그대로 기록한다."""
    out = {}

    for name, spec in cfg.retrievers.items():
        params = dict(getattr(spec, "params", None) or {})
        rule = _rule_for(name, params)
        keys = tuple(rule["file_params"]) + tuple(rule["audit_only_params"])
        rec = {
            k: str(_resolve_param_path(params[k]))
            for k in keys
            if params.get(k)
        }
        if rec:
            out[name] = rec

    return out


def runtime_retriever_fingerprint(
    cfg: PipelineConfig,
    registry: EmbedderRegistry,
) -> dict:
    """
    실제 로드된 임베더의 속성을 읽어 기록하고, yaml 과 대조 가능한 항목은
    대조한다. 큰 임베딩 루프 전에 실행한다.

    속성이 없으면 건너뛰지 않고 실패시킨다. "검증할 수 없음" 이 조용히
    "검증 통과" 로 보이면 안 된다.
    """
    out = {}

    for name, spec in cfg.retrievers.items():
        params = dict(getattr(spec, "params", None) or {})
        rule = _rule_for(name, params)
        embedder = registry.get(name)

        record = {}
        mismatches = []
        unverified = []

        for attr_path, declared_key in rule["runtime_attrs"]:
            try:
                value = _getattr_path(embedder, attr_path)
            except AttributeError:
                if rule.get("generic") and attr_path != "DIM":
                    # generic 규칙: 생성자 param 과 속성 이름이 다를 수 있다(IRRA 의 amp→use_amp).
                    # 실패시키지 않고 미검증으로 기록한다. DIM 은 예외 없이 필수.
                    unverified.append(attr_path)
                    continue
                raise RuntimeError(
                    f"retriever '{name}' 런타임 객체 "
                    f"({type(embedder).__name__}) 에 '{attr_path}' 속성이 "
                    "없어 검증할 수 없습니다. FINGERPRINT_RULES 의 "
                    "runtime_attrs 또는 임베더 구현을 확인하세요."
                )

            record[attr_path] = _norm(value)

            if declared_key is not None and declared_key in params:
                if _norm(params[declared_key]) != _norm(value):
                    mismatches.append(
                        f"{declared_key}: yaml={params[declared_key]!r} "
                        f"runtime={value!r}"
                    )

        if "DIM" in record and int(record["DIM"]) != int(spec.dim):
            mismatches.append(
                f"dim: yaml={spec.dim} runtime={record['DIM']}"
            )

        if name == "siglip2":
            expected_naflex = (
                "naflex" in str(params.get("model_id", "")).lower()
            )
            if bool(record.get("is_naflex")) != expected_naflex:
                mismatches.append(
                    f"naflex: model_id 기준 {expected_naflex}, "
                    f"runtime={record.get('is_naflex')}"
                )

        if mismatches:
            raise RuntimeError(
                "\n".join(
                    [
                        f"retriever '{name}' 런타임 설정이 pipeline.yaml 과 "
                        "다릅니다:",
                        *(f"  {m}" for m in mismatches),
                    ]
                )
            )

        if unverified:
            # 값으로 "unverified" 를 섞지 않는다 (int(record["DIM"]) 계약 유지). 별도 키.
            record["_unverified"] = sorted(unverified)

        out[name] = record
        print(f"runtime[{name}]:", record)

    return out


def assert_upsert_complete(store, submitted: int, label: str) -> None:
    """
    제출한 detection 수 == committed, skipped == 0 을 강제한다.

    QdrantStore 는 벡터 없는 detection 을 경고 한 줄로 건너뛰고 committed
    수만 돌려준다. build_db 에서는 validate_router_result 가 모든 detection
    에 벡터를 보장하므로 skip 이 하나라도 있으면 그 자체가 이상이다.
    출력만 하고 대조하지 않으면 조용한 누락이 된다.
    """
    if submitted == 0:
        return

    stats = getattr(store, "last_upsert_stats", None)
    if not stats:
        raise RuntimeError(
            f"{label}: QdrantStore.last_upsert_stats 가 없습니다. "
            "qdrant_store.py 가 upsert 통계를 기록하는 버전인지 확인하세요."
        )

    seen = int(stats.get("submitted", -1))
    committed = int(stats.get("committed", -1))
    skipped = int(stats.get("skipped", 0))

    if seen != submitted or committed != submitted or skipped != 0:
        raise RuntimeError(
            f"{label} 적재 건수 불일치: submitted={submitted}, "
            f"store_seen={seen}, committed={committed}, skipped={skipped}. "
            "이 batch 는 checkpoint 에 기록하지 않았습니다. 같은 명령으로 "
            "다시 실행하면 직전 checkpoint 부터 재시도합니다."
        )


# ============================================================
# State
# ============================================================

def make_run_info(
    cfg: PipelineConfig,
    batch_size: int,
    total: int,
    stats_path: Path,
):
    """
    Checkpoint fingerprint — the RESUME COMPATIBILITY part only.

    Compatibility is decided by content, not by file identity. The crop-
    metadata JSON enters this fingerprint only through its sha256 (plus size,
    which the hash already implies). Its path and mtime are recorded
    separately by make_audit_info() and are never compared: copying, moving,
    or regenerating a byte-identical filter_stats.json must not force a full
    re-embed. The GUI cannot pass --fresh, so a spurious mismatch would
    dead-end it. A genuinely different file (e.g. filter_stats_dedup.json)
    still mismatches because its content hash differs.

    "retrievers" carries the declared fingerprint of EVERY embedder
    (declared_retriever_fingerprint). Checkpoint files enter it as content
    hashes, never as paths. "siglip2" duplicates part of it in the older
    flat shape so a state.json written before "retrievers" existed still
    compares — see load_checkpoint() and LEGACY_OPTIONAL_COMPAT_KEYS.

    Only the keys in CHECKPOINT_COMPAT_KEYS take part in the comparison.
    Adding a key here without adding it there makes it audit-only.

    The pipeline file hash already detects any YAML change.

    Only the Router batch size is included. The Qdrant upsert batch size is
    deliberately excluded: it changes how many points are sent per request,
    not which points are produced, and point IDs are deterministic. Adding it
    here would invalidate a valid checkpoint for no reason.
    """
    stats_path = stats_path.resolve()
    stats_stat = stats_path.stat()

    retrievers = declared_retriever_fingerprint(cfg)

    if "siglip2" not in retrievers:
        raise RuntimeError(
            "pipeline.yaml 에 retrievers.siglip2 가 없습니다."
        )

    sig = retrievers["siglip2"]

    return {
        "pipeline_sha256":
            sha256_file(
                CONFIG_PATH
            ),

        # 구버전 state.json 과의 비교용 평면 사본. 내용은 retrievers.siglip2 와 같다.
        "siglip2": {
            "model_id":
                str(sig["params"]["model_id"]),

            "max_num_patches":
                int(sig["params"]["max_num_patches"]),

            "dim":
                int(sig["dim"]),
        },

        "retrievers":
            retrievers,

        "stats_sha256":
            sha256_file(
                stats_path
            ),

        "stats_size":
            stats_stat.st_size,

        "total":
            total,

        "batch_size":
            batch_size,
    }


def make_audit_info(
    cfg: PipelineConfig,
    stats_path: Path,
):
    """
    Provenance recorded next to the checkpoint but NOT compared on resume.

    This is where the file-identity fields live. They answer "which file, on
    which machine, at what time produced this DB" for audit purposes, without
    turning a copied or regenerated filter_stats.json into a forced rebuild.
    """
    stats_path = stats_path.resolve()
    stats_stat = stats_path.stat()

    return {
        "stats_path":
            str(stats_path),

        "stats_mtime_ns":
            stats_stat.st_mtime_ns,

        "config_path":
            str(CONFIG_PATH),

        "retriever_paths":
            declared_retriever_paths(cfg),

        # 어떤 지문 규칙이 쓰였는지 + 임베더 구현 파일 해시 (audit 전용; compat 비교 대상 아님)
        "fingerprint_rules":
            fingerprint_rule_sources(cfg),

        "retriever_sources":
            retriever_source_hashes(cfg),

        "recorded_at":
            time.strftime(
                "%Y-%m-%dT%H:%M:%S%z"
            ),
    }


# Keys of run_info that decide whether an existing checkpoint may be resumed.
#
# Comparison is restricted to these keys on BOTH sides, so a state.json
# written by an older build_db.py — whose run_info also carried stats_path
# and stats_mtime_ns — still resumes as long as these values agree. That is
# what keeps this change from itself forcing a one-time full re-embed.
CHECKPOINT_COMPAT_KEYS = (
    "pipeline_sha256",
    "siglip2",
    "retrievers",
    "stats_sha256",
    "stats_size",
    "total",
    "batch_size",
)

# 구버전 state.json 에는 없는 compat 키. 이전 실행이 기록하지 않은 값은
# 대조할 수 없으므로, 없으면 경고를 남기고 resume 을 허용한다 (legacy).
# 다음 checkpoint 부터는 새 형식으로 기록되어 정식 비교 대상이 된다.
LEGACY_OPTIONAL_COMPAT_KEYS = frozenset({
    "retrievers",
})


def _compat_subset(info: dict) -> dict:
    return {
        key: info.get(key)
        for key in CHECKPOINT_COMPAT_KEYS
    }


def _assert_resume_target(store, collection: str, saved_index: int) -> None:
    """resume 대상 컬렉션이 존재하고 비어 있지 않아야 한다."""
    exists = bool(store.client.collection_exists(collection))
    count = int(store.client.count(collection, exact=True).count) if exists else 0
    if not exists or count == 0:
        raise RuntimeError(
            f"resume(checkpoint next_index={saved_index:,}) 인데 컬렉션 '{collection}' 이 "
            f"{'없습니다' if not exists else '비어 있습니다'}. checkpoint 와 Qdrant 가 어긋났습니다 "
            "(DB 교체/삭제?). 처음부터 다시 적재하려면 --fresh (별도 --checkpoint-dir 권장), "
            "아니면 Qdrant 데이터를 먼저 복구하세요."
        )


def load_checkpoint(
    run_info: dict,
    audit_info: dict = None,
):
    if not STATE_PATH.exists():
        return {
            "run_info": run_info,
            "next_index": 0,
        }

    with open(
        STATE_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        state = json.load(f)

    old = state.get(
        "run_info"
    ) or {}

    old_compat = _compat_subset(old)
    new_compat = _compat_subset(run_info)

    legacy_missing = sorted(
        key
        for key in LEGACY_OPTIONAL_COMPAT_KEYS
        if key not in old
    )

    # Name the offending keys so a GUI user can tell "the crop list
    # changed" apart from "pipeline.yaml changed" without reading code.
    changed = [
        key
        for key in CHECKPOINT_COMPAT_KEYS
        if key not in legacy_missing
        and _canon(old_compat[key]) != _canon(new_compat[key])
    ]

    if changed:
        prev_audit = state.get("audit") or {}
        prev_stats = prev_audit.get("stats_path", "(unknown)")
        cur_stats = (audit_info or {}).get("stats_path", "(unknown)")
        raise RuntimeError(
            f"기존 embedding checkpoint ({CHECKPOINT_DIR}, build "
            f"{state.get('embedding_build_id') or '(legacy, id 없음)'}) 와 현재 설정이 다릅니다.\n"
            f"  달라진 항목: {', '.join(changed)}\n"
            f"  이전 stats : {prev_stats}\n"
            f"  현재 stats : {cur_stats}\n"
            "  다른 crop 목록이나 다른 yaml 로 **새 build** 를 시작하는 것이면 "
            "--checkpoint-dir <새 폴더> 를 쓰세요 (권장; 기존 build 의 checkpoint 는 보존).\n"
            "  정말 이 checkpoint 를 버리고 처음부터 하려면 --fresh (이 폴더의 checkpoint 가 삭제됩니다)."
        )

    if legacy_missing:
        print(
            "WARNING: legacy checkpoint — 이전 실행이 기록하지 않아 대조하지 "
            f"못한 항목: {', '.join(legacy_missing)}"
        )
        print(
            "         SOLIDER/IRRA/DINOv2 설정이 이전 실행과 같다고 가정하고 "
            "이어갑니다. 확실히 하려면 --fresh 로 새로 구축하세요."
        )

    return state


def make_build_id(run_info: dict) -> str:
    """
    embedding_build_id = emb_<날짜>_<compat 지문 sha256 앞 8자리>.

    해시 부분이 결정적이라 같은 설정·같은 입력이면 같은 id 가 나온다.
    resume 은 state.json 에 저장된 id 를 재사용하므로 하나의 논리적 build 가
    여러 id 로 갈라지지 않는다.
    """
    digest = hashlib.sha256(
        json.dumps(
            _compat_subset(run_info),
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:8]

    return f"emb_{time.strftime('%Y%m%d')}_{digest}"


def resolve_build_id(
    state: dict,
    run_info: dict,
    saved_index: int,
):
    """
    이번 실행이 쓸 embedding_build_id 를 정한다.

    resume 이면 state.json 에 저장된 id 를 무조건 재사용한다. make_build_id 의
    날짜 접두어 때문에, 밤에 중단하고 다음날 이어가면 같은 지문이라도 새 id 가
    나오므로 여기서 재사용하지 않으면 한 build 가 두 provenance 로 갈라진다.

    반환: (build_id, legacy_build)
      legacy_build : resume 인데 이전 실행이 id 를 기록하지 않은 경우.
                     그 앞서 적재된 point 는 id 없는 legacy 로 남는다.
    """
    saved = state.get("embedding_build_id")

    if saved:
        return str(saved), False

    return make_build_id(run_info), saved_index > 0

def write_build_manifest(
    manifest_dir: Path,
    manifest: dict,
):
    """
    불변 manifest. 같은 id 의 파일이 이미 있으면 compat 지문이 같은지만
    확인하고 덮어쓰지 않는다. 다르면 id 충돌이므로 실패시킨다.

    반환: (경로, 새로 만들었는지)
    """
    path = manifest_dir / f"{manifest['embedding_build_id']}.json"

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)

        if _canon(existing.get("compat")) != _canon(manifest["compat"]):
            raise RuntimeError(
                f"build manifest 가 이미 존재하는데 compat 지문이 다릅니다: "
                f"{path}\nmanifest 는 불변이어야 합니다. id 가 충돌했다면 "
                "기존 파일을 조사한 뒤 --fresh 로 새 build 를 시작하세요."
            )

        return path, False

    atomic_write_json(path, manifest)
    return path, True


# ============================================================
# Main
# ============================================================

def main():

    # Allow CLI args to update module-level path globals.
    # Must be declared before any use of these names in this function.
    global CONFIG_PATH, CHECKPOINT_DIR, STATE_PATH, MANIFEST_DIR

    # Fixed-shape inference benefits from Tensor Core paths and cuDNN's
    # one-time kernel selection. These flags do not affect CPU execution.
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
    except (ImportError, AttributeError):
        pass

    # GUI 의 "중단" 은 Windows 에서 CTRL_BREAK_EVENT(SIGBREAK) 로 들어온다
    # (pipeline_page.py ProcessWorker.stop). Python 은 SIGINT 만
    # KeyboardInterrupt 로 바꾸고 SIGBREAK 는 CRT 기본 동작(즉시 종료)에
    # 맡기므로, 아래의 except KeyboardInterrupt 가 실행되지 않고 종료 코드도
    # 셸 관례(130)와 다르게 나온다. 두 신호를 같은 경로로 합친다.
    if hasattr(signal, "SIGBREAK"):
        signal.signal(
            signal.SIGBREAK,
            signal.default_int_handler,
        )

    parser = argparse.ArgumentParser(
        description=(
            "Full embedding + Qdrant build"
        )
    )

    parser.add_argument(
        "--stats",
        default=str(DEFAULT_STATS_PATH),
        help=(
            "Crop metadata JSON to build from "
            "(default: data/crops/filter_stats.json). "
            "Use data/crops/filter_stats_dedup.json after dedup."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Router outer batch size (RTX 5090 profile default: 512)",
    )

    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=256,
        help=(
            "Qdrant upsert batch size, independent of Router batch size "
            "(default: 256)"
        ),
    )

    parser.add_argument(
        "--upsert-parallel",
        type=int,
        default=4,
        help="Qdrant upload worker count (default: 4)",
    )

    parser.add_argument(
        "--decode-workers",
        type=int,
        default=min(16, max(1, os.cpu_count() or 1)),
        help="Parallel image decoders; each crop is decoded once (default: up to 16)",
    )

    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help=(
            "Pipeline YAML config file "
            "(default: pipeline.yaml next to build_db.py). "
            "Pass a different file to use a separate config for image vs video builds."
        ),
    )

    parser.add_argument(
        "--checkpoint-dir",
        default=str(CHECKPOINT_DIR),
        help=(
            "Directory for the embedding checkpoint state.json "
            "(default: data/embedding_checkpoint). "
            "Use a separate directory for each DB build to avoid checkpoint collision "
            "between image and video builds."
        ),
    )

    parser.add_argument(
        "--manifest-dir",
        default=str(MANIFEST_DIR),
        help=(
            "Directory for immutable build manifests, one JSON per "
            "embedding_build_id (default: data/build_manifests)."
        ),
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Reset the embedding checkpoint and process again from index 0",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "Delete the Qdrant collections and create them from scratch. "
            "컬렉션 이름은 yaml collection_prefix 로 정해진다 — prefix 를 바꾸지 않은 채 "
            "쓰면 현재 운영 컬렉션(forensic_person/forensic_object)을 지운다."
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Resolve path globals from CLI args.                                  #
    # These must be updated before any function that reads them            #
    # (load_checkpoint, CHECKPOINT_DIR.mkdir, etc.).                       #
    # ------------------------------------------------------------------ #
    CONFIG_PATH = Path(args.config).expanduser().resolve()
    CHECKPOINT_DIR = Path(args.checkpoint_dir).expanduser().resolve()
    STATE_PATH = CHECKPOINT_DIR / "state.json"
    MANIFEST_DIR = Path(args.manifest_dir).expanduser().resolve()

    stats_path = Path(args.stats).expanduser().resolve()

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be at least 1"
        )

    if args.upsert_batch_size <= 0:
        raise ValueError(
            "upsert-batch-size must be at least 1"
        )

    if args.upsert_parallel <= 0 or args.decode_workers <= 0:
        raise ValueError(
            "upsert-parallel and decode-workers must be at least 1"
        )

    # --------------------------------------------------------
    # 1. Config
    # --------------------------------------------------------

    print("=" * 70)
    print("LOAD CONFIG")
    print("=" * 70)

    cfg = PipelineConfig.load(
        str(CONFIG_PATH)
    )

    person_cfg, object_cfg = make_collection_configs(cfg)

    print(
        "collections:",
        {
            "person": person_cfg.collection,
            "object": object_cfg.collection,
        },
    )

    print(
        "person vectors:",
        list(person_cfg.retrievers.keys()),
    )

    print(
        "object vectors:",
        list(object_cfg.retrievers.keys()),
    )

    # --------------------------------------------------------
    # 2. RF-DETR crop metadata
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOAD CROP METADATA")
    print("=" * 70)

    if not stats_path.is_file():
        raise FileNotFoundError(
            f"crop metadata not found: {stats_path}"
        )

    print(
        "stats:",
        stats_path,
    )

    with open(
        stats_path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    crops = data["crops"]

    # --------------------------------------------------------
    # Global detection_id uniqueness check
    #
    # from_rfdetr() only detects duplicate IDs within one call.
    # build_db.py calls it once per Router batch, so duplicate IDs
    # across different batches would otherwise go unnoticed.
    #
    # Qdrant Point IDs are derived from detection_id, so duplicate
    # detection IDs would overwrite the same point.
    #
    # Reuse make_detection_id() here so the validation rule cannot
    # drift from the actual ID generation rule.
    # --------------------------------------------------------

    crop_ids = [
        make_detection_id(
            record,
            int(record.get("frame_idx", 0)),
        )
        for record in crops
    ]

    counts = Counter(crop_ids)

    duplicates = [
        crop_id
        for crop_id, count in counts.items()
        if count > 1
    ]

    if duplicates:
        raise RuntimeError(
            f"Duplicate detection IDs found in {stats_path.name}: "
            f"{len(duplicates)} unique duplicated IDs. "
            f"Qdrant points would be overwritten. "
            f"Examples: {duplicates[:3]}"
        )

    total = len(crops)

    if total == 0:
        raise RuntimeError(
            "crop metadata contains 0 records."
        )

    print(
        "total crops:",
        total,
    )

    print(
        "unique detection ids:",
        len(counts),
    )

    # --------------------------------------------------------
    # 3. Checkpoint
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.fresh:
        if STATE_PATH.exists():
            STATE_PATH.unlink()

        print(
            "embedding checkpoint reset"
        )

    if (
        args.recreate
        and STATE_PATH.exists()
        and not args.fresh
    ):
        raise RuntimeError(
            "--recreate deletes and recreates the existing Qdrant collections.\n"
            "An embedding checkpoint already exists, so use "
            "--recreate --fresh together."
        )

    run_info = make_run_info(
        cfg,
        args.batch_size,
        total,
        stats_path,
    )

    audit_info = make_audit_info(
        cfg,
        stats_path,
    )

    state = load_checkpoint(
        run_info,
        audit_info,
    )

    saved_index = int(
        state.get(
            "next_index",
            0,
        )
    )

    # --------------------------------------------------------
    # embedding_build_id
    #
    # resume 이면 state.json 의 id 를 그대로 쓴다. 구버전 checkpoint 에는
    # 없으므로 지금 만들되, 그 앞서 적재된 point 들은 id 없는 legacy 로 남는다.
    # --------------------------------------------------------

    build_id, legacy_build = resolve_build_id(
        state,
        run_info,
        saved_index,
    )

    print(
        "embedding_build_id:",
        build_id,
    )

    if legacy_build:
        print(
            "WARNING: legacy checkpoint — 이전 실행은 embedding_build_id 를 "
            "기록하지 않았습니다. 이미 적재된 point 는 provenance 없는 legacy "
            "로 남고, 이번 실행부터 적재되는 point 에만 id 가 붙습니다."
        )

    # Replay the previous batch once when resuming.
    # Point IDs are deterministic, so repeated upserts overwrite safely.
    if saved_index > 0:
        start_index = max(
            0,
            saved_index
            - args.batch_size,
        )

        print(
            f"resume: checkpoint={saved_index:,}"
        )

        print(
            f"safety replay from={start_index:,}"
        )

    else:
        start_index = 0

    # --------------------------------------------------------
    # 4. Determine input format
    # --------------------------------------------------------

    _, fmt = from_rfdetr(
        crops[:1],
        load_mode="path",
    )

    print(
        "input_format:",
        fmt,
    )

    # --------------------------------------------------------
    # 5. Qdrant
    #
    # Validate the Qdrant connection and collection schemas first.
    # This avoids loading every GPU model only to fail later on Qdrant setup.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("QDRANT INITIALIZE")
    print("=" * 70)

    person_store = QdrantStore(
        person_cfg
    )

    object_store = QdrantStore(
        object_cfg
    )

    # resume 인데 대상 컬렉션이 없거나 비어 있으면 checkpoint 와 DB 의 연속성이 깨진 것이다
    # (DB 볼륨 교체/삭제 등). 빈 컬렉션을 새로 만들고 중간부터 이어가면 앞부분이 통째로
    # 빠진 채 BUILD COMPLETED 까지 간다. 여기서 멈춘다.
    if saved_index > 0 and not args.fresh:
        for store, name in ((person_store, person_cfg.collection), (object_store, object_cfg.collection)):
            _assert_resume_target(store, name, saved_index)

    person_store.ensure_collection(
        recreate=args.recreate
    )

    object_store.ensure_collection(
        recreate=args.recreate
    )

    print(
        "Qdrant ready:",
        person_cfg.collection,
        list(person_cfg.retrievers.keys()),
    )

    print(
        "Qdrant ready:",
        object_cfg.collection,
        list(object_cfg.retrievers.keys()),
    )

    # 두 store 모두 이 build 의 id 를 payload 에 붙인다.
    person_store.embedding_build_id = build_id
    object_store.embedding_build_id = build_id

    # --------------------------------------------------------
    # 6. Embedders + Router
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOAD EMBEDDERS")
    print("=" * 70)

    registry = EmbedderRegistry(
        cfg
    )

    # Large-scale embedding 전에 모든 retriever 의 런타임 설정을 yaml 과
    # 대조하고 기록한다. (예전에는 SigLIP2 만 확인했다.)
    runtime_fp = runtime_retriever_fingerprint(
        cfg,
        registry,
    )

    # resume 이면 이전 실행의 런타임 지문과도 대조한다. yaml 이 같아도
    # 코드/환경 차이(예: DINOv2 dtype, 임베더 기본값 변경)로 공간이 달라질
    # 수 있고, 그 벡터를 기존 point 와 섞으면 안 된다.
    previous_rt = state.get("runtime_fingerprint")

    if previous_rt is not None:
        rt_diff = [
            name
            for name in sorted(set(previous_rt) | set(runtime_fp))
            if _canon(previous_rt.get(name)) != _canon(runtime_fp.get(name))
        ]

        if rt_diff:
            raise RuntimeError(
                "이어서 진행하려는 checkpoint 는 다른 런타임 설정으로 "
                f"만들어졌습니다: {', '.join(rt_diff)}\n"
                "구/신 벡터가 한 컬렉션에 섞이므로 중단합니다. "
                "--fresh 로 새로 구축하세요."
            )

    elif saved_index > 0:
        print(
            "WARNING: legacy checkpoint — 이전 실행의 런타임 지문이 없어 "
            "대조하지 못했습니다."
        )

    # --------------------------------------------------------
    # build manifest (불변)
    # --------------------------------------------------------

    manifest = {
        "schema_version": 1,
        "embedding_build_id": build_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "compat": _canon(_compat_subset(run_info)),
        "runtime_fingerprint": _canon(runtime_fp),
        "collections": {
            "person": person_cfg.collection,
            "object": object_cfg.collection,
        },
        "audit": _canon(audit_info),
    }

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path, manifest_created = write_build_manifest(
        MANIFEST_DIR,
        manifest,
    )

    print(
        "build manifest:",
        manifest_path,
        "(created)" if manifest_created else "(existing, verified)",
    )

    router = Router(
        cfg,
        registry,
        input_format=fmt,
    )

    print(
        "Router ready"
    )

    # --------------------------------------------------------
    # 7. Full embedding
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FULL EMBEDDING START")
    print("=" * 70)

    print(
        f"start : {start_index:,}"
    )

    print(
        f"total : {total:,}"
    )

    print(
        f"batch : router={args.batch_size} "
        f"upsert={args.upsert_batch_size} "
        f"upload_workers={args.upsert_parallel} "
        f"decode_workers={args.decode_workers}"
    )

    run_start = time.time()

    processed_this_run = 0

    try:

        for start in range(
            start_index,
            total,
            args.batch_size,
        ):

            end = min(
                start
                + args.batch_size,
                total,
            )

            batch_start_time = (
                time.time()
            )

            # --------------------------------------------
            # RF-DETR metadata -> Detection
            # --------------------------------------------

            batch_records = crops[
                start:end
            ]

            detections, batch_fmt = (
                from_rfdetr(
                    batch_records,
                    load_mode="path",
                )
            )

            if batch_fmt != fmt:
                raise RuntimeError(
                    "input_format change detected: "
                    f"{fmt} -> {batch_fmt}"
                )

            if (
                len(detections)
                != len(batch_records)
            ):
                raise RuntimeError(
                    "from_rfdetr conversion count mismatch: "
                    f"{len(batch_records)} -> "
                    f"{len(detections)}"
                )

            # --------------------------------------------
            # Core routing
            #
            # all    -> SigLIP2
            # person -> IRRA + SOLIDER
            # object -> DINOv2
            # --------------------------------------------

            vectors = router.embed(
                detections
            )

            # Router routing sanity check
            validate_router_result(
                detections,
                vectors,
                cfg,
            )

            # --------------------------------------------
            # Count vectors for logging
            # --------------------------------------------

            vector_counts = {}

            for vector_map in vectors:
                for name in vector_map:
                    vector_counts[name] = (
                        vector_counts.get(
                            name,
                            0,
                        )
                        + 1
                    )

            # --------------------------------------------
            # Qdrant
            # --------------------------------------------

            person_detections = []
            person_vectors = []

            object_detections = []
            object_vectors = []

            for det, vector_map in zip(
                detections,
                vectors,
            ):
                if is_person(det, cfg):
                    # Store only siglip2 / irra / solider in the person collection
                    filtered = {
                        name: vector
                        for name, vector in vector_map.items()
                        if name in person_cfg.retrievers
                    }
                    person_detections.append(det)
                    person_vectors.append(filtered)

                else:
                    # Store only siglip2 / dinov2 in the object collection
                    filtered = {
                        name: vector
                        for name, vector in vector_map.items()
                        if name in object_cfg.retrievers
                    }
                    object_detections.append(det)
                    object_vectors.append(filtered)

            uploaded_person = 0
            uploaded_object = 0

            if person_detections:
                uploaded_person = person_store.upsert(
                    person_detections,
                    person_vectors,
                    batch_size=args.upsert_batch_size,
                    parallel=args.upsert_parallel,
                )

            if object_detections:
                uploaded_object = object_store.upsert(
                    object_detections,
                    object_vectors,
                    batch_size=args.upsert_batch_size,
                    parallel=args.upsert_parallel,
                )

            # 제출 == committed, skipped == 0. 어긋나면 checkpoint 를 쓰기 전에
            # 중단한다 — 그래야 재실행이 이 batch 를 다시 시도한다.
            assert_upsert_complete(
                person_store,
                len(person_detections),
                "person",
            )

            assert_upsert_complete(
                object_store,
                len(object_detections),
                "object",
            )

            uploaded = uploaded_person + uploaded_object

            # --------------------------------------------
            # Checkpoint
            # Write checkpoint state only after Qdrant upsert succeeds.
            # --------------------------------------------

            state = {
                "run_info":
                    run_info,

                "audit":
                    audit_info,

                "embedding_build_id":
                    build_id,

                "runtime_fingerprint":
                    runtime_fp,

                "next_index":
                    end,
            }

            atomic_write_json(
                STATE_PATH,
                state,
            )

            processed_this_run += (
                end - start
            )

            # --------------------------------------------
            # Progress
            # --------------------------------------------

            batch_elapsed = (
                time.time()
                - batch_start_time
            )

            elapsed = (
                time.time()
                - run_start
            )

            rate = (
                processed_this_run
                / elapsed
                if elapsed > 0
                else 0
            )

            remaining = (
                total - end
            )

            eta = (
                remaining / rate
                if rate > 0
                else 0
            )

            counts_text = ", ".join(
                f"{name}={count}"
                for name, count
                in sorted(
                    vector_counts.items()
                )
            )

            profile_text = ", ".join(
                f"{name}={seconds:.1f}s"
                for name, seconds in getattr(router, "last_profile", {}).items()
            )

            print(
                f"[{end:,}/{total:,}] "
                f"{end / total * 100:6.2f}% | "
                f"batch {batch_elapsed:6.1f}s | "
                f"{rate:6.2f} crop/s | "
                f"ETA {format_seconds(eta)} | "
                f"Qdrant={uploaded} "
                f"(person={uploaded_person}, object={uploaded_object}) | "
                f"{counts_text} | {profile_text}"
            )

            # Release batch references
            del vectors
            del detections

    except KeyboardInterrupt:

        print(
            "\nCtrl+C detected."
        )

        print(
            "Progress up to the latest checkpoint has been saved."
        )

        print(
            "Run the same command again to resume from the checkpoint."
        )

        # 중단은 성공이 아니다.
        #
        # GUI 는 이 스크립트를 subprocess 로 실행하고 종료 코드로 성패를
        # 판별한다. 여기서 return 하면 exit code 0 이 되어, 사용자가 중단시킨
        # 실행이 "완료" 로 표시된다.
        #
        # 130 = 128 + SIGINT. Ctrl+C 중단에 대한 셸 관례값이다.
        sys.exit(130)

    # --------------------------------------------------------
    # 8. Finished
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BUILD COMPLETED")
    print("=" * 70)

    print(
        "total:",
        f"{total:,}",
    )

    print(
        "collections:",
        f"{person_cfg.collection} / {object_cfg.collection}",
    )

    print(
        "stats:",
        stats_path,
    )

    print(
        "checkpoint:",
        STATE_PATH,
    )

    print(
        "embedding_build_id:",
        build_id,
    )

    print(
        "manifest:",
        manifest_path,
    )


if __name__ == "__main__":
    main()