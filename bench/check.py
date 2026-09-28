"""bench/check.py — 구성 요소 계약(공통 입출력 규격) 검사 + DB 호환성 검사.

새 모델을 yaml 에 등록했을 때 "파이프라인이 그 모듈을 그대로 쓸 수 있는가"를 모델을 다 돌리기 전에 확인한다.

  python bench/check.py [--config pipeline.yaml] [--tracking-config pipeline_tracking.yaml]*
                        [--method leiden | --method-config y.yaml] [--instantiate] [--db] [--json out.json]

정적 검사 (모델 로드 없음):
  retriever   module import · class 존재 · BaseEmbedder 상속(또는 embed_crops) · DIM 클래스값 vs yaml dim ·
              __init__ 이 params 를 받는지(필수 누락/알 수 없는 키) · supports_text 면 embed_text
  detector    BaseDetector 상속(또는 detect) · params vs __init__            (tracker/stitcher: update·reset / stitch)
  clusterer   BaseClusterer 상속 · cluster · params() · required_vectors
--instantiate: 실제로 만들어 작은 입력으로 출력 규격 확인
  retriever   2장 → (2, dim) float32 유한값 (+ 텍스트 1건 → (1, dim))
  detector    빈 프레임 1장 → Detection 리스트 (bbox 4 · confidence · class_id · class_name)
  clusterer   합성 두 덩어리 (N=80) → ClusterResult.labels 길이 N
--db: pipeline.yaml 의 retriever 선언 지문(ingest.build_db.declared_retriever_fingerprint) 을
      운영 DB point 의 embedding_build_id → data/build_manifests/<id>.json 의 compat 지문과 대조.
      다르면 그 컬렉션의 벡터는 지금 yaml 의 임베더로 만든 것이 아니다 (검색·클러스터링 결과를 비교하면 안 됨 → 재적재).
종료 코드: FAIL 이 하나라도 있으면 1.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST_DIR = PROJECT_ROOT / "data" / "build_manifests"


class Row(dict):
    """검사 결과 한 줄: target, check, status(OK|FAIL|WARN|SKIP), detail."""


def row(target: str, check: str, status: str, detail: str = "") -> Row:
    return Row(target=target, check=check, status=status, detail=detail)


# ---------------------------------------------------------------- 공통
def import_class(module: str, cls: str) -> Tuple[Optional[type], Optional[str]]:
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001
        return None, f"import 실패: {type(exc).__name__}: {exc}"
    obj = getattr(mod, cls, None)
    if obj is None:
        return None, f"'{module}' 에 '{cls}' 없음"
    if not inspect.isclass(obj):
        return None, f"'{cls}' 는 클래스가 아님"
    return obj, None


def check_init_params(cls: type, params: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """(필수인데 빠진 키, 생성자가 모르는 키). **kwargs 가 있으면 모르는 키는 없음."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return [], []
    names = [p for p in sig.parameters.values() if p.name != "self"]
    var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in names)
    accepted = {p.name for p in names if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    required = [p.name for p in names if p.default is inspect.Parameter.empty
                and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)]
    missing = [k for k in required if k not in params]
    unknown = [] if var_kw else [k for k in params if k not in accepted]
    return missing, unknown


def has_methods(cls: type, names: Sequence[str], base: Optional[type] = None) -> List[str]:
    """없거나(callable 아님), base 가 주어지면 base 의 것을 그대로 물려받아 재정의하지 않은 메서드 이름."""
    out = []
    for n in names:
        fn = getattr(cls, n, None)
        if not callable(fn):
            out.append(n)
        elif base is not None and fn is getattr(base, n, None):
            out.append(f"{n}(미구현: {base.__name__} 의 것 그대로)")
    return out


# ---------------------------------------------------------------- retriever
def check_retrievers(config_path: str, instantiate: bool, log: Callable[[str], Any] = print) -> List[Row]:
    rows: List[Row] = []
    try:
        from config import PipelineConfig
        cfg = PipelineConfig.load(config_path)
    except Exception as exc:  # noqa: BLE001
        return [row("pipeline.yaml", "load", "FAIL", f"{type(exc).__name__}: {exc}")]
    try:
        from embedders.base import BaseEmbedder
    except Exception:  # noqa: BLE001
        BaseEmbedder = None  # type: ignore
    registry = None
    for name, spec in cfg.retrievers.items():
        t = f"retriever:{name}"
        cls, err = import_class(spec.module, spec.class_name)
        if cls is None:
            rows.append(row(t, "import", "FAIL", err or ""))
            continue
        rows.append(row(t, "import", "OK", f"{spec.module}.{spec.class_name}"))
        is_base = BaseEmbedder is not None and issubclass(cls, BaseEmbedder)
        # BaseEmbedder 상속이면 실제 구현 지점은 _encode (embed_crops 는 base 가 제공), 아니면 embed_crops 자체가 있어야 한다
        missing_m = has_methods(cls, ["_encode"], BaseEmbedder) if is_base else has_methods(cls, ["embed_crops"])
        rows.append(row(t, "contract", "OK" if not missing_m else "FAIL",
                        ("BaseEmbedder + _encode 구현" if is_base else "embed_crops 있음") if not missing_m else f"메서드 없음: {missing_m}"))
        dim = getattr(cls, "DIM", None)
        if isinstance(dim, int) and dim > 0:
            rows.append(row(t, "dim", "OK" if dim == int(spec.dim) else "FAIL", f"class DIM {dim} vs yaml {spec.dim}"))
        else:
            rows.append(row(t, "dim", "SKIP", f"DIM 은 인스턴스에서 결정 (yaml {spec.dim}); --instantiate 로 확인"))
        missing, unknown = check_init_params(cls, dict(spec.params or {}))
        rows.append(row(t, "params", "FAIL" if (missing or unknown) else "OK",
                        (f"필수 누락 {missing} " if missing else "") + (f"알 수 없는 키 {unknown}" if unknown else "") or f"{len(spec.params or {})} 개"))
        if spec.supports_text:
            rows.append(row(t, "text", "OK" if callable(getattr(cls, "embed_text", None)) else "FAIL", "supports_text → embed_text"))
        if instantiate:
            try:
                import numpy as np
                from PIL import Image
                if registry is None:
                    from registry import EmbedderRegistry
                    registry = EmbedderRegistry(cfg)
                t0 = time.time()
                obj = registry.get(name)
                imgs = [Image.new("RGB", (64, 128), (120, 60, 30)), Image.new("RGB", (80, 160), (30, 120, 200))]
                vec = obj.embed_crops(imgs)
                ok = (isinstance(vec, np.ndarray) and vec.shape == (2, int(spec.dim)) and vec.dtype == np.float32 and np.isfinite(vec).all())
                norms = np.linalg.norm(vec, axis=1) if isinstance(vec, np.ndarray) and vec.ndim == 2 else None
                rows.append(row(t, "instantiate", "OK" if ok else "FAIL",
                                f"embed_crops(2장) → {getattr(vec, 'shape', None)} {getattr(vec, 'dtype', None)} · L2 norm {np.round(norms, 3).tolist() if norms is not None else '?'} · {time.time() - t0:.1f}s"))
                if spec.supports_text and callable(getattr(obj, "embed_text", None)):
                    tv = obj.embed_text(["a person in a red jacket"])
                    tok = isinstance(tv, np.ndarray) and tv.shape == (1, int(spec.dim)) and np.isfinite(tv).all()
                    rows.append(row(t, "instantiate_text", "OK" if tok else "FAIL", f"embed_text(1) → {getattr(tv, 'shape', None)}"))
            except Exception as exc:  # noqa: BLE001
                rows.append(row(t, "instantiate", "FAIL", f"{type(exc).__name__}: {exc}"))
    if registry is not None:
        try:
            registry.release()
        except Exception:  # noqa: BLE001
            pass
    return rows


# ---------------------------------------------------------------- detector / tracker / stitcher
_VIDEO_BLOCKS = {
    "detector": ("BaseDetector", ["detect"]),
    "tracker": ("BaseTracker", ["update", "reset"]),
    "stitcher": ("BaseStitcher", ["stitch"]),
}


def check_tracking_yaml(path: str, instantiate: bool, log: Callable[[str], Any] = print) -> List[Row]:
    rows: List[Row] = []
    try:
        import yaml
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
    except Exception as exc:  # noqa: BLE001
        return [row(path, "load", "FAIL", f"{type(exc).__name__}: {exc}")]
    from detect import base as dbase
    for block, (base_name, methods) in _VIDEO_BLOCKS.items():
        spec = raw.get(block)
        if not isinstance(spec, dict):
            continue
        t = f"{Path(path).name}:{block}"
        module, cls_name, params = str(spec.get("module", "")), str(spec.get("class", "")), dict(spec.get("params") or {})
        cls, err = import_class(module, cls_name)
        if cls is None:
            rows.append(row(t, "import", "FAIL", err or ""))
            continue
        rows.append(row(t, "import", "OK", f"{module}.{cls_name}"))
        base = getattr(dbase, base_name, None)
        is_base = base is not None and issubclass(cls, base)
        missing_m = has_methods(cls, methods, base if is_base else None)
        rows.append(row(t, "contract", "OK" if not missing_m else "FAIL",
                        (f"{base_name} + {','.join(methods)} 재정의" if is_base else "메서드 있음 " + ",".join(methods)) if not missing_m
                        else f"메서드 없음: {missing_m}"))
        missing, unknown = check_init_params(cls, params)
        rows.append(row(t, "params", "FAIL" if (missing or unknown) else "OK",
                        (f"필수 누락 {missing} " if missing else "") + (f"알 수 없는 키 {unknown}" if unknown else "") or f"{len(params)} 개"))
        if instantiate and block == "detector":
            try:
                import numpy as np
                from detect.loader import load_component
                t0 = time.time()
                det = load_component({"module": module, "class": cls_name, "params": params})
                frame = np.zeros((320, 320, 3), dtype=np.uint8)
                out = det.detect(frame, frame_idx=0)
                ok = isinstance(out, list)
                bad = []
                for d in out[:5]:
                    for attr in ("bbox", "confidence", "class_id", "class_name"):
                        if not hasattr(d, attr):
                            bad.append(attr)
                    if hasattr(d, "bbox") and len(d.bbox) != 4:
                        bad.append("bbox!=4")
                rows.append(row(t, "instantiate", "OK" if (ok and not bad) else "FAIL",
                                f"detect(빈 320×320) → {type(out).__name__}[{len(out) if isinstance(out, list) else '?'}]"
                                f"{(' 필드 문제 ' + str(sorted(set(bad)))) if bad else ''} · conf_threshold={getattr(det, 'conf_threshold', None)} · {time.time() - t0:.1f}s"))
            except Exception as exc:  # noqa: BLE001
                rows.append(row(t, "instantiate", "FAIL", f"{type(exc).__name__}: {exc}"))
    return rows


# ---------------------------------------------------------------- clusterer
def check_clusterer(method: Optional[str], method_config: Optional[str], instantiate: bool,
                    log: Callable[[str], Any] = print) -> List[Row]:
    rows: List[Row] = []
    from clustering.base import BaseClusterer, ClusterResult, load_clusterer, resolve_clusterer_spec
    try:
        spec = resolve_clusterer_spec(method, None, None, [], method_config)
    except Exception as exc:  # noqa: BLE001
        return [row(f"clusterer:{method or method_config}", "spec", "FAIL", f"{type(exc).__name__}: {exc}")]
    t = f"clusterer:{spec['class']}"
    cls, err = import_class(spec["module"], spec["class"])
    if cls is None:
        return [row(t, "import", "FAIL", err or "")]
    rows.append(row(t, "import", "OK", f"{spec['module']}.{spec['class']}"))
    is_base = issubclass(cls, BaseClusterer)
    missing_m = has_methods(cls, ["cluster"], BaseClusterer if is_base else None)
    rows.append(row(t, "contract", "OK" if not missing_m else "FAIL",
                    ("BaseClusterer + cluster 재정의" if is_base else "cluster 있음") if not missing_m else f"메서드 없음: {missing_m}"))
    missing, unknown = check_init_params(cls, dict(spec.get("params") or {}))
    rows.append(row(t, "params", "FAIL" if (missing or unknown) else "OK",
                    (f"필수 누락 {missing} " if missing else "") + (f"알 수 없는 키 {unknown}" if unknown else "") or f"{len(spec.get('params') or {})} 개"))
    if instantiate:
        try:
            import numpy as np
            obj = load_clusterer(spec)
            rng = np.random.default_rng(0)
            n, d = 80, 16
            a = rng.normal(0, 0.05, (n // 2, d)) + np.eye(d)[0]
            b = rng.normal(0, 0.05, (n // 2, d)) + np.eye(d)[1]
            x = np.vstack([a, b]).astype(np.float32)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            ids = [f"p{i}" for i in range(n)]
            vectors = {name: x for name in dict.fromkeys(["primary", *getattr(obj, "required_vectors", ())])}
            t0 = time.time()
            res = obj.cluster(ids, x, vectors, log=lambda *_a, **_k: None)
            ok = isinstance(res, ClusterResult) and len(res.labels) == n
            k = len({l for l in res.labels if l is not None and l >= 0}) if ok else None
            rows.append(row(t, "instantiate", "OK" if ok else "FAIL",
                            f"cluster(합성 2덩어리 N={n}) → labels {len(res.labels) if ok else '?'} · 군집 {k} · "
                            f"required_vectors={list(getattr(obj, 'required_vectors', ()))} · {time.time() - t0:.1f}s"))
        except Exception as exc:  # noqa: BLE001
            rows.append(row(t, "instantiate", "FAIL", f"{type(exc).__name__}: {exc}"))
    return rows


# ---------------------------------------------------------------- DB 호환성
def scroll_build_ids(url: str, collection: str, limit: int = 2000, batch: int = 500) -> Dict[str, int]:
    """point payload 의 embedding_build_id 분포 (앞 limit 개)."""
    import requests
    counts: Dict[str, int] = {}
    offset = None
    seen = 0
    while seen < limit:
        body: Dict[str, Any] = {"limit": min(batch, limit - seen), "with_payload": ["embedding_build_id"], "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        r = requests.post(f"{url.rstrip('/')}/collections/{collection}/points/scroll", json=body, timeout=60)
        r.raise_for_status()
        res = r.json().get("result") or {}
        pts = res.get("points") or []
        for p in pts:
            bid = (p.get("payload") or {}).get("embedding_build_id") or "(legacy: id 없음)"
            counts[bid] = counts.get(bid, 0) + 1
        seen += len(pts)
        offset = res.get("next_page_offset")
        if not pts or offset is None:
            break
    return counts


def compare_manifest(declared: Dict[str, Any], manifest: Dict[str, Any], canon: Callable[[Any], Any]) -> Dict[str, str]:
    """retriever 별 'same' / 'different' / 'missing'."""
    compat = (manifest.get("compat") or {}).get("retrievers") or {}
    out: Dict[str, str] = {}
    for name, fp in declared.items():
        if name not in compat:
            out[name] = "missing"
        else:
            out[name] = "same" if canon(compat[name]) == canon(fp) else "different"
    return out


def check_db(config_path: str, manifest_dir: Path = MANIFEST_DIR, sample: int = 2000, log: Callable[[str], Any] = print) -> List[Row]:
    rows: List[Row] = []
    try:
        from config import PipelineConfig
        from ingest.build_db import _canon, declared_retriever_fingerprint
        from report_common import load_pipeline_settings
        cfg = PipelineConfig.load(config_path)
        settings = load_pipeline_settings(config_path)
        declared = declared_retriever_fingerprint(cfg)
    except Exception as exc:  # noqa: BLE001
        return [row("db", "fingerprint", "FAIL", f"{type(exc).__name__}: {exc}")]
    rows.append(row("db", "declared", "OK", "retrievers " + ", ".join(f"{n}(dim {v['dim']}, files {len(v['files'])})" for n, v in declared.items())))
    for target in ("person", "object"):
        collection = settings.collection_for(target)
        t = f"db:{collection}"
        try:
            counts = scroll_build_ids(settings.qdrant_url, collection, sample)
        except Exception as exc:  # noqa: BLE001
            rows.append(row(t, "scroll", "FAIL", f"Qdrant 조회 실패: {type(exc).__name__}: {exc}"))
            continue
        if not counts:
            rows.append(row(t, "scroll", "WARN", "point 없음"))
            continue
        rows.append(row(t, "build_ids", "OK", f"표본 {sum(counts.values()):,}: " + ", ".join(f"{k}×{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:6])))
        for bid in sorted(counts):
            if bid.startswith("(legacy"):
                rows.append(row(t, "manifest", "WARN", f"embedding_build_id 없는 point {counts[bid]:,} — 지문 대조 불가"))
                continue
            path = manifest_dir / f"{bid}.json"
            if not path.is_file():
                rows.append(row(t, "manifest", "WARN", f"{bid}: manifest 파일 없음 ({path})"))
                continue
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                rows.append(row(t, "manifest", "WARN", f"{bid}: 읽기 실패 {exc}"))
                continue
            cmp = compare_manifest(declared, manifest, _canon)
            relevant = {n: s for n, s in cmp.items() if getattr(cfg.retrievers[n], "accepts_" + target)()}
            # different = 다른 임베더로 만든 벡터, missing = 그 named vector 를 이 build 가 만들지 않음 → 둘 다 호환 아님
            bad = [n for n, s in relevant.items() if s in ("different", "missing")]
            status = "FAIL" if bad else "OK"
            same_yaml = (manifest.get("compat") or {}).get("pipeline_sha256") == settings.config_sha256
            rows.append(row(t, "compat", status, f"{bid} ({counts[bid]:,} pt): " + ", ".join(f"{n}={s}" for n, s in relevant.items())
                            + (" · yaml 동일" if same_yaml else " · yaml 은 바뀜(지문으로 판단)")))
    return rows


# ---------------------------------------------------------------- spec 단위 검사 (register.py 가 쓴다)
def check_detector_spec(spec: Dict[str, Any], instantiate: bool, target: str = "detector") -> List[Row]:
    rows: List[Row] = []
    from detect import base as dbase
    module, cls_name, params = str(spec.get("module", "")), str(spec.get("class", "")), dict(spec.get("params") or {})
    cls, err = import_class(module, cls_name)
    if cls is None:
        return [row(target, "import", "FAIL", err or "")]
    rows.append(row(target, "import", "OK", f"{module}.{cls_name}"))
    is_base = issubclass(cls, dbase.BaseDetector)
    missing_m = has_methods(cls, ["detect"], dbase.BaseDetector if is_base else None)
    rows.append(row(target, "contract", "OK" if not missing_m else "FAIL",
                    ("BaseDetector + detect 재정의" if is_base else "detect 있음") if not missing_m else f"메서드 없음: {missing_m}"))
    missing, unknown = check_init_params(cls, params)
    rows.append(row(target, "params", "FAIL" if (missing or unknown) else "OK",
                    (f"필수 누락 {missing} " if missing else "") + (f"알 수 없는 키 {unknown}" if unknown else "") or f"{len(params)} 개"))
    if instantiate:
        try:
            import numpy as np
            from detect.loader import load_component
            t0 = time.time()
            det = load_component({"module": module, "class": cls_name, "params": params})
            out = det.detect(np.zeros((320, 320, 3), dtype=np.uint8), frame_idx=0)
            ok = isinstance(out, list)
            bad = []
            for d in out[:5]:
                for attr in ("bbox", "confidence", "class_id", "class_name"):
                    if not hasattr(d, attr):
                        bad.append(attr)
                if hasattr(d, "bbox") and len(d.bbox) != 4:
                    bad.append("bbox!=4")
            if not hasattr(det, "conf_threshold"):
                bad.append("conf_threshold 속성 없음 (채점·러너가 읽음)")
            rows.append(row(target, "instantiate", "OK" if (ok and not bad) else "FAIL",
                            f"detect(빈 320×320) → {type(out).__name__}[{len(out) if isinstance(out, list) else '?'}]"
                            f"{(' 문제 ' + str(sorted(set(bad)))) if bad else ''} · {time.time() - t0:.1f}s"))
        except Exception as exc:  # noqa: BLE001
            rows.append(row(target, "instantiate", "FAIL", f"{type(exc).__name__}: {exc}"))
    return rows


def check_clusterer_spec(spec: Dict[str, Any], instantiate: bool, target: Optional[str] = None) -> List[Row]:
    rows: List[Row] = []
    from clustering.base import BaseClusterer, ClusterResult, load_clusterer
    t = target or f"clusterer:{spec.get('class')}"
    cls, err = import_class(str(spec.get("module", "")), str(spec.get("class", "")))
    if cls is None:
        return [row(t, "import", "FAIL", err or "")]
    rows.append(row(t, "import", "OK", f"{spec['module']}.{spec['class']}"))
    is_base = issubclass(cls, BaseClusterer)
    missing_m = has_methods(cls, ["cluster"], BaseClusterer if is_base else None)
    rows.append(row(t, "contract", "OK" if not missing_m else "FAIL",
                    ("BaseClusterer + cluster 재정의" if is_base else "cluster 있음") if not missing_m else f"메서드 없음: {missing_m}"))
    missing, unknown = check_init_params(cls, dict(spec.get("params") or {}))
    rows.append(row(t, "params", "FAIL" if (missing or unknown) else "OK",
                    (f"필수 누락 {missing} " if missing else "") + (f"알 수 없는 키 {unknown}" if unknown else "") or f"{len(spec.get('params') or {})} 개"))
    if instantiate:
        try:
            import numpy as np
            obj = load_clusterer(spec)
            rng = np.random.default_rng(0)
            n, d = 80, 16
            a = rng.normal(0, 0.05, (n // 2, d)) + np.eye(d)[0]
            b = rng.normal(0, 0.05, (n // 2, d)) + np.eye(d)[1]
            x = np.vstack([a, b]).astype(np.float32)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            ids = [f"p{i}" for i in range(n)]
            vectors = {name: x for name in dict.fromkeys(["primary", *getattr(obj, "required_vectors", ())])}
            t0 = time.time()
            res = obj.cluster(ids, x, vectors, log=lambda *_a, **_k: None)
            ok = isinstance(res, ClusterResult) and len(res.labels) == n
            k = len({lab for lab in res.labels if lab is not None and lab >= 0}) if ok else None
            rows.append(row(t, "instantiate", "OK" if ok else "FAIL",
                            f"cluster(합성 2덩어리 N={n}) → labels {len(res.labels) if ok else '?'} · 군집 {k} · "
                            f"required_vectors={list(getattr(obj, 'required_vectors', ()))} · {time.time() - t0:.1f}s"))
        except Exception as exc:  # noqa: BLE001
            rows.append(row(t, "instantiate", "FAIL", f"{type(exc).__name__}: {exc}"))
    return rows


def check_embedder_spec(spec: Dict[str, Any], dim: int, supports_text: bool, instantiate: bool, target: Optional[str] = None) -> List[Row]:
    rows: List[Row] = []
    t = target or f"retriever:{spec.get('class')}"
    try:
        from embedders.base import BaseEmbedder
    except Exception:  # noqa: BLE001
        BaseEmbedder = None  # type: ignore
    cls, err = import_class(str(spec.get("module", "")), str(spec.get("class", "")))
    if cls is None:
        return [row(t, "import", "FAIL", err or "")]
    rows.append(row(t, "import", "OK", f"{spec['module']}.{spec['class']}"))
    is_base = BaseEmbedder is not None and issubclass(cls, BaseEmbedder)
    missing_m = has_methods(cls, ["_encode"], BaseEmbedder) if is_base else has_methods(cls, ["embed_crops"])
    rows.append(row(t, "contract", "OK" if not missing_m else "FAIL",
                    ("BaseEmbedder + _encode 구현" if is_base else "embed_crops 있음") if not missing_m else f"메서드 없음: {missing_m}"))
    cdim = getattr(cls, "DIM", None)
    if isinstance(cdim, int) and cdim > 0:
        rows.append(row(t, "dim", "OK" if cdim == int(dim) else "FAIL", f"class DIM {cdim} vs yaml {dim}"))
    else:
        rows.append(row(t, "dim", "SKIP", f"DIM 은 인스턴스에서 결정 (yaml {dim})"))
    missing, unknown = check_init_params(cls, dict(spec.get("params") or {}))
    rows.append(row(t, "params", "FAIL" if (missing or unknown) else "OK",
                    (f"필수 누락 {missing} " if missing else "") + (f"알 수 없는 키 {unknown}" if unknown else "") or f"{len(spec.get('params') or {})} 개"))
    if supports_text:
        rows.append(row(t, "text", "OK" if callable(getattr(cls, "embed_text", None)) else "FAIL", "supports_text → embed_text"))
    if instantiate:
        try:
            import numpy as np
            from PIL import Image
            from detect.loader import load_component
            t0 = time.time()
            obj = load_component(spec)
            imgs = [Image.new("RGB", (64, 128), (120, 60, 30)), Image.new("RGB", (80, 160), (30, 120, 200))]
            vec = obj.embed_crops(imgs)
            ok = isinstance(vec, np.ndarray) and vec.shape == (2, int(dim)) and vec.dtype == np.float32 and np.isfinite(vec).all()
            rows.append(row(t, "instantiate", "OK" if ok else "FAIL", f"embed_crops(2장) → {getattr(vec, 'shape', None)} {getattr(vec, 'dtype', None)} · {time.time() - t0:.1f}s"))
            if supports_text and callable(getattr(obj, "embed_text", None)):
                tv = obj.embed_text(["a person in a red jacket"])
                tok = isinstance(tv, np.ndarray) and tv.shape == (1, int(dim)) and np.isfinite(tv).all()
                rows.append(row(t, "instantiate_text", "OK" if tok else "FAIL", f"embed_text(1) → {getattr(tv, 'shape', None)}"))
        except Exception as exc:  # noqa: BLE001
            rows.append(row(t, "instantiate", "FAIL", f"{type(exc).__name__}: {exc}"))
    return rows


# ---------------------------------------------------------------- 출력
def render(rows: Sequence[Row]) -> str:
    w = max((len(r["target"]) for r in rows), default=10)
    lines = []
    for r in rows:
        lines.append(f"{r['status']:<4} {r['target']:<{w}}  {r['check']:<16} {r['detail']}")
    fails = sum(1 for r in rows if r["status"] == "FAIL")
    warns = sum(1 for r in rows if r["status"] == "WARN")
    lines.append(f"— 검사 {len(rows)} · FAIL {fails} · WARN {warns}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="구성 요소 계약·호환성 검사")
    p.add_argument("--config", default=str(PROJECT_ROOT / "pipeline.yaml"))
    p.add_argument("--tracking-config", action="append", default=None, help="detector/tracker/stitcher yaml (여러 번; 기본 pipeline_tracking*.yaml 전부)")
    p.add_argument("--method", default=None, help="내장 클러스터러 이름")
    p.add_argument("--method-config", default=None, help="clusterer: yaml")
    p.add_argument("--no-retrievers", action="store_true")
    p.add_argument("--no-tracking", action="store_true")
    p.add_argument("--no-clusterer", action="store_true")
    p.add_argument("--instantiate", action="store_true", help="실제로 만들어 작은 입력으로 출력 규격 확인 (모델 로드)")
    p.add_argument("--db", action="store_true", help="운영 DB 의 임베딩 지문과 yaml 대조")
    p.add_argument("--json", default=None, help="결과 json 저장")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows: List[Row] = []
    if not args.no_retrievers:
        rows += check_retrievers(args.config, args.instantiate)
    if not args.no_tracking:
        paths = args.tracking_config or [str(p) for p in sorted(PROJECT_ROOT.glob("pipeline_tracking*.yaml"))]
        for path in paths:
            rows += check_tracking_yaml(path, args.instantiate)
    if not args.no_clusterer:
        if args.method or args.method_config:
            rows += check_clusterer(args.method, args.method_config, args.instantiate)
        else:
            from clustering.base import BUILTIN_METHODS
            for m in sorted(BUILTIN_METHODS):
                rows += check_clusterer(m, None, args.instantiate)
    if args.db:
        rows += check_db(args.config)
    print(render(rows))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": rows},
                                              ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    return 1 if any(r["status"] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
