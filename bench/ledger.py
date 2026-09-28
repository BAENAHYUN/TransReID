"""bench/ledger.py — 평가 실행 원장 (append-only JSONL).

검출·임베딩·통합검색·클러스터링 평가 스크립트가 결과를 한 곳에 남긴다:
  bench/ledger.jsonl    한 줄 = 한 실행의 한 method/variant

엔트리 스키마 (schema_version 1)
  run_id        "<stage>_<YYYYmmddTHHMMSS>_<fp8>"  (fp8 = fingerprint 앞 8자)
  fingerprint   sha1(stage, name, component, params, gt, metrics)[:16] — 이관(import) 중복 방지 키
  created_at    ISO 8601
  stage         detect | embed | search | cluster
  producer      만든 스크립트 (detect_eval_prw · prw_eval · prw_eval_unified · prw_cluster_gt_eval)
  name          method / variant 이름 (yolo26m_test, solider, single:solider, Leiden_exact_0.97 …)
  component     평가된 구성 요소 {module, class, params} 또는 {model: …} / {variant: …}
  params        하이퍼파라미터 + 평가 조건 (conf, iou, knn, threshold, weights …)
  gt            정답 집합 식별 {dataset, split, frames/boxes/points …}
  metrics       평평한 dict, 표준 이름 (METRIC_KEYS 참고; 백분율 지표는 % 단위 그대로)
  timing        {fps, latency_ms_mean, elapsed_sec, cluster_sec …}
  versions      {torch, cuda, gpu, rfdetr, ultralytics …}
  env           {git_commit, git_dirty, python, platform, host, cwd} (이관 엔트리는 imported_from)
  inputs        [{role, path, sha256, size}] 평가에 들어간 파일
  config        {path, sha256} 사용한 pipeline yaml
  report        사이드카 json 경로
  command       실행 인자 (argv)
  seed / note

CLI
  python bench/ledger.py import [--results-dir eval/results]      기존 산출물 이관 (멱등: fingerprint 중복 건너뜀)
  python bench/ledger.py list [--stage S] [--name 부분문자열] [--last N]
  python bench/ledger.py show RUN_ID
  python bench/ledger.py table --stage S [--format md|csv] [--latest] [--names a,b] [--metrics k1,k2]
공통: --ledger PATH (기본 bench/ledger.jsonl)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION = 1
DEFAULT_LEDGER = PROJECT_ROOT / "bench" / "ledger.jsonl"
# e2e = 전체 파이프라인(검출→crop→임베딩→DB→검색) 평가. 나머지는 단독 평가(고정 GT 입력).
STAGES = ("detect", "track", "embed", "search", "cluster", "e2e", "object", "qwen")

# 단계별 표(기준표) 기본 열 — metrics 의 표준 이름
METRIC_KEYS: Dict[str, List[str]] = {
    "detect": ["ap50", "ap50_95", "max_recall", "precision", "recall", "f1", "fp_duplicate", "fp_background", "fn",
               "recall_h75_119", "recall_h120_199", "fps"],
    "embed": ["map", "rank1", "rank5", "rank10"],
    "search": ["map", "rank1", "rank5", "rank10", "pool_recall"],
    "cluster": ["pair_precision", "pair_recall", "pair_f1", "b3_precision", "b3_recall", "b3_f1", "purity",
                "mixed_clusters", "pids_split", "noise_ratio"],
    "e2e": ["map", "map_db", "rank1", "rank5", "rank10", "recall_at_k", "det_ceiling", "distractor_ratio", "sec_per_query"],
    "track": ["idf1", "hota", "deta", "assa", "mota", "idsw", "fragments", "splits", "over_merges", "idf1_before", "idsw_before", "idsw_ratio"],
    "object": ["map", "rank1", "map_labeled", "pair_auc", "pair_f1", "pair_threshold", "pair_acc_at_threshold", "cluster_pair_precision",
               "cluster_pair_recall", "queries", "pairs_same", "pairs_diff"],
    "qwen": ["p10_before", "p10_after", "p10_gain_pp", "p5_before", "p5_after", "p20_before", "p20_after", "false_drop_rate", "unknown_ratio",
             "sec_per_candidate", "queries", "candidates"],
}

# verify 허용 오차 (지표 이름 → 절대 오차). 없는 지표는 비교하지 않는다(정보용: fps, sec …).
VERIFY_TOLERANCES: Dict[str, Dict[str, float]] = {
    "detect": {"ap50": 0.005, "ap50_95": 0.005, "max_recall": 0.005, "precision": 0.005, "recall": 0.005, "f1": 0.005,
               "recall_h75_119": 0.01, "recall_h120_199": 0.01, "recall_h200p": 0.01},
    "embed": {"map": 0.1, "rank1": 0.2, "rank5": 0.2, "rank10": 0.2},
    "search": {"map": 0.1, "rank1": 0.2, "rank5": 0.2, "rank10": 0.2, "pool_recall": 0.2},
    "cluster": {"pair_precision": 0.005, "pair_recall": 0.005, "pair_f1": 0.005, "b3_precision": 0.005, "b3_recall": 0.005,
                "b3_f1": 0.005, "purity": 0.005, "noise_ratio": 0.005},
    "e2e": {"map": 0.1, "map_db": 0.1, "rank1": 0.2, "rank5": 0.2, "rank10": 0.2, "recall_at_k": 0.2, "det_ceiling": 0.001},
    "track": {"idf1": 0.005, "hota": 0.005, "mota": 0.005, "idsw": 0, "over_merges": 0, "splits": 0},
    "object": {"map": 0.1, "rank1": 0.2, "map_labeled": 0.1, "pair_auc": 0.005, "pair_f1": 0.005, "cluster_pair_precision": 0.005},
    # Qwen 생성은 완전 결정적이지 않다 → 검증 전 P@K 만 엄격, 검증 후·오탈락은 느슨하게
    "qwen": {"p10_before": 0.01, "p10_after": 5.0, "false_drop_rate": 0.1, "unknown_ratio": 0.1},
}

_BUCKET_KEYS = {"<50": "recall_h_lt50", "50–74": "recall_h50_74", "75–119": "recall_h75_119",
                "120–199": "recall_h120_199", "200+": "recall_h200p"}
_CLUSTER_PARAM_KEYS = ("method", "vector", "knn", "score_threshold", "threshold", "resolution", "mutual_knn",
                       "max_cluster_size", "refine_rounds", "min_cluster_size", "seed", "eps", "min_faces", "weights",
                       "combined_vectors", "secondary_vector", "secondary_threshold", "identity_safe", "knn_mode", "exact",
                       "max_points", "target")


# ---------------------------------------------------------------- 기본 유틸
def resolve_ledger_path(value: Any = None, disabled: bool = False) -> Optional[Path]:
    """--ledger 값 → 경로. None = 기본 원장, '' / none / off / - = 기록 안 함."""
    if disabled:
        return None
    if value is None:
        return DEFAULT_LEDGER
    text = str(value).strip()
    if not text or text.lower() in ("none", "off", "-"):
        return None
    return Path(text).expanduser().resolve()


def file_sha256(path: Any, chunk: int = 1 << 20) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def git_info(root: str = str(PROJECT_ROOT)) -> Dict[str, Any]:
    info: Dict[str, Any] = {"git_commit": None, "git_dirty": None}
    try:
        r = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=root, capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            info["git_commit"] = r.stdout.strip() or None
        r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                           capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            info["git_dirty"] = bool(r.stdout.strip())
    except Exception:  # git 없음 / 타임아웃 — 원장 기록을 막지 않는다
        pass
    return info


def env_info() -> Dict[str, Any]:
    return {"python": platform.python_version(), "platform": platform.platform(), "host": socket.gethostname(),
            "cwd": os.getcwd(), **git_info()}


def hardware_info() -> Dict[str, Any]:
    """공정한 비교·재현을 위한 하드웨어 기록: GPU 이름/VRAM/CUDA, CPU 코어, RAM."""
    info: Dict[str, Any] = {"cpu": platform.processor() or platform.machine(), "os": platform.platform()}
    try:
        import psutil
        info["cpu_cores"] = psutil.cpu_count(logical=False)
        info["cpu_threads"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            info["gpu"] = prop.name
            info["vram_gb"] = round(prop.total_memory / 1e9, 1)
            info["cuda"] = torch.version.cuda
            info["torch"] = torch.__version__
        else:
            info["gpu"] = None
    except Exception:
        pass
    return info


def retriever_fingerprint_sha(config_path: Any, name: str, log: Callable[[str], Any] = print) -> str:
    """임베더(retriever) 선언 지문의 sha1 — module/class/params + 가중치 파일 내용 sha256
    (ingest.build_db.declared_retriever_fingerprint 재사용). yaml 의 다른 줄이 바뀌어도 그대로, 같은 경로의
    checkpoint 를 바꿔 끼우면 달라진다. 캐시(임베딩·벡터)의 호환성 키로 쓴다. 실패하면 ''."""
    try:
        from config import PipelineConfig
        from ingest.build_db import declared_retriever_fingerprint
        fp = declared_retriever_fingerprint(PipelineConfig.load(str(config_path))).get(name)
        if not fp:
            return ""
        return hashlib.sha1(json.dumps(fp, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")).hexdigest()
    except Exception as exc:  # noqa: BLE001
        log(f"[compat] retriever '{name}' 지문 계산 실패 ({type(exc).__name__}: {exc})")
        return ""


def versions_info() -> Dict[str, Any]:
    """이미 로드된 라이브러리 버전만 (임포트를 새로 하지 않아 가볍다)."""
    info: Dict[str, Any] = {}
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            info["torch"] = torch.__version__
            info["cuda"] = bool(torch.cuda.is_available())
            if info["cuda"]:
                info["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    for mod in ("transformers", "ultralytics", "rfdetr", "open_clip", "timm", "qdrant_client"):
        m = sys.modules.get(mod)
        if m is not None and getattr(m, "__version__", None):
            info[mod] = m.__version__
    return info


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "item"):  # numpy scalar
        try:
            return o.item()
        except Exception:
            pass
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def _dumps(value: Any, **kw: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default, **kw)


TIMING_METRICS = frozenset({"fps", "latency_ms_mean", "latency_ms_p50", "sec", "sec_per_query", "cluster_sec", "elapsed_sec", "sec_per_candidate", "wall_sec_per_candidate",
                            "load_sec", "search_sec"})


def fingerprint(entry: Dict[str, Any]) -> str:
    """이관 중복 방지 키: stage/name/component/params/gt + 시간 계열을 뺀 metrics (fps·sec 는 실행마다 달라 같은 결과를 다른 키로 만든다)."""
    metrics = {k: v for k, v in (entry.get("metrics") or {}).items() if k not in TIMING_METRICS}
    key = [entry.get("stage"), entry.get("name"), entry.get("component"), entry.get("params"), entry.get("gt"), metrics]
    return hashlib.sha1(_dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if value:
        try:
            return datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now().astimezone()


def _clean(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {k: v for k, v in (d or {}).items() if v is not None}


def make_entry(stage: str, producer: str, name: str, *, component: Optional[Dict[str, Any]] = None,
               params: Optional[Dict[str, Any]] = None, gt: Optional[Dict[str, Any]] = None,
               metrics: Optional[Dict[str, Any]] = None, timing: Optional[Dict[str, Any]] = None,
               versions: Optional[Dict[str, Any]] = None, inputs: Optional[Sequence[Dict[str, Any]]] = None,
               config: Optional[Dict[str, Any]] = None, report: Any = None, command: Optional[Sequence[str]] = None,
               seed: Any = None, note: Optional[str] = None, created_at: Any = None,
               env: Optional[Dict[str, Any]] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """extra: 최상위에 덧붙일 키 (bench / hardware / weights / verify …). fingerprint 에는 들어가지 않는다."""
    if stage not in STAGES:
        raise ValueError(f"stage 는 {STAGES} 중 하나: {stage!r}")
    if not name:
        raise ValueError("name 이 비었습니다")
    when = _parse_time(created_at)
    entry: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "run_id": None, "fingerprint": None,
        "created_at": when.isoformat(), "stage": stage, "producer": producer, "name": str(name),
        "component": _clean(component), "params": _clean(params), "gt": _clean(gt), "metrics": _clean(metrics),
        "timing": _clean(timing), "versions": _clean(versions), "env": env if env is not None else env_info(),
        "inputs": [dict(i) for i in (inputs or [])], "config": _clean(config),
        "report": str(report) if report else None, "command": list(command) if command is not None else None,
        "seed": seed, "note": note,
    }
    fp = fingerprint(entry)
    entry["fingerprint"] = fp
    entry["run_id"] = f"{stage}_{when.strftime('%Y%m%dT%H%M%S')}_{fp[:8]}"
    for key, value in (extra or {}).items():
        entry[key] = value
    return entry


# ---------------------------------------------------------------- 읽기 / 쓰기
def read_entries(path: Any, errors: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    out: List[Dict[str, Any]] = []
    with p.open(encoding="utf-8") as f:
        for no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                if errors is not None:
                    errors.append(f"{p}:{no}: {exc}")
                continue
            if isinstance(d, dict):
                out.append(d)
    return out


def existing_fingerprints(path: Any) -> set:
    return {e.get("fingerprint") for e in read_entries(path) if e.get("fingerprint")}


class _FileLock:
    """원장 append 를 프로세스 간 직렬화 (병렬 trial). <원장>.lock 파일에 배타 잠금 (Windows msvcrt / POSIX fcntl)."""

    def __init__(self, path: Path, timeout: float = 60.0):
        self.path = Path(str(path) + ".lock")
        self.timeout = timeout
        self.fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a+")
        deadline = time.time() + self.timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.time() > deadline:
                    raise TimeoutError(f"원장 잠금 대기 초과: {self.path}")
                time.sleep(0.05)

    def __exit__(self, *exc):
        try:
            if os.name == "nt":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        finally:
            self.fh.close()


def append_entries(path: Any, entries: Iterable[Dict[str, Any]], dedupe: bool = False) -> Tuple[int, int]:
    """엔트리를 원장에 추가 (파일 잠금 아래). dedupe=True 면 같은 fingerprint 가 이미 있으면 건너뜀. (기록 수, 건너뛴 수)"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    entries = list(entries)
    written = skipped = 0
    with _FileLock(p):
        seen = existing_fingerprints(p) if dedupe else set()
        with p.open("a", encoding="utf-8", newline="\n") as f:
            for e in entries:
                fp = e.get("fingerprint")
                if dedupe and fp in seen:
                    skipped += 1
                    continue
                f.write(_dumps(e) + "\n")
                seen.add(fp)
                written += 1
    return written, skipped


def record(builder: Callable[[], Sequence[Dict[str, Any]]], ledger: Any = None, disabled: bool = False,
           log: Callable[[str], Any] = print) -> Optional[Path]:
    """평가 스크립트용: 원장 경로를 정하고 builder() 의 엔트리를 추가. 실패해도 예외를 밖으로 내지 않는다."""
    try:
        path = resolve_ledger_path(ledger, disabled=disabled)
        if path is None:
            return None
        entries = list(builder())
        n, _ = append_entries(path, entries)
        ids = ", ".join(e["run_id"] for e in entries[:4]) + (" …" if len(entries) > 4 else "")
        log(f"[ledger] {n} 건 기록 → {path} ({ids})")
        return path
    except Exception as exc:  # noqa: BLE001 — 원장 실패가 평가 결과를 깨지 않게 (경로 해석·builder·append·log 전부)
        try:
            log(f"[ledger] 기록 실패 ({type(exc).__name__}: {exc}) — 결과 파일은 정상")
        except Exception:  # noqa: BLE001
            pass
        return None


def latest_by_name(entries: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """이름별 가장 최근 엔트리 (created_at 기준; 이관 행은 append 순서와 시간 순서가 다를 수 있다. 같으면 뒤가 이김)."""
    out: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        name = e.get("name", "")
        cur = out.get(name)
        if cur is None or str(e.get("created_at", "")) >= str(cur.get("created_at", "")):
            out[name] = e
    return out


# ---------------------------------------------------------------- 스크립트별 빌더
def _role_inputs(summary: Dict[str, Any], role: str) -> List[Dict[str, Any]]:
    return [dict(i) for i in (summary.get("inputs") or []) if i.get("role") == role]


def entries_from_detect_summary(summary: Dict[str, Any], report: Any = None, command: Optional[Sequence[str]] = None,
                                env: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """detect_eval_prw 의 사이드카(detect_eval_report.json) → method 당 엔트리."""
    cfg = summary.get("config") or {}
    gt = summary.get("gt") or {}
    metas = summary.get("metas") or {}
    entries = []
    for name, r in (summary.get("results") or {}).items():
        meta = metas.get(name) or {}
        det = meta.get("detector") or {}
        st = meta.get("stats") or {}
        op = r.get("operating") or {}
        metrics: Dict[str, Any] = {
            "ap50": r.get("ap50"), "ap50_95": r.get("ap50_95"), "max_recall": r.get("max_recall"),
            "precision": op.get("precision"), "recall": op.get("recall"), "f1": op.get("f1"),
            "fp_duplicate": op.get("fp_duplicate"), "fp_background": op.get("fp_background"), "fn": op.get("fn"),
            "dets_per_frame": op.get("dets_per_frame"), "frames": r.get("frames"), "gt_boxes": r.get("gt_boxes"),
            "detections": r.get("detections"), "latency_ms_mean": st.get("latency_ms_mean"), "fps": st.get("fps"),
        }
        for bucket, key in _BUCKET_KEYS.items():
            rb = (r.get("recall_by_size") or {}).get(bucket)
            if isinstance(rb, dict):
                metrics[key] = rb.get("recall")
        det_params = dict(det.get("params") or {})
        component = {"module": det.get("module"), "class": det.get("class"), "params": det_params}
        params = {**det_params, "operating_threshold": r.get("operating_threshold"), "iou": cfg.get("iou"),
                  "class_name": meta.get("class_name"), "config_conf_threshold": det.get("config_conf_threshold")}
        split = meta.get("split") or {}
        gt_d = {"dataset": "PRW", "split": split.get("split"), "limit": split.get("limit"), "every": split.get("every"),
                "frames": r.get("frames"), "boxes": r.get("gt_boxes"), "boxes_unlabeled_pid": gt.get("boxes_unlabeled_pid")}
        timing = {k: st.get(k) for k in ("load_sec", "latency_ms_mean", "latency_ms_p50", "fps", "elapsed_sec") if k in st}
        config = {"path": det.get("config_path"), "sha256": det.get("config_sha256")} if det.get("config_path") else {}
        entries.append(make_entry("detect", summary.get("producer") or "detect_eval_prw", name, component=component,
                                  params=params, gt=gt_d, metrics=metrics, timing=timing, versions=meta.get("versions"),
                                  inputs=_role_inputs(summary, f"detections:{name}"), config=config, report=report,
                                  command=command, note=meta.get("kind"), created_at=summary.get("generated_at"), env=env))
    return entries


def _guess_cluster_method(file_name: str) -> Optional[str]:
    low = file_name.lower()
    for key in ("leiden", "dbscan"):
        if key in low:
            return key
    return None


def cluster_component_from_report(assignments_path: Any) -> Dict[str, Any]:
    """assignments.jsonl 옆의 <…>_report.json (clustering 드라이버 산출물) 에서 component/params/timing 을 읽는다."""
    p = Path(assignments_path)
    out: Dict[str, Any] = {"component": {}, "params": {}, "timing": {}, "config": {}, "report": None}
    suffix = "_assignments.jsonl"
    if not p.name.endswith(suffix):
        return out
    rp = p.with_name(p.name[:-len(suffix)] + "_report.json")
    if not rp.is_file():
        return out
    try:
        d = json.loads(rp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    cfg = d.get("config") or {}
    stats = d.get("stats") or {}
    plugin = cfg.get("plugin")
    if isinstance(plugin, dict):
        out["component"] = {"module": plugin.get("module"), "class": plugin.get("class"), "params": plugin.get("params") or {}}
    else:
        out["component"] = {"method": cfg.get("method") or _guess_cluster_method(p.name), "vector": cfg.get("vector")}
    out["params"] = {k: cfg[k] for k in _CLUSTER_PARAM_KEYS if cfg.get(k) is not None}
    out["timing"] = {k: v for k, v in stats.items() if k.endswith("_sec") and isinstance(v, (int, float))}
    if cfg.get("config_path"):
        out["config"] = {"path": cfg.get("config_path"), "sha256": cfg.get("config_sha256")}
    out["report"] = str(rp)
    return out


def entries_from_cluster_summary(summary: Dict[str, Any], report: Any = None, command: Optional[Sequence[str]] = None,
                                 env: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """prw_cluster_gt_eval 의 사이드카(cluster_gt_report.json) → method 당 엔트리."""
    cfg = summary.get("config") or {}
    gt = summary.get("gt") or {}
    entries = []
    for name, r in (summary.get("results") or {}).items():
        pn = r.get("pairs_noise_as_singletons") or {}
        pc = r.get("pairs_clustered_only") or {}
        b3 = r.get("bcubed") or {}
        metrics = {
            "pair_precision": pn.get("precision"), "pair_recall": pn.get("recall"), "pair_f1": pn.get("f1"),
            "pair_precision_clustered": pc.get("precision"), "pair_recall_clustered": pc.get("recall"), "pair_f1_clustered": pc.get("f1"),
            "b3_precision": b3.get("precision"), "b3_recall": b3.get("recall"), "b3_f1": b3.get("f1"),
            "purity": r.get("purity"), "inverse_purity": r.get("inverse_purity"), "noise_ratio": r.get("noise_ratio"),
            "labeled_points": r.get("labeled_points"), "noise_points": r.get("noise_points"),
            "pure_clusters": r.get("pure_clusters"), "mixed_clusters": r.get("mixed_clusters"),
            "mixed_cluster_points": r.get("mixed_cluster_points"), "pids_split": r.get("pids_split"),
            "pids_all_noise": r.get("pids_all_noise"), "clusters_per_pid": r.get("clusters_per_pid"),
            "ari": r.get("ari_noise_as_singletons"), "nmi": r.get("nmi_noise_as_singletons"),
            "ari_clustered": r.get("ari_clustered_only"), "nmi_clustered": r.get("nmi_clustered_only"),
        }
        path = (cfg.get("methods") or {}).get(name)
        info = cluster_component_from_report(path) if path else cluster_component_from_report("")
        params = {**info["params"], "iou": cfg.get("iou"), "min_pid_size": cfg.get("min_pid_size"), "pid_split": cfg.get("pid_split")}
        gt_d = {"dataset": "PRW", "sources": cfg.get("sources"), "collection": cfg.get("collection"),
                "evaluated_points": gt.get("evaluated_points"), "pids": gt.get("pids"),
                "iou": cfg.get("iou"), "min_pid_size": cfg.get("min_pid_size"), "pid_split": cfg.get("pid_split")}
        inputs = _role_inputs(summary, f"assignments:{name}")
        if info["report"]:
            inputs.append({"role": "cluster_report", "path": info["report"]})
        entries.append(make_entry("cluster", summary.get("producer") or "prw_cluster_gt_eval", name,
                                  component=info["component"], params=params, gt=gt_d, metrics=metrics,
                                  timing=info["timing"], inputs=inputs, config=info["config"], report=report,
                                  command=command, seed=info["params"].get("seed"),
                                  created_at=summary.get("generated_at"), env=env))
    return entries


def entry_from_embedding_result(result: Dict[str, Any], model: str, *, component: Optional[Dict[str, Any]] = None,
                                params: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None,
                                versions: Optional[Dict[str, Any]] = None, report: Any = None,
                                command: Optional[Sequence[str]] = None, created_at: Any = None,
                                env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """prw_eval 의 결과 dict(mAP/Rank-k, %) → 엔트리 하나."""
    metrics = {"map": result.get("mAP"), "rank1": result.get("Rank-1"), "rank5": result.get("Rank-5"),
               "rank10": result.get("Rank-10"), "valid_queries": result.get("valid_queries")}
    gt = {"dataset": "PRW", "protocol": "query_box → test-frame GT crops", "gallery_size": result.get("gallery_size"),
          "query_total": result.get("query_total")}
    return make_entry("embed", "prw_eval", model, component=component or {"model": model}, params=params, gt=gt,
                      metrics=metrics, versions=versions, config=config, report=report, command=command,
                      created_at=created_at, env=env)


def entries_from_unified_result(out: Dict[str, Any], report: Any = None, command: Optional[Sequence[str]] = None,
                                env: Optional[Dict[str, Any]] = None,
                                versions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """prw_eval_unified 의 결과 dict(unified_eval.json) → variant 당 엔트리."""
    params_common = {"weights": out.get("weights"), "rrf_k": out.get("rrf_k"), "prefetch": out.get("prefetch"),
                     "pool": out.get("pool")}
    gt = {"dataset": "PRW", "protocol": out.get("protocol"), "gallery_size": out.get("gallery_size"),
          "query_total": out.get("query_total")}
    config = {"path": out.get("config"), "sha256": out.get("config_sha256")} if out.get("config") else {}
    entries = []
    for variant, r in (out.get("results") or {}).items():
        metrics = {"map": r.get("mAP"), "rank1": r.get("Rank-1"), "rank5": r.get("Rank-5"), "rank10": r.get("Rank-10"),
                   "pool_recall": r.get("pool_recall(%)"), "all_positives_in_pool": r.get("queries_all_positives_in_pool(%)"),
                   "any_positive_in_pool": r.get("queries_any_positive_in_pool(%)"), "valid_queries": r.get("valid_queries"),
                   "sec": r.get("sec")}
        entries.append(make_entry("search", "prw_eval_unified", variant, component={"variant": variant},
                                  params=params_common, gt=gt, metrics=metrics, versions=versions, config=config,
                                  report=report, command=command, note=r.get("note"),
                                  created_at=out.get("generated_at"), env=env))
    return entries


def entry_from_e2e_result(out: Dict[str, Any], report: Any = None, command: Optional[Sequence[str]] = None,
                          env: Optional[Dict[str, Any]] = None, versions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """prw_e2e_search_eval 의 결과 dict → 엔트리 하나 (stage=e2e, 전체 파이프라인)."""
    m = out.get("metrics") or {}
    cfg = out.get("config") or {}
    gt = out.get("gt") or {}
    component = {"variant": out.get("name"), "stage1": cfg.get("stage1"), "rerank": cfg.get("rerank"),
                 "pipeline": "detect→crop→embed→qdrant→search"}
    params = {k: cfg.get(k) for k in ("limit", "pool", "gallery", "scope", "pid_split", "max_queries") if cfg.get(k) is not None}
    gt_d = {"dataset": "PRW", "protocol": "e2e: query_box → 운영 DB(검출 crop), point→pid IoU≥0.5 매칭",
            "collection": gt.get("collection"), "queries": gt.get("queries"), "valid_queries": gt.get("valid_queries"),
            "gt_positives": gt.get("gt_positives"), "db_positives": gt.get("db_positives"), "db_points": gt.get("db_points")}
    config = {"path": cfg.get("config_path"), "sha256": cfg.get("config_sha256")} if cfg.get("config_path") else {}
    inputs = [{"role": "matches_cache", "path": gt.get("matches_cache"), "sha256": gt.get("matches_sha256")}] if gt.get("matches_cache") else []
    return make_entry("e2e", out.get("producer") or "prw_e2e_search_eval", str(out.get("name")), component=component,
                      params=params, gt=gt_d, metrics=m, timing=out.get("timing"), versions=versions, inputs=inputs,
                      config=config, report=report, command=command, created_at=out.get("generated_at"), env=env)



# ---------------------------------------------------------------- P6 정답(사람 라벨) 평가 3종
def _p6_entry(stage: str, out: Dict[str, Any], component: Dict[str, Any], params: Dict[str, Any], gt_extra: Dict[str, Any],
              report: Any, command: Optional[Sequence[str]], versions: Optional[Dict[str, Any]], env: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    m = out.get("metrics") or {}
    gt = dict(out.get("gt") or {})
    gt.update(gt_extra)
    timing = {k: m[k] for k in ("elapsed_sec", "sec_per_candidate", "wall_sec_per_candidate") if m.get(k) is not None}
    return make_entry(stage, out.get("producer") or stage, str(out.get("name")), component=component, params=params, gt=gt, metrics=m,
                      timing=timing, versions=versions, report=report, command=command, created_at=out.get("generated_at"), env=env,
                      note="pseudo GT" if out.get("pseudo_gt") else None)


def entry_from_track_result(out: Dict[str, Any], report: Any = None, command: Optional[Sequence[str]] = None,
                            env: Optional[Dict[str, Any]] = None, versions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """track_gt_eval 의 결과 dict → 엔트리 (stage=track: 추적기 before / 스티처 after, metrics 는 after + *_before)."""
    cfg = out.get("config") or {}
    component = {"pipeline": "detect→track→stitch", "tracking_config": cfg.get("tracking_config"), "pred_file": cfg.get("pred_file"),
                 "processed_root": cfg.get("processed_root")}
    params = {"videos": cfg.get("videos"), "iou": cfg.get("iou")}
    return _p6_entry("track", out, component, params, {"dataset": "semi-GT tracks"}, report, command, versions, env)


def entry_from_object_result(out: Dict[str, Any], report: Any = None, command: Optional[Sequence[str]] = None,
                             env: Optional[Dict[str, Any]] = None, versions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """object_pair_eval 의 결과 dict → 엔트리 (stage=object: 객체 트랙 벡터의 검색 mAP·쌍 AUC·클러스터 일치)."""
    cfg = out.get("config") or {}
    component = {"vector": cfg.get("vector"), "collection": cfg.get("collection")}
    params = {"threshold": cfg.get("threshold"), "assignments": cfg.get("assignments")}
    return _p6_entry("object", out, component, params, {"dataset": "object pairs"}, report, command, versions, env)


def entry_from_qwen_result(out: Dict[str, Any], report: Any = None, command: Optional[Sequence[str]] = None,
                           env: Optional[Dict[str, Any]] = None, versions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """qwen_verify_eval 의 결과 dict → 엔트리 (stage=qwen: P@K 전/후, 오탈락률, UNKNOWN, 후보당 초)."""
    cfg = out.get("config") or {}
    component = {"model_id": cfg.get("model_id"), "reranker": cfg.get("reranker_used"), "verify_mode": cfg.get("verify_mode")}
    params = {k: cfg.get(k) for k in ("top_k", "alpha", "threshold", "no_reranker", "rescore")}
    return _p6_entry("qwen", out, component, params, {"dataset": "qwen judgements"}, report, command, versions, env)


# ---------------------------------------------------------------- 이관
def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return d if isinstance(d, dict) else None


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()


def collect_results(results_dir: Any, log: Callable[[str], Any] = print) -> List[Dict[str, Any]]:
    """eval/results 아래 기존 산출물을 엔트리로 변환 (원장에 쓰지는 않음)."""
    root = Path(results_dir)
    entries: List[Dict[str, Any]] = []

    def imported(p: Path) -> Dict[str, Any]:
        return {"imported_from": str(p), "imported_at": datetime.now().astimezone().isoformat()}

    for p in sorted(root.rglob("detect_eval_report.json")):
        d = _load_json(p)
        if d and d.get("results"):
            got = entries_from_detect_summary(d, report=p, env=imported(p))
            log(f"[import] detect  {p} → {len(got)}")
            entries += got
    for p in sorted(root.rglob("cluster_gt_report.json")):
        d = _load_json(p)
        if d and d.get("results"):
            got = entries_from_cluster_summary(d, report=p, env=imported(p))
            log(f"[import] cluster {p} → {len(got)}")
            entries += got
    for p in sorted(root.glob("embedding_*.json")):
        d = _load_json(p)
        if d and "mAP" in d:
            model = str(d.get("model") or p.stem.replace("embedding_", ""))
            entries.append(entry_from_embedding_result(d, model, report=p, created_at=_mtime_iso(p), env=imported(p)))
            log(f"[import] embed   {p} → 1")
    for p in sorted(root.rglob("unified_eval.json")):
        d = _load_json(p)
        if d and d.get("results"):
            got = entries_from_unified_result(d, report=p, env=imported(p))
            log(f"[import] search  {p} → {len(got)}")
            entries += got
    for p in sorted(root.rglob("e2e_search.json")):
        d = _load_json(p)
        if d and d.get("metrics") and d.get("name"):
            entries.append(entry_from_e2e_result(d, report=p, env=imported(p)))
            log(f"[import] e2e     {p} → 1")
    for fname, builder, tag in (("track_eval.json", entry_from_track_result, "track"), ("object_pair_eval.json", entry_from_object_result, "object"),
                                ("qwen_verify_eval.json", entry_from_qwen_result, "qwen")):
        for p in sorted(root.rglob(fname)):
            d = _load_json(p)
            if d and d.get("metrics") and d.get("name") and not d.get("pseudo_gt") and not d.get("unlabeled_only"):
                entries.append(builder(d, report=p, env=imported(p)))
                log(f"[import] {tag:<7} {p} → 1")
    return entries


def import_results(results_dir: Any, ledger: Any = None, log: Callable[[str], Any] = print) -> Tuple[int, int, Path]:
    path = resolve_ledger_path(ledger) or DEFAULT_LEDGER
    entries = collect_results(results_dir, log)
    n, skipped = append_entries(path, entries, dedupe=True)
    log(f"[import] 기록 {n} · 중복 건너뜀 {skipped} → {path}")
    return n, skipped, path


# ---------------------------------------------------------------- 조회 / 표
def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
    return str(value)


def filter_entries(entries: Iterable[Dict[str, Any]], stage: Optional[str] = None, name: Optional[str] = None,
                   names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    out = []
    for e in entries:
        if stage and e.get("stage") != stage:
            continue
        if name and name.lower() not in str(e.get("name", "")).lower():
            continue
        if names and e.get("name") not in names:
            continue
        out.append(e)
    return out


def render_table(entries: Sequence[Dict[str, Any]], stage: str, metrics: Optional[Sequence[str]] = None,
                 fmt_name: str = "md") -> str:
    keys = list(metrics) if metrics else METRIC_KEYS.get(stage, [])
    header = ["name", "created", *keys, "run_id"]
    rows = [[e.get("name"), str(e.get("created_at", ""))[:16], *[fmt((e.get("metrics") or {}).get(k)) for k in keys],
             e.get("run_id")] for e in entries]
    if fmt_name == "csv":
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(header)
        for row in rows:
            w.writerow(row)
        return buf.getvalue()
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def list_lines(entries: Sequence[Dict[str, Any]]) -> List[str]:
    lines = []
    for e in entries:
        keys = METRIC_KEYS.get(e.get("stage", ""), [])[:4]
        m = e.get("metrics") or {}
        vals = " ".join(f"{k}={fmt(m.get(k))}" for k in keys)
        lines.append(f"{e.get('run_id')}  {str(e.get('created_at', ''))[:19]}  {e.get('stage', ''):<7} {str(e.get('name', '')):<26} {vals}")
    return lines


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="평가 실행 원장 (bench/ledger.jsonl)")
    p.add_argument("--ledger", default=None, help=f"원장 경로 (기본 {DEFAULT_LEDGER}); 하위 명령 앞에 둔다")
    # 공통 옵션은 모든 하위 명령이 받는다 (GUI 가 명령과 무관하게 같은 인자를 넘길 수 있게). 해당 없는 옵션은 무시.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--stage", choices=STAGES, default=None, help="table 은 필수, list 는 필터")
    common.add_argument("--name", default=None, help="list: 이름 부분 문자열")
    common.add_argument("--names", default=None, help="쉼표 구분 이름 목록")
    common.add_argument("--last", type=int, default=0, help="list: 마지막 N 건만")
    common.add_argument("--latest", action="store_true", help="table: 이름별 마지막 엔트리만")
    common.add_argument("--format", choices=["md", "csv"], default="md", help="table 출력 형식")
    common.add_argument("--metrics", default=None, help="table: 쉼표 구분 지표 열 (기본 METRIC_KEYS)")
    common.add_argument("--results-dir", default=str(PROJECT_ROOT / "eval" / "results"), help="import: 이관할 폴더")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("import", parents=[common], help="eval/results 산출물 이관 (멱등)")
    sub.add_parser("list", parents=[common], help="엔트리 목록")
    s = sub.add_parser("show", parents=[common], help="엔트리 하나 (json)")
    s.add_argument("run_id")
    sub.add_parser("table", parents=[common], help="단계별 지표 표 (--stage 필수)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    path = resolve_ledger_path(args.ledger) or DEFAULT_LEDGER
    if args.cmd == "import":
        import_results(args.results_dir, path)
        return 0
    errors: List[str] = []
    entries = read_entries(path, errors)
    for err in errors:
        print(f"[ledger] 손상된 줄 건너뜀: {err}", file=sys.stderr)
    if args.cmd == "list":
        rows = filter_entries(entries, stage=args.stage, name=args.name)
        if args.last:
            rows = rows[-args.last:]
        print("\n".join(list_lines(rows)) if rows else f"(엔트리 없음: {path})")
        return 0
    if args.cmd == "show":
        for e in entries:
            if e.get("run_id") == args.run_id:
                print(_dumps(e, indent=2))
                return 0
        print(f"run_id 없음: {args.run_id}", file=sys.stderr)
        return 1
    if args.cmd == "table":
        if not args.stage:
            print("table 은 --stage 가 필요합니다 (detect | embed | search | cluster)", file=sys.stderr)
            return 2
        names = [n.strip() for n in args.names.split(",") if n.strip()] if args.names else None
        rows = filter_entries(entries, stage=args.stage, names=names)
        if args.latest:
            rows = list(latest_by_name(rows).values())
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()] if args.metrics else None
        sys.stdout.write(render_table(rows, args.stage, metrics, args.format))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
