"""bench/run.py — 단일 러너: 한 명령으로 한 단계의 평가를 고정 GT 로 돌리고 원장(bench/ledger.jsonl)에 기록한다.
verify 로 이전 실행을 같은 설정으로 다시 돌려 허용 오차 안에서 재현되는지 검증한다.

  python bench/run.py detect  --config pipeline_tracking_yolo26.yaml [--name N] [--split test] [--limit 0] [--every 1]
                              [--module M --class C] [--param k=v]* [--conf-threshold 0.05] [--operating-threshold T]
                              [--compare 이름=detections.jsonl]* [--reuse-detections detections.jsonl] [--no-images]
  python bench/run.py embed   --model solider [--config pipeline.yaml] [--batch-size 32]
  python bench/run.py search  [--config pipeline.yaml] [--weights siglip2=1,irra=1.5,solider=1.5] [--rrf-k 2]
                              [--prefetch 200] [--pool 200] [--pools 50,100,500,1000]
  python bench/run.py cluster --method leiden | --method-config y.yaml [--module M --class C] [--param knn=30]*
                              [--target person] [--sources prw_image] [--vector solider] [--max-points 0]
                              [--min-cluster-size 2] [--name N] [--pid-split f:part]
  python bench/run.py e2e     [--stage1 siglip2 irra] [--rerank solider|none] [--limit 200] [--pool 200]
                              [--max-queries 0] [--gallery test] [--pid-split f:part] [--name N]
  python bench/run.py verify  RUN_ID [--tol 지표=오차]*          이전 실행 재현 (러너/스크립트 실행 모두 가능)
  python bench/run.py show-cmd RUN_ID                             그 실행을 재현하는 러너 명령만 출력
공통: --ledger PATH | --no-ledger, --runs-dir bench/runs (실행별 산출물 폴더), --dry-run (명령만 출력)

실행 폴더 bench/runs/<stage>_<시각>_<이름>/ 에 산출물·log.txt·ledger_part.jsonl(스크립트가 쓴 원장 조각)이 남고,
러너는 조각에 bench(인자·명령·폴더)·hardware·weights(가중치 sha256)·env 를 덧붙여 본 원장에 기록한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench import ledger  # noqa: E402

PY = sys.executable
DEFAULT_RUNS_DIR = PROJECT_ROOT / "bench" / "runs"
DEFAULT_PIPELINE = PROJECT_ROOT / "pipeline.yaml"
DEFAULT_TRACKING = PROJECT_ROOT / "pipeline_tracking.yaml"
WEIGHT_KEYS = ("weights", "ckpt", "checkpoint", "model_path", "irra_ckpt", "solider_ckpt", "clip_pt", "weight_path")
WEIGHT_SUFFIXES = (".pt", ".pth", ".safetensors", ".ckpt", ".bin", ".onnx")
WEIGHTS_SHA_CACHE = PROJECT_ROOT / "bench" / "cache" / "weights_sha.json"


# ---------------------------------------------------------------- 유틸
def encode_param(key: str, value: Any) -> str:
    """--param key=value 로 되돌릴 수 있게 인코딩 (수신측은 json.loads 시도 후 문자열)."""
    if isinstance(value, str):
        try:
            json.loads(value)
            return f"{key}={json.dumps(value, ensure_ascii=False)}"    # "0.5" 같은 문자열은 따옴표로 보호
        except ValueError:
            return f"{key}={value}"
    return f"{key}={json.dumps(value, ensure_ascii=False)}"


def slug(text: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "-_.+") else "_" for c in str(text))
    return keep.strip("_") or "run"


def weights_sha256(path: Path) -> Optional[str]:
    """큰 가중치 파일의 sha256 (경로·크기·mtime 으로 캐시)."""
    try:
        st = path.stat()
    except OSError:
        return None
    key = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    cache: Dict[str, str] = {}
    if WEIGHTS_SHA_CACHE.is_file():
        try:
            cache = json.loads(WEIGHTS_SHA_CACHE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache = {}
    if key in cache:
        return cache[key]
    digest = ledger.file_sha256(path)
    if digest:
        cache[key] = digest
        WEIGHTS_SHA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        WEIGHTS_SHA_CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8", newline="\n")
    return digest


def resolve_weight_path(value: str) -> Optional[Path]:
    p = Path(value).expanduser()
    candidates = [p, PROJECT_ROOT / p, PROJECT_ROOT / "weights" / p, PROJECT_ROOT / "weights" / "yolo" / p]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def weights_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """component.params 에서 가중치 파일을 찾아 sha256 기록 (재현성: 같은 이름의 다른 파일을 잡아낸다)."""
    out: Dict[str, Any] = {}
    for key, value in (params or {}).items():
        if not isinstance(value, str):
            continue
        if key in WEIGHT_KEYS or value.lower().endswith(WEIGHT_SUFFIXES):
            p = resolve_weight_path(value)
            out[key] = {"path": str(p) if p else value, "sha256": weights_sha256(p) if p else None,
                        "size": p.stat().st_size if p else None}
    return out


def retriever_weights(config_path: str, name: str) -> Dict[str, Any]:
    """pipeline.yaml retriever 의 선언 지문(ingest.build_db.declared_retriever_fingerprint) — 파일 params 는 내용 sha256."""
    try:
        from config import PipelineConfig
        from ingest.build_db import declared_retriever_fingerprint
        fp = declared_retriever_fingerprint(PipelineConfig.load(config_path))
        return fp.get(name) or {}
    except Exception as exc:  # noqa: BLE001 — 지문 실패가 벤치를 막지 않게
        return {"error": f"{type(exc).__name__}: {exc}"}


def sh(cmd: Sequence[str], log_path: Optional[Path], dry: bool = False, echo: Callable[[str], Any] = print) -> int:
    """하위 스크립트를 프로젝트 루트에서 실행하고 출력을 화면과 log.txt 에 같이 남긴다."""
    shown = " ".join(_quote(c) for c in cmd)
    echo(f"$ {shown}")
    if dry:
        return 0
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    log_f = log_path.open("a", encoding="utf-8") if log_path else None
    try:
        if log_f:
            log_f.write(f"$ {shown}\n")
        proc = subprocess.Popen(list(cmd), cwd=str(PROJECT_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            echo(line.rstrip("\n"))
            if log_f:
                log_f.write(line)
        code = proc.wait()
        if log_f:
            log_f.write(f"[exit {code}]\n")
    finally:
        if log_f:
            log_f.close()
    if code != 0:
        raise RuntimeError(f"하위 명령 실패 (exit {code}): {shown}")
    return code


def _quote(s: str) -> str:
    return f'"{s}"' if (" " in s or not s) else s


# ---------------------------------------------------------------- 단계별 명령
class Stage:
    """한 단계 실행 계획: 명령 목록을 만들고(런타임에 2단계 명령이 결정될 수도 있음) 결과 엔트리를 돌려준다."""

    def __init__(self, stage: str, a: Dict[str, Any], run_dir: Path, dry: bool, echo: Callable[[str], Any] = print):
        self.stage, self.a, self.run_dir, self.dry, self.echo = stage, a, run_dir, dry, echo
        self.part_ledger = run_dir / "ledger_part.jsonl"
        self.log_path = run_dir / "log.txt"
        self.commands: List[List[str]] = []

    def run_cmd(self, cmd: List[str]) -> None:
        self.commands.append(list(cmd))
        sh(cmd, None if self.dry else self.log_path, self.dry, self.echo)

    # ---- detect
    def detect(self) -> None:
        a = self.a
        name = a.get("name") or "detector"
        det_path = a.get("reuse_detections")
        if det_path:
            det_path = str(Path(det_path).resolve())
        else:
            cmd = [PY, "eval/detect_eval_prw.py", "--mode", "run", "--name", name, "--split", a.get("split") or "test",
                   "--limit", str(a.get("limit") or 0), "--every", str(a.get("every") or 1),
                   "--output-dir", str(self.run_dir), "--no-score", "--no-ledger"]
            if a.get("config"):
                cmd += ["--detector-config", str(a["config"])]
            else:
                cmd += ["--detector-config", ""]
            if a.get("module"):
                cmd += ["--module", a["module"]]
            if a.get("cls"):
                cmd += ["--class", a["cls"]]
            for k, v in (a.get("params") or {}).items():
                cmd += ["--param", encode_param(k, v)]
            if a.get("conf_threshold") is not None:
                cmd += ["--conf-threshold", str(a["conf_threshold"])]
            if a.get("data_root"):
                cmd += ["--data-root", a["data_root"]]
            self.run_cmd(cmd)
            det_path = str(self.run_dir / name / "detections.jsonl")
        cmd = [PY, "eval/detect_eval_prw.py", "--mode", "score", "--method", f"{name}={det_path}", *[str(c) for c in (a.get("compare") or [])],
               "--output-dir", str(self.run_dir), "--ledger", str(self.part_ledger)]
        if a.get("operating_threshold") is not None:
            cmd += ["--operating-threshold", str(a["operating_threshold"])]
        if a.get("no_images"):
            cmd += ["--no-images"]
        if a.get("data_root"):
            cmd += ["--data-root", a["data_root"]]
        self.run_cmd(cmd)

    # ---- embed
    def embed(self) -> None:
        a = self.a
        cmd = [PY, "eval/prw_eval.py", "--model", a["model"], "--config", str(a.get("config") or DEFAULT_PIPELINE),
               "--save-result", str(self.run_dir / f"embedding_{a['model']}.json"), "--ledger", str(self.part_ledger)]
        if a.get("batch_size"):
            cmd += ["--batch-size", str(a["batch_size"])]
        if a.get("data_root"):
            cmd += ["--data-root", a["data_root"]]
        for k in ("backbone", "semantic_weight", "neck_feat"):
            if a.get(k) is not None:
                cmd += [f"--{k.replace('_', '-')}", str(a[k])]
        self.run_cmd(cmd)

    # ---- search (통합 검색 조합, 단독 평가)
    def search(self) -> None:
        a = self.a
        cmd = [PY, "eval/prw_eval_unified.py", "--config", str(a.get("config") or DEFAULT_PIPELINE),
               "--out-json", str(self.run_dir / "unified_eval.json"), "--out-csv", str(self.run_dir / "unified_eval.csv"),
               "--ledger", str(self.part_ledger)]
        if a.get("weights"):
            w = a["weights"]
            cmd += ["--weights", ",".join(f"{k}={v}" for k, v in w.items()) if isinstance(w, dict) else str(w)]
        for k in ("rrf_k", "prefetch", "pool", "pools", "models", "batch_size"):
            if a.get(k) is not None:
                cmd += [f"--{k.replace('_', '-')}", str(a[k])]
        if a.get("data_root"):
            cmd += ["--data-root", a["data_root"]]
        self.run_cmd(cmd)

    # ---- cluster (플러그인 실행 → GT 평가)
    def cluster(self) -> None:
        a = self.a
        target = a.get("target") or "person"
        cdir = self.run_dir / "cluster"
        cmd = [PY, "clustering/cluster_qdrant.py", "--config", str(a.get("config") or DEFAULT_PIPELINE), "--target", target,
               "--output-dir", str(cdir), "--vector-cache", str(a.get("vector_cache") or "outputs/clustering/cache/bench"),
               "--min-cluster-size", str(a.get("min_cluster_size") or 2)]
        if a.get("sources"):
            cmd += ["--sources", a["sources"]]
        if a.get("vector"):
            cmd += ["--vector", a["vector"]]
        if a.get("method_config"):
            cmd += ["--method-config", str(a["method_config"])]
        elif a.get("method"):
            cmd += ["--method", a["method"]]
        if a.get("module"):
            cmd += ["--module", a["module"]]
        if a.get("cls"):
            cmd += ["--class", a["cls"]]
        for k, v in (a.get("params") or {}).items():
            cmd += ["--param", encode_param(k, v)]
        if a.get("max_points"):
            cmd += ["--max-points", str(a["max_points"])]
        self.run_cmd(cmd)
        if self.dry:
            assignments = cdir / target / f"{target}_<method>_assignments.jsonl"
        else:
            found = sorted((cdir / target).glob(f"{target}_*_assignments.jsonl"))
            if not found:
                raise RuntimeError(f"assignments 가 없습니다: {cdir / target}")
            assignments = found[-1]
        label = a.get("name") or a.get("method") or assignments.name.split("_")[1]
        cmd = [PY, "eval/prw_cluster_gt_eval.py", "--method", f"{label}={assignments}", "--config", str(a.get("config") or DEFAULT_PIPELINE),
               "--output-dir", str(self.run_dir / "gt"), "--no-images", "--ledger", str(self.part_ledger)]
        if a.get("sources"):
            cmd += ["--sources", a["sources"]]
        if a.get("pid_split"):
            cmd += ["--pid-split", a["pid_split"]]
        if a.get("data_root"):
            cmd += ["--data-root", a["data_root"]]
        self.run_cmd(cmd)

    # ---- e2e (전체 파이프라인 검색)
    def e2e(self) -> None:
        a = self.a
        cmd = [PY, "eval/prw_e2e_search_eval.py", "--config", str(a.get("config") or DEFAULT_PIPELINE),
               "--output-dir", str(self.run_dir), "--ledger", str(self.part_ledger)]
        if a.get("stage1"):
            cmd += ["--stage1", *a["stage1"]]
        if a.get("rerank") is not None:
            cmd += ["--rerank", str(a["rerank"])]
        for k in ("limit", "pool", "max_queries", "gallery", "pid_split", "name"):
            if a.get(k) not in (None, "", 0):
                cmd += [f"--{k.replace('_', '-')}", str(a[k])]
        if a.get("data_root"):
            cmd += ["--data-root", a["data_root"]]
        self.run_cmd(cmd)

    # ---- P6: 사람 정답(준정답) 평가 3종 — 평가 스크립트가 원장 조각을 직접 쓴다
    def track(self) -> None:
        a = self.a
        name = a.get("name") or default_name("track", a)
        processed = a.get("processed_root")
        if a.get("tracking_config"):
            proc_root = self.run_dir / "processed"
            videos = list(a.get("videos") or [])
            if not videos:
                gt_dir = Path(a.get("gt_dir") or PROJECT_ROOT / "eval" / "gt" / "tracks")
                videos = sorted(p.name for p in gt_dir.iterdir() if (p / "boxes.jsonl").is_file()) if gt_dir.is_dir() else []
            if a.get("restitch"):
                # 스티처만 교체 비교: 기존 처리 결과의 tracks.jsonl + sushi_input(검출 pickle) 위에서 SUSHI(+yaml 의 link_windows 옵션)만 다시 돌린다.
                # 검출·추적을 다시 하면 박스가 달라져 semi-GT(고정 박스)와 맞지 않는다 — 실측: 재추적 시 FP 6,850 (2026-09-28).
                from video.batch_preprocess_videos_parallel import stitcher_link_args
                src_root = Path(processed or PROJECT_ROOT / "outputs" / "processed_videos")
                for v in videos:
                    src = src_root / v
                    out_dir = proc_root / v
                    sushi_in = src / "sushi_input" / v
                    if not (sushi_in / "processed_data").is_dir():
                        # 옛 처리 결과에 SUSHI 입력(검출 pickle)이 없으면 어댑터로 다시 만든다 (영상 파일 필요)
                        try:
                            video_file = _video_file(src, v, a.get("videos_root"))
                        except SystemExit:
                            if not self.dry:
                                raise
                            video_file = Path(a.get("videos_root") or PROJECT_ROOT / "data" / "videos") / f"{v}.mp4"   # dry-run 은 경로만 보여준다
                        sushi_in = out_dir / "sushi_input" / v
                        self.run_cmd([PY, "video/sushi_adapter.py", "--video", str(video_file), "--tracks", str(src / "tracks.jsonl"),
                                      "--sushi-root", str(a.get("sushi_root") or "./third_party/SUSHI"), "--output-root", str(out_dir / "sushi_input")])
                    cmd = [PY, "video/sushi_inference.py", "--input-root", str(sushi_in), "--sushi-root", str(a.get("sushi_root") or "./third_party/SUSHI"),
                           "--checkpoint", str(a.get("checkpoint") or "./third_party/SUSHI/pretrained_models/mot17private.pth"),
                           "--tracks", str(src / "tracks.jsonl"), "--output", str(out_dir / "stitched_tracks.json")]
                    cmd += stitcher_link_args(a["tracking_config"])
                    self.run_cmd(cmd)
                processed = str(proc_root)
                videos_for_preprocess = []
            else:
                videos_for_preprocess = videos
            for v in videos_for_preprocess:
                # 같은 GT 영상을 이 yaml 로 다시 검출·추적·스티칭 (검출기·추적기 교체 비교 — 박스가 달라지므로 semi-GT 와는 부분적으로만 맞음)
                cmd = [PY, "video/batch_preprocess_videos_parallel.py", "--processed-root", str(proc_root), "--work-root", str(self.run_dir / "work"),
                       "--tracking-config", str(a["tracking_config"]), "--pattern", f"{v}.", "--workers", "1"]   # "stem." 로 clip1 ≠ clip10
                if a.get("videos_root"):
                    cmd += ["--videos-root", str(a["videos_root"])]
                self.run_cmd(cmd)
            if videos_for_preprocess:
                processed = str(proc_root)
        cmd = [PY, "eval/track_gt_eval.py", "eval", "--output-dir", str(self.run_dir), "--ledger", str(self.part_ledger), "--name", name, "--record-pseudo"]
        if a.get("gt_dir"):
            cmd += ["--gt-dir", str(a["gt_dir"])]
        if a.get("videos"):
            cmd += ["--videos", *[str(v) for v in a["videos"]]]
        if processed:
            cmd += ["--processed-root", str(processed)]
        if a.get("pred_file"):
            cmd += ["--pred-file", str(a["pred_file"])]
        if a.get("tracking_config"):
            cmd += ["--tracking-config", str(a["tracking_config"])]
        for k in ("iou", "max_gap", "min_coverage"):
            if a.get(k) is not None:
                cmd += [f"--{k.replace('_', '-')}", str(a[k])]
        self.run_cmd(cmd)

    def object(self) -> None:
        a = self.a
        name = a.get("name") or default_name("object", a)
        cmd = [PY, "eval/object_pair_eval.py", "eval", "--output-dir", str(self.run_dir), "--ledger", str(self.part_ledger), "--name", name,
               "--vector", str(a.get("vector") or "dinov2"), "--record-pseudo"]
        for k in ("gt_dir", "collection", "threshold", "assignments", "min_coverage"):
            if not _empty(a.get(k)):
                cmd += [f"--{k.replace('_', '-')}", str(a[k])]
        self.run_cmd(cmd)

    def qwen(self) -> None:
        a = self.a
        name = a.get("name") or default_name("qwen", a)
        cmd = [PY, "eval/qwen_verify_eval.py", "eval", "--output-dir", str(self.run_dir), "--ledger", str(self.part_ledger), "--name", name, "--record-pseudo"]
        if a.get("allow_unlabeled"):
            cmd.append("--allow-unlabeled")
        for k in ("gt_dir", "top_k", "alpha", "threshold", "verify_mode", "model_id", "max_queries", "qwen_dir", "reranker_model_id", "dtype", "max_pixels", "min_coverage", "batch_size"):
            if not _empty(a.get(k)):
                cmd += [f"--{k.replace('_', '-')}", str(a[k])]
        if a.get("no_reranker"):
            cmd.append("--no-reranker")
        if a.get("rescore"):
            cmd.append("--rescore")
        self.run_cmd(cmd)

    def execute(self) -> List[Dict[str, Any]]:
        if not self.dry:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        getattr(self, self.stage)()
        if self.dry:
            return []
        entries = ledger.read_entries(self.part_ledger)
        if not entries:
            raise RuntimeError(f"원장 조각이 비었습니다: {self.part_ledger} (하위 스크립트 로그 {self.log_path})")
        hardware = ledger.hardware_info()
        env = ledger.env_info()
        for e in entries:
            comp = e.get("component") or {}
            if self.stage in ("embed",):
                weights = retriever_weights(str(self.a.get("config") or DEFAULT_PIPELINE), str(comp.get("model")))
            elif self.stage in ("search", "e2e"):
                weights = {}
                cfg_path = str(self.a.get("config") or DEFAULT_PIPELINE)
                names = list(comp.get("stage1") or []) + ([comp["rerank"]] if comp.get("rerank") else [])
                for n in dict.fromkeys(names):
                    weights[n] = retriever_weights(cfg_path, n)
            else:
                weights = weights_from_params(comp.get("params") or {})
            e["bench"] = {"stage": self.stage, "args": self.a, "run_dir": str(self.run_dir), "python": PY,
                          "commands": self.commands, "elapsed_sec": round(time.time() - started, 2)}
            e["hardware"] = hardware
            e["weights"] = weights
            e["env"] = env
        return entries


# ---------------------------------------------------------------- 엔트리 → 러너 인자 (verify / show-cmd)
def args_from_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """원장 엔트리에서 같은 실행을 재현할 러너 인자를 만든다. 러너로 돌린 것은 bench.args 그대로."""
    bench = entry.get("bench") or {}
    if bench.get("args"):
        return dict(bench["args"])
    stage = entry.get("stage")
    comp = entry.get("component") or {}
    params = dict(entry.get("params") or {})
    gt = entry.get("gt") or {}
    cfg = entry.get("config") or {}
    if stage == "detect":
        if not comp.get("module") or comp.get("module") == "qdrant":
            raise ValueError("운영 DB(import-qdrant) 엔트리는 재현 대상이 아닙니다")
        det_params = dict(comp.get("params") or {})
        return {"name": entry["name"], "module": comp["module"], "cls": comp["class"], "params": det_params,
                "conf_threshold": det_params.get("conf_threshold"), "split": gt.get("split") or "test",
                "limit": gt.get("limit") or 0, "every": gt.get("every") or 1,
                "operating_threshold": params.get("operating_threshold"), "no_images": True}
    if stage == "cluster":
        if not comp.get("module"):
            raise ValueError("플러그인(module/class) 정보가 없는 클러스터링 엔트리는 재현할 수 없습니다 (clustering/cluster_qdrant.py 로 다시 실행)")
        # 플러그인 생성자 인자는 component.params 그대로 (보고서 params 의 score_threshold/target/pid_split 등은 생성자 인자가 아님)
        plugin_params = dict(comp.get("params") or {})
        target = str(params.get("target") or "person")
        out = {"name": entry["name"], "module": comp["module"], "cls": comp["class"], "params": plugin_params,
               "target": target, "sources": ",".join(gt.get("sources") or ["prw_image"]), "vector": params.get("vector"),
               "min_cluster_size": params.get("min_cluster_size") or 2, "max_points": params.get("max_points") or 0,
               "config": cfg.get("path") or str(DEFAULT_PIPELINE)}
        if params.get("pid_split") or gt.get("pid_split"):
            out["pid_split"] = params.get("pid_split") or gt.get("pid_split")
        return out
    if stage == "embed":
        out = {"model": comp.get("model"), "config": cfg.get("path") or str(DEFAULT_PIPELINE)}
        for k in ("batch_size", "backbone", "semantic_weight", "neck_feat"):
            if params.get(k) is not None:
                out[k] = params[k]
        return out
    if stage == "search":
        return {"config": cfg.get("path") or str(DEFAULT_PIPELINE), "weights": params.get("weights"), "rrf_k": params.get("rrf_k"),
                "prefetch": params.get("prefetch"), "pool": params.get("pool"), "variant": entry["name"]}
    if stage == "e2e":
        return {"config": cfg.get("path") or str(DEFAULT_PIPELINE), "stage1": comp.get("stage1"), "rerank": comp.get("rerank") or "none",
                "limit": params.get("limit"), "pool": params.get("pool"), "max_queries": params.get("max_queries") or 0,
                "gallery": params.get("gallery"), "pid_split": params.get("pid_split"), "name": entry["name"]}
    if stage in ("track", "object", "qwen"):
        out = {"name": entry["name"]}
        if stage == "track":
            out.update({"tracking_config": comp.get("tracking_config"), "processed_root": comp.get("processed_root"),
                        "pred_file": comp.get("pred_file"), "videos": params.get("videos"), "gt_dir": params.get("gt_dir"),
                        "restitch": bool(params.get("restitch")), "iou": params.get("iou"), "min_coverage": params.get("min_coverage"),
                        "max_gap": params.get("max_gap") if isinstance(params.get("max_gap"), int) else None})
        elif stage == "object":
            out.update({"vector": comp.get("vector"), "collection": comp.get("collection"), "threshold": params.get("threshold"),
                        "assignments": params.get("assignments"), "gt_dir": params.get("gt_dir"), "min_coverage": params.get("min_coverage")})
        else:
            out.update({"model_id": comp.get("model_id"), "verify_mode": comp.get("verify_mode"), "top_k": params.get("top_k"),
                        "alpha": params.get("alpha"), "threshold": params.get("threshold"), "no_reranker": bool(params.get("no_reranker")),
                        "gt_dir": params.get("gt_dir"), "reranker_model_id": comp.get("reranker_model_id"), "dtype": comp.get("dtype"),
                        "max_pixels": comp.get("max_pixels"), "rescore": bool(params.get("rescore")),
                        "batch_size": comp.get("batch_size") if (comp.get("batch_size") or 1) > 1 else None})
        return {k: v for k, v in out.items() if not _empty(v)}
    raise ValueError(f"알 수 없는 stage: {stage}")


def runner_command(stage: str, a: Dict[str, Any]) -> List[str]:
    """러너 인자 dict → `python bench/run.py …` 토큰 (사람이 복사해 쓰는 용도)."""
    cmd = [PY, "bench/run.py", stage]
    for k, v in a.items():
        if _empty(v) or k == "variant":
            continue
        flag = "--class" if k == "cls" else f"--{k.replace('_', '-')}"
        if k == "params":
            for pk, pv in v.items():
                cmd += ["--param", encode_param(pk, pv)]
        elif k == "weights" and isinstance(v, dict):
            cmd += [flag, ",".join(f"{a_}={b}" for a_, b in v.items())]
        elif isinstance(v, bool):
            cmd += [flag]
        elif isinstance(v, list):
            cmd += [flag, *[str(x) for x in v]]
        else:
            cmd += [flag, str(v)]
    return cmd


# ---------------------------------------------------------------- verify
def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def compare_metrics(old: Dict[str, Any], new: Dict[str, Any], tol: Dict[str, float]) -> Tuple[str, List[Dict[str, Any]]]:
    """(status, rows). status: PASS = 비교한 모든 지표가 허용 오차 안 · FAIL = 하나라도 벗어나거나 원본에 있던 지표가 재실행에 없음 ·
    UNVERIFIED = 비교할 수 있는 지표가 하나도 없음 (통과로 기록하지 않는다)."""
    rows = []
    status = "PASS"
    for key, t in tol.items():
        a, b = old.get(key), new.get(key)
        if not _num(a):
            continue                      # 원본에 없던 지표는 비교 대상 아님
        if not _num(b):
            rows.append({"metric": key, "before": a, "after": None, "delta": None, "tol": t, "ok": False, "reason": "재실행 결과에 없음"})
            status = "FAIL"
            continue
        delta = float(b) - float(a)
        ok = abs(delta) <= t
        if not ok:
            status = "FAIL"
        rows.append({"metric": key, "before": a, "after": b, "delta": round(delta, 6), "tol": t, "ok": ok})
    if not rows:
        status = "UNVERIFIED"
    return status, rows


def find_entry(entries: Sequence[Dict[str, Any]], run_id: str) -> Dict[str, Any]:
    exact = [e for e in entries if e.get("run_id") == run_id]
    if exact:
        return exact[-1]
    prefix = [e for e in entries if str(e.get("run_id", "")).startswith(run_id)]
    if len(prefix) == 1:
        return prefix[0]
    if len(prefix) > 1:
        raise SystemExit(f"run_id 접두어가 여러 개와 맞습니다: {[e['run_id'] for e in prefix][:6]}")
    raise SystemExit(f"run_id 없음: {run_id}")


def pick_result(entries: Sequence[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for e in entries:
        if e.get("name") == name:
            return e
    return entries[0]


# ---------------------------------------------------------------- CLI
def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ledger", default=None, help=f"원장 (기본 {ledger.DEFAULT_LEDGER})")
    p.add_argument("--no-ledger", action="store_true")
    p.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    p.add_argument("--dry-run", action="store_true", help="하위 명령만 출력")
    p.add_argument("--data-root", default=None)


def add_stage_options(p: argparse.ArgumentParser) -> None:
    """모든 단계가 같은 옵션 집합을 받는다 (GUI 가 단계와 무관하게 같은 인자를 넘길 수 있게). 해당 없는 옵션은 무시."""
    g = p.add_argument_group("공통")
    g.add_argument("--config", default=None, help="detect: detector 블록 yaml (기본 pipeline_tracking.yaml) / 그 외: pipeline.yaml")
    g.add_argument("--name", default=None, help="결과(원장 name) 이름")
    g.add_argument("--module", default=None, help="검출기/클러스터러 module 덮어쓰기")
    g.add_argument("--class", dest="cls", default=None, help="검출기/클러스터러 class 덮어쓰기")
    g.add_argument("--param", action="extend", nargs="+", default=[],
                   help="검출기/클러스터러 params 덮어쓰기 key=value (여러 개: --param a=1 b=2 또는 --param 반복)")
    g.add_argument("--pid-split", default=None, help="cluster/e2e: bench/splits 파일:부분")
    g = p.add_argument_group("detect")
    g.add_argument("--split", choices=["test", "train", "all"], default=None)
    g.add_argument("--limit", type=int, default=None, help="detect: 프레임 수 제한 (0 전부) / e2e: 순위 길이 K")
    g.add_argument("--every", type=int, default=None)
    g.add_argument("--conf-threshold", type=float, default=None, help="실행 임계값 (기본 0.05, AP 스윕용)")
    g.add_argument("--operating-threshold", type=float, default=None)
    g.add_argument("--compare", action="extend", nargs="+", default=[], help="같이 채점할 이름=detections.jsonl (여러 개)")
    g.add_argument("--reuse-detections", default=None, help="이미 있는 detections.jsonl 로 채점만")
    g.add_argument("--no-images", action="store_true")
    g = p.add_argument_group("embed")
    g.add_argument("--model", default=None, help="embed: siglip2 | irra | solider")
    g.add_argument("--batch-size", type=int, default=None)
    g.add_argument("--backbone", default=None)
    g.add_argument("--semantic-weight", type=float, default=None)
    g.add_argument("--neck-feat", default=None)
    g = p.add_argument_group("search")
    g.add_argument("--weights", default=None, help="siglip2=1.0,irra=1.5,solider=1.5")
    g.add_argument("--rrf-k", type=float, default=None)
    g.add_argument("--prefetch", type=int, default=None)
    g.add_argument("--pool", type=int, default=None, help="search/e2e: 재정렬 후보 수")
    g.add_argument("--pools", default=None)
    g.add_argument("--models", default=None, help="search: 평가할 모델 (쉼표; 기본 핵심 3종 + yaml person retriever 전부)")
    g = p.add_argument_group("cluster")
    g.add_argument("--method", default=None, help="내장 클러스터러 (leiden | dbscan_v6)")
    g.add_argument("--method-config", default=None, help="clusterer: yaml")
    g.add_argument("--target", default=None, help="person | object (기본 person)")
    g.add_argument("--sources", default=None, help="payload source 필터 (기본 prw_image)")
    g.add_argument("--vector", default=None)
    g.add_argument("--max-points", type=int, default=None)
    g.add_argument("--min-cluster-size", type=int, default=None)
    g.add_argument("--vector-cache", default=None)
    g = p.add_argument_group("e2e")
    g.add_argument("--stage1", nargs="*", default=None)
    g.add_argument("--rerank", default=None)
    g.add_argument("--max-queries", type=int, default=None)
    g.add_argument("--gallery", choices=["test", "all"], default=None)
    g = p.add_argument_group("track / object / qwen (P6 사람 정답)")
    g.add_argument("--gt-dir", default=None, help="정답 폴더 (track: eval/gt/tracks · object: eval/gt/object_pairs · qwen: eval/gt/qwen)")
    g.add_argument("--processed-root", default=None, help="track: 예측을 읽을 영상 파이프라인 출력 루트")
    g.add_argument("--videos", nargs="*", default=None, help="track: 영상 stem 목록 (기본 gt-dir 전부)")
    g.add_argument("--videos-root", default=None, help="track: --tracking-config 재추적 시 원본 영상 폴더")
    g.add_argument("--tracking-config", default=None, help="track: 이 yaml 로 GT 영상을 다시 추적·스티칭한 뒤 평가 (pipeline_tracking*.yaml)")
    g.add_argument("--pred-file", default=None, help="track: 다른 추적기 출력 파일 (jsonl/json)")
    g.add_argument("--restitch", action="store_true", help="track: --tracking-config 와 함께 — 검출·추적은 기존 출력을 쓰고 스티처(SUSHI + link_windows)만 다시 돌린다 (스티처 비교용, 빠름)")
    g.add_argument("--sushi-root", default=None, help="track --restitch: SUSHI 루트 (기본 ./third_party/SUSHI)")
    g.add_argument("--checkpoint", default=None, help="track --restitch: SUSHI 체크포인트 (기본 mot17private.pth)")
    g.add_argument("--collection", default=None, help="object: Qdrant 컬렉션 (기본 forensic_object)")
    g.add_argument("--threshold", type=float, default=None, help="object: 쌍 임계값(0.97) / qwen: 판정 임계값(0.5)")
    g.add_argument("--assignments", default=None, help="object: 트랙 클러스터 assignments.jsonl")
    g.add_argument("--top-k", type=int, default=None, help="qwen: Qwen 이 볼 상위 후보 수 (20)")
    g.add_argument("--alpha", type=float, default=None, help="qwen: 속성 점수 가중 (0.7)")
    g.add_argument("--verify-mode", choices=["flag", "filter"], default=None, help="qwen: flag 표시만 / filter FAIL 제거")
    g.add_argument("--no-reranker", action="store_true", help="qwen: Qwen3-VL-Reranker 생략")
    g.add_argument("--model-id", default=None, help="qwen: Instruct 모델 id")
    g.add_argument("--qwen-dir", default=None, help="qwen: Qwen 결과 캐시 폴더 (재사용)")
    g.add_argument("--allow-unlabeled", action="store_true", help="qwen: 라벨 없는 쿼리도 실행 (시간·UNKNOWN 만)")
    g.add_argument("--iou", type=float, default=None, help="track: 매칭 IoU (0.5)")
    g.add_argument("--max-gap", type=int, default=None, help="track: 구간 분할 프레임 간격 (proposals 값)")
    g.add_argument("--min-coverage", type=float, default=None, help="track/object/qwen: 정식 평가 최소 검토율 (1.0)")
    g.add_argument("--reranker-model-id", default=None, help="qwen: 재랭커 모델 id")
    g.add_argument("--dtype", default=None, help="qwen: bfloat16 …")
    g.add_argument("--max-pixels", type=int, default=None, help="qwen: 이미지 최대 픽셀")
    g.add_argument("--rescore", action="store_true", help="qwen: 계약이 맞는 캐시로 alpha/threshold 만 재채점")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="단일 러너 (bench/run.py)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for stage, help_text in (("detect", "검출기 평가 (PRW GT)"), ("embed", "임베더 단독 Re-ID 평가 (GT crop)"),
                             ("search", "통합 검색 조합 단독 평가 (GT crop, 모든 변형)"),
                             ("cluster", "클러스터링 플러그인 실행 + GT 평가"), ("e2e", "전체 파이프라인 검색 평가 (운영 DB)"),
                             ("track", "추적·스티칭 준정답 평가 (IDF1/HOTA/IDSW)"), ("object", "객체 재출현 쌍 평가 (객체 임베더)"),
                             ("qwen", "Qwen 후처리 평가 (P@K 변화·오탈락률)")):
        s = sub.add_parser(stage, help=help_text)
        add_common(s)
        add_stage_options(s)

    s = sub.add_parser("verify", help="이전 실행 재현 검증")
    add_common(s)
    s.add_argument("run_id", nargs="?", default=None)
    s.add_argument("--run-id", dest="run_id_opt", default=None, help="run_id (GUI 용; 위치 인자 대신)")
    s.add_argument("--tol", action="extend", nargs="+", default=[], help="지표=허용오차 (여러 개; 기본 ledger.VERIFY_TOLERANCES)")
    add_stage_options(s)     # GUI 가 넘기는 공통 인자를 무시하기 위해

    s = sub.add_parser("show-cmd", help="엔트리를 재현하는 러너 명령 출력")
    s.add_argument("run_id")
    s.add_argument("--ledger", default=None)
    return p


STAGE_KEYS = {
    "detect": {"config", "name", "module", "cls", "params", "split", "limit", "every", "conf_threshold", "operating_threshold",
               "compare", "reuse_detections", "no_images", "data_root"},
    "embed": {"config", "model", "batch_size", "backbone", "semantic_weight", "neck_feat", "data_root"},
    "search": {"config", "weights", "rrf_k", "prefetch", "pool", "pools", "models", "batch_size", "data_root"},
    "cluster": {"config", "name", "module", "cls", "params", "method", "method_config", "target", "sources", "vector",
                "max_points", "min_cluster_size", "vector_cache", "pid_split", "data_root"},
    "e2e": {"config", "name", "stage1", "rerank", "limit", "pool", "max_queries", "gallery", "pid_split", "data_root"},
    "track": {"name", "gt_dir", "processed_root", "videos", "videos_root", "tracking_config", "restitch", "sushi_root", "checkpoint", "pred_file",
              "iou", "max_gap", "min_coverage", "data_root"},
    "object": {"name", "gt_dir", "vector", "collection", "threshold", "assignments", "min_coverage", "data_root"},
    "qwen": {"name", "gt_dir", "top_k", "alpha", "threshold", "verify_mode", "no_reranker", "model_id", "max_queries", "qwen_dir", "allow_unlabeled",
             "reranker_model_id", "dtype", "max_pixels", "rescore", "min_coverage", "batch_size", "data_root"},
}


def parse_params(items: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--param 은 key=value: {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except ValueError:
            out[k.strip()] = v
    return out


def stage_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Namespace → 그 단계에 해당하는 인자만 (GUI 가 넘긴 다른 단계 옵션은 버린다)."""
    a = {k: v for k, v in vars(args).items() if k not in ("cmd", "ledger", "no_ledger", "runs_dir", "dry_run")}
    if "param" in a:
        a["params"] = parse_params(a.pop("param") or [])
    keep = STAGE_KEYS.get(args.cmd)
    if keep is not None:
        a = {k: v for k, v in a.items() if k in keep}
    if args.cmd == "detect" and a.get("config") is None and not a.get("module"):
        a["config"] = str(DEFAULT_TRACKING)
    if args.cmd == "embed" and not a.get("model"):
        raise SystemExit("embed 는 --model 이 필요합니다 (siglip2 | irra | solider)")
    if args.cmd == "cluster":
        a.setdefault("sources", "prw_image")
        a["sources"] = a["sources"] or "prw_image"
        a["target"] = a.get("target") or "person"
    if args.cmd == "cluster" and a["target"] != "person":
        raise SystemExit(f"cluster --target {a['target']}: PRW 에는 사람 GT 만 있어 러너로 평가할 수 없습니다 (clustering/cluster_qdrant.py 로 직접 실행)")
    if args.cmd == "search" and a.get("weights"):
        a["weights"] = {k.strip(): float(v) for k, v in (kv.split("=", 1) for kv in a["weights"].split(",") if kv.strip())}
    return {k: v for k, v in a.items() if not _empty(v)}


def _empty(v: Any) -> bool:
    """None / '' / [] / {} / False 만 '없음'. 숫자 0 은 값이다 (0 == False 함정 방지)."""
    return v is None or v is False or (isinstance(v, (str, list, dict)) and len(v) == 0)


def _video_file(src: Path, stem: str, videos_root: Any = None) -> Path:
    """처리 결과 폴더의 status.json 에 적힌 영상 경로, 없으면 videos_root(기본 data/videos)/<stem>.*"""
    try:
        st = json.loads((src / "status.json").read_text(encoding="utf-8-sig"))
        if st.get("video") and Path(st["video"]).is_file():
            return Path(st["video"])
    except Exception:  # noqa: BLE001
        pass
    root = Path(videos_root) if videos_root else PROJECT_ROOT / "data" / "videos"
    hits = sorted(root.glob(f"{stem}.*")) if root.is_dir() else []
    if not hits:
        raise SystemExit(f"영상 파일을 찾지 못했습니다: {root / stem}.* (SUSHI 입력을 다시 만들려면 원본 영상이 필요)")
    return hits[0]


def default_name(stage: str, a: Dict[str, Any]) -> str:
    if a.get("name"):
        return slug(a["name"])
    if stage == "detect":
        return slug(Path(a["config"]).stem if a.get("config") else (a.get("cls") or "detector"))
    if stage == "embed":
        return slug(a["model"])
    if stage == "search":
        return "unified"
    if stage == "cluster":
        return slug(a.get("method") or (a.get("cls") or "cluster"))
    if stage == "e2e":
        return slug("+".join(a.get("stage1") or ["default"]) + "__" + str(a.get("rerank") or "default"))
    if stage == "track":
        return slug((Path(a["tracking_config"]).stem + ("_restitch" if a.get("restitch") else "")) if a.get("tracking_config") else "tracks")
    if stage == "object":
        return slug(a.get("vector") or "object")
    if stage == "qwen":
        bs = int(a.get("batch_size") or 1)
        return slug("qwen_" + str(a.get("verify_mode") or "flag") + (f"_b{bs}" if bs > 1 else ""))
    return stage


def run_stage(stage: str, a: Dict[str, Any], runs_dir: Path, dry: bool, echo: Callable[[str], Any] = print) -> Tuple[List[Dict[str, Any]], Path]:
    import uuid
    # 같은 초에 같은 이름으로 병렬 시작해도 폴더가 겹치지 않게 (P3 병렬 trial)
    run_dir = runs_dir / f"{stage}_{time.strftime('%Y%m%dT%H%M%S')}_{default_name(stage, a)}_{uuid.uuid4().hex[:6]}"
    entries = Stage(stage, a, run_dir, dry, echo).execute()
    return entries, run_dir


def is_pseudo(entry: Dict[str, Any]) -> bool:
    """라벨 없는 결과(pseudo GT / unlabeled) — 실행 폴더에는 남기되 기본 원장에는 넣지 않는다."""
    return str(entry.get("note") or "") in ("pseudo GT", "unlabeled")


def summarize(entries: Sequence[Dict[str, Any]], echo: Callable[[str], Any] = print) -> None:
    for e in entries:
        keys = ledger.METRIC_KEYS.get(e.get("stage", ""), [])[:6]
        m = e.get("metrics") or {}
        echo(f"[run] {e['run_id']}  {e['name']}  " + " ".join(f"{k}={ledger.fmt(m.get(k))}" for k in keys))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    ledger_path = ledger.resolve_ledger_path(getattr(args, "ledger", None), disabled=getattr(args, "no_ledger", False))
    entries_all = ledger.read_entries(ledger.resolve_ledger_path(getattr(args, "ledger", None)) or ledger.DEFAULT_LEDGER)

    if args.cmd == "show-cmd":
        e = find_entry(entries_all, args.run_id)
        print(" ".join(_quote(c) for c in runner_command(e["stage"], args_from_entry(e))))
        return 0

    runs_dir = Path(args.runs_dir).resolve()
    if args.cmd == "verify":
        run_id = args.run_id or getattr(args, "run_id_opt", None)
        if not run_id:
            raise SystemExit("verify 는 run_id 가 필요합니다 (위치 인자 또는 --run-id)")
        target = find_entry(entries_all, run_id)
        stage = target["stage"]
        a = args_from_entry(target)
        want_name = target["name"]
        if stage in ("detect", "cluster", "e2e", "track", "object", "qwen"):
            a["name"] = f"{want_name}__verify"
            want_name = a["name"]
        if stage == "detect":
            a.pop("compare", None)
            a.pop("reuse_detections", None)
        if args.data_root:
            a["data_root"] = args.data_root
        tol = dict(ledger.VERIFY_TOLERANCES.get(stage, {}))
        for item in args.tol:
            k, v = item.split("=", 1)
            tol[k.strip()] = float(v)
        print(f"[verify] {target['run_id']} ({stage} · {target['name']}) 를 같은 설정으로 다시 실행")
        print("[verify] " + " ".join(_quote(c) for c in runner_command(stage, a)))
        entries, run_dir = run_stage(stage, a, runs_dir, args.dry_run)
        if args.dry_run:
            return 0
        compare_name = want_name if stage != "search" else target["name"]
        matched = [e for e in entries if e.get("name") == compare_name]
        if not matched:
            status, rows, new = "UNVERIFIED", [], None
            print(f"\n[verify] UNVERIFIED — 재실행 결과에 '{compare_name}' 이(가) 없음 (있는 것: {[e.get('name') for e in entries][:8]})")
        else:
            new = matched[0]
            status, rows = compare_metrics(target.get("metrics") or {}, new.get("metrics") or {}, tol)
            print(f"\n[verify] {status}  ({target['run_id']} → {new['run_id']})")
            for r in rows:
                if r.get("after") is None:
                    print(f"  NG  {r['metric']:<16} {ledger.fmt(r['before'])} → (없음)")
                else:
                    print(f"  {'ok ' if r['ok'] else 'NG '} {r['metric']:<16} {ledger.fmt(r['before'])} → {ledger.fmt(r['after'])}  Δ {r['delta']:+.4f} (허용 ±{r['tol']})")
            if not rows:
                print("  비교할 공통 지표가 없습니다 → UNVERIFIED (통과로 기록하지 않음)")
        verify = {"against": target["run_id"], "status": status, "passed": status == "PASS", "rows": rows, "tolerances": tol,
                  "compared_name": compare_name,
                  "same_git_commit": bool(new) and (target.get("env") or {}).get("git_commit") == (new.get("env") or {}).get("git_commit"),
                  "same_host": bool(new) and (target.get("env") or {}).get("host") == (new.get("env") or {}).get("host"),
                  "same_config_sha": bool(new) and (target.get("config") or {}).get("sha256") == (new.get("config") or {}).get("sha256")}
        for e in entries:                       # verify 판정은 비교한 행에만; 같이 나온 다른 variant 는 동반 결과로만 표시
            if new is not None and e is new:
                e["verify"] = verify
                e["note"] = f"verify of {target['run_id']}: {status}"
            else:
                e["note"] = f"verify run of {target['run_id']} (동반 결과, 비교 대상 아님)"
        (run_dir / "verify.json").write_text(ledger._dumps({"target": target, "result": new, "verify": verify}, indent=2),
                                             encoding="utf-8", newline="\n")
        if ledger_path is not None:
            n, _ = ledger.append_entries(ledger_path, entries)
            print(f"[ledger] {n} 건 기록 → {ledger_path}")
        summarize(entries)
        return 0 if status == "PASS" else 1

    stage = args.cmd
    a = stage_args(args)
    entries, run_dir = run_stage(stage, a, runs_dir, args.dry_run)
    if args.dry_run:
        return 0
    if ledger_path is not None:
        keep = [e for e in entries if not is_pseudo(e)]
        if len(keep) < len(entries):
            print(f"[ledger] 라벨 없는(pseudo/unlabeled) {len(entries) - len(keep)} 건은 기본 원장에 기록하지 않음 (실행 폴더 ledger_part.jsonl 에만)")
        n, _ = ledger.append_entries(ledger_path, keep) if keep else (0, 0)
        print(f"[ledger] {n} 건 기록 → {ledger_path}")
    summarize(entries)
    print(f"[run] 폴더: {run_dir}")
    print(f"RESULT_SUMMARY: {run_dir / 'ledger_part.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
