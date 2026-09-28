"""bench/register.py — P7: 새 모델 등록기. 어댑터 1개 + yaml 블록 → 계약 검사 → 단계 벤치 자동 → 원장 순위.

  python bench/register.py template detector|clusterer|embedder --name NAME [--out PATH] [--dim D]      어댑터 스켈레톤 생성
  python bench/register.py detector  --name NAME --module M --class C [--param k=v …] [--limit 300] [--check-only] [--no-bench]
  python bench/register.py clusterer --name NAME --module M --class C [--param k=v …] [--max-points 3000] [--check-only] [--no-bench]
  python bench/register.py embedder  --name NAME --module M --class C --dim D [--scope all|person|object] [--supports-text]
                                     [--weight 1.0] [--param k=v …] [--check-only] [--no-bench]

흐름
  1) 계약 검사 (bench/check.py: import · base 상속/메서드 재정의 · 생성자 인자 · 실제 생성해 작은 입력으로 출력 규격)  — FAIL 이면 여기서 멈춤
  2) yaml 등록 (원본 불변): 검출기 → pipeline_tracking_<이름>.yaml (tracker/stitcher 는 pipeline_tracking.yaml 그대로)
                             클러스터러 → clusterer_<이름>.yaml
                             임베더 → pipeline_<이름>.yaml (pipeline.yaml 사본의 retrievers: 에 블록 추가, 주석 보존, 로더 검증)
  3) 단계 벤치 (bench/run.py detect|cluster|embed) → 원장 기록
  4) 원장에서 같은 단계의 이름별 최근 행과 비교해 순위·채택 기준(bench/criteria.py) 출력 → GUI 벤치마크 탭에서도 보임
코드 수정 없이 되는 예: 같은 어댑터 클래스에 다른 가중치 (YOLO26Detector + weights=yolo26s.pt), 같은 임베더 클래스에 다른 model_id.
한계: 임베더는 단독 Re-ID 평가(GT crop)까지만 자동 — 운영 DB 적재(ingest/build_db.py) 와 통합 검색·e2e 는 수동 (README 6.1).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench import check as CHK  # noqa: E402
from bench import criteria, ledger  # noqa: E402

PY = sys.executable
KINDS = ("detector", "clusterer", "embedder")
STAGE_OF = {"detector": "detect", "clusterer": "cluster", "embedder": "embed"}
OBJECTIVE_OF = {"detect": "ap50", "cluster": "b3_f1", "embed": "map"}


def slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(text)).strip("_")
    if not s or s[0].isdigit():
        s = "m_" + s
    return s


def parse_params(items: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--param 은 key=value: {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except ValueError:
            out[k.strip()] = v
    return out


# ---------------------------------------------------------------- yaml 등록
def _header(kind: str, name: str, spec: Dict[str, Any]) -> str:
    return (f"# bench/register.py 등록 ({time.strftime('%Y-%m-%d %H:%M')}): {kind} '{name}' = {spec.get('module')}.{spec.get('class')}. "
            f"원본 설정 파일은 바꾸지 않았다.\n")


def write_detector_yaml(name: str, spec: Dict[str, Any], root: Path, template: Path, overwrite: bool = False) -> Path:
    import yaml
    raw = yaml.safe_load(template.read_text(encoding="utf-8-sig")) or {}
    raw["detector"] = {"module": spec["module"], "class": spec["class"], "params": dict(spec.get("params") or {})}
    out = root / f"pipeline_tracking_{slug(name)}.yaml"
    if out.exists() and not overwrite:
        raise FileExistsError(f"이미 있음: {out} (--overwrite)")
    out.write_text(_header("detector", name, spec) + f"# tracker/stitcher 는 {template.name} 그대로\n"
                   + yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
    return out


def write_clusterer_yaml(name: str, spec: Dict[str, Any], root: Path, overwrite: bool = False) -> Path:
    import yaml
    out = root / f"clusterer_{slug(name)}.yaml"
    if out.exists() and not overwrite:
        raise FileExistsError(f"이미 있음: {out} (--overwrite)")
    block = {"clusterer": {"module": spec["module"], "class": spec["class"], "params": dict(spec.get("params") or {})}}
    out.write_text(_header("clusterer", name, spec) + yaml.safe_dump(block, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
    return out


def retriever_block(name: str, spec: Dict[str, Any], dim: int, scope: str, supports_text: bool, weight: float) -> str:
    import yaml
    tool = {"person": "human", "object": "object"}.get(scope, "common")
    lines = [f"  {name}:", f"    tool: {tool}", f"    scope: {scope}", f"    supports_text: {'true' if supports_text else 'false'}",
             f"    dim: {int(dim)}", f"    weight: {float(weight)}", f"    module: {spec['module']}", f"    class: {spec['class']}"]
    params = dict(spec.get("params") or {})
    if params:
        lines.append("    params:")
        dumped = yaml.safe_dump(params, allow_unicode=True, sort_keys=False, default_flow_style=False)
        lines += ["      " + ln for ln in dumped.rstrip("\n").splitlines()]
    else:
        lines.append("    params: {}")
    return "\n".join(lines) + "\n"


def write_embedder_yaml(name: str, spec: Dict[str, Any], dim: int, scope: str, supports_text: bool, weight: float, root: Path,
                        src: Path, overwrite: bool = False) -> Path:
    """pipeline.yaml 사본의 retrievers: 절 끝에 블록을 끼워 넣는다 (주석 보존) → 로더로 검증."""
    text = src.read_text(encoding="utf-8-sig")
    lines = text.splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines) if re.match(r"^retrievers:\s*(#.*)?$", ln.rstrip("\r\n"))), None)
    if start is None:
        raise ValueError(f"{src}: retrievers: 절이 없습니다")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if re.match(r"^[A-Za-z_]", lines[i]):
            end = i
            break
    existing = [m.group(1) for ln in lines[start + 1:end] for m in [re.match(r"^  ([A-Za-z0-9_]+):\s*(#.*)?$", ln.rstrip("\r\n"))] if m]
    if name in existing:
        raise ValueError(f"retrievers.{name} 이(가) 이미 {src.name} 에 있습니다 — 다른 --name 을 쓰세요 (같은 이름을 두 번 정의할 수 없음)")
    nl = "\r\n" if lines[0].endswith("\r\n") else "\n"
    block = retriever_block(name, spec, dim, scope, supports_text, weight).replace("\n", nl)
    insert_at = end
    while insert_at > start + 1 and lines[insert_at - 1].strip() == "":
        insert_at -= 1
    new_lines = lines[:insert_at] + [nl, block] + lines[insert_at:]
    out = root / f"pipeline_{slug(name)}.yaml"
    if out.exists() and not overwrite:
        raise FileExistsError(f"이미 있음: {out} (--overwrite)")
    out.write_text(_header("embedder", name, spec).replace("\n", nl) + "".join(new_lines), encoding="utf-8", newline="")
    from config import PipelineConfig
    cfg = PipelineConfig.load(out)          # 형식 오류면 여기서 예외
    if name not in cfg.retrievers:
        raise ValueError(f"{out}: 로더가 retrievers.{name} 을 읽지 못했습니다")
    return out


# ---------------------------------------------------------------- 벤치 · 순위
def bench_command(kind: str, name: str, yaml_path: Path, limit: Optional[int] = None, max_points: Optional[int] = None,
                  ledger_path: Optional[str] = None) -> List[str]:
    if kind == "detector":
        cmd = [PY, "bench/run.py", "detect", "--config", str(yaml_path), "--name", name, "--no-images"]
        if limit:
            cmd += ["--limit", str(limit)]
    elif kind == "clusterer":
        cmd = [PY, "bench/run.py", "cluster", "--method-config", str(yaml_path), "--name", name]
        if max_points:
            cmd += ["--max-points", str(max_points)]
    else:
        cmd = [PY, "bench/run.py", "embed", "--model", name, "--config", str(yaml_path)]
    if ledger_path:
        cmd += ["--ledger", str(ledger_path)]
    return cmd


# ---------------------------------------------------------------- 임베더: 소규모 운영 DB 적재 + e2e 비교 (별도 collection prefix, 운영 컬렉션은 건드리지 않음)
INGEST_ROOT = PROJECT_ROOT / "bench" / "ingest"
DEFAULT_PRW_STATS = [PROJECT_ROOT / "data" / "prw_crops_p25h75" / "filter_stats_dedup.json",
                     PROJECT_ROOT / "data" / "prw_crops_p25h75" / "filter_stats.json"]
PROD_STAGE1 = ("siglip2", "irra")
PROD_RERANK = "solider"


def set_collection_prefix(yaml_path: Path, prefix: str) -> str:
    """pipeline 사본의 최상위 collection_prefix 를 바꾼다(없으면 맨 앞에 넣는다) → 적재가 <prefix>_person/_object 새 컬렉션으로 간다. 로더로 검증."""
    text = yaml_path.read_text(encoding="utf-8-sig")
    lines = text.splitlines(keepends=True)
    nl = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"
    done = False
    for i, ln in enumerate(lines):
        if re.match(r"^collection_prefix\s*:", ln):
            lines[i] = f"collection_prefix: {prefix}{nl}"
            done = True
            break
    if not done:
        lines.insert(0, f"collection_prefix: {prefix}   # bench/register.py — 표본 적재용 별도 컬렉션{nl}")
    yaml_path.write_text("".join(lines), encoding="utf-8", newline="")
    from config import PipelineConfig
    cfg = PipelineConfig.load(yaml_path)
    if cfg.collection_prefix != prefix:
        raise ValueError(f"{yaml_path}: collection_prefix 를 {prefix} 로 바꾸지 못했습니다 (로더가 읽은 값 {cfg.collection_prefix})")
    return cfg.person_collection()


def sample_prw_frames(stats_path: Path, n_frames: int, frames: Sequence[str], out_dir: Path, sample_file: Optional[Path] = None) -> Tuple[Path, Dict[str, Any]]:
    """PRW test 프레임 목록에서 고르게 n_frames 개를 뽑아(결정적; 같은 n 이면 같은 표본 → 임베더끼리 비교 가능) 그 프레임의 crop 만 남긴 stats JSON 을 만든다."""
    frames = sorted(frames)
    n = max(1, min(int(n_frames), len(frames)))
    sample_file = sample_file or (INGEST_ROOT / "samples" / f"prw_test_{n}.json")
    if sample_file.is_file():
        chosen = set(json.loads(sample_file.read_text(encoding="utf-8"))["frames"])
    else:
        step = len(frames) / n
        chosen = {frames[int(i * step)] for i in range(n)}
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        sample_file.write_text(json.dumps({"n": n, "of": len(frames), "frames": sorted(chosen)}, ensure_ascii=False, indent=1), encoding="utf-8")
    data = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    keep = [c for c in data.get("crops", []) if Path(str(c.get("image_id", ""))).stem in chosen]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "filter_stats_sample.json"
    out.write_text(json.dumps({"crops": keep, "filtered": [], "sample": {"source_stats": str(stats_path), "frames": len(chosen), "sample_file": str(sample_file)}},
                              ensure_ascii=False), encoding="utf-8")
    return out, {"frames": len(chosen), "crops": len(keep), "sample_file": str(sample_file), "source_stats": str(stats_path)}


def ingest_commands(name: str, yaml_path: Path, sample_stats: Path, work_dir: Path, n_frames: int, max_queries: int = 0,
                    ledger_path: Optional[str] = None) -> Dict[str, List[str]]:
    """① build_db 로 표본 적재(별도 checkpoint/manifest) ② 같은 컬렉션에서 e2e 검색: 새 임베더 단독 vs 운영 조합(siglip2+irra→solider)."""
    build = [PY, "ingest/build_db.py", "--config", str(yaml_path), "--stats", str(sample_stats), "--checkpoint-dir", str(work_dir / "checkpoint"),
             "--manifest-dir", str(work_dir / "manifests")]
    common = [PY, "eval/prw_e2e_search_eval.py", "--config", str(yaml_path), "--matches-cache", str(work_dir / "prw_gt_matches.jsonl"),
              "--output-dir", str(work_dir / "e2e"), "--gallery", "test"]
    if max_queries:
        common += ["--max-queries", str(max_queries)]
    if ledger_path:
        common += ["--ledger", str(ledger_path)]
    e2e_new = common + ["--stage1", name, "--rerank", "none", "--name", f"{name}__sample{n_frames}"]
    e2e_prod = common + ["--stage1", *PROD_STAGE1, "--rerank", PROD_RERANK, "--name", f"prod__sample{n_frames}"]
    return {"build": build, "e2e_new": e2e_new, "e2e_prod": e2e_prod}


def _run_logged(cmd: List[str], root: Path, log, tail_lines: int = 8) -> Tuple[int, float, str]:
    log("[register] $ " + " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(root), text=True, encoding="utf-8", errors="replace", capture_output=True)
    tail = "\n".join((proc.stdout or "").splitlines()[-tail_lines:])
    log(tail)
    if proc.returncode != 0:
        log((proc.stderr or "")[-800:])
    return proc.returncode, round(time.time() - t0, 1), tail


def ingest_and_compare(name: str, yaml_path: Path, n_frames: int, *, root: Path = PROJECT_ROOT, stats_path: Optional[Path] = None,
                       max_queries: int = 0, ledger_path: Optional[Path] = None, data_root: Optional[Path] = None, log=print) -> Dict[str, Any]:
    """임베더 등록 뒤: 별도 prefix 로 표본 적재 → e2e 검색 비교. 결과 dict (원장에는 e2e 행 2개가 남는다: <name>__sampleN, prod__sampleN)."""
    stats = stats_path or next((p for p in DEFAULT_PRW_STATS if p.is_file()), None)
    if stats is None or not Path(stats).is_file():
        raise FileNotFoundError("PRW crop stats 가 없습니다 (data/prw_crops_p25h75/filter_stats*.json) — --stats 로 지정")
    from eval import prw_eval
    data_root = data_root or (root / "data" / "PRW")
    frames = prw_eval.load_frame_list(data_root / "frame_test.mat")
    work = INGEST_ROOT / slug(name)
    work.mkdir(parents=True, exist_ok=True)
    collection = set_collection_prefix(yaml_path, f"bench_{slug(name)}")
    sample_stats, info = sample_prw_frames(Path(stats), n_frames, frames, work)
    log(f"[register] 표본 적재: PRW test 프레임 {info['frames']} / crop {info['crops']} → 컬렉션 {collection} (운영 forensic_* 은 건드리지 않음)")
    cmds = ingest_commands(name, yaml_path, sample_stats, work, info["frames"], max_queries, str(ledger_path) if ledger_path else None)
    out: Dict[str, Any] = {"collection": collection, "yaml": str(yaml_path), "sample": info, "commands": cmds, "steps": {}}
    for step in ("build", "e2e_new", "e2e_prod"):
        code, sec, tail = _run_logged(cmds[step], root, log)
        out["steps"][step] = {"exit": code, "sec": sec}
        if code != 0:
            out["error"] = f"{step} 실패 (exit {code})"
            log(f"[register] {out['error']}")
            return out
    entries = ledger.read_entries(ledger_path or ledger.DEFAULT_LEDGER)
    latest = ledger.latest_by_name(ledger.filter_entries(entries, stage="e2e"))
    rows = {}
    for key in (f"{name}__sample{info['frames']}", f"prod__sample{info['frames']}"):
        e = latest.get(key)
        rows[key] = {k: (e.get("metrics") or {}).get(k) for k in ("map", "map_db", "rank1", "det_ceiling")} if e else None
    out["e2e"] = rows
    log(f"[register] e2e 비교 (같은 표본 컬렉션 {collection}): " + " · ".join(f"{k}: mAP {ledger.fmt((v or {}).get('map'))} / R1 {ledger.fmt((v or {}).get('rank1'))}" for k, v in rows.items()))
    return out



def rank_in_ledger(stage: str, name: str, ledger_path: Optional[Path] = None) -> Dict[str, Any]:
    entries = ledger.read_entries(ledger_path or ledger.DEFAULT_LEDGER)
    latest = ledger.latest_by_name(ledger.filter_entries(entries, stage=stage))
    objective = OBJECTIVE_OF.get(stage, "")
    rows = sorted(latest.values(), key=lambda e: -(float((e.get("metrics") or {}).get(objective) or 0.0)))
    pos = next((i + 1 for i, e in enumerate(rows) if e.get("name") == name), None)
    mine = latest.get(name)
    def gt_size(e: Dict[str, Any]) -> Any:
        g = e.get("gt") or {}
        return g.get("frames") or g.get("evaluated_points") or g.get("query_total")

    return {"objective": objective, "position": pos, "total": len(rows), "entry": mine,
            "status": criteria.evaluate(mine)["status"] if mine else None,
            "top": [(e.get("name"), (e.get("metrics") or {}).get(objective), gt_size(e)) for e in rows[:5]],
            "gt_size": gt_size(mine) if mine else None}


# ---------------------------------------------------------------- 템플릿
DETECTOR_TEMPLATE = '''"""{cls} — bench/register.py 가 만든 검출기 어댑터 스켈레톤 (계약: detect.base.BaseDetector).

detect(frame, *, frame_idx, timestamp_sec=None) → List[Detection]
  frame: HxWx3 uint8 (파이프라인은 BGR; 모델이 RGB 면 여기서 바꾼다)
  Detection(frame_idx, bbox=(x1, y1, x2, y2) 픽셀, confidence 0~1, class_id=COCO 1-based id, class_name, timestamp_sec)
필수 속성: conf_threshold (채점·러너가 읽는다). filter_forensic=True 면 FORENSIC_CLASSES 만 남긴다.
등록: python bench/register.py detector --name {name} --module {module} --class {cls} --param weights=... --limit 300
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from detect.base import FORENSIC_CLASSES, BaseDetector, Detection, coco_id_by_name


class {cls}(BaseDetector):
    def __init__(self, weights: str = "", conf_threshold: float = 0.2, filter_forensic: bool = True,
                 input_color: str = "bgr", device: Optional[str] = None):
        self.weights = weights
        self.conf_threshold = float(conf_threshold)
        self.filter_forensic = bool(filter_forensic)
        self.input_color = input_color
        self.device = device
        self._model = None            # TODO: 지연 로드 권장 (첫 detect 에서 로드)
        self._coco = coco_id_by_name()

    def _load(self):
        if self._model is None:
            raise NotImplementedError("TODO: 모델 로드")   # 예: self._model = SomeLib.load(self.weights, device=self.device)

    def detect(self, frame, *, frame_idx: int, timestamp_sec: Optional[float] = None) -> List[Detection]:
        self._load()
        image = frame if self.input_color == "bgr" else frame[..., ::-1]
        raw = []                        # TODO: [(x1, y1, x2, y2, conf, class_name), ...] 를 모델에서 얻는다
        out: List[Detection] = []
        for x1, y1, x2, y2, conf, name in raw:
            if conf < self.conf_threshold:
                continue
            if self.filter_forensic and name not in FORENSIC_CLASSES:
                continue
            out.append(Detection(frame_idx=frame_idx, bbox=(float(x1), float(y1), float(x2), float(y2)), confidence=float(conf),
                                 class_id=int(self._coco.get(name, -1)), class_name=str(name), timestamp_sec=timestamp_sec))
        return out
'''

CLUSTERER_TEMPLATE = '''"""{cls} — bench/register.py 가 만든 클러스터러 스켈레톤 (계약: clustering.base.BaseClusterer).

cluster(ids, primary, vectors, log) → ClusterResult(labels, stats)
  primary: (N, D) float32, 행 L2 정규화 (벡터 없는 point 는 0 행). vectors: name → (N, D_name) (required_vectors 에 적은 것이 들어온다)
  labels: ids 와 같은 순서의 int 또는 None(노이즈). name = 파일명·payload 키 접두어. params() = 보고서에 실리는 하이퍼파라미터.
등록: python bench/register.py clusterer --name {name} --module {module} --class {cls} --param knn=30 --max-points 3000
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from clustering.base import BaseClusterer, ClusterResult


class {cls}(BaseClusterer):
    name = "{name}"
    required_vectors = ()                 # primary 외에 필요한 named vector (예: ("irra",))

    def __init__(self, knn: int = 30, threshold: float = 0.9, seed: int = 42):
        self.knn = int(knn)
        self.threshold = float(threshold)
        self.seed = int(seed)

    def params(self) -> Dict[str, Any]:
        return dict(knn=self.knn, threshold=self.threshold, seed=self.seed)

    def cluster(self, ids: Sequence[Any], primary: np.ndarray, vectors: Dict[str, np.ndarray], log=print) -> ClusterResult:
        n = len(ids)
        labels: List[Optional[int]] = [None] * n      # TODO: 군집 번호를 채운다 (None = 노이즈)
        # 힌트: from clustering.cluster_leiden_qdrant import exact_topk  → (idx, sim) kNN 그래프
        return ClusterResult(labels=labels, stats={{"points": n}})
'''

EMBEDDER_TEMPLATE = '''"""{cls} — bench/register.py 가 만든 임베더 스켈레톤 (계약: embedders.base.BaseEmbedder).

DIM 클래스 변수와 _encode(images: List[PIL.Image]) → (len(images), DIM) 만 구현하면 전처리·배치·L2 정규화·검증은 BaseEmbedder 가 한다.
텍스트 검색(supports_text: true)을 지원하려면 embed_text(texts) → (N, DIM) 도 구현.
등록: python bench/register.py embedder --name {name} --module {module} --class {cls} --dim {dim} --scope person --param model_id=...
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PIL import Image

from embedders.base import BaseEmbedder


class {cls}(BaseEmbedder):
    DIM = {dim}

    def __init__(self, model_id: str = "", device: Optional[str] = None, batch_size: int = 32, l2_normalize: bool = True):
        super().__init__(device=device, batch_size=batch_size, l2_normalize=l2_normalize)
        self.model_id = model_id
        self._model = None                # TODO: 지연 로드 권장

    def _encode(self, images: List[Image.Image]) -> np.ndarray:
        # TODO: RGB PIL 이미지 리스트 → (len(images), DIM) float32
        raise NotImplementedError("TODO: 모델 추론")
'''


def write_template(kind: str, name: str, out: Optional[Path] = None, dim: int = 512, root: Path = PROJECT_ROOT) -> Tuple[Path, str, str]:
    """스켈레톤 파일을 만들고 (경로, module, class) 를 돌려준다."""
    base = slug(name)
    cls = "".join(p.capitalize() for p in base.split("_")) + {"detector": "Detector", "clusterer": "Clusterer", "embedder": "Embedder"}[kind]
    default = {"detector": root / "detect" / "detectors" / f"{base}_detector.py",
               "clusterer": root / "clustering" / "methods" / f"{base}.py",
               "embedder": root / "embedders" / f"{base}_embedder.py"}[kind]
    path = Path(out) if out else default
    try:
        rel = path.resolve().relative_to(root.resolve())
        module = ".".join(rel.with_suffix("").parts)
    except ValueError:
        module = path.stem
    if path.exists():
        raise FileExistsError(f"이미 있음: {path}")
    tmpl = {"detector": DETECTOR_TEMPLATE, "clusterer": CLUSTERER_TEMPLATE, "embedder": EMBEDDER_TEMPLATE}[kind]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tmpl.format(cls=cls, name=base, module=module, dim=int(dim)), encoding="utf-8", newline="\n")
    return path, module, cls


# ---------------------------------------------------------------- 등록 절차
def run_checks(kind: str, spec: Dict[str, Any], dim: Optional[int], supports_text: bool, instantiate: bool) -> List[Dict[str, Any]]:
    if kind == "detector":
        return CHK.check_detector_spec(spec, instantiate, target=f"detector:{spec['class']}")
    if kind == "clusterer":
        return CHK.check_clusterer_spec(spec, instantiate)
    return CHK.check_embedder_spec(spec, int(dim or 0), supports_text, instantiate)


def register(kind: str, name: str, spec: Dict[str, Any], *, root: Path = PROJECT_ROOT, dim: Optional[int] = None, scope: str = "all",
             supports_text: bool = False, weight: float = 1.0, limit: Optional[int] = None, max_points: Optional[int] = None,
             check_only: bool = False, no_bench: bool = False, overwrite: bool = False, instantiate: bool = True,
             tracking_template: Optional[Path] = None, pipeline_path: Optional[Path] = None, ledger_path: Optional[Path] = None,
             ingest_frames: int = 0, stats_path: Optional[Path] = None, e2e_max_queries: int = 0, log=print) -> Dict[str, Any]:
    result: Dict[str, Any] = {"kind": kind, "name": name, "spec": spec, "checks": [], "yaml": None, "bench": None, "rank": None}
    log(f"[register] {kind} '{name}' = {spec['module']}.{spec['class']} params={json.dumps(spec.get('params') or {}, ensure_ascii=False)}")
    rows = run_checks(kind, spec, dim, supports_text, instantiate)
    result["checks"] = rows
    log(CHK.render(rows))
    if any(r["status"] == "FAIL" for r in rows):
        result["error"] = "계약 검사 FAIL — 어댑터/params 를 고친 뒤 다시 실행"
        log(f"[register] {result['error']}")
        return result
    if check_only:
        log("[register] --check-only: yaml 등록·벤치 생략")
        return result
    if kind == "detector":
        out = write_detector_yaml(name, spec, root, tracking_template or (root / "pipeline_tracking.yaml"), overwrite)
    elif kind == "clusterer":
        out = write_clusterer_yaml(name, spec, root, overwrite)
    else:
        out = write_embedder_yaml(name, spec, int(dim or 0), scope, supports_text, weight, root, pipeline_path or (root / "pipeline.yaml"), overwrite)
    result["yaml"] = str(out)
    log(f"[register] yaml 등록 → {out}")
    if no_bench:
        return result
    cmd = bench_command(kind, name, out, limit, max_points, str(ledger_path) if ledger_path else None)
    log("[register] 벤치: " + " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(root), text=True, encoding="utf-8", errors="replace", capture_output=True)
    tail = "\n".join((proc.stdout or "").splitlines()[-12:])
    log(tail)
    result["bench"] = {"exit": proc.returncode, "sec": round(time.time() - t0, 1), "command": cmd}
    if proc.returncode != 0:
        result["error"] = f"벤치 실패 (exit {proc.returncode}); stderr 끝: {(proc.stderr or '')[-600:]}"
        log(f"[register] {result['error']}")
        return result
    rank = rank_in_ledger(STAGE_OF[kind], name, ledger_path)
    result["rank"] = {k: v for k, v in rank.items() if k != "entry"}
    if rank["position"]:
        log(f"[register] 원장 순위: {rank['position']}/{rank['total']} ({rank['objective']}), 채택 기준 {criteria.STATUS_LABEL.get(rank['status'], rank['status'])}"
            f" · 이 벤치의 GT 크기 {rank.get('gt_size')}")
        for i, (n, v, g) in enumerate(rank["top"], 1):
            log(f"    {i}. {n:<32} {rank['objective']}={ledger.fmt(v)}  (GT {g})")
        sizes = {g for _, _, g in rank["top"] if g is not None}
        if rank.get("gt_size") is not None and len(sizes | {rank['gt_size']}) > 1:
            log("[register] 주의: GT 크기가 다른 행이 섞여 있습니다 — 같은 조건으로 비교하려면 전체 프레임 벤치(--limit 0 / --max-points 0)를 돌리세요.")
        log("[register] GUI 벤치마크 탭에서 같은 표를 볼 수 있습니다.")
    else:
        log("[register] 원장에서 이름을 찾지 못했습니다 (벤치 로그 확인)")
    if kind == "embedder" and ingest_frames:
        try:
            result["ingest"] = ingest_and_compare(name, out, int(ingest_frames), root=root, stats_path=stats_path, max_queries=e2e_max_queries,
                                                  ledger_path=ledger_path, log=log)
        except Exception as exc:  # noqa: BLE001
            result["ingest"] = {"error": f"{type(exc).__name__}: {exc}"}
            log(f"[register] 표본 적재/e2e 실패: {result['ingest']['error']}")
    return result


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="새 모델 등록기 (P7)")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template", help="어댑터 스켈레톤 생성")
    t.add_argument("kind", choices=KINDS)
    t.add_argument("--name", required=True)
    t.add_argument("--out", default=None)
    t.add_argument("--dim", type=int, default=512)
    for kind in KINDS:
        s = sub.add_parser(kind, help=f"{kind} 등록: 검사 → yaml → 벤치 → 순위")
        s.add_argument("--name", required=True, help="원장·yaml 에 쓸 이름 (영숫자·_)")
        s.add_argument("--module", required=True)
        s.add_argument("--class", dest="cls", required=True)
        s.add_argument("--param", action="extend", nargs="+", default=[], help="생성자 params key=value (여러 개)")
        s.add_argument("--check-only", action="store_true")
        s.add_argument("--no-bench", action="store_true", help="yaml 등록까지만")
        s.add_argument("--no-instantiate", action="store_true", help="정적 검사만 (모델 로드 생략)")
        s.add_argument("--overwrite", action="store_true")
        s.add_argument("--root", default=str(PROJECT_ROOT), help="yaml 을 쓸 폴더 (기본 프로젝트 루트 → 드롭다운에 등장)")
        s.add_argument("--ledger", default=None)
        # kind 별
        s.add_argument("--limit", type=int, default=300, help="detector: 벤치 프레임 수 (0 = 전체 6,112)")
        s.add_argument("--max-points", type=int, default=3000, help="clusterer: 벤치 point 수 (0 = 전체)")
        s.add_argument("--tracking-template", default=None, help="detector: tracker/stitcher 를 가져올 yaml (기본 pipeline_tracking.yaml)")
        s.add_argument("--dim", type=int, default=None, help="embedder: 벡터 차원 (필수)")
        s.add_argument("--scope", choices=["all", "person", "object"], default="person")
        s.add_argument("--supports-text", action="store_true")
        s.add_argument("--weight", type=float, default=1.0)
        s.add_argument("--pipeline", default=None, help="embedder: 사본의 원본 pipeline.yaml (기본 프로젝트 것)")
        s.add_argument("--ingest-frames", type=int, default=0, help="embedder: PRW test 프레임 N 개의 crop 을 별도 컬렉션(bench_<이름>_*)에 적재하고 e2e 검색을 운영 조합과 비교 (0 = 생략)")
        s.add_argument("--stats", default=None, help="embedder --ingest-frames: crop stats JSON (기본 data/prw_crops_p25h75/filter_stats_dedup.json)")
        s.add_argument("--e2e-max-queries", type=int, default=0, help="embedder --ingest-frames: e2e 쿼리 수 제한 (0 = 전부)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "template":
        path, module, cls = write_template(args.kind, args.name, Path(args.out) if args.out else None, args.dim)
        print(f"[template] {path}\n다음: 파일의 TODO 를 채운 뒤\n  python bench/register.py {args.kind} --name {slug(args.name)} --module {module} --class {cls}"
              + (f" --dim {args.dim}" if args.kind == "embedder" else "") + " --param ...")
        return 0
    kind = args.cmd
    if kind == "embedder" and not args.dim:
        raise SystemExit("embedder 는 --dim 이 필요합니다")
    spec = {"module": args.module, "class": args.cls, "params": parse_params(args.param)}
    res = register(kind, slug(args.name), spec, root=Path(args.root).resolve(), dim=args.dim, scope=args.scope, supports_text=args.supports_text,
                   weight=args.weight, limit=args.limit or None, max_points=args.max_points or None, check_only=args.check_only,
                   no_bench=args.no_bench, overwrite=args.overwrite, instantiate=not args.no_instantiate,
                   tracking_template=Path(args.tracking_template) if args.tracking_template else None,
                   pipeline_path=Path(args.pipeline) if args.pipeline else None, ledger_path=Path(args.ledger) if args.ledger else None,
                   ingest_frames=int(getattr(args, "ingest_frames", 0) or 0), stats_path=Path(args.stats) if getattr(args, "stats", None) else None,
                   e2e_max_queries=int(getattr(args, "e2e_max_queries", 0) or 0))
    if res.get("yaml"):
        print(f"RESULT_SUMMARY: {res['yaml']}")
    return 1 if res.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
