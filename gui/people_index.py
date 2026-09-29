#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gui/people_index.py — '인물 분류' 페이지의 데이터: 클러스터 결과(assignments) + DB 페이로드 → 사람별·파일별 색인.

  assignments jsonl (point_id → cluster_id)  +  Qdrant 페이로드 (point_id → image_id/video, crop_path, source, score)
  ⇒ clusters: {cluster_id: {members, files{file: [pid]}, rep(대표 crop point), size}}
     files   : {file: {cluster_id: [pid]}}
파일 = 사진이면 image_id, 영상이면 video 이름. 폴더 = image_id 의 앞 경로(예 PRW, coco_train2017) 또는 'videos'.
이름은 <run>/person/person_names.json 에 {cluster_id: 이름} 으로 둔다 (Immich 의 사람 이름 붙이기).
색인은 <run>/person/people_index.json 에 캐시한다 (DB 조회 없이 다시 열림).

자동 라벨: 같은 폴더의 labels_qwen*/ (clustering/label_clusters_qwen.py, '노란 반팔에 검은 바지') 와
labels_vec*/ (clustering/label_clusters_from_vectors.py, '노란색 상의') 의 cluster_labels.jsonl 을 읽어
이름이 없는 사람에게 붙인다. 우선순위: 확정(labeled) > 추정(tentative), 같은 등급이면 Qwen 문장 > 색상.
옛 label_leiden_clusters_siglip2.py 출력(labels/, status 없음)은 PRW 에서 퇴화한 라벨이라 읽지 않는다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
NOISE = "__noise__"

# 자동 라벨 종류: (폴더 이름 접두, 표시 이름, 우선순위(작을수록 먼저), 만드는 스크립트)
LABEL_KINDS: List[Tuple[str, str, int, str]] = [
    ("labels_qwen", "Qwen 문장", 0, "clustering/label_clusters_qwen.py"),
    ("labels_vec", "색상(SigLIP2)", 1, "clustering/label_clusters_from_vectors.py"),
]
LABEL_TOOLS = {"qwen": "labels_qwen", "vec": "labels_vec"}       # 자동 라벨 붙이기 버튼의 도구 키 → 폴더 접두
LABEL_OK_STATUS = ("labeled", "tentative", "class_only")   # class_only: 물건인데 색을 못 정해 종류만 (예: 자전거)
STATUS_RANK = {"labeled": 0, "tentative": 1, "class_only": 2}
TARGETS = {"person": "사람", "object": "물건"}


def find_runs(root: Path = ROOT, target: str = "person") -> List[Dict[str, Any]]:
    """outputs/clustering/*/<target>/<target>_*_assignments.jsonl → 최근순."""
    out: List[Dict[str, Any]] = []
    base = root / "outputs" / "clustering"
    if not base.is_dir():
        return out
    for run in sorted(base.iterdir()):
        folder = run / target
        if not folder.is_dir():
            continue
        for a in sorted(folder.glob(f"{target}_*_assignments.jsonl")):
            method = a.name[len(target) + 1:-len("_assignments.jsonl")]
            try:
                mtime = a.stat().st_mtime
            except OSError:
                continue
            out.append({"run": run.name, "method": method, "target": target, "assignments": a, "folder": folder, "mtime": mtime,
                        "cached": (folder / f"people_index_{method}.json").is_file(),
                        "labels": [s["dir"] for s in find_labels({"folder": folder})]})
    # 바로 열리는 것(캐시 있음)을 앞에, 그다음 최근순
    out.sort(key=lambda r: (not r["cached"], -r["mtime"]))
    return out


def read_assignments(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            pid = str(d.get("point_id") or "")
            if pid:
                out[pid] = d
    return out


def file_of(payload: Dict[str, Any]) -> Tuple[str, str]:
    """(파일 이름, 폴더). 사진은 image_id, 영상은 video 이름."""
    if payload.get("media_type") == "video" or payload.get("video"):
        return str(payload.get("video") or payload.get("video_name") or Path(str(payload.get("video_path") or "")).name or "?"), "videos"
    image_id = str(payload.get("image_id") or "")
    p = Path(image_id.replace("\\", "/"))
    folder = p.parent.as_posix() if p.parent.as_posix() not in ("", ".") else str(payload.get("source") or "")
    return image_id or "?", folder


def build_index(assignments: Dict[str, Dict[str, Any]], payloads: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    clusters: Dict[str, Dict[str, Any]] = {}
    files: Dict[str, Dict[str, Any]] = {}
    missing = 0
    for pid, a in assignments.items():
        pl = payloads.get(pid)
        if pl is None:
            missing += 1
            continue
        cid = NOISE if a.get("noise") or not a.get("cluster_id") else str(a["cluster_id"])
        fname, folder = file_of(pl)
        crop = str(pl.get("crop_path") or "")
        score = float(pl.get("score") or 0.0)
        c = clusters.setdefault(cid, {"members": [], "files": {}, "rep": None, "rep_score": -1.0, "folders": {}})
        c["members"].append(pid)
        c["files"].setdefault(fname, []).append(pid)
        c["folders"][folder] = c["folders"].get(folder, 0) + 1
        if crop and score > c["rep_score"]:
            c["rep"], c["rep_score"] = pid, score
        f = files.setdefault(fname, {"folder": folder, "clusters": {}})
        f["clusters"].setdefault(cid, []).append(pid)
    points = {pid: {"file": file_of(pl)[0], "folder": file_of(pl)[1], "crop": str(pl.get("crop_path") or ""), "score": float(pl.get("score") or 0.0),
                    "time": pl.get("time_mmss") or pl.get("timestamp_sec"), "track": pl.get("track_key")}
              for pid, pl in payloads.items() if pid in assignments}
    for c in clusters.values():
        c["size"] = len(c["members"])
        c["n_files"] = len(c["files"])
    return {"built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_points": len(points), "n_missing": missing,
            "clusters": clusters, "files": files, "points": points}


def fetch_payloads(collection: str, ids: Iterable[str], url: str, batch: int = 512,
                   progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Dict[str, Any]]:
    """Qdrant 에서 point 페이로드만 가져온다 (벡터 없음)."""
    from qdrant_client import QdrantClient

    client = QdrantClient(url=url, timeout=120)
    ids = list(ids)
    out: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        for p in client.retrieve(collection_name=collection, ids=chunk, with_payload=True, with_vectors=False):
            out[str(p.id)] = dict(p.payload or {})
        if progress:
            progress(min(len(ids), i + batch), len(ids))
    return out


def qdrant_settings(config: Optional[Path] = None, target: str = "person") -> Tuple[str, str]:
    """(qdrant url, collection) — pipeline.yaml 에서."""
    from report_common import load_pipeline_settings

    s = load_pipeline_settings(str(config) if config else None, require=True)
    return str(s.qdrant_url), str(s.collection_for(target))


def cache_path(run: Dict[str, Any]) -> Path:
    return run["folder"] / f"people_index_{run['method']}.json"


def load_or_build(run: Dict[str, Any], *, fetch: Optional[Callable[[Iterable[str], Callable[[int, int], None]], Dict[str, Dict[str, Any]]]] = None,
                  progress: Optional[Callable[[int, int], None]] = None, rebuild: bool = False) -> Dict[str, Any]:
    cp = cache_path(run)
    if cp.is_file() and not rebuild:
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except ValueError:
            pass
    assignments = read_assignments(run["assignments"])
    if fetch is None:
        url, collection = qdrant_settings(target=str(run.get("target") or "person"))
        payloads = fetch_payloads(collection, assignments.keys(), url, progress=progress or (lambda a, b: None))
    else:
        payloads = fetch(assignments.keys(), progress or (lambda a, b: None))
    index = build_index(assignments, payloads)
    cp.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    return index


def names_path(run: Dict[str, Any]) -> Path:
    return run["folder"] / "person_names.json"


def load_names(run: Dict[str, Any]) -> Dict[str, str]:
    p = names_path(run)
    if not p.is_file():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return {str(k): str(v) for k, v in (d or {}).items()} if isinstance(d, dict) else {}


def save_name(run: Dict[str, Any], cluster_id: str, name: str) -> Dict[str, str]:
    names = load_names(run)
    name = name.strip()
    if name:
        names[cluster_id] = name
    else:
        names.pop(cluster_id, None)
    names_path(run).write_text(json.dumps(names, ensure_ascii=False, indent=1), encoding="utf-8")
    return names


def short_id(cluster_id: str) -> str:
    """'leiden:person:b6e58eddc91a14f1' → '#b6e58edd'."""
    if cluster_id == NOISE:
        return "미분류"
    tail = cluster_id.split(":")[-1]
    return "#" + tail[:8]


def inherited_names(run: Dict[str, Any], clusters: Dict[str, Dict[str, Any]], root: Path = ROOT,
                    min_share: float = 0.5, runs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    """다른 결과에서 붙인 이름을 이 결과의 군집으로 옮긴다. 같은 DB 의 point id 는 결과가 달라도 같으므로
    예전 군집 A 와 지금 군집 B 가 서로 과반을 공유하면(|A∩B|/|A| ≥ min_share 이고 |A∩B|/|B| ≥ min_share) 같은 사람으로 본다.
    반환: {지금 cluster_id: {name, from_run, share}}. 이 결과에서 직접 붙인 이름이 있으면 그쪽이 우선이다(화면에서)."""
    member_of: Dict[str, str] = {}
    size: Dict[str, int] = {}
    for cid, c in clusters.items():
        if cid == NOISE:
            continue
        size[cid] = len(c.get("members", []))
        for pid in c.get("members", []):
            member_of[str(pid)] = cid
    if not member_of:
        return {}
    target = str(run.get("target") or Path(run["folder"]).name or "person")
    out: Dict[str, Dict[str, Any]] = {}
    for other in (runs if runs is not None else find_runs(root, target)):
        if str(other["assignments"]) == str(run["assignments"]):
            continue
        names = load_names(other)
        if not names:
            continue
        old_members: Dict[str, List[str]] = {}
        for pid, a in read_assignments(other["assignments"]).items():
            cid = str(a.get("cluster_id") or "")
            if cid in names and not a.get("noise"):
                old_members.setdefault(cid, []).append(pid)
        for old_cid, pids in old_members.items():
            hits: Dict[str, int] = {}
            for pid in pids:
                cur = member_of.get(pid)
                if cur:
                    hits[cur] = hits.get(cur, 0) + 1
            if not hits:
                continue
            cur, n = max(hits.items(), key=lambda kv: kv[1])
            share = min(n / len(pids), n / max(1, size[cur]))
            if share >= min_share and share > out.get(cur, {}).get("share", 0.0):
                out[cur] = {"name": names[old_cid], "from_run": other["run"], "share": round(share, 3)}
    return out


def last_run_path(root: Path = ROOT) -> Path:
    return Path(root) / "outputs" / "clustering" / "people_last_run.json"


def load_last_run(root: Path = ROOT, target: str = "person") -> Optional[str]:
    """마지막으로 불러온 결과의 assignments 경로 (대상별). 없으면 None."""
    try:
        d = json.loads(last_run_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    v = d.get(target) if isinstance(d, dict) else None
    return str(v) if v else None


def save_last_run(run: Dict[str, Any], root: Path = ROOT) -> None:
    p = last_run_path(root)
    try:
        d = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except (OSError, ValueError):
        d = {}
    if not isinstance(d, dict):
        d = {}
    d[str(run.get("target") or "person")] = str(run["assignments"])
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass


def run_title(run: Dict[str, Any]) -> str:
    """묶음 결과 콤보 항목: '실행 · 방법 · 시각 · 캐시 · 라벨: 문장·색상'."""
    from datetime import datetime

    when = datetime.fromtimestamp(run["mtime"]).strftime("%m-%d %H:%M")
    kinds = []
    dirs = run.get("labels") or []
    if any(d.startswith("labels_qwen") for d in dirs):
        kinds.append("문장")
    if any(d.startswith("labels_vec") for d in dirs):
        kinds.append("색상")
    return (f"{run['run']} · {run['method']} · {when}" + (" · 캐시" if run.get("cached") else "")
            + (f" · 라벨: {'·'.join(kinds)}" if kinds else " · 라벨 없음"))


def display_name(cluster_id: str, names: Dict[str, str], labels: Optional[Dict[str, Dict[str, Any]]] = None,
                 inherited: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    """이 결과에서 붙인 이름 > 다른 결과에서 이어받은 이름 > 자동 라벨 > #id."""
    if names.get(cluster_id):
        return names[cluster_id]
    if inherited and inherited.get(cluster_id, {}).get("name"):
        return str(inherited[cluster_id]["name"])
    if labels:
        auto = labels.get(cluster_id)
        if auto and auto.get("name"):
            return str(auto["name"])
    return short_id(cluster_id)


# ---- 자동 라벨 (cluster_labels.jsonl) ----
def label_kind(dirname: str) -> Optional[Tuple[str, int]]:
    """폴더 이름 → (표시 이름, 우선순위). labels_qwen_sample4b 처럼 접미가 붙어도 같은 종류로 본다."""
    for prefix, title, priority, _script in LABEL_KINDS:
        if dirname == prefix or dirname.startswith(prefix + "_"):
            return title, priority
    return None


def find_labels(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    """<run folder>/labels_*/cluster_labels.jsonl 목록 — 우선순위, 기본 폴더 이름(접미 없음) 우선, 최근순."""
    folder = Path(run["folder"])
    out: List[Dict[str, Any]] = []
    if not folder.is_dir():
        return out
    for d in folder.iterdir():
        kind = label_kind(d.name) if d.is_dir() else None
        if kind is None:
            continue
        p = d / "cluster_labels.jsonl"
        if not p.is_file():
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        out.append({"dir": d.name, "path": p, "kind": kind[0], "priority": kind[1], "mtime": mtime,
                    "exact": d.name in {k[0] for k in LABEL_KINDS}})
    out.sort(key=lambda s: (s["priority"], not s["exact"], -s["mtime"]))
    return out


def read_labels(path: Path) -> Dict[str, Dict[str, Any]]:
    """cluster_labels.jsonl → {cluster_id: record}. 이름이 있고 status 가 labeled/tentative 인 줄만 (옛 형식은 status 가 없어 걸러진다)."""
    out: Dict[str, Dict[str, Any]] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, dict):
            continue
        cid, name = str(rec.get("cluster_id") or ""), str(rec.get("cluster_name") or "").strip()
        if cid and name and rec.get("status") in LABEL_OK_STATUS:
            out[cid] = rec
    return out


def load_labels(run: Dict[str, Any], clusters: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
    """cluster_id → {name, status, kind, dir, confidence, desc, others:[{name, kind, status}]}.
    여러 라벨 파일이 있으면 확정 > 추정, 같은 등급이면 Qwen 문장 > 색상 을 대표로 두고 나머지는 others 에 남긴다."""
    keep = set(clusters) if clusters is not None else None
    best: Dict[str, Dict[str, Any]] = {}
    for order, src in enumerate(find_labels(run)):
        for cid, rec in read_labels(src["path"]).items():
            if keep is not None and cid not in keep:
                continue
            status = str(rec.get("status"))
            entry = {"name": str(rec["cluster_name"]).strip(), "status": status, "kind": src["kind"], "dir": src["dir"],
                     "confidence": float(rec.get("label_confidence") or 0.0), "desc": str(rec.get("cluster_description") or "")}
            rank = (STATUS_RANK.get(status, 3), src["priority"], order)
            cur = best.get(cid)
            if cur is None:
                best[cid] = {**entry, "_rank": rank, "others": []}
            elif rank < cur["_rank"]:
                best[cid] = {**entry, "_rank": rank, "others": [_other(cur)] + cur["others"]}
            else:
                cur["others"].append(_other(entry))
    for e in best.values():
        e.pop("_rank", None)
        seen = set()
        uniq = []
        for o in e["others"]:
            key = (o["kind"], o["name"])
            if key not in seen and o["name"] != e["name"]:
                seen.add(key)
                uniq.append(o)
        e["others"] = uniq
    return best


def _other(e: Dict[str, Any]) -> Dict[str, Any]:
    return {k: e[k] for k in ("name", "kind", "status", "dir")}


def _kind_tag(e: Dict[str, Any]) -> str:
    """'Qwen 문장' — 기본 폴더가 아니면(labels_qwen_sample4b 등) 폴더 이름을 덧붙여 구분한다."""
    d = str(e.get("dir") or "")
    return e["kind"] if not d or d in {k[0] for k in LABEL_KINDS} else f"{e['kind']} · {d}"


def label_line(entry: Optional[Dict[str, Any]]) -> str:
    """상세/툴팁용 한 줄: '노란 반팔에 검은 바지 (Qwen 문장) · 노란색 상의 (색상(SigLIP2))'."""
    if not entry or not entry.get("name"):
        return ""
    parts = [f"{entry['name']} ({_kind_tag(entry)})"] + [f"{o['name']} ({_kind_tag(o)})" for o in entry.get("others", [])]
    return " · ".join(parts)


def label_output_dir(run: Dict[str, Any], tool: str) -> Path:
    """자동 라벨 출력 폴더: leiden 은 8/8b 단계와 같은 labels_vec / labels_qwen, 다른 방법은 labels_vec_<method> (같은 폴더의 두 방법이 서로 덮어쓰지 않게)."""
    prefix = LABEL_TOOLS[tool]
    method = str(run.get("method") or "")
    return Path(run["folder"]) / (prefix if method in ("", "leiden") else f"{prefix}_{method}")


def auto_label_command(run: Dict[str, Any], tool: str = "vec", python: Optional[str] = None, root: Path = ROOT) -> List[str]:
    """'자동 라벨 붙이기' 가 돌리는 명령. vec = 색상(SigLIP2 벡터, 빠름) / qwen = 문장(Qwen3-VL, GPU·느림)."""
    if tool not in LABEL_TOOLS:
        raise ValueError(f"unknown label tool: {tool}")
    script = next(s for p, _t, _pr, s in LABEL_KINDS if p == LABEL_TOOLS[tool])
    target = str(run.get("target") or Path(run["folder"]).name or "person")
    out_dir = label_output_dir(run, tool)
    cmd = [python or sys.executable, "-u", str(Path(root) / script),
           "--assignments", str(run["assignments"]), "--target", target, "--output-dir", str(out_dir)]
    # Qwen 은 군집당 1~2초라 다시 돌리면 수십 분 — 이미 결과가 있으면 빠진 군집만 채운다 (처음부터 다시 쓰면
    # 도중에 죽을 때 기존 라벨이 사라진다). 색상 라벨은 1~3분·결정적이라 그냥 다시 만든다.
    if tool == "qwen" and (out_dir / "cluster_labels.jsonl").is_file():
        cmd.append("--resume")
    return cmd


def resolve_crop(crop: str, root: Path = ROOT) -> Optional[Path]:
    if not crop:
        return None
    p = Path(crop)
    if not p.is_absolute():
        p = root / p
    return p if p.is_file() else None
