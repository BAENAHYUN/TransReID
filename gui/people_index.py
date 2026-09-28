#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gui/people_index.py — '인물 분류' 페이지의 데이터: 클러스터 결과(assignments) + DB 페이로드 → 사람별·파일별 색인.

  assignments jsonl (point_id → cluster_id)  +  Qdrant 페이로드 (point_id → image_id/video, crop_path, source, score)
  ⇒ clusters: {cluster_id: {members, files{file: [pid]}, rep(대표 crop point), size}}
     files   : {file: {cluster_id: [pid]}}
파일 = 사진이면 image_id, 영상이면 video 이름. 폴더 = image_id 의 앞 경로(예 PRW, coco_train2017) 또는 'videos'.
이름은 <run>/person/person_names.json 에 {cluster_id: 이름} 으로 둔다 (Immich 의 사람 이름 붙이기).
색인은 <run>/person/people_index.json 에 캐시한다 (DB 조회 없이 다시 열림).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
NOISE = "__noise__"


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
            out.append({"run": run.name, "method": method, "assignments": a, "folder": folder, "mtime": mtime,
                        "cached": (folder / f"people_index_{method}.json").is_file()})
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
        url, collection = qdrant_settings()
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


def display_name(cluster_id: str, names: Dict[str, str]) -> str:
    return names.get(cluster_id) or short_id(cluster_id)


def resolve_crop(crop: str, root: Path = ROOT) -> Optional[Path]:
    if not crop:
        return None
    p = Path(crop)
    if not p.is_absolute():
        p = root / p
    return p if p.is_file() else None
