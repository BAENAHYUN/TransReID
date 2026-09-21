#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""클러스터 결과(assignments.jsonl)를 군집별 폴더로 내보낸다.

한 군집 = 한 폴더. 폴더 이름에 순위·크기·라벨을 붙이고 crop 파일을 복사(또는 하드링크)한다.
Leiden(cluster_leiden_qdrant.py) / DBSCAN(cluster_dbscan_qdrant.py) assignments 는 형식이 같다
(point_id / cluster_id / noise). crop 경로는 Qdrant payload(crop_path) 에서 읽는다.

라벨 출처 (둘 다 선택. 없으면 순위·크기·군집 id 만):
  * --gt-matches : eval/prw_cluster_gt_eval.py 의 gt_matches.jsonl (point_id → GT pid).
                   군집별 다수 pid · 순도(purity) · 섞인 pid 수를 폴더 이름에 붙이고, 파일 이름 앞에
                   pid 를 붙여 폴더 안에서 다른 인물이 바로 눈에 띄게 한다 (pidNA = GT 없음).
  * --labels     : label_leiden_clusters_siglip2.py 의 cluster_labels.jsonl/json (cluster_name).

--group-by gt-pid : 군집 대신 GT 인물(pid) 단위로 폴더를 만들고, 파일 이름 앞에 군집 순위(c0001_)를
                   붙인다. 한 인물이 몇 개 군집으로 갈라졌는지 폴더 하나에서 본다. --gt-matches 필수.

폴더 이름 예: c0001_n652_pid0123_pur64_mix9_black-jacket_15778620
  c0001 순위(크기 내림차순) · n652 point 수 · pid0123 GT 다수 인물 · pur64 순도 % · mix9 섞인 pid 수
  · (라벨 slug) · 군집 id 앞 8자.   특수 폴더: _small_n.. (최소 크기 미만) / _noise_n.. / _no_gt_n..

산출물: <output-dir>/<폴더들>/*.jpg, manifest.csv, index.html, export_report.json(사이드카)
마지막에 RESULT_SUMMARY / RESULT_HTML 마커를 한 번씩 출력한다 (GUI '결과 열기' 계약).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

from build_leiden_gallery import (CROP_KEYS, QdrantHTTP, first_payload, load_assignments, resolve_path,
                                  split_groups)
from report_common import (CONFIG_HELP, DEFAULT_CONFIG_PATH, applied_config, common_summary, esc, file_info,
                           is_noise, load_pipeline_settings, path_href, resolve, result_markers, table, warning,
                           write_json)

PRODUCER = "export_cluster_folders.py"
REPORT_NAME = "export_report.json"
MANIFEST_NAME = "manifest.csv"
INDEX_NAME = "index.html"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
WINDOWS_MAX_PATH = 259           # 기본 설정의 Windows 에서 이보다 긴 경로는 열리지 않는다
MANIFEST_FIELDS = ("folder", "file", "point_id", "cluster_id", "cluster_rank", "noise", "gt_pid", "label",
                   "src", "action")
PLACED_ACTIONS = ("copied", "linked", "copied_fallback", "kept", "planned")   # 파일 이름이 정해진 action
KIND_LABEL = {"cluster": "군집", "small": "작은 군집 모음", "noise": "noise", "pid": "GT 인물", "no_gt": "GT 없음"}


# ----------------------------------------------------------------------------------------------------
# 이름 조각
# ----------------------------------------------------------------------------------------------------
def slugify(text: Any, max_len: int = 40) -> str:
    """폴더/파일 이름에 쓸 조각. 글자·숫자(한글 포함)만 남기고 나머지는 '-' 로 잇는다."""
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text)).lower()
    value = re.sub(r"[^\w]+", "-", value).strip("-_")
    if len(value) > max_len:
        value = value[:max_len].rstrip("-_")
    return value


def short_id(cluster_id: Any, length: int = 8) -> str:
    """'leiden:person:15778620d6e6813e' → '15778620'. 마지막 ':' 뒤 조각의 앞부분."""
    tail = str(cluster_id).rsplit(":", 1)[-1]
    return slugify(tail, length) or "x"


def pid_token(pid: Any) -> str:
    """GT pid 를 이름 조각으로. 정수는 4자리 0 채움(정렬용), 그 외는 slug."""
    try:
        number = int(str(pid).strip())
    except (TypeError, ValueError):
        return slugify(pid, 12) or "x"
    return f"{number:04d}" if number >= 0 else f"m{-number:04d}"


def gt_prefix(pid: Any) -> str:
    return f"pid{pid_token(pid)}_" if pid is not None else "pidNA_"


def folder_name(rank: int, size: int, cluster_id: Any, gt: Optional[Dict[str, Any]] = None,
                label: Optional[str] = None, max_len: int = 64) -> str:
    """c0001_n652[_pid0123_pur64[_mix9]|_gtNA][_label-slug]_<id8>. 라벨 slug 만 잘라서 max_len 을 지킨다."""
    parts = [f"c{rank:04d}", f"n{size}"]
    if gt is not None:
        if gt.get("top_pid") is None:
            parts.append("gtNA")
        else:
            parts.append(f"pid{pid_token(gt['top_pid'])}")
            parts.append(f"pur{round(gt['purity'] * 100)}")
            if gt.get("distinct", 0) > 1:
                parts.append(f"mix{gt['distinct']}")
    tail = short_id(cluster_id)
    base = "_".join(parts)
    budget = max_len - len(base) - len(tail) - 2
    slug = slugify(label, budget) if label and budget >= 4 else ""
    return "_".join([base] + ([slug] if slug else []) + [tail])


def unique_name(used: set, name: str, point_id: Any) -> str:
    """같은 폴더 안에서 파일 이름이 겹치면 point_id 앞 8자를 덧붙인다."""
    if name not in used:
        used.add(name)
        return name
    stem, suffix = os.path.splitext(name)
    marker = slugify(point_id, 8) or "dup"
    candidate = f"{stem}_{marker}{suffix}"
    counter = 2
    while candidate in used:
        candidate = f"{stem}_{marker}-{counter}{suffix}"
        counter += 1
    used.add(candidate)
    return candidate


# ----------------------------------------------------------------------------------------------------
# 라벨 입력
# ----------------------------------------------------------------------------------------------------
def load_labels(path: Path) -> Dict[str, Dict[str, Any]]:
    """cluster_labels.jsonl(.json) → {cluster_id: {name, confidence, description}}. cluster_name 없는 행은 건너뛴다."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        records = json.loads(text)
        if not isinstance(records, list):
            raise ValueError(f"labels json 은 record 배열이어야 한다: {path}")
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    out: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        cid, name = record.get("cluster_id"), record.get("cluster_name")
        if cid is None or not str(name or "").strip():
            continue
        out[str(cid)] = dict(name=str(name).strip(), confidence=record.get("label_confidence"),
                             description=record.get("cluster_description"))
    return out


def load_gt_matches(path: Path) -> Dict[str, Any]:
    """gt_matches.jsonl → {point_id: pid}. _meta 행, pid 없는 행(unmatched), status 가 'labeled' 가 아닌 행
    (unlabeled: PRW pid -2 = 미표기 보행자), 음수 pid 는 제외한다."""
    out: Dict[str, Any] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or record.get("_meta"):
                continue
            pid = record.get("pid")
            if record.get("point_id") is None or pid is None:
                continue
            if record.get("status") is not None and record.get("status") != "labeled":
                continue
            if isinstance(pid, (int, float)) and not isinstance(pid, bool) and pid < 0:
                continue
            out[str(record["point_id"])] = pid
    return out


def gt_summary(members: Sequence[Dict[str, Any]], gt: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """군집 안 GT 가 붙은 point 의 다수 pid 와 순도. GT 입력이 없으면 None."""
    if gt is None:
        return None
    counts = Counter(gt[str(m["point_id"])] for m in members if str(m["point_id"]) in gt)
    if not counts:
        return dict(labeled=0, distinct=0, top_pid=None, top_count=0, purity=None)
    pid, top = counts.most_common(1)[0]
    labeled = sum(counts.values())
    return dict(labeled=labeled, distinct=len(counts), top_pid=pid, top_count=top, purity=top / labeled)


# ----------------------------------------------------------------------------------------------------
# 폴더 계획 (순수 함수 — 파일을 만들지 않는다)
# ----------------------------------------------------------------------------------------------------
def member_items(members: Sequence[Dict[str, Any]], gt: Optional[Dict[str, Any]],
                 extra_prefix: str = "") -> List[Tuple[Dict[str, Any], str]]:
    items = []
    for row in members:
        prefix = extra_prefix
        if gt is not None:
            prefix += gt_prefix(gt.get(str(row["point_id"])))
        items.append((row, prefix))
    items.sort(key=lambda item: (item[1], str(item[0]["point_id"])))
    return items


def cap_members(items: List[Tuple[Dict[str, Any], str]], max_per_folder: int, rng: random.Random):
    if max_per_folder > 0 and len(items) > max_per_folder:
        sampled = rng.sample(items, max_per_folder)
        sampled.sort(key=lambda item: (item[1], str(item[0]["point_id"])))
        return sampled, True
    return items, False


def special_folder(name: str, kind: str, items, gt, max_per_folder, rng) -> Dict[str, Any]:
    folder = dict(name=name, kind=kind, cluster_id=None, rank=None, size=len(items),
                  gt=gt_summary([row for row, _ in items], gt), label=None)
    folder["members"], folder["sampled"] = cap_members(items, max_per_folder, rng)
    return folder


def plan_clusters(rows, gt, labels, min_cluster_size: int, max_per_folder: int, seed: int, max_name_len: int):
    """군집 단위 폴더. 최소 크기 미만 군집은 _small 로, noise 는 _noise 로 모은다."""
    groups, noise = split_groups(rows)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    rng = random.Random(seed)
    folders: List[Dict[str, Any]] = []
    ranks: Dict[str, int] = {}
    small_items: List[Tuple[Dict[str, Any], str]] = []
    for cid, members in ordered:
        if len(members) < min_cluster_size:
            small_items.extend(member_items(members, gt, f"{short_id(cid)}_"))
            continue
        rank = len(ranks) + 1
        ranks[cid] = rank
        summary = gt_summary(members, gt)
        label = (labels or {}).get(cid)
        folder = dict(name=folder_name(rank, len(members), cid, summary, label["name"] if label else None,
                                       max_name_len),
                      kind="cluster", cluster_id=cid, rank=rank, size=len(members), gt=summary, label=label)
        folder["members"], folder["sampled"] = cap_members(member_items(members, gt), max_per_folder, rng)
        folders.append(folder)
    if small_items:
        small_items.sort(key=lambda item: (item[1], str(item[0]["point_id"])))
        folders.append(special_folder(f"_small_n{len(small_items)}", "small", small_items, gt, max_per_folder, rng))
    if noise:
        folders.append(special_folder(f"_noise_n{len(noise)}", "noise", member_items(noise, gt), gt,
                                      max_per_folder, rng))
    return folders, ranks


def plan_by_pid(rows, gt: Dict[str, Any], max_per_folder: int, seed: int):
    """GT 인물 단위 폴더. 파일 앞에 군집 순위(c0001_) 또는 noise_ 를 붙인다. GT 없는 point 는 _no_gt 로."""
    groups, _ = split_groups(rows)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ranks = {cid: index for index, (cid, _) in enumerate(ordered, 1)}
    rng = random.Random(seed)
    by_pid: Dict[Any, List[Tuple[Dict[str, Any], str]]] = defaultdict(list)
    no_gt: List[Tuple[Dict[str, Any], str]] = []
    for row in rows:
        prefix = "noise_" if is_noise(row) else f"c{ranks[str(row.get('cluster_id'))]:04d}_"
        pid = gt.get(str(row["point_id"]))
        (no_gt if pid is None else by_pid[pid]).append((row, prefix))
    folders: List[Dict[str, Any]] = []
    for pid, items in sorted(by_pid.items(), key=lambda kv: (-len(kv[1]), pid_token(kv[0]))):
        items.sort(key=lambda item: (item[1], str(item[0]["point_id"])))
        clusters = {str(row.get("cluster_id")) for row, _ in items if not is_noise(row)}
        noise_points = sum(1 for row, _ in items if is_noise(row))
        folder = dict(name=f"pid{pid_token(pid)}_n{len(items)}_k{len(clusters)}", kind="pid", cluster_id=None,
                      rank=None, size=len(items), gt=dict(labeled=len(items), distinct=1, top_pid=pid,
                                                          top_count=len(items), purity=1.0),
                      label=None, pid=pid, clusters=len(clusters), noise_points=noise_points)
        folder["members"], folder["sampled"] = cap_members(items, max_per_folder, rng)
        folders.append(folder)
    if no_gt:
        no_gt.sort(key=lambda item: (item[1], str(item[0]["point_id"])))
        folders.append(special_folder(f"_no_gt_n{len(no_gt)}", "no_gt", no_gt, gt, max_per_folder, rng))
    return folders, ranks


# ----------------------------------------------------------------------------------------------------
# 파일 배치
# ----------------------------------------------------------------------------------------------------
def place_one(src: Path, dst: Path, mode: str) -> str:
    """이미 같은 크기·mtime 이면 'kept'. hardlink 실패(다른 드라이브 등)는 복사로 대체한다."""
    if dst.exists():
        s, d = src.stat(), dst.stat()
        if d.st_size == s.st_size and d.st_mtime_ns == s.st_mtime_ns:
            return "kept"
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "linked"
        except OSError:
            shutil.copy2(src, dst)
            return "copied_fallback"
    shutil.copy2(src, dst)
    return "copied"


def place_files(folders, payloads, project_root: Path, out_dir: Path, mode: str, ranks, gt, labels,
                dry_run: bool = False, progress_every: int = 100):
    """폴더마다 파일을 놓는다. dry_run 이면 폴더·파일을 만들지 않고 이름만 정한다 (action=planned)."""
    counts: Counter = Counter()
    manifest: List[Dict[str, Any]] = []
    for index, folder in enumerate(folders, 1):
        folder_dir = out_dir / folder["name"]
        if not dry_run:
            folder_dir.mkdir(parents=True, exist_ok=True)
        used: set = set()
        folder_counts: Counter = Counter()
        files: List[Dict[str, Any]] = []
        for row, prefix in folder["members"]:
            pid = str(row["point_id"])
            cid = row.get("cluster_id")
            payload = payloads.get(pid, {})
            raw = first_payload(payload, CROP_KEYS)
            src = resolve_path(raw, project_root)
            label = (labels or {}).get(str(cid)) if cid is not None else None
            entry = dict(folder=folder["name"], file="", point_id=pid, cluster_id="" if cid is None else cid,
                         cluster_rank=ranks.get(str(cid), "") if cid is not None else "",
                         noise=int(is_noise(row)), gt_pid="" if gt is None or gt.get(pid) is None else gt[pid],
                         label=label["name"] if label else "", src=raw or "", action="missing")
            if src is not None:
                suffix = src.suffix.lower() if src.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
                name = unique_name(used, f"{prefix}{src.stem}{suffix}", pid)
                dst = folder_dir / name
                entry["file"] = name
                entry["action"] = "planned" if dry_run else place_one(src, dst, mode)
                if len(str(dst)) > WINDOWS_MAX_PATH:
                    counts["long_path"] += 1
                files.append(dict(name=name, src=src, point_id=pid))
            counts[entry["action"]] += 1
            folder_counts[entry["action"]] += 1
            manifest.append(entry)
        folder["counts"] = dict(folder_counts)
        folder["files"] = files
        if progress_every and (index % progress_every == 0 or index == len(folders)):
            done = sum(counts[k] for k in PLACED_ACTIONS)
            print(f"[{index:,}/{len(folders):,}] 폴더 처리 · 파일 {done:,} · 누락 {counts['missing']:,}")
    return manifest, counts


def stale_folders(out_dir: Path, planned: set) -> List[str]:
    if not out_dir.is_dir():
        return []
    return sorted(p.name for p in out_dir.iterdir() if p.is_dir() and p.name not in planned)


def clean_previous(out_dir: Path, warnings: List[Dict[str, Any]]) -> int:
    """이전 export_report.json 에 기록된 폴더만 지운다. 기록이 없으면 아무것도 지우지 않는다."""
    report_path = out_dir / REPORT_NAME
    if not report_path.is_file():
        warnings.append(warning("CLEAN_NO_REPORT", f"--clean: 이전 {REPORT_NAME} 이 없어 지운 폴더 없음"))
        return 0
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        warnings.append(warning("CLEAN_BAD_REPORT", f"--clean: 이전 {REPORT_NAME} 을 읽지 못함 ({exc})"))
        return 0
    removed = 0
    root = out_dir.resolve()
    for folder in (data.get("folders") or []) if isinstance(data, dict) else []:
        name = folder.get("name") if isinstance(folder, dict) else None
        if not name or "/" in name or "\\" in name or name in (".", ".."):
            continue
        target = (out_dir / name)
        if target.is_dir() and target.resolve().parent == root:
            shutil.rmtree(target)
            removed += 1
    return removed


# ----------------------------------------------------------------------------------------------------
# 출력 파일
# ----------------------------------------------------------------------------------------------------
def write_manifest(path: Path, manifest: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for entry in manifest:
            writer.writerow({key: entry.get(key, "") for key in MANIFEST_FIELDS})


def percent(value: Optional[float]) -> str:
    return "" if value is None else f"{value * 100:.0f}%"


def file_url(folder: Dict[str, Any], entry: Dict[str, Any], index_path: Path, dry_run: bool) -> str:
    """index.html 에서 파일로 가는 링크. dry-run 은 원본 crop, 실제 실행은 폴더 안 복사본."""
    if dry_run:
        return path_href(entry["src"], index_path)
    return quote(folder["name"], safe="") + "/" + quote(entry["name"], safe="")


def folder_row_html(folder: Dict[str, Any], group_by: str, index_path: Path, dry_run: bool) -> str:
    name = folder["name"]
    href = quote(name, safe="") + "/"
    counts = folder.get("counts") or {}
    placed = sum(counts.get(k, 0) for k in PLACED_ACTIONS)
    gt = folder.get("gt")
    label = folder.get("label") or {}
    files = folder.get("files") or []
    thumb = ""
    if files:
        url = file_url(folder, files[0], index_path, dry_run)
        thumb = f'<a href="{url}"><img loading="lazy" src="{url}" alt=""></a>'
    folder_cell = f'<code>{esc(name)}</code>' if dry_run else f'<a href="{href}">{esc(name)}</a>'
    rank = folder["rank"] if folder.get("rank") is not None else KIND_LABEL.get(folder["kind"], folder["kind"])
    if group_by == "gt-pid":
        is_pid = folder["kind"] == "pid"
        extra = (f'<td>{esc(folder.get("clusters")) if is_pid else ""}</td>'
                 f'<td>{esc(folder.get("noise_points")) if is_pid else ""}</td>')
    else:
        top_pid = "" if not gt or gt.get("top_pid") is None else esc(gt["top_pid"])
        purity = "" if not gt else percent(gt.get("purity"))
        distinct = "" if not gt else esc(gt.get("distinct"))
        label_name = esc(label.get("name")) if label else ""
        confidence = label.get("confidence") if label else None
        confidence_text = "" if confidence is None else f"{confidence:.2f}"
        extra = (f'<td>{top_pid}</td><td>{purity}</td><td>{distinct}</td>'
                 f'<td>{label_name}</td><td>{confidence_text}</td>')
    sampled = " <span class=\"muted\">(표본)</span>" if folder.get("sampled") else ""
    return (f'<tr><td>{esc(rank)}</td><td>{folder_cell}{sampled}</td>'
            f'<td>{esc(KIND_LABEL.get(folder["kind"], folder["kind"]))}</td><td>{folder["size"]:,}</td>'
            f'<td>{placed:,}</td><td>{counts.get("missing", 0):,}</td>{extra}<td class="thumb">{thumb}</td></tr>')


def folder_files_html(folder: Dict[str, Any], index_path: Path, dry_run: bool, list_limit: int) -> str:
    """폴더 하나의 파일 이름 목록 (접이식). list_limit 개까지만 보이고 나머지는 개수로 적는다 (0 = 전부)."""
    files = folder.get("files") or []
    shown = files if list_limit <= 0 else files[:list_limit]
    items = "".join(f'<li><a href="{file_url(folder, entry, index_path, dry_run)}" title="point {esc(entry["point_id"])}">'
                    f'{esc(entry["name"])}</a></li>' for entry in shown)
    more = (f'<li class="muted">… 외 {len(files) - len(shown):,}개 (manifest.csv 참조)</li>'
            if len(files) > len(shown) else "")
    missing = (folder.get("counts") or {}).get("missing", 0)
    miss = f" · 누락 {missing:,}" if missing else ""
    return (f'<details><summary><code>{esc(folder["name"])}</code> · 파일 {len(files):,}{miss}</summary>'
            f'<ol>{items}{more}</ol></details>')


def write_index(path: Path, title: str, summary: Dict[str, Any], folders: List[Dict[str, Any]],
                group_by: str, warnings: List[Dict[str, Any]], dry_run: bool = False, list_limit: int = 200) -> None:
    if group_by == "gt-pid":
        heads = "<th>군집 수</th><th>noise point</th>"
        legend = ("폴더: <code>pid0123_n90_k3</code> = GT 인물 0123 · point 90 · 군집 3개로 갈라짐. "
                  "파일 앞 <code>c0001_</code> 는 그 crop 이 속한 군집 순위, <code>noise_</code> 는 노이즈.")
    else:
        heads = "<th>GT 다수 pid</th><th>순도</th><th>pid 수</th><th>라벨</th><th>신뢰도</th>"
        legend = ("폴더: <code>c0001_n652_pid0123_pur64_mix9_라벨_15778620</code> = 순위 · point 수 · GT 다수 인물 "
                  "· 순도 % · 섞인 pid 수 · 라벨 slug · 군집 id 앞 8자. 파일 앞 <code>pid0123_</code> 는 그 crop 의 "
                  "GT 인물, <code>pidNA_</code> 는 GT 없음. <code>_small</code> 은 최소 크기 미만 군집 모음(파일 앞에 "
                  "군집 id), <code>_noise</code> 는 노이즈.")
    if dry_run:
        legend += (" <b>미리보기(dry-run)</b>: 폴더·파일을 만들지 않았다. 아래 파일 이름은 실제 실행 때 붙을 이름이고, "
                   "링크와 썸네일은 원본 crop 을 가리킨다.")
    warn_html = ""
    if warnings:
        items = "".join(f"<li><b>{esc(w.get('code'))}</b> {esc(w.get('message'))}</li>" for w in warnings)
        warn_html = f'<div class="warn"><b>경고 {len(warnings)}</b><ul>{items}</ul></div>'
    rows = "".join(folder_row_html(folder, group_by, path, dry_run) for folder in folders)
    file_lists = "".join(folder_files_html(folder, path, dry_run, list_limit) for folder in folders)
    limit_note = "전부" if list_limit <= 0 else f"폴더당 앞 {list_limit:,}개"
    html_text = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>{esc(title)}</title>
<style>
body{{font-family:'Malgun Gothic',Segoe UI,sans-serif;margin:20px;color:#222;background:#fafafa}}
h1{{font-size:20px;margin:0 0 8px}} .muted{{color:#777;font-size:12px}}
table{{border-collapse:collapse;background:#fff}} th,td{{border:1px solid #ddd;padding:4px 8px;font-size:13px;vertical-align:top}}
th{{background:#f0f0f0;text-align:left}} td.thumb img{{height:72px;display:block}}
.warn{{background:#fff4e0;border:1px solid #f0c060;padding:8px 12px;margin:12px 0}} .warn ul{{margin:4px 0 0 18px}}
code{{background:#eee;padding:0 3px}} .legend{{margin:10px 0 16px;font-size:13px}}
h2{{font-size:16px;margin:24px 0 8px}} details{{margin:2px 0}} summary{{cursor:pointer;font-size:13px}}
details ol{{columns:3;font-size:12px;margin:4px 0 8px;padding-left:2.2em}} details li{{break-inside:avoid}}
</style></head><body>
<h1>{esc(title)}</h1>
<p class="muted">생성 {esc(summary.get('generated_at'))} · {esc(PRODUCER)}</p>
{table({k: summary[k] for k in ('대상', '컬렉션', 'assignments', '폴더 단위', '실행 방식', '출력 폴더', 'point 수',
                                  '폴더 수', '파일 처리', '라벨 출처', 'GT 출처', '이전 잔재 폴더') if k in summary})}
<p class="legend">{legend}</p>
{warn_html}
<table><tr><th>순위</th><th>폴더</th><th>종류</th><th>point</th><th>파일</th><th>누락</th>{heads}<th>미리보기</th></tr>
{rows}</table>
<h2>폴더별 파일 이름 <span class="muted">({limit_note} · 파일 {sum(len(f.get('files') or []) for f in folders):,}개)</span></h2>
{file_lists}
</body></html>
"""
    path.write_text(html_text, encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="클러스터 assignments 를 군집별 폴더로 내보낸다 (라벨 포함)")
    p.add_argument("--assignments", default="outputs/clustering/leiden_image_prw/person/person_leiden_assignments.jsonl")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=CONFIG_HELP)
    p.add_argument("--target", choices=["person", "object"], default=None)
    p.add_argument("--collection", default=None, help=CONFIG_HELP)
    p.add_argument("--qdrant-url", default=None, help=CONFIG_HELP)
    p.add_argument("--api-key", default=None)
    p.add_argument("--project-root", default=".")
    p.add_argument("--output-dir", default=None, help="기본: assignments 옆 folders/")
    p.add_argument("--gt-matches", default=None, help="eval/prw_cluster_gt_eval.py 의 gt_matches.jsonl (point_id→pid)")
    p.add_argument("--labels", default=None, help="label_leiden_clusters_siglip2.py 의 cluster_labels.jsonl/json")
    p.add_argument("--group-by", choices=["cluster", "gt-pid"], default="cluster")
    p.add_argument("--mode", choices=["copy", "hardlink"], default="copy")
    p.add_argument("--min-cluster-size", type=int, default=1, help="미만 군집은 _small 폴더로 (group-by cluster)")
    p.add_argument("--noise", choices=["folder", "skip"], default="folder")
    p.add_argument("--max-per-folder", type=int, default=0, help="0 이면 전부. 표본만 볼 때 사용")
    p.add_argument("--max-name-len", type=int, default=64, help="폴더 이름 길이 상한 (라벨 slug 를 잘라 맞춘다)")
    p.add_argument("--clean", action="store_true", help="이전 export_report.json 에 기록된 폴더를 먼저 지운다")
    p.add_argument("--dry-run", action="store_true",
                   help="폴더·파일을 만들지 않고 폴더 이름·파일 이름 계획만 index.html / manifest.csv 로 낸다 (링크는 원본 crop)")
    p.add_argument("--list-limit", type=int, default=200, help="index.html 에 폴더마다 보여줄 파일 이름 수 (0 = 전부)")
    p.add_argument("--title", default=None)
    p.add_argument("--qdrant-batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p


def parse_args(argv=None) -> argparse.Namespace:
    p = build_parser()
    args = p.parse_args(argv)
    for name, minimum in (("min_cluster_size", 1), ("max_per_folder", 0), ("max_name_len", 24), ("qdrant_batch_size", 1),
                          ("list_limit", 0)):
        if getattr(args, name) < minimum:
            p.error(f"--{name.replace('_', '-')} must be >= {minimum}")
    if args.group_by == "gt-pid" and not args.gt_matches:
        p.error("--group-by gt-pid 는 --gt-matches 가 필요하다")
    return args


def infer_target(assignments: str) -> Optional[str]:
    return next((v for v in ("person", "object") if Path(assignments).name.startswith(v + "_")), None)


def main(argv=None) -> int:
    args = parse_args(argv)
    target = args.target or infer_target(args.assignments)
    if args.collection in (None, "") and target is None:
        build_parser().error("--target 또는 --collection 지정")
    try:
        settings = load_pipeline_settings(args.config, require=not (
            args.collection not in (None, "") and args.qdrant_url not in (None, "")))
        args.collection = resolve(args.collection, settings.collection_for(target)
                                  if settings and target else None, "collection")
        args.qdrant_url = resolve(args.qdrant_url, settings.qdrant_url if settings else None, "qdrant_url")
    except (OSError, ValueError) as exc:
        build_parser().error(str(exc))

    assignments_path = Path(args.assignments).resolve()
    project_root = Path(args.project_root).resolve()
    out_dir = (Path(args.output_dir) if args.output_dir else assignments_path.parent / "folders").resolve()
    index_path = out_dir / INDEX_NAME
    report_path = out_dir / REPORT_NAME
    warnings: List[Dict[str, Any]] = []
    inputs = [file_info("assignments", assignments_path)]
    if settings:
        inputs.insert(0, file_info("config", settings.config_path))
    else:
        warnings.append(warning("CONFIG_UNAVAILABLE", f"config 없음: {args.config}; 명시 CLI 사용"))

    parse_errors = dict(count=0, first_line=None)
    rows = load_assignments(assignments_path, parse_errors)
    if parse_errors["count"]:
        warnings.append(warning("PARSE_ERRORS", f"깨진 assignments 줄: {parse_errors}", target))
    if not rows:
        warnings.append(warning("NO_ASSIGNMENTS", f"유효한 assignments 없음: {assignments_path}", target))
    if args.noise == "skip":
        rows = [row for row in rows if not is_noise(row)]

    gt: Optional[Dict[str, Any]] = None
    if args.gt_matches:
        gt_path = Path(args.gt_matches).resolve()
        if gt_path.is_file():
            gt = load_gt_matches(gt_path)
            inputs.append(file_info("gt_matches", gt_path))
            if not gt:
                warnings.append(warning("GT_EMPTY", f"GT 매칭이 비어 있음: {gt_path}", target))
        elif args.group_by == "gt-pid":
            build_parser().error(f"--gt-matches 파일 없음: {gt_path}")
        else:
            warnings.append(warning("GT_MISSING", f"GT 매칭 파일 없음 — GT 라벨 없이 진행: {gt_path}", target))
    labels: Optional[Dict[str, Dict[str, Any]]] = None
    if args.labels:
        labels_path = Path(args.labels).resolve()
        if labels_path.is_file():
            try:
                labels = load_labels(labels_path)
                inputs.append(file_info("labels", labels_path))
            except (OSError, ValueError) as exc:
                warnings.append(warning("LABELS_UNREADABLE", f"라벨 파일을 읽지 못함 ({exc}): {labels_path}", target))
        else:
            warnings.append(warning("LABELS_MISSING", f"라벨 파일 없음 — 라벨 없이 진행: {labels_path}", target))

    if args.group_by == "gt-pid":
        if args.min_cluster_size > 1:
            warnings.append(warning("MIN_SIZE_IGNORED", "--group-by gt-pid 에서는 --min-cluster-size 를 쓰지 않는다"))
        folders, ranks = plan_by_pid(rows, gt or {}, args.max_per_folder, args.seed)
    else:
        folders, ranks = plan_clusters(rows, gt, labels, args.min_cluster_size, args.max_per_folder, args.seed,
                                       args.max_name_len)
    if labels is not None:
        matched = sum(1 for cid in ranks if cid in labels)
        if labels and not matched:
            warnings.append(warning("LABELS_UNMATCHED", "라벨의 cluster_id 가 assignments 와 하나도 맞지 않음 — 다른 실행 결과?",
                                    target))
    if gt is not None:
        covered = sum(1 for row in rows if str(row["point_id"]) in gt)
        if rows and not covered:
            warnings.append(warning("GT_UNMATCHED", "GT 매칭의 point_id 가 assignments 와 하나도 맞지 않음", target))

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean and args.dry_run:
        warnings.append(warning("CLEAN_IGNORED", "--dry-run 에서는 --clean 을 적용하지 않는다 (아무것도 지우지 않음)"))
    removed = clean_previous(out_dir, warnings) if args.clean and not args.dry_run else 0
    stale = stale_folders(out_dir, {folder["name"] for folder in folders})
    if stale:
        warnings.append(warning("STALE_FOLDERS", f"이번 계획에 없는 하위 폴더 {len(stale)}개 (이전 실행 잔재?). "
                                                 f"--clean 은 이전 {REPORT_NAME} 에 기록된 폴더를 지운다", target))

    ids: List[Any] = []
    seen: set = set()
    for folder in folders:
        for row, _ in folder["members"]:
            if str(row["point_id"]) not in seen:
                seen.add(str(row["point_id"]))
                ids.append(row["point_id"])

    print("=" * 88)
    print("CLUSTER FOLDER EXPORT")
    print("=" * 88)
    print(f"assignments   : {assignments_path}")
    print(f"rows          : {len(rows):,}   folders: {len(folders):,}   group-by: {args.group_by}   mode: {args.mode}")
    print(f"labels        : {'없음' if labels is None else f'{len(labels):,} 군집'}   GT: "
          f"{'없음' if gt is None else f'{len(gt):,} point'}")
    print(f"output        : {out_dir}")
    print(f"Qdrant fetch  : {len(ids):,} ids from {args.collection}")

    payloads = QdrantHTTP(args.qdrant_url, args.api_key).retrieve_points(args.collection, ids, args.qdrant_batch_size) \
        if ids else {}
    if len(payloads) < len(ids):
        warnings.append(warning("PAYLOAD_MISSING", f"Qdrant 에 없는 point {len(ids) - len(payloads):,}개 — crop 경로를 모른다",
                                target))
    manifest, counts = place_files(folders, payloads, project_root, out_dir, args.mode, ranks, gt, labels,
                                   dry_run=args.dry_run)
    if counts["missing"]:
        warnings.append(warning("MISSING_CROPS", f"crop 파일을 찾지 못한 point {counts['missing']:,}개 (manifest action=missing)",
                                target))
    if counts["copied_fallback"]:
        warnings.append(warning("HARDLINK_FALLBACK", f"hardlink 실패로 복사한 파일 {counts['copied_fallback']:,}개", target))
    if counts["long_path"]:
        warnings.append(warning("LONG_PATH", f"경로 길이 {WINDOWS_MAX_PATH} 초과 파일 {counts['long_path']:,}개 — "
                                             "출력 폴더를 더 짧은 위치로", target))

    write_manifest(out_dir / MANIFEST_NAME, manifest)
    placed = {k: counts[k] for k in PLACED_ACTIONS + ("missing",) if counts[k]}
    title = args.title or (f"군집 폴더 내보내기 · {target or args.collection} · {args.group_by}"
                           + (" (미리보기)" if args.dry_run else ""))
    summary = common_summary(PRODUCER, index_path, inputs, warnings)
    summary.update({
        "target": target, "collection": args.collection, "assignments_path": str(assignments_path),
        "group_by": args.group_by, "mode": args.mode, "output_dir": str(out_dir), "points": len(rows),
        "folder_count": len(folders), "counts": dict(counts), "removed_folders": removed, "stale_folders": stale,
        "label_source": str(Path(args.labels).resolve()) if labels is not None else None,
        "gt_source": str(Path(args.gt_matches).resolve()) if gt is not None else None,
        "min_cluster_size": args.min_cluster_size, "noise": args.noise, "max_per_folder": args.max_per_folder,
        "dry_run": args.dry_run, "list_limit": args.list_limit, "manifest": str(out_dir / MANIFEST_NAME),
        "config": applied_config(settings, collection=args.collection, qdrant_url=args.qdrant_url, target=target),
        "folders": [dict(name=f["name"], kind=f["kind"], cluster_id=f["cluster_id"], rank=f["rank"], size=f["size"],
                         sampled=f.get("sampled", False), counts=f.get("counts", {}), gt=f.get("gt"),
                         label=(f["label"] or {}).get("name") if f.get("label") else None,
                         pid=f.get("pid"), clusters=f.get("clusters"), noise_points=f.get("noise_points"))
                    for f in folders],
        # index.html 표 머리용 (한글 키)
        "대상": target, "컬렉션": args.collection, "assignments": str(assignments_path), "폴더 단위": args.group_by,
        "실행 방식": "미리보기 (dry-run · 파일 안 만듦)" if args.dry_run else args.mode,
        "출력 폴더": str(out_dir), "point 수": len(rows), "폴더 수": len(folders),
        "파일 처리": placed or {"없음": 0}, "라벨 출처": summary_source(labels, args.labels),
        "GT 출처": summary_source(gt, args.gt_matches), "이전 잔재 폴더": len(stale),
    })
    write_index(index_path, title, summary, folders, args.group_by, warnings, args.dry_run, args.list_limit)
    for key in ("대상", "컬렉션", "assignments", "폴더 단위", "실행 방식", "출력 폴더", "point 수", "폴더 수",
                "파일 처리", "라벨 출처", "GT 출처", "이전 잔재 폴더"):
        summary.pop(key, None)
    write_json(report_path, summary)

    print("-" * 88)
    if args.dry_run:
        print("dry-run       : 폴더·파일을 만들지 않았습니다 — 이름 계획만 index.html / manifest.csv 에")
    print(f"folders       : {len(folders):,}   files: {sum(counts[k] for k in PLACED_ACTIONS):,}"
          f"   ({', '.join(f'{k} {v:,}' for k, v in placed.items()) or '없음'})")
    print(f"index         : {index_path}")
    print(f"manifest      : {out_dir / MANIFEST_NAME}")
    print(f"warnings      : {len(warnings)}")
    result_markers(report_path, index_path)
    return 0


def summary_source(loaded, raw) -> str:
    if loaded is None:
        return "없음" if not raw else f"없음 (파일 못 읽음: {raw})"
    return f"{raw} ({len(loaded):,})"


if __name__ == "__main__":
    sys.exit(main())
