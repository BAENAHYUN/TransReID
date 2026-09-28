#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_leiden_gallery_track.py

Track-aware Leiden gallery: one representative crop per track per cluster.

Rationale:
  - cluster_leiden_track_centroid.py assigns cluster_id at TRACK level, then
    propagates to all crop point_ids in that track.
  - A naive gallery that shows all crops has N crops per track → same person
    appears N times, making it hard to spot over-merging.
  - This gallery collapses each track to ONE representative crop, so Cluster N
    shows exactly N distinct tracks, making cross-person contamination obvious.

Representative crop selection (per track, in priority order):
  1. Highest det_conf / score / confidence field
  2. Frame closest to the temporal median of the track
  3. First point_id seen

Usage:
    python build_leiden_gallery_track.py \
        --assignments outputs/clustering/leiden_track_sweep/t096/person/person_leiden_assignments.jsonl \
        --collection forensic_person \
        --output-dir outputs/clustering/leiden_track_sweep/t096/gallery_track \
        --tracks-per-cluster 60
"""
from __future__ import annotations

import argparse
import html
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

CROP_KEYS = ("crop_path", "selected_crop_path", "image_path", "path")
VIDEO_KEYS = ("video_stem", "video_path", "video", "source_path")
CONF_KEYS = ("det_conf", "score", "confidence", "det_score")
FRAME_KEYS = ("frame_idx", "frame_index", "frame")


def chunks(seq: Sequence[Any], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


class QdrantHTTP:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"Content-Type": "application/json"})
        if api_key:
            self.s.headers.update({"api-key": api_key})

    def post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        r = self.s.post(self.base_url + path, json=body, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f"Qdrant POST {path} → {r.status_code}\n{r.text[:2000]}")
        return r.json()

    def retrieve_points(
        self, collection: str, ids: Sequence[Any], batch_size: int = 256
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for batch in chunks(list(ids), batch_size):
            data = self.post(
                f"/collections/{collection}/points",
                {"ids": list(batch), "with_payload": True, "with_vector": False},
            )
            for row in data.get("result") or []:
                out[str(row["id"])] = row.get("payload") or {}
        return out


# ---------------------------------------------------------------------------
# Track key — must match cluster_leiden_track_centroid.py
# ---------------------------------------------------------------------------

def track_key(payload: Dict[str, Any]) -> str:
    stem = str(payload.get("video_stem", payload.get("video", "unk")))
    cid = payload.get("canonical_person_id")
    if cid is not None and str(cid).strip():
        return f"{stem}||canonical||{cid}"
    tid = str(
        payload.get("long_track_id",
        payload.get("track_id",
        payload.get("selected_track_id", "0")))
    )
    return f"{stem}||track||{tid}"


def first_val(d: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for k in keys:
        v = d.get(k)
        if v is not None and str(v).strip():
            return v
    return None


def resolve_path(raw: Optional[str], project_root: Path) -> Optional[Path]:
    if not raw:
        return None
    p = Path(raw)
    if p.is_file():
        return p
    for base in (project_root, Path(".")):
        q = (base / raw.replace("\\", "/")).resolve()
        if q.is_file():
            return q
    return None


def safe_name(point_id: Any, suffix: str) -> str:
    clean = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(point_id))
    return clean[:140] + suffix


def esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


# ---------------------------------------------------------------------------
# Track representative selection
# ---------------------------------------------------------------------------

def pick_representative(
    pids: List[Any],
    payloads: Dict[str, Dict[str, Any]],
) -> Any:
    """Return the point_id that best represents this track."""
    best_pid = pids[0]

    # Strategy 1: highest confidence
    best_conf = -1.0
    for pid in pids:
        pl = payloads.get(str(pid), {})
        conf_raw = first_val(pl, CONF_KEYS)
        if conf_raw is not None:
            try:
                conf = float(conf_raw)
                if conf > best_conf:
                    best_conf = conf
                    best_pid = pid
            except (ValueError, TypeError):
                pass

    if best_conf >= 0:
        return best_pid

    # Strategy 2: frame closest to temporal median
    frames: List[Tuple[int, Any]] = []
    for pid in pids:
        pl = payloads.get(str(pid), {})
        f_raw = first_val(pl, FRAME_KEYS)
        if f_raw is not None:
            try:
                frames.append((int(f_raw), pid))
            except (ValueError, TypeError):
                pass

    if frames:
        frames.sort(key=lambda x: x[0])
        median_idx = len(frames) // 2
        return frames[median_idx][1]

    # Strategy 3: first pid
    return pids[0]


# ---------------------------------------------------------------------------
# Group cluster members by track, return sorted track list
# ---------------------------------------------------------------------------

def group_by_track(
    members: List[Dict[str, Any]],
    payloads: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, List[Any]]]:
    """
    Returns list of (track_key, [point_ids...]) sorted by (video_stem, track_id).
    """
    track_pids: Dict[str, List[Any]] = defaultdict(list)
    for row in members:
        pid = row["point_id"]
        pl = payloads.get(str(pid), {})
        tk = track_key(pl)
        track_pids[tk].append(pid)

    def sort_key(item):
        tk = item[0]
        parts = tk.split("||")
        stem = parts[0] if parts else tk
        rest = parts[-1] if len(parts) > 1 else ""
        try:
            return (stem, int(rest))
        except ValueError:
            return (stem, rest)

    return sorted(track_pids.items(), key=sort_key)


# ---------------------------------------------------------------------------
# Load / split
# ---------------------------------------------------------------------------

def load_assignments(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "point_id" not in row:
                raise RuntimeError(f"point_id missing at line {n}")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No assignments in {path}")
    return rows


def split_groups(rows: List[Dict[str, Any]]):
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    noise: List[Dict[str, Any]] = []
    for row in rows:
        cid = row.get("cluster_id")
        if row.get("noise") or cid is None:
            noise.append(row)
        else:
            groups[str(cid)].append(row)
    return dict(groups), noise


def choose_clusters(
    groups: Dict[str, List[Dict[str, Any]]],
    top_n: int, medium_n: int, small_n: int, seed: int,
) -> List[Tuple[str, List[Dict[str, Any]], str]]:
    rng = random.Random(seed)
    # Sort by cluster_tracks (from JSONL) then cluster_size then key
    def sort_key(kv):
        members = kv[1]
        n_tracks = members[0].get("cluster_tracks", 1) if members else 1
        n_pts = len(members)
        return (-n_tracks, -n_pts, kv[0])

    ordered = sorted(groups.items(), key=sort_key)
    selected = []
    used: set = set()

    for cid, members in ordered[:top_n]:
        selected.append((cid, members, "largest"))
        used.add(cid)

    n_tracks_list = [
        (members[0].get("cluster_tracks", 1) if members else 1, cid, members)
        for cid, members in ordered if cid not in used
    ]
    medium = [(c, m) for nt, c, m in n_tracks_list if 5 <= nt <= 80]
    if len(medium) > medium_n:
        medium = rng.sample(medium, medium_n)
    for cid, members in medium:
        selected.append((cid, members, "medium"))
        used.add(cid)

    small = [(c, m) for nt, c, m in n_tracks_list if 2 <= nt <= 4 and c not in used]
    if len(small) > small_n:
        small = rng.sample(small, small_n)
    for cid, members in small:
        selected.append((cid, members, "small"))

    return selected


# ---------------------------------------------------------------------------
# HTML render
# ---------------------------------------------------------------------------

CSS = """
:root{--bg:#0b1020;--panel:#141b2d;--panel2:#19233b;--line:#2b3758;
  --text:#eef3ff;--muted:#9cacc9;--accent:#7aa2ff;--good:#5bd5a6;--warn:#ffcf6b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
  font-family:Segoe UI,Arial,sans-serif}
.wrap{max-width:1700px;margin:auto;padding:26px}h1{margin:0 0 6px;font-size:28px}
.topsub{color:var(--muted);margin-bottom:18px}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:18px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.stat .k{color:var(--muted);font-size:11px}.stat .v{font-size:22px;font-weight:700;margin-top:5px}
.cluster{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  margin:16px 0;padding:16px}
.cluster-head{display:flex;justify-content:space-between;gap:12px;
  align-items:flex-start;margin-bottom:11px}
.cluster h2{margin:0;font-size:18px}.sub{color:var(--muted);font-size:11px;
  margin-top:3px;word-break:break-all}
.counts{text-align:right;white-space:nowrap}
.count-tracks{font-size:18px;font-weight:700;color:var(--good)}
.count-pts{font-size:12px;color:var(--muted);margin-top:2px}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:9px}
.thumb{background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:7px}
.thumb img{width:100%;height:180px;object-fit:contain;background:#090d16;border-radius:6px;display:block}
.track-badge{display:inline-block;background:#1e2e55;color:var(--accent);
  font-size:9px;font-weight:700;border-radius:4px;padding:1px 5px;margin:4px 0 2px}
.meta{font-size:10px;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.path{font-size:9px;color:var(--muted);margin-top:3px;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.missingbox{height:180px;display:flex;align-items:center;justify-content:center;
  border-radius:6px;background:#351d27;color:#ffadb7;font-weight:700;font-size:11px}
.note{background:var(--panel);border-left:4px solid var(--accent);border-radius:9px;
  padding:11px 14px;color:#dbe5fa;margin:0 0 16px}
.noise-section{opacity:.85}
@media(max-width:800px){.stats{grid-template-columns:repeat(2,1fr)}}
"""


def main():
    p = argparse.ArgumentParser(
        description="Track-aware Leiden gallery: 1 representative crop per track per cluster"
    )
    p.add_argument("--assignments",
                   default=r"outputs\clustering\leiden_track_centroid\person\person_leiden_assignments.jsonl")
    p.add_argument("--collection", default="forensic_person")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--api-key", default=None)
    p.add_argument("--project-root", default=".")
    p.add_argument("--output-dir",
                   default=r"outputs\clustering\leiden_track_centroid\person\gallery_track")
    p.add_argument("--top-clusters", type=int, default=20,
                   help="Show N largest clusters")
    p.add_argument("--medium-clusters", type=int, default=10,
                   help="Show N randomly sampled medium clusters (5-80 tracks)")
    p.add_argument("--small-clusters", type=int, default=10,
                   help="Show N randomly sampled small clusters (2-4 tracks)")
    p.add_argument("--tracks-per-cluster", type=int, default=60,
                   help="Max tracks (= images) shown per cluster")
    p.add_argument("--noise-samples", type=int, default=60,
                   help="Noise track representatives to show")
    p.add_argument("--qdrant-batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    assignments_path = Path(args.assignments).resolve()
    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    assets_dir = out_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRACK-AWARE LEIDEN GALLERY")
    print("=" * 80)

    rows = load_assignments(assignments_path)
    groups, noise_rows = split_groups(rows)
    selected = choose_clusters(
        groups, args.top_clusters, args.medium_clusters, args.small_clusters, args.seed
    )

    print(f"assignments    : {len(rows):,}")
    print(f"clusters       : {len(groups):,}")
    print(f"noise points   : {len(noise_rows):,}")
    print(f"selected       : {len(selected):,} clusters to render")

    # Collect all point_ids we need payloads for
    all_pids_needed = set()
    for _, members, _ in selected:
        for row in members:
            all_pids_needed.add(row["point_id"])
    rng = random.Random(args.seed)
    noise_sample = noise_rows if len(noise_rows) <= args.noise_samples else rng.sample(noise_rows, args.noise_samples)
    for row in noise_sample:
        all_pids_needed.add(row["point_id"])

    print(f"Qdrant fetch   : {len(all_pids_needed):,} point_ids ...")
    q = QdrantHTTP(args.qdrant_url, args.api_key)
    payloads = q.retrieve_points(args.collection, list(all_pids_needed), args.qdrant_batch)
    print(f"payloads fetched: {len(payloads):,}")

    copied = 0
    missing = 0
    sections_html = []

    def render_cluster_section(rank: int, cid: str, kind: str, members: List[Dict[str, Any]]) -> str:
        nonlocal copied, missing

        # Group members by track, pick representative per track
        track_list = group_by_track(members, payloads)  # [(tk, [pids...]), ...]
        n_tracks_total = len(track_list)
        n_pts_total = sum(len(pids) for _, pids in track_list)

        # Cap at tracks_per_cluster — random sample so not biased toward early-video tracks
        if len(track_list) > args.tracks_per_cluster:
            local_rng = random.Random(args.seed + rank)
            track_list = local_rng.sample(track_list, args.tracks_per_cluster)
            track_list.sort(key=lambda x: x[0])  # re-sort by (stem, track_id) after sample
            truncated = True
        else:
            truncated = False

        slug = f"cluster_{rank:03d}"
        group_dir = assets_dir / slug
        group_dir.mkdir(parents=True, exist_ok=True)

        cards = []
        for tk, pids in track_list:
            rep_pid = pick_representative(pids, payloads)
            payload = payloads.get(str(rep_pid), {})

            raw_crop = first_val(payload, CROP_KEYS)
            src = resolve_path(raw_crop, project_root)

            # Parse track key for display
            tk_parts = tk.split("||")
            stem_display = tk_parts[0] if tk_parts else tk
            id_display = tk_parts[-1] if len(tk_parts) > 1 else ""

            frame = first_val(payload, FRAME_KEYS) or ""
            conf_raw = first_val(payload, CONF_KEYS)
            conf_display = f"{float(conf_raw):.3f}" if conf_raw is not None else ""
            canonical = payload.get("canonical_person_id", "")
            track_id = payload.get("long_track_id", payload.get("track_id", ""))
            n_crops = len(pids)

            if src is None:
                missing += 1
                cards.append(
                    f'<div class="thumb missing">'
                    f'<div class="missingbox">이미지 없음</div>'
                    f'<div class="track-badge">track: {esc(id_display)}</div>'
                    f'<div class="meta">{esc(stem_display)}</div>'
                    f'<div class="path">{esc(raw_crop or "crop_path 없음")}</div>'
                    f'</div>'
                )
                continue

            suffix = src.suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                suffix = ".jpg"
            dst = group_dir / safe_name(rep_pid, suffix)
            if not dst.exists():
                shutil.copy2(src, dst)
            copied += 1
            rel = dst.relative_to(out_dir).as_posix()

            cards.append(
                f'<div class="thumb">'
                f'<a href="{esc(rel)}" target="_blank">'
                f'<img src="{esc(rel)}" loading="lazy" title="{esc(tk)}">'
                f'</a>'
                f'<div class="track-badge">track {esc(id_display)} · {n_crops}crop</div>'
                f'<div class="meta" title="{esc(stem_display)}">{esc(Path(stem_display).stem if Path(stem_display).parts[:-0] and len(Path(stem_display).parts) > 1 else stem_display)}</div>'
                f'<div class="meta">canonical: {esc(canonical) or esc(track_id)}</div>'
                f'<div class="meta">frame {esc(frame)}{" · conf " + esc(conf_display) if conf_display else ""}</div>'
                f'</div>'
            )

        truncation_note = (
            f'<div class="note" style="margin:6px 0 10px;font-size:11px;">'
            f'⚠ {n_tracks_total}개 tracks 중 상위 {args.tracks_per_cluster}개만 표시</div>'
            if truncated else ""
        )

        return (
            f'<section class="cluster">'
            f'<div class="cluster-head">'
            f'<div><h2>Cluster {rank} · <span style="color:var(--warn)">{kind}</span></h2>'
            f'<div class="sub">{esc(cid)}</div></div>'
            f'<div class="counts">'
            f'<div class="count-tracks">{n_tracks_total} tracks</div>'
            f'<div class="count-pts">{n_pts_total:,} crops</div>'
            f'</div></div>'
            f'{truncation_note}'
            f'<div class="gallery">{"".join(cards)}</div>'
            f'</section>'
        )

    def render_noise_section(rows: List[Dict[str, Any]]) -> str:
        nonlocal copied, missing
        track_list = group_by_track(rows, payloads)
        slug = "noise"
        group_dir = assets_dir / slug
        group_dir.mkdir(parents=True, exist_ok=True)

        cards = []
        for tk, pids in track_list:
            rep_pid = pick_representative(pids, payloads)
            payload = payloads.get(str(rep_pid), {})
            raw_crop = first_val(payload, CROP_KEYS)
            src = resolve_path(raw_crop, project_root)
            tk_parts = tk.split("||")
            stem_display = tk_parts[0] if tk_parts else tk
            id_display = tk_parts[-1] if len(tk_parts) > 1 else ""
            n_crops = len(pids)

            if src is None:
                missing += 1
                cards.append(
                    f'<div class="thumb missing">'
                    f'<div class="missingbox">이미지 없음</div>'
                    f'<div class="track-badge">{esc(id_display)}</div>'
                    f'</div>'
                )
                continue

            suffix = src.suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                suffix = ".jpg"
            dst = group_dir / safe_name(rep_pid, suffix)
            if not dst.exists():
                shutil.copy2(src, dst)
            copied += 1
            rel = dst.relative_to(out_dir).as_posix()

            cards.append(
                f'<div class="thumb">'
                f'<a href="{esc(rel)}" target="_blank"><img src="{esc(rel)}" loading="lazy"></a>'
                f'<div class="track-badge">noise · {n_crops}crop</div>'
                f'<div class="meta">{esc(stem_display)}</div>'
                f'<div class="meta">{esc(id_display)}</div>'
                f'</div>'
            )

        return (
            f'<section class="cluster noise-section">'
            f'<div class="cluster-head">'
            f'<div><h2>Noise sample</h2>'
            f'<div class="sub">전체 noise {len(noise_rows):,} rows 중 {len(rows):,} tracks 샘플</div></div>'
            f'<div class="counts"><div class="count-tracks">{len(track_list)} tracks</div></div>'
            f'</div>'
            f'<div class="gallery">{"".join(cards)}</div>'
            f'</section>'
        )

    for rank, (cid, members, kind) in enumerate(selected, 1):
        sections_html.append(render_cluster_section(rank, cid, kind, members))

    if noise_sample:
        sections_html.append(render_noise_section(noise_sample))

    # Build summary stats
    n_clusters = len(groups)
    n_noise_pts = len(noise_rows)
    # noise track count from JSONL (cluster_tracks field on noise rows is always 1,
    # so count unique point_ids as proxy — exact count needs full payload fetch)
    n_noise_tracks = None  # omitted: would require fetching all noise payloads

    html_out = (
        f'<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Track Gallery – {esc(Path(args.assignments).parts[-3] if len(Path(args.assignments).parts) >= 3 else "Leiden")}</title>'
        f'<style>{CSS}</style></head><body><div class="wrap">'
        f'<h1>Track-Aware Leiden Gallery</h1>'
        f'<div class="topsub">{esc(args.assignments)} | {esc(args.collection)} | '
        f'1 representative crop / track</div>'
        f'<div class="note">각 카드 = 하나의 track · 대표 crop 1장. '
        f'같은 cluster 안에 다른 사람이 섞여 있으면 즉시 육안으로 확인 가능.</div>'
        f'<div class="stats">'
        f'<div class="stat"><div class="k">Total clusters</div><div class="v">{n_clusters:,}</div></div>'
        f'<div class="stat"><div class="k">Noise points</div><div class="v">{n_noise_pts:,}</div></div>'
        f'<div class="stat"><div class="k">Copied images</div><div class="v">{copied:,}</div></div>'
        f'<div class="stat"><div class="k">Missing images</div><div class="v">{missing:,}</div></div>'
        f'</div>'
        f'{"".join(sections_html)}'
        f'</div></body></html>'
    )

    # Update copied/missing in stats (rendered after sections)
    html_path = out_dir / "leiden_track_gallery.html"
    html_path.write_text(html_out, encoding="utf-8")

    report = {
        "assignments": len(rows),
        "clusters": n_clusters,
        "noise_points": n_noise_pts,
        "noise_tracks": n_noise_tracks,
        "selected_clusters": len(selected),
        "copied_images": copied,
        "missing_images": missing,
        "html": str(html_path),
    }
    (out_dir / "gallery_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 80)
    print("GALLERY COMPLETE")
    print("=" * 80)
    print(f"HTML           : {html_path}")
    print(f"copied images  : {copied:,}")
    print(f"missing images : {missing:,}")
    print(f"report         : {out_dir / 'gallery_report.json'}")


if __name__ == "__main__":
    main()
