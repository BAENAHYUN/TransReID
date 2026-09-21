#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from config import PipelineConfig
from registry import EmbedderRegistry
from router import Router

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "pipeline.yaml"


def parse_args():
    p = argparse.ArgumentParser(
        description="Conservative SOLIDER post-merge for fragmented person tracks"
    )
    p.add_argument("--video", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--review-dir", default="./outputs/person_track_postmerge")
    p.add_argument("--samples-per-track", type=int, default=5)
    p.add_argument("--similarity-threshold", type=float, default=0.95)
    p.add_argument("--conflict-iou-threshold", type=float, default=0.10)
    p.add_argument("--min-width", type=int, default=14)
    p.add_argument("--min-height", type=int, default=25)
    p.add_argument("--min-area", type=int, default=500)
    return p.parse_args()


def load_rows(path: Path) -> List[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("tracks"), list):
        return obj["tracks"]
    raise TypeError("tracks JSON must be list or {'tracks': [...]} object")


def l2(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        raise ValueError("zero vector")
    return v / n


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = aa + ba - inter
    return 0.0 if den <= 0 else inter / den


def sample_rows(rows, n):
    rows = sorted(rows, key=lambda r: int(r["frame_idx"]))
    if len(rows) <= n:
        return rows
    idxs = np.linspace(0, len(rows)-1, n).round().astype(int)
    return [rows[int(i)] for i in sorted(set(map(int, idxs)))]


def crop_from_row(cap, row, args):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_idx"]))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, row["bbox"])
    x1 = max(0, min(w, int(math.floor(x1))))
    y1 = max(0, min(h, int(math.floor(y1))))
    x2 = max(0, min(w, int(math.ceil(x2))))
    y2 = max(0, min(h, int(math.ceil(y2))))

    cw, ch = x2-x1, y2-y1
    if cw < args.min_width or ch < args.min_height or cw*ch < args.min_area:
        return None

    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def tracks_conflict(rows_a, rows_b, iou_threshold):
    a = defaultdict(list)
    b = defaultdict(list)
    for r in rows_a:
        a[int(r["frame_idx"])].append(r)
    for r in rows_b:
        b[int(r["frame_idx"])].append(r)

    examples = []
    for f in sorted(set(a) & set(b)):
        for ra in a[f]:
            for rb in b[f]:
                iou = bbox_iou(ra["bbox"], rb["bbox"])
                if iou < iou_threshold:
                    examples.append((f, iou))
                    if len(examples) >= 5:
                        return True, examples
    return bool(examples), examples


def component_can_merge(comp_a, comp_b, pair_info, threshold):
    # Complete-link + cannot-link:
    # EVERY cross pair must be non-conflicting and above threshold.
    for a in comp_a:
        for b in comp_b:
            key = tuple(sorted((a, b)))
            info = pair_info[key]
            if info["conflict"]:
                return False
            if info["similarity"] < threshold:
                return False
    return True


def write_html(path, groups, preview_paths, pair_rows):
    sections = []
    for canonical, members in sorted(groups.items()):
        cards = []
        for tid in members:
            for p in preview_paths.get(tid, [])[:3]:
                cards.append(
                    f'<div class="card"><b>track {tid}</b>'
                    f'<img src="{html.escape(Path(p).resolve().as_uri())}"></div>'
                )
        sections.append(
            f'<section><h2>canonical_person_id {canonical}</h2>'
            f'<p>members: {members}</p><div class="grid">{"".join(cards)}</div></section>'
        )

    rows = "".join(
        f"<tr><td>{x['track_a']}</td><td>{x['track_b']}</td>"
        f"<td>{x['similarity']:.4f}</td><td>{x['conflict']}</td></tr>"
        for x in pair_rows
    )

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Person Post-Merge Review</title>
<style>
body{{background:#09101d;color:#eef4ff;font-family:Segoe UI,Arial;padding:24px}}
section{{background:#121b2d;border:1px solid #293753;border-radius:14px;padding:16px;margin:16px 0}}
.grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}}
.card{{background:#070c16;border:1px solid #293753;border-radius:10px;padding:8px}}
.card img{{width:100%;height:220px;object-fit:contain}}
table{{width:100%;border-collapse:collapse}}
th,td{{border:1px solid #293753;padding:7px}}
</style></head><body>
<h1>Person Track Post-Merge · Complete-Link</h1>
<table><tr><th>A</th><th>B</th><th>cosine</th><th>conflict</th></tr>{rows}</table>
{"".join(sections)}
</body></html>"""
    path.write_text(doc, encoding="utf-8")


def main():
    args = parse_args()

    video = Path(args.video)
    tracks_path = Path(args.tracks)
    output = Path(args.output)
    review_dir = Path(args.review_dir)
    crop_root = review_dir / "preview_crops"
    crop_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(tracks_path)
    person_rows = [r for r in rows if str(r.get("final_db_route","")).lower() == "person"]

    grouped = defaultdict(list)
    for r in person_rows:
        grouped[int(r.get("long_track_id", r.get("track_id")))].append(r)

    if not grouped:
        raise RuntimeError("No final_db_route=person tracks")

    cfg = PipelineConfig.load(CONFIG_PATH)
    reg = EmbedderRegistry(cfg)
    router = Router(cfg, reg, input_format="rgb")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")

    track_vectors: Dict[int, np.ndarray] = {}
    preview_paths = defaultdict(list)

    try:
        for tid in sorted(grouped):
            embs = []
            for idx, row in enumerate(sample_rows(grouped[tid], args.samples_per_track), 1):
                crop = crop_from_row(cap, row, args)
                if crop is None:
                    continue
                dst = crop_root / f"track_{tid:04d}_{idx:02d}_f{int(row['frame_idx']):06d}.jpg"
                cv2.imwrite(str(dst), crop)
                preview_paths[tid].append(str(dst))

                vecs = router.embed_query_image(str(dst), scope="person", names=["solider"])
                embs.append(l2(vecs["solider"]))

            if embs:
                track_vectors[tid] = l2(np.mean(np.stack(embs), axis=0))
                print(f"[EMBED] track={tid} samples={len(embs)}")
            else:
                print(f"[SKIP] track={tid}: no valid crop")
    finally:
        cap.release()
        try:
            reg.release()
        except Exception:
            pass

    tids = sorted(track_vectors)
    pair_info = {}
    pair_rows = []

    for i, a in enumerate(tids):
        for b in tids[i+1:]:
            sim = float(np.dot(track_vectors[a], track_vectors[b]))
            conflict, examples = tracks_conflict(
                grouped[a], grouped[b], args.conflict_iou_threshold
            )
            info = {
                "track_a": a,
                "track_b": b,
                "similarity": sim,
                "conflict": conflict,
                "conflict_examples": examples,
            }
            pair_info[(a, b)] = info
            pair_rows.append(info)

    # Agglomerative complete-link constrained clustering.
    clusters = [[tid] for tid in tids]

    while True:
        best = None
        best_score = -1.0

        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                ca, cb = clusters[i], clusters[j]
                if not component_can_merge(ca, cb, pair_info, args.similarity_threshold):
                    continue

                sims = [
                    pair_info[tuple(sorted((a,b)))]["similarity"]
                    for a in ca for b in cb
                ]
                score = min(sims)  # complete-link
                if score > best_score:
                    best_score = score
                    best = (i, j)

        if best is None:
            break

        i, j = best
        merged = sorted(clusters[i] + clusters[j])
        clusters = [
            c for k, c in enumerate(clusters)
            if k not in (i, j)
        ]
        clusters.append(merged)

    clusters = sorted(clusters, key=lambda c: min(c))

    groups = {}
    canonical_of = {}
    for members in clusters:
        canonical = min(members)
        groups[canonical] = members
        for tid in members:
            canonical_of[tid] = canonical

    revised = []
    for r in rows:
        out = dict(r)
        if str(r.get("final_db_route","")).lower() == "person":
            tid = int(r.get("long_track_id", r.get("track_id")))
            out["canonical_person_id"] = int(canonical_of.get(tid, tid))
            out["canonical_person_members"] = groups.get(
                canonical_of.get(tid, tid), [tid]
            )
        revised.append(out)

    output.write_text(
        json.dumps(revised, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    review_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_person_tracks": len(grouped),
        "embedded_tracks": len(track_vectors),
        "canonical_person_tracks": len(groups),
        "similarity_threshold": args.similarity_threshold,
        "groups": [
            {"canonical_person_id": k, "members": v}
            for k, v in sorted(groups.items())
        ],
        "pair_decisions": pair_rows,
    }
    (review_dir / "person_track_postmerge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_html(
        review_dir / "person_track_postmerge.html",
        groups,
        preview_paths,
        pair_rows,
    )

    print()
    print("=" * 88)
    print("PERSON TRACK POST-MERGE COMPLETE")
    print("=" * 88)
    print("input person tracks     :", len(grouped))
    print("canonical person tracks :", len(groups))
    print("groups                  :", groups)
    print("threshold               :", args.similarity_threshold)
    print("=" * 88)


if __name__ == "__main__":
    main()
