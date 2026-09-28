#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOT = ROOT / "data" / "videos"
PROCESSED_ROOT = ROOT / "outputs" / "processed_videos"
OUT_ROOT = ROOT / "outputs" / "final_db_candidates"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video-stem", default="Normal_Videos_015_x264")
    p.add_argument("--person-per-track", type=int, default=5)
    p.add_argument("--object-per-track", type=int, default=5)
    p.add_argument("--object-min-width", type=int, default=10)
    p.add_argument("--object-min-height", type=int, default=16)
    p.add_argument("--object-min-area", type=int, default=220)
    p.add_argument("--object-min-sharpness", type=float, default=12.0)
    p.add_argument("--merge-max-gap-frames", type=int, default=30)
    p.add_argument("--merge-max-center-distance", type=float, default=80.0)
    return p.parse_args()


def find_video(video_stem):
    matches = [
        p for p in VIDEO_ROOT.rglob(video_stem + ".*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    if not matches:
        raise FileNotFoundError(video_stem)
    return sorted(matches)[0]


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def clamp_bbox(bbox, w, h):
    x1, y1, x2, y2 = map(float, bbox)
    x1 = max(0, min(w, int(math.floor(x1))))
    y1 = max(0, min(h, int(math.floor(y1))))
    x2 = max(0, min(w, int(math.ceil(x2))))
    y2 = max(0, min(h, int(math.ceil(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def sharpness(crop):
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def nclip(v, lo, hi):
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (float(v) - lo) / (hi - lo)))


def quality(row, crop, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = row["bbox_clamped"]
    bw, bh = x2 - x1, y2 - y1
    area_ratio = (bw * bh) / max(w * h, 1)
    conf = float(row.get("confidence") or 0.0)
    shp = sharpness(crop)
    route = str(row.get("final_db_route", "")).lower()

    if route == "person":
        size_score = nclip(area_ratio, 0.002, 0.06)
        shp_score = nclip(shp, 20, 220)
        val = float(row.get("final_person_score") or 0.0)
        score = 0.30 * conf + 0.25 * size_score + 0.20 * shp_score + 0.25 * val
    else:
        size_score = nclip(area_ratio, 0.0005, 0.03)
        shp_score = nclip(shp, 20, 220)
        sem = float(row.get("object_semantic_score") or 0.0)
        score = 0.25 * conf + 0.20 * size_score + 0.15 * shp_score + 0.40 * sem

    return {
        "quality_score": float(score),
        "sharpness": float(shp),
        "area_ratio": float(area_ratio),
        "crop_width": int(bw),
        "crop_height": int(bh),
    }


# ---------------------------------------------------------------------------
# Perceptual hash (DCT-based) — no external library needed, uses OpenCV+numpy
# ---------------------------------------------------------------------------

def phash_cv2(img_bgr, hash_size: int = 8) -> np.ndarray:
    """DCT-based perceptual hash. Returns bool array of length hash_size²."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Resize to (hash_size*4) × (hash_size*4) for stable DCT
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4),
                         interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    # Keep top-left hash_size × hash_size low-freq coefficients
    dct_low = dct[:hash_size, :hash_size]
    median = np.median(dct_low)
    return (dct_low > median).flatten()


def phash_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    """Hamming distance between two phash arrays (0 = identical, 64 = max)."""
    return int(np.count_nonzero(h1 != h2))


# ---------------------------------------------------------------------------
# Diverse top-k selection: temporal gap + pHash visual diversity
# ---------------------------------------------------------------------------

def diverse_topk(rows: list, k: int, scope: str = "object") -> list:
    """
    Select up to k representative crops with two diversity criteria.

    1. Temporal gap — crops must be at least `min_gap` frames apart.
       min_gap = max(1, span // (k + 1)) so picks spread across the full track.

    2. Visual diversity via pHash (DCT-based, no external lib).
       Crops within `phash_threshold` Hamming distance of any already-selected
       crop are skipped. Threshold is scope-aware:
         person → 6   (stricter: same outfit from same angle, skip)
         object → 10  (relaxed: object may have limited viewpoint variation)

    Two-pass fallback strategy:
      Pass 1: full temporal + pHash constraints
      Pass 2: temporal gap halved, pHash threshold unchanged
      (No Pass 3 — intentional: force quality over filling the quota)

    Edge-case safety:
      If both passes yield 0 results (e.g. single-frame track), return the
      single highest-quality crop as a guaranteed minimum.
    """
    rows = sorted(rows, key=lambda r: r["quality_score"], reverse=True)
    if len(rows) <= k:
        return rows

    frames = [int(r["frame_idx"]) for r in rows]
    span = max(frames) - min(frames) if len(frames) > 1 else 0

    # span / (k+1): k picks distributed over the full track duration
    min_gap = max(1, span // max(k + 1, 1))

    # person: stricter visual dedup (same-person, same-angle shots unwanted)
    # object: slightly relaxed (view diversity more limited for static objects)
    phash_threshold = 6 if scope == "person" else 10

    def _try_select(candidates, gap, phash_thr, existing_hashes):
        """Select from candidates, skipping already-picked hashes."""
        picked, used_frames = [], []
        used_hashes = list(existing_hashes)  # carry forward hashes from prior pass

        for r in candidates:
            if len(picked) >= k:
                break
            fi = int(r["frame_idx"])

            # Temporal check
            if used_frames and any(abs(fi - u) < gap for u in used_frames):
                continue

            # pHash visual check
            crop = r.get("_crop")
            if crop is not None and crop.size > 0:
                h = phash_cv2(crop)
                if any(phash_distance(h, uh) <= phash_thr for uh in used_hashes):
                    continue
                used_hashes.append(h)

            picked.append(r)
            used_frames.append(fi)

        return picked

    # Pass 1 — full constraints
    picked = _try_select(rows, min_gap, phash_threshold, [])

    # Pass 2 — halve temporal gap, keep pHash (fill remaining slots only)
    if len(picked) < k:
        picked_ids = set(id(r) for r in picked)
        remaining = [r for r in rows if id(r) not in picked_ids]
        existing_hashes = []
        for r in picked:
            crop = r.get("_crop")
            if crop is not None and crop.size > 0:
                existing_hashes.append(phash_cv2(crop))
        relaxed_gap = max(1, min_gap // 2)
        extra = _try_select(remaining, relaxed_gap, phash_threshold, existing_hashes)
        picked = (picked + extra)[:k]

    # Edge-case: if still 0 selected (e.g. entire track is one duplicate frame),
    # return the single best-quality crop as a guaranteed minimum.
    if not picked:
        picked = rows[:1]

    return picked


def bbox_center(bbox):
    x1, y1, x2, y2 = map(float, bbox)
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def center_distance(a, b):
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def hard_object_quality_pass(r, args):
    w = int(r.get("crop_width", 0))
    h = int(r.get("crop_height", 0))
    area = w * h
    shp = float(r.get("sharpness", 0.0))
    return (
        w >= args.object_min_width
        and h >= args.object_min_height
        and area >= args.object_min_area
        and shp >= args.object_min_sharpness
    )


def merge_split_object_tracks(groups, args):
    """
    Merge obvious split tracks of the same object when:
      - same label
      - close in time
      - spatially close
    This prevents track 6 / 7-like duplicate DB identities.
    """
    object_keys = sorted(
        [k for k in groups if k[0] == "object"],
        key=lambda k: (
            min(int(r["frame_idx"]) for r in groups[k]),
            k[1],
        ),
    )

    parent = {k: k for k in object_keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, ka in enumerate(object_keys):
        ra = groups[ka]
        label_a = str(
            ra[0].get("object_track_label")
            or ra[0].get("class_name")
            or "object"
        ).lower()
        end_a = max(ra, key=lambda r: int(r["frame_idx"]))

        for kb in object_keys[i + 1:]:
            rb = groups[kb]
            label_b = str(
                rb[0].get("object_track_label")
                or rb[0].get("class_name")
                or "object"
            ).lower()

            if label_a != label_b:
                continue

            start_b = min(rb, key=lambda r: int(r["frame_idx"]))
            gap = int(start_b["frame_idx"]) - int(end_a["frame_idx"])

            if gap < 0:
                continue
            if gap > args.merge_max_gap_frames:
                break

            dist = center_distance(
                end_a.get("bbox_clamped", end_a.get("bbox")),
                start_b.get("bbox_clamped", start_b.get("bbox")),
            )

            if dist <= args.merge_max_center_distance:
                union(ka, kb)

    merged = defaultdict(list)
    merge_map = {}

    for k in object_keys:
        root = find(k)
        merged[root].extend(groups[k])
        merge_map[k] = root

    # Keep person groups unchanged.
    out = defaultdict(list)
    for k, rows in groups.items():
        if k[0] == "person":
            out[k].extend(rows)

    for root, rows in merged.items():
        new_key = ("object", root[1])
        for r in rows:
            rr = dict(r)
            rr["merged_object_track_id"] = int(root[1])
            rr["merged_from_track_ids"] = sorted({
                int(x.get("long_track_id", x.get("track_id")))
                for x in rows
            })
            out[new_key].append(rr)

    return out, merge_map


def write_html(out_dir, summary, selected):
    groups = defaultdict(list)
    for r in selected:
        if r["candidate_scope"] == "person":
            group_id = int(
                r.get("canonical_person_id",
                      r.get("long_track_id", r.get("track_id")))
            )
        else:
            group_id = int(r.get("long_track_id", r.get("track_id")))
        groups[(r["candidate_scope"], group_id)].append(r)

    sections = []
    for (scope, tid), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: int(r["selected_rank"]))
        label = rows[0]["selected_label"]
        cards = []

        for r in rows:
            p = Path(r["selected_crop_path"])
            try:
                rel = p.resolve().relative_to(out_dir.resolve()).as_posix()
            except Exception:
                rel = p.as_uri()

            extra = (
                f'person {float(r.get("final_person_score") or 0):.3f}'
                if scope == "person"
                else f'semantic {float(r.get("object_semantic_score") or 0):.3f}'
            )

            # mm:ss timestamp from frame_idx and stored fps
            fps = float(r.get("fps", 25.0))
            ts_sec = int(r["frame_idx"]) / max(fps, 1.0)
            minutes = int(ts_sec // 60)
            seconds = int(ts_sec % 60)
            time_text = f"{minutes:02d}:{seconds:02d}"

            cards.append(f"""
            <div class="card">
              <img src="{html.escape(rel)}">
              <div class="meta">
                <b>#{r["selected_rank"]} &middot; {time_text}</b>
                <span>frame {r["frame_idx"]}</span>
                <span>Q {float(r["quality_score"]):.3f}</span>
                <span>det {float(r.get("confidence") or 0):.3f}</span>
                <span>{extra}</span>
                <span>{int(r["crop_width"])}&times;{int(r["crop_height"])}</span>
              </div>
            </div>""")

        sections.append(f"""
        <section>
          <div class="head">
            <h2>{scope.upper()} Track {tid} &middot; {html.escape(label)}</h2>
            <span>{len(rows)} selected</span>
          </div>
          <div class="grid">{''.join(cards)}</div>
        </section>""")

    doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Final DB Candidates</title>
<style>
:root{{--bg:#09101d;--panel:#121b2d;--line:#293753;--text:#eef4ff;--muted:#9aabc9}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,Arial,sans-serif}}
main{{max-width:1750px;margin:auto;padding:30px}}.top{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:22px 0}}
.metric{{background:var(--panel);border:1px solid var(--line);padding:16px;border-radius:14px}}.metric span{{color:var(--muted);font-size:12px}}.metric b{{display:block;font-size:28px}}
section{{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;margin:18px 0}}
.head{{display:flex;justify-content:space-between;align-items:center;gap:20px}}
.grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-top:14px}}
.card{{border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#070c16}}
.card img{{width:100%;height:340px;object-fit:contain;background:#04070d}}
.meta{{padding:10px;display:grid;gap:4px;font-size:13px}}.meta span{{color:var(--muted)}}
@media(max-width:1200px){{.top,.grid{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:700px){{.top,.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Final DB Candidates</h1>
<div class="top">
<div class="metric"><span>Person tracks</span><b>{summary["person_tracks"]}</b></div>
<div class="metric"><span>Object tracks</span><b>{summary["object_tracks"]}</b></div>
<div class="metric"><span>Person crops</span><b>{summary["person_selected_crops"]}</b></div>
<div class="metric"><span>Object crops</span><b>{summary["object_selected_crops"]}</b></div>
<div class="metric"><span>Total crops</span><b>{summary["total_selected_crops"]}</b></div>
</div>
{''.join(sections)}
</main></body></html>"""
    (out_dir / "final_db_candidates.html").write_text(doc, encoding="utf-8")


def main():
    args = parse_args()
    video_stem = args.video_stem
    video_path = find_video(video_stem)

    per_root = PROCESSED_ROOT / video_stem

    canonical_path = per_root / "final_routed_tracks_canonical.json"
    fallback_path = per_root / "final_routed_tracks.json"

    final_path = (
        canonical_path
        if canonical_path.is_file()
        else fallback_path
    )

    if not final_path.is_file():
        raise FileNotFoundError(final_path)

    print(f"[ROUTE] using: {final_path}")

    final_rows = load_json(final_path)
    object_rows = load_json(
        per_root / "object_validated_tracks_margin.json"
    )

    obj_idx = {}
    for r in object_rows:
        key = (
            int(r.get("long_track_id", r.get("track_id", -1))),
            int(r.get("frame_idx", -1)),
        )
        obj_idx[key] = r

    if final_path == canonical_path:
        person_rows_check = [
            r for r in final_rows
            if str(r.get("final_db_route", "")).lower() == "person"
        ]
        missing_canonical = [
            r for r in person_rows_check
            if r.get("canonical_person_id") is None
        ]
        if missing_canonical:
            raise RuntimeError(
                "canonical route file selected, but some person rows "
                "have no canonical_person_id"
            )

    merged = []
    for r in final_rows:
        route = str(r.get("final_db_route", "")).lower()

        if route == "person":
            out = dict(r)
            out["candidate_scope"] = "person"
            out["candidate_reason"] = "final_db_route=person"
            merged.append(out)

        elif route == "object":
            key = (
                int(r.get("long_track_id", r.get("track_id", -1))),
                int(r.get("frame_idx", -1)),
            )
            sem = obj_idx.get(key)
            if not sem:
                continue
            if str(sem.get("final_db_route_v2", "")).lower() != "object":
                continue

            out = dict(r)
            for k in (
                "object_track_label",
                "object_class_purity",
                "object_semantic_score",
                "object_semantic_decision",
                "final_db_route_v2",
            ):
                out[k] = sem.get(k)
            out["candidate_scope"] = "object"
            out["candidate_reason"] = "object_semantic_decision=good"
            merged.append(out)

    out_dir = OUT_ROOT / video_stem
    crop_root = out_dir / "crops"
    crop_root.mkdir(parents=True, exist_ok=True)

    by_frame = defaultdict(list)
    for r in merged:
        by_frame[int(r["frame_idx"])].append(r)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    # Read FPS from the video — used for mm:ss timestamps in HTML
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    print(f"[FPS] {fps:.3f}")

    enriched = []
    frame_ids = sorted(by_frame)
    for i, frame_idx in enumerate(frame_ids, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]

        for r in by_frame[frame_idx]:
            bb = clamp_bbox(r["bbox"], w, h)
            if bb is None:
                continue
            x1, y1, x2, y2 = bb
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            item = dict(r)
            item["bbox_clamped"] = bb
            item.update(quality(item, crop, frame.shape))
            item["_crop"] = crop
            item["_fps"] = fps        # store for later use in diverse_topk & HTML
            enriched.append(item)

        if i == 1 or i % 100 == 0 or i == len(frame_ids):
            print(f"[CROP] {i}/{len(frame_ids)} frames | {len(enriched)} records")

    cap.release()

    groups = defaultdict(list)
    for r in enriched:
        if r["candidate_scope"] == "person":
            group_id = int(
                r.get("canonical_person_id",
                      r.get("long_track_id", r.get("track_id")))
            )
        else:
            group_id = int(r.get("long_track_id", r.get("track_id")))
        groups[(r["candidate_scope"], group_id)].append(r)

    # Hard quality filter for object candidates BEFORE track-level selection.
    filtered_groups = defaultdict(list)
    object_quality_rejected = []

    for key, rows in groups.items():
        scope, _ = key
        if scope == "person":
            filtered_groups[key].extend(rows)
            continue

        for r in rows:
            if hard_object_quality_pass(r, args):
                filtered_groups[key].append(r)
            else:
                object_quality_rejected.append({
                    "track_id": int(r.get("long_track_id", r.get("track_id"))),
                    "frame_idx": int(r["frame_idx"]),
                    "crop_width": int(r.get("crop_width", 0)),
                    "crop_height": int(r.get("crop_height", 0)),
                    "sharpness": float(r.get("sharpness", 0.0)),
                    "quality_score": float(r.get("quality_score", 0.0)),
                })

    # Merge obvious split object tracks (e.g. same handbag split into 6/7).
    filtered_groups, merge_map = merge_split_object_tracks(
        filtered_groups,
        args,
    )

    selected = []
    for (scope, tid), rows in sorted(filtered_groups.items()):
        if not rows:
            continue

        k = args.person_per_track if scope == "person" else args.object_per_track
        # Pass scope so diverse_topk applies the right pHash threshold
        for rank, r in enumerate(diverse_topk(rows, k, scope=scope), 1):
            label = "person" if scope == "person" else str(
                r.get("object_track_label") or r.get("class_name") or "object"
            )
            dst = (
                crop_root / scope /
                f"track_{tid:04d}_{label.replace(' ', '_')}" /
                f"{rank:02d}_frame_{int(r['frame_idx']):06d}.jpg"
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), r["_crop"])

            clean = {kk: vv for kk, vv in r.items() if kk not in ("_crop", "_fps")}
            clean["selected_rank"] = rank
            clean["selected_crop_path"] = str(dst.resolve())
            clean["selected_label"] = label
            clean["selected_track_id"] = int(tid)
            clean["fps"] = float(fps)      # persist fps for HTML rendering
            if scope == "person":
                clean["canonical_person_id"] = int(tid)
            selected.append(clean)

    persons = [r for r in selected if r["candidate_scope"] == "person"]
    objects = [r for r in selected if r["candidate_scope"] == "object"]

    summary = {
        "video_stem": video_stem,
        "video_path": str(video_path.resolve()),
        "route_file": str(final_path.resolve()),
        "fps": float(fps),
        "person_tracks": len({
            int(r.get("selected_track_id",
                      r.get("canonical_person_id",
                            r.get("long_track_id"))))
            for r in persons
        }),
        "object_tracks": len({int(r.get("selected_track_id", r["long_track_id"])) for r in objects}),
        "person_selected_crops": len(persons),
        "object_selected_crops": len(objects),
        "total_selected_crops": len(selected),
        "object_quality_rejected_records": len(object_quality_rejected),
        "object_track_merge_map": {
            f"{k[0]}:{k[1]}": f"{v[0]}:{v[1]}"
            for k, v in merge_map.items()
        },
        "object_quality_thresholds": {
            "min_width": args.object_min_width,
            "min_height": args.object_min_height,
            "min_area": args.object_min_area,
            "min_sharpness": args.object_min_sharpness,
        },
    }

    (out_dir / "final_db_candidates.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "candidates": selected,
                "object_quality_rejected": object_quality_rejected,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_html(out_dir, summary, selected)

    print()
    print("=" * 84)
    print("FINAL DB CANDIDATE SELECTION COMPLETE")
    print("=" * 84)
    for k, v in summary.items():
        print(f"{k:24s}: {v}")
    print("json                    :", out_dir / "final_db_candidates.json")
    print("html                    :", out_dir / "final_db_candidates.html")
    print("=" * 84)


if __name__ == "__main__":
    main()
