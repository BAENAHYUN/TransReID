
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
VIDEO_ROOT = ROOT / "data" / "videos"
PROCESSED_ROOT = ROOT / "outputs" / "processed_videos"
REPORT_ROOT = ROOT / "outputs" / "track_quality_report"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def find_video(video_stem: str) -> Path:
    direct = [p for p in VIDEO_ROOT.glob(video_stem + ".*")
              if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    if direct:
        return sorted(direct)[0]

    recursive = [p for p in VIDEO_ROOT.rglob(video_stem + ".*")
                 if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    if recursive:
        return sorted(recursive)[0]

    raise FileNotFoundError(f"video not found: {video_stem}")


def load_tracks(video_stem: str) -> List[dict]:
    p = PROCESSED_ROOT / video_stem / "final_routed_tracks.json"
    if not p.is_file():
        raise FileNotFoundError(p)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError("final_routed_tracks.json must contain a JSON array")
    return [
        dict(r) for r in data
        if str(r.get("final_db_route", "")).lower() in {"person", "object"}
    ]


def clamp_bbox(bbox, width: int, height: int):
    x1, y1, x2, y2 = map(float, bbox)
    x1 = max(0, min(width, int(math.floor(x1))))
    y1 = max(0, min(height, int(math.floor(y1))))
    x2 = max(0, min(width, int(math.ceil(x2))))
    y2 = max(0, min(height, int(math.ceil(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def laplacian_sharpness(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def border_penalty(x1, y1, x2, y2, w, h) -> float:
    # 1.0 = no touching border, 0.0 = severe clipping
    touches = 0
    margin_x = max(2, int(w * 0.01))
    margin_y = max(2, int(h * 0.01))
    if x1 <= margin_x:
        touches += 1
    if y1 <= margin_y:
        touches += 1
    if x2 >= w - margin_x:
        touches += 1
    if y2 >= h - margin_y:
        touches += 1
    return max(0.0, 1.0 - 0.25 * touches)


def center_score(x1, y1, x2, y2, w, h) -> float:
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    nx = abs(cx - w * 0.5) / max(w * 0.5, 1)
    ny = abs(cy - h * 0.5) / max(h * 0.5, 1)
    d = min(1.0, math.sqrt(nx * nx + ny * ny) / math.sqrt(2))
    return 1.0 - d


def normalize_clip(v, lo, hi):
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def quality_score(row: dict, crop: np.ndarray, frame_shape: Tuple[int, int, int], bbox):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    area_ratio = (bw * bh) / max(w * h, 1)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharp_raw = laplacian_sharpness(gray)

    conf = float(row.get("confidence") or 0.0)
    person_score = row.get("final_person_score")
    person_score = float(person_score) if person_score is not None else 0.0

    route = str(row.get("final_db_route", "")).lower()

    if route == "person":
        # Person crops need more body area than tiny object crops.
        size_score = normalize_clip(area_ratio, 0.0025, 0.08)
        sharp_score = normalize_clip(sharp_raw, 20.0, 220.0)
        val_score = person_score
    else:
        size_score = normalize_clip(area_ratio, 0.0007, 0.03)
        sharp_score = normalize_clip(sharp_raw, 20.0, 250.0)
        val_score = 0.5

    border = border_penalty(x1, y1, x2, y2, w, h)
    center = center_score(x1, y1, x2, y2, w, h)

    score = (
        0.30 * conf
        + 0.25 * size_score
        + 0.20 * sharp_score
        + 0.15 * border
        + 0.05 * center
        + 0.05 * val_score
    )

    # Hard low-quality guards
    too_small = (
        (route == "person" and (bw < 24 or bh < 48 or area_ratio < 0.0015))
        or (route == "object" and (bw < 12 or bh < 12 or area_ratio < 0.00025))
    )

    if too_small:
        bucket = "LOW_QUALITY"
    elif score >= 0.62:
        bucket = "GOOD"
    elif score >= 0.45:
        bucket = "LOW_QUALITY"
    else:
        bucket = "REJECT"

    return {
        "quality_score": float(score),
        "sharpness": sharp_raw,
        "area_ratio": float(area_ratio),
        "size_score": float(size_score),
        "sharp_score": float(sharp_score),
        "border_score": float(border),
        "center_score": float(center),
        "quality_bucket": bucket,
        "crop_width": int(bw),
        "crop_height": int(bh),
    }


def temporal_diverse_pick(rows: List[dict], k: int, best=True) -> List[dict]:
    if not rows or k <= 0:
        return []

    ordered = sorted(
        rows,
        key=lambda r: r["quality_score"],
        reverse=best,
    )

    picked = []
    used_frames = []

    frame_vals = [int(r["frame_idx"]) for r in rows]
    span = max(frame_vals) - min(frame_vals) if len(frame_vals) > 1 else 0
    min_gap = max(1, span // max(k * 2, 1))

    for r in ordered:
        fi = int(r["frame_idx"])
        if all(abs(fi - x) >= min_gap for x in used_frames):
            picked.append(r)
            used_frames.append(fi)
            if len(picked) >= k:
                break

    if len(picked) < k:
        for r in ordered:
            if r in picked:
                continue
            picked.append(r)
            if len(picked) >= k:
                break

    return picked


def save_crop(path: Path, crop: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)


def run(video_stem: str, best_n: int, worst_n: int):
    video_path = find_video(video_stem)
    rows = load_tracks(video_stem)

    out_dir = REPORT_ROOT / video_stem
    crop_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_frame: Dict[int, List[dict]] = {}
    for r in rows:
        by_frame.setdefault(int(r["frame_idx"]), []).append(r)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    enriched = []
    frame_keys = sorted(by_frame)

    for i, frame_idx in enumerate(frame_keys, 1):
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

            q = quality_score(r, crop, frame.shape, bb)
            item = dict(r)
            item.update(q)
            item["bbox_clamped"] = [x1, y1, x2, y2]
            item["_crop"] = crop
            enriched.append(item)

        if i == 1 or i % 100 == 0 or i == len(frame_keys):
            print(f"[ANALYZE] frames {i:,}/{len(frame_keys):,} | records={len(enriched):,}")

    cap.release()

    grouped: Dict[Tuple[str, int], List[dict]] = {}
    for r in enriched:
        route = str(r["final_db_route"]).lower()
        tid = int(r["long_track_id"])
        grouped.setdefault((route, tid), []).append(r)

    track_results = []
    for (route, tid), track_rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        best = temporal_diverse_pick(track_rows, best_n, best=True)
        worst = temporal_diverse_pick(track_rows, worst_n, best=False)

        for kind, picks in (("best", best), ("worst", worst)):
            for rank, r in enumerate(picks, 1):
                ext = crop_dir / route / f"track_{tid:04d}" / kind / f"{rank:02d}_frame_{int(r['frame_idx']):06d}.jpg"
                save_crop(ext, r["_crop"])
                r[f"{kind}_crop_path"] = str(ext.resolve())

        bucket_counts = {}
        for r in track_rows:
            b = r["quality_bucket"]
            bucket_counts[b] = bucket_counts.get(b, 0) + 1

        good_count = bucket_counts.get("GOOD", 0)
        low_count = bucket_counts.get("LOW_QUALITY", 0)
        reject_count = bucket_counts.get("REJECT", 0)

        track_bucket = "GOOD"
        if good_count == 0 and low_count > 0:
            track_bucket = "LOW_QUALITY"
        elif good_count == 0 and low_count == 0:
            track_bucket = "REJECT"

        track_results.append({
            "route": route,
            "track_id": tid,
            "records": len(track_rows),
            "frame_start": min(int(r["frame_idx"]) for r in track_rows),
            "frame_end": max(int(r["frame_idx"]) for r in track_rows),
            "time_start": min(float(r.get("timestamp_sec") or 0.0) for r in track_rows),
            "time_end": max(float(r.get("timestamp_sec") or 0.0) for r in track_rows),
            "good": good_count,
            "low_quality": low_count,
            "reject": reject_count,
            "track_bucket": track_bucket,
            "avg_quality": float(np.mean([r["quality_score"] for r in track_rows])),
            "best": best,
            "worst": worst,
        })

    summary = {
        "video_stem": video_stem,
        "video_path": str(video_path.resolve()),
        "total_records": len(enriched),
        "tracks": len(track_results),
        "person_tracks": sum(t["route"] == "person" for t in track_results),
        "object_tracks": sum(t["route"] == "object" for t in track_results),
        "good_tracks": sum(t["track_bucket"] == "GOOD" for t in track_results),
        "low_quality_tracks": sum(t["track_bucket"] == "LOW_QUALITY" for t in track_results),
        "reject_tracks": sum(t["track_bucket"] == "REJECT" for t in track_results),
    }

    json_ready = []
    for t in track_results:
        clean = dict(t)
        clean["best"] = [{k: v for k, v in r.items() if k != "_crop"} for r in t["best"]]
        clean["worst"] = [{k: v for k, v in r.items() if k != "_crop"} for r in t["worst"]]
        json_ready.append(clean)

    (out_dir / "track_quality_summary.json").write_text(
        json.dumps({"summary": summary, "tracks": json_ready}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_html(out_dir, summary, track_results)
    print()
    print("=" * 88)
    print("TRACK QUALITY REPORT COMPLETE")
    print("=" * 88)
    for k, v in summary.items():
        print(f"{k:20s}: {v}")
    print("html_report         :", out_dir / "track_quality_report.html")
    print("=" * 88)


def rel(path_str: str, base: Path) -> str:
    p = Path(path_str)
    try:
        return p.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return p.as_uri() if p.is_absolute() else p.as_posix()


def fmt_time(sec: float):
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:05.2f}"


def crop_card(r: dict, base: Path, label: str):
    path = r.get(f"{label.lower()}_crop_path", "")
    score = float(r.get("quality_score") or 0)
    bucket = r.get("quality_bucket", "")
    return f"""
    <div class="crop-card">
      <img src="{html.escape(rel(path, base))}" alt="crop">
      <div class="crop-meta">
        <b>{html.escape(label)} · frame {int(r.get('frame_idx',0))}</b>
        <span class="bucket {bucket.lower()}">{html.escape(bucket)}</span>
        <div>Q {score:.3f}</div>
        <div>conf {float(r.get('confidence') or 0):.3f}</div>
        <div>{int(r.get('crop_width',0))}×{int(r.get('crop_height',0))}</div>
        <div>sharp {float(r.get('sharpness') or 0):.1f}</div>
      </div>
    </div>
    """


def write_html(out_dir: Path, summary: dict, tracks: List[dict]):
    sections = []
    for t in tracks:
        best_html = "".join(crop_card(r, out_dir, "Best") for r in t["best"])
        worst_html = "".join(crop_card(r, out_dir, "Worst") for r in t["worst"])
        sections.append(f"""
        <section class="track">
          <div class="track-head">
            <div>
              <h2>{t['route'].upper()} Track {t['track_id']}</h2>
              <div class="muted">
                records {t['records']} ·
                frame {t['frame_start']} → {t['frame_end']} ·
                time {fmt_time(t['time_start'])} → {fmt_time(t['time_end'])}
              </div>
            </div>
            <span class="track-bucket {t['track_bucket'].lower()}">{t['track_bucket']}</span>
          </div>

          <div class="stats">
            <span>avg quality <b>{t['avg_quality']:.3f}</b></span>
            <span>GOOD <b>{t['good']}</b></span>
            <span>LOW <b>{t['low_quality']}</b></span>
            <span>REJECT <b>{t['reject']}</b></span>
          </div>

          <h3>Best samples</h3>
          <div class="crop-grid">{best_html}</div>

          <h3>Worst samples</h3>
          <div class="crop-grid worst-grid">{worst_html}</div>
        </section>
        """)

    doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(summary['video_stem'])} · Track Quality Report</title>
<style>
:root {{
  --bg:#09101d; --panel:#121b2d; --line:#293753; --text:#eef4ff; --muted:#99a8c5;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,Arial,sans-serif}}
main{{max-width:1750px;margin:auto;padding:30px}}
h1{{font-size:32px;margin:0 0 8px}}
h2{{margin:0;font-size:24px}} h3{{margin:18px 0 10px}}
.muted{{color:var(--muted)}}
.top-grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:24px 0}}
.metric{{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:16px}}
.metric .k{{color:var(--muted);font-size:12px;text-transform:uppercase}}
.metric .v{{font-size:28px;font-weight:800;margin-top:4px}}
.track{{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;margin:20px 0}}
.track-head{{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}}
.track-bucket,.bucket{{display:inline-block;padding:5px 9px;border-radius:999px;font-size:12px;font-weight:800}}
.good{{background:#153d2e;color:#82f0bd;border:1px solid #2e8c67}}
.low_quality{{background:#433619;color:#ffd27b;border:1px solid #91712e}}
.reject{{background:#451d25;color:#ff98aa;border:1px solid #954150}}
.stats{{display:flex;gap:16px;flex-wrap:wrap;margin:14px 0;color:var(--muted)}}
.crop-grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}}
.worst-grid{{grid-template-columns:repeat(3,minmax(0,1fr))}}
.crop-card{{border:1px solid var(--line);border-radius:13px;overflow:hidden;background:#0a111e}}
.crop-card img{{width:100%;height:330px;object-fit:contain;background:#050912}}
.crop-meta{{padding:10px;font-size:13px;display:grid;grid-template-columns:1fr 1fr;gap:5px}}
.crop-meta b{{grid-column:1/-1}}
@media(max-width:1200px){{.top-grid{{grid-template-columns:repeat(3,1fr)}}.crop-grid{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:700px){{.top-grid,.crop-grid,.worst-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<main>
  <h1>{html.escape(summary['video_stem'])}</h1>
  <div class="muted">{html.escape(summary['video_path'])}</div>

  <div class="top-grid">
    <div class="metric"><div class="k">Records</div><div class="v">{summary['total_records']}</div></div>
    <div class="metric"><div class="k">Tracks</div><div class="v">{summary['tracks']}</div></div>
    <div class="metric"><div class="k">Person</div><div class="v">{summary['person_tracks']}</div></div>
    <div class="metric"><div class="k">Object</div><div class="v">{summary['object_tracks']}</div></div>
    <div class="metric"><div class="k">Good tracks</div><div class="v">{summary['good_tracks']}</div></div>
    <div class="metric"><div class="k">Low/Reject</div><div class="v">{summary['low_quality_tracks'] + summary['reject_tracks']}</div></div>
  </div>

  {''.join(sections)}
</main>
</body>
</html>"""

    (out_dir / "track_quality_report.html").write_text(doc, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-stem", default="Normal_Videos_015_x264")
    ap.add_argument("--best", type=int, default=5)
    ap.add_argument("--worst", type=int, default=3)
    args = ap.parse_args()

    run(args.video_stem, args.best, args.worst)


if __name__ == "__main__":
    main()
