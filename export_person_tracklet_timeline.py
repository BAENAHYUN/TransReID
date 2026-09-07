from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
PERSON_ROOT = ROOT / "data" / "video_tracks" / "person"
OUT_ROOT = ROOT / "data" / "validation" / "person_tracklet_timeline_review"


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                print(f"[WARN] {path}:{line_no}: {exc}")
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def safe_int(v, default=-1):
    try:
        return int(v)
    except Exception:
        return default


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def frame_idx(row):
    for key in ("frame_idx", "frame_number", "frame"):
        if row.get(key) is not None:
            return safe_int(row.get(key), -1)
    return -1


def timestamp(row):
    for key in ("timestamp_sec", "time_sec"):
        if row.get(key) is not None:
            return safe_float(row.get(key), 0.0)
    return 0.0


def crop_path(row):
    raw = row.get("crop_path") or row.get("path")
    if not raw:
        return None
    p = Path(str(raw))
    if not p.is_absolute():
        p = ROOT / p
    return p


def evenly_spaced(rows, n):
    rows = sorted(rows, key=lambda r: (timestamp(r), frame_idx(r)))
    if len(rows) <= n:
        return rows
    if n <= 1:
        return [rows[len(rows)//2]]

    idxs = []
    for i in range(n):
        idx = round(i * (len(rows) - 1) / (n - 1))
        if idx not in idxs:
            idxs.append(idx)
    return [rows[i] for i in idxs]


def fit(im, w, h):
    im = im.convert("RGB")
    out = ImageOps.contain(im, (w, h))
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(out, ((w-out.width)//2, (h-out.height)//2))
    return canvas


def make_sheet(video, tid, rows, out_dir, samples):
    chosen = evenly_spaced(rows, samples)

    cols = 4
    tile_w = 300
    tile_h = 330
    image_h = 245
    header_h = 75

    nrows = max(1, math.ceil(len(chosen) / cols))
    canvas = Image.new(
        "RGB",
        (cols * tile_w, header_h + nrows * tile_h),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text(
        (10, 8),
        f"{video} | RAW track_{tid:04d} | rows={len(rows)} | sampled={len(chosen)}",
        fill="black",
        font=font,
    )
    draw.text(
        (10, 30),
        "TRACKLET PURITY CHECK: all tiles should show one visually consistent person.",
        fill="black",
        font=font,
    )

    for i, row in enumerate(chosen):
        rr, cc = divmod(i, cols)
        x0 = cc * tile_w
        y0 = header_h + rr * tile_h

        p = crop_path(row)
        if p and p.exists():
            try:
                with Image.open(p) as im:
                    panel = fit(im, tile_w - 20, image_h)
                canvas.paste(panel, (x0 + 10, y0 + 5))
            except Exception as exc:
                draw.text((x0 + 10, y0 + 20), str(exc)[:40], fill="black", font=font)

        draw.text(
            (x0 + 10, y0 + image_h + 15),
            f"frame={frame_idx(row)}  t={timestamp(row):.1f}s",
            fill="black",
            font=font,
        )
        draw.text(
            (x0 + 10, y0 + image_h + 35),
            p.name if p else "crop_path=N/A",
            fill="black",
            font=font,
        )

    out = out_dir / f"track_{tid:04d}_timeline.jpg"
    canvas.save(out, quality=95)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="Normal_Videos_439_x264")
    ap.add_argument(
        "--tracks",
        nargs="+",
        type=int,
        default=[5, 6, 9, 35, 30, 32],
    )
    ap.add_argument("--samples", type=int, default=12)
    args = ap.parse_args()

    video_dir = PERSON_ROOT / args.video
    tracks_path = video_dir / "tracks.jsonl"
    if not tracks_path.exists():
        raise FileNotFoundError(tracks_path)

    rows = load_jsonl(tracks_path)
    grouped = defaultdict(list)
    for row in rows:
        tid = safe_int(row.get("track_id"), -1)
        if tid >= 0:
            grouped[tid].append(row)

    out_dir = OUT_ROOT / args.video
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("RAW PERSON TRACKLET TIMELINE REVIEW")
    print("=" * 92)
    print("video :", args.video)
    print("input :", tracks_path)
    print("output:", out_dir)

    for tid in args.tracks:
        if tid not in grouped:
            print(f"[MISS] track_{tid:04d}")
            continue
        out = make_sheet(
            args.video,
            tid,
            grouped[tid],
            out_dir,
            args.samples,
        )
        print(f"[OK] track_{tid:04d} rows={len(grouped[tid])} -> {out.name}")

    print("\nInterpretation:")
    print("  same visual person across timeline -> tracklet internally clean")
    print("  obvious appearance/person switch   -> ByteTrack ID switch / impure tracklet")
    print("  IMPORTANT: impure raw tracklets must be split before stitching.")


if __name__ == "__main__":
    main()
