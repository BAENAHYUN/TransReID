from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
PERSON_ROOT = ROOT / "data" / "video_tracks" / "person"
OUT_ROOT = ROOT / "data" / "validation" / "person_stitch_v4_5_visual_review"

DEFAULT_VIDEO = "Normal_Videos_924_x264"


def safe_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def safe_float(v: Any, default=None):
    try:
        return float(v)
    except Exception:
        return default


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                print(f"[WARN] parse error {path.name}:{line_no}: {exc}")
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def crop_path(row: dict) -> Path | None:
    raw = row.get("crop_path") or row.get("path")
    if not raw:
        return None
    p = Path(str(raw))
    if not p.is_absolute():
        p = ROOT / p
    return p


def frame_idx(row: dict) -> int:
    for key in ("frame_idx", "frame_number", "frame"):
        if row.get(key) is not None:
            return safe_int(row.get(key), -1)
    return -1


def timestamp(row: dict) -> float:
    for key in ("timestamp_sec", "time_sec"):
        if row.get(key) is not None:
            x = safe_float(row.get(key))
            if x is not None:
                return x
    return 0.0


def fit(img: Image.Image, w: int, h: int) -> Image.Image:
    img = img.convert("RGB")
    fitted = ImageOps.contain(img, (w, h))
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(fitted, ((w - fitted.width)//2, (h - fitted.height)//2))
    return canvas


def representative_row(rows: List[dict]) -> dict:
    rows = sorted(rows, key=lambda r: (timestamp(r), frame_idx(r)))
    return rows[len(rows)//2] if rows else {}


def first_meta(rows: List[dict]) -> dict:
    rows = sorted(rows, key=lambda r: (timestamp(r), frame_idx(r)))
    return rows[0] if rows else {}


def fmt(v: Any, digits: int = 3) -> str:
    x = safe_float(v)
    return "-" if x is None else f"{x:.{digits}f}"


def make_sheet(video: str, pid: int, track_rows: Dict[int, List[dict]], out_dir: Path):
    tids = sorted(
        track_rows,
        key=lambda tid: (
            timestamp(sorted(track_rows[tid], key=timestamp)[0]),
            tid
        ),
    )

    cols = 4
    tile_w = 340
    tile_h = 410
    image_h = 235
    header_h = 100
    tracks_per_page = 16
    pages = max(1, math.ceil(len(tids) / tracks_per_page))
    font = ImageFont.load_default()

    for page in range(pages):
        page_tids = tids[page*tracks_per_page:(page+1)*tracks_per_page]
        nrows = max(1, math.ceil(len(page_tids)/cols))

        canvas = Image.new("RGB", (cols*tile_w, header_h+nrows*tile_h), "white")
        draw = ImageDraw.Draw(canvas)

        draw.text(
            (10, 8),
            f"{video} | V4.5 person_{pid:04d} | tracklets={len(tids)} | page={page+1}/{pages}",
            fill="black", font=font
        )
        draw.text(
            (10, 30),
            "FALSE MERGE CHECK: every tile should be the SAME real person.",
            fill="black", font=font
        )
        draw.text(
            (10, 52),
            "Check: rule / fused / global / anchor / gallery / endpoint / gap",
            fill="black", font=font
        )

        for i, tid in enumerate(page_tids):
            rows = track_rows[tid]
            rep = representative_row(rows)
            meta = first_meta(rows)

            rr = i // cols
            cc = i % cols
            x0 = cc*tile_w
            y0 = header_h + rr*tile_h

            p = crop_path(rep)
            if p is not None and p.exists():
                try:
                    panel = fit(Image.open(p), tile_w-20, image_h)
                    canvas.paste(panel, (x0+10, y0+6))
                except Exception:
                    pass

            from_tid = meta.get("stitched_from_track_id")
            rule = meta.get("stitch_rule") or "seed"

            lines = [
                f"T{tid} n={len(rows)} frame={frame_idx(rep)} t={timestamp(rep):.1f}s from={from_tid if from_tid is not None else '-'}",
                f"rule={rule} fused={fmt(meta.get('stitch_fused_score'))} gap={fmt(meta.get('stitch_gap_sec'),1)}s",
                f"global={fmt(meta.get('stitch_global_similarity'))} gmax={fmt(meta.get('stitch_global_max_similarity'))} anchor={fmt(meta.get('stitch_anchor_global_similarity'))}",
                f"gallery={fmt(meta.get('stitch_gallery_global_topk_mean'))} epTopK={fmt(meta.get('stitch_endpoint_topk_mean'))} epMax={fmt(meta.get('stitch_endpoint_max'))}",
                f"margin={fmt(meta.get('stitch_identity_margin'))} scale={fmt(meta.get('stitch_bbox_scale_ratio'))}",
            ]

            base_y = y0 + image_h + 18
            for j, text in enumerate(lines):
                draw.text((x0+10, base_y+j*21), text[:82], fill="black", font=font)

        out = out_dir / f"person_{pid:04d}_v4_5_page_{page+1:02d}.jpg"
        canvas.save(out, quality=94)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument(
        "--all",
        action="store_true",
        help="include singleton identities too",
    )
    ap.add_argument(
        "--min-tracklets",
        type=int,
        default=3,
        help=(
            "minimum tracklets per stitched identity to export "
            "(default: 3 for first-pass high-risk review)"
        ),
    )
    ap.add_argument(
        "--all-merged",
        action="store_true",
        help="export every merged identity (equivalent to --min-tracklets 2)",
    )
    ap.add_argument(
        "--person-ids",
        nargs="*",
        type=int,
        default=None,
        help="optional exact stitched person IDs to export",
    )
    args = ap.parse_args()

    video_dir = PERSON_ROOT / args.video
    stitched_path = video_dir / "stitched_tracks_v4_5.jsonl"
    if not stitched_path.exists():
        raise FileNotFoundError(f"V4.5 stitched output not found: {stitched_path}")

    rows = load_jsonl(stitched_path)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pid = safe_int(row.get("person_id"))
        tid = safe_int(row.get("original_track_id"))
        if pid >= 0 and tid >= 0:
            grouped[pid][tid].append(row)

    if args.person_ids:
        selected = [
            pid for pid in args.person_ids
            if pid in grouped
        ]
    else:
        min_tracklets = 1 if args.all else (2 if args.all_merged else args.min_tracklets)
        selected = [
            pid for pid, track_rows in grouped.items()
            if len(track_rows) >= min_tracklets
        ]

    selected.sort(
        key=lambda pid: (-len(grouped[pid]), pid)
    )

    out_dir = OUT_ROOT / args.video
    out_dir.mkdir(parents=True, exist_ok=True)

    report = []

    print("=" * 92)
    print("PERSON STITCHING V4.5 VISUAL REVIEW")
    print("=" * 92)
    print("video :", args.video)
    print("input :", stitched_path)
    print("output:", out_dir)
    print("selected identities:", len(selected))
    print(
        "selection mode    :",
        (
            f"person_ids={args.person_ids}"
            if args.person_ids
            else (
                "all"
                if args.all
                else (
                    "all merged"
                    if args.all_merged
                    else f"tracklets >= {args.min_tracklets}"
                )
            )
        ),
    )
    print()

    for pid in selected:
        track_rows = grouped[pid]
        make_sheet(args.video, pid, track_rows, out_dir)
        tids = sorted(track_rows)

        merge_meta = []
        for tid in tids:
            meta = first_meta(track_rows[tid])
            if meta.get("stitch_rule"):
                merge_meta.append(meta)

        gaps = [
            safe_float(m.get("stitch_gap_sec"))
            for m in merge_meta
            if safe_float(m.get("stitch_gap_sec")) is not None
        ]
        fused = [
            safe_float(m.get("stitch_fused_score"))
            for m in merge_meta
            if safe_float(m.get("stitch_fused_score")) is not None
        ]

        report.append({
            "video": args.video,
            "person_id": pid,
            "tracklet_count": len(tids),
            "merge_edge_count": max(0, len(tids) - 1),
            "track_ids": ",".join(map(str, tids)),
            "max_gap_sec": "" if not gaps else round(max(gaps), 3),
            "min_fused_score": "" if not fused else round(min(fused), 6),
        })
        print(f"[OK] person_{pid:04d} tracklets={len(tids)} tracks={tids}")

    report_path = out_dir / "selected_person_ids.csv"
    with report_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "person_id",
                "tracklet_count",
                "merge_edge_count",
                "track_ids",
                "max_gap_sec",
                "min_fused_score",
            ]
        )
        w.writeheader()
        w.writerows(report)

    print("\nDONE")
    print("report:", report_path)
    print("\n판정:")
    print("  모든 타일 같은 사람 -> 정상")
    print("  다른 사람 하나라도 섞임 -> False Merge")


if __name__ == "__main__":
    main()