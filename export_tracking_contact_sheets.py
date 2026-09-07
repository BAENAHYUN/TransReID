from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
PERSON_ROOT = ROOT / "data" / "video_tracks" / "person"
OBJECT_ROOT = ROOT / "data" / "video_tracks" / "object"
OUT_ROOT = ROOT / "data" / "validation" / "tracking_visual_review"

PRESETS = [
    ("person", "Normal_Videos_758_x264"),
    ("person", "Normal_Videos_310_x264"),
    ("person", "Normal_Videos_935_x264"),
    ("object", "Normal_Videos_576_x264"),
    ("object", "Normal_Videos_901_x264"),
    ("person", "Normal_Videos_696_x264"),
    ("object", "Normal_Videos_696_x264"),
]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                print(f"[WARN] JSONL parse error {path} line={line_no}: {exc}")
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def track_id(row: dict) -> int | None:
    for key in ("track_id", "original_track_id", "tracker_id"):
        if row.get(key) is not None:
            try:
                return int(row[key])
            except Exception:
                return None
    return None


def frame_idx(row: dict) -> int:
    for key in ("frame_idx", "frame_number", "frame"):
        if row.get(key) is not None:
            try:
                return int(row[key])
            except Exception:
                pass
    return -1


def crop_path(row: dict) -> Path | None:
    raw = row.get("crop_path") or row.get("path")
    if not raw:
        return None
    p = Path(str(raw))
    if not p.is_absolute():
        p = ROOT / p
    return p


def evenly_sample(items: List[dict], k: int) -> List[dict]:
    if len(items) <= k:
        return items[:]
    if k <= 1:
        return [items[len(items) // 2]]

    idxs = []
    for i in range(k):
        pos = round(i * (len(items) - 1) / (k - 1))
        idxs.append(pos)

    seen = set()
    out = []
    for idx in idxs:
        if idx not in seen:
            out.append(items[idx])
            seen.add(idx)
    return out


def fit_image(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    img = img.convert("RGB")
    fitted = ImageOps.contain(img, (box_w, box_h))
    canvas = Image.new("RGB", (box_w, box_h), "white")
    x = (box_w - fitted.width) // 2
    y = (box_h - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def label_for_track(kind: str, tid: int, items: List[dict], class_summary: dict) -> str:
    if kind == "person":
        return f"track_{tid:04d} | rows={len(items)}"

    info = class_summary.get(str(tid), {}) if isinstance(class_summary, dict) else {}
    final_class = info.get("final_class_name") or "unknown"
    counts = info.get("class_counts") or {}
    conf_sums = info.get("class_confidence_sums") or {}

    if counts:
        counts_txt = ", ".join(f"{k}:{v}" for k, v in counts.items())
    else:
        counts_txt = "-"

    if conf_sums:
        top = sorted(
            ((str(k), float(v)) for k, v in conf_sums.items()),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        conf_txt = ", ".join(f"{k}:{v:.2f}" for k, v in top)
    else:
        conf_txt = "-"

    return (
        f"track_{tid:04d} | rows={len(items)} | final={final_class}\n"
        f"votes={counts_txt}\n"
        f"confidence_sum(top)={conf_txt}"
    )


def make_track_sheet(
    kind: str,
    tid: int,
    items: List[dict],
    class_summary: dict,
    out_path: Path,
    samples_per_track: int,
):
    samples = evenly_sample(items, samples_per_track)

    tile_w = 260
    tile_h = 280
    image_h = 220
    header_h = 95

    width = max(1, len(samples)) * tile_w
    height = header_h + tile_h

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    title = label_for_track(kind, tid, items, class_summary)
    draw.multiline_text((10, 8), title, fill="black", font=font, spacing=4)

    for i, row in enumerate(samples):
        x0 = i * tile_w
        p = crop_path(row)
        fi = frame_idx(row)

        if p is not None and p.exists():
            try:
                img = Image.open(p)
                panel = fit_image(img, tile_w - 20, image_h)
                canvas.paste(panel, (x0 + 10, header_h))
            except Exception:
                draw.rectangle(
                    [x0 + 10, header_h, x0 + tile_w - 10, header_h + image_h],
                    outline="black",
                )
                draw.text((x0 + 20, header_h + 90), "IMAGE LOAD ERROR", fill="black", font=font)
        else:
            draw.rectangle(
                [x0 + 10, header_h, x0 + tile_w - 10, header_h + image_h],
                outline="black",
            )
            draw.text((x0 + 20, header_h + 90), "MISSING CROP", fill="black", font=font)

        draw.text(
            (x0 + 10, header_h + image_h + 5),
            f"sample {i+1}/{len(samples)} | frame={fi}",
            fill="black",
            font=font,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def make_overview(
    kind: str,
    title: str,
    track_items: List[Tuple[int, List[dict]]],
    class_summary: dict,
    out_path: Path,
    max_tracks: int,
):
    selected = track_items[:max_tracks]

    tile_w = 220
    tile_h = 270
    crop_h = 190
    cols = 5
    rows = max(1, math.ceil(len(selected) / cols))
    header_h = 55

    canvas = Image.new("RGB", (cols * tile_w, header_h + rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 10), title, fill="black", font=font)

    for idx, (tid, items) in enumerate(selected):
        r = idx // cols
        c = idx % cols
        x0 = c * tile_w
        y0 = header_h + r * tile_h

        middle = items[len(items) // 2]
        p = crop_path(middle)

        if p is not None and p.exists():
            try:
                img = Image.open(p)
                panel = fit_image(img, tile_w - 16, crop_h)
                canvas.paste(panel, (x0 + 8, y0 + 8))
            except Exception:
                pass

        if kind == "object":
            info = class_summary.get(str(tid), {}) if isinstance(class_summary, dict) else {}
            final_class = info.get("final_class_name") or "unknown"
            txt = f"T{tid} | n={len(items)} | {final_class}"
        else:
            txt = f"T{tid} | n={len(items)}"

        draw.text((x0 + 8, y0 + crop_h + 14), txt, fill="black", font=font)
        draw.text(
            (x0 + 8, y0 + crop_h + 34),
            f"frame={frame_idx(middle)}",
            fill="black",
            font=font,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def select_tracks(
    by_track: Dict[int, List[dict]],
    kind: str,
    class_summary: dict,
    detail_limit: int,
):
    ordered = [(tid, sorted(items, key=frame_idx)) for tid, items in by_track.items()]

    shortest = sorted(ordered, key=lambda x: (len(x[1]), x[0]))
    longest = sorted(ordered, key=lambda x: (-len(x[1]), x[0]))

    selected = []
    seen = set()

    # 1) object: ambiguous class-voting tracks first
    if kind == "object":
        ambiguous = []
        for tid, items in ordered:
            info = class_summary.get(str(tid), {}) if isinstance(class_summary, dict) else {}
            counts = info.get("class_counts") or {}
            if isinstance(counts, dict) and len(counts) > 1:
                vals = sorted((int(v) for v in counts.values()), reverse=True)
                total = sum(vals)
                dom = vals[0] / total if total else 1.0
                close = len(vals) >= 2 and vals[1] > 0 and (vals[0] / vals[1] <= 1.5)
                if dom < 0.70 or close:
                    ambiguous.append((tid, items))

        for item in ambiguous:
            if item[0] not in seen:
                selected.append(("ambiguous_vote", item))
                seen.add(item[0])

    # 2) shortest tracks -> fragmentation inspection
    shortest_quota = max(10, detail_limit // 2)
    for item in shortest[:shortest_quota]:
        if len(selected) >= detail_limit:
            break
        if item[0] not in seen:
            selected.append(("short", item))
            seen.add(item[0])

    # 3) longest tracks -> ID mixing inspection
    for item in longest:
        if len(selected) >= detail_limit:
            break
        if item[0] not in seen:
            selected.append(("long", item))
            seen.add(item[0])

    return ordered, shortest, longest, selected


def process_one(
    kind: str,
    video: str,
    samples_per_track: int,
    detail_limit: int,
    overview_limit: int,
):
    root = PERSON_ROOT if kind == "person" else OBJECT_ROOT
    video_dir = root / video

    if not video_dir.exists():
        print(f"[WARN] missing: {video_dir}")
        return []

    tracks_path = video_dir / "tracks.jsonl"
    rows = load_jsonl(tracks_path)
    class_summary = load_json(video_dir / "track_class_summary.json") if kind == "object" else {}

    by_track: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        tid = track_id(row)
        if tid is not None:
            by_track[tid].append(row)

    if not by_track:
        print(f"[WARN] no tracks: {kind} {video}")
        return []

    ordered, shortest, longest, selected = select_tracks(
        by_track=by_track,
        kind=kind,
        class_summary=class_summary,
        detail_limit=detail_limit,
    )

    out_dir = OUT_ROOT / kind / video
    out_dir.mkdir(parents=True, exist_ok=True)

    make_overview(
        kind,
        f"{kind.upper()} {video} - SHORTEST TRACKS (fragmentation review)",
        shortest,
        class_summary,
        out_dir / "00_overview_shortest_tracks.jpg",
        overview_limit,
    )

    make_overview(
        kind,
        f"{kind.upper()} {video} - LONGEST TRACKS (ID mixing review)",
        longest,
        class_summary,
        out_dir / "01_overview_longest_tracks.jpg",
        overview_limit,
    )

    report_rows = []

    for reason, (tid, items) in selected:
        out_name = f"{reason}_track_{tid:04d}.jpg"
        make_track_sheet(
            kind=kind,
            tid=tid,
            items=items,
            class_summary=class_summary,
            out_path=out_dir / out_name,
            samples_per_track=samples_per_track,
        )

        info = class_summary.get(str(tid), {}) if kind == "object" else {}
        report_rows.append({
            "kind": kind,
            "video": video,
            "track_id": tid,
            "reason": reason,
            "rows": len(items),
            "first_frame": frame_idx(items[0]),
            "last_frame": frame_idx(items[-1]),
            "final_class": info.get("final_class_name") if isinstance(info, dict) else "",
            "class_counts": json.dumps(info.get("class_counts", {}), ensure_ascii=False)
                if isinstance(info, dict) else "",
            "sheet": str((out_dir / out_name).resolve()),
        })

    with (out_dir / "selected_tracks.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "kind", "video", "track_id", "reason", "rows",
            "first_frame", "last_frame", "final_class", "class_counts", "sheet"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(report_rows)

    print(
        f"[OK] {kind:6s} {video} | "
        f"tracks={len(by_track)} selected={len(report_rows)} "
        f"-> {out_dir}"
    )

    return report_rows


def main():
    ap = argparse.ArgumentParser(
        description="의심 영상의 Person/Object track을 contact sheet로 시각 검증"
    )
    ap.add_argument("--video", default=None, help="영상 stem 1개만 검사")
    ap.add_argument("--type", choices=["person", "object", "both"], default="both")
    ap.add_argument("--samples-per-track", type=int, default=5)
    ap.add_argument("--detail-limit", type=int, default=30)
    ap.add_argument("--overview-limit", type=int, default=60)
    args = ap.parse_args()

    jobs = []

    if args.video:
        if args.type in ("person", "both"):
            jobs.append(("person", args.video))
        if args.type in ("object", "both"):
            jobs.append(("object", args.video))
    else:
        jobs = PRESETS[:]

    print("=" * 88)
    print("TRACKING VISUAL REVIEW EXPORT")
    print("=" * 88)
    print("output root:", OUT_ROOT)
    print("jobs       :", len(jobs))
    print()

    all_rows = []
    for kind, video in jobs:
        all_rows.extend(
            process_one(
                kind=kind,
                video=video,
                samples_per_track=args.samples_per_track,
                detail_limit=args.detail_limit,
                overview_limit=args.overview_limit,
            )
        )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_csv = OUT_ROOT / "selected_tracks_all.csv"

    if all_rows:
        with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
            fields = [
                "kind", "video", "track_id", "reason", "rows",
                "first_frame", "last_frame", "final_class", "class_counts", "sheet"
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)

    print("\nDONE")
    print("summary:", summary_csv)
    print("\n먼저 각 영상 폴더의")
    print("  00_overview_shortest_tracks.jpg")
    print("  01_overview_longest_tracks.jpg")
    print("두 장부터 확인하세요.")


if __name__ == "__main__":
    main()
