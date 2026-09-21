#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--output-root", default="./outputs/track_validation")
    p.add_argument("--samples-per-track", type=int, default=5)
    p.add_argument("--pad-ratio", type=float, default=0.12)
    return p.parse_args()


def load_tracks(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("tracks"), list):
            return obj["tracks"]
    except Exception:
        pass

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def sample_rows(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    rows = sorted(rows, key=lambda r: int(r["frame_idx"]))
    if len(rows) <= n:
        return rows
    idxs = np.linspace(0, len(rows) - 1, n).round().astype(int)
    out, seen = [], set()
    for i in idxs:
        i = int(i)
        if i not in seen:
            seen.add(i)
            out.append(rows[i])
    return out


def crop_with_padding(frame, bbox, pad_ratio):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, bbox)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    px = bw * pad_ratio
    py = bh * pad_ratio

    xx1 = max(0, int(np.floor(x1 - px)))
    yy1 = max(0, int(np.floor(y1 - py)))
    xx2 = min(w, int(np.ceil(x2 + px)))
    yy2 = min(h, int(np.ceil(y2 + py)))

    if xx2 <= xx1 or yy2 <= yy1:
        return None

    return frame[yy1:yy2, xx1:xx2].copy()


def make_html(manifest):
    cards = []

    for tr in manifest["tracks"]:
        sample_html = []
        for s in tr["samples"]:
            rel = s["crop_relpath"].replace("\\", "/")
            sample_html.append(
                '<div class="sample">'
                f'<img src="{rel}">'
                f'<div class="cap">frame {s["frame_idx"]} · '
                f'{s["timestamp_sec"]:.2f}s · conf {s["confidence"]:.3f}</div>'
                '</div>'
            )

        short_ids = ", ".join(str(x) for x in tr["short_track_ids"])
        tid = tr["long_track_id"]

        cards.append(
            f'<section class="track-card" data-track="{tid}">'
            '<div class="head"><div>'
            f'<h2>Long Track #{tid}</h2>'
            f'<div class="meta">Short IDs: {short_ids} · '
            f'detections: {tr["num_detections"]} · '
            f'frames: {tr["frame_start"]} → {tr["frame_end"]}</div>'
            '</div><div class="buttons">'
            f'<button class="person" onclick="markTrack({tid},\'person\',this)">사람</button>'
            f'<button class="notperson" onclick="markTrack({tid},\'not_person\',this)">사람 아님</button>'
            f'<button class="uncertain" onclick="markTrack({tid},\'uncertain\',this)">보류</button>'
            '</div></div>'
            f'<div class="samples">{"".join(sample_html)}</div>'
            '</section>'
        )

    video_json = json.dumps(manifest["video"], ensure_ascii=False)
    tracks_json = json.dumps(manifest["tracks_file"], ensure_ascii=False)

    html = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Track Validation</title>
<style>
body{margin:0;background:#0f1115;color:#eef1f6;font-family:Segoe UI,Arial,sans-serif}
header{position:sticky;top:0;background:#0f1115ee;border-bottom:1px solid #303644;padding:16px 22px;z-index:20}
.wrap{max-width:1500px;margin:auto;padding:18px}
.track-card{background:#171a21;border:1px solid #303644;border-radius:12px;margin-bottom:18px;overflow:hidden}
.head{display:flex;justify-content:space-between;align-items:center;gap:15px;padding:14px 16px;border-bottom:1px solid #303644}
h1,h2{margin:0} h2{font-size:18px}
.meta{color:#9ba3b4;font-size:13px;margin-top:5px}
.samples{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;padding:12px}
.sample{background:#0b0d12;border:1px solid #303644;border-radius:8px;overflow:hidden}
.sample img{width:100%;height:250px;object-fit:contain;background:#050609}
.cap{font-size:12px;color:#aab2c0;padding:7px}
button{border:0;border-radius:7px;padding:9px 12px;font-weight:700;cursor:pointer;margin-left:5px}
.person{background:#9ece6a;color:#10130c}
.notperson{background:#f7768e;color:white}
.uncertain{background:#e0af68;color:#15120b}
.selected{outline:3px solid white}
.toolbar{display:flex;gap:10px;align-items:center;margin-top:8px}
#status{color:#9ba3b4;font-size:13px}
</style>
</head>
<body>
<header>
<h1>DB 구축 전 Track Validation</h1>
<div class="toolbar">
<button onclick="downloadResult()">검증 결과 JSON 저장</button>
<span id="status">각 Long Track을 사람 / 사람 아님 / 보류로 표시하세요.</span>
</div>
</header>
<div class="wrap">
""" + "".join(cards) + """
</div>
<script>
const decisions = {};

function markTrack(id, decision, btn){
  decisions[String(id)] = decision;
  const card = btn.closest('.track-card');
  card.querySelectorAll('button').forEach(x => x.classList.remove('selected'));
  btn.classList.add('selected');
  document.getElementById('status').textContent =
    `현재 ${Object.keys(decisions).length}개 Track 검증됨`;
}

function downloadResult(){
  const result = {
    source_video: VIDEO_PATH,
    source_tracks: TRACKS_PATH,
    decisions
  };
  const blob = new Blob([JSON.stringify(result,null,2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'track_validation_result.json';
  a.click();
  URL.revokeObjectURL(a.href);
}

const VIDEO_PATH = __VIDEO__;
const TRACKS_PATH = __TRACKS__;
</script>
</body>
</html>
"""
    html = html.replace("__VIDEO__", video_json)
    html = html.replace("__TRACKS__", tracks_json)
    return html


def main():
    args = parse_args()

    video_path = Path(args.video).resolve()
    tracks_path = Path(args.tracks).resolve()
    out_root = Path(args.output_root).resolve() / video_path.stem
    crops_root = out_root / "crops"

    out_root.mkdir(parents=True, exist_ok=True)
    crops_root.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks(tracks_path)

    person_rows = [
        r for r in tracks
        if str(r.get("class_name", "")).lower() == "person"
    ]

    grouped = defaultdict(list)
    for r in person_rows:
        lid = int(r.get("long_track_id", r.get("track_id")))
        grouped[lid].append(r)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    manifest_tracks = []

    for long_id in sorted(grouped):
        rows = sorted(grouped[long_id], key=lambda r: int(r["frame_idx"]))
        samples = sample_rows(rows, args.samples_per_track)

        track_dir = crops_root / f"long_{long_id:04d}"
        track_dir.mkdir(parents=True, exist_ok=True)

        sample_meta = []

        for i, row in enumerate(samples, 1):
            frame_idx = int(row["frame_idx"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()

            if not ok:
                print(f"[WARN] failed to read frame {frame_idx}")
                continue

            crop = crop_with_padding(frame, row["bbox"], args.pad_ratio)
            if crop is None or crop.size == 0:
                print(f"[WARN] empty crop frame={frame_idx}")
                continue

            img_path = track_dir / f"sample_{i:02d}_frame_{frame_idx:08d}.jpg"
            cv2.imwrite(str(img_path), crop)

            sample_meta.append({
                "frame_idx": frame_idx,
                "timestamp_sec": float(row.get("timestamp_sec", 0.0)),
                "confidence": float(row.get("confidence", 0.0)),
                "bbox": row["bbox"],
                "crop_relpath": str(img_path.relative_to(out_root)),
            })

        manifest_tracks.append({
            "long_track_id": long_id,
            "short_track_ids": sorted({
                int(r.get("short_track_id", r.get("track_id")))
                for r in rows
            }),
            "num_detections": len(rows),
            "frame_start": int(rows[0]["frame_idx"]),
            "frame_end": int(rows[-1]["frame_idx"]),
            "time_start": float(rows[0].get("timestamp_sec", 0.0)),
            "time_end": float(rows[-1].get("timestamp_sec", 0.0)),
            "samples": sample_meta,
        })

    cap.release()

    manifest = {
        "video": str(video_path),
        "tracks_file": str(tracks_path),
        "samples_per_track": args.samples_per_track,
        "tracks": manifest_tracks,
    }

    (out_root / "validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    (out_root / "person_candidates.json").write_text(
        json.dumps(manifest_tracks, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    (out_root / "rejected_candidates.json").write_text(
        "[]",
        encoding="utf-8"
    )

    (out_root / "track_validation.html").write_text(
        make_html(manifest),
        encoding="utf-8"
    )

    print("=" * 72)
    print("TRACK VALIDATION PREP COMPLETE")
    print("=" * 72)
    print(f"person candidate tracks : {len(manifest_tracks)}")
    print(f"output                  : {out_root}")
    print(f"html                    : {out_root / 'track_validation.html'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
