#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
finalize_track_routes.py

validated_tracks.json의 uncertain track을 2차 자동 검증하여
최종 person/object DB route를 확정한다.

2차 검증 특징:
- RF-DETR confidence는 최종 사람 여부 점수에 사용하지 않음
- SigLIP2만 사용
- 더 많은 대표 프레임 사용
- 각 프레임에서 tight crop + context crop 두 시점 평가
- track 전체 median + mean을 robust aggregate
- 최종적으로 review 없이 person/object 중 하나로 자동 route

출력:
- final_routed_tracks.json
- final_route_summary.json
- final_route_report.html
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


POSITIVE_PROMPTS = [
    "a photo of a person",
    "a human being",
    "a pedestrian",
    "a full-body human",
    "a person's head and torso",
    "a person standing or walking",
]

NEGATIVE_PROMPTS = [
    "a photo of an animal",
    "a dog",
    "a cat",
    "a bird",
    "a non-human object",
    "a vehicle",
    "a bag or luggage",
    "an object with no human body",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--report-dir", default="./outputs/final_route_report")
    p.add_argument("--model", default="google/siglip2-base-patch16-224")
    p.add_argument("--device", default="cuda")
    p.add_argument("--samples-per-track", type=int, default=15)

    # Accuracy-first conservative threshold:
    # uncertain track must show reasonably strong human evidence
    # before entering person DB. Otherwise it goes object DB.
    p.add_argument("--person-threshold", type=float, default=0.60)

    p.add_argument("--tight-pad", type=float, default=0.10)
    p.add_argument("--context-pad", type=float, default=0.50)
    return p.parse_args()


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def sample_rows(rows: List[Dict[str, Any]], n: int):
    rows = sorted(rows, key=lambda r: int(r["frame_idx"]))
    if len(rows) <= n:
        return rows

    idxs = np.linspace(0, len(rows) - 1, n).round().astype(int)
    out, seen = [], set()
    for idx in idxs:
        idx = int(idx)
        if idx not in seen:
            seen.add(idx)
            out.append(rows[idx])
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


def load_siglip(model_name, device):
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as e:
        raise RuntimeError(
            "transformers가 없습니다. 설치: pip install transformers sentencepiece"
        ) from e

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return processor, model


@torch.inference_mode()
def siglip_person_score(crop_bgr, processor, model, device) -> float:
    image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    texts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS

    inputs = processor(
        text=texts,
        images=image,
        padding="max_length",
        return_tensors="pt",
    )

    inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    outputs = model(**inputs)

    if hasattr(outputs, "logits_per_image"):
        logits = outputs.logits_per_image[0]
    elif hasattr(outputs, "logits_per_text"):
        logits = outputs.logits_per_text[:, 0]
    else:
        raise RuntimeError("SigLIP logits not found")

    probs = torch.softmax(logits.float(), dim=0)
    n_pos = len(POSITIVE_PROMPTS)
    return float(probs[:n_pos].sum().item())


def robust_track_score(scores: List[float]) -> Tuple[float, float, float, float]:
    if not scores:
        return 0.0, 0.0, 0.0, 0.0

    arr = np.asarray(scores, dtype=np.float32)

    median = float(np.median(arr))
    mean = float(np.mean(arr))
    p75 = float(np.percentile(arr, 75))
    human_vote = float(np.mean(arr >= 0.50))

    # Robust central tendency, not a single lucky frame.
    final = (
        0.50 * median
        + 0.25 * mean
        + 0.15 * p75
        + 0.10 * human_vote
    )
    return float(final), median, mean, human_vote


def make_report_html(summary, report_root: Path):
    rows_html = []

    for r in summary["tracks"]:
        cls = "person" if r["final_db_route"] == "person" else "object"
        rows_html.append(
            f"""
            <tr>
              <td>{r["long_track_id"]}</td>
              <td>{r["source_decision"]}</td>
              <td class="{cls}">{r["final_db_route"]}</td>
              <td>{r["final_person_score"]:.3f}</td>
              <td>{r.get("second_pass_samples", 0)}</td>
              <td>{r.get("resolution", "")}</td>
            </tr>
            """
        )

    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Final Track Route Report</title>
<style>
body{{background:#0f1115;color:#eef1f6;font-family:Segoe UI,Arial,sans-serif;margin:0;padding:24px}}
table{{width:100%;border-collapse:collapse;background:#171a21}}
th,td{{border:1px solid #303644;padding:10px;text-align:center}}
th{{background:#20242d}}
.person{{color:#9ece6a;font-weight:bold}}
.object{{color:#f7768e;font-weight:bold}}
.summary{{display:flex;gap:14px;margin-bottom:18px}}
.card{{background:#171a21;border:1px solid #303644;border-radius:10px;padding:14px;min-width:150px}}
.big{{font-size:24px;font-weight:bold}}
.muted{{color:#9ba3b4}}
</style>
</head>
<body>
<h1>Final Track Route Report</h1>
<div class="summary">
  <div class="card"><div class="muted">Person tracks</div><div class="big">{summary["counts"]["person"]}</div></div>
  <div class="card"><div class="muted">Object tracks</div><div class="big">{summary["counts"]["object"]}</div></div>
  <div class="card"><div class="muted">2nd-pass resolved</div><div class="big">{summary["counts"]["second_pass"]}</div></div>
</div>

<table>
<thead>
<tr>
<th>Long Track</th>
<th>1st Decision</th>
<th>Final Route</th>
<th>Person Score</th>
<th>2nd Pass Samples</th>
<th>Resolution</th>
</tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
</body>
</html>
"""
    (report_root / "final_route_report.html").write_text(html, encoding="utf-8")


def main():
    args = parse_args()

    video_path = Path(args.video).resolve()
    tracks_path = Path(args.tracks).resolve()
    output_path = Path(args.output).resolve()
    report_root = Path(args.report_dir).resolve() / video_path.stem

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    rows = load_json_or_jsonl(tracks_path)

    # group person-origin records by long_track_id
    grouped = defaultdict(list)
    for r in rows:
        if str(r.get("class_name", "")).lower() == "person":
            lid = int(r.get("long_track_id", r.get("track_id")))
            grouped[lid].append(r)

    uncertain_ids = sorted({
        int(r.get("long_track_id", r.get("track_id")))
        for r in rows
        if str(r.get("class_name", "")).lower() == "person"
        and str(r.get("validation_decision", "")).lower() == "uncertain"
    })

    processor = model = None
    cap = None

    second_pass_results = {}

    if uncertain_ids:
        print(f"[INFO] uncertain tracks: {uncertain_ids}")
        print(f"[INFO] loading SigLIP2: {args.model}")

        processor, model = load_siglip(args.model, args.device)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")

        for long_id in uncertain_ids:
            track_rows = sorted(
                grouped[long_id],
                key=lambda r: int(r["frame_idx"])
            )
            sampled = sample_rows(track_rows, args.samples_per_track)

            all_scores = []
            frame_details = []

            for row in sampled:
                frame_idx = int(row["frame_idx"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()

                if not ok:
                    print(f"[WARN] failed frame {frame_idx}")
                    continue

                tight = crop_with_padding(
                    frame, row["bbox"], args.tight_pad
                )
                context = crop_with_padding(
                    frame, row["bbox"], args.context_pad
                )

                view_scores = []

                if tight is not None and tight.size:
                    view_scores.append(
                        siglip_person_score(
                            tight, processor, model, args.device
                        )
                    )

                if context is not None and context.size:
                    view_scores.append(
                        siglip_person_score(
                            context, processor, model, args.device
                        )
                    )

                if not view_scores:
                    continue

                # Both tight and contextual evidence matter.
                frame_score = float(np.mean(view_scores))
                all_scores.append(frame_score)

                frame_details.append({
                    "frame_idx": frame_idx,
                    "timestamp_sec": float(
                        row.get("timestamp_sec", 0.0)
                    ),
                    "view_scores": view_scores,
                    "frame_person_score": frame_score,
                })

            final_score, median, mean, human_vote = robust_track_score(
                all_scores
            )

            final_route = (
                "person"
                if final_score >= args.person_threshold
                else "object"
            )

            second_pass_results[long_id] = {
                "final_person_score": final_score,
                "median_score": median,
                "mean_score": mean,
                "human_vote_ratio": human_vote,
                "final_db_route": final_route,
                "sample_count": len(all_scores),
                "frame_details": frame_details,
            }

            print(
                f"[TRACK {long_id}] "
                f"2nd_pass={final_route:<6} "
                f"score={final_score:.3f} "
                f"median={median:.3f} "
                f"mean={mean:.3f} "
                f"vote={human_vote:.2f}"
            )

        cap.release()

    final_rows = []

    for r in rows:
        out = dict(r)

        if str(r.get("class_name", "")).lower() != "person":
            out["final_db_route"] = "object"
            out["final_validation_decision"] = "non_person"
            out["final_person_score"] = 0.0
            out["route_resolution"] = "original_non_person_class"
            final_rows.append(out)
            continue

        long_id = int(r.get("long_track_id", r.get("track_id")))
        first_decision = str(
            r.get("validation_decision", "")
        ).lower()

        if (
            first_decision == "reject"
            or str(r.get("db_route", "")).lower() == "reject"
            or str(r.get("validation_geometry_gate", "")).lower() == "reject"
        ):
            out["final_db_route"] = "reject"
            out["final_validation_decision"] = "reject"
            out["final_person_score"] = 0.0
            out["route_resolution"] = "geometry_gate_reject"

        elif first_decision == "person":
            out["final_db_route"] = "person"
            out["final_validation_decision"] = "person"
            out["final_person_score"] = float(
                r.get("validation_person_score", 1.0)
            )
            out["route_resolution"] = "first_pass"

        elif first_decision == "non_person":
            out["final_db_route"] = "object"
            out["final_validation_decision"] = "non_person"
            out["final_person_score"] = float(
                r.get("validation_person_score", 0.0)
            )
            out["route_resolution"] = "first_pass"

        elif first_decision == "uncertain":
            res = second_pass_results[long_id]

            out["final_db_route"] = res["final_db_route"]
            out["final_validation_decision"] = (
                "person"
                if res["final_db_route"] == "person"
                else "non_person"
            )
            out["final_person_score"] = float(
                res["final_person_score"]
            )
            out["route_resolution"] = "siglip2_second_pass"

        else:
            # Conservative fallback for malformed/missing validation.
            out["final_db_route"] = "object"
            out["final_validation_decision"] = "non_person"
            out["final_person_score"] = float(
                r.get("validation_person_score", 0.0)
            )
            out["route_resolution"] = "conservative_fallback"

        final_rows.append(out)

    output_path.write_text(
        json.dumps(final_rows, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Track-level summary
    track_summary = []

    all_track_keys = sorted({
        (
            str(r.get("class_name", "")).lower(),
            int(r.get("long_track_id", r.get("track_id")))
        )
        for r in final_rows
    })

    for class_name, lid in all_track_keys:
        subset = [
            r for r in final_rows
            if str(r.get("class_name", "")).lower() == class_name
            and int(r.get("long_track_id", r.get("track_id"))) == lid
        ]

        source_decision = subset[0].get(
            "validation_decision", "non_person"
        )
        route = subset[0]["final_db_route"]
        score = float(subset[0]["final_person_score"])

        track_summary.append({
            "class_name": class_name,
            "long_track_id": lid,
            "source_decision": source_decision,
            "final_db_route": route,
            "final_person_score": score,
            "second_pass_samples": (
                second_pass_results.get(lid, {}).get("sample_count", 0)
                if class_name == "person" else 0
            ),
            "resolution": subset[0].get(
                "route_resolution", ""
            ),
        })

    counts = {
        "person": sum(
            1 for x in track_summary
            if x["final_db_route"] == "person"
        ),
        "object": sum(
            1 for x in track_summary
            if x["final_db_route"] == "object"
        ),
        "reject": sum(
            1 for x in track_summary
            if x["final_db_route"] == "reject"
        ),
        "second_pass": len(second_pass_results),
    }

    summary = {
        "video": str(video_path),
        "source_tracks": str(tracks_path),
        "output_tracks": str(output_path),
        "model": args.model,
        "second_pass_person_threshold": args.person_threshold,
        "counts": counts,
        "tracks": track_summary,
        "second_pass_details": second_pass_results,
    }

    summary_path = report_root / "final_route_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    make_report_html(summary, report_root)

    print()
    print("=" * 72)
    print("FINAL TRACK ROUTING COMPLETE")
    print("=" * 72)
    print(f"person tracks : {counts['person']}")
    print(f"object tracks : {counts['object']}")
    print(f"reject tracks : {counts.get('reject', 0)}")
    print(f"2nd-pass      : {counts['second_pass']}")
    print(f"output        : {output_path}")
    print(f"summary       : {summary_path}")
    print(f"html          : {report_root / 'final_route_report.html'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
