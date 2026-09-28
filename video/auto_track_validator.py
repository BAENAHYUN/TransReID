#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
auto_track_validator.py

DB 구축 전 자동 Track Validation.

입력:
- stitched_tracks.json
- 원본 video

동작:
1) person으로 분류된 각 long_track_id에서 대표 crop 여러 장 추출
2) SigLIP2 image-text scoring으로 person / non_person 자동 판별
3) track 단위 score 집계
4) stitched track 레코드에 validation 결과 추가
5) validated_tracks.json 생성
6) uncertain track만 HTML review 대상으로 저장

기본 모델:
  google/siglip2-base-patch16-224

첫 실행에서 모델이 로컬 캐시에 없으면 다운로드가 필요할 수 있음.
이미 캐시된 모델이면 이후 완전 로컬 실행 가능.

실행 예:
python .\auto_track_validator.py `
  --video ".\data\videos\Normal_Videos_003_x264.mp4" `
  --tracks ".\outputs\video_tracks\stitched_tracks.json" `
  --output ".\outputs\video_tracks\validated_tracks.json" `
  --review-dir ".\outputs\track_validation_auto"
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image


POSITIVE_PROMPTS = [
    "a photo of a person",
    "a human being",
    "a pedestrian",
]

NEGATIVE_PROMPTS = [
    "a photo of an animal",
    "a dog",
    "a cat",
    "a non-human object",
    "a vehicle",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--review-dir", default="./outputs/track_validation_auto")
    p.add_argument("--model", default="google/siglip2-base-patch16-224")
    p.add_argument("--device", default="cuda")
    p.add_argument("--samples-per-track", type=int, default=7)
    p.add_argument("--person-threshold", type=float, default=0.65)
    p.add_argument("--non-person-threshold", type=float, default=0.35)
    p.add_argument("--siglip-weight", type=float, default=0.85)
    p.add_argument("--detector-weight", type=float, default=0.15)
    p.add_argument("--pad-ratio", type=float, default=0.12)
    p.add_argument("--min-person-width", type=float, default=16.0)
    p.add_argument("--min-person-height", type=float, default=40.0)
    p.add_argument("--min-person-area", type=float, default=800.0)
    p.add_argument("--min-person-records", type=int, default=2)
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


def sample_rows(rows: List[Dict[str, Any]], n: int):
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



def geometry_gate_track(rows, args):
    widths, heights, areas = [], [], []

    for r in rows:
        bbox = r.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(float, bbox)
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)

        if bw <= 0.0 or bh <= 0.0:
            continue

        widths.append(bw)
        heights.append(bh)
        areas.append(bw * bh)

    if not widths:
        return False, {
            "passed": False,
            "reason": "no_valid_bbox",
            "record_count": len(rows),
        }

    med_w = float(np.median(widths))
    med_h = float(np.median(heights))
    med_area = float(np.median(areas))

    reasons = []

    if len(rows) < int(args.min_person_records):
        reasons.append(f"records<{int(args.min_person_records)}")
    if med_w < float(args.min_person_width):
        reasons.append(f"median_width<{float(args.min_person_width):g}")
    if med_h < float(args.min_person_height):
        reasons.append(f"median_height<{float(args.min_person_height):g}")
    if med_area < float(args.min_person_area):
        reasons.append(f"median_area<{float(args.min_person_area):g}")

    detail = {
        "passed": not reasons,
        "reason": "ok" if not reasons else ";".join(reasons),
        "record_count": int(len(rows)),
        "median_bbox_width": med_w,
        "median_bbox_height": med_h,
        "median_bbox_area": med_area,
    }
    return not reasons, detail


def load_siglip(model_name: str, device: str):
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
def score_crop_siglip(
    crop_bgr: np.ndarray,
    processor,
    model,
    device: str,
) -> float:
    """
    Return person probability in [0,1].

    We compare one image against positive and negative prompts together.
    SigLIP logits are converted with softmax over the prompt set.
    """
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(crop_rgb)

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
        raise RuntimeError(
            "SigLIP output에 logits_per_image/logits_per_text가 없습니다."
        )

    probs = torch.softmax(logits.float(), dim=0)
    n_pos = len(POSITIVE_PROMPTS)
    person_prob = probs[:n_pos].sum().item()
    return float(person_prob)


def aggregate_track_score(siglip_scores, detector_scores, siglip_weight, detector_weight):
    if not siglip_scores:
        return 0.5, 0.5, 0.5

    siglip_score = float(np.median(siglip_scores))
    detector_score = (
        float(np.median(detector_scores))
        if detector_scores else 0.5
    )

    total_w = siglip_weight + detector_weight
    if total_w <= 0:
        raise ValueError("weights must sum to > 0")

    fused = (
        siglip_score * siglip_weight
        + detector_score * detector_weight
    ) / total_w

    return fused, siglip_score, detector_score


def classify(score, person_threshold, non_person_threshold):
    if score >= person_threshold:
        return "person"
    if score <= non_person_threshold:
        return "non_person"
    return "uncertain"


def make_uncertain_html(review_tracks, root: Path):
    cards = []

    for tr in review_tracks:
        imgs = []
        for s in tr["samples"]:
            rel = s["crop_relpath"].replace("\\", "/")
            imgs.append(
                '<div class="sample">'
                f'<img src="{rel}">'
                f'<div>frame {s["frame_idx"]} · score {s["siglip_person_score"]:.3f}</div>'
                '</div>'
            )

        cards.append(
            '<section class="card">'
            f'<h2>Long Track #{tr["long_track_id"]}</h2>'
            f'<p>fused={tr["fused_person_score"]:.3f} · '
            f'siglip={tr["siglip_person_score"]:.3f} · '
            f'detector={tr["detector_person_score"]:.3f}</p>'
            f'<div class="grid">{"".join(imgs)}</div>'
            '</section>'
        )

    html = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Uncertain Track Review</title>
<style>
body{background:#0f1115;color:#eef1f6;font-family:Segoe UI,Arial,sans-serif;margin:0;padding:20px}
.card{background:#171a21;border:1px solid #303644;border-radius:12px;padding:14px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}
.sample{background:#090b0f;padding:8px;border-radius:8px}
.sample img{width:100%;height:240px;object-fit:contain}
p{color:#aab2c0}
</style>
</head>
<body>
<h1>Uncertain Track Review</h1>
""" + "".join(cards) + """
</body>
</html>"""

    (root / "uncertain_review.html").write_text(html, encoding="utf-8")


def main():
    args = parse_args()

    video_path = Path(args.video).resolve()
    tracks_path = Path(args.tracks).resolve()
    output_path = Path(args.output).resolve()
    review_root = Path(args.review_dir).resolve() / video_path.stem
    crop_root = review_root / "crops"

    review_root.mkdir(parents=True, exist_ok=True)
    crop_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    tracks = load_tracks(tracks_path)

    person_rows = [
        r for r in tracks
        if str(r.get("class_name", "")).lower() == "person"
    ]

    grouped = defaultdict(list)
    for r in person_rows:
        long_id = int(r.get("long_track_id", r.get("track_id")))
        grouped[long_id].append(r)

    processor = model = None

    if grouped:
        print(f"[INFO] loading SigLIP2: {args.model}")
        processor, model = load_siglip(args.model, args.device)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
    else:
        print("[INFO] no person candidate tracks -> skipping SigLIP2 validation")
        cap = None

    track_results = {}
    review_tracks = []

    for long_id in sorted(grouped):
        rows = sorted(grouped[long_id], key=lambda r: int(r["frame_idx"]))

        gate_passed, gate = geometry_gate_track(rows, args)

        if not gate_passed:
            result = {
                "long_track_id": int(long_id),
                "decision": "reject",
                "fused_person_score": 0.0,
                "siglip_person_score": 0.0,
                "detector_person_score": float(np.median([
                    float(r.get("confidence", 0.0))
                    for r in rows
                ])) if rows else 0.0,
                "num_source_detections": len(rows),
                "num_validation_samples": 0,
                "short_track_ids": sorted({
                    int(r.get("short_track_id", r.get("track_id")))
                    for r in rows
                }),
                "samples": [],
                "geometry_gate": gate,
            }
            track_results[int(long_id)] = result

            print(
                f"[TRACK {long_id}] decision=reject "
                f"geometry={gate['reason']} "
                f"median={gate.get('median_bbox_width', 0):.1f}x"
                f"{gate.get('median_bbox_height', 0):.1f} "
                f"area={gate.get('median_bbox_area', 0):.1f}"
            )
            continue

        samples = sample_rows(rows, args.samples_per_track)

        siglip_scores = []
        detector_scores = []
        sample_meta = []

        track_crop_dir = crop_root / f"long_{long_id:04d}"
        track_crop_dir.mkdir(parents=True, exist_ok=True)

        for i, row in enumerate(samples, 1):
            frame_idx = int(row["frame_idx"])

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                print(f"[WARN] cannot read frame={frame_idx}")
                continue

            crop = crop_with_padding(frame, row["bbox"], args.pad_ratio)
            if crop is None or crop.size == 0:
                continue

            sig_score = score_crop_siglip(
                crop, processor, model, args.device
            )

            det_score = float(row.get("confidence", 0.5))

            siglip_scores.append(sig_score)
            detector_scores.append(det_score)

            crop_path = (
                track_crop_dir
                / f"sample_{i:02d}_frame_{frame_idx:08d}.jpg"
            )
            cv2.imwrite(str(crop_path), crop)

            sample_meta.append({
                "frame_idx": frame_idx,
                "timestamp_sec": float(row.get("timestamp_sec", 0.0)),
                "rf_detr_confidence": det_score,
                "siglip_person_score": sig_score,
                "crop_relpath": str(crop_path.relative_to(review_root)),
            })

        fused, siglip_track, detector_track = aggregate_track_score(
            siglip_scores,
            detector_scores,
            args.siglip_weight,
            args.detector_weight,
        )

        decision = classify(
            fused,
            args.person_threshold,
            args.non_person_threshold,
        )

        result = {
            "long_track_id": int(long_id),
            "decision": decision,
            "fused_person_score": fused,
            "siglip_person_score": siglip_track,
            "detector_person_score": detector_track,
            "num_source_detections": len(rows),
            "num_validation_samples": len(sample_meta),
            "short_track_ids": sorted({
                int(r.get("short_track_id", r.get("track_id")))
                for r in rows
            }),
            "samples": sample_meta,
            "geometry_gate": gate,
        }

        track_results[int(long_id)] = result

        if decision == "uncertain":
            review_tracks.append(result)

        print(
            f"[TRACK {long_id}] "
            f"decision={decision:<10} "
            f"fused={fused:.3f} "
            f"siglip={siglip_track:.3f} "
            f"det={detector_track:.3f}"
        )

    if cap is not None:
        cap.release()

    validated = []

    for row in tracks:
        out = dict(row)

        if str(row.get("class_name", "")).lower() == "person":
            long_id = int(row.get("long_track_id", row.get("track_id")))
            res = track_results[long_id]

            out["validation_decision"] = res["decision"]
            out["validation_person_score"] = res["fused_person_score"]
            out["validation_siglip_score"] = res["siglip_person_score"]
            out["validation_detector_score"] = res["detector_person_score"]

            # Final routing field used by embedding/DB stage.
            if res["decision"] == "person":
                out["db_route"] = "person"
            elif res["decision"] == "non_person":
                out["db_route"] = "object"
            elif res["decision"] == "reject":
                out["db_route"] = "reject"
            else:
                out["db_route"] = "review"

            gate = res.get("geometry_gate", {})
            out["validation_geometry_gate"] = (
                "pass" if gate.get("passed", True) else "reject"
            )
            out["validation_geometry_reason"] = gate.get("reason", "ok")
            out["validation_bbox_median_width"] = float(
                gate.get("median_bbox_width", 0.0)
            )
            out["validation_bbox_median_height"] = float(
                gate.get("median_bbox_height", 0.0)
            )
            out["validation_bbox_median_area"] = float(
                gate.get("median_bbox_area", 0.0)
            )
        else:
            out["validation_decision"] = "non_person"
            out["db_route"] = "object"

        validated.append(out)

    output_path.write_text(
        json.dumps(validated, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "video": str(video_path),
        "source_tracks": str(tracks_path),
        "validated_tracks": str(output_path),
        "model": args.model,
        "thresholds": {
            "person": args.person_threshold,
            "non_person": args.non_person_threshold,
        },
        "weights": {
            "siglip": args.siglip_weight,
            "detector": args.detector_weight,
        },
        "track_results": list(track_results.values()),
    }

    (review_root / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    make_uncertain_html(review_tracks, review_root)

    counts = defaultdict(int)
    for x in track_results.values():
        counts[x["decision"]] += 1

    print()
    print("=" * 72)
    print("AUTO TRACK VALIDATION COMPLETE")
    print("=" * 72)
    print(f"person      : {counts['person']}")
    print(f"non_person  : {counts['non_person']}")
    print(f"uncertain   : {counts['uncertain']}")
    print(f"reject      : {counts['reject']}")
    print(f"output      : {output_path}")
    print(f"summary     : {review_root / 'validation_summary.json'}")
    print(f"review html : {review_root / 'uncertain_review.html'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
