#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


POSITIVE_ALIASES = {
    "handbag": ["a handbag", "a purse", "a bag carried by a person"],
    "backpack": ["a backpack", "a rucksack", "a bag worn on the back"],
    "umbrella": ["an umbrella", "a handheld umbrella", "a folded or open umbrella"],
    "suitcase": ["a suitcase", "luggage", "a travel case"],
    "tie": ["a necktie", "a tie worn around the neck", "formal neckwear"],
    "cell phone": ["a cell phone", "a smartphone", "a mobile phone"],
    "bottle": ["a bottle", "a drink bottle", "a plastic or glass bottle"],
    "knife": ["a knife", "a handheld knife", "a bladed hand tool"],
    "car": ["a car", "an automobile", "a passenger vehicle"],
    "bicycle": ["a bicycle", "a bike", "a pedal bicycle"],
    "motorcycle": ["a motorcycle", "a motorbike", "a two-wheeled motor vehicle"],
    "bus": ["a bus", "a passenger bus", "a large public transport vehicle"],
    "truck": ["a truck", "a lorry", "a cargo vehicle"],
}

NEGATIVE_PROMPTS = [
    "a person's body",
    "human skin",
    "clothing",
    "a shirt or jacket",
    "an arm or leg",
    "a person's head",
    "background scenery",
    "an unrecognizable blurry crop",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Object track class stabilization + SigLIP2 semantic validation"
    )
    p.add_argument("--video", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--report-dir", default="./outputs/object_semantic_validation")
    p.add_argument("--model", default="google/siglip2-base-patch16-224")
    p.add_argument("--device", default="cuda")
    p.add_argument("--samples-per-track", type=int, default=7)
    p.add_argument("--tight-pad", type=float, default=0.10)
    p.add_argument("--context-pad", type=float, default=0.35)
    p.add_argument("--good-threshold", type=float, default=0.55)
    p.add_argument("--reject-threshold", type=float, default=0.45)
    p.add_argument("--min-class-purity", type=float, default=0.55)
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


def sample_rows(rows: List[dict], n: int) -> List[dict]:
    rows = sorted(rows, key=lambda r: int(r.get("frame_idx", 0)))
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


def weighted_class_vote(rows: List[dict]) -> Tuple[str, float, dict]:
    scores = defaultdict(float)
    total = 0.0

    for r in rows:
        label = str(r.get("class_name", "object")).strip().lower()
        conf = max(float(r.get("confidence", 0.0)), 1e-6)
        scores[label] += conf
        total += conf

    if not scores:
        return "object", 0.0, {}

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    label, score = ordered[0]
    purity = score / total if total > 0 else 0.0
    return label, float(purity), dict(ordered)


def positive_prompts(label: str) -> List[str]:
    label = str(label).strip().lower()
    aliases = POSITIVE_ALIASES.get(label)
    if aliases:
        return aliases
    return [
        f"a photo of a {label}",
        f"a clear {label}",
        f"the object is a {label}",
    ]


def load_siglip(model_name: str, device: str):
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return processor, model


@torch.inference_mode()
def semantic_score(
    crop_bgr: np.ndarray,
    label: str,
    processor,
    model,
    device: str,
):
    pos = positive_prompts(label)
    neg = NEGATIVE_PROMPTS
    texts = pos + neg

    image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
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
        logits = outputs.logits_per_image[0].float()
    elif hasattr(outputs, "logits_per_text"):
        logits = outputs.logits_per_text[:, 0].float()
    else:
        raise RuntimeError("SigLIP logits not found")

    n_pos = len(pos)
    pos_logits = logits[:n_pos]
    neg_logits = logits[n_pos:]

    pos_mean = float(pos_logits.mean().item())
    neg_mean = float(neg_logits.mean().item())
    margin = pos_mean - neg_mean

    # Stable 0..1 semantic score based on positive-vs-negative margin.
    # margin=0 -> 0.5, positive margin -> >0.5
    score = float(torch.sigmoid(torch.tensor(margin)).item())

    return {
        "score": score,
        "positive_mean_logit": pos_mean,
        "negative_mean_logit": neg_mean,
        "margin": float(margin),
    }


def robust_track_score(scores: List[float]) -> Tuple[float, float, float]:
    if not scores:
        return 0.0, 0.0, 0.0
    arr = np.asarray(scores, dtype=np.float32)
    median = float(np.median(arr))
    mean = float(np.mean(arr))
    p75 = float(np.percentile(arr, 75))
    final = 0.55 * median + 0.30 * mean + 0.15 * p75
    return float(final), median, mean


def classify(score: float, purity: float, args) -> str:
    if purity < args.min_class_purity:
        return "reject"
    if score >= args.good_threshold:
        return "good"
    if score <= args.reject_threshold:
        return "reject"
    return "low_quality"


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))


def main():
    args = parse_args()

    video_path = Path(args.video).resolve()
    tracks_path = Path(args.tracks).resolve()
    output_path = Path(args.output).resolve()
    report_root = Path(args.report_dir).resolve() / video_path.stem
    crop_root = report_root / "crops"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    crop_root.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    rows = load_json_or_jsonl(tracks_path)

    object_rows = [
        r for r in rows
        if str(r.get("final_db_route", "")).lower() == "object"
    ]

    grouped = defaultdict(list)
    for r in object_rows:
        lid = int(r.get("long_track_id", r.get("track_id")))
        grouped[lid].append(r)

    if not grouped:
        raise RuntimeError("No object tracks found in final_db_route=object")

    print(f"[INFO] loading SigLIP2: {args.model}")
    processor, model = load_siglip(args.model, args.device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    results = {}

    for long_id in sorted(grouped):
        track_rows = sorted(
            grouped[long_id],
            key=lambda r: int(r.get("frame_idx", 0)),
        )

        voted_label, purity, class_votes = weighted_class_vote(track_rows)
        sampled = sample_rows(track_rows, args.samples_per_track)

        frame_details = []
        all_scores = []

        track_crop_dir = crop_root / f"long_{long_id:04d}_{safe_name(voted_label)}"
        track_crop_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in enumerate(sampled, 1):
            frame_idx = int(row["frame_idx"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            tight = crop_with_padding(frame, row["bbox"], args.tight_pad)
            context = crop_with_padding(frame, row["bbox"], args.context_pad)

            view_scores = []
            chosen_crop = None

            tight_detail = None
            context_detail = None

            if tight is not None and tight.size:
                tight_detail = semantic_score(
                    tight, voted_label, processor, model, args.device
                )
                view_scores.append(float(tight_detail["score"]))
                chosen_crop = tight

            if context is not None and context.size:
                context_detail = semantic_score(
                    context, voted_label, processor, model, args.device
                )
                view_scores.append(float(context_detail["score"]))
                if chosen_crop is None:
                    chosen_crop = context

            if not view_scores or chosen_crop is None:
                continue

            frame_score = float(np.mean(view_scores))
            all_scores.append(frame_score)

            crop_path = (
                track_crop_dir
                / f"sample_{idx:02d}_frame_{frame_idx:08d}.jpg"
            )
            cv2.imwrite(str(crop_path), chosen_crop)

            frame_details.append({
                "frame_idx": frame_idx,
                "timestamp_sec": float(row.get("timestamp_sec", 0.0)),
                "source_class_name": str(row.get("class_name", "")),
                "source_confidence": float(row.get("confidence", 0.0)),
                "view_scores": view_scores,
                "tight_detail": tight_detail,
                "context_detail": context_detail,
                "semantic_score": frame_score,
                "crop_relpath": str(crop_path.relative_to(report_root)),
            })

        final_score, median, mean = robust_track_score(all_scores)
        decision = classify(final_score, purity, args)

        results[long_id] = {
            "long_track_id": int(long_id),
            "voted_label": voted_label,
            "class_purity": purity,
            "class_votes": class_votes,
            "semantic_score": final_score,
            "semantic_median": median,
            "semantic_mean": mean,
            "decision": decision,
            "num_source_records": len(track_rows),
            "num_samples": len(frame_details),
            "frames": frame_details,
        }

        print(
            f"[TRACK {long_id}] "
            f"label={voted_label:<12} "
            f"purity={purity:.3f} "
            f"semantic={final_score:.3f} "
            f"decision={decision}"
        )

    cap.release()

    # Preserve all rows, but annotate object records with semantic result.
    revised = []
    for r in rows:
        out = dict(r)

        if str(r.get("final_db_route", "")).lower() == "object":
            lid = int(r.get("long_track_id", r.get("track_id")))
            res = results[lid]

            out["object_track_label"] = res["voted_label"]
            out["object_class_purity"] = res["class_purity"]
            out["object_semantic_score"] = res["semantic_score"]
            out["object_semantic_decision"] = res["decision"]

            if res["decision"] == "good":
                out["final_db_route_v2"] = "object"
            elif res["decision"] == "low_quality":
                out["final_db_route_v2"] = "review"
            else:
                out["final_db_route_v2"] = "reject"
        else:
            out["final_db_route_v2"] = out.get("final_db_route")

        revised.append(out)

    output_path.write_text(
        json.dumps(revised, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "video": str(video_path),
        "tracks": list(results.values()),
        "counts": {
            "good": sum(r["decision"] == "good" for r in results.values()),
            "low_quality": sum(
                r["decision"] == "low_quality" for r in results.values()
            ),
            "reject": sum(r["decision"] == "reject" for r in results.values()),
        },
        "thresholds": {
            "good": args.good_threshold,
            "reject": args.reject_threshold,
            "min_class_purity": args.min_class_purity,
        },
    }

    (report_root / "object_semantic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    make_html(summary, report_root)

    print()
    print("=" * 80)
    print("OBJECT SEMANTIC VALIDATION COMPLETE")
    print("=" * 80)
    print("good        :", summary["counts"]["good"])
    print("low_quality :", summary["counts"]["low_quality"])
    print("reject      :", summary["counts"]["reject"])
    print("output      :", output_path)
    print("html        :", report_root / "object_semantic_report.html")
    print("=" * 80)


def make_html(summary: dict, report_root: Path):
    sections = []

    for tr in summary["tracks"]:
        cards = []
        for fr in tr["frames"]:
            rel = fr["crop_relpath"].replace("\\", "/")
            cards.append(
                f"""
                <div class="card">
                  <img src="{html.escape(rel)}">
                  <div class="meta">
                    <b>frame {fr["frame_idx"]}</b>
                    <span>semantic {fr["semantic_score"]:.3f}</span>
                    <span>det {fr["source_confidence"]:.3f}</span>
                    <span>{html.escape(fr["source_class_name"])}</span>
                    <span>tight margin {
                        "-" if not fr.get("tight_detail")
                        else f'{fr["tight_detail"]["margin"]:.3f}'
                    }</span>
                    <span>context margin {
                        "-" if not fr.get("context_detail")
                        else f'{fr["context_detail"]["margin"]:.3f}'
                    }</span>
                  </div>
                </div>
                """
            )

        sections.append(
            f"""
            <section>
              <div class="head">
                <div>
                  <h2>OBJECT Track {tr["long_track_id"]} · {html.escape(tr["voted_label"])}</h2>
                  <p>
                    class purity <b>{tr["class_purity"]:.3f}</b> ·
                    semantic <b>{tr["semantic_score"]:.3f}</b> ·
                    samples {tr["num_samples"]}/{tr["num_source_records"]}
                  </p>
                </div>
                <span class="decision {tr["decision"]}">{tr["decision"]}</span>
              </div>
              <div class="grid">{''.join(cards)}</div>
            </section>
            """
        )

    c = summary["counts"]

    doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Object Semantic Validation</title>
<style>
:root{{--bg:#09101d;--panel:#121b2d;--line:#293753;--text:#eef4ff;--muted:#99a8c5}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,Arial,sans-serif}}
main{{max-width:1750px;margin:auto;padding:30px}}
.top{{display:flex;gap:12px;margin:20px 0;flex-wrap:wrap}}
.metric{{background:var(--panel);border:1px solid var(--line);padding:14px 18px;border-radius:14px;min-width:150px}}
.metric b{{font-size:28px;display:block}}
section{{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;margin:18px 0}}
.head{{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}}
h2{{margin:0}} p{{color:var(--muted)}}
.decision{{padding:7px 11px;border-radius:999px;font-weight:800}}
.good{{background:#153d2e;color:#82f0bd;border:1px solid #2e8c67}}
.low_quality{{background:#433619;color:#ffd27b;border:1px solid #91712e}}
.reject{{background:#451d25;color:#ff98aa;border:1px solid #954150}}
.grid{{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:10px}}
.card{{border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#070c16}}
.card img{{width:100%;height:300px;object-fit:contain;background:#04070d}}
.meta{{display:grid;padding:9px;gap:4px;font-size:12px}}
.meta span{{color:var(--muted)}}
@media(max-width:1300px){{.grid{{grid-template-columns:repeat(4,1fr)}}}}
@media(max-width:800px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body>
<main>
<h1>Object Semantic Validation · Margin Scoring</h1>
<div class="top">
  <div class="metric">GOOD<b>{c["good"]}</b></div>
  <div class="metric">LOW QUALITY<b>{c["low_quality"]}</b></div>
  <div class="metric">REJECT<b>{c["reject"]}</b></div>
</div>
{''.join(sections)}
</main>
</body>
</html>"""

    (report_root / "object_semantic_report.html").write_text(
        doc, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
