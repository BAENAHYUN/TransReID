#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from config import PipelineConfig


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "pipeline.yaml"


def parse_args():
    p = argparse.ArgumentParser(
        description="Build visual HTML for cross-video benchmark Top-1 failures"
    )
    p.add_argument(
        "--benchmark",
        default=str(
            ROOT / "outputs" / "final_db_candidates"
            / "test10_cross_video_benchmark"
            / "cross_video_benchmark.json"
        ),
    )
    p.add_argument("--person-collection", default="forensic_person_test10")
    p.add_argument("--object-collection", default="forensic_object_test10")
    p.add_argument("--top-n", type=int, default=5)
    return p.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fetch_track_representative(client, collection, track_key):
    filt = Filter(
        must=[
            FieldCondition(
                key="track_key",
                match=MatchValue(value=track_key),
            )
        ]
    )

    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=filt,
        limit=32,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        return None

    # Prefer the highest quality selected representative.
    def q(p):
        payload = p.payload or {}
        return float(payload.get("quality_score") or 0.0)

    best = max(points, key=q)
    payload = dict(best.payload or {})
    payload["_point_id"] = str(best.id)
    return payload


def path_uri(path):
    if not path:
        return ""
    try:
        return Path(path).resolve().as_uri()
    except Exception:
        return str(path).replace("\\", "/")


def card(title, payload, rank=None):
    if not payload:
        return f"""
        <div class="card missing">
          <div class="rank">{html.escape(str(rank or ""))}</div>
          <div class="body"><b>{html.escape(title)}</b><br>payload not found</div>
        </div>
        """

    crop = path_uri(payload.get("crop_path"))
    tk = payload.get("track_key")
    quality = payload.get("quality_score")
    frame = payload.get("frame_idx")
    ts = payload.get("timestamp_sec")

    return f"""
    <div class="card">
      <div class="rank">{html.escape(str(rank or ""))}</div>
      <div class="thumb">
        {'<img src="' + html.escape(crop) + '">' if crop else '<div class="noimg">no crop</div>'}
      </div>
      <div class="body">
        <b>{html.escape(title)}</b>
        <div>{html.escape(str(tk or ""))}</div>
        <div>frame={html.escape(str(frame))} · time={html.escape(str(ts))}</div>
        <div>quality={html.escape(str(quality))}</div>
      </div>
    </div>
    """


def main():
    args = parse_args()
    bench_path = Path(args.benchmark)
    data = load_json(bench_path)

    failures = [
        q for q in data.get("queries", [])
        if q.get("rank") != 1
    ]

    cfg = PipelineConfig.load(CONFIG_PATH)
    client = QdrantClient(url=cfg.qdrant.url, timeout=120)

    cache = {}

    def rep(scope, track_key):
        key = (scope, track_key)
        if key in cache:
            return cache[key]

        collection = (
            args.person_collection
            if scope == "person"
            else args.object_collection
        )
        payload = fetch_track_representative(
            client,
            collection,
            track_key,
        )
        cache[key] = payload
        return payload

    unique_expected = Counter(
        q.get("expected_track_key")
        for q in failures
    )

    same_video_wrong = 0
    cross_video_wrong = 0
    rank_none = 0

    blocks = []

    for idx, q in enumerate(failures, 1):
        scope = q["scope"]
        expected = q["expected_track_key"]
        returned = q.get("returned_track_keys") or []

        if q.get("rank") is None:
            rank_none += 1

        expected_video = expected.split("/")[0]
        top1 = returned[0] if returned else ""
        top1_video = top1.split("/")[0] if "/" in top1 else ""

        if top1_video == expected_video:
            same_video_wrong += 1
        else:
            cross_video_wrong += 1

        query_uri = path_uri(q.get("query_path"))

        result_cards = []
        for rank, tk in enumerate(returned[: args.top_n], 1):
            result_cards.append(
                card(
                    f"Rank {rank}",
                    rep(scope, tk),
                    rank=rank,
                )
            )

        blocks.append(f"""
        <section>
          <div class="section-head">
            <div>
              <h2>Failure {idx} · {html.escape(scope.upper())}</h2>
              <div class="expected">Expected: {html.escape(expected)}</div>
              <div>actual rank: {html.escape(str(q.get("rank")))}</div>
              <div>frame: {html.escape(str(q.get("frame_idx")))}</div>
            </div>
          </div>

          <div class="query-row">
            <div class="query-card">
              <h3>Query</h3>
              <img src="{html.escape(query_uri)}">
            </div>
            <div class="query-card">
              <h3>Expected DB Representative</h3>
              {card("Expected", rep(scope, expected))}
            </div>
          </div>

          <h3>Returned Top-{args.top_n}</h3>
          <div class="grid">
            {''.join(result_cards)}
          </div>
        </section>
        """)

    summary = {
        "failure_queries": len(failures),
        "unique_failed_expected_tracks": len(unique_expected),
        "same_video_top1_wrong": same_video_wrong,
        "cross_video_top1_wrong": cross_video_wrong,
        "rank_none": rank_none,
        "failed_track_counts": dict(unique_expected),
    }

    out_dir = bench_path.parent
    summary_path = out_dir / "failure_analysis_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    html_path = out_dir / "failure_review.html"
    doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cross-video Failure Review</title>
<style>
:root{{--bg:#09101d;--panel:#121b2d;--panel2:#070c16;--line:#293753;--text:#eef4ff;--muted:#9aabc9;--bad:#ff9a9a}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,Arial,sans-serif}}
main{{max-width:1800px;margin:auto;padding:28px}}
h1{{margin-bottom:8px}}
.summary{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:20px 0}}
.metric{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}}
.metric span{{font-size:12px;color:var(--muted)}} .metric b{{display:block;font-size:26px;margin-top:5px}}
section{{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;margin:22px 0}}
.expected{{color:var(--bad);font-weight:800}}
.query-row{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:15px}}
.query-card{{background:var(--panel2);border:1px solid var(--line);border-radius:14px;padding:14px}}
.query-card>img{{width:100%;height:380px;object-fit:contain;background:#03060b}}
.grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}}
.card{{position:relative;background:var(--panel2);border:1px solid var(--line);border-radius:13px;overflow:hidden}}
.thumb{{height:300px;background:#03060b}}
.thumb img{{width:100%;height:100%;object-fit:contain}}
.body{{padding:10px;font-size:12px;display:grid;gap:4px;word-break:break-all}}
.rank{{position:absolute;z-index:2;top:7px;left:7px;padding:5px 8px;border-radius:999px;background:#10203b;font-weight:800}}
.missing{{min-height:120px}}
@media(max-width:1100px){{.grid{{grid-template-columns:repeat(2,1fr)}}.summary{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:700px){{.query-row,.grid,.summary{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main>
<h1>Cross-video Benchmark · Failure Review</h1>
<p>Query crop vs expected representative vs returned Top-{args.top_n}</p>

<div class="summary">
  <div class="metric"><span>Failed queries</span><b>{len(failures)}</b></div>
  <div class="metric"><span>Unique failed tracks</span><b>{len(unique_expected)}</b></div>
  <div class="metric"><span>Same-video Top1 errors</span><b>{same_video_wrong}</b></div>
  <div class="metric"><span>Cross-video Top1 errors</span><b>{cross_video_wrong}</b></div>
  <div class="metric"><span>Expected absent Top-{data.get('top_k', 10)}</span><b>{rank_none}</b></div>
</div>

{''.join(blocks)}
</main></body></html>"""

    html_path.write_text(doc, encoding="utf-8")

    print("=" * 90)
    print("FAILURE REVIEW COMPLETE")
    print("=" * 90)
    for k, v in summary.items():
        if k != "failed_track_counts":
            print(f"{k:30s}: {v}")
    print("html   :", html_path)
    print("json   :", summary_path)
    print("=" * 90)


if __name__ == "__main__":
    main()
