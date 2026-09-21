#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import html
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List

import unified_search_4mode as us


ROOT = Path(__file__).resolve().parent
CANDIDATE_ROOT = ROOT / "outputs" / "final_db_candidates"
CONFIG_PATH = ROOT / "pipeline.yaml"


def parse_args():
    p = argparse.ArgumentParser(
        description="Run 4 grouped search tests against TEST Qdrant collections"
    )
    p.add_argument("--video-stem", default="Normal_Videos_015_x264")
    p.add_argument("--person-image", default=None)
    p.add_argument("--object-image", default=None)
    p.add_argument("--person-text", default="a person")
    p.add_argument("--object-text", default=None)
    p.add_argument("--person-collection", default="forensic_person_test")
    p.add_argument("--object-collection", default="forensic_object_test")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--group-size", type=int, default=3)
    p.add_argument("--candidate-k", type=int, default=20)
    p.add_argument("--no-translate", action="store_true")
    return p.parse_args()


def load_candidates(video_stem: str):
    path = CANDIDATE_ROOT / video_stem / "final_db_candidates.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return path, data


def pick_query_images(data: dict):
    rows = data.get("candidates", [])
    people = [r for r in rows if r.get("candidate_scope") == "person"]
    objects = [r for r in rows if r.get("candidate_scope") == "object"]

    if not people:
        raise RuntimeError("No person candidate crop found")
    if not objects:
        raise RuntimeError("No object candidate crop found")

    # Highest quality crop among final selected candidates.
    person = max(people, key=lambda r: float(r.get("quality_score") or 0.0))
    obj = max(objects, key=lambda r: float(r.get("quality_score") or 0.0))
    return person, obj


def ns_image(scope: str, image: str, args):
    return Namespace(
        image=image,
        config=str(CONFIG_PATH),
        scope=scope,
        db="test",
        vector=("solider" if scope == "person" else "dinov2"),
        top_k=args.top_k,
        group_size=args.group_size,
    )


def ns_text(scope: str, text: str, args):
    return Namespace(
        text=text,
        config=str(CONFIG_PATH),
        scope=scope,
        db="test",
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        group_size=args.group_size,
        translate_backend="opus",
        translate_model_id=None,
        no_translate=args.no_translate,
        expand=False,
    )


def rel_or_uri(path: str, base: Path) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return p.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        try:
            return p.resolve().as_uri()
        except Exception:
            return str(p).replace("\\", "/")


def score_text(row: dict) -> str:
    if row.get("rrf_score") is not None:
        return f'RRF {float(row["rrf_score"]):.6f}'
    return f'score {float(row.get("score") or 0.0):.6f}'


def result_cards(result: dict, report_dir: Path) -> str:
    cards = []
    for row in result.get("results", []):
        crop = rel_or_uri(str(row.get("crop_path") or ""), report_dir)
        gs = row.get("group_summary") or {}
        vector_scores = row.get("vector_scores") or {}

        vector_html = " · ".join(
            f"{html.escape(str(k))}={float(v):.4f}"
            for k, v in vector_scores.items()
        )

        cards.append(f"""
        <article class="result-card">
          <div class="rank">#{int(row.get("rank", 0))}</div>
          <div class="thumb">
            {'<img src="' + html.escape(crop) + '">' if crop else '<div class="missing">no crop</div>'}
          </div>
          <div class="body">
            <h3>{html.escape(str(row.get("label") or ""))}</h3>
            <div class="score">{html.escape(score_text(row))}</div>
            <div><b>track</b> {html.escape(str(row.get("track_key") or ""))}</div>
            <div><b>frame</b> {html.escape(str(row.get("frame_idx")))}</div>
            <div><b>time</b> {us.fmt_time(row.get("timestamp_sec"))}</div>
            <div><b>points</b> {html.escape(str(gs.get("count", 0)))}</div>
            <div><b>range</b> {us.fmt_time(gs.get("start_sec"))} ~ {us.fmt_time(gs.get("end_sec"))}</div>
            <div class="vectors">{html.escape(vector_html)}</div>
          </div>
        </article>
        """)
    return "".join(cards) or '<div class="empty">No results</div>'


def query_block(title: str, result: dict, query_desc: str, report_dir: Path) -> str:
    timing = result.get("timing") or {}
    timing_text = " · ".join(
        f"{k}={float(v):.3f}s" for k, v in timing.items()
    )
    vectors = " + ".join(result.get("vectors") or [])

    return f"""
    <section>
      <div class="section-head">
        <div>
          <h2>{html.escape(title)}</h2>
          <p>{html.escape(query_desc)}</p>
        </div>
        <div class="tags">
          <span>{html.escape(str(result.get("collection")))}</span>
          <span>{html.escape(vectors)}</span>
          <span>{html.escape(timing_text)}</span>
        </div>
      </div>
      <div class="results">
        {result_cards(result, report_dir)}
      </div>
    </section>
    """


def write_html(
    path: Path,
    video_stem: str,
    person_image: str,
    object_image: str,
    person_text: str,
    object_text: str,
    results: List[tuple],
):
    report_dir = path.parent
    blocks = [
        query_block(title, res, desc, report_dir)
        for title, res, desc in results
    ]

    person_q = rel_or_uri(person_image, report_dir)
    object_q = rel_or_uri(object_image, report_dir)

    doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>4-Mode TEST Search</title>
<style>
:root{{--bg:#08101d;--panel:#111a2b;--panel2:#0a111e;--line:#293754;--text:#edf4ff;--muted:#9baccc;--accent:#79b8ff}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,Arial,sans-serif}}
main{{max-width:1750px;margin:auto;padding:30px}}
h1{{margin:0 0 6px;font-size:32px}}
.sub{{color:var(--muted);margin-bottom:24px}}
.query-preview{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:22px}}
.query-card{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:15px;display:grid;grid-template-columns:180px 1fr;gap:15px}}
.query-card img{{width:180px;height:180px;object-fit:contain;background:#040812;border-radius:10px}}
.query-card p{{color:var(--muted)}}
section{{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;margin:20px 0}}
.section-head{{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}}
.section-head h2{{margin:0 0 6px}}
.section-head p{{margin:0;color:var(--muted)}}
.tags{{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end}}
.tags span{{border:1px solid var(--line);background:var(--panel2);padding:6px 9px;border-radius:999px;font-size:12px;color:var(--muted)}}
.results{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-top:16px}}
.result-card{{position:relative;border:1px solid var(--line);background:var(--panel2);border-radius:14px;overflow:hidden}}
.rank{{position:absolute;top:8px;left:8px;z-index:2;background:#0b1830;border:1px solid #365b8d;padding:5px 8px;border-radius:999px;font-weight:800}}
.thumb{{height:300px;display:flex;align-items:center;justify-content:center;background:#040812}}
.thumb img{{width:100%;height:100%;object-fit:contain}}
.body{{padding:12px;font-size:13px;display:grid;gap:5px}}
.body h3{{margin:0;font-size:17px}}
.score{{font-weight:800;color:var(--accent);font-size:16px}}
.vectors{{color:var(--muted);word-break:break-all}}
.empty,.missing{{color:var(--muted);padding:30px}}
@media(max-width:1200px){{.results{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:760px){{.query-preview,.results{{grid-template-columns:1fr}}.query-card{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main>
<h1>TEST DB · 4-Mode Search</h1>
<div class="sub">{html.escape(video_stem)} · grouped by track_key · production DB untouched</div>

<div class="query-preview">
  <div class="query-card">
    <img src="{html.escape(person_q)}">
    <div><h2>Person query</h2><p>Image: {html.escape(Path(person_image).name)}</p><p>Text: {html.escape(person_text)}</p></div>
  </div>
  <div class="query-card">
    <img src="{html.escape(object_q)}">
    <div><h2>Object query</h2><p>Image: {html.escape(Path(object_image).name)}</p><p>Text: {html.escape(object_text)}</p></div>
  </div>
</div>

{''.join(blocks)}
</main></body></html>
"""
    path.write_text(doc, encoding="utf-8")


def main():
    args = parse_args()

    # This script is TEST-DB-only.
    # Local unified_search_4mode.py variants differ: some call
    # collection_for_scope(cfg, scope) and therefore never pass args.db.
    # To avoid accidentally falling back to production collections,
    # force ALL collection resolution in this process to TEST collections.
    def _test_collection_for_scope(cfg, scope, *extra, **kwargs):
        if scope == "person":
            return args.person_collection
        if scope == "object":
            return args.object_collection
        raise ValueError("scope must be person or object")

    us.collection_for_scope = _test_collection_for_scope

    candidate_path, data = load_candidates(args.video_stem)
    person_row, object_row = pick_query_images(data)

    person_image = args.person_image or person_row["selected_crop_path"]
    object_image = args.object_image or object_row["selected_crop_path"]
    object_text = args.object_text or (
        f'a {object_row.get("selected_label") or object_row.get("object_track_label") or "object"}'
    )

    print("=" * 92)
    print("4-MODE TEST SEARCH")
    print("=" * 92)
    print("person image :", person_image)
    print("object image :", object_image)
    print("person text  :", args.person_text)
    print("object text  :", object_text)
    print("=" * 92)

    results = []

    # 1) image -> person track, default SOLIDER
    print("[1/4] image -> person")
    r1 = us.run_image_video(ns_image("person", person_image, args))
    results.append((
        "1. Image → Person",
        r1,
        f"query={Path(person_image).name} · SOLIDER grouped track search",
    ))

    # 2) image -> object track, default DINOv2
    print("[2/4] image -> object")
    r2 = us.run_image_video(ns_image("object", object_image, args))
    results.append((
        "2. Image → Object",
        r2,
        f"query={Path(object_image).name} · DINOv2 grouped track search",
    ))

    # 3) text -> person track, SigLIP2 + IRRA RRF
    print("[3/4] text -> person")
    r3 = us.run_text_video(ns_text("person", args.person_text, args))
    results.append((
        "3. Text → Person",
        r3,
        f'query="{args.person_text}" · SigLIP2 + IRRA RRF',
    ))

    # 4) text -> object track, SigLIP2
    print("[4/4] text -> object")
    r4 = us.run_text_video(ns_text("object", object_text, args))
    results.append((
        "4. Text → Object",
        r4,
        f'query="{object_text}" · SigLIP2',
    ))

    out_dir = CANDIDATE_ROOT / args.video_stem / "test_search_4mode"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "video_stem": args.video_stem,
        "candidate_file": str(candidate_path),
        "queries": {
            "person_image": person_image,
            "object_image": object_image,
            "person_text": args.person_text,
            "object_text": object_text,
        },
        "tests": {
            "image_person": r1,
            "image_object": r2,
            "text_person": r3,
            "text_object": r4,
        },
    }

    json_path = out_dir / "search_results.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    html_path = out_dir / "search_results.html"
    write_html(
        html_path,
        args.video_stem,
        person_image,
        object_image,
        args.person_text,
        object_text,
        results,
    )

    print()
    print("=" * 92)
    print("4-MODE TEST SEARCH COMPLETE")
    print("=" * 92)
    print("json :", json_path)
    print("html :", html_path)
    print("=" * 92)


if __name__ == "__main__":
    main()
