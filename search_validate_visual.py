from __future__ import annotations

import argparse
import base64
import html
import mimetypes
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from config import PipelineConfig
from registry import EmbedderRegistry
from router import Router

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "pipeline.yaml"

PERSON_COLLECTION = "forensic_person"
OBJECT_COLLECTION = "forensic_object"


def normalize(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v if n <= 0 else v / n


def canonical_path(path: str) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve()).lower()
    except Exception:
        return str(path).lower()


def make_filter(media_type: str, label: str | None = None):
    must = []
    if media_type != "all":
        must.append(FieldCondition(key="media_type", match=MatchValue(value=media_type)))
    if label:
        must.append(FieldCondition(key="label", match=MatchValue(value=label)))
    return Filter(must=must) if must else None


def get_timestamp(frame_idx: int, video_path: str) -> str:
    if not video_path:
        return ""
    p = Path(video_path)
    if not p.exists():
        return ""
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return ""
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        return ""
    sec = frame_idx / fps
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


def image_to_data_uri(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    try:
        mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"
    except Exception:
        return ""


def run_single_vector_search(client, collection, vector_name, query_vector,
                             limit, media_type, label):
    return client.query_points(
        collection_name=collection,
        using=vector_name,
        query=normalize(query_vector).tolist(),
        query_filter=make_filter(media_type, label),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    ).points


def remove_self_match(hits, query_image):
    if not query_image:
        return hits
    q = canonical_path(query_image)
    return [
        h for h in hits
        if canonical_path((h.payload or {}).get("crop_path", "")) != q
    ]


def weighted_rrf(per_model_hits, weights, rrf_k=60):
    fused = defaultdict(float)
    for model_name, hits in per_model_hits.items():
        w = float(weights.get(model_name, 1.0))
        for rank, hit in enumerate(hits, start=1):
            fused[str(hit.id)] += w / (rrf_k + rank)
    return dict(fused)


def search_and_explain(client, collection, query_vectors, cfg, candidate_k,
                       top_k, media_type, label, query_image):
    info = client.get_collection(collection)
    available_vectors = set(info.config.params.vectors.keys())

    per_model_hits = {}
    weights = {}
    rank_map = defaultdict(dict)
    score_map = defaultdict(dict)
    point_map = {}

    for model_name, vector in query_vectors.items():
        if model_name not in available_vectors:
            continue

        raw_hits = run_single_vector_search(
            client, collection, model_name, vector,
            candidate_k + (5 if query_image else 0),
            media_type, label
        )
        hits = remove_self_match(raw_hits, query_image)[:candidate_k]
        per_model_hits[model_name] = hits
        weights[model_name] = float(cfg.retrievers[model_name].weight)

        for rank, hit in enumerate(hits, start=1):
            pid = str(hit.id)
            rank_map[pid][model_name] = rank
            score_map[pid][model_name] = float(hit.score)
            point_map[pid] = hit

    fused = weighted_rrf(per_model_hits, weights)

    rows = []
    for pid, hit in point_map.items():
        p = hit.payload or {}
        rows.append({
            "point_id": pid,
            "fusion_score": fused.get(pid, 0.0),
            "payload": p,
            "model_ranks": dict(rank_map.get(pid, {})),
            "model_scores": dict(score_map.get(pid, {})),
        })

    rows.sort(key=lambda x: x["fusion_score"], reverse=True)

    # video: same track appears only once in final fusion list
    out, seen = [], set()
    for row in rows:
        p = row["payload"]
        if p.get("media_type") == "video":
            key = p.get("track_key") or f"{p.get('video')}|{p.get('track_id')}|{p.get('label')}"
            if key in seen:
                continue
            seen.add(key)
        out.append(row)
        if len(out) >= top_k:
            break

    return out, per_model_hits


def hit_card(rank, hit, model_name=None):
    p = hit.payload or {}
    crop = p.get("crop_path", "")
    src = image_to_data_uri(crop)
    score = float(hit.score)
    media = p.get("media_type", "")

    extra = ""
    if media == "video":
        video_path = p.get("video_path", "") or ""
        frame = int(p.get("frame_idx", 0) or 0)
        extra = (
            f"<div><b>video</b>: {html.escape(str(p.get('video','')))}</div>"
            f"<div><b>track</b>: {html.escape(str(p.get('track_id','')))}"
            f" / <b>frame</b>: {frame}"
            f" / <b>time</b>: {html.escape(get_timestamp(frame, video_path) or 'N/A')}</div>"
        )

    img_html = (
        f'<img src="{src}" loading="lazy">'
        if src else
        '<div class="missing">이미지 없음</div>'
    )

    return f"""
    <div class="card">
      <div class="rank">#{rank}</div>
      {img_html}
      <div class="meta">
        <div><b>score</b>: {score:.6f}</div>
        <div><b>media</b>: {html.escape(media)}</div>
        {extra}
        <div class="path">{html.escape(crop)}</div>
      </div>
    </div>
    """


def fusion_card(rank, row, model_order):
    p = row["payload"]
    crop = p.get("crop_path", "")
    src = image_to_data_uri(crop)
    media = p.get("media_type", "")

    model_lines = []
    for m in model_order:
        rr = row["model_ranks"].get(m)
        ss = row["model_scores"].get(m)
        if rr is None:
            model_lines.append(f"<div><b>{m}</b>: Top-K 밖</div>")
        else:
            model_lines.append(f"<div><b>{m}</b>: rank {rr}, score {ss:.6f}</div>")

    extra = ""
    if media == "video":
        video_path = p.get("video_path", "") or ""
        frame = int(p.get("frame_idx", 0) or 0)
        extra = (
            f"<div><b>video</b>: {html.escape(str(p.get('video','')))}</div>"
            f"<div><b>track</b>: {html.escape(str(p.get('track_id','')))}"
            f" / <b>frame</b>: {frame}"
            f" / <b>time</b>: {html.escape(get_timestamp(frame, video_path) or 'N/A')}</div>"
        )

    img_html = (
        f'<img src="{src}" loading="lazy">'
        if src else
        '<div class="missing">이미지 없음</div>'
    )

    return f"""
    <div class="card fusion">
      <div class="rank">#{rank}</div>
      {img_html}
      <div class="meta">
        <div><b>fusion</b>: {row["fusion_score"]:.6f}</div>
        {''.join(model_lines)}
        {extra}
        <div class="path">{html.escape(crop)}</div>
      </div>
    </div>
    """


def build_html(query_image, query_text, scope, media, rows, per_model_hits,
               model_order, single_top):
    query_uri = image_to_data_uri(query_image) if query_image else ""
    q_display = html.escape(query_image or query_text or "")

    sections = []

    for model in model_order:
        hits = per_model_hits.get(model, [])
        if not hits:
            continue
        cards = "".join(
            hit_card(i, h, model)
            for i, h in enumerate(hits[:single_top], start=1)
        )
        sections.append(f"""
        <section>
          <h2>{model.upper()} 단독 Top {min(single_top, len(hits))}</h2>
          <div class="grid">{cards}</div>
        </section>
        """)

    fusion_cards = "".join(
        fusion_card(i, row, model_order)
        for i, row in enumerate(rows, start=1)
    )

    query_block = (
        f'<img class="query-img" src="{query_uri}">' if query_uri else ""
    )

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Search Validation Results</title>
<style>
body {{
  font-family: Arial, "Malgun Gothic", sans-serif;
  margin: 24px;
  background: #f5f6f8;
  color: #222;
}}
h1, h2 {{ margin-bottom: 12px; }}
.query {{
  display: flex;
  gap: 20px;
  align-items: flex-start;
  background: white;
  padding: 16px;
  border-radius: 12px;
  margin-bottom: 24px;
}}
.query-img {{
  width: 220px;
  max-height: 320px;
  object-fit: contain;
  background: #eee;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 14px;
}}
.card {{
  position: relative;
  background: white;
  border-radius: 12px;
  padding: 10px;
  box-shadow: 0 1px 5px rgba(0,0,0,.08);
}}
.card img {{
  width: 100%;
  height: 280px;
  object-fit: contain;
  background: #eee;
  border-radius: 8px;
}}
.card.fusion {{
  border: 2px solid #4f73c3;
}}
.rank {{
  position: absolute;
  top: 14px;
  left: 14px;
  background: rgba(0,0,0,.78);
  color: white;
  padding: 5px 9px;
  border-radius: 6px;
  font-size: 18px;
  font-weight: bold;
}}
.meta {{
  padding-top: 8px;
  line-height: 1.45;
  font-size: 13px;
}}
.path {{
  margin-top: 8px;
  font-size: 10px;
  color: #666;
  overflow-wrap: anywhere;
}}
.missing {{
  height: 280px;
  display: grid;
  place-items: center;
  background: #eee;
}}
section {{
  margin-bottom: 34px;
}}
</style>
</head>
<body>
<h1>검색/검증 이미지 모음</h1>
<div class="query">
  {query_block}
  <div>
    <div><b>Query</b>: {q_display}</div>
    <div><b>Scope</b>: {html.escape(scope)}</div>
    <div><b>Media</b>: {html.escape(media)}</div>
  </div>
</div>

<section>
  <h2>최종 Fusion Top {len(rows)}</h2>
  <div class="grid">{fusion_cards}</div>
</section>

{''.join(sections)}
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser(description="검색 검증 + 결과 이미지 HTML 모음")
    q = ap.add_mutually_exclusive_group(required=True)
    q.add_argument("--image")
    q.add_argument("--text")

    ap.add_argument("--scope", choices=["person", "object"], required=True)
    ap.add_argument("--media", choices=["all", "image", "video"], default="all")
    ap.add_argument("--label", default=None)
    ap.add_argument("--candidate-k", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--single-top", type=int, default=10)
    ap.add_argument("--html-out", default=None)
    args = ap.parse_args()

    cfg = PipelineConfig.load(CONFIG_PATH)
    reg = EmbedderRegistry(cfg)
    router = Router(cfg, reg, input_format="rgb")
    client = QdrantClient(url=cfg.qdrant.url)

    query_image = None

    if args.image:
        qp = Path(args.image)
        if not qp.exists():
            raise FileNotFoundError(qp)
        query_image = str(qp)
        vectors = router.embed_query_image(str(qp), scope=args.scope)
    else:
        names = ["siglip2", "irra"] if args.scope == "person" else ["siglip2"]
        vectors = router.embed_query_text(args.text, names=names)

    if args.scope == "person":
        collection = PERSON_COLLECTION
        model_order = ["solider", "irra", "siglip2"]
        label = args.label or "person"
    else:
        collection = OBJECT_COLLECTION
        model_order = ["dinov2", "siglip2"]
        label = args.label

    vectors = {k: v for k, v in vectors.items() if k in model_order}

    rows, per_model_hits = search_and_explain(
        client=client,
        collection=collection,
        query_vectors=vectors,
        cfg=cfg,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        media_type=args.media,
        label=label,
        query_image=query_image,
    )

    print("\n" + "=" * 90)
    print("FINAL FUSION RESULTS")
    print("=" * 90)
    for i, row in enumerate(rows, 1):
        p = row["payload"]
        print(
            f"[{i:02d}] fusion={row['fusion_score']:.6f} "
            f"| {p.get('media_type','')} "
            f"| crop={p.get('crop_path','')}"
        )

    if args.html_out:
        out = Path(args.html_out)
        if not out.is_absolute():
            out = ROOT / out
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = ROOT / "results" / f"validation_{args.scope}_{args.media}_{stamp}.html"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        build_html(
            query_image=args.image,
            query_text=args.text,
            scope=args.scope,
            media=args.media,
            rows=rows,
            per_model_hits=per_model_hits,
            model_order=model_order,
            single_top=args.single_top,
        ),
        encoding="utf-8",
    )

    print(f"\n[+] 결과 이미지 HTML 저장 완료:")
    print(out)
    print("\n[+] 열기:")
    print(f'Start-Process "{out}"')


if __name__ == "__main__":
    main()
