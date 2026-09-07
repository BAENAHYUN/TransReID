from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from config import PipelineConfig
from registry import EmbedderRegistry
from router import Router

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "pipeline.yaml"

PERSON_COLLECTION = "forensic_person"
OBJECT_COLLECTION = "forensic_object"

# 검색 목적에 맞춘 기본 가중치.
# Person image->image에서는 identity 중심 SOLIDER를 가장 크게 두되,
# IRRA/SigLIP2가 동시에 동의하지 않으면 독주하지 못하도록 consensus를 별도 반영한다.
PERSON_WEIGHTS = {
    "solider": 0.50,
    "irra": 0.20,
    "siglip2": 0.30,
}
OBJECT_WEIGHTS = {
    "dinov2": 0.60,
    "siglip2": 0.40,
}


def l2(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v if n <= 1e-12 else v / n


def canonical_path(path: str) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve()).lower()
    except Exception:
        return str(path).lower()


def make_filter(media_type: str, label: str | None):
    must = []
    if media_type != "all":
        must.append(FieldCondition(key="media_type", match=MatchValue(value=media_type)))
    if label:
        must.append(FieldCondition(key="label", match=MatchValue(value=label)))
    return Filter(must=must) if must else None


def parse_query_identity(query_image: str | None):
    if not query_image:
        return None

    p = Path(query_image)
    video = None
    track_id = None
    frame_idx = None

    for part in p.parts:
        if part.lower().endswith("_converted"):
            video = f"{part}.avi"
        m = re.fullmatch(r"track_(\d+)", part, re.I)
        if m:
            track_id = int(m.group(1))

    m = re.match(r"frame_(\d+)_", p.name, re.I)
    if m:
        frame_idx = int(m.group(1))

    if video is None and track_id is None and frame_idx is None:
        return None
    return {"video": video, "track_id": track_id, "frame_idx": frame_idx}


def infer_gt(query_image: str | None):
    q = parse_query_identity(query_image)
    if not q or q["video"] is None or q["track_id"] is None:
        return None
    return {"video": q["video"], "track_id": q["track_id"]}


def is_gt(payload: dict, gt: dict | None):
    if not gt:
        return False
    try:
        return (
            str(payload.get("video", "")) == str(gt["video"])
            and int(payload.get("track_id", -999)) == int(gt["track_id"])
        )
    except Exception:
        return False


def dhash64(path: str):
    try:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
        diff = img[:, 1:] > img[:, :-1]
        value = 0
        for bit in diff.reshape(-1):
            value = (value << 1) | int(bool(bit))
        return value
    except Exception:
        return None


def hamming64(a, b):
    if a is None or b is None:
        return 999
    return int((a ^ b).bit_count())


def remove_self_matches(hits, query_image: str | None):
    if not query_image:
        return hits

    q_path = canonical_path(query_image)
    q_id = parse_query_identity(query_image)
    q_hash = dhash64(query_image)

    out = []
    for hit in hits:
        p = hit.payload or {}
        crop = p.get("crop_path", "")

        # 1) exact path
        if canonical_path(crop) == q_path:
            continue

        # 2) exact SCVD frame identity
        if q_id:
            try:
                same_video = q_id["video"] is not None and str(p.get("video")) == str(q_id["video"])
                same_track = q_id["track_id"] is not None and int(p.get("track_id", -999)) == int(q_id["track_id"])
                same_frame = q_id["frame_idx"] is not None and int(p.get("frame_idx", -999)) == int(q_id["frame_idx"])
                if same_video and same_track and same_frame:
                    continue
            except Exception:
                pass

        # 3) almost identical duplicate
        if crop and q_hash is not None:
            if hamming64(q_hash, dhash64(crop)) <= 1:
                continue

        out.append(hit)
    return out


def query_vector(client, collection, model_name, vector, limit, media, label):
    return client.query_points(
        collection_name=collection,
        using=model_name,
        query=l2(vector).tolist(),
        query_filter=make_filter(media, label),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    ).points


def robust_score_map(scores: list[float]):
    """
    모델마다 cosine score 범위가 다르므로 raw score를 직접 합치지 않는다.
    median/MAD 기반 robust z-score -> sigmoid로 0~1 변환.
    min-max보다 극단값 하나에 덜 흔들린다.
    """
    if not scores:
        return []
    x = np.asarray(scores, dtype=np.float32)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = max(1e-6, 1.4826 * mad)
    z = np.clip((x - med) / scale, -8.0, 8.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32).tolist()


def entity_key(payload: dict, media: str):
    if media == "video" or payload.get("media_type") == "video":
        return (
            str(payload.get("video", "")),
            int(payload.get("track_id", -999) or -999),
            str(payload.get("label", "")),
        )
    return ("point", str(payload.get("crop_path", "")))


def aggregate_tracks_for_model(hits, media: str, top_frames: int = 3):
    """
    핵심 개선:
    기존에는 frame point 단위로 rank를 만든 뒤 마지막에 track 중복 제거.
    v4는 모델별 검색 결과를 먼저 track 단위로 묶고,
    각 track의 상위 여러 frame score를 합쳐 track score를 만든다.

    이 방식은 특정 1프레임의 우연한 고득점보다
    여러 프레임에서 반복적으로 잘 맞는 동일인을 더 높게 올린다.
    """
    if media != "video":
        rows = []
        scores = [float(h.score) for h in hits]
        norms = robust_score_map(scores)
        for i, (h, ns) in enumerate(zip(hits, norms), 1):
            rows.append({
                "key": entity_key(h.payload or {}, media),
                "repr_hit": h,
                "raw_score": float(h.score),
                "norm_score": float(ns),
                "frame_count": 1,
            })
        return rows

    grouped = defaultdict(list)
    for hit in hits:
        grouped[entity_key(hit.payload or {}, media)].append(hit)

    aggregated = []
    for key, ghits in grouped.items():
        ghits = sorted(ghits, key=lambda h: float(h.score), reverse=True)
        top = ghits[:max(1, top_frames)]
        vals = [float(h.score) for h in top]

        # best frame 60% + remaining top frames mean 40%
        best = vals[0]
        mean_top = float(np.mean(vals))
        track_raw = 0.60 * best + 0.40 * mean_top

        aggregated.append({
            "key": key,
            "repr_hit": top[0],
            "raw_score": track_raw,
            "best_score": best,
            "mean_top": mean_top,
            "frame_count": len(ghits),
        })

    aggregated.sort(key=lambda x: x["raw_score"], reverse=True)
    norms = robust_score_map([r["raw_score"] for r in aggregated])
    for row, ns in zip(aggregated, norms):
        row["norm_score"] = float(ns)
    return aggregated


def build_query_vectors(router, query_image, query_text, scope, model_order, flip_tta):
    if query_text is not None:
        names = ["siglip2", "irra"] if scope == "person" else ["siglip2"]
        return {
            k: l2(v)
            for k, v in router.embed_query_text(query_text, names=names).items()
            if k in model_order
        }

    original = router.embed_query_image(query_image, scope=scope)
    original = {k: l2(v) for k, v in original.items() if k in model_order}

    if not flip_tta:
        return original

    # Horizontal flip TTA:
    # 원본/좌우반전 embedding을 평균 후 다시 L2 normalize.
    # 사람의 방향/포즈 변화에 대한 query robustness를 높이기 위한 선택 옵션.
    im = Image.open(query_image).convert("RGB")
    flipped = ImageOps.mirror(im)
    flip_vecs = router.embed_query_image(flipped, scope=scope)

    out = {}
    for name, vec in original.items():
        if name in flip_vecs:
            out[name] = l2(l2(vec) + l2(flip_vecs[name]))
        else:
            out[name] = vec
    return out


def hybrid_fusion(per_model_entities, weights, model_order, top_k, gt):
    """
    최종 점수:
      55% calibrated similarity
      25% rank score
      20% consensus

    single-model candidate는 추가 감점.
    """
    point = {}

    for model, entities in per_model_entities.items():
        n = max(1, len(entities))
        for rank, e in enumerate(entities, 1):
            key = e["key"]
            if key not in point:
                point[key] = {
                    "payload": e["repr_hit"].payload or {},
                    "repr_hit": e["repr_hit"],
                    "model_ranks": {},
                    "model_raw": {},
                    "model_norm": {},
                    "model_frames": {},
                }
            point[key]["model_ranks"][model] = rank
            point[key]["model_raw"][model] = float(e["raw_score"])
            point[key]["model_norm"][model] = float(e["norm_score"])
            point[key]["model_frames"][model] = int(e.get("frame_count", 1))

    rows = []
    active = [m for m in model_order if m in per_model_entities]
    total_w = sum(weights.get(m, 1.0) for m in active) or 1.0

    for key, row in point.items():
        present = [m for m in active if m in row["model_ranks"]]
        coverage = len(present) / max(1, len(active))

        sim_num = 0.0
        sim_den = 0.0
        rank_num = 0.0
        rank_den = 0.0

        for m in active:
            w = weights.get(m, 1.0)
            rank_den += w
            if m not in present:
                continue

            sim_num += w * row["model_norm"][m]
            sim_den += w

            r = row["model_ranks"][m]
            # rank 1=1, rank 2≈0.91, rank 10≈0.63, rank 100≈0.30
            rank_component = 1.0 / (1.0 + 0.10 * (r - 1))
            rank_num += w * rank_component

        similarity = sim_num / sim_den if sim_den else 0.0
        rank_score = rank_num / rank_den if rank_den else 0.0

        # 3/3은 1.0, 2/3은 0.444, 1/3은 0.111
        consensus = coverage ** 2

        final = (
            0.55 * similarity
            + 0.25 * rank_score
            + 0.20 * consensus
        )

        if len(active) >= 3 and len(present) == 1:
            final -= 0.12
        elif len(active) >= 3 and len(present) == 2:
            final -= 0.02

        row.update({
            "key": key,
            "score": float(final),
            "similarity": float(similarity),
            "rank_score": float(rank_score),
            "consensus": float(consensus),
            "consensus_count": len(present),
            "is_gt": is_gt(row["payload"], gt),
        })
        rows.append(row)

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top_k]


def image_uri(path):
    try:
        p = Path(path)
        if not p.exists():
            return ""
        mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")
    except Exception:
        return ""


def video_time(frame_idx, video_path):
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if fps <= 0:
            return ""
        sec = int(frame_idx) / fps
        return f"{int(sec // 60):02d}:{sec % 60:05.2f}"
    except Exception:
        return ""


def make_html(query_image, query_text, rows, model_order, gt, scope, media, flip_tta):
    q_uri = image_uri(query_image) if query_image else ""
    q_img = f'<img class="query" src="{q_uri}">' if q_uri else ""

    cards = []
    for rank, row in enumerate(rows, 1):
        p = row["payload"]
        crop = p.get("crop_path", "")
        uri = image_uri(crop)
        image = f'<img src="{uri}">' if uri else '<div class="missing">image missing</div>'
        gt_cls = " gt" if row["is_gt"] else ""
        gt_badge = '<div class="gtbadge">AUTO GT</div>' if row["is_gt"] else ""

        model_lines = []
        for m in model_order:
            if m in row["model_ranks"]:
                model_lines.append(
                    f'<div><b>{m.upper()}</b> '
                    f'R{row["model_ranks"][m]} '
                    f'raw={row["model_raw"][m]:.4f} '
                    f'cal={row["model_norm"][m]:.3f} '
                    f'frames={row["model_frames"][m]}</div>'
                )
            else:
                model_lines.append(f'<div class="muted"><b>{m.upper()}</b> Top-K 밖</div>')

        media_lines = ""
        if p.get("media_type") == "video":
            frame = int(p.get("frame_idx", 0) or 0)
            media_lines = (
                f'<div><b>Video</b> {html.escape(str(p.get("video","")))}</div>'
                f'<div><b>Track</b> {html.escape(str(p.get("track_id","")))} '
                f'| <b>Frame</b> {frame} '
                f'| <b>Time</b> {video_time(frame, p.get("video_path","")) or "N/A"}</div>'
            )

        cards.append(f"""
        <div class="card{gt_cls}">
          <div class="rank">#{rank}</div>
          {gt_badge}
          {image}
          <div class="meta">
            <div class="score">Hybrid {row["score"]:.4f}</div>
            <div>Similarity {row["similarity"]:.4f}
                 | Rank {row["rank_score"]:.4f}
                 | Consensus {row["consensus_count"]}/{len(model_order)}</div>
            {''.join(model_lines)}
            {media_lines}
            <div class="path">{html.escape(crop)}</div>
          </div>
        </div>
        """)

    gtline = ""
    if gt:
        gtline = f'<div><b>Auto GT</b>: {html.escape(gt["video"])} / track {gt["track_id"]}</div>'

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Search Validation v4</title>
<style>
body{{font-family:Arial,"Malgun Gothic",sans-serif;background:#f4f6f8;margin:24px;color:#20242a}}
.head{{background:#fff;padding:18px;border-radius:14px;display:flex;gap:20px;align-items:flex-start;margin-bottom:22px}}
.query{{width:300px;height:300px;object-fit:contain;background:#eee;border-radius:10px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:18px}}
.card{{position:relative;background:#fff;border:2px solid #d0d7de;border-radius:14px;padding:12px}}
.card.gt{{border:4px solid #23a047;background:#f5fff7}}
.card img{{width:100%;height:360px;object-fit:contain;background:#eee;border-radius:10px}}
.rank{{position:absolute;top:16px;left:16px;background:#111d;color:#fff;padding:6px 11px;border-radius:8px;font-size:20px;font-weight:700}}
.gtbadge{{position:absolute;top:16px;right:16px;background:#23a047;color:#fff;padding:6px 10px;border-radius:8px;font-weight:700}}
.meta{{font-size:13px;line-height:1.5;padding-top:9px}}
.score{{font-size:18px;font-weight:700}}
.path{{font-size:10px;color:#777;overflow-wrap:anywhere;margin-top:7px}}
.muted{{color:#999}}
.note{{background:#fff8dc;border:1px solid #e8cf72;border-radius:10px;padding:12px;margin-bottom:18px}}
.missing{{height:360px;display:grid;place-items:center;background:#eee}}
</style>
</head>
<body>
<h1>검색/검증 v4 — Track Aggregation + Robust Fusion</h1>
<div class="head">
{q_img}
<div>
<div><b>Query</b>: {html.escape(query_image or query_text or "")}</div>
<div><b>Scope</b>: {scope} / <b>Media</b>: {media}</div>
<div><b>Flip TTA</b>: {"ON" if flip_tta else "OFF"}</div>
{gtline}
</div>
</div>

<div class="note">
<b>v4 핵심:</b>
video 검색은 frame point를 바로 fusion하지 않고 먼저 <b>track 단위로 묶어 상위 여러 프레임을 집계</b>합니다.
모델별 점수는 median/MAD 기반으로 robust calibration하고,
그 뒤 calibrated similarity + model rank + consensus를 결합합니다.
</div>

<div class="grid">
{''.join(cards)}
</div>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser(description="고정 DB 기반 검색 정확도 개선/검증 v4")
    q = ap.add_mutually_exclusive_group(required=True)
    q.add_argument("--image")
    q.add_argument("--text")

    ap.add_argument("--scope", choices=["person", "object"], required=True)
    ap.add_argument("--media", choices=["all", "image", "video"], default="all")
    ap.add_argument("--label", default=None)
    ap.add_argument("--candidate-k", type=int, default=300)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--track-top-frames", type=int, default=3)
    ap.add_argument("--flip-tta", action="store_true")
    ap.add_argument("--html-out", default=None)
    args = ap.parse_args()

    cfg = PipelineConfig.load(CONFIG_PATH)
    reg = EmbedderRegistry(cfg)
    router = Router(cfg, reg, input_format="rgb")
    client = QdrantClient(url=cfg.qdrant.url)

    if args.scope == "person":
        collection = PERSON_COLLECTION
        model_order = ["solider", "irra", "siglip2"]
        weights = PERSON_WEIGHTS
        label = args.label or "person"
    else:
        collection = OBJECT_COLLECTION
        model_order = ["dinov2", "siglip2"]
        weights = OBJECT_WEIGHTS
        label = args.label

    if args.image:
        qpath = Path(args.image)
        if not qpath.exists():
            raise FileNotFoundError(qpath)
        query_image = str(qpath)
        query_text = None
    else:
        query_image = None
        query_text = args.text

    gt = infer_gt(query_image)

    query_vectors = build_query_vectors(
        router=router,
        query_image=query_image,
        query_text=query_text,
        scope=args.scope,
        model_order=model_order,
        flip_tta=args.flip_tta,
    )

    info = client.get_collection(collection)
    available = set(info.config.params.vectors.keys())

    per_model_entities = {}

    print("\n" + "=" * 108)
    print("SEARCH VALIDATION V4")
    print("=" * 108)
    print("scope          :", args.scope)
    print("media          :", args.media)
    print("candidate_k    :", args.candidate_k)
    print("track_top_frames:", args.track_top_frames)
    print("flip_tta       :", args.flip_tta)
    print()

    for model in model_order:
        if model not in query_vectors or model not in available:
            continue

        hits = query_vector(
            client=client,
            collection=collection,
            model_name=model,
            vector=query_vectors[model],
            limit=args.candidate_k + (20 if query_image else 0),
            media=args.media,
            label=label,
        )

        hits = remove_self_matches(hits, query_image)[:args.candidate_k]

        entities = aggregate_tracks_for_model(
            hits,
            media=args.media,
            top_frames=args.track_top_frames,
        )
        per_model_entities[model] = entities

        print(f"[{model.upper()}] point_hits={len(hits)} entity_hits={len(entities)}")
        for i, e in enumerate(entities[:5], 1):
            p = e["repr_hit"].payload or {}
            print(
                f"  #{i:02d} raw={e['raw_score']:.6f} cal={e['norm_score']:.4f} "
                f"| {p.get('video','')} track={p.get('track_id')} "
                f"| frames={e.get('frame_count',1)}"
            )
        print()

    rows = hybrid_fusion(
        per_model_entities=per_model_entities,
        weights=weights,
        model_order=model_order,
        top_k=args.top_k,
        gt=gt,
    )

    print("=" * 108)
    print("FINAL HYBRID")
    print("=" * 108)
    for i, row in enumerate(rows, 1):
        p = row["payload"]
        mark = " <-- AUTO GT" if row["is_gt"] else ""
        print(
            f"[{i:02d}] hybrid={row['score']:.6f} "
            f"| sim={row['similarity']:.4f} "
            f"| rank={row['rank_score']:.4f} "
            f"| consensus={row['consensus_count']}/{len(model_order)} "
            f"| {p.get('video','')} track={p.get('track_id')} "
            f"{mark}"
        )

    if gt:
        gt_rank = next((i for i, r in enumerate(rows, 1) if r["is_gt"]), None)
        print(f"\n[AUTO GT] {gt['video']} / track={gt['track_id']} / rank={gt_rank}")

    if args.html_out:
        out = Path(args.html_out)
        if not out.is_absolute():
            out = ROOT / out
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = ROOT / "results" / f"validation_v4_{args.scope}_{args.media}_{stamp}.html"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        make_html(
            query_image=query_image,
            query_text=query_text,
            rows=rows,
            model_order=model_order,
            gt=gt,
            scope=args.scope,
            media=args.media,
            flip_tta=args.flip_tta,
        ),
        encoding="utf-8",
    )

    print("\n[+] HTML:", out)
    print(f'[+] OPEN: Start-Process "{out}"')


if __name__ == "__main__":
    main()
