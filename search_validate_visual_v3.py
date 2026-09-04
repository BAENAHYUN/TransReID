from __future__ import annotations

import argparse
import base64
import html
import math
import mimetypes
import re
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

# 검증용 기본 가중치
# - 사람 이미지 검색은 Re-ID 성격이 있으므로 SOLIDER를 조금 더 주되,
#   단일 모델 독주를 막기 위해 consensus/coverage를 강하게 반영함.
DEFAULT_PERSON_WEIGHTS = {
    "solider": 0.40,
    "irra": 0.25,
    "siglip2": 0.35,
}
DEFAULT_OBJECT_WEIGHTS = {
    "dinov2": 0.55,
    "siglip2": 0.45,
}


def normalize_vector(v):
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



def parse_query_identity(query_image: str | None):
    """
    SCVD crop 경로/파일명에서 query의 video, track, frame을 추출.
    예:
      ...\n017_converted\track_0002\frame_00000030_person_03.jpg
    """
    if not query_image:
        return None

    p = Path(query_image)
    parts = list(p.parts)

    video = None
    track_id = None
    frame_idx = None

    for part in parts:
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

    return {
        "video": video,
        "track_id": track_id,
        "frame_idx": frame_idx,
    }


def dhash64(path: str):
    """
    매우 보수적인 near-duplicate 확인용 64-bit dHash.
    Query 자기 자신/복제본 제거 목적. Re-ID 유사도 계산에는 사용하지 않음.
    """
    try:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
        diff = img[:, 1:] > img[:, :-1]
        value = 0
        for bit in diff.flatten():
            value = (value << 1) | int(bool(bit))
        return value
    except Exception:
        return None


def hamming64(a, b):
    if a is None or b is None:
        return 999
    return int((a ^ b).bit_count())


def make_filter(media_type: str, label: str | None = None):
    must = []
    if media_type != "all":
        must.append(
            FieldCondition(
                key="media_type",
                match=MatchValue(value=media_type),
            )
        )
    if label:
        must.append(
            FieldCondition(
                key="label",
                match=MatchValue(value=label),
            )
        )
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


def infer_gt_from_query(query_image: str | None):
    """
    SCVD query가
      ...\scvd_person_tracks_v1\v017_converted\track_0006\frame_xxx.jpg
    구조면 GT video/track을 자동 추론.
    """
    if not query_image:
        return None

    p = Path(query_image)
    parts = list(p.parts)

    try:
        idx = [x.lower() for x in parts].index("scvd_person_tracks_v1")
        video_dir = parts[idx + 1]
        track_dir = parts[idx + 2]
    except Exception:
        return None

    m = re.match(r"track_(\d+)", track_dir, re.I)
    if not m:
        return None

    return {
        "video": f"{video_dir}.avi",
        "track_id": int(m.group(1)),
        "track_dir": track_dir,
    }


def is_gt(payload: dict, gt: dict | None) -> bool:
    if not gt:
        return False
    return (
        str(payload.get("video", "")) == str(gt["video"])
        and int(payload.get("track_id", -999) or -999) == int(gt["track_id"])
    )


def query_hits(client, collection, vector_name, query_vector,
               limit, media_type, label):
    return client.query_points(
        collection_name=collection,
        using=vector_name,
        query=normalize_vector(query_vector).tolist(),
        query_filter=make_filter(media_type, label),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    ).points


def remove_self_match(hits, query_image):
    """
    Query 자기 자신/중복 point 제거:
      1) crop_path 동일
      2) SCVD video + track + frame 동일
      3) 이미지 dHash가 사실상 동일(Hamming <= 1)

    같은 track의 '다른 frame'은 제거하지 않음.
    """
    if not query_image:
        return hits

    q_path = canonical_path(query_image)
    q_identity = parse_query_identity(query_image)
    q_hash = dhash64(query_image)

    out = []

    for hit in hits:
        p = hit.payload or {}
        crop_path = p.get("crop_path", "")
        crop_canon = canonical_path(crop_path)

        # 1. 동일 경로
        if crop_canon and crop_canon == q_path:
            continue

        # 2. 동일 video + track + frame
        if q_identity:
            hit_video = p.get("video")
            hit_track = p.get("track_id")
            hit_frame = p.get("frame_idx")

            same_video = (
                q_identity.get("video") is not None
                and str(hit_video) == str(q_identity["video"])
            )
            same_track = (
                q_identity.get("track_id") is not None
                and int(hit_track if hit_track is not None else -999)
                == int(q_identity["track_id"])
            )
            same_frame = (
                q_identity.get("frame_idx") is not None
                and int(hit_frame if hit_frame is not None else -999)
                == int(q_identity["frame_idx"])
            )

            if same_video and same_track and same_frame:
                continue

        # 3. 경로/metadata가 달라도 이미지가 사실상 동일한 복제본이면 제거
        # 너무 공격적으로 제거하지 않도록 Hamming <= 1만 사용.
        if crop_path and q_hash is not None:
            hit_hash = dhash64(crop_path)
            if hamming64(q_hash, hit_hash) <= 1:
                continue

        out.append(hit)

    return out


def minmax_scores(hits):
    """
    모델별 score 범위가 달라 직접 합치면 왜곡되므로
    query마다 해당 모델 후보군 내부에서 0~1 정규화.
    """
    if not hits:
        return {}

    vals = np.asarray([float(h.score) for h in hits], dtype=np.float32)
    lo = float(vals.min())
    hi = float(vals.max())

    if abs(hi - lo) < 1e-12:
        return {str(h.id): 1.0 for h in hits}

    return {
        str(h.id): (float(h.score) - lo) / (hi - lo)
        for h in hits
    }


def rank_component(rank: int, rrf_k: int = 60) -> float:
    # 1위가 과도하게 독주하지 않도록 완만한 rank 점수
    return 1.0 / (rrf_k + rank)


def build_hybrid_rows(
    per_model_hits: dict[str, list],
    weights: dict[str, float],
    top_k: int,
    gt: dict | None,
):
    point_map = {}
    rank_map = defaultdict(dict)
    raw_map = defaultdict(dict)
    norm_map = defaultdict(dict)

    model_norm_maps = {
        name: minmax_scores(hits)
        for name, hits in per_model_hits.items()
    }

    for model_name, hits in per_model_hits.items():
        for rank, hit in enumerate(hits, start=1):
            pid = str(hit.id)
            point_map[pid] = hit
            rank_map[pid][model_name] = rank
            raw_map[pid][model_name] = float(hit.score)
            norm_map[pid][model_name] = model_norm_maps[model_name].get(pid, 0.0)

    active_models = list(per_model_hits.keys())
    total_weight = sum(weights.get(m, 1.0) for m in active_models) or 1.0

    rows = []

    for pid, hit in point_map.items():
        payload = hit.payload or {}

        present_models = [
            m for m in active_models
            if m in rank_map[pid]
        ]
        coverage = len(present_models) / max(1, len(active_models))

        # 1) 모델별 raw score의 query 내부 정규화 점수
        sim_sum = 0.0
        sim_w = 0.0
        for m in present_models:
            w = float(weights.get(m, 1.0))
            sim_sum += w * norm_map[pid][m]
            sim_w += w
        similarity_score = sim_sum / sim_w if sim_w else 0.0

        # 2) Weighted RRF를 0~1 근처로 정규화
        rrf_raw = 0.0
        rrf_best = 0.0
        for m in active_models:
            w = float(weights.get(m, 1.0))
            rrf_best += w * rank_component(1)
            if m in rank_map[pid]:
                rrf_raw += w * rank_component(rank_map[pid][m])
        rrf_score = rrf_raw / rrf_best if rrf_best else 0.0

        # 3) 여러 모델이 동시에 동의할수록 가산점
        consensus_bonus = coverage ** 2

        # 4) 한 모델에만 뜬 결과는 명시적으로 패널티
        lone_penalty = 0.15 if len(active_models) >= 2 and len(present_models) == 1 else 0.0

        # 최종
        final_score = (
            0.45 * similarity_score
            + 0.35 * rrf_score
            + 0.20 * consensus_bonus
            - lone_penalty
        )

        rows.append({
            "point_id": pid,
            "payload": payload,
            "final_score": float(final_score),
            "similarity_score": float(similarity_score),
            "rrf_score": float(rrf_score),
            "coverage": float(coverage),
            "consensus_count": len(present_models),
            "model_ranks": dict(rank_map[pid]),
            "model_raw_scores": dict(raw_map[pid]),
            "model_norm_scores": dict(norm_map[pid]),
            "is_gt": is_gt(payload, gt),
        })

    rows.sort(key=lambda x: x["final_score"], reverse=True)

    # 최종 결과는 같은 video/track 1개만 유지
    out = []
    seen_tracks = set()

    for row in rows:
        p = row["payload"]
        if p.get("media_type") == "video":
            key = p.get("track_key") or (
                f"{p.get('video')}|{p.get('track_id')}|{p.get('label')}"
            )
            if key in seen_tracks:
                continue
            seen_tracks.add(key)

        out.append(row)
        if len(out) >= top_k:
            break

    return out


def make_card(rank, row, model_order):
    p = row["payload"]
    crop = p.get("crop_path", "")
    src = image_to_data_uri(crop)
    media = p.get("media_type", "")
    gt_class = " gt" if row["is_gt"] else ""

    model_lines = []
    for m in model_order:
        rr = row["model_ranks"].get(m)
        raw = row["model_raw_scores"].get(m)
        norm = row["model_norm_scores"].get(m)
        if rr is None:
            model_lines.append(
                f'<div class="model muted"><b>{m.upper()}</b>: Top-K 밖</div>'
            )
        else:
            model_lines.append(
                f'<div class="model"><b>{m.upper()}</b>: '
                f'R{rr} / raw {raw:.4f} / norm {norm:.3f}</div>'
            )

    extra = ""
    if media == "video":
        frame = int(p.get("frame_idx", 0) or 0)
        vp = p.get("video_path", "") or ""
        extra = (
            f'<div><b>Video</b>: {html.escape(str(p.get("video","")))}</div>'
            f'<div><b>Track</b>: {html.escape(str(p.get("track_id","")))} '
            f'| <b>Frame</b>: {frame} '
            f'| <b>Time</b>: {html.escape(get_timestamp(frame, vp) or "N/A")}</div>'
        )

    badge = '<span class="gt-badge">GT 정답</span>' if row["is_gt"] else ""

    img_html = (
        f'<img src="{src}" loading="lazy">'
        if src else
        '<div class="missing">이미지 없음</div>'
    )

    return f"""
    <div class="card{gt_class}">
      <div class="rank">#{rank}</div>
      {badge}
      {img_html}
      <div class="meta">
        <div class="final"><b>Hybrid</b> {row["final_score"]:.4f}</div>
        <div><b>Similarity</b> {row["similarity_score"]:.4f}
             | <b>RRF</b> {row["rrf_score"]:.4f}
             | <b>Consensus</b> {row["consensus_count"]}/{len(model_order)}</div>
        {''.join(model_lines)}
        {extra}
        <div class="path">{html.escape(crop)}</div>
      </div>
    </div>
    """


def build_html(query_image, query_text, scope, media, rows, model_order, gt):
    q_uri = image_to_data_uri(query_image) if query_image else ""
    q_display = html.escape(query_image or query_text or "")

    query_img = f'<img class="query-img" src="{q_uri}">' if q_uri else ""

    gt_text = ""
    if gt:
        gt_text = (
            f'<div class="gt-info"><b>자동 GT</b>: '
            f'{html.escape(gt["video"])} / track {gt["track_id"]}</div>'
        )

    cards = "".join(
        make_card(i, row, model_order)
        for i, row in enumerate(rows, start=1)
    )

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>검색 검증 - Hybrid Ranking</title>
<style>
body {{
  font-family: Arial, "Malgun Gothic", sans-serif;
  margin: 24px;
  background: #f4f6f8;
  color: #20242a;
}}
h1 {{ margin: 0 0 16px; }}
.query {{
  display: flex;
  gap: 22px;
  align-items: flex-start;
  background: white;
  padding: 18px;
  border-radius: 14px;
  margin-bottom: 26px;
  box-shadow: 0 1px 6px rgba(0,0,0,.08);
}}
.query-img {{
  width: 300px;
  height: 300px;
  object-fit: contain;
  background: #e9ecef;
  border-radius: 10px;
}}
.gt-info {{
  margin-top: 10px;
  padding: 9px 12px;
  background: #e7f7ea;
  border-radius: 8px;
  font-size: 15px;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(310px, 1fr));
  gap: 18px;
}}
.card {{
  position: relative;
  background: white;
  border: 2px solid #d0d7de;
  border-radius: 14px;
  padding: 12px;
  box-shadow: 0 1px 6px rgba(0,0,0,.08);
}}
.card.gt {{
  border: 4px solid #22a447;
  background: #f6fff7;
}}
.card img {{
  width: 100%;
  height: 360px;
  object-fit: contain;
  background: #e9ecef;
  border-radius: 10px;
}}
.rank {{
  position: absolute;
  top: 16px;
  left: 16px;
  background: rgba(0,0,0,.82);
  color: white;
  padding: 6px 11px;
  border-radius: 8px;
  font-size: 20px;
  font-weight: 700;
  z-index: 2;
}}
.gt-badge {{
  position: absolute;
  top: 16px;
  right: 16px;
  background: #22a447;
  color: white;
  padding: 6px 11px;
  border-radius: 8px;
  font-weight: 700;
  z-index: 2;
}}
.meta {{
  padding-top: 10px;
  line-height: 1.5;
  font-size: 13px;
}}
.final {{
  font-size: 18px;
  margin-bottom: 4px;
}}
.model {{
  font-size: 13px;
}}
.muted {{
  color: #999;
}}
.path {{
  margin-top: 8px;
  color: #777;
  font-size: 10px;
  overflow-wrap: anywhere;
}}
.missing {{
  height: 360px;
  display: grid;
  place-items: center;
  background: #e9ecef;
  border-radius: 10px;
}}
.note {{
  background: #fff8dd;
  border: 1px solid #e6c75b;
  border-radius: 10px;
  padding: 12px 14px;
  margin-bottom: 20px;
}}
</style>
</head>
<body>
<h1>검색/검증 이미지 모음 — Hybrid Ranking</h1>

<div class="query">
  {query_img}
  <div>
    <div><b>Query</b>: {q_display}</div>
    <div><b>Scope</b>: {html.escape(scope)}</div>
    <div><b>Media</b>: {html.escape(media)}</div>
    {gt_text}
  </div>
</div>

<div class="note">
<b>랭킹 방식:</b>
모델별 raw score를 query 내부에서 0~1 정규화한 Similarity +
Weighted RRF +
여러 모델 동시 동의(Consensus)를 합산합니다.
한 모델에만 등장한 후보는 패널티를 줍니다.<br><b>Self-match 제거:</b> 동일 경로 + 동일 video/track/frame + 사실상 동일한 이미지 복제본(dHash)을 제외합니다.
</div>

<h2>최종 Top {len(rows)}</h2>
<div class="grid">
{cards}
</div>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(
        description="검색 검증용 Hybrid Ranking v3 + 강한 self-match 제거 + 시각화"
    )
    q = ap.add_mutually_exclusive_group(required=True)
    q.add_argument("--image")
    q.add_argument("--text")

    ap.add_argument("--scope", choices=["person", "object"], required=True)
    ap.add_argument("--media", choices=["all", "image", "video"], default="all")
    ap.add_argument("--label", default=None)
    ap.add_argument("--candidate-k", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--html-out", default=None)
    args = ap.parse_args()

    cfg = PipelineConfig.load(CONFIG_PATH)
    reg = EmbedderRegistry(cfg)
    router = Router(cfg, reg, input_format="rgb")
    client = QdrantClient(url=cfg.qdrant.url)

    query_image = None
    gt = None

    if args.image:
        qp = Path(args.image)
        if not qp.exists():
            raise FileNotFoundError(qp)
        query_image = str(qp)
        gt = infer_gt_from_query(query_image)
        vectors = router.embed_query_image(query_image, scope=args.scope)
    else:
        names = ["siglip2", "irra"] if args.scope == "person" else ["siglip2"]
        vectors = router.embed_query_text(args.text, names=names)

    if args.scope == "person":
        collection = PERSON_COLLECTION
        model_order = ["solider", "irra", "siglip2"]
        weights = DEFAULT_PERSON_WEIGHTS
        label = args.label or "person"
    else:
        collection = OBJECT_COLLECTION
        model_order = ["dinov2", "siglip2"]
        weights = DEFAULT_OBJECT_WEIGHTS
        label = args.label

    vectors = {k: v for k, v in vectors.items() if k in model_order}

    info = client.get_collection(collection)
    available_vectors = set(info.config.params.vectors.keys())

    per_model_hits = {}

    for model_name, vector in vectors.items():
        if model_name not in available_vectors:
            continue

        raw_hits = query_hits(
            client=client,
            collection=collection,
            vector_name=model_name,
            query_vector=vector,
            limit=args.candidate_k + (5 if query_image else 0),
            media_type=args.media,
            label=label,
        )

        per_model_hits[model_name] = remove_self_match(
            raw_hits,
            query_image,
        )[:args.candidate_k]

    rows = build_hybrid_rows(
        per_model_hits=per_model_hits,
        weights=weights,
        top_k=args.top_k,
        gt=gt,
    )

    print("\n" + "=" * 100)
    print("HYBRID RANKING RESULTS")
    print("=" * 100)

    for i, row in enumerate(rows, start=1):
        p = row["payload"]
        gtmark = " <-- GT" if row["is_gt"] else ""
        print(
            f"[{i:02d}] hybrid={row['final_score']:.6f} "
            f"| sim={row['similarity_score']:.4f} "
            f"| rrf={row['rrf_score']:.4f} "
            f"| consensus={row['consensus_count']}/{len(model_order)} "
            f"| {p.get('video','')} track={p.get('track_id')} "
            f"{gtmark}"
        )

    if gt:
        gt_rank = next(
            (i for i, row in enumerate(rows, start=1) if row["is_gt"]),
            None
        )
        print("\n[GT]")
        print(f"video={gt['video']} track={gt['track_id']} final_rank={gt_rank}")

    if args.html_out:
        out = Path(args.html_out)
        if not out.is_absolute():
            out = ROOT / out
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = ROOT / "results" / f"hybrid_{args.scope}_{args.media}_{stamp}.html"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        build_html(
            query_image=args.image,
            query_text=args.text,
            scope=args.scope,
            media=args.media,
            rows=rows,
            model_order=model_order,
            gt=gt,
        ),
        encoding="utf-8",
    )

    print("\n[+] HTML 저장:")
    print(out)
    print("\n[+] 열기:")
    print(f'Start-Process "{out}"')


if __name__ == "__main__":
    main()
