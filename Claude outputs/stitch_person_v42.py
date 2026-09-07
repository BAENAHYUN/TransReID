"""
Person Tracklet Stitching V4.2
==============================
V4.1 대비 주요 변경:
  1. general rule 폐지 → gap-tier 3단계 (short / medium / long)
  2. 모든 rule에 endpoint hard gate 추가 (endpoint 없으면 즉시 거부)
  3. gallery vote ratio 도입 (몇 %의 gallery member가 동의하는지)
  4. identity global embedding drift cap (신규 track이 전체의 20% 이상 영향 불가)
  5. position score에서 time 이중반영 제거 (fused 공정성 수정)
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import traceback
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PipelineConfig

CONFIG_PATH = ROOT / "pipeline.yaml"
TRACK_ROOT  = ROOT / "data" / "video_tracks" / "person"


# ============================================================
# Basic utilities (unchanged from V4.1)
# ============================================================

def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v if n <= 1e-12 else v / n


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected 2D array, got {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid JSONL: {path} line={line_no}"
                ) from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rgb(path: Path):
    with Image.open(path) as im:
        return im.convert("RGB").copy()


def load_embedder(cfg: PipelineConfig, name: str):
    if name not in cfg.retrievers:
        raise KeyError(f"pipeline.yaml retrievers에 '{name}'이 없습니다.")
    spec = cfg.retrievers[name]
    module = importlib.import_module(spec.module)
    cls = getattr(module, spec.class_name)
    embedder = cls(**dict(spec.params))
    if not hasattr(embedder, "embed_crops"):
        raise TypeError(f"{spec.class_name} does not provide embed_crops()")
    return embedder, spec


# ============================================================
# Tracklet construction (unchanged from V4.1)
# ============================================================

def final_class(records: list[dict]) -> str:
    labels = [
        str(
            r.get("final_class_name")
            or r.get("class_name")
            or r.get("raw_class_name")
            or "unknown"
        )
        for r in records
    ]
    return Counter(labels).most_common(1)[0][0]


def valid_crop_rows(records: list[dict]) -> list[dict]:
    out = []
    for row in records:
        crop = Path(str(row.get("crop_path", "")))
        if crop.exists() and crop.is_file():
            out.append(row)
    return out


def select_global_rows(records: list[dict], top_k: int) -> list[dict]:
    rows = valid_crop_rows(records)
    rows.sort(key=lambda r: int(r.get("frame_idx", 0)))
    if len(rows) <= top_k:
        return rows
    bins = np.array_split(np.arange(len(rows)), top_k)
    selected = []
    for bin_indices in bins:
        if len(bin_indices) == 0:
            continue
        candidates = [rows[int(i)] for i in bin_indices]
        best = max(
            candidates,
            key=lambda r: (float(r.get("confidence", 0.0)), int(r.get("frame_idx", 0))),
        )
        selected.append(best)
    selected.sort(key=lambda r: int(r.get("frame_idx", 0)))
    return selected


def select_start_rows(records: list[dict], k: int) -> list[dict]:
    rows = valid_crop_rows(records)
    rows.sort(key=lambda r: int(r.get("frame_idx", 0)))
    return rows[:k]


def select_end_rows(records: list[dict], k: int) -> list[dict]:
    rows = valid_crop_rows(records)
    rows.sort(key=lambda r: int(r.get("frame_idx", 0)))
    return rows[-k:]


def build_tracklets(rows: list[dict]) -> dict[int, dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("track_id") is None:
            continue
        grouped[int(row["track_id"])].append(row)

    tracklets = {}
    for track_id, items in grouped.items():
        items.sort(key=lambda r: int(r.get("frame_idx", 0)))
        start = items[0]
        end   = items[-1]
        start_sec = float(start.get("timestamp_sec", 0.0))
        end_sec   = float(end.get("timestamp_sec", 0.0))
        if start_sec > end_sec:
            warnings.warn(
                f"track_id={track_id}: start_sec({start_sec}) > end_sec({end_sec})",
                RuntimeWarning,
            )
        tracklets[track_id] = {
            "track_id":    track_id,
            "records":     items,
            "class_name":  "person",
            "start_sec":   start_sec,
            "end_sec":     end_sec,
            "start_frame": int(start.get("frame_idx", 0)),
            "end_frame":   int(end.get("frame_idx", 0)),
            "start_bbox":  list(start.get("bbox") or []),
            "end_bbox":    list(end.get("bbox") or []),
            "observations": len(items),
            "global_embedding": None,
            "start_vectors":    None,
            "end_vectors":      None,
            "global_crops": [],
            "start_crops":  [],
            "end_crops":    [],
        }
    return tracklets


# ============================================================
# SOLIDER embedding (unchanged from V4.1)
# ============================================================

def mean_embedding(
    vectors: np.ndarray | list[np.ndarray],
    *,
    already_normalized: bool = False,
) -> np.ndarray | None:
    if vectors is None:
        return None
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if not already_normalized:
        arr = normalize_rows(arr)
    return normalize(np.mean(arr, axis=0))


def embed_tracklets(
    embedder,
    tracklets: dict[int, dict],
    global_top_k: int,
    endpoint_k: int,
) -> list[dict]:
    path_to_owners: dict[str, list[tuple[int, str]]] = defaultdict(list)
    ordered_paths: list[str] = []
    seen_paths: set[str] = set()
    crop_errors: list[dict] = []

    for track_id in sorted(tracklets):
        t = tracklets[track_id]
        selections = {
            "global": select_global_rows(t["records"], top_k=global_top_k),
            "start":  select_start_rows(t["records"], k=endpoint_k),
            "end":    select_end_rows(t["records"],   k=endpoint_k),
        }
        for kind, selected in selections.items():
            crop_paths = [
                str(Path(str(row["crop_path"])).resolve())
                for row in selected
            ]
            t[f"{kind}_crops"] = crop_paths
            for crop_path in crop_paths:
                path_to_owners[crop_path].append((track_id, kind))
                if crop_path not in seen_paths:
                    seen_paths.add(crop_path)
                    ordered_paths.append(crop_path)

    if not ordered_paths:
        return crop_errors

    valid_paths = []
    images = []
    for crop_path in ordered_paths:
        try:
            images.append(load_rgb(Path(crop_path)))
            valid_paths.append(crop_path)
        except Exception as exc:
            crop_errors.append({
                "crop_path":  crop_path,
                "error_type": type(exc).__name__,
                "error":      str(exc),
            })

    if not valid_paths:
        return crop_errors

    try:
        vecs = np.asarray(
            embedder.embed_crops(images, input_format="rgb"),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError(
            f"SOLIDER embed_crops failed for {len(valid_paths)} crops"
        ) from exc

    if vecs.ndim != 2 or vecs.shape[0] != len(valid_paths):
        raise RuntimeError(
            f"SOLIDER embedding shape mismatch: {vecs.shape}, expected rows={len(valid_paths)}"
        )

    vecs = normalize_rows(vecs)
    bucket: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    for crop_path, vec in zip(valid_paths, vecs):
        for owner in path_to_owners[crop_path]:
            bucket[owner].append(vec)

    for track_id in sorted(tracklets):
        t = tracklets[track_id]
        global_vecs = np.asarray(bucket.get((track_id, "global"), []), dtype=np.float32)
        start_vecs  = np.asarray(bucket.get((track_id, "start"),  []), dtype=np.float32)
        end_vecs    = np.asarray(bucket.get((track_id, "end"),    []), dtype=np.float32)

        t["global_embedding"] = mean_embedding(global_vecs, already_normalized=True)
        if start_vecs.size:
            t["start_vectors"] = normalize_rows(start_vecs.reshape(len(start_vecs), -1))
        if end_vecs.size:
            t["end_vectors"]   = normalize_rows(end_vecs.reshape(len(end_vecs), -1))

    return crop_errors


# ============================================================
# Pairwise endpoint appearance (unchanged from V4.1)
# ============================================================

def endpoint_pairwise_stats(
    prev_end_vectors: np.ndarray | None,
    next_start_vectors: np.ndarray | None,
    pair_top_k: int,
) -> dict:
    if prev_end_vectors is None or next_start_vectors is None:
        return {
            "pair_count": 0,
            "endpoint_max": None,
            "endpoint_mean": None,
            "endpoint_topk_mean": None,
            "endpoint_top_values": [],
        }

    a = normalize_rows(np.asarray(prev_end_vectors, dtype=np.float32))
    b = normalize_rows(np.asarray(next_start_vectors, dtype=np.float32))
    sims = a @ b.T
    flat = sims.reshape(-1)

    if flat.size == 0:
        return {
            "pair_count": 0,
            "endpoint_max": None,
            "endpoint_mean": None,
            "endpoint_topk_mean": None,
            "endpoint_top_values": [],
        }

    k = min(max(1, int(pair_top_k)), int(flat.size))
    top_values = np.sort(flat)[-k:][::-1]

    return {
        "pair_count":       int(flat.size),
        "endpoint_max":     float(np.max(flat)),
        "endpoint_mean":    float(np.mean(flat)),
        "endpoint_topk_mean": float(np.mean(top_values)),
        "endpoint_top_values": [round(float(x), 6) for x in top_values.tolist()],
    }


# ============================================================
# Time / position / scale score (unchanged from V4.1)
# ============================================================

def bbox_center_and_diag(bbox):
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = map(float, bbox)
    if x2 <= x1 or y2 <= y1:
        return None
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    diag = math.hypot(x2 - x1, y2 - y1)
    return cx, cy, diag


def position_score(prev_bbox, next_bbox) -> tuple[float, float | None]:
    a = bbox_center_and_diag(prev_bbox)
    b = bbox_center_and_diag(next_bbox)
    if a is None or b is None:
        return 0.0, None
    ax, ay, ad = a
    bx, by, bd = b
    dist  = math.hypot(ax - bx, ay - by)
    scale = max(1.0, 0.5 * (ad + bd))
    ratio = dist / scale
    score = math.exp(-ratio / 3.0)
    return float(score), float(ratio)


def temporal_score(gap_sec: float, max_gap_sec: float) -> float:
    if max_gap_sec <= 0:
        return 1.0 if gap_sec <= 0 else 0.0
    return max(0.0, 1.0 - (gap_sec / max_gap_sec))


def bbox_scale_score(prev_bbox, next_bbox) -> tuple[float, float | None]:
    a = bbox_center_and_diag(prev_bbox)
    b = bbox_center_and_diag(next_bbox)
    if a is None or b is None:
        return 0.0, None
    prev_diag = max(float(a[2]), 1e-6)
    next_diag = max(float(b[2]), 1e-6)
    ratio = max(prev_diag, next_diag) / min(prev_diag, next_diag)
    score = 1.0 / ratio
    return float(score), float(ratio)


# ============================================================
# V4.2 gallery-level global + vote ratio  ← 변경
# ============================================================

def gallery_global_stats(
    identity: dict,
    tracklet: dict,
    top_k: int,
    vote_min_sim: float,      # ← V4.2 추가: vote 기준 유사도
) -> dict:
    """
    identity gallery 각 tracklet global embedding × 새 tracklet global 비교.

    추가: gallery_vote_ratio
      - gallery member 중 vote_min_sim 이상인 비율
      - 0.0 = 아무도 동의 안 함 / 1.0 = 전원 동의
    """
    track_global = tracklet.get("global_embedding")

    if track_global is None:
        return {
            "gallery_global_count":    0,
            "gallery_global_max":      None,
            "gallery_global_topk_mean": None,
            "gallery_global_values":   [],
            "gallery_global_top_track_ids": [],
            "gallery_vote_ratio":      0.0,   # ← V4.2
            "gallery_n_agree":         0,     # ← V4.2
            "gallery_vote_min_sim":    vote_min_sim,
        }

    scored = []
    for item in identity.get("gallery", []):
        gallery_global = item.get("global_embedding")
        if gallery_global is None:
            continue
        sim = cosine(gallery_global, track_global)
        scored.append((int(item["track_id"]), float(sim)))

    if not scored:
        return {
            "gallery_global_count":    0,
            "gallery_global_max":      None,
            "gallery_global_topk_mean": None,
            "gallery_global_values":   [],
            "gallery_global_top_track_ids": [],
            "gallery_vote_ratio":      0.0,
            "gallery_n_agree":         0,
            "gallery_vote_min_sim":    vote_min_sim,
        }

    scored.sort(key=lambda x: x[1], reverse=True)

    k   = min(max(1, int(top_k)), len(scored))
    top = scored[:k]

    # V4.2: vote ratio
    n_agree    = sum(1 for _, sim in scored if sim >= vote_min_sim)
    vote_ratio = n_agree / len(scored)

    return {
        "gallery_global_count":    len(scored),
        "gallery_global_max":      float(scored[0][1]),
        "gallery_global_topk_mean": float(np.mean([sim for _, sim in top])),
        "gallery_global_values": [
            {"track_id": int(tid), "similarity": round(float(sim), 6)}
            for tid, sim in scored
        ],
        "gallery_global_top_track_ids": [int(tid) for tid, _ in top],
        "gallery_vote_ratio":  float(vote_ratio),  # ← V4.2
        "gallery_n_agree":     int(n_agree),         # ← V4.2
        "gallery_vote_min_sim": float(vote_min_sim), # ← V4.2
    }


# ============================================================
# V4.2 candidate score  ← gallery_vote_min_sim 파라미터 추가
# ============================================================

def make_candidate_score(
    identity: dict,
    predecessor: dict,
    tracklet: dict,
    gap_sec: float,
    max_gap_sec: float,
    pair_top_k: int,
    gallery_global_top_k: int,
    gallery_vote_min_sim: float,   # ← V4.2 추가
    endpoint_weight: float,
    global_weight: float,
    time_weight: float,
    position_weight: float,
    scale_weight: float,
):
    pair_stats    = endpoint_pairwise_stats(
        predecessor.get("end_vectors"),
        tracklet.get("start_vectors"),
        pair_top_k=pair_top_k,
    )
    gallery_stats = gallery_global_stats(
        identity,
        tracklet,
        top_k=gallery_global_top_k,
        vote_min_sim=gallery_vote_min_sim,   # ← V4.2
    )

    endpoint_score = pair_stats["endpoint_topk_mean"]

    predecessor_global = predecessor.get("global_embedding")
    identity_global    = identity.get("global_embedding")
    track_global       = tracklet.get("global_embedding")

    predecessor_global_sim = None
    identity_global_sim    = None

    if predecessor_global is not None and track_global is not None:
        predecessor_global_sim = cosine(predecessor_global, track_global)
    if identity_global is not None and track_global is not None:
        identity_global_sim = cosine(identity_global, track_global)

    available_global = [x for x in (predecessor_global_sim, identity_global_sim) if x is not None]
    global_sim = max(available_global) if available_global else None

    if global_sim is None:
        global_source = None
    elif (predecessor_global_sim is not None and global_sim == predecessor_global_sim):
        global_source = "predecessor"
    else:
        global_source = "identity"

    t_score = temporal_score(gap_sec, max_gap_sec)
    p_score, center_distance_ratio = position_score(
        predecessor.get("end_bbox"), tracklet.get("start_bbox")
    )
    scale_score, scale_ratio = bbox_scale_score(
        predecessor.get("end_bbox"), tracklet.get("start_bbox")
    )

    # V4.2: position에서 time 이중반영 제거 (raw p_score 사용)
    parts = []
    if endpoint_score is not None:
        parts.append((endpoint_weight, endpoint_score))
    if global_sim is not None:
        parts.append((global_weight, global_sim))
    parts.append((time_weight,     t_score))
    parts.append((position_weight, p_score))   # V4.1: p_score*t_score → V4.2: p_score
    parts.append((scale_weight,    scale_score))

    weight_sum = sum(w for w, _ in parts)
    fused = (
        -1.0
        if weight_sum <= 0
        else sum(w * s for w, s in parts) / weight_sum
    )

    return {
        **pair_stats,
        **gallery_stats,
        "predecessor_global_similarity": predecessor_global_sim,
        "identity_global_similarity":    identity_global_sim,
        "global_similarity":             global_sim,
        "global_similarity_source":      global_source,
        "time_score":                    float(t_score),
        "position_score":                float(p_score),
        "center_distance_ratio":         center_distance_ratio,
        "bbox_scale_score":              float(scale_score),
        "bbox_scale_ratio":              scale_ratio,
        "fused_score":                   float(fused),
    }


# ============================================================
# V4.2 결정 함수  ← 완전 재작성 (general 폐지 → gap-tier)
# ============================================================

def decide_candidate_v42(
    score: dict,
    gap_sec: float,
    *,
    max_gap_sec: float,
    max_scale_ratio: float,
    # ── Short gap tier ──────────────────────
    short_gap_sec: float,
    short_min_endpoint_max: float,
    short_min_endpoint_topk: float,
    short_min_position: float,
    short_min_global: float,
    # ── Medium gap tier ─────────────────────
    medium_gap_sec: float,
    medium_min_endpoint_topk: float,
    medium_min_endpoint_max: float,
    medium_min_global: float,
    medium_min_vote_ratio: float,
    medium_fused_threshold: float,
    # ── Long gap tier ───────────────────────
    long_min_endpoint_topk: float,
    long_min_endpoint_max: float,
    long_min_global: float,
    long_min_vote_ratio: float,
    long_min_gallery_topk: float,
    long_fused_threshold: float,
) -> dict:
    """
    V4.2 gap-tier decision:

      Short  (gap ≤ short_gap_sec)            : endpoint + position 위주
      Medium (short_gap_sec < gap ≤ medium_gap_sec) : endpoint + global + gallery vote
      Long   (medium_gap_sec < gap ≤ max_gap_sec)   : 모든 신호 + gallery 과반수

    공통 Hard Gate:
      1. endpoint 없으면 즉시 거부
      2. scale_ratio > max_scale_ratio 이면 거부
    """
    endpoint_topk   = score.get("endpoint_topk_mean")
    endpoint_max    = score.get("endpoint_max")
    scale_ratio     = score.get("bbox_scale_ratio")
    global_sim      = score.get("global_similarity")
    gallery_topk    = score.get("gallery_global_topk_mean")
    vote_ratio      = score.get("gallery_vote_ratio", 1.0)
    position        = score.get("position_score", 0.0)
    fused           = score.get("fused_score", -1.0)

    _base = {
        "short_gap_pass":  False,
        "medium_gap_pass": False,
        "long_gap_pass":   False,
    }

    # ── Hard gate 1: max gap ────────────────────────────────
    if gap_sec > max_gap_sec:
        return {**_base, "accepted": False, "accepted_rule": None,
                "reject_reason": "max_gap",
                "scale_ok": True, "has_endpoint": True, "vote_ratio": vote_ratio}

    # ── Hard gate 2: scale ──────────────────────────────────
    scale_ok = (scale_ratio is None or scale_ratio <= max_scale_ratio)
    if not scale_ok:
        return {**_base, "accepted": False, "accepted_rule": None,
                "reject_reason": "scale",
                "scale_ok": False, "has_endpoint": True, "vote_ratio": vote_ratio}

    # ── Hard gate 3: endpoint 반드시 존재 ───────────────────
    has_endpoint = (endpoint_topk is not None and endpoint_max is not None)
    if not has_endpoint:
        return {**_base, "accepted": False, "accepted_rule": None,
                "reject_reason": "no_endpoint",
                "scale_ok": True, "has_endpoint": False, "vote_ratio": vote_ratio}

    # ── Tier evaluation ─────────────────────────────────────
    short_gap_pass  = False
    medium_gap_pass = False
    long_gap_pass   = False

    if gap_sec <= short_gap_sec:
        short_gap_pass = (
            endpoint_max  >= short_min_endpoint_max
            and endpoint_topk >= short_min_endpoint_topk
            and position      >= short_min_position
            and (global_sim is None or global_sim >= short_min_global)
        )

    elif gap_sec <= medium_gap_sec:
        medium_gap_pass = (
            endpoint_topk >= medium_min_endpoint_topk
            and endpoint_max  >= medium_min_endpoint_max
            and global_sim is not None
            and global_sim    >= medium_min_global
            and vote_ratio    >= medium_min_vote_ratio
            and fused         >= medium_fused_threshold
        )

    else:  # medium_gap_sec < gap <= max_gap_sec
        long_gap_pass = (
            endpoint_topk >= long_min_endpoint_topk
            and endpoint_max  >= long_min_endpoint_max
            and global_sim is not None
            and global_sim    >= long_min_global
            and vote_ratio    >= long_min_vote_ratio
            and gallery_topk is not None
            and gallery_topk  >= long_min_gallery_topk
            and fused         >= long_fused_threshold
        )

    if short_gap_pass:
        rule = "short_gap"
    elif medium_gap_pass:
        rule = "medium_gap"
    elif long_gap_pass:
        rule = "long_gap"
    else:
        rule = None

    return {
        "accepted":        rule is not None,
        "accepted_rule":   rule,
        "reject_reason":   None if rule else "threshold",
        "short_gap_pass":  short_gap_pass,
        "medium_gap_pass": medium_gap_pass,
        "long_gap_pass":   long_gap_pass,
        "scale_ok":        True,
        "has_endpoint":    True,
        "vote_ratio":      float(vote_ratio),
    }


# ============================================================
# V4.2 gallery utilities (unchanged from V4.1)
# ============================================================

def gallery_entry(tracklet: dict) -> dict:
    return {
        "track_id":   int(tracklet["track_id"]),
        "start_sec":  float(tracklet["start_sec"]),
        "end_sec":    float(tracklet["end_sec"]),
        "end_frame":  int(tracklet["end_frame"]),
        "end_vectors": (
            None if tracklet["end_vectors"] is None
            else tracklet["end_vectors"].copy()
        ),
        "end_bbox":        list(tracklet["end_bbox"]),
        "global_embedding": (
            None if tracklet["global_embedding"] is None
            else tracklet["global_embedding"].copy()
        ),
    }


def trim_gallery(gallery: list[dict], gallery_size: int) -> list[dict]:
    gallery = sorted(gallery, key=lambda x: (x["end_sec"], x["track_id"]))
    if gallery_size <= 0:
        return gallery
    return gallery[-gallery_size:]


# ============================================================
# V4.2 stitching  ← drift cap + 새 decide 함수 사용
# ============================================================

def stitch_tracklets(
    tracklets: dict[int, dict],
    max_gap_sec: float,
    gallery_size: int,
    pair_top_k: int,
    gallery_global_top_k: int,
    gallery_vote_min_sim: float,
    identity_drift_cap: float,       # ← V4.2: drift 방지
    # ── Short gap ──────────────────────────
    short_gap_sec: float,
    short_min_endpoint_max: float,
    short_min_endpoint_topk: float,
    short_min_position: float,
    short_min_global: float,
    # ── Medium gap ─────────────────────────
    medium_gap_sec: float,
    medium_min_endpoint_topk: float,
    medium_min_endpoint_max: float,
    medium_min_global: float,
    medium_min_vote_ratio: float,
    medium_fused_threshold: float,
    # ── Long gap ───────────────────────────
    long_min_endpoint_topk: float,
    long_min_endpoint_max: float,
    long_min_global: float,
    long_min_vote_ratio: float,
    long_min_gallery_topk: float,
    long_fused_threshold: float,
    # ── Common ─────────────────────────────
    max_scale_ratio: float,
    calibration_only: bool,
    endpoint_weight: float,
    global_weight: float,
    time_weight: float,
    position_weight: float,
    scale_weight: float,
):
    usable = [
        t for t in tracklets.values()
        if (
            t["global_embedding"] is not None
            and t["start_vectors"] is not None
            and t["end_vectors"]   is not None
        )
    ]
    usable.sort(key=lambda t: (t["start_sec"], t["track_id"]))

    identities   = []
    assignment   = {}
    candidate_log = []
    next_person_id = 1

    for tracklet in usable:
        best = None

        for ident in identities:
            identity_gap = tracklet["start_sec"] - ident["last_end_sec"]
            if identity_gap < 0 or identity_gap > max_gap_sec:
                continue

            for predecessor in ident["gallery"]:
                gap_sec = tracklet["start_sec"] - predecessor["end_sec"]
                if gap_sec < 0 or gap_sec > max_gap_sec:
                    continue

                score = make_candidate_score(
                    identity=ident,
                    predecessor=predecessor,
                    tracklet=tracklet,
                    gap_sec=gap_sec,
                    max_gap_sec=max_gap_sec,
                    pair_top_k=pair_top_k,
                    gallery_global_top_k=gallery_global_top_k,
                    gallery_vote_min_sim=gallery_vote_min_sim,
                    endpoint_weight=endpoint_weight,
                    global_weight=global_weight,
                    time_weight=time_weight,
                    position_weight=position_weight,
                    scale_weight=scale_weight,
                )

                decision = decide_candidate_v42(
                    score,
                    gap_sec,
                    max_gap_sec=max_gap_sec,
                    max_scale_ratio=max_scale_ratio,
                    short_gap_sec=short_gap_sec,
                    short_min_endpoint_max=short_min_endpoint_max,
                    short_min_endpoint_topk=short_min_endpoint_topk,
                    short_min_position=short_min_position,
                    short_min_global=short_min_global,
                    medium_gap_sec=medium_gap_sec,
                    medium_min_endpoint_topk=medium_min_endpoint_topk,
                    medium_min_endpoint_max=medium_min_endpoint_max,
                    medium_min_global=medium_min_global,
                    medium_min_vote_ratio=medium_min_vote_ratio,
                    medium_fused_threshold=medium_fused_threshold,
                    long_min_endpoint_topk=long_min_endpoint_topk,
                    long_min_endpoint_max=long_min_endpoint_max,
                    long_min_global=long_min_global,
                    long_min_vote_ratio=long_min_vote_ratio,
                    long_min_gallery_topk=long_min_gallery_topk,
                    long_fused_threshold=long_fused_threshold,
                )

                would_accept = bool(decision["accepted"])

                if calibration_only:
                    decision = dict(decision)
                    decision["accepted"]      = False
                    decision["accepted_rule"] = None

                # ── candidate log ──────────────────────────────
                log_row = {
                    "track_id":                      tracklet["track_id"],
                    "candidate_person_id":            ident["person_id"],
                    "candidate_predecessor_track_id": predecessor["track_id"],
                    "candidate_identity_last_track_id": ident["last_track_id"],
                    "gap_sec":                        round(float(gap_sec), 4),

                    # endpoint
                    "pair_count":          score["pair_count"],
                    "endpoint_max":        _r(score["endpoint_max"]),
                    "endpoint_mean":       _r(score["endpoint_mean"]),
                    "endpoint_topk_mean":  _r(score["endpoint_topk_mean"]),
                    "endpoint_top_values": score["endpoint_top_values"],

                    # global
                    "predecessor_global_similarity": _r(score["predecessor_global_similarity"]),
                    "identity_global_similarity":    _r(score["identity_global_similarity"]),
                    "global_similarity":             _r(score["global_similarity"]),
                    "global_similarity_source":      score["global_similarity_source"],

                    # gallery
                    "gallery_global_count":    score["gallery_global_count"],
                    "gallery_global_max":      _r(score["gallery_global_max"]),
                    "gallery_global_topk_mean": _r(score["gallery_global_topk_mean"]),
                    "gallery_global_values":   score["gallery_global_values"],
                    "gallery_vote_ratio":      round(float(score["gallery_vote_ratio"]), 4),
                    "gallery_n_agree":         score["gallery_n_agree"],
                    "gallery_vote_min_sim":    score["gallery_vote_min_sim"],

                    # geometry
                    "time_score":             round(float(score["time_score"]), 6),
                    "position_score":         round(float(score["position_score"]), 6),
                    "center_distance_ratio":  _r(score["center_distance_ratio"]),
                    "bbox_scale_score":       round(float(score["bbox_scale_score"]), 6),
                    "bbox_scale_ratio":       _r(score["bbox_scale_ratio"]),
                    "fused_score":            round(float(score["fused_score"]), 6),

                    # decision
                    "short_gap_pass":   decision["short_gap_pass"],
                    "medium_gap_pass":  decision["medium_gap_pass"],
                    "long_gap_pass":    decision["long_gap_pass"],
                    "scale_ok":         decision["scale_ok"],
                    "has_endpoint":     decision["has_endpoint"],
                    "reject_reason":    decision.get("reject_reason"),
                    "accepted_rule":    decision["accepted_rule"],
                    "accepted":         decision["accepted"],
                    "would_accept_with_current_thresholds": would_accept,
                    "calibration_only": bool(calibration_only),
                }
                candidate_log.append(log_row)

                if not decision["accepted"]:
                    continue

                selection_key = (float(score["fused_score"]), -float(gap_sec))
                if best is None or selection_key > best["selection_key"]:
                    best = {
                        "identity":    ident,
                        "predecessor": predecessor,
                        "score":       score,
                        "decision":    decision,
                        "gap_sec":     gap_sec,
                        "selection_key": selection_key,
                    }

        # ── New identity ────────────────────────────────────
        if best is None:
            ident = {
                "person_id":    next_person_id,
                "class_name":   tracklet["class_name"],
                "track_ids":    [tracklet["track_id"]],
                "global_embedding": tracklet["global_embedding"].copy(),
                "global_weight":    float(max(1, tracklet["observations"])),
                "gallery":     [gallery_entry(tracklet)],
                "last_track_id":   tracklet["track_id"],
                "start_sec":       tracklet["start_sec"],
                "last_end_sec":    tracklet["end_sec"],
                "start_frame":     tracklet["start_frame"],
                "last_end_frame":  tracklet["end_frame"],
            }
            identities.append(ident)
            assignment[tracklet["track_id"]] = _empty_assignment(next_person_id)
            next_person_id += 1
            continue

        # ── Merge into existing identity ────────────────────
        ident      = best["identity"]
        predecessor = best["predecessor"]
        score       = best["score"]
        decision    = best["decision"]

        old_weight = float(ident["global_weight"])
        new_weight = float(max(1, tracklet["observations"]))

        # V4.2: drift cap — 신규 track이 identity의 20% 이상 당길 수 없음
        new_weight_capped = min(new_weight, old_weight * identity_drift_cap)

        ident["global_embedding"] = normalize(
            ident["global_embedding"] * old_weight
            + tracklet["global_embedding"] * new_weight_capped
        )
        ident["global_weight"] = old_weight + new_weight_capped

        ident["track_ids"].append(tracklet["track_id"])
        ident["gallery"].append(gallery_entry(tracklet))
        ident["gallery"] = trim_gallery(ident["gallery"], gallery_size)

        ident["last_track_id"]  = tracklet["track_id"]
        ident["last_end_sec"]   = max(ident["last_end_sec"],  tracklet["end_sec"])
        ident["last_end_frame"] = max(ident["last_end_frame"], tracklet["end_frame"])

        assignment[tracklet["track_id"]] = {
            "person_id":                   ident["person_id"],
            "stitched_from_track_id":      predecessor["track_id"],
            "stitch_rule":                 decision["accepted_rule"],
            "stitch_endpoint_topk_mean":   _r(score["endpoint_topk_mean"]),
            "stitch_endpoint_max":         _r(score["endpoint_max"]),
            "stitch_global_similarity":    _r(score["global_similarity"]),
            "stitch_gallery_global_topk_mean": _r(score["gallery_global_topk_mean"]),
            "stitch_gallery_vote_ratio":   round(float(score["gallery_vote_ratio"]), 4),
            "stitch_fused_score":          round(float(score["fused_score"]), 6),
            "stitch_gap_sec":              round(float(best["gap_sec"]), 4),
            "stitch_position_score":       round(float(score["position_score"]), 6),
            "stitch_bbox_scale_ratio":     _r(score["bbox_scale_ratio"]),
        }

    # embedding 없는 tracklet은 독립 person 유지
    for track_id in sorted(tracklets):
        if track_id in assignment:
            continue
        t = tracklets[track_id]
        assignment[track_id] = _empty_assignment(next_person_id)
        identities.append({
            "person_id":    next_person_id,
            "class_name":   t["class_name"],
            "track_ids":    [track_id],
            "global_embedding": t["global_embedding"],
            "global_weight":    0.0,
            "gallery": [gallery_entry(t)] if t["end_vectors"] is not None else [],
            "last_track_id":   track_id,
            "start_sec":       t["start_sec"],
            "last_end_sec":    t["end_sec"],
            "start_frame":     t["start_frame"],
            "last_end_frame":  t["end_frame"],
        })
        next_person_id += 1

    identities.sort(key=lambda x: x["person_id"])
    return identities, assignment, candidate_log


def _r(v, digits: int = 6):
    """None-safe round."""
    return None if v is None else round(float(v), digits)


def _empty_assignment(person_id: int) -> dict:
    return {
        "person_id":               person_id,
        "stitched_from_track_id":  None,
        "stitch_rule":             None,
        "stitch_endpoint_topk_mean": None,
        "stitch_endpoint_max":     None,
        "stitch_global_similarity": None,
        "stitch_gallery_global_topk_mean": None,
        "stitch_gallery_vote_ratio": None,
        "stitch_fused_score":      None,
        "stitch_gap_sec":          None,
        "stitch_position_score":   None,
        "stitch_bbox_scale_ratio": None,
    }


# ============================================================
# Output V4.2  ← 파일명 / stitch_model 업데이트
# ============================================================

def save_outputs(
    video_dir: Path,
    rows: list[dict],
    tracklets: dict[int, dict],
    identities: list[dict],
    assignment: dict[int, dict],
    candidate_log: list[dict],
    crop_errors: list[dict],
    args,
):
    stitched_rows = []
    skipped_output_rows = 0

    for row in rows:
        if row.get("track_id") is None:
            skipped_output_rows += 1
            continue
        track_id = int(row["track_id"])
        if track_id not in assignment:
            skipped_output_rows += 1
            continue

        info = assignment[track_id]
        out  = dict(row)
        out["original_track_id"]    = track_id
        out["person_id"]            = int(info["person_id"])
        out["stitched_id"]          = f"person_{int(info['person_id']):04d}"
        out["class_name"]           = tracklets[track_id]["class_name"]
        out["final_class_name"]     = tracklets[track_id]["class_name"]
        out["stitch_model"]         = "solider_gallery_v4_2"
        out["stitched_from_track_id"]       = info["stitched_from_track_id"]
        out["stitch_rule"]                  = info["stitch_rule"]
        out["stitch_endpoint_topk_mean"]    = info["stitch_endpoint_topk_mean"]
        out["stitch_endpoint_max"]          = info["stitch_endpoint_max"]
        out["stitch_global_similarity"]     = info["stitch_global_similarity"]
        out["stitch_gallery_global_topk_mean"] = info["stitch_gallery_global_topk_mean"]
        out["stitch_gallery_vote_ratio"]    = info["stitch_gallery_vote_ratio"]
        out["stitch_fused_score"]           = info["stitch_fused_score"]
        out["stitch_gap_sec"]               = info["stitch_gap_sec"]
        out["stitch_position_score"]        = info["stitch_position_score"]
        out["stitch_bbox_scale_ratio"]      = info["stitch_bbox_scale_ratio"]
        stitched_rows.append(out)

    stitched_path   = video_dir / "stitched_tracks_v4_2.jsonl"
    candidates_path = video_dir / "stitch_candidates_v4_2.jsonl"
    crop_error_path = video_dir / "stitch_crop_errors_v4_2.jsonl"
    write_jsonl(stitched_path,   stitched_rows)
    write_jsonl(candidates_path, candidate_log)
    write_jsonl(crop_error_path, crop_errors)

    rule_counts = Counter(
        info["stitch_rule"]
        for info in assignment.values()
        if info["stitch_rule"] is not None
    )

    summary = {
        "scope":           "person",
        "stitch_model":    "solider_gallery_v4_2",
        "calibration_only": bool(args.calibration_only),
        "global_top_k":    args.global_top_k,
        "endpoint_k":      args.endpoint_k,
        "pair_top_k":      args.pair_top_k,
        "gallery_size":    args.gallery_size,
        "gallery_global_top_k":   args.gallery_global_top_k,
        "gallery_vote_min_sim":   args.gallery_vote_min_sim,
        "identity_drift_cap":     args.identity_drift_cap,
        "max_gap_sec":     args.max_gap_sec,
        "rules": {
            "short_gap": {
                "max_gap_sec":       args.short_gap_sec,
                "min_endpoint_max":  args.short_min_endpoint_max,
                "min_endpoint_topk": args.short_min_endpoint_topk,
                "min_position":      args.short_min_position,
                "min_global":        args.short_min_global,
            },
            "medium_gap": {
                "max_gap_sec":       args.medium_gap_sec,
                "min_endpoint_topk": args.medium_min_endpoint_topk,
                "min_endpoint_max":  args.medium_min_endpoint_max,
                "min_global":        args.medium_min_global,
                "min_vote_ratio":    args.medium_min_vote_ratio,
                "fused_threshold":   args.medium_fused_threshold,
            },
            "long_gap": {
                "min_endpoint_topk": args.long_min_endpoint_topk,
                "min_endpoint_max":  args.long_min_endpoint_max,
                "min_global":        args.long_min_global,
                "min_vote_ratio":    args.long_min_vote_ratio,
                "min_gallery_topk":  args.long_min_gallery_topk,
                "fused_threshold":   args.long_fused_threshold,
            },
            "max_scale_ratio": args.max_scale_ratio,
        },
        "weights": {
            "endpoint": args.endpoint_weight,
            "global":   args.global_weight,
            "time":     args.time_weight,
            "position": args.position_weight,
            "scale":    args.scale_weight,
        },
        "input_rows":       len(rows),
        "input_tracklets":  len(tracklets),
        "output_person_ids": len(identities),
        "output_rows":      len(stitched_rows),
        "skipped_output_rows": skipped_output_rows,
        "crop_errors":      len(crop_errors),
        "merge_rule_counts": dict(rule_counts),
        "persons": [],
    }

    npz_data = {}
    for track_id, t in sorted(tracklets.items()):
        if t["global_embedding"] is not None:
            npz_data[f"track_{track_id}_global"] = np.asarray(t["global_embedding"], dtype=np.float32)
        if t["start_vectors"] is not None:
            npz_data[f"track_{track_id}_start"]  = np.asarray(t["start_vectors"], dtype=np.float32)
        if t["end_vectors"] is not None:
            npz_data[f"track_{track_id}_end"]    = np.asarray(t["end_vectors"], dtype=np.float32)

    for ident in identities:
        person_id = int(ident["person_id"])
        representative_crops = []
        for track_id in ident["track_ids"]:
            representative_crops.extend(tracklets[track_id]["global_crops"])
        summary["persons"].append({
            "person_id":          person_id,
            "stitched_id":        f"person_{person_id:04d}",
            "final_class_name":   ident["class_name"],
            "track_ids":          ident["track_ids"],
            "start_sec":          round(float(ident["start_sec"]), 4),
            "end_sec":            round(float(ident["last_end_sec"]), 4),
            "gallery_track_ids":  [int(x["track_id"]) for x in ident.get("gallery", [])],
            "representative_crops": representative_crops,
        })
        if ident["global_embedding"] is not None:
            npz_data[f"person_{person_id}_global"] = np.asarray(
                ident["global_embedding"], dtype=np.float32
            )

    summary_path   = video_dir / "stitching_v4_2.json"
    embedding_path = video_dir / "stitch_embeddings_solider_v4_2.npz"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(embedding_path, **npz_data)

    return stitched_path, summary_path, candidates_path, crop_error_path, embedding_path


# ============================================================
# Helpers (unchanged)
# ============================================================

def find_video_dirs() -> list[Path]:
    if not TRACK_ROOT.exists():
        return []
    return sorted(
        p.parent
        for p in TRACK_ROOT.glob("*/tracks.jsonl")
        if p.is_file() and p.stat().st_size > 0
    )


def save_video_error(video_dir: Path, exc: Exception) -> Path:
    error_path = video_dir / "stitch_error_v4_2.json"
    error_path.write_text(
        json.dumps({
            "video":      video_dir.name,
            "error_type": type(exc).__name__,
            "error":      str(exc),
            "traceback":  traceback.format_exc(),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return error_path


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Person Tracklet Stitching V4.2: "
            "gap-tier (short/medium/long) + gallery vote ratio + drift cap"
        )
    )

    ap.add_argument("--max-videos",      type=int,   default=None)
    ap.add_argument("--video",           type=str,   default=None)
    ap.add_argument("--calibration-only", action="store_true")
    ap.add_argument("--global-top-k",    type=int,   default=5)
    ap.add_argument("--endpoint-k",      type=int,   default=3)
    ap.add_argument("--pair-top-k",      type=int,   default=3)
    ap.add_argument("--gallery-size",    type=int,   default=5)
    ap.add_argument("--gallery-global-top-k", type=int, default=3)
    ap.add_argument(
        "--gallery-vote-min-sim", type=float, default=0.70,
        help="gallery vote 기준 유사도 (이 값 이상 = 동의)"
    )
    ap.add_argument(
        "--identity-drift-cap", type=float, default=0.20,
        help="신규 tracklet이 identity weight의 몇 배까지 영향 가능 (0.20 = 20%%)"
    )
    ap.add_argument("--max-gap-sec",     type=float, default=20.0)
    ap.add_argument("--max-scale-ratio", type=float, default=2.5)

    # ── Short gap ────────────────────────────────────────────
    ap.add_argument("--short-gap-sec",          type=float, default=1.5)
    ap.add_argument("--short-min-endpoint-max", type=float, default=0.65)
    ap.add_argument("--short-min-endpoint-topk",type=float, default=0.60)
    ap.add_argument("--short-min-position",     type=float, default=0.90)
    ap.add_argument("--short-min-global",       type=float, default=0.55)

    # ── Medium gap ───────────────────────────────────────────
    ap.add_argument("--medium-gap-sec",           type=float, default=8.0)
    ap.add_argument("--medium-min-endpoint-topk", type=float, default=0.68)
    ap.add_argument("--medium-min-endpoint-max",  type=float, default=0.65)
    ap.add_argument("--medium-min-global",        type=float, default=0.75)
    ap.add_argument("--medium-min-vote-ratio",    type=float, default=0.60)
    ap.add_argument("--medium-fused-threshold",   type=float, default=0.75)

    # ── Long gap ─────────────────────────────────────────────
    ap.add_argument("--long-min-endpoint-topk",  type=float, default=0.72)
    ap.add_argument("--long-min-endpoint-max",   type=float, default=0.70)
    ap.add_argument("--long-min-global",         type=float, default=0.82)
    ap.add_argument("--long-min-vote-ratio",     type=float, default=0.75)
    ap.add_argument("--long-min-gallery-topk",   type=float, default=0.78)
    ap.add_argument("--long-fused-threshold",    type=float, default=0.80)

    # ── Fused weights ────────────────────────────────────────
    ap.add_argument("--endpoint-weight", type=float, default=0.65)
    ap.add_argument("--global-weight",   type=float, default=0.20)
    ap.add_argument("--time-weight",     type=float, default=0.05)
    ap.add_argument("--position-weight", type=float, default=0.05)
    ap.add_argument("--scale-weight",    type=float, default=0.05)

    args = ap.parse_args()

    # ── Validation ───────────────────────────────────────────
    for name, val in {
        "global_top_k": args.global_top_k,
        "endpoint_k":   args.endpoint_k,
        "pair_top_k":   args.pair_top_k,
        "gallery_size": args.gallery_size,
        "gallery_global_top_k": args.gallery_global_top_k,
    }.items():
        if val <= 0:
            raise ValueError(f"--{name.replace('_','-')} must be >= 1")

    for name, val in {
        "max_gap_sec":     args.max_gap_sec,
        "short_gap_sec":   args.short_gap_sec,
        "medium_gap_sec":  args.medium_gap_sec,
    }.items():
        if val < 0:
            raise ValueError(f"--{name.replace('_','-')} must be >= 0")

    if args.short_gap_sec >= args.medium_gap_sec:
        raise ValueError("--short-gap-sec must be < --medium-gap-sec")
    if args.medium_gap_sec >= args.max_gap_sec:
        raise ValueError("--medium-gap-sec must be < --max-gap-sec")

    if args.max_scale_ratio < 1.0:
        raise ValueError("--max-scale-ratio must be >= 1.0")
    if not (0.0 < args.identity_drift_cap <= 1.0):
        raise ValueError("--identity-drift-cap must be in (0, 1]")

    weights = [args.endpoint_weight, args.global_weight, args.time_weight,
               args.position_weight, args.scale_weight]
    if any(w < 0 for w in weights):
        raise ValueError("stitch weights must be >= 0")
    if sum(weights) <= 0:
        raise ValueError("sum of stitch weights must be > 0")

    video_dirs = find_video_dirs()
    if args.video:
        wanted = str(args.video).strip().lower()
        video_dirs = [
            p for p in video_dirs
            if p.name.lower() == wanted
            or any((p.name + ext).lower() == wanted for ext in (".mp4", ".avi", ".mov", ".mkv"))
        ]
    if args.max_videos is not None:
        if args.max_videos <= 0:
            raise ValueError("--max-videos must be >= 1")
        video_dirs = video_dirs[:args.max_videos]

    print("=" * 96)
    print("PERSON TRACKLET STITCHING V4.2 — gap-tier + gallery vote ratio + drift cap")
    print("=" * 96)
    print(f"track root         : {TRACK_ROOT}")
    print(f"videos             : {len(video_dirs)}")
    print(f"calibration only   : {args.calibration_only}")
    print(f"gallery vote min   : {args.gallery_vote_min_sim}")
    print(f"identity drift cap : {args.identity_drift_cap}")
    print(f"max gap sec        : {args.max_gap_sec}")
    print(f"short gap tier     : gap ≤ {args.short_gap_sec}s  ep_max≥{args.short_min_endpoint_max}  ep_topk≥{args.short_min_endpoint_topk}  pos≥{args.short_min_position}  global≥{args.short_min_global}")
    print(f"medium gap tier    : gap ≤ {args.medium_gap_sec}s  ep_topk≥{args.medium_min_endpoint_topk}  ep_max≥{args.medium_min_endpoint_max}  global≥{args.medium_min_global}  vote≥{args.medium_min_vote_ratio}  fused≥{args.medium_fused_threshold}")
    print(f"long gap tier      : gap ≤ {args.max_gap_sec}s  ep_topk≥{args.long_min_endpoint_topk}  ep_max≥{args.long_min_endpoint_max}  global≥{args.long_min_global}  vote≥{args.long_min_vote_ratio}  gallery≥{args.long_min_gallery_topk}  fused≥{args.long_fused_threshold}")
    print(f"max scale ratio    : {args.max_scale_ratio}")

    if not video_dirs:
        print("No non-empty person tracks.jsonl found.")
        return

    cfg = PipelineConfig.load(CONFIG_PATH)
    embedder, spec = load_embedder(cfg, "solider")
    print(f"model              : {spec.name}  ({spec.class_name}  dim={spec.dim})")

    passed = failed = 0

    for idx, video_dir in enumerate(video_dirs, start=1):
        print(f"\n[{idx}/{len(video_dirs)}] {video_dir.name}")
        try:
            rows = load_jsonl(video_dir / "tracks.jsonl")
            if not rows:
                print("  input rows      : 0 — skip")
                passed += 1
                continue

            tracklets  = build_tracklets(rows)
            crop_errors = embed_tracklets(
                embedder=embedder,
                tracklets=tracklets,
                global_top_k=args.global_top_k,
                endpoint_k=args.endpoint_k,
            )

            identities, assignment, candidate_log = stitch_tracklets(
                tracklets=tracklets,
                max_gap_sec=args.max_gap_sec,
                gallery_size=args.gallery_size,
                pair_top_k=args.pair_top_k,
                gallery_global_top_k=args.gallery_global_top_k,
                gallery_vote_min_sim=args.gallery_vote_min_sim,
                identity_drift_cap=args.identity_drift_cap,
                short_gap_sec=args.short_gap_sec,
                short_min_endpoint_max=args.short_min_endpoint_max,
                short_min_endpoint_topk=args.short_min_endpoint_topk,
                short_min_position=args.short_min_position,
                short_min_global=args.short_min_global,
                medium_gap_sec=args.medium_gap_sec,
                medium_min_endpoint_topk=args.medium_min_endpoint_topk,
                medium_min_endpoint_max=args.medium_min_endpoint_max,
                medium_min_global=args.medium_min_global,
                medium_min_vote_ratio=args.medium_min_vote_ratio,
                medium_fused_threshold=args.medium_fused_threshold,
                long_min_endpoint_topk=args.long_min_endpoint_topk,
                long_min_endpoint_max=args.long_min_endpoint_max,
                long_min_global=args.long_min_global,
                long_min_vote_ratio=args.long_min_vote_ratio,
                long_min_gallery_topk=args.long_min_gallery_topk,
                long_fused_threshold=args.long_fused_threshold,
                max_scale_ratio=args.max_scale_ratio,
                calibration_only=args.calibration_only,
                endpoint_weight=args.endpoint_weight,
                global_weight=args.global_weight,
                time_weight=args.time_weight,
                position_weight=args.position_weight,
                scale_weight=args.scale_weight,
            )

            (stitched_path, summary_path, candidates_path,
             crop_error_path, embedding_path) = save_outputs(
                video_dir=video_dir,
                rows=rows,
                tracklets=tracklets,
                identities=identities,
                assignment=assignment,
                candidate_log=candidate_log,
                crop_errors=crop_errors,
                args=args,
            )

            print(f"  input rows      : {len(rows)}")
            print(f"  input tracklets : {len(tracklets)}")
            print(f"  output persons  : {len(identities)}")
            print(f"  merge rules     : {dict(Counter(x['stitch_rule'] for x in assignment.values() if x['stitch_rule']))}")
            print(f"  crop errors     : {len(crop_errors)}")
            print(f"  stitched rows   : {stitched_path}")
            print(f"  summary         : {summary_path}")
            print(f"  candidates      : {candidates_path}")
            passed += 1

        except Exception as exc:
            failed += 1
            error_path = save_video_error(video_dir, exc)
            print(f"  [FAIL] {type(exc).__name__}: {exc}")
            print(f"  error log       : {error_path}")

    print("\n" + "=" * 96)
    print(f"STITCH V4.2 COMPLETE — passed: {passed}  failed: {failed}")
    print("=" * 96)


if __name__ == "__main__":
    main()
