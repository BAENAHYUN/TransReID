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
TRACK_ROOT = ROOT / "data" / "video_tracks" / "object"


# ============================================================
# Basic utilities
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
        raise KeyError(
            f"pipeline.yaml retrievers에 '{name}'이 없습니다."
        )

    spec = cfg.retrievers[name]
    module = importlib.import_module(spec.module)
    cls = getattr(module, spec.class_name)

    embedder = cls(**dict(spec.params))

    if not hasattr(embedder, "embed_crops"):
        raise TypeError(
            f"{spec.class_name} does not provide embed_crops()"
        )

    return embedder, spec


# ============================================================
# Tracklet construction
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
    """
    V3:
    confidence top-k만 뽑으면 한 시간대에 몰릴 수 있으므로,
    track 전체 시간축을 최대 top_k 구간으로 나누고
    각 구간에서 confidence가 가장 높은 crop 1장을 고른다.

    동률이면 더 최신 frame을 우선한다.
    """
    rows = valid_crop_rows(records)
    rows.sort(key=lambda r: int(r.get("frame_idx", 0)))

    if len(rows) <= top_k:
        return rows

    bins = np.array_split(
        np.arange(len(rows)),
        top_k,
    )

    selected = []

    for bin_indices in bins:
        if len(bin_indices) == 0:
            continue

        candidates = [
            rows[int(i)]
            for i in bin_indices
        ]

        best = max(
            candidates,
            key=lambda r: (
                float(r.get("confidence", 0.0)),
                int(r.get("frame_idx", 0)),
            ),
        )

        selected.append(best)

    selected.sort(
        key=lambda r: int(r.get("frame_idx", 0))
    )

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
        # BUG FIX #1:
        # track_id=None row는 stitching 대상이 아님.
        if row.get("track_id") is None:
            continue

        grouped[int(row["track_id"])].append(row)

    tracklets = {}

    for track_id, items in grouped.items():
        items.sort(key=lambda r: int(r.get("frame_idx", 0)))

        start = items[0]
        end = items[-1]

        start_sec = float(start.get("timestamp_sec", 0.0))
        end_sec = float(end.get("timestamp_sec", 0.0))

        # 데이터 이상 감지:
        # frame_idx 정렬 이후에도 start timestamp가 end보다 크면
        # 원본 metadata timestamp가 비정상적인 상태다.
        # stitching을 강제로 중단하지는 않고 경고를 남긴다.
        if start_sec > end_sec:
            warnings.warn(
                (
                    f"track_id={track_id}: "
                    f"start_sec({start_sec}) > end_sec({end_sec})"
                ),
                RuntimeWarning,
            )

        tracklets[track_id] = {
            "track_id": track_id,
            "records": items,
            "class_name": final_class(items),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "start_frame": int(start.get("frame_idx", 0)),
            "end_frame": int(end.get("frame_idx", 0)),
            "start_bbox": list(start.get("bbox") or []),
            "end_bbox": list(end.get("bbox") or []),
            "observations": len(items),

            # V3 embedding fields
            "global_embedding": None,
            "start_vectors": None,
            "end_vectors": None,

            "global_crops": [],
            "start_crops": [],
            "end_crops": [],
        }

    return tracklets


# ============================================================
# DINOv2 embedding
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

    # embed_tracklets()에서는 vecs를 이미 normalize_rows()한 뒤 bucket에 넣으므로
    # 중복 정규화를 피할 수 있다. 범용 호출 시에는 기본값 False로 안전하게 유지.
    if not already_normalized:
        arr = normalize_rows(arr)

    return normalize(np.mean(arr, axis=0))


def embed_tracklets(
    embedder,
    tracklets: dict[int, dict],
    global_top_k: int,
    endpoint_k: int,
) -> list[dict]:
    """
    모든 필요한 crop을 한 번만 DINOv2에 넣고,
    동일 embedding을 global/start/end에서 재사용한다.

    손상 이미지:
      - 해당 crop만 skip
      - errors에 기록
      - 영상 전체는 계속 진행

    embed_crops 자체 실패(OOM 등):
      - 예외를 올려 main의 video-level error log로 기록
    """
    path_to_owners: dict[str, list[tuple[int, str]]] = defaultdict(list)
    ordered_paths: list[str] = []
    seen_paths: set[str] = set()
    crop_errors: list[dict] = []

    for track_id in sorted(tracklets):
        t = tracklets[track_id]

        selections = {
            "global": select_global_rows(
                t["records"],
                top_k=global_top_k,
            ),
            "start": select_start_rows(
                t["records"],
                k=endpoint_k,
            ),
            "end": select_end_rows(
                t["records"],
                k=endpoint_k,
            ),
        }

        for kind, selected in selections.items():
            crop_paths = [
                str(Path(str(row["crop_path"])).resolve())
                for row in selected
            ]

            t[f"{kind}_crops"] = crop_paths

            for crop_path in crop_paths:
                path_to_owners[crop_path].append(
                    (track_id, kind)
                )

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
                "crop_path": crop_path,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    if not valid_paths:
        return crop_errors

    try:
        vecs = np.asarray(
            embedder.embed_crops(
                images,
                input_format="rgb",
            ),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError(
            f"DINOv2 embed_crops failed for "
            f"{len(valid_paths)} crops"
        ) from exc

    if vecs.ndim != 2 or vecs.shape[0] != len(valid_paths):
        raise RuntimeError(
            f"DINOv2 embedding shape mismatch: "
            f"{vecs.shape}, expected rows={len(valid_paths)}"
        )

    vecs = normalize_rows(vecs)

    bucket: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)

    for crop_path, vec in zip(valid_paths, vecs):
        for owner in path_to_owners[crop_path]:
            bucket[owner].append(vec)

    for track_id in sorted(tracklets):
        t = tracklets[track_id]

        global_vecs = np.asarray(
            bucket.get((track_id, "global"), []),
            dtype=np.float32,
        )
        start_vecs = np.asarray(
            bucket.get((track_id, "start"), []),
            dtype=np.float32,
        )
        end_vecs = np.asarray(
            bucket.get((track_id, "end"), []),
            dtype=np.float32,
        )

        t["global_embedding"] = mean_embedding(
            global_vecs,
            already_normalized=True,
        )

        if start_vecs.size:
            t["start_vectors"] = normalize_rows(
                start_vecs.reshape(
                    len(start_vecs),
                    -1,
                )
            )

        if end_vecs.size:
            t["end_vectors"] = normalize_rows(
                end_vecs.reshape(
                    len(end_vecs),
                    -1,
                )
            )

    return crop_errors


# ============================================================
# Pairwise endpoint appearance
# ============================================================

def endpoint_pairwise_stats(
    prev_end_vectors: np.ndarray | None,
    next_start_vectors: np.ndarray | None,
    pair_top_k: int,
) -> dict:
    """
    이전 track END crop N개 × 다음 track START crop M개
    모든 cosine similarity를 계산한다.

    예: 3 x 3 -> 9개 similarity

    stitch 핵심 점수:
      endpoint_topk_mean
        = pair similarity 상위 K개의 평균
    """
    if (
        prev_end_vectors is None
        or next_start_vectors is None
    ):
        return {
            "pair_count": 0,
            "endpoint_max": None,
            "endpoint_mean": None,
            "endpoint_topk_mean": None,
            "endpoint_top_values": [],
        }

    a = normalize_rows(
        np.asarray(prev_end_vectors, dtype=np.float32)
    )
    b = normalize_rows(
        np.asarray(next_start_vectors, dtype=np.float32)
    )

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

    k = min(
        max(1, int(pair_top_k)),
        int(flat.size),
    )

    top_values = np.sort(flat)[-k:][::-1]

    return {
        "pair_count": int(flat.size),
        "endpoint_max": float(np.max(flat)),
        "endpoint_mean": float(np.mean(flat)),
        "endpoint_topk_mean": float(np.mean(top_values)),
        "endpoint_top_values": [
            round(float(x), 6)
            for x in top_values.tolist()
        ],
    }


# ============================================================
# Time / position score
# ============================================================

def bbox_center_and_diag(bbox):
    if not bbox or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = map(float, bbox)

    if x2 <= x1 or y2 <= y1:
        return None

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    w = x2 - x1
    h = y2 - y1

    diag = math.hypot(w, h)

    return cx, cy, diag


def position_score(prev_bbox, next_bbox) -> tuple[float, float | None]:
    """
    bbox center 이동거리를 bbox 크기로 정규화.
    ratio가 작을수록 위치 연속성이 높다.
    """
    a = bbox_center_and_diag(prev_bbox)
    b = bbox_center_and_diag(next_bbox)

    if a is None or b is None:
        return 0.0, None

    ax, ay, ad = a
    bx, by, bd = b

    dist = math.hypot(ax - bx, ay - by)

    scale = max(
        1.0,
        0.5 * (ad + bd),
    )

    ratio = dist / scale
    score = math.exp(-ratio / 3.0)

    return float(score), float(ratio)


def temporal_score(gap_sec: float, max_gap_sec: float) -> float:
    if max_gap_sec <= 0:
        return 1.0 if gap_sec <= 0 else 0.0

    return max(
        0.0,
        1.0 - (gap_sec / max_gap_sec),
    )


# ============================================================
# V4 bbox scale consistency
# ============================================================

def bbox_scale_score(prev_bbox, next_bbox) -> tuple[float, float | None]:
    """
    bbox 대각선 크기 변화 비율을 사용한다.

    ratio = max(diag_prev, diag_next) / min(...)
    - 1.0에 가까울수록 크기 변화가 작다.
    - score = 1 / ratio
    """
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
# V4.1 gallery-level global appearance
# ============================================================

def gallery_global_stats(
    identity: dict,
    tracklet: dict,
    top_k: int,
) -> dict:
    """
    identity gallery의 각 tracklet global embedding과
    새 tracklet global embedding을 모두 비교한다.

    strong-global fallback에서는 특정 predecessor 하나의
    global similarity가 아니라 gallery 상위 K개 평균을
    identity-level appearance evidence로 사용한다.
    """
    track_global = tracklet.get("global_embedding")

    if track_global is None:
        return {
            "gallery_global_count": 0,
            "gallery_global_max": None,
            "gallery_global_topk_mean": None,
            "gallery_global_values": [],
            "gallery_global_top_track_ids": [],
        }

    scored = []

    for item in identity.get("gallery", []):
        gallery_global = item.get("global_embedding")

        if gallery_global is None:
            continue

        sim = cosine(
            gallery_global,
            track_global,
        )

        scored.append(
            (
                int(item["track_id"]),
                float(sim),
            )
        )

    if not scored:
        return {
            "gallery_global_count": 0,
            "gallery_global_max": None,
            "gallery_global_topk_mean": None,
            "gallery_global_values": [],
            "gallery_global_top_track_ids": [],
        }

    scored.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    k = min(
        max(1, int(top_k)),
        len(scored),
    )

    top = scored[:k]

    return {
        "gallery_global_count": len(scored),
        "gallery_global_max": float(scored[0][1]),
        "gallery_global_topk_mean": float(
            np.mean([sim for _, sim in top])
        ),
        "gallery_global_values": [
            {
                "track_id": int(track_id),
                "similarity": round(float(sim), 6),
            }
            for track_id, sim in scored
        ],
        "gallery_global_top_track_ids": [
            int(track_id)
            for track_id, _ in top
        ],
    }


# ============================================================
# V4 candidate score
# ============================================================

def make_candidate_score(
    identity: dict,
    predecessor: dict,
    tracklet: dict,
    gap_sec: float,
    max_gap_sec: float,
    pair_top_k: int,
    gallery_global_top_k: int,
    endpoint_weight: float,
    global_weight: float,
    time_weight: float,
    position_weight: float,
    scale_weight: float,
):
    """
    Identity gallery의 특정 predecessor tracklet과
    새 tracklet을 비교한다.

    appearance:
      - predecessor END x current START pairwise
      - predecessor global x current global
      - identity centroid global x current global

    geometry:
      - time gap
      - bbox center continuity
      - bbox scale consistency
    """
    pair_stats = endpoint_pairwise_stats(
        predecessor.get("end_vectors"),
        tracklet.get("start_vectors"),
        pair_top_k=pair_top_k,
    )

    gallery_stats = gallery_global_stats(
        identity,
        tracklet,
        top_k=gallery_global_top_k,
    )

    endpoint_score = pair_stats["endpoint_topk_mean"]

    predecessor_global = predecessor.get("global_embedding")
    identity_global = identity.get("global_embedding")
    track_global = tracklet.get("global_embedding")

    predecessor_global_sim = None
    identity_global_sim = None

    if predecessor_global is not None and track_global is not None:
        predecessor_global_sim = cosine(
            predecessor_global,
            track_global,
        )

    if identity_global is not None and track_global is not None:
        identity_global_sim = cosine(
            identity_global,
            track_global,
        )

    available_global = [
        x
        for x in (
            predecessor_global_sim,
            identity_global_sim,
        )
        if x is not None
    ]

    global_sim = (
        max(available_global)
        if available_global
        else None
    )

    if global_sim is None:
        global_source = None
    elif (
        predecessor_global_sim is not None
        and global_sim == predecessor_global_sim
    ):
        global_source = "predecessor"
    else:
        global_source = "identity"

    t_score = temporal_score(
        gap_sec,
        max_gap_sec,
    )

    p_score, center_distance_ratio = position_score(
        predecessor.get("end_bbox"),
        tracklet.get("start_bbox"),
    )

    scale_score, scale_ratio = bbox_scale_score(
        predecessor.get("end_bbox"),
        tracklet.get("start_bbox"),
    )

    # gap이 길수록 bbox 위치 연속성은 신뢰도가 낮으므로 감쇠.
    effective_position_score = p_score * t_score

    parts = []

    if endpoint_score is not None:
        parts.append(
            (endpoint_weight, endpoint_score)
        )

    if global_sim is not None:
        parts.append(
            (global_weight, global_sim)
        )

    parts.append(
        (time_weight, t_score)
    )
    parts.append(
        (position_weight, effective_position_score)
    )
    parts.append(
        (scale_weight, scale_score)
    )

    weight_sum = sum(w for w, _ in parts)

    fused = (
        -1.0
        if weight_sum <= 0
        else sum(
            w * score
            for w, score in parts
        ) / weight_sum
    )

    return {
        **pair_stats,
        **gallery_stats,
        "predecessor_global_similarity": predecessor_global_sim,
        "identity_global_similarity": identity_global_sim,
        "global_similarity": global_sim,
        "global_similarity_source": global_source,
        "time_score": float(t_score),
        "position_score": float(p_score),
        "effective_position_score": float(effective_position_score),
        "center_distance_ratio": center_distance_ratio,
        "bbox_scale_score": float(scale_score),
        "bbox_scale_ratio": scale_ratio,
        "fused_score": float(fused),
    }


def decide_candidate(
    score: dict,
    gap_sec: float,
    *,
    fused_threshold: float,
    strong_global_threshold: float,
    strong_global_max_gap_sec: float,
    strong_global_min_endpoint_max: float,
    strong_global_min_position: float,
    short_gap_sec: float,
    short_min_position: float,
    short_min_endpoint_max: float,
    short_min_global: float,
    long_gap_min_sec: float,
    long_gap_max_sec: float,
    long_min_endpoint_max: float,
    long_min_endpoint_topk: float,
    long_min_global: float,
    long_min_gallery_global: float,
    long_min_position: float,
    long_max_scale_ratio: float,
    long_min_fused: float,
    max_scale_ratio: float,
) -> dict:
    """
    V4.2는 V4.1 규칙에 보수적인 long-gap re-entry fallback을 추가한다.

    A. general
       fused >= fused_threshold

    B. strong_global fallback
       global이 매우 강하고, endpoint max/위치/시간도 보조하는 경우

    C. short_gap continuity fallback
       gap이 매우 짧고 위치/endpoint/bbox scale이 자연스러운 경우

    D. long_gap_reentry fallback
       장시간 가림/퇴장 후 재등장 후보를 보수적으로 복구.
       endpoint max/top-k, identity global, gallery global, 위치, scale, fused를
       모두 동시에 통과해야 한다. 현재 기본값은 Normal_Videos_003_x264의
       단일 검증 샘플에서 false candidate 없이 T5 -> T10만 통과하도록 잡은
       validation default이며 일반 데이터셋 calibration을 대체하지 않는다.
    """
    endpoint_max = score.get("endpoint_max")
    endpoint_topk = score.get("endpoint_topk_mean")
    global_sim = score.get("global_similarity")
    gallery_global = score.get("gallery_global_topk_mean")
    position = score.get("position_score", 0.0)
    scale_ratio = score.get("bbox_scale_ratio")

    scale_ok = (
        scale_ratio is not None
        and scale_ratio <= max_scale_ratio
    )

    general_pass = (
        score["fused_score"] >= fused_threshold
    )

    strong_global_pass = bool(
        gallery_global is not None
        and endpoint_max is not None
        and gallery_global >= strong_global_threshold
        and endpoint_max >= strong_global_min_endpoint_max
        and gap_sec <= strong_global_max_gap_sec
        and position >= strong_global_min_position
        and scale_ok
    )

    short_gap_pass = bool(
        global_sim is not None
        and endpoint_max is not None
        and gap_sec <= short_gap_sec
        and position >= short_min_position
        and endpoint_max >= short_min_endpoint_max
        and global_sim >= short_min_global
        and scale_ok
    )

    long_scale_ok = bool(
        scale_ratio is not None
        and scale_ratio <= long_max_scale_ratio
    )

    long_gap_reentry_pass = bool(
        endpoint_max is not None
        and endpoint_topk is not None
        and global_sim is not None
        and gallery_global is not None
        and gap_sec >= long_gap_min_sec
        and gap_sec <= long_gap_max_sec
        and endpoint_max >= long_min_endpoint_max
        and endpoint_topk >= long_min_endpoint_topk
        and global_sim >= long_min_global
        and gallery_global >= long_min_gallery_global
        and position >= long_min_position
        and long_scale_ok
        and score["fused_score"] >= long_min_fused
    )

    if general_pass:
        rule = "general"
    elif strong_global_pass:
        rule = "strong_global"
    elif short_gap_pass:
        rule = "short_gap"
    elif long_gap_reentry_pass:
        rule = "long_gap_reentry"
    else:
        rule = None

    return {
        "general_pass": bool(general_pass),
        "strong_global_pass": strong_global_pass,
        "short_gap_pass": short_gap_pass,
        "long_gap_reentry_pass": long_gap_reentry_pass,
        "scale_ok": bool(scale_ok),
        "long_scale_ok": long_scale_ok,
        "accepted": rule is not None,
        "accepted_rule": rule,
    }


# ============================================================
# Stitching V4 - identity gallery
# ============================================================

def gallery_entry(tracklet: dict) -> dict:
    return {
        "track_id": int(tracklet["track_id"]),
        "start_sec": float(tracklet["start_sec"]),
        "end_sec": float(tracklet["end_sec"]),
        "end_frame": int(tracklet["end_frame"]),
        "end_vectors": (
            None
            if tracklet["end_vectors"] is None
            else tracklet["end_vectors"].copy()
        ),
        "end_bbox": list(tracklet["end_bbox"]),
        "global_embedding": (
            None
            if tracklet["global_embedding"] is None
            else tracklet["global_embedding"].copy()
        ),
    }


def trim_gallery(gallery: list[dict], gallery_size: int) -> list[dict]:
    gallery = sorted(
        gallery,
        key=lambda x: (
            x["end_sec"],
            x["track_id"],
        ),
    )

    if gallery_size <= 0:
        return gallery

    return gallery[-gallery_size:]


def stitch_tracklets(
    tracklets: dict[int, dict],
    max_gap_sec: float,
    gallery_size: int,
    pair_top_k: int,
    gallery_global_top_k: int,
    fused_threshold: float,
    strong_global_threshold: float,
    strong_global_max_gap_sec: float,
    strong_global_min_endpoint_max: float,
    strong_global_min_position: float,
    short_gap_sec: float,
    short_min_position: float,
    short_min_endpoint_max: float,
    short_min_global: float,
    long_gap_min_sec: float,
    long_gap_max_sec: float,
    long_min_endpoint_max: float,
    long_min_endpoint_topk: float,
    long_min_global: float,
    long_min_gallery_global: float,
    long_min_position: float,
    long_max_scale_ratio: float,
    long_min_fused: float,
    max_scale_ratio: float,
    endpoint_weight: float,
    global_weight: float,
    time_weight: float,
    position_weight: float,
    scale_weight: float,
):
    """
    V4:
      - endpoint hard gate 제거
      - identity gallery 사용
      - general / strong-global / short-gap / long-gap-reentry 4개 merge rule
      - bbox scale consistency 추가

    중요:
      identity 전체의 max end_sec보다 새 tracklet start가 이르면
      해당 identity와 시간 중첩이므로 identity 전체를 skip한다.
    """
    usable = [
        t
        for t in tracklets.values()
        if (
            t["global_embedding"] is not None
            and t["start_vectors"] is not None
            and t["end_vectors"] is not None
        )
    ]

    usable.sort(
        key=lambda t: (
            t["start_sec"],
            t["track_id"],
        )
    )

    identities = []
    assignment = {}
    candidate_log = []
    next_object_id = 1

    for tracklet in usable:
        best = None

        for ident in identities:
            if ident["class_name"] != tracklet["class_name"]:
                continue

            # identity 안의 어느 track과도 시간 중첩되면 같은 물체로 합치지 않음.
            identity_gap = (
                tracklet["start_sec"]
                - ident["last_end_sec"]
            )

            if identity_gap < 0:
                continue

            if identity_gap > max_gap_sec:
                # gallery의 모든 항목은 last_end_sec보다 같거나 더 과거이므로
                # 가장 최근 end와도 max gap 초과면 볼 필요 없음.
                continue

            # 최근 gallery member 각각과 비교.
            for predecessor in ident["gallery"]:
                gap_sec = (
                    tracklet["start_sec"]
                    - predecessor["end_sec"]
                )

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
                    endpoint_weight=endpoint_weight,
                    global_weight=global_weight,
                    time_weight=time_weight,
                    position_weight=position_weight,
                    scale_weight=scale_weight,
                )

                decision = decide_candidate(
                    score,
                    gap_sec,
                    fused_threshold=fused_threshold,
                    strong_global_threshold=strong_global_threshold,
                    strong_global_max_gap_sec=strong_global_max_gap_sec,
                    strong_global_min_endpoint_max=strong_global_min_endpoint_max,
                    strong_global_min_position=strong_global_min_position,
                    short_gap_sec=short_gap_sec,
                    short_min_position=short_min_position,
                    short_min_endpoint_max=short_min_endpoint_max,
                    short_min_global=short_min_global,
                    long_gap_min_sec=long_gap_min_sec,
                    long_gap_max_sec=long_gap_max_sec,
                    long_min_endpoint_max=long_min_endpoint_max,
                    long_min_endpoint_topk=long_min_endpoint_topk,
                    long_min_global=long_min_global,
                    long_min_gallery_global=long_min_gallery_global,
                    long_min_position=long_min_position,
                    long_max_scale_ratio=long_max_scale_ratio,
                    long_min_fused=long_min_fused,
                    max_scale_ratio=max_scale_ratio,
                )

                log_row = {
                    "track_id": tracklet["track_id"],
                    "track_class": tracklet["class_name"],
                    "candidate_object_id": ident["object_id"],
                    "candidate_predecessor_track_id": predecessor["track_id"],
                    "candidate_identity_last_track_id": ident["last_track_id"],
                    "gap_sec": round(float(gap_sec), 4),

                    "pair_count": score["pair_count"],
                    "endpoint_max": (
                        None
                        if score["endpoint_max"] is None
                        else round(float(score["endpoint_max"]), 6)
                    ),
                    "endpoint_mean": (
                        None
                        if score["endpoint_mean"] is None
                        else round(float(score["endpoint_mean"]), 6)
                    ),
                    "endpoint_topk_mean": (
                        None
                        if score["endpoint_topk_mean"] is None
                        else round(float(score["endpoint_topk_mean"]), 6)
                    ),
                    "endpoint_top_values": score["endpoint_top_values"],

                    "predecessor_global_similarity": (
                        None
                        if score["predecessor_global_similarity"] is None
                        else round(
                            float(score["predecessor_global_similarity"]),
                            6,
                        )
                    ),
                    "identity_global_similarity": (
                        None
                        if score["identity_global_similarity"] is None
                        else round(
                            float(score["identity_global_similarity"]),
                            6,
                        )
                    ),
                    "global_similarity": (
                        None
                        if score["global_similarity"] is None
                        else round(float(score["global_similarity"]), 6)
                    ),
                    "global_similarity_source": score["global_similarity_source"],

                    "gallery_global_count": score["gallery_global_count"],
                    "gallery_global_max": (
                        None
                        if score["gallery_global_max"] is None
                        else round(float(score["gallery_global_max"]), 6)
                    ),
                    "gallery_global_topk_mean": (
                        None
                        if score["gallery_global_topk_mean"] is None
                        else round(
                            float(score["gallery_global_topk_mean"]),
                            6,
                        )
                    ),
                    "gallery_global_values": score["gallery_global_values"],
                    "gallery_global_top_track_ids": (
                        score["gallery_global_top_track_ids"]
                    ),

                    "time_score": round(float(score["time_score"]), 6),
                    "position_score": round(float(score["position_score"]), 6),
                    "effective_position_score": round(
                        float(score["effective_position_score"]),
                        6,
                    ),
                    "center_distance_ratio": (
                        None
                        if score["center_distance_ratio"] is None
                        else round(float(score["center_distance_ratio"]), 6)
                    ),
                    "bbox_scale_score": round(
                        float(score["bbox_scale_score"]),
                        6,
                    ),
                    "bbox_scale_ratio": (
                        None
                        if score["bbox_scale_ratio"] is None
                        else round(float(score["bbox_scale_ratio"]), 6)
                    ),

                    "fused_score": round(float(score["fused_score"]), 6),

                    "general_pass": decision["general_pass"],
                    "strong_global_pass": decision["strong_global_pass"],
                    "short_gap_pass": decision["short_gap_pass"],
                    "long_gap_reentry_pass": decision["long_gap_reentry_pass"],
                    "scale_ok": decision["scale_ok"],
                    "long_scale_ok": decision["long_scale_ok"],
                    "accepted_rule": decision["accepted_rule"],
                    "accepted": decision["accepted"],
                }

                candidate_log.append(log_row)

                if not decision["accepted"]:
                    continue

                # accepted 후보끼리는 fused를 우선하고,
                # 동점에 가까우면 gap이 짧은 predecessor를 선호.
                selection_key = (
                    float(score["fused_score"]),
                    -float(gap_sec),
                )

                if (
                    best is None
                    or selection_key > best["selection_key"]
                ):
                    best = {
                        "identity": ident,
                        "predecessor": predecessor,
                        "score": score,
                        "decision": decision,
                        "gap_sec": gap_sec,
                        "selection_key": selection_key,
                    }

        # ----------------------------------------------------
        # New identity
        # ----------------------------------------------------
        if best is None:
            ident = {
                "object_id": next_object_id,
                "class_name": tracklet["class_name"],
                "track_ids": [tracklet["track_id"]],

                "global_embedding": tracklet["global_embedding"].copy(),
                "global_weight": float(
                    max(1, tracklet["observations"])
                ),

                "gallery": [
                    gallery_entry(tracklet)
                ],

                "last_track_id": tracklet["track_id"],
                "start_sec": tracklet["start_sec"],
                "last_end_sec": tracklet["end_sec"],
                "start_frame": tracklet["start_frame"],
                "last_end_frame": tracklet["end_frame"],
            }

            identities.append(ident)

            assignment[tracklet["track_id"]] = {
                "object_id": next_object_id,
                "stitched_from_track_id": None,
                "stitch_rule": None,
                "stitch_endpoint_topk_mean": None,
                "stitch_endpoint_max": None,
                "stitch_global_similarity": None,
                "stitch_gallery_global_topk_mean": None,
                "stitch_fused_score": None,
                "stitch_gap_sec": None,
                "stitch_position_score": None,
                "stitch_bbox_scale_ratio": None,
            }

            next_object_id += 1
            continue

        # ----------------------------------------------------
        # Merge into existing identity
        # ----------------------------------------------------
        ident = best["identity"]
        predecessor = best["predecessor"]
        score = best["score"]
        decision = best["decision"]

        old_weight = float(ident["global_weight"])
        new_weight = float(
            max(1, tracklet["observations"])
        )

        ident["global_embedding"] = normalize(
            ident["global_embedding"] * old_weight
            + tracklet["global_embedding"] * new_weight
        )
        ident["global_weight"] = old_weight + new_weight

        ident["track_ids"].append(
            tracklet["track_id"]
        )

        ident["gallery"].append(
            gallery_entry(tracklet)
        )
        ident["gallery"] = trim_gallery(
            ident["gallery"],
            gallery_size,
        )

        ident["last_track_id"] = tracklet["track_id"]
        ident["last_end_sec"] = max(
            ident["last_end_sec"],
            tracklet["end_sec"],
        )
        ident["last_end_frame"] = max(
            ident["last_end_frame"],
            tracklet["end_frame"],
        )

        assignment[tracklet["track_id"]] = {
            "object_id": ident["object_id"],
            "stitched_from_track_id": predecessor["track_id"],
            "stitch_rule": decision["accepted_rule"],
            "stitch_endpoint_topk_mean": (
                None
                if score["endpoint_topk_mean"] is None
                else round(float(score["endpoint_topk_mean"]), 6)
            ),
            "stitch_endpoint_max": (
                None
                if score["endpoint_max"] is None
                else round(float(score["endpoint_max"]), 6)
            ),
            "stitch_global_similarity": (
                None
                if score["global_similarity"] is None
                else round(float(score["global_similarity"]), 6)
            ),
            "stitch_gallery_global_topk_mean": (
                None
                if score["gallery_global_topk_mean"] is None
                else round(
                    float(score["gallery_global_topk_mean"]),
                    6,
                )
            ),
            "stitch_fused_score": round(
                float(score["fused_score"]),
                6,
            ),
            "stitch_gap_sec": round(
                float(best["gap_sec"]),
                4,
            ),
            "stitch_position_score": round(
                float(score["position_score"]),
                6,
            ),
            "stitch_bbox_scale_ratio": (
                None
                if score["bbox_scale_ratio"] is None
                else round(float(score["bbox_scale_ratio"]), 6)
            ),
        }

    # embedding을 만들지 못한 tracklet은 독립 object 유지.
    for track_id in sorted(tracklets):
        if track_id in assignment:
            continue

        t = tracklets[track_id]

        assignment[track_id] = {
            "object_id": next_object_id,
            "stitched_from_track_id": None,
            "stitch_rule": None,
            "stitch_endpoint_topk_mean": None,
            "stitch_endpoint_max": None,
            "stitch_global_similarity": None,
            "stitch_gallery_global_topk_mean": None,
            "stitch_fused_score": None,
            "stitch_gap_sec": None,
            "stitch_position_score": None,
            "stitch_bbox_scale_ratio": None,
        }

        identities.append({
            "object_id": next_object_id,
            "class_name": t["class_name"],
            "track_ids": [track_id],
            "global_embedding": t["global_embedding"],
            "global_weight": 0.0,
            "gallery": (
                [gallery_entry(t)]
                if t["end_vectors"] is not None
                else []
            ),
            "last_track_id": track_id,
            "start_sec": t["start_sec"],
            "last_end_sec": t["end_sec"],
            "start_frame": t["start_frame"],
            "last_end_frame": t["end_frame"],
        })

        next_object_id += 1

    identities.sort(
        key=lambda x: x["object_id"]
    )

    return identities, assignment, candidate_log


# ============================================================
# Output V4.2
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
        out = dict(row)

        out["original_track_id"] = track_id
        out["object_id"] = int(info["object_id"])
        out["stitched_id"] = (
            f"object_{int(info['object_id']):04d}"
        )

        out["class_name"] = tracklets[track_id]["class_name"]
        out["final_class_name"] = tracklets[track_id]["class_name"]

        out["stitch_model"] = "dinov2_gallery_v4_2"
        out["stitched_from_track_id"] = info["stitched_from_track_id"]
        out["stitch_rule"] = info["stitch_rule"]
        out["stitch_endpoint_topk_mean"] = info["stitch_endpoint_topk_mean"]
        out["stitch_endpoint_max"] = info["stitch_endpoint_max"]
        out["stitch_global_similarity"] = info["stitch_global_similarity"]
        out["stitch_gallery_global_topk_mean"] = (
            info["stitch_gallery_global_topk_mean"]
        )
        out["stitch_fused_score"] = info["stitch_fused_score"]
        out["stitch_gap_sec"] = info["stitch_gap_sec"]
        out["stitch_position_score"] = info["stitch_position_score"]
        out["stitch_bbox_scale_ratio"] = info["stitch_bbox_scale_ratio"]

        stitched_rows.append(out)

    stitched_path = (
        video_dir / "stitched_tracks_v4_2.jsonl"
    )
    write_jsonl(stitched_path, stitched_rows)

    candidates_path = (
        video_dir / "stitch_candidates_v4_2.jsonl"
    )
    write_jsonl(candidates_path, candidate_log)

    crop_error_path = (
        video_dir / "stitch_crop_errors_v4_2.jsonl"
    )
    write_jsonl(crop_error_path, crop_errors)

    rule_counts = Counter(
        info["stitch_rule"]
        for info in assignment.values()
        if info["stitch_rule"] is not None
    )

    summary = {
        "scope": "object",
        "stitch_model": "dinov2_gallery_v4_2",

        "global_top_k": args.global_top_k,
        "endpoint_k": args.endpoint_k,
        "pair_top_k": args.pair_top_k,
        "gallery_size": args.gallery_size,
        "gallery_global_top_k": args.gallery_global_top_k,
        "max_gap_sec": args.max_gap_sec,

        "rules": {
            "general": {
                "fused_threshold": args.fused_threshold,
            },
            "strong_global": {
                "global_threshold": args.strong_global_threshold,
                "max_gap_sec": args.strong_global_max_gap_sec,
                "min_endpoint_max": args.strong_global_min_endpoint_max,
                "min_position": args.strong_global_min_position,
            },
            "short_gap": {
                "max_gap_sec": args.short_gap_sec,
                "min_position": args.short_min_position,
                "min_endpoint_max": args.short_min_endpoint_max,
                "min_global": args.short_min_global,
            },
            "long_gap_reentry": {
                "min_gap_sec": args.long_gap_min_sec,
                "max_gap_sec": args.long_gap_max_sec,
                "min_endpoint_max": args.long_min_endpoint_max,
                "min_endpoint_topk_mean": args.long_min_endpoint_topk,
                "min_global": args.long_min_global,
                "min_gallery_global_topk_mean": args.long_min_gallery_global,
                "min_position": args.long_min_position,
                "max_scale_ratio": args.long_max_scale_ratio,
                "min_fused": args.long_min_fused,
                "status": "ONE-VIDEO VALIDATION DEFAULT",
            },
            "max_scale_ratio": args.max_scale_ratio,
        },

        "weights": {
            "endpoint": args.endpoint_weight,
            "global": args.global_weight,
            "time": args.time_weight,
            "position": args.position_weight,
            "scale": args.scale_weight,
        },

        "input_rows": len(rows),
        "input_tracklets": len(tracklets),
        "output_object_ids": len(identities),
        "output_rows": len(stitched_rows),
        "skipped_output_rows": skipped_output_rows,
        "crop_errors": len(crop_errors),
        "merge_rule_counts": dict(rule_counts),
        "objects": [],
    }

    npz_data = {}

    for track_id, t in sorted(tracklets.items()):
        if t["global_embedding"] is not None:
            npz_data[
                f"track_{track_id}_global"
            ] = np.asarray(
                t["global_embedding"],
                dtype=np.float32,
            )

        if t["start_vectors"] is not None:
            npz_data[
                f"track_{track_id}_start"
            ] = np.asarray(
                t["start_vectors"],
                dtype=np.float32,
            )

        if t["end_vectors"] is not None:
            npz_data[
                f"track_{track_id}_end"
            ] = np.asarray(
                t["end_vectors"],
                dtype=np.float32,
            )

    for ident in identities:
        object_id = int(ident["object_id"])
        representative_crops = []

        for track_id in ident["track_ids"]:
            representative_crops.extend(
                tracklets[track_id]["global_crops"]
            )

        summary["objects"].append({
            "object_id": object_id,
            "stitched_id": f"object_{object_id:04d}",
            "final_class_name": ident["class_name"],
            "track_ids": ident["track_ids"],
            "start_sec": round(float(ident["start_sec"]), 4),
            "end_sec": round(float(ident["last_end_sec"]), 4),
            "gallery_track_ids": [
                int(x["track_id"])
                for x in ident.get("gallery", [])
            ],
            "representative_crops": representative_crops,
        })

        if ident["global_embedding"] is not None:
            npz_data[
                f"object_{object_id}_global"
            ] = np.asarray(
                ident["global_embedding"],
                dtype=np.float32,
            )

    summary_path = (
        video_dir / "stitching_v4_2.json"
    )
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    embedding_path = (
        video_dir / "stitch_embeddings_dinov2_v4_2.npz"
    )
    np.savez_compressed(
        embedding_path,
        **npz_data,
    )

    return (
        stitched_path,
        summary_path,
        candidates_path,
        crop_error_path,
        embedding_path,
    )


def find_video_dirs() -> list[Path]:
    if not TRACK_ROOT.exists():
        return []

    return sorted(
        p.parent
        for p in TRACK_ROOT.glob("*/tracks.jsonl")
        if p.is_file()
    )


def save_video_error(
    video_dir: Path,
    exc: Exception,
) -> Path:
    error_path = (
        video_dir / "stitch_error_v4_2.json"
    )

    payload = {
        "video": video_dir.name,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }

    error_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return error_path


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Object Tracklet Stitching V4.2: "
            "DINOv2 identity gallery + pairwise endpoint "
            "+ global + time + position + bbox scale"
        )
    )

    ap.add_argument(
        "--max-videos",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--video",
        type=str,
        default=None,
        help="process only one video track directory by stem/name",
    )

    ap.add_argument(
        "--global-top-k",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--endpoint-k",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--pair-top-k",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--gallery-size",
        type=int,
        default=5,
        help="identity별 최근 tracklet gallery 크기",
    )

    ap.add_argument(
        "--gallery-global-top-k",
        type=int,
        default=2,
        help=(
            "strong-global fallback에서 사용할 "
            "gallery global similarity 상위 K개 평균"
        ),
    )

    ap.add_argument(
        "--max-gap-sec",
        type=float,
        default=20.0,
    )

    # A. General merge
    ap.add_argument(
        "--fused-threshold",
        type=float,
        default=0.70,
    )

    # B. Strong-global fallback
    ap.add_argument(
        "--strong-global-threshold",
        type=float,
        default=0.85,
    )
    ap.add_argument(
        "--strong-global-max-gap-sec",
        type=float,
        default=8.0,
    )
    ap.add_argument(
        "--strong-global-min-endpoint-max",
        type=float,
        default=0.60,
    )
    ap.add_argument(
        "--strong-global-min-position",
        type=float,
        default=0.95,
    )

    # C. Short-gap continuity fallback
    ap.add_argument(
        "--short-gap-sec",
        type=float,
        default=2.5,
    )
    ap.add_argument(
        "--short-min-position",
        type=float,
        default=0.95,
    )
    ap.add_argument(
        "--short-min-endpoint-max",
        type=float,
        default=0.55,
    )
    ap.add_argument(
        "--short-min-global",
        type=float,
        default=0.45,
    )

    # D. Long-gap re-entry fallback (validation default)
    ap.add_argument("--long-gap-min-sec", type=float, default=8.0)
    ap.add_argument("--long-gap-max-sec", type=float, default=20.0)
    ap.add_argument("--long-min-endpoint-max", type=float, default=0.64)
    ap.add_argument("--long-min-endpoint-topk", type=float, default=0.62)
    ap.add_argument("--long-min-global", type=float, default=0.62)
    ap.add_argument("--long-min-gallery-global", type=float, default=0.64)
    ap.add_argument("--long-min-position", type=float, default=0.88)
    ap.add_argument("--long-max-scale-ratio", type=float, default=1.50)
    ap.add_argument("--long-min-fused", type=float, default=0.60)

    ap.add_argument(
        "--max-scale-ratio",
        type=float,
        default=2.5,
        help="fallback rule에서 허용할 bbox 대각선 크기 비율",
    )

    # Fused weights
    ap.add_argument(
        "--endpoint-weight",
        type=float,
        default=0.65,
    )
    ap.add_argument(
        "--global-weight",
        type=float,
        default=0.20,
    )
    ap.add_argument(
        "--time-weight",
        type=float,
        default=0.05,
    )
    ap.add_argument(
        "--position-weight",
        type=float,
        default=0.05,
    )
    ap.add_argument(
        "--scale-weight",
        type=float,
        default=0.05,
    )

    args = ap.parse_args()

    positive_int_args = {
        "global_top_k": args.global_top_k,
        "endpoint_k": args.endpoint_k,
        "pair_top_k": args.pair_top_k,
        "gallery_size": args.gallery_size,
        "gallery_global_top_k": args.gallery_global_top_k,
    }

    for name, value in positive_int_args.items():
        if value <= 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be >= 1"
            )

    nonnegative_args = {
        "max_gap_sec": args.max_gap_sec,
        "strong_global_max_gap_sec": args.strong_global_max_gap_sec,
        "short_gap_sec": args.short_gap_sec,
        "long_gap_min_sec": args.long_gap_min_sec,
        "long_gap_max_sec": args.long_gap_max_sec,
    }

    for name, value in nonnegative_args.items():
        if value < 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be >= 0"
            )

    similarity_args = {
        "fused_threshold": args.fused_threshold,
        "strong_global_threshold": args.strong_global_threshold,
        "strong_global_min_endpoint_max": args.strong_global_min_endpoint_max,
        "strong_global_min_position": args.strong_global_min_position,
        "short_min_position": args.short_min_position,
        "short_min_endpoint_max": args.short_min_endpoint_max,
        "short_min_global": args.short_min_global,
        "long_min_endpoint_max": args.long_min_endpoint_max,
        "long_min_endpoint_topk": args.long_min_endpoint_topk,
        "long_min_global": args.long_min_global,
        "long_min_gallery_global": args.long_min_gallery_global,
        "long_min_position": args.long_min_position,
        "long_min_fused": args.long_min_fused,
    }

    for name, value in similarity_args.items():
        if not (-1.0 <= value <= 1.0):
            raise ValueError(
                f"--{name.replace('_', '-')} must be in [-1, 1]"
            )

    if args.long_gap_max_sec < args.long_gap_min_sec:
        raise ValueError(
            "--long-gap-max-sec must be >= --long-gap-min-sec"
        )

    if args.long_gap_max_sec > args.max_gap_sec:
        raise ValueError(
            "--long-gap-max-sec must be <= --max-gap-sec"
        )

    if args.max_scale_ratio < 1.0:
        raise ValueError(
            "--max-scale-ratio must be >= 1.0"
        )

    if args.long_max_scale_ratio < 1.0:
        raise ValueError(
            "--long-max-scale-ratio must be >= 1.0"
        )

    weights = [
        args.endpoint_weight,
        args.global_weight,
        args.time_weight,
        args.position_weight,
        args.scale_weight,
    ]

    if any(w < 0 for w in weights):
        raise ValueError(
            "stitch weights must be >= 0"
        )

    if sum(weights) <= 0:
        raise ValueError(
            "sum of stitch weights must be > 0"
        )

    video_dirs = find_video_dirs()

    if args.video:
        wanted = Path(args.video).stem
        video_dirs = [
            p for p in video_dirs
            if p.name == args.video or p.name == wanted
        ]
        if not video_dirs:
            raise RuntimeError(
                f"No object track directory matched --video {args.video!r}"
            )

    if args.max_videos is not None:
        if args.max_videos <= 0:
            raise ValueError(
                "--max-videos must be >= 1"
            )
        video_dirs = video_dirs[:args.max_videos]

    print("=" * 96)
    print(
        "OBJECT TRACKLET STITCHING V4.2 "
        "- DINOV2 IDENTITY GALLERY + PAIRWISE + GLOBAL + TIME + POSITION + SCALE"
    )
    print("=" * 96)

    print("track root        :", TRACK_ROOT)
    print("videos            :", len(video_dirs))
    print("global top-k      :", args.global_top_k)
    print("endpoint k        :", args.endpoint_k)
    print("pair top-k        :", args.pair_top_k)
    print("gallery size      :", args.gallery_size)
    print("gallery global k  :", args.gallery_global_top_k)
    print("max gap sec       :", args.max_gap_sec)
    print("fused threshold   :", args.fused_threshold)
    print(
        "strong global     :",
        {
            "global": args.strong_global_threshold,
            "max_gap": args.strong_global_max_gap_sec,
            "endpoint_max": args.strong_global_min_endpoint_max,
            "position": args.strong_global_min_position,
        },
    )
    print(
        "short gap         :",
        {
            "max_gap": args.short_gap_sec,
            "position": args.short_min_position,
            "endpoint_max": args.short_min_endpoint_max,
            "global": args.short_min_global,
        },
    )
    print(
        "long-gap reentry  :",
        {
            "gap": [args.long_gap_min_sec, args.long_gap_max_sec],
            "endpoint_max": args.long_min_endpoint_max,
            "endpoint_topk": args.long_min_endpoint_topk,
            "global": args.long_min_global,
            "gallery_global": args.long_min_gallery_global,
            "position": args.long_min_position,
            "max_scale": args.long_max_scale_ratio,
            "fused": args.long_min_fused,
            "status": "ONE-VIDEO VALIDATION DEFAULT",
        },
    )
    print("max scale ratio   :", args.max_scale_ratio)
    print(
        "weights           :",
        {
            "endpoint": args.endpoint_weight,
            "global": args.global_weight,
            "time": args.time_weight,
            "position": args.position_weight,
            "scale": args.scale_weight,
        },
    )

    if not video_dirs:
        print("No object tracks.jsonl found.")
        return

    cfg = PipelineConfig.load(CONFIG_PATH)
    embedder, spec = load_embedder(
        cfg,
        "dinov2",
    )

    print("model             :", spec.name)
    print("module            :", spec.module)
    print("class             :", spec.class_name)
    print("dim               :", spec.dim)

    passed = 0
    failed = 0

    for idx, video_dir in enumerate(
        video_dirs,
        start=1,
    ):
        print(
            f"\n[{idx}/{len(video_dirs)}] "
            f"{video_dir.name}"
        )

        try:
            tracks_path = (
                video_dir / "tracks.jsonl"
            )
            rows = load_jsonl(tracks_path)

            if not rows:
                print("  input rows      : 0")
                print("  nothing to stitch")
                passed += 1
                continue

            tracklets = build_tracklets(rows)

            crop_errors = embed_tracklets(
                embedder=embedder,
                tracklets=tracklets,
                global_top_k=args.global_top_k,
                endpoint_k=args.endpoint_k,
            )

            (
                identities,
                assignment,
                candidate_log,
            ) = stitch_tracklets(
                tracklets=tracklets,
                max_gap_sec=args.max_gap_sec,
                gallery_size=args.gallery_size,
                pair_top_k=args.pair_top_k,
                gallery_global_top_k=args.gallery_global_top_k,
                fused_threshold=args.fused_threshold,
                strong_global_threshold=args.strong_global_threshold,
                strong_global_max_gap_sec=args.strong_global_max_gap_sec,
                strong_global_min_endpoint_max=args.strong_global_min_endpoint_max,
                strong_global_min_position=args.strong_global_min_position,
                short_gap_sec=args.short_gap_sec,
                short_min_position=args.short_min_position,
                short_min_endpoint_max=args.short_min_endpoint_max,
                short_min_global=args.short_min_global,
                long_gap_min_sec=args.long_gap_min_sec,
                long_gap_max_sec=args.long_gap_max_sec,
                long_min_endpoint_max=args.long_min_endpoint_max,
                long_min_endpoint_topk=args.long_min_endpoint_topk,
                long_min_global=args.long_min_global,
                long_min_gallery_global=args.long_min_gallery_global,
                long_min_position=args.long_min_position,
                long_max_scale_ratio=args.long_max_scale_ratio,
                long_min_fused=args.long_min_fused,
                max_scale_ratio=args.max_scale_ratio,
                endpoint_weight=args.endpoint_weight,
                global_weight=args.global_weight,
                time_weight=args.time_weight,
                position_weight=args.position_weight,
                scale_weight=args.scale_weight,
            )

            (
                stitched_path,
                summary_path,
                candidates_path,
                crop_error_path,
                embedding_path,
            ) = save_outputs(
                video_dir=video_dir,
                rows=rows,
                tracklets=tracklets,
                identities=identities,
                assignment=assignment,
                candidate_log=candidate_log,
                crop_errors=crop_errors,
                args=args,
            )

            print("  input rows      :", len(rows))
            print("  input tracklets :", len(tracklets))
            print("  output objects  :", len(identities))
            print(
                "  merge rules     :",
                dict(
                    Counter(
                        x["stitch_rule"]
                        for x in assignment.values()
                        if x["stitch_rule"] is not None
                    )
                ),
            )
            print(
                "  mapping         :",
                {
                    o["object_id"]: {
                        "class": o["class_name"],
                        "tracks": o["track_ids"],
                    }
                    for o in identities
                },
            )
            print("  crop errors     :", len(crop_errors))
            print("  stitched rows   :", stitched_path)
            print("  stitching json  :", summary_path)
            print("  candidates      :", candidates_path)
            print("  crop error log  :", crop_error_path)
            print("  DINOv2 cache    :", embedding_path)

            passed += 1

        except Exception as exc:
            failed += 1
            error_path = save_video_error(
                video_dir,
                exc,
            )

            print(
                "  [FAIL]",
                type(exc).__name__,
                str(exc),
            )
            print("  error log       :", error_path)
            continue

    print("\n" + "=" * 96)
    print("STITCH V4.2 COMPLETE")
    print("passed :", passed)
    print("failed :", failed)
    print("=" * 96)


if __name__ == "__main__":
    main()