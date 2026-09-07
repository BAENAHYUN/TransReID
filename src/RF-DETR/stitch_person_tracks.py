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
TRACK_ROOT = ROOT / "data" / "video_tracks" / "person"


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


# ============================================================
# V4.5 clothing appearance descriptor
# ============================================================

CLOTHING_ZONES = {
    "upper_left": (0.03, 0.18, 0.37, 0.58),
    "upper_center": (0.30, 0.18, 0.70, 0.58),
    "upper_right": (0.63, 0.18, 0.97, 0.58),
    "lower": (0.12, 0.52, 0.88, 0.95),
}

CLOTHING_ZONE_WEIGHTS = {
    "upper_left": 0.25,
    "upper_center": 0.30,
    "upper_right": 0.25,
    "lower": 0.20,
}


def clothing_zone_histogram(
    image: Image.Image,
    box: tuple[float, float, float, float],
    *,
    h_bins: int = 12,
    s_bins: int = 4,
    v_bins: int = 4,
) -> np.ndarray | None:
    """
    Independent HSV clothing descriptor.

    This is only a negative/veto signal. It never creates a positive identity
    match by itself. Low-saturation black/white clothing is retained because
    S and V are included in the joint histogram.
    """
    rgb = image.convert("RGB")
    w, h = rgb.size

    if w < 4 or h < 8:
        return None

    x1, y1, x2, y2 = box
    left = max(0, min(w - 1, int(round(x1 * w))))
    top = max(0, min(h - 1, int(round(y1 * h))))
    right = max(left + 1, min(w, int(round(x2 * w))))
    bottom = max(top + 1, min(h, int(round(y2 * h))))

    arr = np.asarray(
        rgb.crop((left, top, right, bottom)).convert("HSV"),
        dtype=np.uint8,
    )
    if arr.size == 0:
        return None

    hh = arr[..., 0].astype(np.int32)
    ss = arr[..., 1].astype(np.int32)
    vv = arr[..., 2].astype(np.int32)

    hi = np.minimum((hh * h_bins) // 256, h_bins - 1)
    si = np.minimum((ss * s_bins) // 256, s_bins - 1)
    vi = np.minimum((vv * v_bins) // 256, v_bins - 1)

    idx = (hi * s_bins + si) * v_bins + vi
    hist = np.bincount(
        idx.reshape(-1),
        minlength=h_bins * s_bins * v_bins,
    ).astype(np.float32)

    total = float(hist.sum())
    if total <= 0:
        return None

    return hist / total


def clothing_profile_from_image(
    image: Image.Image,
) -> dict[str, np.ndarray] | None:
    profile = {}

    for name, box in CLOTHING_ZONES.items():
        hist = clothing_zone_histogram(image, box)
        if hist is None:
            return None
        profile[name] = hist

    return profile


def copy_clothing_profile(profile):
    if profile is None:
        return None

    return {
        key: np.asarray(value, dtype=np.float32).copy()
        for key, value in profile.items()
    }


def mean_clothing_profiles(profiles):
    if not profiles:
        return None

    out = {}

    for zone in CLOTHING_ZONES:
        values = [
            np.asarray(p[zone], dtype=np.float32)
            for p in profiles
            if zone in p
        ]
        if not values:
            return None

        hist = np.mean(np.stack(values, axis=0), axis=0)
        total = float(hist.sum())
        if total > 0:
            hist = hist / total
        out[zone] = hist.astype(np.float32)

    return out


def blend_clothing_profiles(a, a_weight: float, b, b_weight: float):
    if a is None:
        return copy_clothing_profile(b)
    if b is None:
        return copy_clothing_profile(a)

    aw = max(0.0, float(a_weight))
    bw = max(0.0, float(b_weight))
    total_weight = aw + bw

    if total_weight <= 0:
        return copy_clothing_profile(a)

    out = {}

    for zone in CLOTHING_ZONES:
        av = np.asarray(a[zone], dtype=np.float32)
        bv = np.asarray(b[zone], dtype=np.float32)
        hist = (av * aw + bv * bw) / total_weight

        total = float(hist.sum())
        if total > 0:
            hist = hist / total

        out[zone] = hist.astype(np.float32)

    return out


def bhattacharyya_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.maximum(np.asarray(a, dtype=np.float32), 0.0)
    bb = np.maximum(np.asarray(b, dtype=np.float32), 0.0)

    sa = float(aa.sum())
    sb = float(bb.sum())

    if sa <= 0 or sb <= 0:
        return 0.0

    aa /= sa
    bb /= sb

    return float(np.sqrt(aa * bb).sum())


def clothing_profile_similarity(a, b):
    if a is None or b is None:
        return None, None, {}

    zone_scores = {}

    for zone in CLOTHING_ZONES:
        if zone not in a or zone not in b:
            return None, None, {}

        zone_scores[zone] = bhattacharyya_similarity(
            a[zone],
            b[zone],
        )

    weighted = sum(
        CLOTHING_ZONE_WEIGHTS[zone] * zone_scores[zone]
        for zone in CLOTHING_ZONES
    )

    upper_min = min(
        zone_scores["upper_left"],
        zone_scores["upper_center"],
        zone_scores["upper_right"],
    )

    return float(weighted), float(upper_min), zone_scores


def compute_tracklet_clothing_profiles(
    tracklets: dict[int, dict],
    *,
    min_crops: int,
) -> list[dict]:
    """
    Use the same time-distributed crops selected for SOLIDER global embedding.

    The veto becomes active only when enough representative crops were
    decoded. This avoids pretending that a one-frame tiny track has a stable
    clothing signature.
    """
    errors = []

    for track_id in sorted(tracklets):
        t = tracklets[track_id]
        profiles = []

        for crop_path in t.get("global_crops", []):
            try:
                with Image.open(crop_path) as im:
                    rgb = im.convert("RGB").copy()

                if rgb.width < 16 or rgb.height < 32:
                    continue

                profile = clothing_profile_from_image(rgb)
                if profile is not None:
                    profiles.append(profile)

            except Exception as exc:
                errors.append({
                    "stage": "clothing_profile",
                    "track_id": int(track_id),
                    "crop_path": str(crop_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })

        t["clothing_profile"] = mean_clothing_profiles(profiles)
        t["clothing_crop_count"] = len(profiles)
        t["clothing_reliable"] = bool(
            t["clothing_profile"] is not None
            and len(profiles) >= min_crops
        )

    return errors


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
            "class_name": "person",
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

            "clothing_profile": None,
            "clothing_crop_count": 0,
            "clothing_reliable": False,
        }

    return tracklets


# ============================================================
# SOLIDER embedding
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
    모든 필요한 crop을 한 번만 SOLIDER에 넣고,
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
            f"SOLIDER embed_crops failed for "
            f"{len(valid_paths)} crops"
        ) from exc

    if vecs.ndim != 2 or vecs.shape[0] != len(valid_paths):
        raise RuntimeError(
            f"SOLIDER embedding shape mismatch: "
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
# V4.5 gallery-level global appearance
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
    V4.5 conservative candidate score.

    Key change from V4.1:
      - V4.1 used max(predecessor_global, identity_global).
      - V4.5 uses a consensus minimum across predecessor, identity centroid,
        and the fixed identity anchor when available.

    This intentionally prefers false-split over false-merge.
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
    anchor_global = identity.get("anchor_embedding")
    track_global = tracklet.get("global_embedding")

    predecessor_global_sim = None
    identity_global_sim = None
    anchor_global_sim = None

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

    if anchor_global is not None and track_global is not None:
        anchor_global_sim = cosine(
            anchor_global,
            track_global,
        )

    available_global = [
        x
        for x in (
            predecessor_global_sim,
            identity_global_sim,
            anchor_global_sim,
        )
        if x is not None
    ]

    global_consensus = (
        min(available_global)
        if available_global
        else None
    )

    global_max = (
        max(available_global)
        if available_global
        else None
    )

    # V4.5 independent clothing appearance consensus.
    track_clothing = tracklet.get("clothing_profile")
    track_clothing_reliable = bool(
        tracklet.get("clothing_reliable", False)
    )

    clothing_detail = {}
    clothing_values = []
    clothing_upper_values = []

    clothing_sources = [
        (
            "predecessor",
            predecessor.get("clothing_profile"),
            bool(predecessor.get("clothing_reliable", False)),
        ),
        (
            "identity",
            identity.get("clothing_profile"),
            bool(identity.get("clothing_reliable", False)),
        ),
        (
            "anchor",
            identity.get("anchor_clothing_profile"),
            bool(identity.get("anchor_clothing_reliable", False)),
        ),
    ]

    for source_name, source_profile, source_reliable in clothing_sources:
        sim = None
        upper_min = None
        zones = {}

        if (
            track_clothing_reliable
            and source_reliable
            and source_profile is not None
            and track_clothing is not None
        ):
            sim, upper_min, zones = clothing_profile_similarity(
                source_profile,
                track_clothing,
            )

        clothing_detail[source_name] = {
            "similarity": sim,
            "upper_min_similarity": upper_min,
            "zone_scores": zones,
        }

        if sim is not None:
            clothing_values.append(float(sim))
        if upper_min is not None:
            clothing_upper_values.append(float(upper_min))

    clothing_consensus = (
        min(clothing_values)
        if clothing_values
        else None
    )
    clothing_upper_consensus = (
        min(clothing_upper_values)
        if clothing_upper_values
        else None
    )

    clothing_gate_applicable = bool(
        track_clothing_reliable
        and clothing_consensus is not None
        and clothing_upper_consensus is not None
    )

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

    effective_position_score = p_score * t_score

    parts = []

    if endpoint_score is not None:
        parts.append(
            (endpoint_weight, endpoint_score)
        )

    if global_consensus is not None:
        parts.append(
            (global_weight, global_consensus)
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
        "anchor_global_similarity": anchor_global_sim,
        "global_similarity": global_consensus,
        "global_max_similarity": global_max,
        "global_similarity_source": "consensus_min",
        "clothing_similarity": clothing_consensus,
        "clothing_upper_min_similarity": clothing_upper_consensus,
        "clothing_gate_applicable": clothing_gate_applicable,
        "clothing_detail": clothing_detail,
        "clothing_track_reliable": track_clothing_reliable,
        "clothing_track_crop_count": int(
            tracklet.get("clothing_crop_count", 0)
        ),
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
    general_max_gap_sec: float,
    general_min_endpoint_topk: float,
    general_min_endpoint_max: float,
    general_min_global: float,
    general_min_gallery_global: float,
    general_min_anchor: float,
    strong_global_threshold: float,
    strong_global_max_gap_sec: float,
    strong_global_min_endpoint_topk: float,
    strong_global_min_endpoint_max: float,
    strong_global_min_gallery_global: float,
    strong_global_min_anchor: float,
    strong_global_min_position: float,
    short_gap_sec: float,
    short_min_position: float,
    short_min_endpoint_topk: float,
    short_min_endpoint_max: float,
    short_min_global: float,
    short_min_anchor: float,
    long_gap_min_sec: float,
    long_gap_max_sec: float,
    long_min_endpoint_topk: float,
    long_min_endpoint_max: float,
    long_min_global: float,
    long_min_gallery_global: float,
    long_min_anchor: float,
    long_min_fused: float,
    long_max_scale_ratio: float,
    max_scale_ratio: float,
    allow_long_gap_auto_merge: bool,
    general_min_clothing: float,
    general_min_clothing_upper: float,
    strong_min_clothing: float,
    strong_min_clothing_upper: float,
    short_min_clothing: float,
    short_min_clothing_upper: float,
    long_min_clothing: float,
    long_min_clothing_upper: float,
) -> dict:
    """
    Person Stitching V4.5 conservative merge rules.

    A. general
       Multi-signal hard gates + fused threshold, max 8s by default.

    B. strong_global
       Very strong appearance consensus for a short gap.

    C. short_gap
       Very short temporal continuity with moderate appearance hard gates.

    D. long_gap_reentry
       8~20s re-entry is candidate-only by default; explicit opt-in is required for auto-merge.

    The intent is forensic-safe behavior: prefer false split over false merge.
    """
    endpoint_topk = score.get("endpoint_topk_mean")
    endpoint_max = score.get("endpoint_max")
    global_sim = score.get("global_similarity")
    gallery_global = score.get("gallery_global_topk_mean")
    anchor_global = score.get("anchor_global_similarity")
    position = score.get("position_score", 0.0)
    scale_ratio = score.get("bbox_scale_ratio")

    clothing_similarity = score.get("clothing_similarity")
    clothing_upper_min = score.get(
        "clothing_upper_min_similarity"
    )
    clothing_gate_applicable = bool(
        score.get("clothing_gate_applicable", False)
    )

    def clothing_veto(min_similarity: float, min_upper: float) -> bool:
        # Veto only when BOTH overall colour distribution and the weakest
        # upper-body zone disagree. Missing/unreliable descriptors never
        # create a fake veto.
        if not clothing_gate_applicable:
            return False
        if clothing_similarity is None or clothing_upper_min is None:
            return False
        return bool(
            clothing_similarity < min_similarity
            and clothing_upper_min < min_upper
        )

    general_clothing_veto = clothing_veto(
        general_min_clothing,
        general_min_clothing_upper,
    )
    strong_clothing_veto = clothing_veto(
        strong_min_clothing,
        strong_min_clothing_upper,
    )
    short_clothing_veto = clothing_veto(
        short_min_clothing,
        short_min_clothing_upper,
    )
    long_clothing_veto = clothing_veto(
        long_min_clothing,
        long_min_clothing_upper,
    )

    scale_ok = bool(
        scale_ratio is not None
        and scale_ratio <= max_scale_ratio
    )

    general_pass = bool(
        endpoint_topk is not None
        and endpoint_max is not None
        and global_sim is not None
        and gallery_global is not None
        and anchor_global is not None
        and gap_sec <= general_max_gap_sec
        and score["fused_score"] >= fused_threshold
        and endpoint_topk >= general_min_endpoint_topk
        and endpoint_max >= general_min_endpoint_max
        and global_sim >= general_min_global
        and gallery_global >= general_min_gallery_global
        and anchor_global >= general_min_anchor
        and not general_clothing_veto
        and scale_ok
    )

    strong_global_pass = bool(
        endpoint_topk is not None
        and endpoint_max is not None
        and global_sim is not None
        and gallery_global is not None
        and anchor_global is not None
        and gap_sec <= strong_global_max_gap_sec
        and global_sim >= strong_global_threshold
        and gallery_global >= strong_global_min_gallery_global
        and anchor_global >= strong_global_min_anchor
        and endpoint_topk >= strong_global_min_endpoint_topk
        and endpoint_max >= strong_global_min_endpoint_max
        and position >= strong_global_min_position
        and not strong_clothing_veto
        and scale_ok
    )

    short_gap_pass = bool(
        endpoint_topk is not None
        and endpoint_max is not None
        and global_sim is not None
        and anchor_global is not None
        and gap_sec <= short_gap_sec
        and position >= short_min_position
        and endpoint_topk >= short_min_endpoint_topk
        and endpoint_max >= short_min_endpoint_max
        and global_sim >= short_min_global
        and anchor_global >= short_min_anchor
        and not short_clothing_veto
        and scale_ok
    )

    long_scale_ok = bool(
        scale_ratio is not None
        and scale_ratio <= long_max_scale_ratio
    )

    long_gap_reentry_candidate = bool(
        endpoint_topk is not None
        and endpoint_max is not None
        and global_sim is not None
        and gallery_global is not None
        and anchor_global is not None
        and gap_sec >= long_gap_min_sec
        and gap_sec <= long_gap_max_sec
        and score["fused_score"] >= long_min_fused
        and endpoint_topk >= long_min_endpoint_topk
        and endpoint_max >= long_min_endpoint_max
        and global_sim >= long_min_global
        and gallery_global >= long_min_gallery_global
        and anchor_global >= long_min_anchor
        and not long_clothing_veto
        and long_scale_ok
    )

    # V4.5 forensic-safe default:
    # long-gap candidates are logged but not automatically merged unless
    # explicitly enabled by the operator.
    long_gap_reentry_pass = bool(
        allow_long_gap_auto_merge
        and long_gap_reentry_candidate
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
        "general_pass": general_pass,
        "strong_global_pass": strong_global_pass,
        "short_gap_pass": short_gap_pass,
        "long_gap_reentry_candidate": long_gap_reentry_candidate,
        "long_gap_reentry_pass": long_gap_reentry_pass,
        "clothing_gate_applicable": clothing_gate_applicable,
        "clothing_similarity": clothing_similarity,
        "clothing_upper_min_similarity": clothing_upper_min,
        "general_clothing_veto": general_clothing_veto,
        "strong_clothing_veto": strong_clothing_veto,
        "short_clothing_veto": short_clothing_veto,
        "long_clothing_veto": long_clothing_veto,
        "scale_ok": scale_ok,
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
        "clothing_profile": copy_clothing_profile(
            tracklet.get("clothing_profile")
        ),
        "clothing_reliable": bool(
            tracklet.get("clothing_reliable", False)
        ),
        "clothing_crop_count": int(
            tracklet.get("clothing_crop_count", 0)
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


def trusted_gallery_decision(
    score: dict,
    gap_sec: float,
    rule: str | None,
    *,
    trusted_max_gap_sec: float,
    trusted_min_fused: float,
    trusted_min_endpoint_topk: float,
    trusted_min_endpoint_max: float,
    trusted_min_global: float,
    trusted_min_gallery_global: float,
    trusted_min_anchor: float,
    trusted_long_min_fused: float,
    trusted_long_min_endpoint_topk: float,
    trusted_long_min_endpoint_max: float,
    trusted_long_min_global: float,
    trusted_long_min_gallery_global: float,
    trusted_long_min_anchor: float,
) -> tuple[bool, str]:
    """
    Decide whether an accepted tracklet is strong enough to update the
    identity centroid and trusted predecessor gallery.

    Important: acceptance and trust are separate. A tracklet may be assigned
    to an identity but remain untrusted, so it cannot drag the centroid or
    become a predecessor for later merges.
    """
    if rule is None:
        return False, "not_merged"

    endpoint_topk = score.get("endpoint_topk_mean")
    endpoint_max = score.get("endpoint_max")
    global_sim = score.get("global_similarity")
    gallery_global = score.get("gallery_global_topk_mean")
    anchor_global = score.get("anchor_global_similarity")
    fused = score.get("fused_score")

    required = [
        endpoint_topk, endpoint_max, global_sim,
        gallery_global, anchor_global, fused,
    ]
    if any(v is None for v in required):
        return False, "missing_signal"

    if rule == "long_gap_reentry":
        ok = bool(
            fused >= trusted_long_min_fused
            and endpoint_topk >= trusted_long_min_endpoint_topk
            and endpoint_max >= trusted_long_min_endpoint_max
            and global_sim >= trusted_long_min_global
            and gallery_global >= trusted_long_min_gallery_global
            and anchor_global >= trusted_long_min_anchor
        )
        return ok, ("trusted_long" if ok else "weak_long_merge")

    ok = bool(
        gap_sec <= trusted_max_gap_sec
        and fused >= trusted_min_fused
        and endpoint_topk >= trusted_min_endpoint_topk
        and endpoint_max >= trusted_min_endpoint_max
        and global_sim >= trusted_min_global
        and gallery_global >= trusted_min_gallery_global
        and anchor_global >= trusted_min_anchor
    )
    return ok, ("trusted_standard" if ok else "weak_standard_merge")



def stitch_tracklets(
    tracklets: dict[int, dict],
    max_gap_sec: float,
    gallery_size: int,
    pair_top_k: int,
    gallery_global_top_k: int,
    fused_threshold: float,
    general_max_gap_sec: float,
    general_min_endpoint_topk: float,
    general_min_endpoint_max: float,
    general_min_global: float,
    general_min_gallery_global: float,
    general_min_anchor: float,
    strong_global_threshold: float,
    strong_global_max_gap_sec: float,
    strong_global_min_endpoint_topk: float,
    strong_global_min_endpoint_max: float,
    strong_global_min_gallery_global: float,
    strong_global_min_anchor: float,
    strong_global_min_position: float,
    short_gap_sec: float,
    short_min_position: float,
    short_min_endpoint_topk: float,
    short_min_endpoint_max: float,
    short_min_global: float,
    short_min_anchor: float,
    long_gap_min_sec: float,
    long_gap_max_sec: float,
    long_min_endpoint_topk: float,
    long_min_endpoint_max: float,
    long_min_global: float,
    long_min_gallery_global: float,
    long_min_anchor: float,
    long_min_fused: float,
    long_max_scale_ratio: float,
    max_scale_ratio: float,
    allow_long_gap_auto_merge: bool,
    general_min_clothing: float,
    general_min_clothing_upper: float,
    strong_min_clothing: float,
    strong_min_clothing_upper: float,
    short_min_clothing: float,
    short_min_clothing_upper: float,
    long_min_clothing: float,
    long_min_clothing_upper: float,
    identity_margin: float,
    identity_update_weight_cap: float,
    trusted_max_gap_sec: float,
    trusted_min_fused: float,
    trusted_min_endpoint_topk: float,
    trusted_min_endpoint_max: float,
    trusted_min_global: float,
    trusted_min_gallery_global: float,
    trusted_min_anchor: float,
    trusted_long_min_fused: float,
    trusted_long_min_endpoint_topk: float,
    trusted_long_min_endpoint_max: float,
    trusted_long_min_global: float,
    trusted_long_min_gallery_global: float,
    trusted_long_min_anchor: float,
    calibration_only: bool,
    endpoint_weight: float,
    global_weight: float,
    time_weight: float,
    position_weight: float,
    scale_weight: float,
):
    """
    V4.5 conservative identity stitching.

    Main safeguards:
      1) fixed anchor embedding per identity
      2) global consensus = minimum(predecessor, identity centroid, anchor)
      3) hard multi-signal gates
      4) long-gap candidate logging with auto-merge disabled by default
      5) best-vs-runner-up identity margin
      6) capped centroid update weight to reduce identity drift
      7) trusted-gallery updates: only high-confidence merges may update
         the centroid or become future predecessor evidence
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
    next_person_id = 1

    def create_identity(tracklet: dict, rejection_reason=None, margin_value=None):
        nonlocal next_person_id

        ident = {
            "person_id": next_person_id,
            "class_name": tracklet["class_name"],
            "track_ids": [tracklet["track_id"]],

            "global_embedding": tracklet["global_embedding"].copy(),
            "global_weight": float(
                min(
                    max(1, tracklet["observations"]),
                    identity_update_weight_cap,
                )
            ),
            # Fixed anchor: never updated after identity creation.
            "anchor_embedding": tracklet["global_embedding"].copy(),

            # Independent clothing evidence follows the same anti-drift rule:
            # fixed anchor + trusted-only centroid updates.
            "clothing_profile": copy_clothing_profile(
                tracklet.get("clothing_profile")
            ),
            "clothing_weight": float(
                min(
                    max(1, tracklet.get("clothing_crop_count", 0)),
                    identity_update_weight_cap,
                )
            ),
            "clothing_reliable": bool(
                tracklet.get("clothing_reliable", False)
            ),
            "anchor_clothing_profile": copy_clothing_profile(
                tracklet.get("clothing_profile")
            ),
            "anchor_clothing_reliable": bool(
                tracklet.get("clothing_reliable", False)
            ),

            # V4.5: gallery is trusted-only predecessor evidence.
            "gallery": [
                gallery_entry(tracklet)
            ],
            "trusted_track_ids": [tracklet["track_id"]],

            "last_track_id": tracklet["track_id"],
            "start_sec": tracklet["start_sec"],
            "last_end_sec": tracklet["end_sec"],
            "start_frame": tracklet["start_frame"],
            "last_end_frame": tracklet["end_frame"],
        }

        identities.append(ident)

        assignment[tracklet["track_id"]] = {
            "person_id": next_person_id,
            "stitched_from_track_id": None,
            "stitch_rule": None,
            "stitch_endpoint_topk_mean": None,
            "stitch_endpoint_max": None,
            "stitch_global_similarity": None,
            "stitch_global_max_similarity": None,
            "stitch_anchor_global_similarity": None,
            "stitch_gallery_global_topk_mean": None,
            "stitch_clothing_similarity": None,
            "stitch_clothing_upper_min_similarity": None,
            "stitch_clothing_gate_applicable": False,
            "stitch_clothing_veto": False,
            "stitch_fused_score": None,
            "stitch_gap_sec": None,
            "stitch_position_score": None,
            "stitch_bbox_scale_ratio": None,
            "stitch_identity_margin": (
                None
                if margin_value is None
                else round(float(margin_value), 6)
            ),
            "stitch_rejection_reason": rejection_reason,
            "stitch_trusted_gallery_update": True,
            "stitch_trusted_gallery_reason": "seed",
        }

        next_person_id += 1

    for tracklet in usable:
        # Keep only the best predecessor candidate per existing identity.
        accepted_by_identity: dict[int, dict] = {}

        for ident in identities:
            identity_gap = (
                tracklet["start_sec"]
                - ident["last_end_sec"]
            )

            # Never merge temporally overlapping identities.
            if identity_gap < 0:
                continue

            if identity_gap > max_gap_sec:
                continue

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
                    general_max_gap_sec=general_max_gap_sec,
                    general_min_endpoint_topk=general_min_endpoint_topk,
                    general_min_endpoint_max=general_min_endpoint_max,
                    general_min_global=general_min_global,
                    general_min_gallery_global=general_min_gallery_global,
                    general_min_anchor=general_min_anchor,
                    strong_global_threshold=strong_global_threshold,
                    strong_global_max_gap_sec=strong_global_max_gap_sec,
                    strong_global_min_endpoint_topk=strong_global_min_endpoint_topk,
                    strong_global_min_endpoint_max=strong_global_min_endpoint_max,
                    strong_global_min_gallery_global=strong_global_min_gallery_global,
                    strong_global_min_anchor=strong_global_min_anchor,
                    strong_global_min_position=strong_global_min_position,
                    short_gap_sec=short_gap_sec,
                    short_min_position=short_min_position,
                    short_min_endpoint_topk=short_min_endpoint_topk,
                    short_min_endpoint_max=short_min_endpoint_max,
                    short_min_global=short_min_global,
                    short_min_anchor=short_min_anchor,
                    long_gap_min_sec=long_gap_min_sec,
                    long_gap_max_sec=long_gap_max_sec,
                    long_min_endpoint_topk=long_min_endpoint_topk,
                    long_min_endpoint_max=long_min_endpoint_max,
                    long_min_global=long_min_global,
                    long_min_gallery_global=long_min_gallery_global,
                    long_min_anchor=long_min_anchor,
                    long_min_fused=long_min_fused,
                    long_max_scale_ratio=long_max_scale_ratio,
                    max_scale_ratio=max_scale_ratio,
                    allow_long_gap_auto_merge=allow_long_gap_auto_merge,
                    general_min_clothing=general_min_clothing,
                    general_min_clothing_upper=general_min_clothing_upper,
                    strong_min_clothing=strong_min_clothing,
                    strong_min_clothing_upper=strong_min_clothing_upper,
                    short_min_clothing=short_min_clothing,
                    short_min_clothing_upper=short_min_clothing_upper,
                    long_min_clothing=long_min_clothing,
                    long_min_clothing_upper=long_min_clothing_upper,
                )

                would_accept = bool(decision["accepted"])

                if calibration_only:
                    decision = dict(decision)
                    decision["accepted"] = False
                    decision["accepted_rule"] = None

                log_row = {
                    "track_id": tracklet["track_id"],
                    "candidate_person_id": ident["person_id"],
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
                    "anchor_global_similarity": (
                        None
                        if score["anchor_global_similarity"] is None
                        else round(
                            float(score["anchor_global_similarity"]),
                            6,
                        )
                    ),
                    "global_similarity": (
                        None
                        if score["global_similarity"] is None
                        else round(float(score["global_similarity"]), 6)
                    ),
                    "global_max_similarity": (
                        None
                        if score["global_max_similarity"] is None
                        else round(float(score["global_max_similarity"]), 6)
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

                    "clothing_similarity": (
                        None
                        if score["clothing_similarity"] is None
                        else round(float(score["clothing_similarity"]), 6)
                    ),
                    "clothing_upper_min_similarity": (
                        None
                        if score["clothing_upper_min_similarity"] is None
                        else round(
                            float(score["clothing_upper_min_similarity"]),
                            6,
                        )
                    ),
                    "clothing_gate_applicable": bool(
                        score["clothing_gate_applicable"]
                    ),
                    "clothing_detail": score["clothing_detail"],
                    "general_clothing_veto": decision[
                        "general_clothing_veto"
                    ],
                    "strong_clothing_veto": decision[
                        "strong_clothing_veto"
                    ],
                    "short_clothing_veto": decision[
                        "short_clothing_veto"
                    ],
                    "long_clothing_veto": decision[
                        "long_clothing_veto"
                    ],

                    "general_pass": decision["general_pass"],
                    "strong_global_pass": decision["strong_global_pass"],
                    "short_gap_pass": decision["short_gap_pass"],
                    "long_gap_reentry_candidate": decision["long_gap_reentry_candidate"],
                    "long_gap_reentry_pass": decision["long_gap_reentry_pass"],
                    "scale_ok": decision["scale_ok"],
                    "long_scale_ok": decision["long_scale_ok"],
                    "accepted_rule": decision["accepted_rule"],
                    "accepted": decision["accepted"],
                    "would_accept_with_current_thresholds": would_accept,
                    "selected_after_margin": False,
                    "margin_rejected": False,
                    "identity_margin": None,
                    "calibration_only": bool(calibration_only),
                }

                candidate_log.append(log_row)
                log_index = len(candidate_log) - 1

                if not decision["accepted"]:
                    continue

                selection_key = (
                    float(score["fused_score"]),
                    float(score.get("global_similarity") or -1.0),
                    -float(gap_sec),
                )

                candidate = {
                    "identity": ident,
                    "predecessor": predecessor,
                    "score": score,
                    "decision": decision,
                    "gap_sec": gap_sec,
                    "selection_key": selection_key,
                    "log_index": log_index,
                }

                person_id = int(ident["person_id"])
                current = accepted_by_identity.get(person_id)

                if (
                    current is None
                    or selection_key > current["selection_key"]
                ):
                    accepted_by_identity[person_id] = candidate

        identity_options = sorted(
            accepted_by_identity.values(),
            key=lambda x: x["selection_key"],
            reverse=True,
        )

        best = identity_options[0] if identity_options else None
        margin_value = None
        rejection_reason = None

        if best is not None and len(identity_options) >= 2:
            first_score = float(best["score"]["fused_score"])
            second_score = float(
                identity_options[1]["score"]["fused_score"]
            )
            margin_value = first_score - second_score

            if margin_value < identity_margin:
                rejection_reason = "identity_margin"
                candidate_log[best["log_index"]]["margin_rejected"] = True
                candidate_log[best["log_index"]]["identity_margin"] = round(
                    float(margin_value),
                    6,
                )
                best = None

        if best is None:
            create_identity(
                tracklet,
                rejection_reason=rejection_reason,
                margin_value=margin_value,
            )
            continue

        candidate_log[best["log_index"]]["selected_after_margin"] = True
        candidate_log[best["log_index"]]["identity_margin"] = (
            None
            if margin_value is None
            else round(float(margin_value), 6)
        )

        # ----------------------------------------------------
        # Merge into existing identity
        # ----------------------------------------------------
        ident = best["identity"]
        predecessor = best["predecessor"]
        score = best["score"]
        decision = best["decision"]

        trusted_update, trusted_reason = trusted_gallery_decision(
            score,
            float(best["gap_sec"]),
            decision["accepted_rule"],
            trusted_max_gap_sec=trusted_max_gap_sec,
            trusted_min_fused=trusted_min_fused,
            trusted_min_endpoint_topk=trusted_min_endpoint_topk,
            trusted_min_endpoint_max=trusted_min_endpoint_max,
            trusted_min_global=trusted_min_global,
            trusted_min_gallery_global=trusted_min_gallery_global,
            trusted_min_anchor=trusted_min_anchor,
            trusted_long_min_fused=trusted_long_min_fused,
            trusted_long_min_endpoint_topk=trusted_long_min_endpoint_topk,
            trusted_long_min_endpoint_max=trusted_long_min_endpoint_max,
            trusted_long_min_global=trusted_long_min_global,
            trusted_long_min_gallery_global=trusted_long_min_gallery_global,
            trusted_long_min_anchor=trusted_long_min_anchor,
        )

        # Every accepted tracklet belongs to the identity, but only a trusted
        # merge may alter future identity evidence. This prevents a weak
        # false merge from contaminating the centroid/gallery chain.
        ident["track_ids"].append(
            tracklet["track_id"]
        )

        if trusted_update:
            old_weight = float(ident["global_weight"])
            new_weight = float(
                min(
                    max(1, tracklet["observations"]),
                    identity_update_weight_cap,
                )
            )

            ident["global_embedding"] = normalize(
                ident["global_embedding"] * old_weight
                + tracklet["global_embedding"] * new_weight
            )
            ident["global_weight"] = old_weight + new_weight

            if (
                ident.get("clothing_profile") is not None
                and tracklet.get("clothing_profile") is not None
                and ident.get("clothing_reliable", False)
                and tracklet.get("clothing_reliable", False)
            ):
                old_clothing_weight = float(
                    ident.get("clothing_weight", 1.0)
                )
                new_clothing_weight = float(
                    min(
                        max(
                            1,
                            tracklet.get("clothing_crop_count", 0),
                        ),
                        identity_update_weight_cap,
                    )
                )

                ident["clothing_profile"] = blend_clothing_profiles(
                    ident["clothing_profile"],
                    old_clothing_weight,
                    tracklet["clothing_profile"],
                    new_clothing_weight,
                )
                ident["clothing_weight"] = (
                    old_clothing_weight + new_clothing_weight
                )

            ident["gallery"].append(
                gallery_entry(tracklet)
            )
            ident["gallery"] = trim_gallery(
                ident["gallery"],
                gallery_size,
            )
            ident["trusted_track_ids"].append(tracklet["track_id"])

        # anchor_embedding intentionally remains fixed.

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
            "person_id": ident["person_id"],
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
            "stitch_global_max_similarity": (
                None
                if score["global_max_similarity"] is None
                else round(float(score["global_max_similarity"]), 6)
            ),
            "stitch_anchor_global_similarity": (
                None
                if score["anchor_global_similarity"] is None
                else round(float(score["anchor_global_similarity"]), 6)
            ),
            "stitch_gallery_global_topk_mean": (
                None
                if score["gallery_global_topk_mean"] is None
                else round(
                    float(score["gallery_global_topk_mean"]),
                    6,
                )
            ),
            "stitch_clothing_similarity": (
                None
                if score["clothing_similarity"] is None
                else round(float(score["clothing_similarity"]), 6)
            ),
            "stitch_clothing_upper_min_similarity": (
                None
                if score["clothing_upper_min_similarity"] is None
                else round(
                    float(score["clothing_upper_min_similarity"]),
                    6,
                )
            ),
            "stitch_clothing_gate_applicable": bool(
                score["clothing_gate_applicable"]
            ),
            "stitch_clothing_veto": False,
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
            "stitch_identity_margin": (
                None
                if margin_value is None
                else round(float(margin_value), 6)
            ),
            "stitch_rejection_reason": None,
            "stitch_trusted_gallery_update": bool(trusted_update),
            "stitch_trusted_gallery_reason": trusted_reason,
        }

    # Tracklets without usable embeddings stay independent.
    for track_id in sorted(tracklets):
        if track_id in assignment:
            continue

        t = tracklets[track_id]

        assignment[track_id] = {
            "person_id": next_person_id,
            "stitched_from_track_id": None,
            "stitch_rule": None,
            "stitch_endpoint_topk_mean": None,
            "stitch_endpoint_max": None,
            "stitch_global_similarity": None,
            "stitch_global_max_similarity": None,
            "stitch_anchor_global_similarity": None,
            "stitch_gallery_global_topk_mean": None,
            "stitch_clothing_similarity": None,
            "stitch_clothing_upper_min_similarity": None,
            "stitch_clothing_gate_applicable": False,
            "stitch_clothing_veto": False,
            "stitch_fused_score": None,
            "stitch_gap_sec": None,
            "stitch_position_score": None,
            "stitch_bbox_scale_ratio": None,
            "stitch_identity_margin": None,
            "stitch_rejection_reason": "missing_embedding",
            "stitch_trusted_gallery_update": False,
            "stitch_trusted_gallery_reason": "missing_embedding",
        }

        identities.append({
            "person_id": next_person_id,
            "class_name": t["class_name"],
            "track_ids": [track_id],
            "global_embedding": t["global_embedding"],
            "global_weight": 0.0,
            "anchor_embedding": (
                None
                if t["global_embedding"] is None
                else t["global_embedding"].copy()
            ),
            "clothing_profile": copy_clothing_profile(
                t.get("clothing_profile")
            ),
            "clothing_weight": 0.0,
            "clothing_reliable": bool(
                t.get("clothing_reliable", False)
            ),
            "anchor_clothing_profile": copy_clothing_profile(
                t.get("clothing_profile")
            ),
            "anchor_clothing_reliable": bool(
                t.get("clothing_reliable", False)
            ),
            "gallery": (
                [gallery_entry(t)]
                if t["end_vectors"] is not None
                else []
            ),
            "trusted_track_ids": (
                [track_id]
                if t["end_vectors"] is not None
                else []
            ),
            "last_track_id": track_id,
            "start_sec": t["start_sec"],
            "last_end_sec": t["end_sec"],
            "start_frame": t["start_frame"],
            "last_end_frame": t["end_frame"],
        })

        next_person_id += 1

    identities.sort(
        key=lambda x: x["person_id"]
    )

    return identities, assignment, candidate_log


# ============================================================
# Output V4.5
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
        out["person_id"] = int(info["person_id"])
        out["stitched_id"] = (
            f"person_{int(info['person_id']):04d}"
        )

        out["class_name"] = tracklets[track_id]["class_name"]
        out["final_class_name"] = tracklets[track_id]["class_name"]

        out["stitch_model"] = "solider_gallery_consensus_v4_5"
        out["stitched_from_track_id"] = info["stitched_from_track_id"]
        out["stitch_rule"] = info["stitch_rule"]
        out["stitch_endpoint_topk_mean"] = info["stitch_endpoint_topk_mean"]
        out["stitch_endpoint_max"] = info["stitch_endpoint_max"]
        out["stitch_global_similarity"] = info["stitch_global_similarity"]
        out["stitch_global_max_similarity"] = info.get(
            "stitch_global_max_similarity"
        )
        out["stitch_anchor_global_similarity"] = info.get(
            "stitch_anchor_global_similarity"
        )
        out["stitch_gallery_global_topk_mean"] = (
            info["stitch_gallery_global_topk_mean"]
        )
        out["stitch_clothing_similarity"] = info.get(
            "stitch_clothing_similarity"
        )
        out["stitch_clothing_upper_min_similarity"] = info.get(
            "stitch_clothing_upper_min_similarity"
        )
        out["stitch_clothing_gate_applicable"] = info.get(
            "stitch_clothing_gate_applicable", False
        )
        out["stitch_clothing_veto"] = info.get(
            "stitch_clothing_veto", False
        )
        out["stitch_fused_score"] = info["stitch_fused_score"]
        out["stitch_gap_sec"] = info["stitch_gap_sec"]
        out["stitch_position_score"] = info["stitch_position_score"]
        out["stitch_bbox_scale_ratio"] = info["stitch_bbox_scale_ratio"]
        out["stitch_identity_margin"] = info.get("stitch_identity_margin")
        out["stitch_rejection_reason"] = info.get("stitch_rejection_reason")
        out["stitch_trusted_gallery_update"] = info.get(
            "stitch_trusted_gallery_update", False
        )
        out["stitch_trusted_gallery_reason"] = info.get(
            "stitch_trusted_gallery_reason"
        )

        stitched_rows.append(out)

    stitched_path = (
        video_dir / "stitched_tracks_v4_5.jsonl"
    )
    write_jsonl(stitched_path, stitched_rows)

    candidates_path = (
        video_dir / "stitch_candidates_v4_5.jsonl"
    )
    write_jsonl(candidates_path, candidate_log)

    crop_error_path = (
        video_dir / "stitch_crop_errors_v4_5.jsonl"
    )
    write_jsonl(crop_error_path, crop_errors)

    rule_counts = Counter(
        info["stitch_rule"]
        for info in assignment.values()
        if info["stitch_rule"] is not None
    )

    summary = {
        "scope": "person",
        "stitch_model": "solider_gallery_consensus_v4_5",
        "calibration_only": bool(args.calibration_only),

        "global_top_k": args.global_top_k,
        "endpoint_k": args.endpoint_k,
        "pair_top_k": args.pair_top_k,
        "gallery_size": args.gallery_size,
        "gallery_global_top_k": args.gallery_global_top_k,
        "max_gap_sec": args.max_gap_sec,

        "rules": {
            "general": {
                "fused_threshold": args.fused_threshold,
                "max_gap_sec": args.general_max_gap_sec,
                "min_endpoint_topk": args.general_min_endpoint_topk,
                "min_endpoint_max": args.general_min_endpoint_max,
                "min_global_consensus": args.general_min_global,
                "min_gallery_global": args.general_min_gallery_global,
                "min_anchor": args.general_min_anchor,
                "clothing_veto": {
                    "min_similarity": args.general_min_clothing,
                    "min_upper_zone": args.general_min_clothing_upper,
                    "logic": "veto_if_both_below",
                },
            },
            "strong_global": {
                "global_threshold": args.strong_global_threshold,
                "max_gap_sec": args.strong_global_max_gap_sec,
                "min_endpoint_topk": args.strong_global_min_endpoint_topk,
                "min_endpoint_max": args.strong_global_min_endpoint_max,
                "min_gallery_global": args.strong_global_min_gallery_global,
                "min_anchor": args.strong_global_min_anchor,
                "min_position": args.strong_global_min_position,
                "clothing_veto": {
                    "min_similarity": args.strong_min_clothing,
                    "min_upper_zone": args.strong_min_clothing_upper,
                    "logic": "veto_if_both_below",
                },
            },
            "short_gap": {
                "max_gap_sec": args.short_gap_sec,
                "min_position": args.short_min_position,
                "min_endpoint_topk": args.short_min_endpoint_topk,
                "min_endpoint_max": args.short_min_endpoint_max,
                "min_global_consensus": args.short_min_global,
                "min_anchor": args.short_min_anchor,
                "clothing_veto": {
                    "min_similarity": args.short_min_clothing,
                    "min_upper_zone": args.short_min_clothing_upper,
                    "logic": "veto_if_both_below",
                },
            },
            "long_gap_reentry": {
                "auto_merge_enabled": bool(args.allow_long_gap_auto_merge),
                "policy": (
                    "experimental_auto_merge"
                    if args.allow_long_gap_auto_merge
                    else "candidate_only_no_auto_merge"
                ),
                "min_gap_sec": args.long_gap_min_sec,
                "max_gap_sec": args.long_gap_max_sec,
                "min_endpoint_topk": args.long_min_endpoint_topk,
                "min_endpoint_max": args.long_min_endpoint_max,
                "min_global_consensus": args.long_min_global,
                "min_gallery_global": args.long_min_gallery_global,
                "min_anchor": args.long_min_anchor,
                "min_fused": args.long_min_fused,
                "max_scale_ratio": args.long_max_scale_ratio,
                "clothing_veto": {
                    "min_similarity": args.long_min_clothing,
                    "min_upper_zone": args.long_min_clothing_upper,
                    "logic": "veto_if_both_below",
                },
            },
            "clothing_profile": {
                "min_crops": args.clothing_min_crops,
                "zones": CLOTHING_ZONES,
                "zone_weights": CLOTHING_ZONE_WEIGHTS,
                "descriptor": "HSV_joint_histogram_12x4x4",
            },
            "identity_margin": args.identity_margin,
            "identity_update_weight_cap": args.identity_update_weight_cap,
            "trusted_gallery": {
                "max_gap_sec": args.trusted_max_gap_sec,
                "min_fused": args.trusted_min_fused,
                "min_endpoint_topk": args.trusted_min_endpoint_topk,
                "min_endpoint_max": args.trusted_min_endpoint_max,
                "min_global_consensus": args.trusted_min_global,
                "min_gallery_global": args.trusted_min_gallery_global,
                "min_anchor": args.trusted_min_anchor,
                "long_min_fused": args.trusted_long_min_fused,
                "long_min_endpoint_topk": args.trusted_long_min_endpoint_topk,
                "long_min_endpoint_max": args.trusted_long_min_endpoint_max,
                "long_min_global_consensus": args.trusted_long_min_global,
                "long_min_gallery_global": args.trusted_long_min_gallery_global,
                "long_min_anchor": args.trusted_long_min_anchor,
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
        "output_person_ids": len(identities),
        "output_rows": len(stitched_rows),
        "skipped_output_rows": skipped_output_rows,
        "crop_errors": len(crop_errors),
        "merge_rule_counts": dict(rule_counts),
        "persons": [],
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
        person_id = int(ident["person_id"])
        representative_crops = []

        for track_id in ident["track_ids"]:
            representative_crops.extend(
                tracklets[track_id]["global_crops"]
            )

        summary["persons"].append({
            "person_id": person_id,
            "stitched_id": f"person_{person_id:04d}",
            "final_class_name": ident["class_name"],
            "track_ids": ident["track_ids"],
            "start_sec": round(float(ident["start_sec"]), 4),
            "end_sec": round(float(ident["last_end_sec"]), 4),
            "gallery_track_ids": [
                int(x["track_id"])
                for x in ident.get("gallery", [])
            ],
            "trusted_track_ids": [
                int(x) for x in ident.get("trusted_track_ids", [])
            ],
            "anchor_track_id": (
                int(ident["track_ids"][0])
                if ident.get("track_ids")
                else None
            ),
            "representative_crops": representative_crops,
        })

        if ident["global_embedding"] is not None:
            npz_data[
                f"person_{person_id}_global"
            ] = np.asarray(
                ident["global_embedding"],
                dtype=np.float32,
            )

    summary_path = (
        video_dir / "stitching_v4_5.json"
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
        video_dir / "stitch_embeddings_solider_v4_5.npz"
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
        if p.is_file() and p.stat().st_size > 0
    )


def save_video_error(
    video_dir: Path,
    exc: Exception,
) -> Path:
    error_path = (
        video_dir / "stitch_error_v4_5.json"
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
            "Person Tracklet Stitching V4.5: "
            "SOLIDER identity gallery + pairwise endpoint "
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
        "--calibration-only",
        action="store_true",
        help=(
            "compute SOLIDER/endpoint/time/position/scale candidate scores "
            "but do not merge different ByteTrack tracklets"
        ),
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

    # A. General merge - conservative hard gates
    ap.add_argument(
        "--fused-threshold",
        type=float,
        default=0.84,
    )
    ap.add_argument(
        "--general-max-gap-sec",
        type=float,
        default=6.0,
    )
    ap.add_argument(
        "--general-min-endpoint-topk",
        type=float,
        default=0.90,
    )
    ap.add_argument(
        "--general-min-endpoint-max",
        type=float,
        default=0.92,
    )
    ap.add_argument(
        "--general-min-global",
        type=float,
        default=0.87,
    )
    ap.add_argument(
        "--general-min-gallery-global",
        type=float,
        default=0.90,
    )
    ap.add_argument(
        "--general-min-anchor",
        type=float,
        default=0.87,
    )

    # B. Strong-global fallback
    ap.add_argument(
        "--strong-global-threshold",
        type=float,
        default=0.95,
    )
    ap.add_argument(
        "--strong-global-max-gap-sec",
        type=float,
        default=3.0,
    )
    ap.add_argument(
        "--strong-global-min-endpoint-topk",
        type=float,
        default=0.88,
    )
    ap.add_argument(
        "--strong-global-min-endpoint-max",
        type=float,
        default=0.93,
    )
    ap.add_argument(
        "--strong-global-min-gallery-global",
        type=float,
        default=0.95,
    )
    ap.add_argument(
        "--strong-global-min-anchor",
        type=float,
        default=0.93,
    )
    ap.add_argument(
        "--strong-global-min-position",
        type=float,
        default=0.92,
    )

    # C. Short-gap continuity fallback
    ap.add_argument(
        "--short-gap-sec",
        type=float,
        default=1.5,
    )
    ap.add_argument(
        "--short-min-position",
        type=float,
        default=0.92,
    )
    ap.add_argument(
        "--short-min-endpoint-topk",
        type=float,
        default=0.90,
    )
    ap.add_argument(
        "--short-min-endpoint-max",
        type=float,
        default=0.93,
    )
    ap.add_argument(
        "--short-min-global",
        type=float,
        default=0.88,
    )
    ap.add_argument(
        "--short-min-anchor",
        type=float,
        default=0.88,
    )

    # D. Long-gap re-entry - strict
    ap.add_argument(
        "--long-gap-min-sec",
        type=float,
        default=8.0,
    )
    ap.add_argument(
        "--long-gap-max-sec",
        type=float,
        default=20.0,
    )
    ap.add_argument(
        "--long-min-endpoint-topk",
        type=float,
        default=0.90,
    )
    ap.add_argument(
        "--long-min-endpoint-max",
        type=float,
        default=0.94,
    )
    ap.add_argument(
        "--long-min-global",
        type=float,
        default=0.92,
    )
    ap.add_argument(
        "--long-min-gallery-global",
        type=float,
        default=0.92,
    )
    ap.add_argument(
        "--long-min-anchor",
        type=float,
        default=0.92,
    )
    ap.add_argument(
        "--long-min-fused",
        type=float,
        default=0.88,
    )
    ap.add_argument(
        "--long-max-scale-ratio",
        type=float,
        default=1.8,
    )
    ap.add_argument(
        "--allow-long-gap-auto-merge",
        action="store_true",
        help=(
            "EXPERIMENTAL: allow 8-20s person re-entry auto-merges. "
            "Default is OFF because validation found false merges even at "
            "very high SOLIDER similarity."
        ),
    )

    # V4.5 independent clothing appearance veto.
    ap.add_argument("--clothing-min-crops", type=int, default=3)
    ap.add_argument("--general-min-clothing", type=float, default=0.60)
    ap.add_argument("--general-min-clothing-upper", type=float, default=0.45)
    ap.add_argument("--strong-min-clothing", type=float, default=0.62)
    ap.add_argument("--strong-min-clothing-upper", type=float, default=0.45)
    ap.add_argument("--short-min-clothing", type=float, default=0.58)
    ap.add_argument("--short-min-clothing-upper", type=float, default=0.43)
    ap.add_argument("--long-min-clothing", type=float, default=0.64)
    ap.add_argument("--long-min-clothing-upper", type=float, default=0.48)

    # Identity-level safeguards
    ap.add_argument(
        "--identity-margin",
        type=float,
        default=0.05,
        help=(
            "best and runner-up accepted identity fused-score margin; "
            "smaller margins are rejected as ambiguous"
        ),
    )
    ap.add_argument(
        "--identity-update-weight-cap",
        type=float,
        default=10.0,
        help="cap per-track contribution to identity centroid to reduce drift",
    )

    # V4.5 trusted-gallery update gates. These do NOT decide identity
    # assignment; they decide whether an accepted merge is allowed to
    # influence future centroid/gallery evidence.
    ap.add_argument("--trusted-max-gap-sec", type=float, default=8.0)
    ap.add_argument("--trusted-min-fused", type=float, default=0.86)
    ap.add_argument("--trusted-min-endpoint-topk", type=float, default=0.90)
    ap.add_argument("--trusted-min-endpoint-max", type=float, default=0.92)
    ap.add_argument("--trusted-min-global", type=float, default=0.93)
    ap.add_argument("--trusted-min-gallery-global", type=float, default=0.93)
    ap.add_argument("--trusted-min-anchor", type=float, default=0.93)

    ap.add_argument("--trusted-long-min-fused", type=float, default=0.87)
    ap.add_argument("--trusted-long-min-endpoint-topk", type=float, default=0.93)
    ap.add_argument("--trusted-long-min-endpoint-max", type=float, default=0.95)
    ap.add_argument("--trusted-long-min-global", type=float, default=0.95)
    ap.add_argument("--trusted-long-min-gallery-global", type=float, default=0.95)
    ap.add_argument("--trusted-long-min-anchor", type=float, default=0.95)

    ap.add_argument(
        "--max-scale-ratio",
        type=float,
        default=2.0,
        help="general/fallback rules bbox diagonal scale ratio limit",
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
        "clothing_min_crops": args.clothing_min_crops,
    }

    for name, value in positive_int_args.items():
        if value <= 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be >= 1"
            )

    nonnegative_args = {
        "max_gap_sec": args.max_gap_sec,
        "general_max_gap_sec": args.general_max_gap_sec,
        "strong_global_max_gap_sec": args.strong_global_max_gap_sec,
        "short_gap_sec": args.short_gap_sec,
        "long_gap_min_sec": args.long_gap_min_sec,
        "long_gap_max_sec": args.long_gap_max_sec,
        "identity_margin": args.identity_margin,
        "identity_update_weight_cap": args.identity_update_weight_cap,
        "trusted_max_gap_sec": args.trusted_max_gap_sec,
    }

    for name, value in nonnegative_args.items():
        if value < 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be >= 0"
            )

    similarity_args = {
        "fused_threshold": args.fused_threshold,
        "general_min_endpoint_topk": args.general_min_endpoint_topk,
        "general_min_endpoint_max": args.general_min_endpoint_max,
        "general_min_global": args.general_min_global,
        "general_min_gallery_global": args.general_min_gallery_global,
        "general_min_anchor": args.general_min_anchor,
        "strong_global_threshold": args.strong_global_threshold,
        "strong_global_min_endpoint_topk": args.strong_global_min_endpoint_topk,
        "strong_global_min_endpoint_max": args.strong_global_min_endpoint_max,
        "strong_global_min_gallery_global": args.strong_global_min_gallery_global,
        "strong_global_min_anchor": args.strong_global_min_anchor,
        "strong_global_min_position": args.strong_global_min_position,
        "short_min_position": args.short_min_position,
        "short_min_endpoint_topk": args.short_min_endpoint_topk,
        "short_min_endpoint_max": args.short_min_endpoint_max,
        "short_min_global": args.short_min_global,
        "short_min_anchor": args.short_min_anchor,
        "long_min_endpoint_topk": args.long_min_endpoint_topk,
        "long_min_endpoint_max": args.long_min_endpoint_max,
        "long_min_global": args.long_min_global,
        "long_min_gallery_global": args.long_min_gallery_global,
        "long_min_anchor": args.long_min_anchor,
        "long_min_fused": args.long_min_fused,
        "trusted_min_fused": args.trusted_min_fused,
        "trusted_min_endpoint_topk": args.trusted_min_endpoint_topk,
        "trusted_min_endpoint_max": args.trusted_min_endpoint_max,
        "trusted_min_global": args.trusted_min_global,
        "trusted_min_gallery_global": args.trusted_min_gallery_global,
        "trusted_min_anchor": args.trusted_min_anchor,
        "trusted_long_min_fused": args.trusted_long_min_fused,
        "trusted_long_min_endpoint_topk": args.trusted_long_min_endpoint_topk,
        "trusted_long_min_endpoint_max": args.trusted_long_min_endpoint_max,
        "trusted_long_min_global": args.trusted_long_min_global,
        "trusted_long_min_gallery_global": args.trusted_long_min_gallery_global,
        "trusted_long_min_anchor": args.trusted_long_min_anchor,
        "general_min_clothing": args.general_min_clothing,
        "general_min_clothing_upper": args.general_min_clothing_upper,
        "strong_min_clothing": args.strong_min_clothing,
        "strong_min_clothing_upper": args.strong_min_clothing_upper,
        "short_min_clothing": args.short_min_clothing,
        "short_min_clothing_upper": args.short_min_clothing_upper,
        "long_min_clothing": args.long_min_clothing,
        "long_min_clothing_upper": args.long_min_clothing_upper,
    }

    for name, value in similarity_args.items():
        if not (-1.0 <= value <= 1.0):
            raise ValueError(
                f"--{name.replace('_', '-')} must be in [-1, 1]"
            )

    if args.long_gap_min_sec > args.long_gap_max_sec:
        raise ValueError(
            "--long-gap-min-sec must be <= --long-gap-max-sec"
        )

    if args.general_max_gap_sec > args.max_gap_sec:
        raise ValueError(
            "--general-max-gap-sec must be <= --max-gap-sec"
        )

    if args.long_gap_max_sec > args.max_gap_sec:
        raise ValueError(
            "--long-gap-max-sec must be <= --max-gap-sec"
        )

    if args.identity_update_weight_cap <= 0:
        raise ValueError(
            "--identity-update-weight-cap must be > 0"
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
        wanted = str(args.video).strip().lower()
        video_dirs = [
            p for p in video_dirs
            if p.name.lower() == wanted
            or (p.name + ".mp4").lower() == wanted
            or (p.name + ".avi").lower() == wanted
            or (p.name + ".mov").lower() == wanted
            or (p.name + ".mkv").lower() == wanted
        ]

    if args.max_videos is not None:
        if args.max_videos <= 0:
            raise ValueError(
                "--max-videos must be >= 1"
            )
        video_dirs = video_dirs[:args.max_videos]

    print("=" * 96)
    print(
        "PERSON TRACKLET STITCHING V4.5 "
        "- CONSENSUS GLOBAL + FIXED ANCHOR + IDENTITY MARGIN + STRICT LONG-GAP"
    )
    print("=" * 96)

    print("track root        :", TRACK_ROOT)
    print("videos            :", len(video_dirs))
    print("calibration only  :", args.calibration_only)
    print("global top-k      :", args.global_top_k)
    print("endpoint k        :", args.endpoint_k)
    print("pair top-k        :", args.pair_top_k)
    print("gallery size      :", args.gallery_size)
    print("gallery global k  :", args.gallery_global_top_k)
    print("max gap sec       :", args.max_gap_sec)
    print(
        "general           :",
        {
            "fused": args.fused_threshold,
            "max_gap": args.general_max_gap_sec,
            "endpoint_topk": args.general_min_endpoint_topk,
            "endpoint_max": args.general_min_endpoint_max,
            "global_consensus": args.general_min_global,
            "gallery_global": args.general_min_gallery_global,
            "anchor": args.general_min_anchor,
        },
    )
    print(
        "strong global     :",
        {
            "global": args.strong_global_threshold,
            "max_gap": args.strong_global_max_gap_sec,
            "endpoint_topk": args.strong_global_min_endpoint_topk,
            "endpoint_max": args.strong_global_min_endpoint_max,
            "gallery_global": args.strong_global_min_gallery_global,
            "anchor": args.strong_global_min_anchor,
            "position": args.strong_global_min_position,
        },
    )
    print(
        "short gap         :",
        {
            "max_gap": args.short_gap_sec,
            "position": args.short_min_position,
            "endpoint_topk": args.short_min_endpoint_topk,
            "endpoint_max": args.short_min_endpoint_max,
            "global": args.short_min_global,
            "anchor": args.short_min_anchor,
        },
    )
    print(
        "long-gap policy   :",
        (
            "EXPERIMENTAL AUTO-MERGE ENABLED"
            if args.allow_long_gap_auto_merge
            else "CANDIDATE-ONLY / AUTO-MERGE DISABLED"
        ),
    )
    print(
        "long-gap reentry  :",
        {
            "gap": [args.long_gap_min_sec, args.long_gap_max_sec],
            "endpoint_topk": args.long_min_endpoint_topk,
            "endpoint_max": args.long_min_endpoint_max,
            "global": args.long_min_global,
            "gallery_global": args.long_min_gallery_global,
            "anchor": args.long_min_anchor,
            "fused": args.long_min_fused,
            "scale": args.long_max_scale_ratio,
        },
    )
    print(
        "clothing veto     :",
        {
            "min_crops": args.clothing_min_crops,
            "general": [
                args.general_min_clothing,
                args.general_min_clothing_upper,
            ],
            "strong": [
                args.strong_min_clothing,
                args.strong_min_clothing_upper,
            ],
            "short": [
                args.short_min_clothing,
                args.short_min_clothing_upper,
            ],
            "long": [
                args.long_min_clothing,
                args.long_min_clothing_upper,
            ],
            "logic": "veto only if overall AND upper-zone are both low",
        },
    )
    print("identity margin   :", args.identity_margin)
    print("centroid weight cap:", args.identity_update_weight_cap)
    print(
        "trusted gallery   :",
        {
            "standard": {
                "max_gap": args.trusted_max_gap_sec,
                "fused": args.trusted_min_fused,
                "endpoint_topk": args.trusted_min_endpoint_topk,
                "endpoint_max": args.trusted_min_endpoint_max,
                "global": args.trusted_min_global,
                "gallery": args.trusted_min_gallery_global,
                "anchor": args.trusted_min_anchor,
            },
            "long": {
                "fused": args.trusted_long_min_fused,
                "endpoint_topk": args.trusted_long_min_endpoint_topk,
                "endpoint_max": args.trusted_long_min_endpoint_max,
                "global": args.trusted_long_min_global,
                "gallery": args.trusted_long_min_gallery_global,
                "anchor": args.trusted_long_min_anchor,
            },
        },
    )
    print("max scale ratio   :", args.max_scale_ratio)
    print(
        "threshold status  : PROVISIONAL V4.5 - clothing veto + long-gap auto-merge OFF"
    )
    if args.calibration_only:
        print(
            "mode note         : no cross-track person merge will be committed"
        )
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
        print("No non-empty person tracks.jsonl found.")
        return

    cfg = PipelineConfig.load(CONFIG_PATH)
    embedder, spec = load_embedder(
        cfg,
        "solider",
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

            clothing_errors = compute_tracklet_clothing_profiles(
                tracklets,
                min_crops=args.clothing_min_crops,
            )
            crop_errors.extend(clothing_errors)

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
                general_max_gap_sec=args.general_max_gap_sec,
                general_min_endpoint_topk=args.general_min_endpoint_topk,
                general_min_endpoint_max=args.general_min_endpoint_max,
                general_min_global=args.general_min_global,
                general_min_gallery_global=args.general_min_gallery_global,
                general_min_anchor=args.general_min_anchor,
                strong_global_threshold=args.strong_global_threshold,
                strong_global_max_gap_sec=args.strong_global_max_gap_sec,
                strong_global_min_endpoint_topk=args.strong_global_min_endpoint_topk,
                strong_global_min_endpoint_max=args.strong_global_min_endpoint_max,
                strong_global_min_gallery_global=args.strong_global_min_gallery_global,
                strong_global_min_anchor=args.strong_global_min_anchor,
                strong_global_min_position=args.strong_global_min_position,
                short_gap_sec=args.short_gap_sec,
                short_min_position=args.short_min_position,
                short_min_endpoint_topk=args.short_min_endpoint_topk,
                short_min_endpoint_max=args.short_min_endpoint_max,
                short_min_global=args.short_min_global,
                short_min_anchor=args.short_min_anchor,
                long_gap_min_sec=args.long_gap_min_sec,
                long_gap_max_sec=args.long_gap_max_sec,
                long_min_endpoint_topk=args.long_min_endpoint_topk,
                long_min_endpoint_max=args.long_min_endpoint_max,
                long_min_global=args.long_min_global,
                long_min_gallery_global=args.long_min_gallery_global,
                long_min_anchor=args.long_min_anchor,
                long_min_fused=args.long_min_fused,
                long_max_scale_ratio=args.long_max_scale_ratio,
                max_scale_ratio=args.max_scale_ratio,
                allow_long_gap_auto_merge=args.allow_long_gap_auto_merge,
                general_min_clothing=args.general_min_clothing,
                general_min_clothing_upper=args.general_min_clothing_upper,
                strong_min_clothing=args.strong_min_clothing,
                strong_min_clothing_upper=args.strong_min_clothing_upper,
                short_min_clothing=args.short_min_clothing,
                short_min_clothing_upper=args.short_min_clothing_upper,
                long_min_clothing=args.long_min_clothing,
                long_min_clothing_upper=args.long_min_clothing_upper,
                identity_margin=args.identity_margin,
                identity_update_weight_cap=args.identity_update_weight_cap,
                trusted_max_gap_sec=args.trusted_max_gap_sec,
                trusted_min_fused=args.trusted_min_fused,
                trusted_min_endpoint_topk=args.trusted_min_endpoint_topk,
                trusted_min_endpoint_max=args.trusted_min_endpoint_max,
                trusted_min_global=args.trusted_min_global,
                trusted_min_gallery_global=args.trusted_min_gallery_global,
                trusted_min_anchor=args.trusted_min_anchor,
                trusted_long_min_fused=args.trusted_long_min_fused,
                trusted_long_min_endpoint_topk=args.trusted_long_min_endpoint_topk,
                trusted_long_min_endpoint_max=args.trusted_long_min_endpoint_max,
                trusted_long_min_global=args.trusted_long_min_global,
                trusted_long_min_gallery_global=args.trusted_long_min_gallery_global,
                trusted_long_min_anchor=args.trusted_long_min_anchor,
                calibration_only=args.calibration_only,
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
            print("  output persons  :", len(identities))
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
                    o["person_id"]: {
                        "class": o["class_name"],
                        "tracks": o["track_ids"],
                    }
                    for o in identities
                },
            )
            print(
                "  trusted updates :",
                sum(
                    1 for x in assignment.values()
                    if x.get("stitch_rule") is not None
                    and x.get("stitch_trusted_gallery_update")
                ),
            )
            print(
                "  clothing vetoes :",
                sum(
                    1 for x in candidate_log
                    if (
                        x.get("general_clothing_veto")
                        or x.get("strong_clothing_veto")
                        or x.get("short_clothing_veto")
                        or x.get("long_clothing_veto")
                    )
                ),
            )
            print(
                "  clothing ready  :",
                sum(
                    1 for t in tracklets.values()
                    if t.get("clothing_reliable")
                ),
                "/",
                len(tracklets),
            )
            print("  crop errors     :", len(crop_errors))
            print("  stitched rows   :", stitched_path)
            print("  stitching json  :", summary_path)
            print("  candidates      :", candidates_path)
            print("  crop error log  :", crop_error_path)
            print("  SOLIDER cache    :", embedding_path)

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
    print("STITCH V4.5 COMPLETE")
    print("passed :", passed)
    print("failed :", failed)
    print("=" * 96)


if __name__ == "__main__":
    main()