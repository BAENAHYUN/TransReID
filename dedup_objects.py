from __future__ import annotations

"""
dedup_objects.py

RF-DETR object duplicate handling for the forensic retrieval project.

Hybrid C policy
---------------
This file deliberately separates hard deduplication from semantic duplicate
judgement.

1) Hard dedup
   - Only an exactly duplicated detection_id is physically removed.
   - The highest-confidence record survives so Qdrant point IDs cannot collide.

2) Soft dedup
   - Geometry + DINOv2 may judge two OBJECT detections to represent the same
     physical object.
   - Those detections are NEVER removed by the semantic judgement.
   - Every member remains an independent crop / embedding / Qdrant point and
     receives the same stable duplicate_group_id.

3) Ambiguous relation
   - Uncertain pairs are preserved and receive ambiguous_group_id metadata.
   - They are not collapsed automatically at search time.

4) Person safety
   - Person detections never participate in bbox/DINO semantic dedup.
   - Only an exact duplicated detection_id can be hard-deduplicated.

Input metadata contract
-----------------------
Expected crop records contain at least:
    image_id
    detection_id
    class_name
    confidence
    bbox = [x1, y1, x2, y2]
    path or crop_path

Output metadata
---------------
The output keeps the input JSON structure and replaces only "crops" with the
hard-deduplicated records. Semantic duplicates remain in "crops".

Optional relation metadata:

    duplicate_group_id:
        stable group id shared by detections confidently judged to represent
        the same physical object. All members remain separate records.

    ambiguous_group_id:
        stable group id shared by uncertain detections that are intentionally
        kept separate.

    duplicate_status:
        "grouped" | "ambiguous" | "grouped_ambiguous"

Hard-dedup audit metadata may also be present on the surviving exact-ID record:
    hard_dedup_count
    class_conflict
    class_candidates
    merged_detection_ids

Important policy
----------------
- Exact duplicated detection_id:
    physically keep only the highest-confidence record.

- Same-class / cross-class semantic duplicate:
    use the existing conservative geometry + DINO decision rules, but convert a
    "duplicate" decision into duplicate_group_id assignment instead of deletion.

- Spatially disjoint boxes are never semantic duplicate candidates.

This makes heuristic mistakes reversible: semantic judgement can change search
presentation, but cannot destroy the underlying detection or embedding.
"""

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT = ROOT / "data" / "crops" / "filter_stats.json"
DEFAULT_OUTPUT = ROOT / "data" / "crops" / "filter_stats_dedup.json"
DEFAULT_REPORT = ROOT / "data" / "crops" / "dedup_report.json"
DEFAULT_CONFIG = ROOT / "pipeline.yaml"

PERSON_LABELS_FALLBACK = {
    "person",
    "pedestrian",
    "people",
    "human",
}


# =============================================================================
# Policy
# =============================================================================

@dataclass(frozen=True)
class DedupPolicy:
    # Candidate generation: same-class
    same_candidate_iou: float
    same_candidate_iom: float
    same_candidate_center: float
    same_candidate_area_ratio: float

    # Candidate generation: cross-class
    cross_candidate_iou: float
    cross_candidate_iom: float
    cross_candidate_center: float
    cross_candidate_area_ratio: float

    # Same-class automatic semantic-group policy.
    # Only STRONG geometry may auto-group, and only when the two boxes cover
    # nearly the same spatial extent. MEDIUM/WEAK never auto-group regardless
    # of DINO similarity.
    same_strong_sim: float
    same_near_same_iou: float
    same_near_same_area_ratio: float

    # Cross-class DINO requirements by geometry tier
    cross_strong_sim: float
    cross_medium_sim: float
    cross_weak_sim: float

    # Cross-class automatic semantic grouping requires nearly the same extent.
    cross_near_same_iou: float
    cross_near_same_area_ratio: float

    # Minimum DINO evidence for an uncertain pair to be kept as ambiguous.
    # IMPORTANT: these floors are used directly. A score below the floor is
    # always separate, regardless of the duplicate threshold for the tier.
    same_ambiguous_floor: float
    cross_ambiguous_floor: float


POLICIES: Dict[str, DedupPolicy] = {
    # Recommended default for this project.
    "balanced": DedupPolicy(
        same_candidate_iou=0.10,
        same_candidate_iom=0.55,
        same_candidate_center=0.22,
        same_candidate_area_ratio=0.25,
        cross_candidate_iou=0.15,
        cross_candidate_iom=0.65,
        cross_candidate_center=0.18,
        cross_candidate_area_ratio=0.35,
        same_strong_sim=0.90,
        same_near_same_iou=0.80,
        same_near_same_area_ratio=0.90,
        cross_strong_sim=0.95,
        cross_medium_sim=0.975,
        cross_weak_sim=0.992,
        cross_near_same_iou=0.99,
        cross_near_same_area_ratio=0.99,
        same_ambiguous_floor=0.90,
        cross_ambiguous_floor=0.95,
    ),

    # Stricter semantic grouping; useful when false grouping is especially costly.
    "conservative": DedupPolicy(
        same_candidate_iou=0.12,
        same_candidate_iom=0.60,
        same_candidate_center=0.20,
        same_candidate_area_ratio=0.30,
        cross_candidate_iou=0.20,
        cross_candidate_iom=0.72,
        cross_candidate_center=0.16,
        cross_candidate_area_ratio=0.40,
        same_strong_sim=0.92,
        same_near_same_iou=0.85,
        same_near_same_area_ratio=0.93,
        cross_strong_sim=0.97,
        cross_medium_sim=0.985,
        cross_weak_sim=0.995,
        cross_near_same_iou=0.995,
        cross_near_same_area_ratio=0.995,
        same_ambiguous_floor=0.91,
        cross_ambiguous_floor=0.96,
    ),

    # More permissive semantic grouping; keep only for controlled experiments.
    "aggressive": DedupPolicy(
        same_candidate_iou=0.08,
        same_candidate_iom=0.50,
        same_candidate_center=0.25,
        same_candidate_area_ratio=0.20,
        cross_candidate_iou=0.12,
        cross_candidate_iom=0.58,
        cross_candidate_center=0.20,
        cross_candidate_area_ratio=0.30,
        same_strong_sim=0.88,
        same_near_same_iou=0.75,
        same_near_same_area_ratio=0.85,
        cross_strong_sim=0.93,
        cross_medium_sim=0.96,
        cross_weak_sim=0.985,
        cross_near_same_iou=0.87,
        cross_near_same_area_ratio=0.87,
        same_ambiguous_floor=0.88,
        cross_ambiguous_floor=0.93,
    ),
}


# =============================================================================
# Record helpers
# =============================================================================

def _label(record: Dict[str, Any]) -> str:
    value = record.get("class_name")

    if value is None or not str(value).strip():
        raise ValueError(
            f"Missing class_name: detection_id={record.get('detection_id')!r}"
        )

    return str(value).strip().lower()


def _confidence(record: Dict[str, Any]) -> float:
    value = record.get("confidence")

    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid confidence: {value!r}"
        ) from exc

    if not math.isfinite(score):
        raise ValueError(
            f"Non-finite confidence: {score!r}"
        )

    return score


def _bbox(
    record: Dict[str, Any],
) -> Tuple[float, float, float, float]:
    value = record.get("bbox")

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(
            f"Invalid bbox: {value!r}"
        )

    try:
        x1, y1, x2, y2 = map(float, value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Non-numeric bbox: {value!r}"
        ) from exc

    if not all(
        math.isfinite(v)
        for v in (x1, y1, x2, y2)
    ):
        raise ValueError(
            f"Non-finite bbox: {value!r}"
        )

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Non-positive bbox: {value!r}"
        )

    return x1, y1, x2, y2


def _image_id(record: Dict[str, Any]) -> str:
    value = record.get("image_id")

    if value is None or not str(value).strip():
        raise ValueError(
            f"Missing image_id: detection_id={record.get('detection_id')!r}"
        )

    # image_id is a logical ID, not an OS-specific absolute path.
    return str(value).strip().replace("\\", "/")


def _detection_id(record: Dict[str, Any]) -> str:
    value = record.get("detection_id")

    if value is None or not str(value).strip():
        raise ValueError(
            f"Missing detection_id: image_id={record.get('image_id')!r}"
        )

    return str(value).strip()


def _crop_path(record: Dict[str, Any]) -> str:
    value = (
        record.get("crop_path")
        or record.get("path")
    )

    if value is None or not str(value).strip():
        raise ValueError(
            f"Missing crop path: detection_id={record.get('detection_id')!r}"
        )

    return str(value).strip()


def _resolve_crop_path(record: Dict[str, Any]) -> Path:
    path = Path(
        _crop_path(record)
    )

    if path.is_absolute():
        return path

    return (ROOT / path).resolve()


def _summary(
    record: Dict[str, Any],
    index: int,
) -> Dict[str, Any]:
    return {
        "index": int(index),
        "image_id": _image_id(record),
        "detection_id": _detection_id(record),
        "class_name": record.get("class_name"),
        "confidence": _confidence(record),
        "bbox": list(_bbox(record)),
        "crop_path": _crop_path(record).replace("\\", "/"),
    }


def _rounded_bbox(record: Dict[str, Any]) -> Tuple[int, int, int, int]:
    return tuple(
        int(round(v))
        for v in _bbox(record)
    )


# =============================================================================
# Geometry
# =============================================================================

def _geometry(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> Dict[str, float]:
    """
    Return geometry features.

    center_distance is normalized by the diagonal of the union envelope.
    shape_similarity is min(aspect_a/aspect_b, aspect_b/aspect_a), in [0, 1].
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    aw = ax2 - ax1
    ah = ay2 - ay1
    bw = bx2 - bx1
    bh = by2 - by1

    area_a = aw * ah
    area_b = bw * bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    union = area_a + area_b - inter
    iou = 0.0 if union <= 0.0 else inter / union

    smaller = min(area_a, area_b)
    iom = 0.0 if smaller <= 0.0 else inter / smaller

    larger = max(area_a, area_b)
    area_ratio = 0.0 if larger <= 0.0 else smaller / larger

    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0

    envelope_w = max(ax2, bx2) - min(ax1, bx1)
    envelope_h = max(ay2, by2) - min(ay1, by1)
    envelope_diag = math.hypot(envelope_w, envelope_h)
    center_distance = (
        0.0
        if envelope_diag <= 0.0
        else math.hypot(acx - bcx, acy - bcy) / envelope_diag
    )

    aspect_a = aw / ah
    aspect_b = bw / bh
    shape_similarity = min(
        aspect_a / aspect_b,
        aspect_b / aspect_a,
    )
    shape_similarity = max(0.0, min(1.0, shape_similarity))

    return {
        "intersection": float(inter),
        "iou": float(iou),
        "iom": float(iom),
        "area_ratio": float(area_ratio),
        "center_distance": float(center_distance),
        "shape_similarity": float(shape_similarity),
    }


def _geometry_tier(g: Dict[str, float]) -> str:
    iou = g["iou"]
    iom = g["iom"]
    area_ratio = g["area_ratio"]
    center = g["center_distance"]
    shape = g["shape_similarity"]

    if (
        iou >= 0.75
        or (iom >= 0.92 and area_ratio >= 0.55)
        or (center <= 0.08 and area_ratio >= 0.70 and shape >= 0.65)
    ):
        return "strong"

    if (
        iou >= 0.45
        or (iom >= 0.80 and area_ratio >= 0.35)
        or (center <= 0.14 and area_ratio >= 0.50 and shape >= 0.50)
    ):
        return "medium"

    return "weak"


def _is_geometry_candidate(
    a: Dict[str, Any],
    b: Dict[str, Any],
    policy: DedupPolicy,
) -> Tuple[bool, Dict[str, Any]]:
    g = _geometry(
        _bbox(a),
        _bbox(b),
    )

    same_class = _label(a) == _label(b)

    # Absolute safety gate: spatially disjoint detections are never merged.
    if g["intersection"] <= 0.0:
        candidate = False

    elif same_class:
        candidate = (
            g["iou"] >= policy.same_candidate_iou
            or (
                g["iom"] >= policy.same_candidate_iom
                and g["area_ratio"] >= policy.same_candidate_area_ratio
            )
            or (
                g["center_distance"] <= policy.same_candidate_center
                and g["area_ratio"] >= policy.same_candidate_area_ratio
            )
        )

    else:
        candidate = (
            g["iou"] >= policy.cross_candidate_iou
            or (
                g["iom"] >= policy.cross_candidate_iom
                and g["area_ratio"] >= policy.cross_candidate_area_ratio
            )
            or (
                g["center_distance"] <= policy.cross_candidate_center
                and g["area_ratio"] >= policy.cross_candidate_area_ratio
            )
        )

    return candidate, {
        "mode": "same_class" if same_class else "cross_class",
        "tier": _geometry_tier(g),
        "iou": round(g["iou"], 6),
        "iom": round(g["iom"], 6),
        "area_ratio": round(g["area_ratio"], 6),
        "center_distance": round(g["center_distance"], 6),
        "shape_similarity": round(g["shape_similarity"], 6),
    }


# =============================================================================
# DINOv2
# =============================================================================

def _load_pipeline_config(
    config_path: Path,
):
    if str(ROOT) not in sys.path:
        sys.path.insert(
            0,
            str(ROOT),
        )

    from config import PipelineConfig

    return PipelineConfig.load(
        str(config_path)
    )


def _load_dinov2(
    cfg,
    *,
    batch_size: Optional[int],
    device: Optional[str],
):
    spec = cfg.retrievers.get(
        "dinov2"
    )

    if spec is None:
        raise RuntimeError(
            "pipeline.yaml does not define retrievers.dinov2."
        )

    module = importlib.import_module(
        spec.module
    )

    cls = getattr(
        module,
        spec.class_name,
    )

    params = dict(
        spec.params
    )

    if batch_size is not None:
        params["batch_size"] = int(
            batch_size
        )

    if device is not None:
        params["device"] = str(
            device
        )

    embedder = cls(
        **params
    )

    actual_dim = int(
        getattr(
            embedder,
            "DIM",
            -1,
        )
    )

    if actual_dim != int(spec.dim):
        raise RuntimeError(
            "DINOv2 dimension mismatch: "
            f"embedder={actual_dim}, config={spec.dim}"
        )

    return embedder


def _collect_candidate_pairs(
    crops: Sequence[Dict[str, Any]],
    *,
    excluded_indices: set[int],
    person_labels: set[str],
    policy: DedupPolicy,
) -> Tuple[
    List[Tuple[int, int, Dict[str, Any]]],
    set[int],
]:
    """
    Geometry finds plausible duplicate pairs.
    DINO runs only for pairs that actually need visual verification.
    """
    groups: Dict[str, List[int]] = defaultdict(list)

    for idx, record in enumerate(crops):
        if idx in excluded_indices:
            continue

        if _label(record) in person_labels:
            continue

        groups[_image_id(record)].append(idx)

    pairs: List[Tuple[int, int, Dict[str, Any]]] = []
    dino_indices: set[int] = set()

    for indices in groups.values():
        for pos_a in range(len(indices) - 1):
            idx_a = indices[pos_a]

            for pos_b in range(pos_a + 1, len(indices)):
                idx_b = indices[pos_b]

                candidate, geometry = _is_geometry_candidate(
                    crops[idx_a],
                    crops[idx_b],
                    policy,
                )

                if not candidate:
                    continue

                # Exact same-class integer bbox is deterministic enough to
                # suppress without a DINO call.
                same_class = _label(crops[idx_a]) == _label(crops[idx_b])
                exact_same_bbox = (
                    same_class
                    and _rounded_bbox(crops[idx_a]) == _rounded_bbox(crops[idx_b])
                )

                geometry["exact_same_bbox"] = bool(exact_same_bbox)

                pairs.append((idx_a, idx_b, geometry))

                if not exact_same_bbox:
                    dino_indices.add(idx_a)
                    dino_indices.add(idx_b)

    return pairs, dino_indices


def _embed_candidates(
    crops: Sequence[Dict[str, Any]],
    indices: Iterable[int],
    cfg,
    *,
    dino_batch_size: Optional[int],
    dino_outer_batch_size: int,
    dino_device: Optional[str],
) -> Tuple[Dict[int, int], np.ndarray]:
    indices = sorted({int(i) for i in indices})

    if not indices:
        return {}, np.empty((0, 0), dtype=np.float32)

    paths: List[str] = []

    for idx in indices:
        path = _resolve_crop_path(crops[idx])

        if not path.is_file():
            raise FileNotFoundError(
                "DINO candidate crop not found: "
                f"{path} "
                f"(detection_id={_detection_id(crops[idx])})"
            )

        paths.append(str(path))

    print()
    print("=" * 72)
    print("LOAD DINOv2 FOR DEDUP")
    print("=" * 72)
    print(f"candidate detections : {len(indices):,}")

    embedder = _load_dinov2(
        cfg,
        batch_size=dino_batch_size,
        device=dino_device,
    )

    dim = int(embedder.DIM)
    vectors = np.empty((len(indices), dim), dtype=np.float32)
    row_by_index: Dict[int, int] = {}

    start_time = time.time()

    for start in range(0, len(indices), dino_outer_batch_size):
        end = min(start + dino_outer_batch_size, len(indices))

        chunk = embedder.embed_crops(
            paths[start:end],
            input_format="rgb",
        )

        chunk = np.asarray(chunk, dtype=np.float32)
        expected = (end - start, dim)

        if chunk.shape != expected:
            raise RuntimeError(
                "Unexpected DINOv2 output shape: "
                f"{chunk.shape}, expected {expected}"
            )

        if not np.isfinite(chunk).all():
            raise RuntimeError(
                "DINOv2 output contains NaN/Inf."
            )

        # Normalize again here so cosine remains correct even if the embedder
        # configuration changes later.
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)

        if np.any(norms <= 0.0):
            raise RuntimeError(
                "DINOv2 produced a zero-norm embedding."
            )

        chunk = chunk / norms
        vectors[start:end] = chunk

        for row, idx in enumerate(indices[start:end], start=start):
            row_by_index[idx] = row

        elapsed = time.time() - start_time
        done = end
        rate = done / elapsed if elapsed > 0.0 else 0.0

        print(
            f"\rDINO candidates: "
            f"{done:,}/{len(indices):,} "
            f"({done / len(indices) * 100:5.1f}%) "
            f"{rate:6.1f} crop/s",
            end="",
            flush=True,
        )

    print()

    return row_by_index, vectors


def _dino_similarity(
    idx_a: int,
    idx_b: int,
    row_by_index: Dict[int, int],
    vectors: np.ndarray,
) -> float:
    row_a = row_by_index[idx_a]
    row_b = row_by_index[idx_b]

    return float(
        np.dot(
            vectors[row_a],
            vectors[row_b],
        )
    )


# =============================================================================
# Pair decision
# =============================================================================

def _required_similarity(
    *,
    same_class: bool,
    tier: str,
    policy: DedupPolicy,
    shape_similarity: float,
) -> float:
    if same_class:
        # Same-class automatic semantic grouping is limited to STRONG
        # geometry. MEDIUM/WEAK remain non-destructive by policy.
        if tier != "strong":
            raise RuntimeError(
                "Safety invariant violated: same-class auto-group threshold "
                "requested for a non-strong geometry tier."
            )
        required = float(policy.same_strong_sim)
    else:
        table = {
            "strong": policy.cross_strong_sim,
            "medium": policy.cross_medium_sim,
            "weak": policy.cross_weak_sim,
        }
        required = float(table[tier])

    # Cross-class pairs with very different shape need slightly stronger visual
    # evidence even when their boxes overlap.
    if not same_class and shape_similarity < 0.45:
        required = min(1.0, required + 0.005)

    return required


def _pair_decision(
    a: Dict[str, Any],
    b: Dict[str, Any],
    geometry: Dict[str, Any],
    *,
    similarity: Optional[float],
    policy: DedupPolicy,
) -> Dict[str, Any]:
    same_class = _label(a) == _label(b)
    tier = str(geometry["tier"])

    # Same-class exact integer bbox: deterministic semantic duplicate relation.
    # Both detections are preserved; the caller will place them in one group.
    if same_class and bool(geometry.get("exact_same_bbox")):
        return {
            "decision": "duplicate",
            "reason": "same_class_exact_bbox",
            "required_similarity": None,
            "dino_similarity": None,
        }

    if similarity is None:
        raise RuntimeError(
            "DINO similarity is required for this candidate pair."
        )

    # ------------------------------------------------------------------
    # Same-class false-grouping safety rule
    # ------------------------------------------------------------------
    # Different instances of the same class can look extremely similar in
    # DINO space (e.g. two horses, two cars). Therefore:
    #   - MEDIUM/WEAK geometry can NEVER auto-group.
    #   - STRONG geometry can auto-group only when the bbox extents are also
    #     nearly the same AND DINO similarity passes the strong threshold.
    # Anything else is ambiguous/separate. All detections survive either way.
    if same_class:
        ambiguous_threshold = policy.same_ambiguous_floor

        if tier != "strong":
            if similarity >= ambiguous_threshold:
                decision = "ambiguous"
                reason = f"same_class_{tier}_forced_ambiguous"
            else:
                decision = "separate"
                reason = f"same_class_{tier}_below_ambiguous_floor"

            return {
                "decision": decision,
                "reason": reason,
                "auto_merge_allowed": False,
                "near_same_box": False,
                "required_similarity": None,
                "dino_similarity": round(float(similarity), 6),
            }

        near_same_box = (
            float(geometry["iou"]) >= policy.same_near_same_iou
            and float(geometry["area_ratio"]) >= policy.same_near_same_area_ratio
        )

        required = _required_similarity(
            same_class=True,
            tier=tier,
            policy=policy,
            shape_similarity=float(geometry["shape_similarity"]),
        )

        if near_same_box and similarity >= required:
            decision = "duplicate"
            reason = "same_class_strong_near_same_box_dino"
        elif similarity >= ambiguous_threshold:
            decision = "ambiguous"
            reason = (
                "same_class_strong_not_same_extent"
                if not near_same_box
                else "same_class_strong_borderline_dino"
            )
        else:
            decision = "separate"
            reason = "same_class_strong_below_ambiguous_floor"

        return {
            "decision": decision,
            "reason": reason,
            "auto_merge_allowed": bool(near_same_box),
            "near_same_box": bool(near_same_box),
            "required_similarity": round(required, 6),
            "dino_similarity": round(float(similarity), 6),
        }

    # ------------------------------------------------------------------
    # Cross-class safety rule
    # ------------------------------------------------------------------
    required = _required_similarity(
        same_class=False,
        tier=tier,
        policy=policy,
        shape_similarity=float(geometry["shape_similarity"]),
    )

    # DINO is not a class classifier. Containment is common for different
    # physical objects (pizza/table, bowl/orange, person/bag, etc.). Therefore
    # cross-class auto-grouping is allowed only when the two boxes cover nearly
    # the same spatial extent. Otherwise strong DINO similarity produces an
    # ambiguous relation. No semantic decision deletes a detection.
    near_same_box = (
        float(geometry["iou"]) >= policy.cross_near_same_iou
        and float(geometry["area_ratio"]) >= policy.cross_near_same_area_ratio
    )

    if near_same_box and similarity >= required:
        decision = "duplicate"
        reason = f"cross_class_near_same_box_{tier}_dino"
    else:
        ambiguous_threshold = policy.cross_ambiguous_floor

        if similarity >= ambiguous_threshold:
            decision = "ambiguous"
            reason = (
                "cross_class_not_same_extent"
                if not near_same_box
                else "cross_class_borderline_dino"
            )
        else:
            decision = "separate"
            reason = "cross_class_below_threshold"

    return {
        "decision": decision,
        "reason": reason,
        "near_same_box": bool(near_same_box),
        "required_similarity": round(required, 6),
        "dino_similarity": round(float(similarity), 6),
    }


# =============================================================================
# Metadata merge helpers
# =============================================================================

def _candidate_entries(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = record.get("class_candidates")
    entries: List[Dict[str, Any]] = []

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue

            name = item.get("class_name")
            conf = item.get("confidence")

            if name is None:
                continue

            try:
                conf_value = float(conf)
            except (TypeError, ValueError):
                continue

            if math.isfinite(conf_value):
                entries.append({
                    "class_name": str(name),
                    "confidence": conf_value,
                })

    # Always include the record's own primary RF-DETR label.
    entries.append({
        "class_name": str(record.get("class_name")),
        "confidence": _confidence(record),
    })

    return entries


def _merged_ids(record: Dict[str, Any]) -> List[str]:
    values: List[str] = []

    raw = record.get("merged_detection_ids")
    if isinstance(raw, list):
        for value in raw:
            if value is not None and str(value).strip():
                values.append(str(value).strip())

    values.append(_detection_id(record))
    return values


def _dedupe_class_candidates(
    entries: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # Keep the highest confidence observed for each class.
    best: Dict[str, Tuple[str, float]] = {}

    for item in entries:
        name = str(item["class_name"]).strip()
        key = name.lower()
        conf = float(item["confidence"])

        old = best.get(key)
        if old is None or conf > old[1]:
            best[key] = (name, conf)

    return [
        {
            "class_name": name,
            "confidence": conf,
        }
        for name, conf in sorted(
            best.values(),
            key=lambda x: (-x[1], x[0].lower()),
        )
    ]


def _merge_survivor_metadata(
    survivor: Dict[str, Any],
    removed: Dict[str, Any],
) -> None:
    """Merge metadata only for HARD exact-detection_id deduplication.

    Semantic duplicates never call this function because both records survive.
    """
    candidates = _dedupe_class_candidates(
        _candidate_entries(survivor)
        + _candidate_entries(removed)
    )

    merged_ids = sorted(set(
        _merged_ids(survivor)
        + _merged_ids(removed)
    ))

    unique_labels = {
        str(item["class_name"]).strip().lower()
        for item in candidates
    }

    survivor["class_candidates"] = candidates
    survivor["class_conflict"] = len(unique_labels) > 1
    survivor["merged_detection_ids"] = merged_ids
    survivor["hard_dedup_count"] = int(
        survivor.get("hard_dedup_count", 0) or 0
    ) + 1


def _refresh_relation_status(record: Dict[str, Any]) -> None:
    """Derive duplicate_status only from SOFT relation metadata."""
    has_duplicate = bool(record.get("duplicate_group_id"))
    has_ambiguous = bool(record.get("ambiguous_group_id"))

    if has_duplicate and has_ambiguous:
        record["duplicate_status"] = "grouped_ambiguous"
    elif has_duplicate:
        record["duplicate_status"] = "grouped"
    elif has_ambiguous:
        record["duplicate_status"] = "ambiguous"
    else:
        record.pop("duplicate_status", None)


# =============================================================================
# Soft duplicate / ambiguous grouping
# =============================================================================

class _DisjointSet:
    def __init__(self, items: Iterable[int]):
        self.parent = {int(x): int(x) for x in items}
        self.rank = {int(x): 0 for x in items}

    def find(self, x: int) -> int:
        parent = self.parent[x]
        if parent != x:
            self.parent[x] = self.find(parent)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _stable_group_id(
    prefix: str,
    image_id: str,
    detection_ids: Sequence[str],
) -> str:
    key = image_id + "|" + "|".join(sorted(detection_ids))
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _apply_relation_groups(
    crops: Sequence[Dict[str, Any]],
    surviving_indices: set[int],
    edges: Sequence[Tuple[int, int]],
    *,
    field_name: str,
    prefix: str,
) -> Dict[str, Any]:
    """Assign stable connected-component group IDs without deleting records."""
    if not surviving_indices or not edges:
        return {
            "group_count": 0,
            "detection_count": 0,
            "groups": [],
        }

    dsu = _DisjointSet(surviving_indices)

    for idx_a, idx_b in edges:
        if idx_a == idx_b:
            continue
        if idx_a not in surviving_indices or idx_b not in surviving_indices:
            continue
        if _image_id(crops[idx_a]) != _image_id(crops[idx_b]):
            raise RuntimeError(
                f"{field_name} edge crosses source images: "
                f"{_image_id(crops[idx_a])!r} != {_image_id(crops[idx_b])!r}"
            )
        dsu.union(idx_a, idx_b)

    touched: set[int] = set()
    for idx_a, idx_b in edges:
        if idx_a in surviving_indices and idx_b in surviving_indices:
            touched.add(idx_a)
            touched.add(idx_b)

    components: Dict[int, List[int]] = defaultdict(list)
    for idx in sorted(touched):
        components[dsu.find(idx)].append(idx)

    groups_report: List[Dict[str, Any]] = []
    detection_count = 0

    for members in components.values():
        if len(members) <= 1:
            continue

        image_id = _image_id(crops[members[0]])
        ids = [_detection_id(crops[idx]) for idx in members]
        group_id = _stable_group_id(prefix, image_id, ids)

        for idx in members:
            crops[idx][field_name] = group_id

        detection_count += len(members)
        groups_report.append({
            field_name: group_id,
            "image_id": image_id,
            "member_indices": members,
            "detection_ids": ids,
        })

    groups_report.sort(key=lambda x: x[field_name])

    return {
        "group_count": len(groups_report),
        "detection_count": detection_count,
        "groups": groups_report,
    }


def _apply_duplicate_groups(
    crops: Sequence[Dict[str, Any]],
    surviving_indices: set[int],
    duplicate_edges: Sequence[Tuple[int, int]],
) -> Dict[str, Any]:
    return _apply_relation_groups(
        crops,
        surviving_indices,
        duplicate_edges,
        field_name="duplicate_group_id",
        prefix="duplicate",
    )


def _apply_ambiguous_groups(
    crops: Sequence[Dict[str, Any]],
    surviving_indices: set[int],
    ambiguous_edges: Sequence[Tuple[int, int]],
) -> Dict[str, Any]:
    return _apply_relation_groups(
        crops,
        surviving_indices,
        ambiguous_edges,
        field_name="ambiguous_group_id",
        prefix="ambiguous",
    )


# =============================================================================
# I/O
# =============================================================================

def _atomic_write_json(
    path: Path,
    data: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def _backup_path(
    input_path: Path,
) -> Path:
    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    return input_path.with_name(
        f"{input_path.stem}_before_dedup_"
        f"{stamp}{input_path.suffix}"
    )


# =============================================================================
# Dedup core
# =============================================================================

def deduplicate(
    crops: Sequence[Dict[str, Any]],
    cfg,
    *,
    policy_name: str,
    policy: DedupPolicy,
    dino_batch_size: Optional[int],
    dino_outer_batch_size: int,
    dino_device: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    started = time.time()

    # Work on copies so the caller still has the untouched RF-DETR records.
    work: List[Dict[str, Any]] = [
        copy.deepcopy(record)
        for record in crops
    ]

    person_labels = {
        str(x).strip().lower()
        for x in (
            getattr(cfg, "person_labels", None)
            or PERSON_LABELS_FALLBACK
        )
    }

    # Validate required fields before doing expensive DINO work.
    for record in work:
        _image_id(record)
        _detection_id(record)
        _label(record)
        _confidence(record)
        _bbox(record)
        _crop_path(record)

        # Relation metadata is derived by this run. Do not inherit stale groups.
        record.pop("duplicate_group_id", None)
        record.pop("ambiguous_group_id", None)
        record.pop("duplicate_status", None)

    # Only HARD exact-detection_id duplicates may enter this set.
    removed: set[int] = set()
    removals: List[Dict[str, Any]] = []
    exact_removed = 0

    # ------------------------------------------------------------------
    # 1. HARD DEDUP: exact duplicate detection_id globally.
    # ------------------------------------------------------------------
    id_groups: Dict[str, List[int]] = defaultdict(list)

    for idx, record in enumerate(work):
        id_groups[_detection_id(record)].append(idx)

    for detection_id, indices in id_groups.items():
        if len(indices) <= 1:
            continue

        ranked = sorted(
            indices,
            key=lambda idx: (-_confidence(work[idx]), idx),
        )
        keep_idx = ranked[0]

        for idx in ranked[1:]:
            removed.add(idx)
            exact_removed += 1

            _merge_survivor_metadata(
                work[keep_idx],
                work[idx],
            )

            removals.append({
                "type": "exact_detection_id",
                "detection_id": detection_id,
                "kept": _summary(work[keep_idx], keep_idx),
                "removed": _summary(work[idx], idx),
            })

    # ------------------------------------------------------------------
    # 2. Geometry candidates for OBJECT detections only.
    #    Exact-ID-removed records are excluded.
    # ------------------------------------------------------------------
    candidate_pairs, dino_indices = _collect_candidate_pairs(
        work,
        excluded_indices=removed,
        person_labels=person_labels,
        policy=policy,
    )

    row_by_index, vectors = _embed_candidates(
        work,
        dino_indices,
        cfg,
        dino_batch_size=dino_batch_size,
        dino_outer_batch_size=dino_outer_batch_size,
        dino_device=dino_device,
    )

    pair_decisions: List[Dict[str, Any]] = []
    duplicate_edges: List[Tuple[int, int]] = []
    ambiguous_edges: List[Tuple[int, int]] = []

    duplicate_pair_count = 0
    same_duplicate_pair_count = 0
    cross_duplicate_pair_count = 0
    ambiguous_pair_count = 0
    separate_pair_count = 0

    for idx_a, idx_b, geometry in candidate_pairs:
        same_class = _label(work[idx_a]) == _label(work[idx_b])
        exact_same_bbox = bool(geometry.get("exact_same_bbox"))

        if exact_same_bbox:
            sim = None
        else:
            sim = _dino_similarity(
                idx_a,
                idx_b,
                row_by_index,
                vectors,
            )

        result = _pair_decision(
            work[idx_a],
            work[idx_b],
            geometry,
            similarity=sim,
            policy=policy,
        )

        decision = str(result["decision"])
        if decision == "duplicate":
            duplicate_pair_count += 1
            duplicate_edges.append((idx_a, idx_b))
            if same_class:
                same_duplicate_pair_count += 1
            else:
                cross_duplicate_pair_count += 1
        elif decision == "ambiguous":
            ambiguous_pair_count += 1
            ambiguous_edges.append((idx_a, idx_b))
        elif decision == "separate":
            separate_pair_count += 1
        else:
            raise RuntimeError(f"Unknown pair decision: {decision!r}")

        pair_decisions.append({
            "image_id": _image_id(work[idx_a]),
            "same_class": bool(same_class),
            "geometry": geometry,
            **result,
            "a": _summary(work[idx_a], idx_a),
            "b": _summary(work[idx_b], idx_b),
        })

    # ------------------------------------------------------------------
    # 3. SOFT DEDUP: assign relation groups. No semantic deletion.
    # ------------------------------------------------------------------
    surviving_indices = {
        idx
        for idx in range(len(work))
        if idx not in removed
    }

    duplicate_group_info = _apply_duplicate_groups(
        work,
        surviving_indices,
        duplicate_edges,
    )

    ambiguous_group_info = _apply_ambiguous_groups(
        work,
        surviving_indices,
        ambiguous_edges,
    )

    for idx in surviving_indices:
        _refresh_relation_status(work[idx])

    deduped = [
        record
        for idx, record in enumerate(work)
        if idx not in removed
    ]

    # ------------------------------------------------------------------
    # 4. Final compatibility / safety checks.
    # ------------------------------------------------------------------
    final_ids = [_detection_id(record) for record in deduped]
    counts = Counter(final_ids)

    duplicate_ids_left = [
        detection_id
        for detection_id, count in counts.items()
        if count > 1
    ]

    if duplicate_ids_left:
        raise RuntimeError(
            "Duplicate detection_id values remain after hard dedup. "
            "build_db.py would reject them. "
            f"Examples: {duplicate_ids_left[:5]}"
        )

    # Hybrid-C invariant: semantic duplicate/ambiguous judgements never remove a
    # record. Therefore every removal must be exact_detection_id hard dedup.
    if len(removed) != exact_removed:
        raise RuntimeError(
            "Hybrid-C invariant violated: non-exact semantic removal occurred."
        )

    non_exact_removals = [
        item
        for item in removals
        if item.get("type") != "exact_detection_id"
    ]
    if non_exact_removals:
        raise RuntimeError(
            "Hybrid-C invariant violated: removals contains semantic deletions."
        )

    expected_after_hard_dedup = len(crops) - exact_removed
    if len(deduped) != expected_after_hard_dedup:
        raise RuntimeError(
            "Hybrid-C count invariant violated: "
            f"expected={expected_after_hard_dedup}, actual={len(deduped)}"
        )

    # Person detections must never receive semantic group metadata.
    person_soft_relations = [
        _detection_id(record)
        for record in deduped
        if _label(record) in person_labels
        and (
            record.get("duplicate_group_id") is not None
            or record.get("ambiguous_group_id") is not None
        )
    ]
    if person_soft_relations:
        raise RuntimeError(
            "Safety invariant violated: person detection received bbox/DINO "
            f"relation metadata. Examples: {person_soft_relations[:5]}"
        )

    elapsed = time.time() - started

    report = {
        "mode": "hybrid_c_hard_exact_soft_semantic",
        "policy_name": policy_name,
        "policy": asdict(policy),
        "original_crop_count": len(crops),
        "deduped_crop_count": len(deduped),
        "removed_count": len(removed),
        "exact_detection_id_removed": exact_removed,

        # Compatibility keys: semantic removals are intentionally always zero.
        "same_class_removed": 0,
        "cross_class_removed": 0,

        "candidate_pair_count": len(candidate_pairs),
        "dino_candidate_detection_count": len(dino_indices),
        "dino_embedded_detection_count": len(row_by_index),
        "duplicate_pair_count": duplicate_pair_count,
        "same_class_duplicate_pair_count": same_duplicate_pair_count,
        "cross_class_duplicate_pair_count": cross_duplicate_pair_count,
        "duplicate_group_count": duplicate_group_info["group_count"],
        "duplicate_grouped_detection_count": duplicate_group_info["detection_count"],
        "ambiguous_pair_count": ambiguous_pair_count,
        "separate_candidate_pair_count": separate_pair_count,
        "ambiguous_group_count": ambiguous_group_info["group_count"],
        "ambiguous_detection_count": ambiguous_group_info["detection_count"],
        "final_unique_detection_id_count": len(counts),
        "elapsed_seconds": round(elapsed, 3),

        # Only hard exact-ID removals appear here.
        "removals": removals,
        "duplicate_groups": duplicate_group_info["groups"],
        "ambiguous_groups": ambiguous_group_info["groups"],
        "duplicate_pairs": [
            item for item in pair_decisions
            if item.get("decision") == "duplicate"
        ],
        "ambiguous_pairs": [
            item for item in pair_decisions
            if item.get("decision") == "ambiguous"
        ],
        "separate_candidate_pairs": [
            item for item in pair_decisions
            if item.get("decision") == "separate"
        ],
        "pair_decisions": pair_decisions,
    }

    return deduped, report


# =============================================================================
# CLI
# =============================================================================

def _validate_policy(policy: DedupPolicy) -> None:
    values = asdict(policy)

    for name, value in values.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"Policy value must be numeric: {name}={value!r}")

        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"Policy value must be between 0 and 1: {name}={value}"
            )

    if not (
        policy.cross_strong_sim
        <= policy.cross_medium_sim
        <= policy.cross_weak_sim
    ):
        raise ValueError(
            "cross-class DINO thresholds must satisfy strong <= medium <= weak"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "RF-DETR Hybrid-C duplicate handling: exact-ID hard dedup + "
            "non-destructive semantic grouping with geometry + DINOv2."
        )
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )

    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
    )

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
    )

    parser.add_argument(
        "--policy",
        choices=sorted(POLICIES),
        default="balanced",
        help=(
            "Semantic-group threshold policy. Default: balanced. "
            "Use conservative when false grouping is especially costly."
        ),
    )

    parser.add_argument(
        "--dino-batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--dino-outer-batch-size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--dino-device",
        default=None,
        help="Optional DINO device override, e.g. cuda or cpu.",
    )

    parser.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Backup the input JSON and replace it with the deduplicated result. "
            "For validation experiments, do NOT use this option."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    config_path = Path(args.config).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input not found: {input_path}"
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config not found: {config_path}"
        )

    if input_path == output_path and not args.in_place:
        raise ValueError(
            "--output must differ from --input unless --in-place is used."
        )

    if args.dino_batch_size is not None and args.dino_batch_size <= 0:
        raise ValueError(
            "--dino-batch-size must be >= 1"
        )

    if args.dino_outer_batch_size <= 0:
        raise ValueError(
            "--dino-outer-batch-size must be >= 1"
        )

    policy = POLICIES[args.policy]
    _validate_policy(policy)

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(
            "filter_stats.json must be a JSON object."
        )

    crops = data.get("crops")
    filtered = data.get("filtered", [])

    if not isinstance(crops, list):
        raise TypeError(
            '"crops" must be a list.'
        )

    if not isinstance(filtered, list):
        raise TypeError(
            '"filtered" must be a list.'
        )

    if not crops:
        raise RuntimeError(
            "No accepted crops found."
        )

    cfg = _load_pipeline_config(
        config_path
    )

    print()
    print("=" * 72)
    print("RF-DETR OBJECT DEDUP")
    print("=" * 72)
    print(f"input             : {input_path}")
    print(f"policy            : {args.policy}")
    print(f"original crops    : {len(crops):,}")
    print("hard dedup        : exact detection_id only")
    print("semantic duplicate: preserve points + duplicate_group_id")
    print("person bbox/DINO  : disabled")
    print("ambiguous         : preserve points + ambiguous_group_id")

    deduped, report = deduplicate(
        crops,
        cfg,
        policy_name=args.policy,
        policy=policy,
        dino_batch_size=args.dino_batch_size,
        dino_outer_batch_size=args.dino_outer_batch_size,
        dino_device=args.dino_device,
    )

    # Preserve every top-level input field. Replace only crops and add a compact
    # dedup summary. This avoids throwing away upstream RF-DETR statistics.
    dedup_data = copy.deepcopy(data)
    dedup_data["crops"] = deduped
    dedup_data["filtered"] = filtered
    dedup_data["dedup"] = {
        "mode": report["mode"],
        "policy_name": report["policy_name"],
        "original_crop_count": report["original_crop_count"],
        "deduped_crop_count": report["deduped_crop_count"],
        "removed_count": report["removed_count"],
        "exact_detection_id_removed": report["exact_detection_id_removed"],
        "duplicate_group_count": report["duplicate_group_count"],
        "duplicate_grouped_detection_count": report["duplicate_grouped_detection_count"],
        "ambiguous_group_count": report["ambiguous_group_count"],
        "ambiguous_detection_count": report["ambiguous_detection_count"],
    }

    if output_path != input_path:
        _atomic_write_json(
            output_path,
            dedup_data,
        )

    backup = None

    if args.in_place:
        backup = _backup_path(
            input_path
        )

        shutil.copy2(
            input_path,
            backup,
        )

        _atomic_write_json(
            input_path,
            dedup_data,
        )

        report["backup"] = str(backup)

    _atomic_write_json(
        report_path,
        report,
    )

    print()
    print("=" * 72)
    print("RF-DETR GEOMETRY + DINO DEDUP COMPLETED")
    print("=" * 72)
    print(f"original crops     : {report['original_crop_count']:,}")
    print(f"deduped crops      : {report['deduped_crop_count']:,}")
    print(f"hard removed       : {report['removed_count']:,}")
    print(f"  exact ID         : {report['exact_detection_id_removed']:,}")
    print(f"semantic removed   : 0")
    print(f"candidate pairs    : {report['candidate_pair_count']:,}")
    print(f"DINO detections    : {report['dino_candidate_detection_count']:,}")
    print(f"duplicate pairs    : {report['duplicate_pair_count']:,}")
    print(f"duplicate groups   : {report['duplicate_group_count']:,}")
    print(f"grouped points     : {report['duplicate_grouped_detection_count']:,}")
    print(f"ambiguous pairs    : {report['ambiguous_pair_count']:,}")
    print(f"separate pairs     : {report['separate_candidate_pair_count']:,}")
    print(f"ambiguous groups   : {report['ambiguous_group_count']:,}")
    print(f"ambiguous points   : {report['ambiguous_detection_count']:,}")

    if backup is not None:
        print(f"backup             : {backup}")
        print("input JSON was replaced with the deduplicated result.")
    else:
        print(f"output             : {output_path}")

    print(f"report             : {report_path}")
    print("Semantic duplicate crop JPG files were not deleted.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
