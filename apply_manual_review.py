from __future__ import annotations

"""
Safely apply manual review decisions under the Hybrid-C duplicate policy.

Hybrid-C contract
-----------------
The preceding dedup_objects.py stage has already separated duplicate handling into:

1) Hard dedup
   - Only exactly duplicated detection_id values are physically removed.
   - This prevents Qdrant point-ID overwrite.

2) Soft semantic duplicate grouping
   - Geometry + DINO duplicate judgements keep every detection as an independent
     crop / embedding / Qdrant point.
   - Related detections share duplicate_group_id.

3) Ambiguous relations
   - Uncertain pairs remain independent and are sent to manual review.

This file applies the manual decisions WITHOUT deleting any additional point.

Inputs
------
1) filter_stats_dedup.json
   - Hybrid-C automatic dedup/grouping output
2) dedup_report.json
   - Hybrid-C automatic audit report
3) manual_review.json
   - decisions created by review_ambiguous.py
4) pipeline.yaml
   - used to enforce the "never semantic-group person" invariant

Outputs
-------
1) filter_stats_reviewed.json
   - reviewed metadata; the input file is never modified
2) manual_review_apply_report.json
   - detailed audit trail

Manual decisions
----------------
duplicate:
    Keep both detections and place them in the same duplicate_group_id. Existing
    automatic duplicate groups are preserved and may be joined transitively.

separate:
    Keep both detections independent with respect to this reviewed edge. If the two
    detections are nevertheless connected by an existing/other duplicate relation,
    the script refuses to continue because the decisions are contradictory.

keep_ambiguous:
    Keep both detections and rebuild an ambiguous_group_id relation. If both are
    already in one duplicate group, the stronger duplicate relation wins and the
    redundant ambiguous edge is omitted.

Important safety properties
---------------------------
- Manual review removes ZERO crop records and ZERO crop JPG files.
- Existing Hybrid-C automatic duplicate groups are preserved.
- Manual duplicate judgements are non-destructive group assignments only.
- Person detections cannot enter duplicate/ambiguous semantic groups.
- Old automatic ambiguous groups are rebuilt only from manual keep_ambiguous.
- Reviews must be complete and match the current dedup_report.json.
- Duplicate detection_id values are rejected before output.
- Uses atomic JSON writes.
- Refuses to overwrite outputs unless --overwrite is explicitly supplied.
- Supports --dry-run for a no-write verification pass.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT = ROOT / "data" / "crops" / "filter_stats_dedup.json"
DEFAULT_DEDUP_REPORT = ROOT / "data" / "crops" / "dedup_report.json"
DEFAULT_REVIEW = ROOT / "data" / "crops" / "manual_review.json"
DEFAULT_OUTPUT = ROOT / "data" / "crops" / "filter_stats_reviewed.json"
DEFAULT_APPLY_REPORT = ROOT / "data" / "crops" / "manual_review_apply_report.json"
DEFAULT_CONFIG = ROOT / "pipeline.yaml"

PERSON_LABELS_FALLBACK = {"person", "pedestrian", "people", "human"}
ALLOWED_DECISIONS = {"duplicate", "separate", "keep_ambiguous"}


# =============================================================================
# Generic helpers
# =============================================================================

def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


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
            f"Invalid confidence: detection_id={record.get('detection_id')!r}, "
            f"value={value!r}"
        ) from exc

    if not math.isfinite(score):
        raise ValueError(
            f"Non-finite confidence: detection_id={record.get('detection_id')!r}"
        )
    return score


def _detection_id(record: Dict[str, Any]) -> str:
    value = record.get("detection_id")
    if value is None or not str(value).strip():
        raise ValueError("Record is missing detection_id.")
    return str(value).strip()


def _image_id(record: Dict[str, Any]) -> str:
    value = record.get("image_id")
    if value is None or not str(value).strip():
        raise ValueError(
            f"Missing image_id: detection_id={record.get('detection_id')!r}"
        )
    return str(value).strip().replace("\\", "/")


def _pair_id_from_values(image_id: str, a_id: str, b_id: str) -> str:
    key = str(image_id) + "|" + "|".join(sorted([str(a_id), str(b_id)]))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]


def _pair_id(pair: Dict[str, Any]) -> str:
    a = pair.get("a") or {}
    b = pair.get("b") or {}
    return _pair_id_from_values(
        str(pair.get("image_id") or ""),
        str(a.get("detection_id") or ""),
        str(b.get("detection_id") or ""),
    )


def _pair_endpoints(pair: Dict[str, Any]) -> Tuple[str, str, str]:
    a = pair.get("a") or {}
    b = pair.get("b") or {}
    image_id = str(pair.get("image_id") or "").strip().replace("\\", "/")
    a_id = str(a.get("detection_id") or "").strip()
    b_id = str(b.get("detection_id") or "").strip()

    if not image_id or not a_id or not b_id:
        raise ValueError("Malformed ambiguous pair in dedup_report.json.")

    return image_id, a_id, b_id


def _load_person_labels(config_path: Path) -> set[str]:
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from config import PipelineConfig  # type: ignore

        cfg = PipelineConfig.load(str(config_path))
        labels = getattr(cfg, "person_labels", None) or PERSON_LABELS_FALLBACK
        return {str(x).strip().lower() for x in labels}
    except Exception as exc:
        raise RuntimeError(
            "Failed to load person_labels from pipeline.yaml. "
            "This check is intentionally mandatory for safety."
        ) from exc


def _represented_ids(record: Dict[str, Any]) -> List[str]:
    """Logical IDs represented by this record after hard exact-ID dedup.

    Under Hybrid-C, merged_detection_ids is hard-dedup lineage only. Semantic
    duplicates remain separate current records and are never placed here.
    """
    values: List[str] = []

    raw = record.get("merged_detection_ids")
    if isinstance(raw, list):
        for value in raw:
            if value is not None and str(value).strip():
                values.append(str(value).strip())

    values.append(_detection_id(record))
    return sorted(set(values))


def _refresh_relation_status(record: Dict[str, Any]) -> None:
    """Derive duplicate_status from non-destructive relation metadata only."""
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


def _clear_soft_relation_metadata(record: Dict[str, Any]) -> None:
    """Clear soft relation fields before deterministic reconstruction.

    Hard-dedup lineage fields such as merged_detection_ids, class_candidates,
    class_conflict and hard_dedup_count are intentionally preserved.
    """
    record.pop("duplicate_group_id", None)
    record.pop("ambiguous_group_id", None)
    record.pop("duplicate_status", None)


def _summary(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "index": int(index),
        "image_id": _image_id(record),
        "detection_id": _detection_id(record),
        "class_name": record.get("class_name"),
        "confidence": _confidence(record),
        "duplicate_group_id": record.get("duplicate_group_id"),
        "ambiguous_group_id": record.get("ambiguous_group_id"),
        "duplicate_status": record.get("duplicate_status"),
        "merged_detection_ids": _represented_ids(record),
    }


# =============================================================================
# Disjoint set
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


def _stable_duplicate_group_id(
    image_id: str,
    detection_ids: Sequence[str],
) -> str:
    return _stable_group_id("duplicate", image_id, detection_ids)


def _stable_ambiguous_group_id(
    image_id: str,
    detection_ids: Sequence[str],
) -> str:
    return _stable_group_id("ambiguous", image_id, detection_ids)


# =============================================================================
# Validation / resolution
# =============================================================================

@dataclass(frozen=True)
class ReviewedPair:
    pair_id: str
    image_id: str
    a_id: str
    b_id: str
    decision: str


def _load_and_validate_reviews(
    dedup_report: Dict[str, Any],
    review_data: Dict[str, Any],
) -> Tuple[List[ReviewedPair], Dict[str, Dict[str, Any]]]:
    # Refuse old A-style / provenance-free manual review files. Pair IDs alone
    # are not sufficient because an old file can coincidentally share pair IDs
    # with a newly generated Hybrid-C report.
    format_version = int(review_data.get("format_version", 0) or 0)
    dedup_mode = str(review_data.get("dedup_mode") or "").strip()
    semantics = review_data.get("review_semantics") or {}
    source_report_sha256 = str(
        review_data.get("source_report_sha256") or ""
    ).strip()

    if format_version < 3:
        raise RuntimeError(
            "manual_review.json is from an older review format. "
            f"Expected format_version >= 3, got {format_version}. "
            "Re-run review_ambiguous.py for the current Hybrid-C report."
        )
    if dedup_mode != "hybrid_c_hard_exact_soft_semantic":
        raise RuntimeError(
            "manual_review.json dedup_mode is not Hybrid-C: "
            f"{dedup_mode!r}"
        )
    if not isinstance(semantics, dict) or semantics.get("semantic_point_deletion") is not False:
        raise RuntimeError(
            "manual_review.json does not explicitly declare non-destructive "
            "Hybrid-C review semantics."
        )
    if not source_report_sha256:
        raise RuntimeError(
            "manual_review.json is missing source_report_sha256 provenance."
        )

    raw_pairs = dedup_report.get("ambiguous_pairs")
    if not isinstance(raw_pairs, list):
        raise TypeError('dedup_report.json: "ambiguous_pairs" must be a list.')

    report_pairs: Dict[str, Dict[str, Any]] = {}
    for pair in raw_pairs:
        if not isinstance(pair, dict):
            raise TypeError("dedup_report.json contains a non-object ambiguous pair.")

        pid = _pair_id(pair)
        if pid in report_pairs:
            raise RuntimeError(f"Duplicate ambiguous pair_id in report: {pid}")
        report_pairs[pid] = pair

    raw_reviews = review_data.get("reviews")
    if not isinstance(raw_reviews, dict):
        raise TypeError('manual_review.json: "reviews" must be an object.')

    unknown = sorted(set(raw_reviews) - set(report_pairs))
    missing = sorted(set(report_pairs) - set(raw_reviews))

    if unknown:
        raise RuntimeError(
            "manual_review.json contains pair IDs not present in the current "
            f"dedup_report.json. The review may be stale. Examples: {unknown[:5]}"
        )

    if missing:
        raise RuntimeError(
            "Manual review is incomplete. Refusing to apply partial decisions. "
            f"Missing {len(missing):,} pair(s). Examples: {missing[:5]}"
        )

    declared_total = review_data.get("total_ambiguous_pairs")
    if declared_total is not None and int(declared_total) != len(report_pairs):
        raise RuntimeError(
            "manual_review.json total_ambiguous_pairs does not match "
            "dedup_report.json."
        )

    declared_pending = review_data.get("pending_count")
    if declared_pending is not None and int(declared_pending) != 0:
        raise RuntimeError(
            f"manual_review.json still reports pending_count={declared_pending}."
        )

    reviewed: List[ReviewedPair] = []

    for pid, pair in report_pairs.items():
        review = raw_reviews[pid]
        if not isinstance(review, dict):
            raise TypeError(f"Review {pid} is not an object.")

        decision = str(review.get("decision") or "").strip()
        if decision not in ALLOWED_DECISIONS:
            raise RuntimeError(
                f"Invalid decision for {pid}: {decision!r}"
            )

        image_id, a_id, b_id = _pair_endpoints(pair)

        review_image = str(review.get("image_id") or "").strip().replace("\\", "/")
        review_a = str(review.get("a_detection_id") or "").strip()
        review_b = str(review.get("b_detection_id") or "").strip()

        if review_image and review_image != image_id:
            raise RuntimeError(
                f"Review image_id mismatch for pair {pid}: "
                f"{review_image!r} != {image_id!r}"
            )

        if review_a and review_b:
            if {review_a, review_b} != {a_id, b_id}:
                raise RuntimeError(
                    f"Review detection IDs mismatch for pair {pid}."
                )

        recomputed = _pair_id_from_values(image_id, a_id, b_id)
        if recomputed != pid:
            raise RuntimeError(f"Internal pair_id mismatch for {pid}")

        reviewed.append(
            ReviewedPair(
                pair_id=pid,
                image_id=image_id,
                a_id=a_id,
                b_id=b_id,
                decision=decision,
            )
        )

    return reviewed, report_pairs


def _build_detection_resolution(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, int]:
    resolution: Dict[str, int] = {}

    for idx, record in enumerate(records):
        for did in _represented_ids(record):
            old = resolution.get(did)
            if old is not None and old != idx:
                raise RuntimeError(
                    "A detection_id is represented by multiple current records: "
                    f"{did!r} -> indices {old}, {idx}"
                )
            resolution[did] = idx

    return resolution


# =============================================================================
# Core application
# =============================================================================

def apply_manual_review(
    source_data: Dict[str, Any],
    dedup_report: Dict[str, Any],
    review_data: Dict[str, Any],
    *,
    person_labels: set[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    crops = source_data.get("crops")
    if not isinstance(crops, list):
        raise TypeError('filter_stats_dedup.json: "crops" must be a list.')
    if not crops:
        raise RuntimeError("No crops found in filter_stats_dedup.json.")

    # Fail closed against accidentally applying Hybrid-C review semantics to an
    # old destructive dedup report.
    mode = str(dedup_report.get("mode") or "").strip()
    if mode != "hybrid_c_hard_exact_soft_semantic":
        raise RuntimeError(
            "dedup_report.json is not a Hybrid-C report. "
            f"expected mode='hybrid_c_hard_exact_soft_semantic', got {mode!r}. "
            "Re-run the updated dedup_objects.py before applying manual review."
        )

    work: List[Dict[str, Any]] = [
        copy.deepcopy(record) for record in crops
    ]

    # ------------------------------------------------------------------
    # 0) Validate input identity / report freshness.
    # ------------------------------------------------------------------
    current_ids = [_detection_id(r) for r in work]
    duplicate_current_ids = [
        did
        for did, count in Counter(current_ids).items()
        if count > 1
    ]
    if duplicate_current_ids:
        raise RuntimeError(
            "filter_stats_dedup.json contains duplicate current detection_id "
            f"values. Examples: {duplicate_current_ids[:5]}"
        )

    expected_count = dedup_report.get("deduped_crop_count")
    if expected_count is not None and int(expected_count) != len(work):
        raise RuntimeError(
            "filter_stats_dedup.json crop count does not match the current "
            "dedup_report.json. Refusing to apply a possibly stale review. "
            f"input={len(work):,}, report={int(expected_count):,}"
        )

    # Hybrid-C's automatic stage may only hard-remove exact duplicated IDs.
    removed_count = int(dedup_report.get("removed_count", 0) or 0)
    exact_removed = int(dedup_report.get("exact_detection_id_removed", 0) or 0)
    same_removed = int(dedup_report.get("same_class_removed", 0) or 0)
    cross_removed = int(dedup_report.get("cross_class_removed", 0) or 0)
    if removed_count != exact_removed or same_removed != 0 or cross_removed != 0:
        raise RuntimeError(
            "Hybrid-C invariant violated in dedup_report.json: only exact "
            "detection_id duplicates may be physically removed."
        )

    reviewed, report_pairs = _load_and_validate_reviews(
        dedup_report,
        review_data,
    )

    # Manual semantic review must never contain a person pair.
    for pair in report_pairs.values():
        a = pair.get("a") or {}
        b = pair.get("b") or {}
        if _label(a) in person_labels or _label(b) in person_labels:
            raise RuntimeError(
                "Safety invariant violated: person detection found in "
                "ambiguous review set."
            )

    resolution = _build_detection_resolution(work)

    # Resolve every reviewed endpoint to one current record. Hard exact-ID dedup
    # may have lineage metadata, but semantic duplicates still have independent IDs.
    resolved_pairs: List[Tuple[ReviewedPair, int, int]] = []
    for item in reviewed:
        if item.a_id not in resolution:
            raise RuntimeError(
                "Reviewed detection cannot be resolved in current metadata: "
                f"{item.a_id}"
            )
        if item.b_id not in resolution:
            raise RuntimeError(
                "Reviewed detection cannot be resolved in current metadata: "
                f"{item.b_id}"
            )

        idx_a = resolution[item.a_id]
        idx_b = resolution[item.b_id]

        image_a = _image_id(work[idx_a])
        image_b = _image_id(work[idx_b])
        if image_a != image_b or image_a != item.image_id:
            raise RuntimeError(
                "Reviewed pair resolved across inconsistent images: "
                f"pair={item.pair_id}, review={item.image_id}, "
                f"A={image_a}, B={image_b}"
            )

        resolved_pairs.append((item, idx_a, idx_b))

    # ------------------------------------------------------------------
    # 1) Capture existing automatic duplicate groups, then rebuild the full
    #    duplicate graph = automatic groups + manual duplicate decisions.
    # ------------------------------------------------------------------
    automatic_groups: Dict[str, List[int]] = defaultdict(list)
    for idx, record in enumerate(work):
        gid = record.get("duplicate_group_id")
        if gid is not None and str(gid).strip():
            automatic_groups[str(gid).strip()].append(idx)

    # Validate automatic groups before touching metadata.
    for gid, members in automatic_groups.items():
        if len(members) < 2:
            raise RuntimeError(
                f"Malformed automatic duplicate group {gid!r}: "
                "fewer than two members."
            )

        images = {_image_id(work[idx]) for idx in members}
        if len(images) != 1:
            raise RuntimeError(
                f"Automatic duplicate group {gid!r} spans multiple images."
            )

        person_members = [
            _detection_id(work[idx])
            for idx in members
            if _label(work[idx]) in person_labels
        ]
        if person_members:
            raise RuntimeError(
                "Safety invariant violated: person detection found in automatic "
                f"duplicate group {gid!r}. Examples: {person_members[:5]}"
            )

    # Clear all soft relation metadata so final IDs are rebuilt deterministically.
    for record in work:
        _clear_soft_relation_metadata(record)

    dsu_dup = _DisjointSet(range(len(work)))

    # Restore automatic group connectivity.
    for members in automatic_groups.values():
        head = members[0]
        for idx in members[1:]:
            dsu_dup.union(head, idx)

    manual_duplicate_edges: List[Tuple[int, int, str]] = []
    duplicate_pairs_already_same_record = 0

    for item, idx_a, idx_b in resolved_pairs:
        if item.decision != "duplicate":
            continue

        if idx_a == idx_b:
            # This should be rare after Hybrid-C because semantic duplicates are
            # separate records, but hard exact-ID lineage can resolve this way.
            duplicate_pairs_already_same_record += 1
            continue

        manual_duplicate_edges.append((idx_a, idx_b, item.pair_id))
        dsu_dup.union(idx_a, idx_b)

    duplicate_components: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(work)):
        duplicate_components[dsu_dup.find(idx)].append(idx)

    final_duplicate_components = [
        members
        for members in duplicate_components.values()
        if len(members) > 1
    ]

    manual_pair_roots = {
        dsu_dup.find(idx_a)
        for idx_a, _idx_b, _pair_id_value in manual_duplicate_edges
    }

    automatic_member_set = {
        idx
        for members in automatic_groups.values()
        for idx in members
    }

    duplicate_groups: List[Dict[str, Any]] = []
    duplicate_detection_count = 0
    manual_duplicate_component_count = 0

    for members in sorted(final_duplicate_components, key=lambda xs: min(xs)):
        images = {_image_id(work[idx]) for idx in members}
        if len(images) != 1:
            raise RuntimeError(
                "Final duplicate component spans multiple source images."
            )

        person_members = [
            _detection_id(work[idx])
            for idx in members
            if _label(work[idx]) in person_labels
        ]
        if person_members:
            raise RuntimeError(
                "Safety invariant violated: person detection entered a semantic "
                f"duplicate group. Examples: {person_members[:5]}"
            )

        image_id = next(iter(images))
        ids = [_detection_id(work[idx]) for idx in members]
        group_id = _stable_duplicate_group_id(image_id, ids)
        root = dsu_dup.find(members[0])

        has_manual = root in manual_pair_roots
        has_automatic = any(idx in automatic_member_set for idx in members)

        if has_manual:
            manual_duplicate_component_count += 1

        if has_manual and has_automatic:
            group_source = "automatic+manual"
        elif has_manual:
            group_source = "manual"
        else:
            group_source = "automatic"

        for idx in members:
            work[idx]["duplicate_group_id"] = group_id

        duplicate_detection_count += len(members)
        duplicate_groups.append(
            {
                "duplicate_group_id": group_id,
                "image_id": image_id,
                "source": group_source,
                "member_indices": sorted(members),
                "detection_ids": sorted(ids),
            }
        )

    duplicate_groups.sort(key=lambda x: x["duplicate_group_id"])

    # A manual 'separate' decision is authoritative. If it is connected anyway by
    # automatic/manual duplicate edges, the graph is contradictory; do not hide it.
    contradictory_separate_pairs: List[str] = []
    for item, idx_a, idx_b in resolved_pairs:
        if item.decision != "separate":
            continue
        if dsu_dup.find(idx_a) == dsu_dup.find(idx_b):
            contradictory_separate_pairs.append(item.pair_id)

    if contradictory_separate_pairs:
        raise RuntimeError(
            "Conflicting duplicate graph: at least one pair marked 'separate' "
            "is still connected through automatic/manual duplicate relations. "
            "No output was written. Examples: "
            f"{contradictory_separate_pairs[:5]}"
        )

    # ------------------------------------------------------------------
    # 2) Rebuild ambiguous groups ONLY from keep_ambiguous decisions.
    # ------------------------------------------------------------------
    ambiguous_edges: List[Tuple[int, int, str]] = []
    keep_ambiguous_pairs_inside_duplicate_group = 0

    for item, idx_a, idx_b in resolved_pairs:
        if item.decision != "keep_ambiguous":
            continue

        if idx_a == idx_b or dsu_dup.find(idx_a) == dsu_dup.find(idx_b):
            # The duplicate relation is stronger than "uncertain". Keep every
            # point, but do not create a redundant ambiguous relation inside one
            # duplicate group.
            keep_ambiguous_pairs_inside_duplicate_group += 1
            continue

        ambiguous_edges.append((idx_a, idx_b, item.pair_id))

    dsu_amb = _DisjointSet(range(len(work)))

    for idx_a, idx_b, pair_id in ambiguous_edges:
        if _image_id(work[idx_a]) != _image_id(work[idx_b]):
            raise RuntimeError(
                f"Ambiguous edge {pair_id} spans multiple images."
            )
        if _label(work[idx_a]) in person_labels or _label(work[idx_b]) in person_labels:
            raise RuntimeError(
                f"Safety invariant violated: person in ambiguous edge {pair_id}."
            )
        dsu_amb.union(idx_a, idx_b)

    ambiguous_components: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(work)):
        ambiguous_components[dsu_amb.find(idx)].append(idx)

    ambiguous_groups: List[Dict[str, Any]] = []
    ambiguous_detection_count = 0

    for members in ambiguous_components.values():
        if len(members) <= 1:
            continue

        images = {_image_id(work[idx]) for idx in members}
        if len(images) != 1:
            raise RuntimeError(
                "Final ambiguous component spans multiple source images."
            )

        person_members = [
            _detection_id(work[idx])
            for idx in members
            if _label(work[idx]) in person_labels
        ]
        if person_members:
            raise RuntimeError(
                "Safety invariant violated: person detection entered an ambiguous "
                f"group. Examples: {person_members[:5]}"
            )

        image_id = next(iter(images))
        ids = [_detection_id(work[idx]) for idx in members]
        group_id = _stable_ambiguous_group_id(image_id, ids)

        for idx in members:
            work[idx]["ambiguous_group_id"] = group_id

        ambiguous_detection_count += len(members)
        ambiguous_groups.append(
            {
                "ambiguous_group_id": group_id,
                "image_id": image_id,
                "member_indices": sorted(members),
                "detection_ids": sorted(ids),
            }
        )

    ambiguous_groups.sort(key=lambda x: x["ambiguous_group_id"])

    # Derive status only after both graphs are final.
    for record in work:
        _refresh_relation_status(record)

    reviewed_crops = work

    # ------------------------------------------------------------------
    # 3) Final Hybrid-C invariants.
    # ------------------------------------------------------------------
    final_current_ids = [_detection_id(r) for r in reviewed_crops]
    final_counts = Counter(final_current_ids)
    duplicate_ids_left = [
        did
        for did, count in final_counts.items()
        if count > 1
    ]
    if duplicate_ids_left:
        raise RuntimeError(
            "Duplicate current detection_id values remain after manual apply. "
            f"Examples: {duplicate_ids_left[:5]}"
        )

    # Manual review under Hybrid-C is strictly non-destructive.
    if len(reviewed_crops) != len(work) or len(reviewed_crops) != len(crops):
        raise RuntimeError(
            "Hybrid-C invariant violated: manual review changed crop count."
        )

    if final_current_ids != current_ids:
        raise RuntimeError(
            "Hybrid-C invariant violated: manual review changed current "
            "detection identity/order."
        )

    final_resolution = _build_detection_resolution(reviewed_crops)
    lost_current_ids = [
        did
        for did in current_ids
        if did not in final_resolution
    ]
    if lost_current_ids:
        raise RuntimeError(
            "Manual application lost detection identities. "
            f"Examples: {lost_current_ids[:5]}"
        )

    # No person may carry semantic soft-relation metadata.
    person_relation_ids = [
        _detection_id(record)
        for record in reviewed_crops
        if _label(record) in person_labels
        and (
            record.get("duplicate_group_id") is not None
            or record.get("ambiguous_group_id") is not None
        )
    ]
    if person_relation_ids:
        raise RuntimeError(
            "Safety invariant violated: person detection received semantic "
            f"group metadata. Examples: {person_relation_ids[:5]}"
        )

    # Every duplicate/ambiguous group must contain at least two records and be
    # reproducible from the actual output metadata.
    output_duplicate_members: Dict[str, List[str]] = defaultdict(list)
    output_ambiguous_members: Dict[str, List[str]] = defaultdict(list)
    for record in reviewed_crops:
        did = _detection_id(record)
        dup_gid = record.get("duplicate_group_id")
        amb_gid = record.get("ambiguous_group_id")
        if dup_gid is not None:
            output_duplicate_members[str(dup_gid)].append(did)
        if amb_gid is not None:
            output_ambiguous_members[str(amb_gid)].append(did)

    malformed_duplicate_groups = {
        gid: ids
        for gid, ids in output_duplicate_members.items()
        if len(ids) < 2
    }
    malformed_ambiguous_groups = {
        gid: ids
        for gid, ids in output_ambiguous_members.items()
        if len(ids) < 2
    }
    if malformed_duplicate_groups:
        raise RuntimeError(
            "Malformed final duplicate groups with fewer than two members: "
            f"{list(malformed_duplicate_groups)[:5]}"
        )
    if malformed_ambiguous_groups:
        raise RuntimeError(
            "Malformed final ambiguous groups with fewer than two members: "
            f"{list(malformed_ambiguous_groups)[:5]}"
        )

    decision_counts = Counter(x.decision for x in reviewed)

    out_data = copy.deepcopy(source_data)
    out_data["crops"] = reviewed_crops
    out_data["manual_review"] = {
        "mode": "hybrid_c_non_destructive",
        "applied": True,
        "reviewed_pair_count": len(reviewed),
        "duplicate_pair_count": int(decision_counts["duplicate"]),
        "separate_pair_count": int(decision_counts["separate"]),
        "keep_ambiguous_pair_count": int(decision_counts["keep_ambiguous"]),
        "manual_duplicate_component_count": manual_duplicate_component_count,
        "manual_removed_point_count": 0,
        "final_crop_count": len(reviewed_crops),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_grouped_detection_count": duplicate_detection_count,
        "ambiguous_group_count": len(ambiguous_groups),
        "ambiguous_detection_count": ambiguous_detection_count,
    }

    report = {
        "mode": "hybrid_c_non_destructive_manual_review",
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_crop_count": len(work),
        "final_crop_count": len(reviewed_crops),
        "manual_removed_point_count": 0,
        "reviewed_pair_count": len(reviewed),
        "decision_counts": {
            "duplicate": int(decision_counts["duplicate"]),
            "separate": int(decision_counts["separate"]),
            "keep_ambiguous": int(decision_counts["keep_ambiguous"]),
        },
        "automatic_duplicate_group_count_before_review": len(automatic_groups),
        "manual_duplicate_component_count": manual_duplicate_component_count,
        "duplicate_pairs_already_same_current_record": (
            duplicate_pairs_already_same_record
        ),
        "keep_ambiguous_pairs_inside_duplicate_group": (
            keep_ambiguous_pairs_inside_duplicate_group
        ),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_grouped_detection_count": duplicate_detection_count,
        "ambiguous_group_count": len(ambiguous_groups),
        "ambiguous_detection_count": ambiguous_detection_count,
        "duplicate_groups": duplicate_groups,
        "ambiguous_groups": ambiguous_groups,
        # Compatibility key retained deliberately; Hybrid-C performs no manual merge.
        "manual_merges": [],
        "invariants": {
            "input_current_detection_ids_unique": True,
            "hybrid_c_dedup_report": True,
            "automatic_semantic_removals_zero": True,
            "all_review_pairs_present": True,
            "review_pair_set_matches_current_dedup_report": True,
            "manual_review_source_report_hash_matches": True,
            "no_person_pairs_in_review_set": True,
            "manual_point_removal_zero": True,
            "all_input_current_detection_ids_still_represented": True,
            "final_current_detection_ids_unique": True,
            "final_crop_count_equals_input": True,
            "detection_identity_and_order_preserved": True,
            "no_person_semantic_groups": True,
            "no_contradictory_separate_vs_duplicate_graph": True,
        },
    }

    return out_data, report


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply manual ambiguous-pair reviews under Hybrid-C without "
            "deleting semantic duplicate points."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--dedup-report", default=str(DEFAULT_DEDUP_REPORT))
    parser.add_argument("--review", default=str(DEFAULT_REVIEW))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--apply-report", default=str(DEFAULT_APPLY_REPORT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and compute the result, but write no files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output/apply-report files.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    dedup_report_path = Path(args.dedup_report).resolve()
    review_path = Path(args.review).resolve()
    output_path = Path(args.output).resolve()
    apply_report_path = Path(args.apply_report).resolve()
    config_path = Path(args.config).resolve()

    _require_file(input_path, "dedup input")
    _require_file(dedup_report_path, "dedup report")
    _require_file(review_path, "manual review")
    _require_file(config_path, "pipeline config")

    # Never allow accidental replacement of source evidence/inputs.
    protected = {
        input_path,
        dedup_report_path,
        review_path,
        config_path,
    }
    if output_path in protected:
        raise ValueError(
            "--output must be a new file and may not overwrite an input."
        )
    if apply_report_path in protected or apply_report_path == output_path:
        raise ValueError(
            "--apply-report must be a separate new file."
        )

    if not args.dry_run and not args.overwrite:
        existing = [
            str(p)
            for p in (output_path, apply_report_path)
            if p.exists()
        ]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing output. "
                "Use --overwrite only after confirming the files are expendable: "
                + ", ".join(existing)
            )

    source_data = _read_json(input_path)
    dedup_report = _read_json(dedup_report_path)
    review_data = _read_json(review_path)

    if not isinstance(source_data, dict):
        raise TypeError("filter_stats_dedup.json must be a JSON object.")
    if not isinstance(dedup_report, dict):
        raise TypeError("dedup_report.json must be a JSON object.")
    if not isinstance(review_data, dict):
        raise TypeError("manual_review.json must be a JSON object.")

    current_report_sha256 = _sha256(dedup_report_path)
    reviewed_report_sha256 = str(
        review_data.get("source_report_sha256") or ""
    ).strip()
    if reviewed_report_sha256 != current_report_sha256:
        raise RuntimeError(
            "manual_review.json was not created from the current dedup_report.json. "
            "Refusing to apply stale review decisions. "
            f"review_hash={reviewed_report_sha256!r}, "
            f"current_hash={current_report_sha256!r}"
        )

    person_labels = _load_person_labels(config_path)

    reviewed_data, report = apply_manual_review(
        source_data,
        dedup_report,
        review_data,
        person_labels=person_labels,
    )

    report["files"] = {
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "dedup_report": str(dedup_report_path),
        "dedup_report_sha256": _sha256(dedup_report_path),
        "manual_review": str(review_path),
        "manual_review_sha256": _sha256(review_path),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "output": str(output_path),
        "apply_report": str(apply_report_path),
    }

    print()
    print("=" * 72)
    print("MANUAL REVIEW APPLY CHECK")
    print("=" * 72)
    print(f"input crops          : {report['input_crop_count']:,}")
    print(f"reviewed pairs       : {report['reviewed_pair_count']:,}")
    print(
        "  duplicate          : "
        f"{report['decision_counts']['duplicate']:,}"
    )
    print(
        "  separate           : "
        f"{report['decision_counts']['separate']:,}"
    )
    print(
        "  keep ambiguous     : "
        f"{report['decision_counts']['keep_ambiguous']:,}"
    )
    print(
        f"manual dup components: "
        f"{report['manual_duplicate_component_count']:,}"
    )
    print(f"points removed       : 0  (Hybrid-C invariant)")
    print(f"final crops          : {report['final_crop_count']:,}")
    print(
        f"duplicate groups     : "
        f"{report['duplicate_group_count']:,}"
    )
    print(
        f"grouped points       : "
        f"{report['duplicate_grouped_detection_count']:,}"
    )
    print(
        f"ambiguous groups     : "
        f"{report['ambiguous_group_count']:,}"
    )
    print(
        f"ambiguous points     : "
        f"{report['ambiguous_detection_count']:,}"
    )

    if args.dry_run:
        print()
        print("DRY RUN PASS - validation succeeded; no files were written.")
        return 0

    _atomic_write_json(output_path, reviewed_data)
    _atomic_write_json(apply_report_path, report)

    print()
    print("PASS - Hybrid-C manual review applied without semantic point deletion.")
    print(f"output       : {output_path}")
    print(f"apply report : {apply_report_path}")
    print("Original JSON/report/review files and crop JPGs were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
