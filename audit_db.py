from __future__ import annotations

"""
audit_db.py

Read-only integrity audit for the RF-DETR -> embedding -> Qdrant pipeline.

Checks:
- source JSON detection_id uniqueness
- expected person/object routing counts
- Qdrant collections / named vector schema / dimensions / distance
- payload indexes
- exact expected point IDs vs actual Qdrant point IDs
- missing / extra / logically duplicated points
- person/object collection contamination
- required payload fields and source-vs-Qdrant payload consistency
- per-point named vector names / dimensions / NaN/Inf / zero vectors
- image/video payload contract
- Hybrid-C dedup metadata preservation:
  duplicate_status, duplicate_group_id, ambiguous_group_id, class_conflict,
  class_candidates, merged_detection_ids
- duplicate / ambiguous group membership preservation
- Hybrid-C safety invariants:
  only exact duplicated detection_id may be hard-removed upstream; semantic
  duplicates must remain as independent points and be related by duplicate_group_id

This script never modifies Qdrant.
"""

import argparse
import hashlib
import json
import math
import random
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from config import PipelineConfig
from rfdetr_adapter import make_detection_id


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "pipeline.yaml"
DEFAULT_STATS = ROOT / "data" / "crops" / "filter_stats_reviewed.json"

DEDUP_KEYS = (
    "duplicate_status",
    "duplicate_group_id",
    "ambiguous_group_id",
    "hard_dedup_count",
    "class_conflict",
    "class_candidates",
    "merged_detection_ids",
)

# detection_id is the canonical logical identity. crop_id is legacy/compatibility
# metadata and is intentionally optional; if present, later checks still require
# crop_id == detection_id and uniqueness.
REQUIRED_PAYLOAD_FIELDS = (
    "image_id",
    "frame_idx",
    "label",
    "is_person",
    "score",
    "bbox",
    "detection_id",
    "bbox_space",
    "media_type",
)

EXPECTED_PAYLOAD_INDEXES = {
    "is_person": "bool",
    "label": "keyword",
    "image_id": "keyword",
    "frame_idx": "integer",
}


@dataclass
class AuditReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def error(self, msg: str) -> None:
        self.errors.append(str(msg))

    def warn(self, msg: str) -> None:
        self.warnings.append(str(msg))

    @property
    def ok(self) -> bool:
        return not self.errors


def _value(v: Any) -> Any:
    return getattr(v, "value", v)


def _norm(v: Any) -> str:
    if v is None:
        return ""
    return str(_value(v)).strip().lower()


def _safe_examples(values: Iterable[Any], n: int = 5) -> List[Any]:
    out = []
    for v in values:
        out.append(v)
        if len(out) >= n:
            break
    return out


def _json_norm(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, Mapping):
        return {str(k): _json_norm(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_norm(x) for x in v]
    return v


def _float_close(a: Any, b: Any, atol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= atol
    except Exception:
        return False


def _bbox_equal(a: Any, b: Any, atol: float = 1e-6) -> bool:
    try:
        aa, bb = list(a), list(b)
    except Exception:
        return False
    return (
        len(aa) == 4
        and len(bb) == 4
        and all(_float_close(x, y, atol) for x, y in zip(aa, bb))
    )


def stable_point_id(collection: str, detection_id: str) -> str:
    key = f"{collection}|detection_id={detection_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def crop_path(record: Mapping[str, Any]) -> Optional[str]:
    for key in ("crop_path", "path"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def infer_media_type(record: Mapping[str, Any]) -> str:
    explicit = record.get("media_type")
    if explicit is not None:
        return str(explicit).strip().lower()

    has_video = any(
        record.get(k) is not None
        for k in ("video", "track_key", "timestamp")
    )
    has_track = record.get("track_id") is not None
    return "video" if (has_video or has_track) else "image"


def collection_names(cfg: PipelineConfig) -> Dict[str, str]:
    return {
        "person": cfg.person_collection(),
        "object": cfg.object_collection(),
    }


def expected_vectors(cfg: PipelineConfig, scope: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name, spec in cfg.retrievers.items():
        s = str(spec.scope).strip().lower()
        if s == "all" or s == scope:
            out[name] = int(spec.dim)
    return out


def record_is_person(record: Mapping[str, Any], person_labels: Set[str]) -> bool:
    return str(record.get("class_name", "")).strip().lower() in person_labels


def load_source(
    stats_path: Path,
    cfg: PipelineConfig,
    report: AuditReport,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    with stats_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not isinstance(data.get("crops"), list):
        raise TypeError('stats JSON must be {"crops": [...]}')

    crops: List[Dict[str, Any]] = data["crops"]
    person_labels = {str(x).lower() for x in cfg.person_labels}
    names = collection_names(cfg)

    seen_detection: Counter[str] = Counter()
    expected_by_point: Dict[str, Dict[str, Any]] = {}
    expected_counts = Counter()
    source_dup = defaultdict(list)
    source_ambig = defaultdict(list)
    dedup_counts = Counter()

    for i, record in enumerate(crops):
        if not isinstance(record, dict):
            report.error(f"source[{i}] is not an object")
            continue

        frame_idx = int(record.get("frame_idx", 0))
        detection_id = make_detection_id(record, frame_idx)
        seen_detection[detection_id] += 1

        scope = "person" if record_is_person(record, person_labels) else "object"
        expected_counts[scope] += 1
        point_id = stable_point_id(names[scope], detection_id)

        if point_id in expected_by_point:
            report.error(
                f"multiple source records map to same expected point_id={point_id}"
            )

        expected_by_point[point_id] = {
            "scope": scope,
            "collection": names[scope],
            "detection_id": detection_id,
            "record": record,
        }

        dup_gid = record.get("duplicate_group_id")
        if dup_gid is not None and str(dup_gid).strip():
            source_dup[str(dup_gid).strip()].append(detection_id)
            if scope == "person":
                report.error(
                    f"{detection_id}: person detection has duplicate_group_id="
                    f"{dup_gid!r}; Hybrid-C forbids semantic person grouping"
                )

        amb_gid = record.get("ambiguous_group_id")
        if amb_gid is not None and str(amb_gid).strip():
            source_ambig[str(amb_gid).strip()].append(detection_id)
            if scope == "person":
                report.error(
                    f"{detection_id}: person detection has ambiguous_group_id="
                    f"{amb_gid!r}; Hybrid-C forbids bbox/DINO person grouping"
                )

        has_dup = bool(dup_gid is not None and str(dup_gid).strip())
        has_amb = bool(amb_gid is not None and str(amb_gid).strip())
        expected_status = (
            "grouped_ambiguous" if has_dup and has_amb
            else "grouped" if has_dup
            else "ambiguous" if has_amb
            else None
        )
        actual_status_raw = record.get("duplicate_status")
        actual_status = (
            str(actual_status_raw).strip().lower()
            if actual_status_raw is not None and str(actual_status_raw).strip()
            else None
        )
        if actual_status != expected_status:
            report.error(
                f"{detection_id}: duplicate_status inconsistent with relation metadata: "
                f"expected={expected_status!r} actual={actual_status!r}"
            )

        for key in DEDUP_KEYS:
            if key in record:
                dedup_counts[key] += 1

    dup = {k: v for k, v in seen_detection.items() if v > 1}
    if dup:
        report.error(
            f"duplicate detection_id in source: {len(dup)}; "
            f"examples={_safe_examples(dup.items())}"
        )

    malformed_dup_groups = {
        gid: ids for gid, ids in source_dup.items() if len(ids) < 2
    }
    malformed_amb_groups = {
        gid: ids for gid, ids in source_ambig.items() if len(ids) < 2
    }
    if malformed_dup_groups:
        report.error(
            "source duplicate_group_id with fewer than two members: "
            f"{len(malformed_dup_groups)}; "
            f"examples={_safe_examples(malformed_dup_groups.items())}"
        )
    if malformed_amb_groups:
        report.error(
            "source ambiguous_group_id with fewer than two members: "
            f"{len(malformed_amb_groups)}; "
            f"examples={_safe_examples(malformed_amb_groups.items())}"
        )

    # Every relation group is local to one logical source image. This matches the
    # Hybrid-C dedup/review graph and prevents accidental cross-image collapsing.
    image_by_detection = {
        make_detection_id(record, int(record.get("frame_idx", 0))):
            str(record.get("image_id", "")).strip().replace("\\", "/")
        for record in crops
        if isinstance(record, dict)
    }
    for kind, groups in (("duplicate", source_dup), ("ambiguous", source_ambig)):
        for gid, ids in groups.items():
            images = {image_by_detection.get(did, "") for did in ids}
            if len(images) != 1:
                report.error(
                    f"source {kind} group spans multiple image_id values: "
                    f"group={gid!r}, images={sorted(images)}"
                )

    if len(expected_by_point) != len(crops):
        report.error(
            f"source point mapping count mismatch: "
            f"crops={len(crops)} expected_points={len(expected_by_point)}"
        )

    report.info["source"] = {
        "path": str(stats_path),
        "crop_count": len(crops),
        "unique_detection_ids": len(seen_detection),
        "expected_person_points": int(expected_counts["person"]),
        "expected_object_points": int(expected_counts["object"]),
        "expected_total_points": int(sum(expected_counts.values())),
        "duplicate_group_count": len(source_dup),
        "duplicate_point_count": sum(len(v) for v in source_dup.values()),
        "ambiguous_group_count": len(source_ambig),
        "ambiguous_point_count": sum(len(v) for v in source_ambig.values()),
        "dedup_metadata_counts": dict(dedup_counts),
    }

    dedup_summary = data.get("dedup")
    manual_summary = data.get("manual_review")
    hybrid_info: Dict[str, Any] = {
        "dedup_present": isinstance(dedup_summary, Mapping),
        "manual_review_present": isinstance(manual_summary, Mapping),
    }

    if isinstance(dedup_summary, Mapping):
        mode = str(dedup_summary.get("mode") or "").strip()
        hybrid_info["dedup_mode"] = mode
        if mode != "hybrid_c_hard_exact_soft_semantic":
            report.error(
                "stats JSON is not marked as Hybrid-C dedup output: "
                f"dedup.mode={mode!r}"
            )

        try:
            original_n = int(dedup_summary.get("original_crop_count"))
            deduped_n = int(dedup_summary.get("deduped_crop_count"))
            removed_n = int(dedup_summary.get("removed_count"))
            exact_n = int(dedup_summary.get("exact_detection_id_removed"))
        except Exception:
            report.error("Hybrid-C dedup summary has invalid count fields")
        else:
            hybrid_info.update({
                "original_crop_count": original_n,
                "deduped_crop_count": deduped_n,
                "hard_removed_count": removed_n,
                "exact_detection_id_removed": exact_n,
            })
            if removed_n != exact_n:
                report.error(
                    "Hybrid-C invariant violated in source summary: removed_count must "
                    f"equal exact_detection_id_removed ({removed_n} != {exact_n})"
                )
            if deduped_n != original_n - exact_n:
                report.error(
                    "Hybrid-C invariant violated in source summary: deduped_crop_count "
                    "must equal original_crop_count - exact_detection_id_removed "
                    f"({deduped_n} != {original_n} - {exact_n})"
                )

    if isinstance(manual_summary, Mapping):
        mode = str(manual_summary.get("mode") or "").strip()
        hybrid_info["manual_review_mode"] = mode
        if mode not in {"hybrid_c_non_destructive", "hybrid_c_non_destructive_manual_review"}:
            report.error(
                "stats JSON manual_review is not marked as Hybrid-C non-destructive: "
                f"mode={mode!r}"
            )
        try:
            manual_removed = int(manual_summary.get("manual_removed_point_count", 0))
            final_n = int(manual_summary.get("final_crop_count", len(crops)))
        except Exception:
            report.error("Hybrid-C manual_review summary has invalid count fields")
        else:
            hybrid_info["manual_removed_point_count"] = manual_removed
            hybrid_info["manual_final_crop_count"] = final_n
            if manual_removed != 0:
                report.error(
                    "Hybrid-C invariant violated: manual review removed semantic points: "
                    f"{manual_removed}"
                )
            if final_n != len(crops):
                report.error(
                    "manual_review final_crop_count does not match stats crops: "
                    f"summary={final_n}, actual={len(crops)}"
                )

    report.info["hybrid_c_source"] = hybrid_info

    return crops, expected_by_point


def audit_schema(
    client: Any,
    collection: str,
    expected_vecs: Dict[str, int],
    distance: str,
    report: AuditReport,
) -> Optional[Any]:
    if not client.collection_exists(collection):
        report.error(f"missing collection: {collection}")
        return None

    info = client.get_collection(collection)
    vectors = info.config.params.vectors

    if not isinstance(vectors, dict):
        report.error(f"{collection}: not a named-vector collection")
        return info

    expected_names = set(expected_vecs)
    actual_names = set(vectors)

    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing:
        report.error(f"{collection}: missing named vectors={missing}")
    if extra:
        report.error(f"{collection}: unexpected named vectors={extra}")

    details = {}
    for name in sorted(expected_names & actual_names):
        actual = vectors[name]
        dim = int(actual.size)
        dist = _norm(actual.distance)
        details[name] = {"dimension": dim, "distance": dist}

        if dim != expected_vecs[name]:
            report.error(
                f"{collection}/{name}: dim mismatch "
                f"expected={expected_vecs[name]} actual={dim}"
            )
        if dist != _norm(distance):
            report.error(
                f"{collection}/{name}: distance mismatch "
                f"expected={_norm(distance)} actual={dist}"
            )

    payload_schema = getattr(info, "payload_schema", None) or {}
    for field, wanted_type in EXPECTED_PAYLOAD_INDEXES.items():
        entry = payload_schema.get(field)
        if entry is None:
            report.error(f"{collection}: missing payload index '{field}'")
            continue
        actual_type = _norm(getattr(entry, "data_type", None))
        if actual_type and actual_type != wanted_type:
            report.error(
                f"{collection}: payload index '{field}' type mismatch "
                f"expected={wanted_type} actual={actual_type}"
            )

    status = _norm(getattr(info, "status", None))
    if status and status not in {"green", "yellow"}:
        report.warn(f"{collection}: status={status}")

    report.info.setdefault("collections", {})[collection] = {
        "status": status,
        "points_count": int(getattr(info, "points_count", 0) or 0),
        "indexed_vectors_count": int(
            getattr(info, "indexed_vectors_count", 0) or 0
        ),
        "segments_count": int(getattr(info, "segments_count", 0) or 0),
        "vectors": details,
    }
    return info


def compare_payload(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    person_labels: Set[str],
    report: AuditReport,
) -> None:
    record = expected["record"]
    detection_id = expected["detection_id"]
    scope = expected["scope"]

    expected_label = str(record.get("class_name", "unknown"))
    if str(payload.get("label", "")) != expected_label:
        report.error(
            f"{detection_id}: label mismatch "
            f"source={expected_label!r} qdrant={payload.get('label')!r}"
        )

    if record.get("image_id") is not None:
        if str(payload.get("image_id", "")) != str(record["image_id"]):
            report.error(f"{detection_id}: image_id mismatch")

    expected_frame = int(record.get("frame_idx", 0))
    try:
        actual_frame = int(payload.get("frame_idx"))
    except Exception:
        actual_frame = None
    if actual_frame != expected_frame:
        report.error(
            f"{detection_id}: frame_idx mismatch "
            f"source={expected_frame} qdrant={payload.get('frame_idx')!r}"
        )

    expected_score = float(record.get("confidence", 1.0))
    if not _float_close(payload.get("score"), expected_score):
        report.error(
            f"{detection_id}: score mismatch "
            f"source={expected_score} qdrant={payload.get('score')!r}"
        )

    if not _bbox_equal(payload.get("bbox"), record.get("bbox", [0, 0, 0, 0])):
        report.error(f"{detection_id}: bbox mismatch")

    wanted_person = scope == "person"
    if payload.get("is_person") is not wanted_person:
        report.error(
            f"{detection_id}: is_person mismatch "
            f"expected={wanted_person} actual={payload.get('is_person')!r}"
        )

    actual_label_person = str(payload.get("label", "")).strip().lower() in person_labels
    if actual_label_person != wanted_person:
        report.error(
            f"{detection_id}: label/scope conflict "
            f"label={payload.get('label')!r} scope={scope}"
        )

    wanted_media = infer_media_type(record)
    actual_media = str(payload.get("media_type", "")).strip().lower()
    if actual_media != wanted_media:
        report.error(
            f"{detection_id}: media_type mismatch "
            f"source={wanted_media!r} qdrant={actual_media!r}"
        )

    if "source" in record and payload.get("source") != record.get("source"):
        report.error(f"{detection_id}: source payload mismatch")

    src_crop = crop_path(record)
    if src_crop is not None and payload.get("crop_path") is not None:
        # Qdrant payload may store portable '/' separators while the source
        # metadata uses Windows '\\'. Treat those as the same path.
        def _norm_path_text(value: Any) -> str:
            text = str(value).strip().replace("\\", "/")
            while "//" in text:
                text = text.replace("//", "/")
            return text.casefold()

        if _norm_path_text(payload["crop_path"]) != _norm_path_text(src_crop):
            report.error(
                f"{detection_id}: crop_path mismatch: "
                f"source={src_crop!r}, qdrant={payload.get('crop_path')!r}"
            )

    for key in DEDUP_KEYS:
        if key not in record:
            continue
        if key not in payload:
            report.error(
                f"{detection_id}: dedup metadata missing in Qdrant payload: {key}"
            )
            continue
        if _json_norm(payload[key]) != _json_norm(record[key]):
            report.error(
                f"{detection_id}: dedup metadata mismatch: {key}"
            )


def audit_vectors(
    collection: str,
    point_id: str,
    vectors: Any,
    expected_vecs: Dict[str, int],
    report: AuditReport,
) -> None:
    if not isinstance(vectors, dict):
        report.error(f"{collection}/{point_id}: vector payload is not a dict")
        return

    actual_names = set(vectors)
    wanted_names = set(expected_vecs)
    if actual_names != wanted_names:
        report.error(
            f"{collection}/{point_id}: vector names mismatch "
            f"expected={sorted(wanted_names)} actual={sorted(actual_names)}"
        )

    for name in sorted(actual_names & wanted_names):
        arr = np.asarray(vectors[name], dtype=np.float32).reshape(-1)
        if arr.size != expected_vecs[name]:
            report.error(
                f"{collection}/{point_id}/{name}: dim mismatch "
                f"expected={expected_vecs[name]} actual={arr.size}"
            )
        if not np.all(np.isfinite(arr)):
            report.error(f"{collection}/{point_id}/{name}: NaN/Inf")
        if arr.size and float(np.linalg.norm(arr)) <= 1e-12:
            report.error(f"{collection}/{point_id}/{name}: all-zero vector")


def scan_collection(
    client: Any,
    collection: str,
    scope: str,
    expected_vecs: Dict[str, int],
    expected_by_point: Mapping[str, Dict[str, Any]],
    person_labels: Set[str],
    report: AuditReport,
    page_size: int,
    with_vectors: bool,
) -> Dict[str, Any]:
    point_ids: Set[str] = set()
    detection_ids = Counter()
    crop_ids = Counter()
    missing_payload_fields = Counter()
    dedup_counts = Counter()
    dup_groups = defaultdict(list)
    ambig = defaultdict(list)
    count = 0
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )

        for point in points:
            count += 1
            pid = str(point.id)

            if pid in point_ids:
                report.error(f"{collection}: point ID returned twice: {pid}")
            point_ids.add(pid)

            payload = point.payload or {}
            if not isinstance(payload, dict):
                report.error(f"{collection}/{pid}: invalid payload type")
                continue

            for key in REQUIRED_PAYLOAD_FIELDS:
                if key not in payload:
                    missing_payload_fields[key] += 1

            det_id = payload.get("detection_id")
            crop_id = payload.get("crop_id")
            if det_id is not None:
                detection_ids[str(det_id)] += 1
            if crop_id is not None:
                crop_ids[str(crop_id)] += 1

            if det_id is not None and crop_id is not None:
                if str(det_id) != str(crop_id):
                    report.error(
                        f"{collection}/{pid}: detection_id != crop_id"
                    )

            for key in DEDUP_KEYS:
                if key in payload:
                    dedup_counts[key] += 1

            dup_group = payload.get("duplicate_group_id")
            if dup_group is not None and str(dup_group).strip():
                dup_groups[str(dup_group).strip()].append(str(det_id or pid))
                if scope == "person":
                    report.error(
                        f"{collection}/{pid}: person point has duplicate_group_id"
                    )

            group = payload.get("ambiguous_group_id")
            if group is not None and str(group).strip():
                ambig[str(group).strip()].append(str(det_id or pid))
                if scope == "person":
                    report.error(
                        f"{collection}/{pid}: person point has ambiguous_group_id"
                    )

            expected = expected_by_point.get(pid)
            if expected is None:
                report.error(
                    f"{collection}: extra point not in source "
                    f"point_id={pid} detection_id={det_id!r}"
                )
            else:
                if expected["scope"] != scope:
                    report.error(
                        f"{collection}/{pid}: wrong collection routing "
                        f"expected_scope={expected['scope']}"
                    )

                compare_payload(payload, expected, person_labels, report)

                wanted_pid = stable_point_id(collection, expected["detection_id"])
                if pid != wanted_pid:
                    report.error(
                        f"{collection}/{pid}: deterministic point ID mismatch "
                        f"expected={wanted_pid}"
                    )

            if scope == "person" and payload.get("is_person") is not True:
                report.error(f"{collection}/{pid}: non-person in person collection")
            if scope == "object" and payload.get("is_person") is not False:
                report.error(f"{collection}/{pid}: person in object collection")

            media = str(payload.get("media_type", "")).strip().lower()
            if media not in {"image", "video"}:
                report.error(f"{collection}/{pid}: invalid media_type={media!r}")
            if media == "video":
                if payload.get("video") in {None, ""}:
                    report.error(f"{collection}/{pid}: video point missing video")
                try:
                    int(payload.get("frame_idx"))
                except Exception:
                    report.error(f"{collection}/{pid}: invalid video frame_idx")

            try:
                bbox = [float(x) for x in payload.get("bbox")]
            except Exception:
                bbox = []
            if len(bbox) != 4 or not all(math.isfinite(x) for x in bbox):
                report.error(f"{collection}/{pid}: invalid bbox={payload.get('bbox')!r}")

            try:
                score = float(payload.get("score"))
            except Exception:
                score = math.nan
            if not math.isfinite(score):
                report.error(f"{collection}/{pid}: invalid score={payload.get('score')!r}")

            if with_vectors:
                audit_vectors(collection, pid, point.vector, expected_vecs, report)

        if next_offset is None:
            break
        offset = next_offset

    for key, n in sorted(missing_payload_fields.items()):
        if n:
            report.error(
                f"{collection}: required payload '{key}' missing from {n} points"
            )

    dup_det = {k: v for k, v in detection_ids.items() if v > 1}
    if dup_det:
        report.error(
            f"{collection}: duplicate detection_id payloads={len(dup_det)}; "
            f"examples={_safe_examples(dup_det.items())}"
        )

    dup_crop = {k: v for k, v in crop_ids.items() if v > 1}
    if dup_crop:
        report.error(
            f"{collection}: duplicate crop_id payloads={len(dup_crop)}; "
            f"examples={_safe_examples(dup_crop.items())}"
        )

    expected_ids = {
        pid for pid, item in expected_by_point.items()
        if item["scope"] == scope
    }
    missing = sorted(expected_ids - point_ids)
    extra = sorted(point_ids - expected_ids)

    if missing:
        report.error(
            f"{collection}: missing points={len(missing)}; "
            f"examples={_safe_examples(missing)}"
        )
    if extra:
        report.error(
            f"{collection}: extra points={len(extra)}; "
            f"examples={_safe_examples(extra)}"
        )
    if count != len(expected_ids):
        report.error(
            f"{collection}: count mismatch "
            f"expected={len(expected_ids)} scanned={count}"
        )

    return {
        "point_count_scanned": count,
        "unique_point_ids": len(point_ids),
        "unique_detection_ids": len(detection_ids),
        "unique_crop_ids": len(crop_ids),
        "actual_point_ids": point_ids,
        "detection_ids": set(detection_ids),
        "duplicate_groups": {
            k: sorted(v) for k, v in dup_groups.items()
        },
        "duplicate_group_count": len(dup_groups),
        "duplicate_point_count": sum(len(v) for v in dup_groups.values()),
        "ambiguous_groups": {
            k: sorted(v) for k, v in ambig.items()
        },
        "ambiguous_group_count": len(ambig),
        "ambiguous_point_count": sum(len(v) for v in ambig.values()),
        "dedup_metadata_counts": dict(dedup_counts),
    }


def source_duplicate_groups(crops: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for record in crops:
        group = record.get("duplicate_group_id")
        if group is None or not str(group).strip():
            continue
        det_id = make_detection_id(dict(record), int(record.get("frame_idx", 0)))
        groups[str(group).strip()].append(det_id)
    return {k: sorted(v) for k, v in groups.items()}


def audit_duplicate_groups(
    crops: Sequence[Mapping[str, Any]],
    person_scan: Mapping[str, Any],
    object_scan: Mapping[str, Any],
    report: AuditReport,
) -> None:
    """Require exact Hybrid-C duplicate-group membership preservation in Qdrant."""
    expected = source_duplicate_groups(crops)
    actual = defaultdict(list)

    for scan in (person_scan, object_scan):
        for group, members in scan.get("duplicate_groups", {}).items():
            actual[group].extend(members)

    actual = {k: sorted(v) for k, v in actual.items()}

    malformed_expected = {k: v for k, v in expected.items() if len(v) < 2}
    malformed_actual = {k: v for k, v in actual.items() if len(v) < 2}
    if malformed_expected:
        report.error(
            "source duplicate groups with fewer than two members: "
            f"{_safe_examples(malformed_expected.items())}"
        )
    if malformed_actual:
        report.error(
            "Qdrant duplicate groups with fewer than two members: "
            f"{_safe_examples(malformed_actual.items())}"
        )

    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        mismatch = sorted(
            k for k in set(expected) & set(actual)
            if expected[k] != actual[k]
        )
        report.error(
            "duplicate-group mismatch: "
            f"missing_groups={len(missing)}, extra_groups={len(extra)}, "
            f"member_mismatches={len(mismatch)}; "
            f"examples_missing={_safe_examples(missing)}, "
            f"examples_extra={_safe_examples(extra)}, "
            f"examples_mismatch={_safe_examples(mismatch)}"
        )

    report.info["duplicate_groups"] = {
        "source_group_count": len(expected),
        "source_point_count": sum(len(v) for v in expected.values()),
        "qdrant_group_count": len(actual),
        "qdrant_point_count": sum(len(v) for v in actual.values()),
        "membership_match": expected == actual,
    }


def source_ambiguous_groups(crops: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for record in crops:
        group = record.get("ambiguous_group_id")
        if group is None:
            continue
        det_id = make_detection_id(dict(record), int(record.get("frame_idx", 0)))
        groups[str(group)].append(det_id)
    return {k: sorted(v) for k, v in groups.items()}


def audit_ambiguous_groups(
    crops: Sequence[Mapping[str, Any]],
    person_scan: Mapping[str, Any],
    object_scan: Mapping[str, Any],
    report: AuditReport,
) -> None:
    expected = source_ambiguous_groups(crops)
    actual = defaultdict(list)

    for scan in (person_scan, object_scan):
        for group, members in scan["ambiguous_groups"].items():
            actual[group].extend(members)

    actual = {k: sorted(v) for k, v in actual.items()}

    if expected == actual:
        return

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatch = sorted(
        k for k in set(expected) & set(actual)
        if expected[k] != actual[k]
    )
    report.error(
        "ambiguous-group mismatch: "
        f"missing_groups={len(missing)}, extra_groups={len(extra)}, "
        f"member_mismatches={len(mismatch)}; "
        f"examples_missing={_safe_examples(missing)}, "
        f"examples_extra={_safe_examples(extra)}, "
        f"examples_mismatch={_safe_examples(mismatch)}"
    )


def _print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_summary(report: AuditReport) -> None:
    _print_header("DB AUDIT SUMMARY")

    source = report.info.get("source", {})
    if source:
        print(f"source crops        : {source.get('crop_count', 0):,}")
        print(f"source unique IDs   : {source.get('unique_detection_ids', 0):,}")
        print(f"expected person     : {source.get('expected_person_points', 0):,}")
        print(f"expected object     : {source.get('expected_object_points', 0):,}")
        print(f"expected total      : {source.get('expected_total_points', 0):,}")
        print(
            "source ambiguous    : "
            f"{source.get('ambiguous_group_count', 0):,} groups / "
            f"{source.get('ambiguous_point_count', 0):,} points"
        )

    for collection, info in report.info.get("collections", {}).items():
        print(
            f"{collection:20s}: "
            f"points={info.get('points_count', 0):,} "
            f"status={info.get('status', '')}"
        )

    for scope, scan in report.info.get("scans", {}).items():
        print(
            f"scan {scope:12s}: "
            f"points={scan.get('point_count_scanned', 0):,} "
            f"unique_point_ids={scan.get('unique_point_ids', 0):,} "
            f"unique_detection_ids={scan.get('unique_detection_ids', 0):,}"
        )

    print()
    print(f"errors   : {len(report.errors):,}")
    print(f"warnings : {len(report.warnings):,}")

    if report.errors:
        _print_header("ERRORS")
        for i, msg in enumerate(report.errors, 1):
            print(f"[{i}] {msg}")

    if report.warnings:
        _print_header("WARNINGS")
        for i, msg in enumerate(report.warnings, 1):
            print(f"[{i}] {msg}")

    _print_header("FINAL RESULT")
    if report.ok:
        print("PASS - no integrity errors detected.")
    else:
        print(
            "FAIL - integrity errors detected. "
            "Do not treat this DB as final until resolved."
        )


def to_serializable(v: Any) -> Any:
    if isinstance(v, set):
        return sorted(v)
    if isinstance(v, dict):
        return {
            str(k): to_serializable(x)
            for k, x in v.items()
            if k not in {"actual_point_ids", "detection_ids"}
        }
    if isinstance(v, list):
        return [to_serializable(x) for x in v]
    return v



# =============================================================================
# Final-strength audit extensions
# =============================================================================

def _quant_expected_signature(quant: Any) -> Dict[str, Any]:
    qtype = _norm(getattr(quant, "type", None))
    if qtype == "scalar":
        quantile = getattr(quant, "quantile", None)
        return {
            "type": "scalar",
            "always_ram": bool(getattr(quant, "always_ram", True)),
            "quantile": None if quantile is None else float(quantile),
        }
    if qtype == "binary":
        return {
            "type": "binary",
            "always_ram": bool(getattr(quant, "always_ram", True)),
        }
    return {"type": "none"}


def _quant_actual_signature(quant: Any) -> Dict[str, Any]:
    if quant is None:
        return {"type": "none"}

    scalar = getattr(quant, "scalar", None)
    if scalar is not None:
        quantile = getattr(scalar, "quantile", None)
        return {
            "type": "scalar",
            "always_ram": bool(getattr(scalar, "always_ram", False)),
            "quantile": None if quantile is None else float(quantile),
        }

    binary = getattr(quant, "binary", None)
    if binary is not None:
        return {
            "type": "binary",
            "always_ram": bool(getattr(binary, "always_ram", False)),
        }

    # Some qdrant-client versions expose the inner config directly.
    tname = _norm(type(quant).__name__)
    if "scalar" in tname:
        quantile = getattr(quant, "quantile", None)
        return {
            "type": "scalar",
            "always_ram": bool(getattr(quant, "always_ram", False)),
            "quantile": None if quantile is None else float(quantile),
        }
    if "binary" in tname:
        return {
            "type": "binary",
            "always_ram": bool(getattr(quant, "always_ram", False)),
        }

    return {"type": tname or "unknown"}


def _quant_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if a.get("type") != b.get("type"):
        return False
    if a.get("type") == "none":
        return True
    if a.get("always_ram") != b.get("always_ram"):
        return False
    if a.get("type") == "scalar":
        x, y = a.get("quantile"), b.get("quantile")
        if x is None or y is None:
            return x is y
        return abs(float(x) - float(y)) <= 1e-6
    return True


def audit_source_content(
    crops: Sequence[Mapping[str, Any]],
    report: AuditReport,
    *,
    check_crop_files: bool,
) -> None:
    """Audit raw-content consistency under Hybrid-C semantics.

    Exact same-class geometry is no longer automatically an integrity error: in
    Hybrid-C those detections are intentionally preserved when they belong to one
    duplicate_group_id. The error is now an *ungrouped* exact duplicate signature.
    """
    crop_paths: Dict[str, List[str]] = defaultdict(list)
    same_class_geom: Dict[Tuple[Any, ...], List[str]] = defaultdict(list)
    box_only_geom: Dict[Tuple[Any, ...], List[Tuple[str, str]]] = defaultdict(list)
    duplicate_group_by_id: Dict[str, Optional[str]] = {}
    ambiguous_group_by_id: Dict[str, Optional[str]] = {}
    missing_crop_paths: List[str] = []
    missing_crop_files: List[str] = []

    for record in crops:
        frame = int(record.get("frame_idx", 0))
        det_id = make_detection_id(dict(record), frame)
        dup_gid = record.get("duplicate_group_id")
        amb_gid = record.get("ambiguous_group_id")
        duplicate_group_by_id[det_id] = (
            str(dup_gid).strip() if dup_gid is not None and str(dup_gid).strip() else None
        )
        ambiguous_group_by_id[det_id] = (
            str(amb_gid).strip() if amb_gid is not None and str(amb_gid).strip() else None
        )

        cp = crop_path(record)
        if cp is None:
            missing_crop_paths.append(det_id)
        else:
            crop_paths[cp.replace("\\", "/")].append(det_id)
            if check_crop_files and not Path(cp).is_file():
                missing_crop_files.append(cp)

        try:
            bbox = tuple(int(round(float(x))) for x in record.get("bbox", ()))
        except Exception:
            bbox = tuple()

        if len(bbox) == 4:
            image_id = str(record.get("image_id", "")).replace("\\", "/")
            label = str(record.get("class_name", "unknown")).strip().lower()
            same_class_geom[(image_id, frame, label, bbox)].append(det_id)
            box_only_geom[(image_id, frame, bbox)].append((label, det_id))

    dup_crop_paths = {k: v for k, v in crop_paths.items() if len(v) > 1}
    dup_same_geom = {k: v for k, v in same_class_geom.items() if len(v) > 1}
    cross_label_same_box = {
        k: v
        for k, v in box_only_geom.items()
        if len(v) > 1 and len({label for label, _ in v}) > 1
    }

    grouped_same_geom: Dict[Tuple[Any, ...], List[str]] = {}
    ungrouped_same_geom: Dict[Tuple[Any, ...], List[str]] = {}
    for signature, ids in dup_same_geom.items():
        gids = {duplicate_group_by_id.get(did) for did in ids}
        if len(gids) == 1 and None not in gids:
            grouped_same_geom[signature] = ids
        else:
            ungrouped_same_geom[signature] = ids

    classified_cross_box: Dict[Tuple[Any, ...], List[Tuple[str, str]]] = {}
    unclassified_cross_box: Dict[Tuple[Any, ...], List[Tuple[str, str]]] = {}
    for signature, members in cross_label_same_box.items():
        ids = [did for _label_name, did in members]
        dup_gids = {duplicate_group_by_id.get(did) for did in ids}
        amb_gids = {ambiguous_group_by_id.get(did) for did in ids}
        duplicate_relation = len(dup_gids) == 1 and None not in dup_gids
        ambiguous_relation = len(amb_gids) == 1 and None not in amb_gids
        if duplicate_relation or ambiguous_relation:
            classified_cross_box[signature] = members
        else:
            unclassified_cross_box[signature] = members

    if missing_crop_paths:
        report.error(
            f"source records missing crop_path/path: {len(missing_crop_paths)}; "
            f"examples={_safe_examples(missing_crop_paths)}"
        )
    if missing_crop_files:
        report.error(
            f"crop files missing on disk: {len(missing_crop_files)}; "
            f"examples={_safe_examples(missing_crop_files)}"
        )
    if dup_crop_paths:
        report.error(
            f"same crop_path is referenced by multiple source records: {len(dup_crop_paths)} paths; "
            f"examples={_safe_examples(dup_crop_paths.items())}"
        )
    if ungrouped_same_geom:
        report.error(
            "Hybrid-C exact same-class geometry exists without one shared "
            "duplicate_group_id: "
            f"{len(ungrouped_same_geom)} signatures; "
            f"examples={_safe_examples(ungrouped_same_geom.items())}"
        )
    if unclassified_cross_box:
        report.warn(
            "same image/frame/bbox appears with multiple class labels but is not "
            "represented by one duplicate_group_id or one ambiguous_group_id: "
            f"{len(unclassified_cross_box)} signatures; "
            f"examples={_safe_examples(unclassified_cross_box.items())}"
        )

    report.info["source_content"] = {
        "missing_crop_path_count": len(missing_crop_paths),
        "missing_crop_file_count": len(missing_crop_files),
        "duplicate_crop_path_count": len(dup_crop_paths),
        "duplicate_same_class_geometry_count": len(dup_same_geom),
        "duplicate_same_class_geometry_grouped_count": len(grouped_same_geom),
        "duplicate_same_class_geometry_ungrouped_count": len(ungrouped_same_geom),
        "cross_label_same_box_count": len(cross_label_same_box),
        "cross_label_same_box_classified_count": len(classified_cross_box),
        "cross_label_same_box_unclassified_count": len(unclassified_cross_box),
        "crop_file_check_enabled": bool(check_crop_files),
    }


def audit_schema_strict(
    client: Any,
    collection: str,
    expected_vecs: Dict[str, int],
    cfg: PipelineConfig,
    report: AuditReport,
) -> Optional[Any]:
    """Schema + storage policy audit, including on_disk / quantization / HNSW."""
    if not client.collection_exists(collection):
        report.error(f"missing collection: {collection}")
        return None

    info = client.get_collection(collection)
    vectors = info.config.params.vectors
    if not isinstance(vectors, dict):
        report.error(f"{collection}: not a named-vector collection")
        return info

    wanted_names = set(expected_vecs)
    actual_names = set(vectors)
    missing = sorted(wanted_names - actual_names)
    extra = sorted(actual_names - wanted_names)
    if missing:
        report.error(f"{collection}: missing named vectors={missing}")
    if extra:
        report.error(f"{collection}: unexpected named vectors={extra}")

    details: Dict[str, Any] = {}
    for name in sorted(wanted_names & actual_names):
        actual = vectors[name]
        dim = int(actual.size)
        dist = _norm(actual.distance)
        wanted_dist = _norm(
            cfg.qdrant.distance_for(name)
            if hasattr(cfg.qdrant, "distance_for")
            else cfg.qdrant.distance
        )

        if dim != int(expected_vecs[name]):
            report.error(
                f"{collection}/{name}: dim mismatch "
                f"expected={expected_vecs[name]} actual={dim}"
            )
        if dist != wanted_dist:
            report.error(
                f"{collection}/{name}: distance mismatch "
                f"expected={wanted_dist} actual={dist}"
            )

        actual_on_disk = getattr(actual, "on_disk", None)
        if actual_on_disk is None:
            report.warn(f"{collection}/{name}: qdrant-client did not expose vector on_disk")
        elif bool(actual_on_disk) != bool(cfg.qdrant.on_disk):
            report.error(
                f"{collection}/{name}: on_disk mismatch "
                f"expected={cfg.qdrant.on_disk} actual={actual_on_disk}"
            )

        expected_quant = _quant_expected_signature(cfg.qdrant.quant_for(name))
        actual_quant_obj = getattr(actual, "quantization_config", None)
        if actual_quant_obj is None:
            actual_quant_obj = getattr(getattr(info, "config", None), "quantization_config", None)
        actual_quant = _quant_actual_signature(actual_quant_obj)
        if not _quant_equal(expected_quant, actual_quant):
            report.error(
                f"{collection}/{name}: quantization mismatch "
                f"expected={expected_quant} actual={actual_quant}"
            )

        wanted_hnsw = cfg.qdrant.hnsw_for(name)
        actual_hnsw = getattr(actual, "hnsw_config", None)
        if actual_hnsw is None:
            actual_hnsw = getattr(getattr(info, "config", None), "hnsw_config", None)
        actual_m = getattr(actual_hnsw, "m", None) if actual_hnsw is not None else None
        actual_ef = (
            getattr(actual_hnsw, "ef_construct", None)
            if actual_hnsw is not None else None
        )
        if actual_m is None or actual_ef is None:
            report.warn(
                f"{collection}/{name}: could not fully inspect HNSW m/ef_construct"
            )
        else:
            if int(actual_m) != int(wanted_hnsw.m):
                report.error(
                    f"{collection}/{name}: HNSW m mismatch "
                    f"expected={wanted_hnsw.m} actual={actual_m}"
                )
            if int(actual_ef) != int(wanted_hnsw.ef_construct):
                report.error(
                    f"{collection}/{name}: HNSW ef_construct mismatch "
                    f"expected={wanted_hnsw.ef_construct} actual={actual_ef}"
                )

        details[name] = {
            "dimension": dim,
            "distance": dist,
            "on_disk": actual_on_disk,
            "quantization": actual_quant,
            "hnsw_m": actual_m,
            "hnsw_ef_construct": actual_ef,
        }

    payload_schema = getattr(info, "payload_schema", None) or {}
    for field, wanted_type in EXPECTED_PAYLOAD_INDEXES.items():
        entry = payload_schema.get(field)
        if entry is None:
            report.error(f"{collection}: missing payload index '{field}'")
            continue
        actual_type = _norm(getattr(entry, "data_type", None))
        if actual_type and actual_type != wanted_type:
            report.error(
                f"{collection}: payload index '{field}' type mismatch "
                f"expected={wanted_type} actual={actual_type}"
            )

    status = _norm(getattr(info, "status", None))
    if status and status not in {"green", "yellow"}:
        report.warn(f"{collection}: status={status}")

    report.info.setdefault("collections", {})[collection] = {
        "status": status,
        "points_count": int(getattr(info, "points_count", 0) or 0),
        "indexed_vectors_count": int(getattr(info, "indexed_vectors_count", 0) or 0),
        "segments_count": int(getattr(info, "segments_count", 0) or 0),
        "vectors": details,
    }
    return info


def _point_vector_map(point: Any) -> Any:
    v = getattr(point, "vector", None)
    if v is None:
        v = getattr(point, "vectors", None)
    return v


def scan_collection_final(
    client: Any,
    collection: str,
    scope: str,
    expected_vecs: Dict[str, int],
    expected_by_point: Mapping[str, Dict[str, Any]],
    person_labels: Set[str],
    report: AuditReport,
    page_size: int,
    with_vectors: bool,
) -> Dict[str, Any]:
    point_ids: Set[str] = set()
    detection_ids = Counter()
    crop_ids = Counter()
    missing_payload_fields = Counter()
    dedup_counts = Counter()
    dup_groups = defaultdict(list)
    ambig = defaultdict(list)
    count = 0
    offset = None

    # Exact float-vector duplicate evidence. These are warnings rather than errors:
    # two independent source images can legitimately be pixel-identical.
    vector_hash_first: Dict[Tuple[str, str], str] = {}
    vector_hash_dup: Dict[str, int] = Counter()
    vector_hash_examples: List[Tuple[str, str, str]] = []
    bundle_hash_first: Dict[str, str] = {}
    bundle_dup_count = 0
    bundle_dup_examples: List[Tuple[str, str]] = []

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )

        for point in points:
            count += 1
            pid = str(point.id)
            if pid in point_ids:
                report.error(f"{collection}: point ID returned twice: {pid}")
            point_ids.add(pid)

            payload = point.payload or {}
            if not isinstance(payload, dict):
                report.error(f"{collection}/{pid}: invalid payload type")
                continue

            for key in REQUIRED_PAYLOAD_FIELDS:
                if key not in payload:
                    missing_payload_fields[key] += 1

            det_id = payload.get("detection_id")
            crop_id = payload.get("crop_id")
            if det_id is not None:
                detection_ids[str(det_id)] += 1
            if crop_id is not None:
                crop_ids[str(crop_id)] += 1
            if det_id is not None and crop_id is not None and str(det_id) != str(crop_id):
                report.error(f"{collection}/{pid}: detection_id != crop_id")

            for key in DEDUP_KEYS:
                if key in payload:
                    dedup_counts[key] += 1
            dup_group = payload.get("duplicate_group_id")
            if dup_group is not None and str(dup_group).strip():
                dup_groups[str(dup_group).strip()].append(str(det_id or pid))
                if scope == "person":
                    report.error(
                        f"{collection}/{pid}: person point has duplicate_group_id"
                    )

            group = payload.get("ambiguous_group_id")
            if group is not None and str(group).strip():
                ambig[str(group).strip()].append(str(det_id or pid))
                if scope == "person":
                    report.error(
                        f"{collection}/{pid}: person point has ambiguous_group_id"
                    )

            expected = expected_by_point.get(pid)
            if expected is None:
                report.error(
                    f"{collection}: extra point not in source "
                    f"point_id={pid} detection_id={det_id!r}"
                )
            else:
                if expected["scope"] != scope:
                    report.error(
                        f"{collection}/{pid}: wrong collection routing "
                        f"expected_scope={expected['scope']}"
                    )
                compare_payload(payload, expected, person_labels, report)
                wanted_pid = stable_point_id(collection, expected["detection_id"])
                if pid != wanted_pid:
                    report.error(
                        f"{collection}/{pid}: deterministic point ID mismatch "
                        f"expected={wanted_pid}"
                    )

            if scope == "person" and payload.get("is_person") is not True:
                report.error(f"{collection}/{pid}: non-person in person collection")
            if scope == "object" and payload.get("is_person") is not False:
                report.error(f"{collection}/{pid}: person in object collection")

            media = str(payload.get("media_type", "")).strip().lower()
            if media not in {"image", "video"}:
                report.error(f"{collection}/{pid}: invalid media_type={media!r}")
            if media == "video":
                if payload.get("video") in {None, ""}:
                    report.error(f"{collection}/{pid}: video point missing video")
                try:
                    int(payload.get("frame_idx"))
                except Exception:
                    report.error(f"{collection}/{pid}: invalid video frame_idx")

            try:
                bbox = [float(x) for x in payload.get("bbox")]
            except Exception:
                bbox = []
            if len(bbox) != 4 or not all(math.isfinite(x) for x in bbox):
                report.error(f"{collection}/{pid}: invalid bbox={payload.get('bbox')!r}")

            try:
                score = float(payload.get("score"))
            except Exception:
                score = math.nan
            if not math.isfinite(score):
                report.error(f"{collection}/{pid}: invalid score={payload.get('score')!r}")

            if with_vectors:
                vmap = _point_vector_map(point)
                audit_vectors(collection, pid, vmap, expected_vecs, report)
                if isinstance(vmap, dict):
                    bundle_parts: List[str] = []
                    for name in sorted(set(vmap) & set(expected_vecs)):
                        try:
                            arr = np.asarray(vmap[name], dtype=np.float32).reshape(-1)
                        except Exception:
                            continue
                        digest = hashlib.sha256(arr.tobytes(order="C")).hexdigest()
                        key = (name, digest)
                        first = vector_hash_first.get(key)
                        if first is None:
                            vector_hash_first[key] = pid
                        elif first != pid:
                            vector_hash_dup[name] += 1
                            if len(vector_hash_examples) < 10:
                                vector_hash_examples.append((name, first, pid))
                        bundle_parts.append(f"{name}:{digest}")
                    if bundle_parts:
                        bundle_digest = hashlib.sha256(
                            "|".join(bundle_parts).encode("ascii")
                        ).hexdigest()
                        first = bundle_hash_first.get(bundle_digest)
                        if first is None:
                            bundle_hash_first[bundle_digest] = pid
                        elif first != pid:
                            bundle_dup_count += 1
                            if len(bundle_dup_examples) < 10:
                                bundle_dup_examples.append((first, pid))

        if next_offset is None:
            break
        offset = next_offset

    for key, n in sorted(missing_payload_fields.items()):
        if n:
            report.error(f"{collection}: required payload '{key}' missing from {n} points")

    dup_det = {k: v for k, v in detection_ids.items() if v > 1}
    if dup_det:
        report.error(
            f"{collection}: duplicate detection_id payloads={len(dup_det)}; "
            f"examples={_safe_examples(dup_det.items())}"
        )
    dup_crop = {k: v for k, v in crop_ids.items() if v > 1}
    if dup_crop:
        report.error(
            f"{collection}: duplicate crop_id payloads={len(dup_crop)}; "
            f"examples={_safe_examples(dup_crop.items())}"
        )

    expected_ids = {pid for pid, item in expected_by_point.items() if item["scope"] == scope}
    missing = sorted(expected_ids - point_ids)
    extra = sorted(point_ids - expected_ids)
    if missing:
        report.error(
            f"{collection}: missing points={len(missing)}; examples={_safe_examples(missing)}"
        )
    if extra:
        report.error(
            f"{collection}: extra points={len(extra)}; examples={_safe_examples(extra)}"
        )
    if count != len(expected_ids):
        report.error(
            f"{collection}: count mismatch expected={len(expected_ids)} scanned={count}"
        )

    if vector_hash_dup:
        report.warn(
            f"{collection}: exact identical named-vector values occur across distinct points: "
            f"{dict(vector_hash_dup)}; examples={vector_hash_examples}"
        )
    if bundle_dup_count:
        report.warn(
            f"{collection}: {bundle_dup_count} distinct point pairs share the complete exact "
            f"named-vector bundle; examples={bundle_dup_examples}. "
            "Check whether their crop images are intentionally identical."
        )

    return {
        "point_count_scanned": count,
        "unique_point_ids": len(point_ids),
        "unique_detection_ids": len(detection_ids),
        "unique_crop_ids": len(crop_ids),
        "actual_point_ids": point_ids,
        "detection_ids": set(detection_ids),
        "duplicate_groups": {k: sorted(v) for k, v in dup_groups.items()},
        "duplicate_group_count": len(dup_groups),
        "duplicate_point_count": sum(len(v) for v in dup_groups.values()),
        "ambiguous_groups": {k: sorted(v) for k, v in ambig.items()},
        "ambiguous_group_count": len(ambig),
        "ambiguous_point_count": sum(len(v) for v in ambig.values()),
        "dedup_metadata_counts": dict(dedup_counts),
        "exact_named_vector_duplicate_occurrences": dict(vector_hash_dup),
        "exact_full_vector_bundle_duplicate_occurrences": bundle_dup_count,
    }


def _balanced_sample_items(
    expected_by_point: Mapping[str, Dict[str, Any]],
    total: int,
    seed: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    if total <= 0:
        return []
    rng = random.Random(seed)
    groups: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {"person": [], "object": []}
    for pid, item in expected_by_point.items():
        groups[item["scope"]].append((pid, item))
    for items in groups.values():
        items.sort(key=lambda x: x[0])

    half = total // 2
    targets = {"person": half, "object": total - half}
    chosen: List[Tuple[str, Dict[str, Any]]] = []
    leftovers: List[Tuple[str, Dict[str, Any]]] = []
    for scope in ("person", "object"):
        items = groups[scope]
        n = min(targets[scope], len(items))
        chosen.extend(rng.sample(items, n) if n else [])
        chosen_ids = {pid for pid, _ in chosen}
        leftovers.extend([x for x in items if x[0] not in chosen_ids])

    need = min(total, len(expected_by_point)) - len(chosen)
    if need > 0 and leftovers:
        chosen.extend(rng.sample(leftovers, min(need, len(leftovers))))
    chosen.sort(key=lambda x: (x[1]["scope"], x[0]))
    return chosen


def _retrieve_points_with_vectors(
    client: Any,
    collection: str,
    point_ids: Sequence[str],
) -> Dict[str, Any]:
    if not point_ids:
        return {}
    records = client.retrieve(
        collection_name=collection,
        ids=list(point_ids),
        with_payload=True,
        with_vectors=True,
    )
    return {str(r.id): r for r in records}


def _cosine(a: Any, b: Any) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    if aa.shape != bb.shape or aa.size == 0:
        return math.nan
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na <= 1e-12 or nb <= 1e-12:
        return math.nan
    return float(np.dot(aa, bb) / (na * nb))


def recompute_spot_check(
    client: Any,
    cfg: PipelineConfig,
    expected_by_point: Mapping[str, Dict[str, Any]],
    report: AuditReport,
    *,
    samples: int,
    seed: int,
    min_cosine: float,
) -> None:
    """Re-run the actual build-time Router on sampled crop files and compare DB vectors."""
    selected = _balanced_sample_items(expected_by_point, samples, seed)
    if not selected:
        report.warn("recompute spot-check skipped: no sample points")
        return

    usable: List[Tuple[str, Dict[str, Any]]] = []
    for pid, item in selected:
        cp = crop_path(item["record"])
        if cp is None or not Path(cp).is_file():
            report.error(
                f"recompute sample cannot read crop file: point_id={pid} crop_path={cp!r}"
            )
        else:
            usable.append((pid, item))
    if not usable:
        report.error("recompute spot-check has zero usable crop files")
        return

    from registry import EmbedderRegistry
    from router import Router
    from rfdetr_adapter import from_rfdetr

    records = [item["record"] for _, item in usable]
    detections, fmt = from_rfdetr(records, load_mode="path")
    if len(detections) != len(records):
        report.error(
            f"recompute from_rfdetr count mismatch: records={len(records)} detections={len(detections)}"
        )
        return

    registry = EmbedderRegistry(cfg)
    recomputed = None
    try:
        router = Router(cfg, registry, input_format=fmt)
        recomputed = router.embed(detections)
    except Exception as e:
        report.error(f"recompute spot-check failed while loading/running embedders: {e}")
        return
    finally:
        # Do not mask audit results if cleanup itself fails.
        try:
            registry.release()
        except Exception as e:
            report.warn(f"EmbedderRegistry.release warning after recompute: {e}")

    if recomputed is None:
        report.error("recompute spot-check produced no result")
        return

    if len(recomputed) != len(usable):
        report.error(
            f"Router recompute count mismatch: expected={len(usable)} actual={len(recomputed)}"
        )
        return

    names = collection_names(cfg)
    by_scope: Dict[str, List[str]] = {"person": [], "object": []}
    for pid, item in usable:
        by_scope[item["scope"]].append(pid)
    stored: Dict[str, Any] = {}
    for scope in ("person", "object"):
        stored.update(_retrieve_points_with_vectors(client, names[scope], by_scope[scope]))

    scores_by_model: Dict[str, List[float]] = defaultdict(list)
    comparisons = 0
    for (pid, item), vector_map in zip(usable, recomputed):
        rec = stored.get(pid)
        if rec is None:
            report.error(f"recompute sample point missing from Qdrant: {pid}")
            continue
        db_vmap = _point_vector_map(rec)
        if not isinstance(db_vmap, dict):
            report.error(f"recompute sample has invalid stored vectors: {pid}")
            continue
        wanted = expected_vectors(cfg, item["scope"])
        for name in sorted(wanted):
            if name not in vector_map:
                report.error(f"recompute {pid}: Router missing vector {name}")
                continue
            if name not in db_vmap:
                report.error(f"recompute {pid}: Qdrant missing vector {name}")
                continue
            sim = _cosine(vector_map[name], db_vmap[name])
            comparisons += 1
            if math.isfinite(sim):
                scores_by_model[name].append(sim)
            if not math.isfinite(sim) or sim < min_cosine:
                report.error(
                    f"recompute vector mismatch: point={pid} model={name} "
                    f"cosine={sim!r} required>={min_cosine}"
                )

    report.info["recompute_spot_check"] = {
        "requested_samples": int(samples),
        "usable_samples": len(usable),
        "comparisons": comparisons,
        "min_cosine_required": float(min_cosine),
        "models": {
            name: {
                "count": len(vals),
                "min_cosine": min(vals) if vals else None,
                "mean_cosine": (sum(vals) / len(vals)) if vals else None,
            }
            for name, vals in sorted(scores_by_model.items())
        },
    }


def _quant_search_params(cfg: PipelineConfig, name: str) -> Any:
    q = cfg.qdrant.quant_for(name)
    if _norm(getattr(q, "type", None)) not in {"scalar", "binary"}:
        return None
    from qdrant_client import models
    oversampling = getattr(q, "oversampling", 2.0)
    kwargs: Dict[str, Any] = {
        "ignore": False,
        "rescore": bool(getattr(q, "rescore", True)),
    }
    if oversampling is not None:
        kwargs["oversampling"] = float(oversampling)
    return models.SearchParams(
        quantization=models.QuantizationSearchParams(**kwargs)
    )


def self_retrieval_smoke_test(
    client: Any,
    cfg: PipelineConfig,
    expected_by_point: Mapping[str, Dict[str, Any]],
    report: AuditReport,
    *,
    samples: int,
    seed: int,
    top_k: int,
) -> None:
    """Use each sampled DB vector as a real ANN query and require self to reappear."""
    selected = _balanced_sample_items(expected_by_point, samples, seed + 1009)
    names = collection_names(cfg)
    by_scope: Dict[str, List[str]] = {"person": [], "object": []}
    item_by_pid: Dict[str, Dict[str, Any]] = {}
    for pid, item in selected:
        by_scope[item["scope"]].append(pid)
        item_by_pid[pid] = item

    stored: Dict[str, Any] = {}
    for scope in ("person", "object"):
        stored.update(_retrieve_points_with_vectors(client, names[scope], by_scope[scope]))

    query_count = 0
    missing_self = 0
    non_top1 = 0
    ranks_by_model: Dict[str, List[int]] = defaultdict(list)

    for pid, item in selected:
        rec = stored.get(pid)
        if rec is None:
            report.error(f"self-search sample point missing from Qdrant: {pid}")
            continue
        vmap = _point_vector_map(rec)
        if not isinstance(vmap, dict):
            report.error(f"self-search sample has invalid vector map: {pid}")
            continue
        collection = names[item["scope"]]
        wanted = expected_vectors(cfg, item["scope"])
        for name in sorted(wanted):
            if name not in vmap:
                report.error(f"self-search {pid}: missing stored vector {name}")
                continue
            arr = np.asarray(vmap[name], dtype=np.float32).reshape(-1)
            kwargs: Dict[str, Any] = {
                "collection_name": collection,
                "query": arr.tolist(),
                "using": name,
                "limit": int(top_k),
                "with_payload": False,
                "with_vectors": False,
            }
            params = _quant_search_params(cfg, name)
            if params is not None:
                kwargs["search_params"] = params
            hits = client.query_points(**kwargs).points
            query_count += 1
            ids = [str(h.id) for h in hits]
            if pid not in ids:
                missing_self += 1
                report.error(
                    f"ANN self-retrieval failed: collection={collection} model={name} "
                    f"point={pid} absent from top-{top_k}; returned={ids[:top_k]}"
                )
                continue
            rank = ids.index(pid) + 1
            ranks_by_model[name].append(rank)
            if rank != 1:
                non_top1 += 1
                report.warn(
                    f"ANN self-retrieval not rank-1: collection={collection} model={name} "
                    f"point={pid} rank={rank}/{top_k}. Exact-vector ties may be legitimate."
                )

    report.info["self_retrieval"] = {
        "requested_samples": int(samples),
        "sampled_points": len(selected),
        "top_k": int(top_k),
        "queries": query_count,
        "missing_self": missing_self,
        "non_top1": non_top1,
        "models": {
            name: {
                "queries_with_self": len(ranks),
                "max_rank": max(ranks) if ranks else None,
                "mean_rank": (sum(ranks) / len(ranks)) if ranks else None,
            }
            for name, ranks in sorted(ranks_by_model.items())
        },
    }


def print_summary_final(report: AuditReport) -> None:
    _print_header("DB AUDIT SUMMARY")
    src = report.info.get("source", {})
    if src:
        print(f"source crops        : {src.get('crop_count', 0):,}")
        print(f"source unique IDs   : {src.get('unique_detection_ids', 0):,}")
        print(f"expected person     : {src.get('expected_person_points', 0):,}")
        print(f"expected object     : {src.get('expected_object_points', 0):,}")
        print(f"expected total      : {src.get('expected_total_points', 0):,}")
        print(
            "source duplicate    : "
            f"{src.get('duplicate_group_count', 0):,} groups / "
            f"{src.get('duplicate_point_count', 0):,} points"
        )
        print(
            "source ambiguous    : "
            f"{src.get('ambiguous_group_count', 0):,} groups / "
            f"{src.get('ambiguous_point_count', 0):,} points"
        )
    for collection, info in report.info.get("collections", {}).items():
        print(
            f"{collection:20s}: points={info.get('points_count', 0):,} "
            f"status={info.get('status', '')}"
        )
    for scope, scan in report.info.get("scans", {}).items():
        print(
            f"scan {scope:12s}: points={scan.get('point_count_scanned', 0):,} "
            f"unique_point_ids={scan.get('unique_point_ids', 0):,} "
            f"unique_detection_ids={scan.get('unique_detection_ids', 0):,}"
        )
        if scan.get("duplicate_group_count") or scan.get("ambiguous_group_count"):
            print(
                f"  groups: duplicate={scan.get('duplicate_group_count', 0):,} "
                f"({scan.get('duplicate_point_count', 0):,} points), "
                f"ambiguous={scan.get('ambiguous_group_count', 0):,} "
                f"({scan.get('ambiguous_point_count', 0):,} points)"
            )

    rec = report.info.get("recompute_spot_check")
    if rec:
        print(
            f"recompute spot      : {rec.get('usable_samples', 0):,} crops / "
            f"{rec.get('comparisons', 0):,} vector comparisons"
        )
        for name, row in rec.get("models", {}).items():
            print(
                f"  {name:12s} cosine min={row.get('min_cosine')} "
                f"mean={row.get('mean_cosine')}"
            )

    selfq = report.info.get("self_retrieval")
    if selfq:
        print(
            f"ANN self-search     : queries={selfq.get('queries', 0):,} "
            f"missing_self={selfq.get('missing_self', 0):,} "
            f"non_top1={selfq.get('non_top1', 0):,}"
        )

    print()
    print(f"errors   : {len(report.errors):,}")
    print(f"warnings : {len(report.warnings):,}")
    if report.errors:
        _print_header("ERRORS")
        for i, msg in enumerate(report.errors, 1):
            print(f"[{i}] {msg}")
    if report.warnings:
        _print_header("WARNINGS")
        for i, msg in enumerate(report.warnings, 1):
            print(f"[{i}] {msg}")
    _print_header("FINAL RESULT")
    if report.ok:
        print("PASS - no integrity errors detected.")
    else:
        print(
            "FAIL - integrity errors detected. "
            "Do not treat this DB as final until resolved."
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Final read-only Qdrant integrity audit for forensic DB."
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--stats", default=str(DEFAULT_STATS))
    ap.add_argument("--page-size", type=int, default=128)
    ap.add_argument(
        "--skip-vectors",
        action="store_true",
        help="Skip full per-point vector content/hash checks.",
    )
    ap.add_argument(
        "--no-crop-file-check",
        action="store_true",
        help="Do not stat every crop_path on local disk.",
    )
    ap.add_argument(
        "--skip-recompute",
        action="store_true",
        help="Skip sampled crop -> Router -> embedding recomputation check.",
    )
    ap.add_argument(
        "--spot-samples",
        type=int,
        default=20,
        help="Total person+object crop samples to re-embed (default: 20).",
    )
    ap.add_argument(
        "--recompute-min-cosine",
        type=float,
        default=0.999,
        help="Minimum cosine between recomputed and stored vector (default: 0.999).",
    )
    ap.add_argument(
        "--skip-self-search",
        action="store_true",
        help="Skip ANN self-retrieval smoke test.",
    )
    ap.add_argument(
        "--self-search-samples",
        type=int,
        default=20,
        help="Total points sampled for ANN self-retrieval (default: 20).",
    )
    ap.add_argument(
        "--self-top-k",
        type=int,
        default=5,
        help="Self point must appear within this ANN top-k (default: 5).",
    )
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    if args.page_size <= 0:
        raise ValueError("--page-size must be >= 1")
    if args.spot_samples < 0 or args.self_search_samples < 0:
        raise ValueError("sample counts must be >= 0")
    if not (0.0 < args.recompute_min_cosine <= 1.000001):
        raise ValueError("--recompute-min-cosine must be in (0, 1]")
    if args.self_top_k <= 0:
        raise ValueError("--self-top-k must be >= 1")

    config_path = Path(args.config).expanduser().resolve()
    stats_path = Path(args.stats).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"stats not found: {stats_path}")

    report = AuditReport()
    cfg = PipelineConfig.load(str(config_path))
    names = collection_names(cfg)
    person_vecs = expected_vectors(cfg, "person")
    object_vecs = expected_vectors(cfg, "object")

    _print_header("LOAD SOURCE")
    print("config:", config_path)
    print("stats :", stats_path)
    print("collections:", names)
    print("person vectors:", person_vecs)
    print("object vectors:", object_vecs)
    crops, expected_by_point = load_source(stats_path, cfg, report)
    audit_source_content(
        crops,
        report,
        check_crop_files=not args.no_crop_file_check,
    )
    print("source crops:", f"{len(crops):,}")

    from qdrant_client import QdrantClient
    client = QdrantClient(url=cfg.qdrant.url, timeout=120)

    _print_header("QDRANT SCHEMA / STORAGE POLICY")
    try:
        client.get_collections()
    except Exception as e:
        report.error(f"Qdrant connection failed: {e}")
        print_summary_final(report)
        return 1

    person_info = audit_schema_strict(client, names["person"], person_vecs, cfg, report)
    object_info = audit_schema_strict(client, names["object"], object_vecs, cfg, report)
    if person_info is None or object_info is None:
        print_summary_final(report)
        return 1

    _print_header("POINT / PAYLOAD / VECTOR SCAN")
    with_vectors = not args.skip_vectors
    print("vector content/hash check:", "ON" if with_vectors else "OFF")
    person_labels = {str(x).lower() for x in cfg.person_labels}

    person_scan = scan_collection_final(
        client,
        names["person"],
        "person",
        person_vecs,
        expected_by_point,
        person_labels,
        report,
        page_size=args.page_size,
        with_vectors=with_vectors,
    )
    print(f"{names['person']}: {person_scan['point_count_scanned']:,} points scanned")

    object_scan = scan_collection_final(
        client,
        names["object"],
        "object",
        object_vecs,
        expected_by_point,
        person_labels,
        report,
        page_size=args.page_size,
        with_vectors=with_vectors,
    )
    print(f"{names['object']}: {object_scan['point_count_scanned']:,} points scanned")
    report.info["scans"] = {"person": person_scan, "object": object_scan}

    cross = person_scan["detection_ids"] & object_scan["detection_ids"]
    if cross:
        report.error(
            f"same logical detection_id exists in both collections: {len(cross)}; "
            f"examples={_safe_examples(sorted(cross))}"
        )

    for scope, info, scan in (
        ("person", person_info, person_scan),
        ("object", object_info, object_scan),
    ):
        server_count = int(getattr(info, "points_count", 0) or 0)
        scanned_count = int(scan["point_count_scanned"])
        if server_count != scanned_count:
            report.error(
                f"{names[scope]}: server points_count != scroll count "
                f"{server_count} != {scanned_count}"
            )

    total_scanned = person_scan["point_count_scanned"] + object_scan["point_count_scanned"]
    if total_scanned != len(crops):
        report.error(
            f"total Qdrant points != source crops: qdrant={total_scanned}, source={len(crops)}"
        )
    report.info["hybrid_c_point_preservation"] = {
        "source_points": len(crops),
        "qdrant_points": total_scanned,
        "preserved": total_scanned == len(crops),
        "meaning": (
            "semantic duplicate groups must not reduce Qdrant point count; "
            "only upstream exact-ID hard dedup may reduce the source before build"
        ),
    }
    audit_duplicate_groups(crops, person_scan, object_scan, report)
    audit_ambiguous_groups(crops, person_scan, object_scan, report)

    if not args.skip_recompute and args.spot_samples > 0:
        _print_header("RECOMPUTE SPOT-CHECK")
        print(
            f"samples={args.spot_samples}, seed={args.seed}, "
            f"required cosine>={args.recompute_min_cosine}"
        )
        recompute_spot_check(
            client,
            cfg,
            expected_by_point,
            report,
            samples=args.spot_samples,
            seed=args.seed,
            min_cosine=args.recompute_min_cosine,
        )
    else:
        report.info["recompute_spot_check"] = {"skipped": True}

    if not args.skip_self_search and args.self_search_samples > 0:
        _print_header("ANN SELF-RETRIEVAL SMOKE TEST")
        print(
            f"samples={args.self_search_samples}, seed={args.seed}, top_k={args.self_top_k}"
        )
        self_retrieval_smoke_test(
            client,
            cfg,
            expected_by_point,
            report,
            samples=args.self_search_samples,
            seed=args.seed,
            top_k=args.self_top_k,
        )
    else:
        report.info["self_retrieval"] = {"skipped": True}

    print_summary_final(report)

    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "ok": report.ok,
                    "error_count": len(report.errors),
                    "warning_count": len(report.warnings),
                    "errors": report.errors,
                    "warnings": report.warnings,
                    "info": to_serializable(report.info),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print("\nreport:", report_path)

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
