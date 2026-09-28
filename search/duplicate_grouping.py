from __future__ import annotations

"""
duplicate_grouping.py
=====================

Hybrid C search-time duplicate grouping helper.

철학
----
1) semantic duplicate는 DB point를 삭제하지 않는다.
2) 같은 duplicate_group_id를 가진 결과만 검색 단계에서 collapse 한다.
3) ambiguous_group_id는 절대 collapse 기준으로 사용하지 않는다.
4) Exact-ID / hard dedup은 이미 사전에 처리되었다고 가정한다.

이 모듈은 image_search.py 와 search_db.py 가 같은 grouping 정책을 공유하도록
만들어진 공용 helper 이다.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class GroupingConfig:
    """런타임에서 사용할 검색 grouping 설정의 최소 공통 표현."""

    enabled: bool = True
    overfetch_factor: int = 3
    max_fetch: int = 200
    group_payload_key: str = "duplicate_group_id"
    ambiguous_payload_key: str = "ambiguous_group_id"


def _coerce_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true/false (got {value!r})")
    return value


def _coerce_positive_int(value: Any, *, name: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum} (got {value!r})")
    return value


def _coerce_nonempty_str(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string (got {value!r})")
    return value.strip()


def resolve_grouping_config(source: Any = None) -> GroupingConfig:
    """
    다양한 입력 형태에서 GroupingConfig 를 만든다.

    지원 입력:
      - None -> 기본값
      - GroupingConfig 자체
      - PipelineConfig (cfg.duplicate_grouping 사용)
      - DuplicateGroupingSpec 류 객체
      - dict 또는 {'duplicate_grouping': {...}} 형태
    """
    if source is None:
        return GroupingConfig()

    if isinstance(source, GroupingConfig):
        return source

    # PipelineConfig-like: cfg.duplicate_grouping
    if hasattr(source, "duplicate_grouping"):
        source = getattr(source, "duplicate_grouping")

    # dict wrapper: {'duplicate_grouping': {...}}
    if isinstance(source, Mapping) and "duplicate_grouping" in source:
        source = source["duplicate_grouping"]

    if isinstance(source, Mapping):
        enabled = _coerce_bool(
            source.get("enabled", True),
            name="duplicate_grouping.enabled",
        )
        overfetch_factor = _coerce_positive_int(
            source.get("overfetch_factor", 3),
            name="duplicate_grouping.overfetch_factor",
        )
        max_fetch = _coerce_positive_int(
            source.get("max_fetch", 200),
            name="duplicate_grouping.max_fetch",
        )
        group_payload_key = _coerce_nonempty_str(
            source.get("group_payload_key", "duplicate_group_id"),
            name="duplicate_grouping.group_payload_key",
        )
        ambiguous_payload_key = _coerce_nonempty_str(
            source.get("ambiguous_payload_key", "ambiguous_group_id"),
            name="duplicate_grouping.ambiguous_payload_key",
        )
    else:
        # dataclass/object-like
        enabled = _coerce_bool(
            getattr(source, "enabled", True),
            name="duplicate_grouping.enabled",
        )
        overfetch_factor = _coerce_positive_int(
            getattr(source, "overfetch_factor", 3),
            name="duplicate_grouping.overfetch_factor",
        )
        max_fetch = _coerce_positive_int(
            getattr(source, "max_fetch", 200),
            name="duplicate_grouping.max_fetch",
        )
        group_payload_key = _coerce_nonempty_str(
            getattr(source, "group_payload_key", "duplicate_group_id"),
            name="duplicate_grouping.group_payload_key",
        )
        ambiguous_payload_key = _coerce_nonempty_str(
            getattr(source, "ambiguous_payload_key", "ambiguous_group_id"),
            name="duplicate_grouping.ambiguous_payload_key",
        )

    if group_payload_key == ambiguous_payload_key:
        raise ValueError(
            "duplicate_grouping.group_payload_key and ambiguous_payload_key "
            "must be different"
        )

    return GroupingConfig(
        enabled=enabled,
        overfetch_factor=overfetch_factor,
        max_fetch=max_fetch,
        group_payload_key=group_payload_key,
        ambiguous_payload_key=ambiguous_payload_key,
    )


def object_retrieval_limit(requested_limit: int, source: Any = None) -> int:
    """
    object 검색에서 collapse 이후 결과 부족을 줄이기 위한 over-fetch 크기.

    규칙:
      overfetch = requested_limit * overfetch_factor
      bounded   = min(overfetch, max_fetch)
      return max(requested_limit, bounded)

    사용자가 max_fetch보다 큰 limit을 요청하면 requested_limit 미만으로 줄이지 않는다.
    """
    requested_limit = _coerce_positive_int(
        requested_limit,
        name="requested_limit",
    )
    cfg = resolve_grouping_config(source)
    if not cfg.enabled:
        return requested_limit

    expanded = requested_limit * cfg.overfetch_factor
    capped = min(expanded, cfg.max_fetch)
    return max(requested_limit, capped)


def object_max_retrieval_limit(
    requested_limit: int,
    source: Any = None,
) -> int:
    """
    Hybrid-C object search의 절대 fetch 상한.

    initial retrieval은 object_retrieval_limit()의 일반 over-fetch를 사용하고,
    collapse 후 Top-K가 부족할 때에만 이 값까지 한 번 더 조회한다.

    max_fetch가 requested_limit보다 작더라도 사용자가 요청한 Top-K보다
    적게 조회해서는 안 되므로 max(requested_limit, max_fetch)를 반환한다.
    """
    requested_limit = _coerce_positive_int(
        requested_limit,
        name="requested_limit",
    )

    cfg = resolve_grouping_config(
        source
    )

    if not cfg.enabled:
        return requested_limit

    return max(
        requested_limit,
        int(cfg.max_fetch),
    )


def _clean_group_id(payload: Mapping[str, Any], key: str) -> str:
    raw = payload.get(key)
    if raw is None:
        return ""
    return str(raw).strip()


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    """대표 선정용 숫자를 안전하게 float로 변환한다. NaN은 default 처리한다."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if out != out:  # NaN
        return float(default)
    return out


def _row_representative_sort_key(
    row: Mapping[str, Any],
    *,
    score_key: str,
) -> Tuple[float, float, str, str]:
    """
    Hybrid-C deterministic representative ordering.

    우선순위:
      1) query-time final score DESC
      2) detector confidence DESC
      3) detection_id ASC
      4) point_id ASC (완전 동률의 마지막 안정화 키)
    """
    score = _safe_float(row.get(score_key, 0.0))
    confidence = _safe_float(
        row.get("confidence", row.get("det_confidence", 0.0))
    )
    detection_id = str(row.get("detection_id") or "")
    point_id = str(row.get("point_id") or "")
    return (-score, -confidence, detection_id, point_id)


def _hit_representative_sort_key(
    hit: Any,
    *,
    score_attr: str,
) -> Tuple[float, float, str, str]:
    payload = _hit_payload_dict(hit)

    score_val = getattr(hit, score_attr, None)
    if score_val is None:
        score_val = getattr(hit, "retrieval_score", 0.0)

    # QdrantStore stores the RF-DETR confidence in payload["score"].
    # Some callers may also expose confidence/det_confidence explicitly, so keep
    # those as higher-priority aliases. Never use hit.score here: hit.score is
    # the query-time retrieval/final score, not detector confidence.
    confidence = payload.get(
        "confidence",
        payload.get(
            "det_confidence",
            payload.get("score", getattr(hit, "confidence", 0.0)),
        ),
    )
    detection_id = str(payload.get("detection_id") or "")
    point_id = str(getattr(hit, "point_id", "") or "")

    return (
        -_safe_float(score_val),
        -_safe_float(confidence),
        detection_id,
        point_id,
    )


def _row_member_view(
    row: Mapping[str, Any],
    *,
    score_key: str,
    group_payload_key: str,
    ambiguous_payload_key: str,
) -> Dict[str, Any]:
    return {
        "point_id": row.get("point_id"),
        "detection_id": row.get("detection_id"),
        "image_id": row.get("image_id"),
        "label": row.get("label"),
        "confidence": _safe_float(
            row.get("confidence", row.get("det_confidence", 0.0))
        ),
        "crop_path": row.get("crop_path"),
        "bbox": list(row.get("bbox") or []),
        "frame_idx": row.get("frame_idx"),
        "track_id": row.get("track_id"),
        "score": _safe_float(row.get(score_key, 0.0)),
        # Preserve member-level relation metadata so collapsing a duplicate group
        # does not hide an ambiguous relation attached only to a non-representative.
        "duplicate_group_id": row.get(group_payload_key),
        "ambiguous_group_id": row.get(ambiguous_payload_key),
        "duplicate_status": row.get("duplicate_status"),
    }


def collapse_duplicate_group_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    score_key: str,
    source: Any = None,
    member_view_fn: Optional[Callable[[Mapping[str, Any], str], Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    dict row 목록을 duplicate_group_id 기준으로 collapse 한다.

    - group key가 없으면 independent result.
    - ambiguous_group_id는 참고만 하고 collapse에 사용하지 않는다.
    - 각 group 대표는 score DESC -> detector confidence DESC -> detection_id ASC
      -> point_id ASC 순으로 결정한다.
    - 반환 rows는 새 dict 들이며, representative에 duplicate_group_members가 붙는다.
    """
    top_k = _coerce_positive_int(top_k, name="top_k")
    cfg = resolve_grouping_config(source)
    view_fn = member_view_fn or (
        lambda row, score_key: _row_member_view(
            row,
            score_key=score_key,
            group_payload_key=cfg.group_payload_key,
            ambiguous_payload_key=cfg.ambiguous_payload_key,
        )
    )

    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: _row_representative_sort_key(
            row,
            score_key=score_key,
        ),
    )

    if not cfg.enabled:
        final_rows = ordered[:top_k]
        return final_rows, {
            "enabled": False,
            "raw_hit_count": len(rows),
            "unique_result_count_before_limit": len(ordered),
            "final_result_count": len(final_rows),
            "collapsed_hit_count": 0,
            "collapsed_group_count": 0,
            "grouped_result_count": 0,
            "underfilled_after_collapse": len(final_rows) < top_k,
            "overfetch_factor": cfg.overfetch_factor,
            "max_fetch": cfg.max_fetch,
            "collapse_key": cfg.group_payload_key,
            "ambiguous_group_collapsed": False,
            "representative_tie_break": [
                "final_score_desc",
                "detector_confidence_desc",
                "detection_id_asc",
                "point_id_asc",
            ],
        }

    collapsed: List[Dict[str, Any]] = []
    representative_pos_by_group: Dict[str, int] = {}
    member_count_by_group: Dict[str, int] = {}
    collapsed_hit_count = 0

    for row in ordered:
        group_id = _clean_group_id(row, cfg.group_payload_key)

        # ambiguous_group_id는 존재해도 collapse 키로 사용하지 않는다.
        if not group_id:
            collapsed.append(row)
            continue

        if group_id not in representative_pos_by_group:
            representative = dict(row)
            representative[cfg.group_payload_key] = group_id
            representative["duplicate_group_retrieved_member_count"] = 1
            representative["duplicate_group_members"] = [
                view_fn(row, score_key)
            ]
            amb_id = _clean_group_id(row, cfg.ambiguous_payload_key)
            if amb_id:
                representative["duplicate_group_ambiguous_group_ids"] = [amb_id]
            representative_pos_by_group[group_id] = len(collapsed)
            member_count_by_group[group_id] = 1
            collapsed.append(representative)
            continue

        rep = collapsed[representative_pos_by_group[group_id]]

        # A duplicate group is local to one source image by Hybrid-C contract.
        # Fail closed here as well so a stale/corrupt DB cannot silently collapse
        # unrelated detections that merely share a malformed group id.
        rep_image = str(rep.get("image_id") or "").strip()
        row_image = str(row.get("image_id") or "").strip()
        if rep_image and row_image and rep_image != row_image:
            raise RuntimeError(
                "duplicate_group_id spans multiple image_id values during search: "
                f"group={group_id!r}, representative={rep_image!r}, member={row_image!r}"
            )

        member_count_by_group[group_id] += 1
        collapsed_hit_count += 1
        rep["duplicate_group_retrieved_member_count"] = (
            int(rep.get("duplicate_group_retrieved_member_count", 1)) + 1
        )
        members = rep.setdefault("duplicate_group_members", [])
        if not isinstance(members, list):
            members = []
            rep["duplicate_group_members"] = members
        members.append(view_fn(row, score_key))

        amb_id = _clean_group_id(row, cfg.ambiguous_payload_key)
        if amb_id:
            agg = rep.setdefault("duplicate_group_ambiguous_group_ids", [])
            if not isinstance(agg, list):
                agg = []
                rep["duplicate_group_ambiguous_group_ids"] = agg
            if amb_id not in agg:
                agg.append(amb_id)
                agg.sort()

    final_rows = collapsed[:top_k]
    collapsed_group_count = sum(
        1 for count in member_count_by_group.values() if count > 1
    )
    grouped_result_count = sum(
        1 for row in final_rows if _clean_group_id(row, cfg.group_payload_key)
    )

    return final_rows, {
        "enabled": True,
        "raw_hit_count": len(rows),
        "unique_result_count_before_limit": len(collapsed),
        "final_result_count": len(final_rows),
        "collapsed_hit_count": collapsed_hit_count,
        "collapsed_group_count": collapsed_group_count,
        "grouped_result_count": grouped_result_count,
        "underfilled_after_collapse": len(final_rows) < top_k,
        "overfetch_factor": cfg.overfetch_factor,
        "max_fetch": cfg.max_fetch,
        "collapse_key": cfg.group_payload_key,
        "ambiguous_group_collapsed": False,
        "representative_tie_break": [
            "final_score_desc",
            "detector_confidence_desc",
            "detection_id_asc",
            "point_id_asc",
        ],
    }


def _hit_payload_dict(hit: Any) -> MutableMapping[str, Any]:
    payload = getattr(hit, "payload", None)
    if isinstance(payload, dict):
        return payload
    payload = {}
    setattr(hit, "payload", payload)
    return payload


def _hit_member_view(
    hit: Any,
    *,
    score_attr: str = "score",
    group_payload_key: str = "duplicate_group_id",
    ambiguous_payload_key: str = "ambiguous_group_id",
) -> Dict[str, Any]:
    score_val = getattr(hit, score_attr, None)
    if score_val is None:
        score_val = getattr(hit, "retrieval_score", 0.0)

    payload = _hit_payload_dict(hit)

    return {
        "point_id": getattr(hit, "point_id", None),
        "detection_id": payload.get("detection_id"),
        "image_id": getattr(hit, "image_id", None),
        "label": getattr(hit, "label", None),
        "confidence": _safe_float(
            payload.get(
                "confidence",
                payload.get("det_confidence", payload.get("score", 0.0)),
            )
        ),
        "crop_path": getattr(hit, "crop_path", None),
        "bbox": list(getattr(hit, "bbox", None) or []),
        "frame_idx": int(getattr(hit, "frame_idx", 0) or 0),
        "track_id": getattr(hit, "track_id", None),
        "score": _safe_float(score_val),
        "retrieval_score": _safe_float(getattr(hit, "retrieval_score", 0.0)),
        "duplicate_group_id": payload.get(group_payload_key),
        "ambiguous_group_id": payload.get(ambiguous_payload_key),
        "duplicate_status": payload.get("duplicate_status"),
    }


def collapse_duplicate_group_hits(
    hits: Sequence[Any],
    *,
    limit: int,
    source: Any = None,
    score_attr: str = "score",
    member_view_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
    set_rank: bool = True,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    SearchHit 류 객체 목록을 duplicate_group_id 기준으로 collapse 한다.

    입력 hits의 기존 순서에 의존하지 않는다. 대표는 공통 deterministic 규칙:
      final score DESC -> detector confidence DESC -> detection_id ASC -> point_id ASC
    로 정한다.
    """
    limit = _coerce_positive_int(limit, name="limit")
    cfg = resolve_grouping_config(source)
    view_fn = member_view_fn or (
        lambda hit: _hit_member_view(
            hit,
            score_attr=score_attr,
            group_payload_key=cfg.group_payload_key,
            ambiguous_payload_key=cfg.ambiguous_payload_key,
        )
    )

    ordered_hits = sorted(
        list(hits),
        key=lambda hit: _hit_representative_sort_key(
            hit,
            score_attr=score_attr,
        ),
    )

    if not cfg.enabled:
        final_hits = list(ordered_hits[:limit])
        if set_rank:
            for rank, hit in enumerate(final_hits, 1):
                setattr(hit, "rank", rank)
        return final_hits, {
            "enabled": False,
            "raw_hit_count": len(hits),
            "unique_result_count_before_limit": len(ordered_hits),
            "final_result_count": len(final_hits),
            "collapsed_hit_count": 0,
            "collapsed_group_count": 0,
            "grouped_result_count": 0,
            "underfilled_after_collapse": len(final_hits) < limit,
            "overfetch_factor": cfg.overfetch_factor,
            "max_fetch": cfg.max_fetch,
            "collapse_key": cfg.group_payload_key,
            "ambiguous_group_collapsed": False,
            "representative_tie_break": [
                "final_score_desc",
                "detector_confidence_desc",
                "detection_id_asc",
                "point_id_asc",
            ],
        }

    kept: List[Any] = []
    representative_by_group: Dict[str, Any] = {}
    member_count_by_group: Dict[str, int] = {}
    collapsed_hit_count = 0

    for hit in ordered_hits:
        payload = _hit_payload_dict(hit)
        group_id = _clean_group_id(payload, cfg.group_payload_key)

        # ambiguous_group_id는 절대 collapse 기준으로 사용하지 않는다.
        if not group_id:
            kept.append(hit)
            continue

        representative = representative_by_group.get(group_id)
        if representative is None:
            representative_by_group[group_id] = hit
            member_count_by_group[group_id] = 1
            payload[cfg.group_payload_key] = group_id
            payload["duplicate_group_members"] = [view_fn(hit)]
            amb_id = _clean_group_id(payload, cfg.ambiguous_payload_key)
            if amb_id:
                payload["duplicate_group_ambiguous_group_ids"] = [amb_id]
            kept.append(hit)
            continue

        rep_payload = _hit_payload_dict(representative)

        rep_image = str(
            getattr(representative, "image_id", None)
            or rep_payload.get("image_id")
            or ""
        ).strip()
        hit_image = str(
            getattr(hit, "image_id", None)
            or payload.get("image_id")
            or ""
        ).strip()
        if rep_image and hit_image and rep_image != hit_image:
            raise RuntimeError(
                "duplicate_group_id spans multiple image_id values during search: "
                f"group={group_id!r}, representative={rep_image!r}, member={hit_image!r}"
            )

        member_count_by_group[group_id] += 1
        collapsed_hit_count += 1

        members = rep_payload.setdefault("duplicate_group_members", [])
        if not isinstance(members, list):
            members = []
            rep_payload["duplicate_group_members"] = members
        members.append(view_fn(hit))

        amb_id = _clean_group_id(payload, cfg.ambiguous_payload_key)
        if amb_id:
            agg = rep_payload.setdefault("duplicate_group_ambiguous_group_ids", [])
            if not isinstance(agg, list):
                agg = []
                rep_payload["duplicate_group_ambiguous_group_ids"] = agg
            if amb_id not in agg:
                agg.append(amb_id)
                agg.sort()

    final_hits = kept[:limit]
    if set_rank:
        for rank, hit in enumerate(final_hits, 1):
            setattr(hit, "rank", rank)

    collapsed_group_count = sum(
        1 for count in member_count_by_group.values() if count > 1
    )
    grouped_result_count = sum(
        1 for hit in final_hits
        if _clean_group_id(_hit_payload_dict(hit), cfg.group_payload_key)
    )

    return final_hits, {
        "enabled": True,
        "raw_hit_count": len(hits),
        "unique_result_count_before_limit": len(kept),
        "final_result_count": len(final_hits),
        "collapsed_hit_count": collapsed_hit_count,
        "collapsed_group_count": collapsed_group_count,
        "grouped_result_count": grouped_result_count,
        "underfilled_after_collapse": len(final_hits) < limit,
        "overfetch_factor": cfg.overfetch_factor,
        "max_fetch": cfg.max_fetch,
        "collapse_key": cfg.group_payload_key,
        "ambiguous_group_collapsed": False,
        "representative_tie_break": [
            "final_score_desc",
            "detector_confidence_desc",
            "detection_id_asc",
            "point_id_asc",
        ],
    }


__all__ = [
    "GroupingConfig",
    "resolve_grouping_config",
    "object_retrieval_limit",
    "object_max_retrieval_limit",
    "collapse_duplicate_group_rows",
    "collapse_duplicate_group_hits",
]

