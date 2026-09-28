#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

import requests


def parse_args():
    p = argparse.ArgumentParser(
        description="Audit production Qdrant schema and only newly ingested video points"
    )
    p.add_argument("--url", default="http://localhost:6333")
    p.add_argument("--video-stem", required=True)
    p.add_argument("--person-collection", default="forensic_person")
    p.add_argument("--object-collection", default="forensic_object")
    p.add_argument("--limit", type=int, default=1000)
    return p.parse_args()


def get_collection(url: str, collection: str) -> dict:
    r = requests.get(f"{url}/collections/{collection}", timeout=30)
    r.raise_for_status()
    return r.json()["result"]


def collection_named_dims(info: dict) -> Dict[str, int]:
    vectors = (
        info.get("config", {})
            .get("params", {})
            .get("vectors", {})
    )

    if isinstance(vectors, dict) and "size" in vectors:
        return {"default": int(vectors["size"])}

    out = {}
    if isinstance(vectors, dict):
        for name, spec in vectors.items():
            if isinstance(spec, dict) and "size" in spec:
                out[name] = int(spec["size"])
    return out


def scroll_video_points(
    url: str,
    collection: str,
    video_stem: str,
    scope: str,
    limit: int,
):
    payload_filter = {
        "must": [
            {"key": "source", "match": {"value": "final_db_candidates"}},
            {"key": "video_stem", "match": {"value": video_stem}},
            {"key": "candidate_scope", "match": {"value": scope}},
        ]
    }

    points = []
    offset = None

    while len(points) < limit:
        body: Dict[str, Any] = {
            "limit": min(256, limit - len(points)),
            "with_payload": True,
            "with_vector": True,
            "filter": payload_filter,
        }
        if offset is not None:
            body["offset"] = offset

        r = requests.post(
            f"{url}/collections/{collection}/points/scroll",
            json=body,
            timeout=60,
        )
        r.raise_for_status()
        result = r.json()["result"]
        batch = result.get("points", [])
        points.extend(batch)

        offset = result.get("next_page_offset")
        if not batch or offset is None:
            break

    return points


def vector_dims(point: dict) -> Dict[str, int]:
    v = point.get("vector")
    if isinstance(v, dict):
        return {k: len(x) for k, x in v.items()}
    if isinstance(v, list):
        return {"default": len(v)}
    return {}


def audit_points(points, scope, expected_dims):
    errors = []
    dims_seen = set()

    required_common = [
        "media_type", "source", "video_stem", "video", "video_path",
        "detection_id", "crop_path", "timestamp_sec",
        "track_id", "long_track_id", "selected_track_id", "selected_rank",
        "track_key", "confidence", "quality_score", "candidate_scope",
        "label", "frame_idx", "bbox",
    ]

    for pt in points:
        p = pt.get("payload") or {}
        missing = [k for k in required_common if k not in p]
        if missing:
            errors.append(f"{pt.get('id')}: missing={missing}")
            continue

        if p.get("source") != "final_db_candidates":
            errors.append(f"{pt.get('id')}: wrong source={p.get('source')}")

        if str(p.get("candidate_scope")).lower() != scope:
            errors.append(
                f"{pt.get('id')}: wrong candidate_scope={p.get('candidate_scope')}"
            )

        track_id = int(p["track_id"])
        selected = int(p["selected_track_id"])
        if track_id != selected:
            errors.append(
                f"{pt.get('id')}: track_id={track_id} != selected_track_id={selected}"
            )

        expected_key = f"{p['video_stem']}/{scope}_{track_id}"
        if p.get("track_key") != expected_key:
            errors.append(
                f"{pt.get('id')}: track_key={p.get('track_key')} "
                f"expected={expected_key}"
            )

        if scope == "person":
            cid = p.get("canonical_person_id")
            if cid is None:
                errors.append(f"{pt.get('id')}: missing canonical_person_id")
            elif int(cid) != track_id:
                errors.append(
                    f"{pt.get('id')}: canonical_person_id={cid} != track_id={track_id}"
                )
        else:
            mid = p.get("merged_object_track_id")
            if mid is None:
                errors.append(f"{pt.get('id')}: missing merged_object_track_id")
            elif int(mid) != track_id:
                errors.append(
                    f"{pt.get('id')}: merged_object_track_id={mid} != track_id={track_id}"
                )

        d = vector_dims(pt)
        dims_seen.add(tuple(sorted(d.items())))
        if d != expected_dims:
            errors.append(
                f"{pt.get('id')}: vector_dims={d} expected={expected_dims}"
            )

    return errors, [dict(x) for x in sorted(dims_seen)]


def main():
    args = parse_args()

    person_expected = {"siglip2": 768, "irra": 512, "solider": 1024}
    object_expected = {"siglip2": 768, "dinov2": 1536}

    person_info = get_collection(args.url, args.person_collection)
    object_info = get_collection(args.url, args.object_collection)

    person_schema = collection_named_dims(person_info)
    object_schema = collection_named_dims(object_info)

    person_points = scroll_video_points(
        args.url,
        args.person_collection,
        args.video_stem,
        "person",
        args.limit,
    )
    object_points = scroll_video_points(
        args.url,
        args.object_collection,
        args.video_stem,
        "object",
        args.limit,
    )

    pe, pd = audit_points(person_points, "person", person_expected)
    oe, od = audit_points(object_points, "object", object_expected)

    schema_errors = []
    if person_schema != person_expected:
        schema_errors.append(
            f"person schema={person_schema} expected={person_expected}"
        )
    if object_schema != object_expected:
        schema_errors.append(
            f"object schema={object_schema} expected={object_expected}"
        )

    print("=" * 92)
    print("PRODUCTION VIDEO QDRANT AUDIT")
    print("=" * 92)
    print("video_stem        :", args.video_stem)
    print()
    print("person collection :", args.person_collection)
    print("person schema     :", person_schema)
    print("person points     :", len(person_points))
    print("person point dims :", pd)
    print("person errors     :", len(pe))
    print()
    print("object collection :", args.object_collection)
    print("object schema     :", object_schema)
    print("object points     :", len(object_points))
    print("object point dims :", od)
    print("object errors     :", len(oe))
    print()
    print("schema errors     :", len(schema_errors))
    print("=" * 92)

    all_errors = schema_errors + pe + oe
    if all_errors:
        print("\nERRORS")
        for e in all_errors[:100]:
            print("-", e)
        sys.exit(1)

    print("PASS: production schema and this video's final DB payloads are consistent.")


if __name__ == "__main__":
    main()
