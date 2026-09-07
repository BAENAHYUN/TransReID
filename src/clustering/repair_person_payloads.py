from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_video_db as bvd
from config import PipelineConfig
from qdrant_store import QdrantStore
from rfdetr_adapter import from_rfdetr

COLLECTION = "forensic_person"
STATS_PATH = ROOT / "data" / "crops" / "filter_stats.json"
SCAN_BATCH = 1024
ADAPTER_BATCH = 2048
UPSERT_BATCH = 128

EXPECTED_TOTAL = 229_005
EXPECTED_IMAGE = 96_660
EXPECTED_VIDEO = 132_345
EXPECTED_DAMAGED = 918
EXPECTED_DAMAGED_IMAGE = 188
EXPECTED_DAMAGED_VIDEO = 730


def detection_media_type(det):
    value = getattr(det, "media_type", None)
    extra = getattr(det, "extra", None)
    extra = extra if isinstance(extra, dict) else {}
    if value is None:
        value = extra.get("media_type")
    if value is None:
        has_video = bool(extra.get("video") or extra.get("video_name"))
        has_track = (
            getattr(det, "track_id", None) is not None
            or extra.get("track_id") is not None
        )
        value = "video" if (has_video or has_track) else "image"
    return str(value).strip().lower()


def count_media(client):
    total = client.get_collection(COLLECTION).points_count
    counts = {}
    for media_type in ("image", "video"):
        f = Filter(
            must=[FieldCondition(key="media_type", match=MatchValue(value=media_type))]
        )
        counts[media_type] = client.count(
            collection_name=COLLECTION,
            count_filter=f,
            exact=True,
        ).count
    counts["missing"] = total - counts["image"] - counts["video"]
    counts["total"] = total
    return counts


def find_damaged_ids(client):
    damaged = set()
    offset = None
    scanned = 0
    print("\n[1/6] Scan forensic_person payloads")
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=SCAN_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            payload = p.payload or {}
            if payload.get("media_type") not in ("image", "video"):
                damaged.add(str(p.id))
        scanned += len(points)
        if scanned % 20000 < SCAN_BATCH:
            print(f"  scanned={scanned:,} damaged={len(damaged):,}")
        if offset is None:
            break
    print(f"  damaged point ids: {len(damaged):,}")
    return damaged


def register_if_damaged(store, damaged_ids, recovered, det):
    point_id, _ = store._stable_point_id(det)
    point_id = str(point_id)
    if point_id in damaged_ids:
        if point_id in recovered:
            raise RuntimeError(f"duplicate reconstructed point id: {point_id}")
        recovered[point_id] = det


def reconstruct_video(store, damaged_ids, recovered):
    print("\n[2/6] Reconstruct VIDEO detections from stitched Person V4.5")
    video_index = bvd.build_video_index()
    records = bvd.discover_kind_records("person", video_index, None)
    for idx, record in enumerate(records, 1):
        det = bvd.record_to_detection(record)
        register_if_damaged(store, damaged_ids, recovered, det)
        if idx % 20000 == 0:
            print(f"  video rows={idx:,}/{len(records):,} matched={len(recovered):,}")
    print(f"  video rows scanned: {len(records):,}")


def reconstruct_images(store, damaged_ids, recovered):
    print("\n[3/6] Reconstruct IMAGE detections from filter_stats.json")
    if not STATS_PATH.is_file():
        raise FileNotFoundError(STATS_PATH)
    with STATS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    crops = data.get("crops", [])
    if not crops:
        raise RuntimeError("filter_stats.json has no crops")

    before = set(recovered)
    for start in range(0, len(crops), ADAPTER_BATCH):
        batch = crops[start:start + ADAPTER_BATCH]
        detections, _ = from_rfdetr(batch, load_mode="path")
        for det in detections:
            if str(det.label).lower() != "person":
                continue
            register_if_damaged(store, damaged_ids, recovered, det)

        done = min(start + ADAPTER_BATCH, len(crops))
        if done % 20000 < ADAPTER_BATCH:
            image_matches = len(set(recovered) - before)
            print(f"  crop metadata={done:,}/{len(crops):,} image matched={image_matches:,}")
        if len(recovered) == len(damaged_ids):
            break


def retrieve_vectors(client, point_ids):
    print("\n[4/6] Retrieve existing vectors for damaged points")
    vectors = {}
    ids = list(point_ids)
    for start in range(0, len(ids), UPSERT_BATCH):
        batch_ids = ids[start:start + UPSERT_BATCH]
        points = client.retrieve(
            collection_name=COLLECTION,
            ids=batch_ids,
            with_payload=False,
            with_vectors=True,
        )
        for p in points:
            if not isinstance(p.vector, dict):
                raise RuntimeError(
                    f"point {p.id}: expected named-vector dict, got {type(p.vector).__name__}"
                )
            vectors[str(p.id)] = {
                name: np.asarray(value, dtype=np.float32)
                for name, value in p.vector.items()
            }
    missing = set(point_ids) - set(vectors)
    if missing:
        raise RuntimeError(
            f"{len(missing)} damaged points could not be retrieved with vectors"
        )
    print(f"  vectors retrieved: {len(vectors):,}")
    return vectors


def restore(store, recovered, vectors):
    print("\n[5/6] Restore ONLY damaged points (no embedding recompute)")
    ids = list(recovered)
    for start in range(0, len(ids), UPSERT_BATCH):
        batch_ids = ids[start:start + UPSERT_BATCH]
        detections = [recovered[pid] for pid in batch_ids]
        vector_maps = [vectors[pid] for pid in batch_ids]
        committed = store.upsert(
            detections,
            vector_maps,
            batch_size=UPSERT_BATCH,
        )
        if committed != len(batch_ids):
            raise RuntimeError(
                f"upsert count mismatch: expected={len(batch_ids)}, actual={committed}"
            )
        print(f"  restored={min(start + len(batch_ids), len(ids)):,}/{len(ids):,}")


def main():
    cfg = PipelineConfig.load(ROOT / "pipeline.yaml")
    person_cfg, _ = bvd.make_collection_configs(cfg)
    client = QdrantClient(url=cfg.qdrant.url, timeout=60)

    print("=" * 78)
    print("FORENSIC_PERSON PAYLOAD REPAIR")
    print("Repairs only points damaged by overwrite_payload.")
    print("NO embedding computation.")
    print("=" * 78)

    before = count_media(client)
    print("\nBEFORE")
    print(before)
    if before["total"] != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Unexpected forensic_person total: {before['total']:,} "
            f"(expected {EXPECTED_TOTAL:,}). Abort."
        )

    damaged_ids = find_damaged_ids(client)
    if len(damaged_ids) != EXPECTED_DAMAGED:
        raise RuntimeError(
            f"Damaged count changed: {len(damaged_ids):,} "
            f"(expected {EXPECTED_DAMAGED:,}). Abort before write."
        )

    store = QdrantStore(person_cfg)
    store.ensure_collection(recreate=False)
    recovered = {}

    reconstruct_video(store, damaged_ids, recovered)
    reconstruct_images(store, damaged_ids, recovered)

    not_found = damaged_ids - set(recovered)
    if not_found:
        raise RuntimeError(
            f"Could not reconstruct {len(not_found):,} damaged point IDs. Abort before write."
        )

    kinds = {"image": 0, "video": 0}
    for det in recovered.values():
        mt = detection_media_type(det)
        if mt not in kinds:
            raise RuntimeError(f"unexpected reconstructed media_type={mt!r}")
        kinds[mt] += 1

    print("\nRECONSTRUCTED")
    print(kinds)
    if (
        kinds["image"] != EXPECTED_DAMAGED_IMAGE
        or kinds["video"] != EXPECTED_DAMAGED_VIDEO
    ):
        raise RuntimeError(
            "Reconstructed damage split mismatch. "
            f"Observed={kinds}; expected image={EXPECTED_DAMAGED_IMAGE}, "
            f"video={EXPECTED_DAMAGED_VIDEO}. Abort before write."
        )

    vectors = retrieve_vectors(client, damaged_ids)
    restore(store, recovered, vectors)

    print("\n[6/6] Verify")
    after = count_media(client)
    print("AFTER")
    print(after)
    if (
        after["total"] != EXPECTED_TOTAL
        or after["image"] != EXPECTED_IMAGE
        or after["video"] != EXPECTED_VIDEO
        or after["missing"] != 0
    ):
        raise RuntimeError(
            "Repair finished but final counts are not exact. "
            f"Observed={after}"
        )

    print("\n" + "=" * 78)
    print("PAYLOAD REPAIR COMPLETE")
    print(f"image : {after['image']:,}")
    print(f"video : {after['video']:,}")
    print(f"total : {after['total']:,}")
    print("missing media_type : 0")
    print("embeddings recomputed : 0")
    print("=" * 78)


if __name__ == "__main__":
    main()
