from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import PipelineConfig
from qdrant_store import QdrantStore

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "pipeline.yaml"
DEFAULT_BACKUP = ROOT / "query_crops_point_ids_backup.json"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Qdrant에서 detection_id가 특정 prefix로 시작하는 point만 "
            "선택 삭제합니다. 기본은 DRY-RUN이며 --apply 때만 삭제합니다."
        )
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--prefix", default="data/query_crops/")
    ap.add_argument("--expected", type=int, default=191490)
    ap.add_argument("--page-size", type=int, default=2000)
    ap.add_argument("--delete-batch-size", type=int, default=1000)
    ap.add_argument("--backup", default=str(DEFAULT_BACKUP))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    cfg = PipelineConfig.load(Path(args.config).resolve())
    store = QdrantStore(cfg)

    if not store.client.collection_exists(store.collection):
        raise RuntimeError(f"collection 없음: {store.collection}")

    ids = []
    samples = []
    scanned = 0
    offset = None

    print("=" * 72)
    print("QUERY_CROPS POINT DELETE AUDIT")
    print("=" * 72)
    print(f"collection : {store.collection}")
    print(f"prefix     : {args.prefix!r}")
    print(f"expected   : {args.expected:,}")
    print(f"mode       : {'APPLY DELETE' if args.apply else 'DRY-RUN'}")

    while True:
        points, next_offset = store.client.scroll(
            collection_name=store.collection,
            limit=args.page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            scanned += 1
            payload = getattr(point, "payload", None) or {}
            detection_id = str(payload.get("detection_id") or "")

            if detection_id.startswith(args.prefix):
                ids.append(str(point.id))
                if len(samples) < 10:
                    samples.append(
                        {
                            "point_id": str(point.id),
                            "detection_id": detection_id,
                            "label": payload.get("label"),
                            "is_person": payload.get("is_person"),
                        }
                    )

        if scanned % 100000 == 0:
            print(f"[SCAN] {scanned:,} | matched={len(ids):,}")

        if next_offset is None:
            break
        offset = next_offset

    print("\n" + "=" * 72)
    print("MATCH RESULT")
    print("=" * 72)
    print(f"Qdrant scanned : {scanned:,}")
    print(f"matched        : {len(ids):,}")

    for i, row in enumerate(samples, 1):
        print(
            f"[sample {i}] {row['point_id']} | "
            f"{row['detection_id']} | {row['label']} | "
            f"person={row['is_person']}"
        )

    if len(ids) != args.expected:
        print(
            f"\n[ABORT] matched={len(ids):,} != expected={args.expected:,}\n"
            "안전상 삭제하지 않습니다."
        )
        return 2

    backup_path = Path(args.backup).resolve()
    backup_path.write_text(
        json.dumps(
            {
                "collection": store.collection,
                "prefix": args.prefix,
                "count": len(ids),
                "point_ids": ids,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[BACKUP] point ID 목록 저장: {backup_path}")

    if not args.apply:
        print(
            "\n[DRY-RUN PASS] 예상 개수와 정확히 일치했습니다.\n"
            "실제 삭제하려면 같은 명령에 --apply 를 추가하세요."
        )
        return 0

    from qdrant_client import models

    deleted = 0
    for start in range(0, len(ids), args.delete_batch_size):
        batch = ids[start:start + args.delete_batch_size]
        store.client.delete(
            collection_name=store.collection,
            points_selector=models.PointIdsList(points=batch),
            wait=True,
        )
        deleted += len(batch)

        if deleted % 10000 == 0 or deleted == len(ids):
            print(f"[DELETE] {deleted:,}/{len(ids):,}")

    remaining = store.client.count(
        collection_name=store.collection,
        exact=True,
    ).count

    print("\n" + "=" * 72)
    print("DELETE COMPLETED")
    print("=" * 72)
    print(f"deleted   : {deleted:,}")
    print(f"remaining : {remaining:,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
