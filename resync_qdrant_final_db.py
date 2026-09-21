#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qdrant re-sync for final_db_candidates (after diverse_topk_v2 upgrade).

Step 1: Delete ALL points where source == 'final_db_candidates' from both
        forensic_person and forensic_object collections.
Step 2: (Manual) Run ingest_final_candidates_qdrant.py for all 150 stems.

Usage:
    python resync_qdrant_final_db.py --dry-run     # preview counts only
    python resync_qdrant_final_db.py               # actually delete
"""
from __future__ import annotations

import argparse
import sys
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

QDRANT_URL = "http://localhost:6333"
COLLECTIONS = ["forensic_person", "forensic_object"]
SOURCE_VALUE = "final_db_candidates"
SCROLL_LIMIT = 1000


def delete_source(client: QdrantClient, collection: str, dry_run: bool) -> int:
    """Delete all points with source == SOURCE_VALUE. Returns count deleted."""
    filt = Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value=SOURCE_VALUE),
            )
        ]
    )

    # Count first
    count_result = client.count(collection_name=collection, count_filter=filt, exact=True)
    total = count_result.count
    print(f"[{collection}] source='{SOURCE_VALUE}' points: {total}")

    if total == 0 or dry_run:
        return total

    # Delete in one call — Qdrant supports filter-based delete
    client.delete(
        collection_name=collection,
        points_selector=filt,
    )
    print(f"[{collection}] DELETED {total} points")
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Count points only, do not delete")
    p.add_argument("--url", default=QDRANT_URL)
    args = p.parse_args()

    client = QdrantClient(url=args.url, timeout=120)

    total_deleted = 0
    for col in COLLECTIONS:
        n = delete_source(client, col, args.dry_run)
        total_deleted += n

    if args.dry_run:
        print(f"\n[DRY-RUN] Would delete {total_deleted} points total. "
              "Re-run without --dry-run to apply.")
    else:
        print(f"\n[DONE] Deleted {total_deleted} points from source='{SOURCE_VALUE}'.")
        print("Next step: re-ingest all 150 stems with ingest_final_candidates_qdrant.py")


if __name__ == "__main__":
    main()
