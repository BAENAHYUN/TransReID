from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from config import PipelineConfig
from qdrant_store import QdrantStore
from rfdetr_adapter import from_rfdetr


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "pipeline.yaml"
DEFAULT_STATS = ROOT / "data" / "crops" / "filter_stats.json"
DEFAULT_CSV = ROOT / "extra_sources.csv"
DEFAULT_JSON = ROOT / "extra_sources.json"


def infer_source(payload: Dict[str, Any]) -> str:
    """
    extra point의 출처를 payload에서 최대한 안정적으로 분류한다.

    우선순위:
      1) detection_id에 '|'가 있으면 첫 토큰
         예: UCF-Crime|... -> UCF-Crime
             UCF-Crime-Normal-obj|... -> UCF-Crime-Normal-obj

      2) detection_id가 경로 형태면 앞 2개 디렉터리
         예: data/query_crops/foo.jpg -> data/query_crops

      3) image_id가 경로 형태면 첫 디렉터리
         예: Normal_Videos_781_x264/frame_... -> Normal_Videos_781_x264

      4) 알 수 없으면 unknown
    """
    detection_id = payload.get("detection_id")
    if detection_id:
        text = str(detection_id).strip()

        if "|" in text:
            return text.split("|", 1)[0].strip() or "unknown"

        norm = text.replace("\\", "/").strip("/")
        parts = [p for p in norm.split("/") if p]
        if len(parts) >= 2:
            return "/".join(parts[:2])
        if len(parts) == 1:
            return parts[0]

    image_id = payload.get("image_id")
    if image_id:
        norm = str(image_id).replace("\\", "/").strip("/")
        parts = [p for p in norm.split("/") if p]
        if parts:
            return parts[0]

    return "unknown"


def build_expected_ids(
    store: QdrantStore,
    crops: List[Dict[str, Any]],
    batch_size: int,
) -> Set[str]:
    expected: Set[str] = set()
    total = len(crops)

    _, expected_fmt = from_rfdetr(crops[:1], load_mode="path")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        detections, batch_fmt = from_rfdetr(
            crops[start:end],
            load_mode="path",
        )

        if batch_fmt != expected_fmt:
            raise RuntimeError(
                f"input_format 변경 감지: {expected_fmt} -> {batch_fmt}"
            )

        if len(detections) != end - start:
            raise RuntimeError(
                f"from_rfdetr 변환 개수 불일치: "
                f"{end-start} -> {len(detections)}"
            )

        for det in detections:
            point_id, _ = store._stable_point_id(det)
            expected.add(str(point_id))

        if end == total or end % (batch_size * 20) == 0:
            print(
                f"[EXPECTED IDS] {end:,}/{total:,} "
                f"unique={len(expected):,}"
            )

    return expected


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "현재 filter_stats.json에 속하지 않는 Qdrant extra point들의 "
            "출처를 read-only로 집계합니다."
        )
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--stats", default=str(DEFAULT_STATS))
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--page-size", type=int, default=1000)
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    ap.add_argument(
        "--samples-per-source",
        type=int,
        default=3,
        help="출처별 샘플 point 개수",
    )
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    stats_path = Path(args.stats).resolve()
    csv_path = Path(args.csv).resolve()
    json_path = Path(args.json).resolve()

    print("=" * 72)
    print("EXTRA POINT SOURCE AUDIT (READ ONLY)")
    print("=" * 72)
    print(f"config : {config_path}")
    print(f"stats  : {stats_path}")

    cfg = PipelineConfig.load(config_path)
    store = QdrantStore(cfg)

    if not store.client.collection_exists(store.collection):
        raise RuntimeError(f"collection 없음: {store.collection}")

    with stats_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    crops = data["crops"]

    print(f"[INFO] source crops={len(crops):,}")
    print(f"[INFO] collection={store.collection}")

    print("\n=== BUILD CURRENT EXPECTED IDS ===")
    expected_ids = build_expected_ids(
        store,
        crops,
        batch_size=args.batch_size,
    )

    source_counts: Counter[str] = Counter()
    source_person_counts: Counter[str] = Counter()
    source_object_counts: Counter[str] = Counter()
    source_label_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    scanned = 0
    extra_total = 0
    current_found = 0
    offset = None

    print("\n=== SCAN QDRANT EXTRA POINTS ===")

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
            pid = str(point.id)

            if pid in expected_ids:
                current_found += 1
                continue

            extra_total += 1
            payload = getattr(point, "payload", None) or {}
            source = infer_source(payload)
            label = str(payload.get("label") or "unknown")
            is_person = payload.get("is_person")

            source_counts[source] += 1
            source_label_counts[source][label] += 1

            if is_person is True:
                source_person_counts[source] += 1
            elif is_person is False:
                source_object_counts[source] += 1

            if len(samples[source]) < args.samples_per_source:
                samples[source].append(
                    {
                        "point_id": pid,
                        "image_id": payload.get("image_id"),
                        "detection_id": payload.get("detection_id"),
                        "frame_idx": payload.get("frame_idx"),
                        "label": payload.get("label"),
                        "is_person": payload.get("is_person"),
                        "track_id": payload.get("track_id"),
                        "bbox": payload.get("bbox"),
                    }
                )

        if scanned % (args.page_size * 20) == 0:
            print(
                f"[SCAN] {scanned:,} | "
                f"current={current_found:,} | extra={extra_total:,}"
            )

        if next_offset is None:
            break
        offset = next_offset

    rows = []
    for source, count in source_counts.most_common():
        labels = source_label_counts[source]
        top_labels = ", ".join(
            f"{label}:{n}"
            for label, n in labels.most_common(10)
        )

        rows.append(
            {
                "source": source,
                "count": count,
                "person": source_person_counts[source],
                "object": source_object_counts[source],
                "top_labels": top_labels,
            }
        )

    print("\n" + "=" * 72)
    print("EXTRA SOURCE SUMMARY")
    print("=" * 72)
    print(f"Qdrant total scanned : {scanned:,}")
    print(f"현재 데이터 발견      : {current_found:,}")
    print(f"extra point 총계      : {extra_total:,}")
    print(f"출처 종류             : {len(rows):,}")

    for row in rows:
        pct = (row["count"] / extra_total * 100.0) if extra_total else 0.0
        print(
            f"\n[{row['source']}]"
            f"\n  count : {row['count']:,} ({pct:.2f}%)"
            f"\n  person: {row['person']:,}"
            f"\n  object: {row['object']:,}"
            f"\n  labels: {row['top_labels']}"
        )

        for i, sample in enumerate(samples[row["source"]], 1):
            print(
                f"  sample{i}: "
                f"detection_id={sample.get('detection_id')!r}, "
                f"image_id={sample.get('image_id')!r}"
            )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "count",
                "person",
                "object",
                "top_labels",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "collection": store.collection,
        "source_crops": len(crops),
        "qdrant_scanned": scanned,
        "current_found": current_found,
        "extra_total": extra_total,
        "sources": [
            {
                **row,
                "samples": samples[row["source"]],
            }
            for row in rows
        ],
    }

    json_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== OUTPUT ===")
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")

    if extra_total != sum(source_counts.values()):
        print("[FAIL] source count 합계가 extra_total과 다릅니다.")
        return 1

    print("[OK] extra point 전체가 출처 그룹으로 집계되었습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
