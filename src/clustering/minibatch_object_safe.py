from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm


COLLECTION = "forensic_object"

SCROLL_BATCH = 512
N_CLUSTERS = 300
KMEANS_BATCH = 512

UPDATE_BATCH = 256
QDRANT_TIMEOUT_SEC = 120
MAX_RETRIES = 5

VECTOR_NAMES = ["siglip2", "dinov2"]

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
ASSIGNMENT_FILE = OUTPUT_DIR / f"minibatch_object_k{N_CLUSTERS}_assignments.jsonl"


def media_filter(media_type: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="media_type",
                match=MatchValue(value=media_type),
            )
        ]
    )


def exact_counts(client: QdrantClient) -> dict:
    total = client.count(
        collection_name=COLLECTION,
        exact=True,
    ).count

    image = client.count(
        collection_name=COLLECTION,
        count_filter=media_filter("image"),
        exact=True,
    ).count

    video = client.count(
        collection_name=COLLECTION,
        count_filter=media_filter("video"),
        exact=True,
    ).count

    return {
        "total": int(total),
        "image": int(image),
        "video": int(video),
        "missing": int(total - image - video),
    }


def print_counts(title: str, counts: dict) -> None:
    print(f"\n----- {title} -----")
    print(f"total              : {counts['total']:,}")
    print(f"image              : {counts['image']:,}")
    print(f"video              : {counts['video']:,}")
    print(f"missing media_type : {counts['missing']:,}")


def iter_points_with_vectors(client: QdrantClient):
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=False,
            with_vectors=VECTOR_NAMES,
        )

        if not points:
            break

        yield points

        if next_offset is None:
            break

        offset = next_offset


def make_batch_vectors(points) -> np.ndarray:
    try:
        siglip = np.asarray(
            [p.vector["siglip2"] for p in points],
            dtype=np.float32,
        )
        dinov2 = np.asarray(
            [p.vector["dinov2"] for p in points],
            dtype=np.float32,
        )
    except KeyError as exc:
        raise RuntimeError(
            f"Required object vector missing: {exc}. "
            f"Expected named vectors={VECTOR_NAMES}"
        ) from exc

    siglip = normalize(siglip)
    dinov2 = normalize(dinov2)

    batch_vec = np.concatenate(
        [siglip, dinov2],
        axis=1,
    ).astype(np.float32, copy=False)

    if not np.isfinite(batch_vec).all():
        raise RuntimeError("NaN/Inf detected in object clustering vectors.")

    return batch_vec


def train_kmeans(client: QdrantClient, total: int) -> MiniBatchKMeans:
    print("\n----- 배치 단위 학습 중 -----")

    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=KMEANS_BATCH,
        random_state=42,
        n_init="auto",
        reassignment_ratio=0.01,
    )

    processed = 0
    first = True

    with tqdm(total=total, unit="point") as pbar:
        for points in iter_points_with_vectors(client):
            batch_vec = make_batch_vectors(points)

            if first and len(batch_vec) < N_CLUSTERS:
                raise RuntimeError(
                    f"First batch size ({len(batch_vec)}) must be >= "
                    f"N_CLUSTERS ({N_CLUSTERS})."
                )

            kmeans.partial_fit(batch_vec)

            first = False
            processed += len(points)
            pbar.update(len(points))

    if processed != total:
        raise RuntimeError(
            f"Training scan count mismatch: processed={processed:,}, total={total:,}"
        )

    print("----- 학습 완료 -----")
    return kmeans


def predict_all(
    client: QdrantClient,
    kmeans: MiniBatchKMeans,
    total: int,
):
    print("\n----- 라벨 예측 중 -----")

    all_ids: list[str | int] = []
    all_labels: list[int] = []

    processed = 0

    with tqdm(total=total, unit="point") as pbar:
        for points in iter_points_with_vectors(client):
            batch_vec = make_batch_vectors(points)
            labels = kmeans.predict(batch_vec)

            all_ids.extend(p.id for p in points)
            all_labels.extend(int(x) for x in labels)

            processed += len(points)
            pbar.update(len(points))

    if processed != total:
        raise RuntimeError(
            f"Prediction scan count mismatch: processed={processed:,}, total={total:,}"
        )

    if len(all_ids) != total or len(all_labels) != total:
        raise RuntimeError(
            f"Assignment size mismatch: ids={len(all_ids):,}, "
            f"labels={len(all_labels):,}, total={total:,}"
        )

    if len(set(map(str, all_ids))) != total:
        raise RuntimeError("Duplicate point IDs detected during prediction scan.")

    return all_ids, all_labels


def print_distribution(labels: list[int]) -> None:
    counts = Counter(labels)

    print(f"\n총 클러스터 수: {len(counts)}")
    print("포인트 수 상위 10개:")

    for cluster_id, count in counts.most_common(10):
        print(f"  cluster {cluster_id:4d}: {count:7,d}개")


def save_assignments_atomic(
    point_ids: list,
    labels: list[int],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tmp_path = ASSIGNMENT_FILE.with_suffix(
        ASSIGNMENT_FILE.suffix + ".tmp"
    )

    with tmp_path.open("w", encoding="utf-8") as f:
        for point_id, cluster_id in zip(point_ids, labels):
            row = {
                "point_id": point_id,
                "cluster_id": int(cluster_id),
            }
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

        f.flush()
        os.fsync(f.fileno())

    tmp_path.replace(ASSIGNMENT_FILE)

    print("\n----- assignment 저장 완료 -----")
    print(ASSIGNMENT_FILE)


def load_assignments(expected_total: int):
    if not ASSIGNMENT_FILE.is_file():
        raise FileNotFoundError(
            "assignment 파일이 없습니다:\n"
            f"{ASSIGNMENT_FILE}\n\n"
            "먼저 --apply-only 없이 전체 clustering을 실행하세요."
        )

    point_ids = []
    labels = []

    with ASSIGNMENT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            if "point_id" not in row or "cluster_id" not in row:
                raise RuntimeError(
                    f"Invalid assignment row at line {line_number}"
                )

            cluster_id = int(row["cluster_id"])

            if not (0 <= cluster_id < N_CLUSTERS):
                raise RuntimeError(
                    f"Invalid cluster_id={cluster_id} at line {line_number}"
                )

            point_ids.append(row["point_id"])
            labels.append(cluster_id)

    if len(point_ids) != expected_total:
        raise RuntimeError(
            f"assignment count mismatch: "
            f"{len(point_ids):,} != DB total {expected_total:,}"
        )

    if len(set(map(str, point_ids))) != expected_total:
        raise RuntimeError(
            "Duplicate point IDs detected in assignment file."
        )

    print(f"assignment loaded: {len(point_ids):,}")
    return point_ids, labels


def set_payload_with_retry(
    client: QdrantClient,
    cluster_id: int,
    point_ids: list,
) -> None:
    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client.set_payload(
                collection_name=COLLECTION,
                payload={"cluster_id": int(cluster_id)},
                points=point_ids,
                wait=True,
            )
            return
        except Exception as exc:
            last_exc = exc

            if attempt >= MAX_RETRIES:
                break

            sleep_sec = min(2 ** attempt, 30)

            print(
                f"\n[retry {attempt}/{MAX_RETRIES}] "
                f"cluster={cluster_id}, points={len(point_ids)} "
                f"error={type(exc).__name__}: {exc}"
            )
            print(f"sleep {sleep_sec}s...")
            time.sleep(sleep_sec)

    raise RuntimeError(
        f"Qdrant set_payload failed after {MAX_RETRIES} retries. "
        f"cluster={cluster_id}, points={len(point_ids)}"
    ) from last_exc


def apply_assignments(
    client: QdrantClient,
    point_ids: list,
    labels: list[int],
) -> None:
    print("\n----- Qdrant payload 업데이트 중 -----")
    print("주의: overwrite_payload 사용 안 함")
    print("      set_payload로 cluster_id만 추가/수정")

    cluster_to_ids: dict[int, list] = defaultdict(list)

    for point_id, label in zip(point_ids, labels):
        cluster_to_ids[int(label)].append(point_id)

    total_chunks = sum(
        math.ceil(len(ids) / UPDATE_BATCH)
        for ids in cluster_to_ids.values()
    )

    with tqdm(
        total=total_chunks,
        unit="batch",
    ) as pbar:
        for cluster_id in sorted(cluster_to_ids):
            ids = cluster_to_ids[cluster_id]

            for start in range(0, len(ids), UPDATE_BATCH):
                chunk = ids[start : start + UPDATE_BATCH]

                set_payload_with_retry(
                    client,
                    cluster_id,
                    chunk,
                )

                pbar.update(1)


def verify_after(
    client: QdrantClient,
    before: dict,
) -> None:
    after = exact_counts(client)
    print_counts("DB 사후 무결성 확인", after)

    if after != before:
        raise RuntimeError(
            "DB payload integrity changed.\n"
            f"before={before}\n"
            f"after ={after}"
        )

    print("\nDB payload 무결성 유지 확인: PASS")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Safe MiniBatchKMeans clustering for forensic_object. "
            "Uses SigLIP2 + DINOv2 and preserves all existing Qdrant payloads."
        )
    )

    ap.add_argument(
        "--apply-only",
        action="store_true",
        help=(
            "기존 assignment JSONL을 읽어 cluster_id payload만 다시 적용. "
            "학습/예측은 수행하지 않음."
        ),
    )

    args = ap.parse_args()

    client = QdrantClient(
        "localhost",
        port=6333,
        timeout=QDRANT_TIMEOUT_SEC,
    )

    before = exact_counts(client)
    print_counts("DB 사전 무결성 확인", before)

    if before["missing"] != 0:
        raise RuntimeError(
            "media_type이 없는 point가 존재합니다. "
            "clustering 전에 DB를 먼저 복구/점검하세요."
        )

    total = before["total"]

    if total < N_CLUSTERS:
        raise RuntimeError(
            f"Total points ({total}) < N_CLUSTERS ({N_CLUSTERS})"
        )

    print(f"\n<전체 포인트 수: {total:,}>")
    print("collection :", COLLECTION)
    print("vectors    :", " + ".join(VECTOR_NAMES))
    print("K          :", N_CLUSTERS)

    if args.apply_only:
        print("\n----- 기존 assignment 로드 -----")
        point_ids, labels = load_assignments(total)
        print_distribution(labels)
    else:
        kmeans = train_kmeans(
            client,
            total,
        )

        point_ids, labels = predict_all(
            client,
            kmeans,
            total,
        )

        print_distribution(labels)

        # Qdrant에 쓰기 전에 assignment부터 안전하게 저장
        save_assignments_atomic(
            point_ids,
            labels,
        )

    apply_assignments(
        client,
        point_ids,
        labels,
    )

    verify_after(
        client,
        before,
    )

    print("\n" + "=" * 78)
    print("MINIBATCH OBJECT CLUSTERING COMPLETE")
    print("=" * 78)
    print(f"points      : {total:,}")
    print(f"clusters    : {N_CLUSTERS}")
    print("vectors     : SigLIP2 + DINOv2")
    print("payload     : preserved")
    print("cluster_id  : added with set_payload")
    print(f"assignment  : {ASSIGNMENT_FILE}")
    print("=" * 78)


if __name__ == "__main__":
    main()
