from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm

COLLECTION = "forensic_person"

SCROLL_BATCH = 512
N_CLUSTERS = 300
KMEANS_BATCH = 512
MAX_ITER = 100

UPDATE_BATCH = 256
QDRANT_TIMEOUT_SEC = 120
MAX_RETRIES = 5

VECTOR_NAMES = ["siglip2", "irra", "solider"]

DEFAULT_ASSIGNMENT_FILE = (
    Path(__file__).resolve().parent
    / "output"
    / f"minibatch_person_k{N_CLUSTERS}_assignments.jsonl"
)


def make_client():
    return QdrantClient(
        "localhost",
        port=6333,
        timeout=QDRANT_TIMEOUT_SEC,
    )


def media_counts(client):
    info = client.get_collection(COLLECTION)
    total = int(info.points_count or 0)

    counts = {}
    for media_type in ("image", "video"):
        f = Filter(
            must=[
                FieldCondition(
                    key="media_type",
                    match=MatchValue(value=media_type),
                )
            ]
        )
        counts[media_type] = client.count(
            collection_name=COLLECTION,
            count_filter=f,
            exact=True,
        ).count

    counts["total"] = total
    counts["missing_media_type"] = total - counts["image"] - counts["video"]
    return counts


def verify_db_before(client):
    counts = media_counts(client)

    print("\n----- DB 사전 무결성 확인 -----")
    print(f"total : {counts['total']:,}")
    print(f"image : {counts['image']:,}")
    print(f"video : {counts['video']:,}")
    print(f"missing media_type : {counts['missing_media_type']:,}")

    if counts["missing_media_type"] != 0:
        raise RuntimeError(
            "media_type이 없는 point가 있습니다. 클러스터링을 시작하지 않습니다."
        )

    if counts["total"] <= 0:
        raise RuntimeError("forensic_person collection이 비어 있습니다.")

    return counts


def batch_vectors(result):
    try:
        s = normalize(
            np.asarray(
                [p.vector["siglip2"] for p in result],
                dtype=np.float32,
            )
        )
        i = normalize(
            np.asarray(
                [p.vector["irra"] for p in result],
                dtype=np.float32,
            )
        )
        o = normalize(
            np.asarray(
                [p.vector["solider"] for p in result],
                dtype=np.float32,
            )
        )
    except KeyError as exc:
        raise RuntimeError(
            f"필수 named vector가 없는 point가 있습니다: {exc}"
        ) from exc

    batch_vec = np.concatenate([s, i, o], axis=1)

    if not np.isfinite(batch_vec).all():
        raise RuntimeError("NaN/Inf가 포함된 vector를 발견했습니다.")

    return batch_vec


def train_kmeans(client, total):
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=KMEANS_BATCH,
        max_iter=MAX_ITER,
        random_state=42,
        verbose=0,
    )

    print("\n----- 배치 단위 학습 중 -----")
    offset = None
    processed = 0

    with tqdm(total=total) as pbar:
        while True:
            result, offset = client.scroll(
                collection_name=COLLECTION,
                limit=SCROLL_BATCH,
                offset=offset,
                with_vectors=VECTOR_NAMES,
                with_payload=False,
            )

            if not result:
                break

            batch_vec = batch_vectors(result)

            if processed == 0 and len(result) < N_CLUSTERS:
                raise RuntimeError(
                    f"첫 배치 크기({len(result)})가 n_clusters({N_CLUSTERS})보다 작습니다."
                )

            kmeans.partial_fit(batch_vec)

            processed += len(result)
            pbar.update(len(result))
            del batch_vec

            if offset is None:
                break

    if processed != total:
        raise RuntimeError(
            f"학습 scroll point 수 불일치: processed={processed:,}, total={total:,}"
        )

    return kmeans


def predict_labels(client, kmeans, total):
    print("----- 학습 완료. 라벨 예측 중 -----")

    all_ids = []
    all_labels = []
    offset = None

    with tqdm(total=total) as pbar:
        while True:
            result, offset = client.scroll(
                collection_name=COLLECTION,
                limit=SCROLL_BATCH,
                offset=offset,
                with_vectors=VECTOR_NAMES,
                with_payload=False,
            )

            if not result:
                break

            batch_vec = batch_vectors(result)
            labels = kmeans.predict(batch_vec)

            all_ids.extend([str(p.id) for p in result])
            all_labels.extend(int(x) for x in labels.tolist())

            pbar.update(len(result))
            del batch_vec, labels

            if offset is None:
                break

    if len(all_ids) != total or len(all_labels) != total:
        raise RuntimeError(
            "예측 point 수 불일치: "
            f"ids={len(all_ids):,}, labels={len(all_labels):,}, total={total:,}"
        )

    if len(set(all_ids)) != total:
        raise RuntimeError("중복 Qdrant point ID를 발견했습니다.")

    return all_ids, all_labels


def print_cluster_distribution(all_labels):
    labels_arr = np.asarray(all_labels, dtype=np.int32)
    unique, counts = np.unique(labels_arr, return_counts=True)

    print(f"\n총 클러스터 수: {len(unique)}")
    print("포인트 수 상위 10개:")

    for cid, cnt in sorted(
        zip(unique, counts),
        key=lambda x: -x[1],
    )[:10]:
        print(f"  cluster {int(cid):>4d}: {int(cnt):>6d}개")

    if len(unique) != N_CLUSTERS:
        print(
            f"[WARNING] 설정 K={N_CLUSTERS}, 실제 사용된 cluster={len(unique)}"
        )


def save_assignments(path, all_ids, all_labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        for pid, cid in zip(all_ids, all_labels):
            f.write(
                json.dumps(
                    {
                        "point_id": str(pid),
                        "cluster_id": int(cid),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    tmp.replace(path)

    print(f"\nassignment 저장: {path}")
    print(f"assignment 수  : {len(all_ids):,}")


def load_assignments(path, expected_total):
    if not path.is_file():
        raise FileNotFoundError(f"assignment 파일이 없습니다: {path}")

    all_ids = []
    all_labels = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
                pid = str(row["point_id"])
                cid = int(row["cluster_id"])
            except Exception as exc:
                raise RuntimeError(
                    f"잘못된 assignment JSONL: {path} line={line_no}"
                ) from exc

            if cid < 0 or cid >= N_CLUSTERS:
                raise RuntimeError(f"cluster_id 범위 오류: {cid}")

            all_ids.append(pid)
            all_labels.append(cid)

    if len(all_ids) != expected_total:
        raise RuntimeError(
            "assignment 수가 현재 DB total과 다릅니다: "
            f"assignment={len(all_ids):,}, DB={expected_total:,}"
        )

    if len(set(all_ids)) != expected_total:
        raise RuntimeError("assignment 파일에 중복 point ID가 있습니다.")

    return all_ids, all_labels


def set_payload_with_retry(client, cluster_id, point_ids):
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

            wait_sec = min(2 ** (attempt - 1), 8)

            print(
                f"\n[retry] cluster={cluster_id} "
                f"batch={len(point_ids)} "
                f"attempt={attempt}/{MAX_RETRIES} "
                f"error={type(exc).__name__}: {exc}"
            )
            print(f"        {wait_sec}초 후 재시도")
            time.sleep(wait_sec)

    raise RuntimeError(
        f"Qdrant set_payload 최종 실패: "
        f"cluster={cluster_id}, points={len(point_ids)}"
    ) from last_exc


def update_qdrant_payload(client, all_ids, all_labels):
    print("\n----- Qdrant payload 업데이트 중 -----")
    print("mode         : set_payload (기존 payload 보존)")
    print(f"update batch : {UPDATE_BATCH}")
    print(f"timeout      : {QDRANT_TIMEOUT_SEC}s")

    cluster_to_ids = defaultdict(list)

    for pid, cid in zip(all_ids, all_labels):
        cluster_to_ids[int(cid)].append(pid)

    total_batches = sum(
        (len(point_ids) + UPDATE_BATCH - 1) // UPDATE_BATCH
        for point_ids in cluster_to_ids.values()
    )

    with tqdm(total=total_batches, desc="payload batches") as pbar:
        for cluster_id in sorted(cluster_to_ids):
            point_ids = cluster_to_ids[cluster_id]

            for start in range(0, len(point_ids), UPDATE_BATCH):
                batch_ids = point_ids[start:start + UPDATE_BATCH]

                set_payload_with_retry(
                    client,
                    cluster_id,
                    batch_ids,
                )

                pbar.update(1)


def verify_db_after(client, before_counts):
    print("\n----- DB 사후 무결성 확인 -----")

    after = media_counts(client)

    print(f"total : {after['total']:,}")
    print(f"image : {after['image']:,}")
    print(f"video : {after['video']:,}")
    print(f"missing media_type : {after['missing_media_type']:,}")

    for key in ("total", "image", "video", "missing_media_type"):
        if after[key] != before_counts[key]:
            raise RuntimeError(
                "payload 업데이트 후 DB 무결성 값이 변경되었습니다: "
                f"{key}: before={before_counts[key]}, after={after[key]}"
            )

    print("DB payload 무결성 유지 확인: PASS")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "forensic_person MiniBatch K-Means clustering "
            "with safe Qdrant payload update"
        )
    )
    ap.add_argument(
        "--apply-only",
        action="store_true",
        help=(
            "저장된 assignment JSONL을 사용해 payload만 적용. "
            "학습/예측을 다시 하지 않음."
        ),
    )
    ap.add_argument(
        "--assignment-file",
        type=Path,
        default=DEFAULT_ASSIGNMENT_FILE,
    )
    args = ap.parse_args()

    client = make_client()
    before_counts = verify_db_before(client)
    total = before_counts["total"]

    print(f"\n<전체 포인트 수: {total:,}>")
    print(f"collection : {COLLECTION}")
    print("vectors    : " + " + ".join(VECTOR_NAMES))
    print(f"K          : {N_CLUSTERS}")

    if args.apply_only:
        print("\n----- 기존 assignment 로드 -----")
        all_ids, all_labels = load_assignments(
            args.assignment_file,
            total,
        )
        print_cluster_distribution(all_labels)

    else:
        kmeans = train_kmeans(client, total)

        all_ids, all_labels = predict_labels(
            client,
            kmeans,
            total,
        )

        print_cluster_distribution(all_labels)

        # payload 단계가 실패해도 다음번에는 --apply-only로
        # 학습/예측을 반복하지 않도록 먼저 저장한다.
        save_assignments(
            args.assignment_file,
            all_ids,
            all_labels,
        )

    update_qdrant_payload(
        client,
        all_ids,
        all_labels,
    )

    verify_db_after(
        client,
        before_counts,
    )

    print("\n" + "=" * 76)
    print("MINIBATCH PERSON CLUSTERING COMPLETE")
    print(f"points      : {total:,}")
    print(f"clusters    : {N_CLUSTERS}")
    print("payload     : preserved")
    print("cluster_id  : added with set_payload")
    print("=" * 76)


if __name__ == "__main__":
    main()
