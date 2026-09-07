from qdrant_client import QdrantClient
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
import numpy as np
from tqdm import tqdm
from collections import defaultdict

COLLECTION = "forensic_person"
SCROLL_BATCH = 512        # Qdrant에서 한 번에 가져올 크롭 수 N_CLUSTERS
N_CLUSTERS = 300
KMEANS_BATCH = 512        # MiniBatch 학습 단위 줄임
MAX_ITER = 100
UPDATE_BATCH = 32         # payload 업데이트 단위 줄임

client = QdrantClient("localhost", port=6333)

info = client.get_collection(COLLECTION)
total = info.points_count
print(f"\n<전체 포인트 수: {total}>")

# ----- 1) MiniBatchKMeans 객체 생성 -----
kmeans = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    batch_size=KMEANS_BATCH,
    max_iter=MAX_ITER,
    random_state=42,
    verbose=0
)

# ----- 2) 배치 단위로 학습 -----
# 전체 벡터 메모리에 다 안 올리고, 배치 단위로 partial_fit 호출 후 순차 학습
print("----- 배치 단위 학습 중 -----")
offset = None
processed = 0

with tqdm(total=total) as pbar:
    while True:
        result, offset = client.scroll(
            collection_name=COLLECTION,
            limit=SCROLL_BATCH,
            offset=offset,
            with_vectors=["siglip2", "irra", "solider"],
            with_payload=False
        )
        if not result:
            break

        s = normalize(np.array([p.vector["siglip2"] for p in result], dtype=np.float32))
        i = normalize(np.array([p.vector["irra"]    for p in result], dtype=np.float32))
        o = normalize(np.array([p.vector["solider"] for p in result], dtype=np.float32))
        batch_vec = np.concatenate([s, i, o], axis=1)

        kmeans.partial_fit(batch_vec)
        processed += len(result)
        pbar.update(len(result))

        # 배치 처리 후 메모리 해제
        del s, i, o, batch_vec

        if offset is None:
            break

print("----- 학습 완료. 라벨 예측 중 -----")

# ----- 3) 전체 데이터에 라벨 배정 -----
# 학습 후 각 포인트에 cluster_id 배정
all_ids = []
all_labels = []
offset = None

with tqdm(total=total) as pbar:
    while True:
        result, offset = client.scroll(
            collection_name=COLLECTION,
            limit=SCROLL_BATCH,
            offset=offset,
            with_vectors=["siglip2", "irra", "solider"],
            with_payload=False
        )
        if not result:
            break

        s = normalize(np.array([p.vector["siglip2"] for p in result], dtype=np.float32))
        i = normalize(np.array([p.vector["irra"]    for p in result], dtype=np.float32))
        o = normalize(np.array([p.vector["solider"] for p in result], dtype=np.float32))
        batch_vec = np.concatenate([s, i, o], axis=1)

        labels = kmeans.predict(batch_vec)
        all_ids.extend([p.id for p in result])
        all_labels.extend(labels.tolist())

        del s, i, o, batch_vec, labels
        pbar.update(len(result))

        if offset is None:
            break

# ----- 4) 클러스터 분포 확인 -----
all_labels_arr = np.array(all_labels)
unique, counts = np.unique(all_labels_arr, return_counts=True)
print(f"\n총 클러스터 수: {len(unique)}")
print("포인트 수 상위 10개:")
for cid, cnt in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
    print(f"  cluster {cid:>4d}: {cnt:>6d}개")

# ----- 5) Qdrant payload 업데이트 -----
# cluster_id별로 포인트 묶어서 한 번에 요청 : 76,326 -> 너무 느림
print("\n----- Qdrant payload 업데이트 중 -----")
cluster_to_ids = defaultdict(list)
for pid, cid in zip(all_ids, all_labels):
    cluster_to_ids[int(cid)].append(pid)
# cluster_to_ids = {42: [id1, id3, ...], 7: [id2, ...], ...}
# 같은 cluster_id의 포인트 id들 리스트로 묶기

for cluster_id, point_ids in tqdm(cluster_to_ids.items()):
    client.overwrite_payload(
        collection_name=COLLECTION,
        payload={"cluster_id": cluster_id},
        points=point_ids
    )
# overwrite_payload는 여러 포인트에 동시에 같은 payload를 씀
# cluster_id당 1번 요청 → 총 300번만 Qdrant 서버에 요청

print("\n ----- 클러스터링 완료 -----")