"""
CLIP-ReID의 Market-1501 rank-1 정확도 자동 평가.
TransReID 때(evaluate_market1501.py)와 완전히 동일한 방법론:
- 같은 카메라의 같은 사람(너무 쉬운 매치)은 제외
- Query 100개로 평가

사용법: python src/RF-DETR/evaluate_market1501_clip.py
"""

import json
import numpy as np
import faiss

EMBEDDINGS_PATH = "data/embeddings_market1501_clip.npy"
METADATA_PATH = "data/metadata_market1501_clip.json"
NUM_QUERIES = 100  # TransReID 평가 때와 동일


def evaluate_rank1():
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # CLIP-ReID는 1280차원
    index.add(embeddings)

    num_queries = min(NUM_QUERIES, len(metadata))

    correct = 0
    evaluated = 0

    for q_idx in range(num_queries):
        query_embedding = embeddings[q_idx:q_idx + 1]
        query_person_id = metadata[q_idx]["person_id"]
        query_camera_id = metadata[q_idx]["camera_id"]

        similarities, indices = index.search(query_embedding, 20)

        top1_person_id = None
        for idx in indices[0]:
            if idx == q_idx:
                continue
            candidate = metadata[idx]
            if candidate["person_id"] == query_person_id and candidate["camera_id"] == query_camera_id:
                continue  # 같은 카메라의 같은 사람은 너무 쉬운 매치, 건너뜀
            top1_person_id = candidate["person_id"]
            break

        if top1_person_id is None:
            continue

        evaluated += 1
        if top1_person_id == query_person_id:
            correct += 1

    rank1_accuracy = correct / evaluated if evaluated > 0 else 0

    print(f"[CLIP-ReID] 평가한 Query 수: {evaluated}개 (전체 {num_queries}개 중)")
    print(f"[CLIP-ReID] Rank-1 정확도: {correct}/{evaluated} = {rank1_accuracy * 100:.2f}%")
    print(f"\n=== 비교 ===")
    print(f"TransReID(같은 방법론, 같은 규모)  : 70.00% (이전 실측)")
    print(f"CLIP-ReID(SIE-OLP, 지금 결과)      : {rank1_accuracy * 100:.2f}%")


if __name__ == "__main__":
    evaluate_rank1()