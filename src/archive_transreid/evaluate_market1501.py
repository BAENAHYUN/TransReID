"""
Market-1501 metadata의 person_id(정답 라벨)를 이용해서, TransReID의
rank-1 정확도를 자동으로 계산.

방식: 저장된 1334개 중 일부를 Query로 뽑고, 나머지를 Gallery로 삼아서
      "Query와 같은 person_id를 가진 사진이 Top-1로 나오는 비율"을 계산.

표준 Market-1501 평가 규칙: 같은 카메라에서 찍힌 같은 사람은 "너무 쉬운 매치"라
                         제외하고 계산 (다른 카메라에서 찾아야 진짜 실력).

사용법: python src/evaluate_market1501.py
"""

import json
import numpy as np
import faiss

EMBEDDINGS_PATH = "data/embeddings_market1501.npy"
METADATA_PATH = "data/metadata_market1501.json"
NUM_QUERIES = 100  # 몇 개를 Query로 뽑아서 평가할지


def evaluate_rank1():
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # 앞쪽 NUM_QUERIES개를 Query로 사용 (이미 무작위로 섞여 저장된 데이터라 편향 없음)
    num_queries = min(NUM_QUERIES, len(metadata))

    correct = 0
    evaluated = 0

    for q_idx in range(num_queries):
        query_embedding = embeddings[q_idx:q_idx + 1]
        query_person_id = metadata[q_idx]["person_id"]
        query_camera_id = metadata[q_idx]["camera_id"]

        # 자기 자신 포함 넉넉하게 검색 (같은 카메라+같은 사람은 나중에 걸러낼 것)
        similarities, indices = index.search(query_embedding, 20)

        # ── 표준 프로토콜: 같은 카메라의 같은 사람(너무 쉬운 매치)은 건너뛰고
        #     "진짜 Top-1"을 찾음 ──
        top1_person_id = None
        for idx in indices[0]:
            if idx == q_idx:
                continue  # 자기 자신은 제외
            candidate = metadata[idx]
            if candidate["person_id"] == query_person_id and candidate["camera_id"] == query_camera_id:
                continue  # 같은 카메라의 같은 사람 = 너무 쉬운 매치, 건너뜀
            top1_person_id = candidate["person_id"]
            break

        if top1_person_id is None:
            continue  # 비교할 다른 카메라 후보가 아예 없으면 이번 Query는 평가 제외

        evaluated += 1
        if top1_person_id == query_person_id:
            correct += 1

    rank1_accuracy = correct / evaluated if evaluated > 0 else 0

    print(f"평가한 Query 수: {evaluated}개 (전체 {num_queries}개 중)")
    print(f"Rank-1 정확도: {correct}/{evaluated} = {rank1_accuracy * 100:.2f}%")
    print(f"\n참고: TransReID 논문 공식 수치(Market-1501, 전체 test set 기준) = 95.1%")
    print(f"지금은 일부(1334개)만 사용한 약식 평가라 논문 수치와 차이 날 수 있음")


if __name__ == "__main__":
    evaluate_rank1()