"""
마일스톤 5: Qdrant에서 유사한 사람 검색
사용법: python src/search_qdrant.py

- load_qdrant_client(): Qdrant 서버에 연결
- search(client, query_embedding, top_k): 가장 비슷한 Top-K 찾기
"""

import sys

sys.path.append("src/RF-DETR")
from reid_clip import load_reid_model, get_embedding

from qdrant_client import QdrantClient

COLLECTION_NAME = "person_embeddings"


def load_qdrant_client():
    """
    로컬에 떠있는 Qdrant 서버에 연결.
    """
    client = QdrantClient(host="localhost", port=6333)
    return client


def search(client, query_embedding, top_k=5):
    """
    query_embedding과 가장 비슷한 Top-K를 Qdrant에서 찾아서 반환.

    반환값: [{"rank": ..., "crop_path": ..., "original_photo": ..., "similarity": ...}, ...]
    """
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=top_k,
    )

    results = []
    for rank, point in enumerate(hits.points, start=1):
        results.append({
            "rank": rank,
            "crop_path": point.payload["crop_path"],
            "original_photo": point.payload["original_photo"],
            "similarity": round(point.score, 4),  # Distance.COSINE 설정이라 score=코사인 유사도
        })

    return results


# ── 이 파일을 단독 실행했을 때는 테스트용으로 동작 ──
if __name__ == "__main__":
    # DB 안에 있는 crop 사진 하나를 Query로 사용 (본인 환경에 맞게 수정 가능)
    TEST_QUERY_IMAGE = "data/crops/George_W_Bush_0001_person_1.jpg"

    reid_model, device = load_reid_model()
    query_embedding = get_embedding(reid_model, device, TEST_QUERY_IMAGE)

    client = load_qdrant_client()
    results = search(client, query_embedding, top_k=5)

    print(f"\nQuery: {TEST_QUERY_IMAGE}")
    print("검색 결과 Top-5:")
    for r in results:
        print(f"  {r['rank']}위: {r['crop_path']} (원본: {r['original_photo']}, 유사도: {r['similarity']})")