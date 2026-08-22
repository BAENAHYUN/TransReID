"""
Face 유사도 None 원인 진단: 특정 crop이 DB에 face_detected=True로 저장되어 있는지 확인
사용법: python src/RF-DETR/diagnose_face_none.py
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_NAME = "forensic_persons"

# Fusion 결과에서 Face가 None으로 나왔던 사진들
TARGET_PHOTOS = [
    "data\\sample_photos\\George_W_Bush_0174.jpg",
    "data\\sample_photos\\George_W_Bush_0003.jpg",
    "data\\sample_photos\\George_W_Bush_0387.jpg",
    "data\\sample_photos\\George_W_Bush_0071.jpg",
]


def diagnose():
    client = QdrantClient(QDRANT_URL)

    for photo in TARGET_PHOTOS:
        # original_image 필드로 정확히 매칭되는 포인트 검색
        results, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="original_image", match=MatchValue(value=photo))]
            ),
            limit=10,
            with_payload=True,
            with_vectors=False,
        )

        if not results:
            print(f"{photo}: DB에서 못 찾음 (경로 형식이 다를 수 있음)")
            continue

        for point in results:
            payload = point.payload
            print(f"{photo}")
            print(f"  face_detected: {payload.get('face_detected')}")
            print(f"  crop_path: {payload.get('crop_path')}")
            print()


if __name__ == "__main__":
    diagnose()