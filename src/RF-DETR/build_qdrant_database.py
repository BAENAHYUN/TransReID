"""
마일스톤 4: data/sample_photos 안의 사진 전부 처리해서 Qdrant Vector DB에 저장
사용법: python src/build_qdrant_database.py

Qdrant는 FAISS와 다르게 "서버"이므로, 미리 Docker로 띄워둬야 함:
  docker run -d -p 6333:6333 -p 6334:6334 --name qdrant_server qdrant/qdrant
"""

import os
import sys

sys.path.append("src/RF-DETR")
from detect_rf import load_detect_model, detect_and_crop
from reid_clip import load_reid_model, get_embedding

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

PHOTOS_DIR = "data/sample_photos"
CROPS_DIR = "data/crops"
COLLECTION_NAME = "person_embeddings"
EMBEDDING_DIM = 1280  # CLIP-ReID(SIE-OLP)로 확인된 실제 차원


def get_qdrant_client():
    """
    로컬에 떠있는 Qdrant 서버에 연결.
    """
    client = QdrantClient(host="localhost", port=6333)
    return client


def create_collection_if_needed(client):
    """
    person_embeddings 컬렉션을 항상 새로 만듦 (기존 데이터가 있으면 완전히 삭제 후 재생성).
    이렇게 해야 재실행할 때마다 예전 결과가 섞이는 문제를 방지할 수 있음.
    """
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"기존 컬렉션 '{COLLECTION_NAME}' 삭제됨")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"컬렉션 '{COLLECTION_NAME}' 새로 생성됨")


def build_database():
    # ── 1. 모델은 딱 한 번씩만 로드 ──
    detect_model = load_detect_model()
    reid_model, device = load_reid_model()

    # ── 2. Qdrant 연결 + 컬렉션 준비 ──
    client = get_qdrant_client()
    create_collection_if_needed(client)

    # ── 3. 사진 목록 가져오기 ──
    photo_files = [
        f for f in os.listdir(PHOTOS_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"처리할 사진 {len(photo_files)}장\n")

    points = []
    point_id = 0

    # ── 4. 사진마다 검출→crop→embedding 반복 ──
    for photo_file in photo_files:
        photo_path = os.path.join(PHOTOS_DIR, photo_file)
        photo_name_no_ext = os.path.splitext(photo_file)[0]

        print(f"--- {photo_file} 처리 중 ---")

        crop_paths = detect_and_crop(
            detect_model,
            photo_path,
            output_dir=CROPS_DIR,
            prefix=f"{photo_name_no_ext}_",
        )

        for crop_path in crop_paths:
            embedding = get_embedding(reid_model, device, crop_path)

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),  # Qdrant는 파이썬 list를 받음 (numpy array 아님)
                    payload={
                        "original_photo": photo_file,
                        "crop_path": crop_path,
                    },
                )
            )
            point_id += 1

        print()

    # ── 5. Qdrant에 한꺼번에 저장 ──
    client.upsert(collection_name=COLLECTION_NAME, wait=True, points=points)

    print(f"완료: 총 {len(points)}개 embedding을 Qdrant에 저장함")

    # 확인
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"컬렉션 상태: {collection_info.points_count}개 포인트 저장됨")


if __name__ == "__main__":
    build_database()