"""
마일스톤 5 (업데이트): Qdrant에서 Person(Re-ID)과 Face 각각 검색
- Named Vector 구조(reid/face)에 맞춰, using="reid" / using="face"로 검색 대상 지정
- 경빈님이 클러스터링 AI 개발을 위해 작성한 통합 테스트 버전을 기반으로 정리함

사용법: python src/RF-DETR/search_qdrant.py
"""

import sys
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient

ROOT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT_DIR / "src" / "RF-DETR")
)

from reid_clip import load_reid_model, get_embedding
from face_insight import load_face_model, detect_faces


# ---------------------------------------------------------
# 설정
# ---------------------------------------------------------

QDRANT_URL = "http://127.0.0.1:6333"

COLLECTION_NAME = "forensic_persons"


# ---------------------------------------------------------
# Qdrant 연결
# ---------------------------------------------------------

def load_qdrant_client():
    return QdrantClient(QDRANT_URL)


# ---------------------------------------------------------
# Query 사진의 Re-ID embedding 추출
# ---------------------------------------------------------

def get_reid_embedding(reid_model, reid_device, image_path):
    embedding = get_embedding(reid_model, reid_device, str(image_path))

    embedding = np.asarray(embedding, dtype=np.float32)

    if embedding.shape != (1280,):
        raise ValueError(f"예상과 다른 Re-ID shape: {embedding.shape}")

    return embedding


# ---------------------------------------------------------
# Query 사진의 Face embedding 추출
# ---------------------------------------------------------

def get_face_embedding(face_model, image_path):
    faces = detect_faces(face_model, str(image_path))

    if not faces:
        return None

    # 여러 얼굴이 검출되면 신뢰도(det_score)가 가장 높은 얼굴 하나만 사용
    face = max(faces, key=lambda x: x["det_score"])

    embedding = np.asarray(face["embedding"], dtype=np.float32)

    if embedding.shape != (512,):
        raise ValueError(f"예상과 다른 Face shape: {embedding.shape}")

    # 벡터 길이를 1로 정규화 (코사인 유사도 계산을 위해)
    norm = np.linalg.norm(embedding)
    if norm > 1e-12:
        embedding = embedding / norm

    return embedding


# ---------------------------------------------------------
# Re-ID(몸 전체 특징) 기준 검색
# ---------------------------------------------------------

def search_reid(client, query_embedding, top_k=5):
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        using="reid",  # Named Vector 중 "reid" 벡터로 검색하겠다는 지정
        limit=top_k,
        with_payload=True,
    )

    return results.points


# ---------------------------------------------------------
# Face(얼굴 특징) 기준 검색
# ---------------------------------------------------------

def search_face(client, query_embedding, top_k=5):
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        using="face",  # Named Vector 중 "face" 벡터로 검색하겠다는 지정
        limit=top_k,
        with_payload=True,
    )

    return results.points


# ---------------------------------------------------------
# 검색 결과 출력
# ---------------------------------------------------------

def print_results(title, points):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    if not points:
        print("결과 없음.")
        return

    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}

        print()
        print(f"[순위 {rank}]")
        print(f"유사도     : {point.score:.4f}")
        print(f"원본 사진  : {payload.get('original_image')}")
        print(f"Crop 경로  : {payload.get('crop_path')}")
        print(f"얼굴 검출  : {payload.get('face_detected')}")


# ---------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------

def main():
    print("=" * 70)
    print("FORENSIC QDRANT 검색 테스트")
    print("=" * 70)

    # ── Query 사진 경로 (본인 환경에 맞게 수정 가능) ──
    query_image = (
        ROOT_DIR / "data" / "crops" / "George_W_Bush_0001_person_1.jpg"
    )

    if not query_image.exists():
        raise FileNotFoundError(f"Query 사진을 찾을 수 없음:\n{query_image}")

    print()
    print(f"Query 사진: {query_image}")

    # ── Qdrant 연결 확인 ──
    print()
    print("[1/4] Qdrant 연결 중...")

    client = load_qdrant_client()
    collections = client.get_collections()

    print("Qdrant 연결 완료.")
    print(collections)

    # ── 모델 로드 ──
    print()
    print("[2/4] CLIP-ReID 로드 중...")
    reid_model, reid_device = load_reid_model()

    print()
    print("[3/4] InsightFace 로드 중...")
    face_model = load_face_model()

    # ── Query embedding 추출 ──
    print()
    print("[4/4] Query embedding 추출 중...")

    print("[Re-ID] 1280D 추출 중...")
    reid_embedding = get_reid_embedding(reid_model, reid_device, query_image)
    print(f"[Re-ID] shape={reid_embedding.shape}")

    # ── Re-ID(몸 전체) 검색 ──
    reid_results = search_reid(client, reid_embedding, top_k=5)
    print_results("Re-ID 검색 결과 (1280D, 몸 전체 특징)", reid_results)

    # ── Face(얼굴) 검색 ──
    print()
    print("[Face] 512D 추출 중...")
    face_embedding = get_face_embedding(face_model, query_image)

    if face_embedding is None:
        print("[Face] 얼굴이 검출되지 않음.")
    else:
        print(f"[Face] shape={face_embedding.shape}")

        face_results = search_face(client, face_embedding, top_k=5)
        print_results("Face 검색 결과 (512D, 얼굴 특징)", face_results)

    print()
    print("=" * 70)
    print("검색 테스트 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()