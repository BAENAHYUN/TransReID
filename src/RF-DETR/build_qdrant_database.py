"""
마일스톤 4/11 (업데이트): data/sample_photos 안의 사진 전부 처리해서 Qdrant Vector DB에 저장
- Person(CLIP-ReID, 1280D) + Face(InsightFace, 512D) + Semantic(SigLIP2, 768D)를
  Named Vector로 동시 저장 (M11: 3-way Fusion 검색을 위한 기반)

사용법: python src/RF-DETR/build_qdrant_database.py

Qdrant는 FAISS와 다르게 "서버"이므로, 미리 Docker로 띄워둬야 함:
  docker run -d -p 6333:6333 -p 6334:6334 --name qdrant_server qdrant/qdrant
"""

import sys
import uuid
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models


# =========================================================
# 프로젝트 경로 설정
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT_DIR / "src" / "RF-DETR")
)

from detect_rf import load_detect_model, detect_and_crop
from reid_clip import load_reid_model, get_embedding
from face_insight import load_face_model, detect_faces
from siglip_semantic import load_semantic_model, get_image_embedding


# =========================================================
# 설정값
# =========================================================

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_NAME = "forensic_persons"

IMAGE_DIR = ROOT_DIR / "data" / "sample_photos"
CROP_DIR = ROOT_DIR / "data" / "crops"

BATCH_SIZE = 32
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RESET_COLLECTION = True


# =========================================================
# Qdrant 연결/컬렉션 관리
# =========================================================

def get_qdrant_client():
    return QdrantClient(QDRANT_URL)


def create_collection(client):
    """
    forensic_persons 컬렉션을 준비.
    RESET_COLLECTION=True면 기존 것 삭제 후 새로 생성.
    """
    if client.collection_exists(COLLECTION_NAME):
        if not RESET_COLLECTION:
            print(f"기존 컬렉션 재사용: {COLLECTION_NAME}")
            return
        print(f"기존 컬렉션 삭제 중: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    print(f"컬렉션 생성 중: {COLLECTION_NAME}")

    # Named vectors: "reid"(1280D), "face"(512D), "semantic"(768D, SigLIP2)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "reid": models.VectorParams(size=1280, distance=models.Distance.COSINE),
            "face": models.VectorParams(size=512, distance=models.Distance.COSINE),
            "semantic": models.VectorParams(size=768, distance=models.Distance.COSINE),
        },
    )

    print("컬렉션 생성 완료.")


# =========================================================
# 얼굴 embedding 추출
# =========================================================

def get_face_embedding(face_model, crop_path):
    """
    crop 사진에서 얼굴을 찾아 embedding을 반환.
    여러 얼굴이 검출되면 신뢰도(det_score)가 가장 높은 얼굴 하나만 사용.
    얼굴이 없으면 (None, None) 반환.
    """
    faces = detect_faces(face_model, str(crop_path))

    if not faces:
        return None, None

    face = max(faces, key=lambda x: x["det_score"])
    embedding = np.asarray(face["embedding"], dtype=np.float32)

    if embedding.shape != (512,):
        raise ValueError(f"예상과 다른 face embedding shape: {embedding.shape}")

    norm = np.linalg.norm(embedding)
    if norm > 1e-12:
        embedding = embedding / norm

    return embedding, face


# =========================================================
# 사진 1장 처리 (검출 -> Person/Face/Semantic embedding -> Qdrant Point 생성)
# =========================================================

def process_image(image_path, detect_model, reid_model, reid_device, face_model,
                   semantic_model, semantic_processor, semantic_device):
    image_path = Path(image_path)
    points = []

    # ── RF-DETR로 사람 검출 + crop ──
    crop_paths = detect_and_crop(
        detect_model,
        str(image_path),
        output_dir=str(CROP_DIR),
        prefix=f"{image_path.stem}_",
    )

    if not crop_paths:
        return points

    # ── crop마다 Person + Face + Semantic embedding 추출 ──
    for crop_path in crop_paths:
        crop_path = Path(crop_path)

        # CLIP-ReID 1280D (몸 전체 특징)
        reid_embedding = get_embedding(reid_model, reid_device, str(crop_path))
        reid_embedding = np.asarray(reid_embedding, dtype=np.float32)

        if reid_embedding.shape != (1280,):
            raise ValueError(f"예상과 다른 ReID embedding shape: {reid_embedding.shape}")

        # InsightFace 512D (얼굴 특징, 얼굴이 안 보이면 None)
        face_embedding, face_info = get_face_embedding(face_model, crop_path)

        # SigLIP2 768D (crop 사진 자체의 장면/속성 특징)
        semantic_embedding = get_image_embedding(
            semantic_model, semantic_processor, semantic_device, str(crop_path)
        )
        semantic_embedding = np.asarray(semantic_embedding, dtype=np.float32)

        if semantic_embedding.shape != (768,):
            raise ValueError(f"예상과 다른 semantic embedding shape: {semantic_embedding.shape}")

        # ── Payload(메타데이터) 구성 ──
        payload = {
            "source": "sample_photos",
            "original_image": str(image_path.relative_to(ROOT_DIR)),
            "crop_path": str(crop_path.relative_to(ROOT_DIR)),
            "crop_filename": crop_path.name,
            "person_detection": True,
            "face_detected": (face_embedding is not None),
        }

        if face_info is not None:
            payload["face_detection_score"] = float(face_info["det_score"])
            payload["face_bbox"] = face_info["bbox"]
            payload["face_age"] = face_info["age"]
            payload["face_gender"] = face_info["gender"]

        # ── Named vectors 구성 (reid/semantic은 항상 있음, face는 있을 때만) ──
        vectors = {
            "reid": reid_embedding.tolist(),
            "semantic": semantic_embedding.tolist(),
        }
        if face_embedding is not None:
            vectors["face"] = face_embedding.tolist()

        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors,
            payload=payload,
        )

        points.append(point)

    return points


# =========================================================
# 배치 업로드
# =========================================================

def upload_batch(client, points):
    if not points:
        return

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )


# =========================================================
# 메인 실행
# =========================================================

def main():
    start_time = time.time()

    print()
    print("=" * 70)
    print("FORENSIC VECTOR DATABASE - SAMPLE DATASET (Person+Face+Semantic)")
    print("=" * 70)

    print()
    print(f"Root       : {ROOT_DIR}")
    print(f"Images     : {IMAGE_DIR}")
    print(f"Qdrant     : {QDRANT_URL}")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Batch size : {BATCH_SIZE}")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("[1/6] Qdrant 연결 중...")
    client = get_qdrant_client()
    print(client.get_collections())
    create_collection(client)

    print()
    print("[2/6] RF-DETR 로드 중...")
    detect_model = load_detect_model()

    print()
    print("[3/6] CLIP-ReID 로드 중...")
    reid_model, reid_device = load_reid_model()

    print()
    print("[4/6] InsightFace 로드 중...")
    face_model = load_face_model()

    print()
    print("[5/6] SigLIP2 로드 중...")
    semantic_model, semantic_processor, semantic_device = load_semantic_model()

    image_files = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    print()
    print(f"[6/6] 발견된 이미지: {len(image_files)}장")

    if not image_files:
        print("에러: 이미지를 찾을 수 없습니다.")
        return

    total_images = len(image_files)
    processed_images = 0
    failed_images = 0
    no_person_images = 0
    total_persons = 0
    total_faces = 0
    batch_points = []

    for index, image_path in enumerate(image_files, start=1):
        elapsed = time.time() - start_time

        print()
        print("=" * 70)
        print(f"[{index}/{total_images}] {image_path.name}")
        print(f"진행률: {index / total_images * 100:.2f}%")
        print(f"경과 시간: {elapsed / 60:.1f}분")
        print("=" * 70)

        try:
            points = process_image(
                image_path, detect_model, reid_model, reid_device, face_model,
                semantic_model, semantic_processor, semantic_device,
            )
            processed_images += 1

            if not points:
                no_person_images += 1
                print("사람 검출 안 됨.")
                continue

            total_persons += len(points)
            for point in points:
                if point.payload.get("face_detected", False):
                    total_faces += 1

            batch_points.extend(points)

            print(f"검출된 사람 수: {len(points)}")
            print(f"대기 중인 배치 포인트: {len(batch_points)}")

            if len(batch_points) >= BATCH_SIZE:
                print()
                print(f"[Qdrant] {len(batch_points)}개 업로드 중...")
                upload_batch(client, batch_points)
                print("[Qdrant] 배치 업로드 완료.")
                batch_points.clear()

        except Exception as e:
            failed_images += 1
            print()
            print(f"에러 발생: {image_path.name}")
            print(f"{type(e).__name__}: {e}")
            continue

    if batch_points:
        print()
        print(f"[Qdrant] 마지막 {len(batch_points)}개 업로드 중...")
        upload_batch(client, batch_points)
        batch_points.clear()
        print("[Qdrant] 마지막 배치 완료.")

    total_time = time.time() - start_time
    info = client.get_collection(COLLECTION_NAME)

    print()
    print()
    print("=" * 70)
    print("구축 완료 (Person+Face+Semantic)")
    print("=" * 70)

    print()
    print(f"발견된 이미지       : {total_images}")
    print(f"처리된 이미지       : {processed_images}")
    print(f"실패한 이미지       : {failed_images}")
    print(f"사람 없는 이미지    : {no_person_images}")
    print(f"검출된 사람(crop)   : {total_persons}")
    print(f"얼굴 검출된 수      : {total_faces}")
    print(f"Qdrant 포인트 수    : {info.points_count}")
    print(f"처리 시간           : {total_time / 60:.2f}분")

    print()
    print(f"컬렉션: {COLLECTION_NAME}")

    print()
    print("=" * 70)
    print("샘플 데이터셋 구축 성공 (Person+Face+Semantic)")
    print("=" * 70)


if __name__ == "__main__":
    main()