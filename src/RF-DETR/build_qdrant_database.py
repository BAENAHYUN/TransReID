"""
마일스톤 4 (업데이트): data/sample_photos 안의 사진 전부 처리해서 Qdrant Vector DB에 저장
- Person(CLIP-ReID, 1280D)과 Face(InsightFace, 512D)를 Named Vector로 동시 저장
- 경빈님이 클러스터링 AI 개발을 위해 통합 테스트하며 개선한 버전을 기반으로 정리함

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

# 이 파일 위치: C:\TransReID\src\RF-DETR\build_qdrant_database.py
# parents[2] = RF-DETR -> src -> TransReID(루트), 2단계 위로 올라가면 루트가 나옴
ROOT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT_DIR / "src" / "RF-DETR")
)

from detect_rf import load_detect_model, detect_and_crop
from reid_clip import load_reid_model, get_embedding
from face_insight import load_face_model, detect_faces


# =========================================================
# 설정값 (query가 아니라 환경설정이므로 상수로 둠)
# =========================================================

QDRANT_URL = "http://127.0.0.1:6333"

COLLECTION_NAME = "forensic_persons"  # 경빈님과 이름 통일 (클러스터링 AI가 이 이름을 찾음)

# 실제 데이터 폴더
IMAGE_DIR = ROOT_DIR / "data" / "sample_photos"

CROP_DIR = ROOT_DIR / "data" / "crops"

BATCH_SIZE = 32

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

# True면 기존 forensic_persons를 삭제하고 새로 구축 (재실행 시 데이터 중복 방지)
RESET_COLLECTION = True


# =========================================================
# Qdrant 연결/컬렉션 관리
# =========================================================

def get_qdrant_client():
    return QdrantClient(QDRANT_URL)


def create_collection(client):
    """
    forensic_persons 컬렉션을 준비.
    RESET_COLLECTION=True면 기존 것 삭제 후 새로 생성,
    False면 기존 컬렉션을 그대로 재사용.
    """
    if client.collection_exists(COLLECTION_NAME):

        if not RESET_COLLECTION:
            print(f"기존 컬렉션 재사용: {COLLECTION_NAME}")
            return

        print(f"기존 컬렉션 삭제 중: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    print(f"컬렉션 생성 중: {COLLECTION_NAME}")

    # Named vectors: "reid"(1280D, 몸 전체 특징)와 "face"(512D, 얼굴 특징)를
    # 하나의 포인트 안에 각각 다른 이름으로 동시에 저장
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "reid": models.VectorParams(
                size=1280,
                distance=models.Distance.COSINE,
            ),
            "face": models.VectorParams(
                size=512,
                distance=models.Distance.COSINE,
            ),
        },
    )

    print("컬렉션 생성 완료.")


# =========================================================
# 얼굴 embedding 추출
# =========================================================

def get_face_embedding(face_model, crop_path):
    """
    crop 사진에서 얼굴을 찾아 embedding을 반환.
    얼굴이 여러 개 검출되면 신뢰도(det_score)가 가장 높은 얼굴 하나만 사용.
    얼굴이 없으면 (None, None) 반환.
    """
    faces = detect_faces(face_model, str(crop_path))

    if not faces:
        return None, None

    # 가장 신뢰도 높은 얼굴 선택
    face = max(faces, key=lambda x: x["det_score"])

    embedding = np.asarray(face["embedding"], dtype=np.float32)

    if embedding.shape != (512,):
        raise ValueError(f"예상과 다른 face embedding shape: {embedding.shape}")

    # 벡터 길이를 1로 정규화 (코사인 유사도 계산을 위해)
    norm = np.linalg.norm(embedding)
    if norm > 1e-12:
        embedding = embedding / norm

    return embedding, face


# =========================================================
# 사진 1장 처리 (검출 -> Person/Face embedding -> Qdrant Point 생성)
# =========================================================

def process_image(image_path, detect_model, reid_model, reid_device, face_model):
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

    # ── crop마다 Person + Face embedding 추출 ──
    for crop_path in crop_paths:
        crop_path = Path(crop_path)

        # CLIP-ReID 1280D (몸 전체 특징)
        reid_embedding = get_embedding(reid_model, reid_device, str(crop_path))
        reid_embedding = np.asarray(reid_embedding, dtype=np.float32)

        if reid_embedding.shape != (1280,):
            raise ValueError(f"예상과 다른 ReID embedding shape: {reid_embedding.shape}")

        # InsightFace 512D (얼굴 특징, 얼굴이 안 보이면 None)
        face_embedding, face_info = get_face_embedding(face_model, crop_path)

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

        # ── Named vectors 구성 (reid는 항상 있음, face는 있을 때만) ──
        vectors = {"reid": reid_embedding.tolist()}
        if face_embedding is not None:
            vectors["face"] = face_embedding.tolist()

        # ── Qdrant Point 생성 (UUID로 ID 발급, 재실행 시 중복/충돌 방지) ──
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
    print("FORENSIC VECTOR DATABASE - SAMPLE DATASET")
    print("=" * 70)

    print()
    print(f"Root       : {ROOT_DIR}")
    print(f"Images     : {IMAGE_DIR}")
    print(f"Qdrant     : {QDRANT_URL}")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Batch size : {BATCH_SIZE}")

    # ── 폴더 준비 ──
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    # ── Qdrant 연결 및 컬렉션 준비 ──
    print()
    print("[1/5] Qdrant 연결 중...")
    client = get_qdrant_client()
    print(client.get_collections())  # 서버 연결 확인
    create_collection(client)

    # ── 모델 로드 (각 1번씩만) ──
    print()
    print("[2/5] RF-DETR 로드 중...")
    detect_model = load_detect_model()

    print()
    print("[3/5] CLIP-ReID 로드 중...")
    reid_model, reid_device = load_reid_model()

    print()
    print("[4/5] InsightFace 로드 중...")
    face_model = load_face_model()

    # ── 사진 목록 ──
    image_files = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    print()
    print(f"[5/5] 발견된 이미지: {len(image_files)}장")

    if not image_files:
        print("에러: 이미지를 찾을 수 없습니다.")
        return

    # ── 처리 통계 변수 ──
    total_images = len(image_files)
    processed_images = 0
    failed_images = 0
    no_person_images = 0
    total_persons = 0
    total_faces = 0
    batch_points = []

    # ── 사진마다 순회하며 처리 ──
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
                image_path, detect_model, reid_model, reid_device, face_model
            )
            processed_images += 1

            if not points:
                no_person_images += 1
                print("사람 검출 안 됨.")
                continue

            # 통계 집계
            total_persons += len(points)
            for point in points:
                if point.payload.get("face_detected", False):
                    total_faces += 1

            batch_points.extend(points)

            print(f"검출된 사람 수: {len(points)}")
            print(f"대기 중인 배치 포인트: {len(batch_points)}")

            # 배치가 다 차면 업로드
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
            continue  # 이 사진만 건너뛰고 다음 사진 계속 처리

    # ── 남은 포인트 업로드 ──
    if batch_points:
        print()
        print(f"[Qdrant] 마지막 {len(batch_points)}개 업로드 중...")
        upload_batch(client, batch_points)
        batch_points.clear()
        print("[Qdrant] 마지막 배치 완료.")

    # ── 최종 결과 출력 ──
    total_time = time.time() - start_time
    info = client.get_collection(COLLECTION_NAME)

    print()
    print()
    print("=" * 70)
    print("구축 완료")
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
    print("샘플 데이터셋 구축 성공")
    print("=" * 70)


if __name__ == "__main__":
    main()