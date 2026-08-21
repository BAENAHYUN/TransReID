"""
마일스톤 7: InsightFace(SCRFD+ArcFace)로 얼굴 검출 + embedding 추출
사용법: python src/InsightFace/face_insight.py

- load_face_model(): 모델을 딱 한 번만 로드 (buffalo_l 팩: SCRFD 검출 + ArcFace 인식)
- detect_faces(model, image_path): 사진 경로를 받아 얼굴들의 embedding 반환
"""

import cv2
from insightface.app import FaceAnalysis


def load_face_model():
    """
    InsightFace 모델(buffalo_l 팩)을 딱 한 번 로드해서 반환.
    처음 실행 시 pretrained weight를 자동으로 다운로드함 (약 330MB).
    buffalo_l = SCRFD(검출) + ArcFace(512차원 embedding) + 나이/성별 추정 포함.
    """
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 -> GPU 사용
    return app


def detect_faces(app, image_path):
    """
    이미 로드된 app을 받아서, 사진 1장(image_path)에서 얼굴을 검출하고
    각 얼굴의 정보(bbox, embedding, 나이, 성별)를 리스트로 반환.
    image_path는 인자로 받음 — 절대 하드코딩하지 않음.

    반환값: [{"bbox": [x1,y1,x2,y2], "embedding": ndarray(512,), "det_score": float,
              "age": int, "gender": int(0=여성,1=남성)}, ...]
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"경고: 이미지를 읽을 수 없음: {image_path}")
        return []

    faces = app.get(image)

    results = []
    for face in faces:
        results.append({
            "bbox": face.bbox.astype(int).tolist(),
            "embedding": face.embedding,  # 512차원 ArcFace embedding
            "det_score": float(face.det_score),
            "age": int(face.age) if face.age is not None else None,
            #성별은 str
            "gender": str(face.sex) if hasattr(face, "sex") and face.sex is not None else None,
        })

    return results


# ── 이 파일을 단독 실행했을 때는 테스트용으로 동작 ──
if __name__ == "__main__":
    TEST_IMAGE_PATH = "data/sample_photos/George_W_Bush_0001.jpg"  # 본인 환경에 맞게 수정 가능

    app = load_face_model()
    faces = detect_faces(app, TEST_IMAGE_PATH)

    print(f"\n검출된 얼굴 수: {len(faces)}")
    for i, face in enumerate(faces):
        print(f"얼굴 {i+1}: bbox={face['bbox']}, 검출신뢰도={face['det_score']:.2f}, "
              f"embedding shape={face['embedding'].shape}")