"""
마일스톤 7: InsightFace(SCRFD+ArcFace)로 얼굴 검출 + embedding 추출
사용법: python src/InsightFace/face_insight.py
- load_face_model(): 모델을 딱 한 번만 로드 (buffalo_l 팩: SCRFD 검출 + ArcFace 인식)
- detect_faces(model, image_path): 사진 경로를 받아 얼굴들의 embedding 반환
주의: GTX 1060(Pascal, compute capability 6.1)은 최신 onnxruntime-gpu 공식 빌드가
컴파일된 CUDA 커널을 제공하지 않아 CPU로 강제 고정함. (dll 로딩은 정상이나
실제 커널 실행 시 CUDNN_STATUS_EXECUTION_FAILED 발생 — onnxruntime-gpu 1.26.0 확인됨)
"""
import cv2
from insightface.app import FaceAnalysis


def load_face_model(det_thresh=0.5):
    """
    InsightFace 모델(buffalo_l 팩)을 딱 한 번 로드해서 반환.
    GTX 1060은 최신 onnxruntime-gpu 공식 빌드의 CUDA 커널을 지원하지 않으므로
    CPUExecutionProvider만 사용.

    det_thresh: SCRFD 검출기 자체의 1차 필터링 기준.
        이 값보다 낮은 confidence의 얼굴은 app.get()이 반환하는 시점에
        이미 제외되어 있으므로, detect_faces()의 det_thresh를 이보다
        낮게 줘도 소용없음 (여기서 이미 걸러졌기 때문).
    """
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=det_thresh)  # ctx_id=-1 -> CPU 강제
    return app


def detect_faces(app, image_path, det_thresh=0.5):
    """
    사진 경로를 받아 얼굴 검출 + ArcFace embedding 추출.
    embedding은 JSON/Qdrant 직렬화를 위해 list[float]로 변환해서 반환.

    det_thresh: 여기서 주는 값은 load_face_model()에서 준 det_thresh보다
        "높거나 같을 때만" 의미가 있음 (2차 상향 필터링용).
        더 낮은 얼굴을 잡고 싶다면 load_face_model(det_thresh=...)를
        더 낮은 값으로 다시 호출해서 app을 새로 만들어야 함.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"경고: 이미지를 읽을 수 없음: {image_path}")
        return []

    try:
        faces = app.get(image)
    except Exception as e:
        print(f"경고: 얼굴 검출 실패: {image_path} ({e})")
        return []

    results = []
    for face in faces:
        if face.det_score < det_thresh:
            continue
        results.append({
            "bbox": face.bbox.astype(int).tolist(),
            "embedding": face.normed_embedding.tolist(),  # numpy -> list (JSON/Qdrant 직렬화용)
            "det_score": float(face.det_score),
            "age": int(face.age) if face.age is not None else None,
            "gender": str(face.sex) if hasattr(face, "sex") and face.sex is not None else None,
        })
    return results


if __name__ == "__main__":
    TEST_IMAGE_PATH = "data/sample_photos/George_W_Bush_0001.jpg"
    app = load_face_model()  # det_thresh 기본값 0.5
    faces = detect_faces(app, TEST_IMAGE_PATH)  # 여기도 0.5 -> 지금은 이전과 동일하게 동작
    print(f"\n검출된 얼굴 수: {len(faces)}")
    for i, face in enumerate(faces):
        print(f"얼굴 {i+1}: bbox={face['bbox']}, 검출신뢰도={face['det_score']:.2f}, "
              f"embedding 길이={len(face['embedding'])}, "
              f"나이={face['age']}, 성별={face['gender']}")

# load_face_model(det_thresh=0.5)가 파라미터를 받아서 app.prepare()에 전달하는 것 하나; 나중에 낮춰서 실험하고 싶으면: det_thresh=0.3 조절 가능
#bbox 관련 — 이후 크롭 예정이면 주의