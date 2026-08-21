"""
마일스톤 1: RF-DETR로 사람 검출 + Crop (함수화 버전)
- load_detect_model(): 모델을 딱 한 번만 로드
- detect_and_crop(model, image_path, output_dir, prefix): 사진 경로를 인자로 받아 crop 결과 반환

참고: RF-DETR은 Ultralytics가 아니라 Roboflow의 별도 패키지(rfdetr)를 씁니다.
      COCO 클래스 기준 "person"의 class_id는 0번입니다 (RT-DETR과 동일 기준).
"""

import os
import cv2
import numpy as np
from rfdetr import RFDETRMedium  # Nano~2XLarge 중 Medium (GTX1060)
from rfdetr.assets.coco_classes import COCO_CLASSES

CONF_THRESHOLD = 0.5  # query가 아니라 환경설정이라 상수로 둠


def load_detect_model():
    """
    RF-DETR 모델을 딱 한 번 로드해서 반환.
    처음 실행 시 pretrained weight를 자동으로 다운로드합니다.
    """
    model = RFDETRMedium()
    model.inference(compile=False, dtype="float16") # ← 이 줄 추가 (최신 함수명)
    return model


def detect_and_crop(model, image_path, output_dir="data/crops", prefix=""):
    """
    이미 로드된 model을 받아서, 사진 1장(image_path)에서 사람을 검출하고 crop.
    image_path는 인자로 받음 — 절대 하드코딩하지 않음.

    반환값: 이번 사진에서 찾은 crop 파일 경로들의 리스트
    """
    os.makedirs(output_dir, exist_ok=True)

    # RF-DETR은 URL/경로/PIL Image/numpy array 다 받을 수 있음. 여기선 경로 그대로 사용.
    detections = model.predict(image_path, threshold=CONF_THRESHOLD)# 기본 해상도(576) 사용

    image = cv2.imread(image_path)  # crop 저장을 위해 OpenCV로도 읽음(BGR)

    crop_paths = []
    person_count = 0

    # detections.class_id, detections.xyxy, detections.confidence 로 순회
    for i in range(len(detections)):
        class_id = int(detections.class_id[i])
        class_name = COCO_CLASSES[class_id]

        if class_name != "person":  # RT-DETR 때와 동일하게 "사람"만 필터링
            continue

        confidence = float(detections.confidence[i])
        x1, y1, x2, y2 = map(int, detections.xyxy[i])

        # 기존 detect.py에서 겪었던 음수/경계초과 좌표 버그 방지 (동일 원칙 재적용)
        img_h, img_w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        cropped = image[y1:y2, x1:x2]
        if cropped is None or cropped.size == 0:
            print(f"경고: 비정상 박스 → 좌표({x1},{y1},{x2},{y2}), 파일: {image_path}")
            continue

        person_count += 1
        save_path = f"{output_dir}/{prefix}person_{person_count}.jpg"
        success = cv2.imwrite(save_path, cropped)

        if not success:
            print(f"경고: {save_path} 저장 실패")
            continue

        crop_paths.append(save_path)
        print(f"사람 {person_count} 검출됨 - 신뢰도: {confidence:.2f}, 저장 위치: {save_path}")

    print(f"완료: 총 {person_count}명 검출됨")
    return crop_paths


# ── 이 파일을 단독 실행했을 때는 테스트용으로 동작 ──
if __name__ == "__main__":
    TEST_IMAGE_PATH = "data/sample_photos/George_W_Bush_0535.jpg"  # 본인 환경에 맞게 수정 가능

    model = load_detect_model()
    crop_paths = detect_and_crop(model, TEST_IMAGE_PATH)

    print(f"\n생성된 crop 파일들: {crop_paths}")