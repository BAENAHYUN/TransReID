"""
마일스톤 1: RF-DETR로 사람 검출 + Crop (함수화 버전)
- load_detect_model(): 모델을 딱 한 번만 로드
- detect_and_crop(model, image_path, output_dir, prefix): 사진 경로를 인자로 받아 crop 결과 반환

참고: RF-DETR은 Ultralytics가 아니라 Roboflow의 별도 패키지(rfdetr)를 씁니다.
      COCO 클래스 기준 "person"의 class_id는 0번입니다 (RT-DETR과 동일 기준).
"""
'''
import os
import cv2
import numpy as np
from rfdetr import RFDETRMedium  # Nano~2XLarge 중 Medium 선택 (속도/정확도 균형, GTX1060 고려)
from rfdetr.assets.coco_classes import COCO_CLASSES

CONF_THRESHOLD = 0.5  # query가 아니라 환경설정이라 상수로 둠
MIN_CROP_WIDTH = 30   # 이보다 좁으면(픽셀) "정보량 부족한 crop"으로 판단해 건너뜀
MIN_CROP_HEIGHT = 50  # 이보다 낮으면 마찬가지


def load_detect_model():
    """
    RF-DETR 모델을 딱 한 번 로드해서 반환.
    처음 실행 시 pretrained weight를 자동으로 다운로드합니다.
    """
    model = RFDETRMedium()
    model.inference(dtype="float16")  # 추론 속도 최적화 (GPU에서 FP16 사용)
    return model


def detect_and_crop(model, image_path, output_dir="data/crops", prefix=""):
    """
    이미 로드된 model을 받아서, 사진 1장(image_path)에서 사람을 검출하고 crop.
    image_path는 인자로 받음 — 절대 하드코딩하지 않음.

    반환값: 이번 사진에서 찾은 crop 파일 경로들의 리스트
    """
    os.makedirs(output_dir, exist_ok=True)

    # RF-DETR은 URL/경로/PIL Image/numpy array 다 받을 수 있음. 여기선 경로 그대로 사용.
    detections = model.predict(image_path, threshold=CONF_THRESHOLD)

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

        # 화면 가장자리에 살짝 걸친 사람 등, crop이 너무 좁아서 정보량이 부족하면 건너뜀
        # (RF-DETR 검출 자체는 정직할 수 있지만, Re-ID/Face 검색엔 쓸모없는 조각이라 제외)
        crop_h, crop_w = cropped.shape[:2]
        if crop_w < MIN_CROP_WIDTH or crop_h < MIN_CROP_HEIGHT:
            print(f"건너뜀: crop이 너무 작음(가로{crop_w}x세로{crop_h}), "
                  f"화면 가장자리에 일부만 걸친 사람일 가능성, 파일: {image_path}")
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
    # 검출은 되는데 활용 불가능한 조각이 나옴 에 관한 오류 수정
    현재 코드는 1차 필터 코드 제안 (가로 30px/세로 50px)
    
    '''
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
from rfdetr import RFDETRMedium  # Nano~2XLarge 중 Medium 선택 (속도/정확도 균형, GTX1060 고려)
from rfdetr.assets.coco_classes import COCO_CLASSES

CONF_THRESHOLD = 0.5  # query가 아니라 환경설정이라 상수로 둠
MIN_CROP_WIDTH = 90    # 62px 같은 "머리카락/귀만 보이는" 애매한 crop도 걸러지도록 상향
MIN_CROP_HEIGHT = 120  # 마찬가지로 상향 (사람 상반신 정도는 되어야 함)


def load_detect_model():
    """
    RF-DETR 모델을 딱 한 번 로드해서 반환.
    처음 실행 시 pretrained weight를 자동으로 다운로드합니다.
    """
    model = RFDETRMedium()
    model.inference(dtype="float16")  # 추론 속도 최적화 (GPU에서 FP16 사용)
    return model


def detect_and_crop(model, image_path, output_dir="data/crops", prefix=""):
    """
    이미 로드된 model을 받아서, 사진 1장(image_path)에서 사람을 검출하고 crop.
    image_path는 인자로 받음 — 절대 하드코딩하지 않음.

    반환값: 이번 사진에서 찾은 crop 파일 경로들의 리스트
    """
    os.makedirs(output_dir, exist_ok=True)

    # RF-DETR은 URL/경로/PIL Image/numpy array 다 받을 수 있음. 여기선 경로 그대로 사용.
    detections = model.predict(image_path, threshold=CONF_THRESHOLD)

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

        # 화면 가장자리에 살짝 걸친 사람 등, crop이 너무 좁아서 정보량이 부족하면 건너뜀
        # (RF-DETR 검출 자체는 정직할 수 있지만, Re-ID/Face 검색엔 쓸모없는 조각이라 제외)
        crop_h, crop_w = cropped.shape[:2]
        if crop_w < MIN_CROP_WIDTH or crop_h < MIN_CROP_HEIGHT:
            print(f"건너뜀: crop이 너무 작음(가로{crop_w}x세로{crop_h}), "
                  f"화면 가장자리에 일부만 걸친 사람일 가능성, 파일: {image_path}")
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