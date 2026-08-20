"""
마일스톤 1~2: RT-DETR로 사람 검출 + Crop (함수화 버전)
- load_detect_model(): 모델을 딱 한 번만 로드
- detect_and_crop(model, image_path, output_dir): 사진 경로를 인자로 받아 crop 결과 반환
"""

from ultralytics import RTDETR
import cv2
import os

CONF_THRESHOLD = 0.5  # query가 아니라 "환경설정"이라 그대로 상수로 둬도 됨


def load_detect_model():
    """
    RT-DETR 모델을 딱 한 번 로드해서 반환.
    M4에서 사진 여러 장 처리할 때, 이 함수는 딱 1번만 호출하고
    나온 model을 계속 재사용해야 함 (매번 부르면 매번 파일을 다시 읽어서 느려짐).
    """
    model = RTDETR("rtdetr-l.pt")
    return model


def detect_and_crop(model, image_path, output_dir="data/crops", prefix=""):
    """
    이미 로드된 model을 받아서, 사진 1장(image_path)에서 사람을 검출하고 crop.
    image_path는 인자로 받음 — 절대 하드코딩하지 않음.
    prefix: 여러 사진을 처리할 때 crop 파일명이 서로 안 겹치게 구분하는 접두어
            (예: "photo1_person_1.jpg" 처럼). 지금은 안 써도 되고 M4에서 활용.

    반환값: 이번 사진에서 찾은 crop 파일 경로들의 리스트
    """
    os.makedirs(output_dir, exist_ok=True)
    results = model(image_path, conf=CONF_THRESHOLD)
    image = cv2.imread(image_path)

    crop_paths = []
    person_count = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id == 0:  # 0번 = person
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 진짜 수정: 좌표가 이미지 경계를 살짝 벗어나면(음수 등)
                # NumPy 음수 인덱스 슬라이싱 문제로 빈 crop이 생기므로, 경계 안으로 고정
                img_h, img_w = image.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)

                cropped = image[y1:y2, x1:x2]

                # 그래도 비정상이면(진짜 찌그러진 박스) 건너뜀 — 이제 이 경고는 거의 안 뜰 것
                if cropped is None or cropped.size == 0:
                    print(f"경고: 여전히 비정상 박스 → 좌표({x1},{y1},{x2},{y2}), "
                          f"원본 이미지 크기({img_w}x{img_h}), 파일: {image_path}")
                    continue

                person_count += 1
                save_path = f"{output_dir}/{prefix}person_{person_count}.jpg"
                success = cv2.imwrite(save_path, cropped)

                if not success:
                    print(f"경고: {save_path} 저장 실패 (cv2.imwrite가 False 반환)")
                    continue

                crop_paths.append(save_path)

                print(f"사람 {person_count} 검출됨 - 신뢰도: {confidence:.2f}, 저장 위치: {save_path}")
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(f"{output_dir}/{prefix}detected_full.jpg", image)
    print(f"완료: 총 {person_count}명 검출됨")

    return crop_paths


# ── 이 파일을 단독 실행했을 때(python src/detect.py)는 테스트용으로 동작 ──
if __name__ == "__main__":
    TEST_IMAGE_PATH = "data/sample_photos/test1.jpg"

    model = load_detect_model()
    crop_paths = detect_and_crop(model, TEST_IMAGE_PATH)

    print(f"\n생성된 crop 파일들: {crop_paths}")