"""
마일스톤 1~2: RT-DETR로 사람 검출 + Crop
사용법: python src/detect.py
"""

from ultralytics import RTDETR
import cv2
import os

# ── 설정 ──────────────────────────────
IMAGE_PATH = "data/sample_photos/test1.jpg"   # 테스트할 사진 경로 (본인 파일명으로 수정)
OUTPUT_DIR = "data/crops"                      # 잘라낸 사람 사진 저장 위치
CONF_THRESHOLD = 0.5                           # 신뢰도 50% 이상만 사람으로 인정

# ── 1. 모델 로드 ──────────────────────────
model = RTDETR("rtdetr-l.pt")

# ── 2. 검출 실행 ──────────────────────────
results = model(IMAGE_PATH, conf=CONF_THRESHOLD)

# ── 3. 결과에서 "사람"만 골라내기 ──────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
image = cv2.imread(IMAGE_PATH)

person_count = 0
for result in results:
    boxes = result.boxes
    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if class_id == 0:  # 0번 = person
            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cropped = image[y1:y2, x1:x2]
            save_path = f"{OUTPUT_DIR}/person_{person_count}.jpg"
            cv2.imwrite(save_path, cropped)

            print(f"사람 {person_count} 검출됨 - 신뢰도: {confidence:.2f}, 저장 위치: {save_path}")

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite(f"{OUTPUT_DIR}/detected_full.jpg", image)

print(f"\n완료: 총 {person_count}명 검출됨")
print(f"박스 그려진 전체 사진: {OUTPUT_DIR}/detected_full.jpg")