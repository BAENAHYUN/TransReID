"""
benchmark_small_person.py
- 화면에서 작게 찍힌 사람들이 실제로 잘 검출되는지 확인
"""
from rfdetr import RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES
import cv2

# CCTV 스타일(사람이 작게 나온) 테스트 이미지 몇 장 필요
TEST_IMAGES = [
    "data/sample_photos/cctv_test_1.jpg",  # 준비 필요
    "data/sample_photos/cctv_test_2.jpg",
]

model = RFDETRMedium()
model.inference(compile=False, dtype="float16")

for img_path in TEST_IMAGES:
    image = cv2.imread(img_path)
    img_h, img_w = image.shape[:2]

    detections = model.predict(img_path, threshold=0.3)  # threshold 낮춰서 일단 다 보기

    print(f"\n=== {img_path} ===")
    for i in range(len(detections)):
        class_id = int(detections.class_id[i])
        if COCO_CLASSES[class_id] != "person":
            continue
        x1, y1, x2, y2 = detections.xyxy[i]
        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / (img_w * img_h) * 100  # 전체 화면 대비 박스 크기 %
        conf = float(detections.confidence[i])
        print(f"박스 크기: 화면의 {area_ratio:.2f}%, confidence: {conf:.3f}")