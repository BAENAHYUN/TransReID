from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Sequence

import cv2
from rfdetr import RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES


CONF_THRESHOLD = 0.5

MIN_PERSON_CROP_WIDTH = 90
MIN_PERSON_CROP_HEIGHT = 120

FORENSIC_TARGET_CLASSES = (
    "person",
    "backpack",
    "handbag",
    "umbrella",
    "suitcase",
    "tie",
    "hat",
    "cell phone",
    "bottle",
    "knife",
    "car",
    "bicycle",
    "motorcycle",
    "bus",
    "truck",
)


def _class_name(class_id: int) -> str:
    """Return a COCO class name for a class id."""
    if isinstance(COCO_CLASSES, dict):
        return str(COCO_CLASSES[class_id])
    return str(COCO_CLASSES[class_id])


def _safe_name(text: str) -> str:
    """Make a short filesystem-safe label."""
    return (
        text.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def _source_token(image_path: str) -> str:
    """Build a stable token from the source path."""
    resolved = str(Path(image_path).resolve())
    digest = hashlib.sha1(
        resolved.encode("utf-8")
    ).hexdigest()[:10]

    stem = _safe_name(
        Path(image_path).stem
    )

    return f"{stem}_{digest}"


def load_detect_model():
    """Load RF-DETR Medium once and prepare it for FP16 inference."""
    model = RFDETRMedium()
    model.inference(dtype="float16")
    return model


# ============================================================
# bbox 시각화
# ============================================================
def _draw_bbox(
    image,
    bbox,
    class_name: str,
    confidence: float,
    color=(0, 255, 0),
):
    """
    원본 crop에는 영향을 주지 않고,
    annotated 이미지에만 bbox와 class/confidence를 표시한다.
    """

    x1, y1, x2, y2 = bbox

    # bbox
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    # 표시할 텍스트
    label = f"{class_name} {confidence:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    # bbox가 이미지 위쪽에 붙어 있어도
    # 텍스트가 이미지 밖으로 나가지 않도록 처리
    text_y = max(
        y1 - 6,
        text_h + 4,
    )

    # 텍스트 배경
    cv2.rectangle(
        image,
        (x1, text_y - text_h - 4),
        (x1 + text_w + 4, text_y + baseline),
        color,
        -1,
    )

    # 텍스트
    cv2.putText(
        image,
        label,
        (x1 + 2, text_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def detect_and_crop(
    model,
    image_path,
    output_dir="data/crops",
    prefix="",
    target_classes: Optional[Sequence[str]] = FORENSIC_TARGET_CLASSES,
):
    """
    Detect target objects in one image and save accepted crops.

    bbox 시각화:
        초록색 = 최종 crop 저장 성공
        빨간색 = 검출됐지만 최종 crop으로 사용되지 않음

    실제 crop은 원본 image에서 생성하므로
    bbox 표시가 crop 이미지에 들어가지 않는다.
    """

    image_path = str(image_path)
    output_dir = str(output_dir)

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ========================================================
    # 원본 이미지 로드
    # ========================================================
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    # ========================================================
    # bbox 표시 전용 이미지
    #
    # 중요:
    # 실제 crop은 image에서 만들고,
    # bbox는 annotated에만 그린다.
    # ========================================================
    annotated = image.copy()

    # ========================================================
    # RF-DETR detection
    # ========================================================
    detections = model.predict(
        image_path,
        threshold=CONF_THRESHOLD,
    )

    allowed = (
        None
        if target_classes is None
        else set(target_classes)
    )

    accepted_crops = []
    filtered_log = []

    img_h, img_w = image.shape[:2]

    source_token = _source_token(
        image_path
    )

    accepted_index = 0

    # ========================================================
    # detection 순회
    # ========================================================
    for i in range(len(detections)):

        class_id = int(
            detections.class_id[i]
        )

        class_name = _class_name(
            class_id
        )

        confidence = float(
            detections.confidence[i]
        )

        # target class 필터
        if (
            allowed is not None
            and class_name not in allowed
        ):
            continue

        # ====================================================
        # bbox
        # ====================================================
        raw_bbox = [
            float(v)
            for v in detections.xyxy[i]
        ]

        x1, y1, x2, y2 = map(
            int,
            raw_bbox,
        )

        # 이미지 범위 clipping
        x1 = max(
            0,
            min(img_w, x1),
        )

        y1 = max(
            0,
            min(img_h, y1),
        )

        x2 = max(
            0,
            min(img_w, x2),
        )

        y2 = max(
            0,
            min(img_h, y2),
        )

        bbox = [
            x1,
            y1,
            x2,
            y2,
        ]

        # ====================================================
        # invalid bbox
        # ====================================================
        if (
            x2 <= x1
            or y2 <= y1
        ):

            filtered_log.append(
                {
                    "source_path": image_path,
                    "image_path": image_path,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": bbox,
                    "reason": "invalid_bbox",
                }
            )

            # 정상적인 사각형 자체가 아니므로
            # bbox는 그리지 않는다.
            continue

        # ====================================================
        # crop
        #
        # 반드시 원본 image에서 crop한다.
        # ====================================================
        cropped = image[
            y1:y2,
            x1:x2
        ]

        # ====================================================
        # empty crop
        # ====================================================
        if (
            cropped is None
            or cropped.size == 0
        ):

            # 최종 crop 사용 불가 → 빨간색
            _draw_bbox(
                annotated,
                bbox,
                class_name,
                confidence,
                color=(0, 0, 255),
            )

            filtered_log.append(
                {
                    "source_path": image_path,
                    "image_path": image_path,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": bbox,
                    "reason": "empty_crop",
                }
            )

            continue

        crop_h, crop_w = (
            cropped.shape[:2]
        )

        # ====================================================
        # person 최소 crop 크기 필터
        #
        # 기존 조건 그대로:
        # width < 90 또는 height < 120 → 제외
        # ====================================================
        if (
            class_name == "person"
            and (
                crop_w < MIN_PERSON_CROP_WIDTH
                or crop_h < MIN_PERSON_CROP_HEIGHT
            )
        ):

            # 검출은 됐지만 필터링 → 빨간색
            _draw_bbox(
                annotated,
                bbox,
                class_name,
                confidence,
                color=(0, 0, 255),
            )

            filtered_log.append(
                {
                    "source_path": image_path,
                    "image_path": image_path,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": bbox,
                    "width": crop_w,
                    "height": crop_h,
                    "reason": "person_crop_too_small",
                }
            )

            continue

        # ====================================================
        # crop 파일명 생성
        # ====================================================
        accepted_index += 1

        class_token = _safe_name(
            class_name
        )

        filename = (
            f"{prefix}"
            f"{source_token}_"
            f"{class_token}_"
            f"{accepted_index:03d}.jpg"
        )

        save_path = str(
            Path(output_dir)
            / filename
        )

        # ====================================================
        # 실제 crop 저장
        # ====================================================
        success = cv2.imwrite(
            save_path,
            cropped,
        )

        # ====================================================
        # 저장 실패
        # ====================================================
        if not success:

            # 검출 + 필터 통과는 했지만
            # 최종 파일 저장 실패 → 빨간색
            _draw_bbox(
                annotated,
                bbox,
                class_name,
                confidence,
                color=(0, 0, 255),
            )

            filtered_log.append(
                {
                    "source_path": image_path,
                    "image_path": image_path,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": bbox,
                    "width": crop_w,
                    "height": crop_h,
                    "reason": "save_failed",
                }
            )

            continue

        # ====================================================
        # 최종 저장 성공
        #
        # 여기까지 온 detection만 초록색
        # ====================================================
        _draw_bbox(
            annotated,
            bbox,
            class_name,
            confidence,
            color=(0, 255, 0),
        )

        # ====================================================
        # accepted crop 기록
        # ====================================================
        accepted_crops.append(
            {
                "source_path": image_path,
                "image_path": image_path,
                "crop_path": save_path,
                "path": save_path,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": bbox,
                "width": crop_w,
                "height": crop_h,
            }
        )

    # ========================================================
    # bbox 전체 이미지 저장
    # ========================================================
    original_stem = _safe_name(
        Path(image_path).stem
    )

    detected_full_path = str(
        Path(output_dir)
        / (
            f"{prefix}"
            f"{original_stem}"
            f"_detected_full.jpg"
        )
    )

    success = cv2.imwrite(
        detected_full_path,
        annotated,
    )

    if success:
        print(
            f"Detected image saved: "
            f"{detected_full_path}"
        )
    else:
        print(
            f"WARNING: Could not save detected image: "
            f"{detected_full_path}"
        )

    return (
        accepted_crops,
        filtered_log,
    )


# ============================================================
# 단일 이미지 테스트
# ============================================================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "RF-DETR single-image smoke test"
        )
    )

    parser.add_argument(
        "image",
        help="Input image path",
    )

    parser.add_argument(
        "--output-dir",
        default="data/crops",
    )

    parser.add_argument(
        "--all-classes",
        action="store_true",
        help=(
            "Detect all COCO classes instead of "
            "the forensic target classes."
        ),
    )

    args = parser.parse_args()

    # 모델 로드
    model = load_detect_model()

    # --all-classes 사용 시 모든 COCO class
    targets = (
        None
        if args.all_classes
        else FORENSIC_TARGET_CLASSES
    )

    # detection + crop
    crops, filtered = detect_and_crop(
        model,
        args.image,
        output_dir=args.output_dir,
        target_classes=targets,
    )

    print(
        f"Accepted crops: {len(crops)}"
    )

    print(
        f"Filtered crops: {len(filtered)}"
    )