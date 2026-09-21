from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Optional, Sequence

import cv2
from rfdetr import RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES


CONF_THRESHOLD = 0.5


def validate_conf_threshold(value) -> float:
    """검출 임계값은 0~1 의 유한한 실수여야 한다. 호출 즉시 검증한다."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"conf_threshold 는 실수여야 합니다: {value!r}") from exc
    if not (math.isfinite(v) and 0.0 <= v <= 1.0):
        raise ValueError(f"conf_threshold 는 0 이상 1 이하여야 합니다: {v!r}")
    return v


# 너비 25 는 MEVID (WACV 2023, arXiv:2211.04656) 3.2.2 에서 person ReID
# 주석에 필요한 최소 해상도로 경험적으로 도출한 값이다. 원 논문은 25x75 를
# 쓰지만, 여기서는 너비만 채택하고 높이는 기존 120 을 유지한다.
# 검출 통계상 병목이 너비였고(width only 121 vs height only 11), 높이를
# 낮출 근거는 확인되지 않았다.
MIN_PERSON_CROP_WIDTH = 25
MIN_PERSON_CROP_HEIGHT = 120


# 객체에는 사람 기준을 쓸 수 없다. cell phone / tie / bottle 은 정상 검출도
# 작게 나오므로 90x120 을 적용하면 유효한 crop 이 대량으로 버려진다.
# 사실상 하한선이며, 데이터를 보고 조정하기 위한 훅이다.
MIN_OBJECT_CROP_WIDTH = 20
MIN_OBJECT_CROP_HEIGHT = 20


FORENSIC_TARGET_CLASSES = (
    "person",
    "backpack",
    "handbag",
    "umbrella",
    "suitcase",
    "tie",
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


def _source_token(image_id: str) -> str:
    """
    Build a stable token from the LOGICAL image id.

    This used to hash the resolved absolute path, which made the crop
    filename depend on the drive letter and dataset location. Since
    detection_id is derived from the crop filename downstream, the same
    source image produced different Qdrant point IDs on different machines.

    Hashing image_id instead keeps the token identical anywhere the same
    logical id is used, while still avoiding filename collisions between
    images that share a stem.
    """
    digest = hashlib.sha1(
        image_id.encode("utf-8")
    ).hexdigest()[:10]

    stem = _safe_name(
        Path(image_id).stem
    )

    return f"{stem}_{digest}"


def _detection_token(detection_id: str) -> str:
    """
    crop 파일명에 쓰는 detection_id 의 짧은 해시.

    예전에는 accepted_index(필터를 통과한 순번)를 썼다. 그 값은 detection
    순서와 필터 통과 여부에 따라 밀리므로 threshold 를 조금만 바꿔도 같은
    detection 이 다른 파일명을 갖게 되고, 옛 filter_stats.json 의 crop_path
    가 엉뚱한 이미지를 가리키게 된다. detection_id 해시는 detection 이 같으면
    항상 같고, 같은 이미지 안에서 중복 제거 뒤에는 유일하다.

    detection_id 자체의 정의(image_id # class # 정수 bbox)는 바꾸지 않는다.
    """
    return hashlib.sha1(
        detection_id.encode("utf-8")
    ).hexdigest()[:10]


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
    image_id: Optional[str] = None,
    source: Optional[str] = None,
    min_person_width: int = MIN_PERSON_CROP_WIDTH,
    min_person_height: int = MIN_PERSON_CROP_HEIGHT,
    min_object_width: int = MIN_OBJECT_CROP_WIDTH,
    min_object_height: int = MIN_OBJECT_CROP_HEIGHT,
    save_annotated: bool = True,
    *,
    conf_threshold: float = CONF_THRESHOLD,
):
    """
    Detect target objects in one image and save accepted crops.

    bbox 시각화:
        초록색 = 최종 crop 저장 성공
        빨간색 = 검출됐지만 최종 crop으로 사용되지 않음

    실제 crop은 원본 image에서 생성하므로
    bbox 표시가 crop 이미지에 들어가지 않는다.

    image_id:
        Logical identity of the source image, e.g.
        "coco_train2017/000000015496.jpg". The caller owns this value because
        only the caller knows how the dataset is laid out. Both the crop
        filename and detection_id are derived from it, so it must not contain
        machine-specific parts such as a drive letter.

        Falls back to the bare filename when omitted, which keeps the
        single-image smoke test below working. Batch runs should always pass
        it explicitly.

    source:
        Provenance label recorded on every emitted record and, downstream,
        written to the integrated DB's "source" field. This module has no way
        to know where an image came from, so there is no default: when it is
        None the key is simply left out rather than guessed.

    min_person_* / min_object_*:
        Minimum crop size per kind. Person crops need enough resolution for
        ReID, but small objects are legitimately small, so the two must not
        share a threshold. Batch runs pass these from the CLI, and the batch
        checkpoint fingerprint includes them: changing the rule must not
        resume a directory built under the old one.

    Duplicate detections:
        RF-DETR has no NMS, so two queries can predict the same box; after
        integer clipping they share one detection_id. Only the highest
        confidence prediction is kept (ties: the first one), BEFORE any crop
        is written, so no duplicate crop file is created. The rest are
        reported in filtered_log with reason "duplicate_detection_id" and
        the kept confidence. detection_id's definition is unchanged:
        image_id # class # integer bbox.

    Crop filenames:
        <source_token>_<class>_<sha1(detection_id)[:10]>.jpg. Deterministic
        per detection, unlike the former accepted-index suffix that shifted
        whenever a threshold changed.

    save_annotated:
        Write the bbox visualization under <output_dir>/detected_full/.
        Batch runs turn this off by default - at 80k images it doubles the
        file count and the write time for a debug artifact.
    """

    image_path = str(image_path)
    output_dir = str(output_dir)

    # Logical id. Never derived from an absolute path.
    if image_id is None:
        image_id = Path(image_path).name

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
    # 임계값은 호출부(RF-DETR_batch --conf-threshold / --config, detect_rf CLI)에서
    # 정한다. 이전에는 모듈 상수에 고정되어 CLI/GUI/yaml 로 바꿀 수 없었다.
    conf_threshold = validate_conf_threshold(conf_threshold)

    detections = model.predict(
        image_path,
        threshold=conf_threshold,
    )

    allowed = (
        None
        if target_classes is None
        else set(target_classes)
    )

    accepted_crops = []
    filtered_log = []

    img_h, img_w = image.shape[:2]

    # crop 파일명과 시각화 파일명의 공통 접두어.
    # 논리 image_id 에서 파생되므로 머신/드라이브가 달라도 동일하다.
    source_token = _source_token(
        image_id
    )

    # ========================================================
    # 1차: 후보 정리 + detection_id 생성 (crop 저장 전)
    #
    # detection_id = 논리 image_id + class + 정수 bbox. 정의는 예전과 같다.
    #
    # RF-DETR 은 NMS 가 없어 두 query 가 같은 box 를 내는 일이 드물게 있고,
    # 정수화/clipping 뒤 detection_id 가 완전히 겹친다. 두 crop 은 같은 원본의
    # 같은 영역이라 픽셀이 동일하므로 하나만 있어야 한다. 저장 전에 걸러야
    # 중복 crop 파일이 생기지 않고, build_db.py 의 전역 중복 검사(그대로
    # 유지)가 정상 데이터에서 걸리지 않는다.
    # ========================================================
    candidates = []

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

        # target class 필터. 기록하지 않고 건너뛴다 (기존 동작 유지).
        if (
            allowed is not None
            and class_name not in allowed
        ):
            continue

        raw_bbox = [
            float(v)
            for v in detections.xyxy[i]
        ]

        x1, y1, x2, y2 = map(
            int,
            raw_bbox,
        )

        # 이미지 범위 clipping
        x1 = max(0, min(img_w, x1))
        y1 = max(0, min(img_h, y1))
        x2 = max(0, min(img_w, x2))
        y2 = max(0, min(img_h, y2))

        bbox = [
            x1,
            y1,
            x2,
            y2,
        ]

        class_token = _safe_name(
            class_name
        )

        # 정수 bbox 는 이미 clipping 되어 있어 부동소수 오차에 흔들리지 않는다.
        detection_id = (
            f"{image_id}#"
            f"{class_token}#"
            f"{x1}_{y1}_{x2}_{y2}"
        )

        candidates.append({
            "index": i,
            "class_id": class_id,
            "class_name": class_name,
            "class_token": class_token,
            "confidence": confidence,
            "bbox": bbox,
            "detection_id": detection_id,
        })

    # 같은 detection_id 가 여러 개면 confidence 가 가장 높은 것 하나만 남긴다.
    # 동점이면 먼저 나온 것 (strict > 비교라 뒤의 동점은 교체하지 않는다).
    # 결정적이다: 같은 입력이면 같은 승자.
    winner_by_id = {}

    for cand in candidates:
        current = winner_by_id.get(
            cand["detection_id"]
        )

        if (
            current is None
            or cand["confidence"] > current["confidence"]
        ):
            winner_by_id[cand["detection_id"]] = cand

    # ========================================================
    # 2차: 중복 제거 → 필터 → crop → 저장
    # ========================================================
    for cand in candidates:

        class_id = cand["class_id"]
        class_name = cand["class_name"]
        class_token = cand["class_token"]
        confidence = cand["confidence"]
        bbox = cand["bbox"]
        detection_id = cand["detection_id"]
        x1, y1, x2, y2 = bbox

        # ====================================================
        # filtered record
        #
        # image_id / source / detection_id 를 accepted 와 같은 형태로 남긴다.
        # 키가 어긋나면 "이 원본의 crop 이 왜 빠졌나" 를 나중에 조인해서
        # 볼 수 없다.
        # ====================================================
        def _filtered_record(reason, **extra):
            record = {
                "image_id": image_id,
                "source_path": image_path,
                "image_path": image_path,
                "detection_id": detection_id,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": bbox,
                "reason": reason,
            }

            if source is not None:
                record["source"] = source

            record.update(extra)

            return record

        # ====================================================
        # duplicate detection_id
        #
        # 승자가 아니면 crop 을 만들지 않는다. 통계에는 남긴다.
        # ====================================================
        winner = winner_by_id[detection_id]

        if winner is not cand:

            _draw_bbox(
                annotated,
                bbox,
                class_name,
                confidence,
                color=(0, 0, 255),
            )

            filtered_log.append(
                _filtered_record(
                    "duplicate_detection_id",
                    kept_confidence=winner["confidence"],
                )
            )

            continue

        # ====================================================
        # invalid bbox
        # ====================================================
        if (
            x2 <= x1
            or y2 <= y1
        ):

            filtered_log.append(
                _filtered_record("invalid_bbox")
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
                _filtered_record("empty_crop")
            )

            continue

        crop_h, crop_w = (
            cropped.shape[:2]
        )

        # ====================================================
        # 최소 crop 크기 필터
        #
        # person 과 object 는 기준이 다르다. 사람은 ReID 를 위해 최소
        # 해상도가 필요하지만, cell phone / tie 같은 객체는 정상 검출도
        # 작게 나온다. 같은 기준을 적용하면 유효한 객체가 버려진다.
        # ====================================================
        if class_name == "person":
            min_w = min_person_width
            min_h = min_person_height
            too_small_reason = "person_crop_too_small"
        else:
            min_w = min_object_width
            min_h = min_object_height
            too_small_reason = "object_crop_too_small"

        if (
            crop_w < min_w
            or crop_h < min_h
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
                _filtered_record(
                    too_small_reason,
                    width=crop_w,
                    height=crop_h,
                    min_width=min_w,
                    min_height=min_h,
                )
            )

            continue

        # ====================================================
        # crop 파일명
        #
        # detection_id 해시 기반. 순번(accepted_index) 을 쓰지 않는 이유는
        # _detection_token() 참조. 중복 제거 뒤라 이미지 안에서 유일하다.
        # ====================================================
        filename = (
            f"{prefix}"
            f"{source_token}_"
            f"{class_token}_"
            f"{_detection_token(detection_id)}.jpg"
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
                _filtered_record(
                    "save_failed",
                    width=crop_w,
                    height=crop_h,
                )
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
        #
        # image_id     논리 ID. 경로/OS 무관
        # source_path  원본을 실제로 열기 위한 물리 경로 (canonical)
        # image_path   기존 호출부 호환용 alias
        # ====================================================
        record = {
            "image_id": image_id,
            "source_path": image_path,
            "image_path": image_path,
            "crop_path": save_path,
            "path": save_path,
            "detection_id": detection_id,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": bbox,
            "width": crop_w,
            "height": crop_h,
        }

        if source is not None:
            record["source"] = source

        accepted_crops.append(record)

    # ========================================================
    # bbox 시각화 이미지 저장
    #
    # crop 과 같은 디렉터리에 쌓으면 8만 장 규모에서 crop 과 섞여
    # 디렉터리 탐색이 무거워진다. 하위 디렉터리로 분리한다.
    #
    # 파일명은 논리 image_id 에서 파생된 source_token 을 쓴다.
    # Path(image_path).stem 기반이면 다른 데이터셋에 같은 파일명이
    # 있을 때 서로 덮어쓴다.
    # ========================================================
    if save_annotated:

        detected_full_dir = (
            Path(output_dir)
            / "detected_full"
        )

        detected_full_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        detected_full_path = str(
            detected_full_dir
            / (
                f"{prefix}"
                f"{source_token}"
                f"_detected_full.jpg"
            )
        )

        if not cv2.imwrite(
            detected_full_path,
            annotated,
        ):
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
        "--image-id",
        default=None,
        help=(
            "Logical image id, e.g. coco_train2017/000000015496.jpg. "
            "Defaults to the bare filename."
        ),
    )

    parser.add_argument(
        "--source",
        default=None,
        help=(
            "Provenance label to record. Omitted from the output when "
            "not given."
        ),
    )

    parser.add_argument(
        "--min-person-width",
        type=int,
        default=MIN_PERSON_CROP_WIDTH,
    )

    parser.add_argument(
        "--min-person-height",
        type=int,
        default=MIN_PERSON_CROP_HEIGHT,
    )

    parser.add_argument(
        "--min-object-width",
        type=int,
        default=MIN_OBJECT_CROP_WIDTH,
    )

    parser.add_argument(
        "--min-object-height",
        type=int,
        default=MIN_OBJECT_CROP_HEIGHT,
    )

    parser.add_argument(
        "--no-annotated",
        action="store_true",
        help="Skip writing the bbox visualization image.",
    )

    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=CONF_THRESHOLD,
        help=f"RF-DETR detection confidence threshold, 0..1 (default: {CONF_THRESHOLD}).",
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
        image_id=args.image_id,
        source=args.source,
        min_person_width=args.min_person_width,
        min_person_height=args.min_person_height,
        min_object_width=args.min_object_width,
        min_object_height=args.min_object_height,
        save_annotated=not args.no_annotated,
        conf_threshold=args.conf_threshold,
    )

    print(
        f"Accepted crops: {len(crops)}"
    )

    print(
        f"Filtered crops: {len(filtered)}"
    )

    if crops:
        print(
            f"Sample detection_id: "
            f"{crops[0]['detection_id']}"
        )

    if not args.no_annotated:
        print(
            f"Annotated image dir: "
            f"{Path(args.output_dir) / 'detected_full'}"
        )