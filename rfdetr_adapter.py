"""
RF-DETR 출력 -> Router 가 먹는 Detection 으로 변환.

detect_and_crop() 이 돌려주는 crop_results 형식:

    {"path": "...jpg", "bbox": [x1,y1,x2,y2], "confidence": 0.87,
     "class_name": "person",
     "image_id": "coco_train2017/000000015496.jpg",
     "source_path": "C:/datasets/coco/train2017/000000015496.jpg",
     "detection_id": "coco_train2017/000000015496.jpg#person#...",
     "source": "COCO"}

이 어댑터가 하는 일은 네 가지다.
  1) 위 dict 를 Detection 으로 변환 (필드 이름 매핑)
  2) crop 을 어떻게 넘길지 결정 — 저장된 파일 경로 vs 원본에서 다시 자른 메모리 배열
  3) 검출 클래스가 파이프라인에서 처리 가능한지 검증
  4) 상류 detection_id 를 존중하고, 예전 record 에만 결정적 fallback 을 제공

품질 필터는 여기에 없다
----------------------
confidence / 크기 필터링은 **detect_and_crop() 한 곳에서만** 한다.
그쪽이 filtered_log 를 남기므로 "이 crop 이 왜 빠졌나"를 한 군데서 추적할 수
있다. 어댑터에도 같은 필터를 두면 두 곳을 다 뒤져야 하고, 한쪽 기준만 바꿨을 때
원인을 찾기 어려워진다. 기준을 조정할 일이 생기면 detect_and_crop 을 고칠 것.

detection_id 계약
-----------------
현재 RF-DETR 경로는 detect_rf.py 가 논리 image_id + class + 정수 bbox 로
detection_id 를 만들어 준다. 이 어댑터는 그 값을 그대로 사용한다.
make_detection_id() 의 path / bbox 조합 로직은 예전 metadata 를 읽기 위한
하위 호환 fallback 이며, 새 crop_results 에서는 사용되지 않는 것이 정상이다.

색 공간 주의
-----------
detect_and_crop 은 cv2 로 읽어 **BGR** 배열을 다루고, cv2.imwrite 로 저장한다.
저장된 jpg 를 PIL 로 다시 읽으면 **RGB** 다. 즉:

    load_mode="path"    -> input_format="rgb"   (파일을 PIL 이 읽음)
    load_mode="memory"  -> input_format="bgr"   (cv2 배열을 그대로 넘김)

이 조합이 어긋나면 R 과 B 가 뒤바뀐 채로 임베딩된다. 에러가 안 나고
검색 품질만 조용히 나빠지므로, 어댑터가 짝을 강제한다.

사용
----
    import json
    from rfdetr_adapter import from_rfdetr, summarize

    crops = json.load(open("data/crops/filter_stats.json", encoding="utf-8"))["crops"]
    dets, fmt = from_rfdetr(crops)
    print(summarize(dets, cfg.person_labels))   # 적재 전 반드시 확인
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from router import Detection


logger = logging.getLogger(__name__)


# RF-DETR(COCO 80) 에 실제로 존재하는 클래스. 오타/없는 클래스를 조기에 잡는다.
COCO80_NAMES = frozenset({
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
})


# --------------------------------------------------------------------------- #
# 클래스 목록 검증
# --------------------------------------------------------------------------- #
def check_target_classes(
    classes: Iterable[str],
    raise_on_missing: bool = False,
) -> List[str]:
    """검출 대상 클래스가 COCO 80 에 실제로 있는지 확인한다.

    COCO 는 원래 91개 카테고리로 정의됐지만 그 중 11개(hat, shoe, eye glasses 등)는
    라벨이 없어 80개만 학습에 쓰인다. 없는 클래스를 target 에 넣으면 **에러 없이**
    영원히 0건이 나온다. 조기에 알려주기 위한 함수.

    FORENSIC_TARGET_CLASSES 를 확정할 때 한 번 돌려볼 것.

    반환: COCO 에 없는 클래스 이름 목록
    """
    missing = sorted({
        c
        for c in classes
        if c not in COCO80_NAMES
    })

    if missing:
        msg = (
            f"RF-DETR(COCO 80)에 없는 클래스가 target 에 있습니다: {missing}\n"
            f"  이 클래스들은 에러 없이 영원히 검출되지 않습니다.\n"
            f"  'hat', 'shoe', 'eye glasses' 는 COCO 91 정의에는 있지만 라벨이 없어 "
            f"80개 학습 대상에서 빠졌습니다."
        )

        if raise_on_missing:
            raise ValueError(msg)

        logger.warning(msg)

    return missing


# --------------------------------------------------------------------------- #
# bbox 검증
# --------------------------------------------------------------------------- #
def validate_bbox(
    r: Dict[str, Any],
) -> Tuple[float, float, float, float]:
    """
    RF-DETR bbox 를 검증하고 (x1, y1, x2, y2) float tuple 로 반환한다.

    검증:
      1) bbox 필드 존재
      2) 정확히 4개 좌표
      3) 모든 좌표가 숫자로 변환 가능
      4) NaN / Inf 없음
      5) xyxy 기준으로 양의 폭/높이

    품질 필터가 아니라 metadata 무결성 검증이다.
    잘못된 bbox 를 (0, 0, 0, 0) 같은 정상값처럼 Qdrant 에 저장하지 않도록
    적재 전에 즉시 실패시킨다.
    """

    hint = (
        r.get("detection_id")
        or r.get("image_id")
        or r.get("path")
        or r.get("source_path")
        or r.get("source_image")
        or r.get("image_path")
        or "<unknown>"
    )

    if "bbox" not in r or r["bbox"] is None:
        raise ValueError(
            f"bbox is missing: {hint}"
        )

    raw_bbox = r["bbox"]

    if isinstance(
        raw_bbox,
        (str, bytes, dict),
    ):
        raise ValueError(
            f"bbox must be a sequence of 4 numeric values: "
            f"{raw_bbox!r} ({hint})"
        )

    try:
        values = list(raw_bbox)
    except TypeError as exc:
        raise ValueError(
            f"bbox must be iterable: "
            f"{raw_bbox!r} ({hint})"
        ) from exc

    if len(values) != 4:
        raise ValueError(
            f"bbox must contain exactly 4 values: "
            f"{raw_bbox!r} ({hint})"
        )

    try:
        bbox = tuple(
            float(v)
            for v in values
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"bbox contains a non-numeric value: "
            f"{raw_bbox!r} ({hint})"
        ) from exc

    if not all(
        math.isfinite(v)
        for v in bbox
    ):
        raise ValueError(
            f"bbox contains NaN/Inf: "
            f"{bbox!r} ({hint})"
        )

    x1, y1, x2, y2 = bbox

    if (
        x2 <= x1
        or y2 <= y1
    ):
        raise ValueError(
            f"invalid xyxy bbox: "
            f"{bbox!r} ({hint})"
        )

    return bbox


# --------------------------------------------------------------------------- #
# 원본 이미지 물리 경로
# --------------------------------------------------------------------------- #
def _resolve_source_path(
    r: Dict[str, Any],
) -> str:
    """memory mode 에서 원본 이미지를 열 실제 경로를 반환한다.

    새 RF-DETR 계약의 canonical key 는 source_path 이다.
    source_image / image_path 는 예전 metadata 호환용 fallback 으로만 받는다.
    """
    value = (
        r.get("source_path")
        or r.get("source_image")
        or r.get("image_path")
    )

    if value is None or not str(value).strip():
        raise ValueError(
            "source image path is missing: "
            f"image_id={r.get('image_id')!r}, path={r.get('path')!r}"
        )

    return str(value)


# --------------------------------------------------------------------------- #
# detection_id
# --------------------------------------------------------------------------- #
def make_detection_id(
    r: Dict[str, Any],
    frame_idx: int = 0,
) -> str:
    """
    같은 detection 에 대해 항상 같은 문자열을 돌려준다.

    우선순위:
      1) r["detection_id"] — 새 RF-DETR 계약. 상류 값을 그대로 존중
      2) crop 파일 경로 — 예전 metadata 호환용 fallback
      3) image_id(또는 legacy source key) + frame_idx + label + bbox 합성

    2~3번은 legacy fallback 이다. 새 detect_rf.py 출력에서는 1번을 타는 것이
    정상이며, adapter 가 detection_id 의미를 다시 정의하지 않는다.
    """

    explicit = r.get(
        "detection_id"
    )

    if (
        explicit is not None
        and str(explicit).strip()
    ):
        return str(explicit).strip()

    path = r.get("path")

    if path:
        return str(path).replace(
            "\\",
            "/",
        )

    # path / explicit detection_id 가 없을 때만 bbox 기반 fallback을 사용한다.
    # 이 경우 잘못된 bbox가 ID에 포함되지 않도록 동일 검증 함수를 거친다.
    bbox = validate_bbox(r)

    coords = "_".join(
        str(
            int(
                round(float(v))
            )
        )
        for v in bbox
    )

    src_value = (
        r.get("image_id")
        or r.get("source_image")
        or r.get("image_path")
        or r.get("source_path")
        or ""
    )

    src = str(src_value).replace(
        "\\",
        "/",
    )

    label = str(
        r.get(
            "class_name",
            "unknown",
        )
    ).lower()

    fidx = int(
        r.get(
            "frame_idx",
            frame_idx,
        )
    )

    return (
        f"{src}#{fidx}#{label}#{coords}"
    )


# --------------------------------------------------------------------------- #
# 변환
# --------------------------------------------------------------------------- #
def from_rfdetr(
    crop_results: Sequence[Dict[str, Any]],
    load_mode: str = "path",
    frame_idx: int = 0,
    track_id_key: Optional[str] = None,
    keep_crop_path: bool = True,
) -> Tuple[List[Detection], str]:
    """crop_results -> (Detection 리스트, 임베더에 넘길 input_format).

    입력은 detect_and_crop() 이 이미 필터링을 마친 결과다. 여기서 추가로
    거르지 않는다.

    load_mode:
        "path"   저장된 crop jpg 경로를 그대로 넘긴다. 임베더가 PIL 로 읽는다 (RGB).
                 간단하지만 디스크 I/O 와 JPEG 재압축 손실이 있다.
        "memory" 원본 이미지를 다시 읽어 bbox 로 잘라 넘긴다 (BGR).
                 JPEG 손실이 없고 디스크 재읽기가 없다. 대신 원본이 그 자리에 있어야 한다.

    track_id_key: 나중에 트래커를 붙였을 때 track id 가 들어 있는 키 이름.
                  지정하면 Detection.track_id 로 옮겨 tracklet 집계가 동작한다.

    반환값의 두 번째 항목을 그대로 Router(input_format=...) 에 넘기면 색 공간이 맞는다.
    """

    if load_mode not in (
        "path",
        "memory",
    ):
        raise ValueError(
            "load_mode 는 'path' 또는 'memory' 여야 합니다."
        )

    input_format = (
        "rgb"
        if load_mode == "path"
        else "bgr"
    )

    dets: List[Detection] = []

    cache: Dict[str, Any] = {}

    if load_mode == "memory":
        import cv2

    seen_ids: Dict[str, int] = {}

    for r in crop_results:

        label = r.get(
            "class_name",
            "unknown",
        )

        # bbox 가 없거나 잘못된 형식이면 여기서 즉시 중단한다.
        # 기존처럼 missing bbox 를 (0, 0, 0, 0) 으로 조용히 바꾸지 않는다.
        bbox = validate_bbox(r)

        this_frame = int(
            r.get(
                "frame_idx",
                frame_idx,
            )
        )

        # 새 RF-DETR 계약에서는 image_id 가 논리 ID 이다.
        # 예전 metadata 에는 없을 수 있으므로 source 계열 key 를 fallback 으로 받는다.
        image_id_value = (
            r.get("image_id")
            or r.get("source_image")
            or r.get("image_path")
            or r.get("source_path")
            or ""
        )
        image_id = str(image_id_value)

        # ---- crop 확보 ----
        if load_mode == "path":

            crop = r["path"]

        else:

            src = _resolve_source_path(r)

            if src not in cache:

                img = cv2.imread(src)

                if img is None:
                    logger.warning(
                        "원본을 읽을 수 없어 건너뜁니다: %s",
                        src,
                    )
                    continue

                cache[src] = img

            img = cache[src]

            ih, iw = img.shape[:2]

            # bbox 가 이미지 밖으로 나가면 numpy 슬라이스가 조용히 빈 배열을
            # 만들거나 잘못된 영역을 준다. 명시적으로 클램프한다.
            x1, y1, x2, y2 = (
                int(round(float(v)))
                for v in bbox
            )

            x1, x2 = sorted((
                max(0, x1),
                min(iw, x2),
            ))

            y1, y2 = sorted((
                max(0, y1),
                min(ih, y2),
            ))

            crop = img[
                y1:y2,
                x1:x2,
            ]

            if crop.size == 0:
                logger.warning(
                    "빈 crop, 건너뜀: %s %s",
                    src,
                    bbox,
                )
                continue

        # ---- detection_id ----
        detection_id = make_detection_id(
            r,
            this_frame,
        )

        if detection_id in seen_ids:

            # 같은 배치 안에서 ID 가 겹치면 Qdrant 에서 뒤엣것이 앞엣것을
            # 덮어쓴다. 조용히 사라지므로 여기서 알린다.
            logger.warning(
                "detection_id 중복: %s (이 배치에서 %d번째). "
                "상류 detection_id / metadata 를 확인하세요.",
                detection_id,
                seen_ids[detection_id] + 1,
            )

        seen_ids[detection_id] = (
            seen_ids.get(
                detection_id,
                0,
            )
            + 1
        )

        # ---- extra payload ----
        extra: Dict[str, Any] = {
            # QdrantStore 가 detection_id -> crop_id 순으로 찾는다.
            # 이 값에서 Point ID(UUID5)가 파생된다.
            "crop_id": detection_id,
        }

        # 저장된 crop 경로
        # Qdrant payload 에는 OS 와 무관하도록 '/' 구분자로 저장한다.
        # 실제 이미지 로딩에 쓰는 crop 변수는 원래 경로를 그대로 유지한다.
        if (
            keep_crop_path
            and "path" in r
        ):
            extra["crop_path"] = (
                str(r["path"])
                .replace("\\", "/")
            )

        # bbox 좌표가 어느 공간 기준인지 명시.
        # 상류에서 지정하지 않았으면 일반 이미지/프레임 좌표로 본다.
        extra["bbox_space"] = str(
            r.get(
                "bbox_space",
                "frame",
            )
        )

        # 이미지/동영상 통합 Qdrant payload용 메타데이터.
        # 값이 실제 입력 record에 존재할 때만 전달한다.
        passthrough_keys = (
            "source",
            "video",
            "track_key",
            "person_idx",
            "split",
            "category",
            "timestamp",
        )

        for key in passthrough_keys:

            value = r.get(key)

            if value is not None:
                extra[key] = value

        # media_type은 상류에서 명시한 값을 우선 사용한다.
        # 없으면 영상 관련 메타데이터/track 존재 여부로 안전하게 추론한다.
        explicit_media_type = r.get(
            "media_type"
        )

        if explicit_media_type is not None:

            media_type = str(
                explicit_media_type
            ).strip().lower()

        else:

            has_video_metadata = any(
                r.get(key) is not None
                for key in (
                    "video",
                    "track_key",
                    "timestamp",
                )
            )

            has_track_metadata = (
                (
                    track_id_key is not None
                    and r.get(track_id_key) is not None
                )
                or r.get("track_id") is not None
            )

            media_type = (
                "video"
                if (
                    has_video_metadata
                    or has_track_metadata
                )
                else "image"
            )

        if media_type not in {
            "image",
            "video",
        }:
            raise ValueError(
                f"media_type은 'image' 또는 'video'여야 합니다: "
                f"{media_type!r}"
            )

        extra["media_type"] = media_type

        # 다중 인물 뭉침 판단 근거.
        # 자동 필터가 아니라 나중에 조사하기 위한 메타데이터.
        if "max_person_iou" in r:

            extra["max_person_iou"] = float(
                r["max_person_iou"]
            )

        # track_id는 명시적으로 전달된 key를 우선 사용하고,
        # 없으면 표준 이름인 "track_id"도 자동으로 받는다.
        resolved_track_id = None

        if track_id_key:

            resolved_track_id = r.get(
                track_id_key
            )

        elif r.get("track_id") is not None:

            resolved_track_id = r.get(
                "track_id"
            )

        if resolved_track_id is not None:

            resolved_track_id = int(
                resolved_track_id
            )

        dets.append(
            Detection(
                crop=crop,
                label=label,
                score=float(
                    r.get(
                        "confidence",
                        1.0,
                    )
                ),
                bbox=bbox,
                image_id=image_id,
                frame_idx=this_frame,
                track_id=resolved_track_id,
                extra=extra,
            )
        )

    return dets, input_format


# --------------------------------------------------------------------------- #
# 배치 버퍼 — GPU 를 놀리지 않기 위한 장치
# --------------------------------------------------------------------------- #
class DetectionBuffer:
    """이미지 여러 장의 detection 을 모아 한 번에 임베딩한다.

    사진 한 장에서 나오는 crop 은 보통 3~10개다. 그때마다 임베더를 호출하면
    GPU 가 대부분 놀고 파이썬 오버헤드만 커진다. 수백 개씩 모아서 넘기면
    처리량이 몇 배 달라진다.

        buf = DetectionBuffer(flush_size=256)
        for path in image_paths:
            crops, _ = detect_and_crop(model, path)
            dets, fmt = from_rfdetr(crops)
            for batch in buf.add(dets):
                handle(batch)
        for batch in buf.close():
            handle(batch)

    주의: load_mode="memory" 로 만든 Detection 은 crop 이 numpy 배열이다.
    flush_size 를 크게 잡으면 그만큼 메모리에 이미지가 쌓인다.
    """

    def __init__(
        self,
        flush_size: int = 256,
    ):

        if flush_size <= 0:
            raise ValueError(
                "flush_size 는 1 이상이어야 합니다."
            )

        self.flush_size = flush_size

        self._buf: List[
            Detection
        ] = []

    def add(
        self,
        detections: Sequence[Detection],
    ) -> Iterable[List[Detection]]:

        self._buf.extend(
            detections
        )

        while (
            len(self._buf)
            >= self.flush_size
        ):

            chunk, self._buf = (
                self._buf[
                    :self.flush_size
                ],
                self._buf[
                    self.flush_size:
                ],
            )

            yield chunk

    def close(
        self,
    ) -> Iterable[List[Detection]]:

        if self._buf:

            yield self._buf

            self._buf = []

    def __len__(
        self,
    ) -> int:

        return len(
            self._buf
        )


# --------------------------------------------------------------------------- #
def summarize(
    detections: Sequence[Detection],
    person_labels: Iterable[str],
) -> Dict[str, Any]:
    """적재 전 sanity check 용 요약.

    **person 이 0 이면 즉시 멈출 것.**
    person_labels 와 RF-DETR 라벨 문자열이 어긋난 것이다. 그 상태로 적재하면
    IRRA/SOLIDER 벡터가 하나도 생기지 않는데, 에러는 나지 않고 검색할 때가
    되어서야 발견된다. by_label 에는 'person' 이 멀쩡히 찍히므로 이 카운트를
    보지 않으면 놓치기 쉽다.

    missing_detection_id 가 0 이 아니면 crop_id 가 안 붙은 detection 이 있다는
    뜻이고, 그것들은 Qdrant 에서 bbox 기반 fallback ID 를 쓰게 된다.
    """

    from collections import Counter

    labels = Counter(
        d.label
        for d in detections
    )

    pl = {
        s.lower()
        for s in person_labels
    }

    n_person = sum(
        v
        for k, v in labels.items()
        if k.lower() in pl
    )

    n_missing_id = sum(
        1
        for d in detections
        if not (
            getattr(
                d,
                "extra",
                None,
            )
            or {}
        ).get("crop_id")
    )

    return {
        "total":
            len(detections),

        "person":
            n_person,

        "object":
            len(detections)
            - n_person,

        "missing_detection_id":
            n_missing_id,

        "by_label":
            dict(
                labels.most_common()
            ),
    }