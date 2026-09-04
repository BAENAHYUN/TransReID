"""
RF-DETR 출력 -> Router 가 먹는 Detection 으로 변환.

detect_and_crop() 이 돌려주는 crop_results 형식:

    {"path": "...jpg", "bbox": [x1,y1,x2,y2], "confidence": 0.87,
     "class_name": "person", "source_image": "...jpg",
     "max_person_iou": 0.12}          # person 에만

이 어댑터가 하는 일은 네 가지다.
  1) 위 dict 를 Detection 으로 변환 (필드 이름 매핑)
  2) crop 을 어떻게 넘길지 결정 — 저장된 파일 경로 vs 원본에서 다시 자른 메모리 배열
  3) 검출 클래스가 파이프라인에서 처리 가능한지 검증
  4) 결정적(deterministic) detection_id 부여

품질 필터는 여기에 없다
----------------------
confidence / 크기 필터링은 **detect_and_crop() 한 곳에서만** 한다.
그쪽이 filtered_log 를 남기므로 "이 crop 이 왜 빠졌나"를 한 군데서 추적할 수
있다. 어댑터에도 같은 필터를 두면 두 곳을 다 뒤져야 하고, 한쪽 기준만 바꿨을 때
원인을 찾기 어려워진다. 기준을 조정할 일이 생기면 detect_and_crop 을 고칠 것.

detection_id 를 왜 여기서 만드는가
---------------------------------
QdrantStore 는 Point ID 를 detection_id 로부터 UUID5 로 만든다. detection_id 가
없으면 bbox 기반 fallback 으로 내려가는데, 그 키는

    image_id | frame_idx | label | bbox(소수점 6자리)

라서 RF-DETR 재실행이나 버전 변경으로 bbox 가 100.000001 처럼 미세하게만
달라져도 **별개 point** 가 된다. 1000만 규모에서 중복이 쌓이면 나중에 정리하기
매우 어렵다. 그래서 적재 전 단계인 여기서 안정적인 ID 를 붙인다.

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
    missing = sorted({c for c in classes if c not in COCO80_NAMES})
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
# detection_id
# --------------------------------------------------------------------------- #
def make_detection_id(r: Dict[str, Any], frame_idx: int = 0) -> str:
    """
    같은 detection 에 대해 항상 같은 문자열을 돌려준다.

    우선순위:
      1) r["detection_id"] — 상류에서 이미 붙였으면 그대로 존중
      2) crop 파일 경로 — detect_and_crop 이 만든 파일명은 이미 고유하다
      3) source_image + frame_idx + label + bbox(정수 픽셀) 합성

    3번의 bbox 는 **정수로 반올림**한다. QdrantStore 의 bbox fallback 이
    소수점 6자리를 쓰는 것과 달리, 픽셀 단위 반올림은 부동소수 오차에
    흔들리지 않아 재실행 안정성이 훨씬 낫다.

    경로 구분자는 '/' 로 통일한다. Windows 에서 만든 데이터를 Linux 서버에
    올려도 같은 ID 가 나와야 중복 적재를 피할 수 있다.
    """
    explicit = r.get("detection_id")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()

    path = r.get("path")
    if path:
        return str(path).replace("\\", "/")

    bbox = r.get("bbox", (0, 0, 0, 0))
    coords = "_".join(str(int(round(float(v)))) for v in bbox)
    src = str(r.get("source_image", "")).replace("\\", "/")
    label = str(r.get("class_name", "unknown")).lower()
    fidx = int(r.get("frame_idx", frame_idx))

    return f"{src}#{fidx}#{label}#{coords}"


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
    if load_mode not in ("path", "memory"):
        raise ValueError("load_mode 는 'path' 또는 'memory' 여야 합니다.")

    input_format = "rgb" if load_mode == "path" else "bgr"
    dets: List[Detection] = []

    cache: Dict[str, Any] = {}
    if load_mode == "memory":
        import cv2

    seen_ids: Dict[str, int] = {}

    for r in crop_results:
        label = r.get("class_name", "unknown")
        bbox = tuple(r.get("bbox", (0, 0, 0, 0)))
        this_frame = int(r.get("frame_idx", frame_idx))

        # ---- crop 확보 ----
        if load_mode == "path":
            crop = r["path"]
        else:
            src = r["source_image"]
            if src not in cache:
                img = cv2.imread(src)
                if img is None:
                    logger.warning("원본을 읽을 수 없어 건너뜁니다: %s", src)
                    continue
                cache[src] = img

            img = cache[src]
            ih, iw = img.shape[:2]

            # bbox 가 이미지 밖으로 나가면 numpy 슬라이스가 조용히 빈 배열을
            # 만들거나 잘못된 영역을 준다. 명시적으로 클램프한다.
            x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
            x1, x2 = sorted((max(0, x1), min(iw, x2)))
            y1, y2 = sorted((max(0, y1), min(ih, y2)))

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                logger.warning("빈 crop, 건너뜀: %s %s", src, bbox)
                continue

        # ---- detection_id ----
        detection_id = make_detection_id(r, this_frame)

        if detection_id in seen_ids:
            # 같은 배치 안에서 ID 가 겹치면 Qdrant 에서 뒤엣것이 앞엣것을
            # 덮어쓴다. 조용히 사라지므로 여기서 알린다.
            logger.warning(
                "detection_id 중복: %s (이 배치에서 %d번째). "
                "crop 파일명이 고유한지 확인하세요.",
                detection_id, seen_ids[detection_id] + 1,
            )
        seen_ids[detection_id] = seen_ids.get(detection_id, 0) + 1

        # ---- extra payload ----
        extra: Dict[str, Any] = {
            # QdrantStore 가 detection_id -> crop_id 순으로 찾는다.
            # 이 값에서 Point ID(UUID5)가 파생된다.
            "crop_id": detection_id,
        }
        if keep_crop_path and "path" in r:
            extra["crop_path"] = r["path"]
        # 다중 인물 뭉침 판단 근거. 자동 필터가 아니라 나중에 조사하기 위한 메타데이터.
        if "max_person_iou" in r:
            extra["max_person_iou"] = float(r["max_person_iou"])

        dets.append(Detection(
            crop=crop,
            label=label,
            score=float(r.get("confidence", 1.0)),
            bbox=bbox,
            image_id=r.get("source_image", ""),
            frame_idx=this_frame,
            track_id=r.get(track_id_key) if track_id_key else None,
            extra=extra,
        ))

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

    def __init__(self, flush_size: int = 256):
        if flush_size <= 0:
            raise ValueError("flush_size 는 1 이상이어야 합니다.")
        self.flush_size = flush_size
        self._buf: List[Detection] = []

    def add(self, detections: Sequence[Detection]) -> Iterable[List[Detection]]:
        self._buf.extend(detections)
        while len(self._buf) >= self.flush_size:
            chunk, self._buf = (
                self._buf[:self.flush_size],
                self._buf[self.flush_size:],
            )
            yield chunk

    def close(self) -> Iterable[List[Detection]]:
        if self._buf:
            yield self._buf
            self._buf = []

    def __len__(self) -> int:
        return len(self._buf)


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

    labels = Counter(d.label for d in detections)
    pl = {s.lower() for s in person_labels}
    n_person = sum(v for k, v in labels.items() if k.lower() in pl)

    n_missing_id = sum(
        1 for d in detections
        if not (getattr(d, "extra", None) or {}).get("crop_id")
    )

    return {
        "total": len(detections),
        "person": n_person,
        "object": len(detections) - n_person,
        "missing_detection_id": n_missing_id,
        "by_label": dict(labels.most_common()),
    }