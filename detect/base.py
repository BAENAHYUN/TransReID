from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

# 포렌식 검색 대상 클래스 (COCO 이름). 검출기 플러그인의 filter_forensic 이 공통으로 쓴다.
FORENSIC_CLASSES = frozenset({
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
})


def coco_id_by_name() -> Dict[str, int]:
    """RF-DETR 이 쓰는 COCO 1-based id 표 (rfdetr.assets.coco_classes) 를 이름→id 로. rfdetr 가 없으면 빈 dict.
    다른 검출기(ultralytics 는 0-based) 도 같은 class_id 를 쓰게 해서 tracks.jsonl 이 검출기와 무관하게 유지되도록 한다."""
    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES
    except Exception:
        return {}
    return {str(name): int(cid) for cid, name in COCO_CLASSES.items()}


@dataclass
class Detection:
    frame_idx: int
    bbox: Tuple[float, float, float, float]  # xyxy
    confidence: float
    class_id: int
    class_name: str
    timestamp_sec: Optional[float] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Track:
    frame_idx: int
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    timestamp_sec: Optional[float] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class LongTrack:
    long_track_id: int
    member_track_ids: List[int]
    class_name: str
    start_frame: int
    end_frame: int
    records: List[Track]


class BaseDetector:
    def detect(
        self,
        frame,
        *,
        frame_idx: int,
        timestamp_sec: float | None = None,
    ) -> List[Detection]:
        raise NotImplementedError


class BaseTracker:
    def update(
        self,
        frame,
        detections: Sequence[Detection],
        frame_idx: int = -1,
    ) -> List[Track]:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class BaseStitcher:
    def stitch(self, tracks: Sequence[Track]) -> List[LongTrack]:
        raise NotImplementedError
