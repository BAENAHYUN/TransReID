from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from detect.base import BaseTracker, Detection, Track


class DeepOCSORTTracker(BaseTracker):
    """
    Deep OC-SORT adapter.
    지금은 비교 실험용으로 남겨두되, 기본 pipeline에서는 사용하지 않는다.
    """

    def __init__(
        self,
        reid_weights: str | None = None,
        device: str = "cuda:0",
        half: bool = True,
        **kwargs,
    ):
        try:
            from boxmot import DeepOcSort
        except Exception as e:
            raise ImportError(
                "Deep OC-SORT backend가 없습니다. `pip install boxmot` 후 다시 실행하세요."
            ) from e

        init_kwargs = {
            "device": device,
            "half": half,
        }
        init_kwargs.update(kwargs)

        if reid_weights:
            init_kwargs["reid_weights"] = reid_weights

        self.tracker = DeepOcSort(**init_kwargs)
        self._cid_to_name: Dict[int, str] = {}

    def update(
        self,
        frame,
        detections: Sequence[Detection],
        frame_idx: int = -1,
    ) -> List[Track]:
        for d in detections:
            self._cid_to_name[int(d.class_id)] = d.class_name

        if detections:
            dets = np.asarray(
                [[*d.bbox, d.confidence, d.class_id] for d in detections],
                dtype=np.float32,
            )
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        outputs = self.tracker.update(dets, frame)

        timestamp_sec = (
            detections[0].timestamp_sec
            if detections and detections[0].timestamp_sec is not None
            else None
        )

        result: List[Track] = []

        for row in outputs:
            if len(row) < 7:
                continue

            x1, y1, x2, y2 = map(float, row[:4])
            track_id = int(row[4])
            conf = float(row[5])
            class_id = int(row[6])

            class_name = self._cid_to_name.get(class_id, str(class_id))

            result.append(
                Track(
                    frame_idx=int(frame_idx),
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name,
                    timestamp_sec=timestamp_sec,
                )
            )

        return result

    def reset(self) -> None:
        reset_fn = getattr(self.tracker, "reset", None)
        if callable(reset_fn):
            reset_fn()

        self._cid_to_name.clear()
