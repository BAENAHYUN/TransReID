import numpy as np

from detect.base import FORENSIC_CLASSES, BaseDetector, Detection

from rfdetr.assets.coco_classes import COCO_CLASSES


_FORENSIC_CLASSES = FORENSIC_CLASSES


def _class_name(class_id: int) -> str:
    return str(
        COCO_CLASSES.get(
            int(class_id),
            f"class_{class_id}"
        )
    )


class RFDETRDetector(BaseDetector):

    def __init__(
        self,
        conf_threshold: float = 0.5,
        filter_forensic: bool = True,
        input_color: str = "bgr",
    ):
        self.conf_threshold = float(conf_threshold)
        self.filter_forensic = bool(filter_forensic)
        # detect/runner.py 는 cv2.VideoCapture 프레임(BGR)을 넘긴다. RF-DETR predict 는
        # RGB ndarray/PIL 을 기대하므로(rfdetr/detr.py: "images should be in RGB channel
        # order") detect() 에서 뒤집는다. 이미 RGB 인 프레임을 주는 호출부는 "rgb".
        if str(input_color).lower() not in ("bgr", "rgb"):
            raise ValueError(f"input_color 는 'bgr' 또는 'rgb': {input_color!r}")
        self.input_color = str(input_color).lower()

        from rfdetr import RFDETRMedium

        self.model = RFDETRMedium()

        try:
            self.model.inference(dtype="float16")
        except Exception:
            pass

    def detect(
        self,
        frame,
        *,
        frame_idx: int,
        timestamp_sec: float | None = None,
    ):
        if (
            self.input_color == "bgr"
            and isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.shape[2] == 3
        ):
            # BGR → RGB. 이전에는 BGR 그대로 들어가 이미지 경로 검출(detect_rf.py,
            # PIL/RGB)과 색공간이 달랐다.
            frame = np.ascontiguousarray(frame[:, :, ::-1])

        r = self.model.predict(
            frame,
            threshold=self.conf_threshold
        )

        xyxy = getattr(r, "xyxy", None)
        cids = getattr(r, "class_id", None)
        confs = getattr(r, "confidence", None)

        xyxy = [] if xyxy is None else list(xyxy)
        cids = [] if cids is None else list(cids)
        confs = [] if confs is None else list(confs)

        out = []

        for box, cid, conf in zip(xyxy, cids, confs):

            conf = float(conf)

            if conf < self.conf_threshold:
                continue

            cid = int(cid)
            name = _class_name(cid)

            if (
                self.filter_forensic
                and name not in _FORENSIC_CLASSES
            ):
                continue

            box = tuple(float(x) for x in box)

            if len(box) != 4:
                continue

            out.append(
                Detection(
                    frame_idx=frame_idx,
                    bbox=box,
                    confidence=conf,
                    class_id=cid,
                    class_name=name,
                    timestamp_sec=timestamp_sec,
                )
            )

        return out