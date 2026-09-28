"""YOLO26 (ultralytics) 검출기 플러그인 — RF-DETR 과 같은 BaseDetector 계약.

yaml 의 detector: 블록에서 module/class 만 바꾸면 영상 파이프라인(detect.runner)과 검출 평가(eval/detect_eval_prw.py)
가 그대로 이 검출기를 쓴다:

  detector:
    module: detect.detectors.yolo26_detector
    class: YOLO26Detector
    params: {weights: yolo26m.pt, conf_threshold: 0.2, imgsz: 640}

* weights: 파일 경로 또는 이름(yolo26n/s/m/l/x.pt). 이름만 주면 weights/yolo/<이름> 을 쓰고, 없으면 ultralytics 가
  GitHub release 에서 그 자리에 내려받는다 (첫 실행 1회, 인터넷 필요).
* class_id 는 RF-DETR 과 같은 COCO 1-based id 로 맞춘다 (detect.base.coco_id_by_name). rfdetr 가 없으면 ultralytics 의
  0-based id 를 그대로 쓴다. class_name 은 어느 경우든 COCO 이름이다.
* 입력: detect.runner 는 cv2 BGR 프레임을 준다. ultralytics 도 ndarray 를 BGR 로 해석하므로 그대로 넘기고,
  input_color="rgb" 인 호출부만 뒤집는다.
* YOLO26 은 NMS-free(end-to-end) 라 iou 인자가 없다. 다른 YOLO 계열 가중치를 주면 ultralytics 기본 NMS 가 붙는다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from detect.base import FORENSIC_CLASSES, BaseDetector, Detection, coco_id_by_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "weights" / "yolo"


def resolve_weights(weights: str, weights_dir: Optional[Path] = None) -> str:
    """존재하는 경로면 그대로, 폴더 없는 이름이면 weights_dir/<이름> (없으면 ultralytics 가 거기로 내려받는다)."""
    weights_dir = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
    p = Path(str(weights).strip())
    if p.is_file():
        return str(p)
    if p.name == str(p):
        weights_dir.mkdir(parents=True, exist_ok=True)
        return str(weights_dir / p.name)
    return str(p)


class YOLO26Detector(BaseDetector):

    def __init__(
        self,
        weights: str = "yolo26m.pt",
        conf_threshold: float = 0.2,
        filter_forensic: bool = True,
        input_color: str = "bgr",
        imgsz: int = 640,
        device: Optional[str] = None,
        half: bool = True,
        classes: Optional[Sequence[str]] = None,
        max_det: int = 300,
    ):
        self.conf_threshold = float(conf_threshold)
        self.filter_forensic = bool(filter_forensic)
        if str(input_color).lower() not in ("bgr", "rgb"):
            raise ValueError(f"input_color 는 'bgr' 또는 'rgb': {input_color!r}")
        self.input_color = str(input_color).lower()
        self.imgsz = int(imgsz)
        self.device = device
        self.half = bool(half)
        self.max_det = int(max_det)
        self.weights = resolve_weights(weights)

        from ultralytics import YOLO

        self.model = YOLO(self.weights)
        names = getattr(self.model, "names", None) or {}
        self.names: Dict[int, str] = {int(k): str(v) for k, v in dict(names).items()}
        self.class_filter_ids: Optional[List[int]] = None
        if classes is not None:
            wanted = {str(c) for c in classes}
            self.class_filter_ids = sorted(i for i, n in self.names.items() if n in wanted)
            if not self.class_filter_ids:
                raise ValueError(f"classes 에 해당하는 모델 클래스가 없습니다: {sorted(wanted)}")
        self._coco_ids = coco_id_by_name()

    def detect(
        self,
        frame,
        *,
        frame_idx: int,
        timestamp_sec: float | None = None,
    ) -> List[Detection]:
        if (
            self.input_color == "rgb"
            and isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.shape[2] == 3
        ):
            frame = np.ascontiguousarray(frame[:, :, ::-1])

        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            # ultralytics 8.4: half= 는 deprecated, quantize=16 이 fp16 (None = fp32)
            quantize=16 if self.half else None,
            classes=self.class_filter_ids,
            max_det=self.max_det,
            verbose=False,
        )
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = np.asarray(boxes.xyxy.cpu().numpy(), dtype=np.float64).reshape(-1, 4)
        confs = np.asarray(boxes.conf.cpu().numpy(), dtype=np.float64).reshape(-1)
        cids = np.asarray(boxes.cls.cpu().numpy()).reshape(-1)

        out: List[Detection] = []
        for box, conf, raw_cid in zip(xyxy, confs, cids):
            conf = float(conf)
            if conf < self.conf_threshold:
                continue
            raw_cid = int(raw_cid)
            name = self.names.get(raw_cid, f"class_{raw_cid}")
            if self.filter_forensic and name not in FORENSIC_CLASSES:
                continue
            cid = self._coco_ids.get(name, raw_cid)
            out.append(
                Detection(
                    frame_idx=frame_idx,
                    bbox=tuple(float(x) for x in box),
                    confidence=conf,
                    class_id=cid,
                    class_name=name,
                    timestamp_sec=timestamp_sec,
                )
            )
        return out
