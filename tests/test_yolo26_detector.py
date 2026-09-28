"""오프라인 테스트: detect/detectors/yolo26_detector.py — 가짜 ultralytics 로 필터·id 매핑·색공간·가중치 경로 확인 (모델 로드 없음)."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detect import base  # noqa: E402
from detect.detectors import yolo26_detector as y26  # noqa: E402

NAMES = {0: "person", 1: "bicycle", 2: "car", 39: "bottle", 56: "chair"}


class _Tensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _Boxes:
    def __init__(self, rows):
        rows = np.asarray(rows, dtype=float).reshape(-1, 6)
        self.xyxy = _Tensor(rows[:, :4])
        self.conf = _Tensor(rows[:, 4])
        self.cls = _Tensor(rows[:, 5])
        self._n = rows.shape[0]

    def __len__(self):
        return self._n


class _Result:
    def __init__(self, rows):
        self.boxes = _Boxes(rows)


class _FakeYOLO:
    rows = []
    instances = []

    def __init__(self, weights):
        self.weights = weights
        self.names = dict(NAMES)
        self.calls = []
        _FakeYOLO.instances.append(self)

    def predict(self, frame, **kw):
        self.calls.append(dict(frame=frame, **kw))
        return [_Result(_FakeYOLO.rows)]


def _install_fake_ultralytics():
    mod = types.ModuleType("ultralytics")
    mod.YOLO = _FakeYOLO
    mod._fake = True
    sys.modules["ultralytics"] = mod


class YOLO26DetectorTests(unittest.TestCase):
    def setUp(self):
        _install_fake_ultralytics()
        _FakeYOLO.instances = []
        _FakeYOLO.rows = [
            [0, 0, 10, 20, 0.9, 0],       # person
            [5, 5, 30, 30, 0.8, 2],       # car
            [1, 1, 3, 3, 0.7, 56],        # chair (forensic 아님)
            [2, 2, 4, 4, 0.1, 0],         # person, conf 미달
            [6, 6, 8, 9, 0.6, 39],        # bottle
        ]
        self.td = tempfile.TemporaryDirectory()
        self.weights_dir = Path(self.td.name) / "w"
        self.coco = {"person": 1, "bicycle": 2, "car": 3, "bottle": 44}

    def tearDown(self):
        self.td.cleanup()

    def make(self, **kw):
        with patch.object(y26, "coco_id_by_name", return_value=kw.pop("coco", self.coco)), \
                patch.object(y26, "DEFAULT_WEIGHTS_DIR", self.weights_dir):
            return y26.YOLO26Detector(**kw)

    def test_filter_forensic_and_threshold_and_coco_ids(self):
        det = self.make(weights="yolo26m.pt", conf_threshold=0.5)
        out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8), frame_idx=7, timestamp_sec=1.5)
        self.assertEqual([(d.class_name, d.class_id, d.confidence) for d in out],
                         [("person", 1, 0.9), ("car", 3, 0.8), ("bottle", 44, 0.6)])
        self.assertEqual(out[0].bbox, (0.0, 0.0, 10.0, 20.0))
        self.assertEqual((out[0].frame_idx, out[0].timestamp_sec), (7, 1.5))
        call = _FakeYOLO.instances[-1].calls[-1]
        self.assertEqual((call["conf"], call["imgsz"], call["verbose"], call["classes"]), (0.5, 640, False, None))

    def test_no_filter_keeps_chair_and_raw_ids_without_rfdetr(self):
        det = self.make(weights="yolo26m.pt", conf_threshold=0.5, filter_forensic=False, coco={})
        out = det.detect(np.zeros((4, 4, 3), dtype=np.uint8), frame_idx=0)
        self.assertEqual([(d.class_name, d.class_id) for d in out], [("person", 0), ("car", 2), ("chair", 56), ("bottle", 39)])

    def test_rgb_input_is_flipped_to_bgr_and_bgr_passthrough(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[..., 0] = 10
        frame[..., 2] = 30
        self.make(weights="yolo26m.pt", input_color="rgb").detect(frame, frame_idx=0)
        passed = _FakeYOLO.instances[-1].calls[-1]["frame"]
        self.assertEqual((int(passed[0, 0, 0]), int(passed[0, 0, 2])), (30, 10))
        self.make(weights="yolo26m.pt", input_color="bgr").detect(frame, frame_idx=0)
        passed = _FakeYOLO.instances[-1].calls[-1]["frame"]
        self.assertEqual((int(passed[0, 0, 0]), int(passed[0, 0, 2])), (10, 30))
        with self.assertRaises(ValueError):
            self.make(weights="yolo26m.pt", input_color="hsv")

    def test_classes_param_maps_names_to_model_ids(self):
        det = self.make(weights="yolo26m.pt", classes=["person", "bottle"])
        self.assertEqual(det.class_filter_ids, [0, 39])
        det.detect(np.zeros((2, 2, 3), dtype=np.uint8), frame_idx=0)
        self.assertEqual(_FakeYOLO.instances[-1].calls[-1]["classes"], [0, 39])
        with self.assertRaises(ValueError):
            self.make(weights="yolo26m.pt", classes=["unicorn"])

    def test_empty_results(self):
        _FakeYOLO.rows = []
        det = self.make(weights="yolo26m.pt")
        self.assertEqual(det.detect(np.zeros((2, 2, 3), dtype=np.uint8), frame_idx=0), [])

    def test_resolve_weights(self):
        existing = Path(self.td.name) / "custom.pt"
        existing.write_bytes(b"x")
        self.assertEqual(y26.resolve_weights(str(existing), self.weights_dir), str(existing))
        self.assertEqual(y26.resolve_weights("yolo26n.pt", self.weights_dir), str(self.weights_dir / "yolo26n.pt"))
        self.assertTrue(self.weights_dir.is_dir())
        nested = str(Path(self.td.name) / "nope" / "missing.pt")
        self.assertEqual(y26.resolve_weights(nested, self.weights_dir), nested)
        det = self.make(weights="yolo26s.pt")
        self.assertEqual(det.weights, str(self.weights_dir / "yolo26s.pt"))


class SharedContractTests(unittest.TestCase):
    def test_forensic_classes_shared_with_rfdetr_detector(self):
        self.assertIn("person", base.FORENSIC_CLASSES)
        self.assertEqual(len(base.FORENSIC_CLASSES), 14)
        try:
            from detect.detectors import rfdetr_detector
        except ImportError:  # rfdetr 미설치 환경
            return
        self.assertIs(rfdetr_detector._FORENSIC_CLASSES, base.FORENSIC_CLASSES)

    def test_coco_id_by_name_is_dict(self):
        ids = base.coco_id_by_name()
        self.assertIsInstance(ids, dict)
        if ids:
            self.assertEqual(ids.get("person"), 1)


if __name__ == "__main__":
    unittest.main()
