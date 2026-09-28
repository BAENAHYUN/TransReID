"""오프라인 테스트: 검출 임계값 플러밍, crop checkpoint 호환/트랜잭션, build_db 지문 규칙·재개 안내.

라이브 Qdrant/GPU/모델 로드 없음. rfdetr 는 가짜 모듈로 대체하고, 모든 파일은 임시 폴더에 만든다.
실행: .venv/Scripts/python.exe -m unittest tests.test_detect_build_config -v
"""
from __future__ import annotations

import importlib.util
import io
import json
import math
import os
import socket
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_fake_rfdetr():
    """detect_rf 는 최상위에서 rfdetr 를 import 한다. 모델 로드 없이 import 되도록 가짜를 넣는다."""
    if "rfdetr" in sys.modules and getattr(sys.modules["rfdetr"], "_fake", False):
        return
    rfdetr = types.ModuleType("rfdetr")
    rfdetr._fake = True

    class _FakeDetections:
        """supervision.Detections 흉내: len() 과 xyxy/class_id/confidence 속성만."""
        xyxy = []
        class_id = []
        confidence = []

        def __len__(self):
            return 0

    class RFDETRMedium:  # noqa: D401
        def __init__(self, *a, **k):
            self.calls = []

        def predict(self, image, threshold=None, **k):
            self.calls.append(threshold)
            return _FakeDetections()

    rfdetr.RFDETRMedium = RFDETRMedium
    assets = types.ModuleType("rfdetr.assets")
    coco = types.ModuleType("rfdetr.assets.coco_classes")
    coco.COCO_CLASSES = {1: "person", 2: "bicycle"}
    assets.coco_classes = coco
    rfdetr.assets = assets
    sys.modules["rfdetr"] = rfdetr
    sys.modules["rfdetr.assets"] = assets
    sys.modules["rfdetr.assets.coco_classes"] = coco


_install_fake_rfdetr()

from detect import detect_rf  # noqa: E402

_spec = importlib.util.spec_from_file_location("rfdetr_batch", ROOT / "detect" / "RF-DETR_batch.py")
rfb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rfb)


def _blocked(*a, **k):
    raise AssertionError("network access is blocked in tests")


class DetectThresholdTests(unittest.TestCase):
    def setUp(self):
        self._patches = [patch("socket.socket", _blocked), patch("socket.create_connection", _blocked)]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_validate_conf_threshold(self):
        self.assertEqual(detect_rf.validate_conf_threshold(0.3), 0.3)
        self.assertEqual(detect_rf.validate_conf_threshold("0.5"), 0.5)
        for bad in (1.5, -0.1, float("nan"), "x", None):
            with self.assertRaises(ValueError):
                detect_rf.validate_conf_threshold(bad)

    def test_detect_and_crop_uses_detector_threshold(self):
        import numpy as np
        import cv2
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "a.jpg"
            cv2.imwrite(str(img), np.zeros((64, 64, 3), dtype=np.uint8))
            default = detect_rf.load_detector()
            custom = detect_rf.load_detector(conf_threshold=0.3)
            with redirect_stdout(io.StringIO()):
                detect_rf.detect_and_crop(default, str(img), output_dir=td, target_classes=None,
                                          image_id="t/a.jpg", save_annotated=False)
                detect_rf.detect_and_crop(custom, str(img), output_dir=td, target_classes=None,
                                          image_id="t/a.jpg", save_annotated=False)
            # 플러그인은 cv2 BGR 프레임을 RGB 로 뒤집어 predict(threshold=...) 에 넘긴다
            self.assertEqual(default.model.calls, [detect_rf.CONF_THRESHOLD])
            self.assertEqual(custom.model.calls, [0.3])
            self.assertFalse(default.filter_forensic)
            with self.assertRaises(ValueError):
                detect_rf.load_detector(conf_threshold=2.0)

    def test_load_detector_spec_and_overrides(self):
        spec = {"module": "detect.detectors.rfdetr_detector", "class": "RFDETRDetector",
                "params": {"conf_threshold": 0.2, "filter_forensic": True}}
        det = detect_rf.load_detector(spec)
        self.assertEqual((det.conf_threshold, det.filter_forensic), (0.2, True))
        det2 = detect_rf.load_detector(spec, conf_threshold=0.5, filter_forensic=False)
        self.assertEqual((det2.conf_threshold, det2.filter_forensic), (0.5, False))
        self.assertEqual(spec["params"]["conf_threshold"], 0.2)  # 원본 spec 은 그대로

    def test_detect_and_crop_consumes_plugin_detections(self):
        import numpy as np
        import cv2
        from detect.base import BaseDetector, Detection

        class Fixed(BaseDetector):
            conf_threshold = 0.5

            def detect(self, frame, *, frame_idx, timestamp_sec=None):
                return [Detection(frame_idx, (2.0, 3.0, 40.0, 60.0), 0.9, 1, "person"),
                        Detection(frame_idx, (5.0, 5.0, 30.0, 30.0), 0.8, 3, "car"),
                        Detection(frame_idx, (2.4, 3.4, 40.2, 60.1), 0.7, 1, "person")]  # 같은 정수 bbox → 중복

        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "a.jpg"
            cv2.imwrite(str(img), np.full((100, 100, 3), 127, dtype=np.uint8))
            with redirect_stdout(io.StringIO()):
                crops, filtered = detect_rf.detect_and_crop(
                    Fixed(), str(img), output_dir=td, target_classes=["person"], image_id="t/a.jpg",
                    source="s", min_person_width=1, min_person_height=1, save_annotated=False)
            self.assertEqual([c["detection_id"] for c in crops], ["t/a.jpg#person#2_3_40_60"])
            self.assertEqual((crops[0]["class_id"], crops[0]["class_name"], crops[0]["confidence"]), (1, "person", 0.9))
            self.assertTrue(Path(crops[0]["crop_path"]).is_file())
            self.assertEqual([f["reason"] for f in filtered], ["duplicate_detection_id"])


class BatchCheckpointTests(unittest.TestCase):
    def _paths(self, td):
        paths = rfb._run_paths(Path(td))
        paths["ckpt_dir"].mkdir(parents=True, exist_ok=True)
        return paths

    def _config(self, thr=0.5, detector=None):
        return rfb._build_config("C:/in", None, "ds", "src", {"person": [25, 120], "object": [20, 20]}, thr, detector)

    def test_build_config_has_threshold_and_detector(self):
        cfg = self._config(0.4)
        self.assertEqual(cfg["conf_threshold"], 0.4)
        self.assertEqual(cfg["detector"], rfb.LEGACY_DETECTOR)
        yolo = rfb._detector_summary({"module": "detect.detectors.yolo26_detector", "class": "YOLO26Detector",
                                      "params": {"weights": "yolo26m.pt", "conf_threshold": 0.05, "filter_forensic": True}})
        self.assertEqual(yolo, {"module": "detect.detectors.yolo26_detector", "class": "YOLO26Detector",
                                "params": {"weights": "yolo26m.pt", "filter_forensic": False}})
        self.assertEqual(rfb._detector_summary(rfb.DEFAULT_DETECTOR_SPEC), rfb.LEGACY_DETECTOR)

    def test_load_state_legacy_without_threshold_or_detector_key(self):
        with tempfile.TemporaryDirectory() as td:
            paths = self._paths(td)
            legacy_cfg = {k: v for k, v in self._config().items() if k not in ("conf_threshold", "detector")}
            state = rfb._new_state(legacy_cfg)
            rfb._atomic_write_json(paths["state"], state)
            loaded = rfb._load_state(paths, self._config(0.5))
            self.assertIsNotNone(loaded)
            with self.assertRaises(SystemExit):
                rfb._load_state(paths, self._config(0.3))
            # 검출기가 다르면 resume 거부
            other = rfb._detector_summary({"module": "m", "class": "C", "params": {}})
            with self.assertRaises(SystemExit):
                rfb._load_state(paths, self._config(0.5, other))
            # 키는 있는데 null → 불일치로 드러나야 한다
            state["config"]["conf_threshold"] = None
            rfb._atomic_write_json(paths["state"], state)
            with self.assertRaises(SystemExit):
                rfb._load_state(paths, self._config(0.5))

    def test_resolve_detector_spec_priority(self):
        with tempfile.TemporaryDirectory() as td:
            y = Path(td) / "y.yaml"
            y.write_text("detector:\n  module: detect.detectors.yolo26_detector\n  class: YOLO26Detector\n"
                         "  params:\n    weights: yolo26m.pt\n    conf_threshold: 0.2\n", encoding="utf-8")
            p = Path(td) / "p.yaml"
            p.write_text("detector:\n  module: detect.detectors.rfdetr_detector\n  class: RFDETRDetector\n"
                         "  params:\n    conf_threshold: 0.5\n", encoding="utf-8")
            spec, origin = rfb._resolve_detector_spec(str(y), str(p))
            self.assertEqual((spec["class"], spec["params"]["weights"], origin), ("YOLO26Detector", "yolo26m.pt", "yaml:y.yaml"))
            spec, origin = rfb._resolve_detector_spec(None, str(p))
            self.assertEqual((spec["class"], origin), ("RFDETRDetector", "yaml:p.yaml"))
            spec, origin = rfb._resolve_detector_spec(None, None)
            self.assertEqual((spec, origin), (rfb.DEFAULT_DETECTOR_SPEC, "default(RF-DETR Medium)"))
            noblock = Path(td) / "n.yaml"
            noblock.write_text("retrievers: {}\n", encoding="utf-8")
            self.assertEqual(rfb._resolve_detector_spec(None, str(noblock))[1], "default(RF-DETR Medium)")
            with self.assertRaises(SystemExit):
                rfb._resolve_detector_spec(str(noblock), None)
            with self.assertRaises(SystemExit):
                rfb._resolve_detector_spec(str(Path(td) / "missing.yaml"), None)

    def test_real_crop_checkpoint_still_resumes(self):
        real = ROOT / "data" / "crops" / "checkpoint" / "state.json"
        if not real.is_file():
            self.skipTest("real checkpoint not present")
        with open(real, "r", encoding="utf-8") as f:
            saved = json.load(f)
        saved_cfg = dict(saved["config"])
        if "conf_threshold" not in saved_cfg:
            saved_cfg["conf_threshold"] = rfb.LEGACY_CONF_THRESHOLD
        if "detector" not in saved_cfg:
            saved_cfg["detector"] = rfb.LEGACY_DETECTOR
        current = rfb._build_config(
            saved["config"]["sample_dir"], saved["config"]["target_classes"],
            saved["config"]["dataset_id"], saved["config"]["source"],
            saved["config"]["min_crop_size"], rfb.LEGACY_CONF_THRESHOLD,
            rfb._detector_summary(rfb.DEFAULT_DETECTOR_SPEC),
        )
        self.assertEqual(saved_cfg, current)

    def test_resolve_conf_threshold_priority(self):
        with tempfile.TemporaryDirectory() as td:
            y = Path(td) / "p.yaml"
            y.write_text("detector:\n  params:\n    conf_threshold: 0.42\n", encoding="utf-8")
            self.assertEqual(rfb._resolve_conf_threshold(0.3, str(y)), (0.3, "cli"))
            self.assertEqual(rfb._resolve_conf_threshold(None, str(y))[0], 0.42)
            self.assertEqual(rfb._resolve_conf_threshold(None, None), (0.5, "default"))
            y2 = Path(td) / "q.yaml"
            y2.write_text("retrievers: {}\n", encoding="utf-8")
            self.assertEqual(rfb._resolve_conf_threshold(None, str(y2))[0], 0.5)
            with self.assertRaises(SystemExit):
                rfb._resolve_conf_threshold(None, str(Path(td) / "missing.yaml"))

    def test_partial_image_rollback(self):
        with tempfile.TemporaryDirectory() as td:
            paths = self._paths(td)
            handles = {k: open(paths[k], "a", encoding="utf-8") for k in rfb.DATA_KEYS}
            try:
                handles["crops"].write('{"a":1}\n'); handles["done"].write("img0.jpg\n")
                sizes = rfb._begin_image_write(paths, handles)
                handles["crops"].write('{"b":2}\n{"c":3}\n')
                handles["filtered"].write('{"f":1}\n')
                # done 을 쓰기 전에 "중단" → rollback
                with redirect_stdout(io.StringIO()):
                    rfb._rollback_partial(paths, handles, sizes)
                handles["crops"].write('{"d":4}\n')  # append 모드 핸들은 새 끝에 이어 쓴다
            finally:
                for h in handles.values():
                    h.close()
            self.assertEqual(paths["crops"].read_text(encoding="utf-8"), '{"a":1}\n{"d":4}\n')
            self.assertEqual(paths["filtered"].read_text(encoding="utf-8"), "")
            self.assertEqual(paths["done"].read_text(encoding="utf-8"), "img0.jpg\n")

    def test_leftover_logs_without_state_are_refused(self):
        with tempfile.TemporaryDirectory() as td:
            paths = self._paths(td)
            paths["crops"].write_text('{"x":1}\n', encoding="utf-8")
            sample = Path(td) / "imgs"; sample.mkdir()
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    rfb.run_batch(sample_dir=sample, output_dir=td, dataset_id="ds", source="s",
                                  target_classes=None, save_annotated=False, limit=None, fresh=False)
            self.assertIn("state.json", str(cm.exception))

    def test_mirror_rejects_broken_filtered_line(self):
        with tempfile.TemporaryDirectory() as td:
            paths = self._paths(td)
            paths["crops"].write_text('{"detection_id":"a#p#1_1_2_2","confidence":0.9,"crop_path":"x"}\n', encoding="utf-8")
            paths["filtered"].write_text('{"reason":"ok"}\n{"broken": \n', encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    rfb._write_stats_mirror(paths)


class BuildDbRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ingest import build_db  # heavy-ish but offline (no model load at import)
        cls.bd = build_db

    def test_explicit_rules_unchanged(self):
        for name in ("siglip2", "irra", "solider", "dinov2"):
            self.assertIs(self.bd._rule_for(name, {"anything": 1}), self.bd.FINGERPRINT_RULES[name])

    def test_generic_rule_classification(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "w.pth"; ckpt.write_bytes(b"weights")
            code = Path(td) / "repo"; code.mkdir()
            params = {"ckpt_path": str(ckpt), "code_root": str(code), "backbone": "vit",
                      "batch_size": 8, "missing": str(Path(td) / "nope.pth")}
            with redirect_stdout(io.StringIO()):
                rule = self.bd._rule_for("newmodel", params)
            self.assertTrue(rule.get("generic"))
            self.assertEqual(rule["file_params"], ("ckpt_path",))
            self.assertEqual(rule["audit_only_params"], ("code_root",))
            attrs = dict(rule["runtime_attrs"])
            self.assertIn("DIM", attrs)
            self.assertIn("backbone", attrs)
            self.assertNotIn("batch_size", attrs)          # PERF_ONLY_PARAMS 제외
            self.assertIn("missing", attrs)                 # 없는 경로는 문자열로 남는다

    def test_yaml_fingerprint_rule(self):
        with tempfile.TemporaryDirectory() as td:
            y = Path(td) / "p.yaml"
            y.write_text(
                "fingerprint:\n  newreid:\n    required_params: [ckpt_path]\n    file_params: [ckpt_path]\n"
                "    audit_only_params: [code_root]\n    runtime_attrs: [[backbone, backbone], [DIM, null]]\n",
                encoding="utf-8")
            with patch.object(self.bd, "CONFIG_PATH", y):
                self.bd._YAML_FINGERPRINT_CACHE.clear()
                rule = self.bd._rule_for("newreid", {})
            self.bd._YAML_FINGERPRINT_CACHE.clear()
            self.assertEqual(rule["source"], "yaml")
            self.assertEqual(rule["file_params"], ("ckpt_path",))
            self.assertEqual(rule["runtime_attrs"], (("backbone", "backbone"), ("DIM", None)))

    def test_resolve_param_path_mirrors_config(self):
        with tempfile.TemporaryDirectory() as td:
            y = Path(td) / "cfg" / "p.yaml"; y.parent.mkdir()
            with patch.object(self.bd, "CONFIG_PATH", y):
                self.assertEqual(self.bd._resolve_param_path("./w.pth"), (y.parent / "w.pth").resolve())
                self.assertEqual(self.bd._resolve_param_path("weights/w.pth"), (Path.cwd() / "weights/w.pth").resolve())
                self.assertEqual(self.bd._resolve_param_path(str(Path(td) / "abs.pth")), (Path(td) / "abs.pth").resolve())

    def test_runtime_generic_unverified_and_dim_required(self):
        bd = self.bd
        cfg = SimpleNamespace(retrievers={"newmodel": SimpleNamespace(dim=4, params={"backbone": "vit", "ghost": 1})})
        good = SimpleNamespace(DIM=4, backbone="vit")
        registry = SimpleNamespace(get=lambda name: good)
        with redirect_stdout(io.StringIO()):
            out = bd.runtime_retriever_fingerprint(cfg, registry)
        self.assertEqual(out["newmodel"]["DIM"], 4)
        self.assertEqual(out["newmodel"]["_unverified"], ["ghost"])
        bad_dim = SimpleNamespace(backbone="vit")
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                bd.runtime_retriever_fingerprint(cfg, SimpleNamespace(get=lambda n: bad_dim))
        mismatch = SimpleNamespace(DIM=4, backbone="swin")
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                bd.runtime_retriever_fingerprint(cfg, SimpleNamespace(get=lambda n: mismatch))

    def test_load_checkpoint_mismatch_message(self):
        bd = self.bd
        with tempfile.TemporaryDirectory() as td:
            ck = Path(td) / "ck"; ck.mkdir()
            state = {"run_info": {k: "old" for k in bd.CHECKPOINT_COMPAT_KEYS}, "next_index": 10,
                     "embedding_build_id": "emb_x", "audit": {"stats_path": "C:/old/filter_stats.json"}}
            (ck / "state.json").write_text(json.dumps(state), encoding="utf-8")
            run_info = {k: "new" for k in bd.CHECKPOINT_COMPAT_KEYS}
            with patch.object(bd, "CHECKPOINT_DIR", ck), patch.object(bd, "STATE_PATH", ck / "state.json"):
                with self.assertRaises(RuntimeError) as cm:
                    bd.load_checkpoint(run_info, {"stats_path": "C:/new/filter_stats.json"})
            msg = str(cm.exception)
            self.assertIn("--checkpoint-dir", msg)
            self.assertIn("emb_x", msg)
            self.assertIn("C:/old/filter_stats.json", msg)
            self.assertLess(msg.index("--checkpoint-dir"), msg.index("--fresh"))

    def test_assert_resume_target(self):
        bd = self.bd
        client = SimpleNamespace(collection_exists=lambda c: False, count=lambda c, exact=True: SimpleNamespace(count=0))
        with self.assertRaises(RuntimeError):
            bd._assert_resume_target(SimpleNamespace(client=client), "forensic_person", 1000)
        client2 = SimpleNamespace(collection_exists=lambda c: True, count=lambda c, exact=True: SimpleNamespace(count=0))
        with self.assertRaises(RuntimeError):
            bd._assert_resume_target(SimpleNamespace(client=client2), "forensic_person", 1000)
        client3 = SimpleNamespace(collection_exists=lambda c: True, count=lambda c, exact=True: SimpleNamespace(count=5))
        bd._assert_resume_target(SimpleNamespace(client=client3), "forensic_person", 1000)


if __name__ == "__main__":
    unittest.main()
