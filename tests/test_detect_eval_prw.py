"""오프라인 테스트: eval/detect_eval_prw.py 의 매칭·AP·동작점·크기별 재현율과 score 모드 end-to-end (모델·Qdrant·네트워크 없음)."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import detect_eval_prw as de  # noqa: E402


class GeometryTests(unittest.TestCase):
    def test_iou_matrix_basic(self):
        a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
        b = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=float)
        m = de.iou_matrix(a, b)
        self.assertEqual(m.shape, (2, 2))
        self.assertAlmostEqual(m[0, 0], 1.0)
        self.assertAlmostEqual(m[0, 1], 25 / 175)
        self.assertAlmostEqual(m[1, 0], 0.0)

    def test_iou_matrix_empty(self):
        self.assertEqual(de.iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape, (0, 3))
        self.assertEqual(de.iou_matrix(np.zeros((2, 4)), np.zeros((0, 4))).shape, (2, 0))

    def test_xywh_to_xyxy(self):
        out = de.xywh_to_xyxy(np.array([[10, 20, 30, 40]]))
        np.testing.assert_allclose(out, [[10, 20, 40, 60]])

    def test_size_bucket_boundaries(self):
        self.assertEqual(de.size_bucket(49.9), "<50")
        self.assertEqual(de.size_bucket(50), "50–74")
        self.assertEqual(de.size_bucket(74.9), "50–74")
        self.assertEqual(de.size_bucket(75), "75–119")
        self.assertEqual(de.size_bucket(120), "120–199")
        self.assertEqual(de.size_bucket(200), "200+")
        self.assertEqual(de.size_bucket(5000), "200+")


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.gts = np.array([[0, 0, 100, 200], [300, 0, 400, 200]], dtype=float)  # xyxy

    def test_greedy_by_confidence_with_duplicate_and_background(self):
        dets = np.array([
            [2, 2, 100, 200, 0.6],      # GT0 와 IoU 높음, conf 낮음
            [0, 0, 100, 200, 0.9],      # GT0 정확, conf 최고 → TP, 위 검출은 중복 FP
            [300, 0, 400, 200, 0.8],    # GT1 → TP
            [600, 600, 650, 700, 0.7],  # 배경 FP
        ])
        m = de.match_frame(dets, self.gts, 0.5)
        np.testing.assert_allclose(m["conf"], [0.9, 0.8, 0.7, 0.6])
        self.assertEqual(m["tp"].tolist(), [True, True, False, False])
        self.assertEqual(m["fp_kind"].tolist(), [0, 0, 2, 1])
        self.assertEqual(m["det_gt"].tolist(), [0, 1, -1, -1])
        np.testing.assert_allclose(m["gt_conf"], [0.9, 0.8])

    def test_no_detections_and_no_gt(self):
        m = de.match_frame(np.zeros((0, 5)), self.gts, 0.5)
        self.assertEqual(m["tp"].shape, (0,))
        self.assertTrue(np.all(np.isnan(m["gt_conf"])))
        m2 = de.match_frame(np.array([[0, 0, 10, 10, 0.5]]), np.zeros((0, 4)), 0.5)
        self.assertEqual(m2["fp_kind"].tolist(), [2])

    def test_iou_threshold_respected(self):
        dets = np.array([[0, 0, 100, 100, 0.9]])  # GT0 와 IoU 0.5 정확
        self.assertTrue(de.match_frame(dets, self.gts, 0.5)["tp"][0])
        self.assertFalse(de.match_frame(dets, self.gts, 0.51)["tp"][0])


class MetricTests(unittest.TestCase):
    def test_ap_perfect_and_empty(self):
        conf = np.array([0.9, 0.8])
        tp = np.array([True, True])
        _, _, ap = de.pr_curve(conf, tp, 2)
        self.assertAlmostEqual(ap, 1.0)
        self.assertEqual(de.pr_curve(np.zeros(0), np.zeros(0, dtype=bool), 2)[2], 0.0)
        self.assertEqual(de.pr_curve(conf, tp, 0)[2], 0.0)

    def test_ap_all_point_interpolation(self):
        # TP(0.9) FP(0.8) TP(0.7), GT 2개: recall 0.5@P1.0, 1.0@P2/3 → AP = 0.5*1 + 0.5*(2/3)
        conf = np.array([0.7, 0.9, 0.8])
        tp = np.array([True, True, False])
        recall, precision, ap = de.pr_curve(conf, tp, 2)
        np.testing.assert_allclose(recall, [0.5, 0.5, 1.0])
        self.assertAlmostEqual(ap, 0.5 + 0.5 * (2 / 3))

    def test_operating_point_counts(self):
        conf = np.array([0.9, 0.8, 0.7, 0.6])
        tp = np.array([True, True, False, False])
        kind = np.array([0, 0, 2, 1])
        o = de.operating_point(conf, tp, kind, n_gt=3, thr=0.65, n_frames=2)
        self.assertEqual((o["dets"], o["tp"], o["fp"], o["fp_duplicate"], o["fp_background"], o["fn"]), (3, 2, 1, 0, 1, 1))
        self.assertAlmostEqual(o["precision"], 2 / 3)
        self.assertAlmostEqual(o["recall"], 2 / 3)
        self.assertAlmostEqual(o["dets_per_frame"], 1.5)
        empty = de.operating_point(conf, tp, kind, n_gt=3, thr=0.95, n_frames=2)
        self.assertIsNone(empty["precision"])
        self.assertEqual(empty["fn"], 3)

    def test_recall_by_size(self):
        heights = np.array([40.0, 60.0, 100.0, 150.0, 300.0])
        gt_conf = np.array([np.nan, 0.3, 0.9, 0.6, 0.55])
        r = de.recall_by_size(heights, gt_conf, 0.5)
        self.assertEqual(r["<50"], dict(gt=1, recalled=0, recall=0.0, recalled_any=0, recall_any=0.0))
        self.assertEqual(r["50–74"]["recall"], 0.0)
        self.assertEqual(r["50–74"]["recall_any"], 1.0)
        self.assertEqual(r["75–119"]["recall"], 1.0)
        self.assertEqual(r["200+"]["recalled"], 1)

    def test_evaluate_method_end_to_end(self):
        gt = {"f1": np.array([[0, 0, 100, 200], [300, 0, 100, 200]], dtype=float),   # xywh
              "f2": np.array([[10, 10, 50, 80]], dtype=float)}
        dets = {"f1": np.array([[0, 0, 100, 200, 0.9], [300, 0, 400, 200, 0.4], [700, 700, 720, 740, 0.8]]),
                "f2": np.zeros((0, 5))}
        res = de.evaluate_method(dets, gt, ["f1", "f2"], 0.5, 0.5, [0.3])
        self.assertEqual(res["gt_boxes"], 3)
        self.assertEqual(res["detections"], 3)
        self.assertAlmostEqual(res["max_recall"], 2 / 3)
        self.assertAlmostEqual(res["operating"]["recall"], 1 / 3)
        self.assertEqual(res["operating"]["fp_background"], 1)
        self.assertEqual(res["per_frame"]["f1"]["fn"], 1)
        self.assertEqual(res["per_frame"]["f2"]["fn"], 1)
        self.assertEqual([o["threshold"] for o in res["operating_points"]], [0.3, 0.5])
        self.assertEqual(len(res["ap_per_iou"]), 10)
        self.assertTrue(0 < res["ap50"] <= 1)


class IOTests(unittest.TestCase):
    def test_detections_roundtrip_and_meta_rewrite(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "d.jsonl"
            rows = [dict(frame="a", width=10, height=10, latency_ms=1.0, dets=[[0, 0, 5, 5, 0.9, 1, "person"]]),
                    dict(frame="b", width=10, height=10, latency_ms=1.0, dets=[])]
            self.assertEqual(de.write_detections(path, dict(producer="t"), rows), 2)
            meta, dets, extra = de.read_detections(path)
            self.assertEqual(meta["producer"], "t")
            self.assertEqual(dets["a"].shape, (1, 5))
            self.assertEqual(dets["b"].shape, (0, 5))
            self.assertEqual(extra["a"]["width"], 10)
            de.rewrite_meta(path, dict(producer="t", stats=dict(fps=3.0)))
            meta2, dets2, _ = de.read_detections(path)
            self.assertEqual(meta2["stats"]["fps"], 3.0)
            self.assertEqual(set(dets2), {"a", "b"})

    def test_parse_named_and_param(self):
        self.assertEqual(de.parse_named([["a=x.jsonl", "b=y.jsonl"], "c=z"]), {"a": Path("x.jsonl"), "b": Path("y.jsonl"), "c": Path("z")})
        self.assertEqual(de.parse_named([[""]]), {})
        with self.assertRaises(ValueError):
            de.parse_named(["nope"])
        self.assertEqual(de.parse_param("conf_threshold=0.3"), ("conf_threshold", 0.3))
        self.assertEqual(de.parse_param("device=cuda:0"), ("device", "cuda:0"))
        self.assertEqual(de.parse_param("flag=true"), ("flag", True))

    def test_frame_from_image_id(self):
        self.assertEqual(de.frame_from_image_id("PRW/c5s2_116999.jpg"), "c5s2_116999")
        self.assertEqual(de.frame_from_image_id("C:\\x\\c1s1_1.jpg"), "c1s1_1")
        self.assertIsNone(de.frame_from_image_id("no_ext"))

    def test_resolve_detector_spec_overrides(self):
        with tempfile.TemporaryDirectory() as td:
            y = Path(td) / "det.yaml"
            y.write_text("detector:\n  module: m\n  class: C\n  params:\n    conf_threshold: 0.2\n    x: 1\n", encoding="utf-8")
            spec = de.resolve_detector_spec(str(y), None, None, ["x=2", "s=abc"], 0.05)
            self.assertEqual((spec["module"], spec["class"]), ("m", "C"))
            self.assertEqual(spec["params"], dict(conf_threshold=0.05, x=2, s="abc"))
            self.assertEqual(spec["config_conf_threshold"], 0.2)
            spec2 = de.resolve_detector_spec(str(y), "other.mod", "Other", [], None)
            self.assertEqual((spec2["module"], spec2["class"], spec2["params"]["conf_threshold"]), ("other.mod", "Other", 0.2))
            with self.assertRaises(ValueError):
                de.resolve_detector_spec(None, None, None, [], None)


class ScoreModeTests(unittest.TestCase):
    """가짜 PRW 폴더(annotations/*.jpg.mat) + detections.jsonl 두 개로 score 모드 전체 경로."""

    def setUp(self):
        import scipy.io
        self.td = tempfile.TemporaryDirectory()
        root = Path(self.td.name)
        self.data_root = root / "PRW"
        ann = self.data_root / "annotations"
        ann.mkdir(parents=True)
        (self.data_root / "frames").mkdir()
        scipy.io.savemat(str(ann / "c1s1_000001.jpg.mat"), {"box_new": np.array([[1, 0, 0, 100, 200], [-2, 300, 0, 100, 60]], dtype=float)})
        scipy.io.savemat(str(ann / "c1s1_000002.jpg.mat"), {"box_new": np.array([[2, 10, 10, 50, 130]], dtype=float)})
        good = root / "good.jsonl"
        de.write_detections(good, dict(producer="t", kind="run", detector=dict(module="m", **{"class": "C"}, params=dict(conf_threshold=0.05), config_conf_threshold=0.2), stats=dict(fps=10.0, latency_ms_mean=100.0)), [
            dict(frame="c1s1_000001", width=1920, height=1080, latency_ms=1, dets=[[0, 0, 100, 200, 0.95, 1, "person"], [300, 0, 400, 60, 0.3, 1, "person"]]),
            dict(frame="c1s1_000002", width=1920, height=1080, latency_ms=1, dets=[[10, 10, 60, 140, 0.7, 1, "person"], [500, 500, 520, 540, 0.6, 1, "person"]]),
        ])
        bad = root / "bad.jsonl"
        de.write_detections(bad, dict(producer="t", kind="import-qdrant", detector=dict(module="qdrant", **{"class": "forensic_person"}, params={})), [
            dict(frame="c1s1_000001", dets=[[0, 0, 100, 200, 0.9, -1, "person"]]),
        ])
        self.good, self.bad, self.out = good, bad, root / "out"

    def tearDown(self):
        self.td.cleanup()

    def run_score(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            summary = de.main([*argv, "--no-ledger"])
        return summary, out.getvalue()

    def test_score_two_methods_intersection_and_outputs(self):
        summary, text = self.run_score(["--mode", "score", "--method", f"good={self.good}", f"db={self.bad}", "--data-root", str(self.data_root),
                                        "--output-dir", str(self.out), "--no-images"])
        self.assertIn("RESULT_SUMMARY:", text)
        self.assertIn("RESULT_HTML:", text)
        self.assertEqual(summary["config"]["frames"], 1)  # 교집합
        self.assertEqual([w["code"] for w in summary["warnings"]], ["FRAME_SET_MISMATCH"])
        self.assertEqual(summary["gt"]["boxes"], 2)
        self.assertEqual(summary["gt"]["boxes_unlabeled_pid"], 1)
        good = summary["results"]["good"]
        self.assertEqual(good["operating_threshold"], 0.2)  # yaml 의 conf_threshold
        self.assertAlmostEqual(good["operating"]["recall"], 1.0)
        self.assertEqual(summary["results"]["db"]["operating_threshold"], 0.5)
        self.assertAlmostEqual(summary["results"]["db"]["operating"]["recall"], 0.5)
        self.assertTrue((self.out / "detect_eval_prw.html").is_file())
        self.assertTrue((self.out / "summary.csv").is_file())
        report = json.loads((self.out / "detect_eval_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["producer"], "detect_eval_prw")
        self.assertNotIn("per_frame", report["results"]["good"])

    def test_score_single_method_all_frames_and_size_buckets(self):
        summary, _ = self.run_score(["--mode", "score", "--method", f"good={self.good}", "--data-root", str(self.data_root),
                                     "--output-dir", str(self.out), "--no-images", "--operating-threshold", "0.5"])
        res = summary["results"]["good"]
        self.assertEqual(res["frames"], 2)
        self.assertEqual(res["gt_boxes"], 3)
        self.assertEqual(res["recall_by_size"]["50–74"], dict(gt=1, recalled=0, recall=0.0, recalled_any=1, recall_any=1.0))
        self.assertEqual(res["recall_by_size"]["120–199"]["recall"], 1.0)
        self.assertEqual(res["operating"]["fp_background"], 1)
        self.assertEqual(summary["warnings"], [])

    def test_argument_errors(self):
        for argv in (["--mode", "score"], ["--mode", "score", "--method", "x", "--data-root", str(self.data_root)],
                     ["--mode", "run", "--detector-config", "", "--data-root", str(self.data_root)],
                     ["--mode", "score", "--method", f"g={self.good}", "--html-name", "sub/x.html"],
                     ["--mode", "score", "--method", f"g={self.good}", "--iou", "1.5"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as e:
                de.main(argv)
            self.assertEqual(e.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
