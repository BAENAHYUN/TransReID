"""eval/prw_cluster_gt_eval.py 순수 함수 테스트 (데이터·Qdrant 없음)."""
import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import prw_cluster_gt_eval as G  # noqa: E402


class MatchTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(G.iou_xyxy_vs_xywh([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(G.iou_xyxy_vs_xywh([0, 0, 10, 10], [5, 0, 10, 10]), 50 / 150)
        self.assertEqual(G.iou_xyxy_vs_xywh([0, 0, 10, 10], [20, 20, 5, 5]), 0.0)
        self.assertEqual(G.iou_xyxy_vs_xywh([0, 0, 0, 0], [0, 0, 0, 0]), 0.0)

    def test_match_detection(self):
        ann = np.array([[-2, 0, 0, 10, 10], [7, 100, 100, 50, 100], [9, 120, 100, 50, 100]], dtype=np.float32)
        pid, iou, row = G.match_detection([100, 100, 150, 200], ann, 0.5)
        self.assertEqual((pid, row), (7, 1))
        self.assertAlmostEqual(iou, 1.0)
        pid, iou, row = G.match_detection([300, 300, 310, 310], ann, 0.5)
        self.assertEqual((pid, row), (None, -1))
        pid, _, _ = G.match_detection([0, 0, 10, 10], ann, 0.5)
        self.assertEqual(pid, -2)  # 미표기 보행자 pid 그대로 반환 (호출측이 제외)
        self.assertEqual(G.match_detection([0, 0, 1, 1], np.empty((0, 5), dtype=np.float32), 0.5)[0], None)

    def test_frame_from_image_id(self):
        self.assertEqual(G.frame_from_image_id("PRW/c5s2_116999.jpg"), "c5s2_116999")
        self.assertEqual(G.frame_from_image_id("PRW\\c1s1_000151.jpg"), "c1s1_000151")
        self.assertIsNone(G.frame_from_image_id("PRW/x.png"))


class MetricTests(unittest.TestCase):
    # 6 points, 2 pids. 클러스터: A={1,2,4}(pid 1,1,2 혼합), B={3}(pid1 단독), 5 노이즈(pid2), 6 노이즈(pid2)
    pid_of = {"1": 1, "2": 1, "3": 1, "4": 2, "5": 2, "6": 2}
    labels = {"1": "A", "2": "A", "3": "B", "4": "A", "5": None, "6": None}
    ids = ["1", "2", "3", "4", "5", "6"]

    def brute_pairs(self, ids, use_noise):
        use = [p for p in ids if use_noise or self.labels[p] is not None]
        t = pr = b = 0
        for a, c in itertools.combinations(use, 2):
            same_pid = self.pid_of[a] == self.pid_of[c]
            same_cl = self.labels[a] is not None and self.labels[a] == self.labels[c]
            t += same_pid; pr += same_cl; b += same_pid and same_cl
        return t, pr, b

    def test_pair_metrics_bruteforce(self):
        for use_noise in (True, False):
            m = G.pair_metrics(self.labels, self.pid_of, self.ids, use_noise)
            t, pr, b = self.brute_pairs(self.ids, use_noise)
            self.assertEqual((m["pairs_true"], m["pairs_pred"], m["pairs_both"]), (t, pr, b))
            self.assertAlmostEqual(m["precision"], b / pr)
            self.assertAlmostEqual(m["recall"], b / t)
        m = G.pair_metrics(self.labels, self.pid_of, self.ids, True)
        self.assertEqual(m["points"], 6)
        self.assertEqual(G.pair_metrics(self.labels, self.pid_of, self.ids, False)["points"], 4)

    def test_bcubed(self):
        b = G.bcubed(self.labels, self.pid_of, self.ids)
        # precision: 1,2 -> 2/3 ; 4 -> 1/3 ; 3 -> 1 ; 5,6 -> 1  => (2/3*2 + 1/3 + 1 + 2)/6
        self.assertAlmostEqual(b["precision"], (2 / 3 * 2 + 1 / 3 + 1 + 2) / 6)
        # recall: pid1 size 3: 1,2 -> 2/3, 3 -> 1/3 ; pid2 size 3: 4 -> 1/3, 5 -> 1/3, 6 -> 1/3
        self.assertAlmostEqual(b["recall"], (2 / 3 * 2 + 1 / 3 + 1 / 3 * 3) / 6)
        perfect = G.bcubed({"1": "x", "2": "x", "3": "y"}, {"1": 1, "2": 1, "3": 2}, ["1", "2", "3"])
        self.assertEqual((perfect["precision"], perfect["recall"], perfect["f1"]), (1.0, 1.0, 1.0))

    def test_purity_metrics(self):
        m = G.purity_metrics(self.labels, self.pid_of, self.ids)
        self.assertEqual(m["clustered_points"], 4)
        self.assertAlmostEqual(m["purity"], (2 + 1) / 4)          # A: max 2 (pid1), B: 1
        self.assertAlmostEqual(m["inverse_purity"], (2 + 1) / 4)  # pid1: max 2 (A), pid2: 1 (A)
        self.assertEqual((m["clusters_with_labeled"], m["pure_clusters"], m["mixed_clusters"], m["mixed_cluster_points"]), (2, 1, 1, 3))
        self.assertEqual((m["pids"], m["pids_clustered"], m["pids_split"], m["pids_all_noise"]), (2, 2, 1, 0))
        self.assertIn("A", m["mixed_detail"])
        self.assertEqual(m["split_detail"]["1"]["clusters"], 2)
        m2 = G.purity_metrics({"1": None, "2": None, "3": "z"}, {"1": 5, "2": 5, "3": 6}, ["1", "2", "3"])
        self.assertEqual(m2["pids_all_noise"], 1)

    def test_ari_nmi_perfect_and_shape(self):
        r = G.ari_nmi({"1": "x", "2": "x", "3": "y"}, {"1": 1, "2": 1, "3": 2}, ["1", "2", "3"])
        self.assertAlmostEqual(r["ari_noise_as_singletons"], 1.0)
        self.assertAlmostEqual(r["nmi_clustered_only"], 1.0)
        r2 = G.ari_nmi(self.labels, self.pid_of, self.ids)
        self.assertEqual(set(r2), {"ari_noise_as_singletons", "nmi_noise_as_singletons", "ari_clustered_only", "nmi_clustered_only"})

    def test_evaluate_method_filters_unknown_points(self):
        labels = dict(self.labels); labels["zz"] = "A"   # GT 없는 point 는 무시
        res = G.evaluate_method(labels, self.pid_of, self.ids + ["not_in_labels"])
        self.assertEqual(res["labeled_points"], 6)
        self.assertEqual(res["noise_points"], 2)


if __name__ == "__main__":
    unittest.main()
