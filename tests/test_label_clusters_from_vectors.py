"""label_clusters_from_vectors 순수 함수 테스트 — 모델·Qdrant 없음."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clustering import label_clusters_from_vectors as lab  # noqa: E402

K = len(lab.COLORS)
IDX = {name: i for i, (name, _) in enumerate(lab.COLORS)}


def scores_with_bias(rng, n, true_colors, bias_color="yellow", bias=0.05, signal=0.03, noise=0.005):
    """crop 마다 실제 색 열에 signal 을, 편향 열(bias_color)에 항상 bias 를 더한 가짜 코사인 점수."""
    base = 0.10 + rng.normal(0, noise, size=(n, K))
    for i, color in enumerate(true_colors):
        base[i, IDX[color]] += signal
    base[:, IDX[bias_color]] += bias
    return base.astype(np.float32)


class PromptTests(unittest.TestCase):
    def test_prompt_texts_cover_every_color_and_template(self):
        bank = lab.BANKS["upper_color"]
        texts = lab.prompt_texts(bank)
        self.assertEqual(len(texts), K * len(bank["templates"]))
        self.assertEqual(texts[0], ("black", "a person wearing a black top"))
        self.assertTrue(all("{c}" not in t for _, t in texts))

    def test_color_text_matrix_averages_templates(self):
        bank = lab.BANKS["upper_color"]
        n_t = len(bank["templates"])
        vectors = np.zeros((K * n_t, 4), dtype=np.float32)
        for k in range(K):
            vectors[k * n_t:(k + 1) * n_t, k % 4] = 1.0 + np.arange(n_t)      # 같은 방향, 다른 크기
        matrix = lab.color_text_matrix(vectors, bank)
        self.assertEqual(matrix.shape, (K, 4))
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-6)
        self.assertEqual(int(np.argmax(matrix[5])), 5 % 4)
        with self.assertRaises(ValueError):
            lab.color_text_matrix(vectors[:-1], bank)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.truth = [lab.COLORS[i % K][0] for i in range(600)]
        self.scores = scores_with_bias(self.rng, 600, self.truth)

    def test_none_is_degenerate_but_zscore_recovers_truth(self):
        top_none, _, _ = lab.crop_decisions(lab.calibrate(self.scores, "none"), 0.3)
        top_z, _, _ = lab.crop_decisions(lab.calibrate(self.scores, "zscore"), 0.3)
        top_c, _, _ = lab.crop_decisions(lab.calibrate(self.scores, "center"), 0.3)
        truth_idx = np.array([IDX[c] for c in self.truth])
        self.assertGreater(np.mean(top_none == IDX["yellow"]), 0.9)         # 편향 열이 다 먹는다
        self.assertGreater(np.mean(top_z == truth_idx), 0.95)
        self.assertGreater(np.mean(top_c == truth_idx), 0.95)
        with self.assertRaises(ValueError):
            lab.calibrate(self.scores, "bogus")

    def test_margin_threshold_scales_with_method(self):
        z = lab.calibrate(self.scores, "zscore")
        raw = lab.calibrate(self.scores, "none")
        self.assertAlmostEqual(lab.margin_threshold(z, 0.3), 0.3 * z.std(), places=6)
        self.assertLess(lab.margin_threshold(raw, 0.3), 0.05)               # 코사인 척도에서는 작아진다
        _, margins, confident = lab.crop_decisions(z, 0.3)
        self.assertEqual(confident.tolist(), (margins >= lab.margin_threshold(z, 0.3)).tolist())
        self.assertTrue(lab.crop_decisions(z, 0.0)[2].all())


class ClusterDecisionTests(unittest.TestCase):
    """보정은 전체 crop 모집단(600장, 12색 균등) 위에서 하고, 군집은 그 안의 index 로 고른다.
    (수십 장짜리 모집단에서는 없는 색의 열이 잡음만 남아 z 점수가 부풀어 오른다 — 실제 DB 규모와 다르다.)"""

    def setUp(self):
        rng = np.random.default_rng(1)
        self.truth = [lab.COLORS[i % K][0] for i in range(600)]
        self.z = lab.calibrate(scores_with_bias(rng, len(self.truth), self.truth), "zscore")
        self.top, _, _ = lab.crop_decisions(self.z, 0.3)
        self.thr = lab.margin_threshold(self.z, 0.3)
        self.red = [i for i, c in enumerate(self.truth) if c == "red"]
        self.blue = [i for i, c in enumerate(self.truth) if c == "blue"]
        self.green = [i for i, c in enumerate(self.truth) if c == "green"]

    def test_pure_cluster_is_labeled(self):
        d = lab.cluster_decision(self.red[:10], self.z, self.top, self.thr, 0.5)
        self.assertEqual((d["label"], d["candidate"], d["n"], d["status"]), (IDX["red"], "red", 10, "labeled"))
        self.assertGreaterEqual(d["agreement"], 0.9)
        self.assertEqual(list(d["distribution"])[0], "red")

    def test_mixed_cluster_is_uncertain(self):
        d = lab.cluster_decision(self.red[:5] + self.blue[:5], self.z, self.top, self.thr, 0.6)
        self.assertIsNone(d["label"])
        self.assertEqual(d["status"], "uncertain")
        self.assertLessEqual(d["agreement"], 0.6)

    def test_two_member_cluster_can_be_labeled(self):
        d = lab.cluster_decision(self.red[:2], self.z, self.top, self.thr, 0.5)            # red + red
        self.assertEqual((d["label"], d["status"]), (IDX["red"], "labeled"))
        d2 = lab.cluster_decision([self.red[0], self.green[0]], self.z, self.top, self.thr, 0.5)   # red + green
        self.assertEqual(d2["status"], "tentative")                                       # 일치율 0.5 는 넘지만 둘이 갈림
        self.assertEqual(d2["agreement"], 0.5)
        self.assertEqual(d2["label"], IDX[d2["candidate"]])

    def test_tentative_when_agreement_ok_but_margin_small(self):
        z = np.zeros((3, K), dtype=np.float32)
        z[:, IDX["blue"]] = 1.0
        z[:, IDX["black"]] = 0.95                                                           # 1·2위 차 0.05
        top = np.full(3, IDX["blue"])
        d = lab.cluster_decision([0, 1, 2], z, top, threshold=0.3, min_share=0.5)
        self.assertEqual((d["status"], d["label"], d["agreement"]), ("tentative", IDX["blue"], 1.0))
        self.assertAlmostEqual(d["margin"], 0.05, places=5)

    def test_compose_name(self):
        votes = {"upper_color": dict(label=IDX["yellow"], status="labeled", candidate="yellow", margin=1.2,
                                     agreement=0.8, n=7)}
        name, conf, desc, status = lab.compose_name(votes, "ko")
        self.assertEqual((name, conf, status), ("노란색 상의", 0.8, "labeled"))
        self.assertIn("upper_color=yellow [labeled]", desc)
        self.assertEqual(lab.compose_name(votes, "en")[0], "yellow top")
        votes["upper_color"]["status"] = "tentative"
        self.assertEqual(lab.compose_name(votes, "ko")[0], "노란색 상의(추정)")
        self.assertEqual(lab.compose_name(votes, "en")[0], "yellow top (tentative)")
        self.assertEqual(lab.compose_name(votes, "ko")[3], "tentative")
        votes["upper_color"].update(label=None, status="uncertain")
        name, conf, _, status = lab.compose_name(votes, "ko")
        self.assertEqual((name, conf, status), ("", 0.0, "uncertain"))          # 빈 이름 → 폴더 이름에 안 붙음
        self.assertEqual(lab.uncertain_word("ko"), "불확실")


class ConsistencyTests(unittest.TestCase):
    def test_gt_consistency_metrics(self):
        ids = [f"p{i}" for i in range(12)]
        gt = {f"p{i}": 1 for i in range(6)}
        gt.update({f"p{i}": 2 for i in range(6, 12)})
        top = np.array([IDX["red"]] * 5 + [IDX["blue"]] + [IDX["blue"]] * 6)
        confident = np.ones(12, dtype=bool)
        m = lab.gt_consistency(gt, ids, top, confident, min_crops=5)
        self.assertEqual(m["identities"], 2)
        self.assertAlmostEqual(m["mean_agreement"], (5 / 6 + 1.0) / 2)
        self.assertEqual(m["label_counts"], {"blue": 7, "red": 5})
        self.assertGreater(m["label_diversity"], 0.0)
        self.assertEqual(m["confident_ratio"], 1.0)
        degenerate = lab.gt_consistency(gt, ids, np.zeros(12, dtype=int), confident)
        self.assertEqual(degenerate["label_diversity"], 0.0)
        self.assertEqual(degenerate["mean_agreement"], 1.0)                # 한 색만 → 일관성은 높아도 다양성 0
        none = lab.gt_consistency({}, ids, top, np.zeros(12, dtype=bool))
        self.assertIsNone(none["mean_agreement"])
        self.assertEqual(none["confident_ratio"], 0.0)

    def test_l2n(self):
        m = lab.l2n(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
        np.testing.assert_allclose(m[0], [0.6, 0.8])
        self.assertTrue(np.isfinite(m).all())

    def test_parse_args_validation(self):
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (["--min-share", "0"], ["--min-share", "1.5"], ["--margin", "-1"], ["--calibration", "x"]):
                with self.assertRaises(SystemExit):
                    lab.parse_args(argv)
        args = lab.parse_args(["--margin", "0.2"])
        self.assertEqual((args.calibration, args.lang, args.min_share), ("zscore", "ko", 0.5))


if __name__ == "__main__":
    unittest.main()
