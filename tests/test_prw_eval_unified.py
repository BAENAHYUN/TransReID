"""eval/prw_eval_unified.py 순위 논리 테스트 (데이터·모델 없음)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import prw_eval  # noqa: E402
from eval import prw_eval_unified as U  # noqa: E402


class RankingTests(unittest.TestCase):
    def test_topk_indices_order(self):
        s = np.array([0.1, 0.9, 0.5, 0.9, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(U.topk_indices(s, 3), [1, 3, 2])
        np.testing.assert_array_equal(U.topk_indices(s, 10), [1, 3, 2, 0, 4])

    def test_rrf_fuse(self):
        lists = {"a": np.array([5, 7, 9]), "b": np.array([7, 5, 11])}
        idx, sc = U.rrf_fuse(lists, {"a": 1.0, "b": 1.5}, k=2)
        # 7: 1/(2+2) + 1.5/(2+1) = 0.25 + 0.5 = 0.75 ; 5: 1/3 + 1.5/4 = 0.7083 ; 9: 1/5 ; 11: 1.5/5
        self.assertEqual(idx.tolist(), [7, 5, 11, 9])
        self.assertAlmostEqual(float(sc[0]), 0.75, places=6)
        idx2, _ = U.rrf_fuse({"a": np.array([1, 2]), "b": np.array([2, 1])}, {"a": 1.0, "b": 1.0}, k=2)
        self.assertEqual(idx2.tolist(), [1, 2])  # 동점이면 인덱스 오름차순

    def test_rerank_by(self):
        pool = np.array([3, 8, 1])
        scores = np.zeros(10); scores[8] = 0.9; scores[1] = 0.5; scores[3] = 0.1
        np.testing.assert_array_equal(U.rerank_by(pool, scores), [8, 1, 3])

    def test_full_ranking_matches_prw_eval(self):
        rng = np.random.default_rng(0)
        G, Q, D = 300, 40, 16
        g_pids = rng.integers(0, 25, size=G)
        g_frames = [f"c1s1_{rng.integers(0, 60)}" for _ in range(G)]
        q_pids = g_pids[:Q].copy()
        q_frames = [g_frames[i] for i in range(Q)]
        g_vecs = U.l2n(rng.normal(size=(G, D)))
        q_vecs = U.l2n(g_vecs[:Q] + 0.3 * rng.normal(size=(Q, D)))
        ref = prw_eval.evaluate_prw(q_vecs, q_pids, q_frames, g_vecs, g_pids, g_frames, max_rank=10)
        sims = q_vecs @ g_vecs.T
        ranked = [np.argsort(-sims[qi], kind="stable") for qi in range(Q)]
        got = U.evaluate_ranked(ranked, q_pids, q_frames, g_pids, g_frames, max_rank=10)
        self.assertEqual(got["valid_queries"], ref["valid_queries"])
        for key in ("mAP", "Rank-1", "Rank-5", "Rank-10"):
            self.assertAlmostEqual(got[key], round(100 * ref[key], 4), places=3, msg=key)
        self.assertAlmostEqual(got["pool_recall(%)"], 100.0)

    def test_partial_ranking_penalises_missing_positives(self):
        g_pids = np.array([1, 1, 1, 2, 2])
        g_frames = ["f1", "f2", "f3", "f4", "f5"]
        q_pids, q_frames = np.array([1]), ["f9"]
        full = [np.array([0, 3, 1, 4, 2])]      # 정답 3개 중 순위 1,3,5
        part = [np.array([0, 3])]               # 후보 2개만 (정답 1개만 포함)
        r_full = U.evaluate_ranked(full, q_pids, q_frames, g_pids, g_frames)
        r_part = U.evaluate_ranked(part, q_pids, q_frames, g_pids, g_frames)
        self.assertAlmostEqual(r_full["mAP"], round(100 * (1 + 2 / 3 + 3 / 5) / 3, 4))
        self.assertAlmostEqual(r_part["mAP"], round(100 * (1 / 3), 4))
        self.assertAlmostEqual(r_part["pool_recall(%)"], round(100 / 3, 4))
        self.assertEqual(r_part["Rank-1"], 100.0)
        self.assertEqual(r_part["queries_all_positives_in_pool(%)"], 0.0)


if __name__ == "__main__":
    unittest.main()
