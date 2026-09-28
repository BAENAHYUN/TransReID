"""오프라인 테스트: eval/prw_e2e_search_eval.py 의 채점 함수 (score_query / aggregate / select_queries / 원장 빌더)."""
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import ledger as L  # noqa: E402
from eval import prw_e2e_search_eval as E  # noqa: E402

M = {
    "p1": {"frame": "f1", "pid": 7, "status": "labeled"},
    "p2": {"frame": "f2", "pid": 7, "status": "labeled"},
    "p3": {"frame": "f3", "pid": 9, "status": "labeled"},
    "p4": {"frame": "f2", "pid": -2, "status": "unlabeled"},
    "p5": {"frame": "t1", "pid": 7, "status": "labeled"},     # train 프레임 (test 집합 밖)
    "p6": {"frame": "f9", "pid": 7, "status": "labeled"},
}
TEST = {"f1", "f2", "f3", "f9"}


class ScoreTests(unittest.TestCase):
    def test_junk_distractor_and_ap(self):
        # 쿼리 pid 7, 프레임 f1. 순위: p1(junk) · x(비PRW) · p5(train) · p2(양성) · p3(음성) · p6(양성)
        s = E.score_query(["p1", "x", "p5", "p2", "p3", "p6"], M, 7, "f1", TEST, n_pos_gt=3, n_pos_db=2)
        self.assertEqual(s["excluded"], 2)               # x, p5
        self.assertEqual(s["considered"], 3)             # p2 p3 p6 (p1 은 junk)
        self.assertEqual(s["first_rank"], 1)
        self.assertEqual(s["hits"], 2)
        # AP 합 = 1/1 + 2/3 = 1.6667 → GT 분모 3 → 0.5556 / DB 분모 2 → 0.8333
        self.assertAlmostEqual(s["ap"], (1 + 2 / 3) / 3, places=6)
        self.assertAlmostEqual(s["ap_db"], (1 + 2 / 3) / 2, places=6)
        self.assertAlmostEqual(s["recall"], 2 / 3, places=6)

    def test_gallery_all_keeps_others_as_negatives(self):
        s = E.score_query(["x", "p5", "p2"], M, 7, "f1", None, n_pos_gt=2, n_pos_db=2)
        self.assertEqual(s["excluded"], 0)
        self.assertEqual(s["considered"], 3)
        self.assertEqual(s["first_rank"], 2)             # p5 는 gallery=all 에서 양성 (pid 7, 다른 프레임)
        self.assertEqual(s["hits"], 2)

    def test_no_db_positive_gives_zero(self):
        s = E.score_query(["p3"], M, 7, "f1", TEST, n_pos_gt=2, n_pos_db=0)
        self.assertEqual((s["ap"], s["ap_db"], s["recall"], s["first_rank"]), (0.0, 0.0, 0.0, None))

    def test_duplicate_detections_count_once_for_gt_metrics(self):
        """같은 GT 의 중복 crop 은 GT 기준 지표에서 한 번만 TP (그 뒤는 비관련 → 순위 벌점), DB 기준 지표에서는 모두 양성."""
        m = dict(M, p7={"frame": "f2", "pid": 7, "status": "labeled"})   # f2 의 pid 7 GT 1개에 검출 2개 (p2, p7)
        gt_frames = Counter({"f2": 1, "f9": 1})
        # GT 양성 2 (f2, f9) 중 검출은 f2 만 (중복 2개) → 순위 p2, p7, p6
        s = E.score_query(["p2", "p7", "p6"], m, 7, "f1", TEST, n_pos_gt=2, n_pos_db=3, gt_frames=gt_frames)
        self.assertEqual(s["duplicates"], 1)
        self.assertEqual(s["hits"], 2)                      # p2, p6
        self.assertEqual(s["hits_db"], 3)                   # p2, p7, p6
        # GT 기준: rel = [T, F(중복), T] → AP = (1/1 + 2/3) / 2
        self.assertAlmostEqual(s["ap"], (1 + 2 / 3) / 2, places=6)
        # DB 기준: rel = [T, T, T] → AP = 1
        self.assertAlmostEqual(s["ap_db"], 1.0, places=6)
        # 상한 없이(gt_frames=None) 세면 중복이 GT 지표를 부풀린다 — 옛 동작
        s0 = E.score_query(["p2", "p7"], m, 7, "f1", TEST, n_pos_gt=1, n_pos_db=2)
        self.assertEqual(s0["ap"], 1.0)
        s1 = E.score_query(["p2", "p7"], m, 7, "f1", TEST, n_pos_gt=2, n_pos_db=2, gt_frames=gt_frames)
        self.assertAlmostEqual(s1["ap"], 0.5, places=6)     # 2 GT 중 1 개만 검출: 중복으로 1.0 이 되지 않는다

    def test_capped_positives(self):
        db = Counter({"f2": 3, "f9": 1, "f1": 2})
        gt = Counter({"f2": 1, "f9": 1, "f1": 1})
        self.assertEqual(E.capped_positives(db, gt, "f1"), 2)         # f2: min(3,1)=1, f9: 1, f1 은 쿼리 프레임 제외
        self.assertEqual(E.capped_positives(Counter(), gt, "f1"), 0)

    def test_aggregate(self):
        rows = [dict(ap=1.0, ap_db=1.0, first_rank=1, hits=2, considered=5, excluded=1, recall=1.0, n_pos_gt=2, n_pos_db=2, sec=0.2),
                dict(ap=0.0, ap_db=0.0, first_rank=None, hits=0, considered=4, excluded=0, recall=0.0, n_pos_gt=3, n_pos_db=0, sec=0.4),
                dict(ap=0.5, ap_db=1.0, first_rank=7, hits=1, considered=5, excluded=5, recall=0.5, n_pos_gt=2, n_pos_db=1, sec=0.3)]
        a = E.aggregate(rows)
        self.assertAlmostEqual(a["map"], 50.0)
        self.assertAlmostEqual(a["map_db"], 100.0)                    # DB 양성이 있는 쿼리(1, 3번)만의 평균
        self.assertAlmostEqual(a["rank1"], 33.3333, places=3)
        self.assertAlmostEqual(a["rank10"], 66.6667, places=3)
        self.assertAlmostEqual(a["recall_at_k"], 50.0)
        self.assertAlmostEqual(a["det_ceiling"], (2 + 0 + 1) / 7, places=4)
        self.assertAlmostEqual(a["distractor_ratio"], 6 / 20, places=4)
        self.assertEqual(a["queries_without_db_positive"], 1)
        self.assertEqual(a["valid_queries"], 3)
        self.assertAlmostEqual(a["sec_per_query"], 0.3, places=4)
        self.assertEqual(E.aggregate([])["valid_queries"], 0)
        # 중복 상한이 적용된 n_pos_db_capped 가 있으면 det_ceiling 은 그것을 쓴다
        rows2 = [dict(r, n_pos_db_capped=1) for r in rows]
        self.assertAlmostEqual(E.aggregate(rows2)["det_ceiling"], 3 / 7, places=4)

    def test_positives_helpers(self):
        db = E.db_positives(M, TEST)
        self.assertEqual(db[7], Counter({"f1": 1, "f2": 1, "f9": 1}))
        self.assertEqual(E.positives_excluding(db[7], "f1"), 2)
        self.assertNotIn(-2, db)
        self.assertEqual(E.db_positives(M, None)[7]["t1"], 1)

    def test_select_queries_even_subset_and_pid_filter(self):
        qs = [{"pid": i % 5, "frame": f"f{i}", "path": ""} for i in range(20)]
        sub = E.select_queries(qs, 5)
        self.assertEqual([q["frame"] for q in sub], ["f0", "f5", "f10", "f14", "f19"])
        self.assertEqual({q["pid"] for q in E.select_queries(qs, 0, {1, 2})}, {1, 2})
        self.assertEqual(len(E.select_queries(qs, 100)), 20)


class LedgerBuilderTests(unittest.TestCase):
    def test_entry_from_e2e_result(self):
        out = {"producer": "prw_e2e_search_eval", "generated_at": "2026-09-28T10:00:00", "name": "siglip2+irra__solider",
               "config": {"config_path": "C:/p.yaml", "config_sha256": "abc", "stage1": ["siglip2", "irra"], "rerank": "solider",
                          "limit": 200, "pool": 200, "gallery": "test", "scope": "person", "pid_split": None, "max_queries": None},
               "gt": {"collection": "forensic_person", "queries": 2057, "valid_queries": 2050, "gt_positives": 10, "db_positives": 9,
                      "db_points": 43343, "matches_cache": "m.jsonl", "matches_sha256": "ff"},
               "metrics": {"map": 60.1, "map_db": 70.2, "rank1": 90.0, "rank5": 95.0, "rank10": 97.0, "recall_at_k": 80.0,
                           "det_ceiling": 0.9, "distractor_ratio": 0.1, "sec_per_query": 0.2, "valid_queries": 2050},
               "timing": {"elapsed_sec": 500.0}}
        e = L.entry_from_e2e_result(out, report="r.json", env={"host": "h"})
        self.assertEqual((e["stage"], e["name"], e["component"]["rerank"], e["params"]["limit"]), ("e2e", "siglip2+irra__solider", "solider", 200))
        self.assertEqual(e["metrics"]["map"], 60.1)
        self.assertEqual(e["gt"]["db_points"], 43343)
        self.assertEqual(e["config"]["sha256"], "abc")
        self.assertEqual(e["inputs"][0]["role"], "matches_cache")
        self.assertIn("e2e", L.STAGES)
        self.assertIn("det_ceiling", L.METRIC_KEYS["e2e"])
        self.assertTrue(e["run_id"].startswith("e2e_20260928T100000_"))


if __name__ == "__main__":
    unittest.main()
