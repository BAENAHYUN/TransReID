"""오프라인 테스트: eval/qwen_verify_eval.py — P@K 전/후, 오탈락·UNKNOWN 집계(flag/filter), Qwen 명령, 후보 payload 형식, 시트, 원장·기준."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import criteria as C  # noqa: E402
from bench import ledger as L  # noqa: E402
from eval import qwen_verify_eval as Q  # noqa: E402

CANDS = [{"rank": r, "point_id": f"p{r}", "crop_path": None, "score": 1 - r / 100} for r in range(1, 11)]
REL = {1: "yes", 2: "yes", 3: "no", 4: "yes", 5: "no", 6: "no", 7: "no", 8: "unsure", 9: "yes", 10: "yes"}
LABELS = {f"q1:{r}": {"relevant": v} for r, v in REL.items()}


def row(pre, rank, verified):
    return {"pre_qwen_rank": pre, "rank": rank, "verified": verified, "attr_score": None if verified is None else (0.9 if verified else 0.2), "point_id": f"p{pre}"}


FLAG_ROWS = [row(2, 1, True), row(9, 2, True), row(1, 3, False), row(3, 4, None), row(4, 5, True), row(5, 6, False), row(6, 7, False),
             row(7, 8, False), row(8, 9, None), row(10, 10, True)]


class ScoreTests(unittest.TestCase):
    def test_precision_at(self):
        rel = {1: True, 2: None, 3: False}
        self.assertEqual(Q.precision_at([1, 2, 3], rel, 3), 0.5)
        self.assertEqual(Q.precision_at([1, 2, 3], rel, 1), 1.0)
        self.assertIsNone(Q.precision_at([2], rel, 5))

    def test_flag_mode(self):
        sc = Q.score_query(CANDS, FLAG_ROWS, LABELS, "q1")
        self.assertEqual((sc["judged"], sc["relevant"], sc["unsure"]), (9, 5, 1))
        self.assertAlmostEqual(sc["p5_before"], 0.6)
        self.assertAlmostEqual(sc["p5_after"], 0.8)
        self.assertAlmostEqual(sc["p10_before"], 5 / 9)
        self.assertAlmostEqual(sc["p10_after"], 5 / 9)
        self.assertEqual((sc["scored"], sc["unknown"], sc["fail"], sc["pass"]), (10, 2, 4, 4))
        self.assertEqual((sc["false_drops"], sc["relevant_judged_by_qwen"]), (1, 5))
        m = Q.aggregate([sc])
        self.assertEqual((m["p5_before"], m["p5_after"], m["p5_gain_pp"]), (60.0, 80.0, 20.0))
        self.assertEqual(m["false_drop_rate"], 0.2)
        self.assertEqual(m["unknown_ratio"], 0.2)
        self.assertEqual((m["queries"], m["candidates"], m["judged"]), (1, 10, 9))

    def test_filter_mode_shorter_list(self):
        rows = [r for r in FLAG_ROWS if r["verified"] is not False]
        for i, r in enumerate(rows, 1):
            r = dict(r)
            r["rank"] = i
            rows[i - 1] = r
        sc = Q.score_query(CANDS, rows, LABELS, "q1")
        self.assertEqual(sc["retained"], 6)
        self.assertAlmostEqual(sc["p10_after"], 0.8)                 # 2,9,3,4,(8 모름),10 → 4/5
        self.assertEqual(sc["fail"], 0)                                # 제거된 행은 결과에 없다
        self.assertEqual(Q.aggregate([sc, sc])["p10_after"], 80.0)

    def test_aggregate_empty_and_none(self):
        m = Q.aggregate([])
        self.assertIsNone(m["p10_before"])
        self.assertIsNone(m["false_drop_rate"])


class PlumbingTests(unittest.TestCase):
    def test_qwen_command_flags(self):
        args = SimpleNamespace(top_k=7, alpha=0.5, threshold=0.4, verify_mode="filter", dtype="bfloat16", max_pixels=1000, no_reranker=True,
                               model_id="Qwen/X", reranker_model_id=None, device=None)
        cmd = Q.qwen_command("py", Path("in.json"), Path("out.json"), args)
        self.assertEqual(cmd[:3], ["py", "-u", str(Q.QWEN_SCRIPT)])
        self.assertIn("--no-reranker", cmd)
        self.assertEqual(cmd[cmd.index("--model-id") + 1], "Qwen/X")
        self.assertEqual(cmd[cmd.index("--verify-mode") + 1], "filter")
        self.assertEqual(cmd[cmd.index("--top-k") + 1], "7")
        args.no_reranker = False
        self.assertNotIn("--no-reranker", Q.qwen_command("py", Path("i"), Path("o"), args))

    def test_candidate_payload_matches_gui_shape(self):
        rows = [{"rank": 1, "score": 0.9, "retrieval_score": 0.8, "point_id": "a", "crop_path": "no/such.jpg", "payload": {"crop_path": "no/such.jpg"}},
                {"rank": 2, "score": None, "retrieval_score": 0.7, "point_id": "b", "crop_path": ""}]
        pl = Q.candidate_payload({"query_id": "q1", "text": "빨간 옷"}, {"query": "빨간 옷", "query_en": "red clothes", "scope": "person", "collection": "forensic_person"}, rows)
        self.assertEqual(pl["search_type"], "text")
        self.assertEqual(pl["crops"][0]["kind"], "text")
        self.assertEqual(pl["crops"][0]["query_text"], "red clothes")
        self.assertEqual([r["pre_qwen_rank"] for r in pl["crops"][0]["results"]], [1, 2])
        self.assertEqual(pl["crops"][0]["results"][1]["pre_qwen_score"], 0.7)
        self.assertEqual(pl["crops"][0]["results"][0]["qdrant_score"], 0.8)

    def test_sheet_and_report(self):
        items = [{"query_id": "q1", "text": "빨간 옷", "query_en": "red clothes", "candidates": [{"rank": 1, "point_id": "a", "crop_path": None, "score": 0.9}]}]
        html = Q.build_sheet(items, 100)
        self.assertIn('data-item="q1:1"', html)
        self.assertIn("red clothes", html)
        self.assertIn("이미지 없음", html)
        sc = Q.score_query(CANDS, FLAG_ROWS, LABELS, "q1")
        sc.update({"query_id": "q1", "text": "t", "qwen_elapsed_sec": 12.0, "labeled": True})
        m = Q.aggregate([sc])
        m["sec_per_candidate"] = 1.2
        out = {"producer": Q.PRODUCER, "generated_at": "2026-09-28T00:00:00", "name": "qwen_flag", "labeled_queries": 1, "unlabeled_only": False,
               "config": {"top_k": 10, "alpha": 0.7, "threshold": 0.5, "verify_mode": "flag", "no_reranker": False, "model_id": "Qwen/Q", "reranker_used": True},
               "gt": {"queries": 1, "labeled_queries": 1, "judged": 9, "candidates": 10}, "metrics": m, "per_query": [sc]}
        md = Q.report_md(out)
        self.assertIn("| P@5 | 60.0 | 80.0 | 20.0 |", md)
        e = L.entry_from_qwen_result(out, report="r.json", command=["eval"])
        self.assertEqual(e["stage"], "qwen")
        self.assertTrue(e["run_id"].startswith("qwen_"))
        self.assertEqual(e["component"]["model_id"], "Qwen/Q")
        self.assertEqual(e["params"]["top_k"], 10)
        self.assertEqual(e["timing"]["sec_per_candidate"], 1.2)
        ev = C.evaluate(e)
        self.assertEqual(ev["status"], "partial")                       # P@10 gain 0 → 미달, 오탈락 0.2 → 미달, 1.2 s/후보 → 통과
        self.assertEqual(sum(1 for c in ev["checks"] if c["ok"]), 1)


if __name__ == "__main__":
    unittest.main()
