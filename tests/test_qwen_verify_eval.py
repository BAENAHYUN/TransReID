"""오프라인 테스트: eval/qwen_verify_eval.py — 불변 후보 id 로 라벨 연결(재랭킹 안전), P@K 전/후(paired), filter 재현, 오탈락 두 정의, UNKNOWN(관찰 행만),
Qwen 명령(항상 flag)·캐시 계약, 후보 payload 형식, 시트, 원장·기준."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import criteria as C  # noqa: E402
from bench import ledger as L  # noqa: E402
from eval import qwen_verify_eval as Q  # noqa: E402

CANDS = [{"rank": r, "cand_id": f"p{r}", "point_id": f"p{r}", "crop_path": None, "score": 1 - r / 100} for r in range(1, 11)]
REL = {1: "yes", 2: "yes", 3: "no", 4: "yes", 5: "no", 6: "no", 7: "no", 8: "unsure", 9: "yes", 10: "yes"}
LABELS = {f"q1:{r}": {"relevant": v, "reviewed": True} for r, v in REL.items()}


def row(orig, rank, verified, pre_rank=None):
    """Qwen 결과 행. pre_qwen_rank 는 재랭커가 있으면 재랭킹 후 순위로 덮어써진다 → 라벨은 point_id 로 연결해야 한다."""
    return {"pre_qwen_rank": pre_rank if pre_rank is not None else orig, "rank": rank, "verified": verified,
            "attr_score": None if verified is None else (0.9 if verified else 0.2), "point_id": f"p{orig}"}


FLAG_ROWS = [row(2, 1, True), row(9, 2, True), row(1, 3, False), row(3, 4, None), row(4, 5, True), row(5, 6, False), row(6, 7, False),
             row(7, 8, False), row(8, 9, None), row(10, 10, True)]


class ScoreTests(unittest.TestCase):
    def test_precision_at(self):
        rel = {"a": True, "b": None, "c": False}
        self.assertEqual(Q.precision_at(["a", "b", "c"], rel, 3), 0.5)
        self.assertEqual(Q.precision_at(["a", "b", "c"], rel, 1), 1.0)
        self.assertIsNone(Q.precision_at(["b"], rel, 5))

    def test_flag_mode(self):
        sc = Q.score_query(CANDS, FLAG_ROWS, LABELS, "q1", "flag")
        self.assertEqual((sc["judged"], sc["relevant"], sc["unsure_or_unlabeled"]), (9, 5, 1))
        self.assertAlmostEqual(sc["p5_before"], 0.6)
        self.assertAlmostEqual(sc["p5_after"], 0.8)
        self.assertAlmostEqual(sc["p10_before"], 5 / 9)
        self.assertAlmostEqual(sc["p10_after"], 5 / 9)
        self.assertEqual((sc["scored"], sc["unknown"], sc["fail"], sc["pass"]), (10, 2, 4, 4))
        self.assertEqual((sc["false_drops"], sc["fails_labeled"], sc["relevant_judged_by_qwen"]), (1, 4, 5))
        m = Q.aggregate([sc])
        self.assertEqual((m["p5_before"], m["p5_after"], m["p5_gain_pp"], m["p5_queries"]), (60.0, 80.0, 20.0, 1))
        self.assertEqual(m["false_drop_rate"], 0.25)          # FAIL 4 중 정답 1 (기준표 정의)
        self.assertEqual(m["lost_relevant_rate"], 0.2)        # 판정된 정답 5 중 FAIL 1
        self.assertEqual(m["unknown_ratio"], 0.2)
        self.assertEqual((m["queries"], m["candidates"], m["judged"]), (1, 10, 9))

    def test_labels_follow_point_id_not_pre_qwen_rank(self):
        # 재랭커가 pre_qwen_rank 를 덮어쓴 상황: 원래 2위(p2, yes) 가 pre_qwen_rank 1 로 기록됨. point_id 로 연결하면 p2 의 라벨(yes)이 붙는다.
        rows = [row(2, 1, True, pre_rank=1), row(1, 2, True, pre_rank=2)]
        sc = Q.score_query(CANDS, rows, LABELS, "q1", "flag")
        self.assertEqual(sc["relevant_judged_by_qwen"], 2)
        rows_bad = [row(3, 1, True, pre_rank=1)]                              # 원래 3위(no) 가 1위로 → 정답 아님
        sc2 = Q.score_query(CANDS, rows_bad, LABELS, "q1", "flag")
        self.assertEqual(sc2["p5_after"], 0.0)

    def test_filter_mode_reproduced_by_evaluator(self):
        sc = Q.score_query(CANDS, FLAG_ROWS, LABELS, "q1", "filter")
        self.assertEqual(sc["retained"], 6)                                  # FAIL 4 제거
        self.assertAlmostEqual(sc["p10_after"], 0.8)                         # 2,9,3,4,(8 모름),10 → 4/5
        self.assertEqual((sc["fail"], sc["false_drops"]), (4, 1))            # 제거해도 판정 이력은 남아 오탈락을 셀 수 있다
        self.assertEqual(Q.aggregate([sc, sc])["p10_after"], 80.0)

    def test_unobserved_tail_rows_are_not_scored(self):
        rows = [row(2, 1, True), row(9, 2, True), row(1, 3, False), row(3, 4, None), row(4, 5, True)]
        rows += [{"pre_qwen_rank": r, "rank": i, "verified": None, "point_id": f"p{r}"} for i, r in enumerate((5, 6, 7, 8, 10), 6)]
        sc = Q.score_query(CANDS, rows, LABELS, "q1", "flag")
        self.assertEqual((sc["scored"], sc["unknown"], sc["fail"], sc["pass"]), (5, 1, 1, 3))
        self.assertEqual(Q.aggregate([sc])["unknown_ratio"], 0.2)

    def test_paired_aggregation_and_empty(self):
        a = Q.score_query(CANDS, FLAG_ROWS, LABELS, "q1", "flag")
        b = dict(a)
        b["p10_after"] = None                                                # after 가 정의되지 않은 쿼리는 짝에서 빠진다
        m = Q.aggregate([a, b])
        self.assertEqual(m["p10_queries"], 1)
        self.assertAlmostEqual(m["p10_before"], round(100 * 5 / 9, 2))
        m0 = Q.aggregate([])
        self.assertIsNone(m0["p10_before"])
        self.assertIsNone(m0["false_drop_rate"])


class PlumbingTests(unittest.TestCase):
    def args(self, **kw):
        base = dict(top_k=7, alpha=0.5, threshold=0.4, verify_mode="filter", dtype="bfloat16", max_pixels=1000, no_reranker=True,
                    model_id="Qwen/X", reranker_model_id=None, device=None)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_qwen_command_always_flag(self):
        cmd = Q.qwen_command("py", Path("in.json"), Path("out.json"), self.args())
        self.assertEqual(cmd[:3], ["py", "-u", str(Q.QWEN_SCRIPT)])
        self.assertEqual(cmd[cmd.index("--verify-mode") + 1], "flag")       # filter 는 평가기가 재현
        self.assertIn("--no-reranker", cmd)
        self.assertEqual(cmd[cmd.index("--model-id") + 1], "Qwen/X")
        self.assertEqual(cmd[cmd.index("--top-k") + 1], "7")
        self.assertNotIn("--no-reranker", Q.qwen_command("py", Path("i"), Path("o"), self.args(no_reranker=False)))
        rc = Q.rescore_command("py", Path("c.json"), Path("o.json"), self.args())
        self.assertIn("--rescore-only", rc)

    def test_cache_key_depends_on_contract_not_alpha(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "q.json"
            inp.write_text('{"a": 1}', encoding="utf-8")
            k1 = Q.cache_key(inp, self.args())
            self.assertEqual(k1, Q.cache_key(inp, self.args(alpha=0.9, threshold=0.1, verify_mode="flag")))   # 채점 인자는 계약 밖
            self.assertNotEqual(k1, Q.cache_key(inp, self.args(top_k=5)))
            self.assertNotEqual(k1, Q.cache_key(inp, self.args(model_id="Qwen/Y")))
            self.assertNotEqual(k1, Q.cache_key(inp, self.args(no_reranker=False)))
            inp.write_text('{"a": 2}', encoding="utf-8")
            self.assertNotEqual(k1, Q.cache_key(inp, self.args()))            # 후보가 바뀌면 계약도 바뀐다

    def test_candidate_payload_matches_gui_shape(self):
        rows = [{"rank": 1, "score": 0.9, "retrieval_score": 0.8, "point_id": "a", "crop_path": "no/such.jpg", "payload": {"crop_path": "no/such.jpg"}},
                {"rank": 2, "score": None, "retrieval_score": 0.7, "point_id": "b", "crop_path": ""}]
        pl = Q.candidate_payload({"query_id": "q1", "text": "빨간 옷"}, {"query": "빨간 옷", "query_en": "red clothes", "scope": "person", "collection": "forensic_person"}, rows)
        self.assertEqual((pl["search_type"], pl["crops"][0]["kind"], pl["crops"][0]["query_text"]), ("text", "text", "red clothes"))
        self.assertEqual([r["pre_qwen_rank"] for r in pl["crops"][0]["results"]], [1, 2])
        self.assertEqual([r["cand_id"] for r in pl["crops"][0]["results"]], ["a", "b"])
        self.assertEqual(pl["crops"][0]["results"][1]["pre_qwen_score"], 0.7)

    def test_sheet_report_ledger_criteria(self):
        items = [{"query_id": "q1", "text": "빨간 옷", "query_en": "red clothes", "candidates": [{"rank": 1, "cand_id": "a", "crop_path": None, "score": 0.9}]}]
        html = Q.build_sheet(items, 100, "m123")
        self.assertIn('data-item="q1:1"', html)
        self.assertIn('data-field="reviewed"', html)
        self.assertIn("m123", html)
        self.assertNotIn("no/such", html)
        sc = Q.score_query(CANDS, FLAG_ROWS, LABELS, "q1", "flag")
        sc.update({"query_id": "q1", "text": "t", "qwen_elapsed_sec": 12.0, "labeled": True, "cache": "ran"})
        m = Q.aggregate([sc])
        m["sec_per_candidate"] = 1.2
        out = {"producer": Q.PRODUCER, "generated_at": "2026-09-28T00:00:00", "name": "qwen_flag", "labeled_queries": 1, "unlabeled_only": False,
               "config": {"top_k": 10, "alpha": 0.7, "threshold": 0.5, "verify_mode": "flag", "no_reranker": False, "model_id": "Qwen/Qwen3-VL-2B-Instruct",
                          "reranker_used": True}, "gt": {"queries": 1, "labeled_queries": 1, "judged": 9, "candidates": 10, "coverage": 0.9},
               "metrics": m, "per_query": [sc], "inputs": [{"role": "labels", "path": "x", "sha1": "y"}]}
        md = Q.report_md(out)
        self.assertIn("| P@5 | 60.0 | 80.0 | 20.0 | 1 |", md)
        e = L.entry_from_qwen_result(out, report="r.json", command=["eval"])
        self.assertEqual((e["stage"], e["component"]["model_id"], e["params"]["top_k"], e["timing"]["sec_per_candidate"]), ("qwen", "Qwen/Qwen3-VL-2B-Instruct", 10, 1.2))
        self.assertEqual(e["inputs"][0]["role"], "labels")
        ev = C.evaluate(e)
        self.assertEqual(ev["status"], "partial")                             # gain 0 → 미달, 오탈락 0.25 → 미달, 1.2 s → 통과
        out["config"]["model_id"] = "Qwen/Qwen3-VL-4B-Instruct"
        m["sec_per_candidate"] = 45.0
        ev4 = C.evaluate(L.entry_from_qwen_result(out))
        self.assertTrue(next(c for c in ev4["checks"] if c["metric"] == "sec_per_candidate")["ok"])   # 4B 는 60 s 기준


if __name__ == "__main__":
    unittest.main()
