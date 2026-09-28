"""eval/qwen_compare_runs.py — 두 Qwen 실행의 후보 단위 비교(판정 일치율·뒤집힘·순위 상관). 파일 픽스처만."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import qwen_compare_runs as C  # noqa: E402


def row(i, verified, attr, final, rank):
    r = {"cand_id": f"c{i}", "rank": rank, "attr_score": attr, "final_score": final}
    if verified is None:
        r["attr_skipped"] = "x"
    else:
        r["verified"] = verified
    return r


def write(dir_, qid, rows, elapsed, scored, batch):
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"{qid}.json").write_text(json.dumps({"crops": [{"results": rows}], "qwen_elapsed_sec": elapsed, "qwen_scored_candidates": scored,
                                                  "qwen_batch_size": batch, "qwen_model": "Q"}), encoding="utf-8")


class CompareTests(unittest.TestCase):
    def test_verdict_and_spearman(self):
        self.assertEqual(C.verdict({"verified": True}), "PASS")
        self.assertEqual(C.verdict({"verified": False}), "FAIL")
        self.assertEqual(C.verdict({"attr_skipped": "no"}), "UNKNOWN")
        self.assertEqual(C.verdict({"rank": 19}), "NA")
        self.assertAlmostEqual(C.spearman([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(C.spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertIsNone(C.spearman([1], [1]))

    def test_compare_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / "a" / "qwen", Path(td) / "b" / "qwen"
            ra = [row(1, True, 1.0, 0.9, 1), row(2, False, 0.0, 0.2, 3), row(3, None, None, None, 2), {"cand_id": "c4", "rank": 4}]
            rb = [row(1, True, 1.0, 0.9, 1), row(2, True, 1.0, 0.8, 2), row(3, None, None, None, 3), {"cand_id": "c4", "rank": 4}]
            write(a, "q01", ra, 60.0, 3, 1)
            write(b, "q01", rb, 12.0, 3, 10)
            write(a, "q02", [row(1, False, 0.0, 0.1, 1)], 20.0, 1, 1)      # q02 는 b 에 없음 → 제외
            out = C.compare_dirs(a, b, top_k=2)
            self.assertEqual(out["queries"], 1)
            s = out["summary"]
            self.assertEqual(s["common"], 3)                        # c4 는 판정 없음(NA) → 제외
            self.assertAlmostEqual(s["agree_rate"], 2 / 3)
            self.assertEqual(s["pass_fail_flips"], 1)
            self.assertEqual(s["unknown_changes"], 0)
            self.assertAlmostEqual(s["sec_per_candidate_a"], 20.0)
            self.assertAlmostEqual(s["sec_per_candidate_b"], 4.0)
            self.assertEqual(out["config_b"]["batch_size"], 10)
            q = out["per_query"][0]
            self.assertEqual(q["verdicts_a"], {"PASS": 1, "FAIL": 1, "UNKNOWN": 1})
            self.assertEqual(q["verdicts_b"], {"PASS": 2, "UNKNOWN": 1})
            self.assertAlmostEqual(q["top2_overlap"], 0.5)          # A top2 = c1,c3 · B top2 = c1,c2
            md = C.report_md(out)
            self.assertIn("| 판정 일치율 (PASS/FAIL/UNKNOWN) | 0.667 |", md)
            self.assertIn("| q01 | 3 |", md)
            rc = C.main(["--a", str(a), "--b", str(b), "--top-k", "2", "--out", str(Path(td) / "r.md")])
            self.assertEqual(rc, 0)
            self.assertTrue((Path(td) / "r.md").is_file() and (Path(td) / "r.json").is_file())


if __name__ == "__main__":
    unittest.main()
