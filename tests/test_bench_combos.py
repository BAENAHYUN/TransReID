"""오프라인 테스트: bench/combos.py — 격자 생성·순위·리더보드·pipeline_best.yaml 생성·best 선택 (Qdrant·GPU 없음)."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import combos as C  # noqa: E402


class GridTests(unittest.TestCase):
    def test_search_grid(self):
        base = {"w_siglip2": 1.0, "w_irra": 1.5, "w_solider": 1.5, "rrf_k": 2.0, "prefetch": 200}
        g = C.search_grid(["siglip2", "irra", "solider"], ["none", "solider", "irra", "siglip2"], [200, 1000], base)
        labels = [c["label"] for c in g]
        self.assertIn("siglip2+irra→solider@200", labels)
        self.assertNotIn("solider→solider@200", labels)                 # 단일 임베더 + 같은 벡터 재정렬은 제외
        self.assertIn("solider→none@200", labels)
        self.assertEqual(len(g), (7 * 4 - 3) * 2)                       # 부분집합 7 × 재정렬 4 − 중복 3, 후보 수 2
        c = next(c for c in g if c["label"] == "irra→none@1000")
        self.assertEqual((c["params"]["stage1"], c["params"]["rerank"], c["params"]["pool"], c["params"]["prefetch"]), ("irra", "none", 1000, 1000))
        self.assertEqual(c["params"]["w_irra"], 1.5)

    def test_cluster_grid(self):
        g = C.cluster_grid(["siglip2", "irra", "solider"], ["leiden", "dbscan_v6"])
        self.assertEqual([c["label"] for c in g], ["leiden@siglip2", "leiden@irra", "leiden@solider", "dbscan_v6@solider"])

    def test_load_params_from(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "best.json"
            p.write_text('{"best": {"params": {"stage1": "irra", "rerank": "none", "pool": 500, "w_irra": 0.7, "rrf_k": 3.0, "prefetch": 850}}}', encoding="utf-8")
            self.assertEqual(C.load_params_from(str(p), "search"), {"w_irra": 0.7, "rrf_k": 3.0, "prefetch": 850})
            self.assertEqual(C.load_params_from(str(p), "cluster")["pool"], 500)
            self.assertEqual(C.load_params_from(None, "search"), {})


class RankTests(unittest.TestCase):
    def rows(self):
        return [{"label": "a", "axes": {}, "params": {}, "metrics": {"map": 80.0, "pool_recall": 95.0}, "feasible": True, "violations": {}},
                {"label": "b", "axes": {}, "params": {}, "metrics": {"map": 90.0, "pool_recall": 70.0}, "feasible": False, "violations": {"pool_recall": 20}},
                {"label": "c", "axes": {}, "params": {}, "metrics": {"map": 85.0, "pool_recall": 92.0}, "feasible": True, "violations": {}}]

    def test_rank_rows_feasible_first(self):
        ranked = C.rank_rows(self.rows(), "map", [])
        self.assertEqual([r["label"] for r in ranked], ["c", "a", "b"])
        self.assertEqual([r["rank"] for r in ranked], [1, 2, 3])

    def test_leaderboard_and_pick_best(self):
        ranked = C.rank_rows(self.rows(), "map", [("pool_recall", ">=", 90.0)])
        md = C.leaderboard_md("search", ranked, "map", [("pool_recall", ">=", 90.0)], "a", {"a": {"map": 79.0}}, {"c": {"value": 86.5}})
        self.assertIn("| 1 | `c` |", md)
        self.assertIn("**(운영)**", md)
        self.assertIn("79.00", md)
        self.assertIn("86.50", md)
        best = C._pick_best(ranked, {"c": {"value": 86.5, "params": {"x": 1}, "metrics": {"map": 86.5, "pool_recall": 93.0}}}, "map")
        self.assertEqual((best["label"], best["source"], best["metrics"]["map"]), ("c", "refined", 86.5))
        best2 = C._pick_best(ranked, {}, "map")
        self.assertEqual((best2["label"], best2["source"]), ("c", "grid"))

    def test_tuned_method_and_slug(self):
        self.assertEqual(C._tuned_method("bench/studies/cluster_leiden_tune_2026/best.json"), "leiden")
        self.assertEqual(C._tuned_method("bench/studies/cluster_dbscan_v6_x/best.json"), "dbscan_v6")
        self.assertIsNone(C._tuned_method("bench/studies/cluster_custom/best.json"))
        self.assertEqual(C._slug("siglip2+irra→solider@200"), "siglip2+irra_to_solider_at_200")


class PipelineBestTests(unittest.TestCase):
    def test_write_pipeline_best_changes_only_weights(self):
        src_text = (ROOT / "pipeline.yaml").read_text(encoding="utf-8-sig")
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "pipeline.yaml"
            src.write_text(src_text, encoding="utf-8")
            dst = Path(td) / "pipeline_best.yaml"
            ok, note = C.write_pipeline_best(src, dst, {"siglip2": 0.59, "irra": 0.5, "solider": 1.89})
            self.assertTrue(ok, note)
            out = dst.read_text(encoding="utf-8")
            self.assertIn("bench/combos.py 가 만든 사본", out.splitlines()[0])
            self.assertIn("weight: 0.59", out)
            self.assertIn("weight: 1.89", out)
            # 가중치 줄 외에는 원본과 같다 (주석 보존)
            orig = [l for l in src_text.splitlines() if "weight:" not in l]
            new = [l for l in out.splitlines()[1:] if "weight:" not in l]
            self.assertEqual(orig, new)
            self.assertEqual(out.count("weight: 1.0"), src_text.count("weight: 1.0") - 1)   # dinov2 만 1.0 유지
        self.assertEqual((ROOT / "pipeline.yaml").read_text(encoding="utf-8-sig"), src_text)  # 원본 불변


class ParserTests(unittest.TestCase):
    def test_parser_accepts_all_stage_options(self):
        p = C.build_parser()
        a = p.parse_args(["search", "--validate", "--refine-trials", "10", "--methods", "leiden", "--pools", "200"])
        self.assertEqual((a.cmd, a.validate, a.refine_trials, a.methods), ("search", True, 10, "leiden"))
        a = p.parse_args(["cluster", "--embedders", "irra", "--adopt", "--constraint", "pair_precision>=0.9", "b3_f1>=0.8"])
        self.assertEqual(a.constraint, ["pair_precision>=0.9", "b3_f1>=0.8"])
        a = p.parse_args(["report", "--stage", "search"])
        self.assertEqual(a.stage, "search")


if __name__ == "__main__":
    unittest.main()
