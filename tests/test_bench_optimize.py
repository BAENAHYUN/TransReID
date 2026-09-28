"""오프라인 테스트: bench/optimize.py — 탐색 공간·제약·Pareto·Optuna 스터디(합성 문제)·검색 목적함수(합성 sims)·검출 스윕 (Qdrant·GPU 없음)."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import optimize as O  # noqa: E402


class SpaceTests(unittest.TestCase):
    def test_parse_space(self):
        s = O.parse_space(["knn=int:10:60", "thr=float:0.9:0.99", "k=float:1:100:log", "m=cat:true,false", "pool=int:100:1000:step50", "s=cat:a,b"])
        self.assertEqual(s["knn"], {"type": "int", "low": 10, "high": 60})
        self.assertEqual(s["thr"], {"type": "float", "low": 0.9, "high": 0.99})
        self.assertTrue(s["k"]["log"])
        self.assertEqual(s["m"]["choices"], [True, False])
        self.assertEqual(s["pool"]["step"], 50)
        self.assertEqual(s["s"]["choices"], ["a", "b"])
        with self.assertRaises(ValueError):
            O.parse_space(["bad"])
        with self.assertRaises(ValueError):
            O.parse_space(["x=weird:1:2"])

    def test_constraints_and_violations(self):
        c = O.parse_constraints(["pair_precision>=0.90", "noise<=0.1", "x>1"])
        self.assertEqual(c[0], ("pair_precision", ">=", 0.9))
        v = O.violations({"pair_precision": 0.85, "noise": 0.05}, c)
        self.assertAlmostEqual(v["pair_precision"], 0.05)
        self.assertEqual(v["noise"], 0.0)
        self.assertEqual(v["x"], float("inf"))          # 지표 없음 = 위반
        with self.assertRaises(ValueError):
            O.parse_constraints(["nope"])

    def test_pareto_front(self):
        rows = [{"metrics": {"a": 1.0, "b": 0.5}, "feasible": True}, {"metrics": {"a": 0.9, "b": 0.9}, "feasible": True},
                {"metrics": {"a": 0.8, "b": 0.8}, "feasible": True}, {"metrics": {"a": 0.5, "b": 1.0}, "feasible": False},
                {"metrics": {"a": None, "b": 1.0}}]
        front = O.pareto_front(rows, "a", "b")
        self.assertEqual([r["metrics"]["a"] for r in front], [1.0, 0.9, 0.5])


class QuadraticProblem(O.Problem):
    stage = "cluster"
    name = "quad"
    baseline = {"x": 0.0, "y": 0.0}

    def evaluate(self, params, part=None):
        x, y = float(params["x"]), float(params["y"])
        val = 1.0 - (x - 0.6) ** 2 - (y - 0.3) ** 2
        return {"b3_f1": val + (0.01 if part == "holdout" else 0.0), "pair_precision": 0.95 - abs(x - 0.6), "cluster_sec": 0.0}

    def recommended_yaml(self, params):
        return f"x: {params['x']}\n"


class StudyTests(unittest.TestCase):
    def test_run_study_finds_optimum_and_writes_trials(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "study"
            space = {"x": {"type": "float", "low": 0.0, "high": 1.0}, "y": {"type": "float", "low": 0.0, "high": 1.0}}
            logs = []
            study = O.run_study(QuadraticProblem(), space, 30, "b3_f1", [("pair_precision", ">=", 0.9)], out, seed=1, log=logs.append)
            self.assertEqual(len(study["rows"]), 30)
            best = study["best"]
            self.assertTrue(best["feasible"])
            self.assertGreater(best["value"], 0.9)                      # 최적값 1.0 근처
            self.assertLess(abs(best["params"]["x"] - 0.6), 0.1)        # 제약 |x-0.6| ≤ 0.05 안
            lines = (out / "trials.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 30)
            row = json.loads(lines[0])
            self.assertEqual(row["params"], {"x": 0.0, "y": 0.0})      # 첫 trial = 운영값(enqueue)
            self.assertFalse(row["feasible"])                            # x=0 → pair_precision 0.35 < 0.9
            self.assertTrue((out / "optuna.db").is_file())
            self.assertTrue(any("trial" in s for s in logs))

    def test_report_and_validation_summary(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "study"
            out.mkdir()
            p = QuadraticProblem()
            rows = [{"number": 0, "params": {"x": 0.6, "y": 0.3}, "metrics": p.evaluate({"x": 0.6, "y": 0.3}), "feasible": True, "violations": {}, "value": 1.0, "penalized": 1.0, "sec": 0},
                    {"number": 1, "params": {"x": 0.0, "y": 0.0}, "metrics": p.evaluate({"x": 0.0, "y": 0.0}), "feasible": False, "violations": {"pair_precision": 0.55}, "value": 0.55, "penalized": -54, "sec": 0}]
            study = {"rows": rows, "best": rows[0], "feasible": 1, "study_name": "s"}
            base = p.evaluate(p.baseline)
            validation = {"holdout_split": "f:holdout", "best": p.evaluate(rows[0]["params"], "holdout"), "baseline": p.evaluate(p.baseline, "holdout"),
                          "delta": 0.45, "delta_tune": 0.45, "consistent": True}
            path = O.write_report(p, study, "b3_f1", [("pair_precision", ">=", 0.9)], base, out, validation, O.pareto_front(rows, "b3_f1", "pair_precision"))
            text = path.read_text(encoding="utf-8")
            self.assertIn("운영값", text)
            self.assertIn("holdout 검증", text)
            self.assertIn("잡음 초과", text)
            self.assertIn("Pareto", text)


class SearchProblemTests(unittest.TestCase):
    def test_evaluate_on_synthetic_sims(self):
        rng = np.random.default_rng(0)
        G, Q, D = 120, 12, 8
        g_pids = np.repeat(np.arange(6), 20)
        g_frames = np.array([f"f{i % 10}" for i in range(G)])
        q_pids = np.arange(6).repeat(2)
        q_frames = np.array([f"q{i}" for i in range(Q)])
        # 같은 pid 끼리 비슷한 벡터
        centers = rng.normal(size=(6, D))
        g = centers[g_pids] + 0.3 * rng.normal(size=(G, D))
        q = centers[q_pids] + 0.3 * rng.normal(size=(Q, D))
        from eval.prw_eval_unified import l2n
        sims = {m: (l2n(q) @ l2n(g).T).astype(np.float32) for m in ("siglip2", "irra", "solider")}
        prob = O.SearchProblem.__new__(O.SearchProblem)
        prob.sims = sims
        prob.meta = dict(g_pids=g_pids, g_frames=g_frames, q_pids=q_pids, q_frames=q_frames)
        prob.parts = {"tune": {0, 1, 2}, "holdout": {3, 4, 5}}
        prob.tune_part = "tune"
        prob.baseline = {"stage1": "siglip2+irra", "rerank": "solider", "w_siglip2": 1.0, "w_irra": 1.5, "w_solider": 1.5, "rrf_k": 2.0, "prefetch": 50, "pool": 50}
        m = prob.evaluate(prob.baseline)
        self.assertEqual(m["valid_queries"], 6)                          # tune 인물 3 × 쿼리 2
        self.assertGreater(m["map"], 50.0)
        self.assertEqual(m["pool_recall"], 100.0)                        # pool 50 ≥ 양성 20
        m2 = prob.evaluate({"stage1": "solider", "rerank": "none", "pool": 10, "prefetch": 10})
        self.assertLess(m2["pool_recall"], 100.0)                        # pool 10 < 양성 20
        mh = prob.evaluate(prob.baseline, part="holdout")
        self.assertEqual(mh["valid_queries"], 6)
        y = prob.recommended_yaml({"stage1": "irra", "rerank": "none", "w_irra": 1.2, "rrf_k": 3.0, "prefetch": 100, "pool": 100})
        self.assertIn("irra", y)


class DetectSweepTests(unittest.TestCase):
    def test_sweep_study_with_fake_problem(self):
        class FakeDetect:
            stage = "detect"
            name = "fake"
            baseline = {"conf_threshold": 0.5}

            def sweep(self):
                rows = []
                for t in (0.2, 0.3, 0.4, 0.5, 0.6):
                    p, r = 0.4 + t, 1.0 - t
                    rows.append({"params": {"conf_threshold": t}, "metrics": {"precision": p, "recall": r, "f1": 2 * p * r / (p + r), "ap50": 0.9, "max_recall": 0.95}})
                return rows

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "sweep"
            logs = []
            study = O.sweep_study(FakeDetect(), "f1", [("recall", ">=", 0.65)], out, log=logs.append)
            self.assertEqual(len(study["rows"]), 5)
            self.assertEqual(study["feasible"], 2)                       # t=0.2, 0.3 만 recall ≥ 0.65
            self.assertEqual(study["best"]["params"]["conf_threshold"], 0.3)
            self.assertTrue((out / "trials.jsonl").is_file())


class CliTests(unittest.TestCase):
    def test_parser_defaults(self):
        p = O.build_parser()
        a = p.parse_args(["cluster", "--method", "dbscan_v6", "--trials", "5"])
        self.assertEqual((a.stage, a.method, a.trials), ("cluster", "dbscan_v6", 5))
        self.assertTrue(a.pid_split.endswith(":tune"))
        a = p.parse_args(["detect", "--detections", "x.jsonl", "--constraint", "recall>=0.85"])
        self.assertEqual(a.constraint, ["recall>=0.85"])
        self.assertEqual(O.DEFAULT_OBJECTIVE["search"], "map")
        self.assertIn("threshold", O.DEFAULT_SPACES["leiden"])           # Leiden 생성자 인자 이름
        # GUI 는 단계와 무관하게 같은 인자를 넘긴다 → 모든 하위 명령이 받아야 한다
        a = p.parse_args(["cluster", "--detections", "x.jsonl", "--cache-dir", "c", "--param", "knn=30", "threshold=0.96", "--validate"])
        self.assertEqual((a.stage, a.detections, a.param, a.validate), ("cluster", "x.jsonl", ["knn=30", "threshold=0.96"], True))
        a = p.parse_args(["search", "--method", "leiden", "--max-points", "0"])
        self.assertEqual(a.stage, "search")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            O.main(["detect", "--no-ledger"])                           # detect 는 --detections 필수


if __name__ == "__main__":
    unittest.main()
