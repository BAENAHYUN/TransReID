"""오프라인 테스트: bench/run.py — 인자 인코딩·엔트리→러너 인자 복원·명령 조립(dry-run)·verify 비교 (모델·Qdrant 없음)."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import ledger as L  # noqa: E402
from bench import run as R  # noqa: E402

ENV = {"git_commit": "abc", "host": "h"}


class EncodeTests(unittest.TestCase):
    def test_encode_param_roundtrip_via_json_parse(self):
        def decode(text):
            k, v = text.split("=", 1)
            try:
                return k, json.loads(v)
            except ValueError:
                return k, v
        for key, value in [("weights", "yolo26m.pt"), ("conf", 0.05), ("flag", True), ("n", 3), ("s", "0.5"), ("s2", "true"),
                           ("lst", [1, 2]), ("d", {"a": 1})]:
            self.assertEqual(decode(R.encode_param(key, value)), (key, value), key)

    def test_slug(self):
        self.assertEqual(R.slug("siglip2+irra__solider"), "siglip2+irra__solider")
        self.assertEqual(R.slug("a b/c:d"), "a_b_c_d")

    def test_weights_from_params_missing_file(self):
        w = R.weights_from_params({"weights": "no_such_file.pt", "conf_threshold": 0.2})
        self.assertEqual(list(w), ["weights"])
        self.assertIsNone(w["weights"]["sha256"])


class ArgsFromEntryTests(unittest.TestCase):
    def test_runner_entry_uses_bench_args(self):
        e = L.make_entry("detect", "t", "x", env=ENV, extra={"bench": {"args": {"name": "x", "limit": 3}}})
        self.assertEqual(R.args_from_entry(e), {"name": "x", "limit": 3})

    def test_detect_entry_reconstructed(self):
        e = L.make_entry("detect", "detect_eval_prw", "yolo26m_test", env=ENV,
                         component={"module": "detect.detectors.yolo26_detector", "class": "YOLO26Detector",
                                    "params": {"weights": "yolo26m.pt", "conf_threshold": 0.05, "imgsz": 640}},
                         params={"operating_threshold": 0.2, "iou": 0.5}, gt={"split": "test", "limit": 0, "every": 1})
        a = R.args_from_entry(e)
        self.assertEqual((a["module"], a["cls"], a["conf_threshold"], a["operating_threshold"], a["split"]),
                         ("detect.detectors.yolo26_detector", "YOLO26Detector", 0.05, 0.2, "test"))
        cmd = R.runner_command("detect", a)
        self.assertIn("--param", cmd)
        self.assertIn("weights=yolo26m.pt", cmd)
        self.assertIn("--class", cmd)

    def test_import_qdrant_entry_not_reproducible(self):
        e = L.make_entry("detect", "t", "prod_db", env=ENV, component={"module": "qdrant", "class": "forensic_person"})
        with self.assertRaises(ValueError):
            R.args_from_entry(e)

    def test_cluster_entry_reconstructed(self):
        e = L.make_entry("cluster", "prw_cluster_gt_eval", "L", env=ENV,
                         component={"module": "clustering.methods.leiden", "class": "LeidenClusterer", "params": {"knn": 30}},
                         params={"knn": 30, "score_threshold": 0.97, "seed": 42, "iou": 0.5, "min_pid_size": 2,
                                 "min_cluster_size": 2, "vector": "solider", "max_points": 3000},
                         gt={"sources": ["prw_image"]}, config={"path": "p.yaml"})
        a = R.args_from_entry(e)
        # 플러그인 생성자 인자는 component.params 그대로 — 보고서의 score_threshold(Leiden 은 threshold) / seed 를 섞지 않는다
        self.assertEqual(a["params"], {"knn": 30})
        self.assertEqual((a["target"], a["sources"], a["vector"], a["max_points"], a["config"]), ("person", "prw_image", "solider", 3000, "p.yaml"))
        self.assertNotIn("pid_split", a)
        from clustering.methods.leiden import LeidenClusterer
        LeidenClusterer(**a["params"])                 # 복원 인자가 실제 생성자와 맞는지
        e2 = L.make_entry("cluster", "prw_cluster_gt_eval", "L", env=ENV, component=e["component"],
                          params={**e["params"], "pid_split": "s.json:tune"}, gt={"sources": ["prw_image"], "pid_split": "s.json:tune"})
        self.assertEqual(R.args_from_entry(e2)["pid_split"], "s.json:tune")

    def test_legacy_cluster_entry_rejected(self):
        e = L.make_entry("cluster", "t", "Leiden_exact_0.97", env=ENV, component={"method": "leiden", "vector": "solider"})
        with self.assertRaises(ValueError):
            R.args_from_entry(e)

    def test_embed_search_e2e_entries(self):
        e = L.make_entry("embed", "prw_eval", "solider", env=ENV, component={"model": "solider"}, params={"backbone": "swin_base"}, config={"path": "p.yaml"})
        self.assertEqual(R.args_from_entry(e), {"model": "solider", "config": "p.yaml", "backbone": "swin_base"})
        s = L.make_entry("search", "u", "stage1_rrf", env=ENV, params={"weights": {"a": 1.0}, "rrf_k": 2, "prefetch": 200, "pool": 200})
        a = R.args_from_entry(s)
        self.assertEqual((a["weights"], a["rrf_k"], a["variant"]), ({"a": 1.0}, 2, "stage1_rrf"))
        self.assertIn("a=1.0", R.runner_command("search", a))
        x = L.make_entry("e2e", "e", "siglip2+irra__solider", env=ENV, component={"stage1": ["siglip2", "irra"], "rerank": "solider"},
                         params={"limit": 200, "pool": 200, "gallery": "test"})
        a = R.args_from_entry(x)
        self.assertEqual((a["stage1"], a["rerank"], a["limit"]), (["siglip2", "irra"], "solider", 200))
        cmd = R.runner_command("e2e", a)
        self.assertEqual(cmd[cmd.index("--stage1") + 1:cmd.index("--stage1") + 3], ["siglip2", "irra"])


class DryRunTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.runs = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def run_dry(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = R.main([*argv, "--dry-run", "--runs-dir", str(self.runs), "--no-ledger"])
        return code, out.getvalue()

    def test_detect_dry_run_commands(self):
        code, text = self.run_dry(["detect", "--config", "pipeline_tracking_yolo26.yaml", "--name", "y", "--limit", "10",
                                   "--param", "imgsz=320", "--operating-threshold", "0.5", "--compare", "a=b.jsonl"])
        self.assertEqual(code, 0)
        self.assertIn("--mode run --name y --split test --limit 10 --every 1", text)
        self.assertIn("--detector-config pipeline_tracking_yolo26.yaml", text)
        self.assertIn("--param imgsz=320", text)
        self.assertIn("--no-score --no-ledger", text)
        self.assertIn("--mode score --method y=", text)
        self.assertIn("a=b.jsonl", text)
        self.assertIn("--operating-threshold 0.5", text)
        self.assertIn("ledger_part.jsonl", text)

    def test_cluster_dry_run_commands(self):
        code, text = self.run_dry(["cluster", "--method", "dbscan_v6", "--param", "eps=0.1", "--max-points", "500", "--name", "d"])
        self.assertEqual(code, 0)
        self.assertIn("clustering/cluster_qdrant.py", text)
        self.assertIn("--method dbscan_v6", text)
        self.assertIn("--param eps=0.1", text)
        self.assertIn("--max-points 500", text)
        self.assertIn("prw_cluster_gt_eval.py --method d=", text)
        self.assertIn("--no-images", text)

    def test_embed_search_e2e_dry_run(self):
        code, text = self.run_dry(["embed", "--model", "irra"])
        self.assertEqual(code, 0)
        self.assertIn("eval/prw_eval.py --model irra", text)
        code, text = self.run_dry(["search", "--weights", "siglip2=1,irra=2", "--rrf-k", "3"])
        self.assertEqual(code, 0)
        self.assertIn("--weights siglip2=1.0,irra=2.0", text)
        self.assertIn("--rrf-k 3.0", text)
        code, text = self.run_dry(["e2e", "--stage1", "solider", "--rerank", "none", "--limit", "50"])
        self.assertEqual(code, 0)
        self.assertIn("eval/prw_e2e_search_eval.py", text)
        self.assertIn("--stage1 solider --rerank none", text)
        self.assertIn("--limit 50", text)


class VerifyTests(unittest.TestCase):
    def test_compare_metrics(self):
        status, rows = R.compare_metrics({"ap50": 0.876, "f1": 0.83, "fps": 39.0, "x": "s"},
                                         {"ap50": 0.8745, "f1": 0.84, "fps": 20.0}, {"ap50": 0.005, "f1": 0.005, "nope": 1.0})
        self.assertEqual(status, "FAIL")
        by = {r["metric"]: r for r in rows}
        self.assertTrue(by["ap50"]["ok"])
        self.assertFalse(by["f1"]["ok"])
        self.assertNotIn("fps", by)
        self.assertNotIn("nope", by)
        self.assertEqual(R.compare_metrics({"map": 89.06}, {"map": 89.15}, {"map": 0.1})[0], "PASS")
        # 원본에 있던 지표가 재실행에 없으면 FAIL, 비교 가능한 지표가 없으면 UNVERIFIED (PASS 아님)
        status, rows = R.compare_metrics({"map": 89.06, "rank1": 97.0}, {"map": 89.1}, {"map": 0.1, "rank1": 0.2})
        self.assertEqual(status, "FAIL")
        self.assertEqual([r for r in rows if r["metric"] == "rank1"][0]["after"], None)
        self.assertEqual(R.compare_metrics({"fps": 10}, {"fps": 12}, {"map": 0.1})[0], "UNVERIFIED")
        self.assertEqual(R.compare_metrics({}, {}, {"map": 0.1})[0], "UNVERIFIED")

    def test_zero_is_kept(self):
        self.assertFalse(R._empty(0))
        self.assertFalse(R._empty(0.0))
        self.assertTrue(R._empty(False))
        self.assertTrue(R._empty(""))
        self.assertTrue(R._empty([]))
        cmd = R.runner_command("detect", {"operating_threshold": 0.0, "limit": 0, "name": ""})
        self.assertIn("--operating-threshold", cmd)
        self.assertIn("0.0", cmd)
        self.assertIn("--limit", cmd)
        self.assertNotIn("--name", cmd)

    def test_find_entry_exact_and_prefix(self):
        a = L.make_entry("embed", "t", "a", env=ENV, created_at="2026-01-01T00:00:00")
        b = L.make_entry("embed", "t", "b", env=ENV, created_at="2026-01-02T00:00:00")
        self.assertIs(R.find_entry([a, b], b["run_id"]), b)
        self.assertIs(R.find_entry([a, b], b["run_id"][:-3]), b)
        with self.assertRaises(SystemExit):
            R.find_entry([a, b], "embed_")          # 둘 다 맞음
        with self.assertRaises(SystemExit):
            R.find_entry([a, b], "nope")

    def test_verify_full_path_with_fake_stage(self):
        """run_stage 를 가짜로 바꿔 verify 의 비교·verify.json·원장 기록 경로를 끝까지 태운다."""
        td = tempfile.TemporaryDirectory()
        orig = R.run_stage
        try:
            ledger_path = Path(td.name) / "ledger.jsonl"
            target = L.make_entry("cluster", "prw_cluster_gt_eval", "leiden_t3000", env=ENV,
                                  component={"module": "clustering.methods.leiden", "class": "LeidenClusterer", "params": {"knn": 30}},
                                  params={"knn": 30, "iou": 0.5, "min_pid_size": 2, "min_cluster_size": 2, "vector": "solider", "max_points": 3000},
                                  gt={"sources": ["prw_image"]}, metrics={"pair_f1": 0.80, "b3_f1": 0.85, "purity": 0.9})
            L.append_entries(ledger_path, [target])
            calls = []

            def fake_run_stage(stage, a, runs_dir, dry, echo=print):
                calls.append((stage, dict(a)))
                run_dir = Path(runs_dir) / "fake"
                run_dir.mkdir(parents=True, exist_ok=True)
                e = L.make_entry("cluster", "prw_cluster_gt_eval", a["name"], env=ENV, component=target["component"],
                                 params=target["params"], gt=target["gt"], metrics={"pair_f1": 0.803, "b3_f1": 0.849, "purity": 0.95})
                other = L.make_entry("cluster", "prw_cluster_gt_eval", "other_variant", env=ENV, metrics={"pair_f1": 0.1})
                return [e, other], run_dir

            R.run_stage = fake_run_stage
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = R.main(["verify", target["run_id"], "--ledger", str(ledger_path), "--runs-dir", td.name])
            text = out.getvalue()
            self.assertEqual(code, 1, text)                              # purity Δ 0.05 > 0.005 → FAIL
            self.assertEqual(calls[0][0], "cluster")
            self.assertEqual(calls[0][1]["name"], "leiden_t3000__verify")
            self.assertIn("[verify] FAIL", text)
            self.assertIn("NG  purity", text)
            self.assertIn("ok  pair_f1", text)
            entries = L.read_entries(ledger_path)
            self.assertEqual(len(entries), 3)
            compared = [e for e in entries if e["name"] == "leiden_t3000__verify"][0]
            other = [e for e in entries if e["name"] == "other_variant"][0]
            v = compared["verify"]
            self.assertEqual((v["against"], v["status"], v["passed"], v["same_git_commit"], v["same_host"]), (target["run_id"], "FAIL", False, True, True))
            self.assertEqual({r["metric"] for r in v["rows"]}, {"pair_f1", "b3_f1", "purity"})
            self.assertNotIn("verify", other)                            # 동반 결과에는 verify 판정을 붙이지 않는다
            self.assertIn("동반 결과", other["note"])
            self.assertTrue((Path(td.name) / "fake" / "verify.json").is_file())
            self.assertIn("verify of", compared["note"])
            # 허용 오차 덮어쓰기 → PASS (--tol 은 여러 값)
            with contextlib.redirect_stdout(io.StringIO()):
                code = R.main(["verify", target["run_id"], "--ledger", str(ledger_path), "--runs-dir", td.name, "--tol", "purity=0.1", "b3_f1=0.01"])
            self.assertEqual(code, 0)
            # 재실행 결과에 비교 대상 이름이 없으면 UNVERIFIED (exit 1, PASS 아님)
            R.run_stage = lambda stage, a, runs_dir, dry, echo=print: ([L.make_entry("cluster", "x", "zzz", env=ENV, metrics={"pair_f1": 1.0})], Path(runs_dir) / "fake")
            with contextlib.redirect_stdout(out):
                code = R.main(["verify", target["run_id"], "--ledger", str(ledger_path), "--runs-dir", td.name])
            self.assertEqual(code, 1)
            self.assertIn("UNVERIFIED", out.getvalue())
        finally:
            R.run_stage = orig
            td.cleanup()

    def test_verify_dry_run_reconstructs_and_renames(self):
        td = tempfile.TemporaryDirectory()
        try:
            ledger_path = Path(td.name) / "ledger.jsonl"
            e = L.make_entry("detect", "detect_eval_prw", "yolo26m_test", env=ENV,
                             component={"module": "detect.detectors.yolo26_detector", "class": "YOLO26Detector",
                                        "params": {"weights": "yolo26m.pt", "conf_threshold": 0.05}},
                             params={"operating_threshold": 0.2}, gt={"split": "test", "limit": 300, "every": 1},
                             metrics={"ap50": 0.871})
            L.append_entries(ledger_path, [e])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = R.main(["verify", e["run_id"], "--ledger", str(ledger_path), "--dry-run", "--runs-dir", td.name])
            text = out.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("--name yolo26m_test__verify", text)
            self.assertIn("--limit 300", text)
            self.assertIn("--mode run", text)
            with contextlib.redirect_stdout(out):
                self.assertEqual(R.main(["show-cmd", e["run_id"], "--ledger", str(ledger_path)]), 0)
            self.assertIn("bench/run.py detect", out.getvalue())
        finally:
            td.cleanup()


if __name__ == "__main__":
    unittest.main()
