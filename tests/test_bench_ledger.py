"""오프라인 테스트: bench/ledger.py — 엔트리 생성·fingerprint·중복 이관·스크립트별 빌더·표·CLI (모델·Qdrant·git 불필요)."""
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

ENV = {"git_commit": "deadbeef0000", "git_dirty": False, "python": "3.11"}   # env_info()(git 호출) 대신 고정값


def detect_summary():
    return {
        "producer": "detect_eval_prw", "generated_at": "2026-09-26T18:00:00+09:00",
        "inputs": [{"role": "detections:yolo", "path": "x/detections.jsonl", "sha256": "ab", "size": 1}],
        "config": {"methods": {"yolo": "x/detections.jsonl"}, "iou": 0.5, "thresholds": [0.2], "frames": 300},
        "gt": {"boxes": 1341, "boxes_unlabeled_pid": 100, "frames": 300},
        "metas": {"yolo": {"kind": "run", "class_name": "person", "split": {"split": "test", "limit": 300, "every": 1},
                           "detector": {"module": "detect.detectors.yolo26_detector", "class": "YOLO26Detector",
                                        "params": {"weights": "yolo26m.pt", "conf_threshold": 0.05}, "config_conf_threshold": 0.2},
                           "versions": {"torch": "2.11"}, "stats": {"fps": 48.8, "latency_ms_mean": 20.5, "elapsed_sec": 10.0}}},
        "results": {"yolo": {"frames": 300, "gt_boxes": 1341, "detections": 3348, "ap50": 0.871, "ap50_95": 0.499,
                             "max_recall": 0.936, "operating_threshold": 0.2,
                             "operating": {"precision": 0.606, "recall": 0.893, "f1": 0.722, "fp_duplicate": 41,
                                           "fp_background": 737, "fn": 144, "dets_per_frame": 6.58},
                             "recall_by_size": {"<50": {"recall": None, "gt": 0}, "75–119": {"recall": 0.784, "gt": 100}}}},
    }


def cluster_summary(assignments_path):
    return {
        "producer": "prw_cluster_gt_eval", "generated_at": "2026-09-25T10:00:00+09:00",
        "inputs": [{"role": "assignments:L", "path": str(assignments_path), "sha256": "cd", "size": 2}],
        "config": {"methods": {"L": str(assignments_path)}, "iou": 0.5, "min_pid_size": 2, "sources": ["prw_image"],
                   "collection": "forensic_person"},
        "gt": {"evaluated_points": 30919, "pids": 933},
        "results": {"L": {"pairs_noise_as_singletons": {"precision": 0.921, "recall": 0.748, "f1": 0.826},
                          "pairs_clustered_only": {"precision": 0.921, "recall": 0.80, "f1": 0.856},
                          "bcubed": {"precision": 0.9, "recall": 0.81, "f1": 0.851},
                          "purity": 0.95, "inverse_purity": 0.9, "noise_ratio": 0.01, "labeled_points": 30919,
                          "noise_points": 300, "pure_clusters": 900, "mixed_clusters": 138, "mixed_cluster_points": 5000,
                          "pids_split": 3, "pids_all_noise": 0, "clusters_per_pid": 1.2,
                          "ari_noise_as_singletons": 0.9, "nmi_noise_as_singletons": 0.97,
                          "ari_clustered_only": 0.91, "nmi_clustered_only": 0.975}},
    }


def unified_out():
    return {"protocol": "PRW GT crops", "gallery_size": 19127, "query_total": 2057,
            "weights": {"siglip2": 1.0, "irra": 1.5, "solider": 1.5}, "rrf_k": 2, "prefetch": 200, "pool": 200,
            "config": "C:/x/pipeline.yaml", "config_sha256": "c95a", "generated_at": "2026-09-20 15:47:32",
            "results": {"single:solider": {"mAP": 89.06, "Rank-1": 97.08, "Rank-5": 98.69, "Rank-10": 98.98,
                                           "valid_queries": 2057, "pool_recall(%)": 100.0,
                                           "queries_all_positives_in_pool(%)": 100.0, "queries_any_positive_in_pool(%)": 100.0,
                                           "note": "solider 단독", "sec": 8.9},
                        "stage1_rrf": {"mAP": 46.07, "Rank-1": 91.0, "Rank-5": 96.2, "Rank-10": 97.1, "valid_queries": 2057,
                                       "pool_recall(%)": 73.3, "queries_all_positives_in_pool(%)": 12.3,
                                       "queries_any_positive_in_pool(%)": 99.4, "note": "RRF", "sec": 3.0}}}


def embedding_result():
    return {"model": "solider", "gallery_size": 19127, "query_total": 2057, "valid_queries": 2057,
            "mAP": 89.0642, "Rank-1": 97.0831, "Rank-5": 98.6874, "Rank-10": 98.9791}


class EntryTests(unittest.TestCase):
    def test_make_entry_fields_and_run_id(self):
        e = L.make_entry("detect", "t", "yolo", metrics={"ap50": 0.5, "x": None}, params={"k": 1},
                         created_at="2026-09-26T18:00:00+09:00", env=ENV)
        self.assertEqual(e["schema_version"], L.SCHEMA_VERSION)
        self.assertTrue(e["run_id"].startswith("detect_20260926T180000_"))
        self.assertEqual(e["run_id"][-8:], e["fingerprint"][:8])
        self.assertEqual(e["metrics"], {"ap50": 0.5})          # None 은 제거
        self.assertEqual(e["env"], ENV)
        self.assertIsNone(e["report"])

    def test_fingerprint_ignores_time_but_not_metrics(self):
        a = L.make_entry("embed", "t", "m", metrics={"map": 1.0}, created_at="2026-01-01T00:00:00", env=ENV)
        b = L.make_entry("embed", "t", "m", metrics={"map": 1.0}, created_at="2026-02-02T00:00:00", env=ENV)
        c = L.make_entry("embed", "t", "m", metrics={"map": 1.1}, created_at="2026-01-01T00:00:00", env=ENV)
        self.assertEqual(a["fingerprint"], b["fingerprint"])
        self.assertNotEqual(a["run_id"], b["run_id"])
        self.assertNotEqual(a["fingerprint"], c["fingerprint"])
        # 시간 계열 지표(fps, sec …)는 실행마다 달라도 같은 결과 → 지문에서 제외
        d = L.make_entry("detect", "t", "m", metrics={"ap50": 0.5, "fps": 10.0, "latency_ms_mean": 20.0}, env=ENV)
        e = L.make_entry("detect", "t", "m", metrics={"ap50": 0.5, "fps": 33.0, "latency_ms_mean": 9.0}, env=ENV)
        self.assertEqual(d["fingerprint"], e["fingerprint"])

    def test_bad_stage_or_name(self):
        with self.assertRaises(ValueError):
            L.make_entry("track", "t", "x", env=ENV)
        with self.assertRaises(ValueError):
            L.make_entry("detect", "t", "", env=ENV)

    def test_resolve_ledger_path(self):
        self.assertEqual(L.resolve_ledger_path(None), L.DEFAULT_LEDGER)
        self.assertIsNone(L.resolve_ledger_path(""))
        self.assertIsNone(L.resolve_ledger_path("none"))
        self.assertIsNone(L.resolve_ledger_path("x.jsonl", disabled=True))
        self.assertTrue(L.resolve_ledger_path("x.jsonl").is_absolute())

    def test_versions_info_only_loaded_modules(self):
        info = L.versions_info()
        self.assertIsInstance(info, dict)
        self.assertNotIn("ultralytics", info) if "ultralytics" not in sys.modules else None


class IOTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.path = Path(self.td.name) / "sub" / "ledger.jsonl"

    def tearDown(self):
        self.td.cleanup()

    def test_append_read_dedupe(self):
        e1 = L.make_entry("embed", "t", "a", metrics={"map": 1.0}, env=ENV)
        e2 = L.make_entry("embed", "t", "b", metrics={"map": 2.0}, env=ENV)
        self.assertEqual(L.append_entries(self.path, [e1, e2]), (2, 0))
        self.assertEqual([e["name"] for e in L.read_entries(self.path)], ["a", "b"])
        self.assertEqual(L.append_entries(self.path, [e1], dedupe=True), (0, 1))
        self.assertEqual(L.append_entries(self.path, [e1]), (1, 0))       # 재실행 증거는 dedupe 없이 남긴다
        self.assertEqual(len(L.read_entries(self.path)), 3)
        self.assertEqual(L.latest_by_name(L.read_entries(self.path))["a"]["run_id"], e1["run_id"])

    def test_read_skips_corrupt_lines(self):
        self.path.parent.mkdir(parents=True)
        e = L.make_entry("embed", "t", "a", env=ENV)
        self.path.write_text(json.dumps(e) + "\n{bad json\n\n" + json.dumps(e) + "\n", encoding="utf-8")
        errors = []
        self.assertEqual(len(L.read_entries(self.path, errors)), 2)
        self.assertEqual(len(errors), 1)

    def test_record_writes_disabled_and_never_raises(self):
        e = L.make_entry("embed", "t", "a", env=ENV)
        logs = []
        self.assertEqual(L.record(lambda: [e], self.path, log=logs.append), self.path)
        self.assertEqual(len(L.read_entries(self.path)), 1)
        self.assertIn(e["run_id"], logs[-1])
        self.assertIsNone(L.record(lambda: [e], self.path, disabled=True, log=logs.append))
        self.assertIsNone(L.record(lambda: [e], "", log=logs.append))
        self.assertEqual(len(L.read_entries(self.path)), 1)

        def boom():
            raise RuntimeError("x")
        self.assertIsNone(L.record(boom, self.path, log=logs.append))     # 실패 → None (경로를 돌려주지 않는다)
        self.assertIn("기록 실패", logs[-1])

        def bad_log(_msg):
            raise RuntimeError("log broken")
        self.assertIsNone(L.record(boom, self.path, log=bad_log))          # log 가 깨져도 예외를 밖으로 내지 않는다

    def test_append_is_serialized_by_file_lock(self):
        e = L.make_entry("embed", "t", "a", env=ENV)
        with L._FileLock(self.path):
            with self.assertRaises(TimeoutError):
                L._FileLock(self.path, timeout=0.3).__enter__()
        self.assertEqual(L.append_entries(self.path, [e]), (1, 0))       # 잠금 해제 후 정상


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_detect_summary(self):
        (e,) = L.entries_from_detect_summary(detect_summary(), report="r.json", command=["--mode", "score"], env=ENV)
        self.assertEqual((e["stage"], e["name"], e["producer"], e["note"]), ("detect", "yolo", "detect_eval_prw", "run"))
        m = e["metrics"]
        self.assertEqual((m["ap50"], m["precision"], m["fp_background"], m["fps"]), (0.871, 0.606, 737, 48.8))
        self.assertEqual(m["recall_h75_119"], 0.784)
        self.assertNotIn("recall_h_lt50", m)
        self.assertEqual(e["component"]["class"], "YOLO26Detector")
        self.assertEqual((e["params"]["conf_threshold"], e["params"]["operating_threshold"], e["params"]["iou"]), (0.05, 0.2, 0.5))
        self.assertEqual((e["gt"]["dataset"], e["gt"]["split"], e["gt"]["frames"], e["gt"]["boxes"]), ("PRW", "test", 300, 1341))
        self.assertEqual(e["timing"]["elapsed_sec"], 10.0)
        self.assertEqual(e["versions"], {"torch": "2.11"})
        self.assertEqual(e["inputs"][0]["role"], "detections:yolo")
        self.assertEqual(e["report"], "r.json")
        self.assertEqual(e["command"], ["--mode", "score"])
        self.assertTrue(e["run_id"].startswith("detect_20260926T180000_"))

    def test_cluster_summary_with_plugin_report(self):
        a = self.root / "person" / "person_leiden_assignments.jsonl"
        a.parent.mkdir()
        a.write_text("", encoding="utf-8")
        (self.root / "person" / "person_leiden_report.json").write_text(json.dumps({
            "config": {"plugin": {"module": "clustering.methods.leiden", "class": "LeidenClusterer", "params": {"knn": 30}},
                       "knn": 30, "score_threshold": 0.97, "seed": 42, "collection": "c", "query_batch_size": 256,
                       "config_path": "p.yaml", "config_sha256": "s"},
            "stats": {"cluster_sec": 1.5, "graph_build_sec": 2.0, "points": 10}}), encoding="utf-8")
        (e,) = L.entries_from_cluster_summary(cluster_summary(a), report="c.json", env=ENV)
        self.assertEqual((e["stage"], e["name"]), ("cluster", "L"))
        self.assertEqual(e["component"]["class"], "LeidenClusterer")
        self.assertEqual((e["params"]["knn"], e["params"]["score_threshold"], e["params"]["iou"], e["params"]["min_pid_size"]), (30, 0.97, 0.5, 2))
        self.assertNotIn("query_batch_size", e["params"])
        self.assertEqual(e["timing"], {"cluster_sec": 1.5, "graph_build_sec": 2.0})
        self.assertEqual(e["seed"], 42)
        self.assertEqual(e["config"], {"path": "p.yaml", "sha256": "s"})
        self.assertEqual((e["metrics"]["pair_precision"], e["metrics"]["b3_f1"], e["metrics"]["mixed_clusters"], e["metrics"]["ari"]), (0.921, 0.851, 138, 0.9))
        self.assertEqual([i["role"] for i in e["inputs"]], ["assignments:L", "cluster_report"])
        self.assertEqual(e["gt"]["evaluated_points"], 30919)

    def test_cluster_summary_legacy_report_and_missing_report(self):
        a = self.root / "person_leiden_assignments.jsonl"
        a.write_text("", encoding="utf-8")
        (self.root / "person_leiden_report.json").write_text(json.dumps({"config": {"vector": "solider", "knn": 30, "resolution": 1.0}, "stats": {}}), encoding="utf-8")
        (e,) = L.entries_from_cluster_summary(cluster_summary(a), env=ENV)
        self.assertEqual(e["component"], {"method": "leiden", "vector": "solider"})
        self.assertEqual(e["params"]["resolution"], 1.0)
        (e2,) = L.entries_from_cluster_summary(cluster_summary(self.root / "nowhere_assignments.jsonl"), env=ENV)
        self.assertEqual(e2["component"], {})
        self.assertEqual(e2["params"], {"iou": 0.5, "min_pid_size": 2})

    def test_embedding_and_unified(self):
        e = L.entry_from_embedding_result(embedding_result(), "solider", params={"backbone": "swin_base"}, env=ENV)
        self.assertEqual((e["stage"], e["name"], e["component"], e["metrics"]["map"], e["metrics"]["rank1"]),
                         ("embed", "solider", {"model": "solider"}, 89.0642, 97.0831))
        self.assertEqual(e["gt"]["gallery_size"], 19127)
        es = L.entries_from_unified_result(unified_out(), report="u.json", env=ENV, versions={"torch": "2"})
        self.assertEqual([x["name"] for x in es], ["single:solider", "stage1_rrf"])
        self.assertEqual((es[1]["metrics"]["map"], es[1]["metrics"]["pool_recall"], es[1]["note"]), (46.07, 73.3, "RRF"))
        self.assertEqual(es[0]["params"]["weights"]["irra"], 1.5)
        self.assertEqual(es[0]["config"]["sha256"], "c95a")
        self.assertEqual(es[0]["versions"], {"torch": "2"})
        self.assertTrue(es[0]["run_id"].startswith("search_20260920T154732_"))


class ImportAndTableTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.results = self.root / "results"
        (self.results / "detect_prw").mkdir(parents=True)
        (self.results / "cluster_gt").mkdir()
        (self.results / "detect_prw" / "detect_eval_report.json").write_text(json.dumps(detect_summary()), encoding="utf-8")
        a = self.results / "person_leiden_assignments.jsonl"
        a.write_text("", encoding="utf-8")
        (self.results / "cluster_gt" / "cluster_gt_report.json").write_text(json.dumps(cluster_summary(a)), encoding="utf-8")
        (self.results / "embedding_solider.json").write_text(json.dumps(embedding_result()), encoding="utf-8")
        (self.results / "unified_eval.json").write_text(json.dumps(unified_out()), encoding="utf-8")
        (self.results / "unrelated.json").write_text("{}", encoding="utf-8")
        self.ledger = self.root / "ledger.jsonl"

    def tearDown(self):
        self.td.cleanup()

    def test_import_is_idempotent_and_marks_imported(self):
        n, skipped, path = L.import_results(self.results, self.ledger, log=lambda s: None)
        self.assertEqual((n, skipped, path), (5, 0, self.ledger))
        entries = L.read_entries(self.ledger)
        self.assertEqual(sorted(e["stage"] for e in entries), ["cluster", "detect", "embed", "search", "search"])
        self.assertTrue(all("imported_from" in e["env"] for e in entries))
        self.assertEqual(L.import_results(self.results, self.ledger, log=lambda s: None)[:2], (0, 5))

    def test_table_and_list_and_cli(self):
        L.import_results(self.results, self.ledger, log=lambda s: None)
        entries = L.read_entries(self.ledger)
        md = L.render_table(L.filter_entries(entries, stage="search"), "search")
        self.assertIn("| name | created | map | rank1 |", md)
        self.assertIn("| stage1_rrf |", md)
        csv_text = L.render_table(L.filter_entries(entries, stage="detect"), "detect", metrics=["ap50", "fps"], fmt_name="csv")
        self.assertEqual(csv_text.splitlines()[0], "name,created,ap50,fps,run_id")
        self.assertIn("0.8710", csv_text)
        self.assertEqual(len(L.filter_entries(entries, name="solider")), 2)     # embed solider + single:solider
        self.assertEqual(len(L.filter_entries(entries, stage="search", names=["stage1_rrf"])), 1)
        self.assertEqual(len(L.list_lines(entries)), 5)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(L.main(["--ledger", str(self.ledger), "list", "--stage", "detect"]), 0)
            self.assertEqual(L.main(["--ledger", str(self.ledger), "table", "--stage", "embed", "--latest"]), 0)
            run_id = entries[0]["run_id"]
            self.assertEqual(L.main(["--ledger", str(self.ledger), "show", run_id]), 0)
        text = out.getvalue()
        self.assertIn("detect_20260926T180000_", text)
        self.assertIn("| solider |", text)
        self.assertIn('"run_id": "%s"' % run_id, text)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(L.main(["--ledger", str(self.ledger), "show", "nope"]), 1)


class ScriptHookTests(unittest.TestCase):
    """평가 스크립트 4개가 --ledger/--no-ledger 를 받는지 (실행은 하지 않음)."""

    def test_detect_and_cluster_and_unified_parsers(self):
        from eval import detect_eval_prw as de
        from eval import prw_cluster_gt_eval as cg
        from eval import prw_eval_unified as un
        for mod in (de, cg, un):
            args = mod.build_parser().parse_args(["--no-ledger", "--ledger", "x.jsonl"] + (["--method", "a=b"] if mod is cg else []))
            self.assertTrue(args.no_ledger)
            self.assertEqual(args.ledger, "x.jsonl")

    def test_prw_eval_source_has_hook(self):
        src = (ROOT / "eval" / "prw_eval.py").read_text(encoding="utf-8")
        self.assertIn('"--no-ledger"', src)
        self.assertIn("entry_from_embedding_result", src)


if __name__ == "__main__":
    unittest.main()
