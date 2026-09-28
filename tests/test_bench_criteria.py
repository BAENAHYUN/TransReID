"""오프라인 테스트: bench/criteria.py — 채택 기준 판정과 채택→yaml 생성 (임시 폴더; 원본 yaml 불변)."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import criteria as C  # noqa: E402
from bench import ledger as L  # noqa: E402

ENV = {"git_commit": "x", "host": "h"}


def detect_entry(**over):
    metrics = {"ap50": 0.876, "max_recall": 0.95, "recall_h75_119": 0.66, "recall_h120_199": 0.864, "fps": 15.6}
    metrics.update(over)
    return L.make_entry("detect", "detect_eval_prw", "rfdetr_medium_test", env=ENV,
                        component={"module": "detect.detectors.rfdetr_detector", "class": "RFDETRDetector", "params": {"conf_threshold": 0.05, "filter_forensic": True}},
                        params={"operating_threshold": 0.54, "iou": 0.5}, metrics=metrics, report="r.json")


class EvaluateTests(unittest.TestCase):
    def test_status_levels(self):
        self.assertEqual(C.evaluate(detect_entry())["status"], "pass")
        self.assertEqual(C.evaluate(detect_entry(recall_h75_119=0.559, recall_h120_199=0.815))["status"], "partial")
        self.assertEqual(C.evaluate(detect_entry(ap50=0.5, max_recall=0.5, recall_h75_119=0.1, recall_h120_199=0.1, fps=1))["status"], "fail")
        e = L.make_entry("embed", "t", "x", env=ENV, metrics={"rank1": 97.0})          # map 없음 → 기준 없음
        self.assertEqual(C.evaluate(e)["status"], "n/a")
        r = C.evaluate(L.make_entry("cluster", "t", "L", env=ENV, metrics={"pair_precision": 0.95, "b3_f1": 0.86, "mixed_clusters": 100}))
        self.assertEqual((r["status"], r["passed"], r["applicable"]), ("pass", 3, 3))
        r = C.evaluate(L.make_entry("search", "t", "s", env=ENV, metrics={"map": 89.4, "pool_recall": 70.0}))
        self.assertEqual(r["status"], "partial")
        self.assertEqual([c["ok"] for c in r["checks"]], [True, False])

    def test_chart_axes_and_rules_cover_all_stages(self):
        for s in L.STAGES:
            self.assertIn(s, C.ADOPTION_RULES)
            self.assertIn(s, C.CHART_AXES)


class AdoptTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.tracking = self.root / "pipeline_tracking.yaml"
        shutil.copyfile(ROOT / "pipeline_tracking.yaml", self.tracking)
        self.pipeline = self.root / "pipeline.yaml"
        shutil.copyfile(ROOT / "pipeline.yaml", self.pipeline)
        self.orig_tracking = (ROOT / "pipeline_tracking.yaml").read_bytes()
        self.orig_pipeline = (ROOT / "pipeline.yaml").read_bytes()

    def tearDown(self):
        self.assertEqual((ROOT / "pipeline_tracking.yaml").read_bytes(), self.orig_tracking)   # 원본 불변
        self.assertEqual((ROOT / "pipeline.yaml").read_bytes(), self.orig_pipeline)
        self.td.cleanup()

    def test_adopt_detect(self):
        import yaml
        res = C.adopt_yaml(detect_entry(), root=self.root, tracking_template=self.tracking)
        self.assertEqual(res["kind"], "detect")
        out = res["files"][0]
        self.assertEqual(out.name, "pipeline_tracking_rfdetr_medium_test.yaml")
        d = yaml.safe_load(out.read_text(encoding="utf-8"))
        self.assertEqual(d["detector"]["class"], "RFDETRDetector")
        self.assertEqual(d["detector"]["params"]["conf_threshold"], 0.54)      # 운영 임계값 = 채점 동작점
        self.assertIn("tracker", d)                                             # tracker/stitcher 는 템플릿에서
        with self.assertRaises(FileExistsError):
            C.adopt_yaml(detect_entry(), root=self.root, tracking_template=self.tracking)
        C.adopt_yaml(detect_entry(), root=self.root, tracking_template=self.tracking, overwrite=True)
        db = L.make_entry("detect", "t", "prod_db", env=ENV, component={"module": "qdrant", "class": "forensic_person"})
        self.assertEqual(C.adopt_yaml(db, root=self.root)["files"], [])

    def test_adopt_cluster_plugin_and_study_params(self):
        import yaml
        e = L.make_entry("cluster", "bench.optimize", "study:leiden_tune", env=ENV,
                         component={"module": "clustering.methods.leiden", "class": "LeidenClusterer", "params": {}},
                         params={"knn": 33, "threshold": 0.95, "mutual_knn": False, "max_cluster_size": 500, "vector": "solider", "iou": 0.5})
        res = C.adopt_yaml(e, root=self.root)
        out = res["files"][0]
        self.assertEqual(out.name, "clusterer_study_leiden_tune.yaml")
        d = yaml.safe_load(out.read_text(encoding="utf-8"))
        self.assertEqual(d["clusterer"]["params"], {"knn": 33, "threshold": 0.95, "mutual_knn": False, "max_cluster_size": 500})   # iou/vector 는 생성자 인자 아님
        self.assertIn("primary vector: solider", out.read_text(encoding="utf-8"))
        legacy = L.make_entry("cluster", "t", "Leiden_exact_0.97", env=ENV, component={"method": "leiden", "vector": "solider"})
        self.assertEqual(C.adopt_yaml(legacy, root=self.root)["files"], [])

    def test_adopt_search_combo(self):
        e = L.make_entry("search", "bench.combos", "combo:solider→none@1000", env=ENV, component={"axes": {"stage1": "solider", "rerank": "none", "pool": 1000}},
                         params={"stage1": "solider", "rerank": "none", "w_solider": 1.89, "w_irra": 0.5, "rrf_k": 3.8, "prefetch": 850, "pool": 1000})
        res = C.adopt_yaml(e, root=self.root, pipeline_path=self.pipeline, name="best_search")
        names = sorted(p.name for p in res["files"])
        self.assertEqual(names, ["pipeline_best_search.search.json", "pipeline_best_search.yaml"])
        side = json.loads((self.root / "pipeline_best_search.search.json").read_text(encoding="utf-8"))
        self.assertEqual((side["stage1"], side["rerank"], side["pool"], side["weights"]), (["solider"], None, 1000, {"solider": 1.89}))
        self.assertIn("weight: 1.89", (self.root / "pipeline_best_search.yaml").read_text(encoding="utf-8"))

    def test_adopt_embed_has_no_yaml(self):
        e = L.make_entry("embed", "prw_eval", "solider", env=ENV, component={"model": "solider"}, metrics={"map": 89.06})
        res = C.adopt_yaml(e, root=self.root)
        self.assertEqual(res["files"], [])
        self.assertIn("pipeline.yaml", res["note"])

    def test_reproduce_command(self):
        cmd = C.reproduce_command(detect_entry())
        self.assertIn("bench/run.py detect", cmd)
        self.assertIn("--class RFDETRDetector", cmd)


if __name__ == "__main__":
    unittest.main()
