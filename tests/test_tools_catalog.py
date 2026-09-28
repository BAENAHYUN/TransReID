"""도구 카탈로그·선택: combo 이름 해석, ★ 선정(상태 우선 → 지표), e2e 우선, 검출기 후보(yaml)·클러스터러 후보, 선택 저장과 폼 기본값 적용."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui import tool_choice as TCH  # noqa: E402
from gui import tools_catalog as TC  # noqa: E402


def entry(name, stage, metrics, component=None):
    return {"stage": stage, "name": name, "metrics": metrics, "component": component or {}, "params": {}, "generated_at": "2026-09-28T00:00:00"}


class CatalogTests(unittest.TestCase):
    def test_parse_combo_and_label(self):
        v = TC.parse_combo("combo:irra+solider→solider@1000")
        self.assertEqual(v, {"stage1": ["irra", "solider"], "rerank": "solider", "pool": 1000})
        self.assertEqual(TC.parse_combo("combo:solider→none@200")["rerank"], None)
        self.assertIsNone(TC.parse_combo("single:solider"))
        self.assertEqual(TC.combo_label(v), "IRRA + SOLIDER → SOLIDER · 후보 1000")

    def test_best_entry_prefers_status_then_metric(self):
        latest = {
            "a": entry("a", "detect", {"ap50": 0.90, "max_recall": 0.5}, {"class": "X"}),
            "b": entry("b", "detect", {"ap50": 0.86, "max_recall": 0.95}, {"class": "X"}),
            "study:c": entry("study:c", "detect", {"ap50": 0.99, "max_recall": 0.99}, {"class": "X"}),
        }
        orig = TC.status_of
        TC.status_of = lambda e: {"a": "fail", "b": "pass", "study:c": "pass"}[e["name"]]   # 채택 판정은 고정해 두고 순서 규칙만 검사
        try:
            best = TC.best_entry(latest, "detect", lambda e: True)
            self.assertEqual(best["name"], "b")                                # 통과가 지표보다 우선, study 제외
            TC.status_of = lambda e: "pass"
            self.assertEqual(TC.best_entry(latest, "detect", lambda e: True)["name"], "a")   # 상태가 같으면 지표
        finally:
            TC.status_of = orig

    def test_search_candidates_prefer_e2e(self):
        latest = {
            "combo:solider→none@1000": entry("combo:solider→none@1000", "search", {"map": 89.4, "pool_recall": 0.9}),
            "combo:siglip2+irra→solider@200": entry("combo:siglip2+irra→solider@200", "search", {"map": 88.9, "pool_recall": 0.9}),
            "prod_e2e": entry("prod_e2e", "e2e", {"map": 58.3, "det_ceiling": 0.9, "map_db": 63.0}, {"stage1": ["siglip2", "irra"], "rerank": "solider"}),
            "solider_only": entry("solider_only", "e2e", {"map": 54.2, "det_ceiling": 0.9, "map_db": 60.0}, {"stage1": ["solider"], "rerank": None}),
        }
        rows = TC.search_candidates(latest)
        self.assertEqual(rows[0]["value"]["stage1"], ["siglip2", "irra"])       # e2e 58.3 이 1위 (검색 단계 89.4 보다 우선)
        self.assertIn("(운영 기본)", rows[0]["label"])
        self.assertEqual(rows[0]["metric"], "e2e map")
        self.assertEqual(rows[1]["value"]["stage1"], ["solider"])
        self.assertIn("검색 단계 mAP 89.40", rows[1]["detail"])

    def test_detector_candidates_from_yaml(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "pipeline.yaml").write_text("detector:\n  module: detect.detectors.rfdetr_detector\n  class: RFDETRDetector\n  params: {conf_threshold: 0.5}\n", encoding="utf-8")
            (root / "pipeline_tracking_yolo26s.yaml").write_text("detector:\n  module: detect.detectors.yolo26_detector\n  class: YOLO26Detector\n  params: {weights: yolo26s.pt}\n", encoding="utf-8")
            cands = TC.detector_candidates(root)
            self.assertEqual([(c["key"], c["class"], c["token"]) for c in cands],
                             [("pipeline.yaml", "RFDETRDetector", ""), ("pipeline_tracking_yolo26s.yaml", "YOLO26Detector", "yolo26s")])
            m = TC._detector_match(cands[1])
            self.assertTrue(m(entry("yolo26s_test", "detect", {}, {"class": "YOLO26Detector"})))
            self.assertFalse(m(entry("yolo26m_test", "detect", {}, {"class": "YOLO26Detector"})))
            self.assertFalse(m(entry("rfdetr", "detect", {}, {"class": "RFDETRDetector"})))

    def test_live_catalog_has_star_per_step(self):
        cat = TC.catalog()
        for step in ("detector", "embedder", "search", "clusterer", "qwen"):
            self.assertIn(step, cat)
            self.assertTrue(cat[step]["candidates"], step)
            self.assertIn(cat[step]["best_key"], [c["key"] for c in cat[step]["candidates"]], step)
        self.assertEqual(cat["clusterer"]["best_key"], "Leiden (기본)")
        vt = cat["video_tracking"]["candidates"]
        self.assertGreaterEqual(len(vt), 4)
        self.assertTrue(all((ROOT / c["value"]).is_file() for c in vt))                       # 조합 yaml 이 실제로 있다
        self.assertEqual(cat["video_tracking"]["best_key"], vt[0]["key"])                     # 성적 없음 → 첫 항목(기본)
        self.assertEqual(cat["embedder"]["best_key"], "solider")
        self.assertEqual(cat["search"]["candidates"][0]["value"]["stage1"], ["siglip2", "irra"])     # e2e 운영 조합이 ★


class ChoiceTests(unittest.TestCase):
    def test_set_get_apply(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "choice.json"
            self.assertIsNone(TCH.get("detector", p))
            TCH.set_choice("detector", "pipeline_tracking_yolo26s.yaml", "pipeline_tracking_yolo26s.yaml", "YOLO26Detector", p)
            self.assertEqual(TCH.get("detector", p)["key"], "pipeline_tracking_yolo26s.yaml")
            spec = {"flag": "--detector-config", "type": "choice", "default": "pipeline.yaml"}
            self.assertEqual(TCH.apply_spec("image_detect", spec, p)["default"], "pipeline_tracking_yolo26s.yaml")
            self.assertEqual(TCH.apply_spec("image_build", spec, p)["default"], "pipeline.yaml")     # 다른 단계는 그대로
            TCH.set_choice("video_tracking", "YOLO26m + BoT-SORT + SUSHI", "pipeline_tracking_yolo26.yaml", "YOLO26m + BoT-SORT + SUSHI", p)
            vspec = {"flag": "--tracking-config", "type": "choice", "default": "pipeline.yaml"}
            self.assertEqual(TCH.apply_spec("video_preprocess", vspec, p)["default"], "pipeline_tracking_yolo26.yaml")   # value(yaml) 가 인자값
            self.assertEqual(TCH.tool_label("video_preprocess", p), "YOLO26m + BoT-SORT + SUSHI")
            TCH.set_choice("clusterer", "DBSCAN v6", "DBSCAN v6", "DBSCAN v6", p)
            self.assertEqual(TCH.tool_label("image_cluster", p), "DBSCAN v6")
            TCH.set_choice("search", "x", {"stage1": ["solider"], "rerank": None, "pool": 1000}, "x", p)
            self.assertEqual(TCH.search_defaults("person", p), {"stage1": ["solider"], "rerank": None, "pool": 1000})
            self.assertIsNone(TCH.search_defaults("object", p))
            TCH.set_choice("qwen", "b10", {"batch_size": 10}, "배치 10", p)
            self.assertEqual(TCH.qwen_batch_default(p), 10)
            TCH.clear("detector", p)
            self.assertIsNone(TCH.get("detector", p))
            eff = TCH.effective("detector", p)                                   # 저장 없음 → 카탈로그 ★ (자동)
            self.assertTrue(eff and eff.get("auto"))

    def test_tools_page(self):
        from PySide6.QtWidgets import QApplication
        from gui.tools_page import ToolsPage

        app = QApplication.instance() or QApplication([])
        cat = {"detector": {"title": "검출기", "desc": "d", "stage": "detect", "group": "image_pipeline", "stage_id": "image_detect", "metric": "ap50",
                            "candidates": [{"key": "a.yaml", "label": "A", "value": "a.yaml", "metric": "ap50", "metric_value": 0.9, "status": "pass", "detail": ""},
                                           {"key": "b.yaml", "label": "B", "value": "b.yaml", "metric": "ap50", "metric_value": 0.8, "status": "partial", "detail": ""}],
                            "best_key": "a.yaml"}}
        with tempfile.TemporaryDirectory() as td:
            TCH.PATH = Path(td) / "choice.json"
            page = ToolsPage(catalog=cat)
            sec = page.sections["detector"]
            self.assertEqual(sec.table.rowCount(), 2)
            self.assertTrue(sec.table.item(0, 1).text().startswith("★ A"))
            self.assertTrue(sec.radios["a.yaml"].isChecked())
            got = []
            page.choiceChanged.connect(lambda s, k: got.append((s, k)))
            sec.radios["b.yaml"].setChecked(True)
            self.assertEqual(got[-1], ("detector", "b.yaml"))
            self.assertEqual(TCH.get("detector")["key"], "b.yaml")
            opened = []
            page.openStep.connect(lambda g, s: opened.append((g, s)))
            sec.reset_btn.click()
            self.assertIsNone(TCH.get("detector"))
            self.assertTrue(sec.radios["a.yaml"].isChecked())


if __name__ == "__main__":
    unittest.main()
