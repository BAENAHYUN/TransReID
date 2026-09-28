"""검색 탭 모델 선택: 백엔드 선택 해석(unified_search_4mode) + GUI 드롭다운(search_gui, Qt offscreen). 모델 로드·Qdrant 없음."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search import unified_search_4mode as us  # noqa: E402


def fake_cfg():
    person = [NS(name="siglip2", supports_text=True), NS(name="irra", supports_text=True), NS(name="solider", supports_text=False)]
    obj = [NS(name="siglip2", supports_text=True), NS(name="dinov2", supports_text=False)]
    return NS(for_person=lambda: person, for_object=lambda: obj)


class BackendSelectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = fake_cfg()

    def test_stage1_default_all_and_explicit(self):
        self.assertEqual(us.resolve_stage1(self.cfg, "person", None), ["siglip2", "irra"])
        self.assertEqual(us.resolve_stage1(self.cfg, "object", []), ["siglip2", "dinov2"])
        self.assertEqual(us.resolve_stage1(self.cfg, "person", [us.COMBO_ALL]), ["siglip2", "irra", "solider"])
        self.assertEqual(us.resolve_stage1(self.cfg, "person", ["solider", "solider", "irra"]), ["solider", "irra"])
        with self.assertRaises(ValueError):
            us.resolve_stage1(self.cfg, "person", ["dinov2"])

    def test_rerank_default_none_and_explicit(self):
        self.assertEqual(us.resolve_rerank(self.cfg, "person", None), "solider")
        self.assertIsNone(us.resolve_rerank(self.cfg, "object", None))
        for off in ("none", "", "NONE", "off"):
            self.assertIsNone(us.resolve_rerank(self.cfg, "person", off))
        self.assertEqual(us.resolve_rerank(self.cfg, "person", "IRRA"), "irra")
        with self.assertRaises(ValueError):
            us.resolve_rerank(self.cfg, "object", "solider")

    def test_text_vectors_only_supports_text(self):
        self.assertEqual(us.resolve_text_vectors(self.cfg, "person", None), ["siglip2", "irra"])
        self.assertEqual(us.resolve_text_vectors(self.cfg, "person", ["irra"]), ["irra"])
        self.assertEqual(us.resolve_text_vectors(self.cfg, "object", [us.COMBO_ALL]), ["siglip2"])
        with self.assertRaises(ValueError):
            us.resolve_text_vectors(self.cfg, "person", ["solider"])
        empty = NS(for_person=lambda: [NS(name="solider", supports_text=False)], for_object=lambda: [])
        with self.assertRaises(ValueError):
            us.resolve_text_vectors(empty, "person", None)

    def test_pipeline_label(self):
        self.assertEqual(us.pipeline_label(["siglip2", "irra"], "solider"), "siglip2+irra -> solider_rerank")
        self.assertEqual(us.pipeline_label(["irra"], None), "irra")

    def test_cli_flags(self):
        p = us.build_parser()
        a = p.parse_args(["crop", "--scope", "person", "--image", "q.jpg", "--stage1", "irra", "solider", "--rerank", "none"])
        self.assertEqual((a.stage1, a.rerank), (["irra", "solider"], "none"))
        a = p.parse_args(["crop", "--scope", "person", "--image", "q.jpg"])
        self.assertEqual((a.stage1, a.rerank), (None, None))
        a = p.parse_args(["text", "--scope", "object", "--text", "x", "--vectors", "siglip2"])
        self.assertEqual(a.vectors, ["siglip2"])
        a = p.parse_args(["text-video", "--scope", "person", "--text", "x", "--vectors", "irra"])
        self.assertEqual(a.vectors, ["irra"])

    def test_merge_group_results_single_vector_sorts_by_its_score(self):
        hit = lambda s, pid: NS(payload={"video": "v", "track_key": pid}, score=s)
        groups = [NS(id="g1", hits=[hit(0.5, "g1")]), NS(id="g2", hits=[hit(0.9, "g2")])]
        rows = us.merge_group_results({"irra": groups}, scope="person")
        self.assertEqual([r.group_id for r in rows], ["g2", "g1"])
        # 두 벡터면 RRF: g1 이 두 목록 모두 1위 → 앞
        rows = us.merge_group_results({"siglip2": groups[::-1], "irra": groups[::-1]}, scope="person")
        self.assertEqual(rows[0].group_id, "g2")


@unittest.skipIf(os.environ.get("SKIP_QT_TESTS") == "1", "Qt tests disabled")
class GuiDropdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])
        import search_gui

        cls.gui = search_gui

    @staticmethod
    def items(combo):
        return [combo.itemText(i) for i in range(combo.count())]

    def test_option_helpers(self):
        g = self.gui
        self.assertEqual(g.stage1_choices("person", ["siglip2", "irra", "solider"]),
                         [["siglip2"], ["irra"], ["solider"], ["siglip2", "irra"], ["siglip2", "irra", "solider"]])
        self.assertEqual(g.stage1_choices("object", ["siglip2", "dinov2"]), [["siglip2"], ["dinov2"], ["siglip2", "dinov2"]])
        self.assertEqual(g.combo_label(["siglip2", "irra"]), "SigLIP2 + IRRA (RRF 조합)")
        self.assertEqual(g.combo_label(["dinov2"]), "DINOv2")
        opts = g.load_search_model_options(str(ROOT / "pipeline.yaml"))
        self.assertIn("solider", opts["person_image"])
        self.assertNotIn("solider", opts["person_text"])
        self.assertEqual(g.load_search_model_options(str(ROOT / "does_not_exist.yaml")), g.FALLBACK_MODEL_OPTIONS)

    def test_image_page_defaults_and_scope_switch(self):
        page = self.gui.ImageSearchPage(str(ROOT / "pipeline.yaml"))
        self.assertEqual(page.image_model_selection(), (["siglip2", "irra"], "solider"))
        self.assertEqual(page.text_model_selection(), ["siglip2", "irra"])
        self.assertIn("SOLIDER", self.items(page.image_stage1))
        self.assertIn("없음", self.items(page.image_rerank))
        page.image_stage1.setCurrentIndex(self.items(page.image_stage1).index("IRRA"))
        page.image_rerank.setCurrentIndex(0)
        self.assertEqual(page.image_model_selection(), (["irra"], None))
        self.assertIn("IRRA", page.image_pipeline.text())
        self.assertIn("재정렬 없음", page.image_pipeline.text())
        page.image_scope.setCurrentIndex(1)   # 객체
        self.assertEqual(page.image_model_selection(), (["siglip2", "dinov2"], None))
        self.assertNotIn("SOLIDER", self.items(page.image_stage1))
        page.text_scope.setCurrentIndex(1)
        self.assertEqual(page.text_model_selection(), ["siglip2"])
        self.assertEqual(self.items(page.text_vectors), ["SigLIP2"])

    def test_video_page_defaults_and_scope_switch(self):
        page = self.gui.VideoSearchPage(str(ROOT / "pipeline.yaml"))
        self.assertEqual(page.image_model_selection(), "solider")
        self.assertEqual(self.items(page.image_vector), ["SigLIP2", "IRRA", "SOLIDER"])
        self.assertEqual(page.text_model_selection(), ["siglip2", "irra"])
        page.text_vectors.setCurrentIndex(1)
        self.assertEqual(page.text_model_selection(), ["irra"])
        self.assertNotIn("RRF", page.text_pipeline.text())
        page.image_scope.setCurrentIndex(1)
        self.assertEqual(page.image_model_selection(), "dinov2")
        self.assertIn("DINOv2", page.image_pipeline.text())


if __name__ == "__main__":
    unittest.main()
