"""Immich 식 셸 (Qt offscreen, 모델·Qdrant 없음):
- gui/shell.AppShell: 섹션/페이지 추가, 선택, 첫 페이지 기본 선택, 아이콘 생성
- gui/search_ui: 자연어/사진 모드 전환이 검색창·대상·설명·고급 행·AI 재확인 대상을 함께 바꾼다; 두 대상 콤보가 묶인다
- search_gui.ResultsPanel: 썸네일 격자 항목(캡션·툴팁), '이 결과로 다시 찾기' 시그널 → 페이지가 사진 모드로 전환
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@unittest.skipIf(os.environ.get("SKIP_QT_TESTS") == "1", "Qt tests disabled")
class ShellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication, QLabel

        cls.app = QApplication.instance() or QApplication([])
        cls.QLabel = QLabel
        from gui import shell as sh

        cls.sh = sh

    def test_add_select_and_icons(self):
        s = self.sh.AppShell(title="T", subtitle="sub")
        s.add_section("검색")
        s.add_page("a", "사진에서 찾기", "search", self.QLabel("A"))
        s.add_page("b", "영상에서 찾기", "video", self.QLabel("B"))
        s.add_section("평가")
        s.add_page("c", "벤치마크", "bench", self.QLabel("C"))
        self.assertEqual(s.keys(), ["a", "b", "c"])
        self.assertEqual(s.current_key(), "a")
        seen = []
        s.pageChanged.connect(seen.append)
        s.select("c")
        self.assertEqual(s.current_key(), "c")
        self.assertEqual(s.stack.currentWidget().text(), "C")
        self.assertEqual(seen, ["c"])
        self.assertEqual(s.label_of("b"), "영상에서 찾기")
        with self.assertRaises(KeyError):
            s.select("zzz")
        with self.assertRaises(ValueError):
            s.add_page("a", "dup", "photo", self.QLabel("dup"))
        for kind in ("search", "video", "photo", "film", "chart", "bench", "gear", "other"):
            self.assertFalse(self.sh.nav_icon(kind).isNull())


@unittest.skipIf(os.environ.get("SKIP_QT_TESTS") == "1", "Qt tests disabled")
class SearchHeaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])
        import search_gui

        cls.gui = search_gui

    def test_mode_switch_image_page(self):
        page = self.gui.ImageSearchPage(str(ROOT / "pipeline.yaml"))
        self.assertEqual(page.search_mode, "text")
        self.assertEqual(page.query_stack.currentIndex(), 0)
        self.assertEqual(page.qwen_source.currentData(), "text")
        self.assertTrue(not page.text_adv.isHidden() and page.crop_adv.isHidden())
        page.mode_crop_btn.setChecked(True)
        self.assertEqual(page.search_mode, "crop")
        self.assertEqual(page.query_stack.currentIndex(), 1)
        self.assertEqual(page.qwen_source.currentData(), "crop")          # AI 재확인 대상이 모드를 따라간다
        self.assertTrue(not page.crop_adv.isHidden() and page.text_adv.isHidden())
        self.assertTrue(page.adv_panel.isHidden())                        # 고급 설정은 접혀 있다
        page.adv_btn.setChecked(True)
        self.assertFalse(page.adv_panel.isHidden())
        # 대상 콤보 두 개는 묶여 있다
        page.image_scope.setCurrentIndex(1)
        self.assertEqual(page.text_scope.currentData(), "object")
        page.text_scope.setCurrentIndex(0)
        self.assertEqual(page.image_scope.currentData(), "person")
        self.assertIs(page.search_tabs, page.header)                     # 옛 이름 호환

    def test_mode_switch_video_page(self):
        page = self.gui.VideoSearchPage(str(ROOT / "pipeline.yaml"))
        self.assertEqual(page.video_qwen_source.currentData(), "text-video")
        self.gui.set_search_mode(page, "crop")
        self.assertEqual(page.video_qwen_source.currentData(), "image-video")
        self.assertTrue(page.mode_crop_btn.isChecked())
        page._set_busy(True)
        self.assertFalse(page.mode_text_btn.isEnabled())
        page._set_busy(False)
        self.assertTrue(page.mode_text_btn.isEnabled())

    def test_results_grid_and_use_as_query(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        from PySide6.QtGui import QColor, QPixmap

        crop = Path(td.name) / "c.jpg"
        pix = QPixmap(40, 90)
        pix.fill(QColor("red"))
        pix.save(str(crop))
        rows = [{"rank": 1, "score": 0.61234, "image_id": "coco/000000000001.jpg", "label": "person", "crop_path": str(crop), "verified": True, "attr_summary": "shirt black"},
                {"rank": 2, "rrf_score": 0.25, "image_id": "coco/x.jpg", "label": "person", "crop_path": str(Path(td.name) / "missing.jpg"), "verified": False, "failed_required": ["shirt.color"]},
                {"rank": 3, "score": 0.1, "image_id": "coco/y.jpg", "label": "person", "crop_path": None, "attr_skipped": "crop file not found"}]
        page = self.gui.ImageSearchPage(str(ROOT / "pipeline.yaml"))
        panel = page.results
        panel.set_rows(rows)
        self.assertEqual(panel.list.count(), 3)
        self.assertEqual(panel.list.item(0).text(), "#1 · 0.61   ✓ AI\n000000000001.jpg")
        self.assertIn("✕ AI", panel.list.item(1).text())
        self.assertIn("? AI", panel.list.item(2).text())
        self.assertFalse(panel.list.item(0).icon().isNull())
        self.assertTrue(panel.list.item(1).icon().isNull())               # 파일 없음 → 빈 아이콘
        self.assertIn("3건", panel.count_label.text())
        self.assertEqual(panel.list.currentRow(), 0)
        self.assertIn("조건에 맞음", panel.detail.toPlainText())
        self.assertIn("원본 필드", panel.detail.toPlainText())
        self.assertTrue(panel.reuse_btn.isEnabled())
        panel.list.setCurrentRow(1)
        self.assertIn("필수 조건 실패: shirt.color", panel.detail.toPlainText())
        panel.list.setCurrentRow(0)
        panel._emit_use_as_query()
        self.assertEqual(page.search_mode, "crop")
        self.assertEqual(Path(page.query_image).resolve(), crop.resolve())
        panel._append_detail_note("재생 오류: x")
        self.assertIn("재생 오류: x", panel.detail.toPlainText())
        panel.clear()
        self.assertEqual(panel.list.count(), 0)
        self.assertFalse(panel.reuse_btn.isEnabled())

    def test_thumb_and_caption_helpers(self):
        g = self.gui
        self.assertIsNone(g.thumb_pixmap(None))
        self.assertIsNone(g.thumb_pixmap(str(Path(tempfile.gettempdir()) / "nope_dir_xyz" / "x.jpg")))   # 없는 드라이브(Z:) 는 탐색 대기로 느리다
        cap = g.result_caption({"rank": 4, "timestamp_sec": 65.0, "video": "cam_01.mp4"}, video=True)
        self.assertEqual(cap, f"#4 · {g.fmt_time(65.0)}\ncam_01.mp4")
        tip = g.result_tooltip({"rank": 4, "score": 0.5, "video": "cam_01.mp4", "timestamp_sec": 65.0, "group_summary": {"start_sec": 60, "end_sec": 70}}, video=True)
        self.assertIn("cam_01.mp4", tip)
        self.assertIn(f"{g.fmt_time(60)} ~ {g.fmt_time(70)}", tip)


if __name__ == "__main__":
    unittest.main()
