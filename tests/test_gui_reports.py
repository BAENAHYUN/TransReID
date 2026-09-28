"""gui/reports_page — 결과 보기(사진/영상 처리 산출물만: 구분·종류·실행) 와 정답 라벨링(시트 상태) (Qt offscreen, 임시 루트)."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("TRANSREID_NO_WEBENGINE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui import reports_page as R  # noqa: E402


def _w(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        gt = self.root / "eval" / "gt"
        _w(gt / "tracks" / "v1" / "sheet.html", "<html>")
        _w(gt / "tracks" / "v1" / "proposals.json", json.dumps({"kind": "track_labels", "manifest": "abc", "n_segments": 5}))
        _w(gt / "tracks" / "v1" / "labels.json", json.dumps({"kind": "track_labels", "manifest": "abc", "reviewed": 3, "items": 5, "labels": {}}))
        _w(gt / "tracks" / "v2" / "sheet.html", "<html>")
        _w(gt / "tracks" / "v2" / "proposals.json", json.dumps({"kind": "track_labels", "manifest": "new", "n_segments": 4}))
        _w(gt / "tracks" / "v2" / "labels.json", json.dumps({"kind": "track_labels", "meta": {"manifest": "old"}, "labels": {"a": {"reviewed": True}, "b": {}}}))
        _w(gt / "object_pairs" / "sheet.html", "<html>")
        _w(gt / "object_pairs" / "proposals.json", json.dumps({"kind": "object_pair_labels", "manifest": "p", "pairs": [1, 2, 3]}))
        _w(gt / "qwen" / "sheet.html", "<html>")
        _w(gt / "qwen" / "proposals.json", json.dumps({"kind": "qwen_labels", "manifest": "q", "top_k": 20, "queries": [1, 2]}))
        _w(self.root / "outputs" / "image_review" / "index_PRW.html", "x")
        _w(self.root / "outputs" / "audit" / "roadmap.html", "x")          # 개발 문서 → 결과 보기에 안 나옴
        time.sleep(0.02)
        _w(self.root / "outputs" / "clustering" / "leiden_img" / "person" / "gallery" / "g.html", "x")
        _w(self.root / "outputs" / "clustering" / "leiden_img" / "person" / "person_leiden_report.json",
           json.dumps({"config": {"sources": ["prw_image"], "collection": "forensic_person"}}))
        time.sleep(0.02)
        _w(self.root / "outputs" / "clustering" / "run_v" / "person" / "gallery_leiden_video_only" / "v.html", "x")
        _w(self.root / "outputs" / "clustering" / "run_v" / "person" / "person_leiden_report.json",
           json.dumps({"config": {"media_type": "video", "collection": "forensic_person"}}))

    def tearDown(self):
        self.td.cleanup()

    def test_sheets(self):
        rows = {r["name"]: r for r in R.scan_sheets(self.root)}
        self.assertEqual(set(rows), {"v1", "v2", "객체 재출현 (같은 개체?)", "Qwen 판정 (설명에 맞는 사람?)"})
        self.assertEqual(rows["v1"]["items"], 5)
        self.assertEqual(rows["v1"]["status"], "labels.json · 검토 3/5")
        self.assertIn("검토 1/2", rows["v2"]["status"])
        self.assertIn("manifest 불일치", rows["v2"]["status"])
        self.assertEqual(rows["객체 재출현 (같은 개체?)"]["items"], 3)
        self.assertTrue(rows["객체 재출현 (같은 개체?)"]["status"].startswith("라벨 없음"))
        self.assertEqual(rows["Qwen 판정 (설명에 맞는 사람?)"]["items"], 40)

    def test_reports_only_pipeline_outputs(self):
        rows = R.scan_reports(self.root)
        self.assertEqual([(r["media"], r["kind"], r["name"]) for r in rows],
                         [("영상", "사람 묶음 갤러리", "run_v · v"), ("사진", "사람 묶음 갤러리", "leiden_img · g"), ("사진", "결과창 인덱스", "index_PRW")])
        self.assertTrue(all("audit" not in r["rel"] for r in rows))          # 개발 문서 제외


@unittest.skipIf(os.environ.get("SKIP_QT_TESTS") == "1", "Qt tests disabled")
class PageTests(ScanTests):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_reports_page(self):
        page = R.ReportsPage(root=self.root)
        self.assertEqual(page.table.rowCount(), 3)
        self.assertEqual(page.table.item(0, 0).text(), "영상")
        self.assertEqual(page.table.cellWidget(0, 4).text(), "열기")
        self.assertEqual(page.stack.currentIndex(), 0)
        page.table.cellWidget(0, 4).click()                                  # GUI 안 뷰어로
        self.assertEqual(page.stack.currentIndex(), 1)
        self.assertEqual(page.viewer.backend, "textbrowser")
        page.viewer.back_btn.click()
        self.assertEqual(page.stack.currentIndex(), 0)
        (self.root / "outputs" / "image_db_html").mkdir(parents=True)
        (self.root / "outputs" / "image_db_html" / "db.html").write_text("x", encoding="utf-8")
        page.refresh()
        self.assertEqual(page.table.rowCount(), 4)

    def test_labeling_page(self):
        page = R.LabelingPage(root=self.root)
        self.assertEqual(page.sheets.rowCount(), 4)
        self.assertEqual(page.sheets.item(0, 1).text(), "v1")
        page.sheets.cellWidget(0, 4).click()
        self.assertEqual(page.stack.currentIndex(), 1)
        self.assertEqual(page.viewer.path, self.root / "eval" / "gt" / "tracks" / "v1" / "sheet.html")


if __name__ == "__main__":
    unittest.main()
