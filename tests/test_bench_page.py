"""오프라인 테스트: gui/bench_page.py — 원장 리더보드 표·채택 색·필터·상세·채택→yaml (Qt offscreen, 그래프 끔, 서브프로세스 없음)."""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from bench import ledger as L  # noqa: E402
from gui.bench_page import BenchPage, STATUS_COLOR  # noqa: E402

ENV = {"git_commit": "x", "host": "h"}


def make_ledger(path: Path):
    entries = [
        L.make_entry("detect", "detect_eval_prw", "yolo26m_test", env=ENV, created_at="2026-09-26T23:49:51",
                     component={"module": "detect.detectors.yolo26_detector", "class": "YOLO26Detector", "params": {"conf_threshold": 0.05}},
                     params={"operating_threshold": 0.5}, metrics={"ap50": 0.8758, "max_recall": 0.9465, "recall_h75_119": 0.559, "recall_h120_199": 0.815, "fps": 39.0}),
        L.make_entry("detect", "detect_eval_prw", "rfdetr_medium_test", env=ENV, created_at="2026-09-26T23:49:51",
                     component={"module": "detect.detectors.rfdetr_detector", "class": "RFDETRDetector", "params": {"conf_threshold": 0.05}},
                     params={"operating_threshold": 0.5}, metrics={"ap50": 0.8759, "max_recall": 0.9536, "recall_h75_119": 0.66, "recall_h120_199": 0.864, "fps": 15.6},
                     extra={"verify": {"status": "PASS", "passed": True}}),
        L.make_entry("detect", "detect_eval_prw", "rfdetr_medium_test", env=ENV, created_at="2026-09-25T10:00:00",
                     component={"module": "detect.detectors.rfdetr_detector", "class": "RFDETRDetector", "params": {}},
                     metrics={"ap50": 0.80, "max_recall": 0.80}),
        L.make_entry("cluster", "prw_cluster_gt_eval", "Leiden_exact_0.97", env=ENV, component={"method": "leiden"},
                     metrics={"pair_precision": 0.921, "pair_recall": 0.748, "b3_f1": 0.851, "mixed_clusters": 138}),
    ]
    L.append_entries(path, entries)
    return entries


class BenchPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.ledger = self.root / "ledger.jsonl"
        self.entries = make_ledger(self.ledger)
        shutil.copyfile(ROOT / "pipeline_tracking.yaml", self.root / "pipeline_tracking.yaml")
        self.page = BenchPage(ledger_path=self.ledger, adopt_root=self.root, charts=False)

    def tearDown(self):
        self.page.deleteLater()
        self.td.cleanup()

    def test_table_rows_colors_and_latest_filter(self):
        p = self.page
        self.assertEqual(p.current_stage(), "detect")
        self.assertEqual(p.table.rowCount(), 2)                       # 이름별 최근만 → rfdetr 2건 중 1건
        names = {p.table.item(i, 1).text() for i in range(p.table.rowCount())}
        self.assertEqual(names, {"yolo26m_test", "rfdetr_medium_test"})
        by = {p.table.item(i, 1).text(): i for i in range(p.table.rowCount())}
        self.assertEqual(p.table.item(by["rfdetr_medium_test"], 0).background().color().name(), STATUS_COLOR["pass"].name())
        self.assertEqual(p.table.item(by["yolo26m_test"], 0).background().color().name(), STATUS_COLOR["partial"].name())
        self.assertTrue(p.table.item(by["rfdetr_medium_test"], 0).text().startswith("✓ 5/5"))
        verify_col = p.table.columnCount() - 2
        self.assertEqual(p.table.item(by["rfdetr_medium_test"], verify_col).text(), "PASS")
        self.assertIn("2 행 · 통과 1", p.count_label.text())
        p.latest_check.setChecked(False)
        self.assertEqual(p.table.rowCount(), 3)
        p.pass_check.setChecked(True)
        self.assertEqual(p.table.rowCount(), 1)
        p.pass_check.setChecked(False)
        p.name_edit.setText("yolo")
        self.assertEqual(p.table.rowCount(), 1)

    def test_stage_switch_and_numeric_sort(self):
        p = self.page
        p.stage_combo.setCurrentIndex([p.stage_combo.itemData(i) for i in range(p.stage_combo.count())].index("cluster"))
        self.assertEqual(p.current_stage(), "cluster")
        self.assertEqual(p.table.rowCount(), 1)
        headers = [p.table.horizontalHeaderItem(j).text() for j in range(p.table.columnCount())]
        self.assertIn("b3_f1", headers)
        p.stage_combo.setCurrentIndex(0)
        p.latest_check.setChecked(False)
        col = [p.table.horizontalHeaderItem(j).text() for j in range(p.table.columnCount())].index("ap50")
        p.table.sortItems(col, Qt.SortOrder.DescendingOrder)
        vals = [p.table.item(i, col).text() for i in range(p.table.rowCount())]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_select_detail_and_adopt(self):
        p = self.page
        by = {p.table.item(i, 1).text(): i for i in range(p.table.rowCount())}
        self.assertFalse(p.adopt_btn.isEnabled())
        p.table.selectRow(by["rfdetr_medium_test"])
        e = p.selected_entry()
        self.assertEqual(e["name"], "rfdetr_medium_test")
        self.assertTrue(p.adopt_btn.isEnabled())
        self.assertIn(e["run_id"], p.detail.toPlainText())
        self.assertIn("채택 기준 통과", p.detail.toPlainText())
        # 채택 → 임시 루트에 pipeline_tracking_<name>.yaml (원본 프로젝트 파일은 건드리지 않음)
        from PySide6.QtWidgets import QMessageBox
        orig = QMessageBox.information
        QMessageBox.information = staticmethod(lambda *a, **k: None)
        try:
            p._adopt()
        finally:
            QMessageBox.information = orig
        out = self.root / "pipeline_tracking_rfdetr_medium_test.yaml"
        self.assertTrue(out.is_file())
        self.assertIn("conf_threshold: 0.5", out.read_text(encoding="utf-8"))
        self.assertIn("[채택]", p.console.toPlainText())
        p._show_cmd()
        self.assertIn("bench/run.py detect", p.detail.toPlainText())

    def test_missing_ledger(self):
        page = BenchPage(ledger_path=self.root / "none.jsonl", adopt_root=self.root, charts=False)
        self.assertEqual(page.table.rowCount(), 0)
        self.assertIn("원장 없음", page.count_label.text())
        page.deleteLater()


if __name__ == "__main__":
    unittest.main()
