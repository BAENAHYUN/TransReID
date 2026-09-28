"""'인물 분류' — 색인(assignments + 페이로드 → 사람별/파일별), 이름 저장, 캐시, 페이지 렌더(Qt offscreen, Qdrant 없음)."""
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

from gui import people_index as PI  # noqa: E402


def _assign(pid, cid, noise=False):
    return {"point_id": pid, "cluster_id": cid, "cluster_size": 3, "raw_leiden_id": 1, "noise": noise}


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        folder = self.root / "outputs" / "clustering" / "run_a" / "person"
        folder.mkdir(parents=True)
        rows = [_assign("p1", "leiden:person:aaaa1111"), _assign("p2", "leiden:person:aaaa1111"), _assign("p3", "leiden:person:bbbb2222"),
                _assign("p4", "", noise=True), _assign("p5", "leiden:person:aaaa1111")]
        (folder / "person_leiden_assignments.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        crops = self.root / "crops"
        crops.mkdir()
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QPixmap, QColor
        self.app = QApplication.instance() or QApplication([])
        for i in range(1, 6):
            pm = QPixmap(30, 60)
            pm.fill(QColor("blue"))
            pm.save(str(crops / f"c{i}.jpg"))
        self.payloads = {
            "p1": {"media_type": "image", "image_id": "PRW/a.jpg", "crop_path": "crops/c1.jpg", "score": 0.9, "source": "prw_image"},
            "p2": {"media_type": "image", "image_id": "PRW/b.jpg", "crop_path": "crops/c2.jpg", "score": 0.95, "source": "prw_image"},
            "p3": {"media_type": "image", "image_id": "PRW/a.jpg", "crop_path": "crops/c3.jpg", "score": 0.5, "source": "prw_image"},
            "p4": {"media_type": "image", "image_id": "coco/z.jpg", "crop_path": "crops/c4.jpg", "score": 0.7, "source": "COCO"},
            "p5": {"media_type": "video", "video": "cam1.mp4", "crop_path": "crops/c5.jpg", "score": 0.8, "time_mmss": "01:02.0", "track_key": "cam1/person_0001"},
        }
        self.fetch = lambda ids, progress: {k: v for k, v in self.payloads.items() if k in set(ids)}

    def tearDown(self):
        self.td.cleanup()

    def test_find_runs_and_build(self):
        runs = PI.find_runs(self.root)
        self.assertEqual([(r["run"], r["method"], r["cached"]) for r in runs], [("run_a", "leiden", False)])
        idx = PI.load_or_build(runs[0], fetch=self.fetch)
        self.assertTrue(PI.cache_path(runs[0]).is_file())
        a = idx["clusters"]["leiden:person:aaaa1111"]
        self.assertEqual(a["size"], 3)
        self.assertEqual(sorted(a["files"]), ["PRW/a.jpg", "PRW/b.jpg", "cam1.mp4"])
        self.assertEqual(a["rep"], "p2")                                   # 검출 점수가 가장 높은 crop
        self.assertEqual(a["folders"], {"PRW": 2, "videos": 1})
        self.assertEqual(idx["files"]["PRW/a.jpg"]["clusters"], {"leiden:person:aaaa1111": ["p1"], "leiden:person:bbbb2222": ["p3"]})
        self.assertEqual(idx["clusters"][PI.NOISE]["members"], ["p4"])
        self.assertEqual(idx["files"]["coco/z.jpg"]["folder"], "coco")
        self.assertEqual(idx["points"]["p5"]["time"], "01:02.0")
        # 캐시 재사용 (fetch 가 불려도 안 됨)
        idx2 = PI.load_or_build(runs[0], fetch=lambda ids, progress: (_ for _ in ()).throw(AssertionError("fetch 호출됨")))
        self.assertEqual(idx2["n_points"], 5)
        self.assertTrue(PI.find_runs(self.root)[0]["cached"])

    def test_names(self):
        run = PI.find_runs(self.root)[0]
        self.assertEqual(PI.display_name("leiden:person:aaaa1111", {}), "#aaaa1111")
        self.assertEqual(PI.display_name(PI.NOISE, {}), "미분류")
        names = PI.save_name(run, "leiden:person:aaaa1111", "김철수")
        self.assertEqual(PI.load_names(run), {"leiden:person:aaaa1111": "김철수"})
        self.assertEqual(PI.display_name("leiden:person:aaaa1111", names), "김철수")
        PI.save_name(run, "leiden:person:aaaa1111", "")
        self.assertEqual(PI.load_names(run), {})

    def test_page(self):
        from gui.people_page import PeoplePage

        page = PeoplePage(root=self.root, fetch=self.fetch)
        self.assertEqual(page.run_combo.count(), 1)
        # 워커(QThread) 경로: start → 끝날 때까지 기다린 뒤 큐에 쌓인 시그널을 처리
        page.load()
        self.assertTrue(page.worker.wait(10000))
        for _ in range(20):
            self.app.processEvents()
        self.assertEqual(page.grid.count(), 2, page.summary.text())
        self.assertTrue(page.load_btn.isEnabled())
        page.index = {}
        page.grid.clear()
        page.load_sync()
        self.assertEqual(page.grid.count(), 2)                              # 미분류 제외
        self.assertTrue(page.grid.item(0).text().startswith("#aaaa1111\n3장 · 파일 3"))
        self.assertFalse(page.grid.item(0).icon().isNull())
        self.assertEqual([page.folder_combo.itemData(i) for i in range(page.folder_combo.count())], ["", "PRW", "coco", "videos"])
        page.grid.setCurrentRow(0)
        self.assertEqual(page.detail_list.count(), 3)                       # 파일 3개
        self.assertTrue(page.search_btn.isEnabled())
        got = []
        page.searchRequested.connect(got.append)
        page._search_person()
        self.assertTrue(got and got[0].endswith("c2.jpg"))
        page.set_name("leiden:person:aaaa1111", "김철수")
        self.assertTrue(page.grid.item(0).text().startswith("김철수"))
        # 폴더 필터
        page.folder_combo.setCurrentIndex(2)                                # coco → 사람 없음(미분류만)
        self.assertEqual(page.grid.count(), 0)
        page.folder_combo.setCurrentIndex(1)                                # PRW
        self.assertEqual(page.grid.count(), 2)
        # 파일별
        page.mode_files.setChecked(True)
        self.assertEqual(page.left.currentIndex(), 1)
        self.assertEqual(page.file_list.count(), 2)                         # PRW/a.jpg, PRW/b.jpg
        self.assertIn("사람 2명: 김철수, #bbbb2222", page.file_list.item(0).text())
        page.file_list.setCurrentRow(0)
        self.assertEqual(page.detail_list.count(), 2)
        self.assertTrue(page.detail_list.item(0).text().startswith("김철수"))


if __name__ == "__main__":
    unittest.main()
