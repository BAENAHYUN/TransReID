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


def _jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _write_labels(folder: Path) -> None:
    """8b(Qwen 문장)·8(색상) 라벨과 옛 형식(status 없음, 퇴화 라벨) 을 실행 폴더에 만든다."""
    _jsonl(folder / "labels_qwen" / "cluster_labels.jsonl", [
        {"cluster_id": "leiden:person:aaaa1111", "cluster_name": "노란 반팔에 검은 바지", "status": "labeled",
         "label_confidence": 1.0, "cluster_description": "upper=노란색 반팔 티셔츠"},
        {"cluster_id": "leiden:person:bbbb2222", "cluster_name": "", "status": "mixed", "label_confidence": 0.0},
    ])
    _jsonl(folder / "labels_vec" / "cluster_labels.jsonl", [
        {"cluster_id": "leiden:person:aaaa1111", "cluster_name": "노란색 상의", "status": "labeled", "label_confidence": 0.96},
        {"cluster_id": "leiden:person:bbbb2222", "cluster_name": "검은색 상의(추정)", "status": "tentative", "label_confidence": 0.5},
        {"cluster_id": "leiden:person:cccc3333", "cluster_name": "", "status": "uncertain", "label_confidence": 0.0},
    ])
    _jsonl(folder / "labels" / "cluster_labels.jsonl", [                      # 옛 label_leiden_clusters_siglip2.py 출력 → 무시
        {"cluster_id": "leiden:person:bbbb2222", "cluster_name": "yellow t-shirt · skirt", "label_confidence": 0.5},
    ])


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

    def test_labels(self):
        run = PI.find_runs(self.root)[0]
        self.assertEqual(run["labels"], [])
        self.assertEqual(PI.load_labels(run), {})
        _write_labels(run["folder"])
        run = PI.find_runs(self.root)[0]
        self.assertEqual(run["labels"], ["labels_qwen", "labels_vec"])           # 옛 labels/ 는 라벨 종류가 아니다
        labels = PI.load_labels(run)
        a = labels["leiden:person:aaaa1111"]
        self.assertEqual((a["name"], a["status"], a["kind"], a["dir"]), ("노란 반팔에 검은 바지", "labeled", "Qwen 문장", "labels_qwen"))
        self.assertEqual(a["others"], [{"name": "노란색 상의", "kind": "색상(SigLIP2)", "status": "labeled", "dir": "labels_vec"}])
        b = labels["leiden:person:bbbb2222"]
        self.assertEqual((b["name"], b["status"], b["kind"], b["others"]), ("검은색 상의(추정)", "tentative", "색상(SigLIP2)", []))
        self.assertNotIn("leiden:person:cccc3333", labels)                       # uncertain(이름 없음) 은 없는 것
        self.assertEqual(PI.label_line(a), "노란 반팔에 검은 바지 (Qwen 문장) · 노란색 상의 (색상(SigLIP2))")
        self.assertEqual(PI.label_line(None), "")
        # 표시 이름: 사용자 이름 > 자동 라벨 > #id
        self.assertEqual(PI.display_name("leiden:person:aaaa1111", {}, labels), "노란 반팔에 검은 바지")
        self.assertEqual(PI.display_name("leiden:person:aaaa1111", {"leiden:person:aaaa1111": "김철수"}, labels), "김철수")
        self.assertEqual(PI.display_name("leiden:person:cccc3333", {}, labels), "#cccc3333")
        self.assertEqual(list(PI.load_labels(run, clusters=["leiden:person:bbbb2222"])), ["leiden:person:bbbb2222"])
        # 같은 종류가 둘이면(labels_qwen_sample4b) 기본 폴더 이름이 먼저 — 확정 등급이 같을 때
        _jsonl(run["folder"] / "labels_qwen_sample4b" / "cluster_labels.jsonl",
               [{"cluster_id": "leiden:person:aaaa1111", "cluster_name": "노란 티셔츠에 검은 긴바지", "status": "labeled", "label_confidence": 1.0}])
        a2 = PI.load_labels(run)["leiden:person:aaaa1111"]
        self.assertEqual((a2["name"], a2["dir"]), ("노란 반팔에 검은 바지", "labels_qwen"))
        self.assertEqual([o["name"] for o in a2["others"]], ["노란 티셔츠에 검은 긴바지", "노란색 상의"])
        self.assertEqual(PI.label_line(a2), "노란 반팔에 검은 바지 (Qwen 문장) · 노란 티셔츠에 검은 긴바지 (Qwen 문장 · labels_qwen_sample4b)"
                                            " · 노란색 상의 (색상(SigLIP2))")
        # 자동 라벨 명령: leiden 은 8/8b 단계와 같은 폴더, 다른 방법은 접미가 붙는다
        cmd = PI.auto_label_command(run, "vec", python="py", root=self.root)
        self.assertEqual(cmd[:2], ["py", "-u"])
        self.assertTrue(cmd[2].endswith(os.path.join("clustering", "label_clusters_from_vectors.py")))
        self.assertEqual(cmd[3:], ["--assignments", str(run["assignments"]), "--target", "person",
                                   "--output-dir", str(run["folder"] / "labels_vec")])
        qcmd = PI.auto_label_command(run, "qwen", python="py", root=self.root)
        self.assertTrue(qcmd[2].endswith("label_clusters_qwen.py"))
        self.assertEqual(qcmd[-2:], [str(run["folder"] / "labels_qwen"), "--resume"])   # 이미 있으면 빠진 군집만 (덮어쓰지 않음)
        self.assertNotIn("--resume", cmd)                                                 # 색상은 그냥 다시 만든다
        fresh = PI.auto_label_command(dict(run, method="dbscan_v6"), "qwen", python="py", root=self.root)
        self.assertEqual(fresh[-1], str(run["folder"] / "labels_qwen_dbscan_v6"))         # 결과가 없으면 처음부터
        self.assertEqual(PI.label_output_dir(dict(run, method="dbscan_v6"), "vec"), run["folder"] / "labels_vec_dbscan_v6")
        # 접미 없는 옛 폴더에 같은 방법의 라벨이 있으면 그 폴더를 계속 쓴다 (09-21 8b 단계가 dbscan 에도 labels_qwen/ 을 썼다)
        with tempfile.TemporaryDirectory() as td:
            f = Path(td)
            _jsonl(f / "labels_qwen" / "cluster_labels.jsonl", [{"cluster_id": "dbscan:person:1", "cluster_name": "x", "status": "labeled"}])
            drun = dict(run, method="dbscan", folder=f)
            self.assertEqual(PI.label_output_dir(drun, "qwen"), f / "labels_qwen")
            self.assertEqual(PI.label_output_dir(drun, "vec"), f / "labels_vec_dbscan")              # 옛 색상 폴더 없음
            self.assertEqual(PI.label_output_dir(dict(drun, method="dbscan_v6"), "qwen"), f / "labels_qwen_dbscan_v6")  # 다른 방법
            _jsonl(f / "labels_qwen_dbscan" / "cluster_labels.jsonl", [{"cluster_id": "dbscan:person:1", "cluster_name": "y", "status": "labeled"}])
            self.assertEqual(PI.label_output_dir(drun, "qwen"), f / "labels_qwen_dbscan")           # 접미 폴더가 있으면 그쪽
        with self.assertRaises(ValueError):
            PI.auto_label_command(run, "nope")

    def test_page(self):
        from gui.people_page import PeoplePage

        page = PeoplePage(root=self.root, fetch=self.fetch, auto_label_on_load=False)
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
        page.searchRequested.connect(lambda *a: got.append(a))
        page._search_person()
        self.assertTrue(got and got[0][0].endswith("c2.jpg"))
        self.assertEqual(got[0][1:], ("image", "person"))                    # 대표 crop 이 사진(PRW/b.jpg) → 사진에서 찾기
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

    def test_page_labels(self):
        from gui import people_page as PP

        page = PP.PeoplePage(root=self.root, fetch=self.fetch, auto_label_on_load=False)
        self.assertFalse(page.label_btn.isEnabled())                          # 불러오기 전
        page.load_sync()
        self.assertTrue(page.label_btn.isEnabled())
        self.assertIn("자동 라벨 없음", page.summary.text())
        self.assertTrue(page.grid.item(0).text().startswith("#aaaa1111\n"))
        # 라벨 파일이 생기면 이름 없는 카드가 라벨을 쓴다 (툴팁·상세·파일별 목록도)
        _write_labels(page.run["folder"])
        page.labels = PI.load_labels(page.run)
        page._render()
        self.assertTrue(page.grid.item(0).text().startswith("노란 반팔에 검은 바지\n3장"))
        self.assertTrue(page.grid.item(1).text().startswith("검은색 상의(추정)\n"))
        self.assertIn("자동 라벨: 노란 반팔에 검은 바지 (Qwen 문장)", page.grid.item(0).toolTip())
        self.assertIn("자동 라벨 2명", page.summary.text())
        page.grid.setCurrentRow(0)
        self.assertTrue(page.detail_title.text().startswith("노란 반팔에 검은 바지"))
        self.assertIn("#aaaa1111 · 자동 라벨: 노란 반팔에 검은 바지 (Qwen 문장) · 노란색 상의 (색상(SigLIP2))", page.detail_sub.text())
        page.set_name("leiden:person:aaaa1111", "김철수")                     # 사용자 이름이 우선
        self.assertTrue(page.grid.item(0).text().startswith("김철수"))
        self.assertIn("이름: 김철수", page.grid.item(0).toolTip())
        page.mode_files.setChecked(True)
        self.assertIn("사람 2명: 김철수, 검은색 상의(추정)", page.file_list.item(0).text())
        page.mode_people.setChecked(True)
        # 자동 라벨 명령은 고른 도구를 따른다
        page.label_tool.setCurrentIndex(1)
        cmd = page.auto_label_command()
        self.assertTrue(cmd[2].endswith("label_clusters_qwen.py"))
        self.assertEqual(cmd[-2:], [str(page.run["folder"] / "labels_qwen"), "--resume"])
        page.label_tool.setCurrentIndex(0)
        self.assertTrue(page.auto_label_command()[2].endswith("label_clusters_from_vectors.py"))
        # 워커: 라벨러 대신 라벨 파일을 새로 쓰는 파이썬 한 줄 → 끝나면 라벨을 다시 읽고 카드가 바뀐다
        new = page.run["folder"] / "labels_qwen" / "cluster_labels.jsonl"
        rec = json.dumps({"cluster_id": "leiden:person:bbbb2222", "cluster_name": "빨간 상의에 청바지", "status": "labeled",
                          "label_confidence": 1.0}, ensure_ascii=False)
        page._start_label_worker([sys.executable, "-c",
                                  f"import pathlib; pathlib.Path({str(new)!r}).write_text({rec!r} + '\\n', encoding='utf-8'); print('RESULT_SUMMARY ok')"])
        self.assertEqual(page.label_btn.text(), "중단")
        self.assertTrue(page.load_btn.isEnabled())                            # 라벨러가 도는 동안에도 다른 결과를 불러올 수 있다
        self.assertTrue(page.label_worker.wait(15000))
        for _ in range(20):
            self.app.processEvents()
        self.assertEqual(page.label_btn.text(), "자동 라벨 붙이기")
        self.assertTrue(page.load_btn.isEnabled() and page.label_tool.isEnabled())
        self.assertIn("자동 라벨 완료 — 사람 2명", page.summary.text())
        self.assertEqual(page.labels["leiden:person:bbbb2222"]["name"], "빨간 상의에 청바지")   # Qwen 확정이 색상 추정을 이긴다
        self.assertTrue(page.grid.item(1).text().startswith("빨간 상의에 청바지\n"))
        self.assertEqual(page.labels["leiden:person:aaaa1111"]["kind"], "색상(SigLIP2)")     # Qwen 파일에서 빠졌으니 색상으로
        # 실패: 종료 코드가 0 이 아니면 상태를 되돌리고 알린다 (대화상자는 막지 않게 바꿔 둔다)
        orig = PP.QMessageBox.warning
        PP.QMessageBox.warning = lambda *a, **k: None
        try:
            page._start_label_worker([sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"])
            self.assertTrue(page.label_worker.wait(15000))
            for _ in range(20):
                self.app.processEvents()
        finally:
            PP.QMessageBox.warning = orig
        self.assertIn("자동 라벨 실패 (종료 코드 3)", page.summary.text())
        self.assertIn("boom", page.summary.text())
        self.assertTrue(page.label_btn.isEnabled() and page.load_btn.isEnabled())
        self.assertEqual(page.label_btn.text(), "자동 라벨 붙이기")

    def test_object_target_and_search_routing(self):
        """물건 군집: 대상을 물건으로 바꾸면 object_* 결과가 보이고, 영상 물건은 '영상에서 찾기' 로 보낸다."""
        from gui import people_page as PP

        folder = self.root / "outputs" / "clustering" / "run_o" / "object"
        folder.mkdir(parents=True)
        (folder / "object_leiden_assignments.jsonl").write_text(
            "".join(json.dumps(_assign(p, "leiden:object:cccc3333")) + "\n" for p in ("o1", "o2")), encoding="utf-8")
        self.payloads.update({
            "o1": {"media_type": "video", "video": "cam2.mp4", "crop_path": "crops/c1.jpg", "score": 0.9, "label": "car"},
            "o2": {"media_type": "video", "video": "cam2.mp4", "crop_path": "crops/c2.jpg", "score": 0.8, "label": "car"},
        })
        page = PP.PeoplePage(root=self.root, fetch=self.fetch, auto_label_on_load=False)
        self.assertEqual([page.run_combo.itemData(i)["run"] for i in range(page.run_combo.count())], ["run_a"])
        page.target_combo.setCurrentIndex(page.target_combo.findData("object"))
        self.assertEqual(page.target, "object")
        self.assertEqual([page.run_combo.itemData(i)["run"] for i in range(page.run_combo.count())], ["run_o"])
        self.assertEqual((page.mode_people.text(), page.search_btn.text()), ("물건별", "이 물건으로 검색"))
        page.load_sync()
        self.assertEqual(page.grid.count(), 1)
        self.assertTrue(page.summary.text().startswith("물건 1개 · "))
        self.assertEqual(page.summary.toolTip(), page.summary.text())                 # 잘려도 툴팁에 전문
        page.grid.setCurrentRow(0)
        req = page.search_request()
        self.assertEqual(req[1:], ("video", "object"))                               # 영상 물건 → 영상에서 찾기 · 물건 대상
        self.assertEqual(page.auto_label_command()[3:7], ["--assignments", str(folder / "object_leiden_assignments.jsonl"),
                                                         "--target", "object"])
        page.target_combo.setCurrentIndex(page.target_combo.findData("person"))      # 되돌리면 사람 결과
        self.assertEqual((page.target, page.mode_people.text(), page.grid.count()), ("person", "사람별", 0))
        self.assertEqual(page.run_combo.itemData(0)["run"], "run_a")

    def test_inherited_names_last_run_and_log(self):
        """다른 결과에서 붙인 이름을 구성원 과반 공유로 이어받고, 마지막으로 연 결과를 기억하고, 라벨러 로그를 파일로 남긴다."""
        from gui import people_page as PP

        run_a = PI.find_runs(self.root)[0]
        PI.save_name(run_a, "leiden:person:aaaa1111", "김철수")                       # run_a: aaaa = p1 p2 p5
        folder_c = self.root / "outputs" / "clustering" / "run_c" / "person"
        folder_c.mkdir(parents=True)
        rows = [_assign("p1", "dbscan:person:zzzz9999"), _assign("p2", "dbscan:person:zzzz9999"),   # p1 p2 → 과반 공유
                _assign("p5", "dbscan:person:yyyy8888"), _assign("p3", "dbscan:person:yyyy8888"),   # p5 만 → 1/3, 1/2
                _assign("p4", "dbscan:person:yyyy8888")]
        (folder_c / "person_dbscan_assignments.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        page = PP.PeoplePage(root=self.root, fetch=self.fetch, auto_label_on_load=False)
        page.run_combo.setCurrentIndex(next(i for i in range(page.run_combo.count()) if page.run_combo.itemData(i)["run"] == "run_c"))
        page.load_sync()
        self.assertEqual(page.inherited, {"dbscan:person:zzzz9999": {"name": "김철수", "from_run": "run_a", "share": 0.667}})
        texts = [page.grid.item(i).text() for i in range(page.grid.count())]
        self.assertTrue(any(t.startswith("김철수\n2장") for t in texts), texts)
        self.assertTrue(any(t.startswith("#yyyy8888\n") for t in texts), texts)      # 공유가 과반이 안 되면 안 옮긴다
        row = next(i for i, t in enumerate(texts) if t.startswith("김철수"))
        self.assertIn("이어받은 이름: 김철수 (run_a 에서, 구성원 공유 67%)", page.grid.item(row).toolTip())
        self.assertIn("이어받은 이름 1명", page.summary.text())
        page.set_name("dbscan:person:zzzz9999", "홍길동")                               # 이 결과의 이름이 우선
        self.assertTrue(any(page.grid.item(i).text().startswith("홍길동") for i in range(page.grid.count())))
        # 마지막으로 연 결과가 다음 페이지의 기본 선택
        self.assertEqual(PI.load_last_run(self.root, "person"), str(folder_c / "person_dbscan_assignments.jsonl"))
        page2 = PP.PeoplePage(root=self.root, fetch=self.fetch, auto_label_on_load=False)
        self.assertEqual(page2.current_run()["run"], "run_c")
        self.assertIn("라벨 없음", page2.run_combo.currentText())
        _write_labels(run_a["folder"])
        self.assertIn("라벨: 문장·색상", PI.run_title(PI.find_runs(self.root)[[r["run"] for r in PI.find_runs(self.root)].index("run_a")]))
        # 라벨러 로그 파일
        page._start_label_worker([sys.executable, "-c", "print('hello log')"])
        self._wait_label(page)
        log = folder_c / "auto_label_custom.log"
        self.assertEqual(page.label_log_path, log)
        text = log.read_text(encoding="utf-8")
        self.assertIn("hello log", text)
        self.assertTrue(text.rstrip().endswith("[exit 0]"))

    def test_route_person_search(self):
        """'이 사람으로 검색' 라우팅: 영상 결과는 영상에서 찾기, 사진 결과는 사진에서 찾기 · 대상 맞춤 · 검색 버튼까지."""
        import search_gui as SG
        from gui.search_ui import _scope_combo

        class FakePage:
            def __init__(self):
                self.image_scope = _scope_combo()
                self.query = None
                self.clicked = 0

                class Btn:
                    def __init__(btn):
                        btn.page = self

                    def click(btn):
                        btn.page.clicked += 1
                self.image_search_btn = Btn()

            def _use_result_as_query(self, path):
                self.query = path

        img, vid, selected = FakePage(), FakePage(), []
        key = SG.route_person_search(img, vid, selected.append, "a.jpg", "video", "object", run=False)
        self.assertEqual((key, selected, vid.query, img.query), ("video_search", ["video_search"], "a.jpg", None))
        self.assertEqual(vid.image_scope.currentData(), "object")
        key = SG.route_person_search(img, vid, selected.append, "b.jpg", "image", "person")
        self.assertEqual((key, img.query, img.image_scope.currentData()), ("image_search", "b.jpg", "person"))
        for _ in range(5):
            self.app.processEvents()
        self.assertEqual((img.clicked, vid.clicked), (1, 0))                    # run=True 면 검색 버튼을 누른다

    def _wait_label(self, page):
        self.assertTrue(page.label_worker.wait(15000))
        for _ in range(20):
            self.app.processEvents()

    def test_auto_label_on_load(self):
        """라벨 파일이 없는 결과(영상이든 사진이든)를 불러오면 색상 라벨러가 저절로 돈다 — 실행마다 한 번, 실패해도 대화상자 없음."""
        from gui import people_page as PP

        calls = []

        def fake_cmd(run, tool, root=None, python=None):
            calls.append((run["run"], tool))
            out = PI.label_output_dir(run, tool) / "cluster_labels.jsonl"
            rec = json.dumps({"cluster_id": "leiden:person:aaaa1111", "cluster_name": "파란색 상의", "status": "labeled",
                              "label_confidence": 0.9}, ensure_ascii=False)
            return [sys.executable, "-c", f"import pathlib; p = pathlib.Path({str(out)!r}); p.parent.mkdir(parents=True, exist_ok=True); "
                                          f"p.write_text({rec!r} + '\\n', encoding='utf-8')"]

        orig_cmd, orig_warn = PP.PI.auto_label_command, PP.QMessageBox.warning
        PP.PI.auto_label_command = fake_cmd
        PP.QMessageBox.warning = lambda *a, **k: (_ for _ in ()).throw(AssertionError("자동 실행 실패에 대화상자"))
        try:
            page = PP.PeoplePage(root=self.root, fetch=self.fetch)
            self.assertTrue(page.auto_check.isChecked())
            page.load_sync()
            self.assertEqual(calls, [("run_a", "vec")])
            self.assertIn("자동으로 만드는 중", page.summary.text())
            self.assertEqual(page.label_btn.text(), "중단")
            self._wait_label(page)
            self.assertEqual(page.labels["leiden:person:aaaa1111"]["name"], "파란색 상의")
            self.assertTrue(page.grid.item(0).text().startswith("파란색 상의\n"))
            self.assertIn("자동 라벨 완료", page.summary.text())
            # 다시 불러와도(라벨 있음) 또 돌지 않는다
            page.load_sync()
            self.assertEqual(len(calls), 1)
            self.assertFalse(page._label_running())

            # 실패하는 자동 실행: 대화상자 없이 요약줄만, 같은 세션에서 반복하지 않는다
            folder2 = self.root / "outputs" / "clustering" / "run_b" / "person"
            folder2.mkdir(parents=True)
            (folder2 / "person_leiden_assignments.jsonl").write_text(
                json.dumps(_assign("p3", "leiden:person:bbbb2222")) + "\n", encoding="utf-8")
            PP.PI.auto_label_command = lambda run, tool, root=None, python=None: (
                calls.append((run["run"], tool)) or [sys.executable, "-c", "import sys; print('no vectors'); sys.exit(2)"])
            page.refresh_runs()
            page.run_combo.setCurrentIndex(next(i for i in range(page.run_combo.count()) if page.run_combo.itemData(i)["run"] == "run_b"))
            page.load_sync()
            self.assertEqual(calls[-1], ("run_b", "vec"))
            self._wait_label(page)
            self.assertIn("자동 라벨 실패 (종료 코드 2)", page.summary.text())
            page.load_sync()
            self.assertEqual(len(calls), 2)                                    # 실패한 실행은 세션 안에서 다시 자동 시작하지 않음

            # 끄면 돌지 않는다
            page._auto_tried.clear()
            page.auto_check.setChecked(False)
            page.load_sync()
            self.assertEqual(len(calls), 2)

            # 라벨러가 도는 사이 다른 결과를 불러오면: 끝난 결과는 지금 화면에 섞지 않는다
            page.auto_check.setChecked(True)
            PP.PI.auto_label_command = fake_cmd
            page.load_sync()                                                    # run_b: 자동 시작 (fake_cmd 는 run_b/labels_vec 에 aaaa 라벨)
            self.assertEqual(calls[-1], ("run_b", "vec"))
            page.run_combo.setCurrentIndex(next(i for i in range(page.run_combo.count()) if page.run_combo.itemData(i)["run"] == "run_a"))
            page.load_sync()                                                    # 도는 중에 run_a 로 전환
            self._wait_label(page)
            self.assertEqual(page.run["run"], "run_a")
            self.assertIn("run_b 자동 라벨 완료", page.summary.text())
            self.assertEqual(page.labels["leiden:person:aaaa1111"]["dir"], "labels_vec")
            self.assertEqual([s["dir"] for s in PI.find_labels({"folder": folder2})], ["labels_vec"])   # run_b 라벨은 run_b 폴더에
        finally:
            PP.PI.auto_label_command, PP.QMessageBox.warning = orig_cmd, orig_warn


if __name__ == "__main__":
    unittest.main()
