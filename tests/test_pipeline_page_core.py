"""파이프라인 탭 단순화 (Qt offscreen, 프로세스 실행 없음):
- 핵심 단계(core) 만 먼저 보이고 '추가 작업 보기' 로 나머지가 펴진다
- basic 필드만 보이고 '고급 옵션 보기' 로 나머지가 펴진다
- str/dir 필드에 choices/choices_dirs 가 있으면 고르거나 직접 칠 수 있는 콤보가 되고 값·플레이스홀더가 맞다
- choices_dirs / choices_exclude / default_prefer / default_first_choice 해석
- 실제 gui_pipelines.json: 이미지·영상 파이프라인 핵심 4단계, 검출 단계 빈 칸 없음(필수 4칸 모두 값 또는 폴더 목록)
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("TRANSREID_NO_WEBENGINE", "1")     # 테스트는 QTextBrowser 폴백 (WebEngine 프로세스 없이)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui import pipeline_page as pp  # noqa: E402


class ChoiceDirTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        for d in ("data/coco/val2017", "data/PRW/frames", "data/PRW/annotations", "data/crops", "data/prw_crops_p25h75", "data/videos"):
            (self.root / d).mkdir(parents=True)
        (self.root / "data" / "note.txt").write_text("x", encoding="utf-8")

    def tearDown(self):
        self.td.cleanup()

    def test_choices_dirs_lists_directories_with_exclude(self):
        spec = {"type": "dir", "default": "", "choices_dirs": ["data/*", "data/*/*"],
                "choices_exclude": ["crops*", "*crop*", "videos", "annotations"]}
        got = [v for v, _ in pp._choice_values(spec, self.root)]
        self.assertEqual(got, ["data/coco", "data/PRW", "data/coco/val2017", "data/PRW/frames"])   # 패턴 순 → 대소문자 무시 정렬

    def test_default_prefer_then_first_choice(self):
        spec = {"type": "dir", "default": "", "choices_dirs": ["data/*/*"], "choices_exclude": ["annotations"],
                "default_prefer": ["data/coco/train2017", "data/coco/val2017"], "default_first_choice": True}
        self.assertEqual(pp.resolve_default(spec, self.root), "data/coco/val2017")      # train2017 없음 → val2017
        spec["default_prefer"] = ["data/nope"]
        self.assertEqual(pp.resolve_default(spec, self.root), "data/coco/val2017")      # 목록 첫 항목
        spec["default"] = "data/given"
        self.assertEqual(pp.resolve_default(spec, self.root), "data/given")             # 명시 default 우선
        self.assertEqual(pp.resolve_default({"type": "str", "default": ""}, self.root), "")


class QtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit
        cls.app = QApplication.instance() or QApplication([])
        cls.QComboBox, cls.QLineEdit = QComboBox, QLineEdit

    def test_editable_combo_and_placeholder(self):
        from PySide6.QtWidgets import QWidget
        host = QWidget()
        f = pp.ArgField({"flag": "--source", "type": "str", "default": "COCO", "choices": ["COCO", "prw_image"]}, host)
        self.assertIsInstance(f.widget, self.QComboBox)
        self.assertTrue(f.widget.isEditable())
        self.assertEqual(f.value(), "COCO")
        self.assertEqual(f.to_argv(), ["--source", "COCO"])
        f.set_value("mine")                                        # 목록에 없는 값도 직접 입력
        self.assertEqual(f.value(), "mine")
        g = pp.ArgField({"flag": "--qdrant-url", "type": "str", "default": "", "placeholder": "비우면 pipeline.yaml"}, host)
        self.assertIsInstance(g.widget, self.QLineEdit)
        self.assertEqual(g.widget.placeholderText(), "비우면 pipeline.yaml")
        self.assertEqual(g.to_argv(), [])

    def _group(self):
        st = lambda i, core=False, args=None: {"id": f"s{i}", "title": f"{i}. 기술 제목 {i}", "script": "report/build_image_results.py",
                                                 "description": f"기술 설명 {i} (foo.json)",
                                                 **({"core": True, "core_title": f"{i}. 쉬운 제목", "core_desc": f"쉬운 설명 {i}"} if core else {}), "args": args or []}
        args = [{"flag": "--in", "label": "입력", "type": "str", "default": "a", "basic": True},
                {"flag": "--req", "label": "필수", "type": "str", "default": "r", "required": True},
                {"flag": "--adv", "label": "고급", "type": "int", "default": 3},
                {"flag": "--adv2", "label": "고급2", "type": "bool", "default": False}]
        return {"id": "g", "title": "G", "stages": [st(1, True, args), st(2), st(3, True), st(4), st(5)]}

    def test_core_list_and_advanced_toggle(self):
        page = pp.PipelineGroupPage(self._group())
        self.assertEqual(page.visible_titles(), ["1. 쉬운 제목", "3. 쉬운 제목"])
        self.assertIsNotNone(page.more_check)
        self.assertIn("(3)", page.more_check.text())
        page.more_check.setChecked(True)
        self.assertEqual(len(page.visible_titles()), 5)
        page.more_check.setChecked(False)
        self.assertEqual(len(page.visible_titles()), 2)
        panel = page.panel                                          # 첫 단계(core, basic 필드 있음) 선택 상태
        self.assertEqual(panel.title.text(), "1. 쉬운 제목")
        self.assertEqual(panel.desc.text(), "쉬운 설명 1")               # 파일명 든 기술 설명 대신
        self.assertIn("1. 기술 제목 1", panel.script_label.text())
        self.assertTrue(panel.script_label.isHidden())                 # 핵심 단계: 파일명 줄은 숨기고 툴팁으로만
        self.assertIn("build_image_results.py", panel.title.toolTip())
        self.assertTrue(panel.adv_check.isVisible() or not panel.adv_check.isHidden())
        adv = [f for f in panel.fields if getattr(f, "advanced", False)]
        self.assertEqual([f.flag for f in adv], ["--adv", "--adv2"])
        self.assertTrue(all(f.container.isHidden() for f in adv))
        panel.adv_check.setChecked(True)
        self.assertFalse(any(f.container.isHidden() for f in adv))
        # basic 없는 단계 → 체크박스 숨김, 전부 표시
        panel.set_stage(self._group()["stages"][1])
        self.assertTrue(panel.adv_check.isHidden())
        self.assertFalse(panel.script_label.isHidden())                # 일반(추가 작업) 단계는 스크립트 줄이 보인다
        self.assertEqual(panel.title.toolTip(), "")

    def test_tools_switch_stage_and_presets(self):
        g = self._group()
        g["stages"][2]["tools"] = [{"label": "기본", "stage": "s3"},
                                   {"label": "다른 도구", "stage": "s4", "set": {"--m": "dbscan"}, "description": "설명 D"}]
        g["stages"][3]["args"] = [{"flag": "--m", "label": "알고리즘", "type": "str", "default": "leiden"},
                                  {"flag": "--x", "label": "기타", "type": "int", "default": 1}]
        g["stages"][3]["script"] = "report/build_image_db_html.py"
        page = pp.PipelineGroupPage(g)
        page.listw.setCurrentRow(1)                                 # 보이는 두 번째 = s3 (core)
        panel = page.panel
        self.assertFalse(panel.tool_row.isHidden())
        self.assertEqual(panel.tool_combo.count(), 2)
        self.assertEqual(panel.stage["id"], "s3")
        panel.tool_combo.setCurrentIndex(1)
        self.assertEqual(panel.stage["id"], "s4")                  # 실행 대상은 s4
        self.assertEqual(panel.title.text(), "3. 쉬운 제목")          # 제목은 핵심 단계 것
        self.assertEqual(panel.desc.text(), "설명 D")
        panel.tool_combo.setCurrentIndex(0)
        self.assertEqual(panel.desc.text(), "쉬운 설명 3")               # 도구 설명이 없으면 핵심 단계의 쉬운 설명
        panel.tool_combo.setCurrentIndex(1)                              # 아래 검사는 다시 '다른 도구' 기준
        self.assertIn("build_image_db_html.py", " ".join(panel._build_cmd()))
        m = next(f for f in panel.fields if f.flag == "--m")
        self.assertEqual(m.value(), "dbscan")                       # set 프리셋이 기본값
        self.assertFalse(m.advanced)                                # 프리셋 필드는 basic
        self.assertTrue(next(f for f in panel.fields if f.flag == "--x").advanced)
        page.listw.setCurrentRow(0)
        self.assertTrue(panel.tool_row.isHidden())                  # 도구 없는 단계
        page.listw.setCurrentRow(1)
        self.assertEqual(panel.tool_combo.currentIndex(), 1)        # 선택 기억
        self.assertEqual(panel.stage["id"], "s4")

    def test_registry_rejects_unknown_tool_stage(self):
        with self.assertRaises(pp.RegistryError):
            pp._check_stages("x", "g", [{"id": "a", "title": "A", "script": "s.py", "tools": [{"label": "t", "stage": "nope"}]}])
        pp._check_stages("x", "g", [{"id": "a", "title": "A", "script": "s.py", "tools": [{"label": "t", "stage": "a"}]}])

    def test_result_shown_inline_after_run(self):
        import tempfile as _tf
        import time as _time
        page = pp.PipelineGroupPage(self._group())
        panel = page.panel
        self.assertEqual(panel.bottom_tabs.count(), 2)
        with _tf.TemporaryDirectory() as td:
            html = Path(td) / "r.html"
            html.write_text("<html><body><h1>결과</h1></body></html>", encoding="utf-8")
            panel._result_path = html
            panel._run_started = _time.time()
            panel._done(0)
            self.assertTrue(panel.open_btn.isEnabled())
            self.assertIs(panel.bottom_tabs.currentWidget(), panel.result_view)      # 결과 탭으로 넘어감
            self.assertEqual(panel.result_view.path, html)
            self.assertTrue(panel.result_view.ext_btn.isEnabled())
            panel._result_path = Path(td) / "missing.html"
            panel._done(0)                                                          # 없는 파일 → 뷰는 그대로, 경고만
            self.assertEqual(panel.result_view.path, html)

    def test_group_without_core_shows_everything(self):
        g = self._group()
        for st in g["stages"]:
            st.pop("core", None)
        page = pp.PipelineGroupPage(g)
        self.assertEqual(len(page.visible_titles()), 5)
        self.assertIsNone(page.more_check)


class LiveRegistryTests(unittest.TestCase):
    def test_core_stages_and_detect_defaults(self):
        groups = {g["id"]: g for g in pp.load_registry()}
        for gid in ("image_pipeline", "video_pipeline"):
            core = [s for s in groups[gid]["stages"] if s.get("core")]
            self.assertEqual(len(core), 4, gid)
            self.assertEqual([s["core_title"][:2] for s in core], ["1.", "2.", "3.", "4."], gid)
            self.assertTrue(all(s.get("core_title") for s in core))
        det = next(s for s in groups["image_pipeline"]["stages"] if s["id"] == "image_detect")
        for a in det["args"]:
            if a.get("required"):
                v = pp.resolve_default(a)
                self.assertTrue(v or a.get("choices_dirs"), a["flag"])   # 값이 있거나 폴더 드롭다운
        cl = next(s for s in groups["image_pipeline"]["stages"] if s["id"] == "image_cluster")
        self.assertEqual([x["label"] for x in cl["tools"]], ["Leiden (기본)", "DBSCAN v6"])   # 두 개만, 파일명 없이
        self.assertEqual({t["stage"] for t in cl["tools"]}, {"image_cluster", "image_cluster_plugin"})
        vc = next(s for s in groups["video_pipeline"]["stages"] if s["id"] == "video_cluster_person")
        self.assertEqual([t["stage"] for t in vc["tools"]], ["video_cluster_person", "video_cluster_object"])
        res = next(s for s in groups["image_pipeline"]["stages"] if s["id"] == "image_results")
        self.assertTrue((ROOT / res["script"]).is_file())
        self.assertTrue(any(a.get("basic") for a in res["args"]))


if __name__ == "__main__":
    unittest.main()
