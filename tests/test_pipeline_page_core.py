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
                                                 **({"core": True, "core_title": f"{i}. 쉬운 제목"} if core else {}), "args": args or []}
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
        self.assertIn("1. 기술 제목 1", panel.script_label.text())
        self.assertTrue(panel.adv_check.isVisible() or not panel.adv_check.isHidden())
        adv = [f for f in panel.fields if getattr(f, "advanced", False)]
        self.assertEqual([f.flag for f in adv], ["--adv", "--adv2"])
        self.assertTrue(all(f.container.isHidden() for f in adv))
        panel.adv_check.setChecked(True)
        self.assertFalse(any(f.container.isHidden() for f in adv))
        # basic 없는 단계 → 체크박스 숨김, 전부 표시
        panel.set_stage(self._group()["stages"][1])
        self.assertTrue(panel.adv_check.isHidden())

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
        res = next(s for s in groups["image_pipeline"]["stages"] if s["id"] == "image_results")
        self.assertTrue((ROOT / res["script"]).is_file())
        self.assertTrue(any(a.get("basic") for a in res["args"]))


if __name__ == "__main__":
    unittest.main()
