"""오프라인 테스트: gui/pipeline_page 의 choice 자동 나열(choices_glob / choices_yaml_key / choices_unique_by / choices_label_key)
과 registry 검증, 그리고 실제 gui_pipelines.json 의 검출기·임베더 드롭다운이 모델 수만큼만 보이는지."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui import pipeline_page as pp  # noqa: E402


def values(items):
    return [v for v, _ in items]


class ChoiceValuesTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        (self.root / "pipeline.yaml").write_text(
            "detector:\n  module: a.rf\n  class: RF\n  params: {conf_threshold: 0.5}\nretrievers:\n  siglip2: {dim: 768}\n  irra: {dim: 512}\n",
            encoding="utf-8")
        (self.root / "pipeline_copy.yaml").write_text(
            "detector:\n  module: a.rf\n  class: RF\n  params: {conf_threshold: 0.2}\nretrievers:\n  siglip2: {dim: 768}\n  irra: {dim: 512}\n",
            encoding="utf-8")
        (self.root / "pipeline_tracking_yolo26.yaml").write_text("detector:\n  module: b.yolo\n  class: YOLO\n", encoding="utf-8")
        (self.root / "pipeline_other.yaml").write_text("retrievers:\n  siglip2: {dim: 1152}\n", encoding="utf-8")
        (self.root / "pipeline_broken.yaml").write_text("detector: [unclosed\n", encoding="utf-8")
        (self.root / "notes.yaml").write_text("detector: {}\n", encoding="utf-8")

    def tearDown(self):
        self.td.cleanup()

    def test_glob_filtered_by_yaml_key(self):
        spec = {"type": "choice", "default": "pipeline.yaml", "choices_glob": "pipeline*.yaml", "choices_yaml_key": "detector"}
        self.assertEqual(values(pp._choice_values(spec, self.root)), ["pipeline.yaml", "pipeline_copy.yaml", "pipeline_tracking_yolo26.yaml"])
        spec = {"type": "choice", "default": "pipeline.yaml", "choices_glob": "pipeline*.yaml", "choices_yaml_key": "retrievers"}
        self.assertEqual(values(pp._choice_values(spec, self.root)), ["pipeline.yaml", "pipeline_copy.yaml", "pipeline_other.yaml"])

    def test_unique_by_subkeys_keeps_default_copy_and_labels_by_class(self):
        spec = {"type": "choice", "default": "pipeline_copy.yaml", "choices_glob": "pipeline*.yaml", "choices_yaml_key": "detector",
                "choices_unique_by": ["module", "class"], "choices_label_key": "class"}
        items = pp._choice_values(spec, self.root)
        # 같은 module/class(RF) 인 pipeline.yaml / pipeline_copy.yaml 중 default 쪽만 남고, YOLO 는 별도 항목
        self.assertEqual(items, [("pipeline_copy.yaml", "RF  (pipeline_copy.yaml)"),
                                 ("pipeline_tracking_yolo26.yaml", "YOLO  (pipeline_tracking_yolo26.yaml)")])

    def test_unique_by_whole_block_and_retriever_label(self):
        spec = {"type": "choice", "default": "pipeline.yaml", "choices_glob": "pipeline*.yaml", "choices_yaml_key": "retrievers",
                "choices_unique_by": "*"}
        items = pp._choice_values(spec, self.root)
        self.assertEqual(values(items), ["pipeline.yaml", "pipeline_other.yaml"])   # copy 는 retrievers 가 같아 합쳐짐
        self.assertEqual(items[0][1], "siglip2, irra  (pipeline.yaml)")

    def test_glob_without_key_lists_all_files_and_keeps_default_first_when_missing(self):
        spec = {"type": "choice", "default": "custom.yaml", "choices_glob": ["pipeline*.yaml", "notes.yaml"]}
        got = values(pp._choice_values(spec, self.root))
        self.assertEqual(got[0], "custom.yaml")
        self.assertEqual(set(got[1:]), {"pipeline.yaml", "pipeline_copy.yaml", "pipeline_tracking_yolo26.yaml", "pipeline_other.yaml",
                                        "pipeline_broken.yaml", "notes.yaml"})

    def test_static_choices_preserved_and_deduplicated(self):
        spec = {"type": "choice", "default": "b", "choices": ["a", "b"], "choices_glob": "nothing*.yaml"}
        self.assertEqual(pp._choice_values(spec, self.root), [("a", "a"), ("b", "b")])
        self.assertEqual(values(pp._choice_values({"type": "choice", "choices": ["x", "x", "y"]}, self.root)), ["x", "y"])

    def test_registry_validation_accepts_glob_choice(self):
        stages = [{"id": "s", "title": "t", "script": "x.py",
                   "args": [{"flag": "--c", "type": "choice", "choices_glob": "pipeline*.yaml"}]}]
        pp._check_stages("test", "g", stages)   # 예외 없음
        with self.assertRaises(pp.RegistryError):
            pp._check_stages("test", "g", [{"id": "s", "title": "t", "script": "x.py", "args": [{"flag": "--c", "type": "choice"}]}])

    def test_live_registry_detector_dropdowns(self):
        """검출기 드롭다운: 검출기 블록이 통째로 같은 사본만 합친다(같은 class 라도 가중치가 다르면 별도 항목, 예 yolo26 / yolo26s).
        영상 1단계는 pipeline_tracking*.yaml 을 파일마다 한 항목으로 보여 스티처 변형(sushi_link)도 고를 수 있다."""
        groups = {g["id"]: g for g in pp.load_registry()}
        for gid, stage_id, flag in (("video_pipeline", "video_preprocess", "--tracking-config"),
                                    ("image_pipeline", "image_detect", "--detector-config"),
                                    ("evaluation", "detect_eval", "--detector-config")):
            stage = next(s for s in groups[gid]["stages"] if s["id"] == stage_id)
            arg = next(a for a in stage["args"] if a["flag"] == flag)
            items = pp._choice_values(arg)
            with self.subTest(stage=stage_id):
                labels = [label for _, label in items]
                self.assertTrue(any(label.startswith("RFDETRDetector") for label in labels), labels)
                self.assertEqual(sum(1 for label in labels if label.startswith("YOLO26Detector")), 2, labels)   # yolo26 + yolo26s
                self.assertIn("pipeline_tracking_yolo26.yaml", values(items))
                self.assertIn("pipeline_tracking_yolo26s.yaml", values(items))
                self.assertEqual(values(items)[0], arg["default"])
                if stage_id == "video_preprocess":
                    self.assertIn("pipeline_tracking_sushi_link.yaml", values(items))
                    self.assertTrue(all(v == arg["default"] or v.startswith("pipeline_tracking") for v in values(items)), values(items))   # 기본(운영 pipeline.yaml) + tracking 변형들
                else:
                    self.assertNotIn("pipeline_tracking_sushi_link.yaml", values(items))       # 검출기 블록이 tracking.yaml 과 같아 합쳐짐
        build = next(s for s in groups["image_pipeline"]["stages"] if s["id"] == "image_build")
        cfg = next(a for a in build["args"] if a["flag"] == "--config")
        items = pp._choice_values(cfg)
        self.assertEqual(values(items)[0], "pipeline.yaml")
        self.assertNotIn("pipeline_tracking.yaml", values(items))   # retrievers 블록 없음
        self.assertTrue(items[0][1].startswith("siglip2"), items[0][1])


if __name__ == "__main__":
    unittest.main()
