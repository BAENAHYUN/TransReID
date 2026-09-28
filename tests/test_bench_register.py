"""오프라인 테스트: bench/register.py — 가짜 어댑터 3종으로 계약 검사·yaml 등록·벤치 명령·순위·템플릿 (Qdrant·GPU 없음)."""
import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import ledger as L  # noqa: E402
from bench import register as R  # noqa: E402

FAKE_DET = '''
from typing import List, Optional
from detect.base import BaseDetector, Detection
class FakeDetector(BaseDetector):
    def __init__(self, weights: str = "w.pt", conf_threshold: float = 0.2, filter_forensic: bool = True):
        self.weights, self.conf_threshold, self.filter_forensic = weights, float(conf_threshold), bool(filter_forensic)
    def detect(self, frame, *, frame_idx: int, timestamp_sec: Optional[float] = None) -> List[Detection]:
        return [Detection(frame_idx=frame_idx, bbox=(1.0, 2.0, 30.0, 80.0), confidence=0.9, class_id=1, class_name="person", timestamp_sec=timestamp_sec)]
class BrokenDetector(BaseDetector):
    def __init__(self, x: int):
        self.x = x
'''
FAKE_CLU = '''
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from clustering.base import BaseClusterer, ClusterResult
class FakeClusterer(BaseClusterer):
    name = "fake"
    def __init__(self, k: int = 2):
        self.k = int(k)
    def params(self): return {"k": self.k}
    def cluster(self, ids, primary, vectors, log=print):
        return ClusterResult(labels=[int(np.argmax(primary[i, :self.k])) for i in range(len(ids))], stats={})
'''
FAKE_EMB = '''
from typing import List, Optional
import numpy as np
from PIL import Image
from embedders.base import BaseEmbedder
class FakeEmbedder(BaseEmbedder):
    DIM = 8
    def __init__(self, model_id: str = "fake", device: Optional[str] = None, batch_size: int = 32, l2_normalize: bool = True):
        super().__init__(device=device, batch_size=batch_size, l2_normalize=l2_normalize)
        self.model_id = model_id
    def _encode(self, images: List[Image.Image]) -> np.ndarray:
        return np.ones((len(images), self.DIM), dtype=np.float32)
'''


class RegisterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.root = Path(cls.td.name)
        pkg = cls.root / "fakeadapters"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "det.py").write_text(FAKE_DET, encoding="utf-8")
        (pkg / "clu.py").write_text(FAKE_CLU, encoding="utf-8")
        (pkg / "emb.py").write_text(FAKE_EMB, encoding="utf-8")
        sys.path.insert(0, str(cls.root))
        shutil.copyfile(ROOT / "pipeline_tracking.yaml", cls.root / "pipeline_tracking.yaml")
        shutil.copyfile(ROOT / "pipeline.yaml", cls.root / "pipeline.yaml")

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.root))
        cls.td.cleanup()

    def test_checks_pass_and_fail(self):
        rows = R.run_checks("detector", {"module": "fakeadapters.det", "class": "FakeDetector", "params": {"weights": "a.pt"}}, None, False, True)
        self.assertTrue(all(r["status"] == "OK" for r in rows), rows)
        rows = R.run_checks("detector", {"module": "fakeadapters.det", "class": "BrokenDetector", "params": {}}, None, False, False)
        self.assertTrue(any(r["status"] == "FAIL" for r in rows))       # 필수 x 누락 + detect 미재정의
        self.assertTrue(any("미구현" in r["detail"] for r in rows if r["check"] == "contract"))
        rows = R.run_checks("clusterer", {"module": "fakeadapters.clu", "class": "FakeClusterer", "params": {"k": 2}}, None, False, True)
        self.assertTrue(all(r["status"] == "OK" for r in rows), rows)
        rows = R.run_checks("embedder", {"module": "fakeadapters.emb", "class": "FakeEmbedder", "params": {"model_id": "x"}}, 8, False, True)
        self.assertTrue(all(r["status"] in ("OK", "SKIP") for r in rows), rows)
        rows = R.run_checks("embedder", {"module": "fakeadapters.emb", "class": "FakeEmbedder", "params": {}}, 16, False, False)
        self.assertTrue(any(r["check"] == "dim" and r["status"] == "FAIL" for r in rows))

    def test_yaml_writers(self):
        import yaml
        out = R.write_detector_yaml("fake_det", {"module": "fakeadapters.det", "class": "FakeDetector", "params": {"weights": "a.pt"}},
                                    self.root, self.root / "pipeline_tracking.yaml")
        d = yaml.safe_load(out.read_text(encoding="utf-8"))
        self.assertEqual((out.name, d["detector"]["class"], d["detector"]["params"]["weights"]), ("pipeline_tracking_fake_det.yaml", "FakeDetector", "a.pt"))
        self.assertIn("tracker", d)
        with self.assertRaises(FileExistsError):
            R.write_detector_yaml("fake_det", {"module": "m", "class": "C"}, self.root, self.root / "pipeline_tracking.yaml")
        out = R.write_clusterer_yaml("fake_clu", {"module": "fakeadapters.clu", "class": "FakeClusterer", "params": {"k": 2}}, self.root)
        self.assertEqual(yaml.safe_load(out.read_text(encoding="utf-8"))["clusterer"]["params"], {"k": 2})
        out = R.write_embedder_yaml("fakeemb", {"module": "fakeadapters.emb", "class": "FakeEmbedder", "params": {"model_id": "x"}}, 8, "person", False, 0.7,
                                    self.root, self.root / "pipeline.yaml")
        from config import PipelineConfig
        cfg = PipelineConfig.load(out)
        spec = cfg.retrievers["fakeemb"]
        self.assertEqual((spec.dim, spec.weight, spec.module, spec.class_name, spec.params["model_id"], spec.scope), (8, 0.7, "fakeadapters.emb", "FakeEmbedder", "x", "person"))
        self.assertEqual(set(cfg.retrievers) >= {"siglip2", "irra", "solider", "dinov2"}, True)
        text = out.read_text(encoding="utf-8")
        self.assertIn("# naflex", text)                                # 원본 주석 보존
        with self.assertRaises(ValueError):                               # 같은 retriever 이름은 overwrite 로도 허용하지 않는다
            R.write_embedder_yaml("siglip2", {"module": "m", "class": "C"}, 8, "all", False, 1.0, self.root, self.root / "pipeline.yaml", overwrite=True)
        self.assertEqual((ROOT / "pipeline.yaml").read_bytes(), (self.root / "pipeline.yaml").read_bytes())   # 원본 불변

    def test_bench_command_and_rank(self):
        cmd = R.bench_command("detector", "x", Path("p.yaml"), limit=300, ledger_path="l.jsonl")
        self.assertEqual(cmd[1:], ["bench/run.py", "detect", "--config", "p.yaml", "--name", "x", "--no-images", "--limit", "300", "--ledger", "l.jsonl"])
        self.assertIn("--method-config", R.bench_command("clusterer", "c", Path("c.yaml"), max_points=3000))
        self.assertEqual(R.bench_command("embedder", "e", Path("p.yaml"))[2:6], ["embed", "--model", "e", "--config"])
        led = self.root / "ledger.jsonl"
        env = {"host": "h"}
        L.append_entries(led, [L.make_entry("detect", "t", "a", env=env, metrics={"ap50": 0.90}),
                               L.make_entry("detect", "t", "b", env=env, metrics={"ap50": 0.80, "max_recall": 0.95}),
                               L.make_entry("detect", "t", "c", env=env, metrics={"ap50": 0.85})])
        r = R.rank_in_ledger("detect", "c", led)
        self.assertEqual((r["objective"], r["position"], r["total"]), ("ap50", 2, 3))
        self.assertEqual(r["top"][0][0], "a")
        self.assertIsNone(R.rank_in_ledger("detect", "zzz", led)["position"])

    def test_register_flow_no_bench_and_check_only(self):
        logs = []
        res = R.register("clusterer", "fake_clu2", {"module": "fakeadapters.clu", "class": "FakeClusterer", "params": {"k": 2}}, root=self.root,
                         no_bench=True, log=logs.append)
        self.assertIsNone(res.get("error"))
        self.assertTrue(Path(res["yaml"]).is_file())
        res = R.register("detector", "broken", {"module": "fakeadapters.det", "class": "BrokenDetector", "params": {}}, root=self.root,
                         instantiate=False, log=logs.append)
        self.assertIn("FAIL", res["error"])
        self.assertIsNone(res["yaml"])
        res = R.register("embedder", "fakeemb2", {"module": "fakeadapters.emb", "class": "FakeEmbedder", "params": {}}, root=self.root, dim=8,
                         check_only=True, log=logs.append)
        self.assertIsNone(res["yaml"])
        self.assertTrue(any("--check-only" in s for s in logs))

    def test_template_and_cli(self):
        out = self.root / "tmpl"
        path, module, cls = R.write_template("detector", "my_det", out / "my_det_detector.py", root=self.root)
        self.assertEqual((path.name, cls), ("my_det_detector.py", "MyDetDetector"))
        self.assertIn("class MyDetDetector(BaseDetector)", path.read_text(encoding="utf-8"))
        self.assertEqual(module, "tmpl.my_det_detector")
        path, module, cls = R.write_template("embedder", "my_emb", out / "my_emb_embedder.py", dim=256, root=self.root)
        self.assertIn("DIM = 256", path.read_text(encoding="utf-8"))
        path, _, cls = R.write_template("clusterer", "my_clu", out / "my_clu.py", root=self.root)
        self.assertIn('name = "my_clu"', path.read_text(encoding="utf-8"))
        with self.assertRaises(FileExistsError):
            R.write_template("clusterer", "my_clu", out / "my_clu.py", root=self.root)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = R.main(["clusterer", "--name", "cli_clu", "--module", "fakeadapters.clu", "--class", "FakeClusterer", "--param", "k=2",
                           "--root", str(self.root), "--no-bench"])
        self.assertEqual(code, 0)
        self.assertIn("RESULT_SUMMARY", buf.getvalue())
        self.assertTrue((self.root / "clusterer_cli_clu.yaml").is_file())
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            R.main(["embedder", "--name", "e", "--module", "m", "--class", "C"])            # --dim 필수
        self.assertEqual(R.slug("yolo26-s v2"), "yolo26_s_v2")
        self.assertEqual(R.slug("2x"), "m_2x")


if __name__ == "__main__":
    unittest.main()
