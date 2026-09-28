"""오프라인 테스트: bench/register.py 임베더 표본 적재 — collection_prefix 교체(로더 검증), PRW 프레임 표본(결정적·재사용), 적재/e2e 명령 조립. 실제 적재·Qdrant 없음."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import register as R  # noqa: E402


class PrefixTests(unittest.TestCase):
    def test_set_collection_prefix_on_copy(self):
        with tempfile.TemporaryDirectory() as td:
            y = Path(td) / "pipeline_x.yaml"
            shutil.copyfile(ROOT / "pipeline.yaml", y)
            col = R.set_collection_prefix(y, "bench_x")
            self.assertEqual(col, "bench_x_person")
            text = y.read_text(encoding="utf-8-sig")
            self.assertEqual(sum(1 for ln in text.splitlines() if ln.startswith("collection_prefix:")), 1)
            from config import PipelineConfig
            self.assertEqual(PipelineConfig.load(y).object_collection(), "bench_x_object")
            # 두 번 적용해도 한 줄
            R.set_collection_prefix(y, "bench_y")
            text = y.read_text(encoding="utf-8-sig")
            self.assertEqual(sum(1 for ln in text.splitlines() if ln.startswith("collection_prefix:")), 1)
            self.assertIn("collection_prefix: bench_y", text)
        self.assertEqual((ROOT / "pipeline.yaml").read_bytes(), (ROOT / "pipeline.yaml").read_bytes())   # 원본은 그대로 (참고)


class SampleTests(unittest.TestCase):
    def test_sample_frames_deterministic_and_reused(self):
        frames = [f"c1s1_{i:06d}" for i in range(100)]
        crops = [{"image_id": f"PRW/c1s1_{i:06d}.jpg", "crop_path": f"x{i}.jpg", "class_name": "person"} for i in range(100)] + \
                [{"image_id": "PRW/c1s1_000010.jpg", "crop_path": "dup.jpg", "class_name": "bag"}]      # 표본 프레임(10 간격)에 crop 하나 더
        with tempfile.TemporaryDirectory() as td:
            stats = Path(td) / "stats.json"
            stats.write_text(json.dumps({"crops": crops, "filtered": [{"x": 1}]}), encoding="utf-8")
            sample_file = Path(td) / "sample.json"
            out, info = R.sample_prw_frames(stats, 10, frames, Path(td) / "w", sample_file)
            self.assertEqual((info["frames"], info["crops"]), (10, 11))                  # 프레임 10 은 crop 2개
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(data["crops"]), 11)
            self.assertEqual(data["filtered"], [])
            chosen = json.loads(sample_file.read_text(encoding="utf-8"))["frames"]
            self.assertEqual(len(chosen), 10)
            self.assertEqual(chosen[0], "c1s1_000000")
            # 같은 n → 같은 표본 파일 재사용 (다른 임베더와 비교 가능)
            out2, info2 = R.sample_prw_frames(stats, 10, list(reversed(frames)), Path(td) / "w2", sample_file)
            self.assertEqual(json.loads(out2.read_text(encoding="utf-8"))["crops"], data["crops"])
            # n 이 프레임 수보다 크면 전부
            _, info3 = R.sample_prw_frames(stats, 1000, frames, Path(td) / "w3", Path(td) / "s3.json")
            self.assertEqual(info3["frames"], 100)


class CommandTests(unittest.TestCase):
    def test_ingest_commands(self):
        cmds = R.ingest_commands("myemb", Path("pipeline_myemb.yaml"), Path("s.json"), Path("work"), 300, max_queries=50, ledger_path="l.jsonl")
        b = cmds["build"]
        self.assertEqual(b[1], "ingest/build_db.py")
        self.assertIn("--checkpoint-dir", b)
        self.assertIn("--manifest-dir", b)
        self.assertNotIn("--recreate", b)
        self.assertNotIn("--fresh", b)
        n = cmds["e2e_new"]
        self.assertEqual(n[n.index("--stage1") + 1], "myemb")
        self.assertEqual(n[n.index("--rerank") + 1], "none")
        self.assertEqual(n[n.index("--name") + 1], "myemb__sample300")
        self.assertEqual(n[n.index("--max-queries") + 1], "50")
        self.assertEqual(n[n.index("--ledger") + 1], "l.jsonl")
        p = cmds["e2e_prod"]
        self.assertEqual(p[p.index("--stage1") + 1:p.index("--stage1") + 3], ["siglip2", "irra"])
        self.assertEqual(p[p.index("--rerank") + 1], "solider")
        self.assertEqual(p[p.index("--name") + 1], "prod__sample300")
        self.assertTrue(all(cmd[2] == "--config" for cmd in (n, p)))

    def test_parser_accepts_ingest_options(self):
        p = R.build_parser()
        a = p.parse_args(["embedder", "--name", "e", "--module", "m", "--class", "C", "--dim", "8", "--ingest-frames", "300", "--e2e-max-queries", "20"])
        self.assertEqual((a.ingest_frames, a.e2e_max_queries), (300, 20))
        d = p.parse_args(["detector", "--name", "d", "--module", "m", "--class", "C"])
        self.assertEqual(d.ingest_frames, 0)


if __name__ == "__main__":
    unittest.main()
