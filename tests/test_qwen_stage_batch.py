"""qwen_stage 배치 관찰(--batch-size) — 모델 없이: 묶음 크기, 단건 폴백, crop 없는 후보 제외, 시그니처/페이로드 키."""
from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verifiers import qwen_stage as q  # noqa: E402


def _obs(color="black"):
    return {"inventory": ["shirt"], "checks": [
        {"object": "shirt", "attribute": "color", "observed": color, "visibility": "sufficient", "evidence": "x"}]}


class FakeQwen:
    def __init__(self, fail_batches=False):
        self.batches = []
        self.singles = []
        self.fail_batches = fail_batches

    def observe_batch(self, paths, constraints):
        self.batches.append(list(paths))
        if self.fail_batches:
            raise RuntimeError("boom")
        return [_obs("black" if i % 2 == 0 else "red") for i, _ in enumerate(paths)]

    def observe(self, path, constraints):
        self.singles.append(path)
        return _obs("black")


class BatchObserveTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.paths = []
        for i in range(5):
            p = Path(self.td.name) / f"c{i}.jpg"
            p.write_bytes(b"x")
            self.paths.append(str(p))
        self.cons = [q.Constraint("shirt", "color", "black", 1.0, True)]

    def tearDown(self):
        self.td.cleanup()

    def _item(self, paths):
        return {"query_text": "black shirt", "results": [
            {"rank": i + 1, "crop_path": p, "qdrant_score": 1.0 - i * 0.1, "point_id": f"p{i}"} for i, p in enumerate(paths)]}

    def test_batches_chunk_and_score_all(self):
        fake = FakeQwen()
        item = self._item(self.paths)
        q.process_item(item, fake, self.cons, top_k=5, alpha=0.7, threshold=0.5, verify_mode="flag", rescore_only=False, batch_size=2)
        self.assertEqual([len(b) for b in fake.batches], [2, 2, 1])
        self.assertEqual(fake.singles, [])
        rows = item["results"]
        self.assertTrue(all(r.get("attr_score") is not None for r in rows))
        by_id = {r["point_id"]: r for r in rows}
        self.assertEqual(by_id["p0"]["attr_score"], 1.0)     # 배치 안 짝수 = black = PASS
        self.assertEqual(by_id["p1"]["attr_score"], 0.0)     # red = 필수 FAIL

    def test_batch_failure_falls_back_to_single(self):
        fake = FakeQwen(fail_batches=True)
        item = self._item(self.paths)
        q.process_item(item, fake, self.cons, top_k=5, alpha=0.7, threshold=0.5, verify_mode="flag", rescore_only=False, batch_size=3)
        self.assertEqual(len(fake.batches), 2)
        self.assertEqual(sorted(fake.singles), sorted(self.paths))
        self.assertTrue(all(r.get("attr_score") == 1.0 for r in item["results"]))

    def test_missing_crop_is_skipped_not_batched(self):
        fake = FakeQwen()
        paths = self.paths[:2] + [str(Path(self.td.name) / "missing.jpg")]
        item = self._item(paths)
        q.process_item(item, fake, self.cons, top_k=5, alpha=0.7, threshold=0.5, verify_mode="flag", rescore_only=False, batch_size=10)
        self.assertEqual(fake.batches, [self.paths[:2]])
        missing = next(r for r in item["results"] if r["point_id"] == "p2")
        self.assertEqual(missing.get("attr_skipped"), "crop file not found")

    def test_batch_size_one_uses_single_observe(self):
        fake = FakeQwen()
        item = self._item(self.paths[:3])
        q.process_item(item, fake, self.cons, top_k=5, alpha=0.7, threshold=0.5, verify_mode="flag", rescore_only=False)
        self.assertEqual(fake.batches, [])
        self.assertEqual(len(fake.singles), 3)

    def test_signatures(self):
        self.assertEqual(inspect.signature(q.run).parameters["batch_size"].default, 1)
        self.assertEqual(inspect.signature(q.process_item).parameters["batch_size"].default, 1)
        self.assertTrue(hasattr(q.QwenVL, "generate_batch") and hasattr(q.QwenVL, "observe_batch"))


if __name__ == "__main__":
    unittest.main()
