"""오프라인 테스트: clustering/cluster_qdrant.fetch_all_vectors 의 벡터 캐시 이름에 임베더 지문과 DB 적재 회차(embedding_build_id)가 들어가는지 (Qdrant·모델 없음)."""
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clustering import cluster_qdrant as CQ  # noqa: E402


class FakeClient:
    def __init__(self, build_id):
        self.build_id = build_id

    def scroll(self, collection, limit=1, with_payload=None, with_vectors=False, **kw):
        if self.build_id is None:
            return [], None
        return [SimpleNamespace(payload={"embedding_build_id": self.build_id} if self.build_id else {})], None


class CacheTagTests(unittest.TestCase):
    def test_db_build_tag(self):
        self.assertEqual(CQ.db_build_tag(FakeClient("emb_20260917_866abdad"), "c"), "_b866abdad")
        self.assertEqual(CQ.db_build_tag(FakeClient(""), "c"), "")                 # legacy point
        self.assertEqual(CQ.db_build_tag(FakeClient(None), "c"), "")               # 빈 컬렉션
        class Boom:
            def scroll(self, *a, **k):
                raise RuntimeError("down")
        self.assertEqual(CQ.db_build_tag(Boom(), "c"), "")

    def test_cache_name_includes_fingerprint_and_build(self):
        seen = {}

        def fake_fetch(q, cfg, ids, cache, log=print):
            seen[cfg.vector] = cache
            return np.zeros((len(ids), 4), dtype=np.float32), {"missing_rows": 0}

        import clustering.cluster_leiden_qdrant as CL
        orig = CL.fetch_vectors_for
        CL.fetch_vectors_for = fake_fetch
        try:
            out = CQ.fetch_all_vectors(FakeClient("emb_20260917_866abdad"), "forensic_person", ["prw_image"], 128, ["a", "b"], ["solider"],
                                       "outputs/clustering/cache/bench/x", "person", log=lambda *_: None, config_path=str(ROOT / "pipeline.yaml"))
        finally:
            CL.fetch_vectors_for = orig
        name = seen["solider"]
        self.assertTrue(name.endswith("_b866abdad.npz"), name)
        self.assertRegex(name, r"_solider_[0-9a-f]{8}_b866abdad\.npz$")
        self.assertEqual(out["solider"]["matrix"].shape, (2, 4))
        CL.fetch_vectors_for = fake_fetch
        try:
            CQ.fetch_all_vectors(FakeClient(""), "forensic_person", None, 128, ["a"], ["solider"], "p", "person", log=lambda *_: None, config_path=None)
        finally:
            CL.fetch_vectors_for = orig
        self.assertEqual(seen["solider"], "p_person_solider.npz")                  # 지문·빌드 id 모두 없으면 옛 이름


if __name__ == "__main__":
    unittest.main()
