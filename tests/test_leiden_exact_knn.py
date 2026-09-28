"""cluster_leiden_qdrant.py 의 정확 kNN 경로 테스트 (Qdrant·GPU 없음; CPU numpy 경로 + fake 벡터 수집)."""
import io
import itertools
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clustering import cluster_leiden_qdrant as L  # noqa: E402


def unit(deg):
    import math
    return np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))], dtype=np.float32)


class ExactTopkTests(unittest.TestCase):
    def test_matches_bruteforce_and_layout(self):
        rng = np.random.default_rng(0)
        M = rng.normal(size=(40, 8)).astype(np.float32)
        M /= np.linalg.norm(M, axis=1, keepdims=True)
        k, thr = 5, 0.2
        with redirect_stdout(io.StringIO()):
            nbr, sc, st = L.exact_topk(M, k, thr, device="cpu", chunk=7)
        self.assertEqual(nbr.shape, (40, k))
        self.assertEqual(st["knn_device"], "cpu")
        S = M @ M.T
        for i in range(40):
            row = S[i].copy(); row[i] = -np.inf
            order = np.argsort(-row, kind="stable")[:k]
            expected = [(int(j), float(row[j])) for j in order if row[j] >= thr]
            got = [(int(j), float(s)) for j, s in zip(nbr[i], sc[i]) if j >= 0]
            self.assertEqual([j for j, _ in got], [j for j, _ in expected], msg=f"row {i}")
            np.testing.assert_allclose([s for _, s in got], [s for _, s in expected], atol=1e-5)
            # threshold 미만 슬롯은 -1 / -1.0 (prefix 채움)
            self.assertTrue(all(nbr[i][len(got):] == -1))
            self.assertTrue(all(sc[i][len(got):] == -1.0))
            self.assertNotIn(i, nbr[i].tolist())  # 자기 자신 제외

    def test_k_larger_than_n(self):
        M = np.vstack([unit(0), unit(10), unit(20)])
        with redirect_stdout(io.StringIO()):
            nbr, sc, _ = L.exact_topk(M, 30, -1.0, device="cpu")
        self.assertEqual(nbr.shape, (3, 2))
        self.assertEqual(sorted(nbr[0].tolist()), [1, 2])
        with redirect_stdout(io.StringIO()):
            nbr1, _, st = L.exact_topk(M[:1], 30, 0.0, device="cpu")
        self.assertEqual(nbr1.shape, (1, 0))
        self.assertEqual(st["knn_device"], "none")


class ReduceEdgesTests(unittest.TestCase):
    def brute(self, nbr, sc, mutual):
        n, k = nbr.shape
        sets = [{int(j) for j in nbr[i] if j >= 0} for i in range(n)]
        score = {}
        for i in range(n):
            for j, s in zip(nbr[i], sc[i]):
                if j >= 0:
                    a, b = min(i, int(j)), max(i, int(j))
                    score[(a, b)] = max(score.get((a, b), -1.0), float(s))
        edges = []
        for (a, b), w in score.items():
            ok = (b in sets[a] and a in sets[b]) if mutual else True
            if ok:
                edges.append(((a, b), w))
        return sorted(edges)

    def test_matches_bruteforce(self):
        rng = np.random.default_rng(1)
        n, k = 30, 4
        nbr = np.full((n, k), -1, dtype=np.int64); sc = np.full((n, k), -1.0, dtype=np.float32)
        for i in range(n):
            m = rng.integers(0, k + 1)
            cand = rng.choice([j for j in range(n) if j != i], size=m, replace=False)
            vals = np.sort(rng.uniform(0.5, 1.0, size=m))[::-1]
            nbr[i, :m] = cand; sc[i, :m] = vals
        for mutual in (False, True):
            with redirect_stdout(io.StringIO()):
                edges, weights, st = L.reduce_edges(nbr, sc, n, mutual, 0.0, {"knn_mode": "test"})
            got = sorted(zip([tuple(e) for e in edges], [round(w, 6) for w in weights]))
            exp = [(e, round(w, 6)) for e, w in self.brute(nbr, sc, mutual)]
            self.assertEqual(got, exp, msg=f"mutual={mutual}")
            self.assertEqual(st["edges"], len(exp))
            self.assertEqual(st["knn_mode"], "test")
            self.assertEqual(st["mutual_knn"], mutual)

    def test_no_hits(self):
        nbr = np.full((3, 2), -1, dtype=np.int64); sc = np.full((3, 2), -1.0, dtype=np.float32)
        with redirect_stdout(io.StringIO()):
            edges, weights, st = L.reduce_edges(nbr, sc, 3, True, 0.0)
        self.assertEqual((edges, weights, st["edges"], st["directed_hits"]), ([], [], 0, 0))


class RobustnessTests(unittest.TestCase):
    def test_missing_rows_never_become_neighbours(self):
        # 0 벡터(결측) 는 threshold <= 0 이어도 이웃이 되지도, 이웃을 갖지도 않는다
        M = np.vstack([unit(0), unit(5), np.zeros(2, dtype=np.float32), unit(90)])
        valid = np.array([True, True, False, True])
        with redirect_stdout(io.StringIO()):
            nbr, sc, _ = L.exact_topk(M, 3, -1.0, device="cpu", valid_mask=valid)
        self.assertTrue(all(nbr[2] == -1))
        for i in (0, 1, 3):
            self.assertNotIn(2, nbr[i].tolist())
        with redirect_stdout(io.StringIO()):
            nbr2, _, _ = L.exact_topk(M, 3, -1.0, device="cpu")          # 마스크 없으면 0 벡터도 후보 (계약 확인용)
        self.assertIn(2, nbr2[0].tolist())

    def test_chunk_rows_budget(self):
        self.assertEqual(L._chunk_rows(1_000_000, 512_000_000), 128)   # 512MB / (1M*4B)
        self.assertEqual(L._chunk_rows(1_000, 512_000_000), 4096)      # 상한
        self.assertEqual(L._chunk_rows(10_000_000, 512_000_000), 64)   # 하한
        self.assertEqual(L._chunk_rows(5, 1, requested=3), 3)

    def test_vector_cache_validation_and_early_exit(self):
        import json as _json
        import tempfile
        ids = ["a", "b", "c"]
        vecs = {"a": unit(0), "b": unit(10), "c": unit(20)}
        cfg = L.Config(target="person", collection="col", vector="solider", sources=["s"], knn=2, score_threshold=0.5,
                       resolution=1.0, min_cluster_size=2, query_batch_size=8, scroll_batch_size=2, seed=0, max_points=None)

        class FakePoint:
            def __init__(self, pid, vec):
                self.id = pid; self.vector = {"solider": vec.tolist()} if vec is not None else {}

        class FakeClient:
            calls = 0
            def __init__(self, *a, **k):
                pass
            def scroll(self, collection_name, limit, offset, with_vectors, with_payload, scroll_filter):
                FakeClient.calls += 1
                pages = [([FakePoint("zz", unit(50)), FakePoint("b", vecs["b"])], 1),
                         ([FakePoint("a", vecs["a"]), FakePoint("c", None)], 2),
                         ([FakePoint("d", unit(70))], None)]
                return pages[offset or 0]

        fake_models = SimpleNamespace(Filter=lambda **k: ("F", k), FieldCondition=lambda **k: k, MatchAny=lambda **k: k)
        fake_mod = SimpleNamespace(QdrantClient=FakeClient, models=fake_models)
        q = L.Qdrant("http://srv-a", None, 5)
        with tempfile.TemporaryDirectory() as tmp:
            cache = str(Path(tmp) / "v.npz")
            with patch.dict(sys.modules, {"qdrant_client": fake_mod}), redirect_stdout(io.StringIO()):
                M, st = L.fetch_vectors_for(q, cfg, ids, cache)
            self.assertEqual(M.shape, (3, 2))
            np.testing.assert_allclose(M[0], unit(0), atol=1e-6)      # 순서 보존 (a 는 두 번째 페이지)
            self.assertEqual(st["missing_vectors"], 1)                 # c 는 벡터 없음
            self.assertEqual(st["missing_rows"].tolist(), [2])
            self.assertEqual(FakeClient.calls, 3)                       # 마지막 페이지까지 읽음 (c 가 미충족)
            # 캐시 재사용: 같은 서버·대상
            with redirect_stdout(io.StringIO()):
                M2, st2 = L.fetch_vectors_for(q, cfg, ids, cache)
            self.assertIn("vector_cache", st2); self.assertEqual(st2["fetch_sec"], 0.0)
            np.testing.assert_array_equal(M2, M)
            # 다른 서버 URL 이면 캐시 무시 → 다시 수집
            q2 = L.Qdrant("http://srv-b", None, 5)
            FakeClient.calls = 0
            with patch.dict(sys.modules, {"qdrant_client": fake_mod}), redirect_stdout(io.StringIO()):
                L.fetch_vectors_for(q2, cfg, ids, cache)
            self.assertEqual(FakeClient.calls, 3)
        # 요청 ID 를 모두 채우면 조기 종료
        FakeClient.calls = 0
        with patch.dict(sys.modules, {"qdrant_client": fake_mod}), redirect_stdout(io.StringIO()):
            M3, st3 = L.fetch_vectors_for(q, cfg, ["a", "b"], None)
        self.assertEqual(FakeClient.calls, 2)
        self.assertEqual(st3["missing_vectors"], 0)


class MemoryGraphTests(unittest.TestCase):
    def test_build_graph_memory_uses_fetched_vectors(self):
        ids = ["a", "b", "c", "d"]
        M = np.vstack([unit(0), unit(5), unit(90), unit(93)])
        cfg = L.Config(target="person", collection="c", vector="solider", sources=["prw_image"], knn=2, score_threshold=0.9,
                       resolution=1.0, min_cluster_size=2, query_batch_size=8, scroll_batch_size=8, seed=0, max_points=None,
                       mutual_knn=True, knn_mode="exact-memory", knn_device="cpu")
        with patch.object(L, "fetch_vectors_for", return_value=(M, {"fetch_sec": 0.0, "missing_vectors": 0})):
            with redirect_stdout(io.StringIO()):
                edges, weights, st = L.build_graph(None, cfg, ids)
        self.assertEqual(sorted(edges), [(0, 1), (2, 3)])  # 0°-5° / 90°-93° 만 0.9 이상
        self.assertEqual(st["knn_mode"], "exact-memory")
        self.assertEqual(st["knn_device"], "cpu")

    def test_dispatch_and_defaults(self):
        cfg = L.Config(target="person", collection="c", vector="v", sources=["s"], knn=1, score_threshold=0.5, resolution=1.0,
                       min_cluster_size=2, query_batch_size=1, scroll_batch_size=1, seed=0, max_points=None)
        self.assertEqual(cfg.knn_mode, "exact-memory")
        weird = L.Config(target="person", collection="c", vector="v", sources=["s"], knn=1, score_threshold=0.5, resolution=1.0,
                         min_cluster_size=2, query_batch_size=1, scroll_batch_size=1, seed=0, max_points=None, knn_mode="weird")
        with self.assertRaises(ValueError):
            L.build_graph(None, weird, ["x"])
        p = L.build_parser()
        args = p.parse_args(["--target", "person", "--qdrant-url", "http://x", "--person-collection", "c",
                             "--person-vector", "v", "--person-threshold", "0.9", "--secondary-vector", "irra", "--sources", "s"])
        self.assertEqual((args.knn_mode, args.knn_device, args.vector_cache), ("exact-memory", "auto", None))
        args2 = p.parse_args(["--knn-mode", "hnsw"])
        self.assertEqual(args2.knn_mode, "hnsw")

    def test_query_batch_sends_exact_flag(self):
        q = L.Qdrant("http://x", None, 5)
        captured = {}
        def fake_post(path, body):
            captured["body"] = body
            return {"result": [{"points": []} for _ in body["searches"]]}
        q.post = fake_post
        q.query_batch("c", ["p1", "p2"], "v", 3, 0.9, None, ["s"], exact=True)
        self.assertTrue(all(s["params"]["exact"] is True for s in captured["body"]["searches"]))
        self.assertTrue(all(s["params"]["quantization"]["ignore"] for s in captured["body"]["searches"]))
        q.query_batch("c", ["p1"], "v", 3, 0.9, None, ["s"])
        self.assertFalse(captured["body"]["searches"][0]["params"]["exact"])


if __name__ == "__main__":
    unittest.main()
