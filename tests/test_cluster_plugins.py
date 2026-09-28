"""오프라인 테스트: 클러스터링 플러그인 계약 (clustering/base.py, methods/leiden.py, methods/dbscan_v6.py, cluster_qdrant driver).
Qdrant 없이 합성 벡터(잘 분리된 두 덩어리)로 두 플러그인이 같은 계약으로 동작하는지 확인한다."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clustering import base  # noqa: E402
from clustering import cluster_qdrant as driver  # noqa: E402
from clustering.methods.dbscan_v6 import DBSCANv6Clusterer, combine_rows  # noqa: E402
from clustering.methods.leiden import LeidenClusterer  # noqa: E402


def two_blobs(n_per=40, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(2, dim))
    rows, truth = [], []
    for c in range(2):
        pts = centers[c] + 0.15 * rng.normal(size=(n_per, dim))
        rows.append(pts)
        truth += [c] * n_per
    m = np.concatenate(rows).astype(np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    ids = [f"p{i}" for i in range(m.shape[0])]
    return ids, m, truth


def purity(labels, truth):
    from collections import Counter, defaultdict
    by = defaultdict(Counter)
    for lab, t in zip(labels, truth):
        if lab is not None:
            by[lab][t] += 1
    n = sum(sum(c.values()) for c in by.values())
    return sum(max(c.values()) for c in by.values()) / n if n else 0.0


class SpecTests(unittest.TestCase):
    def test_builtin_and_overrides(self):
        spec = base.resolve_clusterer_spec(None)
        self.assertEqual(spec["class"], "LeidenClusterer")
        spec = base.resolve_clusterer_spec("dbscan_v6", params=["eps=0.2", "combined_vectors=[\"a\",\"b\"]"])
        self.assertEqual((spec["class"], spec["params"]), ("DBSCANv6Clusterer", {"eps": 0.2, "combined_vectors": ["a", "b"]}))
        self.assertEqual(base.BUILTIN_METHODS["dbscan_v6"]["params"], {})  # 원본은 그대로
        spec = base.resolve_clusterer_spec("leiden", module="my.mod", cls="MyC", params=["x=1"])
        self.assertEqual((spec["module"], spec["class"], spec["params"]), ("my.mod", "MyC", {"x": 1}))
        with self.assertRaises(ValueError):
            base.resolve_clusterer_spec("unknown")
        with self.assertRaises(ValueError):
            base.parse_param("noequals")

    def test_yaml_block(self):
        with tempfile.TemporaryDirectory() as td:
            y = Path(td) / "c.yaml"
            y.write_text("clusterer:\n  module: clustering.methods.leiden\n  class: LeidenClusterer\n  params:\n    knn: 5\n", encoding="utf-8")
            spec = base.resolve_clusterer_spec(None, config_path=str(y), params=["threshold=0.5"])
            self.assertEqual(spec["params"], {"knn": 5, "threshold": 0.5})
            bad = Path(td) / "bad.yaml"
            bad.write_text("retrievers: {}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                base.resolve_clusterer_spec(None, config_path=str(bad))

    def test_load_clusterer_contract(self):
        c = base.load_clusterer(base.resolve_clusterer_spec("leiden", params=["knn=5"]))
        self.assertIsInstance(c, LeidenClusterer)
        self.assertEqual(c.knn, 5)
        with self.assertRaises(TypeError):
            base.load_clusterer({"module": "pathlib", "class": "Path", "params": {}})

    def test_normalize_labels_and_stable_ids(self):
        ids = ["a", "b", "c", "d", "e"]
        labels = [0, 0, 1, None, 2]
        assignments, stats = base.normalize_labels("m", "person", ids, labels, min_cluster_size=2)
        self.assertEqual(assignments["a"]["cluster_id"], assignments["b"]["cluster_id"])
        self.assertTrue(assignments["a"]["cluster_id"].startswith("m:person:"))
        self.assertIsNone(assignments["c"]["cluster_id"])   # 크기 1 < 2 → 노이즈
        self.assertTrue(assignments["d"]["noise"])
        self.assertEqual((stats["kept_clusters"], stats["clustered_points"], stats["noise_points"], stats["largest_community"]), (1, 2, 3, 2))
        self.assertEqual(base.stable_cluster_id("m", "person", ["b", "a"]), base.stable_cluster_id("m", "person", ["a", "b"]))
        with self.assertRaises(ValueError):
            base.normalize_labels("m", "person", ids, [0], 1)


class PluginTests(unittest.TestCase):
    def setUp(self):
        self.ids, self.m, self.truth = two_blobs()
        self.quiet = lambda *a, **k: None

    def test_leiden_finds_two_blobs(self):
        # k = 덩어리 크기 − 1 → 각 덩어리가 클리크가 되어 modularity 가 더 쪼개지 않는다 (k 가 작으면 성긴 kNN 그래프를 세분한다)
        c = LeidenClusterer(knn=39, threshold=0.5, mutual_knn=True, knn_device="cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            res = c.cluster(self.ids, self.m, {"solider": self.m}, log=self.quiet)
        self.assertEqual(len(res.labels), len(self.ids))
        self.assertEqual(len(set(res.labels)), 2)
        self.assertEqual(purity(res.labels, self.truth), 1.0)
        self.assertIn("edges", res.stats)
        self.assertEqual(c.params()["score_threshold"], 0.5)

    def test_dbscan_v6_finds_two_blobs_with_combined_vectors(self):
        # greedy 단일 패스 배정은 후보 k 가 덩어리보다 작으면 같은 덩어리도 여러 군집으로 쪼갠다 (알고리즘 특성, ablation 에서 확인).
        # 계약 검증이 목적이므로 k = 덩어리 크기 − 1 로 두어 이웃이 전부 보이게 한다.
        c = DBSCANv6Clusterer(knn=39, score_threshold=0.5, eps=0.5, min_faces=3, combined_vectors=("a", "b"), weights=(0.5, 0.5),
                              knn_device="cpu")
        self.assertEqual(c.required_vectors, ("a", "b"))
        with contextlib.redirect_stdout(io.StringIO()):
            res = c.cluster(self.ids, self.m, {"solider": self.m, "a": self.m, "b": self.m}, log=self.quiet)
        clustered = [l for l in res.labels if l is not None]
        self.assertEqual(len(set(clustered)), 2)
        self.assertEqual(purity(res.labels, self.truth), 1.0)
        self.assertEqual(res.stats["combined_dim"], 2 * self.m.shape[1])
        with self.assertRaises(ValueError):
            c.cluster(self.ids, self.m, {"solider": self.m}, log=self.quiet)   # 결합 벡터 누락

    def test_combine_rows_matches_original_rule(self):
        a = np.array([[3.0, 4.0], [0.0, 0.0]])
        b = np.array([[0.0, 2.0], [1.0, 0.0]])
        out = combine_rows([a, b], [0.25, 0.75])
        np.testing.assert_allclose(np.linalg.norm(out[0]), 1.0, atol=1e-6)
        expect = np.concatenate([0.25 * np.array([0.6, 0.8]), 0.75 * np.array([0.0, 1.0])])
        np.testing.assert_allclose(out[0], expect / np.linalg.norm(expect), atol=1e-6)
        np.testing.assert_allclose(out[1], np.array([0, 0, 0.75, 0]) / 0.75, atol=1e-6)

    def test_missing_vector_rows_become_noise(self):
        m = self.m.copy()
        m[0] = 0.0
        c = LeidenClusterer(knn=39, threshold=0.5, knn_device="cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            res = c.cluster(self.ids, m, {}, log=self.quiet)
        assignments, stats = base.normalize_labels("leiden", "person", self.ids, res.labels, 2)
        self.assertTrue(assignments["p0"]["noise"])
        self.assertEqual(res.stats.get("missing_vectors"), 1)


class DriverTests(unittest.TestCase):
    def test_run_from_vectors_writes_leiden_convention_outputs(self):
        ids, m, truth = two_blobs()
        spec = base.resolve_clusterer_spec("leiden", params=["knn=39", "threshold=0.5", "knn_device=cpu"])
        c = base.load_clusterer(spec)
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            report = driver.run_from_vectors(c, spec, "person", ids, "solider", {"solider": m}, 2, Path(td),
                                             config_extra=dict(sources=["x"]), fetch_stats={"fetch_solider": {"fetch_sec": 0.0}},
                                             log=lambda *a, **k: None)
            out = Path(td) / "person"
            rows = [json.loads(l) for l in (out / "person_leiden_assignments.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), len(ids))
            self.assertEqual({r["point_id"] for r in rows}, set(ids))
            self.assertEqual(len({r["cluster_id"] for r in rows if not r["noise"]}), 2)
            saved = json.loads((out / "person_leiden_report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["config"]["plugin"]["class"], "LeidenClusterer")
            self.assertEqual((saved["config"]["vector"], saved["config"]["knn"], saved["config"]["sources"]), ("solider", 39, ["x"]))
            self.assertTrue(saved["dry_run"])
            self.assertEqual(saved["stats"]["kept_clusters"], 2)
            self.assertEqual(report["stats"]["noise_points"], 0)
            with self.assertRaises(ValueError):
                driver.run_from_vectors(c, spec, "person", ids, "solider", {"solider": m}, 2, Path(td), write_payload=True)

    def test_cli_argument_errors(self):
        for argv in (["--method", "nope"], ["--min-cluster-size", "0"], ["--method-config", "missing.yaml"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as e:
                driver.main(argv)
            self.assertEqual(e.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
