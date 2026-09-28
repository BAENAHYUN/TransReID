"""cluster_dbscan_qdrant / compare_cluster_results 순수 함수 테스트 (네트워크·Qdrant 없음).

핵심: assign_clusters 가 dbscan_person_v6_cl_fixed.py 의 Step 3/4 루프(원문 복사본)와
임의 그래프에서 완전히 같은 배정을 내는지 확인한다.
"""
import io
import json
import math
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clustering import cluster_dbscan_qdrant as dbscan  # noqa: E402
from clustering import compare_cluster_results as compare  # noqa: E402


def reference_assign(all_ids, neighbors_map, MIN_FACES):
    """dbscan_person_v6_cl_fixed.py Step 3 + Step 4 원문 그대로."""
    cluster_map, deferred = {}, []
    next_cid = 0
    for pid in all_ids:
        if pid in cluster_map:
            continue
        neighbors = neighbors_map[pid]
        is_core = len(neighbors) >= MIN_FACES
        assigned = next((cluster_map[n] for n in neighbors if n in cluster_map), None)
        if assigned is not None:
            cluster_map[pid] = assigned
        elif is_core:
            cluster_map[pid] = next_cid
            next_cid += 1
        else:
            deferred.append(pid)
    for pid in deferred:
        if pid in cluster_map:
            continue
        assigned = next((cluster_map[n] for n in neighbors_map[pid] if n in cluster_map), None)
        cluster_map[pid] = assigned if assigned is not None else -1
    return cluster_map


def reference_combined(s, i, o, W=(0.1, 0.3, 0.6)):
    """원본 combined_vec (sklearn normalize) 그대로."""
    from sklearn.preprocessing import normalize
    s, i, o = normalize([s])[0], normalize([i])[0], normalize([o])[0]
    return normalize([np.concatenate([W[0] * s, W[1] * i, W[2] * o])])[0]


class AssignTests(unittest.TestCase):
    def test_matches_reference_on_random_graphs(self):
        rng = random.Random(1)
        for _ in range(40):
            n = rng.randint(1, 60)
            ids = [f"p{j}" for j in range(n)]
            neighbors = {}
            for pid in ids:
                others = [x for x in ids if x != pid]
                k = rng.randint(0, min(6, len(others)))
                neighbors[pid] = rng.sample(others, k)
            for min_faces in (1, 2, 3):
                got, _ = dbscan.assign_clusters(ids, neighbors, min_faces)
                self.assertEqual(got, reference_assign(ids, neighbors, min_faces))

    def test_stats(self):
        ids = ["a", "b", "c", "d", "e"]
        nb = {"a": ["b", "c", "d"], "b": ["a"], "c": ["a"], "d": ["a"], "e": []}
        cm, st = dbscan.assign_clusters(ids, nb, 3)
        self.assertEqual(cm, {"a": 0, "b": 0, "c": 0, "d": 0, "e": -1})
        self.assertEqual((st["core_points"], st["deferred"], st["deferred_joined"], st["raw_clusters"]), (1, 1, 0, 1))

    def test_normalize_assignments(self):
        ids = ["a", "b", "c", "e"]
        cm = {"a": 0, "b": 0, "c": 0, "e": -1}
        deg = {"a": 3, "b": 1, "c": 1, "e": 0}
        assignments, st = dbscan.normalize_assignments("person", "m", ids, cm, deg, 3)
        self.assertTrue(assignments["e"]["noise"])
        self.assertIsNone(assignments["e"]["cluster_id"])
        self.assertEqual(assignments["a"]["cluster_size"], 3)
        self.assertTrue(assignments["a"]["is_core"])
        self.assertFalse(assignments["b"]["is_core"])
        self.assertTrue(assignments["a"]["cluster_id"].startswith("dbscan:person:"))
        self.assertEqual(assignments["a"]["cluster_id"], assignments["b"]["cluster_id"])
        self.assertEqual(st, dict(kept_clusters=1, singleton_clusters=0, largest_community=3,
                                  clustered_points=3, noise_points=1))
        self.assertEqual(dbscan.stable_cluster_id("person", "m", ["b", "a"]),
                         dbscan.stable_cluster_id("person", "m", ["a", "b"]))


class VectorTests(unittest.TestCase):
    def test_combined_matches_reference(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("sklearn 없음")
        rng = np.random.default_rng(0)
        for _ in range(5):
            s, i, o = rng.normal(size=8), rng.normal(size=5), rng.normal(size=6)
            got = dbscan.combined_vec([s, i, o], (0.1, 0.3, 0.6))
            np.testing.assert_allclose(got, reference_combined(s, i, o), atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.norm(got)), 1.0, places=12)

    def test_l2_zero_vector(self):
        np.testing.assert_array_equal(dbscan.l2([0, 0, 0]), np.zeros(3))

    def test_l2_tiny_norm_matches_sklearn(self):
        # sklearn: norm < 10*eps 이면 나누지 않는다 → [1e-20, 0] 은 그대로
        tiny = np.array([1e-20, 0.0])
        np.testing.assert_array_equal(dbscan.l2(tiny), tiny)
        try:
            from sklearn.preprocessing import normalize
        except ImportError:
            self.skipTest("sklearn 없음")
        np.testing.assert_allclose(dbscan.l2(tiny), normalize([tiny])[0])
        np.testing.assert_allclose(dbscan.l2([3.0, 4.0]), normalize([[3.0, 4.0]])[0])

    def test_find_neighbors_with_fake_client(self):
        from types import SimpleNamespace

        def u(deg):
            return np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))], dtype=np.float32)
        ids = ["p", "q", "r", "s"]
        matrix = np.vstack([u(0), u(10), u(40), u(5)])
        index = {pid: k for k, pid in enumerate(ids)}
        hits_by_pos = {"p": ["p", "q", "r", "s", "zz"], "q": ["q", "p", "r"], "r": ["r"], "s": ["s", "p", "q"]}

        class FakeClient:
            def __init__(self):
                self.requests = []

            def query_batch_points(self, collection_name, requests):
                self.requests.append((collection_name, requests))
                batch_pids = ids[len(self.requests[0][1]) * (len(self.requests) - 1):][:len(requests)]
                return [SimpleNamespace(points=[SimpleNamespace(id=h, score=0.9) for h in hits_by_pos[pid]])
                        for pid in batch_pids]

        eps = 1 - math.cos(math.radians(12))
        cfg = dbscan.Config(target="person", collection="c", vector="solider", combined_vectors=["siglip2", "irra", "solider"],
                            weights=[0.1, 0.3, 0.6], sources=["prw_image"], knn=25, score_threshold=1 - eps, eps=eps,
                            min_faces=3, query_batch_size=3, scroll_batch_size=10, max_points=None, payload_key="k")
        client = FakeClient()
        neighbors, stats = dbscan.find_neighbors(client, cfg, ids, index, matrix, matrix, log=lambda *a: None)
        self.assertEqual(neighbors, {"p": ["q", "s"], "q": ["p"], "r": [], "s": ["p", "q"]})
        self.assertEqual(stats["neighbor_pairs"], 5)
        self.assertEqual(stats["knn_hits"], 12)
        self.assertEqual(len(client.requests), 2)  # 4 개를 배치 3 으로 → 2 회
        req = client.requests[0][1][0]
        self.assertEqual((req.using, req.limit, req.score_threshold), ("solider", 25, 1 - eps))
        self.assertTrue(req.params.exact)
        self.assertTrue(req.params.quantization.ignore)
        self.assertEqual(req.filter.must[0].key, "source")
        self.assertEqual(req.filter.must[0].match.value, "prw_image")
        np.testing.assert_allclose(req.query, matrix[0].tolist(), atol=1e-6)


class DeferredTests(unittest.TestCase):
    def test_deferred_join_and_noise_propagation_stats(self):
        ids = ["x", "y", "z", "w"]
        nb = {"x": ["y"], "y": ["x", "z", "w"], "z": ["y"], "w": []}
        cm, st = dbscan.assign_clusters(ids, nb, 2)
        self.assertEqual(cm, {"x": 0, "y": 0, "z": 0, "w": -1})
        self.assertEqual((st["deferred"], st["deferred_joined"], st["deferred_noise_propagated"]), (2, 1, 0))
        ids2 = ["u", "v"]
        nb2 = {"u": [], "v": ["u"]}
        cm2, st2 = dbscan.assign_clusters(ids2, nb2, 2)
        self.assertEqual(cm2, {"u": -1, "v": -1})  # 원본: 앞선 보류점의 -1 이 전파된다
        self.assertEqual((st2["deferred_joined"], st2["deferred_noise_propagated"]), (0, 1))
        self.assertEqual(cm2, reference_assign(ids2, nb2, 2))

    def test_single_core_cluster_survives(self):
        # core 인 a 가 클러스터 0 을 만들지만, b/c/d 는 자기 이웃 목록에 a 가 없어 합류하지 않는다 (비대칭 kNN)
        ids = ["a", "b", "c", "d", "e"]
        nb = {"a": ["b", "c", "d"], "b": ["e"], "c": ["e"], "d": ["e"], "e": ["b", "c", "d"]}
        cm, _ = dbscan.assign_clusters(ids, nb, 3)
        self.assertEqual(cm["a"], 0)
        self.assertEqual({cm["b"], cm["c"], cm["d"], cm["e"]}, {1})
        assignments, st = dbscan.normalize_assignments("person", "m", ids, cm, {i: len(nb[i]) for i in ids}, 3)
        self.assertEqual(st["singleton_clusters"], 1)
        self.assertFalse(assignments["a"]["noise"])
        self.assertEqual(cm, reference_assign(ids, nb, 3))

    def test_filter_neighbors(self):
        def u(deg):
            return np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
        ids = ["p", "q", "r", "s"]
        matrix = np.vstack([u(0), u(10), u(40), u(5)]).astype(np.float32)
        index = {pid: k for k, pid in enumerate(ids)}
        eps = 1 - math.cos(math.radians(12))
        got = dbscan.filter_neighbors("p", ["p", "q", "r", "s", "zz"], index, matrix, eps)
        self.assertEqual(got, ["q", "s"])  # 순서 유지, 자기 자신·미지 id·먼 점 제외
        self.assertEqual(dbscan.filter_neighbors("nope", ["p"], index, matrix, eps), [])
        self.assertEqual(dbscan.filter_neighbors("p", [], index, matrix, eps), [])


class CompareTests(unittest.TestCase):
    A = {"1": "x", "2": "x", "3": "x", "4": "y", "5": "y", "6": None, "7": "z", "8": "z"}
    B = {"1": "m", "2": "m", "3": "n", "4": "n", "5": "n", "6": "n", "7": None, "8": None}
    IDS = list(A)

    def test_method_stats(self):
        st = compare.method_stats(self.A, self.IDS)
        self.assertEqual((st["clusters"], st["clustered"], st["noise"], st["largest"], st["singleton_clusters"]),
                         (3, 7, 1, 3, 0))

    def test_pair_counts_bruteforce(self):
        pc = compare.pair_counts(self.A, self.B, self.IDS)
        ids = self.IDS
        pa = pb = both = 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sa = self.A[ids[i]] is not None and self.A[ids[i]] == self.A[ids[j]]
                sb = self.B[ids[i]] is not None and self.B[ids[i]] == self.B[ids[j]]
                pa += sa
                pb += sb
                both += (sa and sb)
        self.assertEqual((pc["pairs_a"], pc["pairs_b"], pc["pairs_both"]), (pa, pb, both))
        self.assertAlmostEqual(pc["b_pair_precision"], both / pb)
        self.assertAlmostEqual(pc["b_pair_recall"], both / pa)

    def test_noise_crosstab(self):
        self.assertEqual(compare.noise_crosstab(self.A, self.B, self.IDS),
                         dict(both_noise=0, a_noise_only=1, b_noise_only=2, both_clustered=5))

    def test_agreement(self):
        same = compare.agreement(self.A, self.A, self.IDS)
        self.assertAlmostEqual(same["ari_noise_as_singletons"], 1.0)
        self.assertAlmostEqual(same["nmi_both_clustered"], 1.0)
        self.assertAlmostEqual(same["pair_jaccard"], 1.0)
        agr = compare.agreement(self.A, self.B, self.IDS)
        self.assertEqual(agr["both_clustered"], 5)
        self.assertTrue(0 <= agr["nmi_noise_as_singletons"] <= 1)
        self.assertLess(agr["ari_noise_as_singletons"], 1.0)

    def test_encode_noise_unique(self):
        codes = compare.encode({"a": None, "b": None, "c": "k"}, ["a", "b", "c"])
        self.assertNotEqual(codes[0], codes[1])
        self.assertEqual(codes[2], 0)

    def test_cluster_mapping_categories(self):
        rows, summary = compare.cluster_mapping(self.A, self.B, self.IDS)
        by = {r["cluster_id"]: r for r in rows}
        self.assertEqual(by["x"]["category"], "mixed")
        self.assertEqual(by["y"]["category"], "merged_in_other")
        self.assertEqual(by["z"]["category"], "all_noise_in_other")
        self.assertEqual(by["x"]["parts"], 2)
        self.assertAlmostEqual(by["x"]["purity"], 2 / 3)
        self.assertAlmostEqual(by["x"]["best_jaccard"], 2 / 3)
        self.assertEqual(summary["categories"]["mixed"]["clusters"], 1)
        self.assertEqual(summary["points"], 7)
        self.assertEqual([r["cluster_id"] for r in rows][0], "x")  # 크기 내림차순

        rows2, _ = compare.cluster_mapping(self.B, self.A, self.IDS)
        by2 = {r["cluster_id"]: r for r in rows2}
        self.assertEqual(by2["m"]["category"], "merged_in_other")
        self.assertEqual(by2["n"]["category"], "mixed")

        A = {"1": "x", "2": "x", "3": "y", "4": "y", "5": "q", "6": "q", "7": "q"}
        B = {"1": "m", "2": "m", "3": "n", "4": "n", "5": "r", "6": "r", "7": None}
        rows3, s3 = compare.cluster_mapping(A, B, list(A))
        self.assertEqual({r["cluster_id"]: r["category"] for r in rows3},
                         {"x": "identical", "y": "identical", "q": "split_in_other"})
        self.assertEqual(s3["best_jaccard_histogram"]["1.0"], 2)

    def test_pick_examples(self):
        rows_ab, _ = compare.cluster_mapping(self.A, self.B, self.IDS)
        rows_ba, _ = compare.cluster_mapping(self.B, self.A, self.IDS)
        ex = compare.pick_examples(rows_ab, rows_ba, 3, min_size=2)
        self.assertEqual([r["cluster_id"] for r in ex["a_split_by_b"]], ["x"])
        self.assertEqual([r["cluster_id"] for r in ex["b_split_by_a"]], ["n"])
        self.assertEqual([r["cluster_id"] for r in ex["a_cluster_b_noise"]], ["z"])
        self.assertEqual(ex["identical"], [])

    def test_sample_members_quota(self):
        pts = [f"p{i}" for i in range(30)]
        labels_to = {p: ("g1" if i < 20 else "g2" if i < 28 else None) for i, p in enumerate(pts)}
        groups = compare.sample_members(pts, labels_to, 12, random.Random(0))
        self.assertEqual([k for k, _, _ in groups], ["g1", "g2", None])  # 큰 그룹부터, 노이즈 마지막
        self.assertEqual([n for _, _, n in groups], [20, 8, 2])
        self.assertLessEqual(sum(len(c) for _, c, _ in groups), 12)
        self.assertTrue(all(len(c) >= 1 for _, c, _ in groups))

    def test_example_groups_fetch_set_equals_render_set(self):
        # 무작위 추출이 필요한 큰 클러스터: 조회 집합(example_ids) 이 곧 렌더링 표본이어야 한다
        A = {f"p{i}": "big" for i in range(80)}
        B = {f"p{i}": ("m" if i < 50 else "n" if i < 75 else None) for i in range(80)}
        ids = list(A)
        rows_ab, _ = compare.cluster_mapping(A, B, ids)
        rows_ba, _ = compare.cluster_mapping(B, A, ids)
        examples = compare.pick_examples(rows_ab, rows_ba, 3)
        members_a = {"big": ids}
        members_b = {"m": ids[:50], "n": ids[50:75]}
        groups = compare.build_example_groups(examples, members_a, members_b, A, B, 10, random.Random(1))
        need = compare.example_ids(groups)
        rendered = {pid for entries in groups.values() for _, sampled in entries for _, chosen, _ in sampled for pid in chosen}
        self.assertEqual(need, rendered)
        self.assertTrue(need)
        # payload 가 없으면 고른 point 마다 placeholder 하나 (같은 point 가 여러 사례에 나오면 그만큼 반복)
        html = compare.examples_html(groups, {"a": "A", "b": "B"}, {}, Path("."), False, 100, 70, Path("x.html"))
        expected = sum(len(chosen) for entries in groups.values() for _, sampled in entries for _, chosen, _ in sampled)
        self.assertEqual(html.count("class='ph'"), expected)
        self.assertIn("B: m", html)

    def test_median_even(self):
        st = compare.method_stats({"1": "x", "2": "x", "3": "y", "4": "y", "5": "y", "6": "y", "7": "y", "8": "y",
                                   "9": "y", "10": "y", "11": "y", "12": "y"}, [str(i) for i in range(1, 13)])
        self.assertEqual(st["median_cluster_size"], 6.0)  # 크기 [2, 10] 의 중앙값

    def test_load_labels_rejects_bad_point_id_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x_assignments.jsonl"
            p.write_text('{"point_id": [], "cluster_id": "a"}\n{"point_id": "ok", "cluster_id": "a"}\n'
                         '{"point_id": true, "cluster_id": "a"}\n', encoding="utf-8")
            errors = dict(count=0, first_line=None)
            labels, dup = compare.load_labels(p, errors)
            self.assertEqual(labels, {"ok": "a"})
            self.assertEqual(errors["count"], 2)

    def test_html_name_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.jsonl"
            a.write_text('{"point_id": "1", "cluster_id": "x"}\n', encoding="utf-8")
            for bad in ("sub/report.html", "compare_report.json", "report.txt"):
                with self.assertRaises(SystemExit):
                    with redirect_stdout(io.StringIO()):
                        compare.parse_args(["--a", str(a), "--b", str(a), "--no-images", "--html-name", bad])

    def test_apply_min_cluster_size(self):
        labels = {"1": "x", "2": "x", "3": "s", "4": None, "5": "t", "6": "t", "7": "t"}
        self.assertEqual(compare.apply_min_cluster_size(labels, 1), 0)
        self.assertEqual(labels["3"], "s")
        self.assertEqual(compare.apply_min_cluster_size(labels, 2), 1)  # 단독 클러스터 s 만 노이즈로
        self.assertIsNone(labels["3"])
        self.assertEqual(labels["1"], "x")
        self.assertEqual(compare.apply_min_cluster_size(labels, 3), 2)  # x(2) 도 노이즈로
        self.assertEqual([labels[k] for k in ("5", "6", "7")], ["t", "t", "t"])

    def test_end_to_end_no_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            def write(name, labels):
                p = tmp / name
                with p.open("w", encoding="utf-8") as f:
                    for pid, cid in labels.items():
                        f.write(json.dumps({"point_id": pid, "cluster_id": cid, "cluster_size": 1,
                                            "noise": cid is None}) + "\n")
                return p
            a = write("person_leiden_assignments.jsonl", self.A)
            b = write("person_dbscan_assignments.jsonl", {**self.B, "9": "extra"})
            (tmp / "person_leiden_report.json").write_text(
                json.dumps({"config": {"vector": "solider", "knn": 30}, "stats": {}, "dry_run": True}), encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
                summary = compare.main(["--a", str(a), "--b", str(b), "--no-images", "--output-dir", str(tmp / "out"),
                                        "--examples", "2", "--min-example-size", "2"])
            text = out.getvalue()
            self.assertIn("RESULT_SUMMARY:", text)
            self.assertIn("RESULT_HTML:", text)
            self.assertEqual(summary["config"]["labels"], {"a": "Leiden", "b": "DBSCAN"})
            self.assertEqual(summary["metrics"]["common_points"], 8)
            self.assertEqual(summary["metrics"]["only_b"], 1)
            codes = {w["code"] for w in summary["warnings"]}
            self.assertIn("POINT_SET_MISMATCH", codes)
            self.assertIn("NO_REPORT", codes)
            html = (tmp / "out" / "compare_clusters.html").read_text(encoding="utf-8")
            self.assertIn("Leiden", html)
            self.assertIn("클러스터 대응", html)
            self.assertTrue((tmp / "out" / "mapping_a_to_b.jsonl").exists())
            side = json.loads((tmp / "out" / "compare_report.json").read_text(encoding="utf-8"))
            self.assertEqual(side["producer"], "compare_cluster_results")
            self.assertIn("agreement", side["metrics"])
            self.assertEqual(side["examples"]["a_split_by_b"][0]["cluster_id"], "x")


if __name__ == "__main__":
    unittest.main()
