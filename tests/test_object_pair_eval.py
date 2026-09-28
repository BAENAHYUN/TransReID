"""오프라인 테스트: eval/object_pair_eval.py — 쌍 제안, identity 그룹·모순, 검색 mAP, 쌍 AUC/F1, 클러스터 일치, 캐시 왕복·정합성, 검토율·manifest, eval end-to-end(Qdrant 없음)."""
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

from bench import criteria as C  # noqa: E402
from bench import ledger as L  # noqa: E402
from eval import object_pair_eval as O  # noqa: E402


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


KEYS = ["v1/object_1", "v1/object_2", "v2/object_3", "v2/object_4", "v3/object_5", "v3/object_6"]
MAT = np.stack([unit([1, 0, 0]), unit([0.99, 0.1, 0]), unit([0, 1, 0]), unit([0, 0.98, 0.2]), unit([0, 0, 1]), unit([0.1, 0, 0.99])])
META = {k: {"track_key": k, "quality": 0.5, "crop_path": None, "label": "bag", "video_stem": k.split("/")[0], "timestamp_sec": i,
            "frame_idx": i, "point_ids": [f"pt{i}"], "n_points": 1} for i, k in enumerate(KEYS)}
CLUSTERS = {"v1/object_1": "c1", "v1/object_2": "c1", "v2/object_3": "c2", "v2/object_4": "c2", "v3/object_5": None, "v3/object_6": None}


def pairs_fixture():
    return [{"pair_id": "p1", "a": "v1/object_1", "b": "v1/object_2", "sim": 0.995, "source": "cluster", "verdict": "same"},
            {"pair_id": "p2", "a": "v2/object_3", "b": "v2/object_4", "sim": 0.98, "source": "cluster", "verdict": "same"},
            {"pair_id": "p3", "a": "v1/object_1", "b": "v2/object_3", "sim": 0.0, "source": "random", "verdict": "different"},
            {"pair_id": "p4", "a": "v3/object_5", "b": "v3/object_6", "sim": 0.99, "source": "knn", "verdict": "different"},
            {"pair_id": "p5", "a": "v1/object_2", "b": "v2/object_4", "sim": 0.1, "source": "random", "verdict": "unsure"}]


class ProposalTests(unittest.TestCase):
    def test_track_clusters_from_assignments(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.jsonl"
            with p.open("w", encoding="utf-8") as f:
                f.write(json.dumps({"point_id": "pt0", "cluster_id": 5, "noise": False}) + "\n")
                f.write(json.dumps({"point_id": "pt1", "cluster_id": 5, "noise": False}) + "\n")
                f.write(json.dumps({"point_id": "pt2", "cluster_id": None, "noise": True}) + "\n")
            cl = O.track_clusters(META, p)
            self.assertEqual((cl["v1/object_1"], cl["v1/object_2"], cl["v2/object_3"]), ("5", "5", None))
        self.assertTrue(all(v is None for v in O.track_clusters(META, None).values()))

    def test_propose_pairs_sources(self):
        pairs = O.propose_pairs(KEYS, MAT, CLUSTERS, n_cluster=10, n_knn=2, n_random=1, seed=1, per_cluster=2, knn_min_sim=0.5, random_max_sim=0.5)
        by = {}
        for p in pairs:
            by.setdefault(p["source"], []).append(p)
        self.assertEqual(len(by["cluster"]), 2)
        self.assertEqual([tuple(sorted((p["a"], p["b"]))) for p in by["knn"]], [("v3/object_5", "v3/object_6")])
        self.assertEqual(len(by["random"]), 1)
        self.assertLessEqual(by["random"][0]["sim"], 0.5)
        self.assertEqual(len({p["pair_id"] for p in pairs}), len(pairs))
        for p in pairs:
            self.assertLess(p["a"], p["b"])
            self.assertIsInstance(p["sim"], float)


class EvalTests(unittest.TestCase):
    def test_groups_contradictions_and_metrics(self):
        pairs = pairs_fixture()
        groups, contra = O.identity_groups(pairs)
        self.assertEqual(sorted(sorted(m) for m in groups.values()), [["v1/object_1", "v1/object_2"], ["v2/object_3", "v2/object_4"]])
        self.assertEqual(contra, 0)
        pairs2 = pairs + [{"pair_id": "p6", "a": "v1/object_2", "b": "v2/object_3", "sim": 0.5, "source": "knn", "verdict": "same"},
                          {"pair_id": "p7", "a": "v1/object_1", "b": "v2/object_4", "sim": 0.5, "source": "knn", "verdict": "different"}]
        groups2, contra2 = O.identity_groups(pairs2)
        self.assertEqual((len(groups2), contra2), (1, 2))                                  # p3, p7 이 같은 그룹 안의 '다름' → 모순 2
        ret = O.retrieval_metrics(KEYS, MAT, groups)
        self.assertEqual((ret["map"], ret["rank1"], ret["queries"], ret["gallery"]), (100.0, 100.0, 4, 6))
        self.assertEqual(O.retrieval_metrics(KEYS, MAT, groups, ["v1/object_1", "v1/object_2", "v2/object_3"])["queries"], 3)
        pm = O.pair_metrics(pairs, 0.97)
        self.assertEqual((pm["pairs_same"], pm["pairs_diff"], pm["pairs_unsure"]), (2, 2, 1))
        self.assertEqual(pm["pair_auc"], 0.75)
        self.assertEqual(pm["pair_acc_at_threshold"], 0.75)
        ca = O.cluster_agreement(pairs, CLUSTERS)
        self.assertEqual((ca["cluster_pair_precision"], ca["cluster_pair_recall"]), (1.0, 1.0))
        self.assertAlmostEqual(O._ap([True, False, True], 2), (1 + 2 / 3) / 2)
        self.assertIsNone(O.pair_metrics([p for p in pairs if p["verdict"] == "same"], 0.9)["pair_auc"])

    def test_labeled_pairs_requires_review(self):
        raw = [{k: v for k, v in p.items() if k != "verdict"} for p in pairs_fixture()]
        labels = {"p1": {"verdict": "same"}, "p2": {"verdict": "same", "reviewed": False}, "p3": {"note": "x", "reviewed": True}}
        lp = O.labeled_pairs(raw, labels)
        self.assertEqual([p["verdict"] for p in lp], ["same", None, None, None, None])       # p2 미검토, p3 판정 없음

    def test_cache_roundtrip_validation_and_eval_offline(self):
        with tempfile.TemporaryDirectory() as td:
            gt = Path(td) / "gt"
            O.save_tracks(gt, "dinov2", KEYS, MAT, META, "forensic_object")
            keys, mat, meta, head = O.load_tracks(gt, "dinov2", "forensic_object")
            self.assertEqual((keys, head["n_tracks"], head["collection"]), (KEYS, 6, "forensic_object"))
            self.assertTrue(np.allclose(mat, MAT))
            with self.assertRaises(SystemExit):
                O.load_tracks(gt, "dinov2", "other_collection")                              # 컬렉션 불일치
            with self.assertRaises(FileNotFoundError):
                O.load_tracks(gt, "siglip2")
            props = [{k: v for k, v in p.items() if k != "verdict"} for p in pairs_fixture()]
            props.append({"pair_id": "p9", "a": "v9/object_9", "b": "v1/object_1", "sim": 0.3, "source": "knn"})   # 캐시에 없는 트랙 → 제외
            manifest = "abc123def456"
            (gt / "proposals.json").write_text(json.dumps({"manifest": manifest, "vector": "dinov2", "pairs": props}), encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                out = O.main(["eval", "--gt-dir", str(gt), "--vector", "dinov2", "--output-dir", str(Path(td) / "res"), "--ledger", str(Path(td) / "l.jsonl")])
            self.assertTrue(out["pseudo_gt"])
            self.assertEqual(out["name"], "dinov2_track__pseudo")
            self.assertEqual(out["metrics"]["map"], 100.0)
            self.assertEqual(out["metrics"]["pairs_missing"], 1)
            self.assertFalse((Path(td) / "l.jsonl").exists())
            labels = {"kind": "object_pair_labels", "meta": {"manifest": manifest},
                      "labels": {p["pair_id"]: {"verdict": p["verdict"], "reviewed": True} for p in pairs_fixture()}}
            labels["labels"]["p9"] = {"verdict": "different", "reviewed": True}
            (gt / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
            with contextlib.redirect_stdout(buf):
                out = O.main(["eval", "--gt-dir", str(gt), "--vector", "dinov2", "--output-dir", str(Path(td) / "res"), "--name", "t",
                              "--ledger", str(Path(td) / "l.jsonl")])
            self.assertFalse(out["pseudo_gt"])
            self.assertEqual(out["name"], "t")
            self.assertEqual((out["metrics"]["pairs_unsure"], out["metrics"]["identities"], out["metrics"]["coverage"]), (1, 2, 1.0))
            self.assertAlmostEqual(next(p["sim"] for p in out["pairs"] if p["pair_id"] == "p1"), float(MAT[0] @ MAT[1]), places=4)   # 현재 벡터로 재계산
            entries = L.read_entries(Path(td) / "l.jsonl")
            self.assertEqual((entries[0]["stage"], entries[0]["component"]["vector"], entries[0]["metrics"]["map"]), ("object", "dinov2", 100.0))
            self.assertTrue(any(i["role"] == "track_cache" for i in entries[0]["inputs"]))
            self.assertEqual(C.evaluate(entries[0])["status"], "incomplete")               # 객체 mAP 기준값(기존) 미정
            labels["meta"]["manifest"] = "wrong"
            (gt / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stdout(buf):
                    O.main(["eval", "--gt-dir", str(gt), "--vector", "dinov2", "--output-dir", str(Path(td) / "res"), "--no-ledger"])


if __name__ == "__main__":
    unittest.main()
