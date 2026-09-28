"""오프라인 테스트: bench/splits.py — 결정적 인물 분할, 파일 형식, --pid-split 인자 파싱."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import splits as S  # noqa: E402


class SplitTests(unittest.TestCase):
    def test_split_ids_deterministic_and_disjoint(self):
        ids = list(range(100))
        a = S.split_ids(ids, {"tune": 0.3, "holdout": 0.7}, seed=42)
        b = S.split_ids(ids, {"tune": 0.3, "holdout": 0.7}, seed=42)
        c = S.split_ids(ids, {"tune": 0.3, "holdout": 0.7}, seed=1)
        self.assertEqual(a, b)
        self.assertNotEqual(a["tune"], c["tune"])
        self.assertEqual(len(a["tune"]), 30)
        self.assertEqual(len(a["holdout"]), 70)
        self.assertEqual(set(a["tune"]) | set(a["holdout"]), set(ids))
        self.assertFalse(set(a["tune"]) & set(a["holdout"]))
        self.assertEqual(a["tune"], sorted(a["tune"]))

    def test_split_ids_validation(self):
        with self.assertRaises(ValueError):
            S.split_ids([1, 2], {"a": 0.5, "b": 0.6}, 0)
        with self.assertRaises(ValueError):
            S.split_ids([1, 2], {}, 0)
        with self.assertRaises(ValueError):
            S.split_ids([1, 2], {"a": 1.5, "b": -0.5}, 0)      # 합은 1 이지만 비율 자체가 범위 밖

    def test_load_split_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.json"
            p.write_text(json.dumps({"parts": {"tune": [1, 2], "holdout": [2, 3]}}), encoding="utf-8")
            with self.assertRaises(ValueError):            # 겹침
                S.load_split(p)
            p.write_text(json.dumps({"parts": {"tune": [1, 2], "holdout": [3]}, "digest": "000000000000"}), encoding="utf-8")
            with self.assertRaises(ValueError):            # digest 불일치
                S.load_split(p)
            p.write_text(json.dumps({"parts": {}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                S.load_split(p)

    def test_make_and_load_split_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "query_info.txt").write_text("".join(f"{pid} 1 2 3 4 c1s1_{i:06d}\n" for i, pid in enumerate([5, 5, 6, 7, 8, 9, 9, 10])), encoding="utf-8")
            d = S.make_split(root, {"tune": 0.5, "holdout": 0.5}, seed=3)
            self.assertEqual(d["counts"]["tune"]["pids"] + d["counts"]["holdout"]["pids"], 6)
            self.assertEqual(d["counts"]["tune"]["queries"] + d["counts"]["holdout"]["queries"], 8)
            out = root / "s.json"
            out.write_text(json.dumps(d), encoding="utf-8")
            self.assertEqual(S.load_split(out)["digest"], d["digest"])
            self.assertEqual(S.load_split_arg(f"{out}:tune"), set(d["parts"]["tune"]))
            with self.assertRaises(ValueError):
                S.load_split_arg(f"{out}:nope")
            with self.assertRaises(ValueError):
                S.parse_split_arg(str(out))
            path, part = S.parse_split_arg("C:\\x\\s.json:holdout")     # 드라이브 문자와 구분
            self.assertEqual((str(path), part), ("C:\\x\\s.json", "holdout"))

    def test_cli_make(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "query_info.txt").write_text("".join(f"{pid} 1 2 3 4 f{i}\n" for i, pid in enumerate(range(20))), encoding="utf-8")
            out = root / "split.json"
            self.assertEqual(S.main(["make", "--data-root", str(root), "--seed", "7", "--out", str(out)]), 0)
            d = S.load_split(out)
            self.assertEqual(d["seed"], 7)
            self.assertEqual(sorted(d["parts"]), ["holdout", "tune"])
            self.assertEqual(S.main(["show", str(out)]), 0)


if __name__ == "__main__":
    unittest.main()
