"""export_cluster_folders 테스트 — 네트워크·Qdrant 없음. 생성 파일은 TemporaryDirectory 안에만."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clustering import export_cluster_folders as exp  # noqa: E402

CID_A = "leiden:person:aaaaaaaa11111111"
CID_B = "leiden:person:bbbbbbbb22222222"
CID_C = "leiden:person:cccccccc33333333"


def row(pid, cid, noise=False):
    return dict(point_id=pid, cluster_id=cid, noise=noise, cluster_size=1)


ROWS = [row("p1", CID_A), row("p2", CID_A), row("p3", CID_A),
        row("p4", CID_B), row("p5", CID_B),
        row("p6", CID_C),
        row("p7", None, True), row("p8", None, True)]
GT = {"p1": 7, "p2": 7, "p3": 12, "p4": 3, "p5": 3, "p7": 7}          # p6 unmatched, p8 not in file
LABELS = {CID_A: dict(name="black jacket · jeans", confidence=0.61, description="d")}


class FakeQdrant:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    def retrieve_points(self, collection, ids, batch_size=256):
        self.calls.append((collection, [str(i) for i in ids], batch_size))
        return {str(i): self.payloads[str(i)] for i in ids if str(i) in self.payloads}


class NameTests(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(exp.slugify("Black Jacket · Jeans"), "black-jacket-jeans")
        self.assertEqual(exp.slugify("검정 재킷/청바지"), "검정-재킷-청바지")
        self.assertEqual(exp.slugify("  --  "), "")
        self.assertEqual(exp.slugify(None), "")
        self.assertEqual(exp.slugify("abcdef-ghijkl", 8), "abcdef-g")
        self.assertEqual(exp.slugify("abcdefg-hijkl", 8), "abcdefg")     # 잘린 끝의 '-' 는 떼어낸다

    def test_short_id_and_pid_token(self):
        self.assertEqual(exp.short_id(CID_A), "aaaaaaaa")
        self.assertEqual(exp.short_id("plain"), "plain")
        self.assertEqual(exp.pid_token(7), "0007")
        self.assertEqual(exp.pid_token("754"), "0754")
        self.assertEqual(exp.pid_token(-2), "m0002")
        self.assertEqual(exp.pid_token("id x"), "id-x")
        self.assertEqual(exp.gt_prefix(None), "pidNA_")
        self.assertEqual(exp.gt_prefix(12), "pid0012_")

    def test_folder_name_variants(self):
        pure = dict(labeled=3, distinct=1, top_pid=7, top_count=3, purity=1.0)
        mixed = dict(labeled=3, distinct=2, top_pid=7, top_count=2, purity=2 / 3)
        none = dict(labeled=0, distinct=0, top_pid=None, top_count=0, purity=None)
        self.assertEqual(exp.folder_name(1, 652, CID_A), "c0001_n652_aaaaaaaa")
        self.assertEqual(exp.folder_name(2, 3, CID_A, pure), "c0002_n3_pid0007_pur100_aaaaaaaa")
        self.assertEqual(exp.folder_name(3, 3, CID_A, mixed, "black jacket · jeans"),
                         "c0003_n3_pid0007_pur67_mix2_black-jacket-jeans_aaaaaaaa")
        self.assertEqual(exp.folder_name(4, 1, CID_A, none), "c0004_n1_gtNA_aaaaaaaa")
        long_label = "very long descriptive label that keeps going and going"
        name = exp.folder_name(5, 10, CID_A, None, long_label, max_len=40)
        self.assertLessEqual(len(name), 40)
        self.assertTrue(name.startswith("c0005_n10_") and name.endswith("_aaaaaaaa"))
        # 예산이 너무 작으면 slug 를 통째로 생략하고 기본 조각은 유지한다
        self.assertEqual(exp.folder_name(6, 10, CID_A, mixed, long_label, max_len=24),
                         "c0006_n10_pid0007_pur67_mix2_aaaaaaaa")

    def test_unique_name(self):
        used = set()
        self.assertEqual(exp.unique_name(used, "a.jpg", "point-1"), "a.jpg")
        self.assertEqual(exp.unique_name(used, "a.jpg", "point-1"), "a_point-1.jpg")
        self.assertEqual(exp.unique_name(used, "a.jpg", "point-1"), "a_point-1-2.jpg")


class InputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="export_inputs_")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_load_labels_jsonl_and_json(self):
        records = [dict(cluster_id=CID_A, cluster_name="black jacket", label_confidence=0.5),
                   dict(cluster_id=CID_B, cluster_name="   "),          # 빈 이름 → 건너뜀
                   dict(cluster_name="no id"), "junk"]
        jsonl = self.root / "labels.jsonl"
        jsonl.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        loaded = exp.load_labels(jsonl)
        self.assertEqual(set(loaded), {CID_A})
        self.assertEqual(loaded[CID_A]["name"], "black jacket")
        self.assertEqual(loaded[CID_A]["confidence"], 0.5)
        as_json = self.root / "labels.json"
        as_json.write_text(json.dumps(records[:1]), encoding="utf-8")
        self.assertEqual(exp.load_labels(as_json), loaded)
        bad = self.root / "bad.json"
        bad.write_text(json.dumps({"cluster_id": CID_A}), encoding="utf-8")
        with self.assertRaises(ValueError):
            exp.load_labels(bad)

    def test_load_gt_matches(self):
        lines = [dict(_meta=True, iou=0.5), dict(point_id="p1", pid=7, status="labeled"),
                 dict(point_id="p2", pid=None, status="unmatched"),
                 dict(point_id="p3", pid=-2, status="unlabeled"),        # PRW 미표기 보행자 → GT 없음
                 dict(point_id="p4", pid=-2),                            # status 없이 음수 pid 도 제외
                 dict(point_id="p5", pid=9),                             # status 없는 옛 형식은 그대로 인정
                 dict(pid=3), "junk-not-dict"]
        path = self.root / "gt.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines) + "\n\n", encoding="utf-8")
        self.assertEqual(exp.load_gt_matches(path), {"p1": 7, "p5": 9})

    def test_gt_summary(self):
        members = [row("p1", CID_A), row("p2", CID_A), row("p3", CID_A), row("px", CID_A)]
        summary = exp.gt_summary(members, GT)
        self.assertEqual((summary["labeled"], summary["distinct"], summary["top_pid"], summary["top_count"]), (3, 2, 7, 2))
        self.assertAlmostEqual(summary["purity"], 2 / 3)
        self.assertIsNone(exp.gt_summary(members, None))
        self.assertEqual(exp.gt_summary([row("px", CID_A)], GT)["top_pid"], None)


class PlanTests(unittest.TestCase):
    def test_plan_clusters_all(self):
        folders, ranks = exp.plan_clusters(ROWS, GT, LABELS, 1, 0, 42, 64)
        self.assertEqual(ranks, {CID_A: 1, CID_B: 2, CID_C: 3})
        names = [f["name"] for f in folders]
        self.assertEqual(names, ["c0001_n3_pid0007_pur67_mix2_black-jacket-jeans_aaaaaaaa",
                                 "c0002_n2_pid0003_pur100_bbbbbbbb", "c0003_n1_gtNA_cccccccc", "_noise_n2"])
        first = folders[0]
        self.assertEqual([(r["point_id"], p) for r, p in first["members"]],
                         [("p1", "pid0007_"), ("p2", "pid0007_"), ("p3", "pid0012_")])
        self.assertEqual(first["label"]["name"], "black jacket · jeans")
        noise = folders[-1]
        self.assertEqual(noise["kind"], "noise")
        self.assertEqual([(r["point_id"], p) for r, p in noise["members"]], [("p7", "pid0007_"), ("p8", "pidNA_")])
        self.assertEqual(noise["gt"]["top_pid"], 7)

    def test_plan_clusters_min_size_and_no_gt(self):
        folders, ranks = exp.plan_clusters(ROWS, None, None, 2, 0, 42, 64)
        self.assertEqual(ranks, {CID_A: 1, CID_B: 2})
        names = [f["name"] for f in folders]
        self.assertEqual(names, ["c0001_n3_aaaaaaaa", "c0002_n2_bbbbbbbb", "_small_n1", "_noise_n2"])
        small = folders[2]
        self.assertEqual([(r["point_id"], p) for r, p in small["members"]], [("p6", "cccccccc_")])
        self.assertIsNone(small["gt"])
        # GT 없이도 파일 접두어 없음
        self.assertEqual(folders[0]["members"][0][1], "")

    def test_plan_clusters_cap(self):
        folders, _ = exp.plan_clusters(ROWS, GT, None, 1, 2, 7, 64)
        first = folders[0]
        self.assertTrue(first["sampled"])
        self.assertEqual(len(first["members"]), 2)
        self.assertEqual(first["size"], 3)
        self.assertFalse(folders[1]["sampled"])

    def test_plan_by_pid(self):
        folders, ranks = exp.plan_by_pid(ROWS, GT, 0, 42)
        self.assertEqual(ranks, {CID_A: 1, CID_B: 2, CID_C: 3})
        names = [f["name"] for f in folders]
        self.assertEqual(names, ["pid0007_n3_k1", "pid0003_n2_k1", "pid0012_n1_k1", "_no_gt_n2"])
        seven = folders[0]
        self.assertEqual([(r["point_id"], p) for r, p in seven["members"]],
                         [("p1", "c0001_"), ("p2", "c0001_"), ("p7", "noise_")])
        self.assertEqual((seven["clusters"], seven["noise_points"], seven["pid"]), (1, 1, 7))
        no_gt = folders[-1]
        self.assertEqual([(r["point_id"], p) for r, p in no_gt["members"]], [("p6", "c0003_"), ("p8", "noise_")])


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        for name in ("requests.Session.post", "requests.Session.get", "socket.socket", "socket.create_connection"):
            guard = patch(name, side_effect=AssertionError("NETWORK FORBIDDEN"))
            guard.start()
            self.addCleanup(guard.stop)
        temp = tempfile.TemporaryDirectory(prefix="export_e2e_")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.crops = self.root / "crops"
        self.crops.mkdir()
        self.payloads = {}
        for r in ROWS:
            pid = r["point_id"]
            if pid == "p8":
                self.payloads[pid] = {"crop_path": "crops/does_not_exist.jpg"}      # 누락 crop
                continue
            (self.crops / f"{pid}_crop.jpg").write_bytes(b"\xff\xd8" + pid.encode() + b"\xff\xd9")
            self.payloads[pid] = {"crop_path": f"crops/{pid}_crop.jpg"}
        self.assignments = self.root / "person_test_assignments.jsonl"
        self.assignments.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n", encoding="utf-8")
        self.gt = self.root / "gt.jsonl"
        self.gt.write_text(json.dumps({"_meta": True}) + "\n" + "\n".join(
            json.dumps(dict(point_id=k, pid=v, status="labeled")) for k, v in GT.items()) + "\n", encoding="utf-8")
        self.labels = self.root / "cluster_labels.jsonl"
        self.labels.write_text(json.dumps(dict(cluster_id=CID_A, cluster_name="black jacket · jeans",
                                               label_confidence=0.61)) + "\n", encoding="utf-8")
        self.fake = FakeQdrant(self.payloads)

    def run_main(self, *extra):
        argv = ["--assignments", str(self.assignments), "--project-root", str(self.root),
                "--collection", "forensic_person", "--qdrant-url", "http://fake:6333", "--config",
                str(self.root / "missing.yaml"), *extra]
        out = io.StringIO()
        with patch.object(exp, "QdrantHTTP", lambda url, key=None: self.fake), contextlib.redirect_stdout(out):
            code = exp.main(argv)
        return code, out.getvalue()

    def test_copy_export_with_gt_and_labels(self):
        out_dir = self.root / "folders"
        code, stdout = self.run_main("--output-dir", str(out_dir), "--gt-matches", str(self.gt),
                                     "--labels", str(self.labels))
        self.assertEqual(code, 0)
        self.assertEqual(stdout.count("RESULT_SUMMARY:"), 1)
        self.assertEqual(stdout.count("RESULT_HTML:"), 1)
        self.assertIn(str(out_dir / exp.INDEX_NAME), stdout)
        a_dir = out_dir / "c0001_n3_pid0007_pur67_mix2_black-jacket-jeans_aaaaaaaa"
        self.assertEqual(sorted(p.name for p in a_dir.iterdir()),
                         ["pid0007_p1_crop.jpg", "pid0007_p2_crop.jpg", "pid0012_p3_crop.jpg"])
        self.assertEqual((a_dir / "pid0007_p1_crop.jpg").read_bytes(), (self.crops / "p1_crop.jpg").read_bytes())
        self.assertTrue((out_dir / "c0002_n2_pid0003_pur100_bbbbbbbb" / "pid0003_p4_crop.jpg").is_file())
        self.assertTrue((out_dir / "c0003_n1_gtNA_cccccccc" / "pidNA_p6_crop.jpg").is_file())
        noise_dir = out_dir / "_noise_n2"
        self.assertEqual(sorted(p.name for p in noise_dir.iterdir()), ["pid0007_p7_crop.jpg"])   # p8 누락
        report = json.loads((out_dir / exp.REPORT_NAME).read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "warning")
        self.assertEqual([w["code"] for w in report["warnings"]], ["CONFIG_UNAVAILABLE", "MISSING_CROPS"])
        self.assertEqual(report["counts"], {"copied": 7, "missing": 1})
        self.assertEqual(report["folder_count"], 4)
        self.assertEqual([f["name"] for f in report["folders"]][:2],
                         ["c0001_n3_pid0007_pur67_mix2_black-jacket-jeans_aaaaaaaa", "c0002_n2_pid0003_pur100_bbbbbbbb"])
        self.assertEqual(report["folders"][0]["label"], "black jacket · jeans")
        self.assertEqual({i["role"] for i in report["inputs"]}, {"assignments", "gt_matches", "labels"})
        # 한글 표 머리 키는 사이드카에 남기지 않는다
        self.assertNotIn("대상", report)
        manifest = (out_dir / exp.MANIFEST_NAME).read_text(encoding="utf-8-sig").splitlines()
        self.assertEqual(manifest[0], ",".join(exp.MANIFEST_FIELDS))
        self.assertEqual(len(manifest), 1 + len(ROWS))
        missing = [line for line in manifest if line.endswith(",missing")]
        self.assertEqual(len(missing), 1)
        self.assertIn(",p8,", missing[0])
        html = (out_dir / exp.INDEX_NAME).read_text(encoding="utf-8")
        self.assertIn("c0001_n3_pid0007_pur67_mix2_black-jacket-jeans_aaaaaaaa", html)
        self.assertIn("MISSING_CROPS", html)
        self.assertIn("pid0007_p1_crop.jpg", html)          # 미리보기 썸네일 링크
        self.assertIn("c0001_n3_pid0007_pur67_mix2_black-jacket-jeans_aaaaaaaa/pid0007_p1_crop.jpg", html)
        self.assertEqual(html.count("<details>"), 4)        # 폴더마다 파일 이름 목록
        self.assertFalse(report["dry_run"])
        self.assertEqual(self.fake.calls[0][0], "forensic_person")
        self.assertEqual(sorted(self.fake.calls[0][1]), sorted(r["point_id"] for r in ROWS))

    def test_rerun_keeps_files_and_flags_stale(self):
        out_dir = self.root / "folders"
        (out_dir / "old_leftover").mkdir(parents=True)
        self.run_main("--output-dir", str(out_dir), "--gt-matches", str(self.gt))
        code, _ = self.run_main("--output-dir", str(out_dir), "--gt-matches", str(self.gt))
        self.assertEqual(code, 0)
        report = json.loads((out_dir / exp.REPORT_NAME).read_text(encoding="utf-8"))
        self.assertEqual(report["counts"], {"kept": 7, "missing": 1})
        self.assertEqual(report["stale_folders"], ["old_leftover"])
        self.assertIn("STALE_FOLDERS", [w["code"] for w in report["warnings"]])

    def test_clean_removes_only_recorded_folders(self):
        out_dir = self.root / "folders"
        self.run_main("--output-dir", str(out_dir), "--gt-matches", str(self.gt))
        (out_dir / "user_folder").mkdir()
        # 다른 계획(GT 없음) → 폴더 이름이 바뀐다. --clean 이 이전 기록 폴더만 지운다.
        code, _ = self.run_main("--output-dir", str(out_dir), "--clean")
        self.assertEqual(code, 0)
        names = sorted(p.name for p in out_dir.iterdir() if p.is_dir())
        self.assertEqual(names, ["_noise_n2", "c0001_n3_aaaaaaaa", "c0002_n2_bbbbbbbb", "c0003_n1_cccccccc", "user_folder"])
        report = json.loads((out_dir / exp.REPORT_NAME).read_text(encoding="utf-8"))
        self.assertEqual(report["removed_folders"], 4)
        self.assertEqual(report["stale_folders"], ["user_folder"])
        self.assertTrue((out_dir / "c0001_n3_aaaaaaaa" / "p1_crop.jpg").is_file())

    def test_hardlink_mode(self):
        out_dir = self.root / "linked"
        code, _ = self.run_main("--output-dir", str(out_dir), "--mode", "hardlink", "--noise", "skip")
        self.assertEqual(code, 0)
        report = json.loads((out_dir / exp.REPORT_NAME).read_text(encoding="utf-8"))
        self.assertEqual(report["points"], 6)
        self.assertNotIn("_noise_n2", [f["name"] for f in report["folders"]])
        actions = report["counts"]
        self.assertEqual(sum(actions.get(k, 0) for k in ("linked", "copied_fallback")), 6)
        if actions.get("linked"):
            linked = out_dir / "c0001_n3_aaaaaaaa" / "p1_crop.jpg"
            self.assertEqual(os.stat(linked).st_nlink, 2)

    def test_group_by_pid_export(self):
        out_dir = self.root / "by_pid"
        code, _ = self.run_main("--output-dir", str(out_dir), "--gt-matches", str(self.gt), "--group-by", "gt-pid",
                                "--min-cluster-size", "3")
        self.assertEqual(code, 0)
        self.assertEqual(sorted(p.name for p in (out_dir / "pid0007_n3_k1").iterdir()),
                         ["c0001_p1_crop.jpg", "c0001_p2_crop.jpg", "noise_p7_crop.jpg"])
        self.assertEqual(sorted(p.name for p in (out_dir / "_no_gt_n2").iterdir()), ["c0003_p6_crop.jpg"])
        report = json.loads((out_dir / exp.REPORT_NAME).read_text(encoding="utf-8"))
        self.assertIn("MIN_SIZE_IGNORED", [w["code"] for w in report["warnings"]])
        html = (out_dir / exp.INDEX_NAME).read_text(encoding="utf-8")
        self.assertIn("pid0007_n3_k1", html)
        self.assertIn("군집 수", html)

    def test_missing_optional_inputs_warn_but_run(self):
        out_dir = self.root / "warn"
        code, _ = self.run_main("--output-dir", str(out_dir), "--gt-matches", str(self.root / "nope.jsonl"),
                                "--labels", str(self.root / "nope_labels.jsonl"))
        self.assertEqual(code, 0)
        report = json.loads((out_dir / exp.REPORT_NAME).read_text(encoding="utf-8"))
        codes = [w["code"] for w in report["warnings"]]
        self.assertIn("GT_MISSING", codes)
        self.assertIn("LABELS_MISSING", codes)
        self.assertTrue((out_dir / "c0001_n3_aaaaaaaa").is_dir())

    def test_labels_from_other_run_warn(self):
        other = self.root / "other_labels.jsonl"
        other.write_text(json.dumps(dict(cluster_id="leiden:person:zzzz", cluster_name="x")) + "\n", encoding="utf-8")
        out_dir = self.root / "other"
        self.run_main("--output-dir", str(out_dir), "--labels", str(other))
        report = json.loads((out_dir / exp.REPORT_NAME).read_text(encoding="utf-8"))
        self.assertIn("LABELS_UNMATCHED", [w["code"] for w in report["warnings"]])

    def test_dry_run_writes_only_html_and_manifest(self):
        out_dir = self.root / "preview"
        code, stdout = self.run_main("--output-dir", str(out_dir), "--gt-matches", str(self.gt),
                                     "--labels", str(self.labels), "--dry-run", "--clean")
        self.assertEqual(code, 0)
        self.assertEqual(sorted(p.name for p in out_dir.iterdir()),
                         sorted([exp.INDEX_NAME, exp.MANIFEST_NAME, exp.REPORT_NAME]))   # 폴더·파일 없음
        self.assertEqual(stdout.count("RESULT_HTML:"), 1)
        self.assertIn("dry-run", stdout)
        report = json.loads((out_dir / exp.REPORT_NAME).read_text(encoding="utf-8"))
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["counts"], {"planned": 7, "missing": 1})
        self.assertEqual(report["removed_folders"], 0)
        codes = [w["code"] for w in report["warnings"]]
        self.assertIn("CLEAN_IGNORED", codes)
        self.assertIn("MISSING_CROPS", codes)
        manifest = (out_dir / exp.MANIFEST_NAME).read_text(encoding="utf-8-sig").splitlines()
        self.assertEqual(sum(1 for line in manifest if line.endswith(",planned")), 7)
        html = (out_dir / exp.INDEX_NAME).read_text(encoding="utf-8")
        self.assertIn("c0001_n3_pid0007_pur67_mix2_black-jacket-jeans_aaaaaaaa", html)
        self.assertIn("pid0007_p1_crop.jpg", html)                        # 실제 실행 때 붙을 파일 이름
        self.assertIn("../crops/p1_crop.jpg", html)                        # 링크·썸네일은 원본 crop
        self.assertNotIn('href="c0001_', html)                             # 없는 폴더로의 링크 없음
        self.assertEqual(html.count("<details>"), 4)
        self.assertIn("미리보기", html)

    def test_list_limit_truncates_file_lists(self):
        out_dir = self.root / "limited"
        code, _ = self.run_main("--output-dir", str(out_dir), "--dry-run", "--list-limit", "1")
        self.assertEqual(code, 0)
        html = (out_dir / exp.INDEX_NAME).read_text(encoding="utf-8")
        self.assertIn("외 2개", html)                                       # 3개짜리 군집에서 1개만 표시
        self.assertIn("p1_crop.jpg", html)
        self.assertNotIn("p3_crop.jpg", html)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            exp.parse_args(["--list-limit", "-1"])

    def test_gt_pid_requires_matches(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                exp.parse_args(["--group-by", "gt-pid"])
            with self.assertRaises(SystemExit):
                exp.parse_args(["--min-cluster-size", "0"])
            with self.assertRaises(SystemExit):
                self.run_main("--group-by", "gt-pid", "--gt-matches", str(self.root / "nope.jsonl"))


if __name__ == "__main__":
    unittest.main()
