"""오프라인 테스트: eval/track_gt_eval.py — IDF1/HOTA/CLEAR 지표, 구간 분할(id 재사용), GT 구성(라벨·split·ignore), 시트·평가 end-to-end(영상 없음), 원장·채택 기준."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import criteria as C  # noqa: E402
from bench import ledger as L  # noqa: E402
from eval import track_gt_eval as T  # noqa: E402


def box(x, y=0):
    return [float(x), float(y), float(x) + 10.0, float(y) + 20.0]


def seq(pid, frames, x0):
    return {f: (pid, box(x0 + f)) for f in frames}


def scene(*tracks):
    out = {}
    for t in tracks:
        for f, item in t.items():
            out.setdefault(f, []).append(item)
    return out


class MetricTests(unittest.TestCase):
    def test_perfect(self):
        gt = scene(seq("A", range(10), 0), seq("B", range(10), 100))
        pr = scene(seq("1", range(10), 0), seq("2", range(10), 100))
        m = T.evaluate(gt, pr, {})
        self.assertEqual((m["idf1"], m["hota"], m["mota"], m["idsw"], m["splits"], m["over_merges"], m["fp"], m["fn"]), (1.0, 1.0, 1.0, 0, 0, 0, 0, 0))

    def test_id_switch(self):
        gt = scene(seq("A", range(10), 0), seq("B", range(10), 100))
        pr = scene(seq("1", range(5), 0), seq("3", range(5, 10), 0), seq("2", range(10), 100))
        m = T.evaluate(gt, pr, {})
        self.assertEqual(m["idsw"], 1)
        self.assertEqual(m["splits"], 1)
        self.assertAlmostEqual(m["idf1"], 0.75)
        self.assertAlmostEqual(m["assa"], 0.75, places=3)
        self.assertEqual(m["deta"], 1.0)
        self.assertAlmostEqual(m["hota"], 0.75 ** 0.5, places=3)

    def test_over_merge_and_fragment(self):
        gt = scene(seq("A", range(10), 0), seq("B", range(10, 20), 100))
        pr = scene(seq("1", range(10), 0), seq("1", range(10, 20), 100))          # 한 pred id 가 두 사람 → 과병합
        m = T.evaluate(gt, pr, {})
        self.assertEqual(m["over_merges"], 1)
        self.assertEqual(m["idsw"], 0)
        self.assertAlmostEqual(m["idf1"], 0.5)
        pr2 = scene(seq("1", list(range(4)) + list(range(6, 10)), 0), seq("2", range(10, 20), 100))   # 중간 2 프레임 끊김
        m2 = T.evaluate(gt, pr2, {})
        self.assertEqual(m2["fragments"], 1)
        self.assertEqual(m2["fn"], 2)
        self.assertEqual(m2["idsw"], 0)

    def test_ignore_removes_overlapping_predictions(self):
        gt = scene(seq("A", range(5), 0))
        ignore = {f: [box(100 + f)] for f in range(5)}
        pr = scene(seq("1", range(5), 0), seq("9", range(5), 100))
        m = T.evaluate(gt, pr, ignore)
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["ignored_pred_boxes"], 5)
        self.assertEqual(m["pred_ids"], 1)

    def test_iou_matrix(self):
        m = T.iou_matrix([box(0)], [box(0), box(5), box(50)])
        self.assertAlmostEqual(m[0, 0], 1.0)
        self.assertAlmostEqual(m[0, 1], 1 / 3, places=4)
        self.assertEqual(m[0, 2], 0.0)
        self.assertEqual(T.iou_matrix([], [box(0)]).shape, (0, 1))


def rows_fixture():
    rows = []
    for f in range(10):
        rows.append({"frame_idx": f, "track_id": 1, "short_track_id": 1, "long_track_id": 7, "bbox": box(f), "confidence": 0.9,
                     "timestamp_sec": f / 30, "final_db_route": "person", "class_name": "person"})
    for f in range(10, 20):
        rows.append({"frame_idx": f, "track_id": 2, "short_track_id": 2, "long_track_id": 7, "bbox": box(f), "confidence": 0.9,
                     "timestamp_sec": f / 30, "final_db_route": "person", "class_name": "person"})
    for f in range(5):
        rows.append({"frame_idx": f, "track_id": 3, "short_track_id": 3, "long_track_id": 8, "bbox": box(200 + f), "confidence": 0.5,
                     "timestamp_sec": f / 30, "final_db_route": "reject", "class_name": "person"})
    return rows


class SegmentTests(unittest.TestCase):
    def test_segments_and_pred_variants(self):
        segs = T.segment_rows(rows_fixture())
        self.assertEqual(sorted(segs), ["1.0", "2.0", "3.0"])
        self.assertEqual(segs["1.0"]["long_id"], 7)
        self.assertEqual(segs["3.0"]["route"], "reject")
        self.assertEqual(segs["2.0"]["first_frame"], 10)
        self.assertEqual(segs["1.0"]["n_frames"], 10)
        ids = lambda pred: {p for items in pred.values() for p, _ in items}  # noqa: E731
        self.assertEqual(ids(T.pred_from_rows(rows_fixture(), "raw")), {"1", "2", "3"})
        self.assertEqual(ids(T.pred_from_rows(rows_fixture(), "segment")), {"1.0", "2.0", "3.0"})
        self.assertEqual(ids(T.pred_from_rows(rows_fixture(), "long")), {"7", "8"})
        self.assertEqual(T.sample_indices(list(range(100)), 3), [0, 50, 99])
        self.assertEqual(T.sample_indices([4, 5], 3), [4, 5])

    def test_id_reuse_splits_into_segments(self):
        # 추적기 id 1 이 두 번 쓰임: 0~9 (긴 트랙 7) 와 50~59 (긴 트랙 9, 다른 사람) — 시간 간격으로도, 긴 트랙 변경으로도 끊긴다
        rows = [r for r in rows_fixture() if r["track_id"] == 1]
        rows += [{"frame_idx": f, "track_id": 1, "short_track_id": 1, "long_track_id": 9, "bbox": box(300 + f), "confidence": 0.9,
                  "timestamp_sec": f / 30, "final_db_route": "person"} for f in range(50, 60)]
        rows += [{"frame_idx": f, "track_id": 1, "short_track_id": 1, "long_track_id": 9, "bbox": box(300 + f), "confidence": 0.9,
                  "timestamp_sec": f / 30, "final_db_route": "person"} for f in range(60, 65)]          # 같은 긴 트랙, 연속 → 같은 구간
        segs = T.segment_rows(rows, max_gap=30)
        self.assertEqual(sorted(segs), ["1.0", "1.1"])
        self.assertEqual((segs["1.1"]["first_frame"], segs["1.1"]["last_frame"], segs["1.1"]["long_id"]), (50, 64, 9))
        raw = T.pred_from_rows(rows, "raw")
        self.assertEqual({p for items in raw.values() for p, _ in items}, {"1"})
        # raw 는 두 사람을 한 id 로 → 과병합 1, 구간은 0
        gt = scene(seq("A", range(10), 0), seq("B", range(50, 65), 300))
        self.assertEqual(T.evaluate(gt, raw, {})["over_merges"], 1)
        self.assertEqual(T.evaluate(gt, T.pred_from_rows(rows, "segment"), {})["over_merges"], 0)
        # 같은 긴 트랙이라도 max_gap 보다 긴 공백이면 새 구간
        segs2 = T.segment_rows(rows, max_gap=5)
        self.assertEqual(sorted(segs2), ["1.0", "1.1"])
        rows2 = [r for r in rows if r["frame_idx"] not in (55, 56, 57, 58, 59, 60)]
        self.assertEqual(sorted(T.segment_rows(rows2, max_gap=5)), ["1.0", "1.1", "1.2"])

    def test_build_gt_defaults_labels_split(self):
        segs = {sid: {"long_id": t["long_id"], "frames": t["frames"], "boxes": t["boxes"]} for sid, t in T.segment_rows(rows_fixture()).items()}
        proposals = {"short_tracks": [{"segment_id": "1.0", "default_gt_id": "P7", "default_status": "person"},
                                      {"segment_id": "2.0", "default_gt_id": "P7", "default_status": "person"},
                                      {"segment_id": "3.0", "default_gt_id": "", "default_status": "ignore"}]}
        gt, ig, info = T.build_gt(segs, proposals, {})
        self.assertEqual(info["gt_ids"], 1)
        self.assertEqual(info["ignored_segments"], 1)
        self.assertEqual(len(ig), 5)
        self.assertFalse(info["labeled"])
        labels = {"2.0": {"gt_id": "P8", "status": "person"},
                  "1.0": {"gt_id": "P7", "status": "person", "split_frame": "5", "split_gt_id": "P9"},
                  "3.0": {"gt_id": "P3", "status": "person"}}
        gt, ig, info = T.build_gt(segs, proposals, labels)
        self.assertEqual(info["gt_ids"], 4)
        self.assertEqual(info["ignored_segments"], 0)
        self.assertEqual(info["splits_labeled"], 1)
        self.assertEqual(gt[4][0][0], "P7")
        self.assertEqual(gt[5][0][0], "P9")
        self.assertEqual(len(ig), 0)


class EndToEndTests(unittest.TestCase):
    def test_sheet_and_eval_without_video(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc = td / "processed" / "vid1"
            proc.mkdir(parents=True)
            (proc / "final_routed_tracks.json").write_text(json.dumps(rows_fixture()), encoding="utf-8")
            gt_dir = td / "gt"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                T.main(["sheet", "--videos", "vid1", "--processed-root", str(td / "processed"), "--videos-root", str(td / "novideos"), "--gt-dir", str(gt_dir)])
            self.assertTrue((gt_dir / "vid1" / "sheet.html").is_file())
            self.assertTrue((gt_dir / "vid1" / "boxes.jsonl").is_file())
            html = (gt_dir / "vid1" / "sheet.html").read_text(encoding="utf-8")
            self.assertIn('data-item="1.0"', html)
            self.assertIn("이미지 없음", html)
            props = json.loads((gt_dir / "vid1" / "proposals.json").read_text(encoding="utf-8"))
            self.assertEqual((props["n_segments"], props["n_person_long"], props["max_gap"]), (3, 1, T.DEFAULT_MAX_GAP))
            # pseudo: 스티처 = 정답 → after 완벽, before(구간) 는 1.0→2.0 이어붙임에서 IDSW 1
            with contextlib.redirect_stdout(buf):
                out = T.main(["eval", "--gt-dir", str(gt_dir), "--processed-root", str(td / "processed"), "--output-dir", str(td / "res"), "--no-ledger"])
            self.assertTrue(out["pseudo_gt"])
            self.assertEqual(out["after"]["idf1"], 1.0)
            self.assertEqual(out["before"]["idsw"], 1)
            self.assertEqual(out["raw"]["idsw"], 1)
            self.assertEqual(out["metrics"]["idsw_ratio"], 0.0)
            self.assertEqual(out["metrics"]["idsw_raw"], 1)
            self.assertTrue((td / "res" / "stitched_pseudo" / "track_eval.json").is_file())
            # 라벨: 2.0 구간은 다른 사람 → 스티처 과병합 1, 추적기·구간은 0
            labels = {"kind": "track_labels", "labels": {"1.0": {"gt_id": "P7", "status": "person"}, "2.0": {"gt_id": "P8", "status": "person"},
                                                          "3.0": {"status": "ignore"}}}
            (gt_dir / "vid1" / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
            with contextlib.redirect_stdout(buf):
                out = T.main(["eval", "--gt-dir", str(gt_dir), "--processed-root", str(td / "processed"), "--output-dir", str(td / "res"),
                              "--name", "t", "--ledger", str(td / "ledger.jsonl")])
            self.assertFalse(out["pseudo_gt"])
            self.assertEqual(out["after"]["over_merges"], 1)
            self.assertEqual(out["before"]["over_merges"], 0)
            self.assertEqual(out["before"]["idsw"], 0)
            self.assertEqual(out["gt"]["labeled_videos"], 1)
            entries = L.read_entries(td / "ledger.jsonl")
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["stage"], "track")
            self.assertTrue(entries[0]["run_id"].startswith("track_"))
            self.assertEqual(entries[0]["metrics"]["over_merges"], 1)
            ev = C.evaluate(entries[0])
            self.assertEqual(ev["status"], "fail")            # 과병합 1 → 미달 (idsw_ratio 는 None → 건너뜀)
            self.assertIn("report.md", [p.name for p in (td / "res" / "t").iterdir()])

    def test_load_routed_fallback_to_tracks_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "v"
            d.mkdir()
            with (d / "tracks.jsonl").open("w", encoding="utf-8") as f:
                for r in rows_fixture()[:3]:
                    f.write(json.dumps({k: r[k] for k in ("frame_idx", "track_id", "bbox", "confidence")}) + "\n")
            rows, src = T.load_routed(Path(td), "v")
            self.assertEqual(src, "tracks.jsonl")
            segs = T.segment_rows(rows)
            self.assertEqual(segs["1.0"]["long_id"], 1)              # long id 없으면 short 와 같음
            with self.assertRaises(SystemExit):
                T.load_routed(Path(td), "missing")


if __name__ == "__main__":
    unittest.main()
