"""오프라인 테스트: eval/track_gt_eval.py — IDF1/HOTA/CLEAR(TrackEval 정의) 지표, 구간 분할(id 재사용·창 경계), ignore(MOT 방식), GT 구성(검토·split·중복),
시트·평가 end-to-end(영상 없음), manifest, 원장·채택 기준."""
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
from eval import gt_sheet as S  # noqa: E402
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
        self.assertEqual((m["idf1"], m["hota"], m["mota"], m["idsw"], m["fragments"], m["splits"], m["over_merges"], m["fp"], m["fn"]), (1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0))

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
        pr2 = scene(seq("1", list(range(4)) + list(range(6, 10)), 0), seq("2", range(10, 20), 100))   # 중간 2 프레임 끊김 → 단절 1
        m2 = T.evaluate(gt, pr2, {})
        self.assertEqual(m2["fragments"], 1)
        self.assertEqual(m2["fn"], 2)
        self.assertEqual(m2["idsw"], 0)
        pr3 = scene(seq("1", range(1, 10), 0), seq("2", range(10, 20), 100))                       # 첫 프레임만 놓침 → 단절 아님 (TrackEval)
        m3 = T.evaluate(gt, pr3, {})
        self.assertEqual((m3["fragments"], m3["fn"]), (0, 1))

    def test_clear_prefers_previous_timestep_only(self):
        # 프레임 0~4: A↔1. 프레임 5~9: A 만 있고 예측 1 은 사라지고 예측 9 가 겹침 → 9 로 매칭(IDSW 1). 프레임 10: 1 과 9 가 둘 다 A 에 겹치면 직전(9) 우선.
        gt = scene(seq("A", range(11), 0))
        pr = scene(seq("1", range(5), 0), seq("9", range(5, 11), 0), {10: ("1", box(10))})
        m = T.evaluate(gt, pr, {})
        self.assertEqual(m["idsw"], 1)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["splits"], 1)

    def test_ignore_mot_style(self):
        gt = scene(seq("A", range(5), 0))
        ignore = {f: [box(100 + f)] for f in range(5)}
        pr = scene(seq("1", range(5), 0), seq("9", range(5), 100))
        m = T.evaluate(gt, pr, ignore)
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["ignored_pred_boxes"], 5)
        self.assertEqual(m["pred_ids"], 1)
        # ignore 박스가 정상 GT 와 겹치는 위치: 정상 GT 에 더 잘 맞는 예측은 살아남는다 (일대일 공동 매칭)
        gt2 = scene(seq("A", range(3), 0))
        ignore2 = {f: [[3.0 + f, 0.0, 13.0 + f, 20.0]] for f in range(3)}     # A 와 70 % 겹치는 ignore
        pr2 = scene(seq("1", range(3), 0))
        m2 = T.evaluate(gt2, pr2, ignore2)
        self.assertEqual((m2["tp"], m2["ignored_pred_boxes"]), (3, 0))

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
        ids = lambda pred: {p for items in pred.values() for p, _ in items}  # noqa: E731
        self.assertEqual(ids(T.pred_from_rows(rows_fixture(), "raw")[0]), {"1", "2", "3"})
        self.assertEqual(ids(T.pred_from_rows(rows_fixture(), "segment")[0]), {"1.0", "2.0", "3.0"})
        self.assertEqual(ids(T.pred_from_rows(rows_fixture(), "long")[0]), {"7", "8"})

    def test_window_boundary_segment_spans_long_ids(self):
        # 추적기 id 0 이 0~19 연속인데 스티처가 창 경계(10)에서 long 1 → 8 로 갈랐다 → 구간은 하나(스티처와 무관), long_ids 두 개
        rows = [{"frame_idx": f, "track_id": 0, "short_track_id": 0, "long_track_id": 1 if f < 10 else 8, "bbox": box(f), "confidence": 0.9,
                 "timestamp_sec": f / 30, "final_db_route": "person"} for f in range(20)]
        segs = T.segment_rows(rows)
        self.assertEqual(sorted(segs), ["0.0"])
        self.assertEqual(segs["0.0"]["long_ids"], [1, 8])
        self.assertIn(segs["0.0"]["long_id"], (1, 8))
        # pseudo GT(구간 기본값 = 한 사람) 에 대해 after(long) 는 갈라짐 1 · IDSW 1, before(구간) 는 완벽
        gt = scene(seq("P", range(20), 0))
        after, _ = T.pred_from_rows(rows, "long")
        before, _ = T.pred_from_rows(rows, "segment")
        self.assertEqual((T.evaluate(gt, after, {})["splits"], T.evaluate(gt, after, {})["idsw"]), (1, 1))
        self.assertEqual(T.evaluate(gt, before, {})["idf1"], 1.0)

    def test_id_reuse_and_gap(self):
        rows = [r for r in rows_fixture() if r["track_id"] == 1]
        rows += [{"frame_idx": f, "track_id": 1, "short_track_id": 1, "long_track_id": 9, "bbox": box(300 + f), "confidence": 0.9,
                  "timestamp_sec": f / 30, "final_db_route": "person"} for f in range(50, 65)]
        segs = T.segment_rows(rows, max_gap=30)
        self.assertEqual(sorted(segs), ["1.0", "1.1"])
        self.assertEqual((segs["1.1"]["first_frame"], segs["1.1"]["last_frame"], segs["1.1"]["long_id"]), (50, 64, 9))
        raw, _ = T.pred_from_rows(rows, "raw")
        gt = scene(seq("A", range(10), 0), seq("B", range(50, 65), 300))
        self.assertEqual(T.evaluate(gt, raw, {})["over_merges"], 1)                       # raw 는 두 사람을 한 id 로
        self.assertEqual(T.evaluate(gt, T.pred_from_rows(rows, "segment")[0], {})["over_merges"], 0)
        rows2 = [r for r in rows if r["frame_idx"] not in (55, 56, 57, 58, 59, 60)]
        self.assertEqual(sorted(T.segment_rows(rows2, max_gap=5)), ["1.0", "1.1", "1.2"])

    def test_duplicate_id_in_frame_is_counted(self):
        rows = rows_fixture()
        rows.append({"frame_idx": 3, "track_id": 5, "short_track_id": 5, "long_track_id": 7, "bbox": box(500), "confidence": 0.8, "final_db_route": "person"})
        pred, dup = T.pred_from_rows(rows, "long")
        self.assertEqual(dup, 1)                                                          # 프레임 3 에 long 7 이 둘 → 동시 존재 과병합 신호
        self.assertEqual(len([p for p, _ in pred[3] if p == "7"]), 1)

    def test_build_gt_reviewed_defaults_split_dup(self):
        segs = {sid: {"long_id": t["long_id"], "frames": t["frames"], "boxes": t["boxes"]} for sid, t in T.segment_rows(rows_fixture()).items()}
        proposals = {"short_tracks": [{"segment_id": "1.0", "default_gt_id": "P7", "default_status": "person"},
                                      {"segment_id": "2.0", "default_gt_id": "P7", "default_status": "person"},
                                      {"segment_id": "3.0", "default_gt_id": "", "default_status": "ignore"}]}
        gt, ig, info = T.build_gt(segs, proposals, {})
        self.assertEqual((info["gt_ids"], info["ignored_segments"], info["coverage"], info["labeled"]), (1, 1, 0.0, False))
        self.assertEqual(len(ig), 5)
        # 검토 안 한 라벨(reviewed 없음·판정 필드 없음)은 무시된다
        gt, ig, info = T.build_gt(segs, proposals, {"2.0": {"gt_id": "P8", "status": "person", "reviewed": False}})
        self.assertEqual((info["gt_ids"], info["coverage"]), (1, 0.0))
        labels = {"2.0": {"gt_id": "P8", "status": "person", "reviewed": True},
                  "1.0": {"gt_id": "P7", "status": "person", "split_frame": "5", "split_gt_id": "P9", "reviewed": True},
                  "3.0": {"gt_id": "P3", "status": "person", "reviewed": True}}
        gt, ig, info = T.build_gt(segs, proposals, labels)
        self.assertEqual((info["gt_ids"], info["ignored_segments"], info["splits_labeled"], info["coverage"]), (4, 0, 1, 1.0))
        self.assertEqual(gt[4][0][0], "P7")
        self.assertEqual(gt[5][0][0], "P9")
        # 같은 프레임에 같은 gt_id 두 박스(라벨 오류) → 하나만 남기고 센다
        labels2 = {"1.0": {"gt_id": "P7", "status": "person", "reviewed": True}, "2.0": {"gt_id": "P7", "status": "person", "reviewed": True},
                   "3.0": {"gt_id": "P7", "status": "person", "reviewed": True}}
        gt, ig, info = T.build_gt(segs, proposals, labels2)
        self.assertEqual(info["gt_dup_boxes"], 5)


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
            html = (gt_dir / "vid1" / "sheet.html").read_text(encoding="utf-8")
            self.assertIn('data-item="1.0"', html)
            self.assertIn('data-field="reviewed"', html)
            self.assertIn("이미지 없음", html)
            props = json.loads((gt_dir / "vid1" / "proposals.json").read_text(encoding="utf-8"))
            self.assertEqual((props["n_segments"], props["n_person_long"], props["max_gap"]), (3, 1, T.DEFAULT_MAX_GAP))
            self.assertTrue(props["manifest"])
            self.assertIn(props["manifest"], html)
            # pseudo: 제안값 = 정답 → after 완벽, before(구간)·raw 는 1→2 이어붙임에서 IDSW 1; 이름에 __pseudo
            with contextlib.redirect_stdout(buf):
                out = T.main(["eval", "--gt-dir", str(gt_dir), "--processed-root", str(td / "processed"), "--output-dir", str(td / "res"), "--no-ledger"])
            self.assertTrue(out["pseudo_gt"])
            self.assertEqual(out["name"], "stitched__pseudo")
            self.assertEqual(out["after"]["idf1"], 1.0)
            self.assertEqual((out["before"]["idsw"], out["raw"]["idsw"], out["metrics"]["idsw_ratio"]), (1, 1, 0.0))
            self.assertTrue((td / "res" / "stitched__pseudo" / "track_eval.json").is_file())
            # 라벨(검토 완료): 2.0 은 다른 사람 → 스티처 과병합 1, 추적기·구간은 0; manifest 가 맞아야 한다
            labels = {"kind": "track_labels", "meta": {"manifest": props["manifest"]},
                      "labels": {"1.0": {"gt_id": "P7", "status": "person", "reviewed": True}, "2.0": {"gt_id": "P8", "status": "person", "reviewed": True},
                                 "3.0": {"status": "ignore", "reviewed": True}}}
            (gt_dir / "vid1" / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
            with contextlib.redirect_stdout(buf):
                out = T.main(["eval", "--gt-dir", str(gt_dir), "--processed-root", str(td / "processed"), "--output-dir", str(td / "res"),
                              "--name", "t", "--ledger", str(td / "ledger.jsonl")])
            self.assertFalse(out["pseudo_gt"])
            self.assertEqual(out["name"], "t")
            self.assertEqual((out["after"]["over_merges"], out["before"]["over_merges"], out["before"]["idsw"]), (1, 0, 0))
            self.assertEqual(out["gt"]["coverage"], 1.0)
            entries = L.read_entries(td / "ledger.jsonl")
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["stage"], "track")
            self.assertTrue(any(i["role"].startswith("gt:vid1:labels") for i in entries[0]["inputs"]))
            ev = C.evaluate(entries[0])
            self.assertEqual(ev["status"], "incomplete")           # IDF1/HOTA 기준값(기존) 미정 → 확인 불가 (통과로 표시하지 않음)
            self.assertFalse(next(c for c in ev["checks"] if c["metric"] == "over_merges")["ok"])
            # 부분 검토 → pseudo 취급 (원장 기록 안 함)
            labels["labels"]["3.0"] = {"status": "ignore", "reviewed": False}
            (gt_dir / "vid1" / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
            with contextlib.redirect_stdout(buf):
                out = T.main(["eval", "--gt-dir", str(gt_dir), "--processed-root", str(td / "processed"), "--output-dir", str(td / "res"), "--no-ledger"])
            self.assertTrue(out["pseudo_gt"] and out["partial_gt"])
            # manifest 불일치 → 중단, --ignore-manifest 로 강행
            labels["meta"]["manifest"] = "deadbeef0000"
            (gt_dir / "vid1" / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stdout(buf):
                    T.main(["eval", "--gt-dir", str(gt_dir), "--processed-root", str(td / "processed"), "--output-dir", str(td / "res"), "--no-ledger"])
            with contextlib.redirect_stdout(buf):
                T.main(["eval", "--gt-dir", str(gt_dir), "--processed-root", str(td / "processed"), "--output-dir", str(td / "res"), "--no-ledger", "--ignore-manifest"])

    def test_pred_file_without_long_id_and_multi_video_guard(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc = td / "processed" / "vid1"
            proc.mkdir(parents=True)
            (proc / "final_routed_tracks.json").write_text(json.dumps(rows_fixture()), encoding="utf-8")
            gt_dir = td / "gt"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                T.main(["sheet", "--videos", "vid1", "--processed-root", str(td / "processed"), "--videos-root", str(td / "no"), "--gt-dir", str(gt_dir)])
            pred = td / "pred.jsonl"
            with pred.open("w", encoding="utf-8") as f:
                for r in rows_fixture():
                    f.write(json.dumps({k: r[k] for k in ("frame_idx", "track_id", "bbox", "confidence")}) + "\n")
            with contextlib.redirect_stdout(buf):
                out = T.main(["eval", "--gt-dir", str(gt_dir), "--pred-file", str(pred), "--output-dir", str(td / "res"), "--no-ledger"])
            self.assertTrue(out["config"]["after_is_raw"])
            self.assertEqual(out["after"]["idsw"], out["raw"]["idsw"])
            (gt_dir / "vid2").mkdir()
            for n in ("boxes.jsonl", "proposals.json"):
                (gt_dir / "vid2" / n).write_bytes((gt_dir / "vid1" / n).read_bytes())
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stdout(buf):
                    T.main(["eval", "--gt-dir", str(gt_dir), "--pred-file", str(pred), "--output-dir", str(td / "res"), "--no-ledger"])

    def test_load_routed_fallback_to_tracks_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "v"
            d.mkdir()
            with (d / "tracks.jsonl").open("w", encoding="utf-8") as f:
                for r in rows_fixture()[:3]:
                    f.write(json.dumps({k: r[k] for k in ("frame_idx", "track_id", "bbox", "confidence")}) + "\n")
            rows, src = T.load_routed(Path(td), "v")
            self.assertEqual(src, "tracks.jsonl")
            self.assertEqual(T.segment_rows(rows)["1.0"]["long_id"], 1)
            with self.assertRaises(SystemExit):
                T.load_routed(Path(td), "missing")

    def test_sheet_helpers(self):
        self.assertEqual(S.manifest_of("k", ["b", "a"]), S.manifest_of("k", ["a", "b"]))
        self.assertNotEqual(S.manifest_of("k", ["a"]), S.manifest_of("k", ["a", "b"]))
        self.assertTrue(S.is_reviewed({"reviewed": True}))
        self.assertFalse(S.is_reviewed({"reviewed": False, "verdict": "same"}))
        self.assertTrue(S.is_reviewed({"verdict": "same"}))                 # 옛 파일: 판정 필드가 있으면 검토로 본다
        self.assertFalse(S.is_reviewed({"gt_id": "P1"}))
        self.assertEqual(S.display_name(r"C:\a\b\video.mp4"), "video.mp4")


if __name__ == "__main__":
    unittest.main()
