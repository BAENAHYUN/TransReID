"""오프라인 테스트: P6 단계(track/object/qwen)가 원장·기준·러너·GUI 정의에 연결됐는지 — dry-run 명령, 엔트리→러너 인자, 이름 규칙, 원장 import."""
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
from bench import run as R  # noqa: E402

P6 = ("track", "object", "qwen")


def dry(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        R.main(list(argv) + ["--dry-run", "--no-ledger"])
    return buf.getvalue()


class WiringTests(unittest.TestCase):
    def test_stage_tables_cover_p6(self):
        for s in P6:
            self.assertIn(s, L.STAGES)
            self.assertTrue(L.METRIC_KEYS[s])
            self.assertTrue(L.VERIFY_TOLERANCES[s])
            self.assertTrue(C.ADOPTION_RULES[s])
            self.assertIn(s, C.CHART_AXES)
            self.assertIn(s, R.STAGE_KEYS)
            for metric, _op, _t, _label in C.ADOPTION_RULES[s]:
                self.assertIn(metric, L.METRIC_KEYS[s], f"{s}.{metric} 는 기준표 열에 있어야 한다")
            x, y = C.CHART_AXES[s]
            self.assertIn(x, L.METRIC_KEYS[s])
            self.assertIn(y, L.METRIC_KEYS[s])
        self.assertIn("sec_per_candidate", L.TIMING_METRICS)

    def test_gui_definitions(self):
        d = json.loads((ROOT / "gui_pipelines.json").read_text(encoding="utf-8"))
        ev = d["evaluation"]
        ids = [s["id"] for s in ev]
        self.assertEqual(len(ev), 14)
        self.assertEqual(ids[-3:], ["track_gt_eval", "object_pair_eval", "qwen_verify_eval"])
        for s in ev[-3:]:
            self.assertTrue((ROOT / s["script"]).is_file(), s["script"])
            self.assertEqual(s["args"][0]["choices"], ["sheet", "eval"])
        ledger_stage = next(a for a in ev[ids.index("bench_ledger")]["args"] if a["flag"] == "--stage")
        runner_cmd = next(a for a in ev[ids.index("bench_run")]["args"] if a["flag"] == "")
        for s in P6:
            self.assertIn(s, ledger_stage["choices"])
            self.assertIn(s, runner_cmd["choices"])


class RunnerTests(unittest.TestCase):
    def test_track_dry_run(self):
        out = dry(["track", "--videos", "v1", "v2", "--tracking-config", "pipeline_tracking_yolo26.yaml", "--gt-dir", "eval/gt/tracks"])
        self.assertEqual(out.count("video/batch_preprocess_videos_parallel.py"), 2)
        self.assertIn("--pattern v1", out)
        self.assertIn("--workers 1", out)
        self.assertIn("eval/track_gt_eval.py eval", out)
        self.assertIn("--videos v1 v2", out)
        self.assertIn("--tracking-config pipeline_tracking_yolo26.yaml", out)
        self.assertIn("--name pipeline_tracking_yolo26", out)
        self.assertIn("--record-pseudo", out)
        out2 = dry(["track", "--processed-root", "outputs/processed_videos"])
        self.assertNotIn("batch_preprocess", out2)
        self.assertIn("--processed-root outputs/processed_videos", out2)
        self.assertIn("--name tracks", out2)
        # --restitch: 검출·추적은 기존 출력, 스티처(SUSHI + link 옵션)만 다시 → sushi_inference 명령 (영상마다), 재추적 없음
        out3 = dry(["track", "--videos", "v1", "--tracking-config", "pipeline_tracking_sushi_link.yaml", "--restitch"])
        self.assertNotIn("batch_preprocess", out3)
        self.assertEqual(out3.count("video/sushi_inference.py"), 1)
        self.assertIn("--link-windows", out3)
        self.assertIn("tracks.jsonl", out3)
        self.assertIn("--name pipeline_tracking_sushi_link_restitch", out3)
        toks = out3.split("eval/track_gt_eval.py")[1].split()
        self.assertTrue(toks[toks.index("--processed-root") + 1].replace("\\", "/").endswith("/processed"))   # 평가는 run 폴더의 processed 를 읽음

    def test_object_and_qwen_dry_run(self):
        out = dry(["object", "--vector", "siglip2", "--threshold", "0.9"])
        self.assertIn("eval/object_pair_eval.py eval", out)
        self.assertIn("--vector siglip2", out)
        self.assertIn("--threshold 0.9", out)
        self.assertIn("--name siglip2", out)
        out = dry(["qwen", "--verify-mode", "filter", "--no-reranker", "--top-k", "5", "--max-queries", "2", "--model-id", "Qwen/X"])
        self.assertIn("eval/qwen_verify_eval.py eval", out)
        self.assertIn("--verify-mode filter", out)
        self.assertIn("--no-reranker", out)
        self.assertIn("--top-k 5", out)
        self.assertIn("--max-queries 2", out)
        self.assertIn("--model-id Qwen/X", out)
        self.assertIn("--name qwen_filter", out)

    def test_args_from_entry_and_command(self):
        track = L.entry_from_track_result({"name": "t", "generated_at": "2026-09-28T01:02:03", "metrics": {"idf1": 0.9, "over_merges": 0, "idsw_ratio": 0.2},
                                           "config": {"tracking_config": "pipeline_tracking.yaml", "videos": ["v1"], "iou": 0.5}, "gt": {"videos": 1}})
        a = R.args_from_entry(track)
        self.assertEqual(a["tracking_config"], "pipeline_tracking.yaml")
        self.assertEqual(a["videos"], ["v1"])
        self.assertNotIn("pred_file", a)
        cmd = R.runner_command("track", a)
        self.assertEqual(cmd[2], "track")
        self.assertIn("--tracking-config", cmd)
        ev = C.evaluate(track)
        self.assertEqual(ev["status"], "incomplete")                          # idsw/idsw_before 누락 + IDF1/HOTA 기준값 미정 → 확인 불가
        self.assertEqual({c["metric"]: c.get("reason") for c in ev["checks"] if c["ok"] is None}, {"idsw": "missing", "idf1": "baseline", "hota": "missing"})
        obj = L.entry_from_object_result({"name": "o", "metrics": {"map": 80.0, "pair_auc": 0.95, "cluster_pair_precision": 0.5},
                                          "config": {"vector": "dinov2", "collection": "forensic_object", "threshold": 0.97}, "gt": {}})
        self.assertEqual(R.args_from_entry(obj), {"name": "o", "vector": "dinov2", "collection": "forensic_object", "threshold": 0.97})
        self.assertEqual(C.evaluate(obj)["status"], "incomplete")              # 객체 mAP 기준값 미정
        full = L.entry_from_track_result({"name": "t2", "metrics": {"idf1": 0.9, "hota": 0.8, "over_merges": 0, "idsw": 4, "idsw_before": 10}, "config": {}, "gt": {}})
        ev2 = C.evaluate(full)
        self.assertEqual(ev2["status"], "incomplete")                         # 기준값 미정이면 다 있어도 통과 아님
        self.assertTrue(next(c for c in ev2["checks"] if c["metric"] == "idsw")["ok"])   # 4 ≤ 0.5 × 10
        self.assertEqual(next(c for c in ev2["checks"] if c["metric"] == "idsw")["target"], 5.0)
        qwen = L.entry_from_qwen_result({"name": "q", "metrics": {"p10_gain_pp": 12.0, "false_drop_rate": 0.05, "sec_per_candidate": 20.0},
                                         "config": {"model_id": "Qwen/Q", "verify_mode": "flag", "top_k": 20, "alpha": 0.7, "threshold": 0.5, "no_reranker": False}, "gt": {}})
        a = R.args_from_entry(qwen)
        self.assertEqual(a["model_id"], "Qwen/Q")
        self.assertNotIn("no_reranker", a)                 # False 는 생략
        self.assertEqual(C.evaluate(qwen)["status"], "pass")
        self.assertEqual(R.default_name("qwen", {"verify_mode": "filter"}), "qwen_filter")

    def test_pseudo_rows_are_flagged_and_kept_out_of_main_ledger(self):
        e = L.entry_from_track_result({"name": "t", "metrics": {"idf1": 1.0}, "pseudo_gt": True, "config": {}, "gt": {}})
        self.assertTrue(R.is_pseudo(e))
        u = L.entry_from_qwen_result({"name": "q", "metrics": {"unknown_ratio": 0.2}, "unlabeled_only": True, "config": {}, "gt": {}})
        self.assertTrue(R.is_pseudo(u))
        self.assertEqual(u["note"], "unlabeled")
        real = L.entry_from_track_result({"name": "t", "metrics": {"idf1": 0.9}, "config": {}, "gt": {}})
        self.assertFalse(R.is_pseudo(real))
        out = dry(["qwen", "--allow-unlabeled", "--max-queries", "1"])
        self.assertIn("--allow-unlabeled", out)
        self.assertIn("--record-pseudo", out)

    def test_import_skips_pseudo_and_unlabeled(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a").mkdir()
            (root / "b").mkdir()
            (root / "c").mkdir()
            (root / "a" / "track_eval.json").write_text(json.dumps({"name": "x", "metrics": {"idf1": 1.0}, "pseudo_gt": True, "config": {}, "gt": {}}), encoding="utf-8")
            (root / "b" / "track_eval.json").write_text(json.dumps({"name": "y", "metrics": {"idf1": 0.8}, "pseudo_gt": False, "config": {}, "gt": {}}), encoding="utf-8")
            (root / "c" / "qwen_verify_eval.json").write_text(json.dumps({"name": "z", "metrics": {"p10_after": 50.0}, "unlabeled_only": True, "config": {}, "gt": {}}), encoding="utf-8")
            got = L.collect_results(root, log=lambda *_: None)
            self.assertEqual([(e["stage"], e["name"]) for e in got], [("track", "y")])


if __name__ == "__main__":
    unittest.main()
