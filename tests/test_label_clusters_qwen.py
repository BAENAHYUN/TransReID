"""label_clusters_qwen 순수 함수 테스트 — 모델·Qdrant 없음."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clustering import label_clusters_qwen as lq  # noqa: E402


def members(n):
    return [dict(point_id=f"p{i}", cluster_id="c") for i in range(n)]


class RepresentativeTests(unittest.TestCase):
    def test_small_cluster_returns_all(self):
        self.assertEqual([m["point_id"] for m in lq.choose_representatives(members(4), 6, 1)], ["p0", "p1", "p2", "p3"])

    def test_spread_and_determinism(self):
        a = lq.choose_representatives(members(50), 6, 7)
        b = lq.choose_representatives(members(50), 6, 7)
        ids = [m["point_id"] for m in a]
        self.assertEqual(ids, [m["point_id"] for m in b])
        self.assertEqual(len(ids), 6)
        self.assertEqual(len(set(ids)), 6)
        self.assertTrue({"p0", "p25", "p49"} <= set(ids))                       # 앞·중간·뒤 고정
        self.assertNotEqual(ids, [m["point_id"] for m in lq.choose_representatives(members(50), 6, 8)])

    def test_count_smaller_than_anchors(self):
        self.assertEqual(len(lq.choose_representatives(members(50), 2, 1)), 2)


class MontageTests(unittest.TestCase):
    def test_montage_geometry(self):
        images = [Image.new("RGB", (40, 120), (200, 0, 0)), Image.new("RGB", (60, 90), (0, 200, 0)),
                  Image.new("RGB", (400, 50), (0, 0, 200))]                        # 마지막은 가로가 매우 긴 crop
        montage = lq.build_montage(images, height=100, gap=4)
        self.assertEqual(montage.size[1], 100)
        widths = [round(40 * 100 / 120), round(60 * 100 / 90), 200]                 # 세로의 2배로 잘림
        self.assertEqual(montage.size[0], sum(widths) + 4 * 2)
        self.assertEqual(montage.getpixel((0, 50)), (200, 0, 0))
        self.assertEqual(montage.getpixel((widths[0] + 1, 50)), (255, 255, 255))   # 간격은 흰색


class ReplyTests(unittest.TestCase):
    def test_prompt_mentions_count_and_forbids_sensitive_attributes(self):
        ko = lq.build_prompt(6, "ko")
        self.assertIn("6장", ko)
        self.assertIn("성별", ko)
        self.assertIn('"consistent"', ko)
        en = lq.build_prompt(3, "en")
        self.assertIn("3 CCTV crops", en)
        self.assertIn("gender", en)

    def test_extract_json_object_variants(self):
        obj = {"upper": "노란색 티셔츠", "lower": "검은 바지", "items": "없음", "consistent": True, "name": "노란 반팔에 검은 바지"}
        plain = json.dumps(obj, ensure_ascii=False)
        self.assertEqual(lq.extract_json_object(plain), obj)
        self.assertEqual(lq.extract_json_object("```json\n" + plain + "\n```"), obj)
        self.assertEqual(lq.extract_json_object("결과입니다: " + plain + " 끝."), obj)
        self.assertEqual(lq.extract_json_object('{"a": {"b": 1}} trailing {'), {"a": {"b": 1}})
        self.assertIsNone(lq.extract_json_object("no json here"))
        self.assertIsNone(lq.extract_json_object(""))
        self.assertIsNone(lq.extract_json_object("[1, 2, 3]"))

    def test_clean_name_and_to_bool(self):
        self.assertEqual(lq.clean_name('  "노란  반팔에\n검은 바지".  '), "노란 반팔에 검은 바지")
        self.assertEqual(len(lq.clean_name("가" * 80)), lq.MAX_NAME_CHARS)
        self.assertEqual(lq.clean_name(None), "")
        self.assertIs(lq.to_bool("True"), True)
        self.assertIs(lq.to_bool("아니오"), False)
        self.assertIs(lq.to_bool(1), True)
        self.assertIsNone(lq.to_bool("maybe"))
        self.assertIsNone(lq.to_bool(None))

    def test_interpret_reply_labeled_mixed_and_errors(self):
        raw = json.dumps({"upper": "흰색 셔츠", "lower": "청바지", "items": "없음", "consistent": True, "name": "흰 셔츠에 청바지"},
                         ensure_ascii=False)
        r = lq.interpret_reply(raw, keep_mixed=False)
        self.assertEqual((r["status"], r["name"], r["upper"]), ("labeled", "흰 셔츠에 청바지", "흰색 셔츠"))
        mixed = json.dumps({"upper": "다양", "lower": "다양", "items": "없음", "consistent": "false", "name": "혼합"}, ensure_ascii=False)
        r = lq.interpret_reply(mixed, keep_mixed=False)
        self.assertEqual((r["status"], r["name"], r["qwen_name"]), ("mixed", "", "혼합"))     # 폴더 이름에 안 붙음
        r = lq.interpret_reply(mixed, keep_mixed=True)
        self.assertEqual((r["status"], r["name"]), ("mixed", "혼합"))
        no_name = json.dumps({"upper": "빨간 후드", "lower": "검은 바지", "consistent": True}, ensure_ascii=False)
        r = lq.interpret_reply(no_name, keep_mixed=False)
        self.assertEqual((r["status"], r["name"]), ("labeled", "빨간 후드 검은 바지"))        # upper+lower 로 대체
        r = lq.interpret_reply("모델이 그냥 말로 답함", keep_mixed=False)
        self.assertEqual(r["status"], "parse_error")
        r = lq.interpret_reply(json.dumps({"consistent": True}), keep_mixed=False)
        self.assertEqual(r["status"], "parse_error")

    def test_make_record_fields(self):
        reply = lq.interpret_reply(json.dumps({"upper": "a", "lower": "b", "items": "c", "consistent": True, "name": "a b"}), False)
        rec = lq.make_record("leiden:person:abcdef0123456789", 3, 42, reply, "raw text", ["p1", "p2"], "montage/x.jpg", "m", 0.5)
        self.assertEqual((rec["cluster_name"], rec["display_name"], rec["status"], rec["label_confidence"]), ("a b", "a b", "labeled", 1.0))
        self.assertEqual(rec["cluster_description"], "upper=a, lower=b, items=c, consistent=True")
        self.assertEqual(rec["qwen"]["raw"], "raw text")
        self.assertEqual(rec["representative_point_ids"], ["p1", "p2"])
        bad = lq.make_record("c", 1, 2, dict(status="parse_error", name="", upper="", lower="", items="", consistent=None), "?", [], None, "m", 0.1)
        self.assertEqual((bad["cluster_name"], bad["display_name"], bad["label_confidence"]), ("", "응답 해석 실패", 0.0))
        mixed = lq.make_record("c", 1, 2, lq.interpret_reply(json.dumps({"name": "x", "consistent": False}), False), "?", [], None, "m", 0.1)
        self.assertEqual((mixed["cluster_name"], mixed["display_name"], mixed["label_confidence"]), ("", "x", 0.0))


class ResumeTests(unittest.TestCase):
    def test_load_existing_skips_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cluster_labels.jsonl"
            lines = [dict(cluster_id="a", status="labeled", cluster_name="x"), dict(cluster_id="b", status="parse_error"),
                     dict(cluster_id="c", status="mixed", cluster_name=""), "junk", dict(status="labeled")]
            path.write_text("\n".join(json.dumps(l) if isinstance(l, dict) else l for l in lines) + "\n", encoding="utf-8")
            existing = lq.load_existing(path)
            self.assertEqual(set(existing), {"a", "c"})
            self.assertEqual(lq.load_existing(Path(tmp) / "missing.jsonl"), {})

    def test_parse_args_validation(self):
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (["--representatives", "0"], ["--montage-height", "10"], ["--max-clusters", "-1"]):
                with self.assertRaises(SystemExit):
                    lq.parse_args(argv)
        args = lq.parse_args(["--max-clusters", "5"])
        self.assertEqual((args.max_clusters, args.representatives, args.lang), (5, 6, "ko"))


if __name__ == "__main__":
    unittest.main()
