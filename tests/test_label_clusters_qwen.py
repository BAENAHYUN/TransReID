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
        """이름은 모델의 name 이 아니라 항목 답으로 짓는다 (v1 에서 name 칸이 프롬프트 예시를 베꼈다)."""
        raw = json.dumps({"upper": "흰색 셔츠", "lower": "청바지", "items": "없음", "consistent": True, "name": "흰 셔츠에 청바지"},
                         ensure_ascii=False)
        r = lq.interpret_reply(raw, keep_mixed=False)
        self.assertEqual((r["status"], r["name"], r["upper"]), ("labeled", "흰 셔츠에 청바지", "흰색 셔츠"))
        # 실제 사례: 항목은 검은 옷 + 노란 가방인데 name 은 프롬프트 예시 그대로
        copied = json.dumps({"upper": "검은색 상의", "lower": "검은색 바지", "items": "노란색 가방", "consistent": True,
                             "name": "노란 반팔에 검은 바지"}, ensure_ascii=False)
        r = lq.interpret_reply(copied, keep_mixed=False)
        self.assertEqual((r["status"], r["name"], r["qwen_name"]), ("labeled", "검은 상의에 검은 바지", "노란 반팔에 검은 바지"))
        # v2 응답: 색을 따로 받는다
        v2 = json.dumps({"upper_color": "회색", "upper": "민소매", "lower_color": "베이지색", "lower": "반바지", "items": "없음",
                         "consistent": True}, ensure_ascii=False)
        r = lq.interpret_reply(v2, keep_mixed=False)
        self.assertEqual((r["status"], r["name"], r["upper"], r["upper_color"]), ("labeled", "회색 민소매에 베이지 반바지", "회색 민소매", "회색"))
        # 섞인 군집: consistent=false 이거나 '다를 수 있는' 같은 답
        mixed = json.dumps({"upper": "검은색 상의", "lower": "색상이 다를 수 있는 하의", "consistent": True, "name": "노란 반팔에 검은 바지"},
                           ensure_ascii=False)
        r = lq.interpret_reply(mixed, keep_mixed=False)
        self.assertEqual((r["status"], r["name"]), ("mixed", ""))                             # 폴더 이름에 안 붙음
        self.assertEqual(lq.interpret_reply(mixed, keep_mixed=True)["name"], "검은 상의")
        r = lq.interpret_reply(json.dumps({"upper": "빨간 후드", "lower": "검은 바지", "consistent": False}, ensure_ascii=False), False)
        self.assertEqual((r["status"], r["name"]), ("mixed", ""))
        r = lq.interpret_reply(json.dumps({"upper": "빨간 후드", "lower": "검은 바지", "consistent": True}, ensure_ascii=False), False)
        self.assertEqual((r["status"], r["name"]), ("labeled", "빨간 후드티에 검은 바지"))
        r = lq.interpret_reply(json.dumps({"upper_color": "모름", "upper": "상의", "lower_color": "모름", "lower": "바지",
                                           "consistent": True}, ensure_ascii=False), False)
        self.assertEqual((r["status"], r["name"]), ("unclear", ""))                          # 색을 하나도 못 읽음
        self.assertEqual(lq.interpret_reply("모델이 그냥 말로 답함", keep_mixed=False)["status"], "parse_error")
        self.assertEqual(lq.interpret_reply(json.dumps({"consistent": True}), keep_mixed=False)["status"], "unclear")

    def test_make_record_fields(self):
        reply = lq.interpret_reply(json.dumps({"upper": "노란색 반팔 티셔츠", "lower": "검은색 긴바지", "items": "가방",
                                               "consistent": True}, ensure_ascii=False), False)
        rec = lq.make_record("leiden:person:abcdef0123456789", 3, 42, reply, "raw text", ["p1", "p2"], "montage/x.jpg", "m", 0.5)
        self.assertEqual((rec["cluster_name"], rec["display_name"], rec["status"], rec["label_confidence"]),
                         ("노란 반팔에 검은 바지", "노란 반팔에 검은 바지", "labeled", 1.0))
        self.assertEqual(rec["cluster_description"], "upper=노란색 반팔 티셔츠, lower=검은색 긴바지, items=가방, consistent=True")
        self.assertEqual((rec["qwen"]["raw"], rec["prompt_version"], rec["name_rule"]), ("raw text", lq.PROMPT_VERSION, lq.NAME_RULE))
        self.assertEqual(rec["representative_point_ids"], ["p1", "p2"])
        bad = lq.make_record("c", 1, 2, dict(status="parse_error", name="", upper="", lower="", items="", consistent=None), "?", [], None, "m", 0.1)
        self.assertEqual((bad["cluster_name"], bad["display_name"], bad["label_confidence"]), ("", "응답 해석 실패", 0.0))
        mixed = lq.make_record("c", 1, 2, lq.interpret_reply(json.dumps({"upper": "검은 상의", "consistent": False}, ensure_ascii=False), False),
                               "?", [], None, "m", 0.1)
        self.assertEqual((mixed["cluster_name"], mixed["display_name"], mixed["label_confidence"]), ("", "혼합(불일치)", 0.0))

    def test_rename_records(self):
        """--rename: 저장된 원문으로 이름·상태만 다시 짓는다 (GPU 불필요). 원문 없는 기록은 그대로."""
        raw = json.dumps({"upper": "검은색 상의", "lower": "검은색 바지", "items": "노란색 가방", "consistent": True,
                          "name": "노란 반팔에 검은 바지"}, ensure_ascii=False)
        old = [dict(cluster_id="dbscan:person:7e01ea13", rank=6, cluster_size=118, cluster_name="노란 반팔에 검은 바지",
                    status="labeled", qwen=dict(raw=raw), montage="montage/c0006.jpg", representative_point_ids=["a"], model_id="m"),
               dict(cluster_id="dbscan:person:0000", rank=7, cluster_size=3, cluster_name="", status="no_images", qwen={}),
               dict(cluster_id="leiden:object:1111", rank=1, cluster_size=5, cluster_name="x", status="labeled",
                    qwen=dict(raw=json.dumps({"category": "승용차", "color": "흰색", "consistent": True}, ensure_ascii=False)))]
        new, changes = lq.rename_records(old)
        self.assertEqual([r["cluster_name"] for r in new], ["검은 상의에 검은 바지", "", "흰 승용차"])
        self.assertEqual((new[0]["rank"], new[0]["cluster_size"], new[0]["montage"], new[0]["prompt_version"]), (6, 118, "montage/c0006.jpg", 1))
        self.assertIs(new[1], old[1])
        self.assertEqual(changes["renamed"], 2)
        self.assertEqual(lq.target_of_record(old[2]), "object")


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
        self.assertFalse(args.rename)
        self.assertTrue(lq.parse_args(["--rename"]).rename)


class ObjectTests(unittest.TestCase):
    """물건 군집: 종류·색·특징을 묻고(번호판·글자·사람 제외), 응답을 upper/lower/items 자리에 둔다."""

    def test_object_prompt(self):
        ko = lq.build_prompt(6, "ko", "object")
        self.assertIn('"category"', ko)
        self.assertNotIn("예:", ko)                                                    # 예시 값은 두지 않는다 (v1 에서 베낌)
        self.assertIn("번호판", ko)
        self.assertNotIn('"upper"', ko)
        self.assertIn('"category"', lq.build_prompt(3, "en", "object"))
        self.assertIn('"upper"', lq.build_prompt(6, "ko"))                        # 사람은 그대로

    def test_object_reply(self):
        raw = '{"category": "승용차", "color": "흰색", "details": "없음", "consistent": true, "name": "흰색 승용차"}'
        r = lq.interpret_reply(raw, False, "object")
        self.assertEqual((r["status"], r["name"], r["upper"], r["lower"], r["items"]), ("labeled", "흰 승용차", "승용차", "흰색", "없음"))
        r2 = lq.interpret_reply('{"category": "자전거", "color": "검은색", "consistent": true}', False, "object")
        self.assertEqual(r2["name"], "검은 자전거")                                   # 이름 = 색 + 종류 (항목으로 짓는다)
        r3 = lq.interpret_reply('{"category": "흰색 트럭", "color": "모름", "consistent": true}', False, "object")
        self.assertEqual(r3["name"], "흰 트럭")                                       # 종류 문장 앞의 색도 읽는다
        rec = lq.make_record("leiden:object:1", 1, 5, r, raw, ["a"], None, "m", 0.1, "object")
        self.assertTrue(rec["cluster_description"].startswith("category=승용차, color=흰색, details=없음"))
        person = lq.make_record("leiden:person:1", 1, 5, lq.interpret_reply(
            '{"upper": "노란 반팔", "lower": "검은 바지", "items": "없음", "consistent": true, "name": "노란 반팔"}', False),
            "", ["a"], None, "m", 0.1)
        self.assertTrue(person["cluster_description"].startswith("upper=노란 반팔, lower=검은 바지"))


if __name__ == "__main__":
    unittest.main()
