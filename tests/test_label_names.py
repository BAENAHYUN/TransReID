"""clustering/label_names — 라벨 문장의 색·옷 종류 읽기와 짧은 이름 짓기 (Qwen 이름 불일치 수정, 2026-09-29)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clustering import label_names as N  # noqa: E402


class ColorTests(unittest.TestCase):
    def test_color_of(self):
        self.assertEqual(N.color_of("노란색 반팔 티셔츠"), ("노", "노란"))
        self.assertEqual(N.color_of("검정 코트"), ("검", "검은"))
        self.assertEqual(N.color_of("청록색 반팔"), ("청록", "청록"))              # 청록 > 청
        self.assertEqual(N.color_of("청색 상의"), ("파", "파란"))
        self.assertEqual(N.color_of("beige색 바지"), ("베이지", "베이지"))           # 영어 뒤 한글
        self.assertEqual(N.color_of("purple 반팔 티셔츠"), ("보라", "보라"))
        self.assertIsNone(N.color_of("colored shirt"))                              # 'red' 가 단어 속에 있어도 아님
        self.assertIsNone(N.color_of("청바지"))                                      # 청바지의 '청' 은 색이 아님
        self.assertIsNone(N.color_of("모름"))
        self.assertIsNone(N.color_of(""))
        self.assertEqual(N.color_of("흰 셔츠와 검은 바지"), ("흰", "흰"))            # 가장 앞의 색

    def test_stems_and_vague(self):
        self.assertEqual(N.color_stems("white 가방, red 색의 신발"), {"흰", "빨"})
        self.assertEqual(N.color_stems("검은 반팔에 청바지"), {"검"})
        self.assertTrue(N.is_vague("색상이 다를 수 있는 상의"))
        self.assertTrue(N.is_vague("색과 종류가 다른 반팔 티셔츠"))
        self.assertTrue(N.is_vague("various colors"))
        self.assertFalse(N.is_vague("검은색 바지"))


class NameTests(unittest.TestCase):
    def test_person_name(self):
        self.assertEqual(N.person_name("노란색", "반팔 티셔츠", "검은색", "긴바지"), "노란 반팔에 검은 바지")
        self.assertEqual(N.person_name("", "노란색 반팔 티셔츠", "", "검은색 긴바지"), "노란 반팔에 검은 바지")   # v1 형식
        self.assertEqual(N.person_name("회색", "민소매", "베이지색", "반바지"), "회색 민소매에 베이지 반바지")
        self.assertEqual(N.person_name("모름", "상의", "검은색", "바지"), "검은 바지")
        self.assertEqual(N.person_name("", "청록색 반팔 티셔츠", "", "청바지"), "청록 반팔에 청바지")
        self.assertEqual(N.person_name("검은색", "반팔", "파란색", "청바지"), "검은 반팔에 청바지")
        self.assertEqual(N.person_name("흰색", "셔츠", "검은색", "청바지"), "흰 셔츠에 검은 청바지")
        self.assertEqual(N.person_name("", "purple 반팔 티셔츠", "", "black 짧은 바지"), "보라 반팔에 검은 반바지")
        self.assertEqual(N.person_name("빨간색", "원피스", "", "원피스"), "빨간 원피스")
        self.assertEqual(N.person_name("", "검은 정장", "", "검은 정장 바지"), "검은 정장에 검은 바지")
        self.assertEqual(N.person_name("모름", "", "모름", ""), "")
        # 색을 못 읽고 무늬만 말하면 무늬로 (구체적인 무늬가 먼저)
        self.assertEqual(N.person_name("", "색상이 어두운 톤의 상의, 패턴은 밝은 색상의 꽃무늬가", "", "검은색 바지"), "꽃무늬 상의에 검은 바지")
        self.assertEqual(N.color_of("흰색 줄무늬 셔츠"), ("흰", "흰"))                 # 색이 있으면 색
        self.assertLessEqual(len(N.person_name("노란색", "반팔 티셔츠", "검은색", "긴바지", max_chars=5)), 5)

    def test_object_name(self):
        self.assertEqual(N.object_name("흰색", "승용차"), "흰 승용차")
        self.assertEqual(N.object_name("", "흰색 승용차"), "흰 승용차")
        self.assertEqual(N.object_name("모름", "청소차"), "청소차")                   # 종류 안의 '청' 을 지우지 않음
        self.assertEqual(N.object_name("빨간색", "자동차"), "빨간 자동차")
        self.assertEqual(N.object_name("모름", ""), "물체")


if __name__ == "__main__":
    unittest.main()
