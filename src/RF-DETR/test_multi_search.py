"""
test_multi_search.py — 단일 쿼리 vs 개선된 쿼리 vs multi-query 비교 테스트
실행: cd TransReID && python src/RF-DETR/test_multi_search.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from forensic_video_search import ForensicSearcher

searcher = ForensicSearcher()

# ── 1. 단일 쿼리 스타일 비교 ─────────────────────────────────────────
style_tests = [
    ("한국어 단문",  "빨간 재킷을 입은 남성"),
    ("영어 단문",    "red jacket man"),
    ("영어 중간",    "a man wearing a red jacket and black pants"),
    ("CUHK 스타일",  "The person is wearing a red long-sleeved jacket and black trousers."),
    ("CUHK 상세",    "The man is dressed in a red upper garment and dark trousers, and appears to be carrying a bag."),
]

print("\n" + "═"*65)
print("  ① 단일 쿼리 스타일별 비교 (1위 결과)")
print("═"*65)
for label, q in style_tests:
    r = searcher.search(q, top_k=3, by_track=True)
    if r:
        top = r[0]
        print(f"\n  [{label}]")
        print(f"  쿼리  : {q}")
        print(f"  1위   : [{top['similarity']:.4f}]  {top['video']} / {top['track']}")
        print(f"  경로  : {top.get('best_path','')}")
    else:
        print(f"\n  [{label}] → 결과 없음")

# ── 2. multi-query 테스트 ────────────────────────────────────────────
print("\n" + "═"*65)
print("  ② Multi-Query 평균 (빨간 재킷 남성)")
print("═"*65)
red_queries = [
    "The person is wearing a red jacket and black pants.",
    "A male in a red coat with dark trousers.",
    "Red upper clothing with dark lower clothing, male.",
]
results = searcher.search_multi(red_queries, top_k=5)
searcher.print_results(results, "MULTI ← 빨간 재킷 남성")

# ── 3. 다른 의상으로도 테스트 ────────────────────────────────────────
print("\n" + "═"*65)
print("  ③ Multi-Query (흰 셔츠 남성)")
print("═"*65)
white_queries = [
    "The person is wearing a white shirt and dark pants.",
    "A male dressed in a white top and black or dark trousers.",
    "White upper garment dark lower garment male person.",
]
results2 = searcher.search_multi(white_queries, top_k=5)
searcher.print_results(results2, "MULTI ← 흰 셔츠 남성")
