"""
test_negative_search.py — negative 쿼리 감산 정확도 테스트
실행: python src/RF-DETR/test_negative_search.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from forensic_video_search import ForensicSearcher

s = ForensicSearcher()

print("\n" + "═"*65)
print("  일반 검색 vs negative 감산 비교 (빨간 상의)")
print("═"*65)

# 일반 검색
r_normal = s.search("The person is wearing a red short-sleeved t-shirt", top_k=5)
print("\n[일반 검색] red short-sleeved t-shirt")
for x in r_normal:
    print(f"  {x['rank']}위 [{x['similarity']:.4f}]  {x['video']}/{x['track']}")
    print(f"       → {x.get('best_path','')}")

# negative 감산
r_neg = s.search_with_negative(
    positive = "The person is wearing a red short-sleeved t-shirt",
    negative = "The person is wearing a red jacket or long-sleeved coat",
    alpha    = 0.5,
    top_k    = 5,
)
print("\n[negative 감산 α=0.5] pos=티셔츠  neg=재킷/코트")
for x in r_neg:
    print(f"  {x['rank']}위 [{x['similarity']:.4f}]  pos={x['pos_score']:.4f}  neg={x['neg_score']:.4f}  {x['video']}/{x['track']}")
    print(f"       → {x.get('best_path','')}")

# alpha 강하게
r_neg2 = s.search_with_negative(
    positive = "The person is wearing a red short-sleeved t-shirt",
    negative = "The person is wearing a red jacket or long-sleeved coat",
    alpha    = 0.8,
    top_k    = 5,
)
print("\n[negative 감산 α=0.8]")
for x in r_neg2:
    print(f"  {x['rank']}위 [{x['similarity']:.4f}]  pos={x['pos_score']:.4f}  neg={x['neg_score']:.4f}  {x['video']}/{x['track']}")
    print(f"       → {x.get('best_path','')}")
