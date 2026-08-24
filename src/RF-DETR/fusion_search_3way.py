"""
마일스톤 11: Person(Re-ID) + Face + Semantic 3-way Fusion 검색
- M8(2-way)을 확장해서 semantic 신호까지 종합
- 이미지 Query뿐 아니라, 텍스트 Query도 지원 (SigLIP2가 텍스트/이미지를 같은 벡터공간에 놓기 때문)

사용법: python src/RF-DETR/fusion_search_3way.py
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.append("src/RF-DETR")

from reid_clip import load_reid_model, get_embedding
from face_insight import load_face_model, detect_faces
from siglip_semantic import load_semantic_model, get_image_embedding, get_text_embedding
from search_qdrant import (
    load_qdrant_client,
    get_reid_embedding,
    get_face_embedding,
    search_reid,
    search_face,
    COLLECTION_NAME,
)

# ── Fusion 가중치 (query가 아니라 환경설정이라 상수로 둠) ──
# Person(몸)이 가장 안정적인 주 신호, Face는 보완 신호, Semantic은 장면 맥락 보조 신호
REID_WEIGHT = 0.5
FACE_WEIGHT = 0.3
SEMANTIC_WEIGHT = 0.2

TOP_K_PER_SIGNAL = 50


def rank_score(rank, total=TOP_K_PER_SIGNAL):
    return (total - rank + 1) / total


def cosine_similarity(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search_semantic(client, query_embedding, top_k=TOP_K_PER_SIGNAL):
    """
    semantic(SigLIP2) Named Vector 기준 검색.
    """
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        using="semantic",
        limit=top_k,
        with_payload=True,
    )
    return results.points


def get_actual_similarity(client, point_id, query_embedding, vector_name):
    """
    Top-K 검색에 없더라도, 해당 point_id의 실제 벡터를 가져와서
    query와의 진짜 코사인 유사도를 계산 (M8에서 검증했던 방식과 동일).
    """
    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_vectors=True,
    )
    if not points:
        return None

    vectors = points[0].vector
    if not isinstance(vectors, dict) or vector_name not in vectors:
        return None  # 이 신호 자체가 없는 포인트 (예: 얼굴 미검출)

    return cosine_similarity(query_embedding, vectors[vector_name])


def fusion_search_3way(client, reid_embedding, face_embedding, semantic_embedding, top_k=5):
    """
    Person + Face + Semantic 세 신호를 순위 기반으로 합쳐서, 최종 Top-K 반환.
    reid_embedding: 필수
    face_embedding: 없으면(None) Face 신호는 건너뜀
    semantic_embedding: 필수 (이미지 Query든 텍스트 Query든 항상 있음)
    """
    # ── 1. 각 신호에서 후보 검색 ──
    reid_results = search_reid(client, reid_embedding, top_k=TOP_K_PER_SIGNAL)

    face_results = []
    if face_embedding is not None:
        face_results = search_face(client, face_embedding, top_k=TOP_K_PER_SIGNAL)

    semantic_results = search_semantic(client, semantic_embedding, top_k=TOP_K_PER_SIGNAL)

    # ── 2. Fusion 점수 누적 ──
    fusion_scores = defaultdict(float)
    point_info = {}
    all_candidate_ids = set()

    def add_signal(results, weight, key_name):
        topk_ids = set()
        for rank, point in enumerate(results, start=1):
            topk_ids.add(point.id)
            all_candidate_ids.add(point.id)
            fusion_scores[point.id] += rank_score(rank) * weight
            if point.id not in point_info:
                point_info[point.id] = {"payload": point.payload}
            point_info[point.id][key_name] = point.score
        return topk_ids

    reid_topk_ids = add_signal(reid_results, REID_WEIGHT, "reid_similarity")
    face_topk_ids = add_signal(face_results, FACE_WEIGHT, "face_similarity")
    semantic_topk_ids = add_signal(semantic_results, SEMANTIC_WEIGHT, "semantic_similarity")

    # ── 3. Re-ID 후보인데 Face/Semantic Top-K 밖인 경우, 실제 유사도 직접 계산 (M8 방식) ──
    for point_id in reid_topk_ids:
        if face_embedding is not None and point_id not in face_topk_ids:
            actual = get_actual_similarity(client, point_id, face_embedding, "face")
            if actual is not None:
                fusion_scores[point_id] += actual * FACE_WEIGHT
                point_info[point_id]["face_similarity"] = actual
                point_info[point_id]["face_direct"] = True

        if point_id not in semantic_topk_ids:
            actual = get_actual_similarity(client, point_id, semantic_embedding, "semantic")
            if actual is not None:
                fusion_scores[point_id] += actual * SEMANTIC_WEIGHT
                point_info[point_id]["semantic_similarity"] = actual
                point_info[point_id]["semantic_direct"] = True

    # ── 4. 정렬해서 최종 Top-K ──
    sorted_ids = sorted(fusion_scores.keys(), key=lambda pid: fusion_scores[pid], reverse=True)

    results = []
    for rank, point_id in enumerate(sorted_ids[:top_k], start=1):
        info = point_info[point_id]

        def fmt(key, direct_key):
            val = info.get(key)
            if val is None:
                return "N/A"
            note = "(직접계산)" if info.get(direct_key) else ""
            return f"{round(val, 4)}{note}"

        results.append({
            "rank": rank,
            "fusion_score": round(fusion_scores[point_id], 4),
            "reid_similarity": fmt("reid_similarity", "reid_direct"),
            "face_similarity": fmt("face_similarity", "face_direct"),
            "semantic_similarity": fmt("semantic_similarity", "semantic_direct"),
            "original_image": info["payload"].get("original_image"),
            "crop_path": info["payload"].get("crop_path"),
        })

    return results


def print_fusion_results(results, title="3-way Fusion 검색 결과 (Person+Face+Semantic)"):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    for r in results:
        print()
        print(f"[순위 {r['rank']}] Fusion 점수: {r['fusion_score']}")
        print(f"  원본 사진        : {r['original_image']}")
        print(f"  Re-ID 유사도     : {r['reid_similarity']}")
        print(f"  Face 유사도      : {r['face_similarity']}")
        print(f"  Semantic 유사도  : {r['semantic_similarity']}")


def text_search_semantic_only(client, text_query_embedding, top_k=5):
    """
    텍스트 Query 전용 검색. Re-ID/Face는 텍스트를 다룰 수 없으므로
    Semantic(SigLIP2) 신호 하나만으로 검색.
    """
    results = search_semantic(client, text_query_embedding, top_k=top_k)

    formatted = []
    for rank, point in enumerate(results, start=1):
        formatted.append({
            "rank": rank,
            "semantic_similarity": round(point.score, 4),
            "original_image": point.payload.get("original_image"),
            "crop_path": point.payload.get("crop_path"),
        })

    return formatted


def print_text_search_results(results, query_text):
    print()
    print("=" * 70)
    print(f"텍스트 Query 검색 결과: \"{query_text}\" (Semantic 신호만 사용)")
    print("=" * 70)
    for r in results:
        print()
        print(f"[순위 {r['rank']}] Semantic 유사도: {r['semantic_similarity']}")
        print(f"  원본 사진: {r['original_image']}")
        print(f"  Crop 경로: {r['crop_path']}")


# ── 이 파일을 단독 실행했을 때는 테스트용으로 동작 ──
if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parents[2]
    TEST_QUERY_IMAGE = ROOT_DIR / "data" / "crops" / "George_W_Bush_0001_person_1.jpg"

    print("모델 로드 중...")
    reid_model, reid_device = load_reid_model()
    face_model = load_face_model()
    semantic_model, semantic_processor, semantic_device = load_semantic_model()
    client = load_qdrant_client()

    print("\nQuery embedding 추출 중 (이미지 Query)...")
    reid_embedding = get_reid_embedding(reid_model, reid_device, TEST_QUERY_IMAGE)
    face_embedding = get_face_embedding(face_model, TEST_QUERY_IMAGE)
    semantic_embedding = get_image_embedding(
        semantic_model, semantic_processor, semantic_device, str(TEST_QUERY_IMAGE)
    )

    results = fusion_search_3way(client, reid_embedding, face_embedding, semantic_embedding, top_k=5)
    print_fusion_results(results, title="이미지 Query 기반 3-way Fusion 결과")

    # ── 텍스트 Query 테스트 (Semantic 신호만 사용) ──
    print("\n\n" + "#" * 70)
    print("# 텍스트 Query 테스트 (Person/Face 없이 Semantic만)")
    print("#" * 70)

    TEST_TEXT_QUERIES = [
        "a man wearing a red tie",               # 직접 눈으로 확인한 사실 (근거 있음)
        "a blue background with a white star",    # 직접 눈으로 확인한 사실 (근거 있음)
        "a man wearing a suit and tie",            # 너무 일반적, 변별력 낮을 것으로 예상 (baseline 겸용)
        "a woman in a colorful dress",              # 대조군: 이 데이터셋(부시)엔 없어야 정상
        "a child playing outdoors",                  # 대조군: 역시 없어야 정상
    ]

    for text_query in TEST_TEXT_QUERIES:
        text_embedding = get_text_embedding(semantic_model, semantic_processor, semantic_device, text_query)
        text_results = text_search_semantic_only(client, text_embedding, top_k=5)
        print_text_search_results(text_results, text_query)

    # ── 텍스트 Query 테스트 ──
    # 텍스트는 Re-ID/Face가 다룰 수 없는 개념이라, Semantic(SigLIP2) 신호만으로 검색
    print("\n\nQuery embedding 추출 중 (텍스트 Query: 'a man in a suit')...")
    text_semantic_embedding = get_text_embedding(
        semantic_model, semantic_processor, semantic_device, "a man in a suit"
    )

    text_results = search_semantic(client, text_semantic_embedding, top_k=5)

    print()
    print("=" * 70)
    print("텍스트 Query 기반 Semantic 검색 결과 ('a man in a suit')")
    print("=" * 70)
    for rank, point in enumerate(text_results, start=1):
        print(f"\n[순위 {rank}] Semantic 유사도: {point.score:.4f}")
        print(f"  원본 사진: {point.payload.get('original_image')}")
        print(f"  Crop 경로: {point.payload.get('crop_path')}")