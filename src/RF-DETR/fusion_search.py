"""
마일스톤 8: Person(Re-ID) + Face 검색 결과를 하나로 합치는 Fusion
- Rank-based Weighted Fusion: 각 신호에서 몇 등이었는지를 점수화해서 합산
  (Re-ID와 Face는 유사도 값의 분포 범위가 서로 달라서, 점수를 그냥 더하면
   한쪽으로 왜곡될 수 있음 -> "순위"를 기준으로 합치는 게 더 공정함)

사용법: python src/RF-DETR/fusion_search.py
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.append("src/RF-DETR")

from reid_clip import load_reid_model, get_embedding
from face_insight import load_face_model, detect_faces
from search_qdrant import (
    load_qdrant_client,
    get_reid_embedding,
    get_face_embedding,
    search_reid,
    search_face,
    COLLECTION_NAME, # COLLECTION_NAME이라는 변수 자체는 search_qdrant.py 안에 있고 여기로 자동으로 넘어오지 않습니다. 새
)

# ── Fusion 가중치 (query가 아니라 환경설정이라 상수로 둠) ──
# Re-ID(몸 전체)가 항상 존재하는 신호라 기본 가중치를 좀 더 둠,
# Face는 보완적 신호(얼굴 안 보이면 아예 없을 수 있음)라 상대적으로 낮게.
REID_WEIGHT = 0.6
FACE_WEIGHT = 0.4

TOP_K_PER_SIGNAL = 50  # 20 -> 50으로 상향: Re-ID에서 상위권이어도 Face에서 밀려나
                        # 반영이 안 됐던 문제(0174/0003/0387/0071)를 완화하기 위함


def rank_score(rank, total=TOP_K_PER_SIGNAL):
    """
    순위를 0~1 사이 점수로 변환. 1등 = 1.0, 꼴등에 가까울수록 0에 가까워짐.
    """
    return (total - rank + 1) / total


import numpy as np


def cosine_similarity(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_actual_face_similarity(client, point_id, query_face_embedding):
    """
    Top-K 검색 결과에 없더라도, 이 point_id가 실제로 가진 face 벡터를
    직접 가져와서 query와의 진짜 코사인 유사도를 계산.
    (face 벡터 자체가 없는 포인트면 None 반환 -> 이건 진짜 "정보 없음")
    """
    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_vectors=True,
    )

    if not points:
        return None

    vectors = points[0].vector
    if not isinstance(vectors, dict) or "face" not in vectors:
        return None  # 이 사람은 애초에 얼굴이 검출 안 됐던 경우 (진짜 정보 없음)

    return cosine_similarity(query_face_embedding, vectors["face"])


def fusion_search(client, reid_embedding, face_embedding, top_k=5):
    """
    Re-ID와 Face(있으면) 검색 결과를 순위 기반으로 합쳐서, 최종 Top-K 반환.

    ⚠️ 설계 원칙: Re-ID Top-K 후보인데 Face Top-K 밖으로 밀린 사람은,
    "0으로 근사"하지 않고 실제 Face 벡터를 직접 조회해서 정확한 유사도를 계산함.
    (얼굴 자체가 없는 경우만 진짜 "정보 없음"으로 처리)
    """
    # ── 1. 각 신호에서 넉넉하게 후보 검색 ──
    reid_results = search_reid(client, reid_embedding, top_k=TOP_K_PER_SIGNAL)

    face_results = []
    face_query_attempted = face_embedding is not None
    if face_embedding is not None:
        face_results = search_face(client, face_embedding, top_k=TOP_K_PER_SIGNAL)

    # ── 2. point.id를 key로, 각 신호의 순위 점수를 누적 ──
    fusion_scores = defaultdict(float)
    point_info = {}
    reid_candidate_ids = set()

    for rank, point in enumerate(reid_results, start=1):
        fusion_scores[point.id] += rank_score(rank) * REID_WEIGHT
        reid_candidate_ids.add(point.id)
        point_info[point.id] = {
            "payload": point.payload,
            "reid_similarity": point.score,
            "face_similarity": None,
            "face_in_topk": False,
        }

    face_topk_ids = set()
    for rank, point in enumerate(face_results, start=1):
        face_topk_ids.add(point.id)
        fusion_scores[point.id] += rank_score(rank) * FACE_WEIGHT
        if point.id in point_info:
            point_info[point.id]["face_similarity"] = point.score
            point_info[point.id]["face_in_topk"] = True
        else:
            point_info[point.id] = {
                "payload": point.payload,
                "reid_similarity": None,
                "face_similarity": point.score,
                "face_in_topk": True,
            }

    # ── 2-1. Re-ID 후보인데 Face Top-K 밖인 사람: "0 근사" 대신 실제 유사도를 직접 계산 ──
    if face_query_attempted:
        for point_id in reid_candidate_ids:
            if point_id not in face_topk_ids:
                actual_similarity = get_actual_face_similarity(client, point_id, face_embedding)

                if actual_similarity is not None:
                    # 진짜 유사도 값을 Fusion 점수에 정확하게 반영
                    # (rank_score 대신, 유사도 값 자체를 0~1 범위 점수로 사용)
                    fusion_scores[point_id] += actual_similarity * FACE_WEIGHT
                    point_info[point_id]["face_similarity"] = actual_similarity
                    point_info[point_id]["face_computed_directly"] = True
                # actual_similarity가 None이면 -> 이 사람은 얼굴 자체가 없는 것, 그대로 None 유지

    # ── 3. Fusion 점수 기준으로 정렬해서 최종 Top-K ──
    sorted_ids = sorted(fusion_scores.keys(), key=lambda pid: fusion_scores[pid], reverse=True)

    results = []
    for rank, point_id in enumerate(sorted_ids[:top_k], start=1):
        info = point_info[point_id]

        if info["face_similarity"] is not None:
            note = "(직접계산)" if info.get("face_computed_directly") else ""
            face_display = f"{round(info['face_similarity'], 4)}{note}"
        elif not face_query_attempted:
            face_display = "N/A(Query에 얼굴 없음)"
        else:
            face_display = "N/A(이 인물은 얼굴 미검출)"

        results.append({
            "rank": rank,
            "fusion_score": round(fusion_scores[point_id], 4),
            "reid_similarity": round(info["reid_similarity"], 4) if info["reid_similarity"] else None,
            "face_similarity": face_display,
            "original_image": info["payload"].get("original_image"),
            "crop_path": info["payload"].get("crop_path"),
        })

    return results


def print_fusion_results(results):
    print()
    print("=" * 70)
    print("Fusion 검색 결과 (Person + Face 통합)")
    print("=" * 70)

    for r in results:
        print()
        print(f"[순위 {r['rank']}] Fusion 점수: {r['fusion_score']}")
        print(f"  원본 사진    : {r['original_image']}")
        print(f"  Re-ID 유사도 : {r['reid_similarity']}")
        print(f"  Face 유사도  : {r['face_similarity']}")


# ── 이 파일을 단독 실행했을 때는 테스트용으로 동작 ──
if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parents[2]
    TEST_QUERY_IMAGE = ROOT_DIR / "data" / "crops" / "George_W_Bush_0001_person_1.jpg"

    print("모델 로드 중...")
    reid_model, reid_device = load_reid_model()
    face_model = load_face_model()
    client = load_qdrant_client()

    print("\nQuery embedding 추출 중...")
    reid_embedding = get_reid_embedding(reid_model, reid_device, TEST_QUERY_IMAGE)
    face_embedding = get_face_embedding(face_model, TEST_QUERY_IMAGE)

    results = fusion_search(client, reid_embedding, face_embedding, top_k=5)
    print_fusion_results(results)