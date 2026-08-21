"""
마일스톤 6: 전체 파이프라인 연결 (RF-DETR -> CLIP-ReID -> Qdrant) + JSON 출력
사용법: python src/main.py --query 사진경로.jpg

전체 사진(배경 포함, 사람 위치 모름)을 넣으면:
1. RF-DETR로 사람 검출+crop
2. CLIP-ReID로 embedding 추출
3. Qdrant에서 유사 인물 검색
4. 팀 합의 JSON 스키마로 결과 저장
"""

import sys
import os
import time
import json
import argparse

sys.path.append("src/RF-DETR")
from detect_rf import load_detect_model, detect_and_crop
from reid_clip import load_reid_model, get_embedding
from search_qdrant import load_qdrant_client, search

RESULTS_OUTPUT_PATH = "data/results.json"


def build_result_json(query_type, query_value, results, search_time_ms, dataset_name="George_W_Bush_sample"):
    """
    팀에서 합의한 JSON 스키마에 맞춰 결과를 구성.
    지금은 이미지 검색만 하므로 video 관련 필드는 0으로 채워둠
    (나중에 영상/텍스트 기능 추가되면 그 값들만 채우면 됨).
    """
    formatted_results = []

    for r in results:
        rank = r["rank"]
        rank_score = round(1.0 / rank, 4)

        formatted_results.append({
            "source_type": f"{query_type}_image",
            "dataset": dataset_name,
            "split": "test",
            "source": os.path.abspath(r["crop_path"]),
            "filename": os.path.basename(r["crop_path"]),
            "image": os.path.basename(r["crop_path"]),
            "embedding_id": r.get("embedding_id", -1),
            "query": query_value,
            "similarity": r["similarity"],
            "vector_id": r.get("embedding_id", -1),
            "rank": rank,
            "source_rank": rank,
            "rank_score": rank_score,
            "result_type": "image",
            "raw_score": r["similarity"],
            "fusion_score": r["similarity"],  # 지금은 단일 신호(person), fusion 없음
            "unified_rank": rank,
        })

    return {
        "query_type": query_type,
        "query": query_value,
        "fusion_method": None,
        "image_weight": 1.0,
        "video_weight": 0.0,
        "search_time_ms": round(search_time_ms, 3),
        "image_result_count": len(formatted_results),
        "video_result_count": 0,
        "video_segment_count": 0,
        "results": formatted_results,
    }


def full_pipeline(query_image_path, top_k=10):
    """
    이미지 1장을 받아서 RF-DETR -> Crop -> CLIP-ReID -> Qdrant 검색까지 전부 실행.
    """
    start_time = time.time()

    # ── 1. 모델 로드 ──
    detect_model = load_detect_model()
    reid_model, device = load_reid_model()

    # ── 2. Query 사진에서 사람 검출 + crop ──
    crops = detect_and_crop(detect_model, query_image_path, output_dir="data/query_crops")
    if not crops:
        print("Query 사진에서 사람을 찾지 못했습니다.")
        return None

    query_crop_path = crops[0]  # 여러 명 있으면 일단 첫 번째 사람 기준

    # ── 3. embedding 추출 ──
    query_embedding = get_embedding(reid_model, device, query_crop_path)

    # ── 4. Qdrant로 유사 검색 ──
    client = load_qdrant_client()
    raw_results = search(client, query_embedding, top_k=top_k)

    search_time_ms = (time.time() - start_time) * 1000

    # ── 5. 합의된 JSON 스키마로 결과 구성 ──
    result_json = build_result_json(
        query_type="image",
        query_value=os.path.basename(query_image_path),
        results=raw_results,
        search_time_ms=search_time_ms,
    )

    return result_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="이미지 기반 인물 검색 (RF-DETR+CLIP-ReID+Qdrant)")
    parser.add_argument("--query", required=True, help="검색할 사진 경로")
    parser.add_argument("--top_k", type=int, default=10, help="검색 결과 개수 (기본 10)")
    args = parser.parse_args()

    result_json = full_pipeline(args.query, top_k=args.top_k)

    if result_json:
        os.makedirs(os.path.dirname(RESULTS_OUTPUT_PATH), exist_ok=True)
        with open(RESULTS_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)

        print(f"\n검색 완료: {result_json['image_result_count']}개 결과")
        print(f"소요 시간: {result_json['search_time_ms']}ms")
        print(f"결과 저장 위치: {RESULTS_OUTPUT_PATH}")