"""
마일스톤 5: FAISS로 embedding DB에서 검색
- load_search_index(): 저장해둔 embeddings.npy를 FAISS 인덱스로 올림
- search(index, metadata, query_embedding, top_k): 가장 비슷한 Top-K 찾기
"""

import json
import numpy as np
import faiss

EMBEDDINGS_PATH = "data/embeddings.npy"
METADATA_PATH = "data/metadata.json"


def load_search_index():
    """
    build_database.py가 만들어둔 embeddings.npy를 읽어서 FAISS 인덱스로 올림.
    IndexFlatIP = 정확 검색(브루트포스), 지금 규모(668개)에선 충분히 빠름.

    반환값: (index, metadata) — 인덱스와, 각 벡터가 어느 사진인지 알려주는 정보
    """
    embeddings = np.load(EMBEDDINGS_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    dimension = embeddings.shape[1]  # 3840

    # IndexFlatIP는 내적(inner product) 기반 유사도 검색.
    # TransReID embedding은 미리 정규화가 안 되어 있을 수 있어서, cosine similarity와
    # 동일하게 만들기 위해 벡터 길이를 1로 맞춰주는 정규화를 먼저 함.
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"FAISS 인덱스 생성 완료: {index.ntotal}개 벡터 등록됨")

    return index, metadata


def search(index, metadata, query_embedding, top_k=3):
    """
    query_embedding(하나의 사람 embedding)과 가장 비슷한 Top-K를 찾아서 반환.

    반환값: [{"crop_path": ..., "original_photo": ..., "similarity": ...}, ...]
    """
    query_embedding = query_embedding.astype("float32").reshape(1, -1)
    faiss.normalize_L2(query_embedding)  # DB와 동일하게 정규화

    similarities, indices = index.search(query_embedding, top_k)

    results = []
    for rank in range(top_k):
        idx = indices[0][rank]
        sim = float(similarities[0][rank])
        info = metadata[idx]

        results.append({
            "rank": rank + 1,
            "crop_path": info["crop_path"],
            "original_photo": info["original_photo"],
            "similarity": round(sim, 4),
        })

    return results


# ── 이 파일을 단독 실행했을 때는 테스트용으로 동작 ──
if __name__ == "__main__":
    import sys
    sys.path.append("src")  # detect.py, reid.py를 가져오기 위함
    from reid import load_reid_model, get_embedding

    # 테스트: DB 안에 있는 사진 중 하나를 Query로 써서, 자기 자신이 1등으로 나오는지 확인
    # 초기에는 이걸로 테스트 했었음
    # TEST_QUERY_IMAGE = "data/crops/George_W_Bush_0083_detected_full.jpg"  # 본인 환경에 맞게 수정 가능
    # 쿼리의 경로를 실제 CROP 파일로 변경해서 실행
    TEST_QUERY_IMAGE = "data/crops/George_W_Bush_0082_person_2.jpg"
    index, metadata = load_search_index()

    reid_model, device = load_reid_model()
    query_embedding = get_embedding(reid_model, device, TEST_QUERY_IMAGE)

    results = search(index, metadata, query_embedding, top_k=3)

    print(f"\nQuery: {TEST_QUERY_IMAGE}")
    print("검색 결과 Top-3:")
    for r in results:
        print(f"  {r['rank']}위: {r['crop_path']} (원본: {r['original_photo']}, 유사도: {r['similarity']})")
