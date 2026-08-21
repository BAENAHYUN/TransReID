"""
브루트포스(IndexFlatIP) 검색이 대규모(2만~10만)에서도 충분히 빠른지
지금 GPU(GTX 1060)로 직접 검증하는 스크립트.

지금 있는 668개 embedding을 복제해서 인위적으로 규모를 늘린 뒤,
검색 속도를 실측합니다 (A팀이 Market-1501로 실측했던 것과 같은 방식).
"""

import time
import numpy as np
import faiss

embeddings = np.load("data/embeddings.npy")
print(f"원본 규모: {embeddings.shape[0]}개")

for target_size in [20000, 100000]:
    # 지금 있는 668개를 복제해서 target_size만큼 늘림 (인위적 규모 확대)
    repeat_count = (target_size // embeddings.shape[0]) + 1
    big_embeddings = np.tile(embeddings, (repeat_count, 1))[:target_size].astype("float32")

    faiss.normalize_L2(big_embeddings)
    index = faiss.IndexFlatIP(big_embeddings.shape[1])
    index.add(big_embeddings)

    # Query 1개로 검색 속도 측정 (10번 반복해서 평균)
    query = big_embeddings[0:1].copy()

    start = time.time()
    for _ in range(10):
        index.search(query, 3)
    elapsed_ms = (time.time() - start) / 10 * 1000

    print(f"{target_size:>7,}개 규모 → 검색 1회 평균: {elapsed_ms:.2f} ms")