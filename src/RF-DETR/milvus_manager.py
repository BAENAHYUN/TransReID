"""
milvus_manager.py — Milvus 벡터 인덱스 인터페이스 (MilvusClient API)

FAISS IndexFlatIP 의 드롭인 대체.
pymilvus 2.4+ MilvusClient API 사용 (Deprecation 경고 없음).

설치:
  pip install pymilvus

환경변수:
  FORENSIC_MILVUS_URI = "http://localhost:19530"   (기본값)
"""

from __future__ import annotations

import os
import numpy as np
from typing import Tuple

from pymilvus import MilvusClient, DataType

# ── 연결 설정 ─────────────────────────────────────────────────────────
_URI = os.environ.get(
    "FORENSIC_MILVUS_URI",
    f"http://{os.environ.get('FORENSIC_MILVUS_HOST','localhost')}:{os.environ.get('FORENSIC_MILVUS_PORT','19530')}",
)

COLLECTION_NAME = "forensic_vectors"
DIM             = 512
NLIST           = 1024
NPROBE          = 64


# ═══════════════════════════════════════════════════════════════════════
# MilvusManager
# ═══════════════════════════════════════════════════════════════════════

class MilvusManager:
    """
    Milvus 컬렉션 관리 + 검색.

    FAISS 호환 인터페이스:
        sims, ids = milvus.search(query_vector, top_k)

    고유 기능:
        milvus.insert(ids, vectors, ...)
        milvus.ntotal
        milvus.search_by_track(q, k, threshold)
    """

    def __init__(
        self,
        uri:             str  = _URI,
        collection_name: str  = COLLECTION_NAME,
        auto_create:     bool = True,
    ):
        self._collection_name = collection_name
        print(f"[*] Milvus 연결: {uri}")
        self._client = MilvusClient(uri=uri)
        print("[+] Milvus 연결 성공")

        if auto_create:
            self._ensure_collection()

    # ── 컬렉션 생성/로드 ─────────────────────────────────────────────
    def _ensure_collection(self):
        if self._client.has_collection(self._collection_name):
            print(f"[*] 기존 컬렉션 로드: {self._collection_name}")
            self._client.load_collection(self._collection_name)
        else:
            print(f"[*] 새 컬렉션 생성: {self._collection_name}")
            self._create_collection()
        n = self._client.get_collection_stats(self._collection_name)["row_count"]
        print(f"[+] 컬렉션 준비: {int(n):,} 벡터")

    def _create_collection(self):
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("milvus_id", DataType.INT64,    is_primary=True)
        schema.add_field("vector",    DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("video",     DataType.VARCHAR,   max_length=200)
        schema.add_field("track",     DataType.VARCHAR,   max_length=100)
        schema.add_field("source",    DataType.VARCHAR,   max_length=20)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name   = "vector",
            metric_type  = "IP",
            index_type   = "IVF_FLAT",
            index_name   = "vector_index",
            params       = {"nlist": NLIST},
        )

        self._client.create_collection(
            collection_name = self._collection_name,
            schema          = schema,
            index_params    = index_params,
        )
        self._client.load_collection(self._collection_name)

    # ── 프로퍼티 ─────────────────────────────────────────────────────
    @property
    def ntotal(self) -> int:
        return int(self._client.get_collection_stats(self._collection_name)["row_count"])

    # ── FAISS 호환 search ────────────────────────────────────────────
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        FAISS IndexFlatIP.search() 와 동일한 반환 형식.

        Returns:
            similarities: np.ndarray shape (1, top_k)
            indices:      np.ndarray shape (1, top_k)  ← milvus_id
        """
        top_k = min(top_k, self.ntotal)
        results = self._client.search(
            collection_name = self._collection_name,
            data            = query_vector.tolist(),
            anns_field      = "vector",
            search_params   = {"metric_type": "IP", "params": {"nprobe": NPROBE}},
            limit           = top_k,
            output_fields   = ["video", "track", "source"],
        )

        ids  = np.full((1, top_k), -1, dtype=np.int64)
        sims = np.zeros((1, top_k), dtype=np.float32)

        for i, hit in enumerate(results[0]):
            if i >= top_k:
                break
            ids[0, i]  = hit["id"]
            sims[0, i] = hit["distance"]

        return sims, ids

    # ── 삽입 ─────────────────────────────────────────────────────────
    def insert(
        self,
        milvus_ids: list[int],
        vectors:    np.ndarray,
        videos:     list[str],
        tracks:     list[str],
        sources:    list[str],
        batch_size: int = 5000,
    ) -> int:
        total    = len(milvus_ids)
        inserted = 0

        for start in range(0, total, batch_size):
            end   = min(start + batch_size, total)
            data  = [
                {
                    "milvus_id": milvus_ids[i],
                    "vector":    vectors[i].tolist(),
                    "video":     videos[i],
                    "track":     tracks[i],
                    "source":    sources[i],
                }
                for i in range(start, end)
            ]
            self._client.insert(collection_name=self._collection_name, data=data)
            inserted += (end - start)
            print(f"  Milvus 삽입: {inserted:,}/{total:,} ({inserted/total*100:.1f}%)", end="\r")

        self._client.flush(self._collection_name)
        print(f"\n[+] Milvus 삽입 완료: {inserted:,}개")
        return inserted

    # ── 트랙 단위 집계 검색 ─────────────────────────────────────────
    def search_by_track(
        self,
        query_vector:   np.ndarray,
        top_k:          int   = 200,
        irra_threshold: float = 0.42,
    ) -> list[dict]:
        """트랙 최고 점수 기준으로 집계된 결과 반환."""
        search_k = min(top_k * 20, self.ntotal)
        results  = self._client.search(
            collection_name = self._collection_name,
            data            = query_vector.tolist(),
            anns_field      = "vector",
            search_params   = {"metric_type": "IP", "params": {"nprobe": NPROBE}},
            limit           = search_k,
            output_fields   = ["video", "track", "source"],
        )

        track_best: dict[str, dict] = {}
        for hit in results[0]:
            score = hit["distance"]
            if score < irra_threshold:
                continue
            vid = hit["entity"].get("video", "")
            trk = hit["entity"].get("track", "")
            src = hit["entity"].get("source", "")
            key = f"{vid}/{trk}"
            if key not in track_best or score > track_best[key]["similarity"]:
                track_best[key] = {
                    "milvus_id":  hit["id"],
                    "video":      vid,
                    "track":      trk,
                    "source":     src,
                    "similarity": round(float(score), 4),
                }

        ranked = sorted(track_best.values(), key=lambda x: x["similarity"], reverse=True)
        return [{"rank": i + 1, **item} for i, item in enumerate(ranked[:top_k])]

    # ── 정리 ─────────────────────────────────────────────────────────
    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ═══════════════════════════════════════════════════════════════════════
# 연결 테스트 CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"[*] Milvus 연결 테스트: {_URI}")
    try:
        mgr = MilvusManager()
        print(f"[+] 컬렉션 '{COLLECTION_NAME}': {mgr.ntotal:,} 벡터")
        mgr.close()
    except Exception as e:
        print(f"[!] 연결 실패: {e}")
        print("    docker-compose up -d 로 Milvus 실행 후 재시도")
