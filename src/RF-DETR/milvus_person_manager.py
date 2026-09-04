"""
milvus_person_manager.py — 인물 멀티벡터 Milvus 인터페이스

컬렉션: forensic_person_v2
벡터:
  siglip_vector  : FLOAT_VECTOR(768)   SigLIP2 semantic
  irra_vector    : FLOAT_VECTOR(512)   IRRA person_text
  solider_vector : FLOAT_VECTOR(1024)  SOLIDER person_reid (선택)

검색:
  search_text(irra_text_vec)           → IRRA 단독
  search_image(siglip_vec, irra_vec)   → SigLIP2 + IRRA 하이브리드
  search_full(siglip_vec, irra_vec, solider_vec) → 3-vector 하이브리드 (RRF)

환경변수:
  FORENSIC_MILVUS_URI  기본: http://localhost:19530
"""

from __future__ import annotations

import os
import numpy as np
from typing import List, Dict, Tuple, Optional

from pymilvus import (
    MilvusClient, DataType,
    AnnSearchRequest, RRFRanker, WeightedRanker,
)

_URI = os.environ.get(
    "FORENSIC_MILVUS_URI",
    f"http://{os.environ.get('FORENSIC_MILVUS_HOST','localhost')}:{os.environ.get('FORENSIC_MILVUS_PORT','19530')}",
)

COLLECTION_NAME = "forensic_person_v2"
DIM_SIGLIP      = 768
DIM_IRRA        = 512
DIM_SOLIDER     = 1024
NLIST           = 1024
NPROBE          = 64


class PersonMilvusManager:
    """
    인물 멀티벡터 Milvus 컬렉션 관리 + 검색.

    DB 구축:
        mgr.insert(ids, siglip_vecs, irra_vecs, videos, tracks, sources)
        mgr.insert_solider(ids, solider_vecs)   # pass2 에서 추가

    검색:
        mgr.search_text(irra_vec, top_k)
        mgr.search_image(siglip_vec, irra_vec, top_k)
        mgr.search_full(siglip_vec, irra_vec, solider_vec, top_k)
        mgr.search_by_track(query_vecs, mode, top_k, threshold)
    """

    def __init__(
        self,
        uri:             str  = _URI,
        collection_name: str  = COLLECTION_NAME,
        auto_create:     bool = True,
    ):
        self._col = collection_name
        print(f"[*] Milvus 연결: {uri}")
        self._client = MilvusClient(uri=uri)
        print("[+] Milvus 연결 성공")
        if auto_create:
            self._ensure_collection()

    # ── 컬렉션 생성/로드 ──────────────────────────────────────────
    def _ensure_collection(self):
        if self._client.has_collection(self._col):
            print(f"[*] 기존 컬렉션 로드: {self._col}")
            self._client.load_collection(self._col)
        else:
            print(f"[*] 새 컬렉션 생성: {self._col}")
            self._create_collection()
        n = self.ntotal
        print(f"[+] 컬렉션 준비: {n:,} 벡터")

    def _create_collection(self):
        schema = self._client.create_schema(
            auto_id=False, enable_dynamic_field=False
        )
        schema.add_field("milvus_id",      DataType.INT64,        is_primary=True)
        schema.add_field("siglip_vector",  DataType.FLOAT_VECTOR, dim=DIM_SIGLIP)
        schema.add_field("irra_vector",    DataType.FLOAT_VECTOR, dim=DIM_IRRA)
        schema.add_field("solider_vector", DataType.FLOAT_VECTOR, dim=DIM_SOLIDER)
        schema.add_field("video",          DataType.VARCHAR,       max_length=200)
        schema.add_field("track",          DataType.VARCHAR,       max_length=100)
        schema.add_field("source",         DataType.VARCHAR,       max_length=20)

        idx = self._client.prepare_index_params()
        for field, dim in [
            ("siglip_vector",  DIM_SIGLIP),
            ("irra_vector",    DIM_IRRA),
            ("solider_vector", DIM_SOLIDER),
        ]:
            idx.add_index(
                field_name  = field,
                metric_type = "IP",
                index_type  = "IVF_FLAT",
                params      = {"nlist": NLIST},
            )

        self._client.create_collection(
            collection_name = self._col,
            schema          = schema,
            index_params    = idx,
        )
        self._client.load_collection(self._col)

    # ── 프로퍼티 ──────────────────────────────────────────────────
    @property
    def ntotal(self) -> int:
        return int(self._client.get_collection_stats(self._col)["row_count"])

    # ── 삽입 (pass1: SigLIP2 + IRRA) ────────────────────────────
    def insert(
        self,
        milvus_ids:   list[int],
        siglip_vecs:  np.ndarray,          # (N, 768)
        irra_vecs:    np.ndarray,          # (N, 512)
        videos:       list[str],
        tracks:       list[str],
        sources:      list[str],
        batch_size:   int = 2000,
    ) -> int:
        # SOLIDER 없을 때 zero 벡터로 placeholder
        N = len(milvus_ids)
        solider_placeholder = np.zeros((N, DIM_SOLIDER), dtype=np.float32)

        total = inserted = 0
        total = N
        for s in range(0, total, batch_size):
            e = min(s + batch_size, total)
            data = [
                {
                    "milvus_id":      milvus_ids[i],
                    "siglip_vector":  siglip_vecs[i].tolist(),
                    "irra_vector":    irra_vecs[i].tolist(),
                    "solider_vector": solider_placeholder[i].tolist(),
                    "video":          videos[i],
                    "track":          tracks[i],
                    "source":         sources[i],
                }
                for i in range(s, e)
            ]
            self._client.insert(collection_name=self._col, data=data)
            inserted += (e - s)
            print(f"  삽입: {inserted:,}/{total:,}", end="\r")

        self._client.flush(self._col)
        print(f"\n[+] 삽입 완료: {inserted:,}개")
        return inserted

    # ── SOLIDER 벡터 추가 (pass2) ────────────────────────────────
    def insert_solider(
        self,
        milvus_ids:    list[int],
        solider_vecs:  np.ndarray,   # (N, 1024)
        batch_size:    int = 2000,
    ) -> int:
        """기존 레코드에 solider_vector 업서트."""
        total = len(milvus_ids)
        inserted = 0
        for s in range(0, total, batch_size):
            e = min(s + batch_size, total)
            # upsert: primary key가 같으면 업데이트
            # 단, MilvusClient.upsert는 모든 필드를 다시 써야 함 → get + upsert
            ids_batch = milvus_ids[s:e]
            existing = self._client.get(
                collection_name = self._col,
                ids             = ids_batch,
                output_fields   = ["milvus_id","siglip_vector","irra_vector",
                                   "video","track","source"],
            )
            by_id = {int(r["milvus_id"]): r for r in existing}

            data = []
            for i, mid in enumerate(ids_batch):
                r = by_id.get(int(mid))
                if r is None:
                    continue   # pass1 미완료 스킵
                data.append({
                    "milvus_id":      mid,
                    "siglip_vector":  r["siglip_vector"],
                    "irra_vector":    r["irra_vector"],
                    "solider_vector": solider_vecs[s + i].tolist(),
                    "video":          r["video"],
                    "track":          r["track"],
                    "source":         r["source"],
                })

            if data:
                self._client.upsert(collection_name=self._col, data=data)
            inserted += len(data)
            print(f"  SOLIDER 업서트: {inserted:,}/{total:,}", end="\r")

        self._client.flush(self._col)
        print(f"\n[+] SOLIDER 업서트 완료: {inserted:,}개")
        return inserted

    # ── 검색: 텍스트 (IRRA만) ────────────────────────────────────
    def search_text(
        self,
        irra_vec: np.ndarray,
        top_k:    int = 200,
    ) -> list[dict]:
        results = self._client.search(
            collection_name = self._col,
            data            = irra_vec.tolist(),
            anns_field      = "irra_vector",
            search_params   = {"metric_type": "IP", "params": {"nprobe": NPROBE}},
            limit           = min(top_k, self.ntotal),
            output_fields   = ["video", "track", "source"],
        )
        return self._format(results[0])

    # ── 검색: 이미지 (SigLIP2 + IRRA 하이브리드) ─────────────────
    def search_image(
        self,
        siglip_vec: np.ndarray,
        irra_vec:   np.ndarray,
        top_k:      int = 200,
        weights:    tuple[float, float] = (0.5, 0.5),
    ) -> list[dict]:
        limit = min(top_k, self.ntotal)
        reqs = [
            AnnSearchRequest(
                data         = siglip_vec.tolist(),
                anns_field   = "siglip_vector",
                param        = {"metric_type": "IP", "params": {"nprobe": NPROBE}},
                limit        = limit,
            ),
            AnnSearchRequest(
                data         = irra_vec.tolist(),
                anns_field   = "irra_vector",
                param        = {"metric_type": "IP", "params": {"nprobe": NPROBE}},
                limit        = limit,
            ),
        ]
        results = self._client.hybrid_search(
            collection_name = self._col,
            reqs            = reqs,
            ranker          = RRFRanker(k=60),
            limit           = limit,
            output_fields   = ["video", "track", "source"],
        )
        return self._format(results[0])

    # ── 검색: 풀 퓨전 (SigLIP2 + IRRA + SOLIDER) ─────────────────
    def search_full(
        self,
        siglip_vec:  np.ndarray,
        irra_vec:    np.ndarray,
        solider_vec: np.ndarray,
        top_k:       int = 200,
    ) -> list[dict]:
        limit = min(top_k, self.ntotal)
        reqs = [
            AnnSearchRequest(
                data       = siglip_vec.tolist(),
                anns_field = "siglip_vector",
                param      = {"metric_type": "IP", "params": {"nprobe": NPROBE}},
                limit      = limit,
            ),
            AnnSearchRequest(
                data       = irra_vec.tolist(),
                anns_field = "irra_vector",
                param      = {"metric_type": "IP", "params": {"nprobe": NPROBE}},
                limit      = limit,
            ),
            AnnSearchRequest(
                data       = solider_vec.tolist(),
                anns_field = "solider_vector",
                param      = {"metric_type": "IP", "params": {"nprobe": NPROBE}},
                limit      = limit,
            ),
        ]
        results = self._client.hybrid_search(
            collection_name = self._col,
            reqs            = reqs,
            ranker          = RRFRanker(k=60),
            limit           = limit,
            output_fields   = ["video", "track", "source"],
        )
        return self._format(results[0])

    # ── 트랙 집계 검색 ────────────────────────────────────────────
    def search_by_track(
        self,
        query_vecs:     dict,          # {"siglip": vec, "irra": vec, "solider": vec(선택)}
        top_k:          int   = 200,
        threshold:      float = 0.3,
        mode:           str   = "image",  # "text" | "image" | "full"
    ) -> list[dict]:
        if mode == "text":
            raw = self.search_text(query_vecs["irra"], top_k * 20)
        elif mode == "full" and "solider" in query_vecs:
            raw = self.search_full(
                query_vecs["siglip"], query_vecs["irra"], query_vecs["solider"], top_k * 20
            )
        else:
            raw = self.search_image(
                query_vecs["siglip"], query_vecs["irra"], top_k * 20
            )

        track_best: dict[str, dict] = {}
        for hit in raw:
            score = float(hit["score"])
            if score < threshold:
                continue
            key = f"{hit['video']}/{hit['track']}"
            if key not in track_best or score > track_best[key]["similarity"]:
                track_best[key] = {
                    "milvus_id":  hit["id"],
                    "video":      hit["video"],
                    "track":      hit["track"],
                    "source":     hit["source"],
                    "similarity": round(score, 4),
                }

        ranked = sorted(track_best.values(), key=lambda x: x["similarity"], reverse=True)
        return [{"rank": i + 1, **item} for i, item in enumerate(ranked[:top_k])]

    # ── 내부 ──────────────────────────────────────────────────────
    def _format(self, hits) -> list[dict]:
        out = []
        for h in hits:
            entity = h.get("entity", h)
            out.append({
                "id":         h.get("id"),
                "score":      h.get("distance", h.get("score", 0.0)),
                "video":      entity.get("video", ""),
                "track":      entity.get("track", ""),
                "source":     entity.get("source", ""),
            })
        return out

    def close(self):
        self._client.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


if __name__ == "__main__":
    print(f"[*] PersonMilvusManager 연결 테스트: {_URI}")
    try:
        mgr = PersonMilvusManager()
        print(f"[+] 컬렉션 '{COLLECTION_NAME}': {mgr.ntotal:,} 벡터")
        mgr.close()
    except Exception as e:
        print(f"[!] 연결 실패: {e}")
