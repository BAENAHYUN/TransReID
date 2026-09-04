"""
qdrant_object_manager.py — 객체 멀티벡터 Qdrant 인터페이스 (로컬 파일 기반)

컬렉션: forensic_object_local
named vectors:
  siglip : 768-dim  Cosine — SigLIP2 semantic (자연어 검색)
  dino   : 768-dim  Cosine — DINOv2 visual (시각적 유사 객체)

payload: video, track, source, label, confidence, split, category

외부 서버 불필요 — 로컬 파일 DB:
  FORENSIC_QDRANT_PATH  기본: <ROOT>/data/qdrant_local
"""
from __future__ import annotations

import os
import numpy as np
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Prefetch, FusionQuery, Fusion,
)

_HERE    = Path(__file__).resolve().parent
ROOT_DIR = _HERE.parents[1]

_QDRANT_PATH = os.environ.get(
    "FORENSIC_QDRANT_PATH",
    str(ROOT_DIR / "data" / "qdrant_local"),
)

COLLECTION_NAME = "forensic_object"
DIM_SIGLIP = 768
DIM_DINO   = 768


class ObjectQdrantManager:
    """
    객체 멀티벡터 Qdrant 컬렉션 관리 + 검색 (로컬 파일 기반).

    DB 구축:
        mgr.insert(ids, siglip_vecs, dino_vecs, payloads)

    검색:
        mgr.search_text(siglip_vec, top_k)          # 자연어 쿼리
        mgr.search_image(siglip_vec, dino_vec, top_k)  # 이미지 쿼리
        mgr.search_by_label(label, top_k)            # 레이블 필터
    """

    def __init__(
        self,
        path:        str  = _QDRANT_PATH,
        collection:  str  = COLLECTION_NAME,
        auto_create: bool = True,
    ):
        self._col = collection
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"[*] Qdrant 로컬 DB: {path}")
        self._client = QdrantClient(path=path)
        print("[+] Qdrant 연결 성공")
        if auto_create:
            self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self._client.get_collections().collections]
        if self._col in existing:
            print(f"[*] 기존 컬렉션 로드: {self._col}")
        else:
            print(f"[*] 새 컬렉션 생성: {self._col}")
            self._client.create_collection(
                collection_name = self._col,
                vectors_config  = {
                    "siglip": VectorParams(size=DIM_SIGLIP, distance=Distance.COSINE),
                    "dino":   VectorParams(size=DIM_DINO,   distance=Distance.COSINE),
                },
            )
        print(f"[+] 컬렉션 준비: {self.ntotal:,} 포인트")

    @property
    def ntotal(self) -> int:
        return self._client.count(self._col, exact=True).count

    # ── 삽입 ──────────────────────────────────────────────────────
    def insert(
        self,
        point_ids:   list[str],
        siglip_vecs: np.ndarray,    # (N, 768)
        dino_vecs:   np.ndarray,    # (N, 768)
        payloads:    list[dict],
        batch_size:  int = 256,
    ) -> int:
        N = len(point_ids)
        inserted = 0

        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            points = [
                PointStruct(
                    id      = point_ids[i],
                    vector  = {
                        "siglip": siglip_vecs[i].tolist(),
                        "dino":   dino_vecs[i].tolist(),
                    },
                    payload = payloads[i],
                )
                for i in range(s, e)
            ]
            self._client.upsert(collection_name=self._col, points=points)
            inserted += (e - s)
            print(f"  삽입: {inserted:,}/{N:,}", end="\r")

        print(f"\n[+] 삽입 완료: {inserted:,}개")
        return inserted

    # ── 존재 확인 ──────────────────────────────────────────────────
    def existing_ids(self, point_ids: list[str]) -> set[str]:
        result = set()
        batch  = 512
        for s in range(0, len(point_ids), batch):
            hits = self._client.retrieve(
                collection_name = self._col,
                ids             = point_ids[s : s + batch],
                with_vectors    = False,
                with_payload    = False,
            )
            result.update(h.id for h in hits)
        return result

    # ── 검색: 자연어 (SigLIP2만) ───────────────────────────────────
    def search_text(
        self,
        siglip_vec: np.ndarray,    # (768,)
        top_k: int = 200,
        label: str | None = None,
    ) -> list[dict]:
        results = self._client.query_points(
            collection_name = self._col,
            query           = siglip_vec.tolist(),
            using           = "siglip",
            limit           = top_k,
            with_payload    = True,
        ).points
        hits = self._fmt(results)
        if label:
            hits = [h for h in hits if h.get("label", "").lower() == label.lower()]
        return hits

    # ── 검색: 이미지 (siglip + dino RRF) ──────────────────────────
    def search_image(
        self,
        siglip_vec: np.ndarray,
        dino_vec:   np.ndarray,
        top_k: int = 200,
        label: str | None = None,
    ) -> list[dict]:
        results = self._client.query_points(
            collection_name = self._col,
            prefetch = [
                Prefetch(query=siglip_vec.tolist(), using="siglip", limit=top_k * 2),
                Prefetch(query=dino_vec.tolist(),   using="dino",   limit=top_k * 2),
            ],
            query        = FusionQuery(fusion=Fusion.RRF),
            limit        = top_k,
            with_payload = True,
        ).points
        hits = self._fmt(results)
        if label:
            hits = [h for h in hits if h.get("label", "").lower() == label.lower()]
        return hits

    # ── 내부 ──────────────────────────────────────────────────────
    def _fmt(self, hits) -> list[dict]:
        out = []
        for h in hits:
            p = h.payload or {}
            out.append({
                "id":         h.id,
                "score":      getattr(h, "score", 0.0),
                "video":      p.get("video",      ""),
                "track":      p.get("track",      ""),
                "source":     p.get("source",     ""),
                "label":      p.get("label",      ""),
                "confidence": p.get("confidence", 0.0),
                "split":      p.get("split",      ""),
                "category":   p.get("category",   ""),
                "frame":      p.get("frame",       ""),
                "path":       p.get("path",        ""),
            })
        return out

    def close(self):
        self._client.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


if __name__ == "__main__":
    print(f"[*] ObjectQdrantManager 연결 테스트: {_QDRANT_PATH}")
    try:
        mgr = ObjectQdrantManager()
        print(f"[+] 컬렉션 '{COLLECTION_NAME}': {mgr.ntotal:,} 포인트")
        mgr.close()
    except Exception as e:
        print(f"[!] 연결 실패: {e}")
