"""
qdrant_person_manager.py — 인물 멀티벡터 Qdrant 인터페이스 (로컬 파일 기반)

컬렉션: forensic_person_local
named vectors:
  siglip  : 768-dim  Cosine — SigLIP2 semantic
  irra    : 512-dim  Cosine — IRRA person+text
  solider : 1024-dim Cosine — SOLIDER person ReID

payload: video, track, source  (상세 메타는 PostgreSQL frames/tracks)

외부 서버 불필요 — 로컬 파일 DB:
  FORENSIC_QDRANT_PATH  기본: <ROOT>/data/qdrant_local
"""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, PointVectors,
    Prefetch, FusionQuery, Fusion,
)

_HERE    = Path(__file__).parent
ROOT_DIR = _HERE.parents[1]

_QDRANT_PATH = os.environ.get(
    "FORENSIC_QDRANT_PATH",
    str(ROOT_DIR / "data" / "qdrant_local"),
)

COLLECTION_NAME = "forensic_person"
DIM_SIGLIP      = 768
DIM_IRRA        = 512
DIM_SOLIDER     = 1024


class PersonQdrantManager:
    """
    인물 멀티벡터 Qdrant 컬렉션 관리 + 검색 (로컬 파일 기반).

    DB 구축:
        mgr.insert(ids, siglip_vecs, irra_vecs, videos, tracks, sources)
        mgr.update_solider(ids, solider_vecs)   # pass2 에서 추가

    검색:
        mgr.search_text(irra_vec, top_k)
        mgr.search_image(siglip_vec, irra_vec, top_k)
        mgr.search_full(siglip_vec, irra_vec, solider_vec, top_k)
        mgr.search_by_track(query_vecs, mode, top_k, threshold)
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

    # ── 컬렉션 생성 ───────────────────────────────────────────────
    def _ensure_collection(self):
        existing = [c.name for c in self._client.get_collections().collections]
        if self._col in existing:
            print(f"[*] 기존 컬렉션 로드: {self._col}")
        else:
            print(f"[*] 새 컬렉션 생성: {self._col}")
            self._client.create_collection(
                collection_name = self._col,
                vectors_config  = {
                    "Siglip2":  VectorParams(size=DIM_SIGLIP,  distance=Distance.COSINE),
                    "irra":    VectorParams(size=DIM_IRRA,    distance=Distance.COSINE),
                    "solider": VectorParams(size=DIM_SOLIDER, distance=Distance.COSINE),
                },
            )
        n = self.ntotal
        print(f"[+] 컬렉션 준비: {n:,} 포인트")

    # ── 프로퍼티 ──────────────────────────────────────────────────
    @property
    def ntotal(self) -> int:
        return self._client.count(self._col, exact=True).count

    # ── 삽입 (pass1: siglip + irra, solider = zeros) ─────────────
    def insert(
        self,
        point_ids:   list[str],
        siglip_vecs: np.ndarray,    # (N, 768)
        irra_vecs:   np.ndarray,    # (N, 512)
        payloads:    list[dict],    # {video, track, source, split, category, ...}
        batch_size:  int = 256,
    ) -> int:
        N = len(point_ids)
        solider_zero = [0.0] * DIM_SOLIDER
        inserted = 0

        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            points = [
                PointStruct(
                    id      = point_ids[i],
                    vector  = {
                        "siglip2":  siglip_vecs[i].tolist(),
                        "irra":    irra_vecs[i].tolist(),
                        "solider": solider_zero,
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

    # ── SOLIDER 벡터 업데이트 (pass2) ────────────────────────────
    def update_solider(
        self,
        point_ids:    list[str],
        solider_vecs: np.ndarray,   # (N, 1024)
        batch_size:   int = 256,
    ) -> int:
        N = len(point_ids)
        updated = 0

        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            self._client.update_vectors(
                collection_name = self._col,
                points = [
                    PointVectors(
                        id     = point_ids[i],
                        vector = {"solider": solider_vecs[i].tolist()},
                    )
                    for i in range(s, e)
                ],
            )
            updated += (e - s)
            print(f"  SOLIDER 업데이트: {updated:,}/{N:,}", end="\r")

        print(f"\n[+] SOLIDER 업데이트 완료: {updated:,}개")
        return updated

    # ── 존재 여부 확인 ────────────────────────────────────────────
    def existing_ids(self, point_ids: list[str]) -> set[str]:
        result = set()
        batch = 512
        for s in range(0, len(point_ids), batch):
            hits = self._client.retrieve(
                collection_name = self._col,
                ids             = point_ids[s:s + batch],
                with_vectors    = False,
                with_payload    = False,
            )
            result.update(h.id for h in hits)
        return result

    # ── 검색: 텍스트 (IRRA만) ────────────────────────────────────
    def search_text(
        self,
        irra_vec: np.ndarray,
        top_k:    int = 200,
    ) -> list[dict]:
        results = self._client.query_points(
            collection_name = self._col,
            query           = irra_vec.tolist(),
            using           = "irra",
            limit           = top_k,
            with_payload    = True,
        ).points
        return self._fmt(results)

    # ── 검색: 이미지 (siglip + irra RRF) ─────────────────────────
    def search_image(
        self,
        siglip_vec: np.ndarray,
        irra_vec:   np.ndarray,
        top_k:      int = 200,
    ) -> list[dict]:
        results = self._client.query_points(
            collection_name = self._col,
            prefetch = [
                Prefetch(query=siglip_vec.tolist(), using="siglip", limit=top_k * 2),
                Prefetch(query=irra_vec.tolist(),   using="irra",   limit=top_k * 2),
            ],
            query        = FusionQuery(fusion=Fusion.RRF),
            limit        = top_k,
            with_payload = True,
        ).points
        return self._fmt(results)

    # ── 검색: 풀 퓨전 (siglip + irra + solider RRF) ──────────────
    def search_full(
        self,
        siglip_vec:  np.ndarray,
        irra_vec:    np.ndarray,
        solider_vec: np.ndarray,
        top_k:       int = 200,
    ) -> list[dict]:
        results = self._client.query_points(
            collection_name = self._col,
            prefetch = [
                Prefetch(query=siglip_vec.tolist(),  using="siglip",  limit=top_k * 2),
                Prefetch(query=irra_vec.tolist(),    using="irra",    limit=top_k * 2),
                Prefetch(query=solider_vec.tolist(), using="solider", limit=top_k * 2),
            ],
            query        = FusionQuery(fusion=Fusion.RRF),
            limit        = top_k,
            with_payload = True,
        ).points
        return self._fmt(results)

    # ── 트랙 집계 검색 ────────────────────────────────────────────
    def search_by_track(
        self,
        query_vecs: dict,
        top_k:      int   = 200,
        threshold:  float = 0.3,
        mode:       str   = "image",
    ) -> list[dict]:
        pool = top_k * 30
        if mode == "text":
            raw = self.search_text(query_vecs["irra"], pool)
        elif mode == "full" and "solider" in query_vecs:
            raw = self.search_full(
                query_vecs["siglip"], query_vecs["irra"], query_vecs["solider"], pool
            )
        else:
            raw = self.search_image(query_vecs["siglip"], query_vecs["irra"], pool)

        track_best: dict[str, dict] = {}
        for h in raw:
            score = float(h["score"])
            if score < threshold:
                continue
            key = f"{h['video']}/{h['track']}"
            if key not in track_best or score > track_best[key]["similarity"]:
                track_best[key] = {
                    "point_id":  h["id"],
                    "video":     h["video"],
                    "track":     h["track"],
                    "source":    h.get("source", ""),
                    "similarity": round(score, 4),
                }

        ranked = sorted(track_best.values(), key=lambda x: x["similarity"], reverse=True)
        return [{"rank": i + 1, **item} for i, item in enumerate(ranked[:top_k])]

    # ── 내부 ──────────────────────────────────────────────────────
    def _fmt(self, hits) -> list[dict]:
        out = []
        for h in hits:
            p = h.payload or {}
            out.append({
                "id":       h.id,
                "score":    getattr(h, "score", 0.0),
                "video":    p.get("video", ""),
                "track":    p.get("track", ""),
                "source":   p.get("source", ""),
                "split":    p.get("split", ""),
                "category": p.get("category", ""),
            })
        return out

    def close(self):
        self._client.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


if __name__ == "__main__":
    print(f"[*] PersonQdrantManager 연결 테스트: {_QDRANT_PATH}")
    try:
        mgr = PersonQdrantManager()
        print(f"[+] 컬렉션 '{COLLECTION_NAME}': {mgr.ntotal:,} 포인트")
        mgr.close()
    except Exception as e:
        print(f"[!] 연결 실패: {e}")
