"""
build_track_embeddings.py — Track-level 대표 임베딩 생성

각 track 안의 frame-level 벡터를 평균내어
Qdrant forensic_person_local 에 payload로 저장하고,
별도 컬렉션 forensic_person_tracks 에도 삽입합니다.

평균 방식:
  - simple   : 단순 평균 (기본)
  - quality  : 크롭 면적 기반 품질 가중 평균 (--mode quality)

저장:
  ① forensic_person_local  payload["track_emb_siglip/irra/solider"] 업데이트
  ② forensic_person_tracks 컬렉션 (track 1건 = 1포인트)

사용법:
  python build_track_embeddings.py
  python build_track_embeddings.py --mode quality
  python build_track_embeddings.py --validate-only
"""
from __future__ import annotations

import argparse
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE    = Path(__file__).resolve().parent
ROOT_DIR = _HERE.parents[1]

sys.path.insert(0, str(_HERE))
from qdrant_person_manager import PersonQdrantManager, _QDRANT_PATH, COLLECTION_NAME

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, PointStruct, VectorParams,
    Prefetch, FusionQuery, Fusion,
)

TRACK_COLLECTION = "forensic_person_tracks"
DIM_SIGLIP  = 768
DIM_IRRA    = 512
DIM_SOLIDER = 1024

SCROLL_BATCH = 1000   # Qdrant scroll 배치 크기


def l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1
    return x / n


def track_uuid(video: str, track: str, source: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"track|{source}|{video}|{track}"))


# ── Qdrant 전체 스캔 ──────────────────────────────────────────────────
def scroll_all(client: QdrantClient) -> list[dict]:
    """forensic_person_local 전체 포인트를 벡터+페이로드 포함해서 읽어옴."""
    points = []
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name = COLLECTION_NAME,
            offset          = offset,
            limit           = SCROLL_BATCH,
            with_payload    = True,
            with_vectors    = True,
        )
        for p in result:
            payload = p.payload or {}
            vecs    = p.vector or {}
            points.append({
                "id":      p.id,
                "video":   payload.get("video",    ""),
                "track":   payload.get("track",    ""),
                "source":  payload.get("source",   ""),
                "split":   payload.get("split",    ""),
                "category":payload.get("category", ""),
                "path":    payload.get("path",     ""),
                "siglip":  np.array(vecs.get("siglip",  []), dtype=np.float32),
                "irra":    np.array(vecs.get("irra",     []), dtype=np.float32),
                "solider": np.array(vecs.get("solider",  []), dtype=np.float32),
            })
        if next_offset is None:
            break
        offset = next_offset

    print(f"[+] 전체 포인트 스캔 완료: {len(points):,}개")
    return points


# ── 크롭 품질 점수 (면적 기반) ────────────────────────────────────────
def crop_quality(path_str: str) -> float:
    """
    크롭 파일명 또는 크기로 품질 추정.
    실제로는 PIL로 이미지를 열어 픽셀 수를 반환.
    파일이 없으면 1.0 반환 (단순 평균으로 폴백).
    """
    try:
        from PIL import Image
        p = Path(path_str)
        if not p.exists():
            return 1.0
        with Image.open(p) as img:
            w, h = img.size
        return float(w * h)
    except Exception:
        return 1.0


# ── Track별 집계 ──────────────────────────────────────────────────────
def aggregate_tracks(points: list[dict], mode: str = "simple") -> list[dict]:
    """
    포인트 리스트 → track별 대표 임베딩 계산.
    반환: list[{track_id, video, track, source, split, category,
                siglip, irra, solider, n_frames, best_path}]
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in points:
        key = f"{p['source']}|{p['video']}|{p['track']}"
        groups[key].append(p)

    tracks = []
    for key, frames in groups.items():
        source, video, track = key.split("|", 2)

        if mode == "quality":
            weights = np.array([crop_quality(f["path"]) for f in frames], dtype=np.float32)
            weights /= weights.sum() + 1e-9
        else:
            weights = np.ones(len(frames), dtype=np.float32) / len(frames)

        def wavg(vec_key: str) -> np.ndarray:
            vecs = np.stack([f[vec_key] for f in frames])   # (N, D)
            avg  = (vecs * weights[:, None]).sum(axis=0)
            return l2(avg[None])[0]

        siglip_avg  = wavg("siglip")
        irra_avg    = wavg("irra")
        solider_avg = wavg("solider")

        # 대표 경로: siglip norm이 가장 높은 frame (= 가장 잘 인식된 crop)
        norms = [np.linalg.norm(f["siglip"]) for f in frames]
        best  = frames[int(np.argmax(norms))]

        tracks.append({
            "track_id":  track_uuid(video, track, source),
            "video":     video,
            "track":     track,
            "source":    source,
            "split":     frames[0].get("split",    ""),
            "category":  frames[0].get("category", ""),
            "n_frames":  len(frames),
            "best_path": best["path"],
            "siglip":    siglip_avg,
            "irra":      irra_avg,
            "solider":   solider_avg,
        })

    print(f"[+] 트랙 집계 완료: {len(tracks):,}개 트랙")
    return tracks


# ── forensic_person_tracks 컬렉션 생성 / 삽입 ────────────────────────
def ensure_track_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if TRACK_COLLECTION not in existing:
        print(f"[*] 새 컬렉션 생성: {TRACK_COLLECTION}")
        client.create_collection(
            collection_name = TRACK_COLLECTION,
            vectors_config  = {
                "siglip":  VectorParams(size=DIM_SIGLIP,  distance=Distance.COSINE),
                "irra":    VectorParams(size=DIM_IRRA,    distance=Distance.COSINE),
                "solider": VectorParams(size=DIM_SOLIDER, distance=Distance.COSINE),
            },
        )
    else:
        n = client.count(TRACK_COLLECTION, exact=True).count
        print(f"[*] 기존 컬렉션 로드: {TRACK_COLLECTION} ({n:,} 포인트)")


def upsert_tracks(client: QdrantClient, tracks: list[dict], batch_size: int = 256):
    total = len(tracks)
    done  = 0
    for s in range(0, total, batch_size):
        batch = tracks[s : s + batch_size]
        points = [
            PointStruct(
                id     = t["track_id"],
                vector = {
                    "siglip":  t["siglip"].tolist(),
                    "irra":    t["irra"].tolist(),
                    "solider": t["solider"].tolist(),
                },
                payload = {
                    "video":     t["video"],
                    "track":     t["track"],
                    "source":    t["source"],
                    "split":     t["split"],
                    "category":  t["category"],
                    "n_frames":  t["n_frames"],
                    "best_path": t["best_path"],
                },
            )
            for t in batch
        ]
        client.upsert(collection_name=TRACK_COLLECTION, points=points)
        done += len(batch)
        print(f"  [{done:>6,}/{total:,}] 삽입 중...", end="\r")
    print(f"\n[+] {TRACK_COLLECTION} 삽입 완료: {done:,}개")


# ── 검증 출력 ─────────────────────────────────────────────────────────
def validate(client: QdrantClient):
    frame_n = client.count(COLLECTION_NAME,  exact=True).count
    track_n = client.count(TRACK_COLLECTION, exact=True).count
    print(f"\n[검증]")
    print(f"  forensic_person_local  (frame) : {frame_n:,}개")
    print(f"  forensic_person_tracks (track) : {track_n:,}개")

    # 샘플 트랙 norm 확인
    sample, _ = client.scroll(
        collection_name = TRACK_COLLECTION,
        limit           = 5,
        with_payload    = True,
        with_vectors    = True,
    )
    print(f"\n[샘플 트랙 5개]")
    for p in sample:
        v = p.vector or {}
        sv  = np.array(v.get("siglip",  []), dtype=np.float32)
        iv  = np.array(v.get("irra",    []), dtype=np.float32)
        sol = np.array(v.get("solider", []), dtype=np.float32)
        pay = p.payload or {}
        print(f"  {pay.get('video','')[:25]:25}  track={pay.get('track','')[:10]:10}  "
              f"frames={pay.get('n_frames',0):4d}  "
              f"siglip={np.linalg.norm(sv):.3f}  irra={np.linalg.norm(iv):.3f}  "
              f"solider={np.linalg.norm(sol):.3f}")


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Track 대표 임베딩 생성")
    ap.add_argument("--mode", choices=["simple", "quality"], default="simple",
                    help="집계 방식: simple=단순평균, quality=품질가중평균")
    ap.add_argument("--validate-only", action="store_true",
                    help="집계 없이 통계만 출력")
    args = ap.parse_args()

    client = QdrantClient(path=_QDRANT_PATH)

    if args.validate_only:
        validate(client)
        client.close()
        return

    print("=" * 72)
    print(f"TRACK EMBEDDING BUILD  (mode={args.mode})")
    print("=" * 72)

    # 전체 frame 포인트 스캔
    points = scroll_all(client)

    # solider가 전부 zeros인 포인트 비율 체크
    zero_sol = sum(1 for p in points if np.linalg.norm(p["solider"]) < 0.01)
    if zero_sol > 0:
        print(f"[!] SOLIDER zeros: {zero_sol:,}/{len(points):,} — "
              f"pass2 미완료 포인트 포함. 평균에 영향 가능.")

    # track 집계
    tracks = aggregate_tracks(points, mode=args.mode)

    # forensic_person_tracks 저장
    ensure_track_collection(client)
    upsert_tracks(client, tracks)

    validate(client)
    client.close()

    print("\n" + "=" * 72)
    print("TRACK EMBEDDING BUILD COMPLETE")
    print(f"  {len(tracks):,}개 트랙 → forensic_person_tracks")
    print(f"  다음: python cluster_persons.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
