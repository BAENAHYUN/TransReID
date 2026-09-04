"""
migrate_to_pg_milvus.py — SQLite + FAISS → PostgreSQL + Milvus 마이그레이션

기존 데이터를 한 번에 새 DB로 이전합니다.
실행 전에 docker-compose up -d 로 PostgreSQL + Milvus 가 기동되어 있어야 합니다.

사용법:
  cd C:\\Users\\Karsel\\Desktop\\Final_tool\\TransReID
  python src/RF-DETR/migrate_to_pg_milvus.py

옵션:
  --dry-run          실제 삽입 없이 파일 검증만
  --batch-size N     배치 크기 (기본 2000)
  --skip-milvus      PostgreSQL만 마이그레이션
  --skip-postgres    Milvus만 마이그레이션
"""

from __future__ import annotations

import sys
import json
import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
import faiss
import psycopg2
import psycopg2.extras

# ── 경로 설정 ─────────────────────────────────────────────────────────
_RFDETR = Path(__file__).parent
_ROOT   = _RFDETR.parents[1]
sys.path.insert(0, str(_ROOT / "src" / "IRRA"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_RFDETR))

from db_manager      import PostgresMetadataDB, _DEFAULT_DSN
from milvus_manager  import MilvusManager

# ── 기본 파일 경로 ────────────────────────────────────────────────────
DEFAULT_SQLITE = str(_ROOT / "data" / "irra_index" / "metadata.db")
DEFAULT_FAISS  = str(_ROOT / "data" / "irra_index" / "irra.faiss")


# ═══════════════════════════════════════════════════════════════════════
# 마이그레이션 함수
# ═══════════════════════════════════════════════════════════════════════

def migrate(
    sqlite_path: str   = DEFAULT_SQLITE,
    faiss_path:  str   = DEFAULT_FAISS,
    pg_dsn:      str   = _DEFAULT_DSN,
    batch_size:  int   = 2000,
    dry_run:     bool  = False,
    skip_pg:     bool  = False,
    skip_milvus: bool  = False,
):
    t0 = time.time()
    print("\n" + "═"*60)
    print("  SQLite + FAISS → PostgreSQL + Milvus 마이그레이션")
    print("═"*60)

    # ── 1. SQLite 읽기 ─────────────────────────────────────────────
    print(f"\n[1/4] SQLite 읽기: {sqlite_path}")
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    rows = sqlite_conn.execute(
        "SELECT vector_id, data FROM metadata ORDER BY vector_id"
    ).fetchall()
    sqlite_conn.close()

    total = len(rows)
    print(f"      {total:,}개 레코드 확인")

    if total == 0:
        print("[!] SQLite 비어 있음. 종료.")
        return

    # JSON 파싱
    records: list[dict] = []
    for row in rows:
        rec = json.loads(row["data"])
        rec["_vector_id"] = int(row["vector_id"])
        records.append(rec)

    # ── 2. FAISS 읽기 ──────────────────────────────────────────────
    print(f"\n[2/4] FAISS 읽기: {faiss_path}")
    index = faiss.read_index(faiss_path)
    n_vectors = index.ntotal
    print(f"      {n_vectors:,}개 벡터 (dim={index.d})")

    if n_vectors != total:
        print(f"[!] 경고: SQLite({total}) ≠ FAISS({n_vectors}) — 불일치")

    # 전체 벡터 추출
    print("      벡터 추출 중 ...")
    vectors = np.zeros((n_vectors, index.d), dtype=np.float32)
    index.reconstruct_n(0, n_vectors, vectors)
    print(f"      추출 완료: {vectors.shape}")

    if dry_run:
        print("\n[Dry-run] 검증 완료. 실제 삽입 없이 종료.")
        return

    # ── 3. PostgreSQL 마이그레이션 ────────────────────────────────
    if not skip_pg:
        print(f"\n[3/4] PostgreSQL 삽입: {pg_dsn}")
        conn = psycopg2.connect(pg_dsn)

        try:
            with conn.cursor() as cur:
                # 기존 데이터 확인
                cur.execute("SELECT COUNT(*) FROM frames")
                existing = cur.fetchone()[0]
                if existing > 0:
                    print(f"      기존 {existing:,}개 프레임 발견 — 스킵 (UPSERT)")

            # UPSERT 배치 삽입
            inserted_pg = 0
            for start in range(0, total, batch_size):
                batch = records[start : start + batch_size]
                data = [
                    (
                        rec["_vector_id"],
                        rec.get("video", ""),
                        rec.get("track", ""),
                        rec.get("source", "video"),
                        rec.get("frame", ""),
                        rec.get("person", ""),
                        rec.get("path", ""),
                    )
                    for rec in batch
                ]
                with conn.cursor() as cur:
                    psycopg2.extras.execute_values(
                        cur,
                        """
                        INSERT INTO frames
                            (milvus_id, video, track, source, frame, person, path)
                        VALUES %s
                        ON CONFLICT (milvus_id) DO UPDATE SET
                            video  = EXCLUDED.video,
                            track  = EXCLUDED.track,
                            source = EXCLUDED.source,
                            frame  = EXCLUDED.frame,
                            person = EXCLUDED.person,
                            path   = EXCLUDED.path
                        """,
                        data,
                        page_size=batch_size,
                    )
                conn.commit()
                inserted_pg += len(batch)
                pct = inserted_pg / total * 100
                print(f"  PG 삽입: {inserted_pg:,}/{total:,} ({pct:.1f}%)", end="\r")

            print(f"\n[+] PostgreSQL 프레임 삽입 완료: {inserted_pg:,}개")

            # tracks 테이블 자동 생성
            print("      tracks 집계 테이블 구성 중 ...")
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO tracks (video, track, source, n_frames, best_path)
                    SELECT
                        video,
                        track,
                        source,
                        COUNT(*) AS n_frames,
                        MIN(path)  AS best_path
                    FROM frames
                    GROUP BY video, track, source
                    ON CONFLICT (video, track) DO UPDATE SET
                        n_frames  = EXCLUDED.n_frames,
                        best_path = EXCLUDED.best_path
                """)
            conn.commit()

            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM tracks")
                n_tracks = cur.fetchone()[0]
            print(f"[+] tracks 테이블: {n_tracks:,}개 트랙")

        finally:
            conn.close()
    else:
        print("\n[3/4] PostgreSQL 건너뜀 (--skip-postgres)")

    # ── 4. Milvus 마이그레이션 ────────────────────────────────────
    if not skip_milvus:
        print(f"\n[4/4] Milvus 삽입 ...")
        mgr = MilvusManager()

        if mgr.ntotal > 0:
            print(f"      기존 {mgr.ntotal:,}개 벡터 존재 — 컬렉션 삭제 후 재삽입")
            from pymilvus import utility
            utility.drop_collection(mgr._collection_name)
            mgr._ensure_collection()

        milvus_ids = [rec["_vector_id"] for rec in records]
        vids       = [rec.get("video",  "") for rec in records]
        trks       = [rec.get("track",  "") for rec in records]
        srcs       = [rec.get("source", "video") for rec in records]

        mgr.insert(
            milvus_ids=milvus_ids,
            vectors=vectors[:total],
            videos=vids,
            tracks=trks,
            sources=srcs,
            batch_size=batch_size,
        )
        mgr.close()
    else:
        print("\n[4/4] Milvus 건너뜀 (--skip-milvus)")

    # ── 완료 ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'═'*60}")
    print(f"  마이그레이션 완료 — {elapsed:.1f}초")
    print(f"  SQLite  : {total:,}개 레코드")
    print(f"  FAISS   : {n_vectors:,}개 벡터")
    print(f"  → PostgreSQL frames 테이블 ✓")
    print(f"  → Milvus '{MilvusManager.__module__}' 컬렉션 ✓")
    print(f"{'═'*60}\n")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SQLite+FAISS → PostgreSQL+Milvus 마이그레이션")
    ap.add_argument("--sqlite",        default=DEFAULT_SQLITE, help="기존 SQLite .db 경로")
    ap.add_argument("--faiss",         default=DEFAULT_FAISS,  help="기존 FAISS .faiss 경로")
    ap.add_argument("--pg-dsn",        default=_DEFAULT_DSN,   help="PostgreSQL DSN")
    ap.add_argument("--batch-size",    type=int, default=2000,  help="배치 크기 (기본 2000)")
    ap.add_argument("--dry-run",       action="store_true",    help="실제 삽입 없이 검증만")
    ap.add_argument("--skip-postgres", action="store_true",    help="PostgreSQL 건너뜀")
    ap.add_argument("--skip-milvus",   action="store_true",    help="Milvus 건너뜀")
    args = ap.parse_args()

    migrate(
        sqlite_path = args.sqlite,
        faiss_path  = args.faiss,
        pg_dsn      = args.pg_dsn,
        batch_size  = args.batch_size,
        dry_run     = args.dry_run,
        skip_pg     = args.skip_postgres,
        skip_milvus = args.skip_milvus,
    )
