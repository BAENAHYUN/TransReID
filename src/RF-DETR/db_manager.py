"""
db_manager.py — PostgreSQL 메타데이터 인터페이스

SQLite MetadataDB 의 드롭인 대체.
기존 코드는 db[vector_id] 그대로 사용 가능.

추가 기능:
  - 감사 로그 (search_audit)
  - 다중 사용자 (users)
  - 트랙 집계 (tracks)

설치:
  pip install psycopg2-binary

환경변수 (또는 .env):
  FORENSIC_PG_DSN = "postgresql://forensic:forensic_secret@localhost:5432/forensic_db"
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import psycopg2
import psycopg2.pool
import psycopg2.extras

# ── DSN 기본값 ───────────────────────────────────────────────────────
_DEFAULT_DSN = os.environ.get(
    "FORENSIC_PG_DSN",
    "postgresql://forensic:forensic_secret@localhost:5432/forensic_db",
)


# ═══════════════════════════════════════════════════════════════════════
# PostgresMetadataDB  — MetadataDB 드롭인 대체
# ═══════════════════════════════════════════════════════════════════════

class PostgresMetadataDB:
    """
    frames 테이블을 읽는 read-optimised wrapper.

    Interface:
        db = PostgresMetadataDB()
        record = db[milvus_id]   # → dict (video, track, source, frame, person, path)
        n      = len(db)         # → 총 프레임 수
    """

    def __init__(self, dsn: str = _DEFAULT_DSN, pool_min: int = 1, pool_max: int = 10):
        self._pool = psycopg2.pool.ThreadedConnectionPool(
            pool_min, pool_max, dsn=dsn,
        )

    # ── list-compatible interface ──────────────────────────────────────
    def __len__(self) -> int:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM frames")
                return cur.fetchone()[0]

    def __getitem__(self, milvus_id: int) -> Dict[str, Any]:
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    "SELECT video, track, source, frame, person, path "
                    "FROM frames WHERE milvus_id = %s",
                    (int(milvus_id),),
                )
                row = cur.fetchone()
        if row is None:
            raise KeyError(f"milvus_id {milvus_id} not found in frames")
        return dict(row)

    # ── 트랙 프레임 목록 ──────────────────────────────────────────────
    def get_track_frames(self, video: str, track: str) -> list[Dict[str, Any]]:
        """특정 트랙의 모든 프레임 반환."""
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    "SELECT milvus_id, frame, path FROM frames "
                    "WHERE video=%s AND track=%s ORDER BY frame",
                    (video, track),
                )
                return [dict(r) for r in cur.fetchall()]

    # ── 전체 트랙 목록 (클러스터링 입력용) ───────────────────────────
    def list_tracks(self) -> list[Dict[str, Any]]:
        """tracks 테이블의 모든 트랙 반환."""
        with self._get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    "SELECT id, video, track, source, n_frames, best_path, cluster_id "
                    "FROM tracks ORDER BY video, track"
                )
                return [dict(r) for r in cur.fetchall()]

    # ── 클러스터 ID 업데이트 ──────────────────────────────────────────
    def update_cluster_ids(self, assignments: dict[tuple[str, str], int]) -> int:
        """
        {(video, track): cluster_id} 로 tracks 테이블 업데이트.
        반환: 업데이트된 행 수
        """
        updated = 0
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                for (video, track), cluster_id in assignments.items():
                    cur.execute(
                        "UPDATE tracks SET cluster_id=%s WHERE video=%s AND track=%s",
                        (cluster_id, video, track),
                    )
                    updated += cur.rowcount
            conn.commit()
        return updated

    # ── 내부 ──────────────────────────────────────────────────────────
    def _get_conn(self):
        return _PooledConnection(self._pool)

    def close(self):
        self._pool.closeall()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ═══════════════════════════════════════════════════════════════════════
# AuditLogger — 검색 감사 로그
# ═══════════════════════════════════════════════════════════════════════

class AuditLogger:
    """
    search_audit 테이블에 검색 이력을 기록.

    Usage:
        logger = AuditLogger()

        # 검색 시작
        audit_id = logger.begin(
            investigator="홍길동",
            query_type="text",
            query_content="red jacket man",
            irra_threshold=0.42,
            qwen_threshold=0.45,
        )

        # ... 검색 실행 ...

        # 검색 완료
        logger.finish(
            audit_id=audit_id,
            irra_candidates=173,
            qwen_passed=16,
            final_saved=11,
            result_dir="/path/to/results",
        )
    """

    def __init__(self, dsn: str = _DEFAULT_DSN):
        self._dsn = dsn
        self._conn: Optional[psycopg2.extensions.connection] = None

    def _ensure_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._dsn)

    def begin(
        self,
        investigator:   str,
        query_type:     str,
        query_content:  str  = "",
        irra_threshold: float = 0.42,
        qwen_threshold: float = 0.45,
        session_id:     str  = "",
        client_ip:      str  = "",
    ) -> int:
        """감사 레코드 생성 → audit_id 반환."""
        self._ensure_conn()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO search_audit
                    (investigator, query_type, query_content,
                     irra_threshold, qwen_threshold,
                     session_id, client_ip, started_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                RETURNING id
                """,
                (investigator, query_type, query_content,
                 irra_threshold, qwen_threshold,
                 session_id, client_ip),
            )
            audit_id = cur.fetchone()[0]
        self._conn.commit()
        return audit_id

    def finish(
        self,
        audit_id:       int,
        irra_candidates: int = 0,
        qwen_passed:    int = 0,
        final_saved:    int = 0,
        result_dir:     str = "",
    ) -> None:
        """감사 레코드 완료 처리."""
        self._ensure_conn()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE search_audit
                SET irra_candidates=%s, qwen_passed=%s, final_saved=%s,
                    result_dir=%s, finished_at=NOW()
                WHERE id=%s
                """,
                (irra_candidates, qwen_passed, final_saved, result_dir, audit_id),
            )
        self._conn.commit()

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ═══════════════════════════════════════════════════════════════════════
# 내부 유틸
# ═══════════════════════════════════════════════════════════════════════

class _PooledConnection:
    """psycopg2 pool 에서 커넥션을 빌려 with 문으로 반납."""

    def __init__(self, pool: psycopg2.pool.ThreadedConnectionPool):
        self._pool = pool
        self._conn = None

    def __enter__(self):
        self._conn = self._pool.getconn()
        return self._conn

    def __exit__(self, exc_type, *_):
        if exc_type:
            self._conn.rollback()
        self._pool.putconn(self._conn)


# ═══════════════════════════════════════════════════════════════════════
# 연결 테스트 CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"[*] PostgreSQL 연결 테스트: {_DEFAULT_DSN}")
    try:
        db = PostgresMetadataDB()
        n  = len(db)
        print(f"[+] 연결 성공 — frames: {n:,}건")
        if n > 0:
            rec = db[0]
            print(f"    첫 번째 레코드: {rec}")
        db.close()
    except Exception as e:
        print(f"[!] 연결 실패: {e}")
        print("    docker-compose up -d 로 PostgreSQL 실행 후 재시도")
