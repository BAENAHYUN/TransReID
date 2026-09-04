-- ============================================================
-- ForensicSearch DB 초기화 스크립트
-- PostgreSQL 16 · docker-compose 자동 실행
-- ============================================================

-- ── 확장 ──────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- password_hash 생성용

-- ============================================================
-- 1. 영상 테이블
-- ============================================================
CREATE TABLE IF NOT EXISTS videos (
    id         SERIAL       PRIMARY KEY,
    name       TEXT         UNIQUE NOT NULL,   -- 'Normal_Videos_042_x264'
    source     TEXT         NOT NULL DEFAULT 'video',  -- 'video' | 'scvd'
    fps        FLOAT        NOT NULL DEFAULT 30.0,
    added_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 2. 프레임 테이블  (SQLite metadata 1:1 대응)
-- ============================================================
CREATE TABLE IF NOT EXISTS frames (
    milvus_id  BIGINT       PRIMARY KEY,       -- Milvus vector ID (= 기존 FAISS vector_id)
    video      TEXT         NOT NULL,
    track      TEXT         NOT NULL,
    source     TEXT         NOT NULL DEFAULT 'video',
    frame      TEXT         NOT NULL,
    person     TEXT,
    path       TEXT         NOT NULL,
    created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_frames_video_track ON frames (video, track);
CREATE INDEX IF NOT EXISTS idx_frames_track        ON frames (track);

-- ============================================================
-- 3. 트랙 집계 테이블  (클러스터링 입력용 — 마이그레이션 후 populate)
-- ============================================================
CREATE TABLE IF NOT EXISTS tracks (
    id           SERIAL  PRIMARY KEY,
    video        TEXT    NOT NULL,
    track        TEXT    NOT NULL,
    source       TEXT    NOT NULL DEFAULT 'video',
    n_frames     INTEGER NOT NULL DEFAULT 0,
    best_path    TEXT,
    cluster_id   INTEGER,                      -- 클러스터링 후 할당
    UNIQUE (video, track)
);

CREATE INDEX IF NOT EXISTS idx_tracks_cluster ON tracks (cluster_id);

-- ============================================================
-- 4. 사용자 (수사관) 테이블
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL       PRIMARY KEY,
    username      TEXT         UNIQUE NOT NULL,
    badge_number  TEXT,
    role          TEXT         NOT NULL DEFAULT 'investigator',  -- 'admin' | 'investigator'
    password_hash TEXT         NOT NULL,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    last_login    TIMESTAMPTZ
);

-- 기본 관리자 계정 (최초 1회만 삽입)
INSERT INTO users (username, badge_number, role, password_hash)
VALUES (
    'admin',
    'ADMIN-001',
    'admin',
    crypt('changeme', gen_salt('bf'))
)
ON CONFLICT (username) DO NOTHING;

-- ============================================================
-- 5. 검색 감사 로그  (법적 필수 · 절대 삭제 불가)
-- ============================================================
CREATE TABLE IF NOT EXISTS search_audit (
    id               BIGSERIAL    PRIMARY KEY,
    user_id          INTEGER      REFERENCES users(id),
    investigator     TEXT,                         -- username snapshot
    query_type       TEXT         NOT NULL,        -- 'text' | 'image' | 'video'
    query_content    TEXT,                         -- 텍스트 내용 or 파일명
    irra_threshold   FLOAT,
    qwen_threshold   FLOAT,
    irra_candidates  INTEGER,
    qwen_passed      INTEGER,
    final_saved      INTEGER,
    result_dir       TEXT,
    started_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    finished_at      TIMESTAMPTZ,
    session_id       TEXT,
    client_ip        TEXT
);

-- 감사 로그는 UPDATE/DELETE 금지 (Row-Level Security)
ALTER TABLE search_audit ENABLE ROW LEVEL SECURITY;

CREATE POLICY audit_insert_only ON search_audit
    FOR INSERT WITH CHECK (true);

CREATE POLICY audit_select_all ON search_audit
    FOR SELECT USING (true);

-- ============================================================
-- 6. 편의 뷰 — 감사 로그 요약
-- ============================================================
CREATE OR REPLACE VIEW v_audit_summary AS
SELECT
    sa.id,
    sa.investigator,
    sa.query_type,
    sa.query_content,
    sa.irra_candidates,
    sa.qwen_passed,
    sa.final_saved,
    sa.started_at,
    EXTRACT(EPOCH FROM (sa.finished_at - sa.started_at))::INT AS duration_sec,
    sa.result_dir
FROM search_audit sa
ORDER BY sa.started_at DESC;
