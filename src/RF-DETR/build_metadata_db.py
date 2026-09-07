"""
build_metadata_db.py — 인물 트랙 메타데이터 SQLite 구축

scvd_person_tracks_v1 디렉터리를 스캔하여 frames / tracks 테이블을 생성합니다.
Qdrant는 벡터 검색 담당, SQLite는 경로/타임스탬프/split/category 담당.

사용법:
  python build_metadata_db.py
  python build_metadata_db.py --fps 30        # FPS 직접 지정 (기본: 영상에서 자동 감지)
  python build_metadata_db.py --validate-only  # 통계만 출력

결과:
  data/forensic.db
    - frames : 개별 크롭 레코드 (frame_num, confidence, path, timestamp_sec)
    - tracks : 트랙 집계 (start/end 타임스탬프, best_path, n_frames)
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import uuid
from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
ROOT_DIR = _HERE.parents[1]

TRACK_ROOT  = ROOT_DIR / "data" / "scvd_person_tracks_v1"
SCVD_ROOT   = ROOT_DIR / "data" / "scvd" / "SCVD_converted"
VIDEO_ROOT  = ROOT_DIR / "data" / "videos"   # mp4 원본 (FPS 감지용)
DB_PATH     = ROOT_DIR / "data" / "forensic.db"

IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_FPS = 30.0

# ── 스키마 ────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    id          TEXT PRIMARY KEY,
    video       TEXT NOT NULL,
    track       TEXT NOT NULL,
    source      TEXT NOT NULL,
    split       TEXT,
    category    TEXT,
    path        TEXT NOT NULL,
    frame_num   INTEGER,
    confidence  REAL,
    timestamp_sec REAL
);

CREATE TABLE IF NOT EXISTS tracks (
    id          TEXT PRIMARY KEY,
    video       TEXT NOT NULL,
    track       TEXT NOT NULL,
    source      TEXT NOT NULL,
    split       TEXT,
    category    TEXT,
    best_path   TEXT,
    n_frames    INTEGER,
    start_frame INTEGER,
    end_frame   INTEGER,
    start_sec   REAL,
    end_sec     REAL
);

CREATE INDEX IF NOT EXISTS idx_frames_video_track ON frames(video, track);
CREATE INDEX IF NOT EXISTS idx_tracks_video ON tracks(video);
CREATE INDEX IF NOT EXISTS idx_tracks_split ON tracks(split);
CREATE INDEX IF NOT EXISTS idx_tracks_category ON tracks(category);
"""


# ── SCVD 메타 인덱스 (split / category) ───────────────────────────────
def build_scvd_info() -> dict[str, dict]:
    """video_stem → {split, category}"""
    idx: dict[str, dict] = {}
    if SCVD_ROOT.exists():
        for p in SCVD_ROOT.rglob("*"):
            if not p.is_file():
                continue
            try:
                parts = p.relative_to(SCVD_ROOT).parts
                idx[p.stem] = {
                    "split":    parts[0] if len(parts) > 1 else "",
                    "category": parts[1] if len(parts) > 2 else "",
                }
            except Exception:
                pass
    return idx


# ── FPS 감지 ──────────────────────────────────────────────────────────
_fps_cache: dict[str, float] = {}

def get_fps(video_stem: str, default: float = DEFAULT_FPS) -> float:
    if video_stem in _fps_cache:
        return _fps_cache[video_stem]
    # mp4 파일 탐색
    candidates = list(VIDEO_ROOT.rglob(f"{video_stem}.*")) if VIDEO_ROOT.exists() else []
    candidates += list(ROOT_DIR.rglob(f"{video_stem}.mp4"))
    for p in candidates:
        if p.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}:
            try:
                import cv2
                cap = cv2.VideoCapture(str(p))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps > 0:
                    _fps_cache[video_stem] = fps
                    return fps
            except Exception:
                pass
    _fps_cache[video_stem] = default
    return default


# ── 파일명 파싱 ───────────────────────────────────────────────────────
_FRAME_RE = re.compile(r"frame_(\d+)(?:.*?_([\d.]+))?$")

def parse_frame_name(stem: str) -> tuple[int, float]:
    """frame_00000180_person_05 또는 frame_00000180_0.92 → (frame_num, confidence)"""
    m = _FRAME_RE.match(stem)
    if m:
        fnum = int(m.group(1))
        conf_str = m.group(2)
        try:
            conf = float(conf_str) if conf_str and float(conf_str) <= 1.0 else 0.0
        except (TypeError, ValueError):
            conf = 0.0
        return fnum, conf
    return 0, 0.0


def frame_id(path: Path) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "frame|scvd|" + str(path.resolve())))

def track_id(video: str, track: str, source: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"track|{source}|{video}|{track}"))


# ── 메인 스캔 ─────────────────────────────────────────────────────────
def scan_tracks(default_fps: float) -> tuple[list[dict], list[dict]]:
    """
    scvd_person_tracks_v1 전체 스캔.
    반환: (frame_rows, track_rows)
    """
    if not TRACK_ROOT.exists():
        print(f"[!] 트랙 루트 없음: {TRACK_ROOT}")
        return [], []

    info = build_scvd_info()
    frame_rows: list[dict] = []
    track_rows:  list[dict] = []

    video_dirs = sorted(TRACK_ROOT.iterdir())
    for video_dir in video_dirs:
        if not video_dir.is_dir():
            continue
        video_stem = video_dir.name                       # 'n170_converted'
        video      = video_stem + ".avi"                  # 'n170_converted.avi' (payload와 일치)
        meta       = info.get(video_stem, {})
        split      = meta.get("split",    "")
        category   = meta.get("category", "")
        fps        = get_fps(video_stem, default_fps)

        for track_dir in sorted(video_dir.iterdir()):
            if not track_dir.is_dir():
                continue
            track = track_dir.name   # 'track_0005'
            tid   = track_id(video, track, "SCVD")

            imgs = sorted(
                p for p in track_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            )
            if not imgs:
                continue

            frames_meta = []
            for img in imgs:
                fnum, conf = parse_frame_name(img.stem)
                ts = fnum / fps
                fid = frame_id(img)
                frames_meta.append({
                    "id":            fid,
                    "video":         video,
                    "track":         track,
                    "source":        "SCVD",
                    "split":         split,
                    "category":      category,
                    "path":          str(img),
                    "frame_num":     fnum,
                    "confidence":    conf,
                    "timestamp_sec": round(ts, 3),
                })

            frame_rows.extend(frames_meta)

            # 트랙 집계
            frame_nums = [f["frame_num"]  for f in frames_meta]
            confs      = [f["confidence"] for f in frames_meta]
            best_idx   = confs.index(max(confs)) if confs else 0

            start_f = min(frame_nums)
            end_f   = max(frame_nums)
            track_rows.append({
                "id":          tid,
                "video":       video,
                "track":       track,
                "source":      "SCVD",
                "split":       split,
                "category":    category,
                "best_path":   frames_meta[best_idx]["path"],
                "n_frames":    len(frames_meta),
                "start_frame": start_f,
                "end_frame":   end_f,
                "start_sec":   round(start_f / fps, 3),
                "end_sec":     round(end_f   / fps, 3),
            })

    return frame_rows, track_rows


# ── DB 삽입 ───────────────────────────────────────────────────────────
def insert_all(db: sqlite3.Connection,
               frame_rows: list[dict], track_rows: list[dict]):
    cur = db.cursor()

    # frames
    cur.executemany("""
        INSERT OR REPLACE INTO frames
        (id, video, track, source, split, category,
         path, frame_num, confidence, timestamp_sec)
        VALUES
        (:id, :video, :track, :source, :split, :category,
         :path, :frame_num, :confidence, :timestamp_sec)
    """, frame_rows)
    print(f"  frames 삽입: {len(frame_rows):,}개")

    # tracks
    cur.executemany("""
        INSERT OR REPLACE INTO tracks
        (id, video, track, source, split, category,
         best_path, n_frames, start_frame, end_frame, start_sec, end_sec)
        VALUES
        (:id, :video, :track, :source, :split, :category,
         :best_path, :n_frames, :start_frame, :end_frame, :start_sec, :end_sec)
    """, track_rows)
    print(f"  tracks 삽입: {len(track_rows):,}개")

    db.commit()


def fmt_ts(sec: float) -> str:
    """123.4 → '00:02:03'"""
    s = int(sec)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def validate(db: sqlite3.Connection):
    cur = db.cursor()
    fn = cur.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    tn = cur.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    print(f"\n[검증]")
    print(f"  frames : {fn:,}개")
    print(f"  tracks : {tn:,}개")
    rows = cur.execute(
        "SELECT video, track, split, category, n_frames, start_sec, end_sec "
        "FROM tracks LIMIT 5"
    ).fetchall()
    print(f"\n[샘플 트랙 5개]")
    for r in rows:
        video, track, split, cat, n, s, e = r
        print(f"  {video[:30]:30}  {track:12}  "
              f"split={split:6}  cat={cat:12}  "
              f"frames={n:4d}  [{fmt_ts(s)}~{fmt_ts(e)}]")


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="인물 트랙 메타데이터 DB 구축")
    ap.add_argument("--fps",           type=float, default=DEFAULT_FPS,
                    help=f"기본 FPS (영상 감지 실패 시 사용, 기본={DEFAULT_FPS})")
    ap.add_argument("--validate-only", action="store_true", help="통계만 출력")
    ap.add_argument("--db",            type=Path, default=DB_PATH,
                    help=f"SQLite 출력 경로 (기본: {DB_PATH})")
    args = ap.parse_args()

    print("=" * 72)
    print("METADATA DB BUILD")
    print("=" * 72)

    args.db.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(args.db))
    db.executescript(SCHEMA)

    if args.validate_only:
        validate(db)
        db.close()
        return

    print(f"\n[스캔] {TRACK_ROOT}")
    frame_rows, track_rows = scan_tracks(default_fps=args.fps)
    print(f"  발견: frames={len(frame_rows):,}  tracks={len(track_rows):,}")

    if not frame_rows:
        print("[!] 크롭 없음. scvd_person_tracks_v1 디렉터리 확인 필요.")
        db.close()
        return

    print("\n[DB 삽입]")
    insert_all(db, frame_rows, track_rows)
    validate(db)
    db.close()

    print(f"\n[+] 완료: {args.db}")
    print("=" * 72)
    print("METADATA DB BUILD COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
