"""
ingest_scvd_v2.py — SCVD 전체 인제스트 파이프라인 오케스트레이터

단계:
  1) scvd_person_track_v2.py       — 인물 트랙 크롭 추출
  2) embed_scvd_person_pass1_v2.py — SigLIP2(768) + IRRA(512) → Qdrant + PostgreSQL
  3) embed_scvd_person_pass2_v2.py — SOLIDER(1024) 업데이트 → Qdrant
  4) embed_scvd_person_pass2_v2.py --validate-only  — 통계 / 검증

  * 객체 인제스트: --with-objects 플래그 (별도 구현 필요)

사용법:
  python ingest_scvd_v2.py                       # 전체 파이프라인
  python ingest_scvd_v2.py --test                # smoke: 영상 3개, 크롭 10개
  python ingest_scvd_v2.py --force               # 기존 레코드 재삽입
  python ingest_scvd_v2.py --max-videos 5        # 영상 5개만
  python ingest_scvd_v2.py --validate-only       # 통계/검증만
  python ingest_scvd_v2.py --skip-tracks         # 크롭 추출 건너뜀
  python ingest_scvd_v2.py --skip-embed          # 임베딩 건너뜀 (검증만)
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
HERE     = Path(__file__).resolve().parent


def run(name: str, args: list[str] | None = None):
    cmd = [sys.executable, "-u", str(HERE / name)]
    if args:
        cmd.extend(args)

    print("\n" + "#" * 80)
    print("RUN:", " ".join(cmd))
    print("#" * 80)

    subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)


def main():
    ap = argparse.ArgumentParser(description="SCVD 인제스트 파이프라인 (Qdrant + PostgreSQL 백엔드)")
    ap.add_argument("--max-videos",    type=int, help="처리 영상 수 제한 (디버그)")
    ap.add_argument("--max-crops",     type=int, help="임베딩 크롭 수 제한 (디버그)")
    ap.add_argument("--test",          action="store_true",
                    help="smoke test: 영상 3개, 크롭 10개")
    ap.add_argument("--force",         action="store_true",
                    help="기존 데이터 무시하고 전체 재삽입")
    ap.add_argument("--with-objects",  action="store_true",
                    help="객체 인제스트 포함")
    ap.add_argument("--skip-tracks",   action="store_true",
                    help="트랙 추출 건너뜀 (이미 완료된 경우)")
    ap.add_argument("--skip-embed",    action="store_true",
                    help="임베딩 건너뜀 (검증만)")
    ap.add_argument("--validate-only", action="store_true",
                    help="pass2 검증/통계만 실행 (트랙/임베딩 건너뜀)")
    args = ap.parse_args()

    max_videos = args.max_videos
    max_crops  = args.max_crops

    if args.test:
        max_videos = max_videos or 3
        max_crops  = max_crops  or 10

    video_args: list[str] = []
    if max_videos:
        video_args += ["--max-videos", str(max_videos)]
    if args.force:
        video_args.append("--force")

    crop_args: list[str] = []     # pass1 전용 (--force 포함)
    solider_args: list[str] = []  # pass2 전용 (--force 없음, 항상 update)
    if max_crops:
        crop_args    += ["--max-crops", str(max_crops)]
        solider_args += ["--max-crops", str(max_crops)]
    if args.force:
        crop_args.append("--force")

    print("=" * 80)
    print("SCVD INGEST V2  (Qdrant 벡터 + PostgreSQL 메타데이터)")
    print("정책: Train ONLY / Test 보류 / 안전한 재실행(이어하기 지원)")
    print("=" * 80)

    if args.validate_only:
        run("embed_scvd_person_pass2_v2.py", ["--validate-only"])
        return

    # ── Step 1: 인물 트랙 크롭 추출 ──────────────────────────────
    if not args.skip_tracks:
        run("scvd_person_track_v2.py", video_args)
    else:
        print("\n[SKIP] 트랙 추출 (--skip-tracks)")

    # ── Step 2: 객체 인제스트 (옵션) ─────────────────────────────
    if args.with_objects:
        run("ingest_scvd_objects_v2.py", video_args)
    else:
        print("\n[SKIP] 객체 인제스트 (--with-objects 로 활성화)")

    # ── Step 3: SigLIP2+IRRA → Qdrant + PostgreSQL ───────────────
    if not args.skip_embed:
        run("embed_scvd_person_pass1_v2.py", crop_args)
    else:
        print("\n[SKIP] SigLIP2+IRRA 임베딩 (--skip-embed)")

    # ── Step 4: SOLIDER → Qdrant update_vectors ──────────────────
    if not args.skip_embed:
        run("embed_scvd_person_pass2_v2.py", solider_args)
    else:
        print("\n[SKIP] SOLIDER 업데이트 (--skip-embed)")

    # ── Step 5: 검증 / 통계 출력 ─────────────────────────────────
    run("embed_scvd_person_pass2_v2.py", ["--validate-only"])

    print("\n" + "=" * 80)
    print("SCVD INGEST V2 COMPLETE")
    print("  - 인물 벡터: Qdrant forensic_person_local (로컬 파일 DB)")
    print("    siglip  768-dim (SigLIP2)")
    print("    irra    512-dim (IRRA)")
    print("    solider 1024-dim (SOLIDER)")
    print("  - 메타데이터: PostgreSQL (frames + tracks)")
    if not args.with_objects:
        print("  - 객체 벡터: 건너뜀 (--with-objects 참조)")
    print("=" * 80)


if __name__ == "__main__":
    main()
