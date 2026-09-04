"""
search_and_rerank.py — IRRA 1차 검색 + Qwen3-VL 재순위 + 결과 폴더 저장

입력 모드:
  --query  "자연어 설명"        텍스트로 검색
  --image  path/to/img.jpg     이미지와 비슷한 인물 검색
  --video  path/to/clip.mp4    영상 속 인물과 비슷한 인물 검색

파이프라인:
  ① 쿼리 임베딩 계산 (IRRA용 + Qwen용)
  ② IRRA FAISS → 임계값 이상 후보 수집 (기본 top-200 중 irra≥0.42)
  ③ Qwen3-VL → 이미지 직접 보고 코사인 유사도 재계산 → qwen≥0.45 필터
  ④ 단일 프레임 트랙 제거 → 결과 폴더 저장 + 개수 출력

사용법:
  python src/RF-DETR/search_and_rerank.py --query "red short-sleeved t-shirt male"
  python src/RF-DETR/search_and_rerank.py --image data/query/suspect.jpg
  python src/RF-DETR/search_and_rerank.py --video data/query/clip.mp4 --video-frames 16
"""

import sys
import shutil
import argparse
import re
import tempfile
import numpy as np
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────────────────
_RFDETR = Path(__file__).parent
_ROOT   = _RFDETR.parents[1]          # TransReID 루트
sys.path.insert(0, str(_ROOT / "src" / "IRRA"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_RFDETR))  # RF-DETR 디렉토리 최우선

from forensic_video_search import (
    ForensicSearcher,
    encode_text,
    encode_image_irra,
    avg_normalize,
)
from qwen_vlm import (
    load_embedding_model,
    get_qwen_image_embedding,
    get_qwen_text_embedding,
)

# 감사 로그 (PostgreSQL 미기동 시 자동 비활성화)
try:
    from db_manager import AuditLogger
    _AUDIT_AVAILABLE = True
except ImportError:
    _AUDIT_AVAILABLE = False

OUTPUT_BASE = _ROOT / "search_results"


# ══════════════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════════════

def cosine_sim(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def frames_to_hms(frame_no: int, fps: float) -> str:
    total_sec = frame_no / fps
    h = int(total_sec // 3600)
    m = int((total_sec % 3600) // 60)
    s = int(total_sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_track_timerange(best_path: str, fps: float = 30.0):
    track_dir = Path(best_path).parent
    if not track_dir.exists():
        track_dir = _ROOT / track_dir
    if not track_dir.exists():
        return "??:??:??", "??:??:??", fps, 0

    frame_nums = []
    for jpg in track_dir.glob("*.jpg"):
        stem  = jpg.stem
        parts = stem.split("_")
        if len(parts) >= 2 and parts[0] == "frame" and parts[1].isdigit():
            frame_nums.append(int(parts[1]))

    if not frame_nums:
        return "??:??:??", "??:??:??", fps, 0

    mn, mx = min(frame_nums), max(frame_nums)
    return (
        frames_to_hms(mn, fps),
        frames_to_hms(mx, fps),
        fps,
        (0 if mn == mx else len(set(frame_nums))),
    )


def detect_fps(video_name: str, source: str = "video") -> float:
    try:
        import cv2
        candidates = []
        if source == "scvd":
            candidates += [
                _ROOT / "data" / "SCVD" / f"{video_name}.mp4",
                _ROOT / "data" / "SCVD" / f"{video_name}.avi",
            ]
        else:
            candidates += [
                _ROOT / "data" / "videos" / f"{video_name}.mp4",
                _ROOT / "data" / "videos" / f"{video_name}.avi",
                _ROOT / "data" / "videos" / f"{video_name}",
            ]
        for p in candidates:
            if p.exists():
                cap = cv2.VideoCapture(str(p))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps and fps > 0:
                    return fps
    except Exception:
        pass
    return 30.0


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s가-힣-]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text[:60]


def extract_video_frames(video_path: str, n_frames: int = 16) -> list:
    """동영상에서 n_frames장을 균등 간격으로 추출 → PIL Image 리스트."""
    import cv2
    from PIL import Image
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"영상에서 프레임을 읽을 수 없습니다: {video_path}")
    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    frames  = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


# ══════════════════════════════════════════════════════════════════════
# 쿼리 임베딩 계산
# ══════════════════════════════════════════════════════════════════════

def build_irra_emb(mode, searcher, query=None, image_path=None, pil_frames=None):
    """IRRA 512-dim 쿼리 임베딩 반환."""
    if mode == "text":
        return encode_text(searcher.model, query)
    elif mode == "image":
        return encode_image_irra(searcher.model, image_path)
    else:  # video
        embs = [encode_image_irra(searcher.model, f) for f in pil_frames]
        print(f"      IRRA: {len(embs)}프레임 임베딩 평균")
        return avg_normalize(embs)


def build_qwen_emb(mode, emb_model, query=None, image_path=None, pil_frames=None):
    """Qwen3-VL 쿼리 임베딩 반환."""
    if mode == "text":
        return get_qwen_text_embedding(emb_model, query)
    elif mode == "image":
        return get_qwen_image_embedding(emb_model, image_path)
    else:  # video — 프레임을 임시 파일로 저장 후 임베딩
        embs = []
        with tempfile.TemporaryDirectory() as tmp:
            for idx, pil in enumerate(pil_frames):
                p = Path(tmp) / f"frame_{idx:04d}.jpg"
                pil.save(str(p), quality=95)
                embs.append(get_qwen_image_embedding(emb_model, str(p)))
        print(f"      Qwen: {len(embs)}프레임 임베딩 평균")
        arr = np.vstack(embs)
        avg = arr.mean(axis=0)
        norm = np.linalg.norm(avg)
        return (avg / norm if norm > 0 else avg).astype("float32")


# ══════════════════════════════════════════════════════════════════════
# 메인 파이프라인
# ══════════════════════════════════════════════════════════════════════

def search_and_rerank(
    mode,
    query           = None,
    image_path      = None,
    video_path      = None,
    video_frames    = 16,
    irra_k          = 200,
    irra_threshold  = 0.42,
    qwen_threshold  = 0.45,
    out_dir         = None,
    investigator    = "anonymous",  # 수사관 이름 (감사 로그)
):
    pil_frames = None

    # ── 감사 로그 시작 ───────────────────────────────────────────────
    _audit_id = None
    _audit_logger = None
    if _AUDIT_AVAILABLE:
        try:
            _audit_logger = AuditLogger()
            _audit_id = _audit_logger.begin(
                investigator   = investigator,
                query_type     = mode,
                query_content  = query or image_path or video_path or "",
                irra_threshold = irra_threshold,
                qwen_threshold = qwen_threshold,
            )
        except Exception as _e:
            print(f"[감사로그] PostgreSQL 미연결 — 로컬 모드 ({_e})")

    # ── ① 라벨 & 사전 처리 ────────────────────────────────────────────
    if mode == "text":
        label     = query
        slug_base = slugify(query)
    elif mode == "image":
        label      = Path(image_path).name
        slug_base  = slugify(Path(image_path).stem)
        image_path = str(Path(image_path).resolve())
    else:  # video
        label     = Path(video_path).name
        slug_base = slugify(Path(video_path).stem)
        print(f"\n[0/3] 동영상 프레임 추출 ({video_frames}장) ...")
        pil_frames = extract_video_frames(video_path, n_frames=video_frames)
        print(f"      {len(pil_frames)}장 추출 완료")

    # ── ② IRRA 1차 검색 ───────────────────────────────────────────────
    print(f"\n[1/3] IRRA 검색 중 (top-{irra_k}, threshold≥{irra_threshold}) ...")
    print(f"      모드: {mode.upper()}  |  쿼리: {label}")
    searcher = ForensicSearcher()

    irra_emb = build_irra_emb(
        mode, searcher,
        query=query, image_path=image_path, pil_frames=pil_frames,
    )

    candidates_all = searcher._search_by_emb(irra_emb, top_k=irra_k, by_track=True)
    candidates     = [c for c in candidates_all if c["similarity"] >= irra_threshold]
    print(f"      {len(candidates_all)}개 검색 → irra≥{irra_threshold}: {len(candidates)}개")

    if not candidates:
        print("[!] IRRA 후보 없음. irra-threshold를 낮추거나 irra-k를 높여보세요.")
        return []

    # ── ③ Qwen3-VL 재순위 ─────────────────────────────────────────────
    print("[2/3] Qwen3-VL Embedding 모델 로드 중 ...")
    emb_model = load_embedding_model()

    print("      쿼리 임베딩 계산 ...")
    qwen_emb = build_qwen_emb(
        mode, emb_model,
        query=query, image_path=image_path, pil_frames=pil_frames,
    )

    print(f"      이미지 임베딩 계산 ({len(candidates)}개) ...")
    reranked = []
    for i, c in enumerate(candidates):
        img_path = Path(c.get("best_path") or c.get("path", ""))
        if not img_path.exists():
            img_path = _ROOT / img_path
        if not img_path.exists():
            continue

        qwen_sim = cosine_sim(qwen_emb, get_qwen_image_embedding(emb_model, str(img_path)))
        reranked.append({
            **c,
            "irra_score": c["similarity"],
            "qwen_score": round(qwen_sim, 4),
            "best_path":  str(img_path),
        })
        print(f"  [{i+1:>3}/{len(candidates)}] irra={c['similarity']:.4f}  "
              f"qwen={qwen_sim:.4f}  {c['track']}", end="\r")
    print()

    reranked.sort(key=lambda x: x["qwen_score"], reverse=True)
    passed = [r for r in reranked if r["qwen_score"] >= qwen_threshold]
    for i, r in enumerate(passed):
        r["rank"] = i + 1
    print(f"      qwen≥{qwen_threshold} 통과: {len(passed)}개")

    # ── ④ 결과 저장 ───────────────────────────────────────────────────
    print("[3/3] 결과 저장 중 ...")
    if out_dir:
        out = Path(out_dir)
    else:
        out = OUTPUT_BASE / f"{slug_base}_irra{irra_k}_th{qwen_threshold}"
    out.mkdir(parents=True, exist_ok=True)

    fps_cache = {}
    log_lines = [
        f"모드    : {mode.upper()}",
        f"쿼리    : {label}",
        f"IRRA k  : {irra_k}  threshold: {irra_threshold}",
        f"Qwen th : {qwen_threshold}",
        "",
        f"{'순위':<4} {'Qwen':>6} {'IRRA':>6}  시간범위              영상/트랙",
        "-" * 70,
    ]

    copied = 0
    for r in passed:
        src = Path(r["best_path"])
        if not src.exists():
            continue

        vid = r["video"]
        if vid not in fps_cache:
            fps_cache[vid] = detect_fps(vid, r.get("source", "video"))
        fps = fps_cache[vid]

        t_start, t_end, _, n_frames = get_track_timerange(str(src), fps)
        if n_frames <= 1:
            continue   # 단일 프레임 제거

        time_range = f"{t_start}~{t_end}"
        fname = (
            f"{r['rank']:02d}_"
            f"qwen{r['qwen_score']:.4f}_"
            f"irra{r['irra_score']:.4f}_"
            f"{r['video']}_{r['track']}{src.suffix}"
        )
        shutil.copy2(src, out / fname)
        copied += 1

        log_lines.append(
            f"{r['rank']:<4} {r['qwen_score']:>6.4f} {r['irra_score']:>6.4f}"
            f"  [{time_range}]  {r['video']}/{r['track']}"
        )
        print(f"  {r['rank']:>3}위  qwen={r['qwen_score']:.4f}  irra={r['irra_score']:.4f}"
              f"  [{time_range}]  {r['video']}/{r['track']}")

    log_lines += ["", f"총 {copied}개 저장"]
    (out / "_result.txt").write_text("\n".join(log_lines), encoding="utf-8")

    print(f"\n{'─'*55}")
    print(f"  쿼리     : {label}")
    print(f"  IRRA 후보: {len(candidates)}개")
    print(f"  Qwen 통과: {len(passed)}개  (qwen≥{qwen_threshold})")
    print(f"  최종 저장: {copied}개  (단일프레임 제거 후)")
    print(f"  저장 위치: {out.resolve()}")
    print(f"{'─'*55}")
    # ── 감사 로그 완료 ──────────────────────────────────────────────
    if _audit_logger and _audit_id:
        try:
            _audit_logger.finish(
                audit_id        = _audit_id,
                irra_candidates = len(candidates),
                qwen_passed     = len(passed),
                final_saved     = copied,
                result_dir      = str(out.resolve()),
            )
            _audit_logger.close()
        except Exception:
            pass

    return passed


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="IRRA + Qwen3-VL 재순위 검색")

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--query",  "-q", help="자연어 텍스트 쿼리")
    grp.add_argument("--image",  "-I", help="쿼리 이미지 파일 경로")
    grp.add_argument("--video",  "-V", help="쿼리 동영상 파일 경로")

    ap.add_argument("--video-frames",   type=int,   default=16,   help="동영상에서 추출할 프레임 수 (기본 16)")
    ap.add_argument("--irra-k",   "-i", type=int,   default=200,  help="IRRA 1차 후보 수 (기본 200)")
    ap.add_argument("--irra-threshold", type=float, default=0.42, help="IRRA 최소 점수 (기본 0.42)")
    ap.add_argument("--qwen-threshold", type=float, default=0.45, help="Qwen 최소 점수 (기본 0.45)")
    ap.add_argument("--out",            default=None,             help="결과 저장 폴더")
    ap.add_argument("--investigator",   default="anonymous",      help="수사관 이름 (감사 로그)")

    args = ap.parse_args()

    if args.query:
        mode = "text"
    elif args.image:
        mode = "image"
    else:
        mode = "video"

    search_and_rerank(
        mode           = mode,
        query          = args.query,
        image_path     = args.image,
        video_path     = args.video,
        video_frames   = args.video_frames,
        irra_k         = args.irra_k,
        irra_threshold = args.irra_threshold,
        qwen_threshold = args.qwen_threshold,
        out_dir        = args.out,
        investigator   = args.investigator,
    )
