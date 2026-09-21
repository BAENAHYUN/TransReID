"""
group_tracklets.py
------------------
tracks.jsonl (프레임별 row) -> tracklets.json (SUSHI 입력용 track 단위)

사용법:
    python group_tracklets.py
    python group_tracklets.py --input outputs/video_tracks/tracks.jsonl \
                               --output outputs/video_tracks/tracklets.json
    python group_tracklets.py --stats-only   # 그룹핑 통계만 출력
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT  = Path("outputs/video_tracks/tracks.jsonl")
DEFAULT_OUTPUT = Path("outputs/video_tracks/tracklets.json")


# ─────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}:{lineno} JSON 파싱 실패: {e}") from e
    return rows


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# ─────────────────────────────────────────────
# 그룹핑
# ─────────────────────────────────────────────

def group_by_track(rows: list[dict]) -> dict[int, list[dict]]:
    """track_id 기준으로 프레임별 row를 묶는다."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        tid = row.get("track_id")
        if tid is None:
            raise KeyError(f"track_id 없는 row: {row}")
        groups[int(tid)].append(row)
    return dict(groups)


def build_tracklet(track_id: int, rows: list[dict]) -> dict:
    """
    프레임별 row 리스트 → SUSHI 입력 포맷 tracklet dict.

    SUSHI 권장 입력:
      track_id, class_name, start_frame, end_frame,
      frames, bboxes, confidences
    """
    # 프레임 순 정렬
    rows_sorted = sorted(rows, key=lambda r: int(r.get("frame_idx", r.get("frame", 0))))

    frames      : list[int]         = []
    bboxes      : list[list[float]] = []
    confidences : list[float]       = []

    class_name = "person"   # fallback

    for r in rows_sorted:
        fidx = int(r.get("frame_idx", r.get("frame", 0)))

        # bbox — xyxy 기대, xywh도 자동 변환
        bbox_raw = r.get("bbox") or r.get("xyxy") or r.get("tlbr")
        if bbox_raw is None:
            xywh = r.get("xywh") or r.get("tlwh")
            if xywh is None:
                raise KeyError(f"track_id={track_id} frame={fidx}: bbox 필드 없음")
            x, y, w, h = [float(v) for v in xywh]
            bbox_raw = [x, y, x + w, y + h]

        bbox = [float(v) for v in bbox_raw]
        if len(bbox) != 4:
            raise ValueError(f"track_id={track_id} frame={fidx}: bbox 길이 {len(bbox)} (기대 4)")

        conf = float(r.get("confidence", r.get("score", r.get("conf", 1.0))))
        cn   = str(r.get("class_name", r.get("label", class_name))).lower()
        class_name = cn   # 마지막으로 본 클래스 유지

        frames.append(fidx)
        bboxes.append(bbox)
        confidences.append(conf)

    if not frames:
        raise RuntimeError(f"track_id={track_id}: 유효한 프레임 없음")

    start_frame = frames[0]
    end_frame   = frames[-1]
    span        = end_frame - start_frame + 1
    missing     = span - len(frames)

    return {
        "track_id"    : track_id,
        "class_name"  : class_name,
        "start_frame" : start_frame,
        "end_frame"   : end_frame,
        "span"        : span,
        "rows"        : len(frames),
        "missing"     : missing,
        "frames"      : frames,
        "bboxes"      : bboxes,
        "confidences" : confidences,
    }


# ─────────────────────────────────────────────
# 통계 출력
# ─────────────────────────────────────────────

def print_stats(tracklets: list[dict]) -> None:
    print(f"\n{'─'*60}")
    print(f"  tracklets : {len(tracklets)}")
    total_rows    = sum(t["rows"]    for t in tracklets)
    total_missing = sum(t["missing"] for t in tracklets)
    print(f"  total rows    : {total_rows:,}")
    print(f"  total missing : {total_missing:,}")
    print(f"{'─'*60}")
    print(f"  {'track_id':>10}  {'rows':>6}  {'span':>6}  {'missing':>7}  {'miss%':>6}  class")
    print(f"  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*8}")
    for t in sorted(tracklets, key=lambda x: x["track_id"]):
        pct = t["missing"] / t["span"] * 100 if t["span"] > 0 else 0
        flag = " ⚠" if pct > 20 else ""
        print(
            f"  {t['track_id']:>10}  {t['rows']:>6,}  {t['span']:>6,}"
            f"  {t['missing']:>7,}  {pct:>5.1f}%  {t['class_name']}{flag}"
        )
    print(f"{'─'*60}\n")
    print("⚠ = missing 비율 20% 초과 (SUSHI long-term association 권장)\n")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="tracks.jsonl → tracklets.json (SUSHI 입력)")
    ap.add_argument("--input",      type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output",     type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--stats-only", action="store_true", help="통계만 출력하고 저장 안 함")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"[ERROR] 입력 파일 없음: {args.input}", file=sys.stderr)
        return 1

    print(f"읽는 중: {args.input}")
    rows = load_jsonl(args.input)
    print(f"  총 {len(rows):,} rows")

    groups = group_by_track(rows)
    print(f"  track_id 종류: {sorted(groups)}")

    tracklets = []
    for tid in sorted(groups):
        t = build_tracklet(tid, groups[tid])
        tracklets.append(t)

    print_stats(tracklets)

    if args.stats_only:
        print("--stats-only: 저장 생략")
        return 0

    # SUSHI가 읽는 최소 필드만 남긴 clean 버전도 함께 저장
    sushi_tracklets = [
        {
            "track_id"    : t["track_id"],
            "class_name"  : t["class_name"],
            "start_frame" : t["start_frame"],
            "end_frame"   : t["end_frame"],
            "frames"      : t["frames"],
            "bboxes"      : t["bboxes"],
            "confidences" : t["confidences"],
        }
        for t in tracklets
    ]

    # 전체 버전 (stats 포함)
    save_json(args.output, tracklets)
    print(f"저장: {args.output}  ({len(tracklets)} tracklets)")

    # SUSHI 전용 slim 버전
    sushi_path = args.output.with_stem(args.output.stem + "_sushi")
    save_json(sushi_path, sushi_tracklets)
    print(f"저장: {sushi_path}  (SUSHI 입력용 slim)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
