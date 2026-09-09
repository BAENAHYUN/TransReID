"""
object_crop_quality.py

객체 track crop 품질을 평가하고, 각 track에서 DB 임베딩에 사용할 대표 crop을 고른다.

목표
----
1) 해상도가 너무 작은 crop 제거
2) 심하게 흐린 crop 제거
3) 원본 프레임 bbox metadata가 있으면 person-object overlap 계산
4) 각 object track에서 quality 상위 K장 선택
5) 원본 파일은 삭제하지 않고 CSV/JSON으로 선택 결과만 기록

기본 입력 구조
--------------
data/scvd_object_tracks_v1/
  <video_stem>/
    <label>_<track_id>/
      frame_00000030_0.6489.jpg

출력
----
outputs/object_crop_quality/
  crop_quality.csv
  selected_crops.json
  rejected_crops.json

주의
----
- person overlap은 원본 full-frame bbox metadata가 있을 때만 계산한다.
- metadata가 없으면 overlap을 임의 추정하지 않고 "unknown"으로 둔다.
- threshold는 프로젝트 데이터에 맞춰 튜닝해야 하는 시작값이다.

실행
----
python .\src\RF-DETR\object_crop_quality.py

예:
python .\src\RF-DETR\object_crop_quality.py `
  --top-k 5 `
  --min-width 64 `
  --min-height 64 `
  --blur-threshold 60 `
  --person-overlap-soft 0.15 `
  --person-overlap-hard 0.35
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parents[2]

DEFAULT_OBJECT_ROOT = ROOT_DIR / "data" / "scvd_object_tracks_v1"
DEFAULT_OUTPUT = ROOT_DIR / "outputs" / "object_crop_quality"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FRAME_RE = re.compile(r"frame_(\d+)", re.IGNORECASE)

PERSON_ROOT_CANDIDATES = (
    ROOT_DIR / "data" / "scvd_person_tracks_v2",
    ROOT_DIR / "data" / "scvd_person_tracks_v1",
    ROOT_DIR / "data" / "person_tracks",
)

# JSON/JSONL에서 자주 쓰는 키 후보
FRAME_KEYS = ("frame_idx", "frame", "frame_id", "frame_index")
BBOX_KEYS = ("bbox", "xyxy", "box")
LABEL_KEYS = ("label", "class_name", "class", "category", "name")
TRACK_KEYS = ("track_id", "tracker_id", "id")


@dataclass
class CropScore:
    path: str
    video: str
    track: str
    label: str
    frame_idx: int

    width: int
    height: int
    area: int

    blur_var: float
    resolution_score: float
    sharpness_score: float

    person_overlap: Optional[float]
    overlap_score: float

    quality_score: float
    accepted: bool
    reasons: List[str]


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def parse_frame_idx(path: Path) -> int:
    m = FRAME_RE.search(path.stem)
    return int(m.group(1)) if m else -1


def parse_label(track_name: str) -> str:
    # bench_0001 -> bench
    m = re.match(r"(.+)_\d+$", track_name)
    return m.group(1) if m else track_name


def laplacian_variance(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def resolution_score(w: int, h: int, target_side: int = 224) -> float:
    """
    짧은 변 기준.
    224 이상이면 1.0, 그보다 작으면 선형 감소.
    """
    return clamp01(min(w, h) / max(target_side, 1))


def sharpness_score(blur_var: float, threshold: float) -> float:
    """
    blur threshold를 0.5점 근처 기준으로 쓰는 완만한 score.
    """
    if threshold <= 0:
        return 1.0

    # threshold -> 0.5, 2*threshold -> 약 0.67
    x = blur_var / threshold
    return clamp01(x / (1.0 + x))


def bbox_xyxy(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if value is None:
        return None

    if isinstance(value, dict):
        # x1/y1/x2/y2
        if all(k in value for k in ("x1", "y1", "x2", "y2")):
            return (
                float(value["x1"]),
                float(value["y1"]),
                float(value["x2"]),
                float(value["y2"]),
            )
        # left/top/right/bottom
        if all(k in value for k in ("left", "top", "right", "bottom")):
            return (
                float(value["left"]),
                float(value["top"]),
                float(value["right"]),
                float(value["bottom"]),
            )

    if isinstance(value, (list, tuple)) and len(value) >= 4:
        x1, y1, x2, y2 = map(float, value[:4])
        return (x1, y1, x2, y2)

    return None


def get_any(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def frame_of(record: Dict[str, Any]) -> Optional[int]:
    v = get_any(record, FRAME_KEYS)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def label_of(record: Dict[str, Any]) -> Optional[str]:
    v = get_any(record, LABEL_KEYS)
    return None if v is None else str(v).lower()


def bbox_of(record: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    for k in BBOX_KEYS:
        if k in record:
            b = bbox_xyxy(record[k])
            if b is not None:
                return b

    # nested detection
    for key in ("detection", "det", "object", "person"):
        child = record.get(key)
        if isinstance(child, dict):
            b = bbox_of(child)
            if b is not None:
                return b

    return None


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []

    try:
        if path.suffix.lower() == ".jsonl":
            out = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        out.append(obj)
                    elif isinstance(obj, list):
                        out.extend(x for x in obj if isinstance(x, dict))
            return out

        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            for key in ("records", "detections", "tracks", "frames", "items"):
                if isinstance(obj.get(key), list):
                    return [x for x in obj[key] if isinstance(x, dict)]
            return [obj]
    except Exception:
        return []

    return []


def find_metadata_files(video_dir: Path) -> List[Path]:
    candidates = []
    for name in (
        "tracks.jsonl",
        "detections.jsonl",
        "objects.jsonl",
        "tracks.json",
        "detections.json",
        "metadata.json",
    ):
        p = video_dir / name
        if p.is_file():
            candidates.append(p)
    return candidates


def build_bbox_index(video_dir: Path) -> Dict[int, List[Dict[str, Any]]]:
    """
    object video directory의 metadata에서 frame별 detection index를 만든다.
    """
    out: Dict[int, List[Dict[str, Any]]] = {}
    for meta in find_metadata_files(video_dir):
        for r in load_json_records(meta):
            fi = frame_of(r)
            if fi is None:
                continue
            out.setdefault(fi, []).append(r)
    return out


def find_person_video_dir(video_stem: str, explicit_root: Optional[Path]) -> Optional[Path]:
    roots = [explicit_root] if explicit_root else list(PERSON_ROOT_CANDIDATES)

    for root in roots:
        if root is None or not root.exists():
            continue
        direct = root / video_stem
        if direct.exists():
            return direct

        # 이름이 조금 다를 경우 한 번 더
        hits = list(root.glob(f"{video_stem}*"))
        for h in hits:
            if h.is_dir():
                return h
    return None


def build_person_bbox_index(
    video_stem: str,
    explicit_root: Optional[Path],
) -> Dict[int, List[Tuple[float, float, float, float]]]:
    d = find_person_video_dir(video_stem, explicit_root)
    if d is None:
        return {}

    frame_boxes: Dict[int, List[Tuple[float, float, float, float]]] = {}

    for meta in find_metadata_files(d):
        for r in load_json_records(meta):
            fi = frame_of(r)
            if fi is None:
                continue

            lbl = label_of(r)
            if lbl is not None and lbl != "person":
                continue

            b = bbox_of(r)
            if b is not None:
                frame_boxes.setdefault(fi, []).append(b)

    return frame_boxes


def intersection_over_object(
    obj: Tuple[float, float, float, float],
    person: Tuple[float, float, float, float],
) -> float:
    ox1, oy1, ox2, oy2 = obj
    px1, py1, px2, py2 = person

    ix1 = max(ox1, px1)
    iy1 = max(oy1, py1)
    ix2 = min(ox2, px2)
    iy2 = min(oy2, py2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    obj_area = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
    if obj_area <= 0:
        return 0.0

    return clamp01(inter / obj_area)


def match_object_bbox(
    frame_records: List[Dict[str, Any]],
    label: str,
    track_name: str,
) -> Optional[Tuple[float, float, float, float]]:
    """
    metadata가 있으면 현재 track과 가장 잘 맞는 object bbox를 찾는다.
    우선 track_id / label을 활용하고, 그래도 하나면 그 bbox 사용.
    """
    label = label.lower()

    track_id_guess = None
    m = re.search(r"_(\d+)$", track_name)
    if m:
        track_id_guess = m.group(1)

    labeled = []
    for r in frame_records:
        lbl = label_of(r)
        if lbl is not None and lbl != label:
            continue

        b = bbox_of(r)
        if b is None:
            continue

        tid = get_any(r, TRACK_KEYS)
        if track_id_guess is not None and tid is not None:
            if str(tid).lstrip("0") == track_id_guess.lstrip("0"):
                return b

        labeled.append(b)

    if len(labeled) == 1:
        return labeled[0]

    return None


def compute_quality(
    crop_path: Path,
    object_bbox: Optional[Tuple[float, float, float, float]],
    person_boxes: List[Tuple[float, float, float, float]],
    *,
    min_width: int,
    min_height: int,
    blur_threshold: float,
    overlap_soft: float,
    overlap_hard: float,
) -> CropScore:
    img = cv2.imread(str(crop_path))
    if img is None:
        return CropScore(
            path=str(crop_path),
            video=crop_path.parents[1].name,
            track=crop_path.parent.name,
            label=parse_label(crop_path.parent.name),
            frame_idx=parse_frame_idx(crop_path),
            width=0,
            height=0,
            area=0,
            blur_var=0.0,
            resolution_score=0.0,
            sharpness_score=0.0,
            person_overlap=None,
            overlap_score=0.0,
            quality_score=0.0,
            accepted=False,
            reasons=["image_read_failed"],
        )

    h, w = img.shape[:2]
    blur = laplacian_variance(img)
    res_score = resolution_score(w, h)
    sharp_score = sharpness_score(blur, blur_threshold)

    overlap: Optional[float] = None
    if object_bbox is not None and person_boxes:
        overlap = max(intersection_over_object(object_bbox, p) for p in person_boxes)

    if overlap is None:
        overlap_score = 1.0
    elif overlap <= overlap_soft:
        overlap_score = 1.0
    elif overlap >= overlap_hard:
        overlap_score = 0.0
    else:
        overlap_score = 1.0 - (
            (overlap - overlap_soft) / max(overlap_hard - overlap_soft, 1e-9)
        )

    reasons: List[str] = []
    accepted = True

    if w < min_width:
        accepted = False
        reasons.append(f"width<{min_width}")

    if h < min_height:
        accepted = False
        reasons.append(f"height<{min_height}")

    if blur < blur_threshold:
        reasons.append(f"blur<{blur_threshold:g}")

    if overlap is not None:
        if overlap >= overlap_hard:
            accepted = False
            reasons.append(f"person_overlap>={overlap_hard:.2f}")
        elif overlap >= overlap_soft:
            reasons.append(f"person_overlap>={overlap_soft:.2f}")

    # 정확도 우선:
    # overlap을 가장 크게, 해상도/선명도를 보조로 둔다.
    # metadata가 없어 overlap unknown이면 1.0으로 가정하되
    # 결과 CSV에서 unknown임을 명확히 남긴다.
    quality = (
        0.50 * overlap_score
        + 0.30 * res_score
        + 0.20 * sharp_score
    )

    # blur가 threshold보다 낮으면 완전 reject하지는 않지만 감점
    if blur < blur_threshold:
        quality *= 0.75

    if not accepted:
        quality *= 0.25

    return CropScore(
        path=str(crop_path),
        video=crop_path.parents[1].name,
        track=crop_path.parent.name,
        label=parse_label(crop_path.parent.name),
        frame_idx=parse_frame_idx(crop_path),
        width=w,
        height=h,
        area=w * h,
        blur_var=blur,
        resolution_score=res_score,
        sharpness_score=sharp_score,
        person_overlap=overlap,
        overlap_score=overlap_score,
        quality_score=float(quality),
        accepted=accepted,
        reasons=reasons,
    )


def write_csv(rows: List[CropScore], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "path",
        "video",
        "track",
        "label",
        "frame_idx",
        "width",
        "height",
        "area",
        "blur_var",
        "resolution_score",
        "sharpness_score",
        "person_overlap",
        "overlap_score",
        "quality_score",
        "accepted",
        "reasons",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for row in rows:
            d = asdict(row)
            d["reasons"] = ";".join(row.reasons)
            if d["person_overlap"] is None:
                d["person_overlap"] = ""
            w.writerow(d)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="object track crop quality 평가 + 대표 crop 선택"
    )
    ap.add_argument(
        "--object-root",
        type=Path,
        default=DEFAULT_OBJECT_ROOT,
    )
    ap.add_argument(
        "--person-root",
        type=Path,
        default=None,
        help="person track metadata root. 생략하면 흔한 경로를 자동 탐색",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-width", type=int, default=64)
    ap.add_argument("--min-height", type=int, default=64)
    ap.add_argument("--blur-threshold", type=float, default=60.0)
    ap.add_argument("--person-overlap-soft", type=float, default=0.15)
    ap.add_argument("--person-overlap-hard", type=float, default=0.35)
    ap.add_argument(
        "--copy-selected",
        action="store_true",
        help="선택된 crop을 outputs/object_crop_quality/selected/로 복사",
    )
    args = ap.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k는 1 이상이어야 합니다.")

    obj_root = args.object_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not obj_root.exists():
        raise SystemExit(f"object root가 없습니다: {obj_root}")

    print("=" * 72)
    print("OBJECT CROP QUALITY")
    print(f"Object root : {obj_root}")
    print(f"Output      : {out_dir}")
    print(f"Top-K/track : {args.top_k}")
    print(
        f"Overlap     : soft={args.person_overlap_soft:.2f}, "
        f"hard={args.person_overlap_hard:.2f}"
    )
    print("=" * 72)

    all_rows: List[CropScore] = []
    selected: Dict[str, List[Dict[str, Any]]] = {}
    rejected: List[Dict[str, Any]] = []

    video_dirs = sorted(p for p in obj_root.iterdir() if p.is_dir())

    total_tracks = 0
    metadata_overlap_count = 0

    for vi, video_dir in enumerate(video_dirs, 1):
        object_meta = build_bbox_index(video_dir)
        person_meta = build_person_bbox_index(video_dir.name, args.person_root)

        track_dirs = sorted(p for p in video_dir.iterdir() if p.is_dir())

        print(
            f"[{vi:3d}/{len(video_dirs):3d}] {video_dir.name} "
            f"tracks={len(track_dirs)} "
            f"object_meta_frames={len(object_meta)} "
            f"person_meta_frames={len(person_meta)}"
        )

        for track_dir in track_dirs:
            total_tracks += 1
            label = parse_label(track_dir.name)

            crops = sorted(
                p for p in track_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )

            track_rows: List[CropScore] = []

            for crop in crops:
                frame_idx = parse_frame_idx(crop)

                obj_bbox = match_object_bbox(
                    object_meta.get(frame_idx, []),
                    label,
                    track_dir.name,
                )
                pboxes = person_meta.get(frame_idx, [])

                row = compute_quality(
                    crop,
                    obj_bbox,
                    pboxes,
                    min_width=args.min_width,
                    min_height=args.min_height,
                    blur_threshold=args.blur_threshold,
                    overlap_soft=args.person_overlap_soft,
                    overlap_hard=args.person_overlap_hard,
                )

                if row.person_overlap is not None:
                    metadata_overlap_count += 1

                all_rows.append(row)
                track_rows.append(row)

            # accepted 우선, quality 내림차순
            ranked = sorted(
                track_rows,
                key=lambda r: (
                    1 if r.accepted else 0,
                    r.quality_score,
                    r.blur_var,
                    r.area,
                ),
                reverse=True,
            )

            good = [r for r in ranked if r.accepted]
            chosen = good[: args.top_k]

            key = f"{video_dir.name}/{track_dir.name}"
            selected[key] = [asdict(r) for r in chosen]

            chosen_paths = {r.path for r in chosen}
            for r in track_rows:
                if r.path not in chosen_paths:
                    rejected.append(asdict(r))

    csv_path = out_dir / "crop_quality.csv"
    selected_path = out_dir / "selected_crops.json"
    rejected_path = out_dir / "rejected_crops.json"

    write_csv(all_rows, csv_path)

    selected_path.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rejected_path.write_text(
        json.dumps(rejected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.copy_selected:
        import shutil

        selected_root = out_dir / "selected"
        for key, items in selected.items():
            safe_key = key.replace("/", "__").replace("\\", "__")
            dst_dir = selected_root / safe_key
            dst_dir.mkdir(parents=True, exist_ok=True)

            for item in items:
                src = Path(item["path"])
                if src.is_file():
                    shutil.copy2(src, dst_dir / src.name)

    accepted_count = sum(1 for r in all_rows if r.accepted)
    selected_count = sum(len(v) for v in selected.values())
    unknown_overlap = sum(1 for r in all_rows if r.person_overlap is None)

    print()
    print("=" * 72)
    print("COMPLETE")
    print(f"Tracks                : {total_tracks:,}")
    print(f"Crops                 : {len(all_rows):,}")
    print(f"Accepted              : {accepted_count:,}")
    print(f"Selected for embedding: {selected_count:,}")
    print(f"Overlap measured      : {metadata_overlap_count:,}")
    print(f"Overlap unknown       : {unknown_overlap:,}")
    print()
    print(f"CSV      : {csv_path}")
    print(f"Selected : {selected_path}")
    print(f"Rejected : {rejected_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
