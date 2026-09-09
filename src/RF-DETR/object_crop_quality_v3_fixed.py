"""
object_crop_quality_v3.py

객체 crop 내부에서 RF-DETR로 person을 다시 검출해 contamination을 계산하고,
해상도/blur와 함께 각 object track의 대표 crop Top-K를 선택한다.

핵심:
    object crop
      -> RF-DETR person 재검출
      -> person union area / crop area
      -> contamination score
      -> blur/resolution score
      -> track별 Top-K 선택

중요:
- 원본 crop은 삭제하지 않는다.
- person overlap은 원본 full-frame metadata가 없어도 계산 가능하다.
- DINOv2는 이 단계의 주체가 아니다. 이 파일은 crop 품질 필터다.
- threshold는 프로젝트 데이터에 맞춰 튜닝해야 하는 시작값이다.

기본 입력:
    data/scvd_object_tracks_v1/<video>/<label_track>/frame_*.jpg

출력:
    outputs/object_crop_quality_v3/
      crop_quality_v3.csv
      selected_crops_v3.json
      rejected_crops_v3.json
      person_contamination_cache.json

실행:
    python .\src\RF-DETR\object_crop_quality_v3.py

추천 시작값:
    python .\src\RF-DETR\object_crop_quality_v3.py `
      --top-k 5 `
      --min-width 64 `
      --min-height 64 `
      --blur-threshold 60 `
      --person-conf 0.35 `
      --person-overlap-soft 0.15 `
      --person-overlap-hard 0.35

RF-DETR 로딩:
- 기본은 rfdetr 패키지의 RFDETRBase를 사용한다.
- 프로젝트에서 다른 checkpoint를 쓰면 --weights 로 지정할 수 있다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parents[2]

DEFAULT_OBJECT_ROOT = ROOT_DIR / "data" / "scvd_object_tracks_v1"
DEFAULT_OUTPUT = ROOT_DIR / "outputs" / "object_crop_quality_v3_fixed"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_RE = re.compile(r"frame_(\d+)", re.IGNORECASE)


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

    person_count: int
    person_overlap: float
    person_max_conf: float
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
    m = re.match(r"(.+)_\d+$", track_name)
    return m.group(1) if m else track_name


def laplacian_variance(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def resolution_score(w: int, h: int, target_side: int = 224) -> float:
    return clamp01(min(w, h) / max(target_side, 1))


def sharpness_score(blur_var: float, threshold: float) -> float:
    if threshold <= 0:
        return 1.0
    x = blur_var / threshold
    return clamp01(x / (1.0 + x))


def union_area_xyxy(
    boxes: Sequence[Tuple[float, float, float, float]],
    width: int,
    height: int,
) -> float:
    """
    crop 내부 person bbox들의 union area를 계산한다.
    단순 area sum은 서로 겹치는 person bbox를 중복 계산할 수 있으므로 union 사용.
    """
    if not boxes or width <= 0 or height <= 0:
        return 0.0

    # crop 해상도가 작으므로 uint8 mask 방식이 단순하고 정확하다.
    mask = np.zeros((height, width), dtype=np.uint8)

    for x1, y1, x2, y2 in boxes:
        ix1 = max(0, min(width, int(math.floor(x1))))
        iy1 = max(0, min(height, int(math.floor(y1))))
        ix2 = max(0, min(width, int(math.ceil(x2))))
        iy2 = max(0, min(height, int(math.ceil(y2))))

        if ix2 > ix1 and iy2 > iy1:
            mask[iy1:iy2, ix1:ix2] = 1

    return float(mask.sum())


class RFDETRPersonDetector:
    """
    rfdetr 패키지를 이용해 object crop 안의 person만 검출한다.

    model.predict(PIL.Image, threshold=...)가 supervision.Detections 또는
    유사 객체를 반환하는 일반적인 RF-DETR API를 기준으로 한다.
    """

    def __init__(
        self,
        conf: float = 0.35,
        weights: Optional[str] = None,
        model_type: str = "base",
    ) -> None:
        self.conf = float(conf)

        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "unknown"

        print(f"[*] RF-DETR person detector 로드 중 (device={self.device})")

        try:
            import rfdetr
        except ImportError as e:
            raise RuntimeError(
                "rfdetr 패키지를 찾을 수 없습니다. "
                "현재 프로젝트에서 RF-DETR를 쓰는 venv가 맞는지 확인하세요."
            ) from e

        mt = model_type.lower().strip()
        class_candidates = {
            "nano": ["RFDETRNano"],
            "small": ["RFDETRSmall"],
            "medium": ["RFDETRMedium"],
            "base": ["RFDETRBase"],
            "large": ["RFDETRLarge"],
        }.get(mt, ["RFDETRBase"])

        model_cls = None
        for name in class_candidates:
            model_cls = getattr(rfdetr, name, None)
            if model_cls is not None:
                break

        if model_cls is None:
            available = [n for n in dir(rfdetr) if n.startswith("RFDETR")]
            raise RuntimeError(
                f"RF-DETR model class를 찾지 못했습니다. 요청 model_type={model_type}, "
                f"사용 가능 후보={available}"
            )

        kwargs: Dict[str, Any] = {}
        if weights:
            # 버전에 따라 parameter명이 다를 수 있어 아래 두 방식 순차 시도
            try:
                self.model = model_cls(pretrain_weights=weights)
            except TypeError:
                try:
                    self.model = model_cls(weights=weights)
                except TypeError:
                    self.model = model_cls()
                    print(
                        "[!] 현재 rfdetr 버전이 weights/pretrain_weights 인자를 받지 않습니다. "
                        "기본 checkpoint로 로드했습니다."
                    )
        else:
            self.model = model_cls(**kwargs)

        print(f"[+] RF-DETR 준비 완료: {model_cls.__name__}")

    @staticmethod
    def _extract_detections(pred: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        supervision.Detections 형태 우선.
        반환: xyxy (N,4), confidence (N,), class_id (N,)
        """
        xyxy = getattr(pred, "xyxy", None)
        conf = getattr(pred, "confidence", None)
        class_id = getattr(pred, "class_id", None)

        if xyxy is None:
            raise RuntimeError(
                "RF-DETR predict 결과에서 xyxy를 찾지 못했습니다. "
                f"반환 타입={type(pred)!r}"
            )

        xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)

        if conf is None:
            conf = np.ones((len(xyxy),), dtype=np.float32)
        else:
            conf = np.asarray(conf, dtype=np.float32).reshape(-1)

        if class_id is None:
            raise RuntimeError(
                "RF-DETR predict 결과에서 class_id를 찾지 못했습니다."
            )
        class_id = np.asarray(class_id).reshape(-1)

        return xyxy, conf, class_id

    def detect_person(
        self,
        image_rgb: np.ndarray,
    ) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
        pil = Image.fromarray(image_rgb)

        pred = self.model.predict(
            pil,
            threshold=self.conf,
        )

        xyxy, confs, class_ids = self._extract_detections(pred)

        # RF-DETR COCO pretrained 체크포인트는 sparse COCO category id를 사용하며
        # person은 class_id=1이다. 다만 fine-tuned/custom checkpoint에서는
        # class_id 체계가 달라질 수 있으므로 class_name metadata를 우선 사용한다.
        class_names = None
        data = getattr(pred, "data", None)
        if isinstance(data, dict):
            raw_names = data.get("class_name")
            if raw_names is not None:
                class_names = np.asarray(raw_names).reshape(-1)

        if class_names is not None and len(class_names) == len(xyxy):
            keep = np.asarray(
                [str(name).strip().lower() == "person" for name in class_names],
                dtype=bool,
            )
        else:
            # COCO pretrained RF-DETR canonical category id: person = 1
            keep = class_ids.astype(int) == 1

        boxes = [
            tuple(map(float, b))
            for b in xyxy[keep]
        ]
        scores = [
            float(x)
            for x in confs[keep]
        ]

        return boxes, scores


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def save_cache(cache: Dict[str, Dict[str, Any]], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def cache_key(path: Path) -> str:
    st = path.stat()
    return f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"


def contamination_for_crop(
    crop_path: Path,
    image_bgr: np.ndarray,
    detector: RFDETRPersonDetector,
    cache: Dict[str, Dict[str, Any]],
) -> Tuple[int, float, float]:
    key = cache_key(crop_path)

    cached = cache.get(key)
    if cached is not None:
        return (
            int(cached.get("person_count", 0)),
            float(cached.get("person_overlap", 0.0)),
            float(cached.get("person_max_conf", 0.0)),
        )

    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    boxes, confs = detector.detect_person(rgb)

    person_area = union_area_xyxy(boxes, w, h)
    overlap = clamp01(person_area / max(float(w * h), 1.0))
    max_conf = max(confs) if confs else 0.0

    cache[key] = {
        "path": str(crop_path.resolve()),
        "person_count": len(boxes),
        "person_overlap": overlap,
        "person_max_conf": max_conf,
        "boxes": [list(map(float, b)) for b in boxes],
        "confidences": [float(c) for c in confs],
    }

    return len(boxes), overlap, max_conf


def compute_quality(
    crop_path: Path,
    detector: RFDETRPersonDetector,
    cache: Dict[str, Dict[str, Any]],
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
            person_count=0,
            person_overlap=0.0,
            person_max_conf=0.0,
            overlap_score=0.0,
            quality_score=0.0,
            accepted=False,
            reasons=["image_read_failed"],
        )

    h, w = img.shape[:2]
    blur = laplacian_variance(img)
    res = resolution_score(w, h)
    sharp = sharpness_score(blur, blur_threshold)

    person_count, overlap, max_conf = contamination_for_crop(
        crop_path,
        img,
        detector,
        cache,
    )

    if overlap <= overlap_soft:
        overlap_score = 1.0
    elif overlap >= overlap_hard:
        overlap_score = 0.0
    else:
        overlap_score = 1.0 - (
            (overlap - overlap_soft)
            / max(overlap_hard - overlap_soft, 1e-9)
        )

    accepted = True
    reasons: List[str] = []

    if w < min_width:
        accepted = False
        reasons.append(f"width<{min_width}")

    if h < min_height:
        accepted = False
        reasons.append(f"height<{min_height}")

    if blur < blur_threshold:
        reasons.append(f"blur<{blur_threshold:g}")

    if overlap >= overlap_hard:
        accepted = False
        reasons.append(f"person_overlap>={overlap_hard:.2f}")
    elif overlap >= overlap_soft:
        reasons.append(f"person_overlap>={overlap_soft:.2f}")

    # 정확도 우선:
    # contamination 신호를 가장 크게 반영한다.
    quality = (
        0.55 * overlap_score
        + 0.25 * res
        + 0.20 * sharp
    )

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
        resolution_score=res,
        sharpness_score=sharp,
        person_count=person_count,
        person_overlap=overlap,
        person_max_conf=max_conf,
        overlap_score=overlap_score,
        quality_score=float(quality),
        accepted=accepted,
        reasons=reasons,
    )


def write_csv(rows: List[CropScore], path: Path) -> None:
    fields = list(asdict(rows[0]).keys()) if rows else [
        "path", "video", "track", "label", "frame_idx",
        "width", "height", "area",
        "blur_var", "resolution_score", "sharpness_score",
        "person_count", "person_overlap", "person_max_conf",
        "overlap_score", "quality_score", "accepted", "reasons",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            d = asdict(row)
            d["reasons"] = ";".join(row.reasons)
            writer.writerow(d)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="RF-DETR crop 내부 person contamination 기반 object crop 품질 필터"
    )
    ap.add_argument(
        "--object-root",
        type=Path,
        default=DEFAULT_OBJECT_ROOT,
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

    ap.add_argument("--person-conf", type=float, default=0.35)
    ap.add_argument("--person-overlap-soft", type=float, default=0.15)
    ap.add_argument("--person-overlap-hard", type=float, default=0.35)

    ap.add_argument(
        "--model-type",
        choices=["nano", "small", "medium", "base", "large"],
        default="base",
    )
    ap.add_argument(
        "--weights",
        default=None,
        help="RF-DETR custom checkpoint 경로. 생략하면 패키지 기본 checkpoint",
    )

    ap.add_argument(
        "--max-crops",
        type=int,
        default=0,
        help="디버그용 처리 상한. 0=전체",
    )
    ap.add_argument(
        "--save-cache-every",
        type=int,
        default=100,
        help="person detector 결과 cache 저장 주기",
    )
    ap.add_argument(
        "--copy-selected",
        action="store_true",
    )

    args = ap.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k는 1 이상이어야 합니다.")
    if not (0 <= args.person_overlap_soft <= args.person_overlap_hard <= 1):
        raise ValueError("overlap threshold는 0 <= soft <= hard <= 1 이어야 합니다.")

    obj_root = args.object_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not obj_root.exists():
        raise SystemExit(f"object root가 없습니다: {obj_root}")

    cache_path = out_dir / "person_contamination_cache.json"
    cache = load_cache(cache_path)

    detector = RFDETRPersonDetector(
        conf=args.person_conf,
        weights=args.weights,
        model_type=args.model_type,
    )

    print("=" * 72)
    print("OBJECT CROP QUALITY V3 FIXED")
    print(f"Object root       : {obj_root}")
    print(f"Output            : {out_dir}")
    print(f"RF-DETR model     : {args.model_type}")
    print(f"Person conf       : {args.person_conf:.2f}")
    print(
        f"Person overlap    : soft={args.person_overlap_soft:.2f}, "
        f"hard={args.person_overlap_hard:.2f}"
    )
    print(f"Top-K / track     : {args.top_k}")
    print(f"Cached detections : {len(cache):,}")
    print("=" * 72)

    all_rows: List[CropScore] = []
    selected: Dict[str, List[Dict[str, Any]]] = {}
    rejected: List[Dict[str, Any]] = []

    video_dirs = sorted(p for p in obj_root.iterdir() if p.is_dir())
    processed = 0
    start_time = time.time()

    stop_all = False

    for vi, video_dir in enumerate(video_dirs, 1):
        if stop_all:
            break

        track_dirs = sorted(p for p in video_dir.iterdir() if p.is_dir())

        print(
            f"[{vi:3d}/{len(video_dirs):3d}] "
            f"{video_dir.name} tracks={len(track_dirs)}"
        )

        for track_dir in track_dirs:
            crops = sorted(
                p for p in track_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )

            track_rows: List[CropScore] = []

            for crop in crops:
                if args.max_crops and processed >= args.max_crops:
                    stop_all = True
                    break

                row = compute_quality(
                    crop,
                    detector,
                    cache,
                    min_width=args.min_width,
                    min_height=args.min_height,
                    blur_threshold=args.blur_threshold,
                    overlap_soft=args.person_overlap_soft,
                    overlap_hard=args.person_overlap_hard,
                )

                all_rows.append(row)
                track_rows.append(row)
                processed += 1

                if (
                    args.save_cache_every > 0
                    and processed % args.save_cache_every == 0
                ):
                    save_cache(cache, cache_path)

                    elapsed = max(time.time() - start_time, 1e-6)
                    rate = processed / elapsed
                    print(
                        f"    crops={processed:,} "
                        f"rate={rate:.2f}/s "
                        f"person-hit={sum(r.person_count > 0 for r in all_rows):,}"
                    )

            ranked = sorted(
                track_rows,
                key=lambda r: (
                    1 if r.accepted else 0,
                    r.quality_score,
                    -r.person_overlap,
                    r.blur_var,
                    r.area,
                ),
                reverse=True,
            )

            chosen = [r for r in ranked if r.accepted][: args.top_k]

            key = f"{video_dir.name}/{track_dir.name}"
            selected[key] = [asdict(r) for r in chosen]

            chosen_paths = {r.path for r in chosen}
            for r in track_rows:
                if r.path not in chosen_paths:
                    rejected.append(asdict(r))

    save_cache(cache, cache_path)

    csv_path = out_dir / "crop_quality_v3.csv"
    selected_path = out_dir / "selected_crops_v3.json"
    rejected_path = out_dir / "rejected_crops_v3.json"

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

        dst_root = out_dir / "selected"
        for key, items in selected.items():
            safe_key = key.replace("/", "__").replace("\\", "__")
            dst_dir = dst_root / safe_key
            dst_dir.mkdir(parents=True, exist_ok=True)

            for item in items:
                src = Path(item["path"])
                if src.is_file():
                    shutil.copy2(src, dst_dir / src.name)

    accepted_count = sum(r.accepted for r in all_rows)
    selected_count = sum(len(v) for v in selected.values())

    person_hit = sum(r.person_count > 0 for r in all_rows)
    soft_hit = sum(r.person_overlap >= args.person_overlap_soft for r in all_rows)
    hard_hit = sum(r.person_overlap >= args.person_overlap_hard for r in all_rows)

    elapsed = time.time() - start_time

    print()
    print("=" * 72)
    print("COMPLETE")
    print(f"Crops processed       : {len(all_rows):,}")
    print(f"Accepted              : {accepted_count:,}")
    print(f"Selected for embedding: {selected_count:,}")
    print(f"Person detected crops : {person_hit:,}")
    print(f"Overlap >= soft       : {soft_hit:,}")
    print(f"Overlap >= hard       : {hard_hit:,}")
    print(f"Elapsed               : {elapsed:.1f}s")
    print()
    print(f"CSV      : {csv_path}")
    print(f"Selected : {selected_path}")
    print(f"Rejected : {rejected_path}")
    print(f"Cache    : {cache_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
