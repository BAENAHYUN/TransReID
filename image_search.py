from __future__ import annotations

"""
image_search.py — 검색 전용 (Qwen 없음)
======================================

쿼리 1장
 -> RF-DETR -> crop N개
 -> crop마다:
    person: SigLIP2 + IRRA RRF -> 후보 200 -> SOLIDER 재정렬 -> 20
    object: SigLIP2 + DINOv2 RRF -> 20
 -> JSON 출력

Qwen 은 이 파일에 없다
--------------------
Qwen 재순위/검증은 qwen_stage.py 가 담당한다. 분리한 이유:

  1) GPU 메모리 — 임베더 4개와 Qwen 을 한 프로세스에 두면 부담이 크다
  2) 재실행 비용 — alpha/threshold 를 바꿔볼 때마다 RF-DETR 과 검색을
     다시 돌리는 것은 낭비다. 검색 결과는 그대로인데 20~30초씩 버린다
  3) 원인 분리 — 결과가 나쁠 때 검색 탓인지 Qwen 탓인지 구분해야 한다

    python image_search.py -i query.jpg --json-out search.json
    python qwen_stage.py --in search.json --out reranked.json
    python qwen_stage.py --in search.json --out t8.json --threshold 0.8
      (두 번째 Qwen 실행은 검색을 다시 하지 않는다)

Qdrant 컬렉션 계약
------------------
최종 DB 는 두 컬렉션으로 분리되어 있다.

    forensic_person
      siglip2 / irra / solider

    forensic_object
      siglip2 / dinov2

이미지와 영상은 컬렉션을 따로 만들지 않고 같은 person/object 컬렉션 안에서
media_type 으로 구분한다.

따라서 검색할 때도 scope 에 맞는 컬렉션을 명시적으로 선택한다.
person_only 필터로 person/object 컬렉션 역할을 대신하지 않는다.

person 검색은 1차 SigLIP2 + IRRA 만 실행한 뒤 후보가 있을 때에만
SOLIDER 를 지연 로드해 재정렬한다.
"""

import argparse
import gc
import inspect
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent
RFDETR_ROOT = ROOT / "src" / "RF-DETR"

for _root in (ROOT, RFDETR_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from search import SearchEngine, SearchHit

logger = logging.getLogger(__name__)

CONFIG = ROOT / "pipeline.yaml"
QUERY_CROP_ROOT = ROOT / "data" / "search_crops"

# detect_and_crop 결과에서 crop 파일 경로를 찾을 때 시도하는 키 순서.
# detect_rf.py 는 "crop_path" 와 "path" 를 둘 다 넣어준다.
CROP_PATH_KEYS = ("crop_path", "path", "image_path")


def load_detector_module():
    """detect_rf 의 위치가 환경마다 달라 순서대로 시도한다."""
    errors: List[str] = []
    for module_name in ("detect.detect_rf", "detect_rf"):
        try:
            return __import__(module_name, fromlist=["*"])
        except Exception as e:  # noqa: BLE001
            errors.append(f"{module_name}: {type(e).__name__} {e}")

    raise ImportError(
        "detect_rf 를 불러올 수 없습니다. 다음 경로를 시도했습니다:\n  "
        + "\n  ".join(errors)
    )


def empty_cuda() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def normalize(v: Any) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x))
    if n <= 0 or not math.isfinite(n):
        raise ValueError("invalid vector")
    return x / n


def cosine(a: Any, b: Any) -> float:
    aa, bb = normalize(a), normalize(b)
    if aa.shape != bb.shape:
        raise ValueError(f"vector dim mismatch: {aa.shape} vs {bb.shape}")
    return float(np.dot(aa, bb))


def crop_path_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    """detect_and_crop 결과 dict 에서 crop 파일 경로를 꺼낸다.

    키가 없거나 파일이 없으면 None 을 돌려준다. 예전에는
    str(meta.get("crop_path") or meta.get("path")) 였는데, 둘 다 없으면
    "None" 이라는 문자열이 되어 엉뚱한 에러가 났다.
    """
    for key in CROP_PATH_KEYS:
        value = meta.get(key)
        if value and Path(str(value)).is_file():
            return str(value)

    for key in CROP_PATH_KEYS:
        value = meta.get(key)
        if value:
            logger.warning("crop 파일이 없습니다: %s", value)
            return None

    logger.warning(
        "crop 경로 키를 찾을 수 없습니다. 있는 키: %s (기대: %s)",
        sorted(meta), CROP_PATH_KEYS,
    )
    return None


def resolve_hit_path(hit: SearchHit) -> Optional[str]:
    """DB payload 의 crop 경로를 이 PC 에서 열 수 있는 경로로 바꾼다.

    적재한 PC 와 검색하는 PC 가 다르면 저장된 절대경로가 무효가 된다.
    Qwen 단계가 이 파일을 직접 열어야 하므로 여기서 해결해 둔다.
    """
    if hit.crop_path and Path(hit.crop_path).is_file():
        return str(hit.crop_path)

    raw = (
        hit.payload.get("crop_path")
        or hit.payload.get("path")
        or hit.payload.get("image_path")
    )
    if not raw:
        return hit.crop_path

    raw = str(raw)
    p = Path(raw)
    if p.is_file():
        return str(p)

    text = raw.replace("\\", "/")
    candidates = [
        ROOT / text.lstrip("./"),
        ROOT / "data" / "crops" / p.name,
        ROOT / "data" / "query_crops" / p.name,
    ]

    parts = [x for x in text.split("/") if x]
    for marker in ("crops", "query_crops"):
        if marker in parts:
            i = parts.index(marker)
            tail = Path(*parts[i + 1:]) if i + 1 < len(parts) else Path(p.name)
            candidates.append(ROOT / "data" / marker / tail)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return hit.crop_path or raw


def hit_to_row(hit: SearchHit) -> Dict[str, Any]:
    return {
        "point_id": hit.point_id,
        "image_id": hit.image_id,
        "label": hit.label,
        "is_person": hit.is_person,
        "crop_path": resolve_hit_path(hit),
        "bbox": [float(x) for x in (hit.bbox or [])],
        "frame_idx": int(hit.frame_idx),
        "track_id": hit.track_id,
        "detection_id": hit.payload.get("detection_id", ""),
        "qdrant_score": float(hit.retrieval_score),
    }


class ImageSearchPipeline:
    def __init__(
        self,
        config: str,
        extra_paths: Optional[Sequence[str]] = None,
    ) -> None:
        self.engine = SearchEngine.from_config(
            config,
            extra_paths=extra_paths,
            project_root=ROOT,
        )
        self.store = self.engine.store
        self.router = self.engine.router
        self.registry = self.engine.registry
        self.cfg = self.engine.cfg

        labels = getattr(self.cfg, "person_labels", None) or (
            "person", "pedestrian", "people", "human"
        )
        self.person_labels = {str(x).lower() for x in labels}

    def is_person(self, label: str) -> bool:
        return label.strip().lower() in self.person_labels

    def select_collection(self, scope: str) -> str:
        """
        scope 에 맞는 Qdrant collection 을 선택한다.

        person -> cfg.person_collection()
        object -> cfg.object_collection()

        이미지/영상은 collection 을 분리하지 않는다.
        같은 person/object collection 안에서 media_type 으로 구분한다.
        """
        if scope == "person":
            collection = self.cfg.person_collection()
        elif scope == "object":
            collection = self.cfg.object_collection()
        else:
            raise ValueError(
                f"scope 는 'person' 또는 'object' 여야 합니다: {scope!r}"
            )

        if not self.store.client.collection_exists(collection):
            raise RuntimeError(
                f"Qdrant collection 이 없습니다: {collection}"
            )

        self.store.collection = collection
        return collection

    def prefetch(self, n: int) -> int:
        return max(int(self.cfg.fusion.prefetch_limit), int(n))

    def get_named_vectors(
        self,
        point_ids: Sequence[str],
        name: str,
    ) -> Dict[str, np.ndarray]:
        records = self.store.client.retrieve(
            collection_name=self.store.collection,
            ids=list(point_ids),
            with_payload=False,
            with_vectors=[name],
        )

        out: Dict[str, np.ndarray] = {}
        for r in records:
            vectors = getattr(r, "vector", None)
            if vectors is None:
                vectors = getattr(r, "vectors", None)

            vec = vectors.get(name) if isinstance(vectors, dict) else vectors

            if vec is not None:
                out[str(r.id)] = np.asarray(vec, dtype=np.float32).reshape(-1)
        return out

    def person_search(
        self,
        query_crop: str,
        candidate_k: int = 200,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        person crop 검색.

        1차:
            SigLIP2 + IRRA RRF -> candidate_k

        2차:
            후보가 있을 때에만 SOLIDER 를 처음 로드/추론
            -> 후보 SOLIDER cosine 재정렬
            -> top_k
        """
        collection = self.select_collection("person")

        # ------------------------------------------------------------
        # 1) 1차 검색용 query vector만 생성.
        #    SOLIDER는 여기서 로드하지 않는다.
        # ------------------------------------------------------------
        qvecs = self.router.embed_query_image(
            query_crop,
            scope="person",
            names=["siglip2", "irra"],
        )

        missing = {"siglip2", "irra"} - set(qvecs)
        if missing:
            raise RuntimeError(
                f"person stage1 query vector missing: {sorted(missing)}"
            )

        points = self.store.fused_search(
            {
                "siglip2": qvecs["siglip2"],
                "irra": qvecs["irra"],
            },
            limit=candidate_k,
            prefetch_limit=self.prefetch(candidate_k),
            person_only=None,
        )
        hits = self.engine._to_hits(points)

        if not hits:
            return []

        # ------------------------------------------------------------
        # 2) 1차 후보가 있을 때에만 SOLIDER를 처음 실행.
        # ------------------------------------------------------------
        solider_query = self.router.embed_query_image(
            query_crop,
            scope="person",
            names=["solider"],
        )

        if "solider" not in solider_query:
            raise RuntimeError(
                "SOLIDER query vector missing"
            )

        candidate_vectors = self.get_named_vectors(
            [h.point_id for h in hits],
            "solider",
        )

        missing_ids = [
            str(h.point_id)
            for h in hits
            if str(h.point_id) not in candidate_vectors
        ]

        # forensic_person 의 모든 point 는 SOLIDER vector를 가져야 한다.
        # 누락을 객체 혼입으로 간주해 조용히 버리지 않는다.
        if missing_ids:
            raise RuntimeError(
                "forensic_person 후보 중 SOLIDER vector 누락: "
                f"{len(missing_ids)}/{len(hits)} "
                f"(collection={collection})"
            )

        rows: List[Dict[str, Any]] = []

        for initial_rank, hit in enumerate(hits, 1):
            vec = candidate_vectors[str(hit.point_id)]

            row = hit_to_row(hit)
            row["initial_rrf_rank"] = initial_rank
            row["solider_score"] = cosine(
                solider_query["solider"],
                vec,
            )
            rows.append(row)

        rows.sort(
            key=lambda x: x["solider_score"],
            reverse=True,
        )

        rows = rows[:top_k]

        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
            row["pre_qwen_rank"] = rank
            row["pre_qwen_score"] = float(
                row["solider_score"]
            )

        return rows

    def object_search(
        self,
        query_crop: str,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        object crop 검색.

        SigLIP2 + DINOv2 RRF -> top_k
        """
        self.select_collection("object")

        qvecs = self.router.embed_query_image(
            query_crop,
            scope="object",
            names=["siglip2", "dinov2"],
        )

        missing = {"siglip2", "dinov2"} - set(qvecs)
        if missing:
            raise RuntimeError(
                f"object query vector missing: {sorted(missing)}"
            )

        points = self.store.fused_search(
            {
                "siglip2": qvecs["siglip2"],
                "dinov2": qvecs["dinov2"],
            },
            limit=top_k,
            prefetch_limit=self.prefetch(top_k),
            person_only=None,
        )
        hits = self.engine._to_hits(points)

        rows: List[Dict[str, Any]] = []

        for rank, hit in enumerate(hits, 1):
            row = hit_to_row(hit)
            row["rank"] = rank
            row["pre_qwen_rank"] = rank
            row["pre_qwen_score"] = float(
                hit.retrieval_score
            )
            rows.append(row)

        return rows

    def release(self) -> None:
        self.registry.release()
        empty_cuda()


def run(
    image: str,
    *,
    config: str,
    crop_dir: str,
    person_candidates: int,
    top_k: int,
    forensic_only: bool,
    min_person_width: Optional[int],
    min_person_height: Optional[int],
    extra_paths: Optional[Sequence[str]],
) -> Dict[str, Any]:
    source = Path(image)
    if not source.is_file():
        raise FileNotFoundError(source)

    # ------------------------------------------------------------
    # 1. 쿼리 1장 -> RF-DETR -> crop N개
    # ------------------------------------------------------------
    detect_rf = load_detector_module()

    out_dir = Path(crop_dir) / source.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    target_classes = None
    if forensic_only:
        target_classes = list(detect_rf.FORENSIC_TARGET_CLASSES)

    crop_kwargs: Dict[str, Any] = {}
    if min_person_width is not None:
        crop_kwargs["min_person_width"] = min_person_width
    if min_person_height is not None:
        crop_kwargs["min_person_height"] = min_person_height

    # TypeError 전체를 "구버전 인자 미지원"으로 오인하지 않는다.
    # 실제 함수 signature를 보고 지원되는 옵션만 넘긴다.
    try:
        detect_params = inspect.signature(
            detect_rf.detect_and_crop
        ).parameters
    except (TypeError, ValueError):
        detect_params = {}

    if detect_params:
        unsupported = [
            key
            for key in crop_kwargs
            if key not in detect_params
        ]

        for key in unsupported:
            logger.warning(
                "detect_rf.detect_and_crop 가 %s 인자를 지원하지 않아 "
                "해당 override를 사용하지 않습니다.",
                key,
            )
            crop_kwargs.pop(key, None)

    detector = detect_rf.load_detect_model()
    try:
        crops, filtered = detect_rf.detect_and_crop(
            detector,
            str(source),
            output_dir=str(out_dir),
            target_classes=target_classes,
            **crop_kwargs,
        )
    finally:
        del detector
        empty_cuda()

    logger.info("RF-DETR accepted=%d filtered=%d", len(crops), len(filtered))

    small_person = sum(
        1 for f in filtered if f.get("reason") == "person_crop_too_small"
    )
    if small_person:
        logger.info(
            "person crop %d건이 최소 크기 미달로 버려졌습니다. "
            "쿼리에서 사람이 작게 찍혀 있으면 "
            "--min-person-width / --min-person-height 를 낮추세요.",
            small_person,
        )

    # ------------------------------------------------------------
    # 2. crop 마다 retrieval
    # ------------------------------------------------------------
    searcher = ImageSearchPipeline(
        config,
        extra_paths=extra_paths,
    )

    outputs = []
    skipped_crops = 0

    try:
        for i, meta in enumerate(crops, 1):
            crop_path = crop_path_from_meta(meta)
            if crop_path is None:
                skipped_crops += 1
                continue

            label = str(
                meta.get("class_name")
                or meta.get("label")
                or "unknown"
            )
            kind = (
                "person"
                if searcher.is_person(label)
                else "object"
            )

            logger.info(
                "[%d/%d] %s label=%s crop=%s",
                i,
                len(crops),
                kind,
                label,
                Path(crop_path).name,
            )

            t0 = time.time()

            try:
                if kind == "person":
                    rows = searcher.person_search(
                        crop_path,
                        candidate_k=person_candidates,
                        top_k=top_k,
                    )
                else:
                    rows = searcher.object_search(
                        crop_path,
                        top_k=top_k,
                    )

                error = None

            except Exception as e:  # noqa: BLE001 — crop 하나가 실패해도 계속
                rows = []
                error = f"{type(e).__name__}: {e}"
                logger.warning(
                    "[crop %d] 검색 실패: %s",
                    i,
                    error,
                )

            outputs.append(
                {
                    "crop_index": i,
                    "query_crop": crop_path,
                    "query_label": label,
                    "kind": kind,
                    "det_confidence": float(
                        meta.get("confidence", 0.0)
                        or 0.0
                    ),
                    "bbox": [
                        float(x)
                        for x in (
                            meta.get("bbox")
                            or []
                        )
                    ],
                    "elapsed_sec": round(
                        time.time() - t0,
                        3,
                    ),
                    "error": error,
                    "results": rows,
                }
            )

        if skipped_crops:
            logger.warning(
                "crop 파일이 없어 건너뛴 detection: %d건",
                skipped_crops,
            )

    finally:
        searcher.release()

    return {
        "query_image": str(source),
        "accepted_crops": len(crops),
        "filtered_crops": len(filtered),
        "skipped_crops": skipped_crops,
        "small_person_filtered": small_person,
        "person_candidates": person_candidates,
        "top_k": top_k,
        "qwen": False,
        "crops": outputs,
    }


def print_results(payload: Dict[str, Any], show: int = 10) -> None:
    print()
    print("=" * 90)
    print(f"query     : {payload['query_image']}")
    print(
        f"RF-DETR   : accepted={payload['accepted_crops']} "
        f"filtered={payload['filtered_crops']}"
        + (f" skipped={payload['skipped_crops']}"
           if payload.get("skipped_crops") else "")
    )
    print(
        f"person    : SigLIP2+IRRA -> {payload['person_candidates']} "
        f"-> SOLIDER -> {payload['top_k']}"
    )
    print(f"object    : SigLIP2+DINOv2 -> {payload['top_k']}")
    print("=" * 90)

    if not payload["crops"]:
        print()
        print("검출된 crop 이 없습니다.")
        if payload.get("small_person_filtered"):
            print(
                f"  person crop {payload['small_person_filtered']}건이 최소 크기 "
                f"미달로 버려졌습니다."
            )
            print("  --min-person-width / --min-person-height 를 낮춰 보세요.")
        print("  --forensic-only 를 빼면 COCO 80개 전체를 검출합니다.")
        return

    for item in payload["crops"]:
        print()
        print(
            f"[crop {item['crop_index']}] "
            f"{item['kind']} / {item['query_label']} / "
            f"{Path(item['query_crop']).name}  ({item['elapsed_sec']:.2f}s)"
        )
        print("-" * 90)

        if item.get("error"):
            print(f"  검색 실패: {item['error']}")
            continue

        if not item["results"]:
            print("  결과 없음")
            continue

        for row in item["results"][:show]:
            if item["kind"] == "person":
                stage = (
                    f"SOLIDER={row.get('solider_score', 0.0):.4f} "
                    f"(RRF rank={row.get('initial_rrf_rank', '-')})"
                )
            else:
                stage = f"RRF={row.get('qdrant_score', 0.0):.6f}"

            print(f"{row['rank']:>3}위 | {row.get('label', ''):<12} | {stage}")
            if row.get("crop_path"):
                print(f"      {row['crop_path']}")

        if len(item["results"]) > show:
            print(f"  ... 외 {len(item['results']) - show}건")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser(
        description=(
            "query 1장 -> RF-DETR -> crop N -> "
            "person(SigLIP2+IRRA -> 200 -> SOLIDER -> 20) / "
            "object(SigLIP2+DINOv2 -> 20). Qwen 은 qwen_stage.py 담당."
        )
    )
    ap.add_argument("-i", "--image", required=True)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--crop-dir", default=str(QUERY_CROP_ROOT))

    ap.add_argument("--person-candidates", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--show", type=int, default=10,
                    help="화면에 출력할 결과 수 (JSON 에는 전부 들어간다)")

    ap.add_argument(
        "--forensic-only",
        action="store_true",
        help="기본은 RF-DETR 모든 COCO class. 이 옵션은 forensic target 만 탐지",
    )
    ap.add_argument("--min-person-width", type=int, default=None,
                    help="person crop 최소 가로 (기본 detect_rf 값 90)")
    ap.add_argument("--min-person-height", type=int, default=None,
                    help="person crop 최소 세로 (기본 detect_rf 값 120)")
    ap.add_argument("--extra-paths", nargs="*", default=None)

    ap.add_argument("--json", action="store_true")
    ap.add_argument("--json-out", default=None,
                    help="qwen_stage.py 의 입력으로 쓸 JSON 경로")
    args = ap.parse_args()

    if args.top_k <= 0:
        ap.error("--top-k must be > 0")
    if args.person_candidates < args.top_k:
        ap.error("--person-candidates must be >= --top-k")
    if args.show <= 0:
        ap.error("--show must be > 0")
    if args.min_person_width is not None and args.min_person_width <= 0:
        ap.error("--min-person-width must be > 0")
    if args.min_person_height is not None and args.min_person_height <= 0:
        ap.error("--min-person-height must be > 0")

    t0 = time.time()
    payload = run(
        args.image,
        config=args.config,
        crop_dir=args.crop_dir,
        person_candidates=args.person_candidates,
        top_k=args.top_k,
        forensic_only=args.forensic_only,
        min_person_width=args.min_person_width,
        min_person_height=args.min_person_height,
        extra_paths=args.extra_paths,
    )
    payload["elapsed_sec"] = round(time.time() - t0, 3)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_results(payload, show=args.show)
        print(f"\nTOTAL: {payload['elapsed_sec']:.2f}s")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        status_stream = sys.stderr if args.json else sys.stdout
        print(f"\nJSON saved: {out}", file=status_stream)
        print(
            f"다음 단계: python qwen_stage.py --in {out} --out reranked.json",
            file=status_stream,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())