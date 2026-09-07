"""
search_person_crop.py — crop 1장 직접 검색 (Qwen 없음)
====================================================

    crop 1장 (이미 잘라놓은 이미지)
      -> person: SigLIP2 + IRRA 융합 -> (선택) SOLIDER 재정렬
         object: SigLIP2 + DINOv2 융합
      -> Qdrant 검색
      -> 결과

image_search.py 와 무엇이 다른가
-----------------------------
    image_search.py         원본 사진 1장 -> RF-DETR -> crop N개 -> crop 별 검색
    search_person_crop.py   이미 잘라놓은 crop 1장 -> 바로 검색

RF-DETR 을 거치지 않으므로 빠르다. 주 용도는 두 가지다.

  1) crop 하나만 빠르게 검색
  2) **정합성 확인** — DB 에 이미 들어 있는 crop 을 쿼리로 넣으면
     자기 자신이 1위(score 거의 1.0)로 나와야 한다. 안 나오면 색인 때와
     질의 때의 전처리가 어긋난 것이다. 가장 값싼 검사다.

        python search_person_crop.py --crop data/crops/xxx_person_001.jpg -k 5

Qwen 은 이 파일에 없다
--------------------
qwen_stage.py 가 담당한다. --json-out 으로 넘기면 된다.

    python search_person_crop.py --crop q.jpg --json-out one.json
    python qwen_stage.py --in one.json --out reranked.json

분리한 이유: 임베더와 Qwen 을 한 프로세스에 두면 GPU 부담이 크고,
alpha/threshold 를 바꿔볼 때마다 검색을 재실행하는 것이 낭비다.

왜 SOLIDER 를 융합에 안 넣는가
---------------------------
    SigLIP2 : 개방 어휘 의미 표현. 옷 색·소지품 같은 시각적 속성
    IRRA    : person ReID 로 파인튜닝된 CLIP. 보행자 외형에 특화
    SOLIDER : 신원 판별에 가장 강하다 (Market mINP 82.95 vs IRRA 29.56)

품질 차가 큰 벡터를 RRF 로 섞으면 약한 쪽 순위가 결과를 끌어내린다.
--solider 로 재정렬 단계로 빼면 SOLIDER 의 강점을 온전히 쓴다.
**미검증이다.** --with-solider (3벡터 융합) 와 비교할 것.

임베더를 직접 만들지 않는다
------------------------
Router/Registry 를 통해 pipeline.yaml 에 등록된 임베더를 쓴다. 색인 때와
질의 때가 같은 코드·같은 전처리를 타야 벡터가 같은 공간에 놓인다.
(예전에 SOLIDER 를 질의 쪽만 256x128 로 처리해 검색이 무작위가 된 적이 있다)

사용
----
    python search_person_crop.py --crop data/crops/xxx_person_001.jpg
    python search_person_crop.py --crop q.jpg -k 50
    python search_person_crop.py --crop q.jpg --solider        # SOLIDER 재정렬
    python search_person_crop.py --crop q.jpg --with-solider   # 3벡터 융합
    python search_person_crop.py --crop q.jpg --names irra     # IRRA 단독
    python search_person_crop.py --crop q.jpg --object         # 객체 crop
    python search_person_crop.py --crop q.jpg --json-out one.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search import SearchEngine, SearchHit

logger = logging.getLogger(__name__)

CONFIG_PATH = ROOT / "pipeline.yaml"

# 융합 1단계에 쓰는 벡터. SOLIDER 는 기본적으로 재정렬 전용이다.
STAGE1_BY_SCOPE = {
    "person": ("siglip2", "irra"),
    "object": ("siglip2", "dinov2"),
}


class CropSearcher:
    """crop 1장 -> Qdrant 검색."""

    def __init__(
        self,
        config_path: str = str(CONFIG_PATH),
        crop_root: Optional[str] = None,
        extra_paths: Optional[List[str]] = None,
    ) -> None:
        self.engine = SearchEngine.from_config(
            config_path,
            extra_paths=extra_paths,
            project_root=ROOT,
            crop_root=crop_root,
        )
        self.cfg = self.engine.cfg

    # ── 임베딩 ─────────────────────────────────────────────────────────────

    def embed(
        self,
        crop_path: str,
        scope: str,
        names: Optional[List[str]] = None,
        with_solider: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """crop 을 임베딩해 (1단계 융합용, 전체) 두 dict 를 돌려준다."""
        all_vecs = self.engine.router.embed_query_image(crop_path, scope=scope)

        if not all_vecs:
            raise RuntimeError(
                f"쿼리 벡터가 비었습니다. pipeline.yaml 에 scope='{scope}' 또는 "
                f"'all' 인 retriever 가 등록되어 있는지 확인하세요."
            )

        if names:
            unknown = set(names) - set(all_vecs)
            if unknown:
                raise ValueError(
                    f"scope='{scope}' 에서 쓸 수 없는 retriever: "
                    f"{sorted(unknown)}. 사용 가능: {sorted(all_vecs)}"
                )
            stage1 = {n: all_vecs[n] for n in names}
            return stage1, all_vecs

        if with_solider:
            return dict(all_vecs), all_vecs

        wanted = STAGE1_BY_SCOPE.get(scope, ())
        stage1 = {n: v for n, v in all_vecs.items() if n in wanted}

        if not stage1:
            logger.warning(
                "%s 가 없어 사용 가능한 벡터 전부로 검색합니다: %s",
                "/".join(wanted), sorted(all_vecs),
            )
            stage1 = dict(all_vecs)

        return stage1, all_vecs

    # ── 검색 ───────────────────────────────────────────────────────────────

    def search(
        self,
        crop_path: str,
        scope: str = "person",
        limit: int = 20,
        names: Optional[List[str]] = None,
        with_solider: bool = False,
        solider_rerank: bool = False,
        solider_pool: int = 200,
    ) -> Dict[str, Any]:
        if scope not in ("person", "object"):
            raise ValueError("scope 는 'person' 또는 'object' 여야 합니다.")

        path = Path(crop_path)
        if not path.is_file():
            raise FileNotFoundError(f"crop 이미지가 없습니다: {path}")

        t0 = time.time()
        stage1, all_vecs = self.embed(
            str(path), scope=scope, names=names, with_solider=with_solider
        )
        t_embed = time.time() - t0

        # SOLIDER 재정렬을 하려면 후보를 넉넉히 가져온다
        fetch = max(limit, solider_pool) if solider_rerank else limit

        t0 = time.time()
        points = self.engine._fetch(
            stage1,
            final_limit=fetch,
            prefetch_limit=None,
            weights=None,
            extra_filter=None,
            person_only=None,     # scope 자동 라우팅
            need=fetch,
        )
        hits = self.engine._to_hits(points)
        t_search = time.time() - t0

        method = "단일" if len(stage1) == 1 else self.cfg.fusion.method
        stages: List[str] = [
            f"{'+'.join(sorted(stage1))} ({method}) -> {len(hits)}건"
        ]

        # ── SOLIDER 재정렬 ──
        t_solider = 0.0
        reranked = False

        if solider_rerank:
            if scope != "person":
                logger.warning(
                    "SOLIDER 는 사람 전용입니다. scope='%s' 에서는 "
                    "재정렬을 건너뜁니다.", scope,
                )
            elif "solider" not in all_vecs:
                logger.warning(
                    "SOLIDER 벡터가 없어 재정렬을 건너뜁니다. "
                    "pipeline.yaml 에 solider 가 등록되어 있는지 확인하세요."
                )
            elif "solider" in stage1:
                logger.warning(
                    "SOLIDER 가 이미 융합에 포함되어 재정렬을 건너뜁니다. "
                    "--with-solider 와 --solider 를 함께 쓰지 마세요."
                )
            else:
                t0 = time.time()
                hits = self._rerank_with_solider(all_vecs["solider"], hits)
                t_solider = time.time() - t0
                reranked = True
                stages.append(f"solider 재정렬 -> 상위 {limit}건")

        hits = self._renumber(hits[:limit])

        return {
            "crop_path": str(path),
            "scope": scope,
            "vectors": sorted(stage1),
            "stages": stages,
            "reranked_by_solider": reranked,
            "hits": hits,
            "timing": {
                "embed": round(t_embed, 3),
                "search": round(t_search, 3),
                "solider": round(t_solider, 3),
            },
        }

    # ── SOLIDER 재정렬 ─────────────────────────────────────────────────────

    def _rerank_with_solider(
        self,
        query_vec: np.ndarray,
        hits: List[SearchHit],
    ) -> List[SearchHit]:
        """
        후보의 SOLIDER 벡터를 retrieve 로 받아 코사인으로 재정렬한다.

        Qdrant 를 다시 검색하지 않는다. 후보 200개면 200x1024 내적이라
        사실상 즉시 끝난다.
        """
        if not hits:
            return hits

        try:
            records = self.engine.store.client.retrieve(
                collection_name=self.cfg.collection,
                ids=[h.point_id for h in hits],
                with_vectors=["solider"],
                with_payload=False,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "SOLIDER 벡터 조회 실패 -> 재정렬 건너뜀: %s: %s",
                type(e).__name__, e,
            )
            return hits

        by_id: Dict[str, np.ndarray] = {}
        for rec in records:
            vec = getattr(rec, "vector", None)
            if isinstance(vec, dict):
                vec = vec.get("solider")
            if vec is not None:
                by_id[str(rec.id)] = np.asarray(vec, dtype=np.float32)

        q = self._unit(query_vec)
        scored, missing = [], []

        for h in hits:
            v = by_id.get(h.point_id)
            if v is None:
                # SOLIDER 벡터가 없는 point. siglip2 가 scope='all' 이라
                # 필터 없이 보므로 객체 point 가 섞일 수 있다.
                missing.append(h)
                continue
            h.score = float(np.dot(q, self._unit(v)))
            h.payload["solider_score"] = h.score
            scored.append(h)

        if missing:
            logger.info(
                "SOLIDER 벡터가 없는 후보 %d/%d건은 재정렬에서 제외 (뒤로 배치)",
                len(missing), len(hits),
            )

        scored.sort(key=lambda h: h.score, reverse=True)
        return self._renumber(scored + missing)

    @staticmethod
    def _unit(vec) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32).ravel()
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-12 else v

    @staticmethod
    def _renumber(hits: List[SearchHit]) -> List[SearchHit]:
        for i, h in enumerate(hits, 1):
            h.rank = i
        return hits

    def release(self) -> None:
        self.engine.registry.release()


# ─────────────────────────────────────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────────────────────────────────────
def print_result(res: Dict[str, Any], show: int = 20) -> None:
    hits: List[SearchHit] = res["hits"]
    t = res["timing"]
    total = sum(t.values())

    bar = "=" * 76
    print()
    print(bar)
    print(f"  쿼리 crop : {res['crop_path']}")
    print(f"  scope     : {res['scope']}")
    print(f"  벡터      : {' + '.join(res['vectors'])}")
    for s in res["stages"]:
        print(f"  단계      : {s}")
    print(
        f"  소요      : {total:.2f}s "
        f"(임베딩 {t['embed']:.2f} / 검색 {t['search']:.2f}"
        + (f" / solider {t['solider']:.2f}" if t["solider"] else "")
        + ")"
    )
    print(bar)

    if not hits:
        print("  결과가 없습니다.")
        print("  - 컬렉션에 데이터가 있는지 확인하세요 "
              "(python doctor.py --only qdrant).")
        print()
        return

    for h in hits[:show]:
        parts = [f"score={h.score:.4f}"]
        if "solider_score" in h.payload:
            parts.append(f"solider={h.payload['solider_score']:.4f}")
        parts.append(f"qdrant={h.retrieval_score:.4f}")

        name = Path(h.crop_path).name if h.crop_path else h.point_id[:12]
        print(f"  {h.rank:>3}위 {'  '.join(parts)}")
        print(f"       {h.label:12s} {h.image_id}")
        print(f"       {name}")

    if len(hits) > show:
        print(f"  ... 외 {len(hits) - show}건")
    print()


def to_json(res: Dict[str, Any]) -> Dict[str, Any]:
    """qwen_stage.py 가 그대로 받을 수 있는 형식으로 만든다.

    image_search.py 의 출력과 같은 구조(crops 배열)를 쓰므로, crop 이
    하나뿐이어도 동일한 후속 단계를 탈 수 있다.
    """
    def hit_dict(h: SearchHit, rank: int) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "rank": rank,
            "point_id": h.point_id,
            "image_id": h.image_id,
            "label": h.label,
            "is_person": h.is_person,
            "crop_path": h.crop_path,
            "bbox": [float(x) for x in (h.bbox or [])],
            "frame_idx": int(h.frame_idx),
            "track_id": h.track_id,
            "detection_id": h.payload.get("detection_id", ""),
            "qdrant_score": round(float(h.retrieval_score), 6),
            "pre_qwen_rank": rank,
            "pre_qwen_score": round(float(h.score), 6),
        }
        if "solider_score" in h.payload:
            d["solider_score"] = round(float(h.payload["solider_score"]), 6)
        return d

    hits = res["hits"]
    scope = res["scope"]

    return {
        "query_image": res["crop_path"],
        "accepted_crops": 1,
        "filtered_crops": 0,
        "skipped_crops": 0,
        "top_k": len(hits),
        "qwen": False,
        "crops": [
            {
                "crop_index": 1,
                "query_crop": res["crop_path"],
                "query_label": scope,
                "kind": scope,
                "det_confidence": 1.0,
                "bbox": [],
                "vectors_used": res["vectors"],
                "stages": res["stages"],
                "reranked_by_solider": res["reranked_by_solider"],
                "timing": res["timing"],
                "error": None,
                "results": [hit_dict(h, i) for i, h in enumerate(hits, 1)],
            }
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    ap = argparse.ArgumentParser(
        description="crop 1장 직접 검색 (Qwen 은 qwen_stage.py 담당)"
    )
    ap.add_argument("--crop", "-c", required=True,
                    help="crop 이미지 경로 (이미 잘라놓은 것)")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--limit", "-k", type=int, default=20)
    ap.add_argument("--show", type=int, default=20)

    ap.add_argument("--object", dest="is_object", action="store_true",
                    help="객체 crop 으로 취급 (SigLIP2 + DINOv2). "
                         "기본은 사람 (SigLIP2 + IRRA)")
    ap.add_argument("--names", nargs="*", default=None,
                    help="참여 벡터 직접 지정 (예: --names irra)")

    ap.add_argument("--solider", dest="solider_rerank", action="store_true",
                    help="SigLIP2+IRRA 융합 후 SOLIDER 로 재정렬 (사람 전용)")
    ap.add_argument("--solider-pool", type=int, default=200,
                    help="SOLIDER 재정렬 대상 후보 수")
    ap.add_argument("--with-solider", action="store_true",
                    help="SOLIDER 를 융합에 함께 넣는다 (--solider 와 비교용)")

    ap.add_argument("--crop-root", default=None,
                    help="DB crop 루트 (다른 PC 에서 적재한 DB 를 열 때)")
    ap.add_argument("--extra-paths", nargs="*", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--json-out", default=None,
                    help="qwen_stage.py 의 입력으로 쓸 JSON 경로")
    args = ap.parse_args()

    if args.solider_rerank and args.with_solider:
        ap.error("--solider 와 --with-solider 는 함께 쓸 수 없습니다. "
                 "재정렬(--solider)과 융합(--with-solider) 중 하나를 고르세요.")
    if args.limit <= 0:
        ap.error("--limit must be > 0")
    if args.solider_pool < args.limit:
        ap.error("--solider-pool must be >= --limit")

    scope = "object" if args.is_object else "person"

    searcher = CropSearcher(
        config_path=args.config,
        crop_root=args.crop_root,
        extra_paths=args.extra_paths,
    )

    res = searcher.search(
        args.crop,
        scope=scope,
        limit=args.limit,
        names=args.names,
        with_solider=args.with_solider,
        solider_rerank=args.solider_rerank,
        solider_pool=args.solider_pool,
    )
    searcher.release()

    payload = to_json(res)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_result(res, show=args.show)
        if scope == "person" and not args.solider_rerank and not args.with_solider:
            print("  --solider (재정렬) 와 --with-solider (융합) 를 각각 돌려 "
                  "비교해 보세요.")
            print()

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON saved: {out}")
        print(f"다음 단계: python qwen_stage.py --in {out} --out reranked.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())