from __future__ import annotations

"""
search_db.py — 자연어 검색
=========================

두 가지 입력 방식이 있다.

    # 1) 자유 문장 (한국어 가능)
    python search_db.py -t "우산 들고있는 꽃무늬 옷 입은 여성" -k 50

    # 2) 항목별 입력 -> 서술형 문장 자동 조립  (권장)
    python search_db.py -k 50 `
      --gender 여성 --hair "짧은 검은 곱슬" `
      --top "화려한 꽃무늬 민소매 원피스" `
      --carry "큰 분홍 양산" --place "야외 맑은"

왜 항목별 입력이 권장인가
----------------------
IRRA 는 CUHK-PEDES 로 학습됐고 그 캡션은 평균 20단어가 넘는 서술형이다.
짧은 문장은 학습 분포 밖이라 벡터가 엉뚱한 곳에 놓인다.

자체 측정 (COCO 452,869 point DB, 정답 000000000036):

    "우산 들고있는 꽃무늬 옷 입은 여성"                        -> 23위
    "A woman wearing a flower pattern holding an umbrella."   -> 23위
    "A woman with short dark curly hair wearing a colorful
     floral sleeveless summer dress, holding a large pink
     parasol umbrella, standing outdoors on a sunny day."     ->  1위

번역을 거치든 안 거치든 짧으면 23위였다. **번역 품질이 아니라 길이·구체성
문제다.** 항목별로 받아 서술형으로 조립하면 이 문제가 해소된다.

Qwen 은 이 파일에 없다
--------------------
qwen_stage.py 가 담당한다. --json-out 으로 넘기면 된다.
다만 위 측정에서 정답이 1위로 올라오면 재순위할 것이 없다. 쿼리를 제대로
만드는 것이 Qwen 재순위보다 효과가 크다(자체 실험에서 Qwen 은 23위를
33위로 내렸다).

자연어 검색 벡터와 컬렉션
-----------------------
최종 DB 는 person / object 두 컬렉션으로 분리되어 있다.

    forensic_person
      -> 자연어 검색: SigLIP2 + IRRA
      -> SOLIDER 는 text encoder 가 없어 제외

    forensic_object
      -> 자연어 검색: SigLIP2
      -> DINOv2 는 text encoder 가 없어 제외

기본 자연어 검색은 person 검색이다.
--object-only 를 지정하면 forensic_object 로 전환하고 SigLIP2 만 사용한다.

이미지/영상은 별도 컬렉션으로 나누지 않으며 같은 person/object 컬렉션 안에서
media_type 으로 구분된다.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_translate import (
    BACKENDS,
    SLOT_LABELS,
    SLOT_ORDER,
    QueryDescriptor,
    QueryTranslator,
    has_hangul,
)
from search import SearchEngine, SearchHit

logger = logging.getLogger(__name__)

CONFIG_PATH = ROOT / "pipeline.yaml"

# CLI 인자로 받는 슬롯. query_translate.SLOT_ORDER + extra
CLI_SLOTS = SLOT_ORDER + ("extra",)


class TextSearcher:
    """자연어 쿼리로 Qdrant 를 검색한다."""

    def __init__(
        self,
        config_path: str = str(CONFIG_PATH),
        translate_backend: str = "opus",
        translate_model_id: Optional[str] = None,
        expand_query: bool = False,
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
        self.expand_query = expand_query

        self._translator = QueryTranslator(
            backend=translate_backend, model_id=translate_model_id
        )
        # 번역기를 공유하면 캐시도 함께 쓰인다
        self._descriptor = QueryDescriptor(self._translator)

        self.last_query_en: Optional[str] = None
        self.last_build: Optional[Dict[str, Any]] = None

    def _select_scope(
        self,
        person_only: Optional[bool],
        names: Optional[List[str]],
    ) -> tuple[str, List[str]]:
        """
        자연어 검색 scope / collection / retriever 를 함께 확정한다.

        기본(None)과 True:
            forensic_person
            supports_text=true 인 person retriever
            -> 현재 SigLIP2 + IRRA

        False:
            forensic_object
            supports_text=true 인 object retriever
            -> 현재 SigLIP2
        """
        scope = "object" if person_only is False else "person"

        if scope == "person":
            collection = self.cfg.person_collection()
            specs = self.cfg.for_person()
        else:
            collection = self.cfg.object_collection()
            specs = self.cfg.for_object()

        available = [
            spec.name
            for spec in specs
            if spec.supports_text
        ]

        if not available:
            raise RuntimeError(
                f"scope='{scope}'에서 자연어 검색을 지원하는 retriever가 없습니다."
            )

        if names is None:
            selected = available
        else:
            selected = list(dict.fromkeys(names))

            if not selected:
                raise ValueError(
                    "names를 지정했다면 retriever 이름을 하나 이상 넣어야 합니다."
                )

            unknown = [
                name
                for name in selected
                if name not in available
            ]

            if unknown:
                raise ValueError(
                    f"scope='{scope}' 자연어 검색에서 사용할 수 없는 retriever: "
                    f"{unknown}. 사용 가능: {available}"
                )

        if not self.engine.store.client.collection_exists(collection):
            raise RuntimeError(
                f"Qdrant collection 이 없습니다: {collection}"
            )

        # SearchEngine/QdrantStore 는 실제 검색 시 self.collection 을 읽는다.
        self.engine.store.collection = collection

        return scope, selected

    # ── 쿼리 준비 ───────────────────────────────────────────────────────────

    def from_text(self, query: str, translate: bool = True) -> str:
        """자유 문장 경로. 한국어면 번역한다."""
        english = self._translator.translate(query) if translate else query
        if self.expand_query:
            english = self._translator.expand(english)
        self.last_query_en = english
        self.last_build = None
        return english

    def from_slots(self, slots: Dict[str, str]) -> str:
        """항목별 입력 경로. 서술형 문장으로 조립한다."""
        build = self._descriptor.build(**slots)
        self.last_query_en = str(build["caption"])
        self.last_build = build
        return self.last_query_en

    # ── 검색 ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: Optional[str] = None,
        slots: Optional[Dict[str, str]] = None,
        limit: int = 50,
        names: Optional[List[str]] = None,
        person_only: Optional[bool] = None,
        translate: bool = True,
    ) -> Dict[str, Any]:
        if not query and not slots:
            raise ValueError("query 또는 slots 중 하나는 있어야 합니다.")

        t0 = time.time()
        if slots:
            english = self.from_slots(slots)
            source = "slots"
            original = " / ".join(
                f"{SLOT_LABELS.get(k, k)}={v}"
                for k, v in slots.items() if v
            )
        else:
            query = (query or "").strip()
            if not query:
                raise ValueError("빈 쿼리입니다.")
            english = self.from_text(query, translate=translate)
            source = "text"
            original = query
        t_prepare = time.time() - t0

        if has_hangul(english):
            logger.warning(
                "쿼리에 한글이 남아 있습니다: %r\n"
                "  IRRA/SigLIP2 는 영어 토크나이저를 쓰므로 에러 없이 "
                "무의미한 벡터가 나옵니다. 번역 백엔드를 확인하거나 해당 "
                "항목을 영어로 직접 입력하세요.",
                english,
            )

        word_count = len(english.split())
        if word_count < 12:
            logger.info(
                "쿼리가 %d단어입니다. IRRA 학습 캡션은 평균 20단어가 넘는 "
                "서술형이므로, 항목을 더 채우면 검색 품질이 올라갑니다.",
                word_count,
            )

        scope, selected_names = self._select_scope(
            person_only,
            names,
        )

        t0 = time.time()
        qvecs = self.engine.router.embed_query_text(
            english,
            names=selected_names,
        )
        t_embed = time.time() - t0

        logger.info(
            "텍스트 쿼리: scope=%s collection=%s vectors=%s",
            scope,
            self.engine.store.collection,
            sorted(qvecs),
        )

        t0 = time.time()
        points = self.engine._fetch(
            qvecs,
            final_limit=limit,
            prefetch_limit=None,
            weights=None,
            extra_filter=None,
            # person/object 는 이미 collection 으로 분리되어 있다.
            person_only=None,
            need=limit,
        )
        hits = self.engine._to_hits(points)
        t_search = time.time() - t0

        method = "단일" if len(qvecs) == 1 else self.cfg.fusion.method

        return {
            "source": source,
            "scope": scope,
            "collection": self.engine.store.collection,
            "query": original,
            "query_en": english,
            "word_count": word_count,
            "translated": english != original,
            "build": self.last_build,
            "slots": {k: v for k, v in (slots or {}).items() if v},
            "vectors": sorted(qvecs),
            "fusion_method": method,
            "hits": hits,
            "timing": {
                "prepare": round(t_prepare, 3),
                "embed": round(t_embed, 3),
                "search": round(t_search, 3),
            },
        }

    def release(self) -> None:
        self.engine.registry.release()
        self._translator.release()


# ─────────────────────────────────────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────────────────────────────────────
def print_result(res: Dict[str, Any], show: int = 20) -> None:
    hits: List[SearchHit] = res["hits"]
    t = res["timing"]
    total = sum(t.values())

    bar = "=" * 78
    print()
    print(bar)

    if res["source"] == "slots":
        print("  입력 방식 : 항목별")
        for k in CLI_SLOTS:
            v = res["slots"].get(k)
            if v:
                print(f"    {SLOT_LABELS.get(k, k):>10} : {v}")
    else:
        print(f"  쿼리      : {res['query']}")

    print(f"  검색 문장 : {res['query_en']}")
    print(f"              ({res['word_count']}단어)")
    print(f"  scope     : {res['scope']}")
    print(f"  collection: {res['collection']}")

    build = res.get("build")
    if build and build.get("unmapped"):
        print(f"  변환 실패 : {build['unmapped']}")

    print(f"  벡터      : {' + '.join(res['vectors'])} ({res['fusion_method']})")
    print(
        f"  소요      : {total:.2f}s "
        f"(준비 {t['prepare']:.2f} / 임베딩 {t['embed']:.2f} / "
        f"검색 {t['search']:.2f})"
    )
    print(f"  결과      : {len(hits)}건")
    print(bar)

    if not hits:
        print("  결과가 없습니다.")
        print("  - 컬렉션에 데이터가 있는지 확인하세요 "
              "(python doctor.py --only qdrant).")
        print("  - --person-only / --object-only 필터를 확인하세요.")
        print()
        return

    for h in hits[:show]:
        tag = "인물" if h.is_person else "객체"
        name = Path(h.crop_path).name if h.crop_path else h.point_id[:12]
        print(f"  {h.rank:>3}위  score={h.score:.5f}  {tag}  {h.label}")
        print(f"        {h.image_id}")
        print(f"        {name}")

    if len(hits) > show:
        print(f"  ... 외 {len(hits) - show}건")
    print()


def to_json(res: Dict[str, Any]) -> Dict[str, Any]:
    """qwen_stage.py 가 받을 수 있는 형식.

    image_search.py 와 같은 crops 배열 구조를 쓴다. 텍스트 쿼리는 crop 이
    없으므로 query_text 를 넣고 query_crop 은 비운다. qwen_stage.py 는
    query_text 가 있으면 텍스트 모드로 동작한다.

    slots 를 함께 기록해 두면 나중에 같은 쿼리를 재현할 수 있다.
    """
    def hit_dict(h: SearchHit, rank: int) -> Dict[str, Any]:
        return {
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

    hits = res["hits"]

    return {
        "search_type": "text",
        "input_source": res["source"],
        "scope": res["scope"],
        "collection": res["collection"],
        "query": res["query"],
        "query_en": res["query_en"],
        "word_count": res["word_count"],
        "query_slots": res["slots"],
        "top_k": len(hits),
        "qwen": False,
        "crops": [
            {
                "crop_index": 1,
                "query_text": res["query_en"],     # Qwen 이 이 문장으로 판정
                "query_text_original": res["query"],
                "query_slots": res["slots"],
                "query_crop": None,                # 텍스트 쿼리라 crop 없음
                "query_label": "text",
                "kind": res["scope"],
                "collection": res["collection"],
                "vectors_used": res["vectors"],
                "fusion_method": res["fusion_method"],
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
        description="자연어 검색 (Qwen 은 qwen_stage.py 담당)",
        epilog=(
            "예시:\n"
            '  python search_db.py -t "우산 들고있는 꽃무늬 옷 입은 여성" -k 50\n'
            "  python search_db.py -k 50 --gender 여성 --hair \"짧은 검은 곱슬\" \\\n"
            '      --top "화려한 꽃무늬 민소매 원피스" --carry "큰 분홍 양산"\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument("--text", "-t", default=None,
                    help="자유 문장 쿼리 (한국어 가능). "
                         "항목별 입력과 함께 쓸 수 없다.")

    slot_group = ap.add_argument_group(
        "항목별 입력",
        "채운 항목만 서술형 문장으로 조립된다. 빈 항목은 빠진다. "
        "사전에 없는 어휘는 번역기로 처리된다.",
    )
    slot_group.add_argument("--gender", default=None,
                            help="성별/연령 (여성, 남성, 소녀, 노인 ...)")
    slot_group.add_argument("--hair", default=None,
                            help="머리 (짧은 검은 곱슬, 긴 갈색 ...)")
    slot_group.add_argument("--top", default=None,
                            help="상의 (빨간 후드, 꽃무늬 원피스 ...)")
    slot_group.add_argument("--bottom", default=None,
                            help="하의 (청바지, 검정 반바지 ...)")
    slot_group.add_argument("--footwear", default=None,
                            help="신발 (흰 운동화, 구두 ...)")
    slot_group.add_argument("--accessory", default=None,
                            help="액세서리 (야구모자, 선글라스 ...)")
    slot_group.add_argument("--carry", default=None,
                            help="소지품 (검정 백팩, 큰 우산 ...)")
    slot_group.add_argument("--pose", default=None,
                            help="자세 (걷는, 앉은, 뒤돌아선 ...)")
    slot_group.add_argument("--place", default=None,
                            help="장소/배경 (거리, 야외 맑은, 지하철 ...)")
    slot_group.add_argument("--extra", default=None,
                            help="그 밖의 특징 (자유 서술)")

    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--limit", "-k", type=int, default=50)
    ap.add_argument("--show", type=int, default=20)
    ap.add_argument("--names", nargs="*", default=None,
                    help="참여 retriever 제한 (예: --names irra)")

    filt = ap.add_mutually_exclusive_group()
    filt.add_argument("--person-only", action="store_true", help="인물 검색 (기본값)")
    filt.add_argument("--object-only", action="store_true", help="객체 자연어 검색 (SigLIP2 only)")

    ap.add_argument("--no-translate", dest="translate", action="store_false",
                    default=True,
                    help="자유 문장을 번역 없이 그대로 쓴다 (-t 전용)")
    ap.add_argument("--translate-backend", default="opus", choices=list(BACKENDS))
    ap.add_argument("--translate-model-id", default=None)
    ap.add_argument("--expand", action="store_true",
                    help="자유 문장을 약하게 확장한다 (효과 제한적, -t 전용)")
    ap.add_argument("--dry-run", action="store_true",
                    help="검색 문장만 만들어 보여주고 끝낸다 (모델 로딩 없음)")

    ap.add_argument("--crop-root", default=None,
                    help="DB crop 루트 (다른 PC 에서 적재한 DB 를 열 때)")
    ap.add_argument("--extra-paths", nargs="*", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--json-out", default=None,
                    help="qwen_stage.py / view_results.py 입력으로 쓸 JSON 경로")
    args = ap.parse_args()

    if args.limit <= 0:
        ap.error("--limit must be > 0")
    if args.show <= 0:
        ap.error("--show must be > 0")
    if args.names == []:
        ap.error("--names 뒤에 retriever 이름을 하나 이상 지정하세요.")

    slots = {
        k: (getattr(args, k) or "").strip()
        for k in CLI_SLOTS
    }
    slots = {k: v for k, v in slots.items() if v}

    if args.text and slots:
        ap.error(
            "--text 와 항목별 입력은 함께 쓸 수 없습니다. "
            "둘 중 하나를 고르세요."
        )
    if not args.text and not slots:
        ap.error(
            "--text 또는 항목별 입력(--gender, --top, --carry ...) 중 "
            "하나는 필요합니다."
        )

    # --dry-run 은 검색 문장만 확인한다. Qdrant / 임베더를 올리지 않는다.
    if args.dry_run:
        tr = QueryTranslator(
            backend=args.translate_backend,
            model_id=args.translate_model_id,
        )
        try:
            print()
            if slots:
                desc = QueryDescriptor(tr)
                build = desc.build(**slots)
                print("  입력 방식 : 항목별")
                for k in CLI_SLOTS:
                    if slots.get(k):
                        print(f"    {SLOT_LABELS.get(k, k):>10} : {slots[k]}")
                print()
                print(f"  검색 문장 ({build['word_count']}단어):")
                print(f"    {build['caption']}")
                if build["unmapped"]:
                    print(f"  변환 실패 : {build['unmapped']}")
                en = str(build["caption"])
            else:
                en = tr.translate(args.text) if args.translate else args.text
                if args.expand:
                    en = tr.expand(en)
                print(f"  원문 : {args.text}")
                print(f"  번역 : {en}")

            print()
            if has_hangul(en):
                print(
                    "  경고: 한글이 남아 있습니다. "
                    "이대로 검색하면 결과가 무의미합니다."
                )
                print()
            if len(en.split()) < 12:
                print(
                    "  참고: 12단어 미만입니다. IRRA 학습 캡션은 평균 20단어가 "
                    "넘는 서술형이라, 항목을 더 채우면 검색 품질이 올라갑니다."
                )
                print()
            return 0
        finally:
            tr.release()

    person_only: Optional[bool] = None
    if args.person_only:
        person_only = True
    elif args.object_only:
        person_only = False

    searcher = TextSearcher(
        config_path=args.config,
        translate_backend=args.translate_backend,
        translate_model_id=args.translate_model_id,
        expand_query=args.expand,
        crop_root=args.crop_root,
        extra_paths=args.extra_paths,
    )

    try:
        res = searcher.search(
            query=args.text,
            slots=slots or None,
            limit=args.limit,
            names=args.names,
            person_only=person_only,
            translate=args.translate,
        )
    finally:
        searcher.release()

    payload = to_json(res)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_result(res, show=args.show)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        status_stream = sys.stderr if args.json else sys.stdout
        print(f"JSON saved: {out}", file=status_stream)
        print(
            f"결과 보기 : python view_results.py --in {out} --open",
            file=status_stream,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())