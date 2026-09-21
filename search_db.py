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

항목별 입력에 관하여
----------------------
QueryDescriptor 는 사람 검색용 구조화 입력을 영어 서술형 문장으로 조립하는
선택 기능이다. 자유 문장보다 항상 우수하다고 가정하지 않는다.

자체 한 사례(COCO 452,869 point DB, 정답 000000000036)에서는 더 구체적인
서술형 query의 순위가 23위에서 1위로 개선됐다. 이는 로컬 관찰이며,
"짧은 문장은 항상 나쁘다" 또는 "번역 품질은 중요하지 않다"는 일반 결론으로
사용하지 않는다.

object 검색에는 사람용 QueryDescriptor 슬롯을 사용하지 않고 자유 문장을 쓴다.

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
import hashlib
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
from config import PipelineConfig
from search import SearchEngine, SearchHit

logger = logging.getLogger(__name__)

CONFIG_PATH = ROOT / "pipeline.yaml"

def _sha256_file(path: Path) -> str:
    """검색 결과 재현성 기록용 설정 파일 SHA-256."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# CLI 인자로 받는 슬롯. query_translate.SLOT_ORDER + extra
CLI_SLOTS = SLOT_ORDER + ("extra",)


def _resolve_query_settings(
    cfg: PipelineConfig,
    *,
    translate_enabled: Optional[bool] = None,
    translate_backend: Optional[str] = None,
    translate_model_id: Optional[str] = None,
    translate_max_new_tokens: Optional[int] = None,
    translate_cache_size: Optional[int] = None,
    expand_query: Optional[bool] = None,
) -> Dict[str, Any]:
    """YAML 기본값 + CLI override를 합쳐 실제 query runtime 설정을 만든다."""
    tcfg = cfg.query.translation

    enabled = (
        bool(tcfg.enabled)
        if translate_enabled is None
        else bool(translate_enabled)
    )

    backend_overridden = translate_backend is not None
    backend = (
        str(tcfg.backend)
        if translate_backend is None
        else str(translate_backend).strip().lower()
    )

    if translate_model_id is not None:
        model_id = translate_model_id
    elif backend_overridden and backend != str(tcfg.backend):
        # backend만 바꿨는데 YAML의 다른 backend model_id를 재사용하면 안 된다.
        # None을 넘겨 QueryTranslator의 backend별 공식 기본 checkpoint를 쓴다.
        model_id = None
    else:
        model_id = tcfg.model_id

    max_new_tokens = (
        int(tcfg.max_new_tokens)
        if translate_max_new_tokens is None
        else int(translate_max_new_tokens)
    )
    cache_size = (
        int(tcfg.cache_size)
        if translate_cache_size is None
        else int(translate_cache_size)
    )
    expand = (
        bool(cfg.query.expand)
        if expand_query is None
        else bool(expand_query)
    )

    return {
        "translation_enabled": enabled,
        "backend": backend,
        "model_id": model_id,
        "max_new_tokens": max_new_tokens,
        "cache_size": cache_size,
        "expand_query": expand,
    }


class TextSearcher:
    """자연어 쿼리로 Qdrant 를 검색한다."""

    def __init__(
        self,
        config_path: str = str(CONFIG_PATH),
        translate_enabled: Optional[bool] = None,
        translate_backend: Optional[str] = None,
        translate_model_id: Optional[str] = None,
        translate_max_new_tokens: Optional[int] = None,
        translate_cache_size: Optional[int] = None,
        expand_query: Optional[bool] = None,
        crop_root: Optional[str] = None,
        extra_paths: Optional[List[str]] = None,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()

        self.engine = SearchEngine.from_config(
            self.config_path,
            extra_paths=extra_paths,
            project_root=ROOT,
            crop_root=crop_root,
        )
        self.cfg = self.engine.cfg

        runtime = _resolve_query_settings(
            self.cfg,
            translate_enabled=translate_enabled,
            translate_backend=translate_backend,
            translate_model_id=translate_model_id,
            translate_max_new_tokens=translate_max_new_tokens,
            translate_cache_size=translate_cache_size,
            expand_query=expand_query,
        )

        self.translation_enabled = bool(
            runtime["translation_enabled"]
        )
        self.expand_query = bool(runtime["expand_query"])

        self._translator = QueryTranslator(
            backend=str(runtime["backend"]),
            model_id=runtime["model_id"],
            max_new_tokens=int(runtime["max_new_tokens"]),
            cache_size=int(runtime["cache_size"]),
        )
        # 번역기를 공유하면 캐시도 함께 쓰인다.
        self._descriptor = QueryDescriptor(self._translator)

        self.last_query_en: Optional[str] = None
        self.last_build: Optional[Dict[str, Any]] = None
        self.last_translation_requested: Optional[bool] = None
        self.last_translation_applied: Optional[bool] = None
        self.last_expanded: bool = False

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

    def _retrieval_score_type(
        self,
        vector_names: List[str],
    ) -> str:
        """Qdrant score가 raw distance인지 fusion score인지 명시한다."""
        if len(vector_names) == 1:
            distance = self.cfg.qdrant.distance_for(vector_names[0])
            return f"qdrant_{str(distance).strip().lower()}"

        return f"qdrant_{str(self.cfg.fusion.method).strip().lower()}"

    def _config_metadata(
        self,
        collection: str,
        vector_names: List[str],
    ) -> Dict[str, Any]:
        """실험/검색 결과 재현을 위한 최소 runtime metadata."""
        retrievers: Dict[str, Any] = {}

        for name in vector_names:
            spec = self.cfg.retrievers.get(name)
            if spec is None:
                continue

            retrievers[name] = {
                "dim": int(spec.dim),
                "scope": str(spec.scope),
                "supports_text": bool(spec.supports_text),
                "module": str(spec.module),
                "class": str(spec.class_name),
                "params": dict(spec.params),
            }

        return {
            "config_path": str(self.config_path),
            "config_sha256": _sha256_file(self.config_path),
            "collection": collection,
            "fusion": {
                "method": str(self.cfg.fusion.method),
                "prefetch_limit": int(self.cfg.fusion.prefetch_limit),
                "limit": int(self.cfg.fusion.limit),
            },
            "retrievers": retrievers,
            # query/translation도 pipeline.yaml 기본값과 실제 runtime 값을 함께
            # 기록해 CLI override가 있었는지까지 재현할 수 있게 한다.
            "translation": {
                "yaml_default": {
                    "enabled": bool(
                        self.cfg.query.translation.enabled
                    ),
                    "backend": str(
                        self.cfg.query.translation.backend
                    ),
                    "model_id": self.cfg.query.translation.model_id,
                    "max_new_tokens": int(
                        self.cfg.query.translation.max_new_tokens
                    ),
                    "cache_size": int(
                        self.cfg.query.translation.cache_size
                    ),
                },
                "effective": {
                    "enabled": bool(self.translation_enabled),
                    "backend": str(self._translator.backend),
                    "model_id": self._translator.model_id,
                    "max_new_tokens": int(
                        self._translator.max_new_tokens
                    ),
                    "cache_size": int(
                        self._translator.cache_size
                    ),
                },
            },
            "expand_query": {
                "yaml_default": bool(self.cfg.query.expand),
                "effective": bool(self.expand_query),
            },
        }

    # ── 쿼리 준비 ───────────────────────────────────────────────────────────

    def from_text(
        self,
        query: str,
        *,
        scope: str,
        translate: Optional[bool] = None,
    ) -> str:
        """자유 문장 경로.

        SigLIP2 자체는 multilingual query를 지원한다.
        translate=True이면 현재 시스템 정책에 따라 먼저 영어로 번역한다.
        expand_query는 번역과 별도 단계이며 scope를 보존해 적용한다.
        """
        translate_requested = (
            self.translation_enabled
            if translate is None
            else bool(translate)
        )
        self.last_translation_requested = translate_requested

        prepared = (
            self._translator.translate(query)
            if translate_requested
            else query
        )

        self.last_translation_applied = (
            translate_requested
            and prepared != query
        )

        english = prepared
        self.last_expanded = False

        if self.expand_query:
            expanded = self._translator.expand(
                english,
                scope=scope,
            )
            self.last_expanded = expanded != english
            english = expanded

        self.last_query_en = english
        self.last_build = None
        return english

    def from_slots(self, slots: Dict[str, str]) -> str:
        """항목별 입력 경로. 서술형 문장으로 조립한다."""
        build = self._descriptor.build(**slots)
        self.last_query_en = str(build["caption"])
        self.last_build = build
        self.last_translation_requested = None
        self.last_translation_applied = None
        self.last_expanded = False
        return self.last_query_en

    # ── 검색 ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: Optional[str] = None,
        slots: Optional[Dict[str, str]] = None,
        limit: int = 50,
        names: Optional[List[str]] = None,
        person_only: Optional[bool] = None,
        translate: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if not query and not slots:
            raise ValueError("query 또는 slots 중 하나는 있어야 합니다.")

        if limit <= 0:
            raise ValueError("limit 은 1 이상이어야 합니다.")

        # scope/collection/retriever를 먼저 확정해야 expand()와 경고도
        # 실제 검색 대상에 맞게 동작한다.
        scope, selected_names = self._select_scope(
            person_only,
            names,
        )

        if slots and scope != "person":
            raise ValueError(
                "QueryDescriptor의 구조화 슬롯은 person 검색 전용입니다. "
                "object 검색은 --text 자유 문장을 사용하세요."
            )

        t0 = time.time()

        if slots:
            english = self.from_slots(slots)
            source = "slots"
            original = " / ".join(
                f"{SLOT_LABELS.get(k, k)}={v}"
                for k, v in slots.items()
                if v
            )
        else:
            query = (query or "").strip()
            if not query:
                raise ValueError("빈 쿼리입니다.")

            english = self.from_text(
                query,
                scope=scope,
                translate=translate,
            )
            source = "text"
            original = query

        t_prepare = time.time() - t0

        # SigLIP2는 multilingual이다. 한글 자체를 오류로 취급하지 않는다.
        # 다만 person fusion에 IRRA가 포함된 상태에서 --no-translate로 한국어를
        # 직접 넣으면 두 retriever의 언어 조건이 달라지므로 명시적으로 알린다.
        if has_hangul(english):
            if "irra" in selected_names:
                logger.warning(
                    "검색 문장에 한글이 남아 있습니다: %r\n"
                    "  SigLIP2는 multilingual query를 지원하지만 현재 IRRA 경로는 "
                    "영어 질의를 기준으로 운영합니다. person fusion의 조건을 "
                    "일관되게 하려면 번역을 사용하거나 --names siglip2 로 "
                    "SigLIP2 단독 검색을 사용하세요.",
                    english,
                )
            else:
                logger.info(
                    "한글 query를 multilingual text encoder에 직접 전달합니다: %r",
                    english,
                )

        word_count = len(english.split())

        # 기존 12단어 기준은 IRRA를 포함한 person 검색용 로컬 관찰이다.
        # object/SigLIP2 단독 검색에는 적용하지 않는다.
        if "irra" in selected_names and word_count < 12:
            logger.info(
                "IRRA 포함 person 검색의 query가 %d단어입니다. "
                "12단어 기준은 로컬 실험용 참고값이며 성능 보장은 아닙니다. "
                "필요하면 더 구체적인 외형 설명과 비교하세요.",
                word_count,
            )

        logger.info(
            "텍스트 쿼리: scope=%s collection=%s vectors=%s",
            scope,
            self.engine.store.collection,
            sorted(selected_names),
        )

        # 공개 SearchEngine API를 사용한다.
        # 각 detection을 독립적인 검색 결과로 반환한다.
        retrieval_prefetch_limit = max(
            int(self.cfg.fusion.prefetch_limit),
            int(limit),
        )

        t0 = time.time()
        hits = self.engine.search_text(
            english,
            limit=limit,
            prefetch_limit=retrieval_prefetch_limit,
            names=selected_names,
            person_only=None,
            rerank=False,
            verify=False,
        )

        hits = hits[:limit]
        for rank, hit in enumerate(hits, 1):
            hit.rank = rank

        t_retrieve = time.time() - t0

        vector_names = sorted(selected_names)
        method = (
            "단일"
            if len(vector_names) == 1
            else self.cfg.fusion.method
        )
        collection = self.engine.store.collection
        retrieval_score_type = self._retrieval_score_type(
            vector_names
        )

        translation_meta = {
            "requested": (
                self.last_translation_requested
                if source == "text"
                else None
            ),
            "applied": self.last_translation_applied,
            "backend": str(self._translator.backend),
            "model_id": self._translator.model_id,
            "expanded": bool(self.last_expanded),
        }

        return {
            "source": source,
            "scope": scope,
            "collection": collection,
            "query": original,
            "query_en": english,
            "word_count": word_count,
            "translated": self.last_translation_applied,
            "translation": translation_meta,
            "build": self.last_build,
            "slots": {
                k: v
                for k, v in (slots or {}).items()
                if v
            },
            "vectors": vector_names,
            "fusion_method": method,
            "retrieval_score_type": retrieval_score_type,
            "config": self._config_metadata(
                collection,
                vector_names,
            ),
            "hits": hits,
            "timing": {
                "prepare": round(t_prepare, 3),
                "retrieve": round(t_retrieve, 3),
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
    print(f"  점수 의미 : {res['retrieval_score_type']}")
    print(
        f"  소요      : {total:.2f}s "
        f"(준비 {t['prepare']:.2f} / 임베딩+검색 {t['retrieve']:.2f})"
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
        print(
            f"  {h.rank:>3}위  "
            f"{res['retrieval_score_type']}={h.retrieval_score:.5f}  "
            f"{tag}  {h.label}"
        )
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
        payload = h.payload if isinstance(h.payload, dict) else {}
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
            "detection_id": payload.get("detection_id", ""),
            # 기존 qwen_stage.py 호환을 위해 qdrant_score key는 유지한다.
            "qdrant_score": round(float(h.retrieval_score), 6),
            "qdrant_score_type": res["retrieval_score_type"],
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
        "vectors_used": res["vectors"],
        "fusion_method": res["fusion_method"],
        "retrieval_score_type": res["retrieval_score_type"],
        "translation": res["translation"],
        "config": res["config"],
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
                "retrieval_score_type": res["retrieval_score_type"],
                "translation": res["translation"],
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

    translate_group = ap.add_mutually_exclusive_group()
    translate_group.add_argument(
        "--translate",
        dest="translate",
        action="store_true",
        help="YAML 기본값과 무관하게 자유 문장 번역을 활성화 (-t 전용)",
    )
    translate_group.add_argument(
        "--no-translate",
        dest="translate",
        action="store_false",
        help="YAML 기본값과 무관하게 자유 문장을 번역하지 않음 (-t 전용)",
    )
    ap.set_defaults(translate=None)

    ap.add_argument(
        "--translate-backend",
        default=None,
        choices=list(BACKENDS),
        help="번역 backend override. 생략 시 pipeline.yaml 사용",
    )
    ap.add_argument(
        "--translate-model-id",
        default=None,
        help="번역 model_id override. 생략 시 pipeline.yaml 사용",
    )
    ap.add_argument(
        "--translate-max-new-tokens",
        type=int,
        default=None,
        help="번역 max_new_tokens override",
    )
    ap.add_argument(
        "--translate-cache-size",
        type=int,
        default=None,
        help="번역 LRU cache_size override",
    )

    expand_group = ap.add_mutually_exclusive_group()
    expand_group.add_argument(
        "--expand",
        dest="expand",
        action="store_true",
        help="scope-aware query 확장 활성화 (-t 전용)",
    )
    expand_group.add_argument(
        "--no-expand",
        dest="expand",
        action="store_false",
        help="query 확장 비활성화 (-t 전용)",
    )
    ap.set_defaults(expand=None)
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
    if (
        args.translate_max_new_tokens is not None
        and args.translate_max_new_tokens <= 0
    ):
        ap.error("--translate-max-new-tokens must be > 0")
    if (
        args.translate_cache_size is not None
        and args.translate_cache_size <= 0
    ):
        ap.error("--translate-cache-size must be > 0")

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

    if args.object_only and slots:
        ap.error(
            "항목별 QueryDescriptor 슬롯은 person 검색 전용입니다. "
            "object 검색은 --text 를 사용하세요."
        )

    if slots and args.expand is not None:
        ap.error("--expand / --no-expand 는 자유 문장 --text 전용입니다.")

    if slots and args.translate is not None:
        ap.error("--translate / --no-translate 는 자유 문장 --text 전용입니다.")

    dry_scope = "object" if args.object_only else "person"

    # --dry-run 은 PipelineConfig만 읽고 Qdrant / 임베더는 올리지 않는다.
    if args.dry_run:
        dry_cfg = PipelineConfig.load(args.config)
        dry_runtime = _resolve_query_settings(
            dry_cfg,
            translate_enabled=args.translate,
            translate_backend=args.translate_backend,
            translate_model_id=args.translate_model_id,
            translate_max_new_tokens=args.translate_max_new_tokens,
            translate_cache_size=args.translate_cache_size,
            expand_query=args.expand,
        )

        tr = QueryTranslator(
            backend=str(dry_runtime["backend"]),
            model_id=dry_runtime["model_id"],
            max_new_tokens=int(dry_runtime["max_new_tokens"]),
            cache_size=int(dry_runtime["cache_size"]),
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
                do_translate = bool(
                    dry_runtime["translation_enabled"]
                )
                en = (
                    tr.translate(args.text)
                    if do_translate
                    else args.text
                )
                if dry_runtime["expand_query"]:
                    en = tr.expand(
                        en,
                        scope=dry_scope,
                    )
                print(f"  원문 : {args.text}")
                print(f"  번역 : {en}")

            print()
            if has_hangul(en):
                if dry_scope == "person":
                    print(
                        "  주의: SigLIP2는 한국어 query를 지원하지만, 기본 person "
                        "검색에는 IRRA도 포함됩니다. 두 retriever의 언어 조건을 "
                        "맞추려면 번역을 사용하거나 SigLIP2 단독 검색을 사용하세요."
                    )
                else:
                    print(
                        "  참고: object 기본 검색은 SigLIP2 단독이며 "
                        "multilingual query를 직접 처리할 수 있습니다."
                    )
                print()

            if dry_scope == "person" and len(en.split()) < 12:
                print(
                    "  참고: IRRA 포함 person 검색에서 12단어 미만은 "
                    "로컬 실험용 경고 기준입니다. 더 구체적인 설명과 비교해 보세요."
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
        translate_enabled=args.translate,
        translate_backend=args.translate_backend,
        translate_model_id=args.translate_model_id,
        translate_max_new_tokens=args.translate_max_new_tokens,
        translate_cache_size=args.translate_cache_size,
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