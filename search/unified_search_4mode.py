#수정하셈 
from __future__ import annotations

"""
unified_search_4mode.py
=======================

GUI 연결 전 단계의 통합 검색 CLI.

검색 영역은 DB의 media_type 기준으로 명확히 분리한다.

1) crop
   잘라진 query crop -> 이미지 DB 검색만 수행
   - media_type == "image" 강제
   - person: STEP 1 SigLIP2 + IRRA -> STEP 2 SOLIDER 재정렬
   - object: SigLIP2 + DINOv2

2) text
   자연어 -> 이미지 DB 검색만 수행
   - media_type == "image" 강제
   - person: SigLIP2 + IRRA
   - object: SigLIP2
   - 한글이면 query_translate.py 사용 가능
   - Qwen rerank/verification은 이 파일에서 수행하지 않는다.

3) image-video
   이미지/crop -> stitched video identity 그룹 검색
   - media_type == "video" 강제
   - person: siglip2 / irra / solider 중 1개
   - object: siglip2 / dinov2 중 1개
   - group_by=track_key

4) text-video
   자연어 -> stitched video identity 그룹 검색
   - media_type == "video" 강제
   - person: SigLIP2 + IRRA RRF
   - object: SigLIP2
   - group_by=track_key
   - Qwen rerank/verification은 이 파일에서 수행하지 않는다.

Qwen은 검색 후보 생성과 분리된 별도 후처리 단계(qwen_stage.py)로 유지한다.

예시
----
python unified_search_4mode.py crop --image query_crop.jpg --scope person -k 20
python unified_search_4mode.py crop --image query_crop.jpg --scope object -k 20
python unified_search_4mode.py text -t "검은 상의를 입은 사람" --scope person -k 20
python unified_search_4mode.py text -t "검은 가방" --scope object -k 20
python unified_search_4mode.py image-video --image query.jpg --scope person --vector solider -k 10
python unified_search_4mode.py image-video --image query.jpg --scope object --vector dinov2 -k 10
python unified_search_4mode.py text-video -t "검은 상의를 입은 사람" --scope person -k 10
python unified_search_4mode.py text-video -t "검은 가방" --scope object -k 10
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PipelineConfig
from registry import EmbedderRegistry
from router import Router
from search.search import SearchEngine, SearchHit

CONFIG_PATH = ROOT / "pipeline.yaml"
RRF_K = 60

logger = logging.getLogger("unified_search")


# =============================================================================
# Common helpers
# =============================================================================

def normalize(v: Any) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError("vector contains NaN/Inf")
    n = float(np.linalg.norm(arr))
    if n <= 1e-12:
        raise ValueError("zero vector cannot be normalized")
    return arr / n


def fmt_time(seconds: Any) -> str:
    try:
        sec = float(seconds)
    except (TypeError, ValueError):
        return "N/A"
    if sec < 0:
        return "N/A"
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


def image_filter() -> Filter:
    """이미지 GUI/검색 경로는 image point만 보도록 강제한다."""
    return Filter(
        must=[
            FieldCondition(
                key="media_type",
                match=MatchValue(value="image"),
            )
        ]
    )


def video_filter() -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="media_type",
                match=MatchValue(value="video"),
            )
        ]
    )


def and_filter(
    base: Optional[Filter],
    *conditions: FieldCondition,
) -> Filter:
    must = []
    if base is not None and base.must:
        must.extend(base.must)
    must.extend(conditions)
    return Filter(must=must)


def collection_for_scope(
    cfg: PipelineConfig,
    scope: str,
) -> str:
    if scope == "person":
        return cfg.person_collection()
    if scope == "object":
        return cfg.object_collection()
    raise ValueError("scope must be person or object")


def identity_key_for_scope(scope: str) -> str:
    return "person_id" if scope == "person" else "object_id"


def default_group_vector(scope: str) -> str:
    # identity search 湲곕낯媛�:
    # person = SOLIDER, object = DINOv2
    return "solider" if scope == "person" else "dinov2"


def allowed_image_vectors(scope: str) -> List[str]:
    if scope == "person":
        return ["siglip2", "irra", "solider"]
    return ["siglip2", "dinov2"]


# =============================================================================
# 검색 모델 선택 (GUI/CLI 가 retriever 조합을 고른다; pipeline.yaml 의 retrievers 가 후보)
# =============================================================================

COMBO_ALL = "__all__"     # "가능한 것 전부 RRF 조합" 을 뜻하는 선택값

# 운영 기본 1차 조합. 여기 없는 retriever 를 골라도 된다 (yaml 에 있으면).
DEFAULT_STAGE1 = {
    "person": ["siglip2", "irra"],
    "object": ["siglip2", "dinov2"],
}
DEFAULT_RERANK = {"person": "solider", "object": None}


def scope_specs(cfg: PipelineConfig, scope: str):
    if scope == "person":
        return cfg.for_person()
    if scope == "object":
        return cfg.for_object()
    raise ValueError("scope must be person or object")


def image_retriever_names(cfg: PipelineConfig, scope: str) -> List[str]:
    """이미지 쿼리를 받을 수 있는 retriever (해당 scope 의 전부)."""
    return [s.name for s in scope_specs(cfg, scope)]


def text_retriever_names(cfg: PipelineConfig, scope: str) -> List[str]:
    """자연어 쿼리를 받을 수 있는 retriever (supports_text)."""
    return [s.name for s in scope_specs(cfg, scope) if s.supports_text]


def _dedupe(names: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(str(n).strip() for n in names if str(n).strip()))


def resolve_stage1(cfg: PipelineConfig, scope: str, requested: Optional[Sequence[str]] = None) -> List[str]:
    """crop 검색 1차 후보 retriever 목록.
    None/빈 값 → 운영 기본(DEFAULT_STAGE1 중 yaml 에 있는 것, 없으면 전부); [COMBO_ALL] → 전부; 그 외 → 검증된 목록."""
    available = image_retriever_names(cfg, scope)
    if not available:
        raise ValueError(f"{scope}: pipeline.yaml 에 retriever 가 없습니다")
    if not requested:
        preset = [n for n in DEFAULT_STAGE1.get(scope, []) if n in available]
        return preset or available
    names = _dedupe(requested)
    if names == [COMBO_ALL]:
        return list(available)
    bad = [n for n in names if n not in available]
    if bad:
        raise ValueError(f"{scope} 에서 쓸 수 없는 retriever: {bad} (가능: {available})")
    return names


def resolve_rerank(cfg: PipelineConfig, scope: str, requested: Optional[str] = None) -> Optional[str]:
    """2차 재정렬 벡터. None → 운영 기본(person solider); 'none'/'' → 재정렬 없음; 이름 → 검증."""
    available = image_retriever_names(cfg, scope)
    if requested is None:
        default = DEFAULT_RERANK.get(scope)
        return default if default in available else None
    name = str(requested).strip().lower()
    if name in ("", "none", "no", "off"):
        return None
    if name not in available:
        raise ValueError(f"{scope} 에서 쓸 수 없는 재정렬 retriever: {name!r} (가능: {available})")
    return name


def resolve_text_vectors(cfg: PipelineConfig, scope: str, requested: Optional[Sequence[str]] = None) -> List[str]:
    """자연어 검색에 쓸 retriever 목록. None/빈 값/[COMBO_ALL] → supports_text 전부; 그 외 → 검증된 목록."""
    available = text_retriever_names(cfg, scope)
    if not available:
        raise ValueError(f"{scope}: supports_text=true 인 retriever 가 pipeline.yaml 에 없습니다")
    if not requested:
        return list(available)
    names = _dedupe(requested)
    if names == [COMBO_ALL]:
        return list(available)
    bad = [n for n in names if n not in available]
    if bad:
        raise ValueError(f"{scope} 자연어 검색에 쓸 수 없는 retriever: {bad} (가능: {available})")
    return names


def pipeline_label(stage1: Sequence[str], rerank: Optional[str]) -> str:
    text = "+".join(stage1)
    return f"{text} -> {rerank}_rerank" if rerank else text


def hit_to_dict(h: SearchHit) -> Dict[str, Any]:
    payload = dict(h.payload or {})
    return {
        "rank": int(h.rank),
        "score": float(h.score),
        "retrieval_score": float(h.retrieval_score),
        "point_id": str(h.point_id),
        "image_id": str(h.image_id),
        "label": str(h.label),
        "is_person": bool(h.is_person),
        "crop_path": str(h.crop_path or ""),
        "bbox": [float(x) for x in (h.bbox or [])],
        "frame_idx": int(h.frame_idx),
        "track_id": h.track_id,
        "media_type": payload.get("media_type"),
        "video": payload.get("video") or payload.get("video_name"),
        "video_path": payload.get("video_path"),
        "timestamp_sec": payload.get("timestamp_sec"),
        "time_mmss": payload.get("time_mmss"),
        "person_id": payload.get("person_id"),
        "object_id": payload.get("object_id"),
        "stitched_id": payload.get("stitched_id"),
        "track_key": payload.get("track_key"),
        # 현행 Leiden 군집 id. 구버전 MiniBatch-KMeans 가 남긴 정수 cluster_id 는
        # 다른 알고리즘의 결과라 같은 이름으로 섞지 않고 legacy 로만 노출한다.
        "cluster_id": payload.get("cluster_leiden_id"),
        "legacy_kmeans_cluster_id": payload.get("cluster_id"),
        "payload": payload,
    }


def print_general_hits(
    title: str,
    hits: List[SearchHit],
    *,
    vectors: Iterable[str],
    scope: str,
    collection: str,
    media: str,
    timing: Dict[str, float],
    show: int,
) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"scope      : {scope}")
    print(f"collection : {collection}")
    print(f"media      : {media}")
    print(f"vectors    : {' + '.join(vectors)}")
    print(
        "timing     : "
        + " / ".join(
            f"{k}={v:.3f}s"
            for k, v in timing.items()
            if v is not None
        )
    )
    print(f"results    : {len(hits)}")
    print("=" * 100)

    if not hits:
        print("No results.")
        return

    for h in hits[:show]:
        p = h.payload or {}
        media_type = p.get("media_type", "")
        video = p.get("video") or p.get("video_name") or ""
        timestamp = p.get("timestamp_sec")
        stitched = p.get("stitched_id", "")
        identity = (
            p.get("person_id")
            if scope == "person"
            else p.get("object_id")
        )
        cluster_id = p.get("cluster_leiden_id")
        crop_name = (
            Path(h.crop_path).name
            if h.crop_path
            else str(h.point_id)[:12]
        )

        print(
            f"[{int(h.rank):02d}] "
            f"score={float(h.score):.6f} | "
            f"media={media_type} | label={h.label}"
        )
        print(
            f"     identity={identity} | stitched_id={stitched} "
            f"| cluster_id={cluster_id}"
        )
        if video:
            print(
                f"     video={video} | "
                f"time={fmt_time(timestamp)} | "
                f"frame={h.frame_idx}"
            )
        else:
            print(f"     image_id={h.image_id}")
        print(f"     crop={crop_name}")


def write_json(
    path: Optional[str],
    payload: Dict[str, Any],
) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nJSON saved: {out}")


# =============================================================================
# Korean text translation helper
# =============================================================================

class TextQueryHelper:
    def __init__(
        self,
        backend: str = "opus",
        model_id: Optional[str] = None,
        expand: bool = False,
    ) -> None:
        try:
            from search.query_translate import (
                QueryTranslator,
                has_hangul,
            )
        except Exception as exc:
            raise RuntimeError(
                "text mode requires query_translate.py"
            ) from exc

        self._has_hangul = has_hangul
        self._translator = QueryTranslator(
            backend=backend,
            model_id=model_id,
        )
        self.expand = expand

    def prepare(
        self,
        text: str,
        *,
        translate: bool = True,
    ) -> Dict[str, Any]:
        original = text.strip()
        if not original:
            raise ValueError("text query is empty")

        english = original

        if translate and self._has_hangul(original):
            english = self._translator.translate(original)

        if self.expand:
            english = self._translator.expand(english)

        if self._has_hangul(english):
            logger.warning(
                "�곷Ц 蹂��� �꾩뿉�� �쒓��� �⑥븘 �덉뒿�덈떎: %r",
                english,
            )

        return {
            "original": original,
            "english": english,
            "translated": english != original,
            "word_count": len(english.split()),
        }

    def release(self) -> None:
        try:
            self._translator.release()
        except Exception:
            pass


# =============================================================================
# Mode 1: crop -> image DB only
# =============================================================================

class CropGeneralSearcher:
    def __init__(
        self,
        config_path: str,
    ) -> None:
        self.engine = SearchEngine.from_config(
            config_path,
            project_root=ROOT,
        )
        self.cfg = self.engine.cfg

    def _select_collection(self, scope: str) -> str:
        collection = collection_for_scope(self.cfg, scope)
        if not self.engine.store.client.collection_exists(collection):
            raise RuntimeError(
                f"Qdrant collection does not exist: {collection}"
            )
        self.engine.store.collection = collection
        return collection

    def _rerank_by_vector(
        self,
        name: str,
        query_vec: np.ndarray,
        hits: List[SearchHit],
    ) -> List[SearchHit]:
        """후보를 named vector `name` 의 cosine 으로 다시 정렬한다 (기존 SOLIDER 재정렬을 일반화)."""
        if not hits:
            return hits

        records = self.engine.store.client.retrieve(
            collection_name=self.engine.store.collection,
            ids=[h.point_id for h in hits],
            with_vectors=[name],
            with_payload=False,
        )

        by_id: Dict[str, np.ndarray] = {}

        for rec in records:
            vec = getattr(rec, "vector", None)
            if vec is None:
                vec = getattr(rec, "vectors", None)
            if isinstance(vec, dict):
                vec = vec.get(name)
            if vec is not None:
                by_id[str(rec.id)] = normalize(vec)

        q = normalize(query_vec)

        scored: List[SearchHit] = []
        missing = []

        for h in hits:
            v = by_id.get(str(h.point_id))
            if v is None:
                missing.append(h.point_id)
                continue

            if q.shape != v.shape:
                raise RuntimeError(
                    f"{name} vector dimension mismatch: "
                    f"query={q.shape}, candidate={v.shape}"
                )

            score = float(np.dot(q, v))
            h.score = score
            h.payload[f"{name}_score"] = score
            h.payload["rerank_score"] = score
            h.payload["rerank_vector"] = name
            scored.append(h)

        if missing:
            raise RuntimeError(
                f"Some {self.engine.store.collection} candidates have no {name} vector: "
                f"{len(missing)}/{len(hits)}"
            )

        scored.sort(
            key=lambda h: float(h.score),
            reverse=True,
        )
        for i, h in enumerate(scored, start=1):
            h.rank = i
        return scored

    def search(
        self,
        image_path: str,
        *,
        scope: str,
        limit: int,
        solider_pool: int,
        stage1: Optional[Sequence[str]] = None,
        rerank: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(path)

        collection = self._select_collection(scope)

        # 모델 조합은 호출부(GUI/CLI)가 고른다. 기본은 운영 조합:
        # person: STEP 1 SigLIP2 + IRRA -> STEP 2 SOLIDER rerank / object: SigLIP2 + DINOv2 fusion
        selected = resolve_stage1(self.cfg, scope, stage1)
        rerank_name = resolve_rerank(self.cfg, scope, rerank)
        solider_rerank = rerank_name is not None

        t0 = time.time()
        qvecs = self.engine.router.embed_query_image(
            str(path),
            scope=scope,
            names=selected,
        )
        t_embed = time.time() - t0

        fetch = (
            max(limit, solider_pool)
            if solider_rerank
            else limit
        )

        t0 = time.time()
        points = self.engine._fetch(
            qvecs,
            final_limit=fetch,
            prefetch_limit=None,
            weights=None,
            extra_filter=image_filter(),
            person_only=None,
            need=fetch,
        )
        hits = self.engine._to_hits(points)
        t_search = time.time() - t0

        t_solider = 0.0

        if solider_rerank and hits:
            t0 = time.time()
            rerank_map = self.engine.router.embed_query_image(
                str(path),
                scope=scope,
                names=[rerank_name],
            )
            if rerank_name not in rerank_map:
                raise RuntimeError(
                    f"{rerank_name} query embedding failed"
                )
            hits = self._rerank_by_vector(
                rerank_name,
                rerank_map[rerank_name],
                hits,
            )
            t_solider = time.time() - t0

        hits = hits[:limit]
        for i, h in enumerate(hits, start=1):
            h.rank = i

        return {
            "mode": "crop",
            "scope": scope,
            "collection": collection,
            "media": "image",
            "qwen": False,
            "query_image": str(path),
            "vectors": sorted(qvecs),
            "pipeline": pipeline_label(selected, rerank_name),
            "models": {"stage1": list(selected), "rerank": rerank_name},
            "solider_rerank": rerank_name == "solider",
            "timing": {
                "embed": t_embed,
                "search": t_search,
                "rerank": t_solider,
            },
            "hits": hits,
        }

    def release(self) -> None:
        try:
            self.engine.registry.release()
        except Exception:
            pass


# =============================================================================
# Mode 2: text -> image DB only
# =============================================================================

class TextGeneralSearcher:
    def __init__(
        self,
        config_path: str,
        *,
        translate_backend: str,
        translate_model_id: Optional[str],
        expand: bool,
    ) -> None:
        self.engine = SearchEngine.from_config(
            config_path,
            project_root=ROOT,
        )
        self.cfg = self.engine.cfg
        self.query_helper = TextQueryHelper(
            backend=translate_backend,
            model_id=translate_model_id,
            expand=expand,
        )

    def search(
        self,
        text: str,
        *,
        scope: str,
        limit: int,
        translate: bool,
        vectors: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        collection = collection_for_scope(
            self.cfg,
            scope,
        )

        if not self.engine.store.client.collection_exists(
            collection
        ):
            raise RuntimeError(
                f"Qdrant collection does not exist: {collection}"
            )

        self.engine.store.collection = collection

        # 자연어 검색은 pipeline.yaml 에서 supports_text=true 인 retriever 만 쓴다.
        # 호출부가 그 부분집합을 고를 수 있다 (기본: 전부 RRF — person SigLIP2+IRRA, object SigLIP2).
        selected = resolve_text_vectors(self.cfg, scope, vectors)

        t0 = time.time()
        prepared = self.query_helper.prepare(
            text,
            translate=translate,
        )
        t_prepare = time.time() - t0

        t0 = time.time()
        qvecs = self.engine.router.embed_query_text(
            prepared["english"],
            names=selected,
        )
        t_embed = time.time() - t0

        t0 = time.time()
        points = self.engine._fetch(
            qvecs,
            final_limit=limit,
            prefetch_limit=None,
            weights=None,
            extra_filter=image_filter(),
            person_only=None,
            need=limit,
        )
        hits = self.engine._to_hits(points)
        t_search = time.time() - t0

        return {
            "mode": "text",
            "scope": scope,
            "collection": collection,
            "media": "image",
            "qwen": False,
            "query": prepared["original"],
            "query_en": prepared["english"],
            "translated": prepared["translated"],
            "word_count": prepared["word_count"],
            "vectors": sorted(qvecs),
            "pipeline": pipeline_label(selected, None),
            "models": {"vectors": list(selected)},
            "timing": {
                "prepare": t_prepare,
                "embed": t_embed,
                "search": t_search,
            },
            "hits": hits,
        }

    def release(self) -> None:
        try:
            self.engine.registry.release()
        except Exception:
            pass
        self.query_helper.release()


# =============================================================================
# Grouped video common
# =============================================================================

def scroll_all_payloads(
    client: QdrantClient,
    collection: str,
    filt: Filter,
    page_size: int = 256,
) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=filt,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        payloads.extend(
            (p.payload or {})
            for p in points
        )

        if next_offset is None:
            break

        offset = next_offset

    return payloads


def group_summary(
    payloads: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not payloads:
        return {
            "count": 0,
            "original_tracks": [],
            "start_sec": None,
            "end_sec": None,
            "videos": [],
        }

    original_tracks = sorted({
        int(p["original_track_id"])
        for p in payloads
        if p.get("original_track_id") is not None
    })

    times = []
    for p in payloads:
        value = p.get("timestamp_sec")
        if value is None:
            continue
        try:
            times.append(float(value))
        except (TypeError, ValueError):
            pass

    videos = sorted({
        str(
            p.get("video")
            or p.get("video_name")
            or ""
        )
        for p in payloads
        if (
            p.get("video")
            or p.get("video_name")
        )
    })

    return {
        "count": len(payloads),
        "original_tracks": original_tracks,
        "start_sec": min(times) if times else None,
        "end_sec": max(times) if times else None,
        "videos": videos,
    }


def grouped_query(
    client: QdrantClient,
    *,
    collection: str,
    vector_name: str,
    query_vector: np.ndarray,
    limit: int,
    group_size: int,
) -> list:
    if not hasattr(client, "query_points_groups"):
        raise RuntimeError(
            "qdrant-client has no query_points_groups(). "
            "Upgrade qdrant-client."
        )

    result = client.query_points_groups(
        collection_name=collection,
        query=normalize(query_vector).tolist(),
        using=vector_name,
        query_filter=video_filter(),
        group_by="track_key",
        limit=limit,
        group_size=group_size,
        with_payload=True,
        with_vectors=False,
    )

    return list(
        getattr(result, "groups", [])
        or []
    )


def grouped_result_to_dict(
    *,
    rank: int,
    group_id: str,
    best_payload: Dict[str, Any],
    score: float,
    summary: Dict[str, Any],
    scope: str,
    vector_scores: Optional[Dict[str, float]] = None,
    vector_ranks: Optional[Dict[str, int]] = None,
    rrf_score: Optional[float] = None,
) -> Dict[str, Any]:
    identity_key = identity_key_for_scope(scope)

    return {
        "rank": rank,
        "group_id": group_id,
        "score": score,
        "rrf_score": rrf_score,
        "vector_scores": vector_scores or {},
        "vector_ranks": vector_ranks or {},
        "identity_id": best_payload.get(identity_key),
        "stitched_id": best_payload.get("stitched_id"),
        "label": best_payload.get("label"),
        "track_key": group_id,
        "video": (
            best_payload.get("video")
            or best_payload.get("video_name")
        ),
        "video_path": best_payload.get("video_path"),
        "frame_idx": best_payload.get("frame_idx"),
        "timestamp_sec": best_payload.get("timestamp_sec"),
        "crop_path": best_payload.get("crop_path"),
        "cluster_id": best_payload.get("cluster_leiden_id"),
        "legacy_kmeans_cluster_id": best_payload.get("cluster_id"),
        "group_summary": summary,
        "payload": best_payload,
    }


VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v")


def video_key_of(payload: Optional[Dict[str, Any]]) -> str:
    """payload 의 영상 이름을 경로·확장자 없이 정규화한다 (영상당 상한 계산용).

    source 에 따라 'video' 가 'Normal_Videos_935_x264.mp4'(video_tracks_stitched) 이거나
    'Normal_Videos_935_x264'(final_db_candidates) 라서 같은 영상이 두 키로 갈라지는 것을 막는다.
    알 수 없으면 '' 를 돌려준다.
    """
    payload = payload or {}
    for key in ("video", "video_name", "video_path", "video_relpath"):
        value = payload.get(key)
        if not value:
            continue
        name = str(value).replace("\\", "/").rstrip("/").split("/")[-1]
        stem, dot, ext = name.rpartition(".")
        if dot and ("." + ext).lower() in VIDEO_SUFFIXES:
            name = stem
        if name:
            return name
    return ""


def group_video_key(group: Any) -> str:
    """qdrant query_points_groups 의 group 객체에서 대표 hit 의 영상 키를 뽑는다."""
    hits = list(getattr(group, "hits", []) or [])
    payload = (getattr(hits[0], "payload", None) or {}) if hits else {}
    return video_key_of(payload)


def select_with_per_video_cap(
    items: List[Any],
    *,
    top_k: int,
    per_video_max: int,
    video_of: Callable[[Any], str],
) -> Tuple[List[Any], Dict[str, int], Dict[str, int]]:
    """순위 순서를 유지하면서 영상당 최대 per_video_max 개만 남기고 top_k 개를 고른다.

    per_video_max <= 0 이면 상한 없음(= items[:top_k]).
    반환: (선택된 항목, 영상별로 상한 때문에 접힌 개수, 영상별 후보 총수).
    접힌 항목은 삭제가 아니라 표시에서 숨긴 것이다. 같은 영상의 여러 track 은 대개 서로 다른 사람이므로
    (SOLIDER 유사도 0.9 이상이어도 다른 사람인 경우가 많다) 임베딩으로 합치지 않고 개수만 제한한다.
    """
    top_k = max(0, int(top_k))
    cap = int(per_video_max or 0)
    kept: List[Any] = []
    shown: Dict[str, int] = {}
    hidden: Dict[str, int] = {}
    total: Dict[str, int] = {}
    for item in items:
        key = video_of(item) or ""
        total[key] = total.get(key, 0) + 1
        if len(kept) >= top_k:
            continue
        if cap > 0 and shown.get(key, 0) >= cap:
            hidden[key] = hidden.get(key, 0) + 1
            continue
        shown[key] = shown.get(key, 0) + 1
        kept.append(item)
    return kept, hidden, total


def annotate_video_cap(
    row: Dict[str, Any],
    payload: Optional[Dict[str, Any]],
    hidden_by_video: Dict[str, int],
    total_by_video: Dict[str, int],
) -> Dict[str, Any]:
    """결과 row 에 영상 키와 '같은 영상에서 접힌 개수' 를 붙인다 (GUI 표시용)."""
    key = video_key_of(payload)
    row["video_key"] = key
    row["same_video_hidden"] = int(hidden_by_video.get(key, 0))
    row["same_video_candidates"] = int(total_by_video.get(key, 0))
    return row


def print_grouped_rows(
    title: str,
    rows: List[Dict[str, Any]],
    *,
    scope: str,
    collection: str,
    vectors: List[str],
) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"scope      : {scope}")
    print(f"collection : {collection}")
    print(f"vectors    : {' + '.join(vectors)}")
    print("media      : video")
    print("group_by   : track_key")
    print(f"results    : {len(rows)}")
    print("=" * 100)

    if not rows:
        print("No grouped video results.")
        return

    for row in rows:
        print("\n" + "-" * 100)

        if row.get("rrf_score") is not None:
            print(
                f"[{row['rank']:02d}] "
                f"RRF={row['rrf_score']:.8f}"
            )
            for name in vectors:
                s = row["vector_scores"].get(name)
                r = row["vector_ranks"].get(name)
                print(
                    f"     {name}: "
                    f"score={s if s is not None else 'N/A'} "
                    f"rank={r if r is not None else 'N/A'}"
                )
        else:
            print(
                f"[{row['rank']:02d}] "
                f"score={row['score']:.6f}"
            )

        print(
            f"     identity_id={row['identity_id']} | "
            f"stitched_id={row['stitched_id']} | "
            f"label={row['label']} | "
            f"cluster_id={row['cluster_id']}"
        )
        print(f"     track_key={row['track_key']}")
        print(f"     video={row['video']}")
        if row.get("same_video_hidden"):
            print(
                f"     same video: {row['same_video_hidden']} more track(s) "
                "hidden by --per-video-max"
            )

        gs = row["group_summary"]
        print(
            f"     points={gs['count']} | "
            f"original_tracks={gs['original_tracks']} | "
            f"range={fmt_time(gs['start_sec'])} ~ "
            f"{fmt_time(gs['end_sec'])}"
        )
        print(
            f"     representative frame={row['frame_idx']} | "
            f"time={fmt_time(row['timestamp_sec'])}"
        )
        print(f"     crop={row['crop_path']}")


# =============================================================================
# Mode 3: image -> grouped stitched video identity
# =============================================================================

def run_image_video(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    cfg = PipelineConfig.load(args.config)
    reg = EmbedderRegistry(cfg)
    router = Router(
        cfg,
        reg,
        input_format="rgb",
    )
    client = QdrantClient(
        url=cfg.qdrant.url,
        timeout=120,
    )

    try:
        collection = collection_for_scope(
            cfg,
            args.scope,
        )

        vector_name = (
            args.vector
            or default_group_vector(args.scope)
        )

        allowed = allowed_image_vectors(
            args.scope
        )
        if vector_name not in allowed:
            raise ValueError(
                f"{args.scope} does not support "
                f"{vector_name}. allowed={allowed}"
            )

        t0 = time.time()
        vectors = router.embed_query_image(
            str(image_path),
            scope=args.scope,
            names=[vector_name],
        )
        t_embed = time.time() - t0

        if vector_name not in vectors:
            raise RuntimeError(
                f"Router did not return vector: {vector_name}"
            )

        per_video_max = int(getattr(args, "per_video_max", 0) or 0)
        # 영상당 상한을 걸면 상위 top_k 안에서 접히는 group 이 생기므로 후보 group 을 더 받아 둔다.
        group_limit = (
            args.top_k
            if per_video_max <= 0
            else max(args.top_k * 5, args.top_k + 40)
        )

        t0 = time.time()
        groups = grouped_query(
            client,
            collection=collection,
            vector_name=vector_name,
            query_vector=vectors[vector_name],
            limit=group_limit,
            group_size=args.group_size,
        )
        candidates = [
            g for g in groups
            if getattr(g, "id", None) is not None
            and list(getattr(g, "hits", []) or [])
        ]
        selected, hidden_by_video, total_by_video = select_with_per_video_cap(
            candidates,
            top_k=args.top_k,
            per_video_max=per_video_max,
            video_of=group_video_key,
        )
        t_search = time.time() - t0

        rows = []
        base_filter = video_filter()

        for rank, group in enumerate(
            selected,
            start=1,
        ):
            group_id = str(
                getattr(group, "id", "")
            )
            hits = list(
                getattr(group, "hits", [])
                or []
            )
            if not group_id or not hits:
                continue

            best = hits[0]
            best_payload = best.payload or {}

            exact = and_filter(
                base_filter,
                FieldCondition(
                    key="track_key",
                    match=MatchValue(
                        value=group_id
                    ),
                ),
            )

            payloads = scroll_all_payloads(
                client,
                collection,
                exact,
            )
            summary = group_summary(payloads)

            row = grouped_result_to_dict(
                rank=rank,
                group_id=group_id,
                best_payload=best_payload,
                score=float(best.score),
                summary=summary,
                scope=args.scope,
                vector_scores={
                    vector_name: float(best.score)
                },
                vector_ranks={
                    vector_name: rank
                },
            )
            annotate_video_cap(
                row,
                best_payload,
                hidden_by_video,
                total_by_video,
            )
            rows.append(row)

        return {
            "mode": "image-video",
            "scope": args.scope,
            "collection": collection,
            "media": "video",
            "qwen": False,
            "query_image": str(image_path),
            "vectors": [vector_name],
            "pipeline": pipeline_label([vector_name], None),
            "models": {"vector": vector_name},
            "per_video_max": per_video_max,
            "per_video_hidden": hidden_by_video,
            "candidate_groups": len(candidates),
            "timing": {
                "embed": t_embed,
                "search": t_search,
            },
            "results": rows,
        }

    finally:
        try:
            reg.release()
        except Exception:
            pass


# =============================================================================
# Mode 4: text -> grouped stitched video identity
# =============================================================================

@dataclass
class GroupCandidate:
    group_id: str
    payload: Dict[str, Any] = field(
        default_factory=dict
    )
    vector_scores: Dict[str, float] = field(
        default_factory=dict
    )
    vector_ranks: Dict[str, int] = field(
        default_factory=dict
    )
    rrf_score: float = 0.0


def merge_group_results(
    grouped_by_vector: Dict[str, list],
    *,
    scope: str,
) -> List[GroupCandidate]:
    merged: Dict[str, GroupCandidate] = {}

    for vector_name, groups in grouped_by_vector.items():
        for rank, group in enumerate(
            groups,
            start=1,
        ):
            group_id = getattr(
                group,
                "id",
                None,
            )
            hits = list(
                getattr(group, "hits", [])
                or []
            )

            if group_id is None or not hits:
                continue

            group_id = str(group_id)
            best = hits[0]
            payload = best.payload or {}
            score = float(best.score)

            if group_id not in merged:
                merged[group_id] = GroupCandidate(
                    group_id=group_id,
                    payload=payload,
                )

            item = merged[group_id]
            item.vector_scores[vector_name] = score
            item.vector_ranks[vector_name] = rank
            item.rrf_score += 1.0 / (
                RRF_K + rank
            )

    rows = list(merged.values())

    # 벡터가 하나면 그 벡터의 점수로, 둘 이상이면 RRF 로 정렬한다
    # (기존 동작과 동일: person = siglip2+irra RRF, object = siglip2 단독 점수).
    names = list(grouped_by_vector)
    if len(names) == 1:
        only = names[0]
        rows.sort(
            key=lambda x: -x.vector_scores.get(
                only,
                -1e9,
            )
        )
    else:
        rows.sort(
            key=lambda x: (
                -x.rrf_score,
                min(x.vector_ranks.values())
                if x.vector_ranks
                else 10**9,
            )
        )

    return rows


def run_text_video(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    cfg = PipelineConfig.load(args.config)
    reg = EmbedderRegistry(cfg)
    router = Router(
        cfg,
        reg,
        input_format="rgb",
    )
    client = QdrantClient(
        url=cfg.qdrant.url,
        timeout=120,
    )

    query_helper = TextQueryHelper(
        backend=args.translate_backend,
        model_id=args.translate_model_id,
        expand=args.expand,
    )

    try:
        collection = collection_for_scope(
            cfg,
            args.scope,
        )

        # 호출부가 고른 supports_text retriever 부분집합 (기본: 전부 — person siglip2+irra, object siglip2)
        text_vectors = resolve_text_vectors(
            cfg,
            args.scope,
            getattr(args, "vectors", None),
        )

        t0 = time.time()
        prepared = query_helper.prepare(
            args.text,
            translate=not args.no_translate,
        )
        t_prepare = time.time() - t0

        t0 = time.time()
        vectors = router.embed_query_text(
            prepared["english"],
            names=text_vectors,
        )
        t_embed = time.time() - t0

        query_vectors: Dict[str, np.ndarray] = {}

        for name in text_vectors:
            if name not in vectors:
                raise RuntimeError(
                    "Router did not return "
                    f"text vector '{name}'. "
                    f"returned={list(vectors)}"
                )
            query_vectors[name] = normalize(
                vectors[name]
            )

        t0 = time.time()
        grouped_by_vector: Dict[str, list] = {}

        for name in text_vectors:
            grouped_by_vector[name] = grouped_query(
                client,
                collection=collection,
                vector_name=name,
                query_vector=query_vectors[name],
                limit=args.candidate_k,
                group_size=args.group_size,
            )

        merged = merge_group_results(
            grouped_by_vector,
            scope=args.scope,
        )
        per_video_max = int(getattr(args, "per_video_max", 0) or 0)
        # 영상당 상한: 순위는 그대로 두고 같은 영상의 track 은 최대 per_video_max 개만 남긴다 (0 = 제한 없음).
        final, hidden_by_video, total_by_video = select_with_per_video_cap(
            merged,
            top_k=args.top_k,
            per_video_max=per_video_max,
            video_of=lambda item: video_key_of(item.payload),
        )
        t_search = time.time() - t0

        rows = []
        base_filter = video_filter()

        for rank, item in enumerate(
            final,
            start=1,
        ):
            exact = and_filter(
                base_filter,
                FieldCondition(
                    key="track_key",
                    match=MatchValue(
                        value=item.group_id
                    ),
                ),
            )

            payloads = scroll_all_payloads(
                client,
                collection,
                exact,
            )
            summary = group_summary(
                payloads
            )

            # 벡터 하나면 그 cosine 점수, 둘 이상이면 RRF (merge_group_results 의 정렬 기준과 같다)
            if len(text_vectors) == 1:
                score = item.vector_scores.get(
                    text_vectors[0],
                    item.rrf_score,
                )
            else:
                score = item.rrf_score

            row = grouped_result_to_dict(
                rank=rank,
                group_id=item.group_id,
                best_payload=item.payload,
                score=float(score),
                summary=summary,
                scope=args.scope,
                vector_scores=item.vector_scores,
                vector_ranks=item.vector_ranks,
                rrf_score=(
                    float(item.rrf_score)
                    if len(text_vectors) > 1
                    else None
                ),
            )
            annotate_video_cap(
                row,
                item.payload,
                hidden_by_video,
                total_by_video,
            )
            rows.append(row)

        return {
            "mode": "text-video",
            "scope": args.scope,
            "collection": collection,
            "media": "video",
            "qwen": False,
            "query": prepared["original"],
            "query_en": prepared["english"],
            "translated": prepared["translated"],
            "word_count": prepared["word_count"],
            "vectors": text_vectors,
            "pipeline": pipeline_label(text_vectors, None),
            "models": {"vectors": list(text_vectors)},
            "per_video_max": per_video_max,
            "per_video_hidden": hidden_by_video,
            "candidate_groups": len(merged),
            "timing": {
                "prepare": t_prepare,
                "embed": t_embed,
                "search": t_search,
            },
            "results": rows,
        }

    finally:
        try:
            reg.release()
        except Exception:
            pass
        query_helper.release()


# =============================================================================
# CLI
# =============================================================================

def add_common_scope(
    ap: argparse.ArgumentParser,
) -> None:
    ap.add_argument(
        "--scope",
        required=True,
        choices=["person", "object"],
    )
    ap.add_argument(
        "--config",
        default=str(CONFIG_PATH),
    )
    ap.add_argument(
        "--json-out",
        default=None,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Forensic Visual Retrieval - "
            "4 search modes in one file"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
모드 요약
---------
crop        : crop 이미지 -> 이미지 DB 검색 (media_type=image)
text        : 자연어 -> 이미지 DB 검색 (media_type=image)
image-video : 이미지 -> stitched 영상 identity 검색 (media_type=video)
text-video  : 자연어 -> stitched 영상 identity 검색 (media_type=video)

Qwen rerank/verification은 별도 qwen_stage.py에서 수행한다.
""",
    )

    sub = ap.add_subparsers(
        dest="mode",
        required=True,
    )

    # ------------------------------------------------------------------ crop
    p = sub.add_parser(
        "crop",
        help="crop image -> image DB search (person: SigLIP2+IRRA->SOLIDER, object: SigLIP2+DINOv2)",
    )
    add_common_scope(p)
    p.add_argument(
        "--image",
        "--crop",
        dest="image",
        required=True,
    )
    p.add_argument(
        "--limit",
        "-k",
        type=int,
        default=20,
    )
    p.add_argument(
        "--show",
        type=int,
        default=20,
    )
    p.add_argument(
        "--solider-pool",
        type=int,
        default=200,
        help="2차 재정렬이 있을 때 1차에서 받아 두는 후보 수",
    )
    p.add_argument(
        "--stage1",
        nargs="+",
        default=None,
        metavar="RETRIEVER",
        help=(
            "1차 후보 retriever 이름들 (pipeline.yaml retrievers 중 해당 scope). "
            f"'{COMBO_ALL}' = 전부 RRF. 기본 person siglip2 irra / object siglip2 dinov2"
        ),
    )
    p.add_argument(
        "--rerank",
        default=None,
        help="2차 재정렬 retriever 이름 또는 none. 기본 person solider / object none",
    )

    # ------------------------------------------------------------------ text
    p = sub.add_parser(
        "text",
        help="natural language -> image DB search (person: SigLIP2+IRRA, object: SigLIP2)",
    )
    add_common_scope(p)
    p.add_argument(
        "--text",
        "-t",
        required=True,
    )
    p.add_argument(
        "--limit",
        "-k",
        type=int,
        default=20,
    )
    p.add_argument(
        "--show",
        type=int,
        default=20,
    )
    p.add_argument(
        "--translate-backend",
        default="opus",
    )
    p.add_argument(
        "--translate-model-id",
        default=None,
    )
    p.add_argument(
        "--no-translate",
        action="store_true",
    )
    p.add_argument(
        "--expand",
        action="store_true",
    )
    p.add_argument(
        "--vectors",
        nargs="+",
        default=None,
        metavar="RETRIEVER",
        help=f"자연어 검색에 쓸 supports_text retriever 들 ('{COMBO_ALL}' = 전부 RRF, 기본 전부)",
    )

    # ----------------------------------------------------------- image-video
    p = sub.add_parser(
        "image-video",
        help="image -> grouped stitched video identity search",
    )
    add_common_scope(p)
    p.add_argument(
        "--image",
        required=True,
    )
    p.add_argument(
        "--vector",
        default=None,
        help=(
            "person: siglip2/irra/solider, "
            "object: siglip2/dinov2. "
            "default person=solider, object=dinov2"
        ),
    )
    p.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=10,
    )
    p.add_argument(
        "--group-size",
        type=int,
        default=3,
    )
    p.add_argument(
        "--per-video-max",
        type=int,
        default=0,
        help=(
            "same video 에서 최대 몇 개의 track 을 보일지 "
            "(0 = 제한 없음). 삭제가 아니라 표시에서 접는다."
        ),
    )

    # ------------------------------------------------------------ text-video
    p = sub.add_parser(
        "text-video",
        help="text -> grouped stitched video identity search",
    )
    add_common_scope(p)
    p.add_argument(
        "--text",
        "-t",
        required=True,
    )
    p.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=10,
    )
    p.add_argument(
        "--candidate-k",
        type=int,
        default=100,
    )
    p.add_argument(
        "--group-size",
        type=int,
        default=3,
    )
    p.add_argument(
        "--per-video-max",
        type=int,
        default=0,
        help=(
            "same video 에서 최대 몇 개의 track 을 보일지 "
            "(0 = 제한 없음). 삭제가 아니라 표시에서 접는다."
        ),
    )
    p.add_argument(
        "--translate-backend",
        default="opus",
    )
    p.add_argument(
        "--translate-model-id",
        default=None,
    )
    p.add_argument(
        "--no-translate",
        action="store_true",
    )
    p.add_argument(
        "--expand",
        action="store_true",
    )
    p.add_argument(
        "--vectors",
        nargs="+",
        default=None,
        metavar="RETRIEVER",
        help=f"자연어 영상 검색에 쓸 supports_text retriever 들 ('{COMBO_ALL}' = 전부 RRF, 기본 전부)",
    )

    return ap


def validate_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if hasattr(args, "limit") and args.limit <= 0:
        parser.error("--limit must be > 0")

    if hasattr(args, "top_k") and args.top_k <= 0:
        parser.error("--top-k must be > 0")

    if (
        hasattr(args, "group_size")
        and args.group_size <= 0
    ):
        parser.error("--group-size must be > 0")

    if (
        hasattr(args, "candidate_k")
        and args.candidate_k <= 0
    ):
        parser.error("--candidate-k must be > 0")
    if (
        hasattr(args, "per_video_max")
        and args.per_video_max is not None
        and args.per_video_max < 0
    ):
        parser.error("--per-video-max must be >= 0")

    if (
        hasattr(args, "solider_pool")
        and args.solider_pool <= 0
    ):
        parser.error("--solider-pool must be > 0")

    if (
        getattr(args, "mode", None) == "crop"
        and getattr(args, "scope", None) == "person"
        and args.solider_pool < args.limit
    ):
        parser.error(
            "--solider-pool must be >= --limit"
        )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    # ================================================================= crop
    if args.mode == "crop":
        searcher = CropGeneralSearcher(
            args.config
        )
        try:
            res = searcher.search(
                args.image,
                scope=args.scope,
                limit=args.limit,
                solider_pool=args.solider_pool,
                stage1=args.stage1,
                rerank=args.rerank,
            )
        finally:
            searcher.release()

        print("\npipeline   :", res["pipeline"])
        print_general_hits(
            "CROP -> IMAGE DB SEARCH",
            res["hits"],
            vectors=res["vectors"],
            scope=res["scope"],
            collection=res["collection"],
            media=res["media"],
            timing=res["timing"],
            show=args.show,
        )

        payload = {
            k: v
            for k, v in res.items()
            if k != "hits"
        }
        payload["results"] = [
            hit_to_dict(h)
            for h in res["hits"]
        ]
        write_json(
            args.json_out,
            payload,
        )
        return 0

    # ================================================================= text
    if args.mode == "text":
        searcher = TextGeneralSearcher(
            args.config,
            translate_backend=args.translate_backend,
            translate_model_id=args.translate_model_id,
            expand=args.expand,
        )
        try:
            res = searcher.search(
                args.text,
                scope=args.scope,
                limit=args.limit,
                translate=not args.no_translate,
                vectors=args.vectors,
            )
        finally:
            searcher.release()

        print("\nquery      :", res["query"])
        print("query_en   :", res["query_en"])
        print("pipeline   :", res["pipeline"])

        print_general_hits(
            "TEXT -> IMAGE DB SEARCH",
            res["hits"],
            vectors=res["vectors"],
            scope=res["scope"],
            collection=res["collection"],
            media=res["media"],
            timing=res["timing"],
            show=args.show,
        )

        payload = {
            k: v
            for k, v in res.items()
            if k != "hits"
        }
        payload["results"] = [
            hit_to_dict(h)
            for h in res["hits"]
        ]
        write_json(
            args.json_out,
            payload,
        )
        return 0

    # ========================================================== image-video
    if args.mode == "image-video":
        res = run_image_video(args)

        print_grouped_rows(
            "IMAGE -> VIDEO IDENTITY GROUP SEARCH",
            res["results"],
            scope=res["scope"],
            collection=res["collection"],
            vectors=res["vectors"],
        )

        write_json(
            args.json_out,
            res,
        )
        return 0

    # =========================================================== text-video
    if args.mode == "text-video":
        res = run_text_video(args)

        print("\nquery      :", res["query"])
        print("query_en   :", res["query_en"])

        print_grouped_rows(
            "TEXT -> VIDEO IDENTITY GROUP SEARCH",
            res["results"],
            scope=res["scope"],
            collection=res["collection"],
            vectors=res["vectors"],
        )

        write_json(
            args.json_out,
            res,
        )
        return 0

    parser.error(
        f"unsupported mode: {args.mode}"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
