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
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PipelineConfig
from registry import EmbedderRegistry
from router import Router
from search import SearchEngine, SearchHit

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
        "cluster_id": payload.get("cluster_id"),
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
        cluster_id = p.get("cluster_id")
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
            from query_translate import (
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
    STAGE1 = {
        "person": ["siglip2", "irra"],
        "object": ["siglip2", "dinov2"],
    }

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

    def _rerank_solider(
        self,
        query_vec: np.ndarray,
        hits: List[SearchHit],
    ) -> List[SearchHit]:
        if not hits:
            return hits

        records = self.engine.store.client.retrieve(
            collection_name=self.engine.store.collection,
            ids=[h.point_id for h in hits],
            with_vectors=["solider"],
            with_payload=False,
        )

        by_id: Dict[str, np.ndarray] = {}

        for rec in records:
            vec = getattr(rec, "vector", None)
            if vec is None:
                vec = getattr(rec, "vectors", None)
            if isinstance(vec, dict):
                vec = vec.get("solider")
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
                    "SOLIDER vector dimension mismatch: "
                    f"query={q.shape}, candidate={v.shape}"
                )

            score = float(np.dot(q, v))
            h.score = score
            h.payload["solider_score"] = score
            scored.append(h)

        if missing:
            raise RuntimeError(
                "Some person candidates have no SOLIDER vector: "
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
    ) -> Dict[str, Any]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(path)

        collection = self._select_collection(scope)

        # GUI/운영 경로는 고정한다.
        # person: STEP 1 SigLIP2 + IRRA -> STEP 2 SOLIDER rerank
        # object: SigLIP2 + DINOv2 fusion
        selected = list(self.STAGE1[scope])
        solider_rerank = scope == "person"

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
            solider_map = self.engine.router.embed_query_image(
                str(path),
                scope="person",
                names=["solider"],
            )
            if "solider" not in solider_map:
                raise RuntimeError(
                    "SOLIDER query embedding failed"
                )
            hits = self._rerank_solider(
                solider_map["solider"],
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
            "pipeline": (
                "siglip2+irra -> solider_rerank"
                if scope == "person"
                else "siglip2+dinov2"
            ),
            "solider_rerank": bool(solider_rerank),
            "timing": {
                "embed": t_embed,
                "search": t_search,
                "solider": t_solider,
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

        specs = (
            self.cfg.for_person()
            if scope == "person"
            else self.cfg.for_object()
        )

        available = [
            spec.name
            for spec in specs
            if spec.supports_text
        ]

        # 자연어 검색은 pipeline.yaml에서 supports_text=true인
        # 표준 벡터만 사용한다. person=SigLIP2+IRRA, object=SigLIP2.
        selected = available

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
        "cluster_id": best_payload.get("cluster_id"),
        "group_summary": summary,
        "payload": best_payload,
    }


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

        t0 = time.time()
        groups = grouped_query(
            client,
            collection=collection,
            vector_name=vector_name,
            query_vector=vectors[vector_name],
            limit=args.top_k,
            group_size=args.group_size,
        )
        t_search = time.time() - t0

        rows = []
        base_filter = video_filter()

        for rank, group in enumerate(
            groups,
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

            rows.append(
                grouped_result_to_dict(
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
            )

        return {
            "mode": "image-video",
            "scope": args.scope,
            "collection": collection,
            "media": "video",
            "qwen": False,
            "query_image": str(image_path),
            "vectors": [vector_name],
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

    if scope == "person":
        rows.sort(
            key=lambda x: (
                -x.rrf_score,
                min(x.vector_ranks.values())
                if x.vector_ranks
                else 10**9,
            )
        )
    else:
        rows.sort(
            key=lambda x: -x.vector_scores.get(
                "siglip2",
                -1e9,
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

        text_vectors = (
            ["siglip2", "irra"]
            if args.scope == "person"
            else ["siglip2"]
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
        final = merged[:args.top_k]
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

            if args.scope == "person":
                score = item.rrf_score
            else:
                score = item.vector_scores.get(
                    "siglip2",
                    item.rrf_score,
                )

            rows.append(
                grouped_result_to_dict(
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
                        if args.scope == "person"
                        else None
                    ),
                )
            )

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
            )
        finally:
            searcher.release()

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
            )
        finally:
            searcher.release()

        print("\nquery      :", res["query"])
        print("query_en   :", res["query_en"])

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
