"""
통합 검색: Fusion -> Qwen3-VL 재순위 -> 자동 검증
================================================

파이프라인 위치:
    쿼리(텍스트 또는 이미지)
      -> 임베딩 (SigLIP2 / IRRA / SOLIDER / DINOv2)
      -> Qdrant fused_search
      -> Qwen3-VL 재순위 (상위 20)
      -> Qwen3-VL 자동 검증 (threshold 판정)
      -> 결과

조각들(router, qdrant_store, qwen_rerank)을 잇는 유일한 지점이다.

재순위와 검증은 같은 질문이다
--------------------------
이게 이 파일의 핵심 설계다.

  재순위(텍스트) : "이 이미지가 설명에 부합하는가?" -> P(yes)
  검증(텍스트)   : "이 이미지가 설명에 부합하는가?" -> P(yes) >= threshold

  재순위(이미지) : "이 두 사람이 같은 사람인가?" -> P(yes)
  검증(이미지)   : "이 두 사람이 같은 사람인가?" -> P(yes) >= threshold

**질문이 동일하므로 Qwen 을 두 번 부를 필요가 없다.** 재순위에서 이미 계산한
P(yes) 에 threshold 를 적용하면 그게 검증이다. logprob 방식으로 재순위를
했다면 검증은 추가 비용이 0 이다.

재순위를 끈 상태에서 검증만 켜면 그때는 실제로 Qwen 을 호출한다.

filter 가 아니라 flag 가 기본이다
-------------------------------
threshold 미달 후보를 결과에서 **지우지 않고 표시만** 한다.

수사 검색에서는 놓친 정답(미탐)이 잘못 올라온 후보(오탐)보다 훨씬 나쁘다.
사람이 목록을 보고 판단하는 것이 전제이므로, 시스템이 조용히 결과를
버리면 안 된다. threshold 를 운영 데이터로 캘리브레이션한 뒤에야
verify_mode="filter" 를 검토할 것.

threshold 는 캘리브레이션이 필요하다
---------------------------------
기본값 0.5 는 아무 근거가 없다. 라벨된 쌍(같은 사람 / 다른 사람) 수십 개로
P(yes) 분포를 그려보고, 원하는 오탐률에 맞춰 정할 것.
수사 용도라면 보통 0.8 이상으로 높게 잡아 오탐을 줄인다.

crop_path 가 payload 에 있어야 한다
---------------------------------
Qwen 은 실제 이미지 파일을 봐야 한다. 적재 시 rfdetr_adapter 가
extra["crop_path"] 를 넣어둔 것이 필수다. crop 파일을 지웠거나
keep_crop_path=False 로 적재했다면 재순위/검증을 쓸 수 없다.

사용
----
    engine = SearchEngine.from_config("pipeline.yaml")

    # 텍스트 검색 + 재순위 + 자동 검증
    results = engine.search_text(
        "a man in a red shirt carrying a black backpack",
        rerank=True, verify=True, verify_threshold=0.8,
    )

    for r in results:
        mark = {True: "OK", False: "미달", None: "미검증"}[r.verified]
        print(r.rank, mark, r.score, r.crop_path)

CLI
----
    python search.py --text "a man in a red shirt" --rerank --verify
    python search.py --image query.jpg --rerank --verify --verify-threshold 0.8
    python search.py --text "..." --no-rerank                    # 융합만
    python search.py --text "..." --rerank --verify-mode filter   # 미달 제거
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from config import PipelineConfig
from qdrant_store import QdrantStore
from registry import EmbedderRegistry
from router import Router

logger = logging.getLogger(__name__)

# payload 에서 crop 경로를 찾을 때 시도하는 key 순서
_PATH_KEYS = ("crop_path", "path", "image_path")

VERIFY_MODES = ("flag", "filter")


@dataclass
class SearchHit:
    """검색 결과 한 건."""
    rank: int
    point_id: str
    score: float                          # 최종 정렬에 쓰인 점수
    retrieval_score: float                # Qdrant fusion 점수

    rerank_score: Optional[float] = None  # Qwen P(yes). 재순위 안 했으면 None
    reason: Optional[str] = None          # 재순위 근거 (explain=True 일 때)

    verify_score: Optional[float] = None  # 검증 P(yes)
    verified: Optional[bool] = None       # None = 검증하지 않음
    verify_reason: Optional[str] = None   # 검증 근거

    crop_path: Optional[str] = None
    image_id: str = ""
    label: str = ""
    is_person: Optional[bool] = None
    bbox: List[float] = field(default_factory=list)
    frame_idx: int = 0
    track_id: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        if self.verified is None:
            return "미검증"
        return "확인" if self.verified else "미달"

    def summary(self) -> str:
        parts = [f"retrieval={self.retrieval_score:.4f}"]
        if self.rerank_score is not None:
            parts.append(f"qwen={self.rerank_score:.3f}")
        if self.verify_score is not None:
            parts.append(f"verify={self.verify_score:.3f}")

        name = Path(self.crop_path).name if self.crop_path else self.point_id[:8]
        mark = "" if self.verified is None else f"[{self.verdict}] "

        return (
            f"{self.rank:2d}. {mark}score={self.score:.4f} "
            f"({', '.join(parts)}) {self.label:12s} {name}"
        )


# --------------------------------------------------------------------------- #
class SearchEngine:
    """Fusion 검색 + Qwen 재순위 + 자동 검증을 묶은 검색 진입점."""

    def __init__(
        self,
        cfg: PipelineConfig,
        registry: Optional[EmbedderRegistry] = None,
        store: Optional[QdrantStore] = None,
        reranker: Optional[Any] = None,
        input_format: str = "rgb",
        extra_paths: Optional[Sequence[str]] = None,
        qwen_model_id: Optional[str] = None,
        qwen_dtype: str = "bfloat16",
        release_embedders_before_rerank: bool = False,
        project_root: Optional[Union[str, Path]] = None,
        crop_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.cfg = cfg
        self.registry = registry or EmbedderRegistry(cfg, extra_paths=extra_paths)
        self.store = store or QdrantStore(cfg)
        self.router = Router(cfg, self.registry, input_format=input_format)

        self._reranker = reranker          # None 이면 처음 필요할 때 로드
        self._qwen_model_id = qwen_model_id
        self._qwen_dtype = qwen_dtype
        self.release_embedders_before_rerank = release_embedders_before_rerank

        # crop 경로 재해석용. 다른 PC 에서 적재한 DB 를 열 때 쓴다.
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.crop_root = Path(
            crop_root or (self.project_root / "data" / "crops")
        ).resolve()
        self._path_cache: Dict[str, str] = {}
        self._path_remap_warned = False

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path] = "pipeline.yaml",
        **kwargs,
    ) -> "SearchEngine":
        return cls(PipelineConfig.load(config_path), **kwargs)

    # ------------------------------------------------------------------ #
    # Qwen 지연 로드
    # ------------------------------------------------------------------ #
    def reranker(self):
        """Qwen 을 필요한 시점에 한 번만 로드한다."""
        if self._reranker is not None:
            return self._reranker

        from verifiers.qwen_rerank import QwenReranker, DEFAULT_MODEL

        model_id = self._qwen_model_id or DEFAULT_MODEL
        logger.info("Qwen3-VL 로드: %s", model_id)
        self._reranker = QwenReranker(model_id=model_id, dtype=self._qwen_dtype)
        return self._reranker

    def release_reranker(self) -> None:
        if self._reranker is not None:
            self._reranker.release()
            self._reranker = None

    # ------------------------------------------------------------------ #
    # payload 파싱
    # ------------------------------------------------------------------ #
    @staticmethod
    def _raw_crop_path(payload: Dict[str, Any]) -> Optional[str]:
        for key in _PATH_KEYS:
            value = payload.get(key)
            if value:
                return str(value)
        return None

    def _crop_path(self, payload: Dict[str, Any]) -> Optional[str]:
        """
        payload 의 crop 경로를 **이 PC 에서 실제로 열 수 있는 경로**로 바꾼다.

        적재 시 저장된 경로가 그때 PC 기준이라, 프로젝트를 다른 PC 로 옮기면
        전부 무효가 된다. Qwen 재순위/검증은 파일을 직접 열어야 하므로
        그러면 조용히 전부 건너뛰어진다.

        순서대로 시도한다:
          1) 저장된 경로 그대로
          2) crop_root / 저장된 경로의 crops 이후 부분
          3) crop_root / 파일명
        찾지 못하면 원래 값을 그대로 돌려준다 (_split_usable 이 걸러낸다).
        """
        raw = self._raw_crop_path(payload)
        if raw is None:
            return None

        cached = self._path_cache.get(raw)
        if cached is not None:
            return cached

        resolved = self._resolve_crop_path(raw)
        self._path_cache[raw] = resolved
        return resolved

    def _resolve_crop_path(self, raw: str) -> str:
        text = raw.replace("\\", "/")
        p = Path(text)

        if p.is_file():
            return str(p)

        candidates: List[Path] = []

        # 프로젝트 루트 기준 상대경로로 재해석
        parts = p.parts
        for i, part in enumerate(parts):
            if part.lower() == "crops":
                # .../crops/xxx.jpg -> crop_root/xxx.jpg
                tail = Path(*parts[i + 1:]) if i + 1 < len(parts) else None
                if tail is not None:
                    candidates.append(self.crop_root / tail)
                break

        candidates.append(self.crop_root / p.name)
        candidates.append(self.project_root / text.lstrip("./"))

        for cand in candidates:
            if cand.is_file():
                if not self._path_remap_warned:
                    self._path_remap_warned = True
                    logger.warning(
                        "payload 의 crop 경로가 이 PC 에 없어 재해석했습니다.\n"
                        "  저장된 값 : %s\n"
                        "  실제 파일 : %s\n"
                        "  다른 PC 에서 적재한 DB 입니다. crop_root 를 확인하세요.",
                        raw, cand,
                    )
                return str(cand)

        return raw

    def _to_hits(self, points: Sequence[Any]) -> List[SearchHit]:
        hits: List[SearchHit] = []
        for i, p in enumerate(points, 1):
            payload = dict(getattr(p, "payload", None) or {})
            score = float(getattr(p, "score", 0.0))
            hits.append(SearchHit(
                rank=i,
                point_id=str(getattr(p, "id", "")),
                score=score,
                retrieval_score=score,
                crop_path=self._crop_path(payload),
                image_id=str(payload.get("image_id", "")),
                label=str(payload.get("label", "")),
                is_person=payload.get("is_person"),
                bbox=[float(v) for v in (payload.get("bbox") or [])],
                frame_idx=int(payload.get("frame_idx", 0) or 0),
                track_id=payload.get("track_id"),
                payload=payload,
            ))
        return hits

    # ------------------------------------------------------------------ #
    # 검색 — 텍스트
    # ------------------------------------------------------------------ #
    def search_text(
        self,
        query: str,
        limit: Optional[int] = None,
        prefetch_limit: Optional[int] = None,
        names: Optional[Sequence[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        extra_filter=None,
        person_only: Union[bool, Dict[str, Optional[bool]], None] = None,
        rerank: bool = False,
        rerank_top_k: int = 20,
        alpha: float = 0.7,
        explain: bool = False,
        verify: bool = False,
        verify_top_k: Optional[int] = None,
        verify_threshold: float = 0.5,
        verify_mode: str = "flag",
        verify_explain: bool = False,
    ) -> List[SearchHit]:
        """
        자연어 검색.

        영어로 질의할 것. IRRA 와 SigLIP2 모두 CLIP 계열 영어 토크나이저를
        쓴다. 한국어는 상위 레이어에서 번역해 넘겨야 한다.

        names 로 참여 retriever 를 제한할 수 있다 (예: ["irra"] -> IRRA 단독).
        person_only 는 보통 넘기지 않는다 — QdrantStore 가 pipeline.yaml 의
        scope 에서 자동으로 결정한다.
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("빈 쿼리입니다.")
        self._check_verify_args(verify_mode, verify_threshold)

        qvecs = self.router.embed_query_text(query, names=names)
        logger.info("텍스트 쿼리 벡터: %s", sorted(qvecs))

        final_limit = int(limit or self.cfg.fusion.limit)
        v_top_k = verify_top_k if verify_top_k is not None else rerank_top_k

        points = self._fetch(
            qvecs,
            final_limit=final_limit,
            prefetch_limit=prefetch_limit,
            weights=weights,
            extra_filter=extra_filter,
            person_only=person_only,
            need=max(rerank_top_k if rerank else 0, v_top_k if verify else 0),
        )
        hits = self._to_hits(points)

        if rerank:
            hits = self._rerank_by_text(
                query, hits, top_k=rerank_top_k, alpha=alpha, explain=explain
            )

        if verify:
            hits = self._verify_text(
                query, hits,
                top_k=v_top_k,
                threshold=verify_threshold,
                mode=verify_mode,
                explain=verify_explain,
                reranked=rerank and not explain,
            )

        return self._trim(hits, final_limit)

    # ------------------------------------------------------------------ #
    # 검색 — 이미지
    # ------------------------------------------------------------------ #
    def search_image(
        self,
        image,
        scope: str = "person",
        limit: Optional[int] = None,
        prefetch_limit: Optional[int] = None,
        names: Optional[Sequence[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        extra_filter=None,
        person_only: Union[bool, Dict[str, Optional[bool]], None] = None,
        rerank: bool = False,
        rerank_top_k: int = 20,
        alpha: float = 0.7,
        verify: bool = False,
        verify_top_k: Optional[int] = None,
        verify_threshold: float = 0.5,
        verify_mode: str = "flag",
        verify_explain: bool = False,
    ) -> List[SearchHit]:
        """
        이미지 검색 (같은 사람 / 같은 물건 찾기).

        scope='person' -> SigLIP2 + IRRA + SOLIDER
        scope='object' -> SigLIP2 + DINOv2
        """
        if scope not in ("person", "object"):
            raise ValueError("scope 는 'person' 또는 'object' 여야 합니다.")
        self._check_verify_args(verify_mode, verify_threshold)

        qvecs = self.router.embed_query_image(image, scope=scope)
        if names is not None:
            keep = set(names)
            missing = keep - set(qvecs)
            if missing:
                raise ValueError(
                    f"scope='{scope}' 에서 쓸 수 없는 retriever: {sorted(missing)}. "
                    f"사용 가능: {sorted(qvecs)}"
                )
            qvecs = {k: v for k, v in qvecs.items() if k in keep}
        logger.info("이미지 쿼리 벡터: %s", sorted(qvecs))

        final_limit = int(limit or self.cfg.fusion.limit)
        v_top_k = verify_top_k if verify_top_k is not None else rerank_top_k

        points = self._fetch(
            qvecs,
            final_limit=final_limit,
            prefetch_limit=prefetch_limit,
            weights=weights,
            extra_filter=extra_filter,
            person_only=person_only,
            need=max(rerank_top_k if rerank else 0, v_top_k if verify else 0),
        )
        hits = self._to_hits(points)

        if rerank:
            hits = self._rerank_by_image(
                image, hits, top_k=rerank_top_k, alpha=alpha
            )

        if verify:
            hits = self._verify_image(
                image, hits,
                top_k=v_top_k,
                threshold=verify_threshold,
                mode=verify_mode,
                explain=verify_explain,
                reranked=rerank,
            )

        return self._trim(hits, final_limit)

    # ------------------------------------------------------------------ #
    # 내부: 인자 검사 / 후처리
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_verify_args(mode: str, threshold: float) -> None:
        if mode not in VERIFY_MODES:
            raise ValueError(f"verify_mode 는 {list(VERIFY_MODES)} 중 하나여야 합니다.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("verify_threshold 는 0~1 이어야 합니다.")

    @staticmethod
    def _trim(hits: List[SearchHit], limit: int) -> List[SearchHit]:
        out = hits[:limit]
        for i, h in enumerate(out, 1):
            h.rank = i
        return out

    # ------------------------------------------------------------------ #
    # 내부: 후보 확보
    # ------------------------------------------------------------------ #
    def _fetch(
        self,
        qvecs: Dict[str, np.ndarray],
        final_limit: int,
        prefetch_limit: Optional[int],
        weights: Optional[Dict[str, float]],
        extra_filter,
        person_only,
        need: int,
    ):
        """
        재순위/검증을 할 거면 최종 limit 보다 많이 가져와야 한다.
        Qwen 이 볼 후보 수만큼은 확보해야 재순위가 의미 있다.
        """
        fetch = max(final_limit, need)

        pf = int(prefetch_limit or self.cfg.fusion.prefetch_limit)
        if pf < fetch:
            logger.info(
                "prefetch_limit(%d) < 필요 후보 수(%d) -> %d 로 올립니다.",
                pf, fetch, fetch,
            )
            pf = fetch

        if len(qvecs) == 1:
            # retriever 가 하나면 fusion 이 의미 없다. 단일 검색이 더 정확하다
            # (fusion 은 순위/점수 정규화를 한 번 더 거친다).
            name, vec = next(iter(qvecs.items()))
            logger.info("단일 검색: %s (limit=%d)", name, fetch)
            return self.store.search_single(
                name, vec,
                limit=fetch,
                person_only=person_only,
                extra_filter=extra_filter,
            )

        logger.info(
            "융합 검색: %s method=%s limit=%d prefetch=%d",
            sorted(qvecs), self.cfg.fusion.method, fetch, pf,
        )
        return self.store.fused_search(
            qvecs,
            limit=fetch,
            prefetch_limit=pf,
            weights=weights,
            extra_filter=extra_filter,
            person_only=person_only,
        )

    # ------------------------------------------------------------------ #
    # 내부: 재순위
    # ------------------------------------------------------------------ #
    def _split_usable(
        self,
        hits: List[SearchHit],
        top_k: int,
    ) -> Tuple[List[SearchHit], List[SearchHit], List[SearchHit]]:
        """crop 파일이 실제로 있는 후보만 골라낸다.

        파일이 없는 후보는 Qwen 이 볼 수 없다. 재순위/검증에서 빼고 원래
        순위를 유지한 채 뒤에 붙인다.
        """
        head, tail = hits[:top_k], hits[top_k:]

        usable, unusable = [], []
        for h in head:
            if h.crop_path and Path(h.crop_path).is_file():
                usable.append(h)
            else:
                unusable.append(h)

        if unusable:
            logger.warning(
                "crop 파일이 없어 Qwen 처리에서 제외: %d/%d건 "
                "(적재 시 keep_crop_path=True 였는지, 파일을 지우지 않았는지 확인)",
                len(unusable), len(head),
            )
        return usable, unusable, tail

    def _maybe_release_embedders(self) -> None:
        if self.release_embedders_before_rerank:
            logger.info("Qwen 처리 전 임베더 해제")
            self.registry.release()

    @staticmethod
    def _reorder(
        usable: List[SearchHit],
        results,
        unusable: List[SearchHit],
        tail: List[SearchHit],
    ) -> List[SearchHit]:
        """재순위 결과를 SearchHit 에 반영하고 순위를 다시 매긴다."""
        ordered: List[SearchHit] = []
        for r in results:
            h = usable[r.index]
            h.rerank_score = float(r.score)
            h.score = float(r.blended if r.blended is not None else r.score)
            if getattr(r, "reason", None):
                h.reason = r.reason
            ordered.append(h)

        final = ordered + unusable + tail
        for i, h in enumerate(final, 1):
            h.rank = i
        return final

    def _rerank_by_text(
        self,
        query: str,
        hits: List[SearchHit],
        top_k: int,
        alpha: float,
        explain: bool,
    ) -> List[SearchHit]:
        usable, unusable, tail = self._split_usable(hits, top_k)
        if not usable:
            logger.warning("재순위할 후보가 없습니다. 융합 결과를 그대로 씁니다.")
            return hits

        self._maybe_release_embedders()
        rr = self.reranker()
        results = rr.rerank_text(
            query,
            [h.crop_path for h in usable],
            retrieval_scores=[h.retrieval_score for h in usable],
            top_k=len(usable),
            method="generate" if explain else "logprob",
            alpha=alpha,
            refs=[h.point_id for h in usable],
        )
        return self._reorder(usable, results, unusable, tail)

    def _rerank_by_image(
        self,
        query_image,
        hits: List[SearchHit],
        top_k: int,
        alpha: float,
    ) -> List[SearchHit]:
        usable, unusable, tail = self._split_usable(hits, top_k)
        if not usable:
            logger.warning("재순위할 후보가 없습니다. 융합 결과를 그대로 씁니다.")
            return hits

        self._maybe_release_embedders()
        rr = self.reranker()
        results = rr.rerank_image(
            query_image,
            [h.crop_path for h in usable],
            retrieval_scores=[h.retrieval_score for h in usable],
            top_k=len(usable),
            alpha=alpha,
            refs=[h.point_id for h in usable],
        )
        return self._reorder(usable, results, unusable, tail)

    # ------------------------------------------------------------------ #
    # 내부: 자동 검증
    # ------------------------------------------------------------------ #
    def _apply_verdict(
        self,
        hits: List[SearchHit],
        threshold: float,
        mode: str,
    ) -> List[SearchHit]:
        """verify_score 를 threshold 와 비교해 판정하고, mode 에 따라 처리한다."""
        for h in hits:
            if h.verify_score is not None:
                h.verified = h.verify_score >= threshold

        n_fail = sum(1 for h in hits if h.verified is False)

        if mode == "filter" and n_fail:
            kept = [h for h in hits if h.verified is not False]
            logger.warning(
                "verify_mode='filter': threshold %.2f 미달 %d건을 결과에서 제거했습니다. "
                "threshold 가 캘리브레이션되지 않았다면 정답을 버릴 수 있습니다.",
                threshold, n_fail,
            )
            hits = kept
        elif n_fail:
            logger.info(
                "threshold %.2f 미달 %d건 (표시만, 제거하지 않음)",
                threshold, n_fail,
            )

        for i, h in enumerate(hits, 1):
            h.rank = i
        return hits

    def _verify_text(
        self,
        query: str,
        hits: List[SearchHit],
        top_k: int,
        threshold: float,
        mode: str,
        explain: bool,
        reranked: bool,
    ) -> List[SearchHit]:
        """
        텍스트 쿼리 검증.

        reranked=True 면 재순위에서 이미 같은 질문("이 이미지가 설명에
        부합하는가")의 P(yes) 를 계산했으므로 그 값을 재사용한다.
        Qwen 추가 호출이 0 이다.
        """
        target = hits[:top_k]

        if reranked and all(h.rerank_score is not None for h in target):
            logger.info(
                "검증: 재순위 점수를 재사용합니다 (동일한 질문, Qwen 추가 호출 0회)"
            )
            for h in target:
                h.verify_score = h.rerank_score
        else:
            usable, _unusable, _tail = self._split_usable(hits, top_k)
            if not usable:
                logger.warning("검증할 후보가 없습니다.")
                return hits

            self._maybe_release_embedders()
            rr = self.reranker()
            logger.info("검증: Qwen 호출 %d회", len(usable))
            for h in usable:
                ok, p, reason = rr.verify_description(
                    h.crop_path, query, threshold=threshold, explain=explain
                )
                h.verify_score = p
                if reason:
                    h.verify_reason = reason

        return self._apply_verdict(hits, threshold, mode)

    def _verify_image(
        self,
        query_image,
        hits: List[SearchHit],
        top_k: int,
        threshold: float,
        mode: str,
        explain: bool,
        reranked: bool,
    ) -> List[SearchHit]:
        """
        이미지 쿼리 검증.

        reranked=True 면 재순위에서 이미 "같은 사람인가" 의 P(yes) 를
        계산했으므로 재사용한다. explain=True 면 근거가 필요하므로
        상위 후보에 대해 실제로 호출한다.
        """
        target = hits[:top_k]

        if reranked and not explain and all(
            h.rerank_score is not None for h in target
        ):
            logger.info(
                "검증: 재순위 점수를 재사용합니다 (동일한 질문, Qwen 추가 호출 0회)"
            )
            for h in target:
                h.verify_score = h.rerank_score
        else:
            usable, _unusable, _tail = self._split_usable(hits, top_k)
            if not usable:
                logger.warning("검증할 후보가 없습니다.")
                return hits

            self._maybe_release_embedders()
            rr = self.reranker()
            logger.info("검증: Qwen 호출 %d회", len(usable))
            for h in usable:
                ok, p, reason = rr.verify_same_person(
                    query_image, h.crop_path,
                    threshold=threshold, explain=explain,
                )
                h.verify_score = p
                if reason:
                    h.verify_reason = reason

        return self._apply_verdict(hits, threshold, mode)

    # ------------------------------------------------------------------ #
    # 단건 검증 (사람이 특정 결과를 지목했을 때)
    # ------------------------------------------------------------------ #
    def verify_one(
        self,
        hit: SearchHit,
        query_image=None,
        query_text: Optional[str] = None,
        threshold: float = 0.5,
        explain: bool = True,
    ):
        """검색 결과 한 건을 단독으로 검증한다. 반환: (판정, P(yes), 근거)."""
        if not hit.crop_path or not Path(hit.crop_path).is_file():
            raise FileNotFoundError(
                f"crop 파일이 없어 검증할 수 없습니다: {hit.crop_path}"
            )
        if (query_image is None) == (query_text is None):
            raise ValueError("query_image 또는 query_text 중 하나만 주세요.")

        rr = self.reranker()
        if query_image is not None:
            return rr.verify_same_person(
                query_image, hit.crop_path, threshold=threshold, explain=explain
            )
        return rr.verify_description(
            hit.crop_path, query_text, threshold=threshold, explain=explain
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    import argparse
    import time

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    ap = argparse.ArgumentParser(
        description="Fusion 검색 + Qwen3-VL 재순위 + 자동 검증"
    )
    ap.add_argument("--config", default="pipeline.yaml")
    ap.add_argument("--text", default=None, help="자연어 쿼리 (영어)")
    ap.add_argument("--image", default=None, help="이미지 쿼리 경로")
    ap.add_argument("--scope", default="person", choices=["person", "object"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--names", nargs="*", default=None,
                    help="참여 retriever 제한 (예: --names irra)")

    ap.add_argument("--rerank", dest="rerank", action="store_true", default=False)
    ap.add_argument("--no-rerank", dest="rerank", action="store_false")
    ap.add_argument("--rerank-top-k", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="1.0=Qwen 점수만, 0.0=검색 점수만")
    ap.add_argument("--explain", action="store_true",
                    help="재순위 근거 생성 (느림)")

    ap.add_argument("--verify", action="store_true", help="자동 검증 켜기")
    ap.add_argument("--verify-top-k", type=int, default=None,
                    help="기본값은 --rerank-top-k 와 동일")
    ap.add_argument("--verify-threshold", type=float, default=0.5)
    ap.add_argument("--verify-mode", default="flag", choices=list(VERIFY_MODES),
                    help="flag=표시만(기본), filter=미달 제거")
    ap.add_argument("--verify-explain", action="store_true",
                    help="검증 근거 생성 (느림)")

    ap.add_argument("--qwen-model-id", default=None)
    ap.add_argument("--extra-paths", nargs="*", default=None)
    ap.add_argument("--crop-root", default=None,
                    help="crop 이미지 루트. 다른 PC 에서 적재한 DB 를 열 때 "
                         "payload 의 경로를 여기 기준으로 재해석한다.")
    ap.add_argument("--release-embedders", action="store_true",
                    help="Qwen 처리 전에 임베더를 GPU 에서 내린다")
    args = ap.parse_args()

    if not args.text and not args.image:
        ap.error("--text 또는 --image 가 필요합니다.")
    if args.text and args.image:
        ap.error("--text 와 --image 는 함께 쓸 수 없습니다.")

    engine = SearchEngine.from_config(
        args.config,
        extra_paths=args.extra_paths,
        qwen_model_id=args.qwen_model_id,
        release_embedders_before_rerank=args.release_embedders,
        crop_root=args.crop_root,
    )

    common = dict(
        limit=args.limit,
        names=args.names,
        rerank=args.rerank,
        rerank_top_k=args.rerank_top_k,
        alpha=args.alpha,
        verify=args.verify,
        verify_top_k=args.verify_top_k,
        verify_threshold=args.verify_threshold,
        verify_mode=args.verify_mode,
        verify_explain=args.verify_explain,
    )

    t0 = time.time()
    if args.text:
        hits = engine.search_text(args.text, explain=args.explain, **common)
        header = f'텍스트 쿼리: "{args.text}"'
    else:
        hits = engine.search_image(args.image, scope=args.scope, **common)
        header = f"이미지 쿼리: {args.image} (scope={args.scope})"
    elapsed = time.time() - t0

    print()
    print(header)
    line = f"재순위: {'ON' if args.rerank else 'OFF'}"
    if args.rerank:
        line += f" (top_k={args.rerank_top_k}, alpha={args.alpha})"
    line += f" / 검증: {'ON' if args.verify else 'OFF'}"
    if args.verify:
        line += f" (threshold={args.verify_threshold}, mode={args.verify_mode})"
    print(line)
    print(f"소요: {elapsed:.2f}s / 결과 {len(hits)}건")
    print("-" * 78)

    for h in hits:
        print(h.summary())
        if h.reason:
            print(f"      재순위: {h.reason}")
        if h.verify_reason:
            print(f"      검증  : {h.verify_reason}")

    if args.verify:
        n_ok = sum(1 for h in hits if h.verified is True)
        n_no = sum(1 for h in hits if h.verified is False)
        print()
        print(f"검증 결과: 확인 {n_ok}건 / 미달 {n_no}건 "
              f"/ 미검증 {len(hits) - n_ok - n_no}건")
        print("threshold 는 라벨된 쌍으로 캘리브레이션해야 의미가 있습니다. "
              "현재 값은 임의값입니다.")

    if args.rerank:
        print()
        print("--no-rerank 결과와 비교해 보세요. "
              "재순위가 항상 개선하는 것은 아닙니다.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())