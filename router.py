"""
RF-DETR detection -> scope 별 임베딩 라우팅.

라우터가 아는 것은 두 가지뿐이다.
  1) detection 이 사람인가 객체인가
  2) 각 retriever 의 scope 가 무엇인가

어떤 모델인지, 차원이 얼마인지, 어떻게 전처리하는지는 전혀 모른다.
그래서 임베더를 바꿔도 이 파일은 그대로다.

핵심 최적화: crop 을 하나씩 돌리지 않고 **scope 별로 모아서 배치로** 넘긴다.
GPU 활용률이 여기서 갈린다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from config import PipelineConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    """RF-DETR 출력 한 건 + 잘라낸 이미지."""

    crop: Any                       # np.ndarray | PIL.Image | 경로
    label: str                      # 'person', 'backpack', 'car', ...
    score: float = 1.0
    bbox: tuple = (0, 0, 0, 0)      # (x1, y1, x2, y2)
    image_id: str = ""
    frame_idx: int = 0
    track_id: Optional[int] = None  # 트래커를 붙였다면 tracklet 집계에 사용
    extra: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
class Router:
    def __init__(
        self,
        cfg: PipelineConfig,
        registry,
        input_format: str = "rgb",
    ):
        self.cfg = cfg
        self.registry = registry
        self.input_format = input_format

    # ---------------- 분류 ---------------- #

    def is_person(self, det: Detection) -> bool:
        extra = getattr(det, "extra", None) or {}
        scope = str(extra.get("candidate_scope", "")).strip().lower()

        if scope == "person":
            return True

        if scope == "object":
            return False

        return det.label.lower() in self.cfg.person_labels

    def _targets(self, det: Detection) -> List[str]:
        """이 detection 에 적용할 retriever 이름들."""
        person = self.is_person(det)

        return [
            s.name
            for s in self.cfg.retrievers.values()
            if (
                s.accepts_person()
                if person
                else s.accepts_object()
            )
        ]

    # ---------------- DB 구축용 임베딩 ---------------- #

    def embed(
        self,
        detections: Sequence[Detection],
    ) -> List[Dict[str, np.ndarray]]:
        """
        detection 순서 그대로 {벡터이름: 벡터} 리스트를 반환.

        DB 구축용 함수다.

        person:
            SigLIP2 + IRRA + SOLIDER

        object:
            SigLIP2 + DINOv2

        사람 crop 에 dinov2 키가 없고 객체 crop 에 irra/solider 키가 없는 것이
        정상이다.

        이 함수의 routing 계약은 검색 최적화와 무관하며 변경하지 않는다.
        """
        results: List[Dict[str, np.ndarray]] = [
            dict()
            for _ in detections
        ]

        if not detections:
            return results

        # retriever 별 담당 crop index를 모아 batch 처리한다.
        buckets: Dict[str, List[int]] = {
            name: []
            for name in self.cfg.retrievers
        }

        for i, det in enumerate(detections):
            for name in self._targets(det):
                buckets[name].append(i)

        for name, idxs in buckets.items():
            if not idxs:
                continue

            embedder = self.registry.get(name)

            crops = [
                detections[i].crop
                for i in idxs
            ]

            vecs = embedder.embed_crops(
                crops,
                input_format=self.input_format,
            )

            vecs = np.asarray(
                vecs,
                dtype=np.float32,
            )

            if vecs.ndim != 2:
                raise RuntimeError(
                    f"[{name}] embedding 출력은 2-D (N, DIM)이어야 합니다. "
                    f"현재 shape={vecs.shape}"
                )

            if vecs.shape[0] != len(idxs):
                raise RuntimeError(
                    f"[{name}] 입력 {len(idxs)}개 -> 출력 {vecs.shape[0]}개. "
                    f"임베더가 일부 crop 을 버리고 있습니다."
                )

            if not np.isfinite(vecs).all():
                raise RuntimeError(
                    f"[{name}] embedding에 NaN/Inf가 있습니다."
                )

            for i, v in zip(idxs, vecs):
                results[i][name] = v

            logger.info(
                "[%s] %d crops -> %s",
                name,
                len(idxs),
                vecs.shape,
            )

        return results

    # ---------------- 이미지 질의 벡터 ---------------- #

    def embed_query_image(
        self,
        image,
        scope: str = "person",
        names: Optional[Sequence[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        이미지 질의 -> {벡터이름: 벡터}.

        scope:
            person
            object

        names=None:
            해당 scope의 모든 retriever 실행.

            person
                -> SigLIP2 + IRRA + SOLIDER

            object
                -> SigLIP2 + DINOv2

        names 지정:
            해당 scope 안에서 지정된 retriever만 실행한다.

        예:
            names=["siglip2", "irra"]
                -> person 1차 검색
                -> SOLIDER는 로드하지 않음

            names=["solider"]
                -> person 2차 SOLIDER 재정렬 시에만 실행

        이 선택 기능은 query-time 최적화일 뿐,
        DB 구축용 embed()의 vector 계약에는 영향을 주지 않는다.
        """

        if scope not in {"person", "object"}:
            raise ValueError(
                f"scope는 'person' 또는 'object'여야 합니다: {scope!r}"
            )

        specs = (
            self.cfg.for_person()
            if scope == "person"
            else self.cfg.for_object()
        )

        # ---------------------------------------------------------------
        # 현재 scope에서 사용 가능한 retriever
        # ---------------------------------------------------------------

        available = {
            s.name: s
            for s in specs
        }

        # ---------------------------------------------------------------
        # names를 지정하지 않으면 기존 동작 유지:
        # 해당 scope의 모든 retriever 실행
        # ---------------------------------------------------------------

        if names is None:
            selected_specs = list(specs)

        else:
            selected_names = list(
                dict.fromkeys(names)
            )

            if not selected_names:
                raise ValueError(
                    "names를 지정했다면 retriever 이름을 "
                    "하나 이상 넣어야 합니다."
                )

            unknown = [
                name
                for name in selected_names
                if name not in available
            ]

            if unknown:
                raise ValueError(
                    f"scope='{scope}'에서 사용할 수 없는 retriever: "
                    f"{unknown}. "
                    f"사용 가능: {sorted(available)}"
                )

            selected_specs = [
                available[name]
                for name in selected_names
            ]

        # ---------------------------------------------------------------
        # 선택된 retriever만 실제 로드/추론
        # ---------------------------------------------------------------

        out: Dict[str, np.ndarray] = {}

        for spec in selected_specs:
            name = spec.name

            # registry.get()을 여기서 호출하므로,
            # 선택되지 않은 모델은 인스턴스화되지 않는다.
            embedder = self.registry.get(name)

            vecs = embedder.embed_crops(
                [image],
                input_format=self.input_format,
            )

            vecs = np.asarray(
                vecs,
                dtype=np.float32,
            )

            if vecs.ndim != 2:
                raise RuntimeError(
                    f"[{name}] query embedding 출력은 "
                    f"2-D (N, DIM)이어야 합니다. "
                    f"현재 shape={vecs.shape}"
                )

            if vecs.shape[0] != 1:
                raise RuntimeError(
                    f"[{name}] query 이미지 1개 -> "
                    f"embedding {vecs.shape[0]}개"
                )

            vector = vecs[0]

            if not np.isfinite(vector).all():
                raise RuntimeError(
                    f"[{name}] query embedding에 NaN/Inf가 있습니다."
                )

            out[name] = vector

            logger.info(
                "[query/%s] %s -> %s",
                scope,
                name,
                vector.shape,
            )

        if not out:
            raise RuntimeError(
                f"scope='{scope}'에서 생성된 query vector가 없습니다."
            )

        return out

    # ---------------- 텍스트 질의 벡터 ---------------- #

    def embed_query_text(
        self,
        text: str,
        names: Optional[Sequence[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        자연어 질의 -> {벡터이름: 벡터}.

        supports_text=true 인 retriever만 로드한다.

        따라서:
            SigLIP2 -> 사용
            IRRA    -> 사용
            SOLIDER -> 로드 안 함
            DINOv2  -> 로드 안 함

        supports_text=false 모델은 registry.get()조차 호출하지 않는다.
        """

        out: Dict[str, np.ndarray] = {}

        selected_names = (
            list(names)
            if names is not None
            else list(self.cfg.retrievers.keys())
        )

        for name in selected_names:
            spec = self.cfg.retrievers.get(name)

            if spec is None:
                raise KeyError(
                    f"설정에 없는 retriever: '{name}' "
                    f"(사용 가능: {sorted(self.cfg.retrievers)})"
                )

            # -----------------------------------------------------------
            # 텍스트 미지원 모델은 인스턴스화 전에 제외
            # -----------------------------------------------------------

            if not spec.supports_text:
                logger.debug(
                    "[%s] 텍스트 인코딩 미지원 — 로드하지 않음",
                    name,
                )
                continue

            embedder = self.registry.get(name)

            fn = getattr(
                embedder,
                "embed_text",
                None,
            )

            # supports_text=true인데 구현이 없으면 설정/구현 계약 오류다.
            if fn is None:
                raise RuntimeError(
                    f"retriever '{name}' 은 supports_text=true 이지만 "
                    f"embed_text()가 구현되어 있지 않습니다."
                )

            v = np.asarray(
                fn(text),
                dtype=np.float32,
            )

            vector = (
                v[0]
                if v.ndim == 2
                else v
            )

            vector = np.asarray(
                vector,
                dtype=np.float32,
            ).reshape(-1)

            if not np.isfinite(vector).all():
                raise RuntimeError(
                    f"[{name}] text embedding에 NaN/Inf가 있습니다."
                )

            out[name] = vector

        if not out:
            raise RuntimeError(
                "텍스트 질의를 처리할 수 있는 retriever가 없습니다."
            )

        return out


# --------------------------------------------------------------------------- #
# tracklet 집계 — 규모를 줄이는 가장 큰 레버
# --------------------------------------------------------------------------- #
def aggregate_by_track(
    detections: Sequence[Detection],
    vectors: Sequence[Dict[str, np.ndarray]],
    min_len: int = 1,
):
    """
    같은 track_id 의 임베딩을 평균내어 tracklet 단위로 축약한다.

    30fps 영상에서 사람이 5초 지나가면 거의 동일한 crop 이 150장 생긴다.

    이를 하나로 묶으면:
      - point 수가 10~100배 줄어 저장/검색 비용이 그만큼 감소
      - 흐릿하거나 가려진 프레임의 노이즈가 평균으로 상쇄되어 품질이 오히려 개선

    track_id 가 없는 detection 은 개별 point 로 그대로 남는다.

    반환:
        (대표 Detection 리스트, 평균 벡터 리스트)
    """

    groups: Dict[Any, List[int]] = {}
    singles: List[int] = []

    for i, det in enumerate(detections):
        if det.track_id is None:
            singles.append(i)

        else:
            groups.setdefault(
                (
                    det.image_id,
                    det.track_id,
                ),
                [],
            ).append(i)

    out_dets: List[Detection] = []
    out_vecs: List[Dict[str, np.ndarray]] = []

    for key, idxs in groups.items():
        if len(idxs) < min_len:
            continue

        # 대표는 검출 점수가 가장 높은 프레임
        # 보통 가장 선명한 detection이다.
        best = max(
            idxs,
            key=lambda i: detections[i].score,
        )

        rep = detections[best]

        rep.extra = {
            **rep.extra,
            "track_size": len(idxs),
            "frame_range": [
                min(
                    detections[i].frame_idx
                    for i in idxs
                ),
                max(
                    detections[i].frame_idx
                    for i in idxs
                ),
            ],
        }

        merged: Dict[str, np.ndarray] = {}

        names = {
            name
            for i in idxs
            for name in vectors[i]
        }

        for name in names:
            stack = np.stack([
                vectors[i][name]
                for i in idxs
                if name in vectors[i]
            ]).astype(
                np.float32,
                copy=False,
            )

            if not np.isfinite(stack).all():
                raise RuntimeError(
                    f"[{name}] tracklet embedding에 NaN/Inf가 있습니다."
                )

            mean = stack.mean(
                axis=0
            )

            norm = float(
                np.linalg.norm(mean)
            )

            merged[name] = (
                mean / norm
                if norm > 1e-12
                else mean
            ).astype(
                np.float32,
                copy=False,
            )

        out_dets.append(rep)
        out_vecs.append(merged)

    # track_id 없는 detection은 그대로 유지
    for i in singles:
        out_dets.append(
            detections[i]
        )

        out_vecs.append(
            vectors[i]
        )

    logger.info(
        "tracklet 집계: %d detections -> %d points",
        len(detections),
        len(out_dets),
    )

    return out_dets, out_vecs