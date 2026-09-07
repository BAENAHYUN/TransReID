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
    def __init__(self, cfg: PipelineConfig, registry, input_format: str = "rgb"):
        self.cfg = cfg
        self.registry = registry
        self.input_format = input_format

    # ---------------- 분류 ---------------- #

    def is_person(self, det: Detection) -> bool:
        return det.label.lower() in self.cfg.person_labels

    def _targets(self, det: Detection) -> List[str]:
        """이 detection 에 적용할 retriever 이름들."""
        person = self.is_person(det)
        return [
            s.name for s in self.cfg.retrievers.values()
            if (s.accepts_person() if person else s.accepts_object())
        ]

    # ---------------- 임베딩 ---------------- #

    def embed(self, detections: Sequence[Detection]) -> List[Dict[str, np.ndarray]]:
        """detection 순서 그대로 {벡터이름: 벡터} 리스트를 반환.

        사람 crop 에 dinov2 키가 없고 객체 crop 에 irra 키가 없는 것이 정상이다.
        Qdrant named vector 는 point 마다 일부만 있어도 된다.
        """
        results: List[Dict[str, np.ndarray]] = [dict() for _ in detections]
        if not detections:
            return results

        # retriever 별로 담당 crop 인덱스를 모은다 (배치화)
        buckets: Dict[str, List[int]] = {name: [] for name in self.cfg.retrievers}
        for i, det in enumerate(detections):
            for name in self._targets(det):
                buckets[name].append(i)

        for name, idxs in buckets.items():
            if not idxs:
                continue
            embedder = self.registry.get(name)
            crops = [detections[i].crop for i in idxs]

            vecs = embedder.embed_crops(crops, input_format=self.input_format)
            vecs = np.asarray(vecs, dtype=np.float32)

            if vecs.shape[0] != len(idxs):
                raise RuntimeError(
                    f"[{name}] 입력 {len(idxs)}개 -> 출력 {vecs.shape[0]}개. "
                    f"임베더가 일부 crop 을 버리고 있습니다."
                )
            for i, v in zip(idxs, vecs):
                results[i][name] = v

            logger.info("[%s] %d crops -> %s", name, len(idxs), vecs.shape)

        return results

    # ---------------- 질의 벡터 ---------------- #

    def embed_query_image(self, image, scope: str = "person") -> Dict[str, np.ndarray]:
        """이미지 질의 -> {벡터이름: 벡터}. 해당 scope 의 retriever 만 돈다."""
        specs = self.cfg.for_person() if scope == "person" else self.cfg.for_object()
        out = {}
        for s in specs:
            v = self.registry.get(s.name).embed_crops([image], input_format=self.input_format)
            out[s.name] = np.asarray(v[0], dtype=np.float32)
        return out

    def embed_query_text(self, text: str, names: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        """자연어 질의 -> {벡터이름: 벡터}.

        `embed_text` 를 가진 retriever 만 참여한다 (IRRA, SigLIP2).
        SOLIDER/DINOv2 는 텍스트 정렬이 없으므로 자동으로 빠진다.
        """
        out: Dict[str, np.ndarray] = {}
        for name in (names or self.cfg.retrievers.keys()):
            embedder = self.registry.get(name)
            fn = getattr(embedder, "embed_text", None)
            if fn is None:
                logger.debug("[%s] 텍스트 인코딩 미지원 — 건너뜀", name)
                continue
            v = np.asarray(fn(text), dtype=np.float32)
            out[name] = v[0] if v.ndim == 2 else v
        if not out:
            raise RuntimeError(
                "텍스트 질의를 처리할 수 있는 retriever 가 없습니다. "
                "IRRA 또는 SigLIP2 임베더에 embed_text 를 구현하세요."
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
    """같은 track_id 의 임베딩을 평균내어 tracklet 단위로 축약한다.

    30fps 영상에서 사람이 5초 지나가면 거의 동일한 crop 이 150장 생긴다.
    이를 하나로 묶으면:
      - point 수가 10~100배 줄어 저장/검색 비용이 그만큼 감소
      - 흐릿하거나 가려진 프레임의 노이즈가 평균으로 상쇄되어 품질이 오히려 개선

    track_id 가 없는 detection 은 개별 point 로 그대로 남는다.
    반환: (대표 Detection 리스트, 평균 벡터 리스트)
    """
    groups: Dict[Any, List[int]] = {}
    singles: List[int] = []
    for i, det in enumerate(detections):
        if det.track_id is None:
            singles.append(i)
        else:
            groups.setdefault((det.image_id, det.track_id), []).append(i)

    out_dets: List[Detection] = []
    out_vecs: List[Dict[str, np.ndarray]] = []

    for key, idxs in groups.items():
        if len(idxs) < min_len:
            continue
        # 대표는 검출 점수가 가장 높은 프레임 (보통 가장 선명함)
        best = max(idxs, key=lambda i: detections[i].score)
        rep = detections[best]
        rep.extra = {**rep.extra, "track_size": len(idxs),
                     "frame_range": [min(detections[i].frame_idx for i in idxs),
                                     max(detections[i].frame_idx for i in idxs)]}

        merged: Dict[str, np.ndarray] = {}
        names = {n for i in idxs for n in vectors[i]}
        for n in names:
            stack = np.stack([vectors[i][n] for i in idxs if n in vectors[i]])
            mean = stack.mean(axis=0)
            norm = np.linalg.norm(mean)
            merged[n] = (mean / norm if norm > 1e-12 else mean).astype(np.float32)

        out_dets.append(rep)
        out_vecs.append(merged)

    for i in singles:
        out_dets.append(detections[i])
        out_vecs.append(vectors[i])

    logger.info("tracklet 집계: %d detections -> %d points", len(detections), len(out_dets))
    return out_dets, out_vecs