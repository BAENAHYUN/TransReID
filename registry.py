"""
설정에 적힌 module/class 를 실제 객체로 만드는 지연 로딩 레지스트리.

임베더 인스턴스화는 비싸다 (체크포인트 수백 MB + GPU 메모리). 그래서
실제로 쓰이는 시점에 한 번만 만들고 캐시한다. 예를 들어 검색 스크립트에서
person 질의만 처리한다면 DINOv2 는 끝까지 로드되지 않는다.

설정의 module 경로가 파이썬이 찾을 수 있는 곳에 있어야 하므로,
프로젝트 루트를 sys.path 에 넣어두거나 extra_paths 로 넘긴다.
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from config import PipelineConfig, RetrieverSpec

logger = logging.getLogger(__name__)


class EmbedderRegistry:
    """이름 -> 임베더 인스턴스. 필요할 때 만들고 재사용한다."""

    def __init__(self, cfg: PipelineConfig, extra_paths: Optional[Sequence[str]] = None):
        self.cfg = cfg
        self._cache: Dict[str, object] = {}
        for p in (extra_paths or []):
            p = str(Path(p).resolve())
            if p not in sys.path:
                sys.path.insert(0, p)

    # ---------------- 생성 ---------------- #

    def get(self, name: str):
        """이름으로 임베더를 얻는다. 처음 호출 시에만 실제 로드."""
        if name in self._cache:
            return self._cache[name]

        spec = self.cfg.retrievers.get(name)
        if spec is None:
            raise KeyError(
                f"설정에 없는 retriever: '{name}' "
                f"(사용 가능: {sorted(self.cfg.retrievers)})"
            )

        obj = self._instantiate(spec)
        self._verify_dim(spec, obj)
        self._cache[name] = obj
        return obj

    def _instantiate(self, spec: RetrieverSpec):
        logger.info("임베더 로드: %s (%s.%s)", spec.name, spec.module, spec.class_name)
        try:
            module = importlib.import_module(spec.module)
        except ImportError as e:
            raise ImportError(
                f"retriever '{spec.name}' 의 모듈 '{spec.module}' 을 import 할 수 없습니다.\n"
                f"  프로젝트 루트가 sys.path 에 있는지, 파일이 존재하는지 확인하세요.\n"
                f"  원인: {e}"
            ) from e

        if not hasattr(module, spec.class_name):
            raise AttributeError(
                f"'{spec.module}' 에 '{spec.class_name}' 클래스가 없습니다."
            )

        cls = getattr(module, spec.class_name)
        try:
            return cls(**spec.params)
        except TypeError as e:
            raise TypeError(
                f"retriever '{spec.name}' 생성 실패. pipeline.yaml 의 params 가 "
                f"{spec.class_name}.__init__ 시그니처와 맞는지 확인하세요.\n"
                f"  params: {spec.params}\n  원인: {e}"
            ) from e

    @staticmethod
    def _verify_dim(spec: RetrieverSpec, obj) -> None:
        """설정의 dim 과 실제 모델 차원이 어긋나면 즉시 실패시킨다.

        이게 어긋난 채로 진행되면 Qdrant upsert 단계에서야 터지는데,
        그때는 이미 수십만 장을 임베딩한 뒤다.
        """
        actual = getattr(obj, "DIM", None)
        if actual is None:
            logger.warning("retriever '%s': DIM 속성이 없어 차원 검증을 건너뜁니다.", spec.name)
            return
        if int(actual) != spec.dim:
            raise ValueError(
                f"retriever '{spec.name}': 차원 불일치. "
                f"pipeline.yaml={spec.dim} vs 모델={actual}. 설정을 고치세요."
            )

    # ---------------- 조회 ---------------- #

    def get_many(self, names: Iterable[str]) -> Dict[str, object]:
        return {n: self.get(n) for n in names}

    def load_all(self) -> Dict[str, object]:
        """DB 구축처럼 전부 필요한 경우. 실패를 앞에서 모아 보고 싶을 때도 유용."""
        return self.get_many(self.cfg.retrievers.keys())

    def loaded(self) -> List[str]:
        return sorted(self._cache)

    def release(self, name: Optional[str] = None) -> None:
        """GPU 메모리 회수. name 을 주면 하나만, 없으면 전부."""
        targets = [name] if name else list(self._cache)
        for t in targets:
            self._cache.pop(t, None)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass