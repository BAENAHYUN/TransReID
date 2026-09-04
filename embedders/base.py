"""
모든 임베더가 공유하는 계약과 공통 로직.

계약은 하나뿐이다:

    DIM: int
    embed_crops(crops, input_format="rgb") -> np.ndarray   # (N, DIM) float32

라우터도 Qdrant 계층도 이 이상은 모른다. 그래서 임베더를 추가/교체해도
파이프라인 코드는 그대로다.

새 임베더를 만들 때는 BaseEmbedder 를 상속하고 `_encode(pil_images)` 하나만
구현하면 된다. 빈 입력 처리, 배치 분할, 입력 형식 변환, L2 정규화, 차원 검증은
전부 여기서 처리한다 — 임베더 4개가 같은 실수를 4번 하지 않도록.

색 순서에 관하여
---------------
`input_format` 을 지정하지 않으면 "rgb" 로 간주한다. detector 가 OpenCV 라면
crop 은 BGR 이므로 R/B 가 뒤바뀐 채 임베딩된다 — **에러는 나지 않고 유사도만
조용히 망가진다.** numpy 배열을 형식 지정 없이 넘기면 경고를 띄운다.
파이프라인 전체가 BGR 이면 `DEFAULT_INPUT_FORMAT` 을 "bgr" 로 바꿔라.

투명도에 관하여
--------------
RGBA 입력은 `ALPHA_BACKGROUND` 위에 **합성**한다. PIL 의 `.convert("RGB")` 나
numpy 의 `arr[..., :3]` 은 알파를 합성하지 않고 버리기 때문에, 투명 픽셀에
저장돼 있던 정의되지 않은 RGB 값이 그대로 남는다 (툴에 따라 검정/흰색/잔상).
누끼 이미지에서는 이 배경이 임베딩을 지배할 수 있다.

누끼 이미지를 검색 대상으로 넣는다면 `alpha_bbox_crop` 으로 타이트하게 자른 뒤
넣어라. 안 그러면 유사도가 물체 외형이 아니라 '실루엣 면적 비율'로 결정되기 쉽다.
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, Image.Image, np.ndarray]

# 파이프라인 전체의 기본 색 순서. OpenCV 기반이면 "bgr" 로 바꾼다.
DEFAULT_INPUT_FORMAT = "rgb"

# 투명 배경을 합성할 색. 검정은 어두운 물체와, 흰색은 밝은 물체와 섞여 실루엣이
# 흐려진다. 중간 회색이 양쪽 모두와 대비가 있어 무난하다.
ALPHA_BACKGROUND: Tuple[int, int, int] = (128, 128, 128)

# numpy 입력에 형식이 지정되지 않았을 때 경고를 몇 번까지 띄울지
_FORMAT_WARN_LIMIT = 3
_format_warn_count = 0


# --------------------------------------------------------------------------- #
# 계약
# --------------------------------------------------------------------------- #
@runtime_checkable
class Embedder(Protocol):
    """라우터가 요구하는 최소 인터페이스."""

    DIM: int

    def embed_crops(self, crops: Sequence[ImageInput],
                    input_format: Optional[str] = None) -> np.ndarray:
        """(N, DIM) float32 를 반환. crops 가 비면 (0, DIM)."""
        ...


# --------------------------------------------------------------------------- #
# 공통 구현
# --------------------------------------------------------------------------- #
class BaseEmbedder(ABC):
    """전처리 · 배치 분할 · 정규화 · 검증을 담당하는 공통 뼈대.

    하위 클래스가 할 일:
        DIM 클래스 변수 지정
        _encode(pil_images) 구현 — (len(pil_images), DIM) 반환
    """

    DIM: int = -1

    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 32,
        l2_normalize: bool = True,
    ) -> None:
        if self.DIM <= 0:
            raise ValueError(f"{type(self).__name__}: DIM 클래스 변수를 지정하세요.")
        self.batch_size = batch_size
        self.l2_normalize = l2_normalize
        self.device = device or self._default_device()

    @staticmethod
    def _default_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # ---------------- 하위 클래스 구현 지점 ---------------- #

    @abstractmethod
    def _encode(self, images: List[Image.Image]) -> np.ndarray:
        """RGB PIL 이미지 리스트 -> (len(images), DIM) 배열.

        정규화는 하지 않아도 된다 (여기서 처리). 배치 크기는 이미 잘려서 들어온다.
        """
        raise NotImplementedError

    # ---------------- 공개 API ---------------- #

    def embed_crops(
        self,
        crops: Sequence[ImageInput],
        input_format: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """crop 리스트 -> (N, DIM) float32.

        input_format: numpy 입력이 OpenCV crop 이면 "bgr". 생략하면
                      DEFAULT_INPUT_FORMAT 이 쓰이고, numpy 입력일 때는 경고한다.
        """
        if len(crops) == 0:
            # 사람이 없는 프레임은 흔하다. 예외 대신 빈 배열로 조용히 넘어간다.
            return np.zeros((0, self.DIM), dtype=np.float32)

        fmt = resolve_input_format(crops, input_format, owner=type(self).__name__)
        bs = batch_size or self.batch_size
        outs: List[np.ndarray] = []

        for i in range(0, len(crops), bs):
            chunk = crops[i:i + bs]
            images = [to_pil(c, fmt) for c in chunk]
            vecs = np.asarray(self._encode(images), dtype=np.float32)

            if vecs.ndim != 2 or vecs.shape[0] != len(images):
                raise RuntimeError(
                    f"{type(self).__name__}._encode 반환 형태 이상: "
                    f"{vecs.shape} (기대: ({len(images)}, {self.DIM}))"
                )
            if vecs.shape[1] != self.DIM:
                raise RuntimeError(
                    f"{type(self).__name__}: 실제 차원 {vecs.shape[1]} != 선언된 DIM {self.DIM}. "
                    f"pipeline.yaml 의 dim 도 함께 확인하세요."
                )
            if not np.isfinite(vecs).all():
                # fp16 오버플로/NaN 이 Qdrant 까지 흘러가면 디버깅이 훨씬 어렵다
                raise RuntimeError(
                    f"{type(self).__name__}: 임베딩에 NaN/Inf 가 있습니다. "
                    f"fp16 을 쓰고 있다면 bf16 또는 fp32 로 바꿔 보세요."
                )
            outs.append(vecs)

        result = np.concatenate(outs, axis=0)
        return l2_normalize(result, owner=type(self).__name__) if self.l2_normalize else result

    # 편의 별칭 — 단일 이미지
    def embed_one(self, crop: ImageInput, input_format: Optional[str] = None) -> np.ndarray:
        return self.embed_crops([crop], input_format=input_format)[0]


# --------------------------------------------------------------------------- #
# 어댑터 — 계약을 안 따르는 기존 클래스를 감싸기
# --------------------------------------------------------------------------- #
class EmbedderAdapter:
    """`encode_batch` 등 다른 이름의 메서드를 가진 임베더를 계약에 맞춘다.

    기존 코드를 수정하지 않고 파이프라인에 끼울 때 사용.

        adapted = EmbedderAdapter(my_embedder, dim=512, method="encode_batch")

    BaseEmbedder 경로와 동일하게 차원 검증과 L2 정규화를 수행한다
    (안 그러면 Qdrant 에 정규화된 벡터와 안 된 벡터가 섞인다).
    """

    def __init__(self, inner: Any, dim: int, method: str = "encode_batch",
                 l2_normalize: bool = True):
        if not hasattr(inner, method):
            raise AttributeError(f"{type(inner).__name__} 에 '{method}' 메서드가 없습니다.")
        self.inner = inner
        self.DIM = dim
        self._method = method
        self.l2_normalize = l2_normalize

        # 호출 시점에 TypeError 를 잡아 재시도하면 inner 내부에서 난 TypeError 까지
        # 삼켜서 input_format 없이 다시 호출하게 된다 (색이 뒤집힌 결과가 조용히 나옴).
        # 시그니처를 미리 보고 결정한다.
        fn = getattr(inner, method)
        try:
            params = inspect.signature(fn).parameters
            self._accepts_format = (
                "input_format" in params
                or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
            )
        except (TypeError, ValueError):      # C 확장 등 시그니처를 못 읽는 경우
            self._accepts_format = False
            logger.warning("%s.%s 의 시그니처를 읽을 수 없어 input_format 을 전달하지 "
                           "않습니다. 색 순서를 직접 확인하세요.",
                           type(inner).__name__, method)

    def embed_crops(self, crops: Sequence[ImageInput],
                    input_format: Optional[str] = None) -> np.ndarray:
        if len(crops) == 0:
            return np.zeros((0, self.DIM), dtype=np.float32)

        fmt = resolve_input_format(crops, input_format, owner=type(self.inner).__name__)
        fn = getattr(self.inner, self._method)
        out = fn(crops, input_format=fmt) if self._accepts_format else fn(crops)

        vecs = np.asarray(out, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape != (len(crops), self.DIM):
            raise RuntimeError(
                f"{type(self.inner).__name__}.{self._method} 반환 형태 이상: "
                f"{vecs.shape} (기대: ({len(crops)}, {self.DIM}))"
            )
        return l2_normalize(vecs, owner=type(self.inner).__name__) if self.l2_normalize else vecs

    def __getattr__(self, item):
        # embed_text 같은 임베더 고유 메서드는 그대로 통과시킨다.
        # 주의: 라우터가 hasattr(emb, "embed_text") 로 텍스트 검색 가능 여부를
        # 판단하므로, inner 에 없으면 여기서 AttributeError 가 나야 정상 동작한다.
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(self.inner, item)


# --------------------------------------------------------------------------- #
# 공통 유틸 (임베더들이 각자 구현하지 않도록)
# --------------------------------------------------------------------------- #
def resolve_input_format(crops: Sequence[ImageInput],
                         input_format: Optional[str],
                         owner: str = "") -> str:
    """형식이 명시되지 않았을 때의 기본값 결정 + 조용한 BGR 사고 방지 경고."""
    global _format_warn_count

    if input_format is not None:
        fmt = input_format.lower()
        if fmt not in ("rgb", "bgr"):
            raise ValueError("input_format 은 'rgb' 또는 'bgr' 이어야 합니다.")
        return fmt

    # 경로/PIL 입력은 색 순서가 애매하지 않다. numpy 만 위험하다.
    if any(isinstance(c, np.ndarray) for c in crops) and _format_warn_count < _FORMAT_WARN_LIMIT:
        _format_warn_count += 1
        logger.warning(
            "%s: numpy crop 을 input_format 없이 받아 '%s' 로 간주합니다. "
            "OpenCV crop 이면 R/B 가 뒤바뀐 채 임베딩되며 에러 없이 유사도만 "
            "망가집니다. 호출부에서 input_format 을 명시하세요.",
            owner or "embed_crops", DEFAULT_INPUT_FORMAT,
        )
    return DEFAULT_INPUT_FORMAT


def _has_alpha(im: Image.Image) -> bool:
    return im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)


def flatten_alpha(im: Image.Image,
                  background: Tuple[int, int, int] = ALPHA_BACKGROUND) -> Image.Image:
    """투명 배경을 background 위에 합성한 뒤 RGB 로 변환.

    `.convert("RGB")` 는 알파를 합성하지 않고 버린다 — 투명 픽셀의 RGB 값은
    정의되어 있지 않으므로 결과가 파일 생성 툴에 따라 달라진다.
    """
    if im.mode == "RGB":
        return im
    if not _has_alpha(im):
        return im.convert("RGB")
    rgba = im.convert("RGBA")
    canvas = Image.new("RGB", rgba.size, background)
    canvas.paste(rgba, mask=rgba.split()[-1])
    return canvas


def alpha_bbox_crop(im: Image.Image, margin: int = 0) -> Image.Image:
    """알파의 유효 영역으로 타이트하게 자른다 (알파가 없으면 원본 반환).

    누끼 이미지는 캔버스 대부분이 배경이라, 그대로 임베딩하면 유사도가 물체
    외형이 아니라 '실루엣 면적 비율'로 결정되기 쉽다. detector crop 과 조건을
    맞추려면 임베딩 전에 이걸 통과시켜라.
    """
    if not _has_alpha(im):
        return im
    box = im.convert("RGBA").split()[-1].getbbox()
    if box is None:                     # 전부 투명
        logger.warning("alpha_bbox_crop: 전부 투명한 이미지입니다. 원본을 반환합니다.")
        return im
    if margin:
        left, top, right, bottom = box
        box = (max(0, left - margin), max(0, top - margin),
               min(im.width, right + margin), min(im.height, bottom + margin))
    return im.crop(box)


def to_pil(image: ImageInput, input_format: str = DEFAULT_INPUT_FORMAT) -> Image.Image:
    """경로 / PIL / numpy(HWC or HW) -> RGB PIL Image.

    알파가 있으면 ALPHA_BACKGROUND 위에 합성한다 (버리지 않는다).
    """
    if isinstance(image, (str, Path)):
        return flatten_alpha(Image.open(image))

    if isinstance(image, Image.Image):
        return flatten_alpha(image)

    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:                       # 흑백
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim != 3:
            raise ValueError(f"HxW 또는 HxWxC 배열이어야 합니다. 받은 shape={arr.shape}")

        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError(f"빈 crop 입니다: shape={arr.shape}. "
                             f"detector 의 bbox 클리핑을 확인하세요.")

        alpha = None
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] == 4:                 # RGBA/BGRA — 버리지 않고 합성한다
            alpha = arr[..., 3]
            arr = arr[..., :3]
        elif arr.shape[2] != 3:
            raise ValueError(f"채널 수가 1/3/4 가 아닙니다: {arr.shape[2]}")

        fmt = input_format.lower()
        if fmt == "bgr":
            arr = arr[..., ::-1]
        elif fmt != "rgb":
            raise ValueError("input_format 은 'rgb' 또는 'bgr' 이어야 합니다.")

        arr = _to_uint8(arr)

        if alpha is not None:
            a = _to_uint8(alpha).astype(np.float32)[..., None] / 255.0
            bg = np.array(ALPHA_BACKGROUND, dtype=np.float32)
            arr = (arr.astype(np.float32) * a + bg * (1.0 - a))
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # BGR 역슬라이스로 생긴 negative stride 제거
        return Image.fromarray(np.ascontiguousarray(arr))

    raise TypeError(f"지원하지 않는 입력 타입: {type(image)}")


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """0~1 float / 0~255 float / uint8 을 uint8 로 통일."""
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    # 0~1 인지 0~255 인지 구분. 거의 검은 crop 이 0~1 로 오인되어 255 배로
    # 밝아지는 걸 막기 위해 min 도 함께 본다.
    if arr.max() <= 1.0 and arr.min() >= 0.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def l2_normalize(x: np.ndarray, eps: float = 1e-12, owner: str = "") -> np.ndarray:
    """행 단위 L2 정규화. 영벡터는 그대로 둔다 (0 나눗셈 방지).

    영벡터는 Qdrant 코사인 거리에서 정의되지 않으므로 조용히 넘기지 않고 알린다.
    """
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    n_zero = int((norm < eps).sum())
    if n_zero:
        logger.warning("%s: 영벡터 %d개가 정규화를 통과했습니다. Qdrant 코사인 거리에서 "
                       "정의되지 않으니 삽입 전에 걸러내세요.", owner or "l2_normalize", n_zero)
    return (x / np.maximum(norm, eps)).astype(np.float32)


def min_size_filter(crops: Sequence[ImageInput], min_height: int = 50, min_width: int = 20):
    """너무 작은 crop 은 임베딩 품질이 불안정하다. (keep_indices, kept_crops) 반환.

    클러스터링 전에 노이즈를 줄이는 용도.
    """
    keep, kept = [], []
    for i, c in enumerate(crops):
        if isinstance(c, np.ndarray):
            h, w = c.shape[:2]
        elif isinstance(c, Image.Image):
            w, h = c.size
        else:
            keep.append(i); kept.append(c)      # 경로는 열어보지 않고 통과
            continue
        if h >= min_height and w >= min_width:
            keep.append(i); kept.append(c)
    return keep, kept