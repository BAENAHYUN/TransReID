"""
DINOv2 Embedding Extractor
==========================

파이프라인 위치:
    Object Crop -> [DINOv2] -> 768-d Embedding -> Qdrant

DINOv2(Oquab et al., 2023)는 라벨 없이 대규모 이미지로 자기지도 학습한 ViT다.
**텍스트 정렬이 없다.** 대신 인스턴스 수준 시각 유사도가 강해서
"이 가방과 같은 가방" 류의 판별을 잘한다.
(논문의 instance recognition 절에서 weakly-supervised/CLIP 계열 대비 큰 격차를 보고)

파이프라인에서의 역할 분담:
    SigLIP2 -> "검은 백팩"이라는 말로 찾기 (범주 수준, 텍스트 정렬 O)
    DINOv2  -> "이 백팩과 같은 백팩" 찾기 (인스턴스 수준, 텍스트 정렬 X)
서로 못 하는 걸 하므로 객체 crop 에는 둘 다 붙인다.

`embed_text` 를 구현하지 않는다 -> 라우터가 자연어 질의에서 자동 제외한다.
(SOLIDER 와 같은 이유)

모델 크기별 차원
---------------
    facebook/dinov2-small   384
    facebook/dinov2-base    768   <- 기본
    facebook/dinov2-large  1024
    facebook/dinov2-giant  1536

`facebook/dinov2-with-registers-*` 도 같은 차원이다. 단, 시퀀스가
[CLS, register x4, patch...] 구조라서 patch 토큰 시작 위치가 다르다
(아래 `num_prefix_tokens` 참고). registers 판(Darcet et al., ICLR 2024)은
high-norm artifact 토큰을 줄이므로 **feature 가 mean / cls+mean 이면 권장**이다.
feature="cls" 만 쓸 거면 굳이 바꿀 이유는 없다.

전처리 정책 (측정으로 확정)
--------------------------
`resize_mode="pad"` 가 기본이다. 종횡비를 유지한 채 레터박스로 넣는다.

증강 자기검색 하네스(COCO crop, 갤러리 600개, 사람 제외) 결과:

    설정              R@1     R@5     MRR    catNN
    resize@224      0.775   0.878   0.824   0.763
    pad@224         0.828   0.900   0.863   0.782   <- 채택

사람을 포함한 전체 세트에서는 격차가 +2.1%p 로 작았지만, DINOv2 가 실제로
담당하는 객체만 놓고 보면 +5.3%p 로 벌어진다. raw / whiten128 / whitenfull
세 후처리 모두, 그리고 R@5 · MRR · catNN 모두 같은 방향이었다.

이유: 객체 crop 은 (20, 147) 처럼 세로로 긴 것과 (162, 27) 처럼 가로로 긴 것이
섞여 있다. `resize` 는 각 crop 을 서로 다른 방향으로 뭉개므로 같은 개체라도
임베딩이 갈라진다. 실측 왜곡이 4~7배까지 나왔다 (학습 jitter 범위는 ~1.33배).

`resize` 도 그대로 남겨뒀다. 파이프라인에 정사각형에 가까운 crop 만 들어온다면
여백 연산이 없는 `resize` 가 더 빠르고 성능 차이도 없을 것이다.

주의사항
-------
* **patch_size=14** 다. 입력 해상도가 14의 배수가 아니면 오른쪽/아래 가장자리가
  조용히 잘려나간다 (에러가 안 난다). 이 래퍼는 로드 시점에 막는다.
* 정규화는 ImageNet 상수다 (SOLIDER 의 0.5, IRRA 의 CLIP 상수와 다름).
* 보간은 전 모드 BICUBIC 으로 통일했다 (공식 transform 과 동일).
  torchvision `T.Resize` 기본값은 BILINEAR 이므로 명시하지 않으면 어긋난다.
* bf16 을 fp16 보다 우선한다. 다만 이건 **논문 근거가 아니라 일반적인 안전
  선택**이다. artifact 토큰의 노름이 크긴 하지만(Darcet et al., 2024) fp16
  상한(65504)을 위협할 정도라는 근거는 없다. 지수 범위가 fp32 와 같은 bf16 이
  손해볼 게 없어서 기본으로 뒀을 뿐이다.

TODO (논문 기반으로 채울 것)
---------------------------
1. 평가 하네스: Revisited Oxford/Paris(Radenović et al., CVPR 2018) 의 mAP
   프로토콜을 자체 crop 데이터에 얹기. resize vs pad, cls vs cls+mean,
   해상도(224 vs 518), 레이어 선택은 전부 측정으로만 결론난다.
2. PCA-whitening: 인스턴스 검색의 사실상 표준 후처리
   (Jégou & Chum, ECCV 2012). Qdrant 삽입 직전 단계.
3. GeM pooling(Radenović, Tolias, Chum, TPAMI 2019) 을 feature 옵션으로 추가.

※ 위 인용의 연도/학회는 원문으로 재확인할 것.

사용
----
    emb = DINOv2Embedder(model_id="facebook/dinov2-base")   # resize_mode="pad"
    vecs = emb.embed_crops(object_crops, input_format="bgr")   # (N, 768)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from embedders.base import BaseEmbedder

logger = logging.getLogger(__name__)

# DINOv2 는 ImageNet 통계로 정규화한다
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 모델 id 에 포함된 키워드 -> hidden size
_SIZE_HINTS = {
    "small": 384,
    "base": 768,
    "large": 1024,
    "giant": 1536,
}

VALID_FEATURES = ("cls", "mean", "cls+mean")
VALID_RESIZE = ("native", "resize", "center_crop", "pad")

# 학습 시 RandomResizedCrop ratio jitter 범위(대략 3/4~4/3). 이 밖은 분포 밖 입력이다.
# resize 모드에서 목표 종횡비 대비 왜곡이 이 배수를 넘으면 경고한다.
_MAX_DISTORT = 4.0 / 3.0


class DINOv2Embedder(BaseEmbedder):
    """DINOv2 객체 임베딩 추출기 (추론 전용)."""

    def __init__(
        self,
        model_id: str = "facebook/dinov2-base",
        feature: str = "cls",
        image_size: Union[int, Tuple[int, int]] = 224,
        resize_mode: str = "pad",
        max_num_patches: int = 256,       # native 모드 전용 토큰 예산
        min_side_patches: int = 4,        # native 모드에서 한 변의 최소 패치 수
        device: Optional[str] = None,
        batch_size: int = 64,
        l2_normalize: bool = True,
        fp16: bool = True,                # True 면 bf16 우선, 미지원 시 fp16
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        if feature not in VALID_FEATURES:
            raise ValueError(f"feature 는 {VALID_FEATURES} 중 하나여야 합니다 (받은 값: {feature})")
        if resize_mode not in VALID_RESIZE:
            raise ValueError(f"resize_mode 는 {VALID_RESIZE} 중 하나여야 합니다 (받은 값: {resize_mode})")

        self.model_id = model_id
        self.feature = feature
        self.resize_mode = resize_mode
        self.max_num_patches = int(max_num_patches)
        self.min_side_patches = int(min_side_patches)

        # 왜곡 경고 스로틀링용 (배치마다 수십 줄 찍히는 걸 막는다)
        self._distort_warned = 0

        # DIM 은 모델을 로드해야 확정되지만, BaseEmbedder 가 __init__ 에서 검사하므로
        # 모델 id 로 먼저 추정하고 로드 후 실제 값과 대조한다.
        hidden = self._guess_hidden_size(model_id)
        self.DIM = hidden * 2 if feature == "cls+mean" else hidden

        super().__init__(device=device, batch_size=batch_size, l2_normalize=l2_normalize)

        # 저정밀은 CUDA 에서만. CPU 반정밀은 느리거나 미지원 연산이 있다.
        self.dtype = self._pick_dtype(fp16, self.device)

        self.image_size = self._normalize_image_size(image_size)
        self.model = self._build(cache_dir, local_files_only)
        self._check_patch_divisibility()
        self._warn_on_registers()
        self._verify_token_layout()
        self.transform = self._build_transform()

        size_desc = (f"native(<={self.max_num_patches}patches)"
                     if resize_mode == "native" else str(self.image_size))
        logger.info(
            "DINOv2Embedder ready | %s dim=%d feature=%s img=%s resize=%s "
            "device=%s dtype=%s prefix_tokens=%d",
            model_id, self.DIM, feature, size_desc, resize_mode,
            self.device, str(self.dtype).replace("torch.", ""), self.num_prefix_tokens,
        )

    # ------------------------------------------------------------------ #
    # 초기화
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pick_dtype(low_precision: bool, device) -> torch.dtype:
        """bf16 > fp16 > fp32.

        근거는 논문이 아니라 일반적인 수치 안정성이다. bf16 은 지수 범위가 fp32 와
        같아서 fp16 대비 잃을 게 없다. 정밀도(가수부)는 fp16 이 더 높지만 추론
        임베딩에서는 실측상 차이가 미미하다 — 이 역시 확인해 볼 항목이다.
        """
        if not low_precision or not str(device).startswith("cuda"):
            return torch.float32
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:  # 구버전 torch / 비정상 드라이버
            pass
        logger.warning("bf16 미지원 GPU 입니다. fp16 으로 진행합니다. "
                       "large/giant 에서 임베딩에 NaN/Inf 가 뜨면 fp32 로 내리세요.")
        return torch.float16

    @staticmethod
    def _guess_hidden_size(model_id: str) -> int:
        # 경로 앞부분("/models/large/...")에 걸리지 않도록 마지막 요소만 본다
        low = os.path.basename(model_id.rstrip("/")).lower()
        for key, dim in _SIZE_HINTS.items():
            if key in low:
                return dim
        logger.warning("모델 id '%s' 에서 크기를 추정할 수 없어 base(768)로 가정합니다.", model_id)
        return 768

    @staticmethod
    def _normalize_image_size(size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        if isinstance(size, int):
            return (size, size)
        if len(size) != 2:
            raise ValueError(f"image_size 는 int 또는 (h, w) 여야 합니다 (받은 값: {size})")
        h, w = size
        return (int(h), int(w))

    def _build(self, cache_dir, local_files_only):
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError("transformers 가 필요합니다: pip install transformers") from e

        kwargs = {}
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        if local_files_only:
            kwargs["local_files_only"] = True

        model = AutoModel.from_pretrained(self.model_id, **kwargs)

        hidden = int(model.config.hidden_size)
        expected = hidden * 2 if self.feature == "cls+mean" else hidden
        if expected != self.DIM:
            raise RuntimeError(
                f"차원 불일치: 모델의 hidden_size={hidden}, feature='{self.feature}' 이면 "
                f"출력은 {expected} 인데 추정값은 {self.DIM} 이었습니다.\n"
                f"  pipeline.yaml 의 dim 을 {expected} 로 맞추세요."
            )

        self.patch_size = int(getattr(model.config, "patch_size", 14))

        # with-registers 체크포인트는 [CLS, register x N, patch...] 구조다.
        # 이걸 무시하고 tokens[:, 1:] 로 평균내면 register 토큰이 섞인다.
        # register 는 논문상 출력에서 버리는 토큰이므로 제외가 맞다.
        # !! 검증 필요: HF 구현의 토큰 순서가 정말 [CLS, reg, patch] 인지
        #    (patch 뒤에 붙는 구현도 있을 수 있다). 아래 self-check 로 확인한다.
        self.num_prefix_tokens = 1 + int(getattr(model.config, "num_register_tokens", 0))

        model = model.to(device=self.device, dtype=self.dtype).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def _check_patch_divisibility(self) -> None:
        """patch_size 로 안 나뉘면 가장자리가 조용히 잘린다. 미리 막는다."""
        if self.resize_mode == "native":
            # native 는 이미지마다 해상도를 계산하므로 image_size 를 쓰지 않는다.
            if self.max_num_patches < self.min_side_patches ** 2:
                raise ValueError(
                    f"max_num_patches({self.max_num_patches}) 가 너무 작습니다. "
                    f"min_side_patches({self.min_side_patches})^2 이상이어야 합니다."
                )
            return
        h, w = self.image_size
        bad = [n for n, v in (("height", h), ("width", w)) if v % self.patch_size != 0]
        if bad:
            near_h = round(h / self.patch_size) * self.patch_size
            near_w = round(w / self.patch_size) * self.patch_size
            raise ValueError(
                f"image_size {self.image_size} 가 patch_size {self.patch_size} 의 배수가 "
                f"아닙니다 ({', '.join(bad)}). 에러 없이 가장자리가 잘리므로 막습니다.\n"
                f"  가까운 값: ({near_h}, {near_w}). 흔히 쓰는 값: 224, 336, 448, 518"
            )

    def _warn_on_registers(self) -> None:
        """patch 토큰을 쓰는데 registers 판이 아니면 알려준다."""
        if self.feature == "cls":
            return
        if self.num_prefix_tokens > 1:
            return
        logger.warning(
            "feature='%s' 는 patch 토큰을 평균냅니다. high-norm artifact 토큰이 평균을 "
            "지배할 수 있으니 'facebook/dinov2-with-registers-*' 체크포인트를 권장합니다 "
            "(Darcet et al., ICLR 2024).", self.feature,
        )

    @torch.inference_mode()
    def _verify_token_layout(self) -> None:
        """더미 forward 로 시퀀스 길이가 예상과 맞는지 확인한다.

        num_prefix_tokens 를 config 에서 추정하므로, 구현이 바뀌거나 추정이 틀리면
        register 토큰이 patch 평균에 섞인 채 조용히 돌아간다. 로드 시점에 잡는다.
        (토큰 '순서'까지는 검증하지 못하고 '개수'만 본다.)
        """
        if self.resize_mode == "native":
            return
        h, w = self.image_size
        dummy = torch.zeros(1, 3, h, w, device=self.device, dtype=self.dtype)
        seq = self.model(pixel_values=dummy).last_hidden_state.shape[1]
        expected_patches = (h // self.patch_size) * (w // self.patch_size)
        actual_prefix = seq - expected_patches
        if actual_prefix != self.num_prefix_tokens:
            raise RuntimeError(
                f"토큰 레이아웃 불일치: 시퀀스 길이 {seq}, 예상 patch 수 "
                f"{expected_patches} -> prefix {actual_prefix} 인데 추정값은 "
                f"{self.num_prefix_tokens} 입니다. register 토큰이 patch 평균에 "
                f"섞일 수 있으므로 중단합니다. transformers 버전과 "
                f"config.num_register_tokens 를 확인하세요."
            )

    def _build_transform(self):
        import torchvision.transforms as T

        h, w = self.image_size
        bicubic = T.InterpolationMode.BICUBIC
        # 흑백/RGBA/팔레트 이미지가 들어오면 ToTensor 채널 수가 3이 아니게 되어
        # Normalize 에서 터진다. 입구에서 통일한다.
        to_rgb = T.Lambda(_to_rgb)
        norm = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        if self.resize_mode in ("native", "pad"):
            # 해상도/여백 계산이 이미지마다 다르므로 여기서는 Tensor 화 + 정규화만.
            # 실제 리사이즈는 _encode 에서 수행한다.
            return T.Compose([to_rgb, T.ToTensor(), norm])

        if self.resize_mode == "resize":
            # 종횡비를 무시하고 전체를 목표 크기로. 물체가 잘리지 않는다. (확정 정책)
            return T.Compose([to_rgb, T.Resize((h, w), interpolation=bicubic),
                              T.ToTensor(), norm])

        # center_crop: HF 프로세서 기본 동작. 종횡비는 지키지만 가장자리가 잘린다.
        short = int(round(min(h, w) * 256 / 224))
        return T.Compose([to_rgb, T.Resize(short, interpolation=bicubic),
                          T.CenterCrop((h, w)), T.ToTensor(), norm])

    # ------------------------------------------------------------------ #
    # 추론
    # ------------------------------------------------------------------ #
    def fit_patch_grid(self, height: int, width: int) -> Tuple[int, int]:
        """원본 종횡비를 유지하면서 토큰 예산 안에 들어가는 (H, W) 픽셀 크기를 정한다.

        SigLIP2 NaFlex 와 같은 발상이다. 다만 NaFlex 는 모델 자체가 가변 종횡비로
        **학습된** 반면 DINOv2 는 그렇지 않고 위치 임베딩 보간에 의존할 뿐이다.
        따라서 native 모드는 검증되지 않은 이식이며, 평가 하네스로 확인하기 전에는
        실험용으로만 쓴다.
        """
        p = self.patch_size
        budget = self.max_num_patches
        m = self.min_side_patches
        ar = max(height, 1) / max(width, 1)     # h/w

        # h_p * w_p <= budget, h_p / w_p ~= ar  ->  w_p = sqrt(budget / ar)
        # int() 절삭 대신 반올림. 절삭은 극단 종횡비에서 목표를 크게 벗어난다.
        w_p = max(m, int(round((budget / ar) ** 0.5)))
        h_p = max(m, int(round(w_p * ar)))

        # 예산 초과분을 긴 쪽부터 줄인다
        while h_p * w_p > budget and (h_p > m or w_p > m):
            if h_p >= w_p and h_p > m:
                h_p -= 1
            elif w_p > m:
                w_p -= 1
            else:
                break

        # min_side_patches 에 걸리면 종횡비가 크게 어긋날 수 있다 (100:1 -> 16:1 등)
        got = h_p / w_p
        err = max(got / ar, ar / got)
        if err > 1.15:
            logger.debug("native: 종횡비 %.2f -> %.2f (%.0f%% 오차), grid=%dx%d. "
                         "min_side_patches/max_num_patches 를 조정하세요.",
                         ar, got, (err - 1) * 100, h_p, w_p)
        return h_p * p, w_p * p

    def _pool(self, tokens: torch.Tensor,
              token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, prefix+N, D) -> (B, DIM).

        token_mask: (B, N) bool. True 인 patch 토큰만 평균에 포함한다.
                    pad 모드에서 검은 여백 토큰을 제외하는 데 쓴다.
        """
        cls = tokens[:, 0]                                  # 최종 layernorm 적용됨
        if self.feature == "cls":
            return cls

        patch = tokens[:, self.num_prefix_tokens:]          # register 토큰 제외
        if token_mask is None:
            pooled = patch.mean(dim=1)
        else:
            if token_mask.shape[1] != patch.shape[1]:
                # 브로드캐스팅으로 조용히 통과하는 걸 막는다
                raise RuntimeError(
                    f"pad 마스크 길이 {token_mask.shape[1]} != patch 토큰 수 "
                    f"{patch.shape[1]}. image_size {self.image_size} 와 "
                    f"patch_size {self.patch_size} 조합을 확인하세요."
                )
            m = token_mask.unsqueeze(-1).to(patch.dtype)
            pooled = (patch * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

        if self.feature == "mean":
            return pooled

        # CLS 노름은 patch 평균 노름보다 훨씬 크다. 그대로 concat 한 뒤 전체를
        # L2 정규화하면(BaseEmbedder 가 그렇게 한다) 코사인 유사도를 CLS 절반이
        # 지배하고 mean 절반은 거의 기여하지 않는다 — cls+mean 을 쓰는 의미가
        # 사라진다. 절반씩 먼저 정규화해서 두 신호의 기여를 맞춘다.
        cls = torch.nn.functional.normalize(cls.float(), dim=-1)
        pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
        return torch.cat([cls, pooled], dim=-1)

    @torch.inference_mode()
    def _forward(self, batch: torch.Tensor,
                 token_mask: Optional[torch.Tensor] = None) -> np.ndarray:
        batch = batch.to(device=self.device, dtype=self.dtype)
        tokens = self.model(pixel_values=batch).last_hidden_state   # (B, prefix+N, D)
        feats = self._pool(tokens, token_mask)
        return feats.float().cpu().numpy()

    @torch.inference_mode()
    def _encode(self, images: List[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.DIM), dtype=np.float32)

        if self.resize_mode == "native":
            return self._encode_native(images)
        if self.resize_mode == "pad":
            return self._encode_pad(images)

        # --- resize / center_crop: 전부 같은 크기라 한 번에 stack --- #
        if self.resize_mode == "resize":
            self._warn_if_distorted(images)
        batch = torch.stack([self.transform(im) for im in images])
        return self._forward(batch)

    def _warn_if_distorted(self, images: List[Image.Image]) -> None:
        """학습 분포 밖 종횡비 왜곡을 알린다 (스로틀링)."""
        if self._distort_warned >= 3:
            return
        th, tw = self.image_size
        target_ar = th / tw
        worst, worst_size = 1.0, None
        for im in images:
            ratio = (im.height / max(im.width, 1)) / target_ar
            d = max(ratio, 1.0 / ratio)
            if d > worst:
                worst, worst_size = d, (im.width, im.height)
        if worst > _MAX_DISTORT:
            self._distort_warned += 1
            logger.warning(
                "resize 모드 종횡비 왜곡 %.2fx (예: %s -> %s). 학습 jitter 범위(~%.2fx) "
                "밖이라 분포 밖 입력입니다. 사람 crop 처럼 길쭉한 객체가 많다면 "
                "pad 모드와 mAP 를 비교해 보세요.",
                worst, worst_size, (tw, th), _MAX_DISTORT,
            )

    def _encode_native(self, images: List[Image.Image]) -> np.ndarray:
        """이미지마다 해상도가 다르므로 같은 크기끼리 묶어서 처리."""
        # 서로 다른 shape 은 한 텐서로 stack 할 수 없다. 종횡비가 비슷한 것들은
        # 같은 격자로 떨어지므로, 실제 버킷 수는 배치 크기보다 훨씬 적다.
        buckets: Dict[Tuple[int, int], List[int]] = {}
        for i, im in enumerate(images):
            hw = self.fit_patch_grid(im.height, im.width)
            buckets.setdefault(hw, []).append(i)

        out = np.empty((len(images), self.DIM), dtype=np.float32)
        for (h, w), idxs in buckets.items():
            tensors = [
                self.transform(_to_rgb(images[i]).resize((w, h), Image.BICUBIC))
                for i in idxs
            ]
            feats = self._forward(torch.stack(tensors))
            for slot, i in enumerate(idxs):
                out[i] = feats[slot]

        logger.debug("native 모드: %d개 이미지 -> %d개 해상도 버킷",
                     len(images), len(buckets))
        return out

    def _encode_pad(self, images: List[Image.Image]) -> np.ndarray:
        """레터박스. 검은 여백 토큰은 평균에서 제외한다.

        NaFlex 는 attention mask 로 여백을 처리하지만 HF DINOv2 경로에는 그게 없다.
        최소한 풀링 단계에서라도 빼야 여백이 임베딩을 오염시키지 않는다.
        (feature='cls' 면 마스크는 쓰이지 않는다 — CLS 는 여전히 여백을 본다.)
        """
        h, w = self.image_size
        p = self.patch_size
        gh, gw = h // p, w // p

        tensors, masks = [], []
        for im in images:
            padded, box = _resize_pad(_to_rgb(im), h, w)
            tensors.append(self.transform(padded))
            masks.append(_valid_patch_mask(box, gh, gw, p))

        batch = torch.stack(tensors)
        token_mask = torch.from_numpy(np.stack(masks)).to(self.device)
        return self._forward(batch, token_mask=token_mask)

    # 텍스트 인코더 없음 — embed_text 를 의도적으로 구현하지 않는다.
    # DINOv2 는 언어와 정렬된 공간이 아니다. 자연어 객체 검색은 SigLIP2 담당.


# --------------------------------------------------------------------------- #
# 헬퍼
# --------------------------------------------------------------------------- #
def _to_rgb(im: Image.Image) -> Image.Image:
    return im if im.mode == "RGB" else im.convert("RGB")


def _resize_pad(im: Image.Image, h: int, w: int) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """종횡비를 유지한 채 (h, w) 캔버스 가운데에 넣고, 유효 영역 box 를 함께 반환."""
    scale = min(w / im.width, h / im.height)
    new_w = max(1, int(round(im.width * scale)))
    new_h = max(1, int(round(im.height * scale)))
    im = im.resize((new_w, new_h), Image.BICUBIC)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(im, (left, top))
    return canvas, (left, top, left + new_w, top + new_h)


def _valid_patch_mask(box: Tuple[int, int, int, int], gh: int, gw: int, p: int) -> np.ndarray:
    """패치 중심이 유효 영역 안에 있으면 True. 반환 shape 은 (gh*gw,)."""
    left, top, right, bottom = box
    cx = (np.arange(gw) + 0.5) * p          # (gw,)
    cy = (np.arange(gh) + 0.5) * p          # (gh,)
    mx = (cx >= left) & (cx < right)
    my = (cy >= top) & (cy < bottom)
    mask = my[:, None] & mx[None, :]        # (gh, gw)
    if not mask.any():                      # 극단적으로 납작한 경우 대비
        mask[:] = True
    return mask.reshape(-1)


# --------------------------------------------------------------------------- #
# 스모크 테스트
#   python -m embedders.object.dinov2_embedder --images bag1.jpg bag2.jpg car.jpg
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ap = argparse.ArgumentParser(description="DINOv2 embedding smoke test")
    ap.add_argument("--model-id", default="facebook/dinov2-base")
    ap.add_argument("--feature", default="cls", choices=VALID_FEATURES)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--resize-mode", default="pad", choices=VALID_RESIZE)
    ap.add_argument("--device", default=None)
    ap.add_argument("--images", nargs="+", required=True)
    args = ap.parse_args()

    emb = DINOv2Embedder(
        model_id=args.model_id,
        feature=args.feature,
        image_size=args.image_size,
        resize_mode=args.resize_mode,
        device=args.device,
    )

    vecs = emb.embed_crops(args.images)
    print(f"\nembeddings: {vecs.shape}  norm={np.linalg.norm(vecs[0]):.4f}")

    print("\n쌍별 코사인 유사도 (같은 물체면 높아야 함):")
    sims = vecs @ vecs.T
    names = [os.path.basename(p) for p in args.images]
    print("        " + "  ".join(f"{n[:8]:>8}" for n in names))
    for n, row in zip(names, sims):
        print(f"{n[:8]:>8}" + "  ".join(f"{v:8.4f}" for v in row))