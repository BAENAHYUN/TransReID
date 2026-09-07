"""
SigLIP2 Embedding Extractor
===========================

【이 파일의 정체】
    파이프라인 **모델 파일** 이다.
    경로 : embedders/siglip2_embedder.py
    등록 : pipeline.yaml -> retrievers.siglip2.module / class
    호출 : EmbedderRegistry 가 SigLIP2Embedder(**params) 로 생성하고
           Router 가 embed_crops() 를 호출한다.

    실험용 배치 스크립트(siglip_semantic 을 import 해서 .npz 로 저장하는 것)와는
    **별개 파일** 이다. DB 적재는 반드시 이 파일 경로로만 할 것.
    이 파일은 이미지/텍스트 임베딩에 동일한 L2 정규화 규칙을 적용해
    Qdrant 밖에서 직접 cosine/dot-product 비교할 때도 결과를 일관되게 만든다.

파이프라인 위치:
    Person/Object Crop -> [SigLIP2] -> DIM-d Embedding -> Qdrant (기본 base: 768-d)
    (scope: all — 사람/객체 crop 모두에 적용)

SigLIP2(Google, 2025)는 이미지-텍스트 대조학습 모델이다. 개방 어휘 의미 표현을
담당하며, **텍스트 정렬이 있어 자연어 검색이 가능**하다.

이 파일의 역할:
    SigLIP2 이미지 임베딩 -> 사람/객체 crop 의 의미 특징 벡터
    SigLIP2 텍스트 임베딩 -> 번역/전처리된 검색 query 의 의미 특징 벡터
    두 벡터는 같은 공간에서 cosine similarity 로 비교한다.

    다른 모델의 역할/호출 관계는 이 파일에서 정의하지 않는다.

NaFlex 를 쓰는 이유
------------------
`-naflex` 체크포인트는 원본 종횡비를 유지한 채 인코딩한다. 고정 해상도 모델은
입력을 정사각형으로 리사이즈하는데, **3:1 보행자 crop 을 옆으로 뭉갠다.**
사람 crop 이 대부분인 이 파이프라인에서는 naflex 가 유리할 가능성이 높다.
(확정은 아니므로 고정 해상도 체크포인트와 A/B 해볼 가치가 있다)

임베딩 차원
-----------
모델 이름으로 하드코딩하지 않는다. AutoConfig 에서 vision/text 최종 출력 차원을
읽고 서로 같은지 검증한다. 기본 base-patch16-naflex 는 768-d 이다.

주의사항
-------
* **`max_num_patches` 는 색인할 때와 질의할 때 같아야 한다.** naflex 는 이 값으로
  유효 해상도가 정해지므로, 다르면 같은 이미지도 다른 벡터가 나온다.
  576 은 대략 384x384 상당의 면적이다. pipeline.yaml 에 적힌 값이 유일한 기준이다.
* 텍스트는 반드시 `padding="max_length", max_length=64`. SigLIP 계열은 고정 길이
  패딩으로 학습돼서, 기본 패딩을 쓰면 결과가 조용히 나빠진다.
* transformers 버전에 따라 `get_image_features` / `get_text_features` 반환형이
  Tensor 또는 `BaseModelOutputWithPooling` 계열일 수 있으므로 둘 다 처리한다.
* 입력 정규화는 AutoProcessor 가 모델 설정에 맞게 적용하므로 이 파일에서
  별도의 이미지 Normalize 를 중복 적용하지 않는다.
* 저정밀은 **model.half() 전체 캐스팅 대신 autocast** 를 쓴다. naflex 프로세서는
  pixel_values 외에 pixel_attention_mask(bool)·spatial_shapes(int) 를 함께
  돌려주는데, 모델을 통째로 half 로 내리면 이들 dtype 과 어긋나 transformers
  버전에 따라 조용히 깨지거나 예외가 난다. autocast 는 가중치를 fp32 로 두고
  연산만 fp16 으로 돌리므로 이 문제가 없다.

사용
----
    emb = SigLIP2Embedder(model_id="google/siglip2-base-patch16-naflex")
    vecs = emb.embed_crops(crops, input_format="bgr")   # (N, 768) L2 정규화
    qvec = emb.embed_text("a man with a black backpack")  # (1, 768)
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from embedders.base import BaseEmbedder, l2_normalize

logger = logging.getLogger(__name__)

# SigLIP 계열 텍스트 컨텍스트 길이 (학습 시 고정)
TEXT_MAX_LENGTH = 64


class SigLIP2Embedder(BaseEmbedder):
    """SigLIP2 이미지/텍스트 임베딩 추출기 (추론 전용)."""

    def __init__(
        self,
        model_id: str = "google/siglip2-base-patch16-naflex",
        max_num_patches: int = 576,
        device: Optional[str] = None,
        batch_size: int = 32,
        l2_normalize: bool = True,
        fp16: bool = True,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        l2_normalize_out: Optional[bool] = None,   # 구버전 인자명 호환
    ) -> None:
        self.model_id = model_id
        self.max_num_patches = int(max_num_patches)
        self.is_naflex = "naflex" in model_id.lower()

        if l2_normalize_out is not None:
            logger.warning(
                "l2_normalize_out 는 예전 인자명입니다. l2_normalize 를 쓰세요."
            )
            l2_normalize = bool(l2_normalize_out)

        if not self.is_naflex and max_num_patches != 576:
            logger.warning(
                "'%s' 는 naflex 체크포인트가 아니라 max_num_patches(%d) 가 "
                "무시됩니다. 고정 해상도로 리사이즈되며 보행자 crop 이 "
                "가로로 뭉개집니다.", model_id, max_num_patches,
            )

        # BaseEmbedder 가 __init__ 에서 DIM 을 검사하므로 모델 이름으로
        # 추정하지 않고, Hugging Face config 에서 실제 출력 차원을 먼저 읽는다.
        self.DIM = self._load_config_dim(
            model_id=model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        super().__init__(
            device=device,
            batch_size=batch_size,
            l2_normalize=l2_normalize,
        )

        # 가중치는 fp32 로 두고 autocast 로만 저정밀을 쓴다 (위 주의사항 참고)
        self.use_amp = bool(fp16) and str(self.device).startswith("cuda")

        self.model, self.processor = self._build(cache_dir, local_files_only)

        logger.info(
            "SigLIP2Embedder ready | %s dim=%d naflex=%s patches=%d device=%s amp=%s",
            model_id, self.DIM, self.is_naflex, self.max_num_patches,
            self.device, self.use_amp,
        )

    # ------------------------------------------------------------------ #
    # 초기화
    # ------------------------------------------------------------------ #
    @classmethod
    def _load_config_dim(
        cls,
        model_id: str,
        cache_dir: Optional[str],
        local_files_only: bool,
    ) -> int:
        """모델 이름을 추정하지 않고 config 에서 실제 임베딩 차원을 읽는다."""
        try:
            from transformers import AutoConfig
        except ImportError as e:
            raise ImportError(
                "transformers 가 필요합니다: pip install transformers"
            ) from e

        kwargs: Dict[str, Any] = {}
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        if local_files_only:
            kwargs["local_files_only"] = True

        config = AutoConfig.from_pretrained(model_id, **kwargs)
        dim = cls._config_dim(config)
        if dim is None:
            raise RuntimeError(
                "SigLIP2 config 에서 임베딩 차원을 확인할 수 없습니다: "
                f"{model_id}"
            )
        return int(dim)

    def _build(self, cache_dir, local_files_only):
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "transformers 가 필요합니다: pip install transformers"
            ) from e

        kwargs: Dict[str, Any] = {}
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        if local_files_only:
            kwargs["local_files_only"] = True

        model = AutoModel.from_pretrained(self.model_id, **kwargs)
        processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)

        actual = self._config_dim(model.config)
        if actual is None:
            raise RuntimeError(
                "로드된 SigLIP2 모델 config 에서 임베딩 차원을 확인할 수 없습니다."
            )
        if actual != self.DIM:
            raise RuntimeError(
                f"차원 불일치: 사전 config={self.DIM}, 로드된 모델={actual}.\n"
                f"  pipeline.yaml 의 retrievers.siglip2.dim 을 "
                f"{actual} 로 맞추세요."
            )

        # half() 하지 않는다. autocast 로 처리한다.
        model.float().to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model, processor

    @staticmethod
    def _config_dim(config) -> Optional[int]:
        """SigLIP2 이미지/텍스트 최종 임베딩 차원을 config 에서 확인한다.

        vision tower 는 보통 hidden_size 가 최종 이미지 임베딩 차원이고,
        text tower 는 projection_size 가 있으면 projection 출력 차원이 최종 차원이다.
        두 값이 모두 있으면 반드시 같아야 이미지/텍스트를 같은 공간에서 비교할 수 있다.
        """
        vision_cfg = getattr(config, "vision_config", None)
        text_cfg = getattr(config, "text_config", None)

        vision_dim: Optional[int] = None
        text_dim: Optional[int] = None

        if vision_cfg is not None:
            hidden_size = getattr(vision_cfg, "hidden_size", None)
            if hidden_size is not None:
                vision_dim = int(hidden_size)

        if text_cfg is not None:
            projection_size = getattr(text_cfg, "projection_size", None)
            hidden_size = getattr(text_cfg, "hidden_size", None)

            if projection_size is not None:
                text_dim = int(projection_size)
            elif hidden_size is not None:
                text_dim = int(hidden_size)

        if vision_dim is not None and text_dim is not None:
            if vision_dim != text_dim:
                raise RuntimeError(
                    "SigLIP2 이미지/텍스트 임베딩 차원이 다릅니다: "
                    f"vision={vision_dim}, text={text_dim}"
                )
            return vision_dim

        if vision_dim is not None:
            return vision_dim
        if text_dim is not None:
            return text_dim

        hidden_size = getattr(config, "hidden_size", None)
        return int(hidden_size) if hidden_size is not None else None

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    # ------------------------------------------------------------------ #
    # 출력 정규화
    # ------------------------------------------------------------------ #
    @staticmethod
    def _unwrap_pooled(out) -> torch.Tensor:
        """transformers 버전에 따라 반환 타입이 다르다.

        4.x : get_image_features -> Tensor
        5.x : get_image_features -> BaseModelOutputWithPooling
        """
        if isinstance(out, torch.Tensor):
            return out

        for attr in ("pooler_output", "image_embeds", "text_embeds"):
            v = getattr(out, attr, None)
            if isinstance(v, torch.Tensor):
                return v

        if isinstance(out, (tuple, list)):
            # BaseModelOutputWithPooling 이 tuple/list 형태로 전달될 경우
            # out[0] 은 last_hidden_state (3-D) 일 수 있다.
            # 최종 검색용 pooled embedding 은 보통 (N, DIM) 2-D 텐서이므로
            # 차원을 확인해서 선택한다.
            for v in out:
                if isinstance(v, torch.Tensor) and v.ndim == 2:
                    return v

        raise RuntimeError(
            f"SigLIP2 출력에서 pooled embedding을 찾을 수 없습니다: {type(out)}"
        )

    def _to_device(self, inputs) -> Dict[str, Any]:
        """프로세서 출력을 device 로 옮긴다.

        naflex 는 pixel_values 외에 pixel_attention_mask / spatial_shapes 를
        함께 준다. transformers 버전에 따라 이들이 텐서가 아닌 list 나 ndarray
        로 올 수 있으므로, 텐서가 아닌 값은 그대로 통과시킨다.
        (dtype 은 바꾸지 않는다 — mask 는 bool, spatial_shapes 는 int 여야 한다)
        """
        out: Dict[str, Any] = {}
        for k, v in dict(inputs).items():
            out[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
        return out

    # ------------------------------------------------------------------ #
    # 추론
    # ------------------------------------------------------------------ #
    def _image_inputs(self, images: List[Image.Image]):
        """naflex 는 max_num_patches 를 받고, 고정 해상도 모델은 받지 않는다."""
        kwargs: Dict[str, Any] = {"images": images, "return_tensors": "pt"}
        if self.is_naflex:
            kwargs["max_num_patches"] = self.max_num_patches
        return self.processor(**kwargs)

    @torch.inference_mode()
    def _encode(self, images: List[Image.Image]) -> np.ndarray:
        """RGB PIL 리스트 -> (N, DIM).

        경로/numpy/알파 처리, 배치 분할, L2 정규화, 차원·NaN 검증은
        BaseEmbedder 가 이미 했다. 여기서는 순수 forward 만 한다.
        """
        inputs = self._to_device(self._image_inputs(images))

        with self._autocast():
            out = self.model.get_image_features(**inputs)

        return self._unwrap_pooled(out).float().cpu().numpy()

    @torch.inference_mode()
    def embed_text(
        self,
        text: Union[str, Sequence[str]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """자연어 -> (N, DIM). 이미지 임베딩과 같은 공간이라 바로 코사인 비교 가능.

        라우터는 이 메서드의 **존재 여부**로 텍스트 검색 참여를 판단한다.
        SOLIDER / DINOv2 에는 없으므로 자동으로 빠진다.

        SigLIP 계열은 고정 길이 패딩(64)으로 학습됐다. 기본 패딩을 쓰면
        에러 없이 성능만 떨어지므로 반드시 max_length 로 맞춘다.

        학습 캡션이 영어이므로 한국어 쿼리는 상위 레이어에서 번역할 것.
        """
        texts = [text] if isinstance(text, str) else list(text)
        if not texts:
            return np.zeros((0, self.DIM), dtype=np.float32)

        bs = batch_size or self.batch_size
        outs: List[np.ndarray] = []

        for i in range(0, len(texts), bs):
            chunk = [t.lower() for t in texts[i:i + bs]]   # SigLIP 표준 전처리
            inputs = self.processor(
                text=chunk,
                padding="max_length",
                truncation=True,
                max_length=TEXT_MAX_LENGTH,
                return_tensors="pt",
            )
            inputs = self._to_device(inputs)

            with self._autocast():
                out = self.model.get_text_features(**inputs)

            outs.append(self._unwrap_pooled(out).float().cpu().numpy())

        feats = np.concatenate(outs, axis=0).astype(np.float32)

        if feats.shape[1] != self.DIM:
            raise RuntimeError(
                f"텍스트 임베딩 차원 {feats.shape[1]} != DIM {self.DIM}"
            )
        if not np.isfinite(feats).all():
            raise RuntimeError(
                "텍스트 임베딩에 NaN/Inf 가 있습니다. fp16=False 로 시도해 보세요."
            )

        # 이미지 경로는 BaseEmbedder 가 정규화하므로, 텍스트도 같은 규칙을
        # 따라야 두 벡터를 같은 공간에서 비교할 수 있다.
        if self.l2_normalize:
            feats = l2_normalize(feats, owner=type(self).__name__)
        return feats


# --------------------------------------------------------------------------- #
# 스모크 테스트 — 이미지 하나와 후보 문장들의 유사도를 출력
#   python -m embedders.siglip2_embedder --image photo.jpg \
#       --texts "a woman with a gold crown" "a dog on the beach"
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    ap = argparse.ArgumentParser(description="SigLIP2 embedding smoke test")
    ap.add_argument("--model-id", default="google/siglip2-base-patch16-naflex")
    ap.add_argument("--max-num-patches", type=int, default=576)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-fp16", action="store_true",
                    help="autocast 끄고 순수 fp32 로 실행")
    ap.add_argument("--image", required=True)
    ap.add_argument("--texts", nargs="+", required=True)
    args = ap.parse_args()

    emb = SigLIP2Embedder(
        model_id=args.model_id,
        max_num_patches=args.max_num_patches,
        device=args.device,
        fp16=not args.no_fp16,
    )

    # 경로를 그대로 넘긴다 — BaseEmbedder 가 처리한다
    v = emb.embed_crops([args.image], input_format="rgb")
    t = emb.embed_text(args.texts)
    print(f"\nimage: {v.shape}  text: {t.shape}")
    print(f"norm  image={np.linalg.norm(v[0]):.4f} text={np.linalg.norm(t[0]):.4f}")

    sims = t @ v[0]
    print("\n=== Semantic similarity ===")
    for text, s in sorted(zip(args.texts, sims), key=lambda x: -x[1]):
        print(f"{s:.4f} | {text}")