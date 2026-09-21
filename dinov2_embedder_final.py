#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final DINOv2 Object Embedder
============================

최종 운영 설정
- Model      : facebook/dinov2-with-registers-giant
- Backbone   : ViT-G/14
- Registers  : 4
- Input      : Aspect-Ratio Preserve -> Center Pad 224x224
- Padding    : ImageNet mean RGB ~= (124, 116, 104)
- Normalize  : ImageNet mean/std
- Feature    : CLS token
- Output     : 1536D
- Final norm : L2 Normalize

목적
- RF-DETR 등으로 생성된 object crop의 시각적 유사도 임베딩
- Qdrant named vector `dinov2` 저장용

권장 pipeline.yaml
-----------------
dinov2:
  tool: object
  scope: object
  supports_text: false
  dim: 1536
  weight: 1.0
  module: embedders.object.dinov2_embedder_final
  class: DINOv2FinalEmbedder
  params:
    batch_size: 4

간단 테스트
-----------
python -c "import cv2,numpy as np; from embedders.object.dinov2_embedder_final import DINOv2FinalEmbedder; im=cv2.imread(r'YOUR_IMAGE.jpg'); e=DINOv2FinalEmbedder(batch_size=1); v=e.embed_crops([im],input_format='bgr'); print(v.shape,np.linalg.norm(v[0]))"
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel
import torchvision.transforms as T


MODEL_ID = "facebook/dinov2-with-registers-giant"

IMAGE_SIZE = 224
PATCH_SIZE = 14
REGISTER_TOKENS = 4
OUTPUT_DIM = 1536

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ImageNet mean을 0~255 RGB로 근사.
# Normalize 이후 padding 영역이 거의 0이 되도록 사용.
IMAGENET_MEAN_RGB = tuple(int(round(v * 255)) for v in IMAGENET_MEAN)


ImageInput = Union[
    str,
    Path,
    Image.Image,
    np.ndarray,
]


class DINOv2FinalEmbedder:
    """
    DINOv2 Giant + Registers 기반 최종 Object Embedder.

    입력 지원:
    - np.ndarray (OpenCV BGR/RGB)
    - PIL.Image
    - str / pathlib.Path

    출력:
    - np.ndarray
    - shape = (N, 1536)
    - dtype = float32
    - 각 row L2 norm ~= 1.0
    """

    DIM = OUTPUT_DIM
    MODEL_ID = MODEL_ID

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        batch_size: int = 4,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        fp16: bool = True,
    ) -> None:

        if batch_size <= 0:
            raise ValueError("batch_size는 1 이상이어야 합니다.")

        self.batch_size = int(batch_size)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        if self.device.type == "cuda" and fp16:
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        load_kwargs = {}

        if cache_dir is not None:
            load_kwargs["cache_dir"] = cache_dir

        if local_files_only:
            load_kwargs["local_files_only"] = True

        self.model = AutoModel.from_pretrained(
            MODEL_ID,
            **load_kwargs,
        )

        self.model = self.model.to(
            device=self.device,
            dtype=self.dtype,
        ).eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.model_id = MODEL_ID
        self.feature = "cls"
        self.resize_mode = "pad"
        self.image_size = IMAGE_SIZE
        self.l2_normalize = True

        self._validate_model_contract()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ])

    def _validate_model_contract(self) -> None:
        hidden_size = int(
            getattr(
                self.model.config,
                "hidden_size",
                -1,
            )
        )

        patch_size = int(
            getattr(
                self.model.config,
                "patch_size",
                PATCH_SIZE,
            )
        )

        num_register_tokens = int(
            getattr(
                self.model.config,
                "num_register_tokens",
                REGISTER_TOKENS,
            )
        )

        if hidden_size != OUTPUT_DIM:
            raise RuntimeError(
                "DINOv2 output dimension mismatch: "
                f"expected={OUTPUT_DIM}, actual={hidden_size}"
            )

        if patch_size != PATCH_SIZE:
            raise RuntimeError(
                "DINOv2 patch size mismatch: "
                f"expected={PATCH_SIZE}, actual={patch_size}"
            )

        if num_register_tokens != REGISTER_TOKENS:
            raise RuntimeError(
                "DINOv2 register token mismatch: "
                f"expected={REGISTER_TOKENS}, actual={num_register_tokens}"
            )

        # 224 / 14 = 16 -> patch 256개
        expected_patch_tokens = (
            IMAGE_SIZE // PATCH_SIZE
        ) ** 2

        expected_seq_len = (
            1
            + REGISTER_TOKENS
            + expected_patch_tokens
        )

        with torch.inference_mode():
            dummy = torch.zeros(
                1,
                3,
                IMAGE_SIZE,
                IMAGE_SIZE,
                device=self.device,
                dtype=self.dtype,
            )

            output = self.model(
                pixel_values=dummy
            )

            seq_len = int(
                output.last_hidden_state.shape[1]
            )

        if seq_len != expected_seq_len:
            raise RuntimeError(
                "DINOv2 token layout mismatch: "
                f"expected sequence={expected_seq_len}, actual={seq_len}"
            )

    @staticmethod
    def _pil_from_input(
        image: ImageInput,
        input_format: str = "bgr",
    ) -> Image.Image:

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, (str, Path)):
            with Image.open(image) as im:
                return im.convert("RGB").copy()

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "지원하지 않는 이미지 타입: "
                f"{type(image).__name__}"
            )

        arr = image

        if arr.ndim == 2:
            arr = np.stack(
                [arr, arr, arr],
                axis=-1,
            )

        if arr.ndim != 3:
            raise ValueError(
                f"이미지 shape이 잘못되었습니다: {arr.shape}"
            )

        if arr.shape[2] == 4:
            arr = arr[:, :, :3]

        if arr.shape[2] != 3:
            raise ValueError(
                f"3-channel 이미지만 지원합니다: {arr.shape}"
            )

        if arr.dtype != np.uint8:
            arr = np.clip(
                arr,
                0,
                255,
            ).astype(np.uint8)

        fmt = input_format.lower()

        if fmt == "bgr":
            arr = arr[:, :, ::-1]
        elif fmt != "rgb":
            raise ValueError(
                "input_format은 'bgr' 또는 'rgb'만 가능합니다."
            )

        return Image.fromarray(
            np.ascontiguousarray(arr),
            mode="RGB",
        )

    @staticmethod
    def _aspect_ratio_pad_224(
        image: Image.Image,
    ) -> Image.Image:
        """
        객체 종횡비를 유지하면서 224x224 안에 fit.
        남는 부분은 ImageNet mean RGB로 center padding.
        """

        image = image.convert("RGB")

        w, h = image.size

        if w <= 0 or h <= 0:
            raise ValueError(
                f"잘못된 이미지 크기: {(w, h)}"
            )

        scale = min(
            IMAGE_SIZE / w,
            IMAGE_SIZE / h,
        )

        new_w = max(
            1,
            min(
                IMAGE_SIZE,
                int(round(w * scale)),
            ),
        )

        new_h = max(
            1,
            min(
                IMAGE_SIZE,
                int(round(h * scale)),
            ),
        )

        resized = image.resize(
            (new_w, new_h),
            Image.Resampling.BICUBIC,
        )

        canvas = Image.new(
            "RGB",
            (IMAGE_SIZE, IMAGE_SIZE),
            IMAGENET_MEAN_RGB,
        )

        left = (
            IMAGE_SIZE - new_w
        ) // 2

        top = (
            IMAGE_SIZE - new_h
        ) // 2

        canvas.paste(
            resized,
            (left, top),
        )

        return canvas

    def _preprocess(
        self,
        image: ImageInput,
        *,
        input_format: str,
    ) -> torch.Tensor:

        pil = self._pil_from_input(
            image,
            input_format=input_format,
        )

        pil = self._aspect_ratio_pad_224(
            pil
        )

        return self.transform(pil)

    @torch.inference_mode()
    def _forward_tensor_batch(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:

        batch = batch.to(
            device=self.device,
            dtype=self.dtype,
            non_blocking=True,
        )

        output = self.model(
            pixel_values=batch
        )

        tokens = output.last_hidden_state

        # token layout:
        # [CLS] [REG x4] [PATCH x256]
        # 최종 descriptor는 CLS만 사용.
        feat = tokens[:, 0].float()

        feat = F.normalize(
            feat,
            p=2,
            dim=-1,
        )

        return feat

    def embed_crops(
        self,
        images: Sequence[ImageInput],
        *,
        input_format: str = "bgr",
    ) -> np.ndarray:
        """
        여러 crop 임베딩.

        Parameters
        ----------
        images:
            OpenCV ndarray / PIL.Image / path 리스트

        input_format:
            ndarray 입력일 때 'bgr' 또는 'rgb'

        Returns
        -------
        np.ndarray:
            shape (N, 1536), float32, L2 normalized
        """

        if len(images) == 0:
            return np.zeros(
                (0, OUTPUT_DIM),
                dtype=np.float32,
            )

        output_batches: List[np.ndarray] = []

        for start in range(
            0,
            len(images),
            self.batch_size,
        ):
            chunk = images[
                start:start + self.batch_size
            ]

            tensors = [
                self._preprocess(
                    image,
                    input_format=input_format,
                )
                for image in chunk
            ]

            batch = torch.stack(
                tensors,
                dim=0,
            )

            feat = self._forward_tensor_batch(
                batch
            )

            output_batches.append(
                feat.cpu().numpy().astype(
                    np.float32,
                    copy=False,
                )
            )

        result = np.concatenate(
            output_batches,
            axis=0,
        )

        if result.shape != (
            len(images),
            OUTPUT_DIM,
        ):
            raise RuntimeError(
                "최종 DINO embedding shape mismatch: "
                f"expected={(len(images), OUTPUT_DIM)}, actual={result.shape}"
            )

        if not np.isfinite(result).all():
            raise RuntimeError(
                "최종 DINO embedding에 NaN/Inf가 포함되어 있습니다."
            )

        return result

    # 기존 코드와 호환용 alias
    def embed_images(
        self,
        images: Sequence[ImageInput],
        *,
        input_format: str = "bgr",
    ) -> np.ndarray:
        return self.embed_crops(
            images,
            input_format=input_format,
        )

    def encode(
        self,
        images: Sequence[ImageInput],
        *,
        input_format: str = "bgr",
    ) -> np.ndarray:
        return self.embed_crops(
            images,
            input_format=input_format,
        )

    def embed_one(
        self,
        image: ImageInput,
        *,
        input_format: str = "bgr",
    ) -> np.ndarray:
        """
        이미지 한 장 -> shape (1536,)
        """
        return self.embed_crops(
            [image],
            input_format=input_format,
        )[0]

    def info(self) -> dict:
        return {
            "model_id": MODEL_ID,
            "architecture": "ViT-G/14",
            "register_tokens": REGISTER_TOKENS,
            "input_size": IMAGE_SIZE,
            "preprocessing": "aspect_ratio_preserve_center_pad",
            "padding_rgb": IMAGENET_MEAN_RGB,
            "feature": "cls",
            "dim": OUTPUT_DIM,
            "l2_normalize": True,
            "device": str(self.device),
            "dtype": str(self.dtype),
        }


# 기존 최종 wrapper 이름도 import 가능하게 alias 제공
DINOv2G14RegistersPadEmbedder = DINOv2FinalEmbedder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Final DINOv2 object embedder smoke test"
    )

    parser.add_argument(
        "image",
        nargs="?",
        help="테스트할 이미지 경로",
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    embedder = DINOv2FinalEmbedder(
        device=args.device,
        batch_size=args.batch_size,
    )

    print("INFO:")
    for k, v in embedder.info().items():
        print(f"  {k}: {v}")

    if args.image:
        vec = embedder.embed_one(
            args.image,
            input_format="rgb",
        )

        print("\nEMBEDDING:")
        print("  shape :", vec.shape)
        print("  dtype :", vec.dtype)
        print(
            "  norm  :",
            float(
                np.linalg.norm(vec)
            ),
        )
        print(
            "  finite:",
            bool(
                np.isfinite(vec).all()
            ),
        )
        print(
            "  first 8:",
            vec[:8],
        )
