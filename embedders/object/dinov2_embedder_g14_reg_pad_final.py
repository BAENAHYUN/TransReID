#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final DINOv2 Object Embedder
============================

理쒖쥌 �댁쁺 �ㅼ젙
- Model      : facebook/dinov2-with-registers-giant
- Backbone   : ViT-G/14
- Registers  : 4
- Input      : Aspect-Ratio Preserve -> Center Pad 224x224
- Padding    : ImageNet mean RGB ~= (124, 116, 104)
- Normalize  : ImageNet mean/std
- Feature    : CLS token
- Output     : 1536D
- Final norm : L2 Normalize

紐⑹쟻
- RF-DETR �깆쑝濡� �앹꽦�� object crop�� �쒓컖�� �좎궗�� �꾨쿋��
- Qdrant named vector `dinov2` ���μ슜

沅뚯옣 pipeline.yaml
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

媛꾨떒 �뚯뒪��
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

# ImageNet mean�� 0~255 RGB濡� 洹쇱궗.
# Normalize �댄썑 padding �곸뿭�� 嫄곗쓽 0�� �섎룄濡� �ъ슜.
IMAGENET_MEAN_RGB = tuple(int(round(v * 255)) for v in IMAGENET_MEAN)


ImageInput = Union[
    str,
    Path,
    Image.Image,
    np.ndarray,
]


class DINOv2FinalEmbedder:
    """
    DINOv2 Giant + Registers 湲곕컲 理쒖쥌 Object Embedder.

    �낅젰 吏���:
    - np.ndarray (OpenCV BGR/RGB)
    - PIL.Image
    - str / pathlib.Path

    異쒕젰:
    - np.ndarray
    - shape = (N, 1536)
    - dtype = float32
    - 媛� row L2 norm ~= 1.0
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
            raise ValueError("batch_size�� 1 �댁긽�댁뼱�� �⑸땲��.")

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

        # 224 / 14 = 16 -> patch 256媛�
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
                "吏��먰븯吏� �딅뒗 �대�吏� ����: "
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
                f"�대�吏� shape�� �섎せ�섏뿀�듬땲��: {arr.shape}"
            )

        if arr.shape[2] == 4:
            arr = arr[:, :, :3]

        if arr.shape[2] != 3:
            raise ValueError(
                f"3-channel �대�吏�留� 吏��먰빀�덈떎: {arr.shape}"
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
                "input_format�� 'bgr' �먮뒗 'rgb'留� 媛��ν빀�덈떎."
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
        媛앹껜 醫낇슒鍮꾨� �좎��섎㈃�� 224x224 �덉뿉 fit.
        �⑤뒗 遺�遺꾩� ImageNet mean RGB濡� center padding.
        """

        image = image.convert("RGB")

        w, h = image.size

        if w <= 0 or h <= 0:
            raise ValueError(
                f"�섎せ�� �대�吏� �ш린: {(w, h)}"
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
        # 理쒖쥌 descriptor�� CLS留� �ъ슜.
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
        �щ윭 crop �꾨쿋��.

        Parameters
        ----------
        images:
            OpenCV ndarray / PIL.Image / path 由ъ뒪��

        input_format:
            ndarray �낅젰�� �� 'bgr' �먮뒗 'rgb'

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
                "理쒖쥌 DINO embedding shape mismatch: "
                f"expected={(len(images), OUTPUT_DIM)}, actual={result.shape}"
            )

        if not np.isfinite(result).all():
            raise RuntimeError(
                "理쒖쥌 DINO embedding�� NaN/Inf媛� �ы븿�섏뼱 �덉뒿�덈떎."
            )

        return result

    # 湲곗〈 肄붾뱶�� �명솚�� alias
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
        �대�吏� �� �� -> shape (1536,)
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


# 湲곗〈 理쒖쥌 wrapper �대쫫�� import 媛��ν븯寃� alias �쒓났
DINOv2G14RegistersPadEmbedder = DINOv2FinalEmbedder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Final DINOv2 object embedder smoke test"
    )

    parser.add_argument(
        "image",
        nargs="?",
        help="�뚯뒪�명븷 �대�吏� 寃쎈줈",
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