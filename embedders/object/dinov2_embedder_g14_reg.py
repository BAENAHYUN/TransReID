"""
DINOv2 ViT-g/14 + Registers 전용 object embedder.

고정 운영 설정
---------------
model    : facebook/dinov2-with-registers-giant
feature  : CLS
dim      : 1536
input    : Resize(shortest=256, bicubic) -> CenterCrop(224)
normalize: ImageNet -> L2
precision: bf16 우선
"""

from __future__ import annotations

from typing import Optional

from embedders.object.dinov2_embedder_registers_v2 import DINOv2Embedder


MODEL_ID = "facebook/dinov2-with-registers-giant"
OUTPUT_DIM = 1536


class DINOv2G14RegistersEmbedder(DINOv2Embedder):
    """프로젝트용 DINOv2 ViT-g/14 + Registers 고정 래퍼."""

    DIM = OUTPUT_DIM

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        batch_size: int = 4,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__(
            model_id=MODEL_ID,
            feature="cls",
            image_size=224,
            resize_mode="center_crop",
            device=device,
            batch_size=batch_size,
            l2_normalize=True,
            fp16=True,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            allow_experimental=False,
        )

        if int(self.DIM) != OUTPUT_DIM:
            raise RuntimeError(
                f"DINOv2 ViT-g/14+Registers 출력 차원 불일치: "
                f"expected={OUTPUT_DIM}, actual={self.DIM}"
            )
