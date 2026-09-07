"""
SOLIDER Embedding Extractor
===========================

파이프라인 위치:
    Person Crop -> [SOLIDER] -> 1024-d Embedding -> Qdrant

SOLIDER(CVPR'23, "Beyond Appearance: a Semantic Controllable Self-Supervised
Learning Framework for Human-Centric Visual Tasks")는 LUPerson 대규모 보행자
이미지로 자기지도 학습한 Swin Transformer 백본이다.

IRRA / SigLIP2 와 달리 텍스트 정렬이 없고,
오로지 사람 identity 표현을 생성한다.

따라서 embed_text 는 구현하지 않는다.


지원 체크포인트
--------------

1) SOLIDER 사전학습 backbone
   swin_base.pth

   - LUPerson self-supervised pretraining
   - zero-shot / cross-domain 평가에 적합
   - BNNeck 없음
   - neck_feat="before" 사용


2) SOLIDER-REID fine-tuned checkpoint

   - Market1501 / MSMT17 등으로 fine-tuning
   - state_dict key에 "base." prefix 존재
   - 필요시 BNNeck 사용 가능

체크포인트 종류는 자동 판별한다.


주의
----

* SOLIDER의 semantic_weight 기본값과 ReID 설정이 다를 수 있다.
  ReID에서는 일반적으로 semantic_weight=0.2 를 사용한다.

* SOLIDER 입력 normalization:
      mean = (0.5, 0.5, 0.5)
      std  = (0.5, 0.5, 0.5)

* Swin forward 반환:
      (global_feat, feature_maps)

  global_feat은 이미 GAP + flatten 된 (B, D) feature다.

* 입력 크기:
      384 x 128

* 테스트 전처리는 SOLIDER-REID 평가 경로에 맞춰
  torchvision Resize 기본 interpolation(BILINEAR)을 사용한다.
"""

from __future__ import annotations
import pickle
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ..base import BaseEmbedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SOLIDER normalization
# ---------------------------------------------------------------------------

SOLIDER_MEAN = (0.5, 0.5, 0.5)
SOLIDER_STD = (0.5, 0.5, 0.5)


# ---------------------------------------------------------------------------
# Backbone output dimensions
# Swin 마지막 stage channel = embed_dims * 2^3
# ---------------------------------------------------------------------------

BACKBONE_DIMS = {
    "swin_tiny": 768,
    "swin_small": 768,
    "swin_base": 1024,
}


# ---------------------------------------------------------------------------
# SOLIDER factory names
# ---------------------------------------------------------------------------

_FACTORY = {
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "swin_small": "swin_small_patch4_window7_224",
    "swin_base": "swin_base_patch4_window7_224",
}


class SoliderEmbedder(BaseEmbedder):
    """
    SOLIDER 사람 identity embedding extractor.

    출력:
        (N, D) float32

    기본 설정:
        swin_base -> D=1024
        img_size=(384, 128)
        semantic_weight=0.2
        L2 normalize=True
    """

    def __init__(
        self,
        solider_root: str | Path,
        ckpt_path: str | Path,
        backbone: str = "swin_base",
        semantic_weight: float = 0.2,
        img_size: Tuple[int, int] = (384, 128),
        neck_feat: str = "before",
        device: Optional[str] = None,
        batch_size: int = 32,
        l2_normalize: bool = True,
    ) -> None:

        if backbone not in BACKBONE_DIMS:
            raise ValueError(
                f"backbone 은 {sorted(BACKBONE_DIMS)} 중 하나여야 합니다 "
                f"(받은 값: {backbone})"
            )

        if neck_feat not in ("before", "after"):
            raise ValueError(
                "neck_feat 은 'before' 또는 'after' 여야 합니다."
            )

        # BaseEmbedder가 DIM > 0 을 요구하므로 먼저 지정
        self.DIM = BACKBONE_DIMS[backbone]

        super().__init__(
            device=device,
            batch_size=batch_size,
            l2_normalize=l2_normalize,
        )

        self.backbone = backbone
        self.semantic_weight = float(semantic_weight)
        self.img_size = tuple(img_size)
        self.neck_feat = neck_feat

        self._inject_path(solider_root)

        self.model, self.bottleneck = self._build(
            Path(ckpt_path).expanduser()
        )

        self.transform = self._build_transform()

        # semantic_weight tensor cache
        self._sw_cache: dict[int, torch.Tensor] = {}

        logger.info(
            "SoliderEmbedder ready | "
            "%s dim=%d device=%s img=%s sw=%.2f neck=%s",
            backbone,
            self.DIM,
            self.device,
            self.img_size,
            self.semantic_weight,
            self.neck_feat,
        )

    # =======================================================================
    # SOLIDER import / repository setup
    # =======================================================================

    @staticmethod
    def _stub_unused_imports() -> None:
        """
        SOLIDER swin_transformer.py가 mmcv.runner를 import하지만
        실제 inference path에서는 사용하지 않는 경우가 있다.

        mmcv-full 설치는 torch/CUDA 버전에 매우 민감하므로,
        mmcv가 없을 때만 최소 dummy module을 삽입한다.

        중요:
        cv2는 절대 stub하지 않는다.
        다른 RF-DETR/OpenCV 코드에 전역 부작용을 줄 수 있기 때문이다.
        """

        import types

        try:
            import mmcv  # noqa: F401

        except ImportError:

            mmcv_module = types.ModuleType("mmcv")
            sys.modules["mmcv"] = mmcv_module

            logger.debug(
                "mmcv 미설치 -> SOLIDER import용 dummy module 생성"
            )

        if "mmcv.runner" not in sys.modules:

            try:
                __import__("mmcv.runner")

            except ImportError:

                runner = types.ModuleType("mmcv.runner")

                # SOLIDER inference에서는 사용하지 않음
                runner.load_checkpoint = None

                sys.modules["mmcv.runner"] = runner

                setattr(
                    sys.modules["mmcv"],
                    "runner",
                    runner,
                )

                logger.debug(
                    "mmcv.runner 미설치 -> dummy runner 생성"
                )

    @classmethod
    def _inject_path(
        cls,
        solider_root: str | Path,
    ) -> None:

        root = Path(
            solider_root
        ).expanduser().resolve()

        if not (root / "swin_transformer.py").exists():
            raise FileNotFoundError(
                f"SOLIDER 레포를 찾을 수 없습니다: {root}\n"
                f"  git clone https://github.com/tinyvision/SOLIDER.git {root}"
            )

        cls._stub_unused_imports()

        if str(root) not in sys.path:
            sys.path.insert(
                0,
                str(root),
            )

    # =======================================================================
    # Model build
    # =======================================================================

    def _build(
        self,
        ckpt_path: Path,
    ):

        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"SOLIDER 체크포인트가 없습니다: {ckpt_path}"
            )

        import swin_transformer  # type: ignore

        factory = getattr(
            swin_transformer,
            _FACTORY[self.backbone],
        )

        model = factory(
            img_size=self.img_size,
            drop_path_rate=0.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            convert_weights=False,
            semantic_weight=self.semantic_weight,
        )

        # ---------------------------------------------------------------
        # Output dimension 확인
        # ---------------------------------------------------------------

        num_features = model.num_features

        if isinstance(
            num_features,
            (list, tuple),
        ):
            actual = int(
                num_features[-1]
            )
        else:
            actual = int(
                num_features
            )

        if actual != self.DIM:
            raise RuntimeError(
                f"차원 불일치: {self.backbone} 의 실제 출력은 "
                f"{actual} 인데 BACKBONE_DIMS 는 {self.DIM} 입니다."
            )

        # ---------------------------------------------------------------
        # checkpoint 종류 자동 판별
        # ---------------------------------------------------------------

        kind = self._detect_ckpt_kind(
            ckpt_path
        )

        bottleneck = None

        if kind == "reid":

            logger.info(
                "SOLIDER-REID 파인튜닝 체크포인트로 판별"
            )

            bottleneck = self._load_reid(
                model,
                ckpt_path,
            )

        else:

            logger.info(
                "SOLIDER 사전학습 백본 체크포인트로 판별"
            )

            # init_weights가 teacher/state_dict/module prefix 및
            # relative_position_bias_table interpolation 처리
            model.init_weights(
                str(ckpt_path)
            )

            if self.neck_feat == "after":
                raise ValueError(
                    "사전학습 backbone 체크포인트에는 BNNeck 이 없습니다.\n"
                    "neck_feat='before' 를 사용하거나 "
                    "SOLIDER-REID 체크포인트를 사용하세요."
                )

        model.to(
            self.device
        ).eval()

        for p in model.parameters():
            p.requires_grad_(
                False
            )

        if bottleneck is not None:

            bottleneck.to(
                self.device
            ).eval()

            for p in bottleneck.parameters():
                p.requires_grad_(
                    False
                )

        return (
            model,
            bottleneck,
        )

    # =======================================================================
    # Checkpoint utilities
    # =======================================================================

    @staticmethod
    def _load_ckpt(path: Path) -> dict:
        """
        SOLIDER checkpoint를 state_dict 형태로 로드한다.
    
        우선 weights_only=True를 사용한다.
        공식/신뢰 가능한 checkpoint가 NumPy 객체 등을 포함해서
        weights_only 로딩에 실패한 경우에만 weights_only=False로 fallback한다.
        """
    
        try:
            ckpt = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
    
        except TypeError:
            # 구버전 PyTorch: weights_only 인자 미지원
            ckpt = torch.load(
                path,
                map_location="cpu",
            )
    
        except pickle.UnpicklingError:
            # SOLIDER 공식 checkpoint처럼
            # weights_only=True가 NumPy 객체 때문에 실패하는 경우
            logger.warning(
                "weights_only=True 로드 실패 -> "
                "신뢰 가능한 SOLIDER checkpoint로 간주하고 "
                "weights_only=False로 다시 로드합니다."
            )
    
            ckpt = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
    
        # checkpoint wrapper 제거
        for key in (
            "teacher",
            "state_dict",
            "model",
        ):
            if (
                isinstance(ckpt, dict)
                and key in ckpt
                and isinstance(ckpt[key], dict)
            ):
                ckpt = ckpt[key]
                break
    
        if not isinstance(ckpt, dict):
            raise TypeError(
                "SOLIDER checkpoint의 최종 state_dict가 "
                f"dict가 아닙니다: {type(ckpt)}"
            )
    
        # module. prefix 제거
        cleaned = {}
    
        for k, v in ckpt.items():
            if k.startswith("module."):
                k = k[len("module."):]
    
            cleaned[k] = v
    
        return cleaned
    
    @classmethod
    def _detect_ckpt_kind(
        cls,
        path: Path,
    ) -> str:
        """
        SOLIDER-REID fine-tuned checkpoint인지
        backbone pretraining checkpoint인지 판별한다.

        SOLIDER-REID:
            base.xxx

        backbone:
            layers.xxx
            patch_embed.xxx
            ...
        """

        sd = cls._load_ckpt(
            path
        )

        if any(
            k.startswith("base.")
            for k in sd
        ):
            return "reid"

        return "backbone"

    # =======================================================================
    # SOLIDER-REID loading
    # =======================================================================

    def _load_reid(
        self,
        model,
        ckpt_path: Path,
    ):
        """
        SOLIDER-REID checkpoint에서

            base.*
            bottleneck.*

        을 분리해서 로드한다.

        classifier는 dataset identity 수에 종속되므로 사용하지 않는다.
        """

        import torch.nn as nn

        sd = self._load_ckpt(
            ckpt_path
        )

        backbone_sd = {
            k[len("base."):]: v
            for k, v in sd.items()
            if k.startswith(
                "base."
            )
        }

        if not backbone_sd:
            raise RuntimeError(
                "SOLIDER-REID checkpoint로 판별했지만 "
                "'base.' 가중치가 없습니다."
            )

        missing, unexpected = model.load_state_dict(
            backbone_sd,
            strict=False,
        )

        loaded = (
            len(backbone_sd)
            - len(unexpected)
        )

        if loaded < len(
            model.state_dict()
        ) * 0.5:
            raise RuntimeError(
                f"백본 가중치가 거의 로드되지 않았습니다 ({loaded}개).\n"
                f"backbone='{self.backbone}' 이 checkpoint와 "
                f"맞는지 확인하세요.\n"
                f"unexpected 예시: {unexpected[:3]}"
            )

        if unexpected:
            logger.debug(
                "unexpected keys: %s",
                unexpected[:5],
            )

        if missing:
            logger.debug(
                "missing keys: %s",
                missing[:5],
            )

        logger.info(
            "SOLIDER backbone weight %d개 로드",
            loaded,
        )

        # ---------------------------------------------------------------
        # BNNeck
        # ---------------------------------------------------------------

        bottleneck = None

        if self.neck_feat == "after":

            w = sd.get(
                "bottleneck.weight"
            )

            if w is None:
                raise RuntimeError(
                    "neck_feat='after' 인데 checkpoint에 "
                    "bottleneck weight가 없습니다."
                )

            bottleneck = nn.BatchNorm1d(
                self.DIM
            )

            bn_sd = {
                k[len("bottleneck."):]: v
                for k, v in sd.items()
                if k.startswith(
                    "bottleneck."
                )
            }

            bottleneck.load_state_dict(
                bn_sd,
                strict=False,
            )

            logger.info(
                "SOLIDER BNNeck 로드 "
                "(neck_feat='after')"
            )

        return bottleneck

    # =======================================================================
    # Transform
    # =======================================================================

    def _build_transform(
        self,
    ):

        import torchvision.transforms as T

        h, w = self.img_size

        return T.Compose(
            [
                # SOLIDER-REID test preprocessing과 동일:
                # torchvision Resize 기본 interpolation = BILINEAR
                T.Resize(
                    (h, w)
                ),

                T.ToTensor(),

                T.Normalize(
                    mean=SOLIDER_MEAN,
                    std=SOLIDER_STD,
                ),
            ]
        )

    # =======================================================================
    # semantic weight
    # =======================================================================

    def _semantic_weight_tensor(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        """
        semantic_weight tensor 생성.

        shape:
            (B, 2)

        값:
            [w, 1-w]

        SOLIDER 원본 내부 .cuda() hardcoding을 피하기 위해
        항상 명시적으로 forward에 넘긴다.
        """

        cached = self._sw_cache.get(
            batch_size
        )

        if cached is not None:
            return cached

        w = (
            torch.ones(
                batch_size,
                1,
                device=self.device,
            )
            * self.semantic_weight
        )

        sw = torch.cat(
            [
                w,
                1.0 - w,
            ],
            dim=-1,
        )

        self._sw_cache[
            batch_size
        ] = sw

        return sw

    # =======================================================================
    # Encoding
    # =======================================================================

    @torch.inference_mode()
    def _encode(
        self,
        images: List[Image.Image],
    ) -> np.ndarray:
        """
        PIL Image list -> SOLIDER embedding

        반환:
            (B, DIM) float32 numpy
        """

        batch = torch.stack(
            [
                self.transform(
                    im.convert("RGB")
                )
                for im in images
            ]
        )

        batch = batch.to(
            self.device,
            non_blocking=True,
        )

        sw = self._semantic_weight_tensor(
            batch.shape[0]
        )

        out = self.model(
            batch,
            semantic_weight=sw,
        )

        # SOLIDER forward:
        #
        #   x = avgpool(outs[-1])
        #   x = flatten(x, 1)
        #   return x, outs
        #
        # 따라서 out[0]은 이미 pooling된 global feature
        if isinstance(
            out,
            (tuple, list),
        ):
            global_feat = out[0]

        else:
            global_feat = out

        if global_feat.ndim != 2:
            raise RuntimeError(
                "SOLIDER global feature shape가 예상과 다릅니다: "
                f"{tuple(global_feat.shape)}"
            )

        if global_feat.shape[1] != self.DIM:
            raise RuntimeError(
                "SOLIDER embedding dimension 불일치: "
                f"expected={self.DIM}, "
                f"actual={global_feat.shape[1]}"
            )

        if self.bottleneck is not None:
            global_feat = self.bottleneck(
                global_feat
            )

        return (
            global_feat
            .float()
            .cpu()
            .numpy()
        )

    # -----------------------------------------------------------------------
    # Text encoder 없음
    #
    # SOLIDER는 image-only human representation 모델이다.
    # embed_text를 의도적으로 구현하지 않는다.
    # -----------------------------------------------------------------------


# ===========================================================================
# Smoke Test
# ===========================================================================

if __name__ == "__main__":

    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(levelname)s "
            "%(name)s: "
            "%(message)s"
        ),
    )

    ap = argparse.ArgumentParser(
        description=(
            "SOLIDER embedding smoke test"
        )
    )

    ap.add_argument(
        "--solider-root",
        required=True,
    )

    ap.add_argument(
        "--ckpt",
        required=True,
    )

    ap.add_argument(
        "--backbone",
        default="swin_base",
        choices=sorted(
            BACKBONE_DIMS
        ),
    )

    ap.add_argument(
        "--semantic-weight",
        type=float,
        default=0.2,
    )

    ap.add_argument(
        "--neck-feat",
        default="before",
        choices=[
            "before",
            "after",
        ],
    )

    ap.add_argument(
        "--device",
        default=None,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    ap.add_argument(
        "--images",
        nargs="+",
        required=True,
    )

    args = ap.parse_args()

    emb = SoliderEmbedder(
        solider_root=args.solider_root,
        ckpt_path=args.ckpt,
        backbone=args.backbone,
        semantic_weight=args.semantic_weight,
        neck_feat=args.neck_feat,
        device=args.device,
        batch_size=args.batch_size,
    )

    # BaseEmbedder가 경로 기반 API를 제공하면 우선 사용
    if hasattr(
        emb,
        "embed_image_paths",
    ):
        vecs = emb.embed_image_paths(
            args.images
        )

    else:
        # 기존 BaseEmbedder가 embed_crops에서 path도 지원하는 경우
        vecs = emb.embed_crops(
            args.images
        )

    print(
        f"\nembeddings: {vecs.shape}"
    )

    print(
        "first norm="
        f"{np.linalg.norm(vecs[0]):.4f}"
    )

    # -----------------------------------------------------------------------
    # Cosine similarity
    # L2 normalize된 embedding이면 dot product == cosine similarity
    # -----------------------------------------------------------------------

    print(
        "\n쌍별 코사인 유사도"
    )

    sims = vecs @ vecs.T

    names = [
        os.path.basename(p)
        for p in args.images
    ]

    print(
        "        "
        + "  ".join(
            f"{n[:8]:>8}"
            for n in names
        )
    )

    for name, row in zip(
        names,
        sims,
    ):
        print(
            f"{name[:8]:>8}"
            + "  ".join(
                f"{v:8.4f}"
                for v in row
            )
        )
