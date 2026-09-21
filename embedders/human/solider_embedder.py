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

SOLIDER-REID fine-tuned checkpoint만 지원한다.

예:
    solider_market_swin_base.pth

공식 SOLIDER-REID 모델 계약에 맞춰 state_dict의

    base.*
    bottleneck.*
    classifier.*

구조를 기대한다.

이 adapter는 사람 임베딩 추출만 담당하므로:

    base.*        -> SOLIDER Swin backbone에 로드
    bottleneck.*  -> neck_feat="after"일 때만 로드
    classifier.*  -> 사용하지 않음

SOLIDER self-supervised pretraining checkpoint의 bare backbone이나
teacher checkpoint를 자동 판별/변환하지 않는다. 그런 checkpoint는
공식 SOLIDER-REID 변환/학습 흐름을 거친 뒤 사용해야 한다.


주의
----

* SOLIDER의 semantic_weight 기본값과 ReID 설정이 다를 수 있다.
  ReID에서는 일반적으로 semantic_weight=0.2 를 사용한다.
  semantic_weight 는 논문의 lambda에 해당하며 0.0 <= lambda <= 1.0 범위만 허용한다.

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

안전성 검증
----------

* 공식 SOLIDER-REID checkpoint만 사용한다는 운영 전제에 따라
  backbone state_dict는 현재 모델과 key/shape가 100% 일치해야 한다.

* Semantic Controller(semantic_embed_w.*, semantic_embed_b.*)는
  모델과 checkpoint 양쪽에 모두 존재해야 하며, 로드 후 값까지 일치하는지 확인한다.

* semantic_weight tensor cache는 (batch_size, semantic_weight)를 key로 사용한다.
  운영 중 semantic_weight가 변경되더라도 이전 cache가 잘못 재사용되지 않는다.
"""

from __future__ import annotations
import importlib.util
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

        semantic_weight = float(semantic_weight)

        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError(
                "semantic_weight 는 0.0 이상 1.0 이하여야 합니다 "
                f"(받은 값: {semantic_weight})"
            )

        # BaseEmbedder가 DIM > 0 을 요구하므로 먼저 지정
        self.DIM = BACKBONE_DIMS[backbone]

        super().__init__(
            device=device,
            batch_size=batch_size,
            l2_normalize=l2_normalize,
        )

        self.backbone = backbone
        self.semantic_weight = semantic_weight
        self.img_size = tuple(img_size)
        self.neck_feat = neck_feat

        self._swin_transformer = self._load_swin_module(
            solider_root
        )

        self.model, self.bottleneck = self._build(
            Path(ckpt_path).expanduser()
        )

        self.transform = self._build_transform()

        # semantic_weight tensor cache
        # 안전성: batch size뿐 아니라 현재 semantic_weight까지 key에 포함한다.
        self._sw_cache: dict[Tuple[int, float], torch.Tensor] = {}

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
        SOLIDER swin_transformer.py가 mmcv.runner.load_checkpoint를 import하지만
        이 adapter의 ReID inference 경로에서는 init_weights()를 사용하지 않는다.

        mmcv-full 설치는 torch/CUDA 버전에 민감하므로 mmcv.runner가 없을 때만
        import를 통과시키기 위한 최소 module을 제공한다.

        주의:
        cv2 등 다른 모듈은 stub하지 않는다.
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

                def _unused_load_checkpoint(*args, **kwargs):
                    raise RuntimeError(
                        "이 adapter는 SOLIDER init_weights()를 사용하지 않습니다. "
                        "SOLIDER-REID fine-tuned checkpoint를 사용하세요."
                    )

                runner.load_checkpoint = _unused_load_checkpoint
                sys.modules["mmcv.runner"] = runner
                setattr(sys.modules["mmcv"], "runner", runner)

                logger.debug(
                    "mmcv.runner 미설치 -> SOLIDER import용 dummy runner 생성"
                )

    @classmethod
    def _load_swin_module(
        cls,
        solider_root: str | Path,
    ):
        """
        지정된 SOLIDER root의 swin_transformer.py를 정확히 로드한다.

        sys.path에 의존하는 ``import swin_transformer``를 사용하지 않아
        TransReID 등 다른 프로젝트의 동명 모듈과 충돌하지 않는다.
        """

        root = Path(solider_root).expanduser().resolve()
        module_path = root / "swin_transformer.py"

        if not module_path.is_file():
            raise FileNotFoundError(
                f"SOLIDER 레포를 찾을 수 없습니다: {root}\n"
                f"필요 파일: {module_path}"
            )

        cls._stub_unused_imports()

        module_name = "_transreid_solider_swin_transformer"
        existing = sys.modules.get(module_name)

        if existing is not None:
            existing_file = Path(
                getattr(existing, "__file__", "")
            ).resolve()

            if existing_file == module_path:
                return existing

            # 다른 SOLIDER root로 다시 초기화하는 경우에는 정확한 파일을 재로드한다.
            del sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(
            module_name,
            module_path,
        )

        if spec is None or spec.loader is None:
            raise ImportError(
                f"SOLIDER swin_transformer 모듈을 로드할 수 없습니다: {module_path}"
            )

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

        return module

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

        factory_name = _FACTORY[self.backbone]

        try:
            factory = getattr(
                self._swin_transformer,
                factory_name,
            )
        except AttributeError as exc:
            raise RuntimeError(
                f"SOLIDER backbone factory를 찾을 수 없습니다: {factory_name}"
            ) from exc

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
        # 공식 SOLIDER-REID checkpoint 계약으로 로드
        # ---------------------------------------------------------------

        bottleneck = self._load_reid(
            model,
            ckpt_path,
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
        SOLIDER-REID fine-tuned checkpoint를 state_dict로 로드한다.

        허용:
            - flat state_dict
            - {"state_dict": state_dict}
            - DDP의 module.* prefix

        허용하지 않음:
            - SOLIDER pretraining teacher checkpoint의 자동 변환
            - bare backbone state_dict의 자동 추측
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

        if (
            isinstance(ckpt, dict)
            and "state_dict" in ckpt
            and isinstance(ckpt["state_dict"], dict)
        ):
            ckpt = ckpt["state_dict"]

        if not isinstance(ckpt, dict):
            raise TypeError(
                "SOLIDER-REID checkpoint의 state_dict가 dict가 아닙니다: "
                f"{type(ckpt)}"
            )

        cleaned = {}

        for k, v in ckpt.items():
            if not isinstance(k, str):
                raise TypeError(
                    "SOLIDER-REID checkpoint key가 문자열이 아닙니다: "
                    f"{type(k)}"
                )

            if k.startswith("module."):
                k = k[len("module."):]

            cleaned[k] = v

        if not any(
            k.startswith("base.")
            for k in cleaned
        ):
            sample_keys = list(cleaned)[:5]
            raise RuntimeError(
                "지원하지 않는 SOLIDER checkpoint 형식입니다.\n"
                "이 adapter는 SOLIDER-REID fine-tuned checkpoint의 "
                "'base.*' 구조를 요구합니다.\n"
                "SOLIDER pretraining teacher/bare-backbone checkpoint는 "
                "자동 변환하지 않습니다.\n"
                f"checkpoint key 예시: {sample_keys}"
            )

        return cleaned

    # =======================================================================
    # SOLIDER-REID loading
    # =======================================================================

    def _load_reid(
        self,
        model,
        ckpt_path: Path,
    ):
        """
        공식 SOLIDER-REID fine-tuned checkpoint 계약을 adapter에 연결한다.

        checkpoint:
            base.*         -> SOLIDER Swin backbone
            bottleneck.*   -> BNNeck (neck_feat="after"일 때 사용)
            classifier.*   -> dataset identity 종속이므로 무시
        """

        import torch.nn as nn

        sd = self._load_ckpt(
            ckpt_path
        )

        backbone_sd = {
            k[len("base."):]: v
            for k, v in sd.items()
            if k.startswith("base.")
        }

        model_sd = model.state_dict()

        # ---------------------------------------------------------------
        # 안전성 강화 1: backbone state_dict 100% 일치 강제
        #
        # 이 adapter는 공식 SOLIDER-REID fine-tuned checkpoint만 지원한다.
        # 따라서 "90% 이상이면 허용"하지 않고, 현재 backbone과 checkpoint의
        # key/shape가 완전히 일치할 때만 로드한다.
        # ---------------------------------------------------------------

        non_tensor_keys = sorted(
            key
            for key, value in backbone_sd.items()
            if not torch.is_tensor(value)
        )

        if non_tensor_keys:
            raise RuntimeError(
                "SOLIDER-REID backbone checkpoint에 tensor가 아닌 값이 있습니다.\n"
                f"key 예시: {non_tensor_keys[:5]}"
            )

        model_keys = set(model_sd)
        checkpoint_keys = set(backbone_sd)

        missing = sorted(
            model_keys - checkpoint_keys
        )
        unexpected = sorted(
            checkpoint_keys - model_keys
        )

        shape_mismatch = []
        for key in sorted(model_keys & checkpoint_keys):
            if model_sd[key].shape != backbone_sd[key].shape:
                shape_mismatch.append(
                    (
                        key,
                        tuple(backbone_sd[key].shape),
                        tuple(model_sd[key].shape),
                    )
                )

        if missing or unexpected or shape_mismatch:
            shape_example = (
                shape_mismatch[:3]
                if shape_mismatch
                else []
            )
            raise RuntimeError(
                "SOLIDER-REID backbone checkpoint가 현재 backbone과 "
                "100% 일치하지 않습니다.\n"
                f"missing={len(missing)} 예시={missing[:5]}\n"
                f"unexpected={len(unexpected)} 예시={unexpected[:5]}\n"
                f"shape_mismatch={len(shape_mismatch)} 예시={shape_example}\n"
                f"pipeline.yaml의 backbone='{self.backbone}'과 checkpoint를 "
                "확인하세요."
            )

        # ---------------------------------------------------------------
        # 안전성 강화 2: Semantic Controller 필수 검사
        #
        # 공식 SOLIDER Swin backbone의 controller parameter:
        #   semantic_embed_w.*
        #   semantic_embed_b.*
        #
        # 모델에는 있는데 checkpoint에 없거나, 반대로 checkpoint에만 있는
        # 경우를 모두 즉시 차단한다.
        # ---------------------------------------------------------------

        semantic_prefixes = (
            "semantic_embed_w.",
            "semantic_embed_b.",
        )

        semantic_model_keys = sorted(
            key
            for key in model_sd
            if key.startswith(semantic_prefixes)
        )
        semantic_checkpoint_keys = sorted(
            key
            for key in backbone_sd
            if key.startswith(semantic_prefixes)
        )

        if not semantic_model_keys:
            raise RuntimeError(
                "현재 SOLIDER backbone에서 Semantic Controller parameter를 "
                "찾을 수 없습니다.\n"
                "필요 key prefix: semantic_embed_w.*, semantic_embed_b.*"
            )

        if semantic_model_keys != semantic_checkpoint_keys:
            missing_semantic = sorted(
                set(semantic_model_keys) - set(semantic_checkpoint_keys)
            )
            unexpected_semantic = sorted(
                set(semantic_checkpoint_keys) - set(semantic_model_keys)
            )
            raise RuntimeError(
                "SOLIDER Semantic Controller checkpoint가 모델과 일치하지 "
                "않습니다.\n"
                f"missing_semantic={missing_semantic[:8]}\n"
                f"unexpected_semantic={unexpected_semantic[:8]}"
            )

        # 전체 backbone key/shape가 완전히 일치했으므로 strict=True로 로드한다.
        model.load_state_dict(
            backbone_sd,
            strict=True,
        )

        # 로드 후 controller tensor가 checkpoint와 정확히 같은지 한 번 더 확인한다.
        loaded_model_sd = model.state_dict()
        semantic_value_mismatch = [
            key
            for key in semantic_model_keys
            if not torch.equal(
                loaded_model_sd[key].detach().cpu(),
                backbone_sd[key].detach().cpu(),
            )
        ]

        if semantic_value_mismatch:
            raise RuntimeError(
                "SOLIDER Semantic Controller가 checkpoint 값과 정확히 "
                "로드되지 않았습니다.\n"
                f"불일치 key 예시: {semantic_value_mismatch[:8]}"
            )

        loaded = len(model_sd)
        total = len(model_sd)

        logger.info(
            "SOLIDER-REID backbone strict 로드 | %d/%d (100.0%%) | "
            "semantic controller %d keys verified",
            loaded,
            total,
            len(semantic_model_keys),
        )

        # ---------------------------------------------------------------
        # BNNeck
        # 공식 inference 의미:
        #   before -> global_feat
        #   after  -> bottleneck(global_feat)
        # ---------------------------------------------------------------

        if self.neck_feat == "before":
            return None

        bn_sd = {
            k[len("bottleneck."):]: v
            for k, v in sd.items()
            if k.startswith("bottleneck.")
        }

        required_bn_keys = {
            "weight",
            "bias",
            "running_mean",
            "running_var",
        }
        missing_bn = sorted(
            required_bn_keys - set(bn_sd)
        )

        if missing_bn:
            raise RuntimeError(
                "neck_feat='after'인데 SOLIDER-REID checkpoint의 "
                "BNNeck가 불완전합니다.\n"
                f"누락 key: {missing_bn}"
            )

        bottleneck = nn.BatchNorm1d(
            self.DIM
        )
        # 공식 ReID 구현의 BNNeck 설정과 동일한 의미
        bottleneck.bias.requires_grad_(False)

        missing_bn_load, unexpected_bn = bottleneck.load_state_dict(
            bn_sd,
            strict=False,
        )

        allowed_missing = {
            "num_batches_tracked"
        }
        bad_missing = [
            k
            for k in missing_bn_load
            if k not in allowed_missing
        ]

        if bad_missing or unexpected_bn:
            raise RuntimeError(
                "SOLIDER-REID BNNeck checkpoint가 모델과 일치하지 않습니다.\n"
                f"missing={bad_missing}\n"
                f"unexpected={unexpected_bn}"
            )

        logger.info(
            "SOLIDER-REID BNNeck 로드 (neck_feat='after')"
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

        # 안전성 강화 3:
        # batch_size만 key로 쓰면 런타임에서 semantic_weight가 변경됐을 때
        # 이전 weight tensor가 재사용될 수 있다. 현재 weight까지 cache key에
        # 포함하여 잘못된 재사용을 막는다.
        cache_key = (
            int(batch_size),
            float(self.semantic_weight),
        )

        cached = self._sw_cache.get(
            cache_key
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
            cache_key
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
#
# 프로젝트 루트에서 module mode로 실행한다.
#   python -m embedders.human.solider_embedder ...
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
        help=(
            "SOLIDER semantic controller weight "
            "(0.0~1.0, Re-ID 기본 권장값: 0.2)"
        ),
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
