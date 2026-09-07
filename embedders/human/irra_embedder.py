"""
IRRA Embedding Extractor
========================

파이프라인 위치:
    Person Crop -> [IRRA] -> 512-d Embedding -> Qdrant

IRRA(CVPR'23, "Cross-Modal Implicit Relation Reasoning and Aligning for
Text-to-Image Person Retrieval")는 CLIP ViT-B/16 백본을 text-to-image person
ReID 로 파인튜닝한 모델이다. 사람 전용이므로 scope='person' 으로 등록한다.

프로젝트 고정 계약
------------------
IRRA 내부 구현과 Qdrant 계약을 분리한다.

    inference precision : FP32 고정
    output DIM          : 512
    output dtype        : float32
    finite              : NaN / Inf 금지
    normalization       : BaseEmbedder 의 기존 L2 normalize 계약 유지
    ordering            : 입력 crop 순서와 출력 vector 순서 1:1 유지

Qdrant / Router 는 IRRA 내부 모델 구조를 알 필요가 없다. 이 파일은 위 출력 계약을
지키는 512-d vector만 반환한다.

특징:
  - 이미지 임베딩 512-d, 텍스트 임베딩 512-d 가 같은 공간에 있다.
  - 입력 해상도 384x128, CLIP normalization.
  - 추론에는 IRR(cross-modal interaction) / MLM loss 계산을 사용하지 않는다.

BaseEmbedder 를 상속한다
-----------------------
경로/PIL/numpy 입력 처리, 알파 합성, 배치 분할, L2 정규화, 차원 검증, NaN 검사는
BaseEmbedder 가 담당한다. 이 파일은 `_encode(pil_images)` 를 구현하고 텍스트 검색용
`embed_text()` 를 추가한다.

전처리
------
IRRA 공식 `datasets/build.py` 의 test transform 은 interpolation 을 지정하지 않는다.
즉 torchvision 기본값인 BILINEAR 이다. BICUBIC 으로 바꾸지 않는다.

외부 IRRA repo import
---------------------
IRRA 원본은 `model`, `utils` 같은 일반적인 top-level package 이름을 사용한다.
프로젝트 전체 sys.path / sys.modules 를 영구 오염시키지 않도록 import 순간에만
IRRA root 를 우선하고, 기존 `model*`, `utils*` 모듈을 임시 보관한 뒤 복원한다.
이미 import 된 IRRA class/function 객체는 자신이 로드된 module globals 를 참조하므로
모델 생성과 추론에는 사용할 수 있다.

필요한 것:
  1) IRRA 원본 repo
  2) fine-tuned checkpoint: best.pth
  3) configs.yaml
  4) 선택: OpenAI CLIP ViT-B/16 local .pt
     - 오프라인/재현 가능한 구축을 원하면 clip_pretrained 로 지정한다.

사용:
    emb = IRRAEmbedder(
        irra_root="./IRRA",
        ckpt_path="./weights/IRRA/cuhk_pedes/best.pth",
        config_file="./weights/IRRA/cuhk_pedes/configs.yaml",
        clip_pretrained="./weights/IRRA/ViT-B-16.pt",
    )

    vecs = emb.embed_crops(crops, input_format="rgb")  # (N, 512) float32
    qvec = emb.embed_text("a man in a red shirt")      # (1, 512) float32
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

from ..base import BaseEmbedder, l2_normalize

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# CLIP normalization
# IRRA datasets/build.py 와 동일해야 한다.
# --------------------------------------------------------------------------- #
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# --------------------------------------------------------------------------- #
# IRRA external repo import isolation
#
# IRRA 원본은 다음과 같은 generic top-level import 를 사용한다.
#
#     from model ...
#     from utils ...
#
# 같은 interpreter 에 다른 외부 repo 가 있으면 이름 충돌 가능성이 있으므로
# IRRA symbol 을 가져오는 짧은 구간에서만 namespace 를 임시 교체한다.
# --------------------------------------------------------------------------- #
_IRRA_IMPORT_LOCK = threading.RLock()
_IRRA_GENERIC_NAMESPACES = ("model", "utils")


def _is_irra_generic_module(name: str) -> bool:
    return any(
        name == prefix or name.startswith(prefix + ".")
        for prefix in _IRRA_GENERIC_NAMESPACES
    )


@contextmanager
def _isolated_irra_import_scope(root: Path) -> Iterator[None]:
    """IRRA의 generic `model` / `utils` import를 일시적으로 격리한다.

    주의:
    - sys.path / sys.modules 는 process 전역 상태이므로 import 구간을 lock 으로 감싼다.
    - 기존 generic module 은 삭제하지 않고 보관했다가 정확히 복원한다.
    - 이 context 밖에는 IRRA root 를 sys.path 에 남기지 않는다.
    """

    root = root.resolve()
    root_str = str(root)

    with _IRRA_IMPORT_LOCK:
        old_sys_path = list(sys.path)
        saved_modules = {
            name: module
            for name, module in list(sys.modules.items())
            if _is_irra_generic_module(name)
        }

        # IRRA import 전에 기존 generic namespace 를 잠시 치운다.
        for name in saved_modules:
            sys.modules.pop(name, None)

        # IRRA root 를 import 우선순위 맨 앞에 둔다.
        sys.path[:] = [
            root_str,
            *[
                p
                for p in old_sys_path
                if os.path.abspath(os.path.expanduser(str(p or os.curdir)))
                != root_str
            ],
        ]
        importlib.invalidate_caches()

        try:
            yield
        finally:
            # 이 scope 안에서 새로 생긴 IRRA generic namespace 를 제거한다.
            for name in list(sys.modules):
                if _is_irra_generic_module(name):
                    sys.modules.pop(name, None)

            # import 전 상태를 복원한다.
            sys.modules.update(saved_modules)
            sys.path[:] = old_sys_path
            importlib.invalidate_caches()


# --------------------------------------------------------------------------- #
# 내부 설정 컨테이너
# --------------------------------------------------------------------------- #
@dataclass
class IRRAConfig:
    """IRRAEmbedder 설정.

    `amp` 는 과거 호출부와의 호환을 위해 필드는 유지하지만 프로젝트 최종 정책은
    FP32 inference 고정이다. True 를 넘기면 초기화 단계에서 명확하게 실패한다.
    """

    irra_root: str
    ckpt_path: str
    config_file: Optional[str] = None
    clip_pretrained: Optional[str] = None

    device: Optional[str] = None
    amp: bool = False
    img_size: Tuple[int, int] = (384, 128)
    stride_size: int = 16
    text_length: int = 77
    batch_size: int = 64
    l2_normalize: bool = True
    num_classes: Optional[int] = None


# --------------------------------------------------------------------------- #
# Embedder
# --------------------------------------------------------------------------- #
class IRRAEmbedder(BaseEmbedder):
    """IRRA 이미지/텍스트 임베딩 추출기 (FP32 추론 전용)."""

    DIM = 512

    def __init__(
        self,
        config: Optional[IRRAConfig] = None,
        *,
        irra_root: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        config_file: Optional[str] = None,
        clip_pretrained: Optional[str] = None,
        device: Optional[str] = None,
        amp: bool = False,
        img_size: Tuple[int, int] = (384, 128),
        stride_size: int = 16,
        text_length: int = 77,
        batch_size: int = 64,
        l2_normalize: bool = True,
        num_classes: Optional[int] = None,
    ) -> None:
        """두 가지 호출 방식을 모두 지원한다.

        IRRAEmbedder(irra_root=..., ckpt_path=...)  # registry / pipeline.yaml
        IRRAEmbedder(IRRAConfig(...))               # evaluation
        """

        if config is not None:
            if not isinstance(config, IRRAConfig):
                raise TypeError(
                    "첫 번째 위치 인자는 IRRAConfig 여야 합니다. "
                    "키워드로 넘기려면 IRRAEmbedder(irra_root=..., ckpt_path=...) "
                    f"형태를 쓰세요. (받은 타입: {type(config).__name__})"
                )

            # 호출자 소유 config 를 IRRA 내부에서 변경하지 않는다.
            cfg = replace(config)

        else:
            if not irra_root or not ckpt_path:
                raise ValueError(
                    "irra_root 와 ckpt_path 는 필수입니다. "
                    "pipeline.yaml 의 retrievers.irra.params 를 확인하세요."
                )

            cfg = IRRAConfig(
                irra_root=irra_root,
                ckpt_path=ckpt_path,
                config_file=config_file,
                clip_pretrained=clip_pretrained,
                device=device,
                amp=amp,
                img_size=tuple(img_size),
                stride_size=stride_size,
                text_length=text_length,
                batch_size=batch_size,
                l2_normalize=l2_normalize,
                num_classes=num_classes,
            )

        self._validate_config(cfg)
        self.cfg = cfg

        # BaseEmbedder 가 device / batch_size / l2_normalize 를 세팅한다.
        super().__init__(
            device=cfg.device,
            batch_size=cfg.batch_size,
            l2_normalize=cfg.l2_normalize,
        )

        # 프로젝트 최종 precision 정책:
        # IRRA inference 는 FP32 only. autocast / AMP 를 사용하지 않는다.
        self.use_amp = False

        (
            self._irra_cls,
            self._load_train_configs,
            self._tokenizer_cls,
        ) = self._load_irra_symbols(cfg.irra_root)

        self.args = self._build_args()
        self.model = self._build_model()
        self._validate_model_contract()
        self.transform = self._build_transform()
        self.tokenizer = self._build_tokenizer()

        try:
            self._sot = self.tokenizer.encoder["<|startoftext|>"]
            self._eot = self.tokenizer.encoder["<|endoftext|>"]
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(
                "IRRA SimpleTokenizer 에 SOT/EOT token 이 없습니다. "
                "IRRA 원본 repo / tokenizer 버전을 확인하세요."
            ) from exc

        logger.info(
            "IRRAEmbedder ready | device=%s precision=fp32 "
            "img_size=%s text_length=%d dim=%d",
            self.device,
            tuple(self.args.img_size),
            int(self.args.text_length),
            self.DIM,
        )

    # ======================================================================= #
    # Config / import validation
    # ======================================================================= #

    @staticmethod
    def _validate_config(cfg: IRRAConfig) -> None:
        if not str(cfg.irra_root).strip():
            raise ValueError("irra_root 는 비어 있을 수 없습니다.")

        if not str(cfg.ckpt_path).strip():
            raise ValueError("ckpt_path 는 비어 있을 수 없습니다.")

        if bool(cfg.amp):
            raise ValueError(
                "IRRA inference precision 은 프로젝트 기준 FP32로 고정되었습니다. "
                "amp=True 를 사용할 수 없습니다. amp=False 로 설정하거나 "
                "pipeline.yaml 에서 amp 항목을 제거하세요."
            )

        try:
            h, w = tuple(cfg.img_size)
        except Exception as exc:
            raise ValueError(
                f"img_size 는 (height, width) 2개 값이어야 합니다: {cfg.img_size!r}"
            ) from exc

        if int(h) <= 0 or int(w) <= 0:
            raise ValueError(f"img_size 는 양수여야 합니다: {cfg.img_size}")

        if int(cfg.stride_size) <= 0:
            raise ValueError(
                f"stride_size 는 1 이상이어야 합니다: {cfg.stride_size}"
            )

        if int(cfg.text_length) <= 0:
            raise ValueError(
                f"text_length 는 1 이상이어야 합니다: {cfg.text_length}"
            )

        if int(cfg.batch_size) <= 0:
            raise ValueError(
                f"batch_size 는 1 이상이어야 합니다: {cfg.batch_size}"
            )

        if cfg.num_classes is not None and int(cfg.num_classes) <= 0:
            raise ValueError(
                f"num_classes 는 1 이상이어야 합니다: {cfg.num_classes}"
            )

    @staticmethod
    def _resolve_irra_root(irra_root: str) -> Path:
        root = Path(irra_root).expanduser().resolve()

        if not root.is_dir():
            raise FileNotFoundError(
                f"IRRA 레포를 찾을 수 없습니다: {root}\n"
                f"  git clone https://github.com/anosorae/IRRA.git {root}"
            )

        if not (root / "model" / "build.py").is_file():
            raise FileNotFoundError(
                f"IRRA model/build.py 를 찾을 수 없습니다: {root}"
            )

        if not (root / "utils" / "iotools.py").is_file():
            raise FileNotFoundError(
                f"IRRA utils/iotools.py 를 찾을 수 없습니다: {root}"
            )

        if not (root / "utils" / "simple_tokenizer.py").is_file():
            raise FileNotFoundError(
                f"IRRA utils/simple_tokenizer.py 를 찾을 수 없습니다: {root}"
            )

        return root

    @classmethod
    def _load_irra_symbols(cls, irra_root: str):
        """IRRA symbol 만 가져오고 generic import namespace 는 원상복구한다."""

        root = cls._resolve_irra_root(irra_root)

        try:
            with _isolated_irra_import_scope(root):
                from model.build import IRRA  # type: ignore
                from utils.iotools import load_train_configs  # type: ignore
                from utils.simple_tokenizer import SimpleTokenizer  # type: ignore
        except Exception as exc:
            raise ImportError(
                "IRRA 원본 repo import 에 실패했습니다. "
                f"root={root}"
            ) from exc

        return IRRA, load_train_configs, SimpleTokenizer

    # ======================================================================= #
    # IRRA args
    # ======================================================================= #

    @staticmethod
    def _require_args(args, names: Sequence[str], owner: str) -> None:
        missing = [
            name
            for name in names
            if not hasattr(args, name) or getattr(args, name) is None
        ]

        if missing:
            raise ValueError(
                f"{owner} 에 IRRA 필수 설정이 없습니다: {missing}"
            )

    @staticmethod
    def _resolve_optional_file(path: Optional[str], label: str) -> Optional[str]:
        if path is None:
            return None

        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"{label} 파일이 없습니다: {resolved}")

        return str(resolved)

    def _build_args(self):
        """IRRA constructor 가 요구하는 args 를 구성한다.

        우선순위:
          1) configs.yaml 로 학습 설정을 읽는다.
          2) configs.yaml 에 기록된 img_size / stride / text_length 를 사용한다.
          3) clip_pretrained 가 있으면 pretrain_choice 를 local .pt 로 최종 override 한다.

        config_file 이 없거나 경로가 존재하지 않으면 명시적으로 로그를 남기고
        알려진 기본값으로 구성한다.
        """

        from argparse import Namespace

        cfg = self.cfg
        config_file = (
            str(Path(cfg.config_file).expanduser().resolve())
            if cfg.config_file
            else None
        )

        if config_file and os.path.isfile(config_file):
            args = self._load_train_configs(config_file)
            logger.info("configs.yaml 로드: %s", config_file)

            self._require_args(
                args,
                (
                    "pretrain_choice",
                    "temperature",
                    "loss_names",
                    "img_size",
                    "stride_size",
                ),
                owner="configs.yaml",
            )

            loaded_img_size = tuple(int(x) for x in args.img_size)
            if len(loaded_img_size) != 2:
                raise ValueError(
                    "configs.yaml img_size 는 (height, width) 2개 값이어야 합니다: "
                    f"{args.img_size!r}"
                )

            loaded_stride = int(args.stride_size)
            loaded_text_length = int(
                getattr(args, "text_length", cfg.text_length)
            )

            # caller config 는 이미 replace() 로 복사되어 있다.
            # 여기서도 새 객체로 갱신해서 상태 변경 지점을 명시한다.
            self.cfg = replace(
                cfg,
                img_size=loaded_img_size,
                stride_size=loaded_stride,
                text_length=loaded_text_length,
            )
            cfg = self.cfg
            self._validate_config(cfg)

        else:
            if config_file:
                logger.warning(
                    "configs.yaml 을 찾을 수 없습니다: %s -> "
                    "IRRA 기본 추론 설정으로 구성합니다. "
                    "checkpoint 의 학습 img_size/stride 와 다르면 "
                    "state_dict shape mismatch 로 모델 로드가 실패할 수 있습니다.",
                    config_file,
                )
            else:
                logger.warning(
                    "config_file 이 지정되지 않았습니다 -> "
                    "IRRA 기본 추론 설정으로 구성합니다. "
                    "최종 DB 구축에서는 checkpoint 와 함께 배포된 configs.yaml 사용을 "
                    "권장합니다."
                )

            args = Namespace(
                pretrain_choice="ViT-B/16",
                temperature=0.02,
                cmt_depth=4,
                loss_names="sdm+id+mlm",
                img_size=tuple(cfg.img_size),
                stride_size=int(cfg.stride_size),
                text_length=int(cfg.text_length),
                vocab_size=49408,
                id_loss_weight=1.0,
                mlm_loss_weight=1.0,
            )

        # IRRA 생성자에서 loss_names 에 따라 필요한 설정을 검증한다.
        tasks = {
            t.strip()
            for t in str(args.loss_names).split("+")
            if t.strip()
        }

        required = [
            "pretrain_choice",
            "temperature",
            "loss_names",
            "img_size",
            "stride_size",
        ]

        if "mlm" in tasks:
            required.extend(
                [
                    "cmt_depth",
                    "vocab_size",
                    "mlm_loss_weight",
                ]
            )

        if "id" in tasks:
            required.append("id_loss_weight")

        self._require_args(args, required, owner="IRRA args")

        # 추론에 필요한 값만 강제 정합한다.
        args.training = False
        args.img_size = tuple(int(x) for x in cfg.img_size)
        args.stride_size = int(cfg.stride_size)
        args.text_length = int(cfg.text_length)

        # 오프라인 / 고정 CLIP weight를 쓰는 경우 configs.yaml보다 우선한다.
        local_clip = self._resolve_optional_file(
            cfg.clip_pretrained,
            "IRRA CLIP pretrained",
        )
        if local_clip is not None:
            args.pretrain_choice = local_clip
            logger.info("IRRA CLIP local pretrained 사용: %s", local_clip)

        if not str(args.pretrain_choice).strip():
            raise ValueError("IRRA pretrain_choice 가 비어 있습니다.")

        return args

    # ======================================================================= #
    # Checkpoint / model
    # ======================================================================= #

    def _resolve_state_dict(self):
        path = Path(self.cfg.ckpt_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"IRRA 체크포인트가 없습니다: {path}")

        # torch>=2.6 에서는 weights_only 기본 동작 변화가 있으므로,
        # IRRA 공식/신뢰 가능한 checkpoint를 명시적으로 전체 로드한다.
        try:
            ckpt = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            # 구버전 PyTorch 는 weights_only 인자를 지원하지 않는다.
            ckpt = torch.load(
                path,
                map_location="cpu",
            )

        for key in ("model", "state_dict"):
            if (
                isinstance(ckpt, dict)
                and key in ckpt
                and isinstance(ckpt[key], dict)
            ):
                ckpt = ckpt[key]
                break

        if not isinstance(ckpt, dict):
            raise TypeError(
                "IRRA checkpoint 의 최종 state_dict 가 dict 가 아닙니다: "
                f"{type(ckpt).__name__}"
            )

        # DataParallel prefix 제거
        return {
            k[7:] if k.startswith("module.") else k: v
            for k, v in ckpt.items()
        }

    def _build_model(self):
        state_dict = self._resolve_state_dict()

        num_classes = self.cfg.num_classes
        if num_classes is None:
            classifier_weight = state_dict.get("classifier.weight")

            if classifier_weight is not None:
                num_classes = int(classifier_weight.shape[0])
                logger.info(
                    "num_classes=%d (checkpoint classifier.weight 에서 추론)",
                    num_classes,
                )
            else:
                # id head 를 아래에서 제거하므로 이 값은 classifier 생성에 사용되지 않는다.
                num_classes = 11003
                logger.info(
                    "classifier.weight 없음 -> id task 제거 예정; "
                    "num_classes placeholder=%d",
                    num_classes,
                )

        # checkpoint 에 실제로 존재하는 head만 생성한다.
        # IRRA inference 자체는 encode_image / encode_text 만 사용하지만,
        # 불필요한 head shape mismatch 때문에 model construction/load 가 깨지는 것을 막는다.
        tasks = [
            t.strip()
            for t in str(self.args.loss_names).split("+")
            if t.strip()
        ]

        if "id" in tasks and "classifier.weight" not in state_dict:
            tasks.remove("id")

        if "mlm" in tasks and not any(
            key.startswith("mlm_head.")
            for key in state_dict
        ):
            tasks.remove("mlm")

        self.args.loss_names = "+".join(tasks) if tasks else "sdm"

        model = self._irra_cls(
            self.args,
            num_classes=int(num_classes),
        )

        try:
            missing, unexpected = model.load_state_dict(
                state_dict,
                strict=False,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "IRRA checkpoint state_dict shape mismatch 입니다. "
                "특히 checkpoint 학습 당시 img_size / stride_size 와 현재 설정이 "
                "같은지 확인하세요. "
                f"현재 img_size={tuple(self.args.img_size)}, "
                f"stride_size={self.args.stride_size}"
            ) from exc

        critical = [
            key
            for key in missing
            if key.startswith("base_model.")
        ]
        if critical:
            raise RuntimeError(
                "IRRA checkpoint 에 base_model 가중치가 누락되었습니다.\n"
                f"  누락 예시: {critical[:5]}"
            )

        if missing:
            logger.debug("missing keys (무시 가능): %s", missing[:10])

        if unexpected:
            logger.debug("unexpected keys (무시 가능): %s", unexpected[:10])

        # ---------------------------------------------------------------
        # 프로젝트 최종 precision 정책: FP32 inference
        # ---------------------------------------------------------------
        # 공식 IRRA build_model()은 convert_weights()를 호출해 일부 weight를
        # fp16 계열로 변환하지만, 이 adapter는 기준 DB/query 경로를 FP32로 고정한다.
        # 따라서 model 전체를 float32로 올리고 autocast도 사용하지 않는다.
        model.float().to(self.device).eval()

        for param in model.parameters():
            param.requires_grad_(False)

        return model

    def _validate_model_contract(self) -> None:
        """IRRA model 과 프로젝트 embedding contract 를 초기화 시점에 검증한다."""

        embed_dim = getattr(self.model, "embed_dim", None)
        if embed_dim is None:
            raise RuntimeError("IRRA model.embed_dim 을 확인할 수 없습니다.")

        if int(embed_dim) != self.DIM:
            raise RuntimeError(
                f"IRRA embedding dimension 불일치: "
                f"expected={self.DIM}, actual={embed_dim}"
            )

        base_model = getattr(self.model, "base_model", None)
        if base_model is None:
            raise RuntimeError("IRRA model.base_model 을 찾을 수 없습니다.")

        context_length = getattr(base_model, "context_length", None)
        if context_length is None:
            raise RuntimeError(
                "IRRA base_model.context_length 를 확인할 수 없습니다."
            )

        model_context_length = int(context_length)
        configured_text_length = int(self.args.text_length)

        if configured_text_length != model_context_length:
            raise ValueError(
                "IRRA text_length / CLIP context_length 불일치입니다. "
                f"text_length={configured_text_length}, "
                f"model_context_length={model_context_length}. "
                "checkpoint/configs.yaml 과 CLIP pretrained 조합을 확인하세요."
            )

    # ======================================================================= #
    # Preprocess / tokenizer
    # ======================================================================= #

    def _build_transform(self):
        import torchvision.transforms as T

        h, w = self.args.img_size

        return T.Compose(
            [
                # IRRA 공식 datasets/build.py test transform 은 interpolation 을
                # 지정하지 않는다 -> torchvision 기본 BILINEAR.
                T.Resize((h, w)),
                T.ToTensor(),
                T.Normalize(
                    mean=CLIP_MEAN,
                    std=CLIP_STD,
                ),
            ]
        )

    def _build_tokenizer(self):
        return self._tokenizer_cls()

    # ======================================================================= #
    # BaseEmbedder implementation
    # ======================================================================= #

    @torch.inference_mode()
    def _encode(
        self,
        images: List[Image.Image],
    ) -> np.ndarray:
        """RGB PIL list -> (N, 512) float32.

        BaseEmbedder 공개 경로에서는 빈 입력을 먼저 처리하지만,
        `_encode([])` 를 직접 호출해도 안전하도록 방어한다.
        """

        if not images:
            return np.zeros(
                (0, self.DIM),
                dtype=np.float32,
            )

        batch = torch.stack(
            [
                self.transform(image)
                for image in images
            ]
        )

        batch = batch.to(
            self.device,
            non_blocking=True,
        ).float()

        # FP32 inference fixed: autocast 없음.
        feats = self.model.encode_image(batch)

        return (
            feats
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    # ======================================================================= #
    # Public APIs
    # ======================================================================= #

    def embed_image_paths(
        self,
        paths: Sequence[Union[str, Path]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """이미지 파일 경로 리스트 -> (N, 512) float32."""

        return self.embed_crops(
            paths,
            input_format="rgb",
            batch_size=batch_size,
        )

    @torch.inference_mode()
    def embed_text(
        self,
        captions: Union[str, Sequence[str]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """자연어 query -> (N, 512) float32.

        이미지와 동일한 IRRA embedding space에 놓인다.
        CLIP BPE tokenizer를 사용하므로 프로젝트에서는 영어 query를 기준으로 한다.
        """

        if isinstance(captions, str):
            captions = [captions]

        if len(captions) == 0:
            return np.zeros(
                (0, self.DIM),
                dtype=np.float32,
            )

        bs = batch_size or self.batch_size
        if bs <= 0:
            raise ValueError("batch_size 는 1 이상이어야 합니다.")

        outs: List[np.ndarray] = []

        for i in range(0, len(captions), bs):
            chunk = captions[i:i + bs]

            ids = torch.stack(
                [
                    self._tokenize(caption)
                    for caption in chunk
                ]
            ).to(self.device)

            # FP32 inference fixed: autocast 없음.
            feats = self.model.encode_text(ids)

            outs.append(
                feats
                .float()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )

        result = np.concatenate(
            outs,
            axis=0,
        ).astype(np.float32, copy=False)

        if result.ndim != 2 or result.shape[1] != self.DIM:
            raise RuntimeError(
                "IRRA text embedding shape 오류: "
                f"expected=(N, {self.DIM}), actual={result.shape}"
            )

        if not np.isfinite(result).all():
            raise RuntimeError(
                "IRRA text embedding 에 NaN/Inf 가 있습니다."
            )

        # 이미지 vector는 BaseEmbedder가 동일 policy로 정규화한다.
        if self.l2_normalize:
            result = l2_normalize(
                result,
                owner=type(self).__name__,
            )

        return np.asarray(
            result,
            dtype=np.float32,
        )

    # ======================================================================= #
    # Tokenization
    # ======================================================================= #

    def _tokenize(
        self,
        caption: str,
    ) -> torch.LongTensor:
        """IRRA / CLIP BPE 규칙에 맞춘 fixed-length token tensor 생성.

        정상 caption에서는 공식 IRRA tokenize와 동일하게 SOT + body + EOT를 만든다.
        추가 방어로 사용자가 literal special-token 문자열을 넣어 body 안에 SOT/EOT가
        생기는 경우를 제거한다. EOT는 vocab 최대 id이고 IRRA encode_text가
        `text.argmax(dim=-1)` 로 EOT 위치를 선택하므로 중간 EOT를 남기지 않는다.
        """

        if not isinstance(caption, str):
            raise TypeError(
                f"caption 은 str 이어야 합니다: {type(caption).__name__}"
            )

        body = [
            token
            for token in self.tokenizer.encode(caption)
            if token not in (self._sot, self._eot)
        ]

        tokens = [
            self._sot,
            *body,
            self._eot,
        ]

        n = int(self.args.text_length)

        if len(tokens) > n:
            tokens = tokens[:n]
            tokens[-1] = self._eot

        result = torch.zeros(
            n,
            dtype=torch.long,
        )

        result[:len(tokens)] = torch.tensor(
            tokens,
            dtype=torch.long,
        )

        return result


# --------------------------------------------------------------------------- #
# CLI smoke test
#
# python -m embedders.human.irra_embedder \
#   --irra-root ./IRRA \
#   --ckpt ./weights/IRRA/cuhk_pedes/best.pth \
#   --config-file ./weights/IRRA/cuhk_pedes/configs.yaml \
#   --clip-pretrained ./weights/IRRA/ViT-B-16.pt \
#   --images a.jpg b.jpg
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser(
        description="IRRA FP32 embedding smoke test"
    )

    ap.add_argument(
        "--irra-root",
        required=True,
    )
    ap.add_argument(
        "--ckpt",
        required=True,
    )
    ap.add_argument(
        "--config-file",
        default=None,
    )
    ap.add_argument(
        "--clip-pretrained",
        default=None,
    )
    ap.add_argument(
        "--images",
        nargs="*",
        default=[],
    )
    ap.add_argument(
        "--text",
        default="a man wearing a red shirt and black pants",
    )
    ap.add_argument(
        "--device",
        default=None,
    )

    args = ap.parse_args()

    emb = IRRAEmbedder(
        irra_root=args.irra_root,
        ckpt_path=args.ckpt,
        config_file=args.config_file,
        clip_pretrained=args.clip_pretrained,
        device=args.device,
        amp=False,
    )

    text_vec = emb.embed_text(args.text)
    print(
        f"text embedding : {text_vec.shape}, "
        f"dtype={text_vec.dtype}, "
        f"norm={np.linalg.norm(text_vec[0]):.4f}"
    )

    if args.images:
        image_vecs = emb.embed_crops(
            args.images,
            input_format="rgb",
        )

        print(
            f"image embedding: {image_vecs.shape}, "
            f"dtype={image_vecs.dtype}, "
            f"norm={np.linalg.norm(image_vecs[0]):.4f}"
        )

        similarities = image_vecs @ text_vec[0]

        for path, score in zip(
            args.images,
            similarities,
        ):
            print(
                f"  {score:+.4f}  "
                f"{os.path.basename(path)}"
            )
