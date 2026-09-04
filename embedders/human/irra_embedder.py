"""
IRRA Embedding Extractor
========================

파이프라인 위치:
    Person Crop -> [IRRA] -> 512-d Embedding -> Qdrant

IRRA(CVPR'23, "Cross-Modal Implicit Relation Reasoning and Aligning for
Text-to-Image Person Retrieval")는 CLIP ViT-B/16 백본을 text-to-image person
ReID 로 파인튜닝한 모델이다. **사람 전용** 이므로 scope='person' 으로 등록한다.

특징:
  - 이미지 임베딩 512-d, 텍스트 임베딩 512-d 가 같은 공간에 있다
    -> Qdrant 에 이미지 벡터만 넣어두고 "a man in a red shirt with a black
       backpack" 같은 자연어로 바로 검색할 수 있다.
  - 입력 해상도 384x128 (ReID 표준 세로 비율), CLIP normalize
  - 추론에는 IRR(cross-modal interaction) / MLM 헤드를 쓰지 않는다.
    논문 주장대로 이들은 학습 시에만 관여하고 추론 비용은 0 이다.

BaseEmbedder 를 상속한다
-----------------------
경로/PIL/numpy 입력 처리, 알파 합성, 배치 분할, L2 정규화, 차원 검증, NaN 검사는
전부 BaseEmbedder 가 한다. 이 파일은 `_encode(pil_images)` 만 구현한다.
예전 독립 구현은 문자열 경로를 받지 못해, 라우터가 crop 경로를 넘기면
IRRA 만 TypeError 로 죽었다.

전처리
-----
IRRA 공식 `datasets/build.py` 의 **test** transform 은 interpolation 을 지정하지
않는다. 즉 torchvision 기본값인 **BILINEAR** 다. BICUBIC 이 아니다.
(SOLIDER-REID 의 test transform 도 동일하게 BILINEAR 다.)
여기를 바꾸면 이미 인덱싱한 벡터가 전부 무효가 되므로 함부로 손대지 말 것.

측정된 성능
----------
Market1501 zero-shot: R@1 84.35 / mAP 66.69 / mINP 29.56
  -> CUHK-PEDES 학습분이라 cross-dataset 조건이다. R@1 대비 mINP 가 크게 낮은데,
     이는 "정답 하나는 잘 찾지만 같은 사람의 나머지 등장을 상위로 올리지는
     못한다"는 뜻이다. 전체 등장 이력을 모으는 용도라면 이 점을 감안할 것.

주의 (논문이 보증하지 않는 영역):
  - 논문 실험은 전부 text->image 다. image->image 는 위 자체 측정치가 전부다.
  - 학습 캡션은 평균 20단어 이상의 서술형이다. 짧은 쿼리는 분포 밖이다.
  - 토크나이저가 CLIP BPE(영어) 이므로 **한국어 쿼리는 지원되지 않는다.**
    상위 레이어에서 번역할 것.

필요한 것:
  1) IRRA 원본 레포 (모델 정의 코드) : https://github.com/anosorae/IRRA
  2) 학습된 가중치 best.pth + configs.yaml

사용 (pipeline.yaml 의 params 가 그대로 kwargs 로 들어온다):
    emb = IRRAEmbedder(
        irra_root="./IRRA",
        ckpt_path="./weights/best.pth",
        config_file="./weights/configs.yaml",
    )
    vecs = emb.embed_crops(crops, input_format="rgb")   # (N, 512) L2 정규화
    qvec = emb.embed_text("a man in a red shirt")       # (1, 512)
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

from ..base import BaseEmbedder, l2_normalize

logger = logging.getLogger(__name__)

# CLIP 정규화 상수 (IRRA datasets/build.py 와 동일해야 함)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# --------------------------------------------------------------------------- #
# 내부 설정 컨테이너
# --------------------------------------------------------------------------- #
@dataclass
class IRRAConfig:
    """생성자 인자를 묶어두는 내부용 컨테이너.

    pipeline.yaml 경로에서는 IRRAEmbedder(**params) 로 kwargs 를 직접 받으므로
    보통 쓸 일이 없다. 평가 스크립트처럼 설정을 통째로 넘기고 싶을 때 사용한다.
    """
    irra_root: str
    ckpt_path: str
    config_file: Optional[str] = None

    device: Optional[str] = None
    amp: bool = True                          # autocast(fp16). 가중치는 항상 fp32
    img_size: Tuple[int, int] = (384, 128)    # (height, width)
    stride_size: int = 16
    text_length: int = 77
    batch_size: int = 64
    l2_normalize: bool = True
    num_classes: Optional[int] = None         # None 이면 체크포인트에서 자동 추론


# --------------------------------------------------------------------------- #
# Embedder
# --------------------------------------------------------------------------- #
class IRRAEmbedder(BaseEmbedder):
    """IRRA 이미지/텍스트 임베딩 추출기 (추론 전용)."""

    DIM = 512  # CLIP ViT-B/16 projection dim

    def __init__(
        self,
        config: Optional[IRRAConfig] = None,
        *,
        irra_root: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        config_file: Optional[str] = None,
        device: Optional[str] = None,
        amp: bool = True,
        img_size: Tuple[int, int] = (384, 128),
        stride_size: int = 16,
        text_length: int = 77,
        batch_size: int = 64,
        l2_normalize: bool = True,
        num_classes: Optional[int] = None,
    ) -> None:
        """
        두 가지 호출 방식을 모두 받는다.

            IRRAEmbedder(irra_root=..., ckpt_path=...)   # registry / pipeline.yaml
            IRRAEmbedder(IRRAConfig(...))                # 평가 스크립트
        """
        if config is not None:
            if not isinstance(config, IRRAConfig):
                raise TypeError(
                    "첫 번째 위치 인자는 IRRAConfig 여야 합니다. "
                    "키워드로 넘기려면 IRRAEmbedder(irra_root=..., ckpt_path=...) "
                    f"형태를 쓰세요. (받은 타입: {type(config).__name__})"
                )
            cfg = config
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
                device=device,
                amp=amp,
                img_size=tuple(img_size),
                stride_size=stride_size,
                text_length=text_length,
                batch_size=batch_size,
                l2_normalize=l2_normalize,
                num_classes=num_classes,
            )

        self.cfg = cfg

        # BaseEmbedder 가 device / batch_size / l2_normalize 를 세팅한다.
        # self.device 는 문자열이다 ("cuda" | "cpu"). torch 의 .to() 는 문자열을 받는다.
        super().__init__(
            device=cfg.device,
            batch_size=cfg.batch_size,
            l2_normalize=cfg.l2_normalize,
        )

        self.use_amp = bool(cfg.amp and str(self.device).startswith("cuda"))

        self._inject_irra_path(cfg.irra_root)
        self.args = self._build_args()
        self.model = self._build_model()
        self.transform = self._build_transform()
        self.tokenizer = self._build_tokenizer()
        self._sot = self.tokenizer.encoder["<|startoftext|>"]
        self._eot = self.tokenizer.encoder["<|endoftext|>"]

        logger.info(
            "IRRAEmbedder ready | device=%s amp=%s img_size=%s dim=%d",
            self.device, self.use_amp, tuple(self.args.img_size), self.DIM,
        )

    # ---------------- 초기화 helpers ---------------- #

    @staticmethod
    def _inject_irra_path(irra_root: str) -> None:
        root = os.path.abspath(os.path.expanduser(str(irra_root)))
        if not os.path.isdir(os.path.join(root, "model")):
            raise FileNotFoundError(
                f"IRRA 레포를 찾을 수 없습니다: {root}\n"
                f"  git clone https://github.com/anosorae/IRRA.git {root}"
            )
        if root not in sys.path:
            sys.path.insert(0, root)

    def _build_args(self):
        """IRRA 의 build_model 은 argparse Namespace 를 받는다.

        체크포인트와 함께 배포되는 configs.yaml 이 있으면 그대로 쓰고,
        없으면 공식 기본 설정(sdm+id+mlm, ViT-B/16, 384x128)으로 재구성한다.
        """
        from argparse import Namespace

        cfg = self.cfg
        config_file = (
            os.path.expanduser(str(cfg.config_file)) if cfg.config_file else None
        )

        if config_file and os.path.isfile(config_file):
            from utils.iotools import load_train_configs  # type: ignore
            args = load_train_configs(config_file)
            logger.info("configs.yaml 로드: %s", config_file)
            # 학습 당시 해상도/stride 를 그대로 따라간다.
            # 다르면 position embedding shape 이 어긋난다.
            cfg.img_size = tuple(args.img_size)
            cfg.stride_size = int(args.stride_size)
            cfg.text_length = int(getattr(args, "text_length", cfg.text_length))
        else:
            if cfg.config_file:
                logger.warning(
                    "configs.yaml 을 찾을 수 없습니다: %s -> 기본 설정으로 구성합니다. "
                    "학습 해상도가 384x128 이 아니었다면 position embedding 이 "
                    "어긋납니다.", config_file,
                )
            args = Namespace(
                pretrain_choice="ViT-B/16",
                temperature=0.02,
                cmt_depth=4,
                loss_names="sdm+id+mlm",
                img_size=cfg.img_size,
                stride_size=cfg.stride_size,
                text_length=cfg.text_length,
                vocab_size=49408,
                id_loss_weight=1.0,
                mlm_loss_weight=1.0,
            )

        # 추론에 필요한 값만 강제 정합
        args.training = False
        args.img_size = tuple(cfg.img_size)
        args.stride_size = cfg.stride_size
        args.text_length = cfg.text_length
        return args

    def _resolve_state_dict(self):
        path = os.path.expanduser(str(self.cfg.ckpt_path))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"IRRA 체크포인트가 없습니다: {path}")

        # torch>=2.6 은 weights_only=True 가 기본이라 Namespace 가 든 ckpt 로드에 실패한다.
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # 구버전 torch 는 weights_only 인자가 없음
            ckpt = torch.load(path, map_location="cpu")

        for key in ("model", "state_dict"):
            if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

        if not isinstance(ckpt, dict):
            raise TypeError(
                f"체크포인트 형식이 dict 가 아닙니다: {type(ckpt).__name__}"
            )

        # DataParallel 접두어 제거
        return {
            k[7:] if k.startswith("module.") else k: v
            for k, v in ckpt.items()
        }

    def _build_model(self):
        from model.build import IRRA  # type: ignore

        state_dict = self._resolve_state_dict()

        num_classes = self.cfg.num_classes
        if num_classes is None:
            w = state_dict.get("classifier.weight")
            num_classes = int(w.shape[0]) if w is not None else 11003
            logger.info("num_classes=%d (체크포인트에서 추론)", num_classes)

        # loss_names 에 실제 체크포인트에 있는 헤드만 남겨 shape mismatch 방지.
        # 전부 사라져도 최소 'sdm' 은 남긴다 (IRRA 생성자의 기본 분기).
        tasks = [t.strip() for t in str(self.args.loss_names).split("+") if t.strip()]
        if "id" in tasks and "classifier.weight" not in state_dict:
            tasks.remove("id")
        if "mlm" in tasks and not any(k.startswith("mlm_head.") for k in state_dict):
            tasks.remove("mlm")
        self.args.loss_names = "+".join(tasks) if tasks else "sdm"

        model = IRRA(self.args, num_classes=num_classes)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        critical = [k for k in missing if k.startswith("base_model.")]
        if critical:
            raise RuntimeError(
                "체크포인트에 백본 가중치가 없습니다. 파일이 잘못된 것 같습니다.\n"
                f"  누락 예시: {critical[:5]}"
            )
        if missing:
            logger.debug("missing keys (무시 가능): %s", missing[:10])
        if unexpected:
            logger.debug("unexpected keys (무시 가능): %s", unexpected[:10])

        # 가중치는 항상 fp32. 속도가 필요하면 autocast 로 처리한다.
        # IRRA 는 AMP 로 학습되었으므로 model.half() 전체 캐스팅은 학습 시 수치와
        # 어긋난다.
        model.float().to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def _build_transform(self):
        import torchvision.transforms as T

        h, w = self.args.img_size
        return T.Compose([
            # IRRA 공식 datasets/build.py 의 test transform 은 interpolation 을
            # 지정하지 않는다 -> torchvision 기본값 BILINEAR.
            # BICUBIC 으로 바꾸면 공식 평가 조건에서 벗어나고, 이미 인덱싱한
            # 벡터와도 섞이지 않는다. 손대지 말 것.
            T.Resize((h, w)),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    def _build_tokenizer(self):
        from utils.simple_tokenizer import SimpleTokenizer  # type: ignore
        return SimpleTokenizer()

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    # ---------------- BaseEmbedder 구현 지점 ---------------- #

    @torch.inference_mode()
    def _encode(self, images: List[Image.Image]) -> np.ndarray:
        """RGB PIL 리스트 -> (N, 512).

        경로/numpy/알파 처리, 배치 분할, L2 정규화, 차원·NaN 검증은
        BaseEmbedder 가 이미 했다. 여기서는 순수 forward 만 한다.
        """
        batch = torch.stack([self.transform(im) for im in images])
        batch = batch.to(self.device, non_blocking=True).float()

        with self._autocast():
            feats = self.model.encode_image(batch)  # (B, 512)

        return feats.float().cpu().numpy()

    # ---------------- 추가 공개 API ---------------- #

    def embed_image_paths(
        self,
        paths: Sequence[Union[str, Path]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """이미지 파일 경로 리스트 -> (N, 512).

        BaseEmbedder.embed_crops 가 경로를 직접 처리하므로 얇은 별칭이다.
        평가 스크립트(eval_i2i / eval_t2i)가 이 이름을 쓴다.
        """
        return self.embed_crops(paths, input_format="rgb", batch_size=batch_size)

    @torch.inference_mode()
    def embed_text(
        self,
        captions: Union[str, Sequence[str]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """자연어 쿼리 -> (N, 512). 이미지 임베딩과 같은 공간이라 바로 코사인 비교 가능.

        라우터는 이 메서드의 **존재 여부**로 텍스트 검색 참여를 판단한다.
        SOLIDER / DINOv2 에는 없으므로 자동으로 빠진다.

        영어만 지원한다 (CLIP BPE tokenizer). 한국어는 상위 레이어에서 번역할 것.
        """
        if isinstance(captions, str):
            captions = [captions]
        if len(captions) == 0:
            return np.zeros((0, self.DIM), dtype=np.float32)

        bs = batch_size or self.batch_size
        outs: List[np.ndarray] = []

        for i in range(0, len(captions), bs):
            chunk = captions[i:i + bs]
            ids = torch.stack([self._tokenize(c) for c in chunk]).to(self.device)

            with self._autocast():
                feats = self.model.encode_text(ids)  # (B, 512)

            outs.append(feats.float().cpu().numpy().astype(np.float32))

        result = np.concatenate(outs, axis=0)

        if result.shape[1] != self.DIM:
            raise RuntimeError(
                f"텍스트 임베딩 차원 {result.shape[1]} != DIM {self.DIM}"
            )
        if not np.isfinite(result).all():
            raise RuntimeError("텍스트 임베딩에 NaN/Inf 가 있습니다.")

        # 이미지 경로는 BaseEmbedder 가 정규화하므로, 텍스트도 같은 규칙을 따라야
        # 두 벡터를 같은 공간에서 비교할 수 있다.
        if self.l2_normalize:
            result = l2_normalize(result, owner=type(self).__name__)
        return result

    # ---------------- 내부 ---------------- #

    def _tokenize(self, caption: str) -> torch.LongTensor:
        """IRRA datasets/bases.py 의 tokenize 와 동일 동작.

        encode_text 는 text.argmax(dim=-1) 로 EOT 위치를 찾는다. EOT(49407) 가
        vocab 에서 가장 큰 id 이므로, 시퀀스 안에 EOT 가 정확히 하나만 있어야
        올바른 토큰에서 feature 를 뽑는다. 사용자가 넣은 특수 토큰 문자열이
        본문 중간에서 EOT 로 인코딩되는 경우를 대비해 본문에서 SOT/EOT 를 제거한다.

        ※ 이 부분은 아직 원본 datasets/bases.py 와 직접 대조하지 않았다.
          텍스트 검색을 실제로 쓰기 전에 확인할 것.
        """
        body = [
            t for t in self.tokenizer.encode(caption)
            if t not in (self._sot, self._eot)
        ]
        tokens = [self._sot] + body + [self._eot]

        n = self.args.text_length
        if len(tokens) > n:
            tokens = tokens[:n]
            tokens[-1] = self._eot  # 잘린 경우 마지막 자리를 EOT 로 되돌린다

        result = torch.zeros(n, dtype=torch.long)  # padding = 0
        result[:len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        return result


# --------------------------------------------------------------------------- #
# CLI 스모크 테스트
#   python -m embedders.human.irra_embedder \
#       --irra-root ./IRRA --ckpt ./weights/best.pth \
#       --config-file ./weights/configs.yaml --images a.jpg b.jpg
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    ap = argparse.ArgumentParser(description="IRRA embedding smoke test")
    ap.add_argument("--irra-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-file", default=None)
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--text", default="a man wearing a red shirt and black pants")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true",
                    help="autocast 끄고 순수 fp32 로 실행")
    args = ap.parse_args()

    emb = IRRAEmbedder(
        irra_root=args.irra_root,
        ckpt_path=args.ckpt,
        config_file=args.config_file,
        device=args.device,
        amp=not args.no_amp,
    )

    t = emb.embed_text(args.text)
    print(f"text embedding : {t.shape}, norm={np.linalg.norm(t[0]):.4f}")

    if args.images:
        # 경로를 그대로 넘긴다 — BaseEmbedder 가 처리한다
        v = emb.embed_crops(args.images, input_format="rgb")
        print(f"image embedding: {v.shape}, norm={np.linalg.norm(v[0]):.4f}")
        sims = v @ t[0]
        for p, s in zip(args.images, sims):
            print(f"  {s:+.4f}  {os.path.basename(p)}")
