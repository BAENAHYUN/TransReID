"""
ingest_scvd_objects_registers_v2.py

기존 src/RF-DETR/ingest_scvd_objects_v2.py의 DB 적재 로직은 그대로 재사용하면서,
DINOv2 부분만 논문 기반 Registers V2 임베더로 교체하는 실행 래퍼.

사용 위치:
    TransReID/src/RF-DETR/ingest_scvd_objects_registers_v2.py

전제:
    TransReID/embedders/object/dinov2_embedder_registers_v2.py 가 존재해야 함.

실행:
    python .\src\RF-DETR\ingest_scvd_objects_registers_v2.py --force

동작:
    SigLIP2: 기존 ingest_scvd_objects_v2.py 구현 그대로
    DINOv2: facebook/dinov2-with-registers-base
            feature=cls
            image_size=224
            resize_mode=center_crop
            L2 normalize
            output=768D
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# 프로젝트 root를 sys.path에 추가
# 이 파일은 TransReID/src/RF-DETR/ 아래에 두는 것을 기준으로 한다.
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 기존 ingest 모듈 import
import ingest_scvd_objects_v2 as original_ingest

# 새 DINOv2 Registers 임베더
from embedders.object.dinov2_embedder_registers_v2 import DINOv2Embedder


DINO_MODEL = "facebook/dinov2-with-registers-base"


class DINOv2RegistersEncoder:
    """
    기존 ingest_scvd_objects_v2.DINOv2Encoder와 동일하게
    encode(paths) -> np.ndarray 인터페이스를 제공하는 어댑터.
    """

    def __init__(self, model_name: str = DINO_MODEL):
        print(f"[*] DINOv2 Registers V2 로드 중: {model_name}")

        self.embedder = DINOv2Embedder(
            model_id=model_name,
            feature="cls",
            image_size=224,
            resize_mode="center_crop",
            l2_normalize=True,
            allow_experimental=False,
        )

        # 기존 코드/로그에서 참조할 수 있도록 보존
        self.dev = getattr(self.embedder, "device", "unknown")

        dim = int(getattr(self.embedder, "DIM", 768))
        if dim != 768:
            raise RuntimeError(
                f"DINOv2 Registers V2 출력 차원이 768이 아닙니다: {dim}. "
                "forensic_object의 dinov2 named vector schema와 일치해야 합니다."
            )

        print(
            "[+] DINOv2 Registers V2 준비 완료 "
            f"(model={model_name}, feature=cls, dim={dim}, device={self.dev})"
        )

    def encode(self, paths: List[Path]) -> np.ndarray:
        """
        기존 ingest 코드가 넘기는 Path 배치를 PIL RGB로 열고,
        Registers-aware DINOv2Embedder에 전달한다.
        """
        if not paths:
            return np.zeros((0, 768), dtype=np.float32)

        # BaseEmbedder.embed_crops()의 input_format은 "pil"을 받지 않는다.
        # 따라서 PIL에서 RGB uint8 numpy 배열로 변환한 뒤 input_format="rgb"로 넘긴다.
        images = []
        for p in paths:
            with Image.open(p) as im:
                images.append(np.asarray(im.convert("RGB"), dtype=np.uint8).copy())

        vecs = self.embedder.embed_crops(
            images,
            input_format="rgb",
        )

        vecs = np.asarray(vecs, dtype=np.float32)

        if vecs.ndim != 2 or vecs.shape[1] != 768:
            raise RuntimeError(
                f"예상 DINOv2 embedding shape=(N, 768), 실제={vecs.shape}"
            )

        # V2 embedder가 L2 normalize하도록 설정되어 있지만
        # DB 투입 직전 값이 정상인지 fail-fast 검증한다.
        norms = np.linalg.norm(vecs, axis=1)
        if not np.all(np.isfinite(vecs)):
            raise RuntimeError("DINOv2 embedding에 NaN/Inf가 포함되어 있습니다.")
        if len(norms) and np.max(np.abs(norms - 1.0)) > 1e-3:
            raise RuntimeError(
                "DINOv2 embedding L2 norm 검증 실패: "
                f"min={norms.min():.6f}, max={norms.max():.6f}"
            )

        return vecs


def main():
    # -----------------------------------------------------------------------
    # 기존 ingest의 나머지 로직은 그대로 두고 DINOv2 구현만 교체한다.
    # original_ingest.main() 안에서 DINOv2Encoder()를 생성하면
    # 아래 클래스가 대신 생성된다.
    # -----------------------------------------------------------------------
    original_ingest.DINO_MODEL = DINO_MODEL
    original_ingest.DINOv2Encoder = DINOv2RegistersEncoder

    print("=" * 72)
    print("SCVD OBJECT INGEST — DINOv2 Registers V2 override")
    print("  DINO model : facebook/dinov2-with-registers-base")
    print("  feature    : CLS")
    print("  dim        : 768")
    print("  preprocess : Resize(shortest=256) -> CenterCrop(224)")
    print("  normalize  : L2")
    print("=" * 72)

    original_ingest.main()


if __name__ == "__main__":
    main()