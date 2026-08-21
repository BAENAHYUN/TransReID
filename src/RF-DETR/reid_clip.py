"""
마일스톤 2: CLIP-ReID로 Person Crop에서 embedding 뽑기 (함수화 버전)
- load_reid_model(): 모델을 딱 한 번만 로드
- get_embedding(model, device, image_path): 사진 경로를 인자로 받아 embedding 반환

주의: CLIP-ReID는 codebase가 TransReID에서 파생되어 구조가 비슷하지만,
      make_model_clipreid.py를 씁니다. forward()의 반환값이 TransReID와 다릅니다.
"""

import sys
import torch
from PIL import Image
from torchvision import transforms

sys.path.append("src/CLIP-ReID_official")

from model.make_model_clipreid import make_model
from config import cfg

# ── 고정 설정 (query가 아니라 환경설정) ──
WEIGHT_PATH = "weights/Market1501_clipreid_12x12sie_ViT-B-16_60.pth"  # SIE-OLP weight로 변경, 본인 실제 파일명 확인 필요
CONFIG_PATH = "src/CLIP-ReID_official/configs/person/vit_clipreid.yml"

_transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_reid_model():
    """
    CLIP-ReID(SIE-OLP) 모델을 딱 한 번 로드해서 반환.
    README 학습 명령어 기준(SIE_CAMERA True, SIE_COE 1.0, STRIDE_SIZE [12,12])으로
    설정을 맞춰서 weight 구조와 일치시킴.
    """
    cfg.merge_from_file(CONFIG_PATH)
    cfg.MODEL.SIE_CAMERA = True   # SIE 활성화 (카메라 정보 실제로 사용)
    cfg.MODEL.SIE_COE = 1.0       # README 학습 명령어와 동일하게 맞춤
    cfg.MODEL.STRIDE_SIZE = [12, 12]
    cfg.freeze()

    num_classes = 751
    model = make_model(cfg, num_class=num_classes, camera_num=6, view_num=1)
    model.load_param(WEIGHT_PATH)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"CLIP-ReID(SIE-OLP) 모델 로드 완료, device: {device}")

    return model, device


def get_embedding(model, device, image_path, cam_id=0):
    """
    이미 로드된 model/device를 받아서, 사진 1장(image_path)의 embedding을 반환.
    cam_id: 진짜 카메라 번호(0-indexed)를 알면 넘겨줄 것 — SIE가 켜져있어서
            이 값이 실제로 결과에 영향을 줌 (TransReID 실험에서 확인된 것과 동일 원리).
            모르면 기본값 0 사용(정확도 저하 가능성 있음, 알면 반드시 넘길 것).
    """
    image = Image.open(image_path).convert("RGB")
    input_tensor = _transform(image).unsqueeze(0).to(device)

    cam_label = torch.tensor([cam_id]).to(device)

    with torch.no_grad():
        embedding = model(input_tensor, cam_label=cam_label, view_label=None)

    # CLIP-ReID는 여러 feature를 튜플/리스트로 반환할 수 있어 방어적으로 처리
    if isinstance(embedding, (tuple, list)):
        embedding = embedding[0]

    embedding = embedding.cpu().numpy().flatten()
    return embedding


# ── 이 파일을 단독 실행했을 때는 테스트용으로 동작 ──
if __name__ == "__main__":
    TEST_IMAGE_PATH = "data/crops/person_1.jpg"  # RF-DETR로 만든 crop 사진으로 테스트

    model, device = load_reid_model()
    embedding = get_embedding(model, device, TEST_IMAGE_PATH)

    print(f"\nEmbedding shape: {embedding.shape}")
    print(f"Embedding 앞부분 5개 값: {embedding[:5]}")