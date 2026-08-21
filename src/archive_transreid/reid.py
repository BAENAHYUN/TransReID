"""
마일스톤 3: TransReID로 Person Crop에서 embedding 뽑기 (함수화 버전)
- load_reid_model(): 모델을 딱 한 번만 로드
- get_embedding(model, device, image_path): 사진 경로를 인자로 받아 embedding 반환
"""

import sys
import torch
from PIL import Image
from torchvision import transforms

# ── TransReID 공식 repo의 모델 코드를 가져오기 위한 경로 추가 ──
sys.path.append("TransReID_official")

from model.make_model import make_model
from config import cfg

# ── 고정 설정 (query가 아니라 "환경설정"이라 그대로 상수로 둬도 됨) ──
WEIGHT_PATH = "weights/transreid_market1501.pth"
CONFIG_PATH = "TransReID_official/configs/Market/vit_transreid_stride.yml"

_transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_reid_model():
    """
    TransReID 모델을 딱 한 번 로드해서 반환.
    M4에서 사진 여러 장 처리할 때, 이 함수는 딱 1번만 호출하고
    나온 model/device를 계속 재사용해야 함 (매번 부르면 매우 느려짐).
    """
    cfg.merge_from_file(CONFIG_PATH)
    cfg.freeze()

    num_classes = 751  # Market-1501 학습 시 사람 수, 구조 생성에만 필요
    model = make_model(cfg, num_class=num_classes, camera_num=6, view_num=1)
    model.load_param(WEIGHT_PATH)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"모델 로드 완료, device: {device}")

    return model, device

# 1. 함수 정의 줄 - cam_id=0 파라미터 추가
def get_embedding(model, device, image_path, cam_id=0):   # ← 여기 cam_id=0 추가됨
    """
    이미 로드된 model/device를 받아서, 사진 1장(image_path)의 embedding을 반환.
    image_path는 인자로 받음 — 절대 하드코딩하지 않음.
     cam_id: 카메라 라벨(0-indexed). 모르면 기본값 0 그대로 사용   # ← 이 줄 추가됨
    """
    image = Image.open(image_path).convert("RGB")
    input_tensor = _transform(image).unsqueeze(0).to(device)

    #0을 cam_id로 교체
    #cam_label = torch.tensor([0]).to(device)
    cam_label = torch.tensor([cam_id]).to(device)   # ← 원래는 torch.tensor([0])이었음
    view_label = torch.tensor([0]).to(device)

    with torch.no_grad():
        embedding = model(input_tensor, cam_label=cam_label, view_label=view_label)

    embedding = embedding.cpu().numpy().flatten()
    return embedding


# ── 이 파일을 단독 실행했을 때(python src/reid.py)는 테스트용으로 동작 ──
if __name__ == "__main__":
    TEST_IMAGE_PATH = "data/crops/person_1.jpg"

    model, device = load_reid_model()
    embedding = get_embedding(model, device, TEST_IMAGE_PATH)

    print(f"\nEmbedding shape: {embedding.shape}")
    print(f"Embedding 앞부분 5개 값: {embedding[:5]}")