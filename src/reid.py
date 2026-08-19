"""
마일스톤 3: TransReID로 Person Crop에서 512차원 embedding 뽑기
사용법: python src/reid.py
"""

import sys
import torch
from PIL import Image
from torchvision import transforms

# ── TransReID 공식 repo의 모델 코드를 가져오기 위한 경로 추가 ──
sys.path.append("TransReID_official")

from model.make_model import make_model
from config import cfg

# ── 설정 ──────────────────────────────
CROP_IMAGE_PATH = "data/crops/person_1.jpg"     # 마일스톤1~2에서 만든 crop 사진
WEIGHT_PATH = "weights/transreid_market1501.pth"  # 방금 받은 weight
CONFIG_PATH = "TransReID_official/configs/Market/vit_transreid_stride.yml"  # Market-1501용 설정 파일

# ── 1. 설정 불러오기 ──────────────────────────
cfg.merge_from_file(CONFIG_PATH)
cfg.freeze()

# ── 2. 모델 구조 만들고 weight 로드 ──────────────
num_classes = 751  # Market-1501의 학습 시 사람 수 (구조 생성에만 필요, 추론엔 영향 없음)
model = make_model(cfg, num_class=num_classes, camera_num=6, view_num=1)
model.load_param(WEIGHT_PATH)
model.eval()  # 추론 모드로 전환 (학습 아님)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"모델 로드 완료, device: {device}")

# ── 3. 이미지 전처리 (TransReID가 요구하는 입력 형태로 변환) ──
transform = transforms.Compose([
    transforms.Resize((256, 128)),  # TransReID 기본 입력 크기
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

image = Image.open(CROP_IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가

# ── 4. Embedding 추출 ──────────────────────────
with torch.no_grad():  # 학습 아니므로 gradient 계산 끔 (속도/메모리 절약)
    embedding = model(input_tensor)

embedding = embedding.cpu().numpy().flatten()

# ── 5. 결과 확인 ──────────────────────────
print(f"\nEmbedding shape: {embedding.shape}")
print(f"Embedding 앞부분 5개 값: {embedding[:5]}")