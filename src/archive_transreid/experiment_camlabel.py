"""
Market-1501 데이터로 cam_label=0(더미)이 진짜 문제인지 검증하는 실험.

Market-1501 파일명 규칙: {personID}_c{cameraID}s{seq}_{frame}_{box}.jpg
예: 0002_c1s1_000451_03.jpg -> 카메라 1번

사용법: python src/experiment_camlabel.py

이 실험이 답해주는 것

마지막 출력의 "유사도" 숫자가:

1.0에 아주 가까우면(0.99 이상) 
    → cam_label=0을 넣든 진짜 값을 넣든 결과가 거의 안 바뀐다는 뜻 → 아까 제가 "SIE 때문"이라고 짐작한 게 틀렸다는 증거
눈에 띄게 다르면(0.9 이하 등) 
    → cam_label이 실제로 embedding에 큰 영향을 준다는 뜻 → 지금까지 관찰하신 "각도/표정 편향"이 이것 때문일 가능성이 높아짐

"""

import os
import re
import sys
import torch
import numpy as np

sys.path.append("src")
from reid import load_reid_model, _transform  # 기존 reid.py 재사용
from PIL import Image

MARKET_TEST_DIR = "data/market_test"  # 여기에 10~20장 미리 복사해두기


def parse_camera_id(filename):
    """
    '0002_c1s1_000451_03.jpg' -> 카메라 ID(0-indexed)를 정수로 반환.
    Market-1501 규칙에 안 맞는 파일명이면 None 반환.
    """
    match = re.search(r"_c(\d)s\d", filename)
    if match:
        return int(match.group(1)) - 1  # c1 -> 0, c2 -> 1 ... (0-indexed로 맞춤)
    return None


def get_embedding_with_cam(model, device, image_path, cam_id):
    """
    reid.py의 get_embedding과 거의 동일하지만, cam_label을 더미(0)가 아니라
    진짜 값으로 넣을 수 있게 만든 실험용 버전.
    """
    image = Image.open(image_path).convert("RGB")
    input_tensor = _transform(image).unsqueeze(0).to(device)

    cam_label = torch.tensor([cam_id]).to(device)
    view_label = torch.tensor([0]).to(device)  # view는 Market-1501에 라벨 없어서 그대로 0

    with torch.no_grad():
        embedding = model(input_tensor, cam_label=cam_label, view_label=view_label)

    return embedding.cpu().numpy().flatten()


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    files = [f for f in os.listdir(MARKET_TEST_DIR) if f.lower().endswith(".jpg")]
    print(f"테스트 대상: {len(files)}장\n")

    model, device = load_reid_model()

    dummy_embeddings = {}
    real_embeddings = {}

    for f in files:
        path = os.path.join(MARKET_TEST_DIR, f)
        cam_id = parse_camera_id(f)

        if cam_id is None:
            print(f"경고: {f} 는 Market-1501 파일명 규칙과 안 맞음, 건너뜀")
            continue

        # 더미(0) 버전
        dummy_embeddings[f] = get_embedding_with_cam(model, device, path, cam_id=0)
        # 진짜 카메라 라벨 버전
        real_embeddings[f] = get_embedding_with_cam(model, device, path, cam_id=cam_id)

        print(f"{f}: 실제 카메라={cam_id+1}")

    # 같은 사진 쌍끼리, 더미 버전 vs 진짜 버전의 embedding이 얼마나 다른지 비교
    print("\n--- 더미(0) vs 진짜 cam_label, 같은 사진의 embedding 차이 ---")
    for f in dummy_embeddings:
        sim = cosine_similarity(dummy_embeddings[f], real_embeddings[f])
        print(f"{f}: 두 embedding 간 유사도 = {sim:.4f}  (1.0에 가까울수록 '차이 없음'을 의미)")