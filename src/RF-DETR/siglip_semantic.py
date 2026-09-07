"""
마일스톤 9: SigLIP 2로 이미지 전체의 의미(semantic) embedding 추출
- Person/Face(특정 사람 지문)와 달리, "이 사진이 대충 어떤 내용인지"를 벡터로 변환
- 나중에 텍스트 쿼리("빨간 옷 입은 사람")도 같은 벡터 공간으로 변환 가능 (M10 자연어 검색과 연결됨)

사용법: python src/RF-DETR/siglip_semantic.py
"""

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL_NAME = "google/siglip2-base-patch16-224"


def load_semantic_model():
    """
    SigLIP 2 모델을 딱 한 번 로드해서 반환.
    처음 실행 시 pretrained weight를 자동으로 다운로드함(Hugging Face 캐시).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print(f"SigLIP 2 모델 로드 완료, device: {device}")
    return model, processor, device


def get_image_embedding(model, processor, device, image_path):
    """
    이미지 1장(전체 사진, crop 아니어도 됨)을 semantic embedding으로 변환.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)

    if hasattr(embedding, "pooler_output"):
        embedding = embedding.pooler_output
    elif hasattr(embedding, "last_hidden_state"):
        embedding = embedding.last_hidden_state.mean(dim=1)

    return embedding.cpu().numpy().flatten()


def get_text_embedding(model, processor, device, text):
    """
    텍스트 쿼리(예: "빨간 옷 입은 사람")를 이미지와 같은 벡터 공간으로 변환.
    SigLIP2는 max_length=64로 학습되어 있어서, 이 설정을 반드시 지켜야
    정확한 결과가 나옴 (공식 문서에서 강조하는 부분).
    """
    inputs = processor(
        text=[text],
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        embedding = model.get_text_features(**inputs)

    if hasattr(embedding, "pooler_output"):
        embedding = embedding.pooler_output
    elif hasattr(embedding, "last_hidden_state"):
        embedding = embedding.last_hidden_state.mean(dim=1)

    return embedding.cpu().numpy().flatten()


def cosine_similarity(a, b):
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def sigmoid_probability(model, cosine, device):
    """
    SigLIP 공식 스케일링: cosine * logit_scale + logit_bias 를 거쳐 sigmoid.
    raw cosine은 anisotropy 때문에 절대값 판단이 어려우므로,
    실제 검색 threshold는 이 확률값 기준으로 잡는 게 안전함.
    """
    import torch

    logit_scale = model.logit_scale.exp().item()
    logit_bias = model.logit_bias.item()
    logit = cosine * logit_scale + logit_bias
    return float(torch.sigmoid(torch.tensor(logit)))


# ── 이 파일을 단독 실행했을 때는 테스트용으로 동작 ──
if __name__ == "__main__":
    TEST_IMAGE_PATH = "data/sample_photos/George_W_Bush_0001.jpg"

    # 정답 후보군 (이미지와 실제로 관련 있을 것으로 기대)
    POSITIVE_TEXTS = [
        "a man wearing a suit and tie",
        "an official government event",
        "a man wearing a suit without a tie",
        "a woman wearing a suit and tie",
        "a formal indoor event",
    ]

    # baseline 무관 문장군 — 여기 값들의 평균/최대가 "무관함의 기준선" 역할
    BASELINE_NEGATIVE_TEXTS = [
        "a dog running on the beach",
        "a bowl of fruit",
        "a kitchen with appliances",
        "a mountain landscape at sunset",
    ]

    model, processor, device = load_semantic_model()

    image_embedding = get_image_embedding(model, processor, device, TEST_IMAGE_PATH)
    print(f"\n이미지 embedding shape: {image_embedding.shape}")

    # baseline 계산
    baseline_scores = []
    print(f"\n=== Baseline (무관 문장) ===")
    for text in BASELINE_NEGATIVE_TEXTS:
        text_embedding = get_text_embedding(model, processor, device, text)
        cos = cosine_similarity(image_embedding, text_embedding)
        prob = sigmoid_probability(model, cos, device)
        baseline_scores.append(cos)
        print(f"  '{text}': cosine={cos:.4f}, prob={prob:.4f}")

    baseline_mean = sum(baseline_scores) / len(baseline_scores)
    baseline_max = max(baseline_scores)
    print(f"\n  baseline 평균 cosine: {baseline_mean:.4f} / 최대: {baseline_max:.4f}")

    # 후보 텍스트 평가 — baseline 대비 상대 비교
    print(f"\n=== 후보 텍스트 (baseline 대비 상대평가) ===")
    for text in POSITIVE_TEXTS:
        text_embedding = get_text_embedding(model, processor, device, text)
        cos = cosine_similarity(image_embedding, text_embedding)
        prob = sigmoid_probability(model, cos, device)
        margin = cos - baseline_mean
        flag = "✅" if cos > baseline_max else ("△" if margin > 0 else "❌")
        print(f"  '{text}': cosine={cos:.4f}, prob={prob:.4f}, margin={margin:+.4f} {flag}")