"""
마일스톤 10: Qwen3-VL로 ① 이미지 캡션 생성 ② 텍스트/이미지 임베딩(검색용)
- 캡션 생성: Qwen3-VL-2B-Instruct (생성형 모델)
- 임베딩: Qwen3-VL-Embedding-2B (검색 전용 별도 모델, sentence_transformers로 사용)
  -> SigLIP2와 마찬가지로 텍스트/이미지가 같은 벡터 공간에 놓임

⚠️ 2B급 모델 2개를 쓰므로 GPU 메모리(VRAM) 부담이 큼. GTX 1060(6GB)에서는
   OOM(메모리 부족)이 날 수 있음 — 그 경우 8-bit/4-bit 양자화 로딩을 고려해야 함.

사용법: python src/RF-DETR/qwen_vlm.py
"""

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from sentence_transformers import SentenceTransformer

CAPTION_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"


def load_caption_model():
    """
    이미지 캡션 생성용 Qwen3-VL-2B-Instruct 모델 로드.
    """
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        CAPTION_MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(CAPTION_MODEL_NAME)

    print(f"Qwen3-VL 캡션 모델 로드 완료")
    return model, processor


def generate_caption(model, processor, image_path, prompt="Describe this image.", low_temperature=True):
    """
    이미지 1장을 보고 자연어 캡션(설명 문장)을 생성.
    low_temperature=True면 생성 무작위성을 낮춰서, 더 보수적이고 일관된 답을 유도
    (환각/실행마다 다른 결과 방지에 도움).
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)

    generate_kwargs = {"max_new_tokens": 100}
    if low_temperature:
        # temperature를 낮추고 do_sample=False로 두면, 매번 가장 확률 높은
        # (=가장 "안전한", 근거 부족한 추측을 덜 하는) 답을 선택하게 됨
        generate_kwargs["do_sample"] = False
        generate_kwargs["temperature"] = None  # do_sample=False일 땐 무시되지만 명시적으로 끔

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)

    # 입력 프롬프트 부분을 잘라내고, 새로 생성된 텍스트만 추출
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return caption.strip()


def load_embedding_model():
    """
    검색용 Qwen3-VL-Embedding-2B 모델 로드 (sentence_transformers 사용).
    """
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Qwen3-VL Embedding 모델 로드 완료")
    return model


def get_qwen_text_embedding(embedding_model, text):
    """
    텍스트 쿼리를 embedding으로 변환 (이미지 embedding과 같은 벡터 공간).
    """
    embedding = embedding_model.encode([text])
    return embedding[0]


def get_qwen_image_embedding(embedding_model, image_path):
    """
    이미지를 embedding으로 변환.
    """
    image = Image.open(image_path).convert("RGB")
    embedding = embedding_model.encode([image])
    return embedding[0]


# ── 이 파일을 단독 실행했을 때는 테스트용으로 동작 ──
if __name__ == "__main__":
    import numpy as np

    TEST_IMAGE_PATH = "data/sample_photos/George_W_Bush_0001.jpg"

    print("=== ① 캡션 생성 테스트 ===")
    caption_model, caption_processor = load_caption_model()

    caption = generate_caption(caption_model, caption_processor, TEST_IMAGE_PATH)
    print(f"생성된 캡션(기본 프롬프트): {caption}")

    # 환각(hallucination) 검증: "이 사람이 누구인지 확신하냐"고 되물어서,
    # 모델이 근거 없이 확신하는지 vs 신중하게 답하는지 확인
    caption_check = generate_caption(
        caption_model, caption_processor, TEST_IMAGE_PATH,
        prompt="Describe only what you can visually see in this image: clothing, colors, background. Do not guess the person's identity."
    )
    print(f"\n생성된 캡션(신원 추측 배제 프롬프트): {caption_check}")

    # 환각 방지 강화 버전: 더 명시적으로 "모르면 모른다고 답하라"고 지시
    caption_no_hallucination = generate_caption(
        caption_model, caption_processor, TEST_IMAGE_PATH,
        prompt=(
            "Describe only visually verifiable facts: clothing color/style, "
            "background color/pattern, approximate age range, expression. "
            "If you are not 100% certain about a person's name or identity, "
            "you must NOT guess or state a name. Say 'unidentified person' instead."
        )
    )
    print(f"\n생성된 캡션(환각 방지 강화 프롬프트): {caption_no_hallucination}")

    # 캡션 모델 GPU 메모리 해제 (임베딩 모델과 동시에 못 올릴 수 있어서)
    del caption_model
    torch.cuda.empty_cache()

    print("\n=== ② 텍스트/이미지 임베딩 테스트 ===")
    embedding_model = load_embedding_model()

    image_embedding = get_qwen_image_embedding(embedding_model, TEST_IMAGE_PATH)
    print(f"이미지 embedding shape: {image_embedding.shape}")

    # SigLIP2 때와 동일한 텍스트 세트로 비교 (같은 기준으로 두 모델 비교 가능하게)
    TEST_TEXTS = [
        "a man wearing a suit and tie",
        "an official government event",
        "an outdoor press conference",
        "a dog running on the beach",
        "a bowl of fruit",
        "a man wearing a red tie",
    ]

    print(f"\n=== 텍스트별 유사도 비교 (SigLIP2와 동일한 텍스트 세트) ===")
    for text in TEST_TEXTS:
        text_embedding = get_qwen_text_embedding(embedding_model, text)
        similarity = np.dot(image_embedding, text_embedding) / (
            np.linalg.norm(image_embedding) * np.linalg.norm(text_embedding)
        )
        print(f"  '{text}': {similarity:.4f}")