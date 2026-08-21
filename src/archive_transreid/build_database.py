"""
마일스톤 4: data/sample_photos 안의 사진 전부 처리해서 embedding DB 만들기
사용법: python src/build_database.py

결과물:
- data/embeddings.npy   : 모든 embedding을 모아놓은 배열
- data/metadata.json    : 몇 번째 embedding이 어느 원본 사진/crop 파일인지 연결 정보
"""

import os
import json
import numpy as np

from detect import load_detect_model, detect_and_crop
from reid import load_reid_model, get_embedding

PHOTOS_DIR = "data/sample_photos"
CROPS_DIR = "data/crops"
EMBEDDINGS_PATH = "data/embeddings.npy"
METADATA_PATH = "data/metadata.json"


def build_database():
    # ── 1. 모델은 딱 한 번씩만 로드 (반복문 밖에서) ──
    detect_model = load_detect_model()
    reid_model, device = load_reid_model()

    # ── 2. data/sample_photos 안의 사진 목록 가져오기 ──
    photo_files = [
        f for f in os.listdir(PHOTOS_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"처리할 사진 {len(photo_files)}장: {photo_files}\n")

    all_embeddings = []
    metadata = []  # 각 embedding이 어느 파일에서 나왔는지 기록

    # ── 3. 사진마다 검출→crop→embedding 반복 ──
    for photo_file in photo_files:
        photo_path = os.path.join(PHOTOS_DIR, photo_file)
        photo_name_no_ext = os.path.splitext(photo_file)[0]

        print(f"--- {photo_file} 처리 중 ---")

        # 3-1. 검출 + crop (prefix로 사진마다 파일명 구분)
        crop_paths = detect_and_crop(
            detect_model,
            photo_path,
            output_dir=CROPS_DIR,
            prefix=f"{photo_name_no_ext}_",
        )

        # 3-2. 이 사진에서 나온 crop마다 embedding 추출
        for crop_path in crop_paths:
            embedding = get_embedding(reid_model, device, crop_path)
            all_embeddings.append(embedding)
            metadata.append({
                "original_photo": photo_file,
                "crop_path": crop_path,
            })

        print()

    # ── 4. 결과를 파일로 저장 ──
    all_embeddings = np.array(all_embeddings).astype("float32")
    np.save(EMBEDDINGS_PATH, all_embeddings)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"완료: 총 {len(all_embeddings)}개의 embedding 저장됨")
    print(f"  - {EMBEDDINGS_PATH} (shape: {all_embeddings.shape})")
    print(f"  - {METADATA_PATH}")


if __name__ == "__main__":
    build_database()
    