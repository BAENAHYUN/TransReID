"""
CLIP-ReID용 Market-1501 embedding DB 구축
TransReID 때 했던 것과 정확히 같은 방법론(무작위 샘플링, 진짜 cam_id 사용)으로
공정한 비교(apples-to-apples)가 되도록 만듦.

사용법: python src/RF-DETR/build_database_market1501_clip.py

결과물:
- data/embeddings_market1501_clip.npy
- data/metadata_market1501_clip.json



현재만 이 코드로 임시 - 가벼운 버전으로 CLIP-ReID rank-1하기 위해서 이 코드를 사용하고 잇음
"""

import os
import re
import json
import random
import sys

import numpy as np

sys.path.append("src/RF-DETR")
from reid_clip import load_reid_model, get_embedding

# ── 설정 (TransReID 실험 때와 동일 규모로 맞춤, 공정 비교를 위해) ──
SOURCE_DIR = "data/Market-1501-v15.09.15/bounding_box_test"
EMBEDDINGS_PATH = "data/embeddings_market1501_clip.npy"
METADATA_PATH = "data/metadata_market1501_clip.json"
MAX_IMAGES = 2000
RANDOM_SEED = 42  # TransReID 때와 동일한 시드 -> 똑같은 이미지 집합으로 비교됨


def parse_market1501_filename(filename):
    """
    '0002_c1s1_000451_03.jpg' -> (person_id=2, camera_id=0)
    junk(0000)/distractor(-1)는 (None, None) 반환.
    """
    match = re.match(r"(-?\d+)_c(\d)s\d", filename)
    if not match:
        return None, None

    person_id = int(match.group(1))
    camera_id = int(match.group(2)) - 1

    if person_id <= 0:
        return None, None

    return person_id, camera_id


def build_database():
    reid_model, device = load_reid_model()

    all_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(".jpg")]

    # TransReID 실험 때와 완전히 동일한 시드로 섞어서, 같은 이미지 집합으로 비교되게 함
    random.seed(RANDOM_SEED)
    random.shuffle(all_files)

    print(f"원본 폴더 파일 수: {len(all_files)}장, 이번엔 {MAX_IMAGES}장만 사용 (TransReID와 동일 시드)")

    all_embeddings = []
    metadata = []
    skipped = 0

    for i, filename in enumerate(all_files[:MAX_IMAGES]):
        person_id, camera_id = parse_market1501_filename(filename)

        if person_id is None:
            skipped += 1
            continue

        image_path = os.path.join(SOURCE_DIR, filename)

        # 진짜 camera_id를 넘김 (SIE-OLP가 실제로 활용하도록)
        embedding = get_embedding(reid_model, device, image_path, cam_id=camera_id)

        all_embeddings.append(embedding)
        metadata.append({
            "filename": filename,
            "path": image_path,
            "person_id": person_id,
            "camera_id": camera_id,
        })

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}장 처리됨...")

    all_embeddings = np.array(all_embeddings).astype("float32")
    np.save(EMBEDDINGS_PATH, all_embeddings)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {len(all_embeddings)}개 embedding 저장됨 (junk/distractor {skipped}개 제외)")
    print(f"  - {EMBEDDINGS_PATH}")
    print(f"  - {METADATA_PATH}")


if __name__ == "__main__":
    build_database()