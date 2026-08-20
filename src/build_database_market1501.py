"""
Market-1501용 embedding DB 생성 (RT-DETR 검출 생략, 이미 crop된 사진이라 바로 reid)
사용법: python src/build_database_market1501.py

결과물:
- data/embeddings_market1501.npy
- data/metadata_market1501.json (person_id, camera_id 포함 — 자동 평가용 정답 라벨)
"""

import os
import re
import json
import random
import numpy as np

import sys
sys.path.append("src")
from reid import load_reid_model, get_embedding

# ── 설정 (query가 아니라 환경설정이라 상수로 둠) ──
SOURCE_DIR = "data/Market-1501-v15.09.15/bounding_box_test"  # 본인 경로로 수정
EMBEDDINGS_PATH = "data/embeddings_market1501.npy"
METADATA_PATH = "data/metadata_market1501.json"
MAX_IMAGES = 2000  # 전체(1만 9천장+) 다 하면 오래 걸리니, 일단 이만큼만 (필요시 조정)
RANDOM_SEED = 42  # 매번 같은 결과 재현되도록 고정


def parse_market1501_filename(filename):
    """
    '0002_c1s1_000451_03.jpg' -> (person_id=2, camera_id=0)
    person_id가 -1이거나 0000이면 정식 인물이 아닌 잡음(junk/distractor) 이미지라 None 반환.
    """
    match = re.match(r"(-?\d+)_c(\d)s\d", filename)
    if not match:
        return None, None

    person_id = int(match.group(1))
    camera_id = int(match.group(2)) - 1  # c1 -> 0

    if person_id <= 0:  # -1(distractor) 또는 0000(junk)은 평가 대상 아님
        return None, None

    return person_id, camera_id


def build_database_market1501():
    reid_model, device = load_reid_model()

    all_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(".jpg")]

    # junk(0000)/distractor(-1) 파일명이 폴더 앞쪽에 몰려있어서,
    # 그냥 앞에서부터 자르면 전부 걸러지는 파일만 뽑힘 -> 무작위로 섞어서 방지
    random.seed(RANDOM_SEED)
    random.shuffle(all_files)

    print(f"원본 폴더 파일 수: {len(all_files)}장, 이번엔 {MAX_IMAGES}장만 사용 (무작위 샘플링)")

    all_embeddings = []
    metadata = []
    skipped = 0

    for i, filename in enumerate(all_files[:MAX_IMAGES]):
        person_id, camera_id = parse_market1501_filename(filename)

        if person_id is None:
            skipped += 1
            continue

        image_path = os.path.join(SOURCE_DIR, filename)

        # RT-DETR 검출/crop 생략 — 이미 사람만 잘려있는 사진이라 바로 reid
        embedding = get_embedding(reid_model, device, image_path, cam_id=camera_id)

        all_embeddings.append(embedding)
        metadata.append({
            "filename": filename,
            "path": image_path,
            "person_id": person_id,     # 정답 라벨 (자동 평가용)
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
    build_database_market1501()