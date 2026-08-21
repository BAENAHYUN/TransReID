"""
Market-1501 데이터셋 폴더 전체에서, 카메라별로 몇 장씩 골고루 자동 선별해서
data/market_test/ 로 복사하는 스크립트.

사용법:
1. 아래 SOURCE_DIR을 실제 압축 푼 폴더 경로로 수정
   (예: "bounding_box_test" 폴더 안쪽 경로)
2. python src/select_market_samples.py 실행
"""

import os
import re
import shutil
from collections import defaultdict

# ── 설정 (환경마다 다를 수 있어서 상수로 둠, query 데이터 아님) ──
SOURCE_DIR = "data/Market-1501-v15.09.15/bounding_box_test"  # 본인 압축 해제 경로로 수정
TARGET_DIR = "data/market_test"
IMAGES_PER_CAMERA = 3  # 카메라 하나당 몇 장씩 뽑을지


def parse_camera_id(filename):
    """'0002_c1s1_000451_03.jpg' -> 1 (1-indexed 원본 그대로)"""
    match = re.search(r"_c(\d)s\d", filename)
    if match:
        return int(match.group(1))
    return None


def select_and_copy_samples():
    # 기존 target 폴더 있으면 비우고 새로 시작 (이전 실험 결과 섞이지 않게)
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR, exist_ok=True)

    # ── 1. 카메라별로 파일들을 그룹핑 ──
    camera_groups = defaultdict(list)

    all_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(".jpg")]
    print(f"원본 폴더 전체 파일 수: {len(all_files)}장")

    for f in all_files:
        cam_id = parse_camera_id(f)
        if cam_id is not None:
            camera_groups[cam_id].append(f)

    print(f"발견된 카메라 종류: {sorted(camera_groups.keys())}")

    # ── 2. 카메라마다 IMAGES_PER_CAMERA장씩 뽑아서 복사 ──
    copied_count = 0
    for cam_id in sorted(camera_groups.keys()):
        files_from_this_camera = camera_groups[cam_id][:IMAGES_PER_CAMERA]

        for f in files_from_this_camera:
            src_path = os.path.join(SOURCE_DIR, f)
            dst_path = os.path.join(TARGET_DIR, f)
            shutil.copy(src_path, dst_path)
            copied_count += 1

        print(f"카메라 {cam_id}: {len(files_from_this_camera)}장 복사됨")

    print(f"\n완료: 총 {copied_count}장을 {TARGET_DIR} 에 복사했습니다 "
          f"({len(camera_groups)}개 카메라 x {IMAGES_PER_CAMERA}장)")


if __name__ == "__main__":
    select_and_copy_samples()