import fiftyone as fo
import fiftyone.zoo as foz


# ============================================================
# 설정
# ============================================================

DOWNLOAD_ROOT = r"C:\datasets"

NUM_IMAGES = 100_000
SEED = 42


# ============================================================
# FiftyOne 데이터셋 저장 위치
# ============================================================

fo.config.dataset_zoo_dir = DOWNLOAD_ROOT


# ============================================================
# Open Images V7 - Train 100,000장 다운로드
# ============================================================

print("=" * 70)
print("Open Images V7 - train 100,000 images")
print("=" * 70)

dataset = foz.load_zoo_dataset(
    "open-images-v7",

    split="train",

    # RF-DETR 검증에도 활용 가능
    label_types=["detections"],

    # 랜덤 10만 장
    max_samples=NUM_IMAGES,
    shuffle=True,
    seed=SEED,

    # FiftyOne 내부 dataset 이름
    dataset_name="open-images-v7-train-100k",
)

print()
print("=" * 70)
print("다운로드 완료")
print("=" * 70)

print(f"이미지 수 : {len(dataset):,}")
print(f"저장 루트 : {DOWNLOAD_ROOT}")

print("\n샘플 이미지 경로:")
for sample in dataset.take(5):
    print(sample.filepath)