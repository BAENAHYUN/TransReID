"""
benchmark_resolution.py (수정판)
- optimize_for_inference() 없이, resolution 자체의 순수 효과만 비교
"""
import time
import torch
from rfdetr import RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

TEST_IMAGE_PATH = "data/sample_photos/manypeople.jpg"
CONF_THRESHOLD = 0.5
N_WARMUP = 5
N_RUNS = 30
RESOLUTIONS = [576, 512, 448, 384]


def benchmark_resolution(model, resolution, image_path):
    shape = (resolution, resolution)

    for _ in range(N_WARMUP):
        _ = model.predict(image_path, threshold=CONF_THRESHOLD, shape=shape)
        torch.cuda.synchronize()

    times, person_counts, confidences = [], [], []

    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        detections = model.predict(image_path, threshold=CONF_THRESHOLD, shape=shape)
        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)

        count = 0
        confs = []
        for i in range(len(detections)):
            if COCO_CLASSES[int(detections.class_id[i])] == "person":
                count += 1
                confs.append(float(detections.confidence[i]))
        person_counts.append(count)
        confidences.extend(confs)

    avg_time = sum(times) / len(times)
    avg_person = sum(person_counts) / len(person_counts)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    print(f"[resolution={resolution}] 평균 시간: {avg_time:.2f}ms, "
          f"평균 검출 인원: {avg_person:.1f}명, 평균 confidence: {avg_conf:.3f}")

    return avg_time, avg_person, avg_conf


if __name__ == "__main__":
    model = RFDETRMedium()
    # model.inference(...) 최적화 호출 제거 — resolution마다 shape가 바뀌니 고정 최적화와 충돌함

    results = {}
    for res in RESOLUTIONS:
        results[res] = benchmark_resolution(model, res, TEST_IMAGE_PATH)

    print("\n=== 요약 (기준: 576) ===")
    baseline_time, baseline_person, _ = results[576]
    for res, (t, p, c) in results.items():
        speedup = baseline_time / t
        print(f"resolution={res}: 속도 {speedup:.2f}x, 검출 인원 {p:.1f}명 (기준 {baseline_person:.1f}명)")