import time
import torch
from rfdetr import RFDETRMedium

TEST_IMAGE_PATH = "data/sample_photos/George_W_Bush_0535.jpg"
N_WARMUP = 5
N_RUNS = 30
CONF_THRESHOLD = 0.5


def benchmark(dtype_str, image_path):
    model = RFDETRMedium()
    model.inference(compile=False, dtype=dtype_str)

    # 워밍업 — 첫 몇 번은 CUDA 컨텍스트/커널 초기화 오버헤드가 섞여서 버림
    for _ in range(N_WARMUP):
        _ = model.predict(image_path, threshold=CONF_THRESHOLD)
        torch.cuda.synchronize()

    # 본 측정
    times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()  # 이전 작업 확실히 끝난 시점부터
        start = time.perf_counter()

        _ = model.predict(image_path, threshold=CONF_THRESHOLD)

        torch.cuda.synchronize()  # GPU 작업 종료 시점까지 대기
        end = time.perf_counter()

        times.append((end - start) * 1000)  # ms 단위

    avg = sum(times) / len(times)
    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]

    print(f"[{dtype_str}] 평균: {avg:.2f}ms, 중앙값: {median:.2f}ms, "
          f"최소: {min(times):.2f}ms, 최대: {max(times):.2f}ms")

    del model
    torch.cuda.empty_cache()

    return avg


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"이미지: {TEST_IMAGE_PATH}\n")

    avg_fp32 = benchmark("float32", TEST_IMAGE_PATH)
    avg_fp16 = benchmark("float16", TEST_IMAGE_PATH)

    print(f"\n결과: FP16이 FP32 대비 {'느림' if avg_fp16 > avg_fp32 else '빠름'} "
          f"({avg_fp16/avg_fp32:.2f}x)")