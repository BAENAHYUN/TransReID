import inspect
import transformers
from PIL import Image
from transformers import AutoProcessor

MODEL = "google/siglip2-base-patch16-naflex"

p = AutoProcessor.from_pretrained(MODEL)
ip = p.image_processor

print("=== VERSION ===")
print("transformers       :", transformers.__version__)

print("\n=== PROCESSOR ===")
print("processor          :", type(p).__name__)
print("image_processor    :", type(ip).__name__)
print("model_input_names  :", getattr(ip, "model_input_names", None))
print("default max patches:", getattr(ip, "max_num_patches", None))

print("\n=== CALL ===")
print(inspect.signature(p.__call__))

img = Image.new("RGB", (96, 192), color=(128, 128, 128))

for patches in (256, 576):
    print(f"\n=== max_num_patches={patches} ===")

    out = p(
        images=[img],
        max_num_patches=patches,
        return_tensors="pt",
    )

    for k, v in out.items():
        print(
            k,
            "| type =", type(v).__name__,
            "| dtype =", getattr(v, "dtype", None),
            "| shape =", getattr(v, "shape", None),
        )
