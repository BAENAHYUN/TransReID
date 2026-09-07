import json

from config import PipelineConfig
from registry import EmbedderRegistry
from router import Router
from rfdetr_adapter import from_rfdetr


# ------------------------------------------------------------
# 1. Load config
# ------------------------------------------------------------
cfg = PipelineConfig.load("pipeline.yaml")


# ------------------------------------------------------------
# 2. Load RF-DETR crop metadata
# ------------------------------------------------------------
with open(
    "data/crops/filter_stats.json",
    "r",
    encoding="utf-8",
) as f:
    crops = json.load(f)["crops"]


# ------------------------------------------------------------
# 3. Convert to normalized detections
# ------------------------------------------------------------
dets, fmt = from_rfdetr(crops)

print("input_format:", fmt)
print("total detections:", len(dets))


# ------------------------------------------------------------
# 4. Find one person and one non-person object
# ------------------------------------------------------------
person_det = None
object_det = None

for det in dets:
    label = getattr(det, "label", None)

    if label in cfg.person_labels:
        if person_det is None:
            person_det = det
    else:
        if object_det is None:
            object_det = det

    if person_det is not None and object_det is not None:
        break


if person_det is None:
    raise RuntimeError("No person detection found.")

if object_det is None:
    raise RuntimeError("No object detection found.")


print("\nSelected samples")
print("person:", person_det)
print("object:", object_det)


# ------------------------------------------------------------
# 5. Initialize embedders + router
# ------------------------------------------------------------
registry = EmbedderRegistry(cfg)

router = Router(
    cfg,
    registry,
    input_format=fmt,
)


# ------------------------------------------------------------
# 6. Person smoke test
#
# Expected:
#   SigLIP2  -> YES
#   IRRA     -> YES
#   SOLIDER  -> YES
#   DINOv2   -> NO
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PERSON EMBEDDING TEST")
print("=" * 60)

person_result = router.embed([person_det])

print("type:", type(person_result))
print("result:", person_result)


# ------------------------------------------------------------
# 7. Object smoke test
#
# Expected:
#   SigLIP2  -> YES
#   IRRA     -> NO
#   SOLIDER  -> NO
#   DINOv2   -> YES
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("OBJECT EMBEDDING TEST")
print("=" * 60)

object_result = router.embed([object_det])

print("type:", type(object_result))
print("result:", object_result)


print("\n" + "=" * 60)
print("ROUTER SMOKE TEST COMPLETED")
print("=" * 60)