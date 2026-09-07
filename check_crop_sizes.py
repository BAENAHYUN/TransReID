from pathlib import Path
from PIL import Image
import numpy as np

root = Path(r".\data\crops")

files = [
    p for p in root.rglob("*")
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
]

files = files[:5000]

ws, hs = [], []

for p in files:
    try:
        with Image.open(p) as im:
            w, h = im.size
        ws.append(w)
        hs.append(h)
    except:
        pass

ws = np.array(ws)
hs = np.array(hs)

print("N =", len(ws))
print("width  median =", np.median(ws))
print("width  p90    =", np.percentile(ws, 90))
print("height median =", np.median(hs))
print("height p90    =", np.percentile(hs, 90))
print("area median   =", np.median(ws * hs))
print("area p90      =", np.percentile(ws * hs, 90))
