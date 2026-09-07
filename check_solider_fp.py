import json
import glob
import os
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

THRESHOLD = 0.95
OUT = Path("solider_checks")
OUT.mkdir(exist_ok=True)

for jf in glob.glob("*_person_search.json"):
    with open(jf, encoding="utf-8") as f:
        data = json.load(f)

    for crop in data.get("crops", []):
        qpath = crop.get("crop_path") or crop.get("query_crop")
        if not qpath or not Path(qpath).is_file():
            continue

        qname = os.path.basename(qpath)
        qsrc = qname.rsplit("_person_", 1)[0]

        candidates = []

        for r in crop.get("results", []):
            cpath = r.get("crop_path")
            if not cpath or not Path(cpath).is_file():
                continue

            cname = os.path.basename(cpath)
            csrc = cname.rsplit("_person_", 1)[0]

            score = float(r.get("solider_score", r.get("score", 0)))

            # 같은 원본 이미지 제외
            if qsrc != csrc and score >= THRESHOLD:
                candidates.append((score, cpath))

        if not candidates:
            continue

        candidates.sort(reverse=True)

        # Query + 후보들
        items = [("QUERY", 1.0, qpath)]
        items += [("CAND", s, p) for s, p in candidates]

        cell_w = 300
        cell_h = 500
        cols = 4
        rows = (len(items) + cols - 1) // cols

        canvas = Image.new(
            "RGB",
            (cell_w * cols, cell_h * rows),
            "white"
        )
        draw = ImageDraw.Draw(canvas)

        for i, (kind, score, path) in enumerate(items):
            im = Image.open(path).convert("RGB")
            im = ImageOps.contain(im, (280, 410))

            x = (i % cols) * cell_w
            y = (i // cols) * cell_h

            canvas.paste(
                im,
                (x + (cell_w - im.width)//2, y)
            )

            name = os.path.basename(path)

            draw.text(
                (x + 5, y + 415),
                f"{kind}  score={score:.4f}",
                fill="black"
            )

            draw.text(
                (x + 5, y + 435),
                name[:42],
                fill="black"
            )

        out = OUT / f"{Path(qname).stem}_check.jpg"
        canvas.save(out, quality=95)

        print(f"SAVED: {out}")

print("\n완료")
