from PIL import Image, ImageDraw
from pathlib import Path

tracks = {
    "GT n017 track_0002": Path(r"C:\Users\Karsel\Desktop\Final_tool\TransReID\data\scvd_person_tracks_v1\n017_converted\track_0002"),
    "RANK1 n018 track_0003": Path(r"C:\Users\Karsel\Desktop\Final_tool\TransReID\data\scvd_person_tracks_v1\n018_converted\track_0003"),
}

thumb_w, thumb_h = 220, 320
margin = 12
label_h = 30
title_h = 40

rows = []

for title, folder in tracks.items():
    files = sorted(folder.glob("*.jpg"))
    imgs = []

    for f in files:
        img = Image.open(f).convert("RGB")
        img.thumbnail((thumb_w, thumb_h))

        canvas = Image.new("RGB", (thumb_w, thumb_h + label_h), "white")
        x = (thumb_w - img.width) // 2
        y = (thumb_h - img.height) // 2
        canvas.paste(img, (x, y))

        d = ImageDraw.Draw(canvas)
        d.text((5, thumb_h + 5), f.name, fill="black")
        imgs.append(canvas)

    row_w = len(imgs) * (thumb_w + margin) + margin
    row_h = title_h + thumb_h + label_h + margin

    row = Image.new("RGB", (row_w, row_h), "white")
    d = ImageDraw.Draw(row)
    d.text((margin, 10), title, fill="black")

    for i, img in enumerate(imgs):
        row.paste(img, (margin + i * (thumb_w + margin), title_h))

    rows.append(row)

final_w = max(r.width for r in rows)
final_h = sum(r.height for r in rows)

out = Image.new("RGB", (final_w, final_h), "white")

y = 0
for r in rows:
    out.paste(r, (0, y))
    y += r.height

out_path = Path(r"C:\Users\Karsel\Desktop\Final_tool\TransReID\results\n017_vs_n018_track_compare.jpg")
out_path.parent.mkdir(parents=True, exist_ok=True)
out.save(out_path, quality=95)

print(out_path)
