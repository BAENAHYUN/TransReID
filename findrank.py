import json, sys
from pathlib import Path
d = json.load(open(sys.argv[1], encoding="utf-8"))
key = sys.argv[2]
for item in d["crops"]:
    for r in item["results"]:
        p = str(r.get("crop_path") or "")
        if key in p:
            print(f'{r["rank"]}위  qdrant={r.get("qdrant_score", 0):.5f}  {Path(p).name}')
