import json, glob, os
rows = []
for f in sorted(glob.glob("p_*.json") + glob.glob("o_*.json") + glob.glob("_compat.json")):
    d = json.load(open(f, encoding="utf-8"))
    for c in d.get("crops", []):
        r = c.get("retrieval") or {}
        g = c.get("grouping") or {}
        res = c.get("results") or []
        rows.append((
            os.path.basename(f),
            c.get("kind"),
            ",".join(r.get("stage1_vectors") or []),
            r.get("stage1_k"),
            r.get("rerank_name"),
            r.get("final_score_type"),
            len(res),
            (res[0].get("point_id") if res else "-")[:8],
            round(float(res[0].get("pre_qwen_score", 0)), 4) if res else "-",
            g.get("collapsed_count"),
        ))
        break
hdr = ("file","kind","vectors","k","rerank","final_score","n","top_id","top_score","collapsed")
w = [max(len(str(x[i])) for x in [hdr]+rows) for i in range(len(hdr))]
fmt = "  ".join("{:<%d}" % n for n in w)
print(fmt.format(*hdr)); print("-" * (sum(w)+2*len(w)))
for r in rows: print(fmt.format(*[str(x) for x in r]))
