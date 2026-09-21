#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DINOv2 Giant+Registers CLS: CenterCrop224 vs Aspect-Ratio Pad224 A/B benchmark.

Manifest CSV:
group_id,role,path
obj001,query,C:\...\query.jpg
obj001,positive,C:\...\same1.jpg
obj001,negative,C:\...\other1.jpg

Run:
python .\tools\benchmark_g14_cls_center_vs_pad.py `
  --manifest .\tools\dinov2_benchmark_manifest.csv `
  --batch-size 4
"""

from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel
import torchvision.transforms as T

MODEL_ID = "facebook/dinov2-with-registers-giant"
IMAGE_SIZE = 224
OFFICIAL_RESIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_MEAN_RGB = tuple(int(round(v * 255)) for v in IMAGENET_MEAN)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def load_manifest(path: Path):
    groups = defaultdict(lambda: {"query": [], "positive": [], "negative": []})
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"group_id", "role", "path"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"manifest columns must include {sorted(required)}")
        for row in reader:
            gid = str(row["group_id"]).strip()
            role = str(row["role"]).strip().lower()
            raw = str(row["path"]).strip()
            if not gid or role not in {"query","positive","negative"} or not raw:
                continue
            p = Path(raw).expanduser()
            p = (path.parent / p).resolve() if not p.is_absolute() else p.resolve()
            if not p.is_file():
                raise FileNotFoundError(p)
            if p.suffix.lower() not in EXTS:
                raise ValueError(f"unsupported image: {p}")
            groups[gid][role].append(p)
    if not groups:
        raise ValueError("manifest is empty")
    for gid,g in groups.items():
        if len(g["query"]) != 1:
            raise ValueError(f"{gid}: query must be exactly 1")
        if not g["positive"] or not g["negative"]:
            raise ValueError(f"{gid}: positive and negative are required")
    return dict(groups)

def to_rgb(img):
    return img if img.mode == "RGB" else img.convert("RGB")

class Preprocessor:
    def __init__(self):
        bicubic = T.InterpolationMode.BICUBIC
        self.center = T.Compose([
            T.Lambda(to_rgb),
            T.Resize(OFFICIAL_RESIZE, interpolation=bicubic),
            T.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.norm = T.Compose([
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def center_crop(self, img):
        return self.center(img)

    def aspect_pad(self, img):
        img = to_rgb(img)
        w,h = img.size
        if w <= 0 or h <= 0:
            raise ValueError(f"invalid image size: {(w,h)}")
        scale = min(IMAGE_SIZE / w, IMAGE_SIZE / h)
        nw = max(1, min(IMAGE_SIZE, int(round(w*scale))))
        nh = max(1, min(IMAGE_SIZE, int(round(h*scale))))
        resized = img.resize((nw,nh), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (IMAGE_SIZE,IMAGE_SIZE), IMAGENET_MEAN_RGB)
        left = (IMAGE_SIZE-nw)//2
        top = (IMAGE_SIZE-nh)//2
        canvas.paste(resized, (left,top))
        return self.norm(canvas)

def choose_device(raw):
    return torch.device(raw) if raw else torch.device("cuda" if torch.cuda.is_available() else "cpu")

def choose_dtype(device, fp32):
    if fp32 or device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

@torch.inference_mode()
def embed_paths(model, paths, preprocess_fn, device, dtype, batch_size):
    outs=[]
    for start in range(0,len(paths),batch_size):
        chunk=paths[start:start+batch_size]
        tensors=[]
        for p in chunk:
            with Image.open(p) as im:
                tensors.append(preprocess_fn(im.copy()))
        batch=torch.stack(tensors).to(device=device,dtype=dtype)
        tokens=model(pixel_values=batch).last_hidden_state
        cls=F.normalize(tokens[:,0].float(), dim=-1)
        outs.append(cls.cpu().numpy())
        print(f"  embedded {min(start+batch_size,len(paths)):,}/{len(paths):,}", end="\r")
    print()
    return np.concatenate(outs,axis=0).astype(np.float32,copy=False)

def group_metrics(q,pos,neg):
    ps=pos@q
    ns=neg@q
    labels=np.concatenate([np.ones(len(ps),dtype=np.int32),np.zeros(len(ns),dtype=np.int32)])
    scores=np.concatenate([ps,ns])
    order=np.argsort(-scores)
    ranked=labels[order]
    ranks=np.flatnonzero(ranked==1)+1
    first=int(ranks[0])
    return {
        "R@1": float(first<=1),
        "R@5": float(first<=5),
        "MRR": float(1.0/first),
        "first_positive_rank": first,
        "positive_mean": float(ps.mean()),
        "negative_mean": float(ns.mean()),
        "separation_gap": float(ps.mean()-ns.mean()),
        "strict_margin": float(ps.min()-ns.max()),
        "top1_positive": bool(ranked[0]==1),
    }

def aggregate(per_group):
    vals=list(per_group.values())
    return {
        "groups": len(vals),
        "R@1": float(np.mean([v["R@1"] for v in vals])),
        "R@5": float(np.mean([v["R@5"] for v in vals])),
        "MRR": float(np.mean([v["MRR"] for v in vals])),
        "positive_mean": float(np.mean([v["positive_mean"] for v in vals])),
        "negative_mean": float(np.mean([v["negative_mean"] for v in vals])),
        "separation_gap_mean": float(np.mean([v["separation_gap"] for v in vals])),
        "strict_margin_mean": float(np.mean([v["strict_margin"] for v in vals])),
        "strict_margin_positive_rate": float(np.mean([v["strict_margin"]>0 for v in vals])),
    }

def evaluate(groups, vec_map):
    out={}
    for gid,g in groups.items():
        q=vec_map[str(g["query"][0])]
        pos=np.stack([vec_map[str(p)] for p in g["positive"]])
        neg=np.stack([vec_map[str(p)] for p in g["negative"]])
        out[gid]=group_metrics(q,pos,neg)
    return out

def score_tuple(a):
    return (a["R@1"],a["R@5"],a["MRR"],a["separation_gap_mean"],a["strict_margin_mean"])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--batch-size",type=int,default=4)
    ap.add_argument("--device",default=None)
    ap.add_argument("--fp32",action="store_true")
    ap.add_argument("--local-files-only",action="store_true")
    ap.add_argument("--cache-dir",default=None)
    ap.add_argument("--output-dir",default="outputs/dinov2_g14_center_vs_pad")
    args=ap.parse_args()

    manifest=Path(args.manifest).expanduser().resolve()
    groups=load_manifest(manifest)

    all_paths=[]; seen=set()
    for g in groups.values():
        for role in ("query","positive","negative"):
            for p in g[role]:
                if str(p) not in seen:
                    seen.add(str(p)); all_paths.append(p)

    device=choose_device(args.device)
    dtype=choose_dtype(device,args.fp32)

    print("="*88)
    print("DINOv2 G/14 REGISTERS CLS — CENTER CROP vs ASPECT-RATIO PAD")
    print("="*88)
    print("model        :",MODEL_ID)
    print("groups       :",len(groups))
    print("unique images:",len(all_paths))
    print("device       :",device)
    print("dtype        :",dtype)
    print("padding RGB  :",IMAGENET_MEAN_RGB)

    kwargs={}
    if args.cache_dir: kwargs["cache_dir"]=args.cache_dir
    if args.local_files_only: kwargs["local_files_only"]=True

    print("\n[LOAD MODEL]")
    model=AutoModel.from_pretrained(MODEL_ID,**kwargs)
    model=model.to(device=device,dtype=dtype).eval()
    for p in model.parameters(): p.requires_grad_(False)

    hidden=int(model.config.hidden_size)
    regs=int(getattr(model.config,"num_register_tokens",0))
    patch=int(getattr(model.config,"patch_size",14))
    print("hidden dim   :",hidden)
    print("registers    :",regs)
    print("patch size   :",patch)
    if hidden != 1536: raise RuntimeError(f"expected 1536, got {hidden}")
    if patch != 14: raise RuntimeError(f"expected patch 14, got {patch}")

    prep=Preprocessor()
    results={}
    for key,fn in [("A_center_crop224_cls",prep.center_crop),("C_aspect_pad224_cls",prep.aspect_pad)]:
        print("\n[RUN]",key)
        vecs=embed_paths(model,all_paths,fn,device,dtype,args.batch_size)
        if not np.isfinite(vecs).all(): raise RuntimeError(f"{key}: NaN/Inf")
        vm={str(p):v for p,v in zip(all_paths,vecs)}
        pg=evaluate(groups,vm)
        agg=aggregate(pg)
        results[key]={"aggregate":agg,"per_group":pg}
        print(f"  R@1={agg['R@1']:.4f} R@5={agg['R@5']:.4f} MRR={agg['MRR']:.4f}")
        print(f"  gap={agg['separation_gap_mean']:.6f} margin={agg['strict_margin_mean']:.6f}")

    a=results["A_center_crop224_cls"]["aggregate"]
    c=results["C_aspect_pad224_cls"]["aggregate"]
    winner="C_aspect_pad224_cls" if score_tuple(c)>score_tuple(a) else "A_center_crop224_cls"
    keys=["R@1","R@5","MRR","positive_mean","negative_mean","separation_gap_mean","strict_margin_mean","strict_margin_positive_rate"]
    delta={k:float(c[k]-a[k]) for k in keys}

    out=Path(args.output_dir).expanduser().resolve()/datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True,exist_ok=True)
    payload={
        "model_id":MODEL_ID,
        "hidden_dim":hidden,
        "num_register_tokens":regs,
        "patch_size":patch,
        "manifest":str(manifest),
        "groups":len(groups),
        "unique_images":len(all_paths),
        "results":results,
        "delta_C_minus_A":delta,
        "winner":winner,
    }
    (out/"results.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")

    lines=[
        "DINOv2 Giant+Registers CLS — CenterCrop224 vs AR-Pad224",
        "="*76,
        f"A center_crop: R@1={a['R@1']:.4f} R@5={a['R@5']:.4f} MRR={a['MRR']:.4f} gap={a['separation_gap_mean']:.6f} margin={a['strict_margin_mean']:.6f}",
        f"C aspect_pad : R@1={c['R@1']:.4f} R@5={c['R@5']:.4f} MRR={c['MRR']:.4f} gap={c['separation_gap_mean']:.6f} margin={c['strict_margin_mean']:.6f}",
        "",
        "C - A:",
        *[f"  {k}: {v:+.6f}" for k,v in delta.items()],
        "",
        f"WINNER: {winner}",
    ]
    (out/"summary.txt").write_text("\n".join(lines),encoding="utf-8")

    print("\n"+"="*88)
    print("FINAL")
    print("="*88)
    print("A center_crop R@1/R@5/MRR :",f"{a['R@1']:.4f} / {a['R@5']:.4f} / {a['MRR']:.4f}")
    print("C aspect_pad  R@1/R@5/MRR :",f"{c['R@1']:.4f} / {c['R@5']:.4f} / {c['MRR']:.4f}")
    print("delta C-A gap             :",f"{delta['separation_gap_mean']:+.6f}")
    print("delta C-A strict margin   :",f"{delta['strict_margin_mean']:+.6f}")
    print("WINNER                    :",winner)
    print("output                    :",out)

if __name__=="__main__":
    main()
