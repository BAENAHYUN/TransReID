from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from config import PipelineConfig
from registry import EmbedderRegistry
from router import Router

ROOT=Path(__file__).resolve().parent
CFG=ROOT/"pipeline.yaml"

def norm(v):
    v=np.asarray(v,dtype=np.float32).reshape(-1); n=np.linalg.norm(v)
    return v if n<=0 else v/n

def qfilter(media,label):
    must=[]
    if media!="all": must.append(FieldCondition(key="media_type",match=MatchValue(value=media)))
    if label: must.append(FieldCondition(key="label",match=MatchValue(value=label)))
    return Filter(must=must) if must else None

def fps_time(frame, video_path):
    p=Path(video_path) if video_path else None
    if not p or not p.exists(): return ""
    cap=cv2.VideoCapture(str(p)); fps=float(cap.get(cv2.CAP_PROP_FPS)); cap.release()
    if fps<=0: return ""
    sec=frame/fps
    return f"{int(sec//60):02d}:{sec%60:05.2f}"

def search(client, collection, vectors, cfg, media, label, top_k, candidate_k):
    avail=set(client.get_collection(collection).config.params.vectors.keys())
    filt=qfilter(media,label)
    ranked={}; weights={}
    for name,vec in vectors.items():
        if name not in avail: continue
        ranked[name]=client.query_points(
            collection_name=collection, using=name, query=norm(vec).tolist(),
            query_filter=filt, limit=candidate_k, with_payload=True, with_vectors=False
        ).points
        weights[name]=float(cfg.retrievers[name].weight)

    score=defaultdict(float); point={}
    for name,hits in ranked.items():
        for r,h in enumerate(hits,1):
            k=str(h.id); score[k]+=weights[name]/(60+r); point[k]=h

    ordered=sorted(score.items(),key=lambda x:x[1],reverse=True)
    out=[]; seen=set()
    for pid,fs in ordered:
        h=point[pid]; p=h.payload or {}
        if p.get("media_type")=="video":
            key=p.get("track_key") or f"{p.get('video')}|{p.get('track_id')}|{p.get('label')}"
            if key in seen: continue
            seen.add(key)
        frame=int(p.get("frame_idx",0) or 0)
        out.append({
            "rank":len(out)+1,"fusion_score":fs,"media_type":p.get("media_type",""),
            "label":p.get("label",""),"image_id":p.get("image_id",""),
            "video":p.get("video",""),"video_path":p.get("video_path",""),
            "frame_idx":frame,"timestamp":fps_time(frame,p.get("video_path","")),
            "track_id":p.get("track_id"),"track_key":p.get("track_key",""),
            "crop_path":p.get("crop_path",""),"source":p.get("source",""),
            "split":p.get("split",""),"category":p.get("category","")
        })
        if len(out)>=top_k: break
    return out

def show(title,rows):
    print("\\n"+"="*88); print(title); print("="*88)
    for r in rows:
        print(f"[{r['rank']:02d}] fusion={r['fusion_score']:.6f} | {r['media_type']} | {r['label']}")
        if r["media_type"]=="video":
            print(f"     video={r['video']} | frame={r['frame_idx']} | time={r['timestamp'] or 'N/A'} | track={r['track_id']}")
            print(f"     video_path={r['video_path']}")
        else:
            print(f"     image={r['image_id']}")
        print(f"     crop={r['crop_path']}")

def main():
    ap=argparse.ArgumentParser()
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image"); g.add_argument("--text")
    ap.add_argument("--scope",choices=["person","object","all"],default="all")
    ap.add_argument("--media",choices=["all","image","video"],default="all")
    ap.add_argument("--label",default=None)
    ap.add_argument("--top-k",type=int,default=10)
    ap.add_argument("--candidate-k",type=int,default=100)
    ap.add_argument("--json-out",default=None)
    a=ap.parse_args()

    cfg=PipelineConfig.load(CFG)
    reg=EmbedderRegistry(cfg)
    router=Router(cfg,reg,input_format="rgb")
    client=QdrantClient(url=cfg.qdrant.url)
    output={}

    if a.image:
        q=Path(a.image)
        if not q.exists(): raise FileNotFoundError(q)
        if a.scope in ("person","all"):
            v=router.embed_query_image(str(q),scope="person")
            output["person"]=search(client,"forensic_person",v,cfg,a.media,"person" if not a.label else a.label,a.top_k,max(a.candidate_k,a.top_k))
        if a.scope in ("object","all"):
            v=router.embed_query_image(str(q),scope="object")
            output["object"]=search(client,"forensic_object",v,cfg,a.media,a.label,a.top_k,max(a.candidate_k,a.top_k))
    else:
        if a.scope in ("person","all"):
            v=router.embed_query_text(a.text,names=["siglip2","irra"])
            output["person"]=search(client,"forensic_person",v,cfg,a.media,"person" if not a.label else a.label,a.top_k,max(a.candidate_k,a.top_k))
        if a.scope in ("object","all"):
            v=router.embed_query_text(a.text,names=["siglip2"])
            output["object"]=search(client,"forensic_object",v,cfg,a.media,a.label,a.top_k,max(a.candidate_k,a.top_k))

    if "person" in output: show("PERSON SEARCH RESULTS",output["person"])
    if "object" in output: show("OBJECT SEARCH RESULTS",output["object"])
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
        print("\\nJSON saved:",a.json_out)

if __name__=="__main__":
    main()
