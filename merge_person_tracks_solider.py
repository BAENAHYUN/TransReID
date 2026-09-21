#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, html, json, math
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

from config import PipelineConfig
from registry import EmbedderRegistry
from router import Router

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / 'pipeline.yaml'


def parse_args():
    p = argparse.ArgumentParser(description='Merge fragmented person long tracks using SOLIDER')
    p.add_argument('--video', required=True)
    p.add_argument('--tracks', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--review-dir', default='./outputs/person_track_postmerge')
    p.add_argument('--samples-per-track', type=int, default=5)
    p.add_argument('--similarity-threshold', type=float, default=0.88)
    p.add_argument('--conflict-iou-threshold', type=float, default=0.10)
    p.add_argument('--min-width', type=int, default=14)
    p.add_argument('--min-height', type=int, default=25)
    p.add_argument('--min-area', type=int, default=500)
    return p.parse_args()


def load_rows(path: Path):
    obj = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get('tracks'), list):
        return obj['tracks']
    raise TypeError('tracks JSON must be list or {"tracks": [...]}')


def l2(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        raise ValueError('zero vector')
    return v / n


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw * ih
    aa = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    ba = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    den = aa + ba - inter
    return 0.0 if den <= 0 else inter / den


def sample_rows(rows, n):
    rows = sorted(rows, key=lambda r: int(r['frame_idx']))
    if len(rows) <= n:
        return rows
    idxs = np.linspace(0, len(rows)-1, n).round().astype(int)
    return [rows[int(i)] for i in dict.fromkeys(idxs.tolist())]


def crop_from_row(cap, row, args):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(row['frame_idx']))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, row['bbox'])
    x1 = max(0, min(w, int(math.floor(x1))))
    y1 = max(0, min(h, int(math.floor(y1))))
    x2 = max(0, min(w, int(math.ceil(x2))))
    y2 = max(0, min(h, int(math.ceil(y2))))
    cw, ch = x2-x1, y2-y1
    if cw < args.min_width or ch < args.min_height or cw*ch < args.min_area:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def same_frame_conflict(rows_a, rows_b, iou_threshold):
    a = {int(r['frame_idx']): r for r in rows_a}
    b = {int(r['frame_idx']): r for r in rows_b}
    common = sorted(set(a) & set(b))
    bad = []
    for f in common:
        iou = bbox_iou(a[f]['bbox'], b[f]['bbox'])
        if iou < iou_threshold:
            bad.append((f, iou))
    return bool(bad), bad[:5]


class UF:
    def __init__(self, items):
        self.p = {x:x for x in items}
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def write_html(path, groups, pair_rows, preview_paths):
    sections = []
    for canonical, members in sorted(groups.items()):
        imgs = []
        for tid in members:
            for p in preview_paths.get(tid, [])[:3]:
                imgs.append(f'<div class="img"><div>track {tid}</div><img src="{html.escape(Path(p).resolve().as_uri())}"></div>')
        sections.append(f'<section><h2>canonical_person_id {canonical}</h2><p>members: {members}</p><div class="grid">{"".join(imgs)}</div></section>')
    pair_table = ''.join(
        f"<tr><td>{r['track_a']}</td><td>{r['track_b']}</td><td>{r['similarity']:.4f}</td><td>{r['conflict']}</td><td>{r['merged']}</td></tr>"
        for r in pair_rows
    )
    doc = f'''<!doctype html><html><head><meta charset="utf-8"><title>Person Track Post-Merge</title>
<style>body{{background:#09101d;color:#eef4ff;font-family:Segoe UI,Arial;padding:25px}}section{{background:#121b2d;border:1px solid #293753;border-radius:14px;padding:16px;margin:16px 0}}.grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}}.img{{background:#070c16;border:1px solid #293753;border-radius:10px;padding:8px}}.img img{{width:100%;height:220px;object-fit:contain}}table{{width:100%;border-collapse:collapse;background:#121b2d}}th,td{{border:1px solid #293753;padding:8px;text-align:left}}</style></head><body>
<h1>Person Track Post-Merge</h1><h2>Pair decisions</h2><table><tr><th>A</th><th>B</th><th>cosine</th><th>conflict</th><th>merged</th></tr>{pair_table}</table>{''.join(sections)}</body></html>'''
    path.write_text(doc, encoding='utf-8')


def main():
    args = parse_args()
    video = Path(args.video)
    tracks_path = Path(args.tracks)
    output = Path(args.output)
    review_dir = Path(args.review_dir)
    crop_root = review_dir / 'preview_crops'
    crop_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(tracks_path)
    person_rows = [r for r in rows if str(r.get('final_db_route','')).lower() == 'person']
    grouped = defaultdict(list)
    for r in person_rows:
        grouped[int(r.get('long_track_id', r.get('track_id')))].append(r)
    if not grouped:
        raise RuntimeError('No final_db_route=person tracks found')

    cfg = PipelineConfig.load(CONFIG_PATH)
    reg = EmbedderRegistry(cfg)
    router = Router(cfg, reg, input_format='rgb')
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'cannot open video: {video}')

    track_vectors, preview_paths = {}, defaultdict(list)
    try:
        for tid in sorted(grouped):
            embs = []
            for idx, row in enumerate(sample_rows(grouped[tid], args.samples_per_track), 1):
                crop = crop_from_row(cap, row, args)
                if crop is None:
                    continue
                dst = crop_root / f'track_{tid:04d}_{idx:02d}_f{int(row["frame_idx"]):06d}.jpg'
                cv2.imwrite(str(dst), crop)
                preview_paths[tid].append(str(dst))
                vecs = router.embed_query_image(str(dst), scope='person', names=['solider'])
                if 'solider' not in vecs:
                    raise RuntimeError('Router did not return SOLIDER vector')
                embs.append(l2(vecs['solider']))
            if embs:
                track_vectors[tid] = l2(np.mean(np.stack(embs), axis=0))
                print(f'[EMBED] track={tid} samples={len(embs)}')
            else:
                print(f'[SKIP] track {tid}: no valid crop')
    finally:
        cap.release()
        try: reg.release()
        except Exception: pass

    tids = sorted(track_vectors)
    uf = UF(tids)
    pair_rows = []
    for i, a in enumerate(tids):
        for b in tids[i+1:]:
            sim = float(np.dot(track_vectors[a], track_vectors[b]))
            conflict, details = same_frame_conflict(grouped[a], grouped[b], args.conflict_iou_threshold)
            merged = sim >= args.similarity_threshold and not conflict
            if merged:
                uf.union(a, b)
            pair_rows.append({'track_a':a,'track_b':b,'similarity':sim,'conflict':conflict,'conflict_examples':details,'merged':merged})

    components = defaultdict(list)
    for tid in tids:
        components[uf.find(tid)].append(tid)

    canonical_of, groups = {}, {}
    for members in components.values():
        members = sorted(members)
        canonical = min(members)
        groups[canonical] = members
        for tid in members:
            canonical_of[tid] = canonical

    revised = []
    for r in rows:
        out = dict(r)
        if str(r.get('final_db_route','')).lower() == 'person':
            tid = int(r.get('long_track_id', r.get('track_id')))
            can = canonical_of.get(tid, tid)
            out['canonical_person_id'] = int(can)
            out['canonical_person_members'] = groups.get(can, [tid])
        revised.append(out)
    output.write_text(json.dumps(revised, ensure_ascii=False, indent=2), encoding='utf-8')

    review_dir.mkdir(parents=True, exist_ok=True)
    summary = {'video':str(video),'input_person_tracks':len(grouped),'embedded_tracks':len(track_vectors),'canonical_person_tracks':len(groups),'similarity_threshold':args.similarity_threshold,'groups':[{'canonical_person_id':int(k),'members':v} for k,v in sorted(groups.items())],'pair_decisions':pair_rows}
    summary_path = review_dir / 'person_track_postmerge_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    html_path = review_dir / 'person_track_postmerge.html'
    write_html(html_path, groups, pair_rows, preview_paths)

    print('\n' + '='*88)
    print('PERSON TRACK POST-MERGE COMPLETE')
    print('='*88)
    print('input person tracks     :', len(grouped))
    print('canonical person tracks :', len(groups))
    print('threshold               :', args.similarity_threshold)
    print('output                  :', output)
    print('summary                 :', summary_path)
    print('html                    :', html_path)
    print('='*88)

if __name__ == '__main__':
    main()
