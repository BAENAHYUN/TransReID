import os, sys
from collections import Counter, defaultdict
import numpy as np
from qdrant_client import QdrantClient
from sklearn.preprocessing import normalize
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [CURRENT_DIR, os.path.dirname(CURRENT_DIR)]:
    if p not in sys.path: sys.path.append(p)
from config import PipelineConfig

# ── 파라미터 ──────────────────────────────────────────
VERSION      = "dbscan_person_v5_cl"
PAYLOAD_KEY  = VERSION
COLLECTION   = "forensic_person"
SCROLL_BATCH = 64
CHUNK_SIZE   = 128
EPS          = 0.15        # v2_cl 0.25 → 축소 (순도 향상)
MIN_FACES    = 5           # v2_cl 3 → 증가 (핵심 포인트 조건 강화)
K            = 25
SCORE_THRESH = 1.0 - EPS   # 0.85 (SOLIDER 공간과 EPS 일치)
W_SIGLIP, W_IRRA, W_SOLIDER = 0.1, 0.3, 0.6
# ──────────────────────────────────────────────────────

cfg = PipelineConfig.load(os.path.join(CURRENT_DIR, "pipeline.yaml"))
client = QdrantClient(url=cfg.qdrant.url, timeout=120)

def combined_vec(s, i, o):
    s, i, o = normalize([s])[0], normalize([i])[0], normalize([o])[0]
    return normalize([np.concatenate([W_SIGLIP*s, W_IRRA*i, W_SOLIDER*o])])[0]

def cos_dist(a, b): return float(1.0 - np.dot(a, b))

# ── Step 1: 벡터 수집 ────────────────────────────────
total = client.get_collection(COLLECTION).points_count
print(f"[{VERSION}] EPS={EPS}, MIN_FACES={MIN_FACES}, SCORE_THRESH={SCORE_THRESH}")

all_ids, all_vecs, all_sol = [], {}, {}
offset = None
with tqdm(total=total, desc="벡터 수집") as pbar:
    while True:
        res, offset = client.scroll(
            collection_name=COLLECTION, limit=SCROLL_BATCH,
            offset=offset, with_vectors=["siglip2","irra","solider"], with_payload=False)
        if not res: break
        for p in res:
            all_ids.append(p.id)
            all_vecs[p.id] = combined_vec(p.vector["siglip2"], p.vector["irra"], p.vector["solider"])
            all_sol[p.id]  = normalize([p.vector["solider"]])[0].tolist()
            pbar.update(1)
        if offset is None: break

# ── Step 2: 순차 배정 클러스터링 (Union-Find 없음) ───
# 이유: Union-Find 이행성이 쏠림 원인 → 제거로 체이닝 차단
cluster_map, deferred = {}, []
next_cid = 0

for pid in tqdm(all_ids, desc="핵심 포인트 클러스터링"):
    if pid in cluster_map: continue

    hits = client.query_points(
        collection_name=COLLECTION, query=all_sol[pid], using="solider",
        limit=K, score_threshold=SCORE_THRESH, with_payload=False, with_vectors=False
    ).points

    neighbors = [h.id for h in hits
                 if h.id != pid and h.id in all_vecs
                 and cos_dist(all_vecs[pid], all_vecs[h.id]) <= EPS]

    if len(neighbors) < MIN_FACES:
        deferred.append(pid)
        continue

    # 이미 배정된 이웃 클러스터에 편입 (순차 배정)
    existing = next((cluster_map[n] for n in neighbors if n in cluster_map), None)
    cid = existing if existing is not None else next_cid
    if existing is None: next_cid += 1

    cluster_map[pid] = cid
    for n in neighbors:
        if n not in cluster_map:
            cluster_map[n] = cid

# ── Step 3: 경계 포인트 후처리 ───────────────────────
for pid in tqdm(deferred, desc="경계 포인트 후처리"):
    if pid in cluster_map: continue
    hits = client.query_points(
        collection_name=COLLECTION, query=all_sol[pid], using="solider",
        limit=K, score_threshold=SCORE_THRESH, with_payload=False, with_vectors=False
    ).points
    assigned = next((cluster_map[h.id] for h in hits
                     if h.id != pid and h.id in cluster_map), None)
    cluster_map[pid] = assigned if assigned is not None else -1

# ── Step 4: 결과 출력 ────────────────────────────────
counts     = Counter(cluster_map.values())
noise      = counts.pop(-1, 0)
print(f"\n=== [{VERSION}] ===")
print(f"클러스터 수: {len(counts):,}")
print(f"노이즈:      {noise:,}개 ({noise/total*100:.2f}%)")
print(f"배정:        {total-noise:,}개")
print("상위 10개:")
for cid, cnt in counts.most_common(10):
    print(f"  cluster {cid:>5d}: {cnt:>6d}개")

# ── Step 5: Qdrant 저장 ──────────────────────────────
cid_to_ids = defaultdict(list)
for pid, cid in cluster_map.items():
    cid_to_ids[int(cid)].append(pid)

for cid, pts in tqdm(cid_to_ids.items(), desc="payload 저장"):
    for i in range(0, len(pts), CHUNK_SIZE):
        client.set_payload(
            collection_name=COLLECTION,
            payload={PAYLOAD_KEY: cid},
            points=pts[i:i+CHUNK_SIZE]
        )
print(f"\n[{PAYLOAD_KEY}] 저장 완료")