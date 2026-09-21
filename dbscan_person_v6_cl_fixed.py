import os, sys
from collections import Counter, defaultdict
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest
from sklearn.preprocessing import normalize
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in [CURRENT_DIR, os.path.dirname(CURRENT_DIR)]:
    if p not in sys.path: sys.path.append(p)
# Docker 없이 qdrant-client local mode DB를 직접 사용
LOCAL_QDRANT_PATH = os.path.join(CURRENT_DIR, "qdrant_local")
QDRANT_TIMEOUT = 120
COLLECTION_PERSON = "forensic_person"
SCROLL_BATCH = 1024
CHUNK_SIZE = 1000

# ── 파라미터 ──────────────────────────────────────────
VERSION      = "dbscan_person_v6_cl"
PAYLOAD_KEY  = VERSION
COLLECTION   = COLLECTION_PERSON
EPS          = 0.12
MIN_FACES    = 3
K            = 25
SCORE_THRESH = 1.0 - EPS
QUERY_BATCH  = 256          # 배치 검색 크기 (핵심 추가)
W_SIGLIP, W_IRRA, W_SOLIDER = 0.1, 0.3, 0.6
# ──────────────────────────────────────────────────────

print(f"[Qdrant Local] DB path: {LOCAL_QDRANT_PATH}")
if not os.path.isdir(LOCAL_QDRANT_PATH):
    raise FileNotFoundError(
        f"로컬 Qdrant DB 폴더가 없습니다: {LOCAL_QDRANT_PATH}\n"
        "먼저 migrate_qdrant_to_local.py 를 실행하세요."
    )

client = QdrantClient(
    path=LOCAL_QDRANT_PATH,
    force_disable_check_same_thread=True,
)

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

# ── Step 2: 전체 이웃 배치 검색 (핵심 개선) ──────────
# 포인트마다 개별 query → 배치로 묶어서 query_batch_points
# 76,326번 네트워크 왕복 → 약 300번으로 감소
print("\n이웃 배치 검색 중...")
neighbors_map = {}

for i in tqdm(range(0, len(all_ids), QUERY_BATCH), desc="배치 검색"):
    batch_ids = all_ids[i:i+QUERY_BATCH]
    requests = [
        QueryRequest(
            query=all_sol[pid], using="solider", limit=K,
            score_threshold=SCORE_THRESH, with_payload=False, with_vector=False
        ) for pid in batch_ids
    ]
    results = client.query_batch_points(collection_name=COLLECTION, requests=requests)

    for pid, res in zip(batch_ids, results):
        neighbors = [h.id for h in res.points
                     if h.id != pid and h.id in all_vecs
                     and cos_dist(all_vecs[pid], all_vecs[h.id]) <= EPS]
        neighbors_map[pid] = neighbors

# ── Step 3: 1차 클러스터링 (메모리 내 처리, 빠름) ────
print("\n1차 클러스터링...")
cluster_map, deferred = {}, []
next_cid = 0

for pid in tqdm(all_ids, desc="1차 배정"):
    if pid in cluster_map: continue
    neighbors = neighbors_map[pid]
    is_core = len(neighbors) >= MIN_FACES
    assigned = next((cluster_map[n] for n in neighbors if n in cluster_map), None)

    if assigned is not None:
        cluster_map[pid] = assigned
    elif is_core:
        cluster_map[pid] = next_cid
        next_cid += 1
    else:
        deferred.append(pid)

# ── Step 4: deferred 재처리 (메모리 내, 빠름) ────────
print("\ndeferred 재처리...")
for pid in tqdm(deferred, desc="deferred"):
    if pid in cluster_map: continue
    assigned = next((cluster_map[n] for n in neighbors_map[pid] if n in cluster_map), None)
    cluster_map[pid] = assigned if assigned is not None else -1

# ── Step 5: 결과 ─────────────────────────────────────
counts = Counter(cluster_map.values())
noise  = counts.pop(-1, 0)
print(f"\n=== [{VERSION}] ===")
print(f"클러스터 수: {len(counts):,}")
print(f"노이즈:      {noise:,}개 ({noise/total*100:.2f}%)")
print(f"배정:        {total-noise:,}개")
print("상위 10개:")
for cid, cnt in counts.most_common(10):
    print(f"  cluster {cid:>5d}: {cnt:>6d}개")

# ── Step 6: Qdrant 저장 ──────────────────────────────
cid_to_ids = defaultdict(list)
for pid, cid in cluster_map.items():
    cid_to_ids[int(cid)].append(pid)
for cid, pts in tqdm(cid_to_ids.items(), desc="payload 저장"):
    for j in range(0, len(pts), CHUNK_SIZE):
        client.set_payload(collection_name=COLLECTION, payload={PAYLOAD_KEY: cid}, points=pts[j:j+CHUNK_SIZE])
print(f"\n[{PAYLOAD_KEY}] 저장 완료")