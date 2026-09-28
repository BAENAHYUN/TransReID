"""eval/object_pair_eval.py — 객체 재출현 쌍 준정답 시트(sheet) + 객체 임베더·클러스터 평가(eval)  (P6, 기준표 §3 객체 · §5 객체)

객체(가방·차량 …)에는 PRW 같은 공개 identity GT 가 없다. 그래서 운영 DB(forensic_object, media_type=video) 의 객체 트랙을
단위로 "이 두 트랙이 같은 물건인가" 를 사람이 판정한 쌍(같음 / 다름 / 모름)을 정답으로 쓴다.
쌍 제안 = ① 운영 Leiden 트랙 클러스터(같은 클러스터 안 쌍) ② 임베딩 최근접이지만 다른 클러스터인 쌍(경계 사례) ③ 무작위 저유사도 쌍(쉬운 음성).

sheet : Qdrant 에서 객체 트랙 벡터(트랙 중심 = 정규화 평균)와 대표 crop 을 모아 <gt-dir>/tracks_<vector>.npz + .meta.json 에 저장하고,
        <gt-dir>/{proposals.json, sheet.html} 을 만든다. 라벨 → labels.json (같은 폴더).
eval  : labels.json 의 같음 쌍을 합쳐 identity 그룹을 만들고
        - 검색 mAP/R1: 그룹의 각 트랙을 질의, 갤러리 = DB 의 모든 객체 트랙 (라벨 없는 트랙은 음성으로 가정 — 보수적) / map_labeled = 라벨된 트랙만
        - 쌍 판별: 코사인 유사도의 AUC · 최적 F1 임계값 · 운영 임계값(0.97) 정확도
        - 클러스터 일치: 라벨 쌍에 대한 운영 클러스터의 쌍 정밀도/재현율
        벡터를 바꿔(--vector dinov2 | siglip2) 같은 정답으로 객체 임베더를 비교한다. labels.json 이 없으면 제안 출처를 정답으로 가정하는
        pseudo 모드(원장 기록 안 함). 산출물: <output-dir>/<name>/object_pair_eval.json + report.md, 원장 stage=object.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import gt_sheet as S  # noqa: E402

PRODUCER = "object_pair_eval"
DEFAULT_GT_DIR = ROOT / "eval" / "gt" / "object_pairs"
DEFAULT_OUT = ROOT / "eval" / "results" / "object_pairs"
DEFAULT_ASSIGN = ROOT / "outputs" / "clustering" / "leiden_object_track_centroid_097" / "object" / "object_leiden_assignments.jsonl"
VERDICTS = ("same", "different", "unsure")


# ---------------------------------------------------------------- Qdrant → 트랙 표
def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def fetch_tracks(client: Any, collection: str, vector: str, media_type: str = "video", scroll_batch: int = 512,
                 log: Any = print) -> Tuple[List[str], np.ndarray, Dict[str, Dict[str, Any]]]:
    """객체 영상 point 를 전부 훑어 track_key 별 중심 벡터·대표 payload 를 만든다."""
    from qdrant_client.http import models as qm
    flt = qm.Filter(must=[qm.FieldCondition(key="media_type", match=qm.MatchValue(value=media_type))])
    vecs: Dict[str, List[np.ndarray]] = defaultdict(list)
    meta: Dict[str, Dict[str, Any]] = {}
    offset = None
    n = 0
    while True:
        pts, offset = client.scroll(collection, scroll_filter=flt, limit=scroll_batch, offset=offset, with_payload=True, with_vectors=[vector])
        for p in pts:
            pl = p.payload or {}
            key = str(pl.get("track_key") or "")
            if not key:
                continue
            v = p.vector.get(vector) if isinstance(p.vector, dict) else p.vector
            if v is None:
                continue
            arr = np.asarray(v, dtype=np.float32)
            nrm = float(np.linalg.norm(arr))
            if nrm > 0:
                arr = arr / nrm
            vecs[key].append(arr)
            q = _f(pl.get("quality_score"), -1.0)
            m = meta.get(key)
            if m is None or q > m["quality"]:
                meta[key] = {"track_key": key, "quality": q, "crop_path": pl.get("crop_path"), "label": pl.get("label") or pl.get("class_name"),
                             "video_stem": pl.get("video_stem") or Path(str(pl.get("video") or "")).stem, "timestamp_sec": pl.get("timestamp_sec"),
                             "frame_idx": pl.get("frame_idx"), "point_ids": []}
            meta[key]["point_ids"].append(str(p.id))
            n += 1
        if offset is None:
            break
        if n and n % 5000 < scroll_batch:
            log(f"[fetch] {n:,} points · {len(vecs):,} tracks")
    keys = sorted(vecs)
    if not keys:
        raise SystemExit(f"{collection} 에 media_type={media_type} 객체 point 가 없습니다")
    mat = np.stack([np.mean(np.stack(vecs[k]), axis=0) for k in keys]).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.clip(norms, 1e-12, None)
    for k in keys:
        meta[k]["n_points"] = len(vecs[k])
    log(f"[fetch] {collection}/{vector}: point {n:,} → 트랙 {len(keys):,}")
    return keys, mat, meta


def cache_paths(gt_dir: Path, vector: str) -> Tuple[Path, Path]:
    return gt_dir / f"tracks_{vector}.npz", gt_dir / f"tracks_{vector}.meta.json"


def save_tracks(gt_dir: Path, vector: str, keys: Sequence[str], mat: np.ndarray, meta: Dict[str, Dict[str, Any]]) -> None:
    npz, mj = cache_paths(gt_dir, vector)
    gt_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, keys=np.asarray(list(keys)), vectors=mat)
    S.write_json(mj, {"vector": vector, "generated_at": S.now_iso(), "n_tracks": len(keys), "tracks": meta})


def load_tracks(gt_dir: Path, vector: str) -> Tuple[List[str], np.ndarray, Dict[str, Dict[str, Any]]]:
    npz, mj = cache_paths(gt_dir, vector)
    if not npz.is_file() or not mj.is_file():
        raise FileNotFoundError(f"트랙 캐시가 없습니다: {npz} — sheet 를 먼저 만들거나 eval --refresh")
    d = np.load(npz, allow_pickle=False)
    keys = [str(k) for k in d["keys"]]
    mat = np.asarray(d["vectors"], dtype=np.float32)
    meta = (S.load_json(mj, {}) or {}).get("tracks") or {}
    return keys, mat, meta


def track_clusters(meta: Dict[str, Dict[str, Any]], assignments: Optional[Path]) -> Dict[str, Optional[str]]:
    """assignments.jsonl(point_id → cluster_id) → 트랙별 다수 클러스터 (noise 는 None)."""
    out: Dict[str, Optional[str]] = {k: None for k in meta}
    if not assignments or not Path(assignments).is_file():
        return out
    by_point: Dict[str, Optional[str]] = {}
    with Path(assignments).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cid = r.get("cluster_id")
            by_point[str(r.get("point_id"))] = None if (cid is None or r.get("noise")) else str(cid)
    for k, m in meta.items():
        votes = Counter(by_point.get(pid) for pid in (m.get("point_ids") or []) if by_point.get(pid) is not None)
        out[k] = votes.most_common(1)[0][0] if votes else None
    return out


# ---------------------------------------------------------------- 쌍 제안
def propose_pairs(keys: Sequence[str], mat: np.ndarray, clusters: Dict[str, Optional[str]], *, n_cluster: int, n_knn: int,
                  n_random: int, seed: int = 42, per_cluster: int = 2, knn_min_sim: float = 0.85, random_max_sim: float = 0.8) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    idx = {k: i for i, k in enumerate(keys)}
    pairs: List[Dict[str, Any]] = []
    seen = set()

    def add(a: str, b: str, source: str) -> bool:
        key = tuple(sorted((a, b)))
        if a == b or key in seen:
            return False
        seen.add(key)
        sim = float(mat[idx[a]] @ mat[idx[b]])
        pairs.append({"pair_id": f"p{len(pairs) + 1:03d}", "a": key[0], "b": key[1], "sim": round(sim, 4), "source": source,
                      "cluster_a": clusters.get(key[0]), "cluster_b": clusters.get(key[1])})
        return True

    # ① 클러스터 안 쌍
    by_cluster: Dict[str, List[str]] = defaultdict(list)
    for k, c in clusters.items():
        if c is not None and k in idx:
            by_cluster[c].append(k)
    n_added = 0
    for c in sorted(by_cluster, key=lambda c: (-len(by_cluster[c]), c)):
        members = sorted(by_cluster[c])
        rng.shuffle(members)
        for i in range(min(per_cluster, len(members) - 1)):
            if n_added >= n_cluster:
                break
            if add(members[i], members[i + 1], "cluster"):
                n_added += 1
        if n_added >= n_cluster:
            break
    # ② 최근접이지만 다른 클러스터 (경계 사례) — 유사도 내림차순
    if n_knn > 0 and len(keys) > 1:
        sims = mat @ mat.T
        np.fill_diagonal(sims, -1.0)
        order = np.argsort(-sims, axis=None)
        n_added = 0
        for flat in order:
            i, j = divmod(int(flat), len(keys))
            if i >= j:
                continue
            s = float(sims[i, j])
            if s < knn_min_sim:
                break
            a, b = keys[i], keys[j]
            if clusters.get(a) is not None and clusters.get(a) == clusters.get(b):
                continue
            if add(a, b, "knn"):
                n_added += 1
            if n_added >= n_knn:
                break
    # ③ 무작위 저유사도
    tries = 0
    n_added = 0
    while n_added < n_random and tries < 20000 and len(keys) > 1:
        tries += 1
        a, b = rng.sample(keys, 2)
        if float(mat[idx[a]] @ mat[idx[b]]) > random_max_sim:
            continue
        if add(a, b, "random"):
            n_added += 1
    return pairs


INTRO = """
<b>객체 재출현 쌍 시트</b> — 각 카드는 운영 DB 의 객체 트랙 두 개입니다. <b>같은 물건(같은 개체)인지</b>만 판정합니다.
<ol>
<li><b>같음</b>: 같은 가방·같은 차량 등 동일 개체 (다른 시간·다른 영상에 다시 나타난 것 포함).</li>
<li><b>다름</b>: 종류가 같아도 다른 개체면 다름 (검은 배낭 두 개 ≠ 같음).</li>
<li><b>모름</b>: 너무 작거나 가려져 판단 불가. 평가에서 제외됩니다.</li>
<li>제안 출처 <span class="tag">cluster</span>=운영 클러스터가 같다고 본 쌍, <span class="tag">knn</span>=임베딩은 비슷한데 클러스터가 다른 쌍, <span class="tag">random</span>=무작위. 출처에 끌리지 말고 그림만 보고 판정하세요.</li>
<li>끝나면 <b>labels.json 내려받기</b> → 이 시트와 같은 폴더에 <code>labels.json</code>.</li>
</ol>
"""


def build_sheet(pairs: Sequence[Dict[str, Any]], meta: Dict[str, Dict[str, Any]], thumb_h: int, vector: str) -> str:
    cards = []
    for p in pairs:
        sides = []
        for side in ("a", "b"):
            m = meta.get(p[side], {})
            uri = S.thumb_b64_from_file(m.get("crop_path"), height=thumb_h)
            sides.append(f"""<div class="side">{S.img_tag(uri, p[side])}<div>{S.esc(m.get('label'))} · {S.esc(m.get('video_stem'))} · {S.fmt_time(m.get('timestamp_sec'))}</div><div>{S.esc(p[side])}</div></div>""")
        pid = p["pair_id"]
        radios = "".join(f'<label><input type="radio" name="{pid}__verdict" data-item="{pid}" data-field="verdict" value="{v}"> {lab}</label>'
                         for v, lab in (("same", "같음"), ("different", "다름"), ("unsure", "모름")))
        cards.append(f"""
<div class="card" data-block="{pid}">
  <h2>{pid} <span class="tag">{S.esc(p['source'])}</span> <span class="meta">유사도 {p['sim']:.3f}</span></h2>
  <div class="pair">{sides[0]}<div style="font-size:22px;color:#999">?</div>{sides[1]}
    <div class="fields"><div class="radios">{radios}</div><label>메모 <input type="text" data-item="{pid}" data-field="note" style="width:160px"></label></div>
  </div>
</div>""")
    intro = INTRO + f'<div class="meta">쌍 {len(pairs)} · 벡터 {S.esc(vector)} · 트랙 {len(meta):,}</div>'
    return S.html_document("객체 재출현 쌍 판정", intro, "".join(cards), kind="object_pair_labels", store_key=f"object_pairs:{vector}",
                           export_name="labels.json", meta={"vector": vector, "n_pairs": len(pairs)},
                           extra_js="window.doneRule = v => !!v.verdict;")


def cmd_sheet(args: argparse.Namespace) -> Dict[str, Any]:
    gt_dir = Path(args.gt_dir).resolve()
    from qdrant_client import QdrantClient
    client = QdrantClient(url=args.qdrant_url, timeout=120)
    keys, mat, meta = fetch_tracks(client, args.collection, args.vector)
    save_tracks(gt_dir, args.vector, keys, mat, meta)
    clusters = track_clusters(meta, Path(args.assignments) if args.assignments else None)
    n_clustered = sum(1 for v in clusters.values() if v is not None)
    pairs = propose_pairs(keys, mat, clusters, n_cluster=args.cluster_pairs, n_knn=args.knn_pairs, n_random=args.random_pairs, seed=args.seed)
    html = build_sheet(pairs, meta, args.thumb_height, args.vector)
    S.write_text(gt_dir / "sheet.html", html)
    S.write_json(gt_dir / "proposals.json", {"generated_at": S.now_iso(), "vector": args.vector, "collection": args.collection,
                                             "assignments": args.assignments, "n_tracks": len(keys), "n_clustered_tracks": n_clustered,
                                             "pairs": pairs})
    src = Counter(p["source"] for p in pairs)
    print(f"[sheet] 트랙 {len(keys):,} (클러스터 소속 {n_clustered:,}) · 쌍 {len(pairs)} ({dict(src)}) → {gt_dir / 'sheet.html'}")
    print(f"RESULT_SUMMARY: {gt_dir / 'sheet.html'}")
    return {"n_tracks": len(keys), "pairs": len(pairs), "sources": dict(src)}


# ---------------------------------------------------------------- 평가
class UnionFind:
    def __init__(self) -> None:
        self.p: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def labeled_pairs(pairs: Sequence[Dict[str, Any]], labels: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """proposals + labels → verdict 가 붙은 쌍 목록 (unsure/미라벨 포함, verdict None = 미라벨)."""
    out = []
    for p in pairs:
        lab = labels.get(p["pair_id"], {})
        v = lab.get("verdict")
        v = str(v).strip().lower() if v else None
        out.append({**p, "verdict": v if v in VERDICTS else None})
    return out


def identity_groups(pairs: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    uf = UnionFind()
    for p in pairs:
        if p.get("verdict") == "same":
            uf.union(p["a"], p["b"])
    groups: Dict[str, List[str]] = defaultdict(list)
    for p in pairs:
        if p.get("verdict") == "same":
            for k in (p["a"], p["b"]):
                r = uf.find(k)
                if k not in groups[r]:
                    groups[r].append(k)
    return dict(groups)


def _ap(ranked_pos: Sequence[bool], n_pos: int) -> float:
    if n_pos <= 0:
        return 0.0
    hits = 0
    s = 0.0
    for i, ok in enumerate(ranked_pos, 1):
        if ok:
            hits += 1
            s += hits / i
    return s / n_pos


def retrieval_metrics(keys: Sequence[str], mat: np.ndarray, groups: Dict[str, List[str]], gallery_keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """그룹의 각 트랙을 질의로, 나머지 트랙(갤러리)을 코사인 유사도로 정렬 → AP / R1. gallery_keys 가 있으면 그 안에서만."""
    idx = {k: i for i, k in enumerate(keys)}
    gal = [k for k in (gallery_keys if gallery_keys is not None else keys) if k in idx]
    gal_idx = np.asarray([idx[k] for k in gal], dtype=np.int64)
    member_of: Dict[str, str] = {k: g for g, ms in groups.items() for k in ms}
    aps, r1s = [], []
    n_q = 0
    for g, members in groups.items():
        for q in members:
            if q not in idx:
                continue
            pos = {m for m in members if m != q and m in idx}
            if not pos:
                continue
            sims = mat[gal_idx] @ mat[idx[q]]
            order = np.argsort(-sims)
            ranked = [gal[i] for i in order if gal[i] != q]
            flags = [k in pos for k in ranked]
            n_pos = sum(1 for k in gal if k in pos)
            if n_pos == 0:
                continue
            aps.append(_ap(flags, n_pos))
            r1s.append(1.0 if flags and flags[0] else 0.0)
            n_q += 1
    return {"map": round(100 * float(np.mean(aps)), 2) if aps else None, "rank1": round(100 * float(np.mean(r1s)), 2) if r1s else None,
            "queries": n_q, "gallery": len(gal)}


def pair_metrics(pairs: Sequence[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    pos = [p["sim"] for p in pairs if p.get("verdict") == "same"]
    neg = [p["sim"] for p in pairs if p.get("verdict") == "different"]
    out: Dict[str, Any] = {"pairs_same": len(pos), "pairs_diff": len(neg), "pairs_unsure": sum(1 for p in pairs if p.get("verdict") == "unsure"),
                           "pairs_unlabeled": sum(1 for p in pairs if p.get("verdict") is None)}
    if not pos or not neg:
        out.update({"pair_auc": None, "pair_f1": None, "pair_threshold": None, "pair_acc_at_threshold": None})
        return out
    # AUC = P(sim_pos > sim_neg) (동률 0.5)
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    out["pair_auc"] = round(wins / (len(pos) * len(neg)), 4)
    best = (0.0, None)
    for t in sorted(set(pos + neg)):
        tp = sum(1 for s in pos if s >= t)
        fp = sum(1 for s in neg if s >= t)
        fn = len(pos) - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best[0]:
            best = (f1, t)
    out["pair_f1"], out["pair_threshold"] = round(best[0], 4), (round(best[1], 4) if best[1] is not None else None)
    acc = (sum(1 for s in pos if s >= threshold) + sum(1 for s in neg if s < threshold)) / (len(pos) + len(neg))
    out["pair_acc_at_threshold"] = round(acc, 4)
    out["threshold"] = threshold
    return out


def cluster_agreement(pairs: Sequence[Dict[str, Any]], clusters: Dict[str, Optional[str]]) -> Dict[str, Any]:
    same_cl = lambda p: clusters.get(p["a"]) is not None and clusters.get(p["a"]) == clusters.get(p["b"])  # noqa: E731
    pos = [p for p in pairs if p.get("verdict") == "same"]
    neg = [p for p in pairs if p.get("verdict") == "different"]
    tp = sum(1 for p in pos if same_cl(p))
    fp = sum(1 for p in neg if same_cl(p))
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / len(pos) if pos else None
    return {"cluster_pair_precision": round(prec, 4) if prec is not None else None, "cluster_pair_recall": round(rec, 4) if rec is not None else None,
            "cluster_pairs_tp": tp, "cluster_pairs_fp": fp}


def cmd_eval(args: argparse.Namespace) -> Dict[str, Any]:
    gt_dir = Path(args.gt_dir).resolve()
    started = time.time()
    try:
        if args.refresh:
            raise FileNotFoundError
        keys, mat, meta = load_tracks(gt_dir, args.vector)
    except FileNotFoundError:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=args.qdrant_url, timeout=120)
        keys, mat, meta = fetch_tracks(client, args.collection, args.vector)
        save_tracks(gt_dir, args.vector, keys, mat, meta)
    proposals = S.load_json(gt_dir / "proposals.json", {}) or {}
    raw_pairs = proposals.get("pairs") or []
    if not raw_pairs:
        raise SystemExit(f"proposals.json 이 없거나 비었습니다: {gt_dir} (sheet 먼저)")
    labels = S.read_labels(gt_dir / "labels.json")
    pseudo = not labels
    idx = {k: i for i, k in enumerate(keys)}
    pairs = labeled_pairs(raw_pairs, labels)
    # 현재 벡터로 유사도를 다시 계산 (시트는 다른 벡터로 만들었을 수 있다)
    for p in pairs:
        if p["a"] in idx and p["b"] in idx:
            p["sim"] = round(float(mat[idx[p["a"]]] @ mat[idx[p["b"]]]), 4)
    if pseudo:
        for p in pairs:
            p["verdict"] = "same" if p["source"] == "cluster" else "different"
    assignments = Path(args.assignments) if args.assignments else (Path(proposals["assignments"]) if proposals.get("assignments") else None)
    clusters = track_clusters(meta, assignments)
    groups = identity_groups(pairs)
    labeled_keys = sorted({k for p in pairs if p.get("verdict") in ("same", "different") for k in (p["a"], p["b"])})
    ret = retrieval_metrics(keys, mat, groups)
    ret_l = retrieval_metrics(keys, mat, groups, labeled_keys)
    pm = pair_metrics(pairs, args.threshold)
    ca = cluster_agreement(pairs, clusters)
    metrics: Dict[str, Any] = {"map": ret["map"], "rank1": ret["rank1"], "map_labeled": ret_l["map"], "rank1_labeled": ret_l["rank1"],
                               "queries": ret["queries"], "gallery": ret["gallery"], **pm, **ca, "identities": len(groups),
                               "elapsed_sec": round(time.time() - started, 2)}
    name = args.name or f"{args.vector}_track{'_pseudo' if pseudo else ''}"
    out = {"producer": PRODUCER, "generated_at": S.now_iso(), "name": name, "pseudo_gt": pseudo,
           "config": {"gt_dir": str(gt_dir), "vector": args.vector, "collection": args.collection, "threshold": args.threshold,
                      "assignments": str(assignments) if assignments else None},
           "gt": {"pairs": len(pairs), "pairs_same": pm["pairs_same"], "pairs_diff": pm["pairs_diff"], "pairs_unsure": pm["pairs_unsure"],
                  "identities": len(groups), "labeled_tracks": len(labeled_keys), "db_tracks": len(keys),
                  "protocol": "객체 트랙 쌍 사람 판정(같음/다름) → identity 그룹; 질의 = 그룹 트랙, 갤러리 = DB 객체 트랙 전부(미라벨 = 음성 가정)"},
           "metrics": metrics, "groups": {g: ms for g, ms in groups.items()},
           "pairs": [{k: p.get(k) for k in ("pair_id", "a", "b", "sim", "source", "verdict")} for p in pairs]}
    out_dir = Path(args.output_dir).resolve() / name
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = S.write_json(out_dir / "object_pair_eval.json", out)
    S.write_text(out_dir / "report.md", report_md(out))
    print(f"\n[eval] {name}{' (pseudo — labels.json 없음: cluster 쌍=같음, 나머지=다름 가정, 원장 기록 안 함)' if pseudo else ''}")
    print(f"  쌍 같음 {pm['pairs_same']} / 다름 {pm['pairs_diff']} / 모름 {pm['pairs_unsure']} · identity 그룹 {len(groups)} · 라벨 트랙 {len(labeled_keys)} / DB 트랙 {len(keys):,}")
    print(f"  검색 mAP {metrics['map']} R1 {metrics['rank1']} (질의 {ret['queries']}, 갤러리 {ret['gallery']:,}) · 라벨 갤러리 mAP {metrics['map_labeled']}")
    print(f"  쌍 AUC {pm.get('pair_auc')} · 최적 F1 {pm.get('pair_f1')} @ {pm.get('pair_threshold')} · 운영 임계 {args.threshold} 정확도 {pm.get('pair_acc_at_threshold')}")
    print(f"  클러스터 쌍 정밀도 {ca['cluster_pair_precision']} / 재현율 {ca['cluster_pair_recall']}")
    if not pseudo:
        from bench import ledger
        ledger.record(lambda: [ledger.entry_from_object_result(out, report=json_path, command=args.command_line, versions=ledger.versions_info())],
                      args.ledger, args.no_ledger)
    print(f"RESULT_SUMMARY: {json_path}")
    return out


def report_md(out: Dict[str, Any]) -> str:
    m = out["metrics"]
    g = out["gt"]
    lines = [f"# 객체 재출현 평가 — {out['name']} ({out['generated_at']})", "",
             f"- 벡터 {out['config']['vector']} · 쌍 {g['pairs']} (같음 {g['pairs_same']} / 다름 {g['pairs_diff']} / 모름 {g['pairs_unsure']}) · identity {g['identities']} · DB 트랙 {g['db_tracks']:,}"
             + (" — **pseudo GT (cluster 쌍 = 같음 가정)**" if out["pseudo_gt"] else ""), "",
             "| 지표 | 값 |", "|---|---|"]
    for k, label in (("map", "검색 mAP (갤러리 전체)"), ("rank1", "Rank-1"), ("map_labeled", "mAP (라벨 트랙만)"), ("queries", "질의 수"),
                     ("pair_auc", "쌍 AUC"), ("pair_f1", "쌍 최적 F1"), ("pair_threshold", "최적 임계값"), ("pair_acc_at_threshold", "운영 임계 정확도"),
                     ("cluster_pair_precision", "클러스터 쌍 정밀도"), ("cluster_pair_recall", "클러스터 쌍 재현율")):
        lines.append(f"| {label} | {m.get(k)} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="객체 재출현 쌍 시트 + 객체 임베더 평가 (P6)")
    p.add_argument("cmd", choices=["sheet", "eval"])
    p.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR))
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--collection", default="forensic_object")
    p.add_argument("--vector", default="dinov2", help="객체 벡터 이름 (dinov2 | siglip2 …)")
    p.add_argument("--assignments", default=str(DEFAULT_ASSIGN) if DEFAULT_ASSIGN.is_file() else None,
                   help="운영 트랙 클러스터 assignments.jsonl (쌍 제안 ① · 클러스터 일치)")
    p.add_argument("--cluster-pairs", type=int, default=40)
    p.add_argument("--knn-pairs", type=int, default=25)
    p.add_argument("--random-pairs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thumb-height", type=int, default=170)
    p.add_argument("--threshold", type=float, default=0.97, help="eval: 운영 쌍 임계값 (Leiden 0.97)")
    p.add_argument("--refresh", action="store_true", help="eval: 트랙 캐시를 Qdrant 에서 다시 만든다")
    p.add_argument("--name", default=None)
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--ledger", default=None, help="실행 원장 JSONL (기본 bench/ledger.jsonl; '' 또는 none 이면 기록 안 함)")
    p.add_argument("--no-ledger", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    p = build_parser()
    args = p.parse_args(argv)
    args.command_line = list(argv) if argv is not None else sys.argv[1:]
    return cmd_sheet(args) if args.cmd == "sheet" else cmd_eval(args)


if __name__ == "__main__":
    main()
