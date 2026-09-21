"""
알고리즘 효과 분리 실험 — Leiden vs DBSCAN × SOLIDER 단독 vs 결합 벡터 (PRW person, GT 채점)
==========================================================================================
같은 point 집합·같은 벡터·정확 kNN 위에서 알고리즘만 바꾸고, 임계값을 스윕해 GT(PRW 인물 ID)로 채점한다.

입력
  * 벡터 캐시: cluster_dbscan_qdrant.py --vector-cache 가 만든 npz
      matrix     = 결합 벡터 (L2(0.1·L2(siglip2) ⊕ 0.3·L2(irra) ⊕ 0.6·L2(solider)), 2304-d)
      knn_matrix = SOLIDER 단독 벡터 (L2, 1024-d)
      ids        = scroll 순서 (DBSCAN 순회 순서)
  * GT 매칭: eval/prw_cluster_gt_eval.py 가 만든 prw_gt_matches.jsonl (point_id → pid)

알고리즘 (운영 코드와 같은 규칙, 이웃만 메모리 정확 kNN)
  * Leiden : k=30 이웃(자기 제외) 중 cosine ≥ thr, 상호 kNN 간선, 가중치 = cosine, RBConfiguration resolution 1.0, seed 42,
             크기 < 2 는 노이즈  (cluster_leiden_qdrant.py 와 동일)
  * DBSCAN : 원본 dbscan_person_v6_cl 의 배정 루프(assign_clusters) 그대로. 후보 = 상위 25(자기 포함 → 제외) 중 cosine ≥ 1−eps,
             같은 벡터로 distance ≤ eps 필터, MIN_FACES 3.  "hybrid" = 원본 그대로(SOLIDER 후보 + 결합 거리 필터), 검증용.

참고: 결합 벡터의 cosine 은 성분 cosine 의 가중 평균(가중치 = 0.1²:0.3²:0.6² = 2.2% : 19.6% : 78.3%) 이라 SOLIDER 가 지배적이다.

산출물: <output-dir>/ablation_results.{csv,json}, ablation_prw.html (표 + 정밀도-재현율 산점도)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cluster_dbscan_qdrant import assign_clusters  # noqa: E402
from cluster_leiden_qdrant import leiden as run_leiden  # noqa: E402
from eval.prw_cluster_gt_eval import evaluate_method  # noqa: E402
from report_common import esc, write_json  # noqa: E402


# ---------------------------------------------------------------------------
def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def exact_knn(matrix: np.ndarray, k: int, chunk: int = 2048, log=print) -> Tuple[np.ndarray, np.ndarray]:
    """자기 자신을 포함한 상위 k+1 (내림차순). 반환 idx (N,k+1) int64, sim (N,k+1) float32. GPU(torch) 있으면 사용."""
    n = matrix.shape[0]
    kk = min(k + 1, n)
    idx = np.zeros((n, kk), dtype=np.int64)
    sim = np.zeros((n, kk), dtype=np.float32)
    started = time.time()
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        M = torch.from_numpy(matrix).to(dev)
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            S = M[s:e] @ M.T
            v, i = torch.topk(S, kk, dim=1)
            idx[s:e] = i.cpu().numpy()
            sim[s:e] = v.cpu().numpy()
            if (s // chunk) % 5 == 0:
                log(f"  knn {e:,}/{n:,} ({dev}, {time.time() - started:.0f}s)")
        del M
        if dev == "cuda":
            torch.cuda.empty_cache()
    except ImportError:
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            S = matrix[s:e] @ matrix.T
            part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            vals = np.take_along_axis(S, part, axis=1)
            order = np.argsort(-vals, axis=1, kind="stable")
            idx[s:e] = np.take_along_axis(part, order, axis=1)
            sim[s:e] = np.take_along_axis(vals, order, axis=1)
    log(f"  knn done {n:,} x {kk} ({time.time() - started:.1f}s)")
    return idx, sim


def drop_self(idx: np.ndarray, sim: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """각 행에서 자기 인덱스를 제거하고 앞 k 개만 남긴다 (자기 자신이 없으면 마지막을 버림)."""
    n = idx.shape[0]
    out_i = np.empty((n, k), dtype=np.int64)
    out_s = np.empty((n, k), dtype=np.float32)
    rows = np.arange(n)
    for r in rows:
        mask = idx[r] != r
        ii, ss = idx[r][mask][:k], sim[r][mask][:k]
        if ii.shape[0] < k:  # 드물게 부족하면 -1 패딩
            pad = k - ii.shape[0]
            ii = np.concatenate([ii, np.full(pad, -1)]); ss = np.concatenate([ss, np.full(pad, -1.0, dtype=np.float32)])
        out_i[r], out_s[r] = ii, ss
    return out_i, out_s


# ---------------------------------------------------------------------------
def leiden_labels(ids: Sequence[Any], nbr: np.ndarray, sc: np.ndarray, thr: float, mutual: bool,
                  resolution: float, seed: int, min_cluster_size: int, log=print) -> Tuple[Dict[Any, Optional[str]], Dict[str, Any]]:
    n = len(ids)
    k = nbr.shape[1]
    src = np.repeat(np.arange(n), k)
    dst = nbr.reshape(-1)
    w = sc.reshape(-1)
    valid = (dst >= 0) & (w >= thr)
    src, dst, w = src[valid], dst[valid], w[valid]
    a, b = np.minimum(src, dst), np.maximum(src, dst)
    key = a * n + b
    order = np.argsort(key, kind="stable")
    key, a, b, w = key[order], a[order], b[order], w[order]
    uniq, start, counts = np.unique(key, return_index=True, return_counts=True)
    wmax = np.maximum.reduceat(w, start) if len(w) else np.zeros(0)
    keep = (counts >= 2) if mutual else np.ones(len(uniq), dtype=bool)
    edges = list(zip(a[start][keep].tolist(), b[start][keep].tolist()))
    weights = [float(x) for x in wmax[keep]]
    t0 = time.time()
    if edges:
        membership, s2 = run_leiden(ids, edges, weights, resolution, seed)
    else:
        membership, s2 = list(range(n)), {"raw_communities": n, "largest_community": 1, "leiden_sec": 0.0}
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, c in enumerate(membership):
        groups[int(c)].append(i)
    labels: Dict[Any, Optional[str]] = {}
    for c, mem in groups.items():
        for i in mem:
            labels[ids[i]] = (f"L{c}" if len(mem) >= min_cluster_size else None)
    stats = dict(edges=len(edges), avg_degree=(2 * len(edges) / n) if n else 0.0, raw_communities=s2["raw_communities"],
                 largest=s2["largest_community"], sec=round(time.time() - t0, 2))
    return labels, stats


def dbscan_labels(ids: Sequence[Any], cand_nbr: np.ndarray, cand_sim: np.ndarray, eps: float, min_faces: int,
                  filter_matrix: Optional[np.ndarray] = None) -> Tuple[Dict[Any, Optional[str]], Dict[str, Any]]:
    """cand_nbr/cand_sim: 후보(자기 제외, 점수순). 후보 조건 sim ≥ 1−eps. filter_matrix 가 있으면 그 벡터의
    distance ≤ eps 로 다시 거른다(hybrid). 없으면 같은 벡터의 sim ≥ 1−eps 가 곧 distance ≤ eps."""
    n = len(ids)
    neighbors: Dict[int, List[int]] = {}
    pairs = 0
    t0 = time.time()
    for i in range(n):
        mask = (cand_nbr[i] >= 0) & (cand_sim[i] >= 1.0 - eps)
        cand = cand_nbr[i][mask]
        if filter_matrix is not None and cand.size:
            d = 1.0 - (filter_matrix[cand].astype(np.float64) @ filter_matrix[i].astype(np.float64))
            cand = cand[d <= eps]
        lst = cand.tolist()
        neighbors[i] = lst
        pairs += len(lst)
    cluster_map, st = assign_clusters(list(range(n)), neighbors, min_faces)
    sizes = Counter(v for v in cluster_map.values() if v != -1)
    labels = {ids[i]: (None if c == -1 else f"D{c}") for i, c in cluster_map.items()}
    stats = dict(neighbor_pairs=pairs, avg_degree=pairs / n if n else 0.0, raw_clusters=len(sizes),
                 singletons=sum(1 for s in sizes.values() if s == 1), largest=max(sizes.values(), default=0),
                 core_points=st["core_points"], sec=round(time.time() - t0, 2))
    return labels, stats


def load_gt(path: Path) -> Dict[Any, int]:
    pid_of: Dict[Any, int] = {}
    with path.open(encoding="utf-8") as f:
        for ln in f:
            d = json.loads(ln)
            if d.get("_meta"):
                continue
            if d.get("status") == "labeled":
                pid_of[d["point_id"]] = int(d["pid"])
    sizes = Counter(pid_of.values())
    return {p: pid for p, pid in pid_of.items() if sizes[pid] >= 2}


def load_assignment_labels(path: Path) -> Dict[Any, Optional[str]]:
    from compare_cluster_results import load_labels
    labels, _ = load_labels(path, dict(count=0, first_line=None))
    return labels


# ---------------------------------------------------------------------------
def row_from(res: Dict[str, Any], **head) -> Dict[str, Any]:
    pn, pc, b3 = res["pairs_noise_as_singletons"], res["pairs_clustered_only"], res["bcubed"]
    return dict(**head, labeled=res["labeled_points"], noise_ratio=res["noise_ratio"],
                pair_P=pn["precision"], pair_R=pn["recall"], pair_F1=pn["f1"], pair_R_clustered=pc["recall"], pair_F1_clustered=pc["f1"],
                B3_P=b3["precision"], B3_R=b3["recall"], B3_F1=b3["f1"], purity=res["purity"], inverse_purity=res["inverse_purity"],
                clusters_per_pid=res["clusters_per_pid"], pids_split=res["pids_split"], mixed_clusters=res["mixed_clusters"],
                mixed_points=res["mixed_cluster_points"], ari=res.get("ari_noise_as_singletons"), nmi=res.get("nmi_noise_as_singletons"))


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, int):
        return f"{v:,}"
    return esc(v)


def scatter_svg(rows: List[Dict[str, Any]], xkey: str, ykey: str, title: str) -> str:
    W, H, m = 520, 400, 50
    colors = {"solider": "#2563eb", "combined": "#dc2626", "hybrid": "#7c3aed"}
    def X(v): return m + v * (W - 2 * m)
    def Y(v): return H - m - v * (H - 2 * m)
    parts = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="background:#fff;border:1px solid #dfe3ea">',
             f'<text x="{W/2}" y="20" text-anchor="middle" font-size="13" font-weight="600">{esc(title)}</text>']
    for t in (0, .2, .4, .6, .8, 1):
        parts.append(f'<line x1="{X(t)}" y1="{Y(0)}" x2="{X(t)}" y2="{Y(1)}" stroke="#eef1f6"/>'
                     f'<line x1="{X(0)}" y1="{Y(t)}" x2="{X(1)}" y2="{Y(t)}" stroke="#eef1f6"/>'
                     f'<text x="{X(t)}" y="{Y(0)+14}" font-size="10" text-anchor="middle">{t}</text>'
                     f'<text x="{X(0)-6}" y="{Y(t)+3}" font-size="10" text-anchor="end">{t}</text>')
    parts.append(f'<text x="{W/2}" y="{H-8}" font-size="11" text-anchor="middle">{esc(xkey)}</text>'
                 f'<text x="12" y="{H/2}" font-size="11" text-anchor="middle" transform="rotate(-90 12 {H/2})">{esc(ykey)}</text>')
    for r in rows:
        x, y = r.get(xkey), r.get(ykey)
        if x is None or y is None:
            continue
        c = colors.get(r["vector"], "#111")
        label = f'{r["algorithm"]} {r["vector"]} {r["param"]}'
        if r["algorithm"] == "Leiden":
            parts.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="5" fill="{c}" fill-opacity="0.85"><title>{esc(label)}</title></circle>')
        else:
            parts.append(f'<rect x="{X(x)-4.5:.1f}" y="{Y(y)-4.5:.1f}" width="9" height="9" fill="{c}" fill-opacity="0.85"><title>{esc(label)}</title></rect>')
        parts.append(f'<text x="{X(x)+6:.1f}" y="{Y(y)-5:.1f}" font-size="9" fill="{c}">{esc(str(r["param"]))}</text>')
    parts.append(f'<text x="{W-m}" y="{m-8}" font-size="10" text-anchor="end">● Leiden  ■ DBSCAN  · 파랑=SOLIDER 단독  빨강=결합  보라=hybrid(원본)</text></svg>')
    return "".join(parts)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vector-cache", default="outputs/clustering/dbscan_image_prw/person/cache/vectors_prw_person.npz")
    p.add_argument("--gt-matches", default="eval/results/cache/prw_gt_matches.jsonl")
    p.add_argument("--leiden-thresholds-solider", default="0.90,0.92,0.94,0.95,0.96,0.97")
    p.add_argument("--leiden-thresholds-combined", default="0.80,0.85,0.88,0.90,0.92,0.94,0.95")
    p.add_argument("--dbscan-eps-solider", default="0.03,0.05,0.08,0.10,0.12,0.15")
    p.add_argument("--dbscan-eps-combined", default="0.06,0.08,0.10,0.12,0.15,0.18,0.20")
    p.add_argument("--hybrid-eps", default="0.12", help="원본 DBSCAN(SOLIDER 후보 + 결합 필터) 검증용 eps 목록")
    p.add_argument("--leiden-knn", type=int, default=30)
    p.add_argument("--dbscan-knn", type=int, default=25)
    p.add_argument("--min-faces", type=int, default=3)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-mutual", action="store_true")
    p.add_argument("--reference", action="append", default=[],
                   help="이름=assignments.jsonl : 기존 결과(Qdrant 기반)를 같은 GT 로 채점해 비교 행으로 추가")
    p.add_argument("--output-dir", default="eval/results/cluster_ablation_prw")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    data = np.load(args.vector_cache, allow_pickle=False)
    ids = [str(x) for x in data["ids"].tolist()]
    combined = l2n(data["matrix"])
    solider = l2n(data["knn_matrix"])
    meta = json.loads(str(data["meta"]))
    print(f"[ablation] points={len(ids):,} combined={combined.shape} solider={solider.shape} meta={meta}")
    pid_of = load_gt(Path(args.gt_matches))
    eval_ids = [p for p in ids if p in pid_of]
    print(f"[ablation] GT point {len(eval_ids):,} / 인물 {len(set(pid_of[p] for p in eval_ids)):,}")

    K = max(args.leiden_knn, args.dbscan_knn)
    knn: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, M in (("solider", solider), ("combined", combined)):
        print(f"[knn] {name}")
        idx, sim = exact_knn(M, K)
        knn[name] = drop_self(idx, sim, K)

    rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}

    def record(algorithm, vector, param, labels, stats):
        res = evaluate_method(labels, pid_of, eval_ids)
        clusters = len({v for v in labels.values() if v is not None})
        noise_all = sum(1 for v in labels.values() if v is None)
        row = row_from(res, algorithm=algorithm, vector=vector, param=param, clusters=clusters,
                       noise_all=noise_all, largest=stats.get("largest"))
        rows.append(row)
        details[f"{algorithm}|{vector}|{param}"] = dict(stats=stats, metrics={k: v for k, v in res.items() if k not in ("mixed_detail", "split_detail")})
        print(f"  {algorithm:<7} {vector:<9} {str(param):<6} clusters={clusters:>6,} noise={noise_all/len(ids):5.1%} "
              f"pairP={fmt(row['pair_P'])} pairR={fmt(row['pair_R'])} F1={fmt(row['pair_F1'])} B3F1={fmt(row['B3_F1'])} "
              f"purity={fmt(row['purity'])} cl/pid={fmt(row['clusters_per_pid'])} ({stats.get('sec')}s)")

    mutual = not args.no_mutual
    for vec, thr_text in (("solider", args.leiden_thresholds_solider), ("combined", args.leiden_thresholds_combined)):
        nbr, sc = knn[vec][0][:, :args.leiden_knn], knn[vec][1][:, :args.leiden_knn]
        for thr in [float(x) for x in thr_text.split(",") if x.strip()]:
            labels, st = leiden_labels(ids, nbr, sc, thr, mutual, args.resolution, args.seed, 2, log=lambda *a: None)
            record("Leiden", vec, thr, labels, st)
    for vec, eps_text in (("solider", args.dbscan_eps_solider), ("combined", args.dbscan_eps_combined)):
        nbr, sc = knn[vec][0][:, :args.dbscan_knn - 1], knn[vec][1][:, :args.dbscan_knn - 1]  # 상위 25 에 자기 포함 → 24
        for eps in [float(x) for x in eps_text.split(",") if x.strip()]:
            labels, st = dbscan_labels(ids, nbr, sc, eps, args.min_faces)
            record("DBSCAN", vec, eps, labels, st)
    for eps in [float(x) for x in args.hybrid_eps.split(",") if x.strip()]:
        nbr, sc = knn["solider"][0][:, :args.dbscan_knn - 1], knn["solider"][1][:, :args.dbscan_knn - 1]
        labels, st = dbscan_labels(ids, nbr, sc, eps, args.min_faces, filter_matrix=combined)
        record("DBSCAN", "hybrid", eps, labels, st)
    for ref in args.reference:
        name, path = ref.split("=", 1)
        labels = load_assignment_labels(Path(path))
        record("REF:" + name, "as-run", "-", labels, dict(largest=None, sec=0))

    # 저장
    keys = ["algorithm", "vector", "param", "clusters", "noise_all", "largest", "labeled", "noise_ratio", "pair_P", "pair_R", "pair_F1",
            "pair_R_clustered", "pair_F1_clustered", "B3_P", "B3_R", "B3_F1", "purity", "inverse_purity", "clusters_per_pid",
            "pids_split", "mixed_clusters", "mixed_points", "ari", "nmi"]
    with (out_dir / "ablation_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k) for k in keys])
    write_json(out_dir / "ablation_results.json", dict(points=len(ids), gt_points=len(eval_ids), vector_cache=str(Path(args.vector_cache).resolve()),
                                                       gt_matches=str(Path(args.gt_matches).resolve()), settings=vars(args), rows=rows,
                                                       details=details, elapsed_sec=round(time.time() - started, 1)))

    # HTML
    def tbl(subset):
        cols = [("param", "임계값/eps"), ("clusters", "클러스터"), ("noise_all", "노이즈(전체)"), ("largest", "최대"), ("pair_P", "쌍 P"), ("pair_R", "쌍 R"),
                ("pair_F1", "쌍 F1"), ("pair_R_clustered", "쌍 R(배정만)"), ("B3_P", "B³ P"), ("B3_R", "B³ R"), ("B3_F1", "B³ F1"),
                ("purity", "purity"), ("clusters_per_pid", "클러스터/인물"), ("pids_split", "갈라진 인물"), ("mixed_clusters", "혼합 클러스터")]
        head = "".join(f"<th>{esc(t)}</th>" for _, t in cols)
        body = "".join("<tr>" + "".join(f"<td>{fmt(r.get(k))}</td>" for k, _ in cols) + "</tr>" for r in subset)
        return f"<table><tr>{head}</tr>{body}</table>"
    sections = []
    for algo in ("Leiden", "DBSCAN"):
        for vec in ("solider", "combined", "hybrid"):
            sub = [r for r in rows if r["algorithm"] == algo and r["vector"] == vec]
            if sub:
                sections.append(f"<h3>{esc(algo)} × {esc(vec)}</h3>{tbl(sub)}")
    refs = [r for r in rows if r["algorithm"].startswith("REF:")]
    if refs:
        sections.append("<h3>기존 실행 결과 (Qdrant 기반, 같은 GT 로 채점)</h3>" + tbl(refs).replace("<th>임계값/eps</th>", "<th>—</th>"))
    best = {}
    for algo in ("Leiden", "DBSCAN"):
        for vec in ("solider", "combined"):
            sub = [r for r in rows if r["algorithm"] == algo and r["vector"] == vec and r["B3_F1"] is not None]
            if sub:
                best[(algo, vec)] = max(sub, key=lambda r: r["B3_F1"])
    best_tbl = "<table><tr><th></th><th>SOLIDER 단독</th><th>결합 벡터</th></tr>" + "".join(
        f"<tr><th>{algo}</th>" + "".join(
            (f"<td>B³F1 <b>{fmt(b['B3_F1'])}</b> · 쌍F1 {fmt(b['pair_F1'])} · P {fmt(b['pair_P'])} · R {fmt(b['pair_R'])} · purity {fmt(b['purity'])} "
             f"· 클러스터/인물 {fmt(b['clusters_per_pid'])} · 노이즈 {b['noise_all']/len(ids):.1%} <span class='muted'>({b['param']})</span></td>")
            if (b := best.get((algo, vec))) else "<td>—</td>" for vec in ("solider", "combined")) + "</tr>" for algo in ("Leiden", "DBSCAN")) + "</table>"
    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>알고리즘 효과 분리 (PRW GT)</title>
<style>body{{font-family:'Malgun Gothic',Segoe UI,sans-serif;max-width:1200px;margin:0 auto;padding:22px 26px;background:#f6f7f9;color:#1f2430}}
h1{{font-size:22px}} h2{{font-size:17px;border-bottom:2px solid #d8dbe2;padding-bottom:4px;margin-top:26px}} h3{{font-size:14px;margin:16px 0 6px}}
table{{border-collapse:collapse;background:#fff;font-size:12.5px;margin:6px 0}} th,td{{border:1px solid #dfe3ea;padding:4px 7px;text-align:left}} th{{background:#eef1f6}}
.muted{{color:#6b7280;font-size:12px}} .two{{display:flex;gap:16px;flex-wrap:wrap}} nav a{{color:#2563eb;text-decoration:none}}</style></head><body>
<h1>알고리즘 효과 분리 — Leiden vs DBSCAN × SOLIDER 단독 vs 결합 벡터 (PRW person, GT 채점)</h1>
<nav class="muted"><a href="../cluster_gt_prw/cluster_gt_report.html">GT 평가 보고서</a> · <a href="../cluster_gt_prw/cluster_gt_prw.html">GT 평가 상세</a></nav>
<p class="muted">point {len(ids):,} (PRW 이미지 검출) · GT 인물 ID 가 붙은 {len(eval_ids):,} 점으로 채점 · 정확 kNN(메모리) · Leiden k={args.leiden_knn} 상호={mutual} resolution={args.resolution} · DBSCAN K={args.dbscan_knn} MIN_FACES={args.min_faces}<br>
결합 벡터 cosine = 성분 cosine 가중 평균 (SigLIP2 2.2% · IRRA 19.6% · SOLIDER 78.3%) — 결합 벡터도 SOLIDER 가 지배적.</p>
<h2>1. 각 칸의 최선 설정 (B-cubed F1 기준)</h2>{best_tbl}
<h2>2. 정밀도–재현율</h2><div class="two">{scatter_svg(rows, 'B3_R', 'B3_P', 'B-cubed 재현율 vs 정밀도')}{scatter_svg(rows, 'pair_R', 'pair_P', '쌍 재현율 vs 정밀도 (노이즈=단독)')}</div>
<h2>3. 스윕 전체</h2>{''.join(sections)}
<p class="muted">쌍 P = 같은 클러스터 쌍 중 실제 같은 인물, 쌍 R = 같은 인물 쌍 중 같은 클러스터(노이즈=단독 클러스터), B³ = B-cubed. 원본 파일: ablation_results.csv / .json</p>
</body></html>"""
    (out_dir / "ablation_prw.html").write_text(html, encoding="utf-8", newline="\n")
    print(f"\nhtml : {out_dir / 'ablation_prw.html'}\ncsv  : {out_dir / 'ablation_results.csv'}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
