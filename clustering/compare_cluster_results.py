#!/usr/bin/env python
"""두 클러스터링 결과(assignments.jsonl)를 같은 point 집합 위에서 비교한다.

입력은 cluster_leiden_qdrant.py / cluster_dbscan_qdrant.py 가 쓰는 jsonl
(`point_id`, `cluster_id`(노이즈면 null), `noise`) 이면 무엇이든 된다.

비교 항목
  * 방법별 요약: 클러스터 수, 배정/노이즈, 최대 크기, 크기 분포
  * 일치도: ARI / NMI (노이즈를 각각 단독 클러스터로 본 경우, 양쪽 모두 배정된 점만 본 경우),
    쌍(pair) 단위 정밀도/재현율/F1/Jaccard, 노이즈 2x2 교차표
  * 클러스터 대응: A 의 각 클러스터가 B 에서 그대로(identical) / 더 큰 클러스터에 흡수(merged_in_other) /
    여러 조각으로 분할(split_in_other) / 섞임(mixed) / 전부 노이즈(all_noise_in_other) 인지, 그 반대 방향도
  * 예시 갤러리: 분할·흡수·노이즈 불일치·완전 일치 클러스터의 crop 썸네일 (Qdrant payload 의 crop_path)

산출물: <output-dir>/<html-name>, compare_report.json(사이드카), mapping_a_to_b.jsonl, mapping_b_to_a.jsonl
마지막에 RESULT_SUMMARY / RESULT_HTML 마커를 출력한다 (GUI 규약).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from report_common import (CONFIG_HELP, DEFAULT_CONFIG_PATH, cluster_details, common_summary, esc, file_info,
                           histogram_html, is_noise, iter_assignments, load_json, load_pipeline_settings,
                           path_href, resolve, result_markers, size_histogram, table, warning, write_json)

PRODUCER = "compare_cluster_results"
RESERVED_OUTPUTS = frozenset({"compare_report.json", "mapping_a_to_b.jsonl", "mapping_b_to_a.jsonl"})
CATEGORIES = ("identical", "merged_in_other", "split_in_other", "mixed", "all_noise_in_other")
CATEGORY_KO = {"identical": "그대로 일치", "merged_in_other": "상대가 더 크게 묶음(흡수)",
               "split_in_other": "상대가 쪼갬(부분집합들)", "mixed": "섞임(경계 다름)",
               "all_noise_in_other": "상대는 전부 노이즈"}
JACCARD_BINS = (("1.0", lambda j: j >= 1.0), ("0.9–1.0", lambda j: 0.9 <= j < 1.0), ("0.7–0.9", lambda j: 0.7 <= j < 0.9),
                ("0.5–0.7", lambda j: 0.5 <= j < 0.7), ("<0.5", lambda j: j < 0.5))


# ---------------------------------------------------------------------------
# 순수 계산 (테스트 대상)
# ---------------------------------------------------------------------------
def load_labels(path, errors) -> Tuple[Dict[Any, Optional[str]], int]:
    """point_id -> cluster_id(str) 또는 None(노이즈). 중복 point_id 는 마지막 값이 이기고 개수를 센다."""
    labels: Dict[Any, Optional[str]] = {}
    duplicates = 0
    for row in iter_assignments(path, errors):
        pid = row["point_id"]
        if isinstance(pid, bool) or not isinstance(pid, (str, int)):
            # iter_assignments 는 키 존재만 보므로 자료형은 여기서 걸러 같은 오류 집계에 넣는다
            errors["count"] += 1
            if errors.get("first_line") is None:
                errors["first_line"] = "point_id 자료형 오류"
            continue
        if pid in labels:
            duplicates += 1
        labels[pid] = None if is_noise(row) else str(row["cluster_id"])
    return labels, duplicates


def apply_min_cluster_size(labels: Dict[Any, Optional[str]], min_size: int) -> int:
    """크기 < min_size 인 클러스터의 멤버를 노이즈(None)로 바꾼다. 반환: 바뀐 point 수.

    Leiden 산출물은 min_cluster_size=2 로 이미 단독점을 노이즈로 두므로, DBSCAN 처럼 크기 1
    클러스터를 남기는 방법과 같은 기준으로 비교하려면 양쪽에 같은 값을 적용한다.
    """
    if min_size <= 1:
        return 0
    sizes = Counter(v for v in labels.values() if v is not None)
    small = {cid for cid, n in sizes.items() if n < min_size}
    changed = 0
    for pid, cid in labels.items():
        if cid in small:
            labels[pid] = None
            changed += 1
    return changed


def method_stats(labels: Dict[Any, Optional[str]], ids: Sequence[Any]) -> Dict[str, Any]:
    counts = Counter(labels[p] for p in ids if labels[p] is not None)
    noise = sum(1 for p in ids if labels[p] is None)
    sizes = sorted(counts.values())
    return dict(points=len(ids), clusters=len(counts), clustered=len(ids) - noise, noise=noise,
                noise_ratio=(noise / len(ids)) if ids else None, largest=sizes[-1] if sizes else 0,
                median_cluster_size=float(statistics.median(sizes)) if sizes else 0,
                singleton_clusters=sum(1 for s in sizes if s == 1), size_histogram=size_histogram(sizes))


def encode(labels: Dict[Any, Optional[str]], ids: Sequence[Any]) -> List[int]:
    """클러스터 id -> 정수. 노이즈는 점마다 서로 다른 음수 (= 단독 클러스터)."""
    codes: Dict[str, int] = {}
    out: List[int] = []
    for i, p in enumerate(ids):
        label = labels[p]
        if label is None:
            out.append(-1 - i)
        else:
            out.append(codes.setdefault(label, len(codes)))
    return out


def _c2(n: int) -> int:
    return n * (n - 1) // 2


def pair_counts(labels_a, labels_b, ids) -> Dict[str, Any]:
    ca: Counter = Counter()
    cb: Counter = Counter()
    cab: Counter = Counter()
    for p in ids:
        a, b = labels_a[p], labels_b[p]
        if a is not None:
            ca[a] += 1
        if b is not None:
            cb[b] += 1
        if a is not None and b is not None:
            cab[(a, b)] += 1
    pa = sum(_c2(n) for n in ca.values())
    pb = sum(_c2(n) for n in cb.values())
    both = sum(_c2(n) for n in cab.values())
    precision = (both / pb) if pb else None      # B 가 같은 클러스터로 묶은 쌍 중 A 도 묶은 비율
    recall = (both / pa) if pa else None         # A 가 묶은 쌍 중 B 도 묶은 비율
    f1 = None
    if precision is not None and recall is not None:
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    union = pa + pb - both
    return dict(pairs_a=pa, pairs_b=pb, pairs_both=both, b_pair_precision=precision, b_pair_recall=recall,
                pair_f1=f1, pair_jaccard=(both / union) if union else None)


def noise_crosstab(labels_a, labels_b, ids) -> Dict[str, int]:
    t = dict(both_noise=0, a_noise_only=0, b_noise_only=0, both_clustered=0)
    for p in ids:
        a_noise, b_noise = labels_a[p] is None, labels_b[p] is None
        key = ("both_noise" if a_noise and b_noise else "a_noise_only" if a_noise
               else "b_noise_only" if b_noise else "both_clustered")
        t[key] += 1
    return t


def agreement(labels_a, labels_b, ids) -> Dict[str, Any]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ea, eb = encode(labels_a, ids), encode(labels_b, ids)
    both = [p for p in ids if labels_a[p] is not None and labels_b[p] is not None]
    result: Dict[str, Any] = dict(points=len(ids),
                                  ari_noise_as_singletons=float(adjusted_rand_score(ea, eb)) if ids else None,
                                  nmi_noise_as_singletons=float(normalized_mutual_info_score(ea, eb)) if ids else None,
                                  both_clustered=len(both))
    if both:
        fa, fb = encode(labels_a, both), encode(labels_b, both)
        result["ari_both_clustered"] = float(adjusted_rand_score(fa, fb))
        result["nmi_both_clustered"] = float(normalized_mutual_info_score(fa, fb))
    else:
        result["ari_both_clustered"] = result["nmi_both_clustered"] = None
    result.update(pair_counts(labels_a, labels_b, ids))
    result["noise_crosstab"] = noise_crosstab(labels_a, labels_b, ids)
    return result


def cluster_mapping(labels_from, labels_to, ids) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """from 의 각 클러스터가 to 에서 어떻게 보이는지. rows 는 크기 내림차순."""
    members: Dict[str, List[Any]] = defaultdict(list)
    to_sizes: Counter = Counter()
    for p in ids:
        if labels_from[p] is not None:
            members[labels_from[p]].append(p)
        if labels_to[p] is not None:
            to_sizes[labels_to[p]] += 1

    rows: List[Dict[str, Any]] = []
    for cid, pts in members.items():
        tc = Counter(labels_to[p] for p in pts)
        noise_to = tc.pop(None, 0)
        size = len(pts)
        if tc:
            dominant, dominant_n = tc.most_common(1)[0]
            best_j = max(n / (size + to_sizes[t] - n) for t, n in tc.items())
        else:
            dominant, dominant_n, best_j = None, 0, 0.0
        parts = len(tc)
        if parts == 0:
            category = "all_noise_in_other"
        elif best_j >= 1.0:
            category = "identical"
        elif parts == 1 and noise_to == 0 and dominant_n < to_sizes[dominant]:
            category = "merged_in_other"
        elif all(n == to_sizes[t] for t, n in tc.items()):
            category = "split_in_other"
        else:
            category = "mixed"
        rows.append(dict(cluster_id=cid, size=size, parts=parts, noise_in_other=noise_to, dominant=dominant,
                         dominant_size=dominant_n, purity=dominant_n / size, best_jaccard=best_j,
                         category=category, parts_detail={str(t): n for t, n in tc.most_common()}))
    rows.sort(key=lambda r: (-r["size"], r["cluster_id"]))

    by_cat_clusters = Counter(r["category"] for r in rows)
    by_cat_points = Counter()
    for r in rows:
        by_cat_points[r["category"]] += r["size"]
    total_points = sum(r["size"] for r in rows)
    jac_hist = {name: 0 for name, _ in JACCARD_BINS}
    for r in rows:
        for name, test in JACCARD_BINS:
            if test(r["best_jaccard"]):
                jac_hist[name] += 1
                break
    summary = dict(clusters=len(rows), points=total_points,
                   categories={c: dict(clusters=by_cat_clusters.get(c, 0), points=by_cat_points.get(c, 0))
                               for c in CATEGORIES},
                   mean_purity=(sum(r["purity"] for r in rows) / len(rows)) if rows else None,
                   point_weighted_purity=(sum(r["dominant_size"] for r in rows) / total_points) if total_points else None,
                   best_jaccard_histogram=jac_hist)
    return rows, summary


def pick_examples(rows_ab, rows_ba, k: int, min_size: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    def top(rows, key, pred):
        return sorted([r for r in rows if pred(r) and r["size"] >= min_size], key=key)[:k]
    return {
        "a_split_by_b": top(rows_ab, lambda r: (-r["parts"], -r["size"]), lambda r: r["parts"] >= 2),
        "b_split_by_a": top(rows_ba, lambda r: (-r["parts"], -r["size"]), lambda r: r["parts"] >= 2),
        "a_cluster_b_noise": top(rows_ab, lambda r: (-r["noise_in_other"], -r["size"]), lambda r: r["noise_in_other"] > 0),
        "b_cluster_a_noise": top(rows_ba, lambda r: (-r["noise_in_other"], -r["size"]), lambda r: r["noise_in_other"] > 0),
        "identical": top(rows_ab, lambda r: (-r["size"], r["cluster_id"]), lambda r: r["category"] == "identical"),
    }


def sample_members(pts: Sequence[Any], labels_to, limit: int, rng: random.Random) -> List[Tuple[Optional[str], List[Any], int]]:
    """상대 라벨별로 묶고, 큰 그룹부터 최소 2장씩 보장하며 limit 장을 배분한다. 노이즈 그룹은 마지막.

    반환: [(상대 라벨 또는 None, 고른 point 목록, 그 그룹의 전체 크기), ...]
    """
    groups: Dict[Optional[str], List[Any]] = defaultdict(list)
    for p in pts:
        groups[labels_to[p]].append(p)
    ordered = sorted(groups.items(), key=lambda kv: (kv[0] is None, -len(kv[1]), str(kv[0])))
    total = len(pts)
    quota: Dict[Optional[str], int] = {}
    remaining = limit
    for key, members in ordered:
        share = max(min(2, len(members)), round(limit * len(members) / total)) if total else 0
        quota[key] = min(share, len(members), remaining)
        remaining -= quota[key]
    result = []
    for key, members in ordered:
        take = quota.get(key, 0)
        if take <= 0:
            continue
        chosen = members[:take] if len(members) <= take else rng.sample(members, take)
        result.append((key, chosen, len(members)))
    return result


EXAMPLE_FROM_A = ("a_split_by_b", "a_cluster_b_noise", "identical")


def build_example_groups(examples, members_a, members_b, labels_a, labels_b, per_example: int, rng: random.Random):
    """사례별 표본을 **한 번만** 뽑는다. payload 조회와 HTML 렌더링이 같은 표본을 쓴다."""
    result: Dict[str, List[Tuple[Dict[str, Any], list]]] = {}
    for key, rows in examples.items():
        from_members, to_labels = (members_a, labels_b) if key in EXAMPLE_FROM_A else (members_b, labels_a)
        result[key] = [(r, sample_members(from_members[r["cluster_id"]], to_labels, per_example, rng)) for r in rows]
    return result


def example_ids(example_groups) -> set:
    return {pid for groups in example_groups.values() for _, sampled in groups for _, chosen, _ in sampled for pid in chosen}


# ---------------------------------------------------------------------------
# Qdrant payload / 썸네일
# ---------------------------------------------------------------------------
def fetch_payloads(url: str, api_key: Optional[str], collection: str, ids: Sequence[Any], batch: int = 256,
                   timeout: int = 60) -> Dict[Any, Dict[str, Any]]:
    import requests
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    if api_key:
        session.headers.update({"api-key": api_key})
    out: Dict[Any, Dict[str, Any]] = {}
    ids = list(ids)
    for i in range(0, len(ids), batch):
        part = ids[i:i + batch]
        r = session.post(f"{url.rstrip('/')}/collections/{collection}/points",
                         json={"ids": part, "with_payload": True, "with_vector": False}, timeout=timeout)
        if not r.ok:
            raise RuntimeError(f"points retrieve -> {r.status_code}\n{r.text[:500]}")
        for p in (r.json().get("result") or []):
            out[p["id"]] = p.get("payload") or {}
    return out


def resolve_crop(payload: Dict[str, Any], project_root: Path) -> Optional[Path]:
    for key in ("crop_path", "image_path", "path"):
        raw = payload.get(key)
        if not raw:
            continue
        p = Path(str(raw))
        for candidate in ((p,) if p.is_absolute() else (project_root / p, project_root / str(raw).replace("\\", "/"))):
            if candidate.is_file():
                return candidate.resolve()
    return None


def thumb_data_uri(src: Path, max_side: int, quality: int) -> Optional[str]:
    try:
        import base64
        import io
        from PIL import Image
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
CSS = """
body{font-family:'Malgun Gothic',Segoe UI,sans-serif;margin:0;padding:20px 28px;background:#f6f7f9;color:#1f2430}
h1{font-size:22px;margin:0 0 6px} h2{font-size:17px;margin:28px 0 10px;border-bottom:2px solid #d8dbe2;padding-bottom:4px}
h3{font-size:14px;margin:18px 0 6px} .muted{color:#6b7280;font-size:13px}
table{border-collapse:collapse;background:#fff;font-size:13px;margin:6px 0}
th,td{border:1px solid #dfe3ea;padding:4px 8px;text-align:left;vertical-align:top} th{background:#eef1f6;font-weight:600}
table table{margin:0} .bar{background:#7c9cd9;height:12px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px} .card{background:#fff;border:1px solid #dfe3ea;border-radius:6px;padding:10px 12px}
.ex{background:#fff;border:1px solid #dfe3ea;border-radius:6px;padding:10px 12px;margin:10px 0}
.grp{display:flex;flex-wrap:wrap;gap:4px;align-items:flex-start;margin:6px 0}
.chip{display:inline-block;font-size:12px;padding:2px 8px;border-radius:10px;color:#fff;margin:2px 6px 2px 0}
.grp img{height:%(h)dpx;border:3px solid var(--c);border-radius:3px;background:#ddd}
.grp .ph{height:%(h)dpx;width:60px;border:3px dashed var(--c);font-size:11px;color:#888;display:flex;align-items:center;justify-content:center}
code{font-size:12px;background:#eef1f6;padding:1px 4px;border-radius:3px}
"""
PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#4d7c0f", "#b45309", "#1d4ed8"]


def short(cid: Optional[str]) -> str:
    if cid is None:
        return "노이즈"
    return cid.rsplit(":", 1)[-1][:12] if ":" in cid else cid[:12]


def fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, int):
        return f"{v:,}"
    return esc(v)


def side_by_side(label_a, stats_a, label_b, stats_b) -> str:
    keys = [("points", "대상 point"), ("clusters", "클러스터 수"), ("clustered", "배정 point"), ("noise", "노이즈"),
            ("noise_ratio", "노이즈 비율"), ("largest", "최대 클러스터"), ("median_cluster_size", "중앙값 크기"),
            ("singleton_clusters", "크기 1 클러스터")]
    rows = "".join(f"<tr><th>{esc(ko)}</th><td>{fmt(stats_a.get(k))}</td><td>{fmt(stats_b.get(k))}</td></tr>"
                   for k, ko in keys)
    return (f"<table><tr><th></th><th>{esc(label_a)}</th><th>{esc(label_b)}</th></tr>{rows}</table>"
            f"<div class='two'><div><h3>{esc(label_a)} 크기 분포</h3>{histogram_html(stats_a['size_histogram'])}</div>"
            f"<div><h3>{esc(label_b)} 크기 분포</h3>{histogram_html(stats_b['size_histogram'])}</div></div>")


def agreement_html(label_a, label_b, agr) -> str:
    ct = agr["noise_crosstab"]
    core = {"ARI (노이즈=단독 클러스터)": agr["ari_noise_as_singletons"], "NMI (노이즈=단독 클러스터)": agr["nmi_noise_as_singletons"],
            "양쪽 모두 배정된 point": agr["both_clustered"], "ARI (양쪽 배정 point 만)": agr["ari_both_clustered"],
            "NMI (양쪽 배정 point 만)": agr["nmi_both_clustered"]}
    pairs = {f"{label_a} 가 같은 클러스터로 묶은 쌍": agr["pairs_a"], f"{label_b} 가 묶은 쌍": agr["pairs_b"],
             "양쪽 모두 묶은 쌍": agr["pairs_both"],
             f"쌍 정밀도 ({label_b} 쌍 중 {label_a} 도 묶음)": agr["b_pair_precision"],
             f"쌍 재현율 ({label_a} 쌍 중 {label_b} 도 묶음)": agr["b_pair_recall"],
             "쌍 F1": agr["pair_f1"], "쌍 Jaccard": agr["pair_jaccard"]}
    cross = (f"<table><tr><th></th><th>{esc(label_b)} 배정</th><th>{esc(label_b)} 노이즈</th></tr>"
             f"<tr><th>{esc(label_a)} 배정</th><td>{ct['both_clustered']:,}</td><td>{ct['b_noise_only']:,}</td></tr>"
             f"<tr><th>{esc(label_a)} 노이즈</th><td>{ct['a_noise_only']:,}</td><td>{ct['both_noise']:,}</td></tr></table>")
    def kv(d):
        return "<table>" + "".join(f"<tr><th>{esc(k)}</th><td>{fmt(v)}</td></tr>" for k, v in d.items()) + "</table>"
    return (f"<div class='two'><div class='card'><h3>군집 일치도</h3>{kv(core)}</div>"
            f"<div class='card'><h3>쌍(pair) 단위</h3>{kv(pairs)}</div></div>"
            f"<h3>노이즈 교차표</h3>{cross}")


def mapping_html(label_from, label_to, summary) -> str:
    cats = summary["categories"]
    rows = "".join(f"<tr><th>{esc(CATEGORY_KO[c])}</th><td>{cats[c]['clusters']:,}</td><td>{cats[c]['points']:,}</td></tr>"
                   for c in CATEGORIES)
    return (f"<div class='card'><h3>{esc(label_from)} 클러스터 → {esc(label_to)} 에서의 모습</h3>"
            f"<table><tr><th>분류</th><th>클러스터 수</th><th>point 수</th></tr>{rows}</table>"
            f"<table><tr><th>평균 purity</th><td>{fmt(summary['mean_purity'])}</td></tr>"
            f"<tr><th>point 가중 purity</th><td>{fmt(summary['point_weighted_purity'])}</td></tr></table>"
            f"<h3>최적 Jaccard 분포</h3>{histogram_html(summary['best_jaccard_histogram'])}</div>")


def examples_html(example_groups, names, payloads, project_root, inline, thumb, quality, hrefs_base,
                  show_images: bool = True) -> str:
    """show_images=False (--no-images) 면 썸네일 자리 없이 조각별 개수(chip)만 그린다."""
    parts = []
    titles = {"a_split_by_b": (names['a'], names['b'], "{a} 쪽은 한 클러스터, {b} 쪽은 여러 조각으로 쪼갬"),
              "b_split_by_a": (names['b'], names['a'], "{b} 쪽은 한 클러스터, {a} 쪽은 여러 조각으로 쪼갬"),
              "a_cluster_b_noise": (names['a'], names['b'], "{a} 쪽은 묶었는데 {b} 쪽은 노이즈로 둔 멤버가 많은 클러스터"),
              "b_cluster_a_noise": (names['b'], names['a'], "{b} 쪽은 묶었는데 {a} 쪽은 노이즈로 둔 멤버가 많은 클러스터"),
              "identical": (names['a'], names['b'], "양쪽이 완전히 같은 클러스터 (큰 것부터)")}
    thumb_cache: Dict[Any, Optional[str]] = {}
    for key, entries in example_groups.items():
        lf, lt, title = titles[key]
        parts.append(f"<h2>{esc(title.format(a=names['a'], b=names['b']))}</h2>")
        if not entries:
            parts.append("<p class='muted'>해당 사례 없음</p>")
            continue
        for r, groups in entries:
            head = (f"<div class='ex'><b>{esc(lf)}</b> 클러스터 <code>{esc(short(r['cluster_id']))}</code> · 크기 {r['size']:,} · "
                    f"{esc(lt)} 쪽 조각 {r['parts']} 개 · {esc(lt)} 노이즈 {r['noise_in_other']:,} · purity {r['purity']:.2f} · "
                    f"best Jaccard {r['best_jaccard']:.2f} · <span class='muted'>{esc(CATEGORY_KO[r['category']])}</span>")
            body = []
            for gi, (to_label, chosen, n_total) in enumerate(groups):
                color = "#9ca3af" if to_label is None else PALETTE[gi % len(PALETTE)]
                chip = f"<span class='chip' style='background:{color}'>{esc(lt)}: {esc(short(to_label))} ({n_total:,}장 중 {len(chosen)}장)</span>"
                imgs = []
                for pid in (chosen if show_images else ()):
                    payload = payloads.get(pid) or {}
                    src = resolve_crop(payload, project_root)
                    title_txt = esc(payload.get("image_id") or payload.get("crop_path") or pid)
                    if src is None:
                        imgs.append(f"<div class='ph' style='--c:{color}' title='{title_txt}'>없음</div>")
                    elif inline:
                        if pid not in thumb_cache:
                            thumb_cache[pid] = thumb_data_uri(src, thumb, quality)
                        uri = thumb_cache[pid]
                        imgs.append(f"<img style='--c:{color}' src='{uri}' title='{title_txt}'>" if uri
                                    else f"<div class='ph' style='--c:{color}' title='{title_txt}'>실패</div>")
                    else:
                        imgs.append(f"<img style='--c:{color}' src='{path_href(src, hrefs_base)}' title='{title_txt}' loading='lazy'>")
                body.append(f"<div class='grp' style='--c:{color}'>{chip}{''.join(imgs)}</div>")
            parts.append(head + "".join(body) + "</div>")
    return "".join(parts)


def render_html(title, names, stats, agr, map_ab, map_ba, examples_block, inputs, reports, warnings, thumb) -> str:
    warn_html = ("<ul>" + "".join(f"<li>{esc(w['message'])}</li>" for w in warnings) + "</ul>") if warnings else "<p class='muted'>없음</p>"
    inputs_html = table({i["role"]: {"path": i["path"], "sha256": (i.get("sha256") or "없음")[:16], "size": i.get("size")} for i in inputs})
    cfg_html = "".join(f"<div class='card'><h3>{esc(label)} 설정</h3>{table(cluster_details(rep))}</div>"
                       for label, rep in reports.items())
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>{esc(title)}</title>
<style>{CSS % dict(h=thumb)}</style></head><body>
<h1>{esc(title)}</h1>
<p class="muted">A = {esc(names['a'])} · B = {esc(names['b'])} · 공통 point {stats['common']:,} · 생성 {esc(stats['generated_at'])}</p>
<h2>1. 방법별 요약 (공통 point 기준)</h2>{side_by_side(names['a'], stats['a'], names['b'], stats['b'])}
<h2>2. 일치도</h2>{agreement_html(names['a'], names['b'], agr)}
<h2>3. 클러스터 대응</h2><div class="two">{mapping_html(names['a'], names['b'], map_ab)}{mapping_html(names['b'], names['a'], map_ba)}</div>
<p class="muted">purity = 클러스터 멤버 중 상대 방법의 최다 클러스터에 속한 비율(노이즈 제외). best Jaccard = 상대 클러스터와의 최대 |교집합|/|합집합|.</p>
{examples_block}
<h2>입력 · 설정</h2><div class="two">{cfg_html}</div>{inputs_html}
<h2>경고</h2>{warn_html}
</body></html>"""


# ---------------------------------------------------------------------------
def guess_label(path: Path, fallback: str) -> str:
    name = path.name.lower()
    for token, label in (("leiden", "Leiden"), ("dbscan", "DBSCAN"), ("kmeans", "KMeans"), ("hdbscan", "HDBSCAN")):
        if token in name:
            return label
    return fallback


def sibling_report(path: Path) -> Optional[Path]:
    if path.name.endswith("_assignments.jsonl"):
        cand = path.with_name(path.name[:-len("_assignments.jsonl")] + "_report.json")
        if cand.is_file():
            return cand
    return None


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="assignments.jsonl (기준 A, 예: Leiden)")
    p.add_argument("--b", required=True, help="assignments.jsonl (비교 B, 예: DBSCAN)")
    p.add_argument("--a-label", default=None)
    p.add_argument("--b-label", default=None)
    p.add_argument("--a-report", default=None, help="A 의 report JSON (기본: 옆의 *_report.json)")
    p.add_argument("--b-report", default=None)
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=CONFIG_HELP)
    p.add_argument("--target", choices=["person", "object"], default="person")
    p.add_argument("--collection", default=None, help=CONFIG_HELP)
    p.add_argument("--qdrant-url", default=None, help=CONFIG_HELP)
    p.add_argument("--api-key", default=None)
    p.add_argument("--project-root", default=".")
    p.add_argument("--output-dir", default="outputs/clustering/compare")
    p.add_argument("--html-name", default="compare_clusters.html")
    p.add_argument("--title", default=None)
    p.add_argument("--examples", type=int, default=5, help="사례 유형별 클러스터 수")
    p.add_argument("--images-per-example", type=int, default=24)
    p.add_argument("--min-example-size", type=int, default=3)
    p.add_argument("--min-cluster-size", type=int, default=1,
                   help="양쪽 모두 이 크기 미만 클러스터를 노이즈로 취급 (Leiden 기본 2 와 맞추려면 2)")
    p.add_argument("--inline-images", action="store_true", help="썸네일을 base64 로 내장 (단일 파일)")
    p.add_argument("--no-images", action="store_true", help="Qdrant 조회/썸네일 생략 (통계만)")
    p.add_argument("--thumb-size", type=int, default=160)
    p.add_argument("--thumb-quality", type=int, default=70)
    p.add_argument("--seed", type=int, default=42)
    return p


def parse_args(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    try:
        settings = load_pipeline_settings(args.config, require=not args.no_images and not (args.qdrant_url and args.collection))
        if not args.no_images:
            args.qdrant_url = resolve(args.qdrant_url, settings.qdrant_url if settings else None, "qdrant-url")
            args.collection = resolve(args.collection, settings.collection_for(args.target) if settings else None, "collection")
        if args.examples < 0 or args.images_per_example < 1 or args.thumb_size < 16 or args.min_cluster_size < 1:
            raise ValueError("examples >= 0, images-per-example >= 1, thumb-size >= 16, min-cluster-size >= 1")
        html_name = Path(args.html_name)
        if html_name.name != args.html_name or not args.html_name.lower().endswith((".html", ".htm")):
            raise ValueError(f"html-name 은 하위 경로 없는 .html 파일명이어야 합니다: {args.html_name!r}")
        if args.html_name in RESERVED_OUTPUTS:
            raise ValueError(f"html-name 이 예약된 산출물 이름과 겹칩니다: {sorted(RESERVED_OUTPUTS)}")
    except (OSError, ValueError, AttributeError) as exc:
        p.error(str(exc))
    args.pipeline_settings = settings
    return args


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    a_path, b_path = Path(args.a).resolve(), Path(args.b).resolve()
    names = {"a": args.a_label or guess_label(a_path, "A"), "b": args.b_label or guess_label(b_path, "B")}
    if names["a"] == names["b"]:
        names["b"] += " (B)"
    warnings: List[Dict[str, Any]] = []
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / args.html_name
    sidecar = out_dir / "compare_report.json"

    inputs = [file_info("assignments_a", a_path), file_info("assignments_b", b_path)]
    reports: Dict[str, Any] = {}
    for key, path, override in (("a", a_path, args.a_report), ("b", b_path, args.b_report)):
        rp = Path(override).resolve() if override else sibling_report(path)
        if rp is None:
            warnings.append(warning("NO_REPORT", f"{names[key]}: report JSON 없음 (설정 표시 생략)", args.target))
            reports[names[key]] = None
            continue
        inputs.append(file_info(f"report_{key}", rp))
        reports[names[key]] = load_json(rp, warnings, args.target)

    errors_a, errors_b = dict(count=0, first_line=None), dict(count=0, first_line=None)
    labels_a, dup_a = load_labels(a_path, errors_a)
    labels_b, dup_b = load_labels(b_path, errors_b)
    for key, err, dup, labels in (("a", errors_a, dup_a, labels_a), ("b", errors_b, dup_b, labels_b)):
        if err["count"]:
            warnings.append(warning("PARSE_ERRORS", f"{names[key]}: 잘못된 줄 {err['count']} (첫 줄 {err['first_line']})", args.target))
        if dup:
            warnings.append(warning("DUPLICATE_POINT_ID", f"{names[key]}: 중복 point_id {dup}", args.target))
        if not labels:
            warnings.append(warning("NO_ASSIGNMENTS", f"{names[key]}: 유효한 assignments 없음", args.target))

    demoted = {"a": apply_min_cluster_size(labels_a, args.min_cluster_size),
               "b": apply_min_cluster_size(labels_b, args.min_cluster_size)}
    if args.min_cluster_size > 1:
        print(f"[compare] min-cluster-size={args.min_cluster_size}: 노이즈로 바꾼 point "
              f"{names['a']}={demoted['a']:,} {names['b']}={demoted['b']:,}")

    common = [p for p in labels_a if p in labels_b]
    only_a, only_b = len(labels_a) - len(common), len(labels_b) - len(common)
    if only_a or only_b:
        warnings.append(warning("POINT_SET_MISMATCH",
                                f"point 집합이 다름: {names['a']} 에만 {only_a:,}, {names['b']} 에만 {only_b:,} → 공통 {len(common):,} 만 비교",
                                args.target))
    print(f"[compare] {names['a']}={len(labels_a):,} {names['b']}={len(labels_b):,} 공통={len(common):,}")

    stats = {"a": method_stats(labels_a, common), "b": method_stats(labels_b, common), "common": len(common),
             "only_a": only_a, "only_b": only_b}
    agr = agreement(labels_a, labels_b, common) if common else None
    rows_ab, map_ab = cluster_mapping(labels_a, labels_b, common)
    rows_ba, map_ba = cluster_mapping(labels_b, labels_a, common)
    for name, rows in (("mapping_a_to_b.jsonl", rows_ab), ("mapping_b_to_a.jsonl", rows_ba)):
        with (out_dir / name).open("w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    examples = pick_examples(rows_ab, rows_ba, args.examples, args.min_example_size) if common else {}
    examples_block = ""
    if examples:
        members_a: Dict[str, List[Any]] = defaultdict(list)
        members_b: Dict[str, List[Any]] = defaultdict(list)
        for p in common:
            if labels_a[p] is not None:
                members_a[labels_a[p]].append(p)
            if labels_b[p] is not None:
                members_b[labels_b[p]].append(p)
        # 표본은 여기서 한 번만 뽑는다 (조회 집합 == 표시 집합)
        example_groups = build_example_groups(examples, members_a, members_b, labels_a, labels_b,
                                              args.images_per_example, random.Random(args.seed))
        payloads: Dict[Any, Dict[str, Any]] = {}
        if not args.no_images:
            need = example_ids(example_groups)
            try:
                payloads = fetch_payloads(args.qdrant_url, args.api_key, args.collection, sorted(need, key=str))
                missing = [pid for pid in need if pid not in payloads]
                if missing:
                    warnings.append(warning("PAYLOAD_MISSING", f"payload 없는 point {len(missing):,} (예: {missing[0]})", args.target))
            except Exception as exc:  # 네트워크 실패는 경고로 남기고 통계 보고서는 만든다
                warnings.append(warning("PAYLOAD_FETCH_FAILED", f"Qdrant payload 조회 실패: {exc}", args.target))
        examples_block = examples_html(example_groups, names, payloads, Path(args.project_root).resolve(),
                                       args.inline_images, args.thumb_size, args.thumb_quality, html_path,
                                       show_images=not args.no_images)

    title = args.title or f"{names['a']} vs {names['b']} 클러스터 비교 ({args.target})"
    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["generated_at"] = generated
    html_text = render_html(title, names, stats, agr or {}, map_ab, map_ba, examples_block, inputs, reports, warnings,
                            args.thumb_size) if agr else (
        f"<!DOCTYPE html><html lang='ko'><head><meta charset='utf-8'><title>{esc(title)}</title></head>"
        f"<body><h1>{esc(title)}</h1><p>공통 point 가 없어 비교할 수 없습니다.</p></body></html>")
    html_path.write_text(html_text, encoding="utf-8", newline="\n")

    summary = common_summary(PRODUCER, html_path, inputs, warnings)
    summary.update(config=dict(a=str(a_path), b=str(b_path), labels=names, target=args.target,
                               examples=args.examples, images_per_example=args.images_per_example,
                               min_cluster_size=args.min_cluster_size, demoted_to_noise=demoted,
                               inline_images=bool(args.inline_images), no_images=bool(args.no_images),
                               config_path=(args.pipeline_settings.config_path if args.pipeline_settings else None),
                               config_sha256=(args.pipeline_settings.config_sha256 if args.pipeline_settings else None)),
                   metrics=dict(common_points=len(common), only_a=only_a, only_b=only_b, a=stats["a"], b=stats["b"],
                                agreement=agr, mapping_a_to_b=map_ab, mapping_b_to_a=map_ba),
                   examples={k: [dict(cluster_id=r["cluster_id"], size=r["size"], parts=r["parts"],
                                      noise_in_other=r["noise_in_other"], category=r["category"]) for r in v]
                             for k, v in examples.items()},
                   elapsed_sec=round(time.time() - started, 3))
    write_json(sidecar, summary)
    if agr:
        print(json.dumps({k: agr[k] for k in ("ari_noise_as_singletons", "nmi_noise_as_singletons", "ari_both_clustered",
                                              "nmi_both_clustered", "pair_f1", "pair_jaccard")}, ensure_ascii=False, indent=2))
    print(f"html    : {html_path}")
    print(f"sidecar : {sidecar}")
    result_markers(sidecar, html_path)
    return summary


if __name__ == "__main__":
    main()
