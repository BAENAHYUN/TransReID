#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""군집에 '노란색 상의' 같은 외형 라벨을 붙인다 — DB 에 이미 있는 SigLIP2 벡터만 사용.

label_leiden_clusters_siglip2.py(대표 crop 8장을 다시 인코딩, 프롬프트 원점수 argmax)는 PRW 에서
거의 모든 군집이 'yellow/purple t-shirt' 로 나왔다. 원인은 프롬프트 편향: SigLIP 은 프롬프트마다
점수 분포가 달라 특정 색 문장이 항상 높게 나온다. 여기서는
  1. Qdrant 의 siglip2 named vector 를 군집 소속 crop 전부에 대해 가져오고 (재인코딩 없음),
  2. 색상 프롬프트 여러 문장의 텍스트 임베딩을 평균해 색상별 텍스트 벡터를 만든 뒤,
  3. crop × 색상 점수를 **프롬프트별로 보정**(zscore: 전체 crop 에 대한 평균·표준편차로 정규화 /
     center: 평균만 뺌 / none)하고, 1·2위 차이(margin)가 작은 crop 은 기권시키고,
  4. 군집 안 다수결(확신 crop 의 과반)로 라벨을 정한다. 과반이 없으면 '불확실'.
PRW GT(--gt-matches)가 있으면 같은 인물의 crop 이 같은 라벨을 받는 비율(일관성)과 라벨 다양성을
보정 방식별로 함께 보고한다 — 한 색만 나오는 퇴화 라벨은 일관성은 높아도 다양성이 0 에 가깝다.

산출물: <output-dir>/cluster_labels.jsonl(.json) — export_cluster_folders.py --labels 에 바로 쓰는 형식
(cluster_id / cluster_name / label_confidence / cluster_description / scores), crop_labels.csv,
cluster_labels.html, cluster_labels_report.json(사이드카). 마지막에 RESULT_SUMMARY / RESULT_HTML 마커.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from report.build_leiden_gallery import load_assignments, split_groups
from clustering.export_cluster_folders import load_gt_matches
from report_common import (CONFIG_HELP, DEFAULT_CONFIG_PATH, applied_config, common_summary, esc, file_info,
                           is_noise, load_pipeline_settings, resolve, result_markers, table, warning, write_json)

PRODUCER = "label_clusters_from_vectors.py"

COLORS: List[Tuple[str, str]] = [
    ("black", "검은색"), ("white", "흰색"), ("gray", "회색"), ("red", "빨간색"), ("blue", "파란색"),
    ("green", "녹색"), ("yellow", "노란색"), ("orange", "주황색"), ("pink", "분홍색"), ("purple", "보라색"),
    ("brown", "갈색"), ("beige", "베이지색"),
]
# 하의 색 bank 는 뺐다: SigLIP2 전역 임베딩은 색이 '어디에' 있는지 거의 구분하지 못해 하의 프롬프트가
# 상의와 같은 색을 그대로 따라갔다 (첫 실행에서 모든 군집이 '검은색 상의 · 검은색 하의' 꼴). 보행자 crop 에서
# 가장 잘 보이는 색이 상의라 '상의' 로 이름 붙이되, 정확히는 'crop 의 주된 옷 색' 이다.
BANKS: Dict[str, Dict[str, Any]] = {
    "upper_color": dict(part_ko="상의", part_en="top", templates=[
        "a person wearing a {c} top", "a person wearing a {c} shirt",
        "a pedestrian wearing {c} upper body clothing", "a photo of a person in a {c} jacket"]),
}
# 물건: 종류(자동차·자전거·가방…)는 검출기 payload 의 label 로 알고 있으니 색만 SigLIP2 로 정한다.
# 이름 = '<색> <종류>' (예: 빨간색 자동차). part_ko 가 None 이면 군집의 다수 종류를 쓴다.
OBJECT_BANKS: Dict[str, Dict[str, Any]] = {
    "object_color": dict(part_ko=None, part_en=None, templates=[
        "a {c} object", "a photo of a {c} object", "a {c} colored thing", "an object that is {c}"]),
}
# COCO 계열 검출기 label → 한국어 (없으면 영어 그대로)
OBJECT_KO: Dict[str, str] = {
    "person": "사람", "bicycle": "자전거", "car": "자동차", "motorcycle": "오토바이", "airplane": "비행기", "bus": "버스",
    "train": "기차", "truck": "트럭", "boat": "배", "traffic light": "신호등", "fire hydrant": "소화전",
    "stop sign": "정지 표지판", "parking meter": "주차 미터기", "bench": "벤치", "bird": "새", "cat": "고양이", "dog": "개",
    "horse": "말", "sheep": "양", "cow": "소", "elephant": "코끼리", "bear": "곰", "zebra": "얼룩말", "giraffe": "기린",
    "backpack": "백팩", "umbrella": "우산", "handbag": "핸드백", "tie": "넥타이", "suitcase": "여행가방", "frisbee": "원반",
    "skis": "스키", "snowboard": "스노보드", "sports ball": "공", "kite": "연", "baseball bat": "야구방망이",
    "baseball glove": "야구 글러브", "skateboard": "스케이트보드", "surfboard": "서핑보드", "tennis racket": "테니스 라켓",
    "bottle": "병", "wine glass": "와인잔", "cup": "컵", "fork": "포크", "knife": "칼", "spoon": "숟가락", "bowl": "그릇",
    "banana": "바나나", "apple": "사과", "sandwich": "샌드위치", "orange": "오렌지", "broccoli": "브로콜리", "carrot": "당근",
    "hot dog": "핫도그", "pizza": "피자", "donut": "도넛", "cake": "케이크", "chair": "의자", "couch": "소파",
    "potted plant": "화분", "bed": "침대", "dining table": "식탁", "toilet": "변기", "tv": "TV", "laptop": "노트북",
    "mouse": "마우스", "remote": "리모컨", "keyboard": "키보드", "cell phone": "휴대폰", "microwave": "전자레인지",
    "oven": "오븐", "toaster": "토스터", "sink": "싱크대", "refrigerator": "냉장고", "book": "책", "clock": "시계",
    "vase": "꽃병", "scissors": "가위", "teddy bear": "곰인형", "hair drier": "헤어드라이어", "toothbrush": "칫솔",
}
CALIBRATIONS = ("zscore", "center", "none")


def banks_for(target: Optional[str]) -> Dict[str, Dict[str, Any]]:
    return OBJECT_BANKS if target == "object" else BANKS


def object_name(label: Any, lang: str = "ko") -> str:
    text = str(label or "").strip()
    if not text:
        return "물체" if lang == "ko" else "object"
    return OBJECT_KO.get(text.lower(), text) if lang == "ko" else text


def majority_class(point_ids: Sequence[str], classes: Dict[str, Any]) -> Tuple[Optional[str], float]:
    """군집 구성원의 검출 label 중 가장 많은 것과 그 비율. label 이 하나도 없으면 (None, 0)."""
    counts = Counter(str(classes[p]) for p in point_ids if classes.get(p))
    if not counts:
        return None, 0.0
    label, n = counts.most_common(1)[0]
    return label, n / max(1, sum(counts.values()))


# ----------------------------------------------------------------------------------------------------
# 순수 함수
# ----------------------------------------------------------------------------------------------------
def l2n(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def prompt_texts(bank: Dict[str, Any]) -> List[Tuple[str, str]]:
    return [(color, template.format(c=color)) for color, _ in COLORS for template in bank["templates"]]


def color_text_matrix(text_vectors: np.ndarray, bank: Dict[str, Any]) -> np.ndarray:
    """(색상 × 템플릿) 순서의 텍스트 임베딩을 색상별로 평균해 (K, D) 로."""
    n_templates = len(bank["templates"])
    if text_vectors.shape[0] != len(COLORS) * n_templates:
        raise ValueError("텍스트 임베딩 수가 색상 × 템플릿 수와 다르다")
    grouped = text_vectors.reshape(len(COLORS), n_templates, -1).mean(axis=1)
    return l2n(grouped)


def calibrate(scores: np.ndarray, method: str) -> np.ndarray:
    """crop × 색상 점수의 프롬프트(열) 편향을 없앤다."""
    if method == "none":
        return scores
    mean = scores.mean(axis=0, keepdims=True)
    if method == "center":
        return scores - mean
    if method == "zscore":
        std = scores.std(axis=0, keepdims=True)
        return (scores - mean) / np.clip(std, 1e-8, None)
    raise ValueError(f"unknown calibration: {method}")


def margin_threshold(calibrated: np.ndarray, margin: float) -> float:
    """margin 은 '보정 점수 전체 표준편차' 단위. zscore 면 ≈ 그대로, center/none 이면 코사인 척도에 맞춰 줄어든다."""
    return float(margin * calibrated.std()) if calibrated.size else 0.0


def crop_decisions(calibrated: np.ndarray, margin: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """crop 별 1위 색상 index, 1·2위 차이, 확신 여부(margin 이상)."""
    order = np.argsort(-calibrated, axis=1)
    top = order[:, 0]
    top_score = calibrated[np.arange(len(top)), top]
    second = calibrated[np.arange(len(top)), order[:, 1]] if calibrated.shape[1] > 1 else top_score
    margins = top_score - second
    return top, margins, margins >= margin_threshold(calibrated, margin)


def cluster_decision(indices: Sequence[int], calibrated: np.ndarray, crop_top: np.ndarray, threshold: float,
                     min_share: float) -> Dict[str, Any]:
    """군집 소속 crop 의 보정 점수 평균으로 색을 정한다. 1·2위 차이가 threshold 이상이고, crop 단위 1위가
    그 색과 같은 비율(일치율)이 min_share 이상일 때만 채택한다. 작은 군집(2장)도 판단할 수 있다."""
    sub = calibrated[list(indices)]
    mean = sub.mean(axis=0)
    order = np.argsort(-mean)
    best, second = int(order[0]), int(order[1])
    margin = float(mean[best] - mean[second])
    agreement = float(np.mean(crop_top[list(indices)] == best))
    # labeled: 평균 margin 과 일치율 모두 통과 / tentative: 일치율만 통과 (crop 과반이 같은 색을 고르지만
    # 점수 차가 작다 — 이름에 '(추정)') / uncertain: crop 들이 서로 다른 색을 고른다 → 이름 없음
    if margin >= threshold and agreement >= min_share:
        status = "labeled"
    elif agreement >= min_share:
        status = "tentative"
    else:
        status = "uncertain"
    distribution = Counter(int(t) for t in crop_top[list(indices)]).most_common(3)
    return dict(n=len(indices), label=None if status == "uncertain" else best, status=status,
                candidate=COLORS[best][0], margin=margin, agreement=agreement, share=agreement,
                distribution={COLORS[k][0]: v for k, v in distribution})


def color_name(index: Optional[int], lang: str) -> Optional[str]:
    if index is None:
        return None
    return COLORS[index][1 if lang == "ko" else 0]


def uncertain_word(lang: str) -> str:
    return "불확실" if lang == "ko" else "uncertain"


def compose_name(votes: Dict[str, Dict[str, Any]], lang: str, banks: Optional[Dict[str, Dict[str, Any]]] = None,
                 part: Optional[str] = None) -> Tuple[str, float, str, str]:
    """bank 별 결정 → (이름, 신뢰도, 설명, status). 채택된 색이 없으면 이름은 빈 문자열(폴더 이름에 안 붙음).
    tentative 는 이름 뒤에 '(추정)'. 신뢰도 = 채택 bank 의 평균 일치율.
    물건(banks=OBJECT_BANKS)은 part 에 군집의 종류 이름을 주고, 색을 못 정해도 종류만으로 이름을 붙인다
    (status 'class_only' — 예: '자전거')."""
    banks = banks or BANKS
    parts, shares, desc = [], [], []
    status = "uncertain"
    for bank_name, bank in banks.items():
        v = votes.get(bank_name) or {}
        label = color_name(v.get("label"), lang)
        word = (bank["part_ko"] if lang == "ko" else bank["part_en"]) or part or ("물체" if lang == "ko" else "object")
        desc.append(f"{bank_name}={color_name(v.get('label'), 'en') or 'uncertain'} [{v.get('status', 'uncertain')}] "
                    f"(candidate={v.get('candidate')}, margin={v.get('margin', 0.0):.2f}, "
                    f"agreement={v.get('agreement', 0.0):.2f}, n={v.get('n', 0)})")
        if label is None:
            continue
        tentative = v.get("status") == "tentative"
        parts.append(f"{label} {word}" + (("(추정)" if lang == "ko" else " (tentative)") if tentative else ""))
        shares.append(v["agreement"])
        status = "tentative" if tentative or status == "tentative" else "labeled"
    confidence = float(np.mean(shares)) if shares else 0.0
    if not parts and part and any(b["part_ko"] is None for b in banks.values()):
        # 물건인데 색을 못 정했다 — 종류(검출 label)만으로 이름을 붙인다
        return part, 0.0, ", ".join(desc), "class_only"
    return " · ".join(parts), confidence, ", ".join(desc), status


def gt_consistency(gt: Dict[str, Any], ids: Sequence[str], top: np.ndarray, confident: np.ndarray,
                   min_crops: int = 5) -> Dict[str, Any]:
    """같은 GT 인물의 확신 crop 이 같은 라벨을 받는 비율(평균 과반율)과 라벨 다양성(정규화 엔트로피)."""
    by_pid: Dict[Any, List[int]] = defaultdict(list)
    for i, pid in enumerate(ids):
        p = gt.get(str(pid))
        if p is not None and confident[i]:
            by_pid[p].append(int(top[i]))
    agreements = []
    for labels in by_pid.values():
        if len(labels) >= min_crops:
            agreements.append(Counter(labels).most_common(1)[0][1] / len(labels))
    used = Counter(int(t) for t, c in zip(top, confident) if c)
    total = sum(used.values())
    entropy = -sum((v / total) * math.log(v / total) for v in used.values()) if total else 0.0
    diversity = entropy / math.log(len(COLORS)) if total else 0.0
    return dict(identities=len(agreements), mean_agreement=float(np.mean(agreements)) if agreements else None,
                label_diversity=float(diversity), confident_ratio=float(confident.mean()) if len(confident) else 0.0,
                label_counts={COLORS[k][0]: v for k, v in used.most_common()})


# ----------------------------------------------------------------------------------------------------
# 데이터 접근
# ----------------------------------------------------------------------------------------------------
def fetch_vectors(qdrant_url: str, api_key: Optional[str], collection: str, vector_name: str,
                  ids: Sequence[str], batch_size: int = 512) -> Tuple[List[str], np.ndarray]:
    """(찾은 point id — ids 순서 유지, 그 벡터 행렬). DB 에서 지워졌거나 벡터가 없는 point 는 빠진다
    (assignments 가 DB 보다 오래되면 생긴다 — 영상 DB 정리 뒤의 옛 군집 결과 등). 하나도 없으면 예외."""
    from qdrant_client import QdrantClient
    client = QdrantClient(url=qdrant_url, api_key=api_key, prefer_grpc=True, timeout=120)
    rows: Dict[str, np.ndarray] = {}
    started = time.time()
    for start in range(0, len(ids), batch_size):
        batch = list(ids[start:start + batch_size])
        records = client.retrieve(collection, ids=batch, with_payload=False, with_vectors=[vector_name])
        for record in records:
            vec = record.vector.get(vector_name) if isinstance(record.vector, dict) else None
            if vec is not None:
                rows[str(record.id)] = np.asarray(vec, dtype=np.float32)
        if (start // batch_size) % 20 == 0:
            print(f"  vectors {min(start + batch_size, len(ids)):,}/{len(ids):,} ({time.time() - started:.0f}s)")
    found = [str(pid) for pid in ids if str(pid) in rows]
    if not found:
        raise RuntimeError(f"{vector_name} 벡터가 있는 point 가 하나도 없다 ({len(ids):,}개 요청, collection={collection})")
    dim = len(rows[found[0]])
    return found, np.stack([rows[pid] for pid in found]).reshape(len(found), dim)


def fetch_payload_field(qdrant_url: str, api_key: Optional[str], collection: str, field: str, ids: Sequence[str],
                        batch_size: int = 1024) -> Dict[str, Any]:
    """point id → payload[field] (물건의 검출 종류 label 등). 벡터는 읽지 않는다."""
    from qdrant_client import QdrantClient
    client = QdrantClient(url=qdrant_url, api_key=api_key, prefer_grpc=True, timeout=120)
    out: Dict[str, Any] = {}
    for start in range(0, len(ids), batch_size):
        for record in client.retrieve(collection, ids=list(ids[start:start + batch_size]), with_payload=[field], with_vectors=False):
            value = (record.payload or {}).get(field)
            if value not in (None, ""):
                out[str(record.id)] = value
    return out


def drop_missing(ordered: List[Tuple[str, List[Dict[str, Any]]]], used_rows: List[Dict[str, Any]], ids: Sequence[str],
                 found: Sequence[str]) -> Tuple[List[Tuple[str, List[Dict[str, Any]]]], List[Dict[str, Any]], List[str], int]:
    """벡터를 못 찾은 point 를 군집·행·id 목록에서 뺀다. 구성원이 모두 빠진 군집은 없앤다. (ordered, used_rows, ids, 뺀 수)."""
    keep = set(map(str, found))
    n_missing = len(ids) - len(keep)
    if not n_missing:
        return ordered, used_rows, list(ids), 0
    ordered = [(cid, kept) for cid, kept in ((cid, [m for m in members if str(m["point_id"]) in keep]) for cid, members in ordered) if kept]
    used_rows = [row for row in used_rows if str(row["point_id"]) in keep]
    return ordered, used_rows, [pid for pid in ids if pid in keep], n_missing


def load_or_fetch_vectors(cache: Path, ids: Sequence[str], fetch) -> Tuple[List[str], np.ndarray, bool]:
    """(찾은 id, 행렬, 캐시 사용 여부). 캐시 키는 요청한 ids 전체; 찾은 id 는 found 로 따로 저장 (옛 캐시는 전부 찾은 것으로 본다)."""
    if cache.is_file():
        data = np.load(cache, allow_pickle=False)
        if list(data["ids"]) == list(ids):
            found = [str(x) for x in data["found"]] if "found" in data.files else [str(x) for x in data["ids"]]
            return found, data["matrix"], True
    found, matrix = fetch()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=np.asarray(ids), found=np.asarray(found), matrix=matrix.astype(np.float32))
    return list(found), matrix, False


def build_embedder(settings, model_id: Optional[str], device: str):
    from embedders.siglip2_embedder import SigLIP2Embedder
    params: Dict[str, Any] = {}
    retrievers = getattr(settings, "retrievers", None) or {}
    spec = retrievers.get("siglip2") if isinstance(retrievers, dict) else None
    if isinstance(spec, dict):
        params = dict(spec.get("params") or {})
    if model_id:
        params["model_id"] = model_id
    kwargs = {k: params[k] for k in ("model_id", "max_num_patches", "cache_dir", "local_files_only") if k in params}
    kwargs.setdefault("model_id", "google/siglip2-base-patch16-naflex")
    kwargs.setdefault("max_num_patches", 256)
    return SigLIP2Embedder(device=device, batch_size=64, fp16=False, **kwargs)


# ----------------------------------------------------------------------------------------------------
# 출력
# ----------------------------------------------------------------------------------------------------
def write_html(path: Path, title: str, summary: Dict[str, Any], records: List[Dict[str, Any]],
               comparison: Dict[str, Any], warnings: List[Dict[str, Any]]) -> None:
    name_counts = Counter(r["display_name"] for r in records)
    dist_rows = "".join(f"<tr><td>{esc(name)}</td><td>{count:,}</td></tr>" for name, count in name_counts.most_common())
    comp_rows = ""
    for method, metrics in comparison.items():
        if not metrics:
            continue
        agreement = metrics.get("mean_agreement")
        comp_rows += (f"<tr><td>{esc(method)}</td><td>{'' if agreement is None else f'{agreement:.3f}'}</td>"
                      f"<td>{metrics['label_diversity']:.3f}</td><td>{metrics['confident_ratio']:.3f}</td>"
                      f"<td>{esc(', '.join(f'{k} {v:,}' for k, v in list(metrics['label_counts'].items())[:6]))}</td></tr>")
    rows = ""
    for r in records:
        up = next(iter(r["scores"].values()), {})      # 첫 bank (사람=상의 색, 물건=물건 색)
        rows += (f"<tr><td>{r['rank']}</td><td><code>{esc(r['cluster_id'])}</code></td><td>{r['cluster_size']:,}</td>"
                 f"<td><b>{esc(r['display_name'])}</b></td><td>{r['label_confidence']:.2f}</td>"
                 f"<td>{esc(up.get('candidate'))}</td><td>{up.get('margin', 0.0):.2f}</td><td>{up.get('agreement', 0.0):.2f}</td>"
                 f"<td>{esc(up.get('distribution'))}</td></tr>")
    warn_html = ""
    if warnings:
        items = "".join(f"<li><b>{esc(w.get('code'))}</b> {esc(w.get('message'))}</li>" for w in warnings)
        warn_html = f'<div class="warn"><b>경고 {len(warnings)}</b><ul>{items}</ul></div>'
    html_text = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>{esc(title)}</title>
<style>
body{{font-family:'Malgun Gothic',Segoe UI,sans-serif;margin:20px;color:#222;background:#fafafa}}
h1{{font-size:20px;margin:0 0 8px}} h2{{font-size:16px;margin:22px 0 8px}} .muted{{color:#777;font-size:12px}}
table{{border-collapse:collapse;background:#fff}} th,td{{border:1px solid #ddd;padding:4px 8px;font-size:13px;vertical-align:top}}
th{{background:#f0f0f0;text-align:left}} code{{background:#eee;padding:0 3px}}
.warn{{background:#fff4e0;border:1px solid #f0c060;padding:8px 12px;margin:12px 0}} .row{{display:flex;gap:24px;flex-wrap:wrap}}
</style></head><body>
<h1>{esc(title)}</h1>
<p class="muted">생성 {esc(summary.get('generated_at'))} · {esc(PRODUCER)}</p>
{table({k: summary[k] for k in ('collection', 'assignments_path', 'vector', 'model_id', 'calibration', 'margin',
                                  'min_share', 'clusters', 'crops', 'gt_source') if k in summary})}
{warn_html}
<div class="row"><div><h2>라벨 분포 (군집 수)</h2><table><tr><th>라벨</th><th>군집</th></tr>{dist_rows}</table></div>
<div><h2>보정 방식 비교 (crop 단위)</h2>
<p class="muted">일관성 = 같은 GT 인물의 확신 crop 이 같은 상의 색을 받는 평균 비율(5장 이상 인물). 다양성 = 라벨 분포의 정규화 엔트로피(0 = 한 색만, 1 = 균등). 확신 비율 = margin 을 넘긴 crop 비율.</p>
<table><tr><th>보정</th><th>일관성</th><th>다양성</th><th>확신 비율</th><th>상위 라벨</th></tr>{comp_rows}</table></div></div>
<h2>군집별 라벨</h2>
<p class="muted">후보 색 = 군집 crop 들의 보정 점수 평균이 가장 높은 색. margin = 1·2위 평균 차(보정 단위). 일치율 = crop 단위 1위가 후보 색과 같은 비율. 둘 다 기준 이상이면 라벨, 일치율만 넘으면 '(추정)', 일치율도 못 넘으면 '불확실'(폴더 이름에 붙지 않음).</p>
<table><tr><th>순위</th><th>cluster_id</th><th>크기</th><th>라벨</th><th>신뢰도</th><th>후보 색</th><th>margin</th><th>일치율</th><th>crop 1위 분포 (상위 3)</th></tr>
{rows}</table>
</body></html>
"""
    path.write_text(html_text, encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DB 의 SigLIP2 벡터로 군집에 색상 외형 라벨 붙이기 (프롬프트 편향 보정)")
    p.add_argument("--assignments", default="outputs/clustering/leiden_image_prw/person/person_leiden_assignments.jsonl")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=CONFIG_HELP)
    p.add_argument("--target", choices=["person", "object"], default=None)
    p.add_argument("--collection", default=None, help=CONFIG_HELP)
    p.add_argument("--qdrant-url", default=None, help=CONFIG_HELP)
    p.add_argument("--api-key", default=None)
    p.add_argument("--vector", default="siglip2", help="텍스트와 같은 공간의 named vector")
    p.add_argument("--model-id", default=None, help="비우면 pipeline.yaml retrievers.siglip2.params.model_id")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default=None, help="기본: assignments 옆 labels_vec/")
    p.add_argument("--vector-cache", default=None, help="벡터 npz 캐시 (기본: <output-dir>/cache/<vector>.npz)")
    p.add_argument("--gt-matches", default=None, help="eval/prw_cluster_gt_eval.py 의 gt_matches.jsonl — 일관성 지표용")
    p.add_argument("--calibration", choices=CALIBRATIONS, default="zscore")
    p.add_argument("--margin", type=float, default=0.3,
                   help="1·2위 점수 차 하한 (보정 점수 표준편차 단위). crop 확신 판정과 군집 평균 판정에 같이 쓴다")
    p.add_argument("--min-share", type=float, default=0.5, help="군집 라벨 채택에 필요한 crop 일치율")
    p.add_argument("--lang", choices=["ko", "en"], default="ko")
    p.add_argument("--include-noise", action="store_true", help="noise point 도 벡터를 가져와 crop_labels.csv 에 남긴다")
    p.add_argument("--title", default=None)
    return p


def parse_args(argv=None) -> argparse.Namespace:
    p = build_parser()
    args = p.parse_args(argv)
    if not 0.0 < args.min_share <= 1.0:
        p.error("--min-share must be in (0, 1]")
    if args.margin < 0:
        p.error("--margin must be >= 0")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    target = args.target or next((v for v in ("person", "object") if Path(args.assignments).name.startswith(v + "_")), None)
    if args.collection in (None, "") and target is None:
        build_parser().error("--target 또는 --collection 지정")
    try:
        settings = load_pipeline_settings(args.config, require=not (
            args.collection not in (None, "") and args.qdrant_url not in (None, "")))
        args.collection = resolve(args.collection, settings.collection_for(target) if settings and target else None, "collection")
        args.qdrant_url = resolve(args.qdrant_url, settings.qdrant_url if settings else None, "qdrant_url")
    except (OSError, ValueError) as exc:
        build_parser().error(str(exc))

    assignments_path = Path(args.assignments).resolve()
    out_dir = (Path(args.output_dir) if args.output_dir else assignments_path.parent / "labels_vec").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.vector_cache).resolve() if args.vector_cache else out_dir / "cache" / f"{args.vector}.npz"
    html_path = out_dir / "cluster_labels.html"
    report_path = out_dir / "cluster_labels_report.json"
    warnings: List[Dict[str, Any]] = []
    inputs = [file_info("assignments", assignments_path)]
    if settings:
        inputs.insert(0, file_info("config", settings.config_path))
    else:
        warnings.append(warning("CONFIG_UNAVAILABLE", f"config 없음: {args.config}; 명시 CLI 사용"))

    parse_errors = dict(count=0, first_line=None)
    rows = load_assignments(assignments_path, parse_errors)
    if parse_errors["count"]:
        warnings.append(warning("PARSE_ERRORS", f"깨진 assignments 줄: {parse_errors}", target))
    groups, noise = split_groups(rows)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    used_rows = [row for _, members in ordered for row in members] + (noise if args.include_noise else [])
    ids = list(dict.fromkeys(str(row["point_id"]) for row in used_rows))
    index_of = {pid: i for i, pid in enumerate(ids)}
    if not ids:
        build_parser().error(f"assignments 에 군집 point 가 없다: {assignments_path}")

    gt: Optional[Dict[str, Any]] = None
    if args.gt_matches:
        gt_path = Path(args.gt_matches).resolve()
        if gt_path.is_file():
            gt = load_gt_matches(gt_path)
            inputs.append(file_info("gt_matches", gt_path))
        else:
            warnings.append(warning("GT_MISSING", f"GT 매칭 파일 없음 — 일관성 지표 생략: {gt_path}", target))

    print("=" * 88)
    print("CLUSTER COLOR LABELS FROM DB VECTORS")
    print("=" * 88)
    print(f"assignments : {assignments_path}")
    print(f"clusters    : {len(ordered):,}   crops: {len(ids):,}   noise included: {args.include_noise}")
    print(f"collection  : {args.collection}   vector: {args.vector}   calibration: {args.calibration}")

    found, matrix, from_cache = load_or_fetch_vectors(
        cache, ids, lambda: fetch_vectors(args.qdrant_url, args.api_key, args.collection, args.vector, ids))
    print(f"vectors     : {matrix.shape} ({'cache ' + str(cache) if from_cache else 'fetched from Qdrant'})")
    ordered, used_rows, ids, n_missing = drop_missing(ordered, used_rows, ids, found)
    if n_missing:
        if n_missing * 2 > n_missing + len(ids):
            build_parser().error(f"DB 에 없는 point 가 절반을 넘는다 ({n_missing:,}/{n_missing + len(ids):,}) — "
                                 f"assignments 와 collection({args.collection}) 이 맞는지 확인")
        msg = f"DB 에 없는(삭제된) point {n_missing:,}개는 건너뜀 — assignments 가 DB 보다 오래됨"
        print(f"WARNING     : {msg}")
        warnings.append(warning("MISSING_POINTS", msg, target))
    index_of = {pid: i for i, pid in enumerate(ids)}
    matrix = l2n(matrix.astype(np.float32))

    embedder = build_embedder(settings, args.model_id, args.device)
    model_id = embedder.model_id
    if matrix.shape[1] != embedder.DIM:
        build_parser().error(f"벡터 차원 {matrix.shape[1]} != 텍스트 모델 차원 {embedder.DIM} ({model_id}) — 같은 모델이어야 한다")
    banks = banks_for(target)
    primary = next(iter(banks))
    # 물건은 종류(검출 label)를 payload 에서 읽어 이름에 쓴다 (예: 빨간색 자동차)
    classes: Dict[str, Any] = {}
    if target == "object":
        classes = fetch_payload_field(args.qdrant_url, args.api_key, args.collection, "label", ids)
        print(f"classes     : {len(classes):,}/{len(ids):,} points have a detector label")
    text_matrices: Dict[str, np.ndarray] = {}
    for bank_name, bank in banks.items():
        texts = [text for _, text in prompt_texts(bank)]
        text_matrices[bank_name] = color_text_matrix(embedder.embed_text(texts), bank)
    raw_scores = {bank_name: matrix @ tm.T for bank_name, tm in text_matrices.items()}

    # 보정 방식 비교 (첫 bank 기준: 사람=상의, 물건=물건 색) — GT 가 있으면 일관성, 없으면 다양성·확신 비율만
    comparison: Dict[str, Any] = {}
    for method in CALIBRATIONS:
        top, margins, confident = crop_decisions(calibrate(raw_scores[primary], method), args.margin)
        comparison[method] = gt_consistency(gt or {}, ids, top, confident)
        if not gt:
            comparison[method]["mean_agreement"] = None
    calibrated = {bank_name: calibrate(scores, args.calibration) for bank_name, scores in raw_scores.items()}
    decisions = {bank_name: crop_decisions(matrix_c, args.margin) for bank_name, matrix_c in calibrated.items()}
    thresholds = {bank_name: margin_threshold(matrix_c, args.margin) for bank_name, matrix_c in calibrated.items()}

    records: List[Dict[str, Any]] = []
    for rank, (cid, members) in enumerate(ordered, 1):
        indices = [index_of[str(m["point_id"])] for m in members]
        votes = {bank_name: cluster_decision(indices, calibrated[bank_name], decisions[bank_name][0],
                                             thresholds[bank_name], args.min_share)
                 for bank_name in banks}
        extra: Dict[str, Any] = {}
        part = None
        if target == "object":
            cls, share = majority_class([str(m["point_id"]) for m in members], classes)
            part = object_name(cls, args.lang)
            extra = dict(object_class=cls, object_class_share=round(share, 3))
        name, confidence, description, status = compose_name(votes, args.lang, banks, part=part)
        if extra.get("object_class"):
            description += f", class={extra['object_class']} ({extra['object_class_share']:.0%})"
        records.append(dict(cluster_id=cid, rank=rank, cluster_size=len(members), cluster_name=name,
                            display_name=name or uncertain_word(args.lang), status=status,
                            label_confidence=confidence, cluster_description=description, scores=votes,
                            method=dict(calibration=args.calibration, margin=args.margin, vector=args.vector,
                                        model_id=model_id), **extra))

    (out_dir / "cluster_labels.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8", newline="\n")
    write_json(out_dir / "cluster_labels.json", records)
    with (out_dir / "crop_labels.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        prefix = primary.replace("_color", "")        # upper (사람, 예전과 같은 열 이름) / object
        writer.writerow(["point_id", "cluster_id", f"{prefix}_color", f"{prefix}_margin", f"{prefix}_confident", "gt_pid"])
        cluster_of = {str(row["point_id"]): row.get("cluster_id") for row in used_rows}
        up_top, up_margin, up_conf = decisions[primary]
        for i, pid in enumerate(ids):
            writer.writerow([pid, cluster_of.get(pid) or "", COLORS[int(up_top[i])][0], f"{up_margin[i]:.3f}",
                             int(up_conf[i]), "" if gt is None or gt.get(pid) is None else gt[pid]])

    name_counts = Counter(r["display_name"] for r in records)
    status_counts = Counter(r["status"] for r in records)
    uncertain = status_counts.get("uncertain", 0)
    if records and uncertain / len(records) > 0.5:
        warnings.append(warning("MOSTLY_UNCERTAIN", f"색을 못 정한 군집 {uncertain:,}/{len(records):,} — --margin 을 낮추거나 --min-share 조정",
                                target))
    if len(name_counts) <= 2 and len(records) > 10:
        warnings.append(warning("DEGENERATE_LABELS", f"라벨 종류가 {len(name_counts)}개뿐 — 프롬프트 편향 의심 (보정 방식 비교 참고)", target))

    title = args.title or f"군집 색상 라벨 · {target or args.collection} · {args.calibration}"
    summary = common_summary(PRODUCER, html_path, inputs, warnings)
    summary.update(dict(
        target=target, collection=args.collection, assignments_path=str(assignments_path), vector=args.vector,
        model_id=model_id, calibration=args.calibration, margin=args.margin, min_share=args.min_share,
        lang=args.lang, clusters=len(records), crops=len(ids), uncertain_clusters=uncertain,
        status_counts=dict(status_counts),
        gt_source=str(Path(args.gt_matches).resolve()) if gt is not None else None,
        label_counts=dict(name_counts.most_common()), comparison=comparison, vector_cache=str(cache),
        outputs=dict(jsonl=str(out_dir / "cluster_labels.jsonl"), json=str(out_dir / "cluster_labels.json"),
                     crops=str(out_dir / "crop_labels.csv")),
        config=applied_config(settings, collection=args.collection, qdrant_url=args.qdrant_url, target=target),
    ))
    write_html(html_path, title, summary, records, comparison, warnings)
    write_json(report_path, summary)

    print("-" * 88)
    print(f"calibration comparison ({primary}):")
    for method, metrics in comparison.items():
        agreement = metrics.get("mean_agreement")
        print(f"  {method:7s} consistency={'n/a' if agreement is None else f'{agreement:.3f}'}  "
              f"diversity={metrics['label_diversity']:.3f}  confident={metrics['confident_ratio']:.3f}  "
              f"top={list(metrics['label_counts'].items())[:4]}")
    print(f"labels      : {len(records):,} clusters, {len(name_counts)} distinct names, "
          f"status {dict(status_counts)}")
    for name, count in name_counts.most_common(8):
        print(f"  {count:5,}  {name}")
    print(f"warnings    : {len(warnings)}")
    result_markers(report_path, html_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
