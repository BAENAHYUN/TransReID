#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Qwen3-VL 로 군집마다 문장형 외형 이름을 붙인다 — 예: "노란 반팔에 검은 바지".

군집 대표 crop N장(기본 6)을 가로로 이어 붙인 몽타주 한 장을 Qwen3-VL-Instruct 에 보여 주고, 옷차림을
JSON 으로 받는다 (상의 / 하의 / 소지품 / crop 들이 같은 옷차림인지 / 폴더 이름용 요약). 신원·성별·나이 같은
민감 속성은 묻지 않고 옷·소지품·색만 다룬다. 모델이 crop 들이 서로 다르다고 하면(consistent=false) 이름을
폴더에 붙이지 않는다 (--keep-mixed 로 바꿀 수 있다).

산출물: <output-dir>/cluster_labels.jsonl(.json) — export_cluster_folders.py --labels 형식(cluster_id / cluster_name /
label_confidence / cluster_description) + qwen 원문, montage/<rank>_<id8>.jpg (근거 이미지), cluster_labels.html,
cluster_labels_report.json(사이드카). RESULT_SUMMARY / RESULT_HTML 마커. --resume 로 끊긴 실행을 이어 간다.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from report.build_leiden_gallery import CROP_KEYS, QdrantHTTP, first_payload, load_assignments, resolve_path, split_groups
from clustering.export_cluster_folders import short_id
from verifiers.qwen_stage import QwenVL
from report_common import (CONFIG_HELP, DEFAULT_CONFIG_PATH, applied_config, common_summary, esc, file_info,
                           load_pipeline_settings, resolve, result_markers, table, warning, write_json)

PRODUCER = "label_clusters_qwen.py"
# 4B 가 2B 보다 옷 종류·소지품을 조금 더 구체적으로 적지만(샘플 20군집 비교), 로드에 RAM 이 9GB 넘게 필요해
# RAM 16GB 기계에서 다른 앱이 떠 있으면 세그폴트(exit 139)로 죽는다. 기본은 2B, 여유가 있으면 --model-id 로 4B.
DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


class BatchedQwenVL(QwenVL):
    """qwen_stage.QwenVL 에 배치 생성을 더한다. 한 장씩 생성하면 토큰당 파이썬 오버헤드에 묶여 2B 도 8 tok/s 인데,
    16장을 한 번에 넣으면 장당 시간이 1/8 이하로 준다 (decoder-only 라 왼쪽 패딩)."""

    def _load_model(self, transformers, dtype):
        """가능하면 GPU 로 바로 올린다 (device_map). CPU 에 먼저 다 올리는 기본 경로는 RAM 16GB 기계에서 4B(9GB) 로드 때
        RAM 을 꽉 채워 세그폴트가 났다. 안 되면 부모 방식으로 돌아간다."""
        if str(self.device).startswith("cuda"):
            for name in ("AutoModelForImageTextToText", "Qwen3VLForConditionalGeneration"):
                cls = getattr(transformers, name, None)
                if cls is None:
                    continue
                try:
                    return cls.from_pretrained(self.model_id, dtype=dtype, device_map=self.device, low_cpu_mem_usage=True)
                except Exception:  # noqa: BLE001 — 버전별 인자 차이
                    continue
        return super()._load_model(transformers, dtype)

    def generate_batch(self, prompts: Sequence[str], image_paths: Sequence[str]) -> List[str]:
        import torch
        from PIL import Image
        if self._model is None or self._processor is None:
            raise RuntimeError("Qwen 모델/프로세서가 로드되지 않았습니다.")
        self._processor.tokenizer.padding_side = "left"
        texts, images = [], []
        for prompt, path in zip(prompts, image_paths):
            with Image.open(path) as im:
                images.append(im.convert("RGB").copy())
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            texts.append(self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = self._processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in dict(inputs).items()}
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        length = inputs["input_ids"].shape[1]
        return [t.strip() for t in self._processor.batch_decode(out[:, length:], skip_special_tokens=True)]
STATUS_LABEL = {"labeled": "라벨", "mixed": "혼합(불일치)", "parse_error": "응답 해석 실패", "no_images": "이미지 없음"}
MAX_NAME_CHARS = 30


# ----------------------------------------------------------------------------------------------------
# 순수 함수
# ----------------------------------------------------------------------------------------------------
def choose_representatives(members: Sequence[Dict[str, Any]], count: int, seed: int) -> List[Dict[str, Any]]:
    """앞·중간·뒤 + 무작위. 같은 seed 면 같은 선택."""
    if len(members) <= count:
        return list(members)
    rng = random.Random(seed)
    picks = [members[0], members[len(members) // 2], members[-1]][:count]
    chosen = {str(m["point_id"]) for m in picks}
    rest = [m for m in members if str(m["point_id"]) not in chosen]
    need = count - len(picks)
    if need > 0:
        picks.extend(rng.sample(rest, need) if len(rest) > need else rest)
    return picks[:count]


def build_montage(images, height: int = 192, gap: int = 6):
    """crop 들을 같은 높이로 맞춰 가로로 이어 붙인다 (흰 배경). 가로가 세로의 두 배를 넘으면 잘라 맞춘다."""
    from PIL import Image
    resized = []
    for im in images:
        im = im.convert("RGB")
        w, h = im.size
        new_w = int(round(w * height / max(h, 1)))
        new_w = max(24, min(new_w, height * 2))
        resized.append(im.resize((new_w, height), Image.BICUBIC))
    width = sum(r.size[0] for r in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for r in resized:
        canvas.paste(r, (x, 0))
        x += r.size[0] + gap
    return canvas


def build_prompt(n: int, lang: str = "ko") -> str:
    if lang == "en":
        return (
            f"This image shows {n} CCTV crops of the same person side by side (left to right). "
            "Describe only the visible clothing and carried items. Do not guess gender, age, ethnicity or identity. "
            "Reply with exactly one JSON object and nothing else: "
            '{"upper": "color and type of upper clothing", "lower": "color and type of lower clothing", '
            '"items": "bags, hats, umbrellas etc. or \\"none\\"", '
            '"consistent": true if all crops show the same outfit else false, '
            '"name": "folder-name summary in at most 6 words, e.g. yellow t-shirt black pants"}'
        )
    return (
        f"이 이미지는 CCTV 에서 잘라낸 같은 사람의 crop {n}장을 왼쪽부터 가로로 이어 붙인 것입니다. "
        "겉으로 보이는 옷차림과 소지품만 설명하세요. 성별·나이·인종·신원은 추정하지 마세요. "
        "다른 말 없이 JSON 객체 하나만 출력하세요: "
        '{"upper": "상의 색과 종류 (예: 노란색 반팔 티셔츠)", "lower": "하의 색과 종류 (예: 검은색 긴바지)", '
        '"items": "가방·모자·우산 등 소지품, 없으면 \\"없음\\"", '
        '"consistent": crop 들이 모두 같은 옷차림이면 true, 아니면 false, '
        '"name": "폴더 이름용 15자 이내 요약 (예: 노란 반팔에 검은 바지)"}'
    )


def extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """응답에서 첫 JSON 객체를 찾는다. 코드펜스·앞뒤 말은 무시한다."""
    if not raw:
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fence.group(1)] if fence else []
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def clean_name(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().strip('"\'“”‘’.。')
    return text[:MAX_NAME_CHARS].rstrip()


def to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "y", "1", "예", "일치", "같음"):
            return True
        if v in ("false", "no", "n", "0", "아니오", "아님", "불일치", "다름"):
            return False
    return None


def interpret_reply(raw: str, keep_mixed: bool) -> Dict[str, Any]:
    """Qwen 응답 → status / name / 필드. name 이 없으면 upper+lower 로 만든다."""
    data = extract_json_object(raw)
    if data is None:
        return dict(status="parse_error", name="", upper="", lower="", items="", consistent=None)
    upper, lower, items = clean_name(data.get("upper")), clean_name(data.get("lower")), clean_name(data.get("items"))
    name = clean_name(data.get("name")) or " ".join(p for p in (upper, lower) if p)[:MAX_NAME_CHARS]
    consistent = to_bool(data.get("consistent"))
    if not name:
        return dict(status="parse_error", name="", upper=upper, lower=lower, items=items, consistent=consistent)
    status = "mixed" if consistent is False else "labeled"
    return dict(status=status, name=name if (status == "labeled" or keep_mixed) else "", qwen_name=name,
                upper=upper, lower=lower, items=items, consistent=consistent)


def make_record(cid: str, rank: int, size: int, reply: Dict[str, Any], raw: str, reps: Sequence[str],
                montage: Optional[str], model_id: str, elapsed: float) -> Dict[str, Any]:
    status = reply["status"]
    description = ", ".join(f"{k}={reply.get(k) or '-'}" for k in ("upper", "lower", "items")) + \
        f", consistent={reply.get('consistent')}"
    return dict(cluster_id=cid, rank=rank, cluster_size=size, cluster_name=reply.get("name", ""),
                display_name=reply.get("name") or reply.get("qwen_name") or STATUS_LABEL.get(status, status),
                status=status, label_confidence=1.0 if status == "labeled" else 0.5 if reply.get("name") else 0.0,
                cluster_description=description,
                qwen=dict(upper=reply.get("upper"), lower=reply.get("lower"), items=reply.get("items"),
                          consistent=reply.get("consistent"), name=reply.get("qwen_name") or reply.get("name"), raw=raw),
                representative_point_ids=list(reps), montage=montage, model_id=model_id, elapsed_sec=round(elapsed, 3))


def load_existing(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and rec.get("cluster_id") and rec.get("status") not in ("parse_error", "no_images"):
                out[str(rec["cluster_id"])] = rec
    return out


# ----------------------------------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------------------------------
def write_html(path: Path, title: str, summary: Dict[str, Any], records: List[Dict[str, Any]],
               warnings: List[Dict[str, Any]]) -> None:
    from urllib.parse import quote
    status_counts = Counter(r["status"] for r in records)
    rows = ""
    for r in records:
        q = r.get("qwen") or {}
        thumb = f'<a href="{quote(r["montage"], safe="/")}"><img loading="lazy" src="{quote(r["montage"], safe="/")}" alt=""></a>' \
            if r.get("montage") else ""
        rows += (f"<tr><td>{r['rank']}</td><td class='thumb'>{thumb}</td><td><b>{esc(r['display_name'])}</b></td>"
                 f"<td>{esc(q.get('upper'))}</td><td>{esc(q.get('lower'))}</td><td>{esc(q.get('items'))}</td>"
                 f"<td>{'예' if q.get('consistent') is True else '아니오' if q.get('consistent') is False else '?'}</td>"
                 f"<td>{esc(STATUS_LABEL.get(r['status'], r['status']))}</td><td>{r['cluster_size']:,}</td>"
                 f"<td><code>{esc(r['cluster_id'])}</code></td></tr>")
    warn_html = ""
    if warnings:
        items = "".join(f"<li><b>{esc(w.get('code'))}</b> {esc(w.get('message'))}</li>" for w in warnings)
        warn_html = f'<div class="warn"><b>경고 {len(warnings)}</b><ul>{items}</ul></div>'
    html_text = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>{esc(title)}</title>
<style>
body{{font-family:'Malgun Gothic',Segoe UI,sans-serif;margin:20px;color:#222;background:#fafafa}}
h1{{font-size:20px;margin:0 0 8px}} .muted{{color:#777;font-size:12px}}
table{{border-collapse:collapse;background:#fff}} th,td{{border:1px solid #ddd;padding:4px 8px;font-size:13px;vertical-align:top}}
th{{background:#f0f0f0;text-align:left}} td.thumb img{{height:96px;display:block}} code{{background:#eee;padding:0 3px;font-size:11px}}
.warn{{background:#fff4e0;border:1px solid #f0c060;padding:8px 12px;margin:12px 0}}
</style></head><body>
<h1>{esc(title)}</h1>
<p class="muted">생성 {esc(summary.get('generated_at'))} · {esc(PRODUCER)} · 상태 {esc(dict(status_counts))}</p>
{table({k: summary[k] for k in ('collection', 'assignments_path', 'model_id', 'representatives', 'clusters',
                                  'labeled', 'mixed', 'parse_error', 'no_images', 'sec_per_cluster', 'elapsed_sec') if k in summary})}
<p class="muted">몽타주 = 군집 대표 crop 을 왼쪽부터 이어 붙인 것으로 Qwen 이 실제로 본 이미지. '일치' 는 Qwen 이 crop 들이 같은 옷차림이라고 답했는지. 혼합(불일치) 군집의 이름은 폴더에 붙지 않는다.</p>
{warn_html}
<table><tr><th>순위</th><th>몽타주</th><th>이름</th><th>상의</th><th>하의</th><th>소지품</th><th>일치</th><th>상태</th><th>크기</th><th>cluster_id</th></tr>
{rows}</table>
</body></html>
"""
    path.write_text(html_text, encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen3-VL 로 군집 문장형 외형 이름 붙이기")
    p.add_argument("--assignments", default="outputs/clustering/leiden_image_prw/person/person_leiden_assignments.jsonl")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=CONFIG_HELP)
    p.add_argument("--target", choices=["person", "object"], default=None)
    p.add_argument("--collection", default=None, help=CONFIG_HELP)
    p.add_argument("--qdrant-url", default=None, help=CONFIG_HELP)
    p.add_argument("--api-key", default=None)
    p.add_argument("--project-root", default=".")
    p.add_argument("--output-dir", default=None, help="기본: assignments 옆 labels_qwen/")
    p.add_argument("--model-id", default=DEFAULT_MODEL,
                   help="Qwen/Qwen3-VL-2B-Instruct(기본) / Qwen/Qwen3-VL-4B-Instruct(더 구체적, 로드에 RAM 9GB+ 필요)")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default=None, help="비우면 cuda 가 있으면 cuda")
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=16, help="한 번에 생성할 군집 수. VRAM 이 부족하면 줄인다")
    p.add_argument("--representatives", type=int, default=6, help="몽타주에 넣을 crop 수")
    p.add_argument("--montage-height", type=int, default=192)
    p.add_argument("--max-clusters", type=int, default=0, help="큰 군집부터 이 수만큼. 0 이면 전부")
    p.add_argument("--min-cluster-size", type=int, default=2)
    p.add_argument("--lang", choices=["ko", "en"], default="ko")
    p.add_argument("--keep-mixed", action="store_true", help="Qwen 이 불일치라고 한 군집에도 이름을 붙인다")
    p.add_argument("--resume", action="store_true", help="기존 cluster_labels.jsonl 의 군집은 건너뛴다")
    p.add_argument("--qdrant-batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--title", default=None)
    return p


def parse_args(argv=None) -> argparse.Namespace:
    p = build_parser()
    args = p.parse_args(argv)
    for name, minimum in (("representatives", 1), ("montage_height", 64), ("max_clusters", 0), ("min_cluster_size", 1),
                          ("max_new_tokens", 16), ("qdrant_batch_size", 1), ("batch_size", 1)):
        if getattr(args, name) < minimum:
            p.error(f"--{name.replace('_', '-')} must be >= {minimum}")
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
    project_root = Path(args.project_root).resolve()
    out_dir = (Path(args.output_dir) if args.output_dir else assignments_path.parent / "labels_qwen").resolve()
    montage_dir = out_dir / "montage"
    montage_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "cluster_labels.jsonl"
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
    groups, _ = split_groups(rows)
    ordered = [(cid, members) for cid, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
               if len(members) >= args.min_cluster_size]
    if args.max_clusters:
        ordered = ordered[:args.max_clusters]
    if not ordered:
        build_parser().error("라벨을 붙일 군집이 없다")
    existing = load_existing(jsonl_path) if args.resume else {}

    reps_by_cluster = {cid: choose_representatives(members, args.representatives, args.seed + i)
                       for i, (cid, members) in enumerate(ordered) if cid not in existing}
    ids = list(dict.fromkeys(str(r["point_id"]) for reps in reps_by_cluster.values() for r in reps))
    print("=" * 88)
    print("CLUSTER SENTENCE LABELS (Qwen3-VL)")
    print("=" * 88)
    print(f"assignments : {assignments_path}")
    print(f"clusters    : {len(ordered):,} (resume skip {len(existing):,})   representatives/cluster: {args.representatives}")
    print(f"model       : {args.model_id} ({args.dtype})   crops to fetch: {len(ids):,}")
    # 모델을 가장 먼저 올린다. payload 조회·몽타주 생성 뒤에 올리면 RAM 이 빠듯한 기계(16GB, 다른 앱 다수)에서
    # 가중치 로드 중 세그폴트(exit 139)가 났다 — 로드 직후 CPU 쪽 임시 메모리는 풀리므로 순서만 바꾸면 된다.
    qwen = BatchedQwenVL(model_id=args.model_id, dtype=args.dtype, device=args.device, max_new_tokens=args.max_new_tokens)
    if reps_by_cluster:
        t_load = time.time()
        qwen.load()
        print(f"model loaded: {time.time() - t_load:.1f}s")
    payloads = QdrantHTTP(args.qdrant_url, args.api_key).retrieve_points(args.collection, ids, args.qdrant_batch_size) if ids else {}

    from PIL import Image
    # 1) 몽타주를 먼저 전부 만든다 (빠름). crop 이 하나도 없는 군집은 바로 기록한다.
    records: List[Dict[str, Any]] = []
    pending: List[Tuple[int, str, int, List[str], str, str]] = []   # rank, cid, size, used ids, montage rel, prompt
    for rank, (cid, members) in enumerate(ordered, 1):
        if cid in existing:
            records.append(dict(existing[cid], rank=rank))
            continue
        images, used = [], []
        for rep in reps_by_cluster[cid]:
            pid = str(rep["point_id"])
            src = resolve_path(first_payload(payloads.get(pid, {}), CROP_KEYS), project_root)
            if src is None:
                continue
            try:
                with Image.open(src) as im:
                    images.append(im.convert("RGB").copy())
                used.append(pid)
            except OSError:
                continue
        if not images:
            reply = dict(status="no_images", name="", upper="", lower="", items="", consistent=None)
            records.append(make_record(cid, rank, len(members), reply, "", used, None, args.model_id, 0.0))
            continue
        montage_name = f"c{rank:04d}_{short_id(cid)}.jpg"
        build_montage(images, args.montage_height).save(montage_dir / montage_name, "JPEG", quality=82)
        pending.append((rank, cid, len(members), used, f"montage/{montage_name}", build_prompt(len(images), args.lang)))
    print(f"montages    : {len(pending):,} made in {montage_dir}")

    # 2) Qwen 배치 생성. 배치가 실패하면 그 배치만 한 장씩 다시 시도한다.
    started = time.time()
    done = 0
    with jsonl_path.open("a" if args.resume else "w", encoding="utf-8") as stream:
        for rec in records:
            if rec["status"] == "no_images":
                stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            paths = [str(out_dir / item[4]) for item in batch]
            t0 = time.time()
            try:
                raws = qwen.generate_batch([item[5] for item in batch], paths)
            except Exception as exc:  # noqa: BLE001 — 배치 실패(OOM 등) → 한 장씩
                print(f"[경고] 배치 생성 실패 ({type(exc).__name__}: {str(exc)[:120]}) — 한 장씩 재시도")
                raws = []
                for item, path in zip(batch, paths):
                    try:
                        raws.append(qwen.generate(item[5], image_path=path))
                    except Exception as exc2:  # noqa: BLE001
                        raws.append(f"[generate error] {type(exc2).__name__}: {exc2}")
            per_item = (time.time() - t0) / len(batch)
            for (rank, cid, size, used, montage_rel, _), raw in zip(batch, raws):
                rec = make_record(cid, rank, size, interpret_reply(raw, args.keep_mixed), raw, used, montage_rel,
                                  args.model_id, per_item)
                records.append(rec)
                stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stream.flush()
            done += len(batch)
            per = (time.time() - started) / done
            print(f"[{done:,}/{len(pending):,}] {per:.2f}s/cluster · 남은 예상 {per * (len(pending) - done) / 60:.1f}분"
                  f" · 최근: {records[-1]['display_name']}")
    qwen.release()
    elapsed = time.time() - started

    records.sort(key=lambda r: r["rank"])
    write_json(out_dir / "cluster_labels.json", records)
    if args.resume:   # 순위를 다시 매긴 전체 목록으로 jsonl 을 정리해 둔다
        jsonl_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8", newline="\n")
    status_counts = Counter(r["status"] for r in records)
    if status_counts.get("parse_error", 0) > len(records) * 0.2:
        warnings.append(warning("MANY_PARSE_ERRORS", f"응답 해석 실패 {status_counts['parse_error']:,}/{len(records):,} — 프롬프트/모델 확인", target))
    if status_counts.get("no_images", 0):
        warnings.append(warning("NO_IMAGES", f"crop 을 하나도 못 읽은 군집 {status_counts['no_images']:,}개", target))

    title = args.title or f"군집 문장형 라벨 (Qwen3-VL) · {target or args.collection}"
    summary = common_summary(PRODUCER, html_path, inputs, warnings)
    summary.update(dict(
        target=target, collection=args.collection, assignments_path=str(assignments_path), model_id=args.model_id,
        dtype=args.dtype, representatives=args.representatives, montage_height=args.montage_height, lang=args.lang,
        keep_mixed=args.keep_mixed, clusters=len(records), labeled=status_counts.get("labeled", 0),
        mixed=status_counts.get("mixed", 0), parse_error=status_counts.get("parse_error", 0),
        no_images=status_counts.get("no_images", 0), elapsed_sec=round(elapsed, 1),
        sec_per_cluster=round(elapsed / done, 3) if done else None, resumed=len(existing),
        outputs=dict(jsonl=str(jsonl_path), json=str(out_dir / "cluster_labels.json"), montage_dir=str(montage_dir)),
        config=applied_config(settings, collection=args.collection, qdrant_url=args.qdrant_url, target=target),
    ))
    write_html(html_path, title, summary, records, warnings)
    write_json(report_path, summary)
    print("-" * 88)
    print(f"clusters    : {len(records):,}   status {dict(status_counts)}   {elapsed / 60:.1f}분")
    for r in records[:8]:
        print(f"  #{r['rank']:<4} n={r['cluster_size']:<4} {r['display_name']}")
    print(f"warnings    : {len(warnings)}")
    result_markers(report_path, html_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
