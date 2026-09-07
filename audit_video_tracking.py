from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parent
PERSON_ROOT = ROOT / "data" / "video_tracks" / "person"
OBJECT_ROOT = ROOT / "data" / "video_tracks" / "object"
DEFAULT_OUT = ROOT / "data" / "validation" / "video_tracking_audit"

# REVIEW용 휴리스틱. 정답 임계값이 아니라 수동 검토 우선순위용이다.
PERSON_RULES = {
    "all_filtered_min_raw": 20,
    "high_small_filter_ratio": 0.80,
    "low_tracking_yield_ratio": 0.70,
    "many_tracks": 50,
    "short_track_ratio": 0.50,
}
OBJECT_RULES = {
    "all_filtered_min_raw": 20,
    "high_small_filter_ratio": 0.80,
    "low_tracking_yield_ratio": 0.70,
    "high_dup_ratio": 0.20,
    "many_tracks": 50,
    "short_track_ratio": 0.50,
    "class_vote_low_dominance": 0.60,
    "class_vote_close_top_ratio": 1.20,
    "disappeared_class_min_count": 10,
}


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                rows.append({"__parse_error__": f"line {line_no}: {type(exc).__name__}: {exc}"})
                continue
            rows.append(obj if isinstance(obj, dict) else {"__parse_error__": f"line {line_no}: not object"})
    return rows


def i(v: Any, d: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return d


def f(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return d


def r(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def get_track_id(row: dict):
    for k in ("track_id", "original_track_id", "tracker_id"):
        if row.get(k) is not None:
            try:
                return int(row[k])
            except Exception:
                return None
    return None


def get_frame(row: dict):
    for k in ("frame_idx", "frame_number", "frame"):
        if row.get(k) is not None:
            try:
                return int(row[k])
            except Exception:
                return None
    return None


def valid_bbox(row: dict) -> bool:
    b = row.get("bbox")
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return False
    try:
        x1, y1, x2, y2 = map(float, b)
    except Exception:
        return False
    return x2 > x1 and y2 > y1


def crop_exists(row: dict) -> bool:
    p = row.get("crop_path") or row.get("path")
    if not p:
        return False
    q = Path(str(p))
    if not q.is_absolute():
        q = ROOT / q
    return q.is_file()


def flag(flags, kind, video, severity, code, message, value=None):
    flags.append({
        "kind": kind, "video": video, "severity": severity,
        "code": code, "value": value, "message": message,
    })


def track_stats(rows: list[dict]) -> dict:
    parse_errors = sum(1 for x in rows if "__parse_error__" in x)
    valid = [x for x in rows if "__parse_error__" not in x]
    by_track = defaultdict(list)
    missing_tid = invalid_bbox = missing_crop = dup_tf = 0
    seen = set()

    for row in valid:
        tid = get_track_id(row)
        fr = get_frame(row)
        if tid is None:
            missing_tid += 1
        else:
            by_track[tid].append(row)
        if not valid_bbox(row):
            invalid_bbox += 1
        if not crop_exists(row):
            missing_crop += 1
        if tid is not None and fr is not None:
            key = (tid, fr)
            if key in seen:
                dup_tf += 1
            seen.add(key)

    nonmono = 0
    lens = []
    for items in by_track.values():
        lens.append(len(items))
        frames = [get_frame(x) for x in items]
        frames = [x for x in frames if x is not None]
        if any(a > b for a, b in zip(frames, frames[1:])):
            nonmono += 1

    short = sum(1 for n in lens if n <= 2)
    return {
        "rows": len(valid),
        "parse_errors": parse_errors,
        "track_count_from_rows": len(by_track),
        "short_tracks_le2": short,
        "singleton_tracks": sum(1 for n in lens if n == 1),
        "short_track_ratio": r(short, len(by_track)),
        "median_rows_per_track": float(median(lens)) if lens else 0.0,
        "max_rows_per_track": max(lens) if lens else 0,
        "missing_track_id_rows": missing_tid,
        "invalid_bbox_rows": invalid_bbox,
        "missing_crop_rows": missing_crop,
        "duplicate_track_frame_rows": dup_tf,
        "non_monotonic_tracks": nonmono,
    }


def audit_common(kind: str, d: Path, summary: dict, rows: list[dict], flags: list[dict]) -> dict:
    ts = track_stats(rows)
    video = d.name
    saved = i(summary.get("saved_person_crops") if kind == "person" else summary.get("saved_object_crops"))
    meta = i(summary.get("metadata_rows"))
    tracks = i(summary.get("unique_person_tracks") if kind == "person" else summary.get("unique_object_tracks"))

    if saved != meta:
        flag(flags, kind, video, "FAIL", "SUMMARY_CROP_METADATA_MISMATCH", "summary crop 수와 metadata_rows가 다름", f"{saved}!={meta}")
    if ts["rows"] != meta:
        flag(flags, kind, video, "FAIL", "TRACKS_JSONL_ROW_MISMATCH", "실제 tracks.jsonl row 수와 metadata_rows가 다름", f"{ts['rows']}!={meta}")
    if ts["track_count_from_rows"] != tracks:
        flag(flags, kind, video, "REVIEW", "TRACK_COUNT_MISMATCH", "tracks.jsonl 계산 track 수와 summary track 수가 다름", f"{ts['track_count_from_rows']}!={tracks}")

    checks = [
        ("parse_errors", "JSONL_PARSE_ERROR", "tracks.jsonl 파싱 오류"),
        ("missing_track_id_rows", "MISSING_TRACK_ID", "track_id 없는 row"),
        ("invalid_bbox_rows", "INVALID_BBOX", "유효하지 않은 bbox"),
        ("missing_crop_rows", "MISSING_CROP_FILE", "crop_path 파일 없음"),
        ("duplicate_track_frame_rows", "DUPLICATE_TRACK_FRAME", "같은 track/frame 중복"),
        ("non_monotonic_tracks", "NON_MONOTONIC_TRACK", "track 내부 frame 순서 역행"),
    ]
    for key, code, msg in checks:
        if ts[key] > 0:
            flag(flags, kind, video, "FAIL", code, msg, ts[key])
    return ts


def audit_person(d: Path, flags: list[dict]) -> dict:
    s = load_json(d / "raw_detection_summary.json")
    rows = load_jsonl(d / "tracks.jsonl")
    ts = audit_common("person", d, s, rows, flags)

    raw = i(s.get("raw_person_detections"))
    acc = i(s.get("accepted_person_detections"))
    small = i(s.get("filtered_small_person_detections"))
    crops = i(s.get("saved_person_crops"))
    tracks = i(s.get("unique_person_tracks"))
    sr = r(small, raw)
    y = r(crops, acc)

    if raw >= 20 and acc == 0:
        flag(flags, "person", d.name, "REVIEW", "ALL_PERSON_FILTERED", "person detection은 많지만 크기 필터 통과 0", raw)
    if raw >= 20 and sr >= PERSON_RULES["high_small_filter_ratio"]:
        flag(flags, "person", d.name, "REVIEW", "HIGH_PERSON_SMALL_FILTER_RATIO", "person 크기 필터 탈락 비율 높음", round(sr, 4))
    if acc > 0 and crops == 0:
        flag(flags, "person", d.name, "FAIL", "ACCEPTED_BUT_NO_PERSON_CROP", "accepted person이 있는데 crop 0", acc)
    if acc >= 20 and y < PERSON_RULES["low_tracking_yield_ratio"]:
        flag(flags, "person", d.name, "REVIEW", "LOW_PERSON_TRACKING_YIELD", "accepted 대비 track crop 저장 비율 낮음", round(y, 4))
    if tracks >= PERSON_RULES["many_tracks"]:
        flag(flags, "person", d.name, "REVIEW", "MANY_PERSON_TRACKS", "person track 수가 많아 fragmentation 검토 필요", tracks)
    if tracks >= 10 and ts["short_track_ratio"] >= PERSON_RULES["short_track_ratio"]:
        flag(flags, "person", d.name, "REVIEW", "MANY_SHORT_PERSON_TRACKS", "2 crop 이하 짧은 person track 비율 높음", round(ts["short_track_ratio"], 4))

    return {
        "kind": "person", "video": d.name,
        "sampled_frames": i(s.get("sampled_frames")),
        "raw_detections": i(s.get("raw_detection_total")),
        "raw_target_detections": raw,
        "accepted": acc, "small_filtered": small,
        "small_filter_ratio": round(sr, 6),
        "saved_crops": crops, "tracking_yield": round(y, 6),
        "summary_tracks": tracks, **ts,
    }


def vote_stats(info: dict):
    counts = info.get("class_counts") or {}
    confs = info.get("class_confidence_sums") or {}
    counts = counts if isinstance(counts, dict) else {}
    confs = confs if isinstance(confs, dict) else {}
    total = sum(max(0, i(v)) for v in counts.values())
    dom = max([i(v) for v in counts.values()] or [0]) / total if total else 0.0
    vals = sorted([max(0.0, f(v)) for v in confs.values()], reverse=True)
    top_ratio = math.inf
    if len(vals) >= 2 and vals[1] > 0:
        top_ratio = vals[0] / vals[1]
    return dom, top_ratio, len(counts)


def audit_object(d: Path, flags: list[dict]) -> dict:
    s = load_json(d / "raw_detection_summary.json")
    csum = load_json(d / "track_class_summary.json")
    rows = load_jsonl(d / "tracks.jsonl")
    ts = audit_common("object", d, s, rows, flags)

    raw = i(s.get("raw_object_detections"))
    cand = i(s.get("candidate_object_detections", s.get("candidate_object", 0)))
    acc = i(s.get("accepted_object_detections", s.get("accepted_after_dedup", 0)))
    small = i(s.get("filtered_small_object_detections"))
    dup = i(s.get("duplicate_suppressed_detections", s.get("duplicate_suppressed", 0)))
    crops = i(s.get("saved_object_crops"))
    tracks = i(s.get("unique_object_tracks"))
    sr = r(small, raw)
    y = r(crops, acc)
    dr = r(dup, max(cand, acc + dup, 1))

    if raw >= 20 and acc == 0:
        flag(flags, "object", d.name, "REVIEW", "ALL_OBJECT_FILTERED", "object detection은 많지만 accepted 0", raw)
    if raw >= 20 and sr >= OBJECT_RULES["high_small_filter_ratio"]:
        flag(flags, "object", d.name, "REVIEW", "HIGH_OBJECT_SMALL_FILTER_RATIO", "object 크기 필터 탈락 비율 높음", round(sr, 4))
    if acc > 0 and crops == 0:
        flag(flags, "object", d.name, "FAIL", "ACCEPTED_BUT_NO_OBJECT_CROP", "accepted object가 있는데 crop 0", acc)
    if acc >= 20 and y < OBJECT_RULES["low_tracking_yield_ratio"]:
        flag(flags, "object", d.name, "REVIEW", "LOW_OBJECT_TRACKING_YIELD", "accepted 대비 track crop 저장 비율 낮음", round(y, 4))
    if dr >= OBJECT_RULES["high_dup_ratio"]:
        flag(flags, "object", d.name, "REVIEW", "HIGH_DUPLICATE_SUPPRESSION", "dedup 제거 비율 높음", round(dr, 4))
    if tracks >= OBJECT_RULES["many_tracks"]:
        flag(flags, "object", d.name, "REVIEW", "MANY_OBJECT_TRACKS", "object track 수가 많아 fragmentation 검토 필요", tracks)
    if tracks >= 10 and ts["short_track_ratio"] >= OBJECT_RULES["short_track_ratio"]:
        flag(flags, "object", d.name, "REVIEW", "MANY_SHORT_OBJECT_TRACKS", "2 crop 이하 짧은 object track 비율 높음", round(ts["short_track_ratio"], 4))

    multi = amb = lowdom = close = 0
    if isinstance(csum, dict):
        for info in csum.values():
            if not isinstance(info, dict):
                continue
            dom, top_ratio, ncls = vote_stats(info)
            if ncls > 1:
                multi += 1
            suspicious = False
            if ncls > 1 and dom < OBJECT_RULES["class_vote_low_dominance"]:
                lowdom += 1; suspicious = True
            if ncls > 1 and top_ratio != math.inf and top_ratio <= OBJECT_RULES["class_vote_close_top_ratio"]:
                close += 1; suspicious = True
            if suspicious:
                amb += 1
    if amb:
        flag(flags, "object", d.name, "REVIEW", "AMBIGUOUS_CLASS_VOTING", "class voting이 팽팽한 track 존재", amb)

    accepted_classes = s.get("accepted_class_counts") or s.get("accepted_raw_classes") or {}
    final_classes = s.get("tracks_by_class") or s.get("final_tracks_by_class") or {}
    disappeared = {}
    if isinstance(accepted_classes, dict) and isinstance(final_classes, dict):
        for cls, cnt in accepted_classes.items():
            if i(cnt) >= OBJECT_RULES["disappeared_class_min_count"] and i(final_classes.get(cls)) == 0:
                disappeared[str(cls)] = i(cnt)
    if disappeared:
        flag(flags, "object", d.name, "REVIEW", "ACCEPTED_CLASS_DISAPPEARED_AFTER_TRACKING", "accepted detection이 충분했던 class가 최종 track class에서 0", disappeared)

    return {
        "kind": "object", "video": d.name,
        "sampled_frames": i(s.get("sampled_frames")),
        "raw_detections": i(s.get("raw_detection_total")),
        "raw_target_detections": raw,
        "candidate": cand, "accepted": acc,
        "small_filtered": small, "small_filter_ratio": round(sr, 6),
        "duplicate_suppressed": dup, "duplicate_ratio": round(dr, 6),
        "person_excluded": i(s.get("excluded_person_detections")),
        "saved_crops": crops, "tracking_yield": round(y, 6),
        "summary_tracks": tracks,
        "class_voting_multi_class_tracks": multi,
        "class_voting_ambiguous_tracks": amb,
        "class_voting_low_dominance_tracks": lowdom,
        "class_voting_close_vote_tracks": close,
        "disappeared_accepted_classes": json.dumps(disappeared, ensure_ascii=False),
        **ts,
    }


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Person/Object tracking 구조 오류 및 의심 영상 자동 선별")
    ap.add_argument("--type", choices=["all", "person", "object"], default="all")
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows, flags = [], []

    if args.type in ("all", "person") and PERSON_ROOT.exists():
        for d in sorted(x for x in PERSON_ROOT.iterdir() if x.is_dir()):
            rows.append(audit_person(d, flags))
    if args.type in ("all", "object") and OBJECT_ROOT.exists():
        for d in sorted(x for x in OBJECT_ROOT.iterdir() if x.is_dir()):
            rows.append(audit_object(d, flags))

    sev = {"FAIL": 0, "REVIEW": 1}
    flags.sort(key=lambda x: (sev.get(x["severity"], 9), x["kind"], x["video"], x["code"]))
    fail_n = sum(x["severity"] == "FAIL" for x in flags)
    review_n = sum(x["severity"] == "REVIEW" for x in flags)
    flagged = len({(x["kind"], x["video"]) for x in flags})
    counts = Counter(x["code"] for x in flags)

    write_csv(out / "tracking_audit_videos.csv", rows)
    write_csv(out / "tracking_audit_flags.csv", flags)
    (out / "tracking_audit_flag_counts.json").write_text(json.dumps(dict(counts.most_common()), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "tracking_audit_summary.json").write_text(json.dumps({
        "audited_video_dirs": len(rows),
        "person_video_dirs": sum(x["kind"] == "person" for x in rows),
        "object_video_dirs": sum(x["kind"] == "object" for x in rows),
        "fail_flags": fail_n,
        "review_flags": review_n,
        "flagged_video_dirs": flagged,
        "note": "FAIL=구조/파일 일관성 문제, REVIEW=품질 의심 후보. REVIEW 휴리스틱은 GT가 아니며 자동 삭제/수정에 사용하지 말 것.",
        "person_review_rules": PERSON_RULES,
        "object_review_rules": OBJECT_RULES,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 88)
    print("VIDEO TRACKING AUDIT")
    print("=" * 88)
    print("audited video dirs :", len(rows))
    print("FAIL flags         :", fail_n)
    print("REVIEW flags       :", review_n)
    print("flagged video dirs :", flagged)
    print("\noutput:")
    for name in ("tracking_audit_summary.json", "tracking_audit_videos.csv", "tracking_audit_flags.csv", "tracking_audit_flag_counts.json"):
        print(" ", out / name)
    if counts:
        print("\nTOP FLAG CODES")
        for code, n in counts.most_common(20):
            print(f"  {code:45s} {n:6d}")
    print("\n[PASS] 구조적 치명 오류 없음" if fail_n == 0 else "\n[FAIL] 구조적 오류 후보 있음 - FAIL부터 확인")
    if review_n:
        print("[REVIEW] 품질 의심 영상 존재 - 수동 검토 우선순위용")


if __name__ == "__main__":
    main()
