from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parent
PERSON_ROOT = ROOT / "data" / "video_tracks" / "person"
DEFAULT_OUT = ROOT / "data" / "validation" / "person_stitching_audit"

# REVIEW heuristics only. These are not GT-calibrated failure thresholds.
RULES = {
    "large_identity_tracks": 10,
    "heavy_compression_min_tracklets": 20,
    "heavy_compression_person_per_tracklet": 0.25,
    "near_fused_margin": 0.03,
    "low_general_global": 0.70,
    "long_general_gap_sec": 8.0,
    "large_scale_ratio": 2.0,
    "ambiguous_top2_margin": 0.03,
}


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
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
                rows.append({
                    "__parse_error__": (
                        f"{path.name}: line {line_no}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                })
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                rows.append({
                    "__parse_error__": f"{path.name}: line {line_no}: non-object row"
                })
    return rows


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def safe_float(v: Any, default=None):
    try:
        return float(v)
    except Exception:
        return default


def add_flag(
    flags: List[dict],
    video: str,
    severity: str,
    code: str,
    value: Any,
    message: str,
):
    flags.append({
        "video": video,
        "severity": severity,
        "code": code,
        "value": value,
        "message": message,
    })


def write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def audit_video(video_dir: Path, flags: List[dict]) -> dict:
    video = video_dir.name

    tracks_path = video_dir / "tracks.jsonl"
    stitched_path = video_dir / "stitched_tracks_v4_1.jsonl"
    summary_path = video_dir / "stitching_v4_1.json"
    cand_path = video_dir / "stitch_candidates_v4_1.jsonl"
    crop_error_path = video_dir / "stitch_crop_errors_v4_1.jsonl"
    cache_path = video_dir / "stitch_embeddings_solider_v4_1.npz"

    original = load_jsonl(tracks_path)
    stitched = load_jsonl(stitched_path)
    summary = load_json(summary_path)
    candidates = load_jsonl(cand_path)
    crop_errors = load_jsonl(crop_error_path)

    original_parse = sum("__parse_error__" in r for r in original)
    stitched_parse = sum("__parse_error__" in r for r in stitched)
    candidate_parse = sum("__parse_error__" in r for r in candidates)

    original_valid = [r for r in original if "__parse_error__" not in r]
    stitched_valid = [r for r in stitched if "__parse_error__" not in r]
    candidates_valid = [r for r in candidates if "__parse_error__" not in r]

    if original_parse:
        add_flag(
            flags, video, "FAIL", "ORIGINAL_JSONL_PARSE_ERROR",
            original_parse, "tracks.jsonl 파싱 오류",
        )

    if stitched_parse:
        add_flag(
            flags, video, "FAIL", "STITCHED_JSONL_PARSE_ERROR",
            stitched_parse, "stitched_tracks_v4_1.jsonl 파싱 오류",
        )

    if candidate_parse:
        add_flag(
            flags, video, "FAIL", "CANDIDATE_JSONL_PARSE_ERROR",
            candidate_parse, "stitch candidate 로그 파싱 오류",
        )

    if not summary:
        add_flag(
            flags, video, "FAIL", "MISSING_OR_INVALID_SUMMARY",
            str(summary_path), "stitching_v4_1.json이 없거나 읽을 수 없음",
        )

    if not stitched_path.exists():
        add_flag(
            flags, video, "FAIL", "MISSING_STITCHED_OUTPUT",
            str(stitched_path), "stitched output 파일 없음",
        )

    input_rows = safe_int(summary.get("input_rows"), len(original_valid))
    input_tracklets = safe_int(summary.get("input_tracklets"), 0)
    output_persons = safe_int(summary.get("output_person_ids"), 0)
    output_rows = safe_int(summary.get("output_rows"), len(stitched_valid))
    skipped_rows = safe_int(summary.get("skipped_output_rows"), 0)
    crop_error_count = safe_int(summary.get("crop_errors"), len(crop_errors))

    unique_orig_tracks = {
        safe_int(r.get("track_id"), -1)
        for r in original_valid
        if r.get("track_id") is not None
    }
    unique_orig_tracks.discard(-1)

    unique_stitched_orig_tracks = {
        safe_int(r.get("original_track_id"), -1)
        for r in stitched_valid
        if r.get("original_track_id") is not None
    }
    unique_stitched_orig_tracks.discard(-1)

    person_ids = {
        safe_int(r.get("person_id"), -1)
        for r in stitched_valid
        if r.get("person_id") is not None
    }
    person_ids.discard(-1)

    if len(original_valid) != input_rows:
        add_flag(
            flags, video, "FAIL", "INPUT_ROW_COUNT_MISMATCH",
            f"{len(original_valid)}!={input_rows}",
            "실제 tracks.jsonl row 수와 summary input_rows 불일치",
        )

    if len(stitched_valid) != output_rows:
        add_flag(
            flags, video, "FAIL", "OUTPUT_ROW_COUNT_MISMATCH",
            f"{len(stitched_valid)}!={output_rows}",
            "실제 stitched row 수와 summary output_rows 불일치",
        )

    if input_rows != output_rows or skipped_rows != 0:
        add_flag(
            flags, video, "FAIL", "STITCH_ROW_LOSS",
            f"input={input_rows}, output={output_rows}, skipped={skipped_rows}",
            "stitching 과정에서 row가 누락됨",
        )

    if len(unique_orig_tracks) != input_tracklets:
        add_flag(
            flags, video, "FAIL", "INPUT_TRACKLET_COUNT_MISMATCH",
            f"{len(unique_orig_tracks)}!={input_tracklets}",
            "tracks.jsonl의 unique track 수와 summary input_tracklets 불일치",
        )

    if unique_orig_tracks != unique_stitched_orig_tracks:
        missing = sorted(unique_orig_tracks - unique_stitched_orig_tracks)
        extra = sorted(unique_stitched_orig_tracks - unique_orig_tracks)
        add_flag(
            flags, video, "FAIL", "TRACK_ASSIGNMENT_MISMATCH",
            f"missing={missing[:20]}, extra={extra[:20]}",
            "원본 ByteTrack track이 stitched output에 정확히 1회 대응하지 않음",
        )

    if len(person_ids) != output_persons:
        add_flag(
            flags, video, "FAIL", "PERSON_ID_COUNT_MISMATCH",
            f"{len(person_ids)}!={output_persons}",
            "stitched rows의 unique person_id 수와 summary output_person_ids 불일치",
        )

    if crop_error_count != len(crop_errors):
        add_flag(
            flags, video, "FAIL", "CROP_ERROR_COUNT_MISMATCH",
            f"{len(crop_errors)}!={crop_error_count}",
            "crop error 로그 row 수와 summary 값 불일치",
        )

    if crop_error_count > 0:
        add_flag(
            flags, video, "REVIEW", "SOLIDER_CROP_ERRORS",
            crop_error_count,
            "SOLIDER embedding 생성 중 crop 오류가 있음",
        )

    if not cache_path.exists():
        add_flag(
            flags, video, "FAIL", "MISSING_SOLIDER_CACHE",
            str(cache_path), "SOLIDER stitch cache 파일 없음",
        )
        cache_person_keys = 0
    else:
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                cache_person_keys = sum(
                    1 for k in data.files
                    if k.startswith("person_") and k.endswith("_global")
                )
        except Exception as exc:
            cache_person_keys = 0
            add_flag(
                flags, video, "FAIL", "SOLIDER_CACHE_READ_ERROR",
                type(exc).__name__, "SOLIDER npz cache 읽기 실패",
            )

        if cache_person_keys != output_persons:
            add_flag(
                flags, video, "FAIL", "SOLIDER_IDENTITY_CACHE_COUNT_MISMATCH",
                f"{cache_person_keys}!={output_persons}",
                "person identity-global SOLIDER cache 개수 불일치",
            )

    persons = summary.get("persons") or []
    if not isinstance(persons, list):
        persons = []

    # Identity size / aggressive compression review.
    largest_identity = 0
    large_identity_count = 0

    for p in persons:
        if not isinstance(p, dict):
            continue
        tids = p.get("track_ids") or []
        if not isinstance(tids, list):
            tids = []
        n = len(tids)
        largest_identity = max(largest_identity, n)

        if n >= RULES["large_identity_tracks"]:
            large_identity_count += 1
            add_flag(
                flags, video, "REVIEW", "LARGE_STITCHED_PERSON",
                {
                    "person_id": p.get("person_id"),
                    "tracks": n,
                    "track_ids": tids,
                },
                "하나의 person_id에 많은 ByteTrack tracklet이 병합됨 - false merge 육안 검토 권장",
            )

    person_per_tracklet = (
        output_persons / input_tracklets
        if input_tracklets > 0 else 1.0
    )
    compression_ratio = (
        1.0 - person_per_tracklet
        if input_tracklets > 0 else 0.0
    )

    if (
        input_tracklets >= RULES["heavy_compression_min_tracklets"]
        and person_per_tracklet < RULES["heavy_compression_person_per_tracklet"]
    ):
        add_flag(
            flags, video, "REVIEW", "HEAVY_STITCH_COMPRESSION",
            round(person_per_tracklet, 4),
            (
                "tracklet 대비 최종 person 수가 매우 적음 - "
                "강한 fragmentation 복구일 수도 있으나 over-merge 검토 필요"
            ),
        )

    # Deduplicate merge metadata by original_track_id because it repeats on every row.
    merge_by_track: Dict[int, dict] = {}
    for r in stitched_valid:
        tid = r.get("original_track_id")
        if tid is None:
            continue
        tid = safe_int(tid, -1)
        if tid < 0:
            continue
        merge_by_track.setdefault(tid, r)

    fused_threshold = safe_float(
        ((summary.get("rules") or {}).get("general") or {}).get("fused_threshold"),
        0.70,
    )
    if fused_threshold is None:
        fused_threshold = 0.70

    weak_merge_count = 0
    long_general_count = 0
    low_global_general_count = 0
    large_scale_count = 0

    for tid, r in merge_by_track.items():
        rule = r.get("stitch_rule")
        if not rule:
            continue

        fused = safe_float(r.get("stitch_fused_score"))
        glob = safe_float(r.get("stitch_global_similarity"))
        gap = safe_float(r.get("stitch_gap_sec"))
        scale = safe_float(r.get("stitch_bbox_scale_ratio"))

        if fused is not None and fused <= fused_threshold + RULES["near_fused_margin"]:
            weak_merge_count += 1

        if rule == "general":
            if glob is not None and glob < RULES["low_general_global"]:
                low_global_general_count += 1
            if gap is not None and gap > RULES["long_general_gap_sec"]:
                long_general_count += 1

        if scale is not None and scale > RULES["large_scale_ratio"]:
            large_scale_count += 1

    if weak_merge_count:
        add_flag(
            flags, video, "REVIEW", "MERGES_NEAR_FUSED_THRESHOLD",
            weak_merge_count,
            "fused threshold 바로 위에서 승인된 merge가 있음",
        )

    if low_global_general_count:
        add_flag(
            flags, video, "REVIEW", "GENERAL_MERGE_LOW_GLOBAL",
            low_global_general_count,
            "general rule merge 중 SOLIDER global similarity가 낮은 사례가 있음",
        )

    if long_general_count:
        add_flag(
            flags, video, "REVIEW", "LONG_GAP_GENERAL_MERGE",
            long_general_count,
            "8초 초과 gap에서 general rule로 병합된 사례가 있음",
        )

    if large_scale_count:
        add_flag(
            flags, video, "REVIEW", "LARGE_SCALE_CHANGE_MERGE",
            large_scale_count,
            "병합 전후 bbox scale 변화가 큰 사례가 있음",
        )

    # Candidate ambiguity: same incoming track had >1 accepted identity candidate
    # with very similar fused scores.
    accepted_by_track: Dict[int, List[dict]] = defaultdict(list)
    for c in candidates_valid:
        if not bool(c.get("accepted")):
            continue
        tid = safe_int(c.get("track_id"), -1)
        if tid >= 0:
            accepted_by_track[tid].append(c)

    ambiguous_candidate_tracks = 0
    for tid, cs in accepted_by_track.items():
        distinct_persons = {
            safe_int(c.get("candidate_person_id"), -1) for c in cs
        }
        distinct_persons.discard(-1)

        if len(distinct_persons) < 2:
            continue

        scores = sorted(
            [
                safe_float(c.get("fused_score"))
                for c in cs
                if safe_float(c.get("fused_score")) is not None
            ],
            reverse=True,
        )
        if len(scores) >= 2 and (scores[0] - scores[1]) <= RULES["ambiguous_top2_margin"]:
            ambiguous_candidate_tracks += 1

    if ambiguous_candidate_tracks:
        add_flag(
            flags, video, "REVIEW", "AMBIGUOUS_ACCEPTED_IDENTITY_CANDIDATES",
            ambiguous_candidate_tracks,
            "한 tracklet에 대해 여러 person identity 후보가 비슷한 점수로 동시에 통과함",
        )

    rule_counts = summary.get("merge_rule_counts") or {}
    if not isinstance(rule_counts, dict):
        rule_counts = {}

    return {
        "video": video,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "input_tracklets": input_tracklets,
        "output_persons": output_persons,
        "person_per_tracklet": round(person_per_tracklet, 6),
        "compression_ratio": round(compression_ratio, 6),
        "largest_identity_tracks": largest_identity,
        "large_identity_count": large_identity_count,
        "merge_general": safe_int(rule_counts.get("general"), 0),
        "merge_strong_global": safe_int(rule_counts.get("strong_global"), 0),
        "merge_short_gap": safe_int(rule_counts.get("short_gap"), 0),
        "weak_merge_count": weak_merge_count,
        "low_global_general_count": low_global_general_count,
        "long_general_count": long_general_count,
        "large_scale_count": large_scale_count,
        "ambiguous_candidate_tracks": ambiguous_candidate_tracks,
        "crop_errors": crop_error_count,
        "solider_identity_cache_keys": cache_person_keys,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Person SOLIDER Stitching V4.1 구조/품질 audit"
    )
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    flags: List[dict] = []
    videos: List[dict] = []

    video_dirs = sorted(
        p for p in PERSON_ROOT.iterdir()
        if p.is_dir() and (p / "stitched_tracks_v4_1.jsonl").exists()
    ) if PERSON_ROOT.exists() else []

    for video_dir in video_dirs:
        videos.append(audit_video(video_dir, flags))

    severity_order = {"FAIL": 0, "REVIEW": 1}
    flags.sort(
        key=lambda x: (
            severity_order.get(x["severity"], 9),
            x["video"],
            x["code"],
        )
    )

    fail_count = sum(x["severity"] == "FAIL" for x in flags)
    review_count = sum(x["severity"] == "REVIEW" for x in flags)
    flagged_videos = len({x["video"] for x in flags})

    summary = {
        "audited_videos": len(videos),
        "fail_flags": fail_count,
        "review_flags": review_count,
        "flagged_videos": flagged_videos,
        "note": (
            "FAIL은 구조/파일 일관성 문제입니다. REVIEW는 false merge/false split 가능성을 "
            "우선적으로 육안 검토하기 위한 휴리스틱이며 실제 오류 판정이 아닙니다."
        ),
        "review_rules": RULES,
    }

    write_csv(out_root / "person_stitching_videos.csv", videos)
    write_csv(out_root / "person_stitching_flags.csv", flags)

    (out_root / "person_stitching_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    counts = Counter(x["code"] for x in flags)
    (out_root / "person_stitching_flag_counts.json").write_text(
        json.dumps(dict(counts.most_common()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 92)
    print("PERSON STITCHING V4.1 AUDIT")
    print("=" * 92)
    print("audited videos :", len(videos))
    print("FAIL flags     :", fail_count)
    print("REVIEW flags   :", review_count)
    print("flagged videos :", flagged_videos)

    print("\noutput:")
    print(" ", out_root / "person_stitching_summary.json")
    print(" ", out_root / "person_stitching_videos.csv")
    print(" ", out_root / "person_stitching_flags.csv")
    print(" ", out_root / "person_stitching_flag_counts.json")

    if counts:
        print("\nTOP FLAG CODES")
        for code, cnt in counts.most_common(20):
            print(f"  {code:48s} {cnt:6d}")

    if fail_count:
        print("\n[FAIL] 구조적/파일 일관성 문제 후보가 있습니다.")
    else:
        print("\n[PASS] 구조적 치명 오류는 자동 검사에서 발견되지 않았습니다.")

    if review_count:
        print("[REVIEW] 병합 품질 의심 후보가 있습니다. false merge 우선 검토용입니다.")


if __name__ == "__main__":
    main()
