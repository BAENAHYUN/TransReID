from __future__ import annotations

"""
verify_hybrid_c.py

Hybrid-C 변경사항 정적 + 합성 검증.

검증 범위
---------
- 필수 파일 존재
- Python 문법
- pipeline.yaml -> config.py -> cfg.duplicate_grouping 연결
- duplicate_grouping.py 공용 over-fetch / collapse
- image_search.py / search_db.py 하드코딩 제거 및 공용 helper 연결
- dedup_objects.py:
    exact detection_id만 hard remove
    semantic duplicate / ambiguous는 point 보존
    person soft-group 금지
- apply_manual_review.py:
    manual duplicate도 비파괴
    keep_ambiguous 보존
- rfdetr_adapter.py payload passthrough
- audit_db.py dedup metadata 감사 키
- review_ambiguous.py Hybrid-C safety gate

주의
----
이 스크립트는 모델, Qdrant, GPU를 실행하지 않는다.
실제 DB end-to-end 검증은 별도로 audit_db.py를 실행해야 한다.
"""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "dedup_objects.py",
    "apply_manual_review.py",
    "rfdetr_adapter.py",
    "image_search.py",
    "search_db.py",
    "audit_db.py",
    "review_ambiguous.py",
    "pipeline.yaml",
    "config.py",
    "duplicate_grouping.py",
]

PY_FILES = [x for x in REQUIRED_FILES if x.endswith(".py")]


def ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def fail(msg: str) -> None:
    raise AssertionError(msg)


def require(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)


def compile_all() -> None:
    for name in PY_FILES:
        path = ROOT / name
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
    ok("Python syntax")


def check_files() -> None:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    require(not missing, f"missing files: {missing}")
    ok("required files")


def check_config() -> None:
    sys.path.insert(0, str(ROOT))
    from config import PipelineConfig

    raw = yaml.safe_load((ROOT / "pipeline.yaml").read_text(encoding="utf-8"))
    dg_raw = raw.get("duplicate_grouping")
    require(isinstance(dg_raw, dict), "pipeline.yaml duplicate_grouping missing")

    cfg = PipelineConfig.load(ROOT / "pipeline.yaml")
    dg = cfg.duplicate_grouping

    require(bool(dg.enabled) is True, "duplicate_grouping.enabled")
    require(int(dg.overfetch_factor) == int(dg_raw["overfetch_factor"]), "overfetch mismatch")
    require(int(dg.max_fetch) == int(dg_raw["max_fetch"]), "max_fetch mismatch")
    require(str(dg.group_payload_key) == str(dg_raw["group_payload_key"]), "group key mismatch")
    require(
        str(dg.ambiguous_payload_key) == str(dg_raw["ambiguous_payload_key"]),
        "ambiguous key mismatch",
    )
    require(dg.group_payload_key != dg.ambiguous_payload_key, "group keys must differ")
    ok("pipeline.yaml -> config.py duplicate_grouping SSOT")


def check_shared_search_helper() -> None:
    image_text = (ROOT / "image_search.py").read_text(encoding="utf-8")
    search_text = (ROOT / "search_db.py").read_text(encoding="utf-8")

    forbidden = (
        "GROUP_OVERFETCH_FACTOR",
        "GROUP_OVERFETCH_MAX",
        "DUPLICATE_GROUP_OVERFETCH_FACTOR",
        "DUPLICATE_GROUP_MAX_FETCH",
        "def _group_fetch_limit",
        "def _object_retrieval_limit",
        "def collapse_duplicate_groups",
        "def _collapse_duplicate_group_hits",
    )
    for token in forbidden:
        require(token not in image_text, f"image_search.py still contains {token}")
        require(token not in search_text, f"search_db.py still contains {token}")

    for text, name in ((image_text, "image_search.py"), (search_text, "search_db.py")):
        require("from duplicate_grouping import" in text, f"{name}: shared helper import missing")
        require("self.cfg.duplicate_grouping" in text, f"{name}: cfg.duplicate_grouping not used")

    ok("image_search/search_db shared duplicate_grouping helper")


def check_duplicate_grouping_runtime() -> None:
    from config import PipelineConfig
    from duplicate_grouping import (
        collapse_duplicate_group_rows,
        object_retrieval_limit,
    )

    cfg = PipelineConfig.load(ROOT / "pipeline.yaml")

    factor = int(cfg.duplicate_grouping.overfetch_factor)
    cap = int(cfg.duplicate_grouping.max_fetch)

    for k in (1, 20, 50, 100, 300):
        expected = max(k, min(k * factor, cap))
        actual = object_retrieval_limit(k, cfg)
        require(actual == expected, f"overfetch mismatch k={k}: {actual} != {expected}")

    rows = [
        {
            "point_id": "p1",
            "image_id": "img",
            "label": "chair",
            "crop_path": "a.jpg",
            "bbox": [0, 0, 10, 10],
            "frame_idx": 0,
            "track_id": None,
            "score": 0.90,
            cfg.duplicate_grouping.group_payload_key: "G1",
        },
        {
            "point_id": "p2",
            "image_id": "img",
            "label": "couch",
            "crop_path": "b.jpg",
            "bbox": [0, 0, 10, 10],
            "frame_idx": 0,
            "track_id": None,
            "score": 0.95,
            cfg.duplicate_grouping.group_payload_key: "G1",
            cfg.duplicate_grouping.ambiguous_payload_key: "A1",
        },
        {
            "point_id": "p3",
            "image_id": "img",
            "label": "bottle",
            "crop_path": "c.jpg",
            "bbox": [20, 20, 30, 30],
            "frame_idx": 0,
            "track_id": None,
            "score": 0.92,
            cfg.duplicate_grouping.ambiguous_payload_key: "A1",
        },
    ]

    out, meta = collapse_duplicate_group_rows(
        rows,
        top_k=10,
        score_key="score",
        source=cfg,
    )
    require([x["point_id"] for x in out] == ["p2", "p3"], "group collapse representative wrong")
    require(meta["collapsed_hit_count"] == 1, "collapsed count wrong")
    require(meta["ambiguous_group_collapsed"] is False, "ambiguous group must never collapse")
    require(
        len(out[0].get("duplicate_group_members") or []) == 2,
        "group member audit list missing",
    )
    ok("duplicate_grouping runtime semantics")


def check_dedup_objects_core() -> None:
    import dedup_objects as d

    base = [
        {
            "image_id": "img1",
            "detection_id": "A",
            "class_name": "chair",
            "confidence": 0.9,
            "bbox": [0, 0, 100, 100],
            "path": "A.jpg",
        },
        {
            "image_id": "img1",
            "detection_id": "B",
            "class_name": "chair",
            "confidence": 0.8,
            "bbox": [1, 1, 101, 101],
            "path": "B.jpg",
        },
        {
            "image_id": "img1",
            "detection_id": "C",
            "class_name": "table",
            "confidence": 0.7,
            "bbox": [2, 2, 102, 102],
            "path": "C.jpg",
        },
        {
            "image_id": "img1",
            "detection_id": "P",
            "class_name": "person",
            "confidence": 0.95,
            "bbox": [0, 0, 40, 90],
            "path": "P.jpg",
        },
    ]

    originals = (
        d._collect_candidate_pairs,
        d._embed_candidates,
        d._pair_decision,
        d._dino_similarity,
    )
    try:
        d._collect_candidate_pairs = lambda *a, **kw: (
            [
                (0, 1, {"exact_same_bbox": False}),
                (1, 2, {"exact_same_bbox": False}),
            ],
            {0, 1, 2},
        )
        d._embed_candidates = lambda *a, **kw: ({0: 0, 1: 1, 2: 2}, None)
        decisions = iter(
            [
                {"decision": "duplicate", "reason": "test", "dino_similarity": 0.99},
                {"decision": "ambiguous", "reason": "test", "dino_similarity": 0.96},
            ]
        )
        d._pair_decision = lambda *a, **kw: next(decisions)
        d._dino_similarity = lambda *a, **kw: 0.99

        out, report = d.deduplicate(
            base,
            SimpleNamespace(person_labels={"person"}),
            policy_name="balanced",
            policy=d.POLICIES["balanced"],
            dino_batch_size=None,
            dino_outer_batch_size=1,
            dino_device=None,
        )

        require(len(out) == 4, "semantic relation deleted a point")
        by = {x["detection_id"]: x for x in out}
        require(
            by["A"]["duplicate_group_id"] == by["B"]["duplicate_group_id"],
            "semantic duplicate group missing",
        )
        require(
            by["B"]["ambiguous_group_id"] == by["C"]["ambiguous_group_id"],
            "ambiguous group missing",
        )
        require(
            "duplicate_group_id" not in by["P"]
            and "ambiguous_group_id" not in by["P"],
            "person received semantic relation",
        )
        require(report["removed_count"] == 0, "semantic removal occurred")
        require(report["same_class_removed"] == 0, "same-class semantic removal occurred")
        require(report["cross_class_removed"] == 0, "cross-class semantic removal occurred")
    finally:
        (
            d._collect_candidate_pairs,
            d._embed_candidates,
            d._pair_decision,
            d._dino_similarity,
        ) = originals

    base2 = [
        {
            "image_id": "img1",
            "detection_id": "X",
            "class_name": "chair",
            "confidence": 0.5,
            "bbox": [0, 0, 10, 10],
            "path": "x1.jpg",
        },
        {
            "image_id": "img1",
            "detection_id": "X",
            "class_name": "chair",
            "confidence": 0.9,
            "bbox": [0, 0, 10, 10],
            "path": "x2.jpg",
        },
    ]

    originals = (d._collect_candidate_pairs, d._embed_candidates)
    try:
        d._collect_candidate_pairs = lambda *a, **kw: ([], set())
        d._embed_candidates = lambda *a, **kw: ({}, None)
        out, report = d.deduplicate(
            base2,
            SimpleNamespace(person_labels={"person"}),
            policy_name="balanced",
            policy=d.POLICIES["balanced"],
            dino_batch_size=None,
            dino_outer_batch_size=1,
            dino_device=None,
        )
        require(len(out) == 1, "exact ID hard dedup failed")
        require(float(out[0]["confidence"]) == 0.9, "highest confidence survivor not kept")
        require(report["removed_count"] == report["exact_detection_id_removed"] == 1, "hard count wrong")
        require(int(out[0].get("hard_dedup_count", 0)) == 1, "hard_dedup_count missing")
    finally:
        d._collect_candidate_pairs, d._embed_candidates = originals

    ok("dedup_objects Hybrid-C hard/soft separation")


def check_manual_review_core() -> None:
    import apply_manual_review as am

    source = {
        "crops": [
            {
                "image_id": "img1",
                "detection_id": "A",
                "class_name": "chair",
                "confidence": 0.9,
                "bbox": [0, 0, 10, 10],
                "path": "a.jpg",
            },
            {
                "image_id": "img1",
                "detection_id": "B",
                "class_name": "chair",
                "confidence": 0.8,
                "bbox": [0, 0, 10, 10],
                "path": "b.jpg",
            },
            {
                "image_id": "img1",
                "detection_id": "C",
                "class_name": "table",
                "confidence": 0.7,
                "bbox": [1, 1, 11, 11],
                "path": "c.jpg",
            },
        ]
    }

    p1 = {
        "image_id": "img1",
        "a": {"detection_id": "A", "class_name": "chair"},
        "b": {"detection_id": "B", "class_name": "chair"},
    }
    p2 = {
        "image_id": "img1",
        "a": {"detection_id": "B", "class_name": "chair"},
        "b": {"detection_id": "C", "class_name": "table"},
    }
    pid1 = am._pair_id(p1)
    pid2 = am._pair_id(p2)

    report = {
        "mode": "hybrid_c_hard_exact_soft_semantic",
        "deduped_crop_count": 3,
        "removed_count": 0,
        "exact_detection_id_removed": 0,
        "same_class_removed": 0,
        "cross_class_removed": 0,
        "ambiguous_pairs": [p1, p2],
    }
    review = {
        "format_version": 3,
        "dedup_mode": "hybrid_c_hard_exact_soft_semantic",
        "review_semantics": {
            "duplicate": "preserve_both_and_assign_duplicate_group_id",
            "separate": "preserve_both_as_independent_points",
            "keep_ambiguous": "preserve_both_and_assign_ambiguous_group_id",
            "semantic_point_deletion": False,
        },
        "source_report_sha256": "synthetic-test-report",
        "total_ambiguous_pairs": 2,
        "pending_count": 0,
        "reviews": {
            pid1: {
                "decision": "duplicate",
                "image_id": "img1",
                "a_detection_id": "A",
                "b_detection_id": "B",
            },
            pid2: {
                "decision": "keep_ambiguous",
                "image_id": "img1",
                "a_detection_id": "B",
                "b_detection_id": "C",
            },
        },
    }

    out, report_out = am.apply_manual_review(
        source,
        report,
        review,
        person_labels={"person"},
    )
    require(len(out["crops"]) == 3, "manual review deleted points")
    require(out["manual_review"]["manual_removed_point_count"] == 0, "manual removed count nonzero")

    by = {x["detection_id"]: x for x in out["crops"]}
    require(
        by["A"]["duplicate_group_id"] == by["B"]["duplicate_group_id"],
        "manual duplicate grouping missing",
    )
    require(
        by["B"]["ambiguous_group_id"] == by["C"]["ambiguous_group_id"],
        "manual ambiguous grouping missing",
    )
    require(by["B"]["duplicate_status"] == "grouped_ambiguous", "combined status wrong")
    require(report_out["manual_removed_point_count"] == 0, "apply report destructive")
    ok("apply_manual_review non-destructive semantics")


def check_metadata_contracts() -> None:
    adapter = (ROOT / "rfdetr_adapter.py").read_text(encoding="utf-8")
    audit = (ROOT / "audit_db.py").read_text(encoding="utf-8")
    review = (ROOT / "review_ambiguous.py").read_text(encoding="utf-8")

    payload_keys = [
        "duplicate_status",
        "duplicate_group_id",
        "ambiguous_group_id",
        "hard_dedup_count",
        "class_conflict",
        "class_candidates",
        "merged_detection_ids",
    ]
    for key in payload_keys:
        require(f'"{key}"' in adapter, f"rfdetr_adapter missing passthrough: {key}")
        require(f'"{key}"' in audit, f"audit_db missing key: {key}")

    require(
        "hybrid_c_hard_exact_soft_semantic" in review,
        "review_ambiguous Hybrid-C mode gate missing",
    )
    require(
        '"semantic_point_deletion": False' in review,
        "review_ambiguous non-destructive semantics metadata missing",
    )
    require(
        '"format_version": 3' in review,
        "review_ambiguous provenance format_version missing",
    )
    require(
        '"source_report_sha256"' in review,
        "review_ambiguous source report hash missing",
    )
    ok("adapter/audit/reviewer metadata contract")


def main() -> int:
    print("=" * 72)
    print("HYBRID-C INTEGRITY VERIFICATION")
    print("=" * 72)
    check_files()
    compile_all()
    check_config()
    check_shared_search_helper()
    check_duplicate_grouping_runtime()
    check_dedup_objects_core()
    check_manual_review_core()
    check_metadata_contracts()
    print("=" * 72)
    print("PASS - Hybrid-C static + synthetic verification completed.")
    print("NEXT - run audit_db.py against the real Qdrant DB for end-to-end verification.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
