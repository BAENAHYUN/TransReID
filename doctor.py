"""
doctor.py — 파이프라인 자동 점검
================================

코드만 봐서는 알 수 없고 **실행해야만 드러나는 것들**을 한 번에 확인한다.
검색 성능이 이상할 때 원인 후보를 좁히는 데 쓴다.

    python doctor.py                      # 전체 점검
    python doctor.py --skip-qwen          # Qwen 제외 (빠름)
    python doctor.py --only config,qdrant # 특정 항목만
    python doctor.py --only qwen --same-pair IMG_A IMG_B --diff-pair IMG_C IMG_D

점검 항목
--------
config    설정 로드, scope/dim, 메모리 추정
qdrant    서버 연결, 컬렉션 스키마, point 수, is_person 분포
payload   crop_path 존재 여부 + 이 PC 에서 열리는지  <- 가장 중요
fusion    weighted RRF / oversampling 이 실제 적용되는지 (폴백 warning 감지)
search    단일 검색 vs 융합 검색 실제 동작
translate 번역 모델 로드 + 한국어 -> 영어
qwen      Qwen3-VL Embedding 로드 + pair cosine 분리 확인  <- 가장 중요

무엇을 왜 보는가
--------------
* payload : crop_path 가 없거나 이 PC 에서 안 열리면 Qwen 재순위/검증이
            조용히 전부 건너뛰어진다. 에러가 안 나므로 놓치기 쉽다.
* fusion  : qdrant-client 버전에 따라 weighted RRF / oversampling 이
            무시되고 균등 RRF 로 폴백된다. pipeline.yaml 의 weight 설정이
            아무 효과가 없는 상태일 수 있다.
* qwen    : 현재 구현은 Qwen3-VL-Embedding 이미지 임베딩의 cosine similarity 로
            재순위/연관성 검증을 한다. 같은/연관 pair 와 다른/비연관 pair 를 주면
            실제 임베딩이 두 그룹을 분리하는지 확인할 수 있다.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
RFDETR_ROOT = ROOT / "src" / "RF-DETR"

# 프로젝트 루트와 src/RF-DETR 양쪽의 모듈을 직접 import 할 수 있게 한다.
# (RF-DETR 폴더명에 하이픈이 있어 package import 대신 sys.path 등록 사용)
for module_root in (ROOT, RFDETR_ROOT):
    module_root_str = str(module_root)
    if module_root_str not in sys.path:
        sys.path.insert(0, module_root_str)

ALL_CHECKS = ("config", "qdrant", "payload", "fusion", "search", "translate", "qwen")

OK = "OK  "
WARN = "주의"
FAIL = "실패"
SKIP = "생략"


class Report:
    def __init__(self) -> None:
        self.rows: List[Tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        prefix = {OK: "  [OK]  ", WARN: "  [주의] ", FAIL: "  [실패] ", SKIP: "  [생략] "}
        print(f"{prefix[status]}{name}" + (f" — {detail}" if detail else ""))

    def summary(self) -> int:
        n_fail = sum(1 for s, _, _ in self.rows if s == FAIL)
        n_warn = sum(1 for s, _, _ in self.rows if s == WARN)
        n_ok = sum(1 for s, _, _ in self.rows if s == OK)

        print()
        print("=" * 72)
        print(f"  OK {n_ok} / 주의 {n_warn} / 실패 {n_fail}")

        if n_fail:
            print()
            print("  실패 항목:")
            for s, name, detail in self.rows:
                if s == FAIL:
                    print(f"    - {name}: {detail}")
        if n_warn:
            print()
            print("  주의 항목 (동작은 하지만 확인 필요):")
            for s, name, detail in self.rows:
                if s == WARN:
                    print(f"    - {name}: {detail}")
        print("=" * 72)
        return 1 if n_fail else 0


def section(title: str) -> None:
    print()
    print(f"── {title} " + "─" * max(0, 68 - len(title)))


# ───────────────────────────────────────────────────────────────────────────
def check_config(rep: Report, config_path: str) -> Optional[Any]:
    section("config")
    try:
        from config import PipelineConfig
        cfg = PipelineConfig.load(config_path)
    except Exception as e:
        rep.add(FAIL, "설정 로드", f"{type(e).__name__}: {e}")
        return None

    rep.add(OK, "설정 로드", config_path)

    table = {"all": None, "person": True, "object": False}
    for name, spec in cfg.retrievers.items():
        po = table.get(spec.scope, "?")
        rep.add(OK, f"retriever {name}",
                f"dim={spec.dim} scope={spec.scope} -> is_person={po} w={spec.weight}")

    # 파일 존재 확인
    for name, spec in cfg.retrievers.items():
        for key, value in (spec.params or {}).items():
            if not isinstance(value, str):
                continue
            if not any(value.lower().endswith(ext)
                       for ext in (".pth", ".pt", ".yaml", ".yml", ".ckpt")):
                continue
            if Path(value).is_file():
                rep.add(OK, f"{name}.{key}", value)
            else:
                rep.add(FAIL, f"{name}.{key}", f"파일 없음: {value}")

    try:
        mem = cfg.estimate_memory(3_000_000, 7_000_000)
        rep.add(OK, "1000만 규모 추정", str(mem))
    except Exception as e:
        rep.add(WARN, "메모리 추정", f"{type(e).__name__}: {e}")

    if str(cfg.fusion.method).lower() == "dbsf":
        rep.add(WARN, "fusion.method=dbsf",
                "미검증. 품질 차가 큰 벡터를 섞으면 약한 쪽이 결과를 끌어내릴 수 "
                "있다. rrf 와 비교할 것")

    return cfg


def check_qdrant(rep: Report, cfg: Any) -> Optional[Any]:
    section("qdrant")
    try:
        from qdrant_store import QdrantStore
        store = QdrantStore(cfg)
    except Exception as e:
        rep.add(FAIL, "QdrantStore 생성", f"{type(e).__name__}: {e}")
        return None

    try:
        cols = store.client.get_collections()
        names = [c.name for c in cols.collections]
        rep.add(OK, "서버 연결", f"{cfg.qdrant.url} / 컬렉션 {len(names)}개")
    except Exception as e:
        rep.add(FAIL, "서버 연결",
                f"{cfg.qdrant.url} — {type(e).__name__}: {e}. "
                f"Qdrant 가 떠 있는지 확인하세요")
        return None

    if cfg.collection not in names:
        rep.add(FAIL, f"컬렉션 '{cfg.collection}'",
                f"없음. 존재하는 것: {names}. ensure_collection() 먼저 실행")
        return store

    buf = io.StringIO()
    try:
        with redirect_stderr(buf):
            store.ensure_collection()
        rep.add(OK, "스키마 일치", cfg.collection)
    except Exception as e:
        rep.add(FAIL, "스키마 검증", f"{type(e).__name__}: {e}")

    for line in buf.getvalue().splitlines():
        if "검증 불가" in line or "불일치" in line:
            rep.add(WARN, "스키마 경고", line.strip()[-110:])

    try:
        total = store.client.count(cfg.collection).count
        if total == 0:
            rep.add(FAIL, "point 수", "0건. 적재를 먼저 하세요")
            return store
        rep.add(OK, "point 수", f"{total:,}건")
    except Exception as e:
        rep.add(WARN, "point 수", f"{type(e).__name__}: {e}")
        return store

    try:
        from qdrant_client import models as qm
        counts = {}
        for flag in (True, False):
            counts[flag] = store.client.count(
                cfg.collection,
                count_filter=qm.Filter(must=[qm.FieldCondition(
                    key="is_person", match=qm.MatchValue(value=flag))]),
            ).count

        detail = f"인물 {counts[True]:,} / 객체 {counts[False]:,}"
        if counts[True] == 0:
            rep.add(FAIL, "is_person 분포",
                    detail + " — 인물이 0건이면 IRRA/SOLIDER 벡터가 하나도 "
                             "없다. person_labels 불일치를 확인하세요")
        elif counts[False] == 0:
            rep.add(WARN, "is_person 분포",
                    detail + " — 객체가 0건. DINOv2 가 쓰이지 않는다")
        else:
            rep.add(OK, "is_person 분포", detail)
    except Exception as e:
        rep.add(WARN, "is_person 분포", f"{type(e).__name__}: {e}")

    return store


def check_payload(rep: Report, cfg: Any, store: Any, crop_root: Optional[str]) -> None:
    section("payload / crop_path")
    try:
        points, _ = store.client.scroll(
            cfg.collection, limit=50, with_payload=True, with_vectors=False
        )
    except Exception as e:
        rep.add(FAIL, "scroll", f"{type(e).__name__}: {e}")
        return

    if not points:
        rep.add(FAIL, "샘플 조회", "point 가 없습니다")
        return

    keys = sorted({k for p in points for k in (p.payload or {})})
    rep.add(OK, "payload 키", ", ".join(keys))

    for required in ("image_id", "label", "is_person", "bbox"):
        if required not in keys:
            rep.add(FAIL, f"payload.{required}", "없음")

    if "detection_id" not in keys:
        rep.add(WARN, "payload.detection_id",
                "없음 — bbox 기반 fallback ID 를 쓴 것. 재적재 시 중복 위험")

    path_key = next((k for k in ("crop_path", "path", "image_path") if k in keys), None)
    if path_key is None:
        rep.add(FAIL, "crop 경로",
                "payload 에 crop_path 가 없다. Qwen 재순위/검증을 쓸 수 없다. "
                "적재 시 keep_crop_path=True 였는지 확인")
        return

    rep.add(OK, "crop 경로 키", path_key)

    from search import SearchEngine

    engine = SearchEngine.__new__(SearchEngine)
    engine.project_root = ROOT
    engine.crop_root = Path(crop_root or (ROOT / "data" / "crops")).resolve()
    engine._path_cache = {}
    engine._path_remap_warned = True   # 여기서는 warning 대신 직접 집계

    direct = remapped = missing = 0
    example_missing = None

    for p in points:
        raw = (p.payload or {}).get(path_key)
        if not raw:
            missing += 1
            continue
        if Path(str(raw)).is_file():
            direct += 1
            continue
        resolved = engine._resolve_crop_path(str(raw))
        if Path(resolved).is_file():
            remapped += 1
        else:
            missing += 1
            if example_missing is None:
                example_missing = str(raw)

    n = len(points)
    detail = f"{n}건 중 그대로 열림 {direct} / 재해석 성공 {remapped} / 실패 {missing}"

    if missing == n:
        rep.add(FAIL, "crop 파일 접근", detail
                + f"\n            예: {example_missing}"
                + f"\n            crop_root={engine.crop_root}"
                + "\n            --crop-root 로 올바른 경로를 지정하세요")
    elif missing:
        rep.add(WARN, "crop 파일 접근", detail
                + f" (예: {example_missing})")
    elif remapped:
        rep.add(WARN, "crop 파일 접근", detail
                + " — 다른 PC 에서 적재한 DB. 재해석으로 동작하지만 "
                  "crop_root 를 명시하는 편이 안전")
    else:
        rep.add(OK, "crop 파일 접근", detail)


def check_fusion(rep: Report, cfg: Any, store: Any) -> None:
    section("fusion / 양자화 설정 적용 여부")
    try:
        from qdrant_client import models
    except Exception as e:
        rep.add(FAIL, "qdrant_client import", str(e))
        return

    # weighted RRF 지원 확인
    from qdrant_store import _model_field_names

    rrf_cls = getattr(models, "Rrf", None)
    rrf_q = getattr(models, "RrfQuery", None)
    fields = _model_field_names(rrf_cls)

    if rrf_cls is None or rrf_q is None:
        rep.add(WARN, "weighted RRF",
                "클라이언트가 Rrf/RrfQuery 를 지원하지 않음 -> 균등 RRF 폴백. "
                "pipeline.yaml 의 weight 가 무시된다")
    elif "weights" not in fields:
        rep.add(WARN, "weighted RRF",
                "Rrf 에 weights 필드가 없음 -> 균등 RRF 폴백. "
                "pipeline.yaml 의 weight 가 무시된다")
    else:
        weights = {n: s.weight for n, s in cfg.retrievers.items()}
        rep.add(OK, "weighted RRF", f"적용됨 {weights}")

    # oversampling / rescore
    qsp = getattr(models, "QuantizationSearchParams", None)
    if qsp is None:
        rep.add(WARN, "oversampling/rescore",
                "QuantizationSearchParams 미지원 -> 적용되지 않는다")
    else:
        supported = _model_field_names(qsp)
        miss = [f for f in ("rescore", "oversampling") if supported and f not in supported]
        if miss:
            rep.add(WARN, "oversampling/rescore", f"미지원 필드: {miss}")
        else:
            for name in cfg.retrievers:
                q = cfg.qdrant.quant_for(name)
                rep.add(OK, f"{name} 양자화",
                        f"type={q.type} rescore={q.rescore} os={q.oversampling}")

    if str(cfg.fusion.method).lower() == "dbsf":
        rep.add(OK, "fusion.method", "dbsf (weight 는 DBSF 에서 쓰이지 않는다)")


def check_search(rep: Report, cfg: Any, config_path: str,
                 crop_root: Optional[str]) -> None:
    section("search (단일 / 융합)")
    try:
        from search import SearchEngine
        engine = SearchEngine.from_config(
            config_path, project_root=ROOT, crop_root=crop_root
        )
    except Exception as e:
        rep.add(FAIL, "SearchEngine 생성", f"{type(e).__name__}: {e}")
        return

    query = "a person walking"

    try:
        t0 = time.time()
        hits = engine.search_text(query, limit=5, rerank=False, verify=False)
        dt = time.time() - t0
        if not hits:
            rep.add(WARN, "융합 검색", f"결과 0건 ({dt:.2f}s). 데이터가 있는지 확인")
        else:
            rep.add(OK, "융합 검색",
                    f"{len(hits)}건 ({dt:.2f}s) top score={hits[0].score:.4f}")
    except Exception as e:
        rep.add(FAIL, "융합 검색", f"{type(e).__name__}: {e}")
        return

    # 단일 검색 (IRRA 만) — 텍스트 인코더 경로 확인
    try:
        t0 = time.time()
        hits1 = engine.search_text(query, limit=5, names=["irra"], rerank=False)
        dt = time.time() - t0
        rep.add(OK, "단일 검색 (irra)",
                f"{len(hits1)}건 ({dt:.2f}s)")
    except Exception as e:
        rep.add(WARN, "단일 검색 (irra)", f"{type(e).__name__}: {e}")


def check_translate(rep: Report, backend: str, model_id: Optional[str]) -> None:
    section("translate")
    if backend == "none":
        rep.add(SKIP, "번역", "backend=none")
        return

    try:
        from query_translate import QueryTranslator, has_hangul
        tr = QueryTranslator(backend=backend, model_id=model_id)
    except Exception as e:
        rep.add(FAIL, "QueryTranslator 생성", f"{type(e).__name__}: {e}")
        return

    samples = ["빨간 재킷을 입은 남성", "검은 배낭을 멘 사람"]

    try:
        t0 = time.time()
        first = tr.translate(samples[0])
        dt = time.time() - t0
        rep.add(OK, "번역 모델 로드", f"{tr.model_id} ({dt:.1f}s)")
    except Exception as e:
        rep.add(FAIL, "번역 모델 로드",
                f"{tr.model_id} — {type(e).__name__}: {e}. "
                f"모델 이름을 확인하고 sentencepiece 설치 여부를 보세요")
        return

    for ko in samples:
        en = tr.translate(ko)
        if has_hangul(en):
            rep.add(FAIL, "번역 결과", f"{ko!r} -> {en!r} (한글이 남아 있음)")
        elif not en.strip():
            rep.add(FAIL, "번역 결과", f"{ko!r} -> 빈 문자열")
        else:
            rep.add(OK, "번역", f"{ko!r} -> {en!r}")

    passthrough = tr.translate("a man in a red jacket")
    if passthrough == "a man in a red jacket":
        rep.add(OK, "영어 통과", "한글 없으면 번역하지 않음")
    else:
        rep.add(WARN, "영어 통과", f"영어가 변형됨: {passthrough!r}")


def check_qwen(
    rep: Report,
    model_id: Optional[str],
    same_pair: Optional[List[str]],
    diff_pair: Optional[List[str]],
) -> None:
    section("qwen (embedding 재순위 / 검증)")

    try:
        from qwen_vlm import (
            EMBEDDING_MODEL_NAME,
            load_embedding_model,
            get_qwen_image_embedding,
        )
        from qwen_reranker import cosine_similarity
        rep.add(OK, "Qwen 모듈 import", "qwen_vlm + qwen_reranker")
    except Exception as e:
        rep.add(
            FAIL,
            "Qwen 모듈 import",
            f"{type(e).__name__}: {e}. "
            f"{RFDETR_ROOT / 'qwen_vlm.py'} 와 "
            f"{RFDETR_ROOT / 'qwen_reranker.py'} 를 확인하세요",
        )
        return

    actual_model_id = EMBEDDING_MODEL_NAME

    if model_id and model_id != actual_model_id:
        rep.add(
            WARN,
            "model-id",
            f"--qwen-model-id={model_id!r} 는 현재 embedding loader에 적용되지 않습니다. "
            f"실제 사용 모델: {actual_model_id}",
        )
    elif model_id == actual_model_id:
        rep.add(OK, "model-id", f"실제 embedding 모델과 일치: {actual_model_id}")

    try:
        t0 = time.time()
        qwen = load_embedding_model()
        rep.add(
            OK,
            "Qwen embedding 모델 로드",
            f"{actual_model_id} ({time.time() - t0:.1f}s)",
        )
    except Exception as e:
        rep.add(
            FAIL,
            "Qwen embedding 모델 로드",
            f"{actual_model_id} — {type(e).__name__}: {e}",
        )
        return

    if not same_pair and not diff_pair:
        rep.add(
            SKIP,
            "embedding 유사도 검증",
            "--same-pair / --diff-pair 로 이미지 쌍을 주면 cosine 분리를 확인합니다.",
        )
        try:
            del qwen
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return

    scores: Dict[str, float] = {}

    def _validate_pair(label: str, pair: Optional[List[str]]) -> bool:
        if not pair:
            return False
        missing = [p for p in pair if not Path(p).is_file()]
        if missing:
            rep.add(
                FAIL,
                f"{label} 이미지 경로",
                "파일 없음: " + ", ".join(missing),
            )
            return False
        return True

    if _validate_pair("same/related pair", same_pair):
        try:
            t0 = time.time()
            emb_a = get_qwen_image_embedding(qwen, same_pair[0])
            emb_b = get_qwen_image_embedding(qwen, same_pair[1])
            score = cosine_similarity(emb_a, emb_b)
            scores["same"] = score
            rep.add(
                OK,
                "same/related pair cosine",
                f"{score:.4f} ({time.time() - t0:.2f}s)",
            )
        except Exception as e:
            rep.add(
                FAIL,
                "same/related pair embedding",
                f"{type(e).__name__}: {e}",
            )

    if _validate_pair("diff/unrelated pair", diff_pair):
        try:
            t0 = time.time()
            emb_a = get_qwen_image_embedding(qwen, diff_pair[0])
            emb_b = get_qwen_image_embedding(qwen, diff_pair[1])
            score = cosine_similarity(emb_a, emb_b)
            scores["diff"] = score
            rep.add(
                OK,
                "diff/unrelated pair cosine",
                f"{score:.4f} ({time.time() - t0:.2f}s)",
            )
        except Exception as e:
            rep.add(
                FAIL,
                "diff/unrelated pair embedding",
                f"{type(e).__name__}: {e}",
            )

    if "same" in scores and "diff" in scores:
        gap = scores["same"] - scores["diff"]

        if gap > 0.30:
            rep.add(
                OK,
                "Qwen embedding 분리",
                f"same/related {scores['same']:.4f} > "
                f"diff/unrelated {scores['diff']:.4f} "
                f"(차이 {gap:.4f})",
            )
        elif gap > 0.05:
            rep.add(
                WARN,
                "Qwen embedding 분리",
                f"방향은 맞지만 차이가 작음: "
                f"{scores['same']:.4f} > {scores['diff']:.4f} "
                f"(차이 {gap:.4f}). 여러 pair 로 추가 검증하세요.",
            )
        else:
            rep.add(
                WARN,
                "Qwen embedding 분리",
                f"분리 불충분: same/related {scores['same']:.4f}, "
                f"diff/unrelated {scores['diff']:.4f} "
                f"(차이 {gap:.4f}). pair 구성과 모델 적합성을 확인하세요.",
            )

        midpoint = (scores["same"] + scores["diff"]) / 2.0
        rep.add(
            WARN,
            "threshold 참고",
            f"이 두 pair의 중간값은 {midpoint:.4f}. "
            "실제 threshold는 수십~수백 pair로 캘리브레이션하세요.",
        )

    try:
        del qwen
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ───────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="파이프라인 자동 점검")
    ap.add_argument("--config", default=str(ROOT / "pipeline.yaml"))
    ap.add_argument("--crop-root", default=None)
    ap.add_argument("--only", default=None,
                    help=f"쉼표로 구분. 가능: {','.join(ALL_CHECKS)}")
    ap.add_argument("--skip-qwen", action="store_true")
    ap.add_argument("--skip-translate", action="store_true")
    ap.add_argument("--qwen-model-id", default=None,
                    help="기대 모델 ID 확인용. 현재 qwen_vlm.py는 EMBEDDING_MODEL_NAME을 직접 사용")
    ap.add_argument("--translate-backend", default="opus")
    ap.add_argument("--translate-model-id", default=None)
    ap.add_argument("--same-pair", nargs=2, default=None,
                    metavar=("IMG_A", "IMG_B"),
                    help="같은/연관 pair 이미지 두 장 (예: 동일 인물 2장 또는 사람↔해당 소지품)")
    ap.add_argument("--diff-pair", nargs=2, default=None,
                    metavar=("IMG_A", "IMG_B"),
                    help="다른/비연관 pair 이미지 두 장")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.only:
        wanted = {c.strip() for c in args.only.split(",") if c.strip()}
        bad = wanted - set(ALL_CHECKS)
        if bad:
            ap.error(f"알 수 없는 점검 항목: {sorted(bad)}")
    else:
        wanted = set(ALL_CHECKS)
        if args.skip_qwen:
            wanted.discard("qwen")
        if args.skip_translate:
            wanted.discard("translate")

    print()
    print("=" * 72)
    print(f"  파이프라인 점검 — {ROOT}")
    print(f"  항목: {', '.join(c for c in ALL_CHECKS if c in wanted)}")
    print("=" * 72)

    rep = Report()

    cfg = None
    if "config" in wanted:
        cfg = check_config(rep, args.config)
        if cfg is None:
            rep.summary()
            return 1
    else:
        from config import PipelineConfig
        cfg = PipelineConfig.load(args.config)

    store = None
    if "qdrant" in wanted:
        store = check_qdrant(rep, cfg)

    if "payload" in wanted:
        if store is None:
            from qdrant_store import QdrantStore
            try:
                store = QdrantStore(cfg)
            except Exception as e:
                rep.add(FAIL, "QdrantStore", str(e))
        if store is not None:
            try:
                check_payload(rep, cfg, store, args.crop_root)
            except Exception as e:
                rep.add(FAIL, "payload 점검", f"{type(e).__name__}: {e}")

    if "fusion" in wanted and store is not None:
        try:
            check_fusion(rep, cfg, store)
        except Exception as e:
            rep.add(FAIL, "fusion 점검", f"{type(e).__name__}: {e}")

    if "search" in wanted:
        try:
            check_search(rep, cfg, args.config, args.crop_root)
        except Exception as e:
            rep.add(FAIL, "search 점검", f"{type(e).__name__}: {e}")

    if "translate" in wanted:
        try:
            check_translate(rep, args.translate_backend, args.translate_model_id)
        except Exception as e:
            rep.add(FAIL, "translate 점검", f"{type(e).__name__}: {e}")

    if "qwen" in wanted:
        try:
            check_qwen(rep, args.qwen_model_id, args.same_pair, args.diff_pair)
        except Exception as e:
            rep.add(FAIL, "qwen 점검", f"{type(e).__name__}: {e}")

    return rep.summary()


if __name__ == "__main__":
    raise SystemExit(main())