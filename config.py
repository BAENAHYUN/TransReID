"""
pipeline.yaml 로더 + 검증.

구축 스크립트와 검색 스크립트가 이 모듈을 통해 같은 설정을 읽는다.
잘못된 설정은 파이프라인이 반쯤 돌다가 죽는 대신 로드 시점에 바로 실패한다.

이번 개정
--------
  6) Hybrid-C duplicate_grouping 설정 추가.
     pipeline.yaml 의 duplicate_grouping 을 검색 중복 collapse 정책의 SSOT로 사용한다.
     enabled / overfetch_factor / max_fetch / payload key를 로드 시점에 검증한다.

  5) collection_prefix 추가 + person_collection() / object_collection().

     기존에는 pipeline.yaml 에 collection: person_db 가 적혀 있는데
     build_db.py 가 "forensic_person" / "forensic_object" 를 하드코딩해
     덮어쓰고 있었다. 그래서 DB 구축 대상 이름의 SSOT 가 yaml 이 아니라
     Python 코드였고, cfg.collection 을 읽는 관리 스크립트
     (delete_query_crops.py 등)는 존재하지 않는 컬렉션을 가리켰다.

     이제 역할을 나눈다.
       collection_prefix : 구축 구조의 SSOT. person/object 두 이름을 파생.
       collection        : QdrantStore(cfg) 가 직접 읽는 단일 컬렉션 대상.

     접미어 "_person" / "_object" 는 이 모듈에만 존재한다. yaml 에 두 이름을
     따로 적으면 한쪽 오타로 세 번째 컬렉션이 조용히 만들어질 수 있다.

이전 개정
--------
  1) QuantSpec 에 rescore / oversampling 추가.
     qdrant_store 가 이미 이 두 값을 읽는데 dataclass 에 필드가 없어서,
     overrides 에 적으면 QuantSpec(**ov) 가 TypeError 로 죽었다.
  2) overrides 검증.
     - 알 수 없는 retriever 이름 -> ValueError (오타를 조용히 무시하지 않는다)
     - 알 수 없는 quantization / hnsw 키 -> ValueError
     - override 의 quantization.type 도 VALID_QUANT 검사
  3) fusion.prefetch_limit < limit 을 로드 시점에 차단.
     예전에는 첫 검색 때까지 미뤄졌다.
  4) distance 검증 + distance_for(name) 추가.
     qdrant_store 가 vector 별 distance 를 지원하므로 훅만 열어 둔다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

VALID_SCOPES = {"all", "person", "object"}
VALID_FUSION = {"dbsf", "rrf"}
VALID_QUANT = {"scalar", "binary", "none"}
VALID_TRANSLATION_BACKENDS = {"opus", "nllb", "none"}

# Qdrant 가 지원하는 distance. 대소문자는 무시하고 비교한다.
VALID_DISTANCE = {"cosine", "dot", "euclid", "manhattan"}

# pipeline yaml 최상위에서 허용하는 키. 이 로더가 읽는 키 + 다른 스크립트가 읽는
# 선택 섹션(detector/tracker/stitcher: detect/runner, clustering: report_common,
# fingerprint: build_db). 여기 없는 키는 from_dict 가 오타로 간주해 거부한다.
TOP_LEVEL_KEYS = frozenset({
    "retrievers", "collection", "collection_prefix", "person_labels",
    "fusion", "qdrant", "duplicate_grouping", "query", "verifiers",
    "detector", "tracker", "stitcher",
    "clustering", "fingerprint",
})

# score 가 작을수록 가까운 distance. threshold 방향이 반대가 된다.
SMALLER_IS_BETTER_DISTANCE = {"euclid", "manhattan"}


def _check_keys(kind: str, where: str, data: Dict[str, Any], spec_cls) -> None:
    """dataclass 가 받지 않는 키가 섞여 있으면 로드 시점에 막는다."""
    allowed = {f.name for f in dataclass_fields(spec_cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(
            f"{where}: 알 수 없는 {kind} 항목 {unknown}\n"
            f"  허용값={sorted(allowed)}"
        )


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RetrieverSpec:
    """ANN 인덱스에 올라가는 임베더 하나."""
    name: str
    tool: str
    scope: str
    dim: int
    weight: float
    module: str
    class_name: str
    supports_text: bool
    params: Dict[str, Any] = field(default_factory=dict)

    def accepts_person(self) -> bool:
        return self.scope in ("all", "person")

    def accepts_object(self) -> bool:
        return self.scope in ("all", "object")


@dataclass(frozen=True)
class FusionSpec:
    method: str = "dbsf"
    prefetch_limit: int = 100
    limit: int = 20


@dataclass(frozen=True)
class DuplicateGroupingSpec:
    """Hybrid-C 검색 결과 duplicate-group collapse 설정.

    semantic duplicate는 DB point를 삭제하지 않고 duplicate_group_id로 보존한다.
    검색 시에만 같은 duplicate_group_id 결과를 하나로 collapse한다.
    ambiguous_group_id는 collapse key로 사용하지 않는다.
    """

    enabled: bool = True
    overfetch_factor: int = 3
    max_fetch: int = 200
    group_payload_key: str = "duplicate_group_id"
    ambiguous_payload_key: str = "ambiguous_group_id"


@dataclass(frozen=True)
class TranslationSpec:
    """자연어 query 번역 설정."""
    enabled: bool = True
    backend: str = "opus"
    model_id: Optional[str] = None
    max_new_tokens: int = 64
    cache_size: int = 2048


@dataclass(frozen=True)
class QuerySpec:
    """자연어 query 전처리 설정."""
    translation: TranslationSpec = field(default_factory=TranslationSpec)
    expand: bool = False


@dataclass(frozen=True)
class QuantSpec:
    """
    양자화 설정.

    rescore / oversampling 은 검색 시점 설정이라 컬렉션 스키마에는 들어가지
    않지만, qdrant_store._quant_search_params 가 벡터별로 읽는다.
    저차원 벡터(IRRA 512-d)는 고차원보다 INT8 손실이 상대적으로 크므로
    recall 이 아쉬우면 oversampling 을 올린다.
    """
    type: str = "scalar"
    always_ram: bool = True
    quantile: float = 0.99
    rescore: bool = True
    oversampling: float = 2.0


@dataclass(frozen=True)
class HnswSpec:
    m: int = 16
    ef_construct: int = 100


@dataclass(frozen=True)
class QdrantSpec:
    url: str = "http://localhost:6333"
    distance: str = "Cosine"
    on_disk: bool = True
    quantization: QuantSpec = field(default_factory=QuantSpec)
    hnsw: HnswSpec = field(default_factory=HnswSpec)
    overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def quant_for(self, name: str) -> QuantSpec:
        """벡터별 양자화 설정. overrides 에 없으면 기본값."""
        ov = self.overrides.get(name, {}).get("quantization")
        if not ov:
            return self.quantization
        # 부분 override 를 허용한다. 적지 않은 키는 전역값을 물려받는다.
        merged = {
            f.name: getattr(self.quantization, f.name)
            for f in dataclass_fields(QuantSpec)
        }
        merged.update(ov)
        return QuantSpec(**merged)

    def hnsw_for(self, name: str) -> HnswSpec:
        ov = self.overrides.get(name, {}).get("hnsw")
        if not ov:
            return self.hnsw
        merged = {
            f.name: getattr(self.hnsw, f.name)
            for f in dataclass_fields(HnswSpec)
        }
        merged.update(ov)
        return HnswSpec(**merged)

    def distance_for(self, name: str) -> str:
        """
        벡터별 distance. overrides 에 없으면 전역값.

        qdrant_store 가 이 메서드를 있으면 쓰고 없으면 전역으로 폴백한다.
        DINOv2 에 PCA-whitening 을 적용하면 L2 노름이 깨져서 COSINE 과 DOT 의
        의미가 달라진다. 그때 pipeline.yaml 에 한 줄만 추가하면 된다.
        """
        return str(
            self.overrides.get(name, {}).get("distance", self.distance)
        )

    def smaller_is_better(self, name: Optional[str] = None) -> bool:
        raw = self.distance_for(name) if name else self.distance
        return str(raw).strip().lower() in SMALLER_IS_BETTER_DISTANCE


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PipelineConfig:
    collection: str
    collection_prefix: str
    person_labels: frozenset
    retrievers: Dict[str, RetrieverSpec]
    fusion: FusionSpec
    qdrant: QdrantSpec
    duplicate_grouping: DuplicateGroupingSpec = field(
        default_factory=DuplicateGroupingSpec
    )
    query: QuerySpec = field(default_factory=QuerySpec)
    verifiers: Dict[str, Any] = field(default_factory=dict)

    # ---------- 로드 ---------- #
    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"설정 파일이 없습니다: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw, base_dir=path.parent)

    @classmethod
    def from_dict(
        cls,
        raw: Dict[str, Any],
        base_dir: Optional[Path] = None,
    ) -> "PipelineConfig":
        if not isinstance(raw, dict):
            raise ValueError("설정 최상위는 매핑(dict)이어야 합니다.")

        # 최상위 키 오타 차단. 예: collection_prefx 는 무시되고 기본 'forensic' 으로
        # 연결되어 다른 컬렉션을 의도한 build 가 운영 컬렉션에 적재된다.
        # '_' 로 시작하는 키는 메타데이터로 허용한다.
        unknown_top = sorted(
            str(k) for k in raw
            if not str(k).startswith("_") and str(k) not in TOP_LEVEL_KEYS
        )
        if unknown_top:
            raise ValueError(
                f"pipeline yaml 최상위에 알 수 없는 키 {unknown_top}\n"
                f"  허용값={sorted(TOP_LEVEL_KEYS)}\n"
                "  오타라면 고치고, 새 섹션이면 config.TOP_LEVEL_KEYS 에 추가하세요."
            )

        if "retrievers" not in raw or not raw["retrievers"]:
            raise ValueError("설정에 retrievers 가 최소 하나는 있어야 합니다.")

        # ---------------- retrievers ---------------- #
        retrievers: Dict[str, RetrieverSpec] = {}

        for name, r in raw["retrievers"].items():
            for key in ("scope", "supports_text", "dim", "module", "class"):
                if key not in r:
                    raise ValueError(f"retriever '{name}': '{key}' 누락")

            if not isinstance(r["supports_text"], bool):
                raise ValueError(
                    f"retriever '{name}': supports_text 는 true/false 여야 합니다 "
                    f"(받은 값: {r['supports_text']!r})"
                )

            if r["scope"] not in VALID_SCOPES:
                raise ValueError(
                    f"retriever '{name}': scope 는 {sorted(VALID_SCOPES)} "
                    f"중 하나여야 합니다 (받은 값: {r['scope']})"
                )

            dim = r["dim"]
            if (
                not isinstance(dim, int)
                or isinstance(dim, bool)
                or dim <= 0
            ):
                raise ValueError(
                    f"retriever '{name}': dim 은 1 이상의 정수여야 합니다 "
                    f"(받은 값: {dim!r})"
                )

            weight = float(r.get("weight", 1.0))
            if weight < 0:
                raise ValueError(
                    f"retriever '{name}': weight 는 0 이상이어야 합니다 "
                    f"(받은 값: {weight})"
                )

            params = dict(r.get("params") or {})
            if base_dir is not None:
                params = _resolve_paths(params, base_dir)

            retrievers[name] = RetrieverSpec(
                name=name,
                tool=r.get("tool", "common"),
                scope=r["scope"],
                dim=int(r["dim"]),
                weight=weight,
                module=r["module"],
                class_name=r["class"],
                supports_text=r["supports_text"],
                params=params,
            )

        # scope='person' 인 retriever 가 하나도 없으면 사람 검색이 불가능하다.
        if not any(s.accepts_person() for s in retrievers.values()):
            logger.warning(
                "사람 crop 을 처리할 retriever 가 없습니다 "
                "(scope 가 'person' 또는 'all' 인 항목 없음)."
            )
        if not any(s.accepts_object() for s in retrievers.values()):
            logger.warning(
                "객체 crop 을 처리할 retriever 가 없습니다 "
                "(scope 가 'object' 또는 'all' 인 항목 없음)."
            )

        # ---------------- fusion ---------------- #
        f = raw.get("fusion") or {}
        _check_keys("fusion", "fusion", f, FusionSpec)

        method = str(f.get("method", "dbsf")).lower()
        if method not in VALID_FUSION:
            raise ValueError(
                f"fusion.method 는 {sorted(VALID_FUSION)} 중 하나여야 합니다 "
                f"(받은 값: {method})"
            )

        prefetch_limit = int(f.get("prefetch_limit", 100))
        limit = int(f.get("limit", 20))

        if limit <= 0 or prefetch_limit <= 0:
            raise ValueError(
                "fusion.limit 과 fusion.prefetch_limit 은 1 이상이어야 합니다 "
                f"(limit={limit}, prefetch_limit={prefetch_limit})"
            )

        # 개정 3 — 예전에는 첫 검색 때까지 미뤄졌다.
        if prefetch_limit < limit:
            raise ValueError(
                f"fusion.prefetch_limit({prefetch_limit}) 은 "
                f"fusion.limit({limit}) 보다 크거나 같아야 합니다. "
                f"retriever 별 후보 수가 최종 반환 수보다 적으면 "
                f"융합할 재료가 부족해집니다."
            )

        fusion = FusionSpec(
            method=method,
            prefetch_limit=prefetch_limit,
            limit=limit,
        )

        # ---------------- Hybrid-C duplicate grouping ---------------- #
        dg_raw = raw.get("duplicate_grouping") or {}
        if not isinstance(dg_raw, dict):
            raise ValueError(
                "duplicate_grouping 은 mapping이어야 합니다 "
                f"(받은 값: {dg_raw!r})"
            )

        _check_keys(
            "duplicate_grouping",
            "duplicate_grouping",
            dg_raw,
            DuplicateGroupingSpec,
        )

        dg_enabled = dg_raw.get("enabled", True)
        if not isinstance(dg_enabled, bool):
            raise ValueError(
                "duplicate_grouping.enabled 는 true/false 여야 합니다 "
                f"(받은 값: {dg_enabled!r})"
            )

        overfetch_factor = dg_raw.get("overfetch_factor", 3)
        if (
            not isinstance(overfetch_factor, int)
            or isinstance(overfetch_factor, bool)
            or overfetch_factor < 1
        ):
            raise ValueError(
                "duplicate_grouping.overfetch_factor 는 1 이상의 정수여야 합니다 "
                f"(받은 값: {overfetch_factor!r})"
            )

        max_fetch = dg_raw.get("max_fetch", 200)
        if (
            not isinstance(max_fetch, int)
            or isinstance(max_fetch, bool)
            or max_fetch < 1
        ):
            raise ValueError(
                "duplicate_grouping.max_fetch 는 1 이상의 정수여야 합니다 "
                f"(받은 값: {max_fetch!r})"
            )

        if dg_enabled and max_fetch < fusion.limit:
            raise ValueError(
                "duplicate_grouping.max_fetch 는 grouping이 활성화된 경우 "
                "fusion.limit 이상이어야 합니다 "
                f"(max_fetch={max_fetch}, fusion.limit={fusion.limit})"
            )

        group_payload_key = dg_raw.get(
            "group_payload_key",
            "duplicate_group_id",
        )
        ambiguous_payload_key = dg_raw.get(
            "ambiguous_payload_key",
            "ambiguous_group_id",
        )

        for key_name, value in (
            ("group_payload_key", group_payload_key),
            ("ambiguous_payload_key", ambiguous_payload_key),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"duplicate_grouping.{key_name} 는 비어 있지 않은 문자열이어야 "
                    f"합니다 (받은 값: {value!r})"
                )

        group_payload_key = group_payload_key.strip()
        ambiguous_payload_key = ambiguous_payload_key.strip()

        if group_payload_key == ambiguous_payload_key:
            raise ValueError(
                "duplicate_grouping.group_payload_key 와 "
                "ambiguous_payload_key 는 서로 달라야 합니다. "
                "Hybrid C에서는 duplicate group만 collapse하고 ambiguous group은 "
                "collapse하지 않습니다."
            )

        # Unified Hybrid-C payload contract is canonical across dedup, adapter,
        # audit and search. Allowing arbitrary key names here would make search
        # read a key that upstream never writes, silently disabling collapse.
        if group_payload_key != "duplicate_group_id":
            raise ValueError(
                "duplicate_grouping.group_payload_key 는 현재 통합 payload 계약상 "
                "'duplicate_group_id' 이어야 합니다 "
                f"(받은 값: {group_payload_key!r})"
            )
        if ambiguous_payload_key != "ambiguous_group_id":
            raise ValueError(
                "duplicate_grouping.ambiguous_payload_key 는 현재 통합 payload 계약상 "
                "'ambiguous_group_id' 이어야 합니다 "
                f"(받은 값: {ambiguous_payload_key!r})"
            )

        duplicate_grouping = DuplicateGroupingSpec(
            enabled=dg_enabled,
            overfetch_factor=overfetch_factor,
            max_fetch=max_fetch,
            group_payload_key=group_payload_key,
            ambiguous_payload_key=ambiguous_payload_key,
        )

        # ---------------- query / translation ---------------- #
        query_raw = raw.get("query") or {}
        if not isinstance(query_raw, dict):
            raise ValueError(
                f"query 는 mapping이어야 합니다 (받은 값: {query_raw!r})"
            )
        _check_keys("query", "query", query_raw, QuerySpec)

        translation_raw = query_raw.get("translation") or {}
        if not isinstance(translation_raw, dict):
            raise ValueError(
                "query.translation 은 mapping이어야 합니다 "
                f"(받은 값: {translation_raw!r})"
            )
        _check_keys(
            "translation",
            "query.translation",
            translation_raw,
            TranslationSpec,
        )

        enabled = translation_raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(
                "query.translation.enabled 는 true/false 여야 합니다 "
                f"(받은 값: {enabled!r})"
            )

        backend = str(
            translation_raw.get("backend", "opus")
        ).strip().lower()
        if backend not in VALID_TRANSLATION_BACKENDS:
            raise ValueError(
                "query.translation.backend 는 "
                f"{sorted(VALID_TRANSLATION_BACKENDS)} 중 하나여야 합니다 "
                f"(받은 값: {backend!r})"
            )

        model_id = translation_raw.get("model_id")
        if model_id is not None:
            if not isinstance(model_id, str) or not model_id.strip():
                raise ValueError(
                    "query.translation.model_id 는 null 또는 비어 있지 않은 "
                    f"문자열이어야 합니다 (받은 값: {model_id!r})"
                )
            model_id = model_id.strip()

        if backend == "none" and model_id is not None:
            raise ValueError(
                "query.translation.backend='none' 에서는 "
                "model_id 를 지정할 수 없습니다."
            )

        max_new_tokens = translation_raw.get("max_new_tokens", 64)
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens <= 0
        ):
            raise ValueError(
                "query.translation.max_new_tokens 는 1 이상의 정수여야 합니다 "
                f"(받은 값: {max_new_tokens!r})"
            )

        cache_size = translation_raw.get("cache_size", 2048)
        if (
            not isinstance(cache_size, int)
            or isinstance(cache_size, bool)
            or cache_size <= 0
        ):
            raise ValueError(
                "query.translation.cache_size 는 1 이상의 정수여야 합니다 "
                f"(받은 값: {cache_size!r})"
            )

        expand = query_raw.get("expand", False)
        if not isinstance(expand, bool):
            raise ValueError(
                "query.expand 는 true/false 여야 합니다 "
                f"(받은 값: {expand!r})"
            )

        query_cfg = QuerySpec(
            translation=TranslationSpec(
                enabled=enabled,
                backend=backend,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                cache_size=cache_size,
            ),
            expand=expand,
        )

        # ---------------- qdrant ---------------- #
        q = raw.get("qdrant") or {}

        distance = str(q.get("distance", "Cosine"))
        if distance.strip().lower() not in VALID_DISTANCE:
            raise ValueError(
                f"qdrant.distance 는 {sorted(VALID_DISTANCE)} 중 하나여야 "
                f"합니다 (대소문자 무시, 받은 값: {distance!r})"
            )

        quant_raw = q.get("quantization") or {}
        _check_keys("quantization", "qdrant.quantization", quant_raw, QuantSpec)
        quant = QuantSpec(**quant_raw)
        _validate_quant("qdrant.quantization", quant)

        hnsw_raw = q.get("hnsw") or {}
        _check_keys("hnsw", "qdrant.hnsw", hnsw_raw, HnswSpec)
        hnsw = HnswSpec(**hnsw_raw)
        _validate_hnsw("qdrant.hnsw", hnsw)

        overrides = q.get("overrides") or {}
        _validate_overrides(overrides, retrievers, quant, hnsw)

        qdrant = QdrantSpec(
            url=q.get("url", "http://localhost:6333"),
            distance=distance,
            on_disk=bool(q.get("on_disk", True)),
            quantization=quant,
            hnsw=hnsw,
            overrides=overrides,
        )

        return cls(
            collection=raw.get("collection", "forensic_person"),
            # 개정 5 — yaml 에서 후행 공백은 눈에 안 보이지만 Qdrant 컬렉션
            # 이름은 공백을 허용한다. "forensic " 이 "forensic _person" 이
            # 되어도 정상 동작처럼 보여서 추적이 어렵다.
            collection_prefix=str(
                raw.get("collection_prefix", "forensic")
            ).strip(),
            person_labels=frozenset(
                s.lower() for s in (raw.get("person_labels") or ["person"])
            ),
            retrievers=retrievers,
            fusion=fusion,
            qdrant=qdrant,
            duplicate_grouping=duplicate_grouping,
            query=query_cfg,
            verifiers=raw.get("verifiers") or {},
        )

    # ---------- 컬렉션 이름 ---------- #
    #
    # 접미어는 이 두 메서드에만 존재한다. build_db.py 는 컬렉션 이름을
    # 하드코딩하지 않고 여기서 받아 간다.
    #
    def person_collection(self) -> str:
        """person crop 컬렉션 (siglip2 / irra / solider)."""
        return f"{self.collection_prefix}_person"

    def object_collection(self) -> str:
        """object crop 컬렉션 (siglip2 / dinov2)."""
        return f"{self.collection_prefix}_object"

    # ---------- 조회 helper ---------- #
    def vector_config(self) -> Dict[str, int]:
        """Qdrant 컬렉션 생성용 {벡터이름: 차원}."""
        return {name: spec.dim for name, spec in self.retrievers.items()}

    def by_tool(self, tool: str) -> List[RetrieverSpec]:
        """'human tool 이 뭐로 구성돼 있나'를 코드가 아니라 설정으로 답한다."""
        return [s for s in self.retrievers.values() if s.tool == tool]

    def for_person(self) -> List[RetrieverSpec]:
        return [s for s in self.retrievers.values() if s.accepts_person()]

    def for_object(self) -> List[RetrieverSpec]:
        return [s for s in self.retrievers.values() if s.accepts_object()]

    def person_only_map(self) -> Dict[str, Optional[bool]]:
        """
        qdrant_store.fused_search(person_only=...) 에 그대로 넘길 수 있는 매핑.

        QdrantStore 도 scope 로부터 같은 값을 자동 계산하므로 보통은 넘길
        필요가 없다. 로그로 확인하거나 일부만 override 할 때 쓴다.
        """
        table = {"all": None, "person": True, "object": False}
        return {name: table[s.scope] for name, s in self.retrievers.items()}

    def describe(self) -> str:
        lines = [
            f"collection(single): {self.collection}",
            f"collection(build) : {self.person_collection()} / "
            f"{self.object_collection()}",
            "retrievers:",
        ]
        for s in self.retrievers.values():
            lines.append(
                f"  {s.name:<10} tool={s.tool:<7} scope={s.scope:<7} "
                f"dim={s.dim:<5} w={s.weight}"
            )
        lines.append(
            f"  (person crop 총 {sum(s.dim for s in self.for_person())}d, "
            f"object crop 총 {sum(s.dim for s in self.for_object())}d)"
        )
        lines.append(
            f"fusion: {self.fusion.method} "
            f"prefetch={self.fusion.prefetch_limit} limit={self.fusion.limit}"
        )
        lines.append(
            "duplicate_grouping: "
            f"enabled={self.duplicate_grouping.enabled} "
            f"overfetch={self.duplicate_grouping.overfetch_factor}x "
            f"max_fetch={self.duplicate_grouping.max_fetch} "
            f"group_key={self.duplicate_grouping.group_payload_key} "
            f"ambiguous_key={self.duplicate_grouping.ambiguous_payload_key}"
        )
        lines.append(
            f"qdrant: distance={self.qdrant.distance} "
            f"quant={self.qdrant.quantization.type} "
            f"on_disk={self.qdrant.on_disk}"
        )
        lines.append(
            "query: "
            f"translate={self.query.translation.enabled} "
            f"backend={self.query.translation.backend} "
            f"model={self.query.translation.model_id or '<backend-default>'} "
            f"max_new_tokens={self.query.translation.max_new_tokens} "
            f"cache_size={self.query.translation.cache_size} "
            f"expand={self.query.expand}"
        )
        for name in self.retrievers:
            qs = self.qdrant.quant_for(name)
            hs = self.qdrant.hnsw_for(name)
            lines.append(
                f"  {name:<10} distance={self.qdrant.distance_for(name):<8} "
                f"quant={qs.type}/os={qs.oversampling}/rescore={qs.rescore} "
                f"hnsw=m{hs.m}/ef{hs.ef_construct}"
            )
        return "\n".join(lines)

    def estimate_memory(self, n_person: int, n_object: int) -> Dict[str, float]:
        """
        대략적인 벡터 저장량(GB). 규모 산정용 — HNSW 그래프/payload 는 제외.

        벡터별로 양자화 설정이 다를 수 있으므로 retriever 단위로 계산한다.
        always_ram=True 인 양자화본만 RAM 상주분으로 집계한다.
        """
        ratio = {"scalar": 4.0, "binary": 32.0, "none": 1.0}

        raw_bytes = 0.0
        quant_bytes = 0.0
        ram_bytes = 0.0

        for s in self.retrievers.values():
            n = 0
            if s.accepts_person():
                n += n_person
            if s.accepts_object():
                n += n_object

            size = n * s.dim * 4
            qs = self.qdrant.quant_for(s.name)
            q_size = size / ratio[qs.type]

            raw_bytes += size
            quant_bytes += q_size
            if qs.always_ram and qs.type != "none":
                ram_bytes += q_size

        gb = 1024 ** 3
        return {
            "float32_gb": round(raw_bytes / gb, 2),
            "quantized_gb": round(quant_bytes / gb, 2),
            "always_ram_gb": round(ram_bytes / gb, 2),
        }


# --------------------------------------------------------------------------- #
# 검증 helper
# --------------------------------------------------------------------------- #

def _validate_quant(where: str, quant: QuantSpec) -> None:
    if quant.type not in VALID_QUANT:
        raise ValueError(
            f"{where}.type 은 {sorted(VALID_QUANT)} 중 하나여야 합니다 "
            f"(받은 값: {quant.type})"
        )
    if not 0.0 < float(quant.quantile) <= 1.0:
        raise ValueError(
            f"{where}.quantile 은 (0, 1] 범위여야 합니다 "
            f"(받은 값: {quant.quantile})"
        )
    if float(quant.oversampling) < 1.0:
        raise ValueError(
            f"{where}.oversampling 은 1.0 이상이어야 합니다 "
            f"(받은 값: {quant.oversampling})"
        )


def _validate_hnsw(where: str, hnsw: HnswSpec) -> None:
    if int(hnsw.m) <= 0:
        raise ValueError(f"{where}.m 은 양수여야 합니다 (받은 값: {hnsw.m})")
    if int(hnsw.ef_construct) <= 0:
        raise ValueError(
            f"{where}.ef_construct 는 양수여야 합니다 "
            f"(받은 값: {hnsw.ef_construct})"
        )


def _validate_overrides(
    overrides: Dict[str, Any],
    retrievers: Dict[str, RetrieverSpec],
    base_quant: QuantSpec,
    base_hnsw: HnswSpec,
) -> None:
    """
    개정 2 — 예전에는 오타난 retriever 이름이 조용히 무시됐다.

    'siglp2' 라고 적으면 override 가 전혀 적용되지 않은 채 파이프라인이
    정상으로 보이므로, 나중에 성능 차이의 원인을 찾기 매우 어렵다.
    """
    allowed_sections = {"quantization", "hnsw", "distance"}

    for name, ov in overrides.items():
        if name not in retrievers:
            raise ValueError(
                f"qdrant.overrides 에 알 수 없는 retriever '{name}' 이 "
                f"있습니다. 오타인지 확인하세요.\n"
                f"  등록된 retriever={sorted(retrievers)}"
            )

        if not isinstance(ov, dict):
            raise ValueError(
                f"qdrant.overrides['{name}'] 은 dict 여야 합니다 "
                f"(받은 타입: {type(ov).__name__})"
            )

        unknown = sorted(set(ov) - allowed_sections)
        if unknown:
            raise ValueError(
                f"qdrant.overrides['{name}'] 에 알 수 없는 항목 {unknown}\n"
                f"  허용값={sorted(allowed_sections)}"
            )

        if "quantization" in ov:
            section = ov["quantization"] or {}
            _check_keys(
                "quantization",
                f"qdrant.overrides['{name}'].quantization",
                section,
                QuantSpec,
            )
            merged = {
                f.name: getattr(base_quant, f.name)
                for f in dataclass_fields(QuantSpec)
            }
            merged.update(section)
            _validate_quant(
                f"qdrant.overrides['{name}'].quantization",
                QuantSpec(**merged),
            )

        if "hnsw" in ov:
            section = ov["hnsw"] or {}
            _check_keys(
                "hnsw",
                f"qdrant.overrides['{name}'].hnsw",
                section,
                HnswSpec,
            )
            merged = {
                f.name: getattr(base_hnsw, f.name)
                for f in dataclass_fields(HnswSpec)
            }
            merged.update(section)
            _validate_hnsw(
                f"qdrant.overrides['{name}'].hnsw",
                HnswSpec(**merged),
            )

        if "distance" in ov:
            value = str(ov["distance"]).strip().lower()
            if value not in VALID_DISTANCE:
                raise ValueError(
                    f"qdrant.overrides['{name}'].distance 는 "
                    f"{sorted(VALID_DISTANCE)} 중 하나여야 합니다 "
                    f"(받은 값: {ov['distance']!r})"
                )


def _resolve_paths(params: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    """
    params 안의 상대경로(./ 또는 ../ 로 시작)를 설정 파일 기준 절대경로로 바꾼다.

    주의: 기준점은 pipeline.yaml 이 있는 디렉터리다. 프로젝트 루트가 아니다.
    yaml 을 configs/ 같은 하위 폴더로 옮기면 './IRRA' 가 'configs/IRRA' 로
    풀리므로, yaml 위치를 옮길 때는 params 의 상대경로도 함께 조정해야 한다.
    """
    out = {}
    for k, v in params.items():
        if isinstance(v, str) and (v.startswith("./") or v.startswith("../")):
            out[k] = str((base_dir / v).resolve())
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    cfg = PipelineConfig.load(
        sys.argv[1] if len(sys.argv) > 1 else "pipeline.yaml"
    )
    print(cfg.describe())
    print("\nhuman tool =", [s.name for s in cfg.by_tool("human")])
    print("object tool =", [s.name for s in cfg.by_tool("object")])
    print("person_only 자동 매핑 =", cfg.person_only_map())
    print(
        "\n1000만(사람 300만 / 객체 700만) 추정:",
        cfg.estimate_memory(3_000_000, 7_000_000),
    )