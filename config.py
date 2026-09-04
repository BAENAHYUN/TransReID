"""
pipeline.yaml 로더 + 검증.

구축 스크립트와 검색 스크립트가 이 모듈을 통해 같은 설정을 읽는다.
잘못된 설정은 파이프라인이 반쯤 돌다가 죽는 대신 로드 시점에 바로 실패한다.

이번 개정
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

# Qdrant 가 지원하는 distance. 대소문자는 무시하고 비교한다.
VALID_DISTANCE = {"cosine", "dot", "euclid", "manhattan"}

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
    person_labels: frozenset
    retrievers: Dict[str, RetrieverSpec]
    fusion: FusionSpec
    qdrant: QdrantSpec
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
        if "retrievers" not in raw or not raw["retrievers"]:
            raise ValueError("설정에 retrievers 가 최소 하나는 있어야 합니다.")

        # ---------------- retrievers ---------------- #
        retrievers: Dict[str, RetrieverSpec] = {}

        for name, r in raw["retrievers"].items():
            for key in ("scope", "dim", "module", "class"):
                if key not in r:
                    raise ValueError(f"retriever '{name}': '{key}' 누락")

            if r["scope"] not in VALID_SCOPES:
                raise ValueError(
                    f"retriever '{name}': scope 는 {sorted(VALID_SCOPES)} "
                    f"중 하나여야 합니다 (받은 값: {r['scope']})"
                )

            if int(r["dim"]) <= 0:
                raise ValueError(f"retriever '{name}': dim 은 양수여야 합니다")

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
            collection=raw.get("collection", "person_db"),
            person_labels=frozenset(
                s.lower() for s in (raw.get("person_labels") or ["person"])
            ),
            retrievers=retrievers,
            fusion=fusion,
            qdrant=qdrant,
            verifiers=raw.get("verifiers") or {},
        )

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
        lines = [f"collection: {self.collection}", "retrievers:"]
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
            f"qdrant: distance={self.qdrant.distance} "
            f"quant={self.qdrant.quantization.type} "
            f"on_disk={self.qdrant.on_disk}"
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