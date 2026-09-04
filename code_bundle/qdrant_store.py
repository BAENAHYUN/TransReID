"""
Qdrant 계층 — 컬렉션 생성(양자화 포함), 적재, 융합 검색.

설정(pipeline.yaml)만 보고 동작한다.
임베더도 라우터도 모른다.

대규모(약 1000만 point) 전제:
  * 원본 float32 vector는 on_disk
  * 양자화 vector는 검색 가속용
  * quantized search + rescore + oversampling은 Qdrant가 내부 처리
  * 여러 named vector 결과는 서버에서 DBSF/RRF fusion

안전장치:
  1) quantized search -> original vector rescore
  2) detections / vectors 길이 불일치 차단
  3) deterministic point ID
     - detection_id를 정상 경로로 사용
     - bbox 기반 ID는 비상용 fallback
  4) 기존 collection schema 검증
     - named vector 이름 / dim / distance / quantization
     - on_disk
     - HNSW(m, ef_construct)
  5) payload index 항상 보장
  6) upsert/query 전 vector name / dim / NaN·Inf 검증
  7) extra_filter의 must / should / must_not 전체 보존
  8) det.extra가 핵심 payload key를 덮어쓰지 못하게 보호
  9) 배치 단위 "검증 -> 전송" streaming 적재
     - 실패 반경을 최대 1 batch로 제한
     - upsert_stream()은 Iterable[(det, vectors)] 직접 지원
 10) clustering kNN은 query_batch_points로 네트워크 왕복 감소
 11) qdrant-client 버전에 따라 weighted RRF / oversampling 안전 폴백

이번 개정 (임베더 3종 확정 후)
------------------------------
 A) zip(strict=True) 제거 — Python 3.9 호환. 길이 검증은 별도로 이미 수행
 B) knn_edges: distance가 EUCLID 계열이면 threshold 방향이 반대이므로 차단
 C) extra["track_id"]를 충돌이 아니라 track_id fallback 소스로 처리
 D) quantization 검증이 서버 응답 부족으로 무력화될 때 warning
 E) on_disk를 서버가 보고하지 않을 때 warning (1000만 규모에서 중요)
 F) vector별 distance 지원 — cfg.qdrant.distance_for(name)이 있으면 사용
    (DINOv2 PCA-whitening 도입 시 L2 노름이 깨져 COSINE/DOT 의미가 달라진다)
 G) person_only에 dict를 허용 — named vector마다 다른 필터를 걸 수 있다
    (사람 벡터 prefetch가 객체 point를 스캔하며 낭비되는 것을 막는다)

주의: 이 파일은 numpy 전수 계산 대비 검증이 끝나지 않았다.
HNSW 근사 + INT8 양자화 두 단계 손실이 얼마인지 반드시 실측할 것.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from config import PipelineConfig


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# pipeline 기본값
# ---------------------------------------------------------------------------

DEFAULT_SCALAR_QUANTILE = 0.99
DEFAULT_ALWAYS_RAM = True
DEFAULT_RESCORE = True
DEFAULT_OVERSAMPLING = 2.0

# score가 "작을수록 좋은" distance. threshold 비교 방향이 반대가 된다.
_SMALLER_IS_BETTER_DISTANCES = {"euclid", "euclidean", "l2", "manhattan"}

# pipeline.yaml 의 retrievers[*].scope -> is_person 필터 값. (개정 H)
#   person : 사람 crop 전용   -> is_person == True 만
#   object : 객체 crop 전용   -> is_person == False 만
#   all    : 모든 crop 공통   -> 필터를 걸지 않는다 (None)
#
# 'all' 을 False 로 두면 사람 point 가 통째로 빠지므로, None 과 False 를
# 혼동하지 않도록 이 매핑을 단일 진실 공급원으로 삼는다.
_SCOPE_TO_PERSON_ONLY: Dict[str, Optional[bool]] = {
    "all": None,
    "person": True,
    "object": False,
}


# ---------------------------------------------------------------------------
# 공통 helper
# ---------------------------------------------------------------------------

def _value(value: Any) -> Any:
    """Enum/Pydantic 값을 primitive 값으로 변환."""
    return getattr(value, "value", value)


def _norm_name(value: Any) -> str:
    if value is None:
        return ""
    return str(_value(value)).strip().lower()


def _attr_or_default(obj: Any, name: str, default: Any) -> Any:
    """attribute가 없거나 값이 None이면 명시적 기본값을 사용한다."""
    value = getattr(obj, name, None)
    return default if value is None else value


def _model_field_names(model_cls: Any) -> set:
    """Pydantic v1/v2 모두에서 model field 이름을 가져온다."""
    if model_cls is None:
        return set()

    fields = getattr(model_cls, "model_fields", None)
    if fields is None:
        fields = getattr(model_cls, "__fields__", None)
    if not fields:
        return set()

    return set(fields.keys())


def _chunked(iterable: Iterable[Any], size: int) -> Iterator[List[Any]]:
    """Iterable을 메모리에 전체 적재하지 않고 size개씩 자른다."""
    batch: List[Any] = []

    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []

    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def _quantization_config(quant, models):
    """pipeline config -> Qdrant quantization config. quant=None도 안전하다."""

    quant_type = _norm_name(getattr(quant, "type", None))

    if quant_type == "scalar":
        quantile = float(
            _attr_or_default(quant, "quantile", DEFAULT_SCALAR_QUANTILE)
        )
        always_ram = bool(
            _attr_or_default(quant, "always_ram", DEFAULT_ALWAYS_RAM)
        )

        return models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=quantile,
                always_ram=always_ram,
            )
        )

    if quant_type == "binary":
        always_ram = bool(
            _attr_or_default(quant, "always_ram", DEFAULT_ALWAYS_RAM)
        )

        return models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=always_ram)
        )

    return None


def _expected_quant_signature(quant) -> Dict[str, Any]:
    """pipeline.yaml 기준 expected quantization signature."""

    quant_type = _norm_name(getattr(quant, "type", None))

    if quant_type == "scalar":
        return {
            "type": "scalar",
            "quantile": float(
                _attr_or_default(quant, "quantile", DEFAULT_SCALAR_QUANTILE)
            ),
            "always_ram": bool(
                _attr_or_default(quant, "always_ram", DEFAULT_ALWAYS_RAM)
            ),
        }

    if quant_type == "binary":
        return {
            "type": "binary",
            "always_ram": bool(
                _attr_or_default(quant, "always_ram", DEFAULT_ALWAYS_RAM)
            ),
        }

    return {"type": "none"}


def _actual_quant_signature(quant_config) -> Dict[str, Any]:
    """Qdrant 서버가 돌려준 실제 quantization signature."""

    if quant_config is None:
        return {"type": "none"}

    scalar = getattr(quant_config, "scalar", None)

    if scalar is not None:
        quantile = getattr(scalar, "quantile", None)
        always_ram = getattr(scalar, "always_ram", None)

        return {
            "type": "scalar",
            "quantile": None if quantile is None else float(quantile),
            "always_ram": None if always_ram is None else bool(always_ram),
        }

    binary = getattr(quant_config, "binary", None)

    if binary is not None:
        always_ram = getattr(binary, "always_ram", None)

        return {
            "type": "binary",
            "always_ram": None if always_ram is None else bool(always_ram),
        }

    return {"type": _norm_name(type(quant_config).__name__)}


def _compare_quantization(
    name: str,
    expected: Dict[str, Any],
    actual: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    quantization schema 비교.

    반환: (일치 여부, 검증 불가 항목 메시지 목록)

    서버가 always_ram / quantile을 None으로 돌려주면 "확인 불가"이지
    "일치"가 아니다. mismatch로 승격하지는 않되 조용히 넘기지도 않는다.
    (개정 D — 이전에는 무조건 True를 반환해 quantile 불일치를 놓쳤다)
    """

    unverified: List[str] = []

    if expected.get("type") != actual.get("type"):
        return False, unverified

    if expected.get("type") == "none":
        return True, unverified

    expected_ram = expected.get("always_ram")
    actual_ram = actual.get("always_ram")

    if actual_ram is None:
        unverified.append(
            f"{name}.quantization.always_ram: "
            f"서버가 값을 보고하지 않아 검증 불가 "
            f"(expected={expected_ram})"
        )
    elif expected_ram is not None and expected_ram != actual_ram:
        return False, unverified

    if expected.get("type") == "scalar":
        expected_quantile = expected.get("quantile")
        actual_quantile = actual.get("quantile")

        if actual_quantile is None:
            unverified.append(
                f"{name}.quantization.quantile: "
                f"서버가 값을 보고하지 않아 검증 불가 "
                f"(expected={expected_quantile})"
            )
        elif expected_quantile is not None:
            if abs(float(expected_quantile) - float(actual_quantile)) >= 1e-6:
                return False, unverified

    return True, unverified


# ===========================================================================
# QdrantStore
# ===========================================================================

class QdrantStore:

    RESERVED_PAYLOAD_KEYS = frozenset({
        "image_id",
        "frame_idx",
        "label",
        "is_person",
        "score",
        "bbox",
        "detection_id",
        "track_id",
    })

    # extra에 들어와도 충돌로 보지 않고 정규 필드로 승격시키는 key.
    # (개정 C — detection 단계에서 track_id를 extra로 넘기는 호출부를 허용)
    PROMOTABLE_EXTRA_KEYS = frozenset({
        "detection_id",
        "crop_id",
        "track_id",
    })

    def __init__(self, cfg: PipelineConfig, client=None):
        self.cfg = cfg

        if client is None:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=cfg.qdrant.url, timeout=120)

        self.client = client
        self.collection = cfg.collection

    # -----------------------------------------------------------------------
    # 기본 설정
    # -----------------------------------------------------------------------

    def _distance_name(self, name: Optional[str] = None) -> str:
        """
        vector별 distance 이름(소문자).

        cfg.qdrant.distance_for(name)이 있으면 그것을 쓰고, 없으면 전역
        cfg.qdrant.distance로 폴백한다. (개정 F)

        DINOv2에 PCA-whitening을 적용하면 L2 노름이 깨져서 COSINE과 DOT의
        의미가 달라진다. 그때 config에 distance_for만 추가하면 이 파일은
        수정 없이 따라간다.
        """
        per_vector = getattr(self.cfg.qdrant, "distance_for", None)

        if name is not None and callable(per_vector):
            try:
                resolved = per_vector(name)
            except Exception:
                resolved = None

            if resolved is not None:
                return _norm_name(resolved)

        return _norm_name(self.cfg.qdrant.distance)

    def _expected_distance(self, models, name: Optional[str] = None):
        """string / Enum 모두 처리."""

        raw = self._distance_name(name)

        try:
            return getattr(models.Distance, raw.upper())
        except AttributeError as exc:
            raise ValueError(
                f"지원하지 않는 Qdrant distance: {raw}"
                + (f" (vector='{name}')" if name else "")
            ) from exc

    def _smaller_is_better(self, name: Optional[str] = None) -> bool:
        """score가 작을수록 좋은 distance인지."""
        return self._distance_name(name) in _SMALLER_IS_BETTER_DISTANCES

    def _is_quantized(self, name: str) -> bool:
        quant = self.cfg.qdrant.quant_for(name)

        return _norm_name(getattr(quant, "type", None)) in {"scalar", "binary"}

    def _quant_search_params(self, name: str, models):
        """
        quantized vector 검색 설정. fusion 설정과 독립적이다.

        client가 oversampling/rescore 필드를 지원하지 않으면
        지원 가능한 필드만 사용한다.

        참고: 저차원 벡터(IRRA 512-d)는 고차원(SOLIDER 1024-d)보다 INT8
        양자화 손실이 상대적으로 크게 나타난다. recall이 중요한 벡터는
        pipeline.yaml에서 oversampling을 올리거나 양자화를 끄는 것을 검토할 것.
        """

        if not self._is_quantized(name):
            return None

        quant = self.cfg.qdrant.quant_for(name)

        rescore = bool(_attr_or_default(quant, "rescore", DEFAULT_RESCORE))
        oversampling = float(
            _attr_or_default(quant, "oversampling", DEFAULT_OVERSAMPLING)
        )

        if oversampling < 1.0:
            raise ValueError(
                f"{name}.oversampling은 1.0 이상이어야 합니다: {oversampling}"
            )

        qsp_cls = getattr(models, "QuantizationSearchParams", None)
        search_params_cls = getattr(models, "SearchParams", None)

        if qsp_cls is None or search_params_cls is None:
            logger.warning(
                "현재 qdrant-client는 QuantizationSearchParams를 지원하지 않아 "
                "%s에서 explicit rescore/oversampling을 적용하지 않습니다.",
                name,
            )
            return None

        supported = _model_field_names(qsp_cls)

        kwargs: Dict[str, Any] = {}

        if not supported or "ignore" in supported:
            kwargs["ignore"] = False

        if not supported or "rescore" in supported:
            kwargs["rescore"] = rescore
        else:
            logger.warning(
                "현재 qdrant-client의 QuantizationSearchParams에 "
                "'rescore' 필드가 없습니다."
            )

        if not supported or "oversampling" in supported:
            kwargs["oversampling"] = oversampling
        else:
            logger.warning(
                "현재 qdrant-client는 oversampling을 지원하지 않아 "
                "%s 검색에서 제외합니다.",
                name,
            )

        return search_params_cls(quantization=qsp_cls(**kwargs))

    # -----------------------------------------------------------------------
    # vector 검증
    # -----------------------------------------------------------------------

    def _validate_vector(self, name: str, vector: np.ndarray) -> np.ndarray:

        if name not in self.cfg.retrievers:
            raise ValueError(
                f"알 수 없는 named vector '{name}'. "
                f"허용값={list(self.cfg.retrievers.keys())}"
            )

        arr = np.asarray(vector, dtype=np.float32)

        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.ndim != 1:
            raise ValueError(
                f"vector '{name}'은 1-D여야 합니다. 현재 shape={arr.shape}"
            )

        expected_dim = int(self.cfg.retrievers[name].dim)

        if arr.size != expected_dim:
            raise ValueError(
                f"vector '{name}' 차원 불일치: "
                f"expected={expected_dim}, actual={arr.size}"
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"vector '{name}'에 NaN/Inf가 포함되어 있습니다.")

        return arr

    def _validate_vector_map(
        self,
        vectors: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:

        return {
            name: self._validate_vector(name, vector)
            for name, vector in vectors.items()
        }

    # -----------------------------------------------------------------------
    # deterministic point ID
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize_id(value: Any) -> Optional[str]:
        """0 같은 falsy 값도 유효 ID로 취급한다. 공백뿐이면 None."""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @classmethod
    def _get_detection_id(cls, det: Any) -> Optional[str]:
        """
        우선순위:
          1) det.detection_id
          2) det.extra["detection_id"]
          3) det.extra["crop_id"]

        det.detection_id 와 extra["detection_id"] 가 둘 다 있으면서 값이
        다르면 ValueError. (개정 I — 예전에는 조용히 det 쪽을 썼다)
        Point ID 가 여기서 파생되므로, 두 값이 어긋난 채 통과하면 중복
        point 나 덮어쓰기가 발생한다.
        """

        explicit_id = cls._normalize_id(getattr(det, "detection_id", None))
        extra = getattr(det, "extra", None)
        extra_map = extra if isinstance(extra, dict) else {}

        extra_id = cls._normalize_id(extra_map.get("detection_id"))

        if explicit_id is not None and extra_id is not None:
            if explicit_id != extra_id:
                raise ValueError(
                    f"detection_id 충돌: "
                    f"det.detection_id={explicit_id!r}, "
                    f"det.extra['detection_id']={extra_id!r}. "
                    f"Point ID 가 여기서 파생되므로 어느 쪽이 맞는지 "
                    f"detection 단계에서 확정하세요."
                )
            return explicit_id

        if explicit_id is not None:
            return explicit_id

        if extra_id is not None:
            return extra_id

        return cls._normalize_id(extra_map.get("crop_id"))

    @staticmethod
    def _get_track_id(det: Any) -> Optional[int]:
        """
        우선순위: det.track_id -> det.extra["track_id"] (개정 C)

        예전에는 extra["track_id"]가 예약 key 충돌로 예외를 발생시켜,
        detection 단계에서 track_id를 extra로 넘기는 호출부가 있으면
        적재 전체가 실패했다.

        둘 다 있으면서 값이 다르면 ValueError. (개정 I)
        track_id 는 클러스터링 평가의 정답 라벨로 쓰일 수 있어, 조용히
        한쪽을 고르면 평가 결과가 오염된다.
        """

        def to_int(value: Any, source: str) -> Optional[int]:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{source} 를 int 로 변환할 수 없습니다: {value!r}"
                ) from exc

        own = to_int(getattr(det, "track_id", None), "det.track_id")

        extra = getattr(det, "extra", None)
        extra_map = extra if isinstance(extra, dict) else {}
        from_extra = to_int(
            extra_map.get("track_id"), "det.extra['track_id']"
        )

        if own is not None and from_extra is not None and own != from_extra:
            raise ValueError(
                f"track_id 충돌: det.track_id={own}, "
                f"det.extra['track_id']={from_extra}. "
                f"어느 쪽이 맞는지 detection/tracking 단계에서 확정하세요."
            )

        return own if own is not None else from_extra

    def _stable_point_id(self, det: Any) -> Tuple[str, bool]:
        """
        정상 경로: detection_id
        fallback: image_id + frame_idx + label + bbox
        """

        detection_id = self._get_detection_id(det)

        if detection_id is not None:
            stable_key = (
                f"{self.collection}|detection_id={detection_id}"
            )
            used_bbox_fallback = False

        else:
            bbox = ",".join(f"{float(x):.6f}" for x in det.bbox)

            stable_key = (
                f"{self.collection}|"
                f"image_id={det.image_id}|"
                f"frame_idx={int(det.frame_idx)}|"
                f"label={str(det.label).lower()}|"
                f"bbox={bbox}"
            )
            used_bbox_fallback = True

        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))

        return point_id, used_bbox_fallback

    # -----------------------------------------------------------------------
    # collection schema 검증
    # -----------------------------------------------------------------------

    def _validate_existing_collection(self, models) -> None:
        """
        검증:
          * named vector 이름
          * dim
          * distance (vector별)
          * quantization
          * on_disk
          * HNSW m / ef_construct

        HNSW는 사후 update가 가능하므로 mismatch는 warning으로 둔다.
        on_disk는 1000만 규모에서 메모리 정책에 직접 영향을 주므로 error.
        서버가 값을 보고하지 않아 검증 불가한 항목은 warning으로 남긴다.
        """

        info = self.client.get_collection(self.collection)
        actual_vectors = info.config.params.vectors

        if not isinstance(actual_vectors, dict):
            raise RuntimeError(
                f"기존 컬렉션 '{self.collection}'이 named vector 구조가 아닙니다."
            )

        expected_names = set(self.cfg.retrievers.keys())
        actual_names = set(actual_vectors.keys())

        errors: List[str] = []
        unverified: List[str] = []

        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)

        if missing:
            errors.append(f"missing named vectors={missing}")

        if extra:
            errors.append(f"unexpected named vectors={extra}")

        expected_on_disk = bool(self.cfg.qdrant.on_disk)

        for name in sorted(expected_names & actual_names):
            actual = actual_vectors[name]
            spec = self.cfg.retrievers[name]

            # ---- dim ----
            expected_dim = int(spec.dim)

            if int(actual.size) != expected_dim:
                errors.append(
                    f"{name}.dim: expected={expected_dim}, actual={actual.size}"
                )

            # ---- distance (vector별) ----
            wanted_distance = _norm_name(
                self._expected_distance(models, name)
            )
            actual_distance = _norm_name(actual.distance)

            if actual_distance != wanted_distance:
                errors.append(
                    f"{name}.distance: "
                    f"expected={wanted_distance}, actual={actual_distance}"
                )

            # ---- quantization ----
            expected_quant = _expected_quant_signature(
                self.cfg.qdrant.quant_for(name)
            )
            actual_quant = _actual_quant_signature(
                getattr(actual, "quantization_config", None)
            )

            same, quant_unverified = _compare_quantization(
                name, expected_quant, actual_quant
            )
            unverified.extend(quant_unverified)

            if not same:
                errors.append(
                    f"{name}.quantization: "
                    f"expected={expected_quant}, actual={actual_quant}"
                )

            # ---- on_disk ----
            actual_on_disk = getattr(actual, "on_disk", None)

            if actual_on_disk is None:
                # 개정 E — 이전에는 조용히 통과했다.
                # 1000만 규모에서 on_disk 오설정은 메모리 사용량을 직격한다.
                unverified.append(
                    f"{name}.on_disk: 서버가 값을 보고하지 않아 검증 불가 "
                    f"(expected={expected_on_disk}). "
                    f"Qdrant 대시보드에서 직접 확인하세요."
                )
            elif bool(actual_on_disk) != expected_on_disk:
                errors.append(
                    f"{name}.on_disk: "
                    f"expected={expected_on_disk}, actual={bool(actual_on_disk)}"
                )

            # ---- HNSW ----
            expected_hnsw = self.cfg.qdrant.hnsw_for(name)
            actual_hnsw = getattr(actual, "hnsw_config", None)

            if actual_hnsw is not None:
                actual_m = getattr(actual_hnsw, "m", None)
                actual_ef = getattr(actual_hnsw, "ef_construct", None)
                expected_m = getattr(expected_hnsw, "m", None)
                expected_ef = getattr(expected_hnsw, "ef_construct", None)

                if (
                    actual_m is not None
                    and expected_m is not None
                    and int(actual_m) != int(expected_m)
                ):
                    logger.warning(
                        "%s.hnsw.m 불일치: expected=%s actual=%s",
                        name, expected_m, actual_m,
                    )

                if (
                    actual_ef is not None
                    and expected_ef is not None
                    and int(actual_ef) != int(expected_ef)
                ):
                    logger.warning(
                        "%s.hnsw.ef_construct 불일치: expected=%s actual=%s",
                        name, expected_ef, actual_ef,
                    )

        for message in unverified:
            logger.warning("schema 검증 불가: %s", message)

        if errors:
            detail = "\n  - ".join(errors)

            raise RuntimeError(
                f"기존 Qdrant 컬렉션 '{self.collection}'의 schema가 "
                f"pipeline.yaml과 다릅니다.\n"
                f"  - {detail}\n"
                f"의도적으로 새 schema를 적용하려면 "
                f"ensure_collection(recreate=True)를 사용하세요."
            )

    # -----------------------------------------------------------------------
    # payload index
    # -----------------------------------------------------------------------

    @staticmethod
    def _payload_index_type(info, field: str) -> Optional[str]:

        payload_schema = getattr(info, "payload_schema", None) or {}
        existing = payload_schema.get(field)

        if existing is None:
            return None

        return _norm_name(getattr(existing, "data_type", None))

    def _ensure_payload_indexes(self) -> None:
        """
        get_collection()은 최초 한 번만 호출한다.
        create race가 발생한 예외 경로에서만 다시 조회한다.
        """

        desired = {
            "is_person": "bool",
            "label": "keyword",
            "image_id": "keyword",
            "frame_idx": "integer",
        }

        info = self.client.get_collection(self.collection)

        known_types = {
            field: self._payload_index_type(info, field)
            for field in desired.keys()
        }

        for field, expected_type in desired.items():
            actual_type = known_types.get(field)

            if actual_type is not None:
                if actual_type != expected_type:
                    raise RuntimeError(
                        f"payload index '{field}' 타입 불일치: "
                        f"expected={expected_type}, actual={actual_type}"
                    )
                continue

            logger.info("payload index 생성: %s (%s)", field, expected_type)

            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=expected_type,
                    wait=True,
                )
                known_types[field] = expected_type

            except Exception:
                # 동시 프로세스 race인지 확인
                refreshed = self.client.get_collection(self.collection)
                refreshed_type = self._payload_index_type(refreshed, field)

                if refreshed_type == expected_type:
                    logger.info(
                        "payload index '%s'는 다른 프로세스에서 동시에 생성됨",
                        field,
                    )
                    known_types[field] = expected_type
                    continue

                logger.error(
                    "payload index '%s' 생성 실패. 서버가 보고한 타입=%s. "
                    "wait=True 타임아웃이라면 잠시 후 재시도하세요.",
                    field, refreshed_type,
                )
                raise

    # -----------------------------------------------------------------------
    # collection
    # -----------------------------------------------------------------------

    def ensure_collection(self, recreate: bool = False) -> None:

        from qdrant_client import models

        q = self.cfg.qdrant

        vectors_config = {}

        for name, spec in self.cfg.retrievers.items():
            quant = q.quant_for(name)
            hnsw = q.hnsw_for(name)

            vectors_config[name] = models.VectorParams(
                size=spec.dim,
                distance=self._expected_distance(models, name),
                on_disk=q.on_disk,
                quantization_config=_quantization_config(quant, models),
                hnsw_config=models.HnswConfigDiff(
                    m=hnsw.m,
                    ef_construct=hnsw.ef_construct,
                ),
            )

        exists = self.client.collection_exists(self.collection)

        if exists and recreate:
            logger.warning("기존 컬렉션 삭제: %s", self.collection)
            self.client.delete_collection(self.collection)
            exists = False

        if exists:
            self._validate_existing_collection(models)
            logger.info("컬렉션 존재 + schema 일치: %s", self.collection)

        else:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=vectors_config,
            )
            logger.info(
                "컬렉션 생성: %s %s",
                self.collection,
                self.cfg.vector_config(),
            )

        self._ensure_payload_indexes()

    # -----------------------------------------------------------------------
    # point 생성
    # -----------------------------------------------------------------------

    def _build_point(
        self,
        det: Any,
        vec: Dict[str, np.ndarray],
        is_person_fn,
        models,
    ) -> Tuple[Optional[Any], bool]:
        """
        point 하나를 완전히 검증하고 생성한다.

        반환: (PointStruct | None, used_bbox_fallback)
        vec가 비어 있으면 (None, False).
        """

        if not vec:
            return None, False

        checked_vectors = self._validate_vector_map(vec)

        detection_id = self._get_detection_id(det)
        track_id = self._get_track_id(det)

        raw_extra = getattr(det, "extra", None)
        extra_payload: Dict[str, Any] = {}

        if raw_extra:
            if not isinstance(raw_extra, dict):
                raise TypeError(
                    f"det.extra는 dict여야 합니다. "
                    f"현재 type={type(raw_extra).__name__}"
                )

            extra_payload = dict(raw_extra)

            # 정규 필드로 승격된 key는 extra에서 제거한다.
            # (개정 C — track_id를 충돌로 보지 않는다)
            for key in self.PROMOTABLE_EXTRA_KEYS:
                extra_payload.pop(key, None)

            collisions = (
                self.RESERVED_PAYLOAD_KEYS & set(extra_payload.keys())
            )

            if collisions:
                raise ValueError(
                    f"det.extra가 예약 payload key를 덮어쓰려고 합니다: "
                    f"{sorted(collisions)}"
                )

        payload: Dict[str, Any] = {
            "image_id": det.image_id,
            "frame_idx": int(det.frame_idx),
            "label": det.label,
            "is_person": bool(is_person_fn(det)),
            "score": float(det.score),
            "bbox": [float(x) for x in det.bbox],
        }

        if detection_id is not None:
            payload["detection_id"] = detection_id

        if track_id is not None:
            payload["track_id"] = track_id

        payload.update(extra_payload)

        point_id, used_bbox_fallback = self._stable_point_id(det)

        point = models.PointStruct(
            id=point_id,
            vector={
                name: arr.tolist()
                for name, arr in checked_vectors.items()
            },
            payload=payload,
        )

        return point, used_bbox_fallback

    # -----------------------------------------------------------------------
    # 진짜 streaming upsert
    # -----------------------------------------------------------------------

    def upsert_stream(
        self,
        items: Iterable[Tuple[Any, Dict[str, np.ndarray]]],
        batch_size: int = 256,
        is_person_fn=None,
        parallel: int = 1,
        max_retries: int = 3,
    ) -> int:
        """
        Iterable[(detection, vectors)]을 직접 받아 진짜 streaming 적재.

        처리 순서:
          1) raw item을 최대 batch_size개 읽음
          2) batch 전체 PointStruct 검증/생성
          3) batch 안에서 하나라도 실패하면 그 batch는 전송하지 않음
          4) 검증 성공한 batch만 Qdrant 전송
          5) 전송 성공 뒤 committed 카운트 증가

        따라서 오류 발생 시 이미 완료된 이전 batch는 남지만,
        실패 반경은 최대 현재 batch 하나로 제한된다.
        """

        from qdrant_client import models

        if batch_size <= 0:
            raise ValueError("batch_size는 1 이상이어야 합니다.")

        if parallel <= 0:
            raise ValueError("parallel은 1 이상이어야 합니다.")

        if max_retries < 0:
            raise ValueError("max_retries는 0 이상이어야 합니다.")

        if is_person_fn is None:
            labels = {str(x).lower() for x in self.cfg.person_labels}

            def is_person_fn(d):
                return str(d.label).lower() in labels

        committed = 0
        skipped = 0
        bbox_fallback_ids = 0
        batch_index = 0

        for raw_batch in _chunked(items, batch_size):
            batch_index += 1

            points = []
            batch_skipped = 0
            batch_bbox_fallbacks = 0

            # --- 1) batch 전체 검증 / PointStruct 생성 ---
            try:
                for det, vec in raw_batch:
                    point, used_bbox_fallback = self._build_point(
                        det, vec, is_person_fn, models
                    )

                    if point is None:
                        batch_skipped += 1
                        continue

                    if used_bbox_fallback:
                        batch_bbox_fallbacks += 1

                    points.append(point)

            except Exception as exc:
                raise RuntimeError(
                    f"Qdrant 적재 batch 검증 실패. "
                    f"batch_index={batch_index}, "
                    f"committed_points={committed}. "
                    f"현재 batch는 전송되지 않았습니다."
                ) from exc

            # 빈 batch면 서버 호출 생략
            if not points:
                skipped += batch_skipped
                continue

            # --- 2) 검증 완료 batch만 전송 ---
            try:
                self.client.upload_points(
                    collection_name=self.collection,
                    points=points,
                    batch_size=len(points),
                    parallel=parallel,
                    max_retries=max_retries,
                    wait=True,
                )

            except Exception as exc:
                raise RuntimeError(
                    f"Qdrant 적재 batch 전송 실패. "
                    f"batch_index={batch_index}, "
                    f"committed_points={committed}. "
                    f"현재 batch의 서버 반영 여부는 "
                    f"전송 실패 시점에 따라 확인이 필요합니다."
                ) from exc

            # 성공 후에만 stats 반영
            committed += len(points)
            skipped += batch_skipped
            bbox_fallback_ids += batch_bbox_fallbacks

        if skipped:
            logger.warning("벡터 없는 detection %d건 건너뜀", skipped)

        if bbox_fallback_ids:
            logger.warning(
                "%d건이 detection_id 없이 bbox 기반 Point ID fallback을 "
                "사용했습니다. 운영 DB에서는 detection 단계에서 detection_id를 "
                "항상 설정하는 것을 권장합니다.",
                bbox_fallback_ids,
            )

        logger.info("upsert_stream 완료: %d committed points", committed)

        return int(committed)

    # -----------------------------------------------------------------------
    # 호환용 upsert wrapper
    # -----------------------------------------------------------------------

    def upsert(
        self,
        detections: Sequence[Any],
        vectors: Sequence[Dict[str, np.ndarray]],
        batch_size: int = 256,
        is_person_fn=None,
        parallel: int = 1,
        max_retries: int = 3,
    ) -> int:
        """
        기존 호출부 호환용.

        진짜 streaming이 필요하면 호출부에서
        upsert_stream(Iterable[(det, vec)])을 직접 사용한다.
        """

        if len(detections) != len(vectors):
            raise ValueError(
                f"detections / vectors 길이 불일치: "
                f"{len(detections)} != {len(vectors)}"
            )

        # 개정 A — 길이는 위에서 이미 확인했다.
        # zip(strict=True)는 Python 3.10+ 전용이라 제거한다.
        return self.upsert_stream(
            zip(detections, vectors),
            batch_size=batch_size,
            is_person_fn=is_person_fn,
            parallel=parallel,
            max_retries=max_retries,
        )

    # -----------------------------------------------------------------------
    # filter
    # -----------------------------------------------------------------------

    def _build_filter(
        self,
        models,
        person_only: Optional[bool],
        extra_filter=None,
    ):
        """
        extra_filter를 nested Filter로 통째로 넣는다.
        must / should / must_not / min_should 의미를 보존한다.
        """

        if person_only is None:
            return extra_filter

        person_condition = models.FieldCondition(
            key="is_person",
            match=models.MatchValue(value=person_only),
        )

        if extra_filter is None:
            return models.Filter(must=[person_condition])

        return models.Filter(must=[person_condition, extra_filter])

    def _scope_person_only(self, name: str) -> Optional[bool]:
        """
        pipeline.yaml 의 retrievers[name].scope 로부터 is_person 필터를 정한다.
        (개정 H — 호출부가 아무것도 넘기지 않아도 자동으로 라우팅된다)

        이 파일은 임베더 이름을 모른다. 'irra'/'solider' 같은 문자열을 코드에
        박는 대신 설정의 scope 만 읽으므로, 새 임베더를 추가할 때 YAML 한 줄만
        늘리면 된다.
        """
        spec = self.cfg.retrievers[name]
        raw = getattr(spec, "scope", None)

        if raw is None:
            raise ValueError(
                f"retriever '{name}' 에 scope 가 없습니다. "
                f"pipeline.yaml 에 {sorted(_SCOPE_TO_PERSON_ONLY)} 중 하나를 "
                f"지정하세요."
            )

        scope = _norm_name(raw)

        if scope not in _SCOPE_TO_PERSON_ONLY:
            raise ValueError(
                f"retriever '{name}' 의 scope 가 올바르지 않습니다: {raw!r}\n"
                f"  허용값={sorted(_SCOPE_TO_PERSON_ONLY)}"
            )

        return _SCOPE_TO_PERSON_ONLY[scope]

    def _resolve_person_only(
        self,
        person_only: Union[bool, Dict[str, Optional[bool]], None],
        name: str,
    ) -> Optional[bool]:
        """
        named vector 하나에 적용할 is_person 필터를 결정한다.

        우선순위:
          1) person_only 가 dict 이고 해당 key 가 있으면 그 값 (명시적 override)
          2) person_only 가 bool 이면 모든 vector 에 동일 적용 (기존 호환)
          3) None 이면 scope 기반 자동 결정  <- 기본 경로

        사람 벡터(IRRA/SOLIDER)와 객체 벡터(DINOv2)를 한 컬렉션에 섞어 두면,
        필터 없는 prefetch 가 해당 벡터를 갖지 않는 point 까지 스캔하며
        prefetch_limit 을 낭비한다. scope='all' 인 SigLIP2 만 필터가 없다.

        dict 로 override 할 때 값 None 은 "필터 없음"을 뜻하며, key 자체가
        없는 것(-> scope 폴백)과 구별된다.
        """
        if isinstance(person_only, dict):
            if name in person_only:
                return person_only[name]
            return self._scope_person_only(name)

        if person_only is not None:
            return person_only

        return self._scope_person_only(name)

    # -----------------------------------------------------------------------
    # RRF
    # -----------------------------------------------------------------------

    def _build_rrf_query(
        self,
        models,
        names: Sequence[str],
        weights: Optional[Dict[str, float]],
    ):
        """실제 Rrf model에 weights field가 있을 때만 weighted RRF 사용."""

        rrf_cls = getattr(models, "Rrf", None)
        rrf_query_cls = getattr(models, "RrfQuery", None)

        fields = _model_field_names(rrf_cls)

        if (
            rrf_cls is not None
            and rrf_query_cls is not None
            and "weights" in fields
        ):
            resolved_weights = [
                float(
                    (weights or {}).get(
                        name,
                        self.cfg.retrievers[name].weight,
                    )
                )
                for name in names
            ]

            return rrf_query_cls(rrf=rrf_cls(weights=resolved_weights))

        logger.warning(
            "현재 qdrant-client는 weighted RRF를 지원하지 않아 "
            "균등 RRF로 폴백합니다."
        )

        return models.FusionQuery(fusion=models.Fusion.RRF)

    # -----------------------------------------------------------------------
    # Fusion Search
    # -----------------------------------------------------------------------

    def fused_search(
        self,
        query_vectors: Dict[str, np.ndarray],
        person_only: Union[bool, Dict[str, bool], None] = None,
        limit: Optional[int] = None,
        prefetch_limit: Optional[int] = None,
        weights: Optional[Dict[str, float]] = None,
        extra_filter=None,
    ):
        """
        person_only는 bool 또는 {vector_name: bool} dict를 받는다. (개정 G)

        주의: 텍스트 질의는 텍스트 인코더가 있는 벡터(IRRA / SigLIP2)만
        query_vectors에 담긴다. SOLIDER / DINOv2는 텍스트 공간이 없으므로
        라우터가 제외한다. 결과적으로 벡터가 하나뿐이면 fusion이 아니라
        사실상 single search가 된다 — 정상 동작이다.
        """

        from qdrant_client import models

        if not query_vectors:
            raise ValueError("query_vectors가 비어 있습니다.")

        checked_queries = self._validate_vector_map(query_vectors)

        fusion_cfg = self.cfg.fusion

        limit = int(limit if limit is not None else fusion_cfg.limit)
        prefetch_limit = int(
            prefetch_limit
            if prefetch_limit is not None
            else fusion_cfg.prefetch_limit
        )

        if limit <= 0 or prefetch_limit <= 0:
            raise ValueError("limit과 prefetch_limit은 1 이상이어야 합니다.")

        if prefetch_limit < limit:
            raise ValueError(
                "prefetch_limit은 limit보다 크거나 같아야 합니다."
            )

        names = list(checked_queries.keys())

        prefetch = []

        for name in names:
            query_filter = self._build_filter(
                models,
                self._resolve_person_only(person_only, name),
                extra_filter,
            )

            kwargs: Dict[str, Any] = {
                "query": checked_queries[name].tolist(),
                "using": name,
                "limit": prefetch_limit,
                "filter": query_filter,
            }

            search_params = self._quant_search_params(name, models)

            if search_params is not None:
                kwargs["params"] = search_params

            prefetch.append(models.Prefetch(**kwargs))

        method = _norm_name(fusion_cfg.method)

        if method == "rrf":
            query = self._build_rrf_query(models, names, weights)

        elif method == "dbsf":
            query = models.FusionQuery(fusion=models.Fusion.DBSF)

        else:
            raise ValueError(
                f"지원하지 않는 fusion method: {fusion_cfg.method}"
            )

        result = self.client.query_points(
            collection_name=self.collection,
            prefetch=prefetch,
            query=query,
            limit=limit,
            with_payload=True,
        )

        return result.points

    # -----------------------------------------------------------------------
    # Single Search
    # -----------------------------------------------------------------------

    def search_single(
        self,
        name: str,
        vector: np.ndarray,
        limit: int = 20,
        person_only: Union[bool, Dict[str, bool], None] = None,
        extra_filter=None,
    ):

        from qdrant_client import models

        if limit <= 0:
            raise ValueError("limit은 1 이상이어야 합니다.")

        checked = self._validate_vector(name, vector)

        query_filter = self._build_filter(
            models,
            self._resolve_person_only(person_only, name),
            extra_filter,
        )

        search_params = self._quant_search_params(name, models)

        kwargs: Dict[str, Any] = {
            "collection_name": self.collection,
            "query": checked.tolist(),
            "using": name,
            "query_filter": query_filter,
            "limit": limit,
            "with_payload": True,
        }

        if search_params is not None:
            kwargs["search_params"] = search_params

        return self.client.query_points(**kwargs).points

    # -----------------------------------------------------------------------
    # Clustering kNN
    # -----------------------------------------------------------------------

    def knn_edges(
        self,
        point_ids: Sequence[str],
        using: str,
        k: int = 30,
        threshold: float = 0.0,
        query_batch_size: int = 256,
    ):
        """
        query_batch_points()로 묶어서 kNN sparse edge 생성.
        자기 자신 제거 후 최종적으로 최대 k개만 남긴다.

        threshold는 COSINE/DOT처럼 score가 클수록 좋은 설정을 전제로 한다.
        EUCLID 계열에서는 부등호 방향이 반대이므로 차단한다. (개정 B)
        """

        from qdrant_client import models

        if using not in self.cfg.retrievers:
            raise ValueError(
                f"알 수 없는 named vector '{using}'. "
                f"허용값={list(self.cfg.retrievers.keys())}"
            )

        if k <= 0:
            raise ValueError("k는 1 이상이어야 합니다.")

        if query_batch_size <= 0:
            raise ValueError("query_batch_size는 1 이상이어야 합니다.")

        if self._smaller_is_better(using) and threshold != 0.0:
            raise ValueError(
                f"'{using}'의 distance는 "
                f"'{self._distance_name(using)}'이라 score가 작을수록 "
                f"유사합니다. 현재 threshold 비교(score < threshold 제외)는 "
                f"COSINE/DOT 전제이므로 방향이 반대가 됩니다.\n"
                f"  threshold=0.0으로 두고 호출부에서 걸러내거나, "
                f"이 벡터를 COSINE으로 재구성하세요."
            )

        search_params = self._quant_search_params(using, models)

        edges = []

        for start in range(0, len(point_ids), query_batch_size):
            batch_ids = point_ids[start:start + query_batch_size]

            requests = []

            for pid in batch_ids:
                kwargs: Dict[str, Any] = {
                    "query": pid,
                    "using": using,
                    # self가 포함되지 않는 경우도 고려해 k+1 요청
                    "limit": k + 1,
                    "with_payload": False,
                    "with_vector": False,
                }

                if search_params is not None:
                    kwargs["params"] = search_params

                requests.append(models.QueryRequest(**kwargs))

            responses = self.client.query_batch_points(
                collection_name=self.collection,
                requests=requests,
            )

            if len(responses) != len(batch_ids):
                raise RuntimeError(
                    f"Qdrant batch query 응답 개수 불일치: "
                    f"requests={len(batch_ids)}, responses={len(responses)}"
                )

            # 개정 A — 길이는 바로 위에서 확인했다. strict는 3.10+ 전용.
            for pid, response in zip(batch_ids, responses):
                neighbors = []

                for hit in response.points:
                    if str(hit.id) == str(pid):
                        continue

                    if float(hit.score) < threshold:
                        continue

                    neighbors.append((str(hit.id), float(hit.score)))

                # self가 검색 결과에 없더라도 최대 k개로 고정
                neighbors = neighbors[:k]

                for dst_id, score in neighbors:
                    edges.append((str(pid), dst_id, score))

        return edges