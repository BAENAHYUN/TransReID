from __future__ import annotations

"""
qwen_stage.py — 동적 조건 관찰 기반 재순위 · 검증 (독립 파일)
==========================================================

    python search_db.py -k 50 --gender 남성 --top "파란 셔츠에 정장" \
        --accessory 넥타이 --json-out t50.json
    python qwen_stage.py --in t50.json --out r50.json
    python view_results.py --in t50.json r50.json --open

역할 분리
--------
    Query Qwen     자연어 -> 동적 constraint 생성
                   [{object, attribute, expected}, ...]

    Candidate Qwen **expected 값을 모른 채** 이미지를 관찰
                   inventory + {observed, visibility, evidence}

    Python         expected vs observed -> PASS / FAIL / UNKNOWN
                   match x coverage -> 점수 -> 정렬 -> 판정

왜 이 구조인가
------------
이전 두 방식이 모두 실패했다.

  1) "이 이미지가 설명에 맞나? yes/no"
     -> 정답을 23위에서 33위로 내렸다. 전체적 인상 판단이 검색보다 나빴다.

  2) 고정 9슬롯 관찰 + 토큰 겹침 비교
     -> shirt 와 suit 가 top 한 칸에 뭉쳤다. "blue shirt suit" vs
        "blue shirt" 가 부분 점수를 받아, 정장이 없어도 상위에 남았다.
     -> unknown 항목을 분모에서 빼는 바람에, 핵심 조건을 확인하지 못한
        후보가 확인된 후보와 같은 점수를 받았다.

이제 object 를 쪼갠다. shirt / suit / tie 가 각각 독립 constraint 다.
"정장 있음" 은 suit.present = true 이고, inventory 에 suit 가 없고 상체가
충분히 보이면 FAIL 이다. 부분 점수가 아니다.

특정 물체를 코드에 넣지 않는다
--------------------------
suit, helmet, umbrella, backpack, dog, car 를 코드가 알지 못한다.
모두 아래 공통 구조로만 다룬다.

    object      무엇을
    attribute   어떤 속성을 (present / color / pattern / type / ...)
    expected    쿼리가 요구하는 값
    observed    이미지에서 관찰된 값
    visibility  그 판단을 할 만큼 보였는가 (sufficient / insufficient)
    evidence    근거 한 줄

검색어가 "노란 헬멧을 쓴 사람" 으로 바뀌어도 Qwen 이 helmet.present 를
만들어내고 Python 코드는 그대로다.

PASS / FAIL / UNKNOWN
--------------------
    expected=true, observed=true                        -> PASS
    expected=true, observed=false, visibility=충분       -> FAIL
    observed=unknown 또는 visibility=불충분              -> UNKNOWN

UNKNOWN 을 분모에서 그냥 빼면 "확인 못 한 후보"가 "확인된 후보"와 같은
점수를 받는다. 그래서 coverage 를 곱한다.

    match    = Σ(w x 점수) / Σ(w)   [PASS/FAIL 만]
    coverage = Σ(w)[PASS/FAIL] / Σ(w)[전체]
    attr     = match x coverage

    PASS PASS PASS PASS       -> match 1.0, coverage 1.0, attr 1.00
    PASS PASS UNKNOWN UNKNOWN -> match 1.0, coverage 0.5, attr 0.50

유도 질문을 막는 두 가지 장치
-------------------------
1) 후보 Qwen 에게 expected 값을 주지 않는다. 검사할 (object, attribute)
   목록만 준다. "정장이 있어야 한다" 를 알려주지 않는다.

2) inventory 를 함께 받아 교차검증한다. present=true 라고 답했는데
   inventory 에 그 object 가 없으면 UNKNOWN 으로 내린다. 맥락에서
   추론한 답을 걸러내기 위한 것이다.

filter 가 아니라 flag 가 기본이다
-------------------------------
threshold 미달 후보를 지우지 않고 표시만 한다. 수사 검색에서는 놓친 정답이
잘못 올라온 후보보다 나쁘고, 사람이 목록을 검토하는 것이 전제다.
UNKNOWN 후보는 filter 모드에서도 제거하지 않는다.
"""

import argparse
import gc
import json
import logging
import re
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
VERIFY_MODES = ("flag", "filter")

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"

# 존재 여부를 묻는 속성 이름. 값 비교가 아니라 bool 비교를 한다.
PRESENCE_ATTRS = frozenset({"present", "presence", "exists", "visible"})

# 토큰 비교에서 무시할 기능어
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "with", "in", "on", "at",
    "is", "are", "was", "were", "wearing", "holding", "carrying",
    "some", "his", "her", "their", "its", "it", "this", "that",
    "unknown", "none", "n/a", "not", "no", "nothing", "unclear",
})

# 같은 것으로 볼 표현. 각 그룹의 첫 항목으로 정규화한다.
# **특정 데이터셋용 규칙이 아니라 일반 어휘 동의어 사전이다.**
_SYNONYM_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("tshirt", "t-shirt", "tee", "teeshirt"),
    ("sneakers", "trainers", "runners", "athletic shoes"),
    ("pants", "trousers"),
    ("jeans", "denim pants", "denim trousers"),
    ("hoodie", "hooded sweatshirt", "hooded top"),
    ("sweater", "jumper", "pullover"),
    ("dress", "onepiece", "one-piece", "frock", "gown"),
    ("suit", "business suit", "formal suit", "suit jacket"),
    ("blazer", "sport coat", "sports jacket"),
    ("backpack", "rucksack", "knapsack"),
    ("handbag", "purse"),
    ("umbrella", "parasol"),
    ("cellphone", "cell phone", "mobile phone", "smartphone", "phone"),
    ("sunglasses", "shades"),
    ("cap", "baseball cap", "ballcap"),
    ("helmet", "hard hat", "safety helmet"),
    ("floral", "flowery", "flower", "flowered"),
    ("striped", "stripes", "stripe"),
    ("plaid", "checked", "checkered", "check", "tartan"),
    ("polkadot", "polka dot", "dotted", "dots"),
    ("colorful", "colourful", "multicolored", "multicoloured", "vibrant"),
    ("gray", "grey"),
    ("sleeveless", "tank", "strappy"),
    ("outdoors", "outdoor", "outside"),
    ("indoors", "indoor", "inside"),
    ("man", "male"),
    ("woman", "female"),
    ("boy", "young male"),
    ("girl", "young female"),
)

# 색 계열. 같은 계열이면 부분 점수를 준다.
_COLOR_FAMILY: Dict[str, str] = {}
for _fam, _members in {
    "blue": ("blue", "navy", "skyblue", "lightblue", "teal", "cobalt"),
    "red": ("red", "crimson", "maroon", "scarlet", "burgundy"),
    "pink": ("pink", "magenta", "rose", "fuchsia"),
    "green": ("green", "olive", "lime", "khaki", "emerald"),
    "yellow": ("yellow", "gold", "mustard", "amber"),
    "brown": ("brown", "beige", "tan", "camel", "chocolate"),
    "gray": ("gray", "silver", "charcoal", "slate"),
    "purple": ("purple", "violet", "lavender", "plum"),
    "orange": ("orange", "coral", "apricot"),
    "black": ("black", "jet"),
    "white": ("white", "cream", "ivory", "offwhite"),
}.items():
    for _m in _members:
        _COLOR_FAMILY[_m] = _fam

_TRUE_WORDS = frozenset({
    "true", "yes", "present", "1", "visible", "y", "t",
})
_FALSE_WORDS = frozenset({
    "false", "no", "absent", "0", "none", "n", "f", "not present",
})
_UNKNOWN_WORDS = frozenset({
    "unknown", "unclear", "uncertain", "maybe", "n/a", "na", "null", "",
})

# 값 비교 임계. PASS / PARTIAL / FAIL 을 가른다.
PASS_THRESHOLD = 0.8
PARTIAL_THRESHOLD = 0.4
PARTIAL_SCORE = 0.5

# 정밀도를 섞는 비율. 관찰이 쿼리보다 덜 자세할 때 과도한 감점을 막는다.
RECALL_WEIGHT = 0.7
PRECISION_WEIGHT = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# 프롬프트
# ─────────────────────────────────────────────────────────────────────────────
QUERY_PARSE_PROMPT = """Convert this person description into search constraints.

Description: "{query}"

Return a JSON array. Each element has exactly these keys:
  "object"    - a single concrete thing (subject, shirt, suit, tie, umbrella,
                backpack, helmet, shoes, hair, ...)
  "attribute" - what is being specified: "present", "color", "pattern",
                "type", "length", "style", "state"
  "expected"  - the required value. Use true/false for "present".

Rules:
- Split distinct items into separate objects. A blue shirt worn under a suit
  is TWO objects: shirt (color=blue) and suit (present=true).
- Use "subject" as the object for gender, age, pose, and location.
- Only include what the description actually states. Do not add details.
- Do not merge two garments into one object.

Example for "a man in a blue shirt with a suit and a red tie":
[
  {{"object": "subject", "attribute": "gender", "expected": "male"}},
  {{"object": "shirt", "attribute": "color", "expected": "blue"}},
  {{"object": "shirt", "attribute": "present", "expected": true}},
  {{"object": "suit", "attribute": "present", "expected": true}},
  {{"object": "tie", "attribute": "present", "expected": true}},
  {{"object": "tie", "attribute": "color", "expected": "red"}}
]

Output only the JSON array."""


OBSERVE_PROMPT = """You are a visual observer. Report only what is actually visible in this image crop.

Step 1 - Inventory. List every object you can clearly SEE.
Step 2 - Checks. For each requested item below, report what you observe.

Requested items:
{checks}

Return a JSON object:
{{
  "inventory": ["shirt", "pants", "bag"],
  "checks": [
    {{
      "object": "<object>",
      "attribute": "<attribute>",
      "observed": <value, true, false, or "unknown">,
      "visibility": "sufficient" or "insufficient",
      "evidence": "one short sentence about what you actually see"
    }}
  ]
}}

Critical rules:
- Report only DIRECT observation. Never infer an object from context,
  from other objects, or from how formal or casual the scene looks.
- For "present": answer true only if you can see that object itself.
  Answer false only if the relevant body region is clearly visible and the
  object is not there. Otherwise answer "unknown".
- Set visibility to "insufficient" when the region is cropped out, blurred,
  occluded, or too small to judge. Then observed must be "unknown".
- If an object is not in your inventory, its "present" cannot be true.
- You are NOT told what the answer should be. Do not guess a preferred answer.

Output only the JSON object."""


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────
def empty_cuda() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def pick_device(device: Optional[str] = None) -> str:
    if device:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def minmax(xs: Sequence[float]) -> List[float]:
    xs = [float(x) for x in xs]
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [0.5] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


def blend(
    attr: Sequence[float],
    retrieval: Sequence[float],
    alpha: float,
) -> List[float]:
    """속성 점수와 검색 점수를 각각 min-max 정규화한 뒤 섞는다."""
    if alpha >= 1.0:
        return [float(a) for a in attr]
    if alpha <= 0.0:
        return [float(r) for r in retrieval]
    na, nr = minmax(attr), minmax(retrieval)
    return [alpha * a + (1.0 - alpha) * r for a, r in zip(na, nr)]


# ─────────────────────────────────────────────────────────────────────────────
# 토큰 정규화 (Python 담당)
# ─────────────────────────────────────────────────────────────────────────────
_SYNONYM_MAP: Dict[str, str] = {}
for _group in _SYNONYM_GROUPS:
    _canon = _group[0].replace(" ", "")
    for _term in _group:
        _SYNONYM_MAP[_term.replace(" ", "")] = _canon
        _SYNONYM_MAP[_term] = _canon


def normalize_tokens(text: Any) -> set:
    """문구를 비교 가능한 토큰 집합으로. unknown 류는 빈 집합이 된다."""
    if isinstance(text, bool):
        return {"true"} if text else {"false"}

    text = str(text or "").lower().strip()
    if not text:
        return set()

    for term in sorted(_SYNONYM_MAP, key=len, reverse=True):
        if " " in term and term in text:
            text = text.replace(term, _SYNONYM_MAP[term])

    text = re.sub(r"[^a-z0-9\s\-]", " ", text).replace("-", "")

    tokens = set()
    for raw in text.split():
        word = raw.strip()
        if not word or word in _STOPWORDS:
            continue
        tokens.add(_SYNONYM_MAP.get(word, word))
    return tokens


def _overlap(a: set, b: set) -> float:
    """a 의 토큰이 b 에 얼마나 있는지. 같은 색 계열은 0.5점."""
    if not a:
        return 0.0
    hit = 0.0
    for token in a:
        if token in b:
            hit += 1.0
            continue
        fam = _COLOR_FAMILY.get(token)
        if fam and any(_COLOR_FAMILY.get(t) == fam for t in b):
            hit += 0.5
    return hit / len(a)


def value_agreement(expected: Any, observed: Any) -> Optional[float]:
    """값 속성의 일치도 0~1. 비교 불가면 None.

    재현율(요구가 관찰에 있는가)과 정밀도(관찰이 요구 범위 안인가)를 섞는다.
    재현율만 쓰면 관찰이 덜 자세할 때 과도하게 감점된다.
    """
    e = normalize_tokens(expected)
    o = normalize_tokens(observed)
    if not e or not o:
        return None
    return RECALL_WEIGHT * _overlap(e, o) + PRECISION_WEIGHT * _overlap(o, e)


def to_bool(value: Any) -> Optional[bool]:
    """true / false / unknown 을 판별한다. 판별 못 하면 None."""
    if isinstance(value, bool):
        return value
    text = str(value if value is not None else "").strip().lower()
    if text in _UNKNOWN_WORDS:
        return None
    if text in _TRUE_WORDS:
        return True
    if text in _FALSE_WORDS:
        return False
    # "a blue shirt is visible" 처럼 서술로 답한 경우 -> 존재로 본다
    if normalize_tokens(text):
        return True
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Constraint
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Constraint:
    object: str
    attribute: str
    expected: Any
    weight: float = 1.0
    required: bool = False        # FAIL 이면 전체 점수를 0 으로

    @property
    def key(self) -> str:
        return f"{self.object}.{self.attribute}"

    @property
    def is_presence(self) -> bool:
        return self.attribute.strip().lower() in PRESENCE_ATTRS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object": self.object,
            "attribute": self.attribute,
            "expected": self.expected,
            "weight": self.weight,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Optional["Constraint"]:
        obj = str(d.get("object") or "").strip()
        attr = str(d.get("attribute") or "").strip()
        if not obj or not attr:
            return None
        if "expected" not in d:
            return None
        return cls(
            object=obj,
            attribute=attr,
            expected=d["expected"],
            weight=float(d.get("weight", 1.0)),
            required=bool(d.get("required", False)),
        )


def dedupe_constraints(items: List[Constraint]) -> List[Constraint]:
    seen: Dict[str, Constraint] = {}
    for c in items:
        seen.setdefault(c.key, c)
    return list(seen.values())


# ─────────────────────────────────────────────────────────────────────────────
# 비교 · 채점 (Python 담당)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_constraint(
    c: Constraint,
    check: Optional[Dict[str, Any]],
    inventory: Sequence[str],
) -> Dict[str, Any]:
    """
    하나의 constraint 를 판정한다.

    반환: {"verdict", "score", "observed", "visibility", "evidence", "note"}
    score 는 PASS 1.0 / PARTIAL 0.5 / FAIL 0.0 / UNKNOWN None.
    """
    out: Dict[str, Any] = {
        "object": c.object,
        "attribute": c.attribute,
        "expected": c.expected,
        "weight": c.weight,
        "required": c.required,
        "observed": None,
        "visibility": None,
        "evidence": None,
        "verdict": UNKNOWN,
        "score": None,
        "note": None,
    }

    if not check:
        out["note"] = "관찰 결과 없음"
        return out

    observed = check.get("observed")
    visibility = str(check.get("visibility") or "").strip().lower()
    out["observed"] = observed
    out["visibility"] = visibility or None
    out["evidence"] = check.get("evidence")

    # 보이지 않았다면 판정하지 않는다
    if visibility.startswith("insuff"):
        out["note"] = "관찰 불충분"
        return out

    if str(observed).strip().lower() in _UNKNOWN_WORDS:
        out["note"] = "관찰 unknown"
        return out

    # ---- 존재 여부 ----
    if c.is_presence:
        obs_bool = to_bool(observed)
        exp_bool = to_bool(c.expected)

        if obs_bool is None or exp_bool is None:
            out["note"] = "존재 여부 판별 불가"
            return out

        # inventory 교차검증: 목록에 없는데 present=true 라면 맥락 추론일
        # 가능성이 높다. FAIL 로 단정하지 않고 UNKNOWN 으로 내린다.
        if obs_bool and inventory:
            inv_tokens = set()
            for name in inventory:
                inv_tokens |= normalize_tokens(name)
            want = normalize_tokens(c.object)
            if want and not (want & inv_tokens):
                out["verdict"] = UNKNOWN
                out["note"] = (
                    f"present=true 인데 inventory({', '.join(inventory)[:60]}) "
                    f"에 '{c.object}' 가 없음 -> 맥락 추론 의심"
                )
                return out

        if obs_bool == exp_bool:
            out["verdict"] = PASS
            out["score"] = 1.0
        else:
            out["verdict"] = FAIL
            out["score"] = 0.0
            out["note"] = (
                "요구=있음, 관찰=없음" if exp_bool else "요구=없음, 관찰=있음"
            )
        return out

    # ---- 값 속성 ----
    agree = value_agreement(c.expected, observed)
    if agree is None:
        out["note"] = "값 비교 불가"
        return out

    if agree >= PASS_THRESHOLD:
        out["verdict"] = PASS
        out["score"] = 1.0
    elif agree >= PARTIAL_THRESHOLD:
        out["verdict"] = PASS if agree >= PASS_THRESHOLD else "PARTIAL"
        out["verdict"] = "PARTIAL"
        out["score"] = PARTIAL_SCORE
    else:
        out["verdict"] = FAIL
        out["score"] = 0.0

    out["agreement"] = round(agree, 4)
    return out


def score_observation(
    constraints: List[Constraint],
    observation: Dict[str, Any],
) -> Dict[str, Any]:
    """
    관찰 결과로 attr_score 를 계산한다.

        match    = Σ(w x score) / Σ(w)        [PASS/PARTIAL/FAIL 만]
        coverage = Σ(w)[판정됨] / Σ(w)[전체]
        attr     = match x coverage

    coverage 를 곱하는 이유: UNKNOWN 을 분모에서 그냥 빼면 "확인 못 한
    후보"가 "확인된 후보"와 같은 점수를 받는다.
    """
    inventory = observation.get("inventory") or []
    if isinstance(inventory, str):
        inventory = [inventory]
    inventory = [str(x) for x in inventory if str(x).strip()]

    by_key: Dict[str, Dict[str, Any]] = {}
    for chk in observation.get("checks") or []:
        if not isinstance(chk, dict):
            continue
        obj = str(chk.get("object") or "").strip()
        attr = str(chk.get("attribute") or "").strip()
        if obj and attr:
            by_key[f"{obj}.{attr}".lower()] = chk

    details: List[Dict[str, Any]] = []
    weighted_score = 0.0
    resolved_weight = 0.0
    total_weight = 0.0

    for c in constraints:
        total_weight += c.weight
        res = evaluate_constraint(c, by_key.get(c.key.lower()), inventory)
        details.append(res)

        if res["score"] is not None:
            weighted_score += c.weight * float(res["score"])
            resolved_weight += c.weight

    if total_weight <= 0:
        return {
            "attr_score": None, "match": None, "coverage": None,
            "details": details, "inventory": inventory,
            "failed_required": [],
        }

    coverage = resolved_weight / total_weight
    match = (weighted_score / resolved_weight) if resolved_weight > 0 else None
    attr = (match * coverage) if match is not None else None

    # 필수 조건이 FAIL 이면 0 으로 내린다.
    #
    # match x coverage 만으로는 "필수 조건"을 표현할 수 없다. 조건 4개 중
    # 하나가 FAIL 이면 0.75 가 되는데, 그것이 "정답인데 crop 이 잘려 확인
    # 못 한 후보"(coverage 0.5 -> 0.5)보다 높게 나온다. 확실히 틀린 것이
    # 모르는 것보다 위에 오는 셈이다.
    #
    # UNKNOWN 은 0 으로 만들지 않는다. 모르는 것을 틀린 것으로 취급하면
    # 잘린 crop 때문에 정답을 버린다.
    failed_required = [
        f"{d['object']}.{d['attribute']}"
        for d in details
        if d.get("required") and d["verdict"] == FAIL
    ]
    if failed_required and attr is not None:
        attr = 0.0

    return {
        "attr_score": attr,
        "match": match,
        "coverage": coverage,
        "details": details,
        "inventory": inventory,
        "failed_required": failed_required,
    }


def summarize_details(details: List[Dict[str, Any]]) -> str:
    """판정 결과를 한 줄로."""
    mark = {PASS: "O", "PARTIAL": "~", FAIL: "X", UNKNOWN: "?"}
    return "  ".join(
        f"{d['object']}.{d['attribute']}{mark.get(d['verdict'], '?')}"
        for d in details
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSON 파싱
# ─────────────────────────────────────────────────────────────────────────────
def extract_json(raw: str) -> Any:
    """모델 출력에서 JSON 을 뽑는다. 코드펜스와 앞뒤 설명문을 견딘다."""
    text = (raw or "").strip()
    if not text:
        return None

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if start >= 0 and end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.debug("JSON 파싱 실패: %r", text[:200])
        return None


def parse_constraints(raw: str) -> List[Constraint]:
    data = extract_json(raw)
    if isinstance(data, dict):
        # {"constraints": [...]} 형태도 받는다
        for key in ("constraints", "conditions", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        return []

    out: List[Constraint] = []
    for item in data:
        if isinstance(item, dict):
            c = Constraint.from_dict(item)
            if c is not None:
                out.append(c)
    return dedupe_constraints(out)


def parse_observation(raw: str) -> Dict[str, Any]:
    data = extract_json(raw)
    if not isinstance(data, dict):
        return {}

    checks = data.get("checks")
    if not isinstance(checks, list):
        checks = []

    inventory = data.get("inventory")
    if isinstance(inventory, str):
        inventory = [inventory]
    if not isinstance(inventory, list):
        inventory = []

    return {"inventory": inventory, "checks": checks}


# ─────────────────────────────────────────────────────────────────────────────
# Qwen (의미 해석 + 시각 관찰만 담당)
# ─────────────────────────────────────────────────────────────────────────────
class QwenVL:
    """Qwen3-VL 로더. 판정하지 않는다."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        dtype: str = "bfloat16",
        device: Optional[str] = None,
        max_pixels: Optional[int] = 768 * 768,
        max_new_tokens: int = 512,
    ) -> None:
        self.model_id = model_id
        self.device = pick_device(device)
        self._dtype_name = dtype
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    # ---- 로딩 ----

    def _resolve_dtype(self):
        import torch

        if not str(self.device).startswith("cuda"):
            return torch.float32

        table = {
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16,
            "float32": torch.float32, "fp32": torch.float32,
        }
        if self._dtype_name not in table:
            raise ValueError(
                f"dtype 은 {sorted(table)} 중 하나여야 합니다: {self._dtype_name}"
            )
        want = table[self._dtype_name]

        if want is torch.bfloat16:
            try:
                if not torch.cuda.is_bf16_supported():
                    logger.warning("bf16 미지원 GPU -> fp16 으로 내립니다.")
                    return torch.float16
            except Exception:
                return torch.float16
        return want

    def load(self) -> None:
        if self._model is not None:
            return

        try:
            import transformers
            from transformers import AutoProcessor
        except ImportError as e:
            raise ImportError(
                "transformers 가 필요합니다: pip install transformers"
            ) from e

        dtype = self._resolve_dtype()
        logger.info(
            "Qwen 로드: %s (%s, %s)",
            self.model_id, self.device, str(dtype).replace("torch.", ""),
        )
        t0 = time.time()

        proc_kwargs: Dict[str, Any] = {}
        if self.max_pixels is not None:
            proc_kwargs["max_pixels"] = self.max_pixels

        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, **proc_kwargs
            )
        except TypeError:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            logger.warning("이 프로세서는 max_pixels 를 지원하지 않습니다.")

        self._model = self._load_model(transformers, dtype)
        self._model.to(self.device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        logger.info("로드 완료 (%.1fs)", time.time() - t0)

    def _load_model(self, transformers, dtype):
        """transformers 버전에 따라 클래스 이름이 다르다. 순서대로 시도한다."""
        candidates = (
            "AutoModelForImageTextToText",
            "Qwen3VLForConditionalGeneration",
            "Qwen2_5_VLForConditionalGeneration",
            "AutoModelForVision2Seq",
        )

        errors: List[str] = []
        for dtype_key in ("dtype", "torch_dtype"):
            for name in candidates:
                cls = getattr(transformers, name, None)
                if cls is None:
                    continue
                try:
                    model = cls.from_pretrained(
                        self.model_id, **{dtype_key: dtype}
                    )
                    logger.info("모델 클래스: %s (%s)", name, dtype_key)
                    return model
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{name}/{dtype_key}: {type(e).__name__} {e}")

        detail = "\n  ".join(errors[:6]) or "후보 클래스가 transformers 에 없습니다."
        raise RuntimeError(
            f"'{self.model_id}' 를 로드할 수 없습니다.\n  {detail}\n"
            f"모델 id 와 transformers 버전을 확인하세요."
        )

    # ---- 생성 ----

    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        import torch

        content: List[Dict[str, Any]] = []
        images = []

        if image_path:
            from PIL import Image

            with Image.open(image_path) as im:
                images.append(im.convert("RGB").copy())
            content.append({"type": "image"})

        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        proc_kwargs: Dict[str, Any] = {
            "text": [text], "return_tensors": "pt", "padding": True,
        }
        if images:
            proc_kwargs["images"] = images

        inputs = self._processor(**proc_kwargs)
        inputs = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in dict(inputs).items()
        }

        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        return self._processor.batch_decode(
            out[:, prompt_len:], skip_special_tokens=True
        )[0].strip()

    # ---- 쿼리 해석 ----

    def parse_query(self, query: str) -> List[Constraint]:
        raw = self.generate(QUERY_PARSE_PROMPT.format(query=query))
        return parse_constraints(raw)

    # ---- 이미지 관찰 ----

    def observe(
        self,
        image_path: str,
        constraints: List[Constraint],
    ) -> Dict[str, Any]:
        """
        expected 값을 **주지 않고** (object, attribute) 만 알려준다.

        "정장이 있어야 한다" 를 알려주면 모델이 그 답을 맞추려 한다.
        검사 항목만 주고 관찰을 요구한다.
        """
        lines = [
            f"- object: {c.object}, attribute: {c.attribute}"
            for c in constraints
        ]
        prompt = OBSERVE_PROMPT.format(checks="\n".join(lines))
        raw = self.generate(prompt, image_path=image_path)
        return parse_observation(raw)

    def release(self) -> None:
        self._model = None
        self._processor = None
        empty_cuda()


# ─────────────────────────────────────────────────────────────────────────────
# 쿼리 -> constraint 확보
# ─────────────────────────────────────────────────────────────────────────────
def constraints_from_slots(slots: Dict[str, Any]) -> List["Constraint"]:
    """
    Qwen 쿼리 해석이 실패했을 때의 폴백.

    슬롯을 object 로 그대로 쓰고 attribute='description' 으로 둔다.
    object 를 쪼개지 못하므로 정확도가 낮다. 어디까지나 폴백이다.
    """
    out: List[Constraint] = []
    for name, value in (slots or {}).items():
        text = str(value or "").strip()
        if not text:
            continue
        out.append(Constraint(object=str(name), attribute="description",
                              expected=text))
    return out


def resolve_constraints(
    item: Dict[str, Any],
    qwen: Optional[QwenVL],
    reuse: bool,
) -> List[Constraint]:
    """비교에 쓸 constraint 를 확보한다."""
    # 1) 이전 실행에서 저장된 것
    stored = item.get("constraints")
    if reuse and isinstance(stored, list) and stored:
        parsed = [Constraint.from_dict(d) for d in stored if isinstance(d, dict)]
        parsed = [c for c in parsed if c is not None]
        if parsed:
            logger.info("저장된 constraint %d개 재사용", len(parsed))
            return parsed

    # 한국어 원문을 먼저 쓴다.
    #
    # 번역된 영어가 조건을 누락하는 사례가 있었다. 예를 들어
    # "우산 들고있는 꽃무늬 옷 입은 여성" 이
    # "A woman wearing a flower pattern holding an umbrella." 로 번역되면서
    # 색·재질 정보가 사라졌다. Qwen 은 한국어를 직접 읽을 수 있으므로
    # 원문에서 constraint 를 뽑는 편이 손실이 적다.
    query = (
        item.get("query_text_original")
        or item.get("query_text")
        or ""
    )
    query = str(query).strip()

    # 2) Qwen 이 자연어를 해석 -> 전부 required
    #
    # 검색어에 명시적으로 들어간 조건이므로 사용자가 매번 지정할 이유가 없다.
    # "정장을 입은" 이라고 썼으면 suit.present=true 는 필수 조건이다.
    if qwen is not None and query:
        try:
            parsed = qwen.parse_query(query)
        except Exception as e:  # noqa: BLE001
            logger.warning("쿼리 해석 실패: %s: %s", type(e).__name__, e)
            parsed = []
        if parsed:
            for c in parsed:
                c.required = True
            logger.info(
                "constraint %d개 (모두 필수): %s",
                len(parsed),
                ", ".join(f"{c.key}={c.expected}" for c in parsed),
            )
            return parsed
        logger.warning("Qwen 쿼리 해석이 비었습니다. 슬롯 폴백을 씁니다.")

    # 3) 슬롯 폴백 -> required=False (soft)
    #
    # 폴백은 object 를 쪼개지 못한 상태다. "파란 셔츠에 정장" 이
    # top.description 한 칸에 뭉쳐 있고 비교도 토큰 겹침이므로,
    # 여기에 hard constraint 를 걸면 부정확한 조건 때문에 정답이
    # 하위로 강제될 수 있다.
    #
    #     신뢰할 수 있게 구조화된 조건 -> hard
    #     구조화 실패 후 임시 폴백     -> soft
    fallback = constraints_from_slots(item.get("query_slots") or {})
    if fallback:
        logger.warning(
            "슬롯 폴백 constraint %d개를 씁니다 (soft, required 아님). "
            "object 를 쪼개지 못하므로 정확도가 낮습니다.", len(fallback),
        )
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# 판정 (Python 담당)
# ─────────────────────────────────────────────────────────────────────────────
def apply_verdict(
    rows: List[Dict[str, Any]],
    threshold: float,
    mode: str,
) -> List[Dict[str, Any]]:
    for row in rows:
        s = row.get("attr_score")
        row["verified"] = None if s is None else (float(s) >= threshold)

    n_fail = sum(1 for r in rows if r.get("verified") is False)

    if mode == "filter" and n_fail:
        # UNKNOWN(None)은 제거하지 않는다. 판정 못 한 것을 미달로 취급하면
        # 조용한 손실이 생긴다.
        rows = [r for r in rows if r.get("verified") is not False]
        logger.warning(
            "verify_mode='filter': threshold %.2f 미달 %d건을 제거했습니다. "
            "threshold 가 캘리브레이션되지 않았다면 정답을 버릴 수 있습니다.",
            threshold, n_fail,
        )
    elif n_fail:
        logger.info(
            "threshold %.2f 미달 %d건 (표시만, 제거하지 않음)", threshold, n_fail
        )

    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def rank_shift(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    up = down = same = 0
    biggest_up = biggest_down = 0
    for r in rows:
        old, new = r.get("pre_qwen_rank"), r.get("rank")
        if not old or not new:
            continue
        d = int(old) - int(new)
        if d > 0:
            up += 1
            biggest_up = max(biggest_up, d)
        elif d < 0:
            down += 1
            biggest_down = min(biggest_down, d)
        else:
            same += 1
    return {
        "moved_up": up, "moved_down": down, "unchanged": same,
        "biggest_up": biggest_up, "biggest_down": abs(biggest_down),
    }


def process_item(
    item: Dict[str, Any],
    qwen: Optional[QwenVL],
    constraints: List[Constraint],
    top_k: int,
    alpha: float,
    threshold: float,
    verify_mode: str,
    rescore_only: bool,
) -> Dict[str, Any]:
    rows = item.get("results") or []
    if not rows:
        return item

    if not constraints:
        item["qwen_error"] = "constraint 를 만들 수 없습니다"
        return item

    item["constraints"] = [c.to_dict() for c in constraints]

    head, tail = rows[:top_k], rows[top_k:]
    scored: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []

    t0 = time.time()
    for i, row in enumerate(head, 1):
        path = row.get("crop_path")

        if rescore_only:
            observation = row.get("qwen_observation")
            if not observation:
                row.update(
                    attr_score=None, verified=None,
                    attr_skipped="no stored observation",
                )
                unresolved.append(row)
                continue
        else:
            if not path or not Path(path).is_file():
                row.update(
                    attr_score=None, verified=None,
                    attr_skipped="crop file not found",
                )
                unresolved.append(row)
                continue
            try:
                observation = qwen.observe(path, constraints)
            except Exception as e:  # noqa: BLE001 — 후보 하나 실패해도 계속
                logger.warning("관찰 실패 (%s): %s", path, e)
                row.update(
                    attr_score=None, verified=None,
                    attr_skipped=f"{type(e).__name__}: {e}",
                )
                unresolved.append(row)
                continue
            row["qwen_observation"] = observation

        if not observation.get("checks"):
            row.update(
                attr_score=None, verified=None,
                attr_skipped="observation parsing failed",
            )
            unresolved.append(row)
            continue

        result = score_observation(constraints, observation)
        row["attr_score"] = result["attr_score"]
        row["attr_match"] = result["match"]
        row["attr_coverage"] = result["coverage"]
        row["attr_details"] = result["details"]
        row["attr_inventory"] = result["inventory"]
        row["attr_summary"] = summarize_details(result["details"])
        row["failed_required"] = result.get("failed_required") or []
        # view_results.py 호환
        row["qwen_score"] = result["attr_score"]

        if result["attr_score"] is None:
            row["verified"] = None
            row["attr_skipped"] = "판정 가능한 조건이 없음"
            unresolved.append(row)
        else:
            scored.append(row)

        if not rescore_only and i % 5 == 0:
            per = (time.time() - t0) / i
            logger.info(
                "  %d/%d (%.2fs/건, 남음 %.0fs)",
                i, len(head), per, per * (len(head) - i),
            )

    if not scored:
        logger.warning("판정 가능한 후보가 없습니다.")
        item["results"] = apply_verdict(head + tail, threshold, verify_mode)
        return item

    finals = blend(
        [float(r["attr_score"]) for r in scored],
        [
            float(r.get("pre_qwen_score", r.get("qdrant_score", 0.0)))
            for r in scored
        ],
        alpha,
    )
    for row, final in zip(scored, finals):
        row["final_score"] = float(final)

    # 필수 조건 FAIL 후보를 정상 후보 아래로 강제한다.
    #
    # attr_score 를 0 으로 만드는 것만으로는 부족하다. alpha < 1 이면
    # final_score 에 Qdrant 점수가 남아 있어서, 검색 점수가 아주 높은
    # 후보는 명시적 조건을 위반했는데도 다시 위로 올라올 수 있다.
    #
    # UNKNOWN 은 강등 대상이 아니다 (unresolved 로 빠지거나 coverage 로만
    # 반영된다).
    scored.sort(
        key=lambda r: (
            not bool(r.get("failed_required")),   # FAIL 없는 쪽이 위
            r["final_score"],
        ),
        reverse=True,
    )

    n_demoted = sum(1 for r in scored if r.get("failed_required"))
    if n_demoted:
        logger.info(
            "명시적 조건 위반 %d건을 정상 후보 아래로 내렸습니다.", n_demoted
        )

    item["results"] = apply_verdict(
        scored + unresolved + tail, threshold, verify_mode
    )
    return item


def run(
    payload: Dict[str, Any],
    *,
    top_k: int,
    alpha: float,
    threshold: float,
    verify_mode: str,
    rescore_only: bool,
    weights: Dict[str, float],
    required: set,
    soft: set,
    model_id: str,
    dtype: str,
    device: Optional[str],
    max_pixels: Optional[int],
) -> Dict[str, Any]:
    crops = payload.get("crops") or []

    text_items = [
        c for c in crops
        if c.get("results") and (c.get("query_text") or c.get("kind") == "text")
    ]
    other = [c for c in crops if c.get("results") and c not in text_items]

    if other:
        logger.warning(
            "텍스트 쿼리가 아닌 항목 %d건은 건너뜁니다 (kind=%s). "
            "이 파일은 자연어 검색 결과만 처리합니다.",
            len(other), sorted({str(c.get("kind")) for c in other}),
        )

    if not text_items:
        logger.warning(
            "재채점할 텍스트 쿼리 결과가 없습니다. "
            "search_db.py --json-out 으로 만든 JSON 인지 확인하세요."
        )
        payload["qwen"] = False
        return payload

    n_cand = sum(min(top_k, len(c["results"])) for c in text_items)
    logger.info(
        "쿼리 %d건 / 후보 %d건 %s (alpha=%.2f)",
        len(text_items), n_cand,
        "재채점(저장된 관찰 재사용)" if rescore_only else "관찰 + 채점",
        alpha,
    )

    qwen: Optional[QwenVL] = None
    if not rescore_only:
        qwen = QwenVL(
            model_id=model_id, dtype=dtype, device=device,
            max_pixels=max_pixels,
        )
        qwen.load()

    t0 = time.time()
    for item in text_items:
        constraints = resolve_constraints(item, qwen, reuse=rescore_only)

        # object 별 가중치 / 필수 여부 override
        for c in constraints:
            if c.object in weights:
                c.weight = weights[c.object]
            if c.key in weights:
                c.weight = weights[c.key]
            if c.object in required or c.key in required:
                c.required = True
            if c.object in soft or c.key in soft:
                c.required = False

        known = {n for c in constraints for n in (c.object, c.key)}
        for label, given in (("--require", required), ("--soft", soft)):
            missing = given - known
            if missing:
                logger.warning(
                    "%s 로 지정했지만 constraint 에 없는 이름: %s\n"
                    "  실제 조건 목록: %s",
                    label, sorted(missing),
                    ", ".join(c.key for c in constraints),
                )

        process_item(
            item, qwen, constraints,
            top_k=top_k, alpha=alpha, threshold=threshold,
            verify_mode=verify_mode, rescore_only=rescore_only,
        )
    elapsed = time.time() - t0

    if qwen is not None:
        qwen.release()

    shifts = [rank_shift(c["results"][:top_k]) for c in text_items]

    payload["qwen"] = True
    payload["qwen_mode"] = "constraint-observe-compare"
    payload["qwen_model"] = None if rescore_only else model_id
    payload["qwen_top_k"] = top_k
    payload["qwen_alpha"] = alpha
    payload["constraint_weights"] = weights
    payload["required_override"] = sorted(required)
    payload["soft_override"] = sorted(soft)
    payload["verify_threshold"] = threshold
    payload["verify_mode"] = verify_mode
    payload["rescore_only"] = rescore_only
    payload["qwen_elapsed_sec"] = round(elapsed, 3)
    payload["qwen_rank_shift"] = shifts[0] if len(shifts) == 1 else shifts

    if n_cand:
        logger.info("완료: %.2fs (%.2fs/건)", elapsed, elapsed / n_cand)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
def print_results(payload: Dict[str, Any], show: int = 20,
                  verbose: bool = False) -> None:
    print()
    print("=" * 88)
    print(f"  쿼리     : {payload.get('query')}")
    if payload.get("query_en") and payload["query_en"] != payload.get("query"):
        print(f"  검색문장 : {payload['query_en']}")
    print(
        f"  방식     : constraint 관찰 + Python 비교 "
        f"(top_k={payload.get('qwen_top_k')} "
        f"alpha={payload.get('qwen_alpha')} "
        f"threshold={payload.get('verify_threshold')} "
        f"mode={payload.get('verify_mode')})"
    )
    if payload.get("rescore_only"):
        print("  모델     : 미사용 (저장된 관찰 재사용)")
    elif payload.get("qwen_model"):
        print(f"  모델     : {payload['qwen_model']}")
    print(f"  소요     : {payload.get('qwen_elapsed_sec')}s")

    shift = payload.get("qwen_rank_shift")
    if isinstance(shift, dict):
        print(
            f"  순위변화 : 상승 {shift['moved_up']}건 / "
            f"하락 {shift['moved_down']}건 / 유지 {shift['unchanged']}건 "
            f"(최대 상승 {shift['biggest_up']}, 최대 하락 {shift['biggest_down']})"
        )
    print("=" * 88)

    n_ok = n_no = n_skip = 0

    for item in payload.get("crops") or []:
        rows = item.get("results") or []
        if not rows:
            continue

        if item.get("qwen_error"):
            print()
            print(f"  {item['qwen_error']}")
            continue

        cons = item.get("constraints") or []
        if cons:
            print()
            print("  조건 :")
            for c in cons:
                bits = []
                w = c.get("weight", 1.0)
                if abs(float(w) - 1.0) > 1e-9:
                    bits.append(f"w={w}")
                if c.get("required"):
                    bits.append("필수")
                extra = f"  ({', '.join(bits)})" if bits else ""
                print(
                    f"    {c['object']}.{c['attribute']} = "
                    f"{c['expected']}{extra}"
                )

        print()
        for row in rows[:show]:
            verified = row.get("verified")
            if verified is True:
                mark = "[확인]  "
                n_ok += 1
            elif verified is False:
                mark = "[미달]  "
                n_no += 1
            else:
                mark = "[미검증]"
                n_skip += 1

            parts = []
            if row.get("final_score") is not None:
                parts.append(f"final={row['final_score']:.4f}")
            if row.get("attr_score") is not None:
                parts.append(f"attr={row['attr_score']:.4f}")
            if row.get("attr_match") is not None:
                parts.append(f"match={row['attr_match']:.2f}")
            if row.get("attr_coverage") is not None:
                parts.append(f"cov={row['attr_coverage']:.2f}")
            parts.append(f"qdrant={row.get('qdrant_score', 0.0):.4f}")

            old, new = row.get("pre_qwen_rank"), row.get("rank")
            move = ""
            if old and new:
                d = int(old) - int(new)
                if d > 0:
                    move = f"  (up {d})"
                elif d < 0:
                    move = f"  (down {abs(d)})"

            print(f"  {new:>3}위 {mark} {'  '.join(parts)}{move}")

            if row.get("attr_summary"):
                print(f"        {row['attr_summary']}")

            if row.get("failed_required"):
                print(
                    f"        필수 조건 FAIL: "
                    f"{', '.join(row['failed_required'])} -> 점수 0"
                )

            inv = row.get("attr_inventory")
            if inv:
                print(f"        inventory: {', '.join(map(str, inv))[:110]}")

            if verbose:
                for d in row.get("attr_details") or []:
                    note = f"  ({d['note']})" if d.get("note") else ""
                    print(
                        f"          {d['object']}.{d['attribute']}: "
                        f"{d['verdict']}  expected={d['expected']!r} "
                        f"observed={d['observed']!r}{note}"
                    )
                    if d.get("evidence"):
                        print(f"            {d['evidence']}")

            if row.get("crop_path"):
                print(f"        {Path(row['crop_path']).name}")
            if row.get("attr_skipped"):
                print(f"        skipped: {row['attr_skipped']}")

        if len(rows) > show:
            print(f"  ... 외 {len(rows) - show}건")

    print()
    print(f"  검증: 확인 {n_ok}건 / 미달 {n_no}건 / 미검증 {n_skip}건")
    print("  기호: O PASS  ~ PARTIAL  X FAIL  ? UNKNOWN")
    print("  attr = match x coverage. coverage 가 낮으면 조건을 확인하지 "
          "못한 것이다.")
    print("  검색어에 명시된 조건은 자동으로 필수다. FAIL 이면 점수가 0 이 "
          "되고 정상 후보 아래로 내려간다 (UNKNOWN 은 제외).")
    print(
        "  가중치·threshold 만 바꿀 때는 --rescore-only 를 쓰면 Qwen 을 "
        "재호출하지 않습니다."
    )
    print("  판정 근거를 보려면 --verbose 를 쓰세요.")
    print()


def parse_weight_args(pairs: Optional[List[str]]) -> Dict[str, float]:
    """--weight suit=2 또는 --weight suit.present=2 형태."""
    weights: Dict[str, float] = {}
    for raw in pairs or []:
        if "=" not in raw:
            raise ValueError(f"--weight 형식은 name=value 입니다: {raw}")
        name, value = raw.rsplit("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"--weight 이름이 비었습니다: {raw}")
        weights[name] = float(value)
    return weights


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    ap = argparse.ArgumentParser(
        description=(
            "Qwen 이 자연어를 constraint 로 해석하고 이미지를 관찰하며, "
            "Python 이 비교·채점·판정한다. Qwen 에게 판정을 맡기지 않는다."
        )
    )
    ap.add_argument("--in", dest="inp", required=True,
                    help="search_db.py --json-out 으로 만든 JSON")
    ap.add_argument("--out", default=None, help="결과 JSON 저장 경로")

    ap.add_argument("--top-k", type=int, default=20,
                    help="관찰·채점할 상위 후보 수")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="1.0=속성 점수만, 0.0=검색 점수만")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="확인/미달 판정 기준 (근거 없는 시작값)")
    ap.add_argument("--verify-mode", default="flag", choices=list(VERIFY_MODES),
                    help="flag=표시만(기본), filter=미달 제거")

    ap.add_argument("--weight", action="append", default=None,
                    metavar="NAME=VALUE",
                    help="object 또는 object.attribute 가중치 "
                         "(예: --weight suit=2 --weight subject.gender=1.5)")

    ap.add_argument("--require", action="append", default=None,
                    metavar="NAME",
                    help="폴백 constraint 를 강제로 필수로 올린다. "
                         "Qwen 이 해석한 constraint 는 이미 전부 필수이므로 "
                         "보통 쓸 일이 없다. "
                         "(예: --require top.description)")
    ap.add_argument("--soft", action="append", default=None,
                    metavar="NAME",
                    help="반대로 특정 조건을 필수에서 뺀다. Qwen 이 잘못 뽑은 "
                         "조건 때문에 정답이 강등될 때 쓴다. "
                         "(예: --soft place --soft subject.pose)")
    ap.add_argument("--rescore-only", action="store_true",
                    help="저장된 관찰·constraint 로 다시 채점만 한다 "
                         "(Qwen 호출 0회). 가중치/alpha/threshold 실험용")

    ap.add_argument("--model-id", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None, help="cuda | cpu")
    ap.add_argument("--max-pixels", type=int, default=768 * 768,
                    help="crop 을 더 크게 보여주려면 올린다 (관찰 정확도 향상, "
                         "속도 저하)")

    ap.add_argument("--show", type=int, default=20)
    ap.add_argument("--verbose", action="store_true",
                    help="조건별 판정 근거를 모두 출력")
    ap.add_argument("--json", action="store_true", help="stdout 으로 JSON 출력")
    args = ap.parse_args()

    if not 0 <= args.alpha <= 1:
        ap.error("--alpha must be 0~1")
    if not 0 <= args.threshold <= 1:
        ap.error("--threshold must be 0~1")
    if args.top_k <= 0:
        ap.error("--top-k must be > 0")

    try:
        weights = parse_weight_args(args.weight)
    except ValueError as e:
        ap.error(str(e))

    src = Path(args.inp)
    if not src.is_file():
        ap.error(f"입력 JSON 이 없습니다: {src}")

    payload = json.loads(src.read_text(encoding="utf-8"))

    if payload.get("search_type") and payload["search_type"] != "text":
        logger.warning(
            "search_type='%s' 입니다. 이 파일은 자연어 검색(text) 결과를 "
            "처리합니다.", payload["search_type"],
        )

    payload = run(
        payload,
        top_k=args.top_k,
        alpha=args.alpha,
        threshold=args.threshold,
        verify_mode=args.verify_mode,
        rescore_only=args.rescore_only,
        weights=weights,
        required={str(x).strip() for x in (args.require or []) if str(x).strip()},
        soft={str(x).strip() for x in (args.soft or []) if str(x).strip()},
        model_id=args.model_id,
        dtype=args.dtype,
        device=args.device,
        max_pixels=args.max_pixels,
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_results(payload, show=args.show, verbose=args.verbose)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON saved: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())