"""
쿼리 번역 · 선택적 구조화 조립
=============================

    한국어 자유 쿼리 -> [QueryTranslator] -> 영어 자유 문장
    항목별 입력       -> [QueryDescriptor] -> 영어 구조화 문장

이 파일의 핵심 역할은 한국어 검색 query 를 영어로 번역하는 것이다.
QueryDescriptor 는 구조화된 슬롯 입력을 영어 문장으로 조립하는 선택 기능이며,
특정 임베딩 모델의 성능 향상을 보장하거나 다른 모델의 역할을 정의하지 않는다.

권장 사용:
    tr = QueryTranslator()
    en = tr.translate("빨간 재킷을 입은 남성")

선택적 구조화 입력:
    desc = QueryDescriptor(tr)
    result = desc.build(
        gender="여성", hair="짧은 검은 곱슬",
        top="화려한 꽃무늬 민소매 원피스",
        carry="큰 분홍 양산", place="야외 맑은 날",
    )
    caption = result["caption"]
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BACKENDS = ("opus", "nllb", "none")

DEFAULT_OPUS = "Helsinki-NLP/opus-mt-ko-en"
DEFAULT_NLLB = "facebook/nllb-200-distilled-600M"

# 한글 음절 + 자모
_HANGUL = re.compile(r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]")


def has_hangul(text: str) -> bool:
    return bool(_HANGUL.search(text or ""))


# ─────────────────────────────────────────────────────────────────────────────
# 번역
# ─────────────────────────────────────────────────────────────────────────────
class QueryTranslator:
    """한국어 -> 영어 번역기 (지연 로딩).

    backend:
        opus (기본) Helsinki-NLP/opus-mt-ko-en. 약 300MB, 빠르다.
        nllb        facebook/nllb-200-distilled-600M. 크지만 문장 품질이 낫다.
        none        번역하지 않는다 (영어로 직접 질의할 때).

    ※ 모델 이름은 배포명이 바뀔 수 있으므로 확인 후 쓸 것.
    """

    def __init__(
        self,
        backend: str = "opus",
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        max_new_tokens: int = 64,
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend 는 {list(BACKENDS)} 중 하나여야 합니다.")

        self.backend = backend
        self.model_id = model_id or (
            DEFAULT_OPUS if backend == "opus"
            else DEFAULT_NLLB if backend == "nllb"
            else None
        )
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        if int(max_new_tokens) <= 0:
            raise ValueError("max_new_tokens 는 1 이상이어야 합니다.")
        self.max_new_tokens = int(max_new_tokens)

        # 번역 모델은 CPU 고정.
        # device 인자는 기존 호출부 호환성을 위해 남기지만 CPU 외 값은 사용하지 않는다.
        if device not in (None, "cpu"):
            logger.warning(
                "QueryTranslator는 CPU 고정입니다. device=%r 설정은 무시합니다.",
                device,
            )
        self._device = "cpu"
        self._model = None
        self._tokenizer = None
        self._cache: Dict[str, str] = {}

    # ---- 로딩 ----

    def _resolve_device(self) -> str:
        return "cpu"

    def _load(self) -> None:
        if self._model is not None or self.backend == "none":
            return

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers 가 필요합니다: "
                "pip install transformers sentencepiece"
            ) from e

        kwargs = {}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        if self.local_files_only:
            kwargs["local_files_only"] = True

        device = self._resolve_device()
        logger.info("번역 모델 로드: %s (%s)", self.model_id, device)

        tok_kwargs = dict(kwargs)
        if self.backend == "nllb":
            # NLLB 는 원문 언어를 토크나이저에 알려줘야 한다
            tok_kwargs["src_lang"] = "kor_Hang"

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tok_kwargs)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, **kwargs)
        self._model.to(device).eval()

    def release(self) -> None:
        self._model = None
        self._tokenizer = None
        self._cache.clear()

    # ---- 번역 ----

    def translate(self, text: str, force: bool = False) -> str:
        """한국어면 영어로. 한글이 없으면 원문 그대로.

        force=True 면 한글 여부와 무관하게 번역을 시도한다.
        같은 문구를 여러 슬롯에서 반복 번역하지 않도록 캐시한다.
        """
        text = (text or "").strip()
        if not text or self.backend == "none":
            return text

        if not force and not has_hangul(text):
            logger.debug("한글 없음 -> 번역 건너뜀")
            return text

        cached = self._cache.get(text)
        if cached is not None:
            return cached

        self._load()

        import torch

        device = self._resolve_device()
        # 긴 자유 입력이 번역 모델의 최대 입력 길이를 넘지 않도록 안전하게 자른다.
        # max_length 를 직접 고정하지 않고 tokenizer/model 의 설정값을 따른다.
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
        ).to(device)

        gen_kwargs = {"max_new_tokens": self.max_new_tokens, "num_beams": 4}
        if self.backend == "nllb":
            forced = self._forced_bos_id("eng_Latn")
            if forced is not None:
                gen_kwargs["forced_bos_token_id"] = forced

        with torch.inference_mode():
            out = self._model.generate(**inputs, **gen_kwargs)

        result = self._tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

        if not result:
            logger.warning("번역 결과가 비었습니다. 원문을 그대로 씁니다: %r", text)
            return text

        logger.info("번역: %r -> %r", text, result)
        self._cache[text] = result
        return result

    def _forced_bos_id(self, lang_code: str) -> Optional[int]:
        """NLLB 목표 언어 토큰 id. transformers 버전에 따라 접근법이 다르다."""
        tok = self._tokenizer

        getter = getattr(tok, "convert_tokens_to_ids", None)
        if callable(getter):
            tid = getter(lang_code)
            unk = getattr(tok, "unk_token_id", None)
            if tid is not None and tid != unk:
                return int(tid)

        lang_map = getattr(tok, "lang_code_to_id", None)
        if isinstance(lang_map, dict) and lang_code in lang_map:
            return int(lang_map[lang_code])

        logger.warning(
            "NLLB 목표 언어 토큰 '%s' 을 찾지 못했습니다. "
            "번역 결과 언어가 어긋날 수 있습니다.", lang_code,
        )
        return None

    # ---- 자유 문장 확장 (약한 보정) ----

    def expand(self, english: str) -> str:
        """짧은 영어 문장을 단순한 사람 묘사 형태로 감싸는 선택 기능.

        번역 자체에는 필요하지 않으며 자동으로 호출하지 않는다.
        검색 품질 향상은 보장하지 않으므로 호출부에서 필요할 때만 사용한다.
        """
        english = (english or "").strip()
        if not english:
            return english

        words = english.split()
        lower = english.lower()

        starts_with_subject = lower.startswith((
            "a man", "a woman", "a person", "a boy", "a girl",
            "the man", "the woman", "the person",
        ))

        if len(words) >= 12 or starts_with_subject:
            return english

        expanded = f"A person wearing {english}."
        logger.info("쿼리 확장(효과 제한적): %r -> %r", english, expanded)
        return expanded


# ─────────────────────────────────────────────────────────────────────────────
# 어휘 사전 — 1:1 번역만. 문장을 만들지 않는다.
# ─────────────────────────────────────────────────────────────────────────────
# 긴 표현이 짧은 표현에 먹히지 않도록, 치환은 길이 내림차순으로 적용한다.

_COLORS = {
    "검정": "black", "검은": "black", "검정색": "black", "블랙": "black",
    "흰": "white", "하얀": "white", "흰색": "white", "화이트": "white",
    "빨간": "red", "빨강": "red", "붉은": "red", "레드": "red",
    "파란": "blue", "파랑": "blue", "푸른": "blue", "블루": "blue",
    "노란": "yellow", "노랑": "yellow", "옐로": "yellow",
    "초록": "green", "녹색": "green", "그린": "green",
    "분홍": "pink", "핑크": "pink",
    "보라": "purple", "자주": "purple", "퍼플": "purple",
    "주황": "orange", "오렌지": "orange",
    "갈색": "brown", "브라운": "brown", "밤색": "brown",
    "회색": "gray", "그레이": "gray", "쥐색": "gray",
    "베이지": "beige", "남색": "navy", "네이비": "navy",
    "하늘색": "light blue", "연두": "light green",
    "은색": "silver", "금색": "gold", "카키": "khaki",
    "짙은": "dark", "어두운": "dark", "진한": "dark",
    "밝은": "bright", "연한": "light", "옅은": "light",
    "알록달록한": "colorful", "화려한": "colorful", "다채로운": "colorful",
}

# 크기·정도 형용사. 모든 물리 슬롯에 공통으로 적용된다.
_SIZE = {
    "큰": "large", "커다란": "large", "대형": "large",
    "작은": "small", "소형": "small", "자그마한": "small",
    "긴": "long", "짧은": "short",
    "두꺼운": "thick", "얇은": "thin",
    "넓은": "wide", "좁은": "narrow",
    "새": "new", "낡은": "worn", "헌": "worn",
}

_PATTERNS = {
    "꽃무늬": "floral", "플로럴": "floral",
    "체크": "plaid", "격자": "plaid",
    "줄무늬": "striped", "스트라이프": "striped",
    "물방울": "polka dot", "도트": "polka dot",
    "민무늬": "plain", "단색": "solid",
    "무늬": "patterned", "패턴": "patterned",
    "프린트": "printed", "그림": "graphic",
    "로고": "logo", "글자": "lettered",
    "반짝이는": "shiny", "가죽": "leather", "데님": "denim",
    "청": "denim", "니트": "knit", "털": "furry",
}

_UPPER = {
    "티셔츠": "t-shirt", "티": "t-shirt", "반팔": "short-sleeved shirt",
    "긴팔": "long-sleeved shirt", "셔츠": "shirt", "남방": "shirt",
    "블라우스": "blouse", "니트": "knit sweater", "스웨터": "sweater",
    "맨투맨": "sweatshirt", "후드": "hoodie", "후드티": "hoodie",
    "재킷": "jacket", "자켓": "jacket", "점퍼": "jacket",
    "코트": "coat", "패딩": "padded jacket", "가디건": "cardigan",
    "조끼": "vest", "베스트": "vest", "정장": "suit", "양복": "suit",
    "원피스": "dress", "드레스": "dress", "민소매": "sleeveless",
    "탱크톱": "tank top", "나시": "sleeveless top",
    "교복": "school uniform", "유니폼": "uniform",
    "상의": "top", "옷": "clothes",
}

_LOWER = {
    "바지": "pants", "청바지": "jeans", "진": "jeans",
    "반바지": "shorts", "숏팬츠": "shorts",
    "슬랙스": "slacks", "면바지": "chino pants",
    "치마": "skirt", "스커트": "skirt",
    "미니스커트": "miniskirt", "롱스커트": "long skirt",
    "레깅스": "leggings", "트레이닝복": "track pants",
    "정장바지": "dress pants", "하의": "bottoms",
}

_FOOTWEAR = {
    "운동화": "sneakers", "스니커즈": "sneakers",
    "구두": "dress shoes", "하이힐": "high heels", "힐": "heels",
    "부츠": "boots", "슬리퍼": "slippers", "샌달": "sandals",
    "샌들": "sandals", "장화": "rain boots", "신발": "shoes",
}

_CARRY = {
    "가방": "bag", "백팩": "backpack", "배낭": "backpack",
    "책가방": "backpack", "핸드백": "handbag", "손가방": "handbag",
    "숄더백": "shoulder bag", "크로스백": "crossbody bag",
    "캐리어": "suitcase", "여행가방": "suitcase", "트렁크": "suitcase",
    "우산": "umbrella", "양산": "parasol umbrella",
    "휴대폰": "cell phone", "핸드폰": "cell phone", "스마트폰": "cell phone",
    "물병": "bottle", "병": "bottle", "컵": "cup", "텀블러": "tumbler",
    "노트북": "laptop", "책": "book", "서류": "documents",
    "봉투": "bag", "쇼핑백": "shopping bag", "박스": "box", "상자": "box",
    "자전거": "bicycle", "스케이트보드": "skateboard",
    "카메라": "camera", "지갑": "wallet",
}

_HAIR = {
    "긴": "long", "짧은": "short", "단발": "shoulder-length",
    "곱슬": "curly", "웨이브": "wavy", "생머리": "straight",
    "직모": "straight", "묶은": "tied", "포니테일": "ponytail",
    "대머리": "bald", "삭발": "shaved",
    "머리": "hair", "머리카락": "hair",
}

_ACCESSORY = {
    "모자": "hat", "캡": "cap", "야구모자": "baseball cap",
    "안경": "glasses", "선글라스": "sunglasses",
    "마스크": "face mask", "목도리": "scarf", "스카프": "scarf",
    "장갑": "gloves", "시계": "watch", "목걸이": "necklace",
    "귀걸이": "earrings", "헤어밴드": "headband",
}

_PLACE = {
    "실내": "indoors", "안": "indoors",
    "실외": "outdoors", "야외": "outdoors", "밖": "outdoors",
    "길": "on a street", "거리": "on a street", "도로": "on a road",
    "인도": "on a sidewalk", "횡단보도": "at a crosswalk",
    "공원": "in a park", "해변": "at a beach", "바다": "at the sea",
    "지하철": "in a subway station", "역": "at a station",
    "버스": "at a bus stop", "정류장": "at a bus stop",
    "주차장": "in a parking lot", "계단": "on stairs",
    "건물": "near a building", "가게": "in a shop", "상점": "in a shop",
    "시장": "at a market", "학교": "at a school",
    "낮": "in the daytime", "밤": "at night",
    "맑은": "on a sunny day", "흐린": "on a cloudy day",
    "비": "in the rain", "눈": "in the snow",
}

_POSE = {
    "서있는": "standing", "선": "standing", "서서": "standing",
    "걷는": "walking", "걸어가는": "walking", "걸으며": "walking",
    "뛰는": "running", "달리는": "running",
    "앉은": "sitting", "앉아있는": "sitting", "앉아서": "sitting",
    "기댄": "leaning", "누운": "lying down",
    "뒤돌아선": "facing away", "뒤에서": "seen from behind",
    "옆": "seen from the side", "정면": "facing the camera",
}

# 슬롯별로 어떤 사전들을 참조할지. 앞쪽 사전이 우선한다.
_SLOT_LEXICONS: Dict[str, Tuple[Dict[str, str], ...]] = {
    "gender": ({},),
    "hair": (_HAIR, _COLORS, _SIZE),
    "top": (_UPPER, _PATTERNS, _COLORS, _SIZE),
    "bottom": (_LOWER, _PATTERNS, _COLORS, _SIZE),
    "footwear": (_FOOTWEAR, _PATTERNS, _COLORS, _SIZE),
    "carry": (_CARRY, _PATTERNS, _COLORS, _SIZE),
    "accessory": (_ACCESSORY, _PATTERNS, _COLORS, _SIZE),
    "pose": (_POSE,),
    "place": (_PLACE, _SIZE),
}

# 성별/연령은 문장의 주어가 되므로 따로 다룬다.
_SUBJECT = {
    "남성": "man", "남자": "man", "남": "man",
    "여성": "woman", "여자": "woman", "여": "woman",
    "소년": "boy", "남아": "boy", "남학생": "male student",
    "소녀": "girl", "여아": "girl", "여학생": "female student",
    "아이": "child", "어린이": "child", "아동": "child",
    "청년": "young man", "젊은남자": "young man",
    "젊은여자": "young woman", "노인": "elderly person",
    "할아버지": "elderly man", "할머니": "elderly woman",
    "사람": "person",
}

SLOT_ORDER = (
    "gender", "hair", "top", "bottom", "footwear",
    "accessory", "carry", "pose", "place",
)

SLOT_LABELS = {
    "gender": "성별/연령",
    "hair": "머리",
    "top": "상의",
    "bottom": "하의",
    "footwear": "신발",
    "accessory": "액세서리",
    "carry": "소지품",
    "pose": "자세",
    "place": "장소/배경",
}


# ─────────────────────────────────────────────────────────────────────────────
# 서술형 조립
# ─────────────────────────────────────────────────────────────────────────────
class QueryDescriptor:
    """항목별 입력을 영어 구조화 문장으로 조립한다.

    조립 결과 예:

        A woman with short dark curly hair, wearing a colorful floral
        sleeveless dress, wearing sneakers, holding a large pink parasol
        umbrella, standing outdoors on a sunny day.

    빈 슬롯은 문장에서 빠진다. 사전에 없는 단어는 번역기로 폴백하므로
    어떤 어휘가 들어와도 동작한다.
    """

    def __init__(self, translator: Optional[QueryTranslator] = None) -> None:
        # 번역기를 공유하면 캐시도 함께 쓰인다
        self.translator = translator or QueryTranslator()
        self._unmapped: List[str] = []

    # ---- 어휘 변환 ----

    @staticmethod
    def _apply_lexicon(text: str, lexicon: Dict[str, str]) -> str:
        """긴 표현부터 치환한다 (짧은 키에 먹히지 않도록)."""
        for ko in sorted(lexicon, key=len, reverse=True):
            if ko in text:
                text = text.replace(ko, f" {lexicon[ko]} ")
        return text

    def _to_english(self, raw: str, slot: str) -> str:
        """슬롯 값을 영어로. 사전 우선, 남은 한글은 번역기 폴백."""
        text = (raw or "").strip()
        if not text:
            return ""

        for lexicon in _SLOT_LEXICONS.get(slot, ()):
            if lexicon:
                text = self._apply_lexicon(text, lexicon)

        text = re.sub(r"\s+", " ", text).strip()

        # 사전이 못 잡은 한글이 남아 있으면 번역기로 넘긴다
        if has_hangul(text):
            leftover = " ".join(
                w for w in text.split() if has_hangul(w)
            )
            translated = self.translator.translate(leftover)
            if translated and translated != leftover:
                for w in leftover.split():
                    text = text.replace(w, "")
                text = f"{text} {translated}"
                text = re.sub(r"\s+", " ", text).strip()
            else:
                self._unmapped.append(f"{slot}: {leftover}")

        # 한국어 조사·어미의 잔재를 정리
        text = re.sub(r"\s+([,.])", r"\1", text)
        return text.strip(" ,.")

    # ---- 관사 ----

    # 복수/불가산 명사에는 관사를 붙이지 않는다.
    _NO_ARTICLE = (
        "pants", "jeans", "shorts", "slacks", "leggings", "bottoms",
        "shoes", "sneakers", "boots", "heels", "sandals", "slippers",
        "glasses", "sunglasses", "gloves", "earrings", "clothes",
        "documents", "hair",
    )

    @classmethod
    def _with_article(cls, phrase: str) -> str:
        """명사구에 부정관사를 붙인다. 복수/불가산은 그대로."""
        phrase = (phrase or "").strip()
        if not phrase:
            return phrase

        lower = phrase.lower()
        if lower.startswith(("a ", "an ", "the ", "some ")):
            return phrase
        if any(lower.endswith(n) or f"{n} " in lower for n in cls._NO_ARTICLE):
            return phrase

        # 단순 첫 글자 규칙의 대표 예외를 보정한다.
        # 예: "uniform" 은 모음 글자로 시작하지만 /juː/ 소리라 "a uniform" 이 맞다.
        a_exceptions = (
            "uni", "user", "use", "euro", "one",
        )
        an_exceptions = (
            "hour", "honest", "honor", "heir",
        )

        if lower.startswith(an_exceptions):
            article = "an"
        elif lower.startswith(a_exceptions):
            article = "a"
        else:
            article = "an" if lower[0] in "aeiou" else "a"
        return f"{article} {phrase}"

    # ---- 문장 조립 ----

    def _subject(self, gender: str) -> str:
        text = (gender or "").strip()
        if not text:
            return "A person"

        # 사전 직접 매칭 (공백 제거 후에도 시도)
        key = text.replace(" ", "")
        for candidate in (text, key):
            if candidate in _SUBJECT:
                return f"A {_SUBJECT[candidate]}"

        # 부분 일치 (긴 것 우선)
        for ko in sorted(_SUBJECT, key=len, reverse=True):
            if ko in key:
                base = _SUBJECT[ko]
                # 수식어가 함께 있으면 앞에 붙인다 ("젊은 여성" 등)
                rest = self._to_english(key.replace(ko, ""), "gender")
                return f"A {rest} {base}".replace("  ", " ") if rest else f"A {base}"

        # 영어로 직접 입력한 경우
        if not has_hangul(text):
            return f"A {text}"

        translated = self.translator.translate(text)
        return f"A {translated}" if translated else "A person"

    def build(
        self,
        gender: str = "",
        hair: str = "",
        top: str = "",
        bottom: str = "",
        footwear: str = "",
        accessory: str = "",
        carry: str = "",
        pose: str = "",
        place: str = "",
        extra: str = "",
    ) -> Dict[str, object]:
        """
        슬롯 -> 서술형 문장.

        반환: {"caption": str, "slots": {...}, "english": {...},
               "unmapped": [...], "word_count": int}
        """
        self._unmapped = []

        slots = {
            "gender": gender, "hair": hair, "top": top, "bottom": bottom,
            "footwear": footwear, "accessory": accessory, "carry": carry,
            "pose": pose, "place": place, "extra": extra,
        }

        english = {
            k: self._to_english(v, k) if k != "gender" else ""
            for k, v in slots.items()
            if k != "extra"
        }
        english["extra"] = self._to_english(extra, "top") if extra else ""

        subject = self._subject(gender)

        # 절 단위로 모은다. 빈 슬롯은 그냥 빠진다.
        clauses: List[str] = []

        if english.get("hair"):
            hair_en = english["hair"]
            # "short black" 처럼 명사가 빠지면 hair 를 붙인다.
            # ponytail / bald 처럼 자체가 명사인 경우는 그대로 둔다.
            if not any(
                w in hair_en
                for w in ("hair", "ponytail", "bald", "shaved")
            ):
                hair_en = f"{hair_en} hair"
            clauses.append(f"with {hair_en}")

        worn = [english.get(k, "") for k in ("top", "bottom")]
        worn = [self._with_article(w) for w in worn if w]
        if worn:
            clauses.append("wearing " + " and ".join(worn))

        if english.get("footwear"):
            clauses.append(
                f"wearing {self._with_article(english['footwear'])}"
            )

        if english.get("accessory"):
            clauses.append(
                f"wearing {self._with_article(english['accessory'])}"
            )

        if english.get("carry"):
            clauses.append(
                f"holding {self._with_article(english['carry'])}"
            )

        tail: List[str] = []
        if english.get("pose"):
            tail.append(english["pose"])
        if english.get("place"):
            tail.append(english["place"])
        if tail:
            clauses.append(" ".join(tail))

        if english.get("extra"):
            clauses.append(english["extra"])

        if clauses:
            caption = subject + " " + ", ".join(clauses) + "."
        else:
            caption = subject + "."

        caption = re.sub(r"\s+", " ", caption).strip()
        caption = re.sub(r"\s+([,.])", r"\1", caption)

        if self._unmapped:
            logger.warning(
                "영어로 변환하지 못한 표현이 있습니다: %s\n"
                "  사전에 없고 번역도 실패했습니다. 해당 슬롯을 영어로 직접 "
                "입력하거나 다른 표현을 써 보세요.",
                self._unmapped,
            )

        word_count = len(caption.split())
        if word_count < 12:
            logger.debug(
                "조립된 문장이 %d단어입니다. 필요한 슬롯만 채워 사용하세요.",
                word_count,
            )

        return {
            "caption": caption,
            "slots": {k: v for k, v in slots.items() if v},
            "english": {k: v for k, v in english.items() if v},
            "unmapped": list(self._unmapped),
            "word_count": word_count,
        }

    @staticmethod
    def has_any(**slots: str) -> bool:
        """슬롯 중 하나라도 채워졌는지."""
        return any((v or "").strip() for v in slots.values())


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    ap = argparse.ArgumentParser(description="쿼리 번역 / 선택적 구조화 조립 스모크 테스트")
    ap.add_argument("--backend", default="opus", choices=list(BACKENDS))
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--translate", nargs="*", default=None,
                    help="번역할 문장들")
    ap.add_argument("--demo", action="store_true",
                    help="슬롯 조립 예시를 출력한다 (모델 로딩 없음)")
    args = ap.parse_args()

    if args.demo:
        # 사전만으로 처리되는 예시라 번역 모델을 올리지 않는다
        desc = QueryDescriptor(QueryTranslator(backend="none"))

        cases = [
            dict(gender="여성", hair="짧은 검은 곱슬",
                 top="화려한 꽃무늬 민소매 원피스",
                 carry="큰 분홍 양산", place="야외 맑은"),
            dict(gender="남성", hair="짧은 검은",
                 top="빨간 후드", bottom="청바지",
                 footwear="흰 운동화", carry="검정 백팩",
                 pose="걷는", place="거리"),
            dict(gender="소녀", top="노란 티셔츠", bottom="분홍 치마",
                 accessory="야구모자"),
            dict(carry="우산"),          # 슬롯 하나만
        ]

        for i, case in enumerate(cases, 1):
            out = desc.build(**case)
            print()
            print(f"[{i}] 입력: {case}")
            print(f"    문장 ({out['word_count']}단어):")
            print(f"    {out['caption']}")
            if out["unmapped"]:
                print(f"    변환 실패: {out['unmapped']}")
        print()
        raise SystemExit(0)

    if not args.translate:
        ap.error("--translate 또는 --demo 가 필요합니다.")

    tr = QueryTranslator(backend=args.backend, model_id=args.model_id)
    print()
    for q in args.translate:
        print(f"  {q}")
        print(f"    -> {tr.translate(q)}")
        print()