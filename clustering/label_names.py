#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""clustering/label_names.py — 라벨 문장에서 색·옷 종류를 뽑고, 항목 답으로 짧은 이름을 짓는다.

Qwen 이 준 이름(name)을 그대로 쓰면 프롬프트 예시를 베끼거나 자기 항목 답과 색이 어긋났다 (예: 상의=검은색인데
이름='노란 반팔에 검은 바지'). 그래서 이름은 항목(상의 색·종류 / 하의 색·종류, 물건은 색·종류)으로 여기서 짓는다.
옛 형식 답("노란색 반팔 티셔츠")도 색과 종류로 나눠 같은 규칙을 쓴다.

  color_of("노란색 반팔 티셔츠")        -> ("노", "노란")        (색 어간, 이름용 짧은 말)
  garment_of("노란색 반팔 티셔츠", "upper") -> "반팔"
  person_name("노란색", "반팔 티셔츠", "검은색", "긴바지") -> "노란 반팔에 검은 바지"
  object_name("흰색", "승용차")          -> "흰색 승용차"
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

# (어간, 이름용 짧은 말, 표기들) — 긴 표기부터 찾도록 순서를 둔다 (청록 before 청, 주황 before …)
COLOR_TABLE: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("청록", "청록", ("청록", "teal", "turquoise", "민트", "mint")),
    ("남", "남색", ("남색", "네이비", "navy")),
    ("하늘", "하늘색", ("하늘", "sky blue", "light blue")),
    ("주황", "주황", ("주황", "오렌지", "orange")),
    ("베이지", "베이지", ("베이지", "beige", "아이보리", "ivory", "살구")),
    ("카키", "카키", ("카키", "khaki", "올리브", "olive")),
    ("보라", "보라", ("보라", "퍼플", "purple", "violet", "자주")),
    ("분홍", "분홍", ("분홍", "핑크", "pink")),
    # '은색' 은 넣지 않는다 — '붉은색'·'옅은색' 안의 '은색' 을 회색으로 읽었다 (은빛·silver 만)
    ("회", "회색", ("회색", "회", "그레이", "gray", "grey", "은빛", "silver")),
    ("검", "검은", ("검은", "검정", "검", "블랙", "black")),
    ("흰", "흰", ("흰", "하얀", "하양", "화이트", "white", "백색")),
    ("빨", "빨간", ("빨간", "빨강", "붉은", "레드", "red", "적색")),
    ("노", "노란", ("노란", "노랑", "옐로", "yellow", "황색")),
    ("초록", "초록", ("초록", "녹색", "녹", "그린", "green")),
    ("파", "파란", ("파란", "파랑", "청색", "청", "블루", "blue")),
    ("갈", "갈색", ("갈색", "갈", "브라운", "brown")),
]
# 색 대신 이름에 쓰는 무늬 — 색을 못 읽었을 때만 (예: '어두운 톤의 상의, 밝은 꽃무늬' → '꽃무늬 상의')
PATTERN_TABLE: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("꽃무늬", "꽃무늬", ("꽃무늬", "꽃 무늬", "floral", "flower")),
    ("줄무늬", "줄무늬", ("줄무늬", "스트라이프", "stripe")),
    ("체크", "체크", ("체크", "격자", "plaid", "checked")),
    ("무늬", "무늬", ("무늬", "패턴", "pattern")),
]
UNKNOWN_WORDS = ("모름", "알 수 없", "unknown", "불명", "없음", "none", "n/a")
# 섞인 군집이라는 뜻의 답 — 이름을 붙이지 않는다
VAGUE_RE = re.compile(r"다를\s*수|다른\s|서로\s*다|다양|여러\s*가지|여러|섞|혼합|various|different|mixed|varies", re.I)
# 옷 종류 (찾는 순서대로) — 이름에는 짧은 말을 쓴다
UPPER_TYPES = [("민소매", ("민소매", "나시", "탱크", "sleeveless", "tank")), ("반팔", ("반팔", "short sleeve", "short-sleeve")),
               ("긴팔", ("긴팔", "long sleeve", "long-sleeve")), ("원피스", ("원피스", "드레스", "dress")),
               ("정장", ("정장", "suit")), ("코트", ("코트", "coat")), ("재킷", ("재킷", "자켓", "jacket", "점퍼", "jumper")),
               ("후드티", ("후드", "hoodie")), ("니트", ("니트", "스웨터", "sweater")), ("셔츠", ("셔츠", "shirt", "블라우스")),
               ("티셔츠", ("티셔츠", "t-shirt", "tee")), ("조끼", ("조끼", "vest"))]
LOWER_TYPES = [("반바지", ("반바지", "짧은", "숏", "shorts")), ("치마", ("치마", "스커트", "skirt")),
               ("원피스", ("원피스", "드레스", "dress")), ("청바지", ("청바지", "데님", "jeans", "denim")),
               ("레깅스", ("레깅스", "leggings")), ("바지", ("바지", "팬츠", "슬랙스", "pants", "trousers"))]


def _norm(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def is_unknown(text: object) -> bool:
    t = _norm(text)
    return not t or any(t.startswith(w) or t == w for w in UNKNOWN_WORDS)


def is_vague(*texts: object) -> bool:
    return any(VAGUE_RE.search(str(t or "")) for t in texts)


def _find(t: str, form: str) -> int:
    """t 안에서 form 의 위치 (-1 = 없음). 영어는 단어 경계로 찾는다 ('red' 가 'colored' 에 걸리지 않게)."""
    f = form.lower()
    if f.isascii():
        m = re.search(r"(?<![a-z])" + re.escape(f) + r"(?![a-z])", t)     # 'beige색' 은 찾고 'colored' 의 red 는 안 찾음
        return m.start() if m else -1
    return t.find(f)


def color_of(text: object) -> Optional[Tuple[str, str]]:
    """문장에서 가장 앞에 나오는 색 → (어간, 이름용 말). '청바지' 의 '청' 은 색으로 보지 않는다.
    같은 위치면 표의 앞쪽(더 구체적인 색: 청록 > 청)이 이긴다."""
    t = _norm(text).replace("청바지", "")
    if not t or is_unknown(t):
        return None
    best: Optional[Tuple[int, str, str]] = None
    for stem, short, forms in COLOR_TABLE:
        pos = min((p for p in (_find(t, f) for f in forms) if p >= 0), default=-1)
        if pos >= 0 and (best is None or pos < best[0]):
            best = (pos, stem, short)
    if best:
        return best[1], best[2]
    for stem, short, forms in PATTERN_TABLE:         # 색이 하나도 없을 때만 무늬 — 구체적인 무늬(꽃무늬 > 무늬)가 먼저
        if any(_find(t, f) >= 0 for f in forms):
            return stem, short
    return None


def color_stems(text: object) -> set:
    """문장에 나오는 모든 색 어간 (색 비교용)."""
    t = _norm(text).replace("청바지", "")
    out = set()
    for stem, _short, forms in COLOR_TABLE:          # 표 순서(구체적인 색 먼저)대로 찾고 찾은 말은 지운다 — '청록' 의 '청' 을 파랑으로 또 세지 않게
        for f in forms:
            if _find(t, f) >= 0:
                out.add(stem)
                t = t.replace(f.lower(), " ")
    return out


def garment_of(text: object, part: str) -> Optional[str]:
    t = _norm(text)
    for short, forms in (UPPER_TYPES if part == "upper" else LOWER_TYPES):
        if any(f in t for f in forms):
            return short
    return None


def person_name(upper_color: object, upper: object, lower_color: object, lower: object, max_chars: int = 30) -> str:
    """'<상의 색> <상의 종류>에 <하의 색> <하의 종류>'. 색을 모르는 쪽은 뺀다. 둘 다 모르면 빈 문자열.
    색이 따로 없으면(옛 형식) 종류 문장에서 색을 찾는다."""
    uc = color_of(upper_color) if not is_unknown(upper_color) else None
    uc = uc or color_of(upper)
    lc = color_of(lower_color) if not is_unknown(lower_color) else None
    lc = lc or color_of(lower)
    ut = garment_of(upper, "upper") or garment_of(upper_color, "upper") or "상의"
    lt = garment_of(lower, "lower") or garment_of(lower_color, "lower") or "하의"
    parts = []
    if uc:
        parts.append(f"{uc[1]} {ut}")
    if ut == "원피스" and lt in ("원피스", "하의"):
        pass                                     # 원피스는 상의 한 번으로 충분
    elif lt == "청바지":
        parts.append(f"{lc[1]} 청바지" if lc and lc[0] != "파" else "청바지")   # 청바지는 그 자체가 색 (파란 청바지 → 청바지)
    elif lc:
        parts.append(f"{lc[1]} {lt}")
    return "에 ".join(parts)[:max_chars].rstrip()


def object_name(color: object, category: object, max_chars: int = 30) -> str:
    """'<색> <종류>' (예: 흰색 승용차 → '흰 승용차'). 종류 문장이 색으로 시작하면 그 색 말만 떼어 낸다."""
    c = color_of(color) if not is_unknown(color) else None
    cat = re.sub(r"\s+", " ", str(category or "")).strip()
    lead = color_of(cat.split(" ")[0]) if " " in cat else None
    if lead:                                     # '흰색 승용차' → 색은 앞말, 종류는 나머지
        c = c or lead
        cat = cat.split(" ", 1)[1].strip()
    cat = cat or "물체"
    return (f"{c[1]} {cat}" if c else cat)[:max_chars].rstrip()
