#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gui/search_ui.py — 검색 화면 상단 카드(Immich 식): 모드(자연어/사진) · 대상 · 큰 검색창 · AI 재확인 · 고급 설정.

ImageSearchPage / VideoSearchPage 가 공유한다. 여기서 만든 위젯은 page 의 속성으로 붙고 이름은 기존 검색 로직이
쓰는 그대로다 (text, text_scope, image_scope, image_limit / image_top_k, qwen_source / video_qwen_source …).
시그널 연결은 페이지가 한다 — 이 모듈은 배치만 담당한다.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

PREVIEW = 96


def _scope_combo() -> QComboBox:
    c = QComboBox()
    c.addItem("사람", "person")
    c.addItem("물건", "object")
    c.setToolTip("무엇을 찾을지: 사람(person 컬렉션) / 물건(object 컬렉션)")
    return c


def _spin(lo: int, hi: int, value: int, tip: str = "") -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    if tip:
        s.setToolTip(tip)
    return s


def _per_video_spin() -> QSpinBox:
    s = QSpinBox()
    s.setRange(0, 50)
    s.setValue(0)
    s.setSpecialValueText("제한 없음")
    s.setToolTip(
        "같은 영상에서 최대 몇 개의 Track 까지 보일지. 0 = 제한 없음. "
        "접힌 결과는 지워지는 것이 아니라 카드에 '같은 영상 N건 더 있음' 으로 표시됩니다."
    )
    return s


def _form_row(pairs: Sequence[Tuple[str, QWidget]]) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(8)
    for label, widget in pairs:
        lab = QLabel(label)
        lab.setObjectName("advLabel")
        h.addWidget(lab)
        h.addWidget(widget)
        h.addSpacing(10)
    h.addStretch(1)
    return w


def build_search_header(page: Any, *, video: bool) -> QFrame:
    """page 에 검색 위젯을 속성으로 붙이고 헤더 카드를 돌려준다."""
    hdr = QFrame()
    hdr.setObjectName("searchHeader")
    v = QVBoxLayout(hdr)
    v.setContentsMargins(16, 12, 16, 10)
    v.setSpacing(8)

    # ---- 1행: 모드 · 대상 · (오른쪽) AI 재확인 · 고급 ----
    row0 = QHBoxLayout()
    row0.setSpacing(8)
    page.mode_text_btn = QToolButton()
    page.mode_text_btn.setObjectName("modeButton")
    page.mode_text_btn.setText("자연어로 찾기")
    page.mode_text_btn.setCheckable(True)
    page.mode_text_btn.setCursor(Qt.PointingHandCursor)
    page.mode_crop_btn = QToolButton()
    page.mode_crop_btn.setObjectName("modeButton")
    page.mode_crop_btn.setText("사진(crop)으로 찾기")
    page.mode_crop_btn.setCheckable(True)
    page.mode_crop_btn.setCursor(Qt.PointingHandCursor)
    page.mode_group = QButtonGroup(hdr)
    page.mode_group.setExclusive(True)
    page.mode_group.addButton(page.mode_text_btn)
    page.mode_group.addButton(page.mode_crop_btn)
    row0.addWidget(page.mode_text_btn)
    row0.addWidget(page.mode_crop_btn)
    row0.addSpacing(14)
    scope_lab = QLabel("대상")
    scope_lab.setObjectName("advLabel")
    row0.addWidget(scope_lab)
    page.text_scope = _scope_combo()
    page.image_scope = _scope_combo()
    row0.addWidget(page.text_scope)
    row0.addWidget(page.image_scope)
    row0.addStretch(1)

    if video:
        page.video_qwen_btn = QPushButton("AI 재확인 (Qwen)")
        qbtn = page.video_qwen_btn
    else:
        page.qwen_btn = QPushButton("AI 재확인 (Qwen)")
        qbtn = page.qwen_btn
    qbtn.setObjectName("qwenButton")
    qbtn.setToolTip("검색 결과 상위 후보를 Qwen3-VL 이 다시 보고 조건에 맞는지 판정합니다 (별도 프로세스, 후보당 수 초~수십 초)")
    row0.addWidget(qbtn)
    page.adv_btn = QToolButton()
    page.adv_btn.setObjectName("advButton")
    page.adv_btn.setText("고급 설정")
    page.adv_btn.setCheckable(True)
    page.adv_btn.setCursor(Qt.PointingHandCursor)
    row0.addWidget(page.adv_btn)
    v.addLayout(row0)

    # ---- 2행: 검색창 (모드별 스택) ----
    page.query_stack = QStackedWidget()
    text_w = QWidget()
    th = QHBoxLayout(text_w)
    th.setContentsMargins(0, 0, 0, 0)
    th.setSpacing(8)
    page.text = QLineEdit()
    page.text.setObjectName("searchBox")
    page.text.setPlaceholderText("무엇을 찾을까요?  예: 검은 상의를 입은 남성 · 빨간 가방을 든 사람 · 흰색 승용차")
    page.text.setClearButtonEnabled(True)
    th.addWidget(page.text, 1)
    page.text_search_btn = QPushButton("검색")
    page.text_search_btn.setObjectName("primaryButton")
    page.text_search_btn.setMinimumWidth(96)
    th.addWidget(page.text_search_btn)
    page.query_stack.addWidget(text_w)

    crop_w = QWidget()
    ch = QHBoxLayout(crop_w)
    ch.setContentsMargins(0, 0, 0, 0)
    ch.setSpacing(10)
    page.query_preview = QLabel("사진 없음")
    page.query_preview.setObjectName("queryPreview")
    page.query_preview.setFixedSize(PREVIEW, PREVIEW)
    page.query_preview.setAlignment(Qt.AlignCenter)
    ch.addWidget(page.query_preview)
    page.choose_btn = QPushButton("사진 선택…")
    page.choose_btn.setMinimumHeight(36)
    ch.addWidget(page.choose_btn)
    hint = QLabel("찾을 사람·물건이 잘린 사진(crop)을 고르면 비슷한 것을 찾습니다. 검색 결과에서 '이 결과로 다시 찾기' 로도 넣을 수 있습니다.")
    hint.setObjectName("subtleLabel")
    hint.setWordWrap(True)
    ch.addWidget(hint, 1)
    page.image_search_btn = QPushButton("검색")
    page.image_search_btn.setObjectName("primaryButton")
    page.image_search_btn.setMinimumWidth(96)
    ch.addWidget(page.image_search_btn)
    page.query_stack.addWidget(crop_w)
    v.addWidget(page.query_stack)

    # ---- 3행: 파이프라인 설명 (모드별) ----
    page.text_pipeline = QLabel()
    page.text_pipeline.setObjectName("pipelineLabel")
    page.text_pipeline.setWordWrap(True)
    page.image_pipeline = QLabel()
    page.image_pipeline.setObjectName("pipelineLabel")
    page.image_pipeline.setWordWrap(True)
    v.addWidget(page.text_pipeline)
    v.addWidget(page.image_pipeline)

    # ---- 고급 설정 (접힘) ----
    page.adv_panel = QFrame()
    page.adv_panel.setObjectName("advPanel")
    ag = QVBoxLayout(page.adv_panel)
    ag.setContentsMargins(12, 8, 12, 8)
    ag.setSpacing(6)

    page.text_vectors = QComboBox()
    page.text_vectors.setToolTip("자연어를 임베딩할 모델 (pipeline.yaml 에서 supports_text=true 인 것). 여러 개면 RRF 조합")
    if video:
        page.text_top_k = _spin(1, 200, 20, "결과(track 그룹) 수")
        page.text_per_video = _per_video_spin()
        page.text_adv = _form_row([("자연어 검색 모델", page.text_vectors), ("결과 수", page.text_top_k), ("영상당 최대", page.text_per_video)])
        page.image_vector = QComboBox()
        page.image_vector.setToolTip("Crop 을 임베딩해 track 을 찾을 모델. 기본 사람 SOLIDER / 물건 DINOv2")
        page.image_top_k = _spin(1, 200, 20, "결과(track 그룹) 수")
        page.image_per_video = _per_video_spin()
        page.crop_adv = _form_row([("사진 검색 모델", page.image_vector), ("결과 수", page.image_top_k), ("영상당 최대", page.image_per_video)])
        page.video_qwen_source = QComboBox()
        page.video_qwen_source.addItem("사진(crop) 검색 결과", "image-video")
        page.video_qwen_source.addItem("자연어 검색 결과", "text-video")
        page.video_qwen_top_k = _spin(1, 200, 10, "Qwen 이 다시 볼 상위 후보 수")
        page.qwen_adv = _form_row([("AI 재확인 대상", page.video_qwen_source), ("후보 수", page.video_qwen_top_k)])
        page._qwen_source_combo = page.video_qwen_source
        page._mode_keys = {"text": "text-video", "crop": "image-video"}
    else:
        page.text_limit = _spin(1, 500, 20, "결과 수")
        page.text_adv = _form_row([("자연어 검색 모델", page.text_vectors), ("결과 수", page.text_limit)])
        page.image_stage1 = QComboBox()
        page.image_stage1.setToolTip("1차 후보를 뽑는 임베더. pipeline.yaml 의 retrievers 중 선택 (여러 개면 RRF 조합)")
        page.image_rerank = QComboBox()
        page.image_rerank.setToolTip("1차 후보를 다시 정렬할 임베더. '없음' 이면 1차 순위 그대로")
        page.image_limit = _spin(1, 500, 20, "결과 수")
        page.crop_adv = _form_row([("1차 검색 모델", page.image_stage1), ("2차 재정렬", page.image_rerank), ("결과 수", page.image_limit)])
        page.qwen_source = QComboBox()
        page.qwen_source.addItem("사진(crop) 검색 결과", "crop")
        page.qwen_source.addItem("자연어 검색 결과", "text")
        page.qwen_top_k = _spin(1, 200, 20, "Qwen 이 다시 볼 상위 후보 수")
        page.qwen_adv = _form_row([("AI 재확인 대상", page.qwen_source), ("후보 수", page.qwen_top_k)])
        page._qwen_source_combo = page.qwen_source
        page._mode_keys = {"text": "text", "crop": "crop"}
    ag.addWidget(page.text_adv)
    ag.addWidget(page.crop_adv)
    ag.addWidget(page.qwen_adv)
    page.adv_panel.setVisible(False)
    page.adv_btn.toggled.connect(page.adv_panel.setVisible)
    v.addWidget(page.adv_panel)

    # ---- 상태 줄 ----
    status_row = QHBoxLayout()
    page.status = QLabel("준비됨")
    page.status.setObjectName("statusLabel")
    status_row.addWidget(page.status, 1)
    page.progress = QProgressBar()
    page.progress.setObjectName("busyBar")
    page.progress.setRange(0, 0)
    page.progress.setTextVisible(False)
    page.progress.setFixedWidth(160)
    page.progress.setVisible(False)
    status_row.addWidget(page.progress)
    v.addLayout(status_row)

    page.mode_text_btn.toggled.connect(lambda on: set_search_mode(page, "text") if on else None)
    page.mode_crop_btn.toggled.connect(lambda on: set_search_mode(page, "crop") if on else None)
    page.search_mode = "text"
    return hdr


def set_search_mode(page: Any, mode: str) -> None:
    """자연어/사진 모드 전환: 검색창·대상·설명·고급 행을 바꾸고 AI 재확인 대상도 그 모드로 맞춘다."""
    is_text = mode == "text"
    page.search_mode = mode
    btn = page.mode_text_btn if is_text else page.mode_crop_btn
    if not btn.isChecked():
        btn.blockSignals(True)
        btn.setChecked(True)
        btn.blockSignals(False)
    page.query_stack.setCurrentIndex(0 if is_text else 1)
    page.text_scope.setVisible(is_text)
    page.image_scope.setVisible(not is_text)
    page.text_pipeline.setVisible(is_text)
    page.image_pipeline.setVisible(not is_text)
    page.text_adv.setVisible(is_text)
    page.crop_adv.setVisible(not is_text)
    combo = getattr(page, "_qwen_source_combo", None)
    key = getattr(page, "_mode_keys", {}).get(mode)
    if combo is not None and key is not None:
        idx = combo.findData(key)
        if idx >= 0:
            combo.setCurrentIndex(idx)


def link_scope_combos(a: QComboBox, b: QComboBox) -> None:
    """두 대상 콤보(자연어/사진)를 같은 값으로 묶는다 — 화면에는 한 번에 하나만 보인다."""
    def sync(src: QComboBox, dst: QComboBox):
        def _on(idx: int):
            if dst.currentIndex() != idx:
                dst.setCurrentIndex(idx)
        return _on
    a.currentIndexChanged.connect(sync(a, b))
    b.currentIndexChanged.connect(sync(b, a))
