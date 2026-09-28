#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gui/shell.py — Immich 식 셸: 왼쪽 아이콘 사이드바(섹션 + 항목) + 오른쪽 페이지 스택.

search_gui.MainWindow 가 상단 QTabWidget 대신 이것을 centralWidget 으로 쓴다.
페이지 자체(검색/파이프라인/벤치마크)는 그대로이고 셸은 배치·탐색만 담당한다.
아이콘은 외부 파일 없이 QPainter 로 그린다 (기본 = 흐린 색, 선택 = 강조색).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui.gui_theme import T

SIDEBAR_WIDTH = 208
ICON_SIZE = 20


def _paint_icon(kind: str, color: str, size: int = ICON_SIZE) -> QPixmap:
    """작은 선 아이콘. kind ∈ search, video, photo, film, chart, bench, gear."""
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(1.8)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    s = size / 20.0  # 20px 기준 좌표

    def R(x, y, w, h):
        return QRectF(x * s, y * s, w * s, h * s)

    def P(x, y):
        return QPointF(x * s, y * s)

    if kind == "search":
        p.drawEllipse(R(3, 3, 10, 10))
        p.drawLine(P(12, 12), P(17.5, 17.5))
    elif kind == "video":
        p.drawRoundedRect(R(2, 4, 16, 12), 2 * s, 2 * s)
        p.setBrush(QColor(color))
        p.drawPolygon(QPolygonF([P(8, 7), P(13, 10), P(8, 13)]))
    elif kind == "photo":
        p.drawRoundedRect(R(2, 3, 16, 14), 2 * s, 2 * s)
        p.drawEllipse(R(5, 6, 3, 3))
        p.drawPolyline(QPolygonF([P(3, 15), P(8, 10), P(11, 13), P(13.5, 10.5), P(17, 15)]))
    elif kind == "film":
        p.drawRoundedRect(R(3, 2, 14, 16), 2 * s, 2 * s)
        for y in (5.5, 10, 14.5):
            p.drawLine(P(3, y), P(6, y))
            p.drawLine(P(14, y), P(17, y))
        p.drawLine(P(6, 2), P(6, 18))
        p.drawLine(P(14, 2), P(14, 18))
    elif kind == "chart":
        p.drawLine(P(3, 17), P(17, 17))
        for x, h in ((5, 6), (9.5, 10), (14, 13)):
            p.drawRoundedRect(R(x, 17 - h, 3, h), 1 * s, 1 * s)
    elif kind == "bench":
        p.drawPolyline(QPolygonF([P(3, 15), P(7.5, 10), P(11, 12.5), P(17, 5)]))
        p.setBrush(QColor(color))
        for x, y in ((3, 15), (7.5, 10), (11, 12.5), (17, 5)):
            p.drawEllipse(R(x - 1.3, y - 1.3, 2.6, 2.6))
    elif kind == "gear":
        p.drawEllipse(R(6, 6, 8, 8))
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0), (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)):
            p.drawLine(P(10 + dx * 5.2, 10 + dy * 5.2), P(10 + dx * 7.6, 10 + dy * 7.6))
    else:
        p.drawRoundedRect(R(3, 3, 14, 14), 3 * s, 3 * s)
    p.end()
    return pix


def nav_icon(kind: str) -> QIcon:
    icon = QIcon()
    icon.addPixmap(_paint_icon(kind, T["muted"]), QIcon.Normal, QIcon.Off)
    icon.addPixmap(_paint_icon(kind, T["accent"]), QIcon.Normal, QIcon.On)
    icon.addPixmap(_paint_icon(kind, T["accent"]), QIcon.Active, QIcon.On)
    return icon


class AppShell(QWidget):
    """사이드바 + 스택. add_section / add_page 순서대로 사이드바에 쌓인다."""

    pageChanged = Signal(str)

    def __init__(self, title: str = "포렌식 검색", subtitle: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(SIDEBAR_WIDTH)
        self._sb = QVBoxLayout(self.sidebar)
        self._sb.setContentsMargins(10, 14, 10, 10)
        self._sb.setSpacing(2)

        brand = QLabel(title)
        brand.setObjectName("brandTitle")
        self._sb.addWidget(brand)
        self.subtitle = QLabel(subtitle)
        self.subtitle.setObjectName("brandSub")
        self.subtitle.setWordWrap(True)
        self._sb.addWidget(self.subtitle)

        self._sb.addStretch(1)
        self.footer = QLabel("")
        self.footer.setObjectName("navFooter")
        self.footer.setWordWrap(True)
        self._sb.addWidget(self.footer)
        outer.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: Dict[str, QToolButton] = {}
        self._labels: Dict[str, str] = {}
        self._keys: List[str] = []
        self._group.buttonClicked.connect(self._on_button)

    # ---- 구성 ----
    def _insert(self, w: QWidget) -> None:
        # stretch(끝에서 두 번째) 앞에 넣는다
        self._sb.insertWidget(self._sb.count() - 2, w)

    def add_section(self, title: str) -> None:
        lab = QLabel(title)
        lab.setObjectName("navSection")
        self._insert(lab)

    def add_page(self, key: str, label: str, icon_kind: str, widget: QWidget) -> QToolButton:
        if key in self._buttons:
            raise ValueError(f"페이지 key 중복: {key}")
        btn = QToolButton()
        btn.setObjectName("navButton")
        btn.setText(label)
        btn.setIcon(nav_icon(icon_kind))
        btn.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        btn.setCheckable(True)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setProperty("pageKey", key)
        self._group.addButton(btn)
        self._insert(btn)
        self._buttons[key] = btn
        self._labels[key] = label
        self._keys.append(key)
        self.stack.addWidget(widget)
        if len(self._keys) == 1:
            btn.setChecked(True)
            self.stack.setCurrentIndex(0)
        return btn

    # ---- 탐색 ----
    def keys(self) -> List[str]:
        return list(self._keys)

    def label_of(self, key: str) -> str:
        return self._labels.get(key, key)

    def current_key(self) -> Optional[str]:
        idx = self.stack.currentIndex()
        return self._keys[idx] if 0 <= idx < len(self._keys) else None

    def select(self, key: str) -> None:
        if key not in self._buttons:
            raise KeyError(key)
        self._buttons[key].setChecked(True)
        self.stack.setCurrentIndex(self._keys.index(key))
        self.pageChanged.emit(key)

    def _on_button(self, btn: QToolButton) -> None:
        key = str(btn.property("pageKey"))
        if key in self._keys:
            self.stack.setCurrentIndex(self._keys.index(key))
            self.pageChanged.emit(key)
