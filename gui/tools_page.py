#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gui/tools_page.py — '도구': 단계마다 쓸 수 있는 도구를 한눈에 보고 고른다. ★ = 원장(벤치마크) 성적이 가장 좋은 것(기본값).

고르면 gui_tool_choice.json 에 저장되고 다음 실행부터 파이프라인 단계·검색 화면의 기본값이 된다.
'★ 로 되돌리기' 는 저장을 지워 다시 최고 성적 도구를 쓴다. '단계로 이동' 은 그 도구를 쓰는 파이프라인 단계로 간다.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import tool_choice as TCH
from gui import tools_catalog as TC

STATUS_TEXT = {"pass": "통과", "partial": "일부", "incomplete": "확인 불가", "fail": "미달", None: "—", "n-a": "—"}


class ToolSection(QFrame):
    chosen = Signal(str, str)          # step, key
    openStep = Signal(str, str)        # group id, stage id

    def __init__(self, step: str, sec: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("searchHeader")
        self.step = step
        self.sec = sec
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 12, 16, 12)
        v.setSpacing(6)
        head = QHBoxLayout()
        title = QLabel(sec["title"])
        title.setObjectName("resultsTitle")
        head.addWidget(title)
        head.addStretch(1)
        self.state = QLabel("")
        self.state.setObjectName("subtleLabel")
        self.state.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # 긴 상태 문장이 버튼을 밀어내지 않게
        head.addWidget(self.state, 1)
        self.reset_btn = QPushButton("★ 로 되돌리기")
        self.reset_btn.clicked.connect(self._reset)
        head.addWidget(self.reset_btn)
        if sec.get("group") and sec.get("stage_id"):
            go = QPushButton("단계로 이동")
            go.clicked.connect(lambda: self.openStep.emit(str(sec["group"]), str(sec["stage_id"])))
            head.addWidget(go)
        v.addLayout(head)
        desc = QLabel(sec["desc"])
        desc.setObjectName("subtleLabel")
        desc.setWordWrap(True)
        v.addWidget(desc)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["사용", "도구", "성적 (원장 최신)", "상태", "설명"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.table)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.radios: Dict[str, QRadioButton] = {}
        self.fill()

    def fill(self) -> None:
        cands: List[Dict[str, Any]] = self.sec.get("candidates") or []
        best = self.sec.get("best_key")
        chosen = TCH.get(self.step)
        current = chosen["key"] if chosen else best
        self.table.setRowCount(0)
        for b in list(self.radios.values()):
            self.group.removeButton(b)
        self.radios.clear()
        for r, c in enumerate(cands):
            self.table.insertRow(r)
            rb = QRadioButton()
            rb.setChecked(c["key"] == current)
            rb.toggled.connect(lambda on, k=c["key"]: self._pick(k) if on else None)
            self.group.addButton(rb)
            self.radios[c["key"]] = rb
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(8, 0, 0, 0)
            hl.addWidget(rb)
            hl.addStretch(1)
            self.table.setCellWidget(r, 0, holder)
            name = ("★ " if c["key"] == best else "") + str(c["label"])
            for col, text in ((1, name), (2, TC.fmt_metric(c) + (f"  ({c.get('entry_name')})" if c.get("entry_name") else "")),
                              (3, STATUS_TEXT.get(c.get("status"), str(c.get("status")))), (4, str(c.get("detail") or ""))):
                it = QTableWidgetItem(text)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                if c["key"] == best:
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                self.table.setItem(r, col, it)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.setMinimumHeight(min(320, 36 + 30 * max(1, len(cands))))
        self._update_state(chosen)

    def _update_state(self, chosen: Optional[Dict[str, Any]]) -> None:
        best_label = next((c["label"] for c in self.sec.get("candidates", []) if c["key"] == self.sec.get("best_key")), None)
        if chosen:
            self.state.setText(f"선택: {chosen.get('label') or chosen['key']} (저장됨 · 다음 실행부터)")
        else:
            self.state.setText(f"기본: ★ {best_label}" if best_label else "후보 없음")
        self.reset_btn.setEnabled(chosen is not None)

    def _pick(self, key: str) -> None:
        c = next((c for c in self.sec.get("candidates", []) if c["key"] == key), None)
        if c is None:
            return
        TCH.set_choice(self.step, key, c.get("value"), str(c.get("label") or ""))
        self._update_state(TCH.get(self.step))
        self.chosen.emit(self.step, key)

    def _reset(self) -> None:
        TCH.clear(self.step)
        best = self.sec.get("best_key")
        if best in self.radios:
            self.radios[best].blockSignals(True)
            self.radios[best].setChecked(True)
            self.radios[best].blockSignals(False)
        self._update_state(None)
        self.chosen.emit(self.step, str(best))


class ToolsPage(QWidget):
    openStep = Signal(str, str)
    choiceChanged = Signal(str, str)

    def __init__(self, parent: Optional[QWidget] = None, catalog: Optional[Dict[str, Dict[str, Any]]] = None):
        super().__init__(parent)
        self._catalog_override = catalog
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(10)
        head = QHBoxLayout()
        title = QLabel("도구")
        title.setObjectName("resultsTitle")
        head.addWidget(title)
        intro = QLabel("단계마다 쓸 도구를 고릅니다. ★ 는 벤치마크(원장) 성적이 가장 좋은 것이고 따로 고르지 않으면 그것이 기본값입니다. 새 모델은 평가 › 새 모델 등록으로 추가합니다.")
        intro.setObjectName("subtleLabel")
        intro.setWordWrap(True)
        head.addWidget(intro, 1)
        self.refresh_btn = QPushButton("새로고침")
        self.refresh_btn.clicked.connect(self.refresh)
        head.addWidget(self.refresh_btn)
        outer.addLayout(head)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(self.scroll, 1)
        self.sections: Dict[str, ToolSection] = {}
        self.refresh()

    def refresh(self) -> None:
        cat = self._catalog_override if self._catalog_override is not None else TC.cached_catalog(refresh=True)
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)
        self.sections = {}
        for step in TC.STEPS:
            sec = cat.get(step["key"])
            if not sec:
                continue
            w = ToolSection(step["key"], sec)
            w.openStep.connect(self.openStep)
            w.chosen.connect(self.choiceChanged)
            v.addWidget(w)
            self.sections[step["key"]] = w
        v.addStretch(1)
        self.scroll.setWidget(host)
