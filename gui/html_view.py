#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gui/html_view.py — GUI 안에서 결과 HTML(리포트·갤러리·인덱스)을 보여 주는 뷰어.

QtWebEngine(크로미움) 이 있으면 그것을, 없거나 TRANSREID_NO_WEBENGINE=1 이면 QTextBrowser 로 대신한다
(QTextBrowser 는 CSS 가 단순하게 나오지만 내용은 읽힌다). 외부 브라우저로 여는 버튼도 같이 둔다.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QTextBrowser, QVBoxLayout, QWidget


def _webengine_view():
    if os.environ.get("TRANSREID_NO_WEBENGINE") == "1":
        return None
    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView
    except Exception:  # noqa: BLE001 — 모듈 없음/로드 실패
        return None
    try:
        return QWebEngineView()
    except Exception:  # noqa: BLE001
        return None


class HtmlView(QWidget):
    """load(path) 로 로컬 HTML 을 보여 준다. 위에 제목·'브라우저에서 열기' 줄, 선택적으로 '← 목록' 버튼."""

    def __init__(self, parent: Optional[QWidget] = None, *, back_label: Optional[str] = None):
        super().__init__(parent)
        self.path: Optional[Path] = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        bar = QHBoxLayout()
        self.back_btn = QPushButton(back_label or "← 목록")
        self.back_btn.setVisible(bool(back_label))
        bar.addWidget(self.back_btn)
        self.title = QLabel("결과가 여기에 표시됩니다")
        self.title.setObjectName("subtleLabel")
        # 긴 경로가 창의 최소 폭을 키우지 않게: 파일 이름만 보이고 전체 경로는 툴팁
        self.title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        bar.addWidget(self.title, 1)
        self.ext_btn = QPushButton("브라우저에서 열기")
        self.ext_btn.setEnabled(False)
        self.ext_btn.clicked.connect(self.open_external)
        bar.addWidget(self.ext_btn)
        layout.addLayout(bar)

        web = _webengine_view()
        if web is not None:
            self.backend = "webengine"
            self.view = web
        else:
            self.backend = "textbrowser"
            self.view = QTextBrowser()
            self.view.setOpenExternalLinks(True)
        layout.addWidget(self.view, 1)

    def load(self, path: Path) -> None:
        p = Path(path)
        self.path = p
        self.title.setText(p.name)
        self.title.setToolTip(str(p))
        self.ext_btn.setEnabled(p.is_file())
        url = QUrl.fromLocalFile(str(p.resolve()))
        if self.backend == "webengine":
            self.view.load(url)
        else:
            self.view.setSource(url)

    def open_external(self) -> None:
        if self.path is not None and self.path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.path.resolve())))
