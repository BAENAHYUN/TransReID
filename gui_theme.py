#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gui_theme.py — search_gui 의 겉모습 한 곳.

Fusion 스타일 + 팔레트 + QSS + 앱 폰트 + 아이콘을 여기서 정한다.
검색/파이프라인 로직은 건드리지 않는다. search_gui.main() 이 apply_theme(app) 을 부른다.

색 토큰은 build_image_db_html.py 의 dataviz 팔레트와 맞췄다 (accent = person 색 #2a78d6).
폰트는 px 가 아니라 pt 로 준다 — QSS 에 `font-size: 13px` 를 앱 전체에 걸면 QFont.pointSize()
가 -1 이 되어 어딘가에서 `QFont::setPointSize: Point size <= 0 (-1)` 경고가 난다.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPalette
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent
ICON_PATH = ROOT / "assets" / "app_icon.png"

APP_NAME = "Forensic Visual Retrieval"
ORG_NAME = "TransReID"

T = {
    "bg": "#f4f6f8",
    "surface": "#ffffff",
    "surface2": "#eef2f6",
    "line": "#d8dde3",
    "line_strong": "#c3cad2",
    "text": "#1f2933",
    "muted": "#5a6673",
    "accent": "#2a78d6",
    "accent_hover": "#2467b8",
    "accent_pressed": "#1d569a",
    "accent_soft": "#e4eefb",
    "ok": "#2e7d32",
    "danger": "#c62828",
    "console_bg": "#0f1419",
    "console_fg": "#d6deeb",
}


def app_font() -> QFont:
    families = set(QFontDatabase.families())
    for name in ("Malgun Gothic", "맑은 고딕", "Segoe UI", "Noto Sans KR"):
        if name in families:
            f = QFont(name, 10)
            f.setStyleStrategy(QFont.PreferAntialias)
            return f
    f = QFont()
    f.setPointSize(10)
    return f


def palette() -> QPalette:
    p = QPalette()
    c = QColor
    p.setColor(QPalette.Window, c(T["bg"]))
    p.setColor(QPalette.WindowText, c(T["text"]))
    p.setColor(QPalette.Base, c(T["surface"]))
    p.setColor(QPalette.AlternateBase, c(T["surface2"]))
    p.setColor(QPalette.Text, c(T["text"]))
    p.setColor(QPalette.Button, c(T["surface"]))
    p.setColor(QPalette.ButtonText, c(T["text"]))
    p.setColor(QPalette.ToolTipBase, c(T["text"]))
    p.setColor(QPalette.ToolTipText, c(T["surface"]))
    p.setColor(QPalette.Highlight, c(T["accent"]))
    p.setColor(QPalette.HighlightedText, c(T["surface"]))
    p.setColor(QPalette.PlaceholderText, c(T["muted"]))
    p.setColor(QPalette.Disabled, QPalette.Text, c("#9aa5b1"))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, c("#9aa5b1"))
    p.setColor(QPalette.Disabled, QPalette.WindowText, c("#9aa5b1"))
    return p


QSS = """
QMainWindow, QDialog { background: %(bg)s; }
QWidget { color: %(text)s; }
QToolTip { background: %(text)s; color: %(surface)s; border: none; padding: 5px 8px; border-radius: 4px; }

/* ---- 최상위/하위 탭 ---- */
QTabWidget::pane {
    border: 1px solid %(line)s; border-radius: 8px; background: %(surface)s; top: -1px;
}
QTabBar::tab {
    background: transparent; color: %(muted)s;
    padding: 8px 16px; margin-right: 2px;
    border: 1px solid transparent; border-bottom: 3px solid transparent;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
}
QTabBar::tab:hover { background: %(surface2)s; color: %(text)s; }
QTabBar::tab:selected {
    color: %(accent)s; font-weight: 600; background: %(surface)s;
    border-color: %(line)s %(line)s %(accent)s %(line)s;
}

/* ---- 카드 (GroupBox) ---- */
QGroupBox {
    font-weight: 600; color: %(text)s;
    border: 1px solid %(line)s; border-radius: 8px;
    margin-top: 14px; padding-top: 14px; background: %(surface)s;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: %(muted)s; }

/* ---- 버튼 ---- */
QPushButton {
    min-height: 30px; padding: 4px 14px;
    background: %(surface)s; border: 1px solid %(line_strong)s; border-radius: 6px;
}
QPushButton:hover { background: %(surface2)s; border-color: %(accent)s; }
QPushButton:pressed { background: %(accent_soft)s; }
QPushButton:disabled { color: #9aa5b1; background: %(surface2)s; border-color: %(line)s; }
QPushButton#primaryButton {
    min-height: 36px; font-weight: 700; color: %(surface)s;
    background: %(accent)s; border: 1px solid %(accent)s;
}
QPushButton#primaryButton:hover { background: %(accent_hover)s; border-color: %(accent_hover)s; }
QPushButton#primaryButton:pressed { background: %(accent_pressed)s; }
QPushButton#primaryButton:disabled { color: %(surface)s; background: #a9c4e6; border-color: #a9c4e6; }

/* ---- 입력 ---- */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
    background: %(surface)s; border: 1px solid %(line_strong)s; border-radius: 6px;
    padding: 4px 8px; selection-background-color: %(accent)s;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {
    border: 1px solid %(accent)s;
}
QLineEdit:read-only { background: %(surface2)s; color: %(muted)s; }
QComboBox::drop-down { border: none; width: 22px; }
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 18px; border: none; }
QCheckBox::indicator { width: 16px; height: 16px; }

/* ---- 목록 / 스크롤 ---- */
QListWidget { background: %(surface)s; border: 1px solid %(line)s; border-radius: 8px; outline: 0; }
QListWidget::item { padding: 7px 10px; border-radius: 5px; margin: 1px 4px; }
QListWidget::item:hover { background: %(surface2)s; }
QListWidget::item:selected { background: %(accent_soft)s; color: %(accent)s; font-weight: 600; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: %(line_strong)s; border-radius: 5px; min-height: 28px; }
QScrollBar::handle:vertical:hover { background: %(muted)s; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: %(line_strong)s; border-radius: 5px; min-width: 28px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QSplitter::handle { background: %(line)s; }
QSplitter::handle:vertical { height: 4px; }

/* ---- 진행 / 상태 ---- */
QProgressBar#busyBar { border: none; background: %(line)s; border-radius: 3px; max-height: 6px; }
QProgressBar#busyBar::chunk { background: %(accent)s; border-radius: 3px; }
QStatusBar { background: %(surface)s; border-top: 1px solid %(line)s; color: %(muted)s; }
QStatusBar::item { border: none; }
QLabel#statusLabel { padding: 6px 2px; font-weight: 600; color: %(muted)s; }
QLabel#runState { padding: 0 8px; }

/* ---- 검색 화면 요소 (기존 objectName 유지) ---- */
QLabel#queryPreview, QLabel#detailPreview, QLabel#thumbnail {
    border: 1px solid %(line)s; border-radius: 6px; background: %(surface2)s; color: %(muted)s;
}
QLabel#pipelineLabel {
    padding: 8px 10px; border: 1px solid %(line)s; border-radius: 6px;
    background: %(accent_soft)s; color: %(text)s;
}
QFrame#resultCard { border: 1px solid %(line)s; border-radius: 8px; background: %(surface)s; }
QFrame#resultCard:hover { border-color: %(accent)s; }
QLabel#resultTitle { font-weight: 700; }
"""


def apply_theme(app: QApplication) -> None:
    app.setOrganizationName(ORG_NAME)   # QSettings 경로용
    app.setApplicationName(APP_NAME)
    # setApplicationDisplayName 은 쓰지 않는다 — Windows 에서 창 제목 뒤에
    # " - <이름>" 을 덧붙여 제목이 두 번 나온다.
    app.setStyle("Fusion")                 # 플랫폼 무관하게 QSS 가 예측 가능하게 먹는다
    app.setFont(app_font())
    app.setPalette(palette())
    app.setStyleSheet(QSS % T)
    if ICON_PATH.is_file():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
