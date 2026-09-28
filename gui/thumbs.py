#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gui/thumbs.py — 정사각 썸네일 (비율 유지, 가운데 배치). 검색 격자와 인물 분류 카드가 같이 쓴다."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPixmap

from gui.gui_theme import T


def square_thumb(path: Any, size: int) -> Optional[QPixmap]:
    if not path:
        return None
    p = Path(str(path))
    if not p.is_file():
        return None
    pix = QPixmap(str(p))
    if pix.isNull():
        return None
    scaled = pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    canvas = QPixmap(size, size)
    canvas.fill(QColor(T["surface2"]))
    painter = QPainter(canvas)
    painter.drawPixmap((size - scaled.width()) // 2, (size - scaled.height()) // 2, scaled)
    painter.end()
    return canvas
