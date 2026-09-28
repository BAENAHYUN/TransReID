#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
driver.py — search_gui.py (PySide6) 를 띄우고, 창을 캡처하고, 클릭을 보낸다. Windows 전용.

원칙: **실제 마우스 커서와 포그라운드를 건드리지 않는다.**
  - 캡처는 PrintWindow (다른 창에 가려져 있어도 GUI 자신의 내용을 그린다)
  - 클릭은 GUI 창 핸들에 WM_LBUTTONDOWN/UP 을 PostMessage
  SetForegroundWindow + 실제 클릭 방식은 Windows 포그라운드 잠금에 막히면
  클릭이 그 좌표에 있던 **다른 앱**(사용자 브라우저 등)으로 들어간다. 실제로 그랬다.

좌표는 GUI 창 **바깥 프레임 기준 (window coords)** 이며, 캡처 PNG 의 픽셀 좌표와 같다.
launch / tab / step 은 창 크기를 DEFAULT_SIZE 로 맞춘 뒤 동작하므로 이름 좌표가 유효하다.

사용 (프로젝트 루트에서):
  .venv/Scripts/python.exe .claude/skills/run-transreid/driver.py launch [--wait 90]
  ...driver.py status
  ...driver.py ss  out.png
  ...driver.py tab "이미지 파이프라인" [--ss out.png]
  ...driver.py step 4 [--ss out.png]            # 파이프라인 탭의 단계 목록 n 번째
  ...driver.py click 369 43 [--ss out.png]      # 임의 창 좌표
  ...driver.py type "data/PRW/frames"           # 포커스 위젯에 문자열 (먼저 click 으로 입력칸에 포커스)
  ...driver.py key tab --repeat 2               # tab / down / up / space / enter / backspace …
  ...driver.py quit
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # <unit>/.claude/skills/run-transreid/driver.py
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
GUI_SCRIPT = "search_gui.py"
TITLE_SUBSTR = "Forensic Visual Retrieval"
DEFAULT_SIZE = (960, 939)                            # 이름 좌표(TABS/STEP_*)는 이 크기에서 검증됐다 (셸 최소 폭 957)
LOG_PATH = Path(os.environ.get("TEMP", str(ROOT))) / "transreid_gui_driver.log"

# 창 좌표 (프레임 포함). 960x939 에서 캡처로 확인한 값.
# 2026-09-28 Immich 식 셸: 상단 탭 대신 왼쪽 사이드바 (gui/shell.py). 항목은 세로로 놓이고 x 는 모두 100.
# 옛 탭 이름도 같은 항목을 가리키게 남겨 둔다.
_NAV = {
    "사진에서 찾기": (100, 155),
    "영상에서 찾기": (100, 196),
    "영상 처리": (100, 270),
    "사진 처리": (100, 311),
    "결과 보기": (100, 384),
    "평가 / 비교": (100, 457),
    "벤치마크": (100, 498),
}
TABS = dict(_NAV)
TABS.update({
    "이미지 검색": _NAV["사진에서 찾기"],
    "영상 검색": _NAV["영상에서 찾기"],
    "영상 파이프라인": _NAV["영상 처리"],
    "이미지 파이프라인": _NAV["사진 처리"],
})
STEP_X = 340           # 파이프라인 단계 목록 (사이드바 208px 오른쪽)
STEP_Y0 = 140          # 1단계 (그룹 설명이 3~4줄로 접힌 뒤 목록 시작; 핵심 4단계만 보임)
STEP_DY = 32           # 단계 간 간격
# 그룹 설명 줄 수가 다르면 목록 시작 y 가 달라진다. 2026-09-28 캡처: 사진 처리 140. 다른 탭은 같은 값으로 두고 어긋나면 ss 로 확인.
STEP_Y0_BY_TAB = {}
# 검색 페이지: 검색창 (530,152) 클릭 → type → key enter. 결과 격자 첫 칸 (330,340) 근처.

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

WM_CLOSE = 0x0010
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_CHAR = 0x0102
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
MK_LBUTTON = 0x0001
# 키 이름 -> 가상 키 코드 (키보드 포커스를 가진 Qt 위젯에 WM_KEYDOWN/UP 으로 전달)
VKEYS = {"tab": 0x09, "enter": 0x0D, "return": 0x0D, "esc": 0x1B, "space": 0x20, "end": 0x23, "home": 0x24,
         "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28, "delete": 0x2E, "backspace": 0x08}
PW_RENDERFULLCONTENT = 0x0002
SWP_NOMOVE, SWP_NOZORDER, SWP_NOACTIVATE = 0x0002, 0x0004, 0x0010


# ---------------------------------------------------------------- window lookup
def find_windows() -> list[tuple[int, str, int]]:
    """(hwnd, title, pid) — 보이는 최상위 창 중 제목에 TITLE_SUBSTR 이 포함된 것."""
    found: list[tuple[int, str, int]] = []
    proc_t = ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)

    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        if TITLE_SUBSTR in buf.value:
            pid = wt.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            found.append((hwnd, buf.value, pid.value))
        return True

    user32.EnumWindows(proc_t(cb), 0)
    return found


def require_window() -> int:
    wins = find_windows()
    if not wins:
        sys.exit(f"GUI 창 없음 (제목에 '{TITLE_SUBSTR}'). 먼저 `driver.py launch`.")
    return wins[0][0]


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    r = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right - r.left, r.bottom - r.top


def client_origin(hwnd: int) -> tuple[int, int]:
    p = wt.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(p))
    return p.x, p.y


def ensure_size(hwnd: int) -> None:
    """이름 좌표가 맞도록 창 크기를 DEFAULT_SIZE 로 (위치는 유지)."""
    _, _, w, h = window_rect(hwnd)
    if (w, h) != DEFAULT_SIZE:
        user32.SetWindowPos(hwnd, 0, 0, 0, DEFAULT_SIZE[0], DEFAULT_SIZE[1],
                            SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE)
        time.sleep(0.4)


# ---------------------------------------------------------------- capture
class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG),
                ("biPlanes", wt.WORD), ("biBitCount", wt.WORD), ("biCompression", wt.DWORD),
                ("biSizeImage", wt.DWORD), ("biXPelsPerMeter", wt.LONG), ("biYPelsPerMeter", wt.LONG),
                ("biClrUsed", wt.DWORD), ("biClrImportant", wt.DWORD)]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wt.DWORD * 3)]


def screenshot(hwnd: int, out: Path) -> Path:
    from PIL import Image  # .venv 에 있음

    _, _, w, h = window_rect(hwnd)
    hdc = gdi32.CreateCompatibleDC(None)
    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = w
    bmi.bmiHeader.biHeight = -h            # top-down
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bits = ctypes.c_void_p()
    hbmp = gdi32.CreateDIBSection(hdc, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0)
    old = gdi32.SelectObject(hdc, hbmp)
    ok = user32.PrintWindow(hwnd, hdc, PW_RENDERFULLCONTENT)
    gdi32.GdiFlush()
    raw = ctypes.string_at(bits, w * h * 4)
    gdi32.SelectObject(hdc, old)
    gdi32.DeleteObject(hbmp)
    gdi32.DeleteDC(hdc)
    if not ok:
        sys.exit("PrintWindow 실패")
    img = Image.frombuffer("RGBA", (w, h), raw, "raw", "BGRA", 0, 1).convert("RGB")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"screenshot {w}x{h} -> {out}")
    return out


# ---------------------------------------------------------------- input (message based)
def click_window_coords(hwnd: int, wx: int, wy: int) -> None:
    """창 좌표 (프레임 포함) -> client 좌표로 바꿔 WM_LBUTTONDOWN/UP 을 보낸다."""
    left, top, _, _ = window_rect(hwnd)
    ox, oy = client_origin(hwnd)
    cx, cy = (left + wx) - ox, (top + wy) - oy
    if cx < 0 or cy < 0:
        sys.exit(f"client 좌표가 음수: ({cx},{cy}) — 프레임 영역을 클릭하려는 중")
    lparam = (cy << 16) | (cx & 0xFFFF)
    user32.PostMessageW(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)
    time.sleep(0.08)
    user32.PostMessageW(hwnd, WM_LBUTTONUP, 0, lparam)
    time.sleep(0.9)                         # Qt 가 위젯을 다시 그릴 시간
    print(f"click window({wx},{wy}) -> client({cx},{cy})")


def send_key(hwnd: int, name: str, repeat: int = 1) -> None:
    """이름 키(tab/down/space/enter …) 를 WM_KEYDOWN/WM_KEYUP 으로 보낸다. 포커스 위젯이 받는다.
    (Tab = 다음 위젯으로 포커스 이동, Down = 콤보박스 다음 항목 — 팝업을 열지 않고 바꾼다, Space = 체크박스 토글)"""
    vk = VKEYS.get(name.lower())
    if vk is None:
        sys.exit(f"알 수 없는 키 '{name}'. 가능: {sorted(VKEYS)}")
    for _ in range(max(1, repeat)):
        user32.PostMessageW(hwnd, WM_KEYDOWN, vk, 0)
        time.sleep(0.03)
        user32.PostMessageW(hwnd, WM_KEYUP, vk, 0xC0000000)
        time.sleep(0.12)
    time.sleep(0.4)
    print(f"key {name} x{repeat}")


def send_text(hwnd: int, text: str) -> None:
    """문자열을 WM_CHAR 로 한 글자씩 보낸다 (QLineEdit/QSpinBox 등 포커스 위젯에 입력). 백슬래시 대신 / 권장."""
    for ch in text:
        user32.PostMessageW(hwnd, WM_CHAR, ord(ch), 0)
        time.sleep(0.02)
    time.sleep(0.4)
    print(f"type {text!r}")


# ---------------------------------------------------------------- commands
def cmd_launch(args) -> None:
    wins = find_windows()
    if wins:
        hwnd, title, pid = wins[0]
        print(f"already running: pid={pid} hwnd={hwnd} title='{title}'")
    else:
        if not VENV_PY.is_file():
            sys.exit(f"venv python 없음: {VENV_PY}")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        log = open(LOG_PATH, "a", encoding="utf-8")
        log.write(f"\n=== launch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"); log.flush()
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen([str(VENV_PY), GUI_SCRIPT], cwd=str(ROOT), env=env,
                                stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        print(f"spawned pid={proc.pid}, waiting for window (max {args.wait}s) … log: {LOG_PATH}")
        t0 = time.time()
        while time.time() - t0 < args.wait:
            wins = find_windows()
            if wins:
                break
            if proc.poll() is not None:
                sys.exit(f"GUI 프로세스가 창을 띄우기 전에 종료됨 (exit {proc.returncode}). 로그: {LOG_PATH}")
            time.sleep(0.5)
        if not wins:
            sys.exit(f"{args.wait}s 안에 창이 나타나지 않음. 로그: {LOG_PATH}")
        hwnd, title, pid = wins[0]
        print(f"window up after {time.time() - t0:.1f}s: pid={pid} title='{title}'")
    ensure_size(hwnd)
    x, y, w, h = window_rect(hwnd)
    print(f"rect: {x},{y} {w}x{h}")
    if args.ss:
        screenshot(hwnd, args.ss)


def cmd_status(_args) -> None:
    wins = find_windows()
    if not wins:
        print("not running")
        return
    for hwnd, title, pid in wins:
        x, y, w, h = window_rect(hwnd)
        print(f"pid={pid} hwnd={hwnd} rect={x},{y} {w}x{h} title='{title}'")


def cmd_ss(args) -> None:
    screenshot(require_window(), args.out)


def cmd_click(args) -> None:
    hwnd = require_window()
    click_window_coords(hwnd, args.x, args.y)
    if args.ss:
        screenshot(hwnd, args.ss)


def cmd_key(args) -> None:
    hwnd = require_window()
    send_key(hwnd, args.name, args.repeat)
    if args.ss:
        screenshot(hwnd, args.ss)


def cmd_type(args) -> None:
    hwnd = require_window()
    send_text(hwnd, args.text)
    if args.ss:
        screenshot(hwnd, args.ss)


def cmd_tab(args) -> None:
    if args.name not in TABS:
        sys.exit(f"알 수 없는 탭 '{args.name}'. 가능: {list(TABS)}")
    hwnd = require_window()
    ensure_size(hwnd)
    click_window_coords(hwnd, *TABS[args.name])
    if args.ss:
        screenshot(hwnd, args.ss)


def cmd_step(args) -> None:
    hwnd = require_window()
    ensure_size(hwnd)
    if args.tab:
        click_window_coords(hwnd, *TABS[args.tab])
    y0 = STEP_Y0_BY_TAB.get(args.tab or "", STEP_Y0)
    click_window_coords(hwnd, STEP_X, y0 + STEP_DY * (args.n - 1))
    if args.ss:
        screenshot(hwnd, args.ss)


def cmd_quit(args) -> None:
    wins = find_windows()
    if not wins:
        print("not running")
        return
    for hwnd, _title, pid in wins:
        user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
        t0 = time.time()
        while time.time() - t0 < args.wait and any(w[2] == pid for w in find_windows()):
            time.sleep(0.3)
        if any(w[2] == pid for w in find_windows()):
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
            print(f"pid={pid}: WM_CLOSE 무응답 -> taskkill /F")
        else:
            print(f"pid={pid}: closed ({time.time() - t0:.1f}s)")


def main() -> None:
    if os.name != "nt":
        sys.exit("Windows 전용 드라이버입니다 (user32/gdi32).")
    # 창 제목의 '·' 가 cp949 콘솔에서 '??' 로 깨진다. 출력만 utf-8 로 고정.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("launch", help="GUI 기동 (이미 떠 있으면 재사용) + 창 크기 정규화")
    s.add_argument("--wait", type=int, default=90); s.add_argument("--ss", default=None); s.set_defaults(fn=cmd_launch)
    s = sub.add_parser("status"); s.set_defaults(fn=cmd_status)
    s = sub.add_parser("ss", help="PrintWindow 캡처"); s.add_argument("out"); s.set_defaults(fn=cmd_ss)
    s = sub.add_parser("click", help="창 좌표 클릭 (메시지)"); s.add_argument("x", type=int); s.add_argument("y", type=int)
    s.add_argument("--ss", default=None); s.set_defaults(fn=cmd_click)
    s = sub.add_parser("key", help="포커스 위젯에 키 (tab/down/up/space/enter/backspace …)"); s.add_argument("name")
    s.add_argument("--repeat", type=int, default=1); s.add_argument("--ss", default=None); s.set_defaults(fn=cmd_key)
    s = sub.add_parser("type", help="포커스 위젯에 문자열 입력 (WM_CHAR)"); s.add_argument("text")
    s.add_argument("--ss", default=None); s.set_defaults(fn=cmd_type)
    s = sub.add_parser("tab", help="사이드바 페이지 전환 (옛 탭 이름도 허용)"); s.add_argument("name"); s.add_argument("--ss", default=None); s.set_defaults(fn=cmd_tab)
    s = sub.add_parser("step", help="파이프라인 단계 목록 n 번째 선택"); s.add_argument("n", type=int)
    s.add_argument("--tab", default=None, help="먼저 이 탭으로 전환 (예: '이미지 파이프라인')")
    s.add_argument("--ss", default=None); s.set_defaults(fn=cmd_step)
    s = sub.add_parser("quit", help="WM_CLOSE 로 정상 종료 (무응답이면 taskkill)"); s.add_argument("--wait", type=int, default=10)
    s.set_defaults(fn=cmd_quit)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
