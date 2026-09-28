---
name: run-transreid
description: Run, launch, screenshot and drive the TransReID forensic search GUI (search_gui.py, PySide6) on Windows, and validate its pipeline definitions without opening a window. Use when asked to start or open the GUI, take a screenshot of it, click through its tabs/pipeline steps, or check that gui_pipelines.json loads.
---

TransReID 는 RF-DETR → SigLIP2/IRRA/SOLIDER/DINOv2 → Qdrant 포렌식 검색 도구이고,
사용자 표면은 PySide6 데스크톱 GUI `search_gui.py` 다. 에이전트는
`.claude/skills/run-transreid/driver.py` 로 GUI 를 띄우고·캡처하고·클릭한다.
드라이버는 **실제 마우스 커서나 포그라운드를 건드리지 않는다** — 캡처는
`PrintWindow`, 클릭은 창 핸들로 `WM_LBUTTONDOWN/UP` 메시지. 아래 명령은 전부
2026-09-18 이 머신(Windows 11)에서 실행해 동작을 확인한 것이다.

**2026-09-28 화면 구조(Immich 식 셸, `gui/shell.py`)**: 상단 탭이 없고 왼쪽 사이드바에 `검색`(사진에서 찾기 / 영상에서 찾기) ·
`자료 만들기`(영상 처리 / 사진 처리) · `결과`(결과 보기: 라벨링 시트·리포트 열기) · `평가`(평가 / 비교 / 벤치마크) 항목이 세로로 있다. 검색 페이지는 상단 카드(자연어/사진 모드 ·
대상 · 큰 검색창 · AI 재확인 · 고급 설정) + 아래 썸네일 격자 + 오른쪽 상세. 파이프라인 페이지는 핵심 4단계(검출/임베딩/클러스터/결과창)만
보이고 '추가 작업 보기' 로 나머지를 편다. `tab` 은 옛 탭 이름(`이미지 검색` 등)도 새 항목으로 매핑한다.

모든 경로는 프로젝트 루트 `TransReID/` 기준. **Windows 전용** — Linux/xvfb 경로는
시도하지 않았고 드라이버가 user32/gdi32 를 직접 호출한다.

## Prerequisites

이 머신에 이미 있는 것: Windows 11, 프로젝트 venv (`.venv\Scripts\python.exe`,
PySide6 · Pillow 포함), Qdrant 서버 `http://localhost:6333` (검색/파이프라인
탭이 실제로 동작하려면 필요, 창을 띄우고 둘러보는 데는 불필요).

```bash
.venv/Scripts/python.exe -c "import PySide6, PIL; print('ok')"
```

## Setup / Build

없다. venv 가 이미 구성돼 있고 빌드 단계가 없다.

## Run (agent path) — driver.py

```bash
D=.claude/skills/run-transreid/driver.py
.venv/Scripts/python.exe $D launch --ss outputs/gui_shots/main.png          # 기동 (떠 있으면 재사용) + 960x939 로 정규화 + 캡처
.venv/Scripts/python.exe $D status                              # pid / hwnd / rect
.venv/Scripts/python.exe $D tab "이미지 파이프라인" --ss outputs/gui_shots/img_pipe.png
.venv/Scripts/python.exe $D step 4 --tab "이미지 파이프라인" --ss outputs/gui_shots/step4.png
# 자연어 검색 실행 (Qdrant + 모델 필요, 첫 검색 60~90초): 검색창 클릭 → 입력 → Enter → 대기 → 캡처 (2026-09-28 실측)
.venv/Scripts/python.exe $D tab "사진에서 찾기" && .venv/Scripts/python.exe $D click 530 152 && .venv/Scripts/python.exe $D type "검은 상의를 입은 남성" && .venv/Scripts/python.exe $D key enter
.venv/Scripts/python.exe $D ss outputs/gui_shots/search_live.png            # 75초쯤 뒤
.venv/Scripts/python.exe $D quit
```

| command | 동작 |
|---|---|
| `launch [--wait 90] [--ss PNG]` | 창이 없으면 `search_gui.py` 를 분리 프로세스로 기동하고 제목에 `Forensic Visual Retrieval` 가 나타날 때까지 대기(실측 6초). 있으면 재사용. 창 크기를 960×939 로 맞춘다 (셸 최소 폭 957). |
| `status` | 창 목록: pid, hwnd, rect, 제목. 없으면 `not running`. |
| `ss OUT.png` | `PrintWindow` 캡처. 다른 창에 가려져 있어도 GUI 내용이 찍힌다. |
| `tab NAME [--ss]` | 사이드바 페이지 전환. NAME ∈ `사진에서 찾기` `영상에서 찾기` `인물 분류` `영상 처리` `사진 처리` `도구` `결과 보기` `평가 / 비교` `정답 라벨링` `벤치마크` (옛 이름 `이미지 검색` `영상 검색` `영상 파이프라인` `이미지 파이프라인` 도 같은 항목). |
| `step N [--tab NAME] [--ss]` | 파이프라인 탭 왼쪽 단계 목록의 N 번째 **보이는** 항목 선택 (기본은 핵심 4단계만 보임 — 검출/임베딩/클러스터/결과창; '추가 작업 보기' 를 켜면 영상 6, 이미지 12). `--tab` 으로 먼저 페이지 전환. 목록은 x=340, 1단계 y=140, 간격 32 (960×939 캡처 실측; 그룹 설명 줄 수가 바뀌면 `ss` 로 다시 잰다). |
| `click X Y [--ss]` | 임의 **창 좌표**(프레임 포함, 캡처 PNG 의 픽셀 좌표와 동일) 클릭. |
| `type TEXT` / `key NAME` | 포커스 위젯에 문자열(WM_CHAR) / 키(enter, tab, down …). 검색창은 `click 530 152` 로 먼저 포커스. |
| `quit [--wait 10]` | `WM_CLOSE` 로 정상 종료(실측 0.3초). 무응답이면 `taskkill /F`. |

캡처 → `--ss` 로 준 경로 (부모 폴더 자동 생성). 기동 로그 → `%TEMP%\transreid_gui_driver.log`.
**캡처를 실제로 열어 확인한다.** 첫 화면은 사이드바 `사진에서 찾기` 가 선택된 검색 카드(자연어로 찾기 · 검색창) 여야 한다.

이름 좌표(탭·단계)는 909×939 창에서 검증됐고 `launch/tab/step` 이 그 크기를 강제한다.
다른 크기에서 `click` 을 쓰려면 먼저 `ss` 로 찍어 좌표를 읽는다.

## Direct invocation — 창 없이 파이프라인 정의 검증

GUI 의 파이프라인 탭은 전부 `gui_pipelines.json`(SSOT, **git 미추적**)에서 생성된다.
JSON 이 깨지면 GUI 는 뜨지만 파이프라인 탭이 사라진다. 대부분의 변경은 이 JSON 이라
창 없이 로더만 돌려 확인하는 게 빠르다:

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
from gui import pipeline_page as pp
for g in pp.load_registry():
    gid = g.get('id') or g.get('group') or g.get('key')
    steps = g.get('steps') or g.get('stages') or []
    print(gid, len(steps), [s['id'] for s in steps])
"
```

기대 출력: `video_pipeline 6 …`, `image_pipeline 12 …`, `evaluation 14 …` (2026-09-28: 이미지 12 = 핵심 4 `core` 단계 + 추가 8; 핵심 단계는 `core_title`, 폼은 `basic` 필드만 기본 표시).
(2026-09-26 재배치: `pipeline_page.py`/`gui_theme.py` 는 `gui/` 패키지, 단계 스크립트는
`detect/ video/ ingest/ clustering/ search/ verifiers/ report/` 아래에 있고 `gui_pipelines.json` 의
`script` 값이 그 상대경로다.)
단계가 부르는 스크립트의 플래그가 실존하는지는 각 스크립트 `--help` 와 대조한다
(`.venv/Scripts/python.exe <script>.py --help`; 임포트가 무거워 1~2분 걸릴 수 있다).

## Run (human path)

```powershell
.\.venv\Scripts\python.exe search_gui.py     # 창이 뜬다. 닫으면 종료.
```

## Test

별도 테스트 스위트는 돌리지 않았다. 이 스킬의 검증 = 위 Direct invocation 로더
출력 + 드라이버 캡처 육안 확인.

## Gotchas

- **실제 마우스 클릭을 쓰지 말 것.** `SetForegroundWindow` 는 Windows 포그라운드
  잠금에 자주 막히고, 그 상태에서 `SetCursorPos`+`mouse_event` 를 쓰면 클릭이 그
  좌표에 있던 **다른 앱**으로 들어간다. 이 세션에서 GUI 가 두 번째 모니터에서
  Chrome 뒤에 있을 때 클릭이 Chrome 북마크 바로 갔다. 드라이버는 그래서 메시지만 보낸다.
- **`CopyFromScreen` 캡처는 화면에 보이는 것을 찍는다** — GUI 앞에 다른 창이 있으면
  그 창이 찍힌다. `PrintWindow(hwnd, hdc, PW_RENDERFULLCONTENT)` 만 신뢰.
- **좌표는 창 프레임 기준.** `GetWindowRect` 원점 = PrintWindow 비트맵 (0,0). client
  좌표로 바꿀 때 `ClientToScreen` 오프셋(실측 +8, +31)을 빼야 Qt 가 맞는 위젯을 맞힌다.
  드라이버가 처리한다.
- **`.venv\Scripts\python.exe` 는 런처다.** `Popen` 이 돌려주는 pid(예 7464)와 창을
  가진 실제 python pid(예 5952)가 다르다. 프로세스는 항상 창 제목으로 찾는다.
- **창이 두 번째 모니터(x≈1926)에 뜰 수 있다.** PrintWindow 에는 무관하지만 화면
  캡처류 도구는 헛것을 찍는다.
- Qt 가 기동 시 `QFont::setPointSize: Point size <= 0 (-1)` 과
  `qt.multimedia.ffmpeg …` 를 stderr 에 찍는다. 정상.
- 콘솔이 cp949 라 창 제목의 `·` 가 `??` 로 보일 수 있다. 드라이버는 stdout 을 utf-8
  로 재설정한다; 다른 스크립트는 `PYTHONIOENCODING=utf-8` 을 붙인다.

## Troubleshooting

- **`GUI 창 없음 (제목에 'Forensic Visual Retrieval')`**: 안 떠 있다. `driver.py launch`.
- **`launch` 가 `…안에 창이 나타나지 않음`**: `%TEMP%\transreid_gui_driver.log` 를 본다.
  `gui_pipelines.json` 파싱 실패는 창은 뜨되 파이프라인 탭만 빠지고 상태바에
  "gui_pipelines.json 을 읽지 못해 파이프라인 탭이 없습니다" 가 뜬다 → 위 로더 스니펫으로
  JSON 을 확인.
- **탭/단계 클릭이 반응 없음**: 창 크기가 909×939 가 아닐 때 이름 좌표가 빗나간다.
  `driver.py launch` 를 한 번 더 부르면 크기를 맞춘다. 그래도 안 되면 `ss` 로 찍어
  좌표를 읽고 `click X Y`.
