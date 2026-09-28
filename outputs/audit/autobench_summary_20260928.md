# TransReID 자동 벤치마크 도구 — 개발 경과 요약 (2026-09-26 → 09-28)

## 0. 한 줄 요약

"AI 모델을 정량으로 비교하고, 자유롭게 갈아 끼우고, 하이퍼파라미터를 자동으로 찾고, 남이 재현할 수 있고, 새 모델이 오면 최적 조합을 다시 찾는" 도구를 3일에 걸쳐 P0→P7 로드맵으로 만들었다.
남은 것은 사람 손이 필요한 정답 라벨링(P6) 하나뿐이며, 그 라벨을 읽는 시트·평가 스크립트는 이미 준비돼 있다.

| 요구 | 답 | 어디서 |
|---|---|---|
| 정량 비교 | 검출·임베딩·검색·클러스터링·전체 파이프라인(e2e) 5단계는 GT 로 측정되어 원장에 쌓이고, 추적·객체·Qwen 3단계는 정답 시트가 준비됨 | `bench/ledger.jsonl`, GUI 벤치마크 탭 |
| 자유 교체 | 검출기·클러스터러·임베더는 yaml 블록 하나(module/class/params)로 갈아 끼움. 새 모델 등록은 명령 1줄 | `bench/register.py`, 드롭다운 |
| 하이퍼파라미터 자동 | Optuna(TPE) + 격자 조합 탐색, tune 으로 찾고 holdout 으로 검증 | `bench/optimize.py`, `bench/combos.py` |
| 재현 | 원장 행마다 git·yaml·가중치 sha·하드웨어·명령이 남고 `verify` 가 같은 설정으로 재실행해 허용 오차 판정 | `bench/run.py verify` |
| 새 모델 → 최적 조합 | 등록 → 계약 검사 → 단계 벤치 → 원장 순위 → 조합 탐색 | `register.py` → `combos.py` |

## 1. 단계별 경과

| 단계 | 한 일 | 결과 (실측) |
|---|---|---|
| **P0** 검출 재측정 | PRW test 6,112 프레임 전체로 YOLO26m vs RF-DETR-M vs 운영 DB 검출 | AP@0.5 0.876 동률. YOLO26m 은 임계 0.5 에서 작은 사람 재현율 −0.10 → **미채택** (속도 2.5× 빠름은 참고) |
| **P1** 실행 원장 | 모든 평가 결과를 한 JSONL(`bench/ledger.jsonl`)에 append-only 기록. run_id·fingerprint·component·params·GT·metrics·timing·버전·git·yaml sha·명령 | 기존 결과 이관, 평가 스크립트 4종 자동 기록, GUI 평가 4단계 |
| **P2** 러너·검사·분할·e2e | `bench/run.py`(한 명령 → 실행 폴더 + 원장, `verify`), `bench/check.py`(플러그인 계약 + DB 임베딩 지문), `bench/splits.py`(pid 단위 tune 135 / holdout 315), `eval/prw_e2e_search_eval.py`(운영 DB 전체 파이프라인) | 외부 검토 4개 보완점(공통 규격·단독/전체 분리·호환성/캐시·공정 기록) 반영, Codex Astra 리뷰 지적 수정. e2e: 운영 조합 mAP 58.3 / R1 92.0 / 검출 상한 0.908 |
| **P3** 탐색 | Optuna 검색·클러스터 스터디, 검출 임계값 스윕, 제약·Pareto·tune/holdout | 검색: 3-RRF+SOLIDER 재발견 (tune 89.4 / holdout 88.9 vs 70.6). 검출: YOLO 0.42 ≈ RF-DETR 0.54 (F1 0.831). Leiden 추천값은 holdout 쌍 정밀도 0.84 < 0.90 → **거부** |
| **P4** 조합 | 임베더 부분집합×재정렬×후보 수 (50조합) / 벡터×클러스터러 격자 → 리더보드 → 상위 미세조정 → `pipeline_best.yaml` | 운영 검색 조합 34/50위, 최선 `solider→none@1000` mAP 89.36 (holdout 88.91). 클러스터: 운영 leiden@solider 1위 |
| **P5** GUI | 벤치마크 탭(원장 리더보드·채택 기준 색·상세·verify·그래프·PR 곡선) + "채택 → yaml" | 채택 기준 = 기준표 운영값(`bench/criteria.py`) |
| **P7** 등록기 | 어댑터 스켈레톤 → 계약 검사(정적 + 실제 생성) → yaml 등록(원본 불변) → 단계 벤치 → 원장 순위 | YOLO26-small 을 코드 수정 없이 등록: AP@0.5 0.874, 원장 5/10 |
| **P6** 사람 정답 | 정답 시트 3종(HTML, 썸네일 내장, 자동 저장 → labels.json) + 평가 3종 + 원장/러너/GUI 연결 | 시트 준비: 추적 3편(구간 116) · 객체 쌍 75 · Qwen 30×20. **라벨링 대기** (§3) |

테스트: 401개 통과 (`python -m unittest discover -s tests`).

## 2. 도구 구조 (한 장)

```
평가 스크립트 (GT → metrics)                 벤치 도구 (bench/)                       GUI
 eval/detect_eval_prw.py   ─┐                ledger.py   원장 기록·조회·이관            평가 탭 1~14
 eval/prw_eval.py           │ 자동 기록      run.py      한 명령 실행 + verify           벤치마크 탭 (리더보드·그래프·채택)
 eval/prw_eval_unified.py   ├──────────────▶ check.py    계약·DB 지문 검사               드롭다운 = pipeline*.yaml glob
 eval/prw_cluster_gt_eval.py│                splits.py   tune/holdout 분할
 eval/prw_e2e_search_eval.py│                optimize.py Optuna·스윕
 eval/track_gt_eval.py      │ (P6)           combos.py   조합 격자 → pipeline_best.yaml
 eval/object_pair_eval.py   │                criteria.py 채택 기준·채택→yaml
 eval/qwen_verify_eval.py  ─┘                register.py 새 모델 등록기
```

핵심 규칙 세 가지: ① 평가 스크립트는 자기 결과를 원장에 직접 쓴다(러너를 안 거쳐도 남는다) ② `pipeline.yaml` 원본은 절대 바꾸지 않고 사본 yaml 로 교체한다(운영 DB 임베딩 sha 와 묶여 있음) ③ 탐색은 tune, 최종 비교는 holdout.

## 3. 남은 일

1. **라벨링 (사람, 2~3일)** — `eval/gt/README.md` 절차대로: 추적 시트 3편(브라우저에서 gt_id 확인) → 객체 쌍 75 → Qwen 600 판정. 각각 `labels.json` 을 시트 폴더에 두고 `eval` 실행(GUI 평가 12·13·14). 그러면 기준표 §2·§3(객체)·§5(객체)·§6 의 "도구 준비" 가 실측으로 바뀐다.
2. **채택 결정** — 검색 가중치 `pipeline_best.yaml`(mAP +1.3, 운영 조합과 차이 작음), Leiden 추천값(정밀도 미달로 권장 안 함).
3. **Qwen 실측** — 판정 시트 라벨 후 `eval/qwen_verify_eval.py eval --max-queries 5` 부터 (2B, 후보당 약 26 s). 데모 실행에서 `qwen_stage.py` 가 CPU 로 잡히며 프로세스가 죽었다(0xC0000005) — 원인은 §4.
4. **git** — P0~P7 은 커밋됨(`9ecfcf5`, 이후 원격 이력 병합 `95de57d`). P6 파일은 아직 미커밋.

## 4. 알아둘 한계·발견

- **SUSHI 는 512 프레임 창을 독립 처리**해 같은 사람이 창마다 다른 긴 트랙 id 를 받는다(048 영상의 정지 박스 하나가 L1·L8·L18·L28). 추적기 id 도 시간이 지나면 재사용된다(한 id 가 5 명). 그래서 추적 정답의 단위를 "구간"으로 잡았고, 라벨이 붙으면 이 단절이 `splits`/IDSW 로 정량화된다.
- **Qwen 배관 확인 완료 (16:03)**: 처음엔 RTX 5090 이 장치 오류 상태(Status=Error, CUDA 불가)라 `qwen_stage.py` 가 CPU 로 올라가다 죽었고, GPU 복구 뒤 1쿼리·5후보 데모가 통과했다 — CUDA 적재 15 s, 후보당 21.7 s(문서의 26 s 와 일치), UNKNOWN 1/5, 재랭커는 미설치(qwen-vl-utils)로 건너뜀. 관찰하지 않은 꼬리 후보를 UNKNOWN 으로 세던 집계도 고쳤다. 시트·후보 파일·평가 코드는 준비돼 있으니 GPU 가 잡히는 사용자 터미널에서 `eval/qwen_verify_eval.py eval --allow-unlabeled --max-queries 1 --top-k 5` 로 배관을 확인한 뒤 라벨링 후 본 실행을 하면 된다 (`--python` 으로 인터프리터 지정 가능).
- 임베더 교체는 단독 Re-ID 평가 + 표본 적재·e2e 비교까지 자동(`--ingest-frames`)이고 운영 DB 전체 적재(`ingest/build_db.py`)는 수동. 검출기 드롭다운은 검출기 블록이 같은 사본만 합치므로 yolo26s 같은 변형도 항목으로 보인다(09-28 16:50 수정).
- 객체 재출현 pseudo 실행에서 최적 쌍 임계값이 0.74 로 나와 운영 Leiden 0.97 과 차이가 크다 — 라벨 후 재확인 대상.
- **외부 검토 2회 반영(09-28 16:09 / 16:41)**: 추적 지표를 TrackEval 정의로 재구현하고(HOTA·CLEAR·MOT ignore), 라벨 유효성(검토 체크·manifest·검토율·중복/모순 차단)과 Qwen 평가의 후보 연결·filter 재현·캐시 계약, 채택 판정의 `incomplete`(확인 불가)를 넣었다. **Astra 2차(16:41) 반영**: 정식 평가에서 검토 안 된 구간은 ignore(제안값 승격 없음), reviewed 는 bool true 만, manifest·kind 필수, 같은 프레임 같은 gt_id 중복·객체 라벨 모순은 중단, CLEAR 를 TrackEval 의 빈 프레임 건너뛰기·직전 프레임 대응 규칙으로, Qwen 은 쿼리 단위 검토율과 재채점 캐시의 원본 결속, 객체는 assignments 대응률·캐시 검증 강화, 창 연결 근거 기록(`links`), 러너 `track --restitch`(스티처만 재실행, SUSHI 입력 없으면 어댑터 재생성), 벤치마크 탭 그래프의 incomplete·미정 기준선 처리와 채택 버튼은 pass 에서만.
- **SUSHI 창 경계 연결**: `pipeline_tracking_sushi_link.yaml`(stitcher link_windows) + `bench/run.py track --restitch` 로 같은 검출 위에서 비교. 창 경계 연결(restitch, 같은 검출·추적 출력, 3편 pseudo GT = 추적기 연속성 기준): 원 스티처 after IDF1 0.619 / IDSW 41 / 갈라짐 32 / 과병합 8 → 연결 후 IDF1 0.937 / IDSW 19 / 갈라짐 10 / 과병합 7 (048 긴 트랙 34→16, 289 26→18, 100 16→12; 거절된 후보는 간격 6·15·31 프레임). 사람 라벨이 붙어야 '같은 사람' 여부가 확정된다.
- **임베더 표본 적재 자동화(17:10)**: `bench/register.py embedder … --ingest-frames N` — 사본 yaml 의 collection_prefix 를 `bench_<이름>` 으로 바꿔 PRW test 프레임 N 개의 crop 을 별도 컬렉션에 적재하고, 같은 표본에서 e2e 검색을 새 임베더 단독 vs 운영 조합으로 원장에 남긴다(운영 컬렉션 불변, 표본 목록 고정으로 임베더 간 비교 가능). 데모(17:08~17:25, SOLIDER 가중치를 `solider_copy` 로 재등록): 계약 검사 OK → `pipeline_solider_copy.yaml` → 단독 Re-ID mAP 89.06(원장 2/4, solider 와 동률) → PRW test 300 프레임 crop 3,283 개를 `bench_solider_copy_person/_object` 에 적재(약 8분) → 같은 표본에서 e2e: `solider_copy__sample300` mAP(db) 89.15 / R1 63.5 / 0.18 s·q vs `prod__sample300`(siglip2+irra→solider) mAP(db) 87.35 / R1 63.0 / 0.56 s·q. 표본 갤러리는 GT 양성의 4 %(검출 상한 0.041)만 담으므로 GT 분모 mAP(3.5)는 무의미하고 map_db 로 비교한다 — 벤치마크 탭은 이런 행을 '표본 갤러리 — 확인 불가' 로 표시한다.

## 4b. 2026-09-28 저녁 — GUI 개편(Immich 식) · 단계별 도구 선택 · Qwen 배치

사용자 요청: "Immich 처럼 알아보기 쉽게", "빈 칸을 채우고 1~9 단계는 검출/임베딩/클러스터/결과창만", "단계마다 도구를 고르게(클러스터가 Leiden 고정)".

- **셸** (`gui/shell.py`): 상단 탭 → 왼쪽 사이드바 `검색`(사진에서 찾기 · 영상에서 찾기) / `자료 만들기`(영상 처리 · 사진 처리) / `평가`(평가·비교 · 벤치마크). 아이콘은 QPainter 로 그려 외부 파일 없음.
- **검색 화면** (`gui/search_ui.py`, `search_gui.ResultsPanel`): 자연어/사진(crop) 모드 토글 · 대상 · 큰 검색창(Enter 로 검색) · `AI 재확인 (Qwen)` · 접힌 `고급 설정`(모델·결과 수·영상당 최대·AI 대상/후보 수/배치). 결과는 150 px 썸네일 격자(순위·점수·AI 표시), 오른쪽 상세는 사람이 읽는 표 + 원본 필드, `이 결과로 다시 찾기` 로 그 crop 이 사진 검색 query 가 된다. 실제 검색 2건(사진 "검은 상의를 입은 남성", 영상 "검은 옷을 입은 사람")으로 캡처 확인.
- **파이프라인 화면** (`gui/pipeline_page.py`, `gui_pipelines.json`): 핵심 4단계(`core`)만 먼저 — 검출 / 임베딩 / 클러스터 / 결과창; 나머지 8단계는 '추가 작업 보기'. 폼은 `basic` 필드만 보이고 '고급 옵션 보기' 로 편다. 검출 단계 빈 칸 4개는 기본값·폴더 드롭다운(`choices_dirs`, crops 폴더는 훑지 않음)·값 드롭다운으로 채웠고, 빈 문자열 필드 29개에 플레이스홀더. **도구** 드롭다운(`tools`): 클러스터 = Leiden(기본) / Leiden 플러그인 러너 / DBSCAN v6 / 커스텀 yaml(clusterer: 블록); 영상 클러스터·갤러리 = 사람 / 물건. 새 단계 `결과창` = `report/build_image_results.py`(DB 리포트 + 갤러리 + 인덱스 한 번에, 플러그인 러너의 `<target>_<method>_assignments.jsonl` 도 인식).
- **Qwen 배치 관찰** (`verifiers/qwen_stage.py --batch-size`): 단건은 GPU 사용률 15 %·15 tok/s 로 후보당 22 s(파이썬 오버헤드 병목). left padding 배치 10 → 2~4 s/후보. 그러나 4건 중 1건의 판정이 달라져(뒷모습 성별 male→unknown) "같은 결과를 빠르게" 가 아닌 **별개 변형**으로 취급: 캐시 계약·이름(`_bN`)·원장 component 에 기록, GUI 기본 1. 캐시 선채우기: 배치 10 은 30 쿼리 × 20 후보 37 분에 완료, 단건은 진행 중(약 3.7 시간). `eval/qwen_compare_runs.py` 로 두 실행의 판정 일치율·뒤집힘·순위 상관을 라벨 없이 비교(데모 5 후보: 전부 일치, 47 s vs 4.6 s). 어느 쪽이 맞는지는 라벨 후 `qwen_verify_eval.py eval` 이 정한다.
- 테스트 401 · 커밋 858d8f1 → baf9cb7 → e009ed1 → (polish) · 드라이버 좌표 갱신(사이드바, type/key 로 검색 실행).

## 5. 문서·산출물

- 기준표: `outputs/audit/eval_criteria_20260926.md` (`.html`) — 단계별 지표·채택 기준·현재값
- 로드맵: `outputs/audit/autobench_roadmap_20260926.md` (`.html`) — P0~P7 완료 표시
- 도구 설명: `bench/README.md`, 새 모델 등록 `bench/REGISTER_GUIDE.md`, 정답 라벨 `eval/gt/README.md`, 프로젝트 `README.md` 5.1·6.4
