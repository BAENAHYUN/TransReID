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

테스트: 350개 통과 (`python -m unittest discover -s tests`).

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
- 임베더 교체는 단독 Re-ID 평가까지 자동이고 운영 DB 적재(`ingest/build_db.py`)는 수동. 검출기 드롭다운은 class 로 중복을 없애므로 같은 class 변형은 경로 지정.
- 객체 재출현 pseudo 실행에서 최적 쌍 임계값이 0.74 로 나와 운영 Leiden 0.97 과 차이가 크다 — 라벨 후 재확인 대상.

## 5. 문서·산출물

- 기준표: `outputs/audit/eval_criteria_20260926.md` (`.html`) — 단계별 지표·채택 기준·현재값
- 로드맵: `outputs/audit/autobench_roadmap_20260926.md` (`.html`) — P0~P7 완료 표시
- 도구 설명: `bench/README.md`, 새 모델 등록 `bench/REGISTER_GUIDE.md`, 정답 라벨 `eval/gt/README.md`, 프로젝트 `README.md` 5.1·6.4
