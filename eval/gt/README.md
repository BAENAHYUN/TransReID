# eval/gt — 사람이 만드는 정답 (P6)

기준표 §2(추적·스티칭), §3·§5(객체), §6(Qwen)의 "미측정"을 없애기 위한 정답 3종. 시트(HTML)는 스크립트가 만들고, 판정만 사람이 한다.
시트는 브라우저에서 열리는 파일 하나(썸네일 내장)이며, 입력은 자동 저장(localStorage)되고 **labels.json 내려받기** 로 내보낸다.
내보낸 `labels.json` 을 시트와 같은 폴더에 두면 평가 스크립트가 읽는다.

**세 가지 규칙 (2026-09-28 외부 검토 반영)**
1. **검토 체크가 정답의 조건이다.** 항목마다 "검토" 체크가 있고, 필드를 고치면 자동으로 체크된다. 제안값(기본값)을 그대로 두고 확인만 했다면 체크를 눌러야 한다.
   검토되지 않은 항목은 사람 정답으로 쓰지 않으며, 검토율(coverage)이 `--min-coverage`(기본 100 %) 미만이면 평가는 **pseudo**(제안값 = 정답 가정)로 취급되어
   원장에 기록되지 않고 결과 이름에 `__pseudo` 가 붙는다. 부분 라벨로라도 보려면 `--min-coverage 0.5` 처럼 낮춘다(그래도 원장에는 안 들어감).
2. **manifest 가 맞아야 한다.** 시트를 만들 때 항목 집합의 해시(manifest)가 proposals.json 과 시트에 박힌다. labels.json 은 그 manifest 를 품고 나오며,
   시트를 다시 만들어 항목이 달라지면 옛 labels.json 은 평가가 거부한다(`--ignore-manifest` 로 강행 가능). 브라우저 자동 저장도 manifest 별로 분리된다.
3. **채택 판정은 기준 지표가 다 있고 기준값이 정해졌을 때만 통과한다.** 지표가 빠지거나 기준값(기존 IDF1/HOTA, 객체 mAP)이 아직 없으면 벤치마크 탭에 `? 확인 불가` 로 나온다.
   첫 라벨 측정이 끝나면 그 값을 "기존" 으로 `bench/criteria.py` 에 넣어야 `✓/✗` 가 나온다.

| 정답 | 폴더 | 시트 만들기 | 라벨 단위 | 평가 | 원장 stage |
|---|---|---|---|---|---|
| 추적·스티칭 | `tracks/<영상>/` | `eval/track_gt_eval.py sheet --videos …` | 구간(추적기 id 연속 구간) → gt_id / ignore / 바뀌는 프레임 | `eval/track_gt_eval.py eval` | `track` |
| 객체 재출현 | `object_pairs/` | `eval/object_pair_eval.py sheet` | 객체 트랙 쌍 → 같음 / 다름 / 모름 | `eval/object_pair_eval.py eval --vector dinov2` | `object` |
| Qwen 판정 | `qwen/` | `eval/qwen_verify_eval.py sheet` | (쿼리, 후보) → 맞다 / 아니다 / 모름 | `eval/qwen_verify_eval.py eval` | `qwen` |

GUI 평가 탭 12·13·14 단계가 같은 명령이다. 러너(`bench/run.py track|object|qwen`)와 벤치마크 탭도 이 세 단계를 안다.

## 1. 추적·스티칭 (`tracks/`) — 목표 트랙 30~50개, 영상 3~5편

**준정답(semi-GT)**: 파이프라인이 낸 검출 박스 위치는 그대로 두고, "어느 박스 묶음이 같은 사람인가"만 사람이 정한다. 그래서 이 정답은 검출이 아니라 **연관(association)** 을 잰다 —
추적기가 id 를 유지했는가, 스티처가 제대로 이었는가/잘못 합쳤는가. 같은 검출 위에서 추적기·스티처를 바꾼 출력끼리 비교하는 용도이며, 검출기까지 바꾸면 기존 검출기가 놓친 사람은
GT 에 없어 새 검출이 FP 로 잡힌다(검출기 비교는 `eval/detect_eval_prw.py` 의 PRW 박스 GT 로).

- 시트 한 장 = 영상 한 편. 카드 = 스티처의 **긴 트랙**(SUSHI `long_track_id`, 구간의 다수 id), 카드 안의 행 = **구간**(추적기 `track_id` 가 끊기지 않는 연속 구간, 이름 `<track_id>.<k>`).
  구간은 스티처와 무관하게 추적기만으로 자른다. 추적기는 한 사람으로 이었는데 스티처가 긴 트랙을 나눈 구간에는 <span>창 경계</span> 표시(걸친 L 번호)가 붙는다.
- 기본값: 긴 트랙이 person 이면 `gt_id = P<긴 트랙 id>`, 상태 person. 파이프라인이 reject 한 긴 트랙은 접힌 목록에 있고 기본 ignore(묶음 단위 "모두 확인함" 버튼 있음).
- 할 일: ① 다른 카드인데 같은 사람 → 같은 gt_id ② 한 카드 안에 다른 사람 → 그 행의 gt_id 를 바꿈 ③ 사람 아님/판단 불가 → ignore ④ 한 행 안에서 사람이 바뀌면 바뀌는 프레임 + 그 뒤 gt_id ⑤ 행마다 검토 체크.
- **알아둘 것**: SUSHI 는 512 프레임 창을 독립으로 처리한다(`video/sushi_inference.py FRAMES_PER_GRAPH`). 같은 사람이 창마다 다른 긴 트랙 id 를 받는 일이 흔하다
  (예: 048 영상의 정지 박스 하나가 L1·L8·L18·L28). 같은 사람이면 같은 gt_id 를 주면 된다 — 평가가 그 단절을 after 의 `splits`/`IDSW` 로 센다.
  창 경계 연결 후처리(`pipeline_tracking_sushi_link.yaml`, stitcher `link_windows`)를 `bench/run.py track --tracking-config` 로 같은 정답에 비교할 수 있다.
- 평가 출력: `raw`(추적기 id 그대로, 재사용 포함) · `before`(구간 = 추적기 tracklet) · `after`(긴 트랙 = 스티처 출력) 각각
  IDF1/IDP/IDR(Identity), HOTA/DetA/AssA, MOTA·IDSW·단절(fragments) — 모두 TrackEval 정의를 따른다 — 와 보조 진단 갈라짐(splits)·과병합(over_merges)·동시 중복 박스(pred_dup_boxes: 한 프레임에 같은 예측 id 두 개).
  ignore 박스는 MOTChallenge 방식(정상+ignore GT 를 함께 일대일 매칭한 뒤 ignore 에 붙은 예측만 제거). 원장 metrics 는 after 이고 `*_before`, `*_raw`, `idsw_ratio`, `coverage` 를 함께 남긴다.
- 채택 기준(`bench/criteria.py` track): 과병합 0 · 스티칭 후 IDSW ≤ 구간(before) IDSW 의 50 % · IDF1/HOTA ≥ 기존(첫 라벨 측정 뒤 채움 → 그 전엔 "확인 불가").

현재 시트: `Normal_Videos_048_x264` · `Normal_Videos_289_x264` · `Normal_Videos_100_x264`. 다른 영상을 쓰려면 `--videos` 에 stem 을 적고 sheet 를 다시 만든다.
라벨 우선순위(외부 검토 권고): 창 경계 양쪽의 같은 사람, 교차·가림·재등장, 추적기 id 재사용, 동시에 존재하는 다른 사람, reject/ignore 제안의 재확인.

## 2. 객체 재출현 쌍 (`object_pairs/`) — 목표 50쌍+

운영 DB(`forensic_object`, media_type=video)의 객체 **트랙** 2,233개에서 쌍 75개를 제안했다: 운영 Leiden 0.97 클러스터 안 쌍 40 · DINOv2 유사도는 높은데 클러스터가 다른 쌍 25(경계 사례) · 무작위 저유사도 10.
시트에는 출처·유사도를 **보이지 않는다**(편향 방지; proposals.json 에만 있음). 그림만 보고 **같은 개체인지** 판정한다 (같은 종류 ≠ 같음). 판정하면 자동으로 검토 표시된다.

평가: 같음 쌍을 합쳐 identity 그룹(다름인데 같은 그룹에 든 쌍은 `label_contradictions` 로 보고) → 그룹 트랙을 질의로 DB 전체를 검색한 mAP/R1(미라벨 트랙 = 음성 가정 — 편향 방향이 고정되지 않으므로
라벨 트랙만의 `map_labeled` 와 함께 본다), 쌍 유사도 AUC·최적 F1 임계값(같은 라벨에서 고르므로 낙관적)·운영 임계(0.97) 정확도, 운영 클러스터의 쌍 정밀도/재현율(제안 출처가 섞인 표본 성능).
`--vector siglip2` 로 같은 정답에 다른 객체 임베더를 비교한다: 유사도는 현재 벡터로 다시 계산하고, 현재 캐시(`tracks_<vector>.npz` + `.meta.json`, 벡터·컬렉션·트랙 수 검증)에 없는 트랙의 쌍은 제외(`pairs_missing`)한다.
이 75쌍은 오류 분석·쌍 분류 비교용이며 전체 DB 의 identity 성능을 대표하지 않는다 — 더 넓게 보려면 제한된 갤러리 전체에 identity 를 붙이는 라벨이 필요하다.

## 3. Qwen 판정 (`qwen/`) — 30 쿼리 × 20 후보 = 600

`qwen_queries.json` 의 한국어 쿼리 30개를 GUI 와 같은 경로(opus 번역 → SigLIP2+IRRA RRF, person 이미지 DB)로 검색해 상위 20 후보를 뽑았다. 후보 파일 `candidates/<qid>.json` 은 GUI 가 [Qwen 검증 실행] 때
`verifiers/qwen_stage.py` 에 넘기는 형식 그대로이고, 후보마다 불변 id(`cand_id` = point_id)를 붙였다 — 라벨은 검색 순위로 저장되지만 평가는 불변 id 로 연결하므로 재랭커가 순위를 바꿔도 안전하다.
시트에서 후보마다 **쿼리 설명에 맞는 사람인지** 판정한다(판정하면 자동 검토). 쿼리 문장은 영상 내용에 맞게 바꿔도 되고(파일 수정 → sheet 다시), 번역 결과는 시트 제목 옆에 보인다.

평가: 후보 파일마다 `qwen_stage.py` 를 별도 프로세스로 **항상 flag 모드**로 실행해 모든 후보의 판정을 보존하고, filter 모드는 평가기가 재현한다(FAIL 제거 뒤 순위) — 그래서 filter 에서도 오탈락을 셀 수 있다.
결과는 `eval/results/qwen_verify/<이름>/qwen/<qid>.json` 에 캐시되며 캐시 계약(`<qid>.meta.json`) = 후보 파일 해시 + 모델 + 재랭커 + top_k + dtype + max_pixels. 계약이 다르면 다시 돌리고, alpha/threshold 만 다르면
저장된 관찰로 재채점한다(Qwen 호출 없음). 후보당 초는 언제나 원래 관찰 실행의 값이다.
지표: P@5/10/20 검증 전(검색 순위) vs 후(Qwen 순위) — 같은 쿼리에서 둘 다 정의된 경우만 짝지어 평균, "모름" 은 분자·분모 제외 · `false_drop_rate` = FAIL 판정 중 실제 정답 비율(기준표 정의, ≤ 10 %) ·
`lost_relevant_rate` = Qwen 이 PASS/FAIL 로 판정한 정답 중 FAIL 비율 · `unknown_ratio` = 관찰한 후보 중 UNKNOWN(판정 불능 + 처리 실패) · `sec_per_candidate`(2B ≤ 30 s, 4B ≤ 60 s).
600 후보는 2B 기준 약 3.6 시간(후보당 21.7 s 실측)이므로 `--max-queries` 로 나눠 돌린다(캐시 재사용).

## 4. 파일 형식

`labels.json` (시트가 내보냄):
```json
{"kind": "track_labels | object_pair_labels | qwen_labels", "meta": {"manifest": "…", ...}, "labeler": "이니셜", "exported_at": "...",
 "reviewed": 57, "items": 63,
 "labels": {"<항목 id>": {"<필드>": "<값>", ..., "reviewed": true}}}
```
- 추적: 항목 = 구간 id, 필드 gt_id / status(person|ignore) / split_frame / split_gt_id / note / reviewed
- 객체: 항목 = pair_id, 필드 verdict(same|different|unsure) / note / reviewed
- Qwen: 항목 = `<qid>:<rank>`, 필드 relevant(yes|no|unsure) / reviewed

정답 폴더는 git 에 넣는다 (`labels.json`·`proposals.json`·`boxes.jsonl`·후보 파일이 핵심; 트랙 캐시 `tracks_*.npz` 와 Qwen 결과 캐시는 재생성 가능해 .gitignore).
