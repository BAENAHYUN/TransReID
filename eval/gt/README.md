# eval/gt — 사람이 만드는 정답 (P6)

기준표 §2(추적·스티칭), §3·§5(객체), §6(Qwen)의 "미측정"을 없애기 위한 정답 3종. 시트(HTML)는 스크립트가 만들고, 판정만 사람이 한다.
시트는 브라우저에서 열리는 파일 하나(썸네일 내장)이며, 입력은 자동 저장(localStorage)되고 **labels.json 내려받기** 로 내보낸다.
내보낸 `labels.json` 을 시트와 같은 폴더에 두면 평가 스크립트가 읽는다. 라벨 파일이 없으면 평가는 "pseudo 모드"(파이프라인 출력을 정답으로 가정)로 돌아가며 원장에는 기록하지 않는다.

| 정답 | 폴더 | 시트 만들기 | 라벨 단위 | 평가 | 원장 stage |
|---|---|---|---|---|---|
| 추적·스티칭 | `tracks/<영상>/` | `eval/track_gt_eval.py sheet --videos …` | 구간(추적기 id 연속 구간) → gt_id / ignore / 바뀌는 프레임 | `eval/track_gt_eval.py eval` | `track` |
| 객체 재출현 | `object_pairs/` | `eval/object_pair_eval.py sheet` | 객체 트랙 쌍 → 같음 / 다름 / 모름 | `eval/object_pair_eval.py eval --vector dinov2` | `object` |
| Qwen 판정 | `qwen/` | `eval/qwen_verify_eval.py sheet` | (쿼리, 후보) → 맞다 / 아니다 / 모름 | `eval/qwen_verify_eval.py eval` | `qwen` |

GUI 평가 탭 12·13·14 단계가 같은 명령이다. 러너(`bench/run.py track|object|qwen`)와 벤치마크 탭도 이 세 단계를 안다.

## 1. 추적·스티칭 (`tracks/`) — 목표 트랙 30~50개, 영상 3~5편

**준정답(semi-GT)**: 파이프라인이 낸 검출 박스 위치는 그대로 두고, "어느 박스 묶음이 같은 사람인가"만 사람이 정한다. 그래서 이 정답은 검출이 아니라 **연관(association)** 을 잰다 — 추적기가 id 를 유지했는가, 스티처가 제대로 이었는가/잘못 합쳤는가.

- 시트 한 장 = 영상 한 편. 카드 = 스티처의 **긴 트랙**(SUSHI `long_track_id`), 카드 안의 행 = **구간**(추적기 `track_id` 가 끊기지 않고 같은 긴 트랙을 유지하는 연속 구간, 이름 `<track_id>.<k>`). 추적기 id 는 시간이 지나면 다른 사람에게 재사용되므로 id 자체는 단위가 될 수 없다.
- 기본값: 긴 트랙이 person 이면 `gt_id = P<긴 트랙 id>`, 상태 person. 파이프라인이 reject 한 긴 트랙은 접힌 목록에 있고 기본 ignore.
- 할 일: ① 다른 카드인데 같은 사람 → 같은 gt_id ② 한 카드 안에 다른 사람 → 그 행의 gt_id 를 바꿈 ③ 사람 아님/판단 불가 → ignore ④ 한 행 안에서 사람이 바뀌면 바뀌는 프레임 + 그 뒤 gt_id.
- **알아둘 것**: SUSHI 는 512 프레임 창을 독립으로 처리한다(`video/sushi_inference.py FRAMES_PER_GRAPH`). 같은 사람이 창마다 다른 긴 트랙 id 를 받는 일이 흔하다 (예: 048 영상의 정지 박스 하나가 L1·L8·L18·L28). 같은 사람이면 같은 gt_id 를 주면 된다 — 평가가 그 단절을 `splits`/`IDSW` 로 센다.
- 평가 출력: `raw`(추적기 id 그대로) · `before`(구간 = 스티처 입력) · `after`(긴 트랙 = 스티처 출력) 각각 IDF1 / HOTA / MOTA / IDSW / 단절 / 갈라짐 / 과병합. 원장 metrics 는 after 이고 `*_before`, `*_raw`, `idsw_ratio` 를 함께 남긴다.
- 다른 추적 yaml 비교: `bench/run.py track --tracking-config pipeline_tracking_yolo26.yaml` 이 GT 영상만 다시 추적·스티칭한 뒤 같은 정답으로 평가한다 (박스가 달라도 IoU≥0.5 로 대응).

현재 시트: `Normal_Videos_048_x264` · `Normal_Videos_289_x264` · `Normal_Videos_100_x264` (구간 63개 중 person 36, 긴 트랙 47 중 person 28). 다른 영상을 쓰려면 `--videos` 에 stem 을 적고 sheet 를 다시 만든다.

## 2. 객체 재출현 쌍 (`object_pairs/`) — 목표 50쌍+

운영 DB(`forensic_object`, media_type=video)의 객체 **트랙** 2,233개에서 쌍 75개를 제안했다: cluster 40(운영 Leiden 0.97 클러스터 안 쌍) · knn 25(DINOv2 유사도는 높은데 클러스터가 다른 쌍 = 경계 사례) · random 10(무작위 저유사도). 출처에 끌리지 말고 그림만 보고 **같은 개체인지** 판정한다 (같은 종류 ≠ 같음).

평가: 같음 쌍을 합쳐 identity 그룹 → 그룹 트랙을 질의로 DB 전체를 검색한 mAP/R1(미라벨 트랙 = 음성 가정, 보수적) + 라벨 트랙만의 mAP, 쌍 유사도 AUC·최적 F1 임계값·운영 임계(0.97) 정확도, 운영 클러스터의 쌍 정밀도/재현율. `--vector siglip2` 로 같은 정답에 다른 객체 임베더를 비교한다 (`tracks_<vector>.npz` 캐시가 없으면 Qdrant 에서 다시 만든다).

## 3. Qwen 판정 (`qwen/`) — 30 쿼리 × 20 후보 = 600

`qwen_queries.json` 의 한국어 쿼리 30개를 GUI 와 같은 경로(opus 번역 → SigLIP2+IRRA RRF, person 이미지 DB)로 검색해 상위 20 후보를 뽑았다. 후보 파일 `candidates/<qid>.json` 은 GUI 가 [Qwen 검증 실행] 때 `verifiers/qwen_stage.py` 에 넘기는 형식 그대로다. 시트에서 후보마다 **쿼리 설명에 맞는 사람인지** 판정한다. 쿼리 문장은 영상 내용에 맞게 바꿔도 되고(파일 수정 → sheet 다시), 번역 결과는 시트 제목 옆에 보인다.

평가: 후보 파일마다 `qwen_stage.py` 를 별도 프로세스로 실행(결과는 `eval/results/qwen_verify/<이름>/qwen/<qid>.json` 에 캐시) → P@5/10/20 검증 전(검색 순위) vs 후(Qwen 순위; filter 모드는 FAIL 제거), 오탈락률(FAIL 판정 중 실제 정답), UNKNOWN 비율, 후보당 초. "모름" 은 분자·분모에서 뺀다. 600 후보는 2B 기준 4~5 시간이므로 `--max-queries` 로 나눠 돌리고(캐시 재사용) `--rescore` 로 alpha/threshold 만 재채점한다.

## 4. 파일 형식

`labels.json` (시트가 내보냄):
```json
{"kind": "track_labels | object_pair_labels | qwen_labels", "meta": {...}, "labeler": "이니셜", "exported_at": "...",
 "labels": {"<항목 id>": {"<필드>": "<값>", ...}}}
```
- 추적: 항목 = 구간 id, 필드 gt_id / status(person|ignore) / split_frame / split_gt_id / note
- 객체: 항목 = pair_id, 필드 verdict(same|different|unsure) / note
- Qwen: 항목 = `<qid>:<rank>`, 필드 relevant(yes|no|unsure)

정답 폴더는 git 에 넣는다 (시트 HTML 은 크므로 `labels.json`·`proposals.json`·`boxes.jsonl` 만 올려도 평가는 된다; 트랙 캐시 `tracks_*.npz` 는 재생성 가능).
