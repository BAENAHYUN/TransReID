# 할 일 우선순위 맵 (2026-09-28)

의존 흐름: ① git push → ② Qwen 배관 확인(GPU 터미널) → ③ 라벨링(사람: 추적 → 객체 → Qwen; 시트 3종 준비됨) → ④ eval 3종(GUI 평가 12·13·14) → ⑤ 기준표 §2·§3·§5·§6 실측 → ⑥ 채택 기준 절대값 확정 → ⑦ 결정 3건(스티처 512창 단절 보완 / 검색 가중치 / Qwen 모드).

## 0. 2026-09-28 저녁 진행 (자동 진행분)

| 일 | 상태 | 내용 |
|---|---|---|
| GUI Immich 식 셸 | 완료 (e009ed1) | 사이드바 · 검색 헤더 카드 · 썸네일 격자 · 상세 · '이 결과로 다시 찾기'. 실제 검색 캡처 확인 |
| 파이프라인 4단계 + 빈 칸 채우기 + 도구 선택 | 완료 (858d8f1, baf9cb7) | 검출/임베딩/클러스터/결과창, basic/고급, 드롭다운·플레이스홀더, 클러스터 Leiden/DBSCAN/커스텀 |
| Qwen 캐시 선채우기 | 배치 10 완료 · 단건 진행 중 | `eval/results/qwen_verify/qwen_flag_norerank_b10/`, `…/qwen_flag_norerank/` (단건 ≈ 3.7 h). 끝나면 `eval/qwen_compare_runs.py --a …norerank/qwen --b …_b10/qwen` |
| 라벨 후 | 대기 | 위 두 캐시로 `qwen_verify_eval.py eval`(이름 그대로) → 원장 stage=qwen 두 행(단건 vs _b10) → 채택 |

## 1순위 — 오늘 (도구는 끝났고 손만 대면 됨)

| # | 일 | 누가 | 소요 | 내용 |
|---|---|---|---|---|
| ① | git push | 사람 | 5분 | P0~P7 커밋·병합 완료. P6 파일 60개(11 MB) 추가 커밋 후 `git push origin main`. `git add .` 금지(outputs/ 18만 파일) |
| ② | GPU 복구 → Qwen 배관 확인 | — | **완료 (16:03)** | GPU 오류 상태 복구 뒤 1쿼리·5후보 데모 통과: CUDA 적재 15 s, 후보당 21.7 s, UNKNOWN 1/5. 재랭커(qwen-vl-utils)는 미설치라 건너뜀 |
| ③ | 라벨링 — 추적 시트 3편 | 사람 | 1~2시간 | `eval/gt/tracks/{048,289,100}/sheet.html` → gt_id 확인 → labels.json 내려받기 → 같은 폴더. SUSHI 512 프레임 창 때문에 같은 사람이 여러 L 번호로 나뉜 것을 같은 gt_id 로 묶는 게 핵심 |
| ③ | 라벨링 — 객체 쌍 75 | 사람 | 30분 | `eval/gt/object_pairs/sheet.html` · 같은 개체인지만 (같음/다름/모름) |
| ③ | 라벨링 — Qwen 600 | 사람 | 2~3시간 | `eval/gt/qwen/sheet.html` · 쿼리 설명에 맞는 사람인지. 쿼리 문장은 `eval/gt/qwen_queries.json` 수정 후 시트 재생성 가능 |
| ④ | eval 3종 → 원장 | 도구 | 추적·객체 1분, Qwen 4~5시간 GPU | GUI 평가 12·13·14 의 명령을 eval 로. Qwen 은 `--max-queries 5` 분할(캐시). 벤치마크 탭 stage track/object/qwen |

## 2순위 — 라벨 결과가 나온 뒤

| # | 일 | 누가 | 소요 | 내용 |
|---|---|---|---|---|
| ⑤ | 기준표 채우기 | 도구 | 30분 | §2 추적·§3/§5 객체·§6 Qwen "도구 준비" → 실측(원장 자동 추출) |
| ⑥ | 채택 기준 절대값 확정 | 도구 | 10분 | 첫 실측 = 기존 → `bench/criteria.py` 에 IDF1/HOTA(−1 %p), 객체 mAP 기준 |
| ⑦-a | 스티처 512 창 단절 보완 여부 | 도구 | 반나절 | after 의 splits/IDSW 가 크면 창 경계 연결 후처리 또는 다른 스티처를 `bench/run.py track --tracking-config` 로 비교 |
| ⑦-b | 검색 가중치 채택 | 사람 | 5분 | `pipeline_best.yaml`(mAP 89.36 vs 운영 88.9, 이득 작음). Leiden 추천값은 거부 유지 |
| ⑦-c | Qwen 모드·임계 | 도구 | 1시간 | `--rescore` 로 flag/filter·threshold 스윕 → P@10 +10 %p, 오탈락 ≤10 % |

## 3순위 — 도구 보완 (급하지 않음)

- ~~임베더 교체 시 운영 DB 적재 자동화~~ (17:10 완료: `bench/register.py embedder --ingest-frames N` → 별도 컬렉션 bench_<이름>_* 표본 적재 + e2e 비교; 전체 적재는 여전히 수동)
- ~~검출기 드롭다운 중복 제거 개선~~ (16:50 완료: 검출기 블록이 같은 사본만 합침 → yolo26s·sushi_link 변형이 보임)
- ~~클러스터 조합 공정 비교 재실행~~ (16:36 완료: leiden@solider 1위 유지, 미세조정 B³F1 0.937 / holdout 0.876; siglip2·irra 벡터는 클러스터링에 부적합 — 잡음 92 %)
- ~~yolo26s 전체 프레임 벤치~~ (16:16 완료: AP@0.5 0.870 / 최대 재현율 0.944 / 34.8 fps — yolo26m 0.876·39 fps, RF-DETR 0.876·15.6 fps 대비 미채택)
- Astra 지적 중 남긴 한계: ~~클러스터 벡터 캐시 DB build id~~(17:40 해결: 캐시 이름에 embedding_build_id), ~~통합 검색 eval 모델 3개 고정~~(17:30 해결: yaml person retriever 전부), e2e gallery=test 혼합 프로토콜(문서화된 한계)
- 저장소 정리: `github_upload/` 삭제 여부, 루트 `.bak`·로그·임시 json, `outputs/` gitignore (30분)

## 한 줄 판단

- 지금 당장: ① push → ② Qwen 배관 확인 → ③ 추적 시트 라벨링. 도구 쪽은 라벨이 들어오기 전까지 할 일이 없음.
- 막히면: Qwen 이 GPU 에서도 죽으면 로그, 시트가 안 열리면 브라우저 이름. 애매한 사례는 "모름" → 평가 제외.
- 끝나는 조건: 기준표 §0 의 "도구 준비" 4칸이 실측으로, 벤치마크 탭에 track/object/qwen 행이 채택 기준 색으로 보이면 P0~P7 종료.
