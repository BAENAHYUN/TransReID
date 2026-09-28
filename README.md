# TransReID — 디지털 포렌식 이미지·영상 통합 검색 도구

이미지와 영상에서 사람·객체를 검출하고, 여러 임베딩 모델과 Qdrant 벡터 DB로
동일·유사 대상을 검색하는 도구입니다. 검색·DB 구축·클러스터링·평가를 하나의
PySide6 GUI(`search_gui.py`)에서 실행할 수 있고, 모든 단계는 CLI 로도 실행됩니다.

```text
입력 (이미지 / 영상 / 자연어)
  │
  ├─ 검출        RF-DETR (Medium)            영상: + BoT-SORT 추적 + SUSHI 스티칭
  │
  ├─ 임베딩      person : SigLIP2 · IRRA · SOLIDER
  │              object : SigLIP2 · DINOv2 (giant, registers)
  │
  ├─ 벡터 DB     Qdrant  forensic_person {siglip2, irra, solider}
  │                      forensic_object {siglip2, dinov2}
  │
  ├─ 검색        person 이미지 : SigLIP2 + IRRA → RRF → SOLIDER rerank
  │              object 이미지 : SigLIP2 + DINOv2 → RRF
  │              자연어        : (한→영 번역) → SigLIP2 (+ IRRA)
  │              선택          : Qwen3-VL 재순위 / 검증 (별도 프로세스)
  │
  └─ 클러스터링  Leiden (crop 단위 / 영상 track centroid) → 검수 갤러리 HTML
```

설계 문서: `TransReID_Forensic_Search_Tool.md` (별도 보관), 개별 스크립트의 docstring 에
상세 동작이 적혀 있습니다.

---

## 1. 요구 사항

아래는 실제로 동작을 확인한 환경입니다 (2026-09-20 기준).

| 항목 | 검증 환경 | 비고 |
|---|---|---|
| OS | Windows 11 Pro | PowerShell 기준으로 설명. GUI 드라이버 등 일부 도구는 Windows 전용 |
| Python | 3.11.9 | `.venv` 가상환경 |
| GPU | NVIDIA RTX 5090 32 GB, 드라이버 610.88 | CUDA 12.8 용 PyTorch 휠 사용. 임베더 4개 + 검출기를 동시에 올리므로 VRAM 16 GB 이상 권장 |
| PyTorch | 2.11.0+cu128 / torchvision 0.26.0+cu128 | |
| Qdrant | 서버 1.19.0 (Docker), 클라이언트 1.19.0 | 서버·클라이언트 버전을 맞출 것 |
| Docker Desktop | 29.x | Qdrant 실행용 |

디스크 여유 (실측): 수동 배치 가중치 약 2.3 GB, Hugging Face 캐시 약 10 GB
(SigLIP2 1.5 GB, DINOv2 giant 4.3 GB, Qwen3-VL-2B 4.0 GB 등), RF-DETR 캐시 약 0.7 GB.
Qdrant 데이터는 crop 수에 비례합니다 (기준 DB 약 71만 point).

---

## 2. 설치

### 2.1 프로젝트 폴더 준비

```powershell
git clone https://github.com/BAENAHYUN/TransReID.git
cd TransReID
```

> **주의 — 현재 git 저장소만으로는 실행이 안 됩니다.** 실행에 필요한 다수의 파일이
> 아직 커밋되지 않았습니다 (`gui_pipelines.json`, `gui/`, `detect/`, `video/`, `ingest/`,
> `clustering/`, `report/`, `tests/`, `eval/` 대부분, `third_party/SUSHI/` 등). 다른 PC 로 옮길 때는
> **프로젝트 폴더를 통째로 복사**하거나, 먼저 이 파일들을 커밋하세요.
> 자세한 목록은 [11. 저장소 상태 주의](#11-저장소-상태-주의) 참고.

`weights/`, `data/`, `storage/`(Qdrant 데이터), `.venv/` 는 git 에 포함되지 않으므로
아래 절차대로 따로 준비합니다.

### 2.2 Python 가상환경

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

PyTorch 는 CUDA 12.8 휠을 먼저 설치합니다 (`requirements.txt` 의 `+cu128` 핀과 맞춤).

```powershell
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

기본 의존성:

```powershell
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

**`requirements.txt` 에 빠져 있는 필수 패키지**를 추가로 설치합니다. 버전은 검증 환경 기준입니다.

```powershell
# GUI · 검출 · 추적 · 클러스터링 · 번역
pip install PySide6==6.11.2 rfdetr==1.9.3 boxmot==25.0.0 lapx==0.10.0 filterpy==1.4.5 `
            leidenalg==0.12.0 igraph==1.0.0 python-igraph==1.0.0 `
            sentencepiece==0.2.2 sacremoses==0.2.0 jsonlines==4.0.0 imagehash==4.3.2

# SUSHI(영상 track 스티칭) 용 PyTorch Geometric — torch 2.11.0+cu128 전용 휠 인덱스
pip install torch-geometric==2.8.0.post1
pip install torch-scatter torch-sparse pyg-lib -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
```

| 패키지 | 용도 | 없으면 |
|---|---|---|
| PySide6 | GUI | `search_gui.py` 실행 불가 |
| rfdetr | RF-DETR 검출기 | 검출 단계 전체 |
| boxmot, lapx, filterpy | BoT-SORT 추적 | 영상 파이프라인 1단계 |
| leidenalg, igraph | Leiden 클러스터링 | 클러스터링 단계 |
| torch-geometric, torch-scatter, torch-sparse, pyg-lib | SUSHI | 영상 파이프라인 1단계 |
| sentencepiece, sacremoses | OPUS-MT 한→영 번역 토크나이저 | 한국어 자연어 검색 |

`rfdetr` 설치 시 `timm` 이 1.0.x 로 올라갑니다 (검증 환경 1.0.29). `requirements.txt` 의
`timm==0.6.13` 핀보다 새 버전이지만 정상 동작합니다. 그 밖에 검증 환경과 핀이 다른
패키지: `pandas 2.3.3`, `opencv-python-headless 4.14.0.94`, `gdown 5.2.2`, `tqdm 4.70.0`.
정확히 같은 환경을 재현하려면 동작하는 venv 에서 `pip freeze > requirements.lock.txt` 를
만들어 두는 것을 권장합니다.

### 2.3 서드파티 코드

| 경로 | 내용 | 준비 방법 |
|---|---|---|
| `IRRA/` | IRRA 텍스트-이미지 Re-ID 코드 | 저장소에 포함 |
| `third_party/SOLIDER/` | SOLIDER Swin 백본 코드 | 저장소에 포함 |
| `third_party/SUSHI/` | SUSHI track 스티칭 (+ `fast-reid/`) | **git 미추적.** 폴더 복사 또는 `git clone https://github.com/dvl-tum/SUSHI.git third_party/SUSHI` 후 README 대로 fast-reid 준비 |

경로는 `pipeline.yaml` 의 `irra_root` / `solider_root` / `sushi_root` 로 지정되며,
임베더가 import 순간에만 `sys.path` 에 넣으므로 별도 `pip install` 은 필요 없습니다.
SUSHI 는 영상 파이프라인에서만 쓰입니다.

### 2.4 모델 가중치

**수동 배치** (아래 경로는 `pipeline.yaml` 과 스크립트 기본값에 그대로 적혀 있음):

| 경로 | 모델 | 크기 | 출처 |
|---|---|---|---|
| `weights/IRRA/cuhk_pedes/best.pth`, `configs.yaml` | IRRA (CUHK-PEDES) | 1.3 GB | [IRRA GitHub](https://github.com/anosorae/IRRA) → "Model & log for CUHK-PEDES" 압축을 풀어 배치 |
| `weights/IRRA/ViT-B-16.pt` | IRRA 가 쓰는 CLIP 백본 | 335 MB | [OpenAI CLIP ViT-B/16](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) |
| `weights/SOLIDER/solider_market_swin_base.pth` | SOLIDER Swin-Base, Market1501 | 335 MB | [SOLIDER-REID GitHub](https://github.com/tinyvision/SOLIDER-REID) Models 표의 Swin Base / Market1501 |
| `third_party/SUSHI/pretrained_models/mot17private.pth` | SUSHI 스티칭 모델 | 2.3 MB | [SUSHI GitHub](https://github.com/dvl-tum/SUSHI) README |
| `third_party/SUSHI/fastreid-models/model_weights/msmt_bot_R50-ibn.pth` | SUSHI 용 fast-reid | 294 MB | 위와 동일 |

`weights/Market1501_clipreid_12x12sie_ViT-B-16_60.pth` (CLIP-ReID) 는 과거 실험용이며
현재 파이프라인은 사용하지 않습니다.

**첫 실행 시 자동 다운로드** (Hugging Face Hub / Roboflow):

| 모델 ID | 용도 | 캐시 위치 |
|---|---|---|
| `google/siglip2-base-patch16-naflex` | SigLIP2 (768-d) | `%USERPROFILE%\.cache\huggingface` (`HF_HOME` 으로 변경 가능) |
| `facebook/dinov2-with-registers-giant` | DINOv2 (1536-d, 약 4.5 GB) | 위와 동일 |
| `Helsinki-NLP/opus-mt-ko-en` | 한→영 질의 번역 | 위와 동일 |
| `Qwen/Qwen3-VL-2B-Instruct`, `Qwen/Qwen3-VL-Reranker-2B` | Qwen 재순위·검증 (선택) | 위와 동일 |
| RF-DETR Medium | 검출기 | `%USERPROFILE%\.roboflow\models` (`RF_HOME` 으로 변경 가능) |

오프라인 환경이라면 미리 한 번 실행해서 캐시를 채운 뒤 복사하세요. 게이트된
모델이면 `huggingface-cli login` 이 필요할 수 있습니다.

### 2.5 Qdrant 벡터 DB

Qdrant 는 Docker 컨테이너로 띄우고, 데이터는 프로젝트의 `storage/` 폴더에 바인드 마운트합니다.
(`docker-compose.yml` 은 초기 설계의 Milvus + PostgreSQL 스택으로, **현재 파이프라인은
사용하지 않습니다.**)

```powershell
docker run -d --name qdrant_server --restart unless-stopped `
  -p 6333:6333 -p 6334:6334 `
  -v "${PWD}\storage:/qdrant/storage" `
  qdrant/qdrant:v1.19.0
```

확인:

```powershell
curl http://localhost:6333
```

`{"title":"qdrant - vector search engine","version":"1.19.0", ...}` 가 나오면 정상입니다.
웹 대시보드는 `http://localhost:6333/dashboard`. 접속 주소는 `pipeline.yaml` 의
`qdrant.url` (기본 `http://localhost:6333`) 입니다.

컬렉션(`forensic_person`, `forensic_object`)은 `build_db.py` 가 `pipeline.yaml` 의
`collection_prefix` 와 각 retriever 의 `dim` 으로 자동 생성합니다.

### 2.6 데이터 폴더

스크립트 기본값이 가리키는 위치입니다. 필요한 것만 만들면 됩니다.

```text
data/
  videos/                  영상 파이프라인 입력 (mp4 등)
  crops/                   RF-DETR_batch.py 출력: crop jpg, filter_stats.json, checkpoint/
  embedding_checkpoint/    build_db.py 체크포인트 (자동 생성)
  build_manifests/         embedding_build_id 별 manifest (자동 생성)
  PRW/                     PRW 데이터셋 (평가용: frames/, annotations/, query_box/, frame_test.mat)
outputs/                   processed_videos/, clustering/, image_db_html/, image_review/ 등 결과물
storage/                   Qdrant 데이터 (Docker 바인드 마운트)
```

---

## 3. 설정 파일

| 파일 | 역할 |
|---|---|
| `pipeline.yaml` | **단일 진실 공급원.** retriever(모델·차원·가중치 경로), `collection_prefix`, Qdrant 주소·양자화·HNSW, 검출기/추적기/스티처, 질의 번역 설정. DB 구축과 검색이 같은 파일을 읽어 "색인할 때와 다른 모델로 질의" 하는 사고를 막습니다. |
| `gui_pipelines.json` | GUI 파이프라인 탭의 단계 정의 (실행 스크립트, 인자, 기본값). Python 코드 수정 없이 단계를 추가·교체합니다. |
| `pipeline_image.yaml` | `pipeline.yaml` 의 사본 (BOM 만 다름). `--config` 로 골라 쓸 수 있습니다. |

주의:

- `pipeline.yaml` 의 내용 해시가 임베딩 체크포인트와 manifest 에 기록됩니다. 모델·차원을
  바꾸면 **새 `collection_prefix` + 새 `--checkpoint-dir` 로 전체 재구축**이 필요합니다.
  기존 컬렉션과 차원이 다르면 시작 시점에 `siglip2.dim expected=... actual=...` 오류로 멈춥니다.
- `gui_pipelines.json` 은 UTF-8 이며 GUI 가 그대로 읽습니다. JSON 이 깨지면 창은 뜨지만
  파이프라인 탭이 사라집니다 (아래 설치 확인의 로더 명령으로 점검).
- 콘솔이 cp949 라 한글이 깨져 보이면 `$env:PYTHONIOENCODING='utf-8'` 을 먼저 실행하세요.

---

## 4. 설치 확인

```powershell
# CUDA 인식
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 핵심 패키지 import (1~2분 걸릴 수 있음)
.\.venv\Scripts\python.exe -c "import PySide6, rfdetr, boxmot, leidenalg, igraph, torch_geometric, qdrant_client, transformers; print('ok')"

# Qdrant
curl http://localhost:6333

# GUI 파이프라인 정의 로드 (창 없이)
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -c "from gui import pipeline_page as pp; print([(g['id'], len(g['stages'])) for g in pp.load_registry()])"
# 기대 출력: [('video_pipeline', 6), ('image_pipeline', 12), ('evaluation', 14)]

# 단위 테스트 (Qdrant·네트워크 불필요, 약 20초)
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

---

## 5. 실행 — GUI

```powershell
.\.venv\Scripts\python.exe search_gui.py
.\.venv\Scripts\python.exe search_gui.py --config pipeline_image.yaml   # 다른 yaml 로 검색
```

창 제목은 `Forensic Visual Retrieval · Pipeline & Search` 입니다. 상단 탭:

| 탭 | 내용 |
|---|---|
| 이미지 검색 | 1 Crop 기반 검색 / 2 자연어 검색 / 3 Qwen 검증 — 이미지 DB(`media_type=image`) 대상 |
| 영상 검색 | 같은 3개 하위 탭 — 영상 track 그룹(`media_type=video`) 대상, 결과에서 영상 재생 |
| 영상 파이프라인 | 1 전처리+트래킹 → 2 Qdrant 적재 → 3 Person Leiden → 4 Object Leiden → 5·6 군집 갤러리 |
| 화면 구조 (2026-09-28) | Immich 식 셸 `gui/shell.py`: 왼쪽 사이드바 **검색**(사진에서 찾기 · 영상에서 찾기) / **자료 만들기**(영상 처리 · 사진 처리) / **결과**(결과 보기: 라벨링 시트 상태·리포트 HTML 열기, `gui/reports_page.py`) / **평가**(평가·비교 · 벤치마크). 검색 페이지 = 상단 카드(자연어·사진 모드, 대상, 큰 검색창, AI 재확인, 고급 설정) + 썸네일 격자 + 오른쪽 상세('이 결과로 다시 찾기'). 핵심 단계에는 **도구** 드롭다운(클러스터: Leiden / DBSCAN v6 / 커스텀 yaml; 영상 클러스터·갤러리: 사람 / 물건) |
| 이미지 파이프라인 | 핵심 4단계만 먼저 보임: **1 검출**(사진 → crop) → **2 임베딩**(crop → Qdrant) → **3 클러스터**(Leiden) → **4 결과창**(`report/build_image_results.py`: DB 리포트 + 갤러리 + 인덱스 한 번에). '추가 작업 보기' 를 켜면 개별 단계(DB HTML 리포트 · 클러스터링 플러그인 · person/object 갤러리 · 결과 인덱스 · 색상 라벨 · Qwen 라벨 · 폴더 내보내기)가 나온다. 핵심 단계 폼은 basic 필드만 보이고 '고급 옵션 보기' 로 나머지를 편다; 빈 칸 대신 기본값·폴더/값 드롭다운·플레이스홀더 |
| 평가 / 비교 | 1 PRW Person Re-ID 평가 → 2 retriever 조합 비교 |

동작 특성:

- GUI 를 띄우는 것만으로는 모델을 로드하지 않습니다. 첫 검색 버튼을 누를 때 worker 스레드에서
  임베더 4개를 로드하므로 **첫 검색은 1분 가까이** 걸립니다 (이후는 빠름).
- 파이프라인 단계는 각각 `sys.executable -u <script>` 서브프로세스로 실행됩니다. 로그가
  실시간으로 표시되고, 중단 버튼은 CTRL_BREAK → 3초 후 `taskkill /T /F` 순으로 종료합니다.
- `--recreate`, `--fresh` 같은 **파괴적 인자는 JSON 에 적혀 있어도 GUI 가 걸러냅니다.**
  기존 컬렉션·체크포인트를 지우려면 CLI 에서 직접 실행하세요.
- 단계가 `RESULT_HTML:` 마커를 출력하면 "결과 열기" 버튼이 활성화됩니다.
- 시작 시 stderr 에 `QFont::setPointSize`, `qt.multimedia.ffmpeg` 경고가 찍히는 것은 정상입니다.

---

### 5.1 벤치마크 탭 (P5)

원장(`bench/ledger.jsonl`)의 모든 평가 실행을 한 표로 비교합니다. 단계·이름·최근만·통과만 필터, 숫자 정렬, 행 색 = 채택 기준(`bench/criteria.py`, 기준표 운영값) 통과 여부. 행을 고르면 상세 JSON · 그래프(목적 vs 제약 산점도, 검출 PR 곡선) · **verify 실행**(재현 검증) · **재현 명령**(클립보드) · **채택 → yaml**(검출기 `pipeline_tracking_<이름>.yaml`, 클러스터 `clusterer_<이름>.yaml`, 검색 조합 `pipeline_<이름>.yaml` + `.search.json` — 드롭다운에 자동 등장) · 결과 폴더 열기. 새 실행은 평가 탭 5(러너)·9(탐색)·10(조합)에서 돌리고 새로고침합니다.

---

## 6. 실행 — CLI

모든 명령은 프로젝트 루트에서 `.\.venv\Scripts\python.exe` 로 실행합니다. 각 스크립트는
`--help` 로 전체 인자를 보여 줍니다 (torch import 때문에 `--help` 도 수십 초 걸릴 수 있음).

### 6.1 이미지 파이프라인 (사진 → DB)

```powershell
# 1. 배치 검출 + crop  (checkpoint 로 중단·재개 가능, --all 은 --limit 무시하고 전부 처리)
#    검출기는 --detector-config 의 detector: 블록 (없으면 --config 의 RF-DETR). YOLO26: --detector-config pipeline_tracking_yolo26.yaml
.\.venv\Scripts\python.exe detect\RF-DETR_batch.py --sample-dir data\coco\val2017 --output-dir data\crops `
    --dataset-id coco_val2017 --source COCO --all --config pipeline.yaml `
    --min-person-width 25 --min-person-height 120 --min-object-width 20 --min-object-height 20

# 2. 임베딩 + Qdrant 구축  (checkpoint 로 재개, 컬렉션 자동 생성)
.\.venv\Scripts\python.exe ingest\build_db.py --stats data\crops\filter_stats.json --config pipeline.yaml `
    --batch-size 512 --checkpoint-dir data\embedding_checkpoint

# 3. DB HTML 리포트
.\.venv\Scripts\python.exe report\build_image_db_html.py --config pipeline.yaml --source COCO `
    --out outputs\image_db_html\image_db_COCO.html

# 4. Leiden 클러스터링 (crop 단위, person=solider / object=dinov2)
#    --dry-run 을 빼면 cluster_leiden_id 가 Qdrant payload 에 실제로 기록됩니다.
.\.venv\Scripts\python.exe clustering\cluster_leiden_qdrant.py --config pipeline.yaml --target both --sources COCO `
    --knn 30 --mutual-knn --query-batch-size 256 --output-dir outputs\clustering\leiden_image_coco --dry-run

# 4b. 클러스터링 알고리즘 교체 (같은 벡터, 플러그인만 교체: 내장 leiden / dbscan_v6, 또는 clusterer: yaml / --module --class)
#     출력 형식이 4단계와 같아 5·6단계·GT 평가에 그대로 쓴다. payload 는 --write-payload 를 줄 때만 기록
.\.venv\Scripts\python.exe clustering\cluster_qdrant.py --config pipeline.yaml --target person --sources prw_image `
    --method dbscan_v6 --param eps=0.12 --vector-cache outputs\clustering\cache\plugin --output-dir outputs\clustering\plugin_image_prw

# 5·6. 클러스터 갤러리 HTML (person / object)
.\.venv\Scripts\python.exe report\build_leiden_gallery.py --config pipeline.yaml --target person `
    --assignments outputs\clustering\leiden_image_coco\person\person_leiden_assignments.jsonl `
    --output-dir outputs\clustering\leiden_image_coco\person\gallery --inline-images

# 7. 결과 인덱스 HTML (Qdrant 불필요, 3~6단계 산출물만 읽음)
.\.venv\Scripts\python.exe report\build_image_review_index.py --db-html outputs\image_db_html\image_db_COCO.html `
    --cluster-dir outputs\clustering\leiden_image_coco --out outputs\image_review\index_COCO.html
```

`--dataset-id` 는 `image_id → detection_id → Qdrant point ID` 를 결정하므로 한 데이터셋에
하나로 고정하세요. `--source` 는 payload 의 출처 라벨(예: `COCO`, `prw_image`)이며
3·4단계의 `--source(s)` 필터와 같은 값을 써야 합니다.

### 6.2 영상 파이프라인 (영상 → track → DB)

```powershell
# 1. 전처리 + 트래킹 (RF-DETR + BoT-SORT + SUSHI). 영상별 worker 디렉터리에서 격리 실행
#    검출기는 --tracking-config 의 detector: 블록으로 고른다 (pipeline.yaml = RF-DETR, pipeline_tracking_yolo26.yaml = YOLO26)
.\.venv\Scripts\python.exe video\batch_preprocess_videos_parallel.py --videos-root data\videos `
    --processed-root outputs\processed_videos --work-root outputs\parallel_work `
    --tracking-config pipeline.yaml --sushi-root third_party\SUSHI --workers 2 --resume

# 2. track 대표 crop 임베딩 → Qdrant 적재 (point ID 가 결정적이라 재실행해도 중복 없음)
.\.venv\Scripts\python.exe ingest\batch_ingest_all_videos.py --video-dir data\videos `
    --person-per-track 5 --object-per-track 5 --batch-size 16

# 3. Person track centroid Leiden (SOLIDER)
.\.venv\Scripts\python.exe clustering\cluster_leiden_track_centroid.py --collection forensic_person --vector solider `
    --knn 15 --threshold 0.88 --output-dir outputs\clustering\leiden_track_centroid --dry-run

# 4. Object track centroid Leiden (DINOv2)
.\.venv\Scripts\python.exe clustering\cluster_leiden_object_track_centroid.py --collection forensic_object --vector dinov2 `
    --knn 15 --threshold 0.97 --output-dir outputs\clustering\leiden_object_track_centroid_097 --dry-run

# 5·6. 군집 갤러리
.\.venv\Scripts\python.exe report\build_leiden_gallery_track.py --collection forensic_person `
    --assignments outputs\clustering\leiden_track_centroid\person\person_leiden_assignments.jsonl `
    --output-dir outputs\clustering\leiden_track_centroid\person\gallery_track
```

GPU 메모리가 부족하면 `--workers 1` 로 낮추세요.

### 6.3 검색

`search/unified_search_4mode.py` 가 GUI 가 쓰는 통합 검색 CLI 입니다. 하위 명령 뒤에 `--scope person|object` 를 붙입니다.

```powershell
# crop 이미지 → 이미지 DB   (person: SigLIP2+IRRA → SOLIDER rerank, object: SigLIP2+DINOv2)
.\.venv\Scripts\python.exe search\unified_search_4mode.py crop --scope person --image query_crop.jpg --limit 20

# 자연어 → 이미지 DB   (한국어는 pipeline.yaml 의 translation 설정대로 자동 번역)
.\.venv\Scripts\python.exe search\unified_search_4mode.py text --scope person --text "검은 옷에 배낭을 멘 남성" --limit 20

# 이미지 → 영상 track 그룹
.\.venv\Scripts\python.exe search\unified_search_4mode.py image-video --scope person --image query_crop.jpg --vector solider --top-k 20

# 자연어 → 영상 track 그룹
.\.venv\Scripts\python.exe search\unified_search_4mode.py text-video --scope object --text "red car" --top-k 20

# 검색 모델 조합 바꾸기 (GUI 검색 탭의 드롭다운과 동일; 후보는 pipeline.yaml 의 retrievers)
#   crop: --stage1 <1차 retriever…> [--rerank <retriever>|none]   text / text-video: --vectors <supports_text retriever…>
.\.venv\Scripts\python.exe search\unified_search_4mode.py crop --scope person --image query_crop.jpg --stage1 irra solider --rerank none
.\.venv\Scripts\python.exe search\unified_search_4mode.py text-video --scope person --text "검은 가방" --vectors irra
```

결과 JSON 의 `pipeline` / `models` 필드에 사용한 조합이 기록된다 (예 `siglip2+irra -> solider_rerank`).

용도별 개별 스크립트:

| 스크립트 | 입력 | 설명 |
|---|---|---|
| `search/image_search.py -i photo.jpg --top-k 20 --json-out r.json` | 원본 사진 1장 | RF-DETR 로 crop 을 뽑은 뒤 crop 마다 검색. `--stage1-names`, `--rerank-name` 으로 retriever 조합 실험 |
| `search/search.py --text "..." --scope person --rerank --verify` | 자연어·이미지 | 검색 + Qwen3-VL 재순위 + 검증을 한 프로세스에서 |
| `verifiers/qwen_stage.py --in r.json --out r_qwen.json --top-k 20` | 검색 결과 JSON | Qwen 재순위·검증만 별도 실행 (GPU 메모리 분리) |

### 6.4 평가

```powershell
# PRW GT crop 기반 임베딩 단독 Re-ID 평가 (Rank-1/5/10, mAP)
.\.venv\Scripts\python.exe eval\prw_eval.py --model irra --data-root .\data\PRW
.\.venv\Scripts\python.exe eval\prw_eval.py --model solider --data-root .\data\PRW

# 세 모델을 한 번에 돌려 표로 합침
.\eval\run_embedding_eval.ps1

# retriever 조합 비교 (SigLIP2 단독 / IRRA 단독 / RRF / RRF+SOLIDER)
.\.venv\Scripts\python.exe search\image_search.py -i query.jpg --stage1-names siglip2 irra --stage1-k 200 `
    --rerank-name solider --top-k 20 --json-out compare_result.json

# 검출기 평가 (PRW GT 사람 박스, AP@0.5 / AP@[.5:.95] / 동작점 P·R / 높이별 재현율 / FPS)
#   검출기는 pipeline_tracking*.yaml 의 detector: module/class 플러그인. --detector-config 로 고르거나 --module/--class 로 교체
#   1) 검출 실행 (test 6,112 프레임 전체, 채점은 나중에) — YOLO26m 약 4분, RF-DETR-M 약 8분
.\.venv\Scripts\python.exe eval\detect_eval_prw.py --mode run --name yolo26m_test --detector-config pipeline_tracking_yolo26.yaml --split test --no-score
.\.venv\Scripts\python.exe eval\detect_eval_prw.py --mode run --name rfdetr_medium_test --detector-config pipeline_tracking.yaml --split test --no-score
#   2) 운영 DB 에 실제로 들어간 PRW 검출(임계값 0.5, crop 최소 높이 75 필터 반영) 도 같은 형식으로
.\.venv\Scripts\python.exe eval\detect_eval_prw.py --mode import-qdrant --name prod_db_test --split test --no-score
#   3) 한 표로 채점 (운영 임계값은 검출기 yaml 값; --operating-threshold 0.5 로 공통 동작점 비교)
.\.venv\Scripts\python.exe eval\detect_eval_prw.py --mode score `
    --method yolo26m_test=eval\results\detect_prw\yolo26m_test\detections.jsonl `
             rfdetr_medium_test=eval\results\detect_prw\rfdetr_medium_test\detections.jsonl `
             prod_db_test=eval\results\detect_prw\prod_db_test\detections.jsonl

# 실행 원장 — 위 평가 스크립트 4개(검출·임베딩·통합 검색·클러스터링 GT)는 결과를 bench\ledger.jsonl 에도 자동으로 남긴다 (--no-ledger 로 끔)
.\.venv\Scripts\python.exe bench\ledger.py table --stage detect --latest     # 단계별 지표 표 (embed / search / cluster / e2e)
.\.venv\Scripts\python.exe bench\ledger.py list --stage cluster --last 10
.\.venv\Scripts\python.exe bench\ledger.py import                            # eval\results 의 기존 산출물 이관 (멱등)

# 단일 러너 — 한 명령으로 한 단계 평가 → bench\runs\<단계_시각_이름>\ + 원장 (GUI 평가 5단계)
.\.venv\Scripts\python.exe bench\run.py detect  --config pipeline_tracking_yolo26.yaml --name yolo26m_test
.\.venv\Scripts\python.exe bench\run.py cluster --method leiden --param threshold=0.97 --max-points 3000
.\.venv\Scripts\python.exe bench\run.py e2e     --stage1 siglip2 irra --rerank solider         # 전체 파이프라인 검색 (운영 DB, GUI 평가 7단계)
.\.venv\Scripts\python.exe bench\run.py verify  <run_id>                                        # 재현성 검증 PASS/FAIL (GUI 평가 6단계)

# 계약·호환 검사 — 새 모델 yaml 등록 전 규격 확인, 운영 DB 임베딩 지문 대조 (GUI 평가 8단계)
.\.venv\Scripts\python.exe bench\check.py --instantiate --db

# 하이퍼파라미터 탐색 — tune 분할로 찾고 holdout 으로 검증, 추천 yaml 생성 (GUI 평가 9단계). >= 제약은 따옴표
.\.venv\Scripts\python.exe bench\optimize.py search --trials 40 --validate
.\.venv\Scripts\python.exe bench\optimize.py cluster --method leiden --trials 30 --validate
.\.venv\Scripts\python.exe bench\optimize.py detect --detections eval\results\detect_prw\yolo26m_test\detections.jsonl --constraint "recall>=0.85"

# 조합 탐색 — 임베더×재정렬×후보 수 / 벡터×클러스터러 격자 → 리더보드(운영 조합 순위) → 상위 미세조정 → 채택 yaml (GUI 평가 10단계)
.\.venv\Scripts\python.exe bench\combos.py search --validate --refine-trials 20 --top 3 [--adopt]    # → bench\combos\...\pipeline_best.yaml
.\.venv\Scripts\python.exe bench\combos.py cluster --validate --top 2

# 새 모델 등록 — 어댑터 1개(또는 기존 어댑터 + 다른 가중치) → 계약 검사 → yaml → 단계 벤치 → 원장 순위 (GUI 평가 11단계, bench\REGISTER_GUIDE.md)
.\.venv\Scripts\python.exe bench\register.py detector --name yolo26s --module detect.detectors.yolo26_detector --class YOLO26Detector --param weights=yolo26s.pt --limit 300
.\.venv\Scripts\python.exe bench\register.py template embedder --name myemb --dim 768        # 어댑터 스켈레톤 생성

# 사람 정답 3종 (P6, GUI 평가 12~14단계; 라벨 가이드 eval\gt\README.md) — sheet 로 시트를 만들고 브라우저에서 라벨 → labels.json → eval
.\.venv\Scripts\python.exe eval\track_gt_eval.py sheet --videos Normal_Videos_048_x264 Normal_Videos_289_x264 Normal_Videos_100_x264   # 추적 준정답 시트
.\.venv\Scripts\python.exe eval\track_gt_eval.py eval --name botsort_sushi                    # IDF1/HOTA/IDSW/과병합 (raw·before·after) → 원장 stage=track
.\.venv\Scripts\python.exe eval\object_pair_eval.py sheet                                     # 객체 트랙 쌍 시트 (Qdrant)
.\.venv\Scripts\python.exe eval\object_pair_eval.py eval --vector dinov2                      # 객체 임베더 mAP·쌍 AUC·클러스터 일치 → stage=object
.\.venv\Scripts\python.exe eval\qwen_verify_eval.py sheet                                     # 자연어 30 쿼리 × 20 후보 판정 시트
.\.venv\Scripts\python.exe eval\qwen_verify_eval.py eval --max-queries 5                      # Qwen 후처리 P@K 전·후·오탈락률 (캐시, --rescore) → stage=qwen
.\.venv\Scripts\python.exe eval\qwen_verify_eval.py eval --batch-size 10 --allow-unlabeled     # 관찰 배치(2B: 22 s → 2~4 s/후보; 판정 일부 달라져 이름 _b10·캐시 계약 별도)
.\.venv\Scripts\python.exe eval\qwen_compare_runs.py --a eval\results\qwen_verify\qwen_flag_norerank\qwen --b eval\results\qwen_verify\qwen_flag_norerank_b10\qwen   # 두 실행의 판정 일치율·뒤집힘·순위 상관 (라벨 불필요)
.\.venv\Scripts\python.exe bench\run.py track --tracking-config pipeline_tracking_sushi_link.yaml --restitch   # 같은 검출·추적 출력 위에서 스티처(창 경계 연결)만 바꿔 비교
.\.venv\Scripts\python.exe bench\register.py embedder --name myemb --module ... --class ... --dim 768 --ingest-frames 300   # 임베더 등록 + PRW 표본을 bench_myemb_* 컬렉션에 적재 + e2e 비교
```

기준표(`outputs/audit/eval_criteria_20260926.md`)의 현재값은 원장에서 다시 뽑힌다. 러너·원장·검사·분할의 스키마와 사용법은 `bench/README.md`.

---

## 7. 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`tests/` 의 12개 모듈, 187개 테스트가 약 15초에 끝납니다 (2026-09-26 확인). Qdrant 나
네트워크는 필요 없습니다. 프로젝트 루트 밖에서 실행하려면 `PYTHONPATH` 에 루트를 넣으세요.

---

## 8. 문제 해결

| 증상 | 원인 / 조치 |
|---|---|
| 콘솔 한글이 `??` 또는 깨진 문자로 보임 | 콘솔 인코딩이 cp949. `$env:PYTHONIOENCODING='utf-8'` 후 실행 |
| GUI 는 뜨는데 파이프라인 탭이 없음, 상태바에 "gui_pipelines.json 을 읽지 못해..." | JSON 문법 오류. 4장의 로더 명령으로 확인 |
| `Connection refused` / Qdrant 관련 timeout | 컨테이너가 꺼짐. `docker ps` 확인 후 `docker start qdrant_server` |
| `siglip2.dim expected=1152 actual=768` 류 스키마 오류 | `pipeline.yaml` 의 모델이 기존 컬렉션과 다름. yaml 을 되돌리거나 새 `collection_prefix` 로 재구축 |
| `build_db.py` 가 resume 을 거부 | 체크포인트의 stats/yaml 해시가 바뀜. 메시지대로 새 `--checkpoint-dir` 지정 |
| CUDA out of memory | 영상 전처리 `--workers 1`, `build_db.py --batch-size` 축소. Qwen 은 `qwen_stage.py` 로 별도 프로세스에서 |
| 첫 검색·첫 단계가 매우 느림 | 임베더 로드(약 50초) + Hugging Face 첫 다운로드. 두 번째부터는 캐시 사용 |
| `--help` 만 쳐도 오래 걸림 | torch/transformers import 비용. 정상 |
| 경로가 260자를 넘어 파일 저장 실패 | 프로젝트를 짧은 경로(예: `C:\work\TransReID`)에 두거나 Windows 긴 경로 지원 활성화 |
| GUI 에서 `--recreate` 가 안 먹힘 | 의도된 차단. CLI 에서 실행 (기존 컬렉션이 삭제되므로 주의) |

---

## 9. 프로젝트 구조 (주요 파일)

파이프라인 단계별로 폴더를 나눴습니다 (2026-09-26 재배치). 루트에는 공용 커널 모듈과 GUI 진입점만 남습니다.

```text
search_gui.py                 PySide6 GUI 진입점
gui_pipelines.json            GUI 파이프라인 단계 정의 (SSOT)
pipeline.yaml                 모델·컬렉션·Qdrant 설정 (SSOT); pipeline_tracking.yaml 영상 detector/tracker/stitcher 선택
config.py                     pipeline.yaml 로더 + 검증
registry.py / router.py       retriever 지연 로드 (module/class 로 임베더 교체), 질의 임베딩 라우팅
qdrant_store.py               Qdrant 컬렉션 생성·스키마 검증·검색
report_common.py              공용 헬퍼 (yaml 에서 컬렉션·벡터·임계값 로드, 리포트 sidecar)

gui/                          pipeline_page.py (gui_pipelines.json → 파이프라인 탭, 서브프로세스 실행·중단), gui_theme.py
detect/                       검출 단계
  RF-DETR_batch.py              이미지 배치 검출 + crop (checkpoint/resume)
  detect_rf.py / rfdetr_adapter.py   RF-DETR 호출, detection_id·crop 파일명 생성
  detectors/ trackers/ stitchers/    영상용 플러그인 (RF-DETR / BoT-SORT·Deep-OC-SORT / SUSHI), loader.py 가 yaml 의 module/class 로 로드
embedders/                    siglip2_embedder.py, human/{irra,solider}_embedder.py, object/dinov2_embedder_g14_reg_pad_final.py
video/                        영상 전처리: batch_preprocess_videos_parallel.py (검출+추적+스티칭 오케스트레이션),
                              auto_track_validator.py, finalize_track_routes.py, sushi_adapter.py / sushi_inference.py
ingest/                       Qdrant 적재: build_db.py (이미지 crop 임베딩 → Qdrant, checkpoint/manifest),
                              batch_ingest_all_videos.py → build_final_db_candidates_canonical.py + ingest_final_candidates_qdrant.py (영상 track)
clustering/                   base.py (BaseClusterer 계약 + 내장 목록) · methods/{leiden,dbscan_v6}.py (플러그인) · cluster_qdrant.py (플러그인 driver: --method / yaml 로 알고리즘 교체),
                              cluster_leiden_qdrant.py (crop 단위 Leiden, 옵션 전체), cluster_leiden_track_centroid.py / cluster_leiden_object_track_centroid.py (track 단위),
                              cluster_dbscan_qdrant.py (DBSCAN v6 이식), compare_cluster_results.py, label_clusters_{from_vectors,qwen}.py, export_cluster_folders.py
search/                       unified_search_4mode.py (통합 검색 CLI: crop / text / image-video / text-video), search.py (검색 엔진),
                              image_search.py (사진 1장 → crop → 검색), query_translate.py (한→영), duplicate_grouping.py (Hybrid-C 중복 collapse)
verifiers/                    qwen_stage.py / qwen_crop_stage.py (Qwen 재순위·검증), search_text_video.py
report/                       build_image_db_html.py, build_leiden_gallery.py, build_leiden_gallery_track.py, build_image_review_index.py, inline_gallery_html.py
eval/                         prw_eval.py, prw_eval_unified.py, prw_cluster_gt_eval.py, detect_eval_prw.py, prw_e2e_search_eval.py, cluster_ablation_prw.py, run_*.ps1
                              track_gt_eval.py / object_pair_eval.py / qwen_verify_eval.py (P6 사람 정답 시트 + 평가), gt_sheet.py (시트 공용), gt/ (정답·시트·라벨 가이드)
bench/                        자동 벤치마크 도구: ledger.py (실행 원장 — 평가 4종이 자동 기록, table/list/import) + ledger.jsonl (원장, git 추적); README.md
tests/                        unittest 스위트
third_party/SOLIDER, third_party/SUSHI, IRRA/     서드파티 모델 코드
weights/                      수동 배치 가중치 (git 제외)
data/ · outputs/ · storage/   입력 데이터 · 산출물 · Qdrant 데이터 (git 제외)
```

하위 폴더의 실행 스크립트는 시작 시 프로젝트 루트를 `sys.path` 에 넣으므로 어느 cwd 에서든
`python <폴더>\<스크립트>.py` 로 실행됩니다. 공용 모듈은 `from search.search import SearchEngine`,
`from clustering import cluster_leiden_qdrant` 처럼 패키지 경로로 import 합니다.

루트의 `*_BACKUP_*.py`, `*.bak*` 파일과 `minimal_forensic_fix/`, `code_bundle/`,
`ForensicShare/`, `codex_out/` 는 백업·리뷰 산출물이며 실행에 필요하지 않습니다.

---

## 10. 운영 원칙

- DB 구축과 검색은 **같은 `pipeline.yaml`** 을 읽습니다. named vector 를 교차 사용하지 않습니다 (SOLIDER DB ↔ SOLIDER 질의).
- Leiden 클러스터링은 retrieval 이후 결과 확장 단계입니다. `--dry-run` 없이 실행해야 payload 에 기록됩니다.
- 객체 검색의 중복은 DB 에서 지우지 않고 검색 단계에서 collapse 합니다 (`duplicate_grouping`).
- Qwen 재순위·검증은 기본 검색과 분리된 선택 단계입니다. threshold 미달 후보는 지우지 않고 표시만 합니다 (`verify_mode=flag`).
- 파괴적 DB 옵션은 GUI 에서 차단합니다.

---

## 11. 저장소 상태 주의

2026-09-20 기준 git 은 25,770개 파일을 추적하지만, 실행에 필요한 다음 항목이 **미추적**입니다.
새 환경에 배포하려면 폴더째 복사하거나 아래를 커밋하세요.

- GUI: `gui_pipelines.json`, `gui/`, `assets/`
- 이미지 파이프라인: `detect/RF-DETR_batch.py`, `report/`, `clustering/cluster_leiden_qdrant.py`, `report_common.py`, `embedders/object/dinov2_embedder_g14_reg_pad_final.py`
- 영상 파이프라인: `detect/` 플러그인, `video/`, `ingest/batch_ingest_all_videos.py`, `ingest/build_final_db_candidates_canonical.py`, `ingest/ingest_final_candidates_qdrant.py`, `clustering/cluster_leiden_track_centroid.py`, `clustering/cluster_leiden_object_track_centroid.py`, `report/build_leiden_gallery_track.py`, `third_party/SUSHI/`
- 평가·테스트: `eval/` 대부분, `tests/`
- 그 밖에 `search/duplicate_grouping.py`, `clustering/compare_cluster_results.py`, `clustering/cluster_dbscan_qdrant.py` 등

또한 추적 중이던 `build_db.py`, `search.py`, `detect_rf.py`, `rfdetr_adapter.py`, `image_search.py`,
`query_translate.py`, `qwen_stage.py`, `qwen_crop_stage.py`, `unified_search_4mode.py` 는 2026-09-26 재배치로
하위 폴더로 옮겨졌습니다 (git 에는 삭제 + 신규로 보이며 커밋 시 rename 으로 인식됩니다). `config.py`,
`qdrant_store.py`, `search_gui.py`, `pipeline.yaml` 등도 커밋본과 작업본이 크게 다릅니다.
**작업본을 `git checkout` 으로 되돌리지 마세요.** 커밋된 `build_db.py` 는 현재와 전혀 다른
영상 시절 스크립트입니다.
