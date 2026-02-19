# lifeai_claude.md
# lifeai 서버에서 Claude Code가 읽는 작업 지시서

## 서버 환경
- **서버**: lifeai
- **작업 디렉토리**: `/home/jjh0709/gitrepo/VISION-Instance-Seg`
- **Python**: 3.9+
- **프레임워크**: mmdetection 기반

---

## 프로젝트 개요

Gemini API로 생성한 불량 이미지 증강(Gen-AI aug)이 instance segmentation 성능에
미치는 영향을 전통적 증강(Traditional aug)과 비교하는 실험 연구.

- **대상 부품**: Cable, Screw, Casting
- **어노테이션 포맷**: COCO format (instance segmentation, polygon)
- **평가지표**: mAP, mAR

---

## 데이터 현황 (배치 전 확인 필요)

### 현재 lifeai 서버 데이터 경로
```
/home/jjh0709/gitrepo/VISION-Instance-Seg/
├── data/                   # .gitignore (직접 배치)
│   └── Cable/
│       ├── Cable/          # ⚠️ 이중 폴더 문제 → 아래 이동 작업 필요
│       │   ├── train/
│       │   ├── val/
│       │   └── inference/
│       └── (비어있어야 함, Cable/Cable/ → Cable/로 올려야 함)
│   ├── Casting/
│   │   ├── train/
│   │   ├── val/
│   │   └── inference/
│   └── ...
└── data_augmented/         # .gitignore (직접 배치)
    └── Cable/
        └── gen_ai/
            └── images/     # ai_cable_000000.png ~ ai_cable_000099.png
```

### 데이터 이동 작업 (최초 1회)
ahnbi1에서 다운받은 cable_transfer 패키지를 lifeai에 배치:

```bash
# 1. Cable 이중 폴더 해결 (Cable/Cable → Cable/)
cd /home/jjh0709/gitrepo/VISION-Instance-Seg/data
mv Cable/Cable/* Cable/
rmdir Cable/Cable

# 2. cable_transfer 패키지 내용 배치
#    (scp로 받은 cable_transfer/ 기준)

# 원본 train/val: cable_transfer/data/Cable/ → data/Cable/
cp -r cable_transfer/data/Cable/train/images/* data/Cable/train/images/
cp cable_transfer/data/Cable/train/annotations.json data/Cable/train/
cp -r cable_transfer/data/Cable/val/images/* data/Cable/val/images/
cp cable_transfer/data/Cable/val/annotations.json data/Cable/val/

# AI 증강 이미지: cable_transfer/data_augmented/ → data_augmented/
mkdir -p data_augmented/Cable/gen_ai/images
cp cable_transfer/data_augmented/Cable/gen_ai/images/* data_augmented/Cable/gen_ai/images/
```

### 최종 목표 데이터 구조
```
data/{Category}/train/{images/, annotations.json}
data/{Category}/val/{images/, annotations.json}
data/{Category}/inference/{images/, annotations.json}  # (선택)

data_augmented/{Category}/gen_ai/{images/, annotations.json}
data_augmented/{Category}/traditional_aug/{images/, annotations.json}
```

---

## 현재 개발 상태 (ahnbi1 분석 완료)

ahnbi1에서 기존 코드를 분석한 결과는 `docs/legacy/` 참조:
- `docs/legacy/annotation_tool_analysis.md` - 라벨링 툴 분석
- `docs/legacy/gemini_augment_analysis.md` - Gemini 증강 스크립트 분석
- `docs/legacy/traditional_aug_analysis.md` - 전통 증강 스크립트 분석

---

## 작업 목록 (우선순위 순)

### Step 1: 라벨링 툴 리팩토링 (`labeling_server/`)

**목적**: AI 생성 이미지(gen_ai) 라벨링 → `data_augmented/Cable/gen_ai/annotations.json` 생성

**참고 원본**: ahnbi1의 `annotation_tool_v8.py`
- Flask 기반 웹 어노테이션 툴
- 하드코딩된 경로를 커맨드라인 인자로 변환 필요
- Cable만 지원 → Cable/Screw/Casting 다중 카테고리 지원
- 저장 경로: `data_augmented/{Category}/gen_ai/`에 저장하도록 수정

**구현할 파일**:
```
labeling_server/
├── app.py              # Flask 앱 (경로/카테고리 인자화)
├── templates/
│   └── annotation.html
└── static/
    └── (JS, CSS)
```

**실행 방법 (목표)**:
```bash
python labeling_server/app.py --category Cable --split gen_ai
# → data_augmented/Cable/gen_ai/annotations.json 에 저장
```

**주요 변경점**:
- `CONFIG` 딕셔너리를 argparse로 대체
- 카테고리별 저장 경로 동적 결정
- Cable: `thunderbolt` / Screw: `defect` / Casting: `Inclusoes`, `Rechupe`

---

### Step 2: Gemini 증강 스크립트 (`scripts/augmentation/gemini_augment.py`)

**목적**: Cable용 Gemini 증강 이미지 추가 생성 (현재 100장 → 최대 250장 필요)

**참고 원본**: ahnbi1의 `generate_defects.py`
- 현재 Casting, Screw만 지원 → Cable 추가
- API 키: 프로젝트 환경변수 또는 config 파일에서 로드하도록 수정
- 진행상황 JSON으로 저장 (재시작 가능)
- 생성 이미지 저장: `data_augmented/Cable/gen_ai/images/ai_cable_NNNNNN.png`
  - 기존 100장이 이미 `ai_cable_000000 ~ ai_cable_000099` → 새로 생성 시 `ai_cable_000100`부터

**Cable 프롬프트 참고**:
- 결함 유형: `thunderbolt` (케이블 커넥터 결함)
- 레퍼런스 이미지: `data/Cable/train/images/` 에서 정상 + 결함 이미지 선택
- 프롬프트 파일: `scripts/augmentation/prompts/cable_prompt.txt`

---

### Step 3: 전통 증강 스크립트 (`scripts/augmentation/traditional_augment.py`)

**목적**: Albumentations 기반 증강으로 대량 데이터 생성

**참고 원본**: ahnbi1의 `prepare_experiments.py`
- 증강 기법: HorizontalFlip, VerticalFlip, Rotate±15°, ShiftScaleRotate,
             RandomBrightnessContrast, HueSaturationValue, GaussNoise, GaussianBlur
- 어노테이션(segmentation polygon) 자동 변환 포함
- 저장: `data_augmented/{Category}/traditional_aug/{images/, annotations.json}`

**실행 방법 (목표)**:
```bash
python scripts/augmentation/traditional_augment.py \
    --category Cable \
    --n_augment 2750 \
    --seed 42
```

---

### Step 4: 데이터 병합 유틸 (`scripts/data_utils/merge_dataset.py`)

**목적**: 실험 조건별로 train 데이터 조합

```bash
# 예시: 원본 25장 + gen_ai 100장
python scripts/data_utils/merge_dataset.py \
    --category Cable \
    --original_n 25 \
    --gen_ai_n 100 \
    --output_dir experiments/cable_exp1/train
```

**구현 사항**:
- 원본 train에서 N장 샘플링 (seed 고정)
- gen_ai에서 N장 샘플링
- traditional_aug에서 N장 샘플링
- COCO annotations.json 병합 (image_id 충돌 방지)
- 출력: `{output_dir}/images/`, `{output_dir}/annotations.json`

---

### Step 5: 학습/평가 스크립트 (`training/`)

**참고 프레임워크**: mmdetection

**구현할 파일**:
```
training/
├── train.py            # mmdetection 학습 래퍼
├── test.py             # 평가 스크립트 (mAP, mAR 출력)
└── run_experiments.sh  # 전체 실험 자동화 셸 스크립트
```

---

## 실험 계획

### 실험 1: Gen-AI 증강 수에 따른 성능 변화

| 조건 | 원본 | Gen-AI | 합계 |
|------|------|--------|------|
| Baseline | 25장 | 0 | 25 |
| +50  | 25장 | 50  | 75  |
| +100 | 25장 | 100 | 125 |
| +150 | 25장 | 150 | 175 |
| +200 | 25장 | 200 | 225 |
| +250 | 25장 | 250 | 275 |

모델: Mask R-CNN, Cascade Mask R-CNN

### 실험 2: 전통 증강 vs Gen-AI 증강 비교

| # | 구성 | 총 데이터 |
|---|------|----------|
| 1 | 원본 25장 | 25 |
| 2 | 원본 25 + 전통 250 | 275 |
| 3 | 원본 25 + Gen-AI 250 | 275 |
| 4 | 원본 25 + Gen-AI 250 + 전통 250 | 525 |
| 5 | 원본 25 + Gen-AI 250 + 전통 2,750 | 3,025 |

### 실험 3: 7종 모델 비교

| 모델 |
|------|
| Mask R-CNN |
| Cascade R-CNN |
| Cascade Mask R-CNN |
| SOLOv2 |
| Mask DINO |
| + 최신 2종 (선택) |

---

## 주의사항

- `data/`, `data_augmented/`, `results/`는 `.gitignore` → Git에 올리지 않음
- **test/inference 데이터는 절대 수정 금지**
- 어노테이션은 **COCO format** (segmentation polygon, `iscrowd=0`)
- 카테고리 ID 통일:
  - Cable → `thunderbolt` (id: 0 또는 1로 통일 필요, 현재 0)
  - Screw → `defect`
  - Casting → `Inclusoes`, `Rechupe`
- 작업 완료 시 `CLAUDE.md`의 `현재 진행 상황` 업데이트

---

## 즉시 시작할 작업

1. `data/Cable/` 이중 폴더 문제 해결 (Cable/Cable → Cable/)
2. `labeling_server/app.py` 리팩토링 → gen_ai 이미지 100장 라벨링
3. 라벨링 완료 후 → `traditional_augment.py` 작성 및 실행
4. `merge_dataset.py` 작성
5. 실험 1 시작 (Mask R-CNN, Baseline부터)
