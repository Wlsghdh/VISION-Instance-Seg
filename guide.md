# 사용 가이드 (guide.md)

> 서버: lifeai | 작업 디렉토리: `/home/jjh0709/gitrepo/VISION-Instance-Seg`

---

## 목차

1. [디렉토리 구조](#1-디렉토리-구조)
2. [라벨링 서버 (labeling_server)](#2-라벨링-서버)
3. [전통 증강 (traditional_augment.py)](#3-전통-증강)
4. [데이터 배치 방법 (신규 카테고리)](#4-신규-카테고리-데이터-배치)
5. [카테고리 ID 규칙](#5-카테고리-id-규칙)
6. [Git 관리 규칙](#6-git-관리)

---

## 1. 디렉토리 구조

```
VISION-Instance-Seg/
│
├── data/                          # 원본 데이터 (.gitignore)
│   ├── Cable/
│   │   ├── train/
│   │   │   ├── images/            # 26장 (thunderbolt only)
│   │   │   └── annotations.json
│   │   ├── val/                   # 수정 금지 (test 용도)
│   │   └── inference/             # 수정 금지
│   ├── Screw/
│   │   └── train/
│   │       ├── images/            # 57장 (defect)
│   │       └── annotations.json
│   └── Casting/
│       └── train/
│           ├── images/            # 54장 (Inclusoes + Rechupe)
│           └── annotations.json
│
├── data_augmented/                # 증강 데이터 (.gitignore)
│   └── Cable/
│       ├── gen_ai/
│       │   ├── images/            # 105장 (Cable_XXXXXX.jpg)
│       │   └── annotations.json
│       └── traditional_aug/
│           ├── images/            # 2750장 (XXXXXX_augYYYY.jpg)
│           └── annotations.json
│
├── labeling_server/
│   ├── app.py                     # Flask 어노테이션 툴 v9
│   └── templates/
│       └── annotation_template.html
│
├── scripts/
│   └── augmentation/
│       ├── traditional_augment.py # 전통 증강 스크립트
│       └── gemini_augment.py      # (미작업)
│
├── training/                      # 학습 스크립트 (미작업)
├── progress.md                    # 진행 상황
└── guide.md                       # 이 파일
```

---

## 2. 라벨링 서버

### 실행 방법

```bash
# 작업 디렉토리에서 실행
cd /home/jjh0709/gitrepo/VISION-Instance-Seg

# Cable gen_ai 라벨링
python labeling_server/app.py --category Cable --split gen_ai --port 5200

# Screw gen_ai 라벨링
python labeling_server/app.py --category Screw --split gen_ai --port 5201

# Casting gen_ai 라벨링
python labeling_server/app.py --category Casting --split gen_ai --port 5202
```

브라우저 접속: `http://서버IP:5200`

### 인자 설명

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--category` | Cable | 대상 카테고리 (Cable / Screw / Casting) |
| `--split` | gen_ai | 데이터 split 이름 (gen_ai / traditional_aug / 커스텀) |
| `--port` | 5200 | Flask 서버 포트 |
| `--host` | 0.0.0.0 | 서버 호스트 |

### 저장 경로

```
--category Cable --split gen_ai
  이미지:        data_augmented/Cable/gen_ai/images/        (서버 이미지는 이동 없음)
  annotations:   data_augmented/Cable/gen_ai/annotations.json

--category Screw --split gen_ai
  이미지:        data_augmented/Screw/gen_ai/images/
  annotations:   data_augmented/Screw/gen_ai/annotations.json
```

### 사용 흐름

#### A. 서버에 있는 기존 gen_ai 이미지 라벨링/검수

```
1. 서버 실행
   python labeling_server/app.py --category Cable --split gen_ai --port 5200

2. 브라우저 접속

3. 왼쪽 사이드바 "0. 서버 이미지 선택" → [📂 서버 이미지 목록] 클릭

4. 파일 목록에서 라벨링할 이미지 선택
   - 초록 "라벨됨": 기존 annotation 자동 로드됨
   - 회색 "미라벨": 새로 라벨링 필요

5. 클래스 선택 → BBox 그리기 → Brush로 세그멘테이션

6. [✅ 현재 결함 완성] → [💾 현재 이미지 저장]
   → data_augmented/Cable/gen_ai/annotations.json 자동 갱신
```

#### B. 새 이미지 업로드 + 라벨링

```
1. 서버 실행

2. "1. 직접 업로드" → 파일 선택

3. 클래스 선택 → BBox → 세그멘테이션

4. [💾 현재 이미지 저장]
   → 이미지: data_augmented/{category}/{split}/images/에 저장
   → annotation: annotations.json에 추가
```

### 저장 형식 (COCO)

```json
{
  "images": [
    {"id": 0, "file_name": "Cable_000000.jpg", "width": 640, "height": 480}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 0,
      "category_id": 1,
      "bbox": [100, 150, 200, 180],
      "segmentation": [[100,150, 300,150, 300,330, 100,330]],
      "area": 36000,
      "iscrowd": 0
    }
  ],
  "categories": [{"id": 1, "name": "thunderbolt", "supercategory": "thunderbolt"}]
}
```

### API 엔드포인트 요약

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 메인 페이지 |
| GET | `/info` | 서버 상태 (이미지/annotation 수) |
| GET | `/images/list` | 이미지 디렉토리 파일 목록 |
| GET | `/images/serve/<filename>` | 이미지 파일 서빙 |
| GET | `/annotations/for/<filename>` | 특정 이미지의 기존 annotation 조회 |
| POST | `/save` | 새 이미지 업로드 + annotation 저장 |
| POST | `/save/existing` | 기존 이미지 annotation 갱신 |
| POST | `/delete` | 이미지 + annotation 삭제 |
| GET | `/stats` | 카테고리별 상세 통계 |

---

## 3. 전통 증강

### 실행 방법

```bash
cd /home/jjh0709/gitrepo/VISION-Instance-Seg

# Cable: 2750장 생성 (이미 완료)
python scripts/augmentation/traditional_augment.py \
    --category Cable \
    --n_augment 2750 \
    --seed 42

# Screw: 2750장
python scripts/augmentation/traditional_augment.py \
    --category Screw \
    --n_augment 2750 \
    --seed 42

# Casting: 2750장
python scripts/augmentation/traditional_augment.py \
    --category Casting \
    --n_augment 2750 \
    --seed 42
```

### 인자 설명

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--category` | (필수) | Cable / Screw / Casting |
| `--n_augment` | 2750 | 생성할 증강 이미지 수 |
| `--seed` | 42 | 랜덤 시드 (재현성) |

### 입출력 경로

```
입력:
  data/{category}/train/images/        ← 원본 이미지
  data/{category}/train/annotations.json

출력:
  data_augmented/{category}/traditional_aug/images/    ← 증강 이미지
  data_augmented/{category}/traditional_aug/annotations.json
```

### 주의사항

- 출력에는 **증강 이미지만** 포함 (원본 없음)
- 실험 시 원본 + 증강을 합치려면 `merge_dataset.py` 사용 (미작성)
- 이미 출력 폴더가 있으면 덮어씌워짐 → 재실행 전 확인
- 파일명 형식: `{원본stem}_aug{N:04d}.jpg` (예: `000000_aug0000.jpg`)

### 증강 기법

| 변환 | 확률 | 파라미터 |
|------|------|----------|
| HorizontalFlip | 50% | — |
| VerticalFlip | 30% | — |
| Rotate | 50% | ±15° |
| ShiftScaleRotate | 50% | shift 10%, scale 10%, rotate 15° |
| RandomBrightnessContrast | 50% | ±20% |
| HueSaturationValue | 30% | hue±10, sat±20, val±10 |
| GaussNoise | 30% | std_range (0.02~0.10) |
| GaussianBlur | 30% | blur 3~5px |

> Segmentation polygon은 마스크 기반으로 자동 변환 (Albumentations 2.x 호환)

---

## 4. 신규 카테고리 데이터 배치

### Screw / Casting gen_ai 데이터가 들어올 때

#### Step 1: gen_ai 이미지 배치

```bash
# screw_transfer/ 패키지가 있다고 가정
mkdir -p data_augmented/Screw/gen_ai/images

# AI 생성 이미지만 복사 (Screw_XXXXXX.jpg 패턴)
cp screw_transfer/Screw_*.jpg data_augmented/Screw/gen_ai/images/
```

#### Step 2: annotations.json 생성 (Python 스크립트)

```python
import json
from pathlib import Path

SRC_JSON = Path('screw_transfer/annotations.json')
DST_DIR  = Path('data_augmented/Screw/gen_ai')

with open(SRC_JSON) as f:
    d = json.load(f)

# Screw_ 이미지만 필터
gen_ai_imgs = [i for i in d['images'] if i['file_name'].startswith('Screw_')]
gen_ai_ids  = {i['id'] for i in gen_ai_imgs}
gen_ai_anns = [a for a in d['annotations'] if a['image_id'] in gen_ai_ids]

new_data = {
    "images": gen_ai_imgs,
    "annotations": gen_ai_anns,
    "categories": [{"id": 0, "name": "defect", "supercategory": "defect"}]
}

with open(DST_DIR / 'annotations.json', 'w') as f:
    json.dump(new_data, f, indent=2)
```

#### Step 3: 라벨링 서버로 검수

```bash
python labeling_server/app.py --category Screw --split gen_ai --port 5201
# 브라우저에서 각 이미지 검수 및 수정
```

#### Step 4: traditional_aug 실행

```bash
python scripts/augmentation/traditional_augment.py \
    --category Screw \
    --n_augment 2750 \
    --seed 42
```

---

## 5. 카테고리 ID 규칙

| 카테고리 | 결함명 | cat_id | 비고 |
|----------|--------|--------|------|
| Cable | thunderbolt | **1** | break(0)는 제외됨 |
| Screw | defect | **0** | 단일 결함 |
| Casting | Inclusoes | **0** | |
| Casting | Rechupe | **1** | |

> `traditional_augment.py`의 `CATEGORY_CONFIG`에 각 카테고리별 `keep_id`가 설정되어 있음

---

## 6. Git 관리

### 트래킹 대상 (커밋)

```
CLAUDE.md, guide.md, progress.md, lifeai_claude.md
labeling_server/app.py
labeling_server/templates/annotation_template.html
scripts/augmentation/traditional_augment.py
scripts/augmentation/gemini_augment.py
.gitignore
```

### 트래킹 제외 (.gitignore)

```
data/               ← 원본 데이터 (서버 직접 배치)
data_augmented/     ← 증강 데이터 (서버 직접 배치)
results/            ← 실험 결과
work_dirs/          ← mmdetection 학습 출력
*.pth, *.pt, *.ckpt ← 모델 가중치
*.backup_*          ← annotation 자동 백업
wandb/, mlruns/     ← 실험 추적
```

### 커밋 예시

```bash
cd /home/jjh0709/gitrepo/VISION-Instance-Seg

git add CLAUDE.md progress.md guide.md lifeai_claude.md
git add labeling_server/app.py
git add labeling_server/templates/annotation_template.html
git add scripts/augmentation/traditional_augment.py
git add .gitignore

git commit -m "[feat] labeling server v9, traditional_augment 리팩토링, Cable 데이터 정리"
git push origin main
```

---

## 빠른 참조

```bash
# 라벨링 서버 시작
python labeling_server/app.py --category Cable --split gen_ai --port 5200

# 전통 증강 실행
python scripts/augmentation/traditional_augment.py --category Cable --n_augment 2750 --seed 42

# 증강 결과 확인
python3 -c "
import json
for p in ['data_augmented/Cable/gen_ai/annotations.json',
          'data_augmented/Cable/traditional_aug/annotations.json']:
    d = json.load(open(p))
    print(p.split('/')[-2], ':', len(d['images']), '장,', len(d['annotations']), '개 annotation')
"
```
