# 전통 증강 코드 분석: prepare_experiments.py

## 파일 개요

Albumentations 기반 전통적 이미지 증강 스크립트. GenAI 증강 데이터 샘플링과 전통 증강을 결합하여 실험별 데이터셋을 자동 생성함.

**원본 경로 (ahnbi1):**
`/data2/project/2026winter/jjh0709/AA_CV_R/prepare_experiments.py`

---

## 사용 라이브러리

| 라이브러리 | 용도 |
|-----------|------|
| `albumentations` | 이미지 증강 파이프라인 |
| `PIL` (Pillow) | 이미지 로드/저장 |
| `numpy` | 배열 연산 |
| `json`, `shutil`, `pathlib` | 파일 처리 |
| `random` | 랜덤 샘플링 (seed=42 고정) |
| `tqdm` | 진행바 출력 |

---

## 사용한 증강 기법 목록

### Geometric Transformations (어노테이션 자동 변환 포함)

| 증강 기법 | 파라미터 | 확률 |
|----------|---------|------|
| `HorizontalFlip` | - | p=0.5 |
| `VerticalFlip` | - | p=0.3 |
| `Rotate` | limit=±15° | p=0.5 |
| `ShiftScaleRotate` | shift=0.1, scale=0.1, rotate=±15° | p=0.5 |

### Pixel-level Transformations (어노테이션 불변)

| 증강 기법 | 파라미터 | 확률 |
|----------|---------|------|
| `RandomBrightnessContrast` | brightness=0.2, contrast=0.2 | p=0.5 |
| `HueSaturationValue` | hue=10, sat=20, val=10 | p=0.3 |
| `GaussNoise` | var_limit=(10.0, 50.0) | p=0.3 |
| `GaussianBlur` | blur_limit=(3, 5) | p=0.3 |

### Albumentations 파이프라인 설정

```python
A.Compose([
    # Geometric
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    # Pixel-level
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)
```

---

## 마스크 변환 처리 여부

**마스크(segmentation) 변환 지원**: Geometric 변환 시 bbox와 segmentation polygon을 함께 변환.

```python
def augment_image_with_annotations(img_path, annotations, aug_pipeline):
    # segmentation → keypoints 형식으로 변환 후 albumentations에 전달
    keypoints = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]

    transformed = aug_pipeline(
        image=image,
        bboxes=bboxes,
        category_ids=category_ids,
        keypoints=all_keypoints
    )

    # 변환된 keypoints → segmentation polygon으로 복원
    segmentation = []
    for kp in kps:
        segmentation.extend([float(kp[0]), float(kp[1])])
    ann['segmentation'] = [segmentation]

    # area 재계산 (bbox 기반)
    x, y, w, h = bbox
    ann['area'] = w * h
```

**주의**: area는 bbox 면적으로 근사 계산 (정확한 polygon area 아님).

---

## 입출력 구조

### 입력

```
AA_CV_R/
├── train/
│   ├── images/           ← 원본 + GenAI 증강 혼합 (232장)
│   │   ├── original_001.jpg     (원본 26장: 파일명으로 구분)
│   │   ├── Cable_000001.jpg     (Cable GenAI 105장)
│   │   └── thunderbolt_000001.jpg (Thunderbolt GenAI 101장)
│   └── annotations.json
└── val/
    ├── images/
    └── annotations.json
```

### 이미지 타입 분류 방식

```python
def classify_image_type(filename):
    if filename.startswith('Cable_'):       return 'Cable_GenAI'
    elif filename.lower().startswith('thunderbolt_'): return 'Thunderbolt_GenAI'
    else:                                   return 'Original'
```

### 출력

```
experiments/
├── exp_1_original26_genai50/        ← 원본 26 + GenAI 50
│   ├── images/
│   └── annotations.json
├── exp_1_original26_genai100/
├── exp_2_original26_only/           ← Baseline (원본 26장만)
├── exp_2_original26_traditional50/  ← 원본 → 전통 증강으로 총 50장
└── exp_2_original26_genai50_traditional/
```

### 실험 구성

| 실험 유형 | 데이터 구성 |
|----------|-----------|
| exp_1 (GenAI 양 변화) | 원본 26 + GenAI {50/100/150/200}장 |
| exp_2 Baseline | 원본 26장만 |
| exp_2 전통 증강 | 원본 26장 → 전통 증강으로 총 {50/100/150/200}장 |
| exp_2 혼합 | 원본 26 + GenAI N + 원본 2배 전통 증강 |

---

## 현재 문제점

1. **Cable/Thunderbolt만 지원**: 이미지 타입 분류 로직이 `Cable_`, `thunderbolt_` 파일명 prefix 기반
2. **Screw, Casting 미지원**: 다른 카테고리는 전통 증강 불가
3. **입력 경로 하드코딩**: `BASE_DIR`이 ahnbi1 절대 경로 고정
4. **단일 annotations.json**: 전체 데이터(원본+GenAI)가 하나의 JSON에 혼합
5. **이미지 복사 방식**: 심볼릭 링크 미사용으로 디스크 공간 낭비
6. **area 계산 근사값**: polygon area 대신 bbox 면적 사용
7. **카테고리별 실험 데이터셋 분리 없음**: Cable에만 적용 가능한 구조

---

## 리팩토링 시 수정 필요 포인트

리팩토링 목표 파일: `scripts/augmentation/traditional_augment.py`

1. **argparse 추가**: `--category (Cable/Screw/Casting)`, `--num_images`, `--source (original/gen_ai/both)`
2. **다중 카테고리 지원**: 카테고리별로 이미지 경로 및 annotations.json 분리
3. **입력 경로 동적 처리**: `data/{Category}/train/` 및 `data_augmented/{Category}/gen_ai/` 기준
4. **출력 경로 통일**: `data_augmented/{Category}/traditional_aug/images/` + `annotations.json`
5. **심볼릭 링크 사용**: 이미지 복사 대신 링크로 디스크 절약
6. **segmentation area 정확한 계산**: `cv2.contourArea()` 또는 shoelace formula 사용
7. **random seed 인수화**: `--seed` 파라미터로 재현성 보장
