# 실험 1: Hyperparameter 통일 설정

실험 1(생성AI 증강 수에 따른 성능 변화)에서 Mask R-CNN, Cascade Mask R-CNN 두 모델에 **동일 적용**하는 하이퍼파라미터 설정 근거와 계획.

---

## 1. 실험 목적

GenAI로 생성한 결함 이미지를 클래스당 0/25/50/75/100/125장씩 추가하며, instance segmentation 성능(segm/AP)이 어떻게 변하는지 확인한다.

---

## 2. 모델

| 모델 | 프레임워크 | ROI Head | 담당 |
|------|-----------|----------|------|
| Mask R-CNN | detectron2 | StandardROIHeads (1-stage) | 양진우 |
| Cascade Mask R-CNN | detectron2 | CascadeROIHeads (3-stage) | 임대윤 |

두 모델은 backbone(ResNet-50 FPN)이 동일하고 ROI Head 구조만 다르다.
**공정한 비교를 위해 하이퍼파라미터를 통일**한다. HP가 다르면 성능 차이가 모델 구조 때문인지 HP 때문인지 구분할 수 없다.

---

## 3. Learning Rate & Batch Size

### 3-1. 기본 배경

Detectron2 기본 config (Base-RCNN-FPN.yaml):
- `BASE_LR`: 0.02 (SGD, momentum=0.9)
- `IMS_PER_BATCH`: 16 (8 GPU x 2 images/GPU)
- COCO 데이터셋(11만장)으로 **from scratch** 학습 기준

### 3-2. 우리 상황

- **COCO pretrained 가중치로 fine-tuning** (from scratch 아님)
- **데이터**: 6개 카테고리 통합(Unified, 14클래스), 클래스당 원본 20장 제한 (`N_ORIGINAL_TRAIN_PER_CLASS=20`)
  - baseline: **280장** (Cable 20 + Screw 20 + Casting 40 + Console 80 + Cylinder 80 + Wood 40)
  - genai_125: 280 + GenAI 1,744 = **2,024장** (실측)
- **GPU**: NVIDIA A100 80GB x 1 (서버 정책: 1인 1GPU)

### 3-3. Batch Size 선택: 12

| Batch Size | GPU 메모리 사용 (추정) | A100 80GB 활용률 |
|:----------:|:--------------------:|:----------------:|
| 2 | ~4 GB | 5% |
| 4 | ~5 GB | 6% |
| 8 | ~10 GB | 13% |
| **12** | **~15 GB** | **19%** |
| 16 | ~20 GB | 25% |

- batch_size=2는 A100 80GB에서 심각한 자원 낭비
- batch_size=12는 GPU 활용률을 개선하면서도 소량 데이터에서 gradient 안정성 확보
- 참고: A100 80GB 기준 batch_size=8~16이 일반적인 권장 범위

### 3-4. Learning Rate: Linear Scaling Rule

> **참고 논문**: Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017)
> https://arxiv.org/abs/1706.02677

Batch size를 k배 줄이면 lr도 k배 줄여야 한다:

```
원본 (from scratch): lr=0.02, batch_size=16
batch_size=12 → lr = 0.02 × (12/16) = 0.015
```

Fine-tuning은 pretrained 가중치를 보호해야 하므로 **1/10**로 추가 보정:

```
batch_size=12 → 0.015 / 10 = 0.0015
```

### 3-5. 최종 선택

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| `batch_size` | **12** | A100 GPU 활용률 개선 + gradient 안정 |
| `lr` | **0.0015** | Linear Scaling Rule + fine-tuning 보정 |

### 3-6. 참고 자료

- Linear Scaling Rule: Goyal et al., 2017 — https://arxiv.org/abs/1706.02677
- Detectron2 LR Scaling 논의: https://github.com/facebookresearch/detectron2/issues/934
- Detectron2 기본 config: `Base-RCNN-FPN.yaml` (BASE_LR=0.02, IMS_PER_BATCH=16)
- Cascade R-CNN 원본: Cai & Vasconcelos, CVPR 2018 — https://arxiv.org/abs/1712.00726

---

## 4. Max Epochs

### 선택: `max_epochs=1000`

- Early stopping이 자동 종료하므로 충분히 크게 설정
- 실험 1은 조건별 Unified 데이터 양:
  - baseline: 280장
  - genai_25: 280 + ~349 = ~629장
  - genai_125: 280 + ~1,744 = ~2,024장
- 데이터가 적을수록 수렴이 빠르고, 많을수록 느릴 수 있음
- 1000으로 넉넉히 잡고 early stopping에 맡기는 것이 안전
- 실제로 1000까지 돌 일은 거의 없음 (early stopping이 먼저 중단)

---

## 5. Warmup Epochs

### 선택: `warmup_epochs=5`

- 학습 초반에 lr을 0에서 목표값(0.0015)까지 서서히 올림
- 초반 큰 lr로 인한 가중치 발산 방지
- detectron2 기본(1,000 iter) 대비 적절한 범위

| 조건 (batch_size=12 기준) | 1 epoch | warmup 5 epochs |
|--------------------------|:-------:|:---------------:|
| Unified baseline (280장) | ~24 iter | ~120 iter |
| Unified genai_125 (~2,024장) | ~169 iter | ~845 iter |

- 조건별로 iteration 수는 다르지만, **모든 조건에 동일 적용**
- 논문 관례: 조건 간 HP를 다르게 하면 공정 비교 불가

---

## 6. Eval Period

### 선택: `eval_period_epochs=5`

- 5 에폭마다 val 데이터로 segm/AP 측정
- 너무 자주 → 학습 시간 증가 (평가에도 시간 소요)
- 너무 드물게 → 최적 중단 시점을 놓칠 수 있음
- 5 에폭은 일반적인 설정

---

## 7. Early Stopping

### 선택: `patience=15`

- 15번 평가 × 5 에폭 = **75 에폭** 동안 segm/AP 개선 없으면 자동 중단
- patience=10 (50 에폭)도 가능하나, 소량 데이터에서 학습 곡선이 불규칙할 수 있으므로 보수적으로 15 채택
- 모니터링 지표: `segm/AP` (instance segmentation mAP)

---

## 8. Checkpoint Period

### 선택: `checkpoint_period_epochs=50`, `max_periodic_checkpoints=1`

- 50 에폭마다 모델 가중치 저장
- 체크포인트 1개 ≈ 548MB (Cascade Mask R-CNN), 335MB (Mask R-CNN)

**학습 중 — Rotation**: `max_periodic_checkpoints=1`
- detectron2: `PeriodicCheckpointer(max_to_keep=1)`
- mmdet: `CheckpointHook(max_keep_ckpts=3)` (best 보호 버퍼)

**학습 + 평가 후 — Cleanup**: `cleanup_artifacts()` 자동 호출
- 삭제: `model_final.pth`, 주기 체크포인트, `last_checkpoint`
- 보존: `model_best.pth`, eval_results, config, metrics, tensorboard
- 안전장치: `model_best.pth`가 존재할 때만 동작

**디스크 사용량**:

| 시점 | Mask R-CNN | Cascade Mask R-CNN |
|---|---:|---:|
| 학습 중 (피크) | ~1.0 GB | ~1.6 GB |
| 평가 후 cleanup | **~0.34 GB** | **~0.55 GB** |

- 36 runs 총: ~12~20 GB (이전 47 GB → 60% 절감)
- best model은 early stopping이 `model_best.pth`로 저장, cleanup 후에도 유지

---

## 9. Input Size

### 9-1. 원본 이미지 해상도

| 카테고리 | 해상도 |
|---------|--------|
| Cable | 632x406 ~ 1920x1146 |
| Screw | 1080x1440 |
| Casting | 2456x1176 |
| Console | 1920x1280 ~ 3840x2748 |
| Cylinder | 1590x1192 ~ 1600x1200 |

대부분 1000px 이상의 고해상도. Casting, Console은 2000~3800px.

### 9-2. 선택

| 파라미터 | 값 |
|---------|-----|
| `input_min_size` | (640, 672, 704, 736, 768, 800) |
| `input_max_size` | 1333 |

**이유:**
1. detectron2 기본값이므로 COCO pretrained 가중치와 일관성 유지
2. max=800(이전 설정)은 원본 대비 과도한 축소 → 작은 결함이 소실될 위험
3. max=1333이면 결함의 픽셀 정보가 더 많이 보존
4. A100 80GB + batch_size=12면 메모리 여유 충분

---

## 10. LR Decay Schedule

### 선택: Step LR, (70%, 90%) 지점

```
0 ~ 70% 구간: lr = 0.0015
70% 지점:      lr = 0.00015  (1/10)
90% 지점:      lr = 0.000015 (1/100)
```

- detectron2 기본 step decay 방식 사용
- 학습 후반에 lr을 낮춰서 fine-grained 수렴 유도
- early stopping으로 실제 학습 에폭이 정해지므로, 비율 기반(70%/90%)이 절대값보다 적절

---

## 11. 반복 실험 (3-seed)

### 선택: seed = 42, 43, 44

각 (카테고리, 조건, 모델) 조합을 **3회 반복** 실행한다.

seed가 영향을 주는 요소:
- **데이터 샘플링** (시드별로 다른 20장이 클래스당 선택됨 — true independent replication)
- 모델 가중치 초기화 (마지막 레이어)
- 데이터 셔플 순서
- 학습 augmentation 랜덤성

→ `merged_datasets`도 **시드별로 분리**되어 (`seed{N}/` 하위 폴더) 저장됨. 링크(하드링크 → 심볼릭링크 fallback)로 원본 공유하므로 디스크 추가 비용은 거의 없음. 실측: exp1 6 conditions × 3 seeds = **~137 MB**.

3회 반복의 결과를 **평균 ± 표준편차**로 보고한다.

```
예시:
  seed=42: segm_AP = 45.1
  seed=43: segm_AP = 43.8
  seed=44: segm_AP = 44.5
  → 보고: 44.5 ± 0.65
```

표준편차가 크면 해당 조건에서 학습이 불안정하다는 의미.

---

## 12. 최종 설정 요약

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| `lr` | 0.0015 | Linear Scaling Rule + fine-tuning 보정 |
| `batch_size` | 12 | A100 80GB GPU 활용률 개선 |
| `max_epochs` | 1000 | early stopping에 맡기고 넉넉히 |
| `seed` | 42, 43, 44 | 3회 반복 |
| `warmup_epochs` | 5 | 조건 간 동일 적용 |
| `eval_period_epochs` | 5 | 5 에폭마다 평가 |
| `checkpoint_period_epochs` | 50 | 50 에폭마다 주기 저장 |
| `max_periodic_checkpoints` | 1 | 주기 체크포인트 rotation (디스크 절약) |
| `early_stopping_patience` | 15 | 75 에폭 동안 개선 없으면 중단 |
| `early_stopping_metric` | segm/AP | instance segmentation mAP |
| `input_min_size` | (640, 672, 704, 736, 768, 800) | detectron2 기본값 |
| `input_max_size` | 1333 | 고해상도 유지, 작은 결함 보존 |
| `lr_decay_steps` | (70%, 90%) | Step LR decay |
| `n_original_per_class` | 20 | 클래스당 원본 20장 (`N_ORIGINAL_TRAIN_PER_CLASS`) |
| `cleanup_artifacts` | 자동 | 평가 성공 후 model_final + 주기 체크포인트 자동 삭제. model_best만 보존 |
| `merged_datasets` | seed별 분리 | 시드마다 다른 20장 샘플링. 링크(하드→심볼릭)로 디스크 공유 |
| `save_results` | 매 condition마다 | incremental save로 도중 끊김에도 결과 보존 |

모든 실험 조건(baseline ~ genai_125), 모든 모델(Mask R-CNN, Cascade Mask R-CNN)에 **동일 적용**.

---

## 13. 카테고리 & 데이터 조건

### 13-1. 카테고리 (6개, 14클래스 통합)

원본 데이터셋 크기 vs 학습에 사용되는 양 (`N_ORIGINAL_TRAIN_PER_CLASS=20`):

| 카테고리 | 클래스 (개수) | 원본 train (전체) | 학습 사용 (20/cls) | GenAI 보유 |
|----------|---------------|:---------------:|:------------------:|:----------:|
| Cable | thunderbolt (1) | 26장 | **20장** | 126장 |
| Screw | defect (1) | 57장 | **20장** | 256장 |
| Casting | Inclusoes, Rechupe (2) | 54장 | **40장** (2×20) | ~245장 |
| Console | Collision, Dirty, Gap, Scratch (4) | 95장 | **80장** (4×20) | ~499장 |
| Cylinder | Chip, PistonMiss, Porosity, RCS (4) | 138장 | **80장** (4×20) | ~500장 |
| Wood | impurities, pits (2) | 51장 | **40장** (2×20) | ~250장 |
| **합계** | **14 클래스** | **421장** | **280장** | **~1,744장** |

6개 카테고리는 `--category Unified`로 14개 결함 클래스를 한 모델로 통합 학습한다.
`_sample_images_per_class()`로 클래스별 균형 샘플링 — 클래스의 가용 이미지가 20장 미만이면 자동으로 전체 사용.

### 13-2. 데이터 조건 (6개)

| 조건 | 원본 (클래스당) | GenAI (클래스당) | Unified 합계 |
|------|:----:|:---------------:|:------------:|
| baseline | 20장 | 0장 | 280 |
| genai_25 | 20장 | 25장 | ~629 |
| genai_50 | 20장 | 50장 | ~978 |
| genai_75 | 20장 | 75장 | ~1,326 |
| genai_100 | 20장 | 100장 | ~1,675 |
| genai_125 | 20장 | 125장 | ~2,024 |

일부 클래스는 GenAI가 125장 미만. 해당 클래스는 보유량 전부 사용.

---

## 14. 총 학습 횟수

```
1 (Unified, 14클래스 통합) × 6 조건 × 2 모델 × 3 반복 = 36회
```

| 담당 | 모델 | 학습 횟수 |
|------|------|:--------:|
| 양진우 | Mask R-CNN | 18회 |
| 임대윤 | Cascade Mask R-CNN | 18회 |

---

## 15. 평가 지표

3가지 레벨의 지표를 사용한다.

### 15-1. Instance-level (객체 단위) — "이 결함 객체를 찾았나?"

COCO evaluator가 자동 출력하는 표준 지표.

| 지표 | 의미 | IoU 기준 |
|------|------|:--------:|
| **mAP** | Precision-Recall 곡선 면적 평균 (종합 정밀도) | 0.50:0.95 (10단계) |
| **mAP50** | 예측이 정답과 50% 이상 겹치면 정답으로 간주 | 0.50 |
| **mAP75** | 75% 이상 겹쳐야 정답 (마스크 정밀도) | 0.75 |
| **mAR** | 결함을 빠뜨리지 않는 정도 (종합 재현율) | 0.50:0.95 |
| **F1** | mAP와 mAR의 조화평균 | - |

```
mAP  = "찾은 것 중에 맞은 비율" (Precision 기반)
mAR  = "정답 중에 찾은 비율" (Recall 기반)
F1   = 2 × mAP × mAR / (mAP + mAR)
```

### 15-2. Pixel-level (픽셀 단위) — "마스크 경계가 정확한가?"

별도 계산 필요 (evaluate.py에서 구현).

| 지표 | 의미 | 계산 |
|------|------|------|
| **Dice** | 픽셀 단위 F1. 마스크 겹침 정도 | 2×\|예측∩정답\| / (\|예측\|+\|정답\|) |
| **Pixel IoU** | 픽셀 단위 겹침 비율 | \|예측∩정답\| / \|예측∪정답\| |
| **Pixel Precision** | 예측 마스크 중 정답인 비율 | TP_px / (TP_px + FP_px) |
| **Pixel Recall** | 정답 마스크 중 찾은 비율 | TP_px / (TP_px + FN_px) |

Dice와 Pixel IoU의 차이:
```
정답: 1000 픽셀, 예측: 900 픽셀, 겹침: 800 픽셀

Dice     = 2 × 800 / (1000 + 900) = 0.842
Pixel IoU = 800 / (1000 + 900 - 800) = 0.727
```
Dice는 항상 Pixel IoU보다 높다. 같은 마스크여도 측정 방식에 따라 값이 다르다.

### 15-3. Recall-focused (결함 탐지 특화) — "결함을 놓치지 않는가?"

결함 탐지에서는 **놓치는 것(False Negative)이 오탐(False Positive)보다 심각**하다.
놓치면 불량품이 출하되지만, 오탐은 재검사하면 된다.

| 지표 | 의미 | 계산 |
|------|------|------|
| **Miss Rate** | 놓친 결함 비율 (낮을수록 좋음) | 1 - Recall |
| **Recall@FP=1** | 이미지당 오탐 1개 허용 시 재현율 | FROC 곡선에서 추출 |

```
예: val 이미지 100장, 정답 결함 150개

모델이 120개를 찾음 (30개 놓침), 20개 오탐
  Recall = 120/150 = 0.80
  Miss Rate = 1 - 0.80 = 0.20  → 결함의 20%를 놓침

Recall@FP=1:
  이미지당 오탐을 1개만 허용하는 threshold에서의 recall
  → 보수적인 환경에서 얼마나 결함을 찾는지
```

### 15-4. IoU(Intersection over Union) 기준

```
IoU = (예측 마스크 ∩ 정답 마스크) / (예측 마스크 ∪ 정답 마스크)
```

| IoU 기준 | 의미 |
|:--------:|------|
| 0.50 | 절반만 겹쳐도 정답 → "대략 맞추는지" |
| 0.75 | 75% 겹쳐야 정답 → "정밀하게 맞추는지" |
| 0.50:0.95 | 0.50~0.95 0.05 간격 10단계 평균 → "종합 성능" |

### 15-5. 전체 지표 역할 정리

| 역할 | 지표 | 레벨 | 용도 |
|------|------|:----:|------|
| **최종 판단** | segm/mAP | Instance | 종합 성능. "어떤 조건이 가장 좋다"를 결론 |
| **핵심 보고** | segm/mAP50 | Instance | 탐지 능력. 대략적으로 결함을 찾는지 |
| **핵심 보고** | segm/F1 | Instance | Precision × Recall 균형 |
| **핵심 보고** | Dice | Pixel | 마스크 품질. GenAI 마스크가 정확한지 |
| **핵심 보고** | Miss Rate | Recall | 결함을 얼마나 놓치는지 (낮을수록 좋음) |
| 원인 분석 | bbox/mAP | Instance | segm이 낮을 때, 객체 탐지 자체가 문제인지 확인 |
| 원인 분석 | segm/mAP75 | Instance | 마스크 경계 정밀도 |
| 원인 분석 | Pixel Precision/Recall | Pixel | Dice가 낮을 때 원인 파악 |
| 원인 분석 | Recall@FP=1 | Recall | 보수적 환경에서의 탐지 능력 |

### 15-6. 결과 테이블 예시

```
카테고리: Cable / 모델: Mask R-CNN / 3회 평균 ± 표준편차

| 조건     | mAP         | mAP50       | F1          | Dice        | Miss Rate    |
|----------|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
| baseline | 32.1 ± 1.2  | 58.3 ± 2.1  | 60.3 ± 1.9  | 0.72 ± 0.03 | 0.38 ± 0.02  |
| +25      | 35.4 ± 0.9  | 63.1 ± 1.5  | 64.6 ± 1.4  | 0.76 ± 0.02 | 0.34 ± 0.01  |
| +50      | 37.8 ± 1.1  | 66.7 ± 1.8  | 67.9 ± 1.7  | 0.79 ± 0.02 | 0.30 ± 0.02  |
| ...      |             |             |             |             |              |
```

### 15-7. 계산 방법

COCO evaluator 자동 출력 지표:

| 우리 표기 | detectron2 key |
|----------|----------------|
| mAP | segm/AP |
| mAP50 | segm/AP50 |
| mAP75 | segm/AP75 |
| mAR | segm/AR |

별도 계산이 필요한 지표:

```python
# F1: mAP와 mAR로 계산
f1 = 2 * mAP * mAR / (mAP + mAR)

# Dice: 매칭된 (예측, 정답) 마스크 쌍마다 계산 후 평균
dice = 2 * (pred_mask & gt_mask).sum() / (pred_mask.sum() + gt_mask.sum())

# Miss Rate: COCO eval 결과에서 추출
miss_rate = 1 - recall

# Recall@FP=1: score threshold를 조절하며 계산
# 이미지당 평균 FP = 1이 되는 threshold에서의 recall
```

> 별도 계산 지표들은 `training/evaluate.py`에서 구현한다.

### 15-8. 상황별 해석 가이드

| 상황 | 해석 |
|------|------|
| mAP↑ F1↑ Dice↑ Miss Rate↓ | 증강이 전방위적으로 효과적 |
| mAP50↑ mAP75 변화없음 | 대충 찾지만 마스크 정밀도 정체 → GenAI 마스크 품질 문제 |
| mAP↑ Dice↓ | 객체는 더 찾지만 마스크 경계 부정확 → 어노테이션 품질 확인 |
| Miss Rate↓ mAP 변화없음 | 놓치는 결함 감소했지만 오탐도 증가 → F1 정체 |
| bbox/mAP↑ segm/mAP↓ | 탐지는 되지만 마스크가 부정확 → Dice로 확인 |
| 특정 조건부터 mAP 정체/하락 | 증강 데이터 양의 포화점(saturation point) 도달 |
| 3회 표준편차 큼 (>3.0) | 데이터 양 부족으로 학습 불안정 |
| Recall@FP=1 낮음 | 보수적 환경에서 사용 어려움 → threshold 튜닝 필요 |

---

## 16. 실행 환경

- GPU: NVIDIA A100 80GB × 1 (서버 정책: 1인 1GPU)
- tmux 세션에서 실행 (연결 끊겨도 유지)
- 에러 시 해당 조합만 재실행

---

## 17. 결과 정리 방식

- 3회 평균 ± 표준편차 표
- GenAI 0/25/50/75/100/125 조건별 추세 그래프 (x축: GenAI 장수, y축: segm/AP)
- Mask R-CNN vs Cascade Mask R-CNN 비교
- 카테고리별 비교
- 성능 정체/하락 구간 분석
