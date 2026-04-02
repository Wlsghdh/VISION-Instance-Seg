# 학습 가이드

## 환경

- 서버: lifeai, conda env `jjh`, Python 3.11
- GPU: A100 80GB x2, CUDA 12.2
- 프레임워크: detectron2 0.6, mmdet 3.3.0

```bash
cd /home/jjh0709/gitrepo/VISION-Instance-Seg
conda activate jjh
```

---

## 1. 기본 사용법

```bash
python -m training.train \
  --category Cable \
  --experiment exp1 \
  --condition baseline \
  --model mask_rcnn
```

### 필수 인자

| 인자 | 선택지 | 설명 |
|------|--------|------|
| `--category` | 14개 카테고리명 또는 `all` | 대상 카테고리 |
| `--experiment` | `exp1`, `exp2`, `exp3` | 실험 종류 |
| `--condition` | 실험별 조건명 또는 `all` | 데이터 조건 |
| `--model` | 모델명 또는 `all` | 학습 모델 |

### 선택 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--max-epochs` | 300 | 최대 에폭 (early stopping이 자동 종료) |
| `--lr` | 1e-4 | 학습률 |
| `--batch-size` | 2 | 배치 크기 |
| `--seed` | 42 | 랜덤 시드 |
| `--eval-period-epochs` | 5 | N 에폭마다 평가 |
| `--patience` | 15 | Early stopping patience (평가 횟수 기준) |
| `--eval-only` | - | 학습 없이 평가만 실행 |

---

## 2. 카테고리

### 실험 대상 6종 (GenAI 증강 완료)

| 카테고리 | Train | GenAI | 전통 증강 | 결함 클래스 |
|----------|:-----:|:-----:|:---------:|------------|
| Cable | 26장 | 126장 | 2,750장 | thunderbolt |
| Screw | 57장 | 256장 | 250장 | defect |
| Casting | 54장 | 238장 | 250장 | Inclusoes, Rechupe |
| Console | 95장 | 499장 | - | Collision, Dirty, Gap, Scratch |
| Cylinder | 138장 | 511장 | - | Chip, PistonMiss, Porosity, RCS |
| Wood | 51장 | 251장 | - | impurities, pits |

- GenAI는 **클래스당 ~125장** 생성
- `n_genai_per_class`로 클래스별 균형 샘플링

### 전체 14개 카테고리

config에 등록되어 있어 `--category`로 지정 가능하다.

| 카테고리 | Train | 클래스 수 | GenAI |
|----------|:-----:|:---------:|:-----:|
| Cable | 26장 | 1 | O |
| Screw | 57장 | 1 | O |
| Casting | 54장 | 2 | O |
| Console | 95장 | 4 | O |
| Cylinder | 138장 | 4 | O |
| Wood | 51장 | 2 | O |
| Capacitor | 35장 | 1 | - |
| Electronics | 36장 | 1 | - |
| Groove | 50장 | 2 | - |
| Hemisphere | 85장 | 4 | - |
| Lens | 66장 | 5 | - |
| PCB_1 | 47장 | 6 | - |
| PCB_2 | 80장 | 7 | - |
| Ring | 45장 | 3 | - |

---

## 3. 실험 설명

### exp1: 생성AI 증강 수에 따른 성능 변화

GenAI 이미지를 클래스당 0~125장까지 늘려가며 성능 변화를 측정한다.

| 조건 | 원본 | GenAI (클래스당) |
|------|------|-----------------|
| `baseline` | 전체 | 0 |
| `genai_25` | 전체 | 25 |
| `genai_50` | 전체 | 50 |
| `genai_75` | 전체 | 75 |
| `genai_100` | 전체 | 100 |
| `genai_125` | 전체 | 125 |

모델: `mask_rcnn`, `cascade_mask_rcnn`

```bash
# Cable 전체 조건 + 전체 모델
python -m training.train --category Cable --experiment exp1 --condition all --model all

# 특정 조건만
python -m training.train --category Screw --experiment exp1 --condition genai_50 --model mask_rcnn
```

### exp2: 전통 증강 vs 생성AI 증강 비교

| 조건 | 원본 | GenAI (클래스당) | 전통 증강 |
|------|------|-----------------|----------|
| `cond1` | 전체 | 0 | 0 |
| `cond2` | 전체 | 0 | 250 |
| `cond3` | 전체 | 125 | 0 |
| `cond4` | 전체 | 125 | 250 |
| `cond5` | 전체 | 125 | 2750 |

모델: `mask_rcnn`, `cascade_mask_rcnn`, `maskdino`

```bash
python -m training.train --category Casting --experiment exp2 --condition all --model maskdino
```

### exp3: 7종 모델 비교

| 조건 | 구성 |
|------|------|
| `original_only` | 원본만 |
| `with_trad` | 원본 + 전통 3000장 |
| `with_genai_trad` | 원본 + GenAI 125/cls + 전통 2750장 |

모델: 7종 전체 (`mask_rcnn`, `cascade_mask_rcnn`, `maskdino`, `mask2former`, `cascade_rcnn`, `solov2`, `rtmdet_ins`)

```bash
python -m training.train --category Cable --experiment exp3 --condition original_only --model all
```

---

## 4. Early Stopping

모든 학습에 자동 적용된다. 별도 설정 없이 동작한다.

- **모니터링 지표**: `segm/AP` (instance segmentation mAP)
- **patience**: 15회 평가 (기본 5에폭 간격 x 15 = 75에폭 동안 개선 없으면 중단)
- **max_epochs**: 300 (early stopping이 먼저 동작)

```bash
# patience를 20으로 늘리기
python -m training.train --category Cable --experiment exp1 --condition baseline --model mask_rcnn --patience 20

# 최대 에폭 변경
python -m training.train --category Cable --experiment exp1 --condition baseline --model mask_rcnn --max-epochs 500
```

### 학습 결과에 포함되는 항목

| 항목 | 설명 |
|------|------|
| `total_epochs` | 실제 학습된 에폭 수 |
| `early_stopped` | early stopping 발동 여부 |
| `early_stop_epoch` | 학습이 멈춘 에폭 |
| `peak_memory_mb` | GPU 피크 메모리 (MB) |
| `train_time_sec` | 총 학습 시간 (초) |

---

## 5. 데이터 준비만 (학습 없이)

```bash
python -m training.data_pipeline --category Cable --experiment exp1 --condition all
python -m training.data_pipeline --category all --experiment exp2 --condition all --force
```

---

## 6. 평가만 실행

이미 학습된 모델을 평가만 하고 싶을 때:

```bash
python -m training.train --category Cable --experiment exp1 --condition baseline --model mask_rcnn --eval-only
```

---

## 7. 결과 확인

### 결과 저장 위치

```
results/
  training/{experiment}/{condition}/{category}/{model}/   # 체크포인트, config, metrics
  evaluation/results.json                                  # 전체 결과 마스터 파일
  reports/                                                 # 비교 테이블
```

### 리포트 생성

```bash
python -m training.utils.report --experiment exp1 --csv
python -m training.utils.report --experiment exp2 --csv
```

### results.json 예시

```json
{
  "category": "Cable",
  "experiment": "exp1",
  "condition": "genai_50",
  "model": "mask_rcnn",
  "train_time_sec": 1234.5,
  "peak_memory_mb": 12345.6,
  "early_stopped": true,
  "early_stop_epoch": 85.0,
  "total_epochs": 85.0,
  "eval": {
    "segm_AP": 45.123,
    "segm_AP50": 72.456,
    "bbox_AP": 48.789
  }
}
```

---

## 8. 전체 실험 실행 예시

```bash
# exp1: 주요 6개 카테고리 중 Cable
python -m training.train --category Cable --experiment exp1 --condition all --model all

# exp1: 전체 카테고리 (14개)
python -m training.train --category all --experiment exp1 --condition all --model all

# exp2: Casting만, maskdino
python -m training.train --category Casting --experiment exp2 --condition all --model maskdino

# exp3: Console, 7종 모델 비교
python -m training.train --category Console --experiment exp3 --condition all --model all
```

---

## 9. 모델별 참고사항

| 모델 | 프레임워크 | 비고 |
|------|-----------|------|
| mask_rcnn | detectron2 | 기본 모델, 가장 안정적 |
| cascade_mask_rcnn | detectron2 | multi-stage, mask_rcnn보다 정밀 |
| maskdino | detectron2 | MaskDINO repo 필요 (`/home/jjh0709/gitrepo/MaskDINO/`) |
| mask2former | detectron2 | Mask2Former repo 필요 (`/home/jjh0709/gitrepo/Mask2Former/`) |
| cascade_rcnn | mmdet | mmdet Cascade Mask R-CNN |
| solov2 | mmdet | anchor-free, 빠른 추론 |
| rtmdet_ins | mmdet | 실시간 모델 |

---

## 10. 트러블슈팅

**CUDA OOM**: `--batch-size 1`로 줄이기

**detectron2 import 오류**: `conda activate jjh` 확인

**MaskDINO/Mask2Former 오류**: CUDA ops 빌드 확인
```bash
cd /home/jjh0709/gitrepo/MaskDINO/maskdino/modeling/pixel_decoder/ops
python setup.py build_ext --inplace
```

**데이터 병합 재생성**: `--force` 플래그 사용
```bash
python -m training.data_pipeline --category Cable --experiment exp1 --condition all --force
```
