# 학습 Guideline

> 통합 학습 환경 사용 가이드

---

## 0. 사전 조건

```bash
# conda 환경 활성화
conda activate jjh

# 작업 디렉토리 이동
cd /home/jjh0709/gitrepo/VISION-Instance-Seg
```

- Python 3.11, PyTorch 2.5.1+cu121, CUDA 12.2
- detectron2 0.6, mmcv 2.1.0, mmdet 3.3.0 설치 완료
- MaskDINO/Mask2Former repo + CUDA ops 빌드 완료

---

## 1. 기본 명령어 구조

```bash
python -m training.train \
    --category {카테고리} \
    --experiment {실험} \
    --condition {조건} \
    --model {모델} \
    [옵션]
```

| 인자 | 값 | 설명 |
|------|---|------|
| `--category` | Cable, Screw, Casting, all | 대상 카테고리 |
| `--experiment` | exp1, exp2, exp3 | 실험 종류 |
| `--condition` | 실험별 조건명, all | 데이터 조건 |
| `--model` | 모델명, all | 학습 모델 |
| `--eval-only` | (플래그) | 학습 스킵, 평가만 |
| `--max-iter` | 정수 (기본 10000) | 학습 반복 수 |
| `--lr` | 실수 (기본 1e-4) | 학습률 |
| `--batch-size` | 정수 (기본 2) | 배치 크기 |
| `--seed` | 정수 (기본 42) | 랜덤 시드 |
| `--eval-period` | 정수 (기본 500) | 평가 주기 (iteration) |

---

## 2. 사용 가능한 모델 (7종)

| CLI 이름 | 모델명 | 프레임워크 |
|---------|--------|-----------|
| `mask_rcnn` | Mask R-CNN (R50-FPN) | detectron2 |
| `cascade_mask_rcnn` | Cascade Mask R-CNN (R50-FPN) | detectron2 |
| `maskdino` | MaskDINO (R50) | detectron2 + MaskDINO repo |
| `mask2former` | Mask2Former (R50) | detectron2 + Mask2Former repo |
| `cascade_rcnn` | Cascade R-CNN (R50-FPN) | mmdet |
| `solov2` | SOLOv2 (R50-FPN) | mmdet |
| `rtmdet_ins` | RTMDet-Ins (S) | mmdet |

---

## 3. 실험별 조건

### exp1: GenAI 증강 수에 따른 성능 변화

| 조건명 | 원본 | GenAI | 합계 |
|--------|:----:|:-----:|:----:|
| `baseline` | 25 | 0 | 25 |
| `genai_50` | 25 | 50 | 75 |
| `genai_100` | 25 | 100 | 125 |
| `genai_150` | 25 | 150 | 175 |
| `genai_200` | 25 | 200 | 225 |
| `genai_250` | 25 | 250 | 275 |

기본 모델: mask_rcnn, cascade_mask_rcnn

### exp2: 전통 증강 vs GenAI 비교

| 조건명 | 원본 | GenAI | 전통 증강 | 합계 |
|--------|:----:|:-----:|:--------:|:----:|
| `cond1` | 25 | 0 | 0 | 25 |
| `cond2` | 25 | 0 | 250 | 275 |
| `cond3` | 25 | 250 | 0 | 275 |
| `cond4` | 25 | 250 | 250 | 525 |
| `cond5` | 25 | 250 | 2,750 | 3,025 |

기본 모델: mask_rcnn, cascade_mask_rcnn, maskdino

### exp3: 7종 모델 비교

| 조건명 | 구성 |
|--------|------|
| `original_only` | 원본 전체 |
| `with_trad` | 원본 전체 + 전통 증강 3,000 |
| `with_genai_trad` | 원본 전체 + GenAI 250 + 전통 증강 2,750 |

기본 모델: 7종 전체

---

## 4. 실제 사용 예시

### 4-1. E2E 검증 (처음 돌릴 때 — 빠른 테스트)

```bash
# 100 iteration만 돌려서 파이프라인 정상 작동 확인
python -m training.train \
    --category Cable --experiment exp2 --condition cond1 \
    --model mask_rcnn --max-iter 100
```

### 4-2. 단일 모델 학습

```bash
# Screw, exp2, cond3, MaskDINO
python -m training.train \
    --category Screw --experiment exp2 --condition cond3 \
    --model maskdino

# 하이퍼파라미터 커스텀
python -m training.train \
    --category Screw --experiment exp2 --condition cond3 \
    --model maskdino --max-iter 15000 --lr 5e-5 --batch-size 4
```

### 4-3. 한 카테고리의 전체 조건 실행

```bash
# Screw에서 exp2 전체 조건을 MaskDINO로
python -m training.train \
    --category Screw --experiment exp2 --condition all \
    --model maskdino
```

실행 순서: cond1 → cond2 → cond3 → cond4 → cond5 순차 실행

### 4-4. 7종 모델 비교

```bash
# exp3, original_only 조건에서 7모델 비교
python -m training.train \
    --category Screw --experiment exp3 --condition original_only \
    --model all
```

### 4-5. 전체 카테고리 일괄

```bash
# 3개 카테고리 × exp2 cond1 × mask_rcnn
python -m training.train \
    --category all --experiment exp2 --condition cond1 \
    --model mask_rcnn
```

### 4-6. 평가만 (이미 학습된 모델)

```bash
# 특정 모델 평가
python -m training.train \
    --category Cable --experiment exp2 --condition cond1 \
    --model mask_rcnn --eval-only

# 독립 평가 스크립트 (일괄)
python -m training.evaluate \
    --category Screw --experiment exp2 --condition all --model all
```

---

## 5. 데이터 준비만 (학습 없이)

학습 전에 데이터 병합이 잘 되는지 확인할 때 사용.

```bash
# 단일
python -m training.data_pipeline \
    --category Cable --experiment exp2 --condition cond1

# 전체
python -m training.data_pipeline \
    --category all --experiment exp2 --condition all

# 강제 재생성
python -m training.data_pipeline \
    --category Cable --experiment exp2 --condition cond1 --force
```

출력: `results/merged_datasets/{experiment}/{condition}/{category}/`

---

## 6. 결과 확인

### 결과 파일 위치

```
results/
├── merged_datasets/{exp}/{cond}/{cat}/          ← 병합 데이터
│   ├── images/
│   └── annotations.json
├── training/{exp}/{cond}/{cat}/{model}/          ← 학습 결과
│   ├── model_final.pth (detectron2)
│   ├── best_*.pth (mmdet)
│   ├── config.yaml / config.py
│   ├── metrics.json (학습 로그)
│   └── eval_results/results.json
├── evaluation/results.json                       ← 전체 결과 마스터 파일
└── reports/                                      ← CSV, 비교 테이블
```

### 리포트 생성

```bash
# 비교 테이블 (터미널 출력)
python -m training.utils.report --experiment exp2

# CSV 내보내기
python -m training.utils.report --experiment exp2 --csv

# 특정 메트릭 기준
python -m training.utils.report --experiment exp3 --metric bbox_AP
```

---

## 7. 권장 실행 순서

### Step 1: E2E 검증 (필수)

아직 실제 학습을 돌려본 적 없으므로, 먼저 짧은 테스트로 파이프라인 검증.

```bash
# detectron2 계열 검증
python -m training.train \
    --category Cable --experiment exp2 --condition cond1 \
    --model mask_rcnn --max-iter 100

# mmdet 계열 검증
python -m training.train \
    --category Cable --experiment exp2 --condition cond1 \
    --model solov2 --max-iter 100
```

### Step 2: Screw exp2 전체 (데이터 완비)

```bash
# Screw는 유일하게 exp2 전체 조건 충족
python -m training.train \
    --category Screw --experiment exp2 --condition all \
    --model mask_rcnn

python -m training.train \
    --category Screw --experiment exp2 --condition all \
    --model maskdino
```

### Step 3: 데이터 부족 카테고리 GenAI 추가 생성

- Cable: 현재 104장 → 250장 필요 (+146장)
- Casting: 현재 193장 → 250장 필요 (+57장)

### Step 4: 3카테고리 × exp2 전체

```bash
python -m training.train \
    --category all --experiment exp2 --condition all \
    --model mask_rcnn
```

### Step 5: exp3 7모델 비교

```bash
python -m training.train \
    --category all --experiment exp3 --condition original_only \
    --model all
```

---

## 8. 주의사항

- **GPU 메모리**: A100 80GB 기준 batch_size=2~4 권장. MaskDINO/Mask2Former는 메모리 사용량이 큼
- **학습 시간 참고**: MaskDINO 10K iter 기준 약 1~2시간 (A100, 소규모 데이터)
- **`--model all` 사용 시**: 7개 모델 순차 실행이므로 시간이 오래 걸림
- **중단 후 재시작**: detectron2는 `model_final.pth`가 있으면 이미 완료된 것으로 간주, mmdet는 `--resume` 미구현 (현재 항상 처음부터)
- **Cable val 필터링**: thunderbolt만 평가에 사용됨 (자동 처리)
- **config.py 수정**: 실험 조건이나 하이퍼파라미터 기본값을 변경하려면 `training/config.py` 수정
