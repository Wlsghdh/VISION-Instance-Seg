# 학습 Guideline

> 통합 학습 환경 사용 가이드 (Unified 카테고리 + 다중 시드 지원)

---

## 0. 사전 조건

```bash
# 본인 conda env (예시: test)
python --version  # Python 3.12 권장
python -c "import detectron2; print(detectron2.__version__)"  # 0.6

# 또는 jjh의 conda env 사용
/home/jjh0709/.conda/envs/jjh/bin/python --version

# 작업 디렉토리: 본인 클론 위치
cd /project/ahnailab/{user}/VISION-Instance-Seg
```

- 데이터 원본: `/home/jjh0709/gitrepo/VISION-Instance-Seg/data/` (읽기만)
- 결과 저장: 본인 폴더 `results/`
- 환경: PyTorch 2.5.1+cu121, CUDA 12.2 호환, A100 80GB × 1

---

## 1. 기본 명령어 구조

```bash
python -m training.train \
    --category {카테고리} \
    --experiment {실험} \
    --condition {조건} \
    --model {모델} \
    [--multi-seed] \
    [--tag TAG] \
    [옵션]
```

| 인자 | 값 | 설명 |
|------|---|------|
| `--category` | Cable, Screw, ..., Unified, all | Unified는 14클래스 통합 학습 |
| `--experiment` | exp1, exp2, exp3 | 실험 종류 |
| `--condition` | 실험별 조건명, all | 데이터 조건 |
| `--model` | 모델명, all | 학습 모델 |
| `--multi-seed` | (플래그) | seed=[42,43,44] 3회 반복 실행 |
| `--tag` | 문자열 | 결과 경로에 태그 추가 (HP 비교용) |
| `--eval-only` | (플래그) | 학습 스킵, 평가만 실행 |

### 하이퍼파라미터 오버라이드 (옵션)

| 인자 | 기본값 | 설명 |
|------|:------:|------|
| `--lr` | 0.0015 | 학습률 |
| `--batch-size` | 12 | 배치 크기 |
| `--max-epochs` | 1000 | 최대 에폭 |
| `--seed` | 42 | 시드 (단일 시드 모드) |
| `--patience` | 15 | early stopping patience |
| `--eval-period-epochs` | 5 | 평가 주기 |

기본값은 `training/config.py`의 `DEFAULT_HYPERPARAMS` 참조.
모델별 오버라이드는 `MODELS[모델]["hyperparams"]`에서 정의.

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

## 3. 통합 카테고리 (Unified) — 14클래스 통합 학습

6개 카테고리를 합쳐 1개 모델로 14개 결함 클래스를 동시 학습.

### 글로벌 클래스 매핑

| ID | 카테고리 | 클래스 |
|:--:|---------|--------|
| 0 | Cable | thunderbolt |
| 1 | Screw | defect |
| 2-3 | Casting | Inclusoes, Rechupe |
| 4-7 | Console | Collision, Dirty, Gap, Scratch |
| 8-11 | Cylinder | Chip, PistonMiss, Porosity, RCS |
| 12-13 | Wood | impurities, pits |

### 데이터 준비만 (학습 없이)

```bash
python -m training.data_pipeline \
    --category Unified --experiment exp1 --condition baseline

# 강제 재생성
python -m training.data_pipeline \
    --category Unified --experiment exp1 --condition baseline --force
```

출력: `results/merged_datasets/exp1/baseline/Unified/{images/, annotations.json}`

---

## 4. 실험별 조건

### exp1: GenAI 증강 수에 따른 성능 변화

| 조건명 | 원본 | GenAI/클래스 |
|--------|:----:|:------------:|
| `baseline` | 전체 | 0 |
| `genai_25` | 전체 | 25 |
| `genai_50` | 전체 | 50 |
| `genai_75` | 전체 | 75 |
| `genai_100` | 전체 | 100 |
| `genai_125` | 전체 | 125 |

기본 모델: `mask_rcnn`, `cascade_mask_rcnn`

### exp2: 전통 증강 vs GenAI 비교

| 조건명 | 원본 | GenAI/클래스 | 전통 증강 |
|--------|:----:|:-----------:|:---------:|
| `cond1` | 전체 | 0 | 0 |
| `cond2` | 전체 | 0 | 250 |
| `cond3` | 전체 | 125 | 0 |
| `cond4` | 전체 | 125 | 250 |
| `cond5` | 전체 | 125 | 2,750 |

기본 모델: `mask_rcnn`, `cascade_mask_rcnn`, `maskdino`

### exp3: 7종 모델 비교

| 조건명 | 구성 |
|--------|------|
| `original_only` | 원본 전체 |
| `with_trad` | 원본 전체 + 전통 증강 3,000 |
| `with_genai_trad` | 원본 전체 + GenAI 125/클래스 + 전통 증강 2,750 |

기본 모델: 7종 전체

---

## 5. 사용 예시

### 5-1. 단일 시드 학습 (빠른 테스트)

```bash
# Unified, exp1 baseline, Cascade Mask R-CNN
python -m training.train \
    --category Unified --experiment exp1 --condition baseline \
    --model cascade_mask_rcnn

# 6개 조건 전체
python -m training.train \
    --category Unified --experiment exp1 --condition all \
    --model cascade_mask_rcnn

# 두 모델 동시
python -m training.train \
    --category Unified --experiment exp1 --condition all \
    --model all
```

### 5-2. 다중 시드 학습 (정식 실험, 3회 반복)

```bash
# 6 조건 × 2 모델 × 3 시드 = 36회
python -m training.train \
    --category Unified --experiment exp1 --condition all \
    --model all --multi-seed
```

결과 경로:
```
results/training/exp1/{cond}/Unified/{model}/seed42/
results/training/exp1/{cond}/Unified/{model}/seed43/
results/training/exp1/{cond}/Unified/{model}/seed44/
```

### 5-3. 하이퍼파라미터 커스텀

```bash
# lr, batch_size 직접 지정
python -m training.train \
    --category Unified --experiment exp1 --condition baseline \
    --model cascade_mask_rcnn --lr 5e-4 --batch-size 4

# HP 비교 시 결과 분리 (--tag)
python -m training.train \
    --category Unified --experiment exp1 --condition baseline \
    --model cascade_mask_rcnn --lr 5e-4 --batch-size 4 \
    --tag bs4_lr5e-4
```

`--tag` 사용 시 출력 경로: `results/training/exp1/baseline/Unified/cascade_mask_rcnn_bs4_lr5e-4/`

### 5-4. 평가만 (이미 학습된 모델)

```bash
# 특정 모델 평가
python -m training.train \
    --category Unified --experiment exp1 --condition baseline \
    --model cascade_mask_rcnn --eval-only

# 독립 평가 스크립트 (일괄)
python -m training.evaluate \
    --category Unified --experiment exp1 --condition all --model all
```

### 5-5. 카테고리별 학습 (Unified 대신)

Unified 외에 카테고리별로 따로 학습할 수도 있음.

```bash
# Cable만
python -m training.train \
    --category Cable --experiment exp1 --condition baseline \
    --model cascade_mask_rcnn

# 6개 카테고리 전체 (Unified 제외)
python -m training.train \
    --category all --experiment exp1 --condition baseline \
    --model cascade_mask_rcnn
```

---

## 6. 백그라운드 실행 (tmux)

```bash
# tmux 세션 시작
tmux new -s exp1 "python -m training.train \
    --category Unified --experiment exp1 --condition all \
    --model all --multi-seed 2>&1 | tee results/exp1.log"

# 세션 빠져나오기: Ctrl+B → D
# 다시 들어가기:
tmux attach -t exp1
# 종료:
tmux kill-session -t exp1
```

`conda activate`가 tmux 안에서 안 되면 python 절대경로 사용:
```bash
tmux new -s exp1 "/home/{user}/.conda/envs/{env}/bin/python -m training.train ..."
```

---

## 7. 결과 확인

### 결과 파일 위치

```
results/
├── merged_datasets/{exp}/{cond}/{cat}/          ← 병합 데이터
│   ├── images/
│   └── annotations.json
│   (Unified 카테고리는 6개 카테고리 통합)
│
├── _unified_val/                                 ← 통합 val 데이터
│   ├── images/
│   └── annotations.json
│
├── training/{exp}/{cond}/{cat}/{model}/          ← 학습 결과
│   ├── model_final.pth (detectron2)
│   ├── best_*.pth (mmdet)
│   ├── config.yaml / config.py
│   ├── metrics.json (학습 로그)
│   └── eval_results/results.json
│   (다중 시드 모드: model/seed42/, seed43/, seed44/ 하위로 분리)
│
├── evaluation/results.json                       ← 전체 결과 마스터 파일
└── reports/                                      ← CSV, 비교 테이블
```

### 결과 JSON 항목

| 키 | 설명 |
|----|------|
| `category, experiment, condition, model, seed` | 식별자 |
| `eval.segm_AP` | 14클래스 평균 instance segmentation mAP (핵심 지표) |
| `eval.bbox_AP` | bbox 기준 mAP |
| `eval.segm_AP-{class}` | 클래스별 AP |
| `train.peak_memory_mb` | GPU 메모리 (allocated peak) |
| `train.peak_memory_reserved_mb` | GPU 메모리 (reserved peak, nvidia-smi 기준) |
| `train.gpu_utilization_peak_pct` | GPU 활용률 피크 |
| `train.early_stopped` | early stopping 발동 여부 |
| `train.early_stop_epoch` | 중단된 epoch |
| `train.total_epochs` | 학습한 총 epoch |
| `train_time_sec` | 학습 시간 (초) |
| `train.pre_train_gpu` | 학습 전 GPU 상태 (다른 프로세스 영향 확인용) |

### 리포트 생성

```bash
# 비교 테이블 (터미널 출력)
python -m training.utils.report --experiment exp1

# CSV 내보내기
python -m training.utils.report --experiment exp1 --csv

# 특정 메트릭 기준
python -m training.utils.report --experiment exp1 --metric bbox_AP
```

### 학습 로그 확인

```bash
tail -20 results/exp1.log
grep "EarlyStopping" results/exp1.log
```

---

## 8. 디스크 관리

### 현재 사용량

```bash
quota -s   # 본인 쿼타
du -sh results/training/  # 결과 폴더 크기
```

### 체크포인트 정리

학습 후 자동으로 정리되지 않는 경우:

```bash
# model_final.pth만 남기고 중간 체크포인트 삭제
find results/training/exp1/ -name "model_0*.pth" -delete
```

체크포인트 1개 ≈ 548MB. `checkpoint_period_epochs=50`으로 설정되어 있어 50 에폭마다 1개 저장.

---

## 9. 주의사항

- **GPU 메모리**: A100 80GB 기준 batch_size=12에서 ~12GB 사용. MaskDINO/Mask2Former는 더 큼
- **학습 시간 참고**: Cascade Mask R-CNN baseline ~1.5시간 (early stopping 작동 시)
- **`--multi-seed`**: 단순히 `--seed` 3개를 순차로 도는 게 아니라 결과 경로에 `seed{N}` 하위 폴더 생성하여 자동 분리
- **`--model all` 사용 시**: 실험에 정의된 모델만 순차 실행 (exp1은 Mask R-CNN, Cascade Mask R-CNN 2개)
- **중단 후 재시작**: detectron2는 `model_final.pth`가 있으면 이미 완료된 것으로 간주, 다시 돌리려면 폴더 삭제 후 재실행
- **Cable val 필터링**: thunderbolt만 평가에 사용됨 (자동 처리)
- **Unified 카테고리 평가**: 14개 클래스 전체의 평균 mAP 사용 (클래스 단위 평균, 카테고리별 평균 아님)
- **config.py 수정**: 실험 조건이나 하이퍼파라미터 기본값을 변경하려면 `training/config.py` 수정

---

## 10. 디버깅

### tmux 세션이 바로 종료되는 경우
- `conda activate`가 안 됨 → python 절대경로 사용

### detectron2 import 에러
- 본인 conda env에 미설치 → 설치하거나 jjh의 python 사용

### CUDA 에러 / cuDNN 초기화 실패
- PyTorch가 서버 CUDA 드라이버(12.2)와 안 맞음 → `torch==2.5.1+cu121` 사용

### early stopping이 안 걸리는 경우
- 미세 개선으로 카운터가 리셋되는 것일 수 있음 → patience 조정 또는 그대로 진행
- 14클래스 평균 metric이 작은 변동에도 민감할 수 있음

### EarlyStopException이 에러로 기록되는 경우
- `EarlyStopException`은 `BaseException` 상속이어야 detectron2의 `except Exception`에 안 잡힘
- 코드 수정사항: `training/adapters/detectron2_adapter.py` 참조
