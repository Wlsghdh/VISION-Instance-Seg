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
| `--skip-existing` | (플래그) | 이미 학습+평가가 완료된 (model_best + eval_results) 조합 건너뜀 |
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

출력: `results/merged_datasets/exp1/baseline/Unified/seed42/{images/, annotations.json}` (시드별 분리, 링크 사용 — 하드링크 → 심볼릭링크 fallback)

---

## 4. 실험별 조건

> **원본 이미지 수 제한**: 모든 조건에서 **클래스당 20장** (`config.py`의 `N_ORIGINAL_TRAIN_PER_CLASS=20`).
> `_sample_images_per_class()`로 클래스별 균형 샘플링. 클래스의 가용 이미지가 20장 미만이면 자동으로 전체 사용.
>
> **Unified 모드 실측 (14 클래스)**: Cable 20 + Screw 20 + Casting 40 + Console 80 + Cylinder 80 + Wood 40 = **총 280장**

### exp1: GenAI 증강 수에 따른 성능 변화

| 조건명 | 원본 (클래스당) | GenAI/클래스 |
|--------|:--------------:|:------------:|
| `baseline` | 20장 | 0 |
| `genai_25` | 20장 | 25 |
| `genai_50` | 20장 | 50 |
| `genai_75` | 20장 | 75 |
| `genai_100` | 20장 | 100 |
| `genai_125` | 20장 | 125 |

기본 모델: `mask_rcnn`, `cascade_mask_rcnn`

### exp2: 전통 증강 vs GenAI 비교

| 조건명 | 원본 (클래스당) | GenAI/클래스 | 전통 증강 |
|--------|:--------------:|:-----------:|:---------:|
| `cond1` | 20장 | 0 | 0 |
| `cond2` | 20장 | 0 | 250 |
| `cond3` | 20장 | 125 | 0 |
| `cond4` | 20장 | 125 | 250 |
| `cond5` | 20장 | 125 | 2,750 |

기본 모델: `mask_rcnn`, `cascade_mask_rcnn`, `maskdino`

### exp3: 7종 모델 비교

| 조건명 | 구성 |
|--------|------|
| `original_only` | 원본 클래스당 20장 |
| `with_trad` | 원본 20/cls + 전통 증강 3,000 |
| `with_genai_trad` | 원본 20/cls + GenAI 125/cls + 전통 증강 2,750 |

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

# 단일 시드로 먼저 돌렸다가 나중에 다중 시드로 확장하는 경우
# 기존 seed42는 재사용, seed43/seed44만 신규 학습
python -m training.train \
    --category Unified --experiment exp1 --condition all \
    --model all --multi-seed --skip-existing
```

결과 경로 (단일 시드도 동일하게 seed{N}/ 하위 폴더 사용):
```
results/training/exp1/{cond}/Unified/{model}/seed42/   ← 단일 모드도 여기에 저장 (기본)
results/training/exp1/{cond}/Unified/{model}/seed43/   ← --multi-seed 시 추가
results/training/exp1/{cond}/Unified/{model}/seed44/   ← --multi-seed 시 추가
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

`--tag` 사용 시 출력 경로: `results/training/exp1/baseline/Unified/cascade_mask_rcnn_bs4_lr5e-4/seed42/`

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
├── merged_datasets/{exp}/{cond}/{cat}/seed{N}/   ← 병합 데이터 (시드별 분리, 링크 사용)
│   ├── images/                                    원본 데이터에 링크 (하드링크 → 심볼릭링크 fallback)
│   └── annotations.json                           시드별로 다른 데이터 샘플링 (true independent replication)
│
├── _unified_val/                                  ← 통합 val 데이터 (시드 무관)
│   ├── images/
│   └── annotations.json
│
├── training/{exp}/{cond}/{cat}/{model}/seed{N}/  ← 학습 결과 (단일/다중 시드 모두 seed{N}/ 사용)
│   ├── model_best.pth         ← 최종 보존 파일. 학습 후 cleanup으로 이것만 남음
│   ├── config.yaml            ← 학습 설정 스냅샷
│   ├── metrics.json           ← 학습 로그 (epoch별 loss/lr/eval)
│   ├── events.out.tfevents.*  ← TensorBoard 로그
│   └── eval_results/results.json  ← COCO 평가 결과
│
│   (학습 중 임시 — 평가 성공 시 자동 삭제됨):
│   ├── model_final.pth
│   ├── model_0000XXX.pth   (rotation 최신 1개)
│   ├── best_*.pth          (mmdet 계열)
│   └── last_checkpoint
│
├── evaluation/results.json                       ← 전체 결과 마스터 (매 condition 끝날 때마다 incremental save)
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

### 체크포인트 자동 Rotation + Cleanup

학습 중에는 **rotation**, 학습 끝나면 **cleanup**으로 디스크를 최소화한다.

#### 학습 중: Rotation
`max_periodic_checkpoints=1` (config.py 기본값) — 주기 체크포인트는 최신 1개만 자동 유지.
- detectron2: `PeriodicCheckpointer(max_to_keep=1)` — `model_final.pth`는 fvcore에서 명시적으로 보호.
- mmdet: `CheckpointHook(max_keep_ckpts=3)` — `best_*.pth` 보호 버퍼로 최소 3개 유지.
- `model_best.pth`는 rotation 큐에 들어가지 않아 항상 보존됨.

#### 학습 + 평가 후: Cleanup
`run_single()`에서 평가 성공 직후 `cleanup_artifacts()` 자동 호출.
- **삭제**: `model_final.pth`, `model_0000XXX.pth` (주기 체크포인트), `last_checkpoint`
- **보존**: `model_best.pth`, `eval_results/`, `config.yaml`, `metrics.json`, `events.out.tfevents.*`
- **안전장치**: `model_best.pth`가 존재할 때만 동작 (보존할 게 없으면 스킵)

#### 디스크 사용량 비교

| 시점 | Mask R-CNN | Cascade Mask R-CNN |
|---|---:|---:|
| 학습 중 (피크) | ~1.0 GB | ~1.6 GB |
| 평가 직후 (cleanup 적용) | **~0.34 GB** | **~0.55 GB** |
| 절감 | 66% | 66% |

→ 36 runs (multi-seed full): 60 GB → **20 GB**

**`merged_datasets` 디스크**:
- 원본 데이터 파일을 **링크로 연결**하여 디스크 데이터 1번만 저장
- 우선순위: 하드 링크 → 심볼릭 링크 → 풀 카피
  - 같은 파일시스템: 하드 링크 (가장 효율적)
  - 다른 파일시스템 (`/home/jjh0709/...` ↔ `/project/ahnailab/...`): 심볼릭 링크 (76 byte / 파일)
  - 둘 다 실패: 풀 카피 (드뭄)
- 시드별/조건별로 디렉토리는 분리되지만 데이터는 원본 1번만
- **실측**: exp1 6 conditions × 3 seeds = **~137 MB** (이전 풀 카피 시 ~5 GB → 97% 절감)

**수동 정리** (필요시):
```bash
# 특정 실험 통째로 삭제
rm -rf results/training/exp1/

# 학습 도중 강제 cleanup이 필요한 경우 (rotation 안 먹은 경우)
find results/training/exp1/ -regex '.*/model_[0-9]+\.pth' -delete
find results/training/exp1/ -name 'model_final.pth' -delete
```

---

## 9. 주의사항

- **GPU 메모리**: A100 80GB 기준 batch_size=12에서 ~12GB 사용. MaskDINO/Mask2Former는 더 큼
- **학습 시간 참고**: Cascade Mask R-CNN baseline ~1.5시간 (early stopping 작동 시)
- **`--multi-seed`**: 단순히 `--seed` 3개를 순차로 도는 게 아니라 결과 경로에 `seed{N}` 하위 폴더 생성하여 자동 분리
- **`--model all` 사용 시**: 실험에 정의된 모델만 순차 실행 (exp1은 Mask R-CNN, Cascade Mask R-CNN 2개)
- **중단 후 재시작**: 기본 동작은 그냥 재실행하면 출력 디렉토리를 덮어씀. 이미 학습+평가 완료된 조합을 건너뛰고 싶으면 `--skip-existing` 플래그 사용 (model_best.pth + eval_results/results.json 둘 다 있는 조합만 SKIP). 강제 재학습은 폴더 삭제 후 재실행
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
