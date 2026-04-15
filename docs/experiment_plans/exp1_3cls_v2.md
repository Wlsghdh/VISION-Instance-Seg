# Exp1_3cls — GenAI 증강 수에 따른 성능 변화 (3 클래스 축소판) **v2**

**버전**: v2 (실험박사 자문 반영, 리뷰어 재검토 대기)
**작성일**: 2026-04-15
**작성자**: jjh (Claude Code 보조)
**v1 대비 주요 변경**: iter 기반 전면 전환, val/test 분리, Cosine schedule, 팀 동기화 자동화, annotation 분포 수치, nested 샘플링, smoke test 기준

---

## 1. 배경 및 목적

기존 Exp1(Unified 14 클래스)의 overall AP는 쉬운 클래스(PistonMiss 100, Gap 65)가 평균을 끌어올려 baseline이 유리하게 나온 측면이 있음. GenAI가 유효한 클래스(Dirty/Inclusoes/impurities/Porosity)와 무관한 결함이 섞여 있어 신호 왜곡 가능.

→ **Exp2_3cls와 동일한 3 defect(Dirty, Inclusoes, impurities)에 한정**하여 GenAI 수 스윕(baseline + genai_{25,50,75,100,125})을 재실행. Exp2_3cls(cond1~cond3)와 직접 비교 가능한 결과 확보.

## 2. 실험 설정

### 2.1 데이터

**원본 annotation 분포 (실측 필요 — 구현 단계에서 채움)**

| 클래스 | 원본 train 이미지 | 원본 annotations | val 원본 이미지 | val 원본 annotations | GenAI 풀 |
|--------|:---:|:---:|:---:|:---:|:---:|
| Dirty | 20 | TBD | TBD | TBD | TBD |
| Inclusoes | 20 | TBD | TBD | TBD | TBD |
| impurities | 20 | TBD | TBD | TBD | TBD |
| **합계** | 60 | TBD | 82 | 113 | TBD |

> TBD 는 `scripts/data_utils/collect_annotation_stats.py` 실행 후 v2 최종본에서 채움.

**Val/Test 분리 (신규)**:
- 기존 val 82장을 **클래스 balanced stratified split** → `val_dev 41장 / val_test 41장`
- `val_dev` → 학습 중 early-stop & best model 선택
- `val_test` → **최종 보고용** (모델 선택과 독립)
- Exp2_3cls와의 비교용으로 **full val 82장 수치도 부록** 제공
- 분할 스크립트: `scripts/data_utils/stratify_exp1_3cls_val.py` (seed=12345 고정)
- 결과: `results/merged_datasets/_exp1_3cls_val_split.json` (이미지 ID 리스트)

### 2.2 조건 (6개, nested 샘플링)

| 조건 | 원본 | +GenAI/cls | GenAI 총 | train 합 |
|------|:---:|:---:|:---:|:---:|
| baseline | 60 | 0 | 0 | 60 |
| genai_25 | 60 | 25 | 75 | 135 |
| genai_50 | 60 | 50 | 150 | 210 |
| genai_75 | 60 | 75 | 225 | 285 |
| genai_100 | 60 | 100 | 300 | 360 |
| genai_125 | 60 | 125 | 375 | 435 |

**Nested 보장**: `genai_N` ⊃ `genai_N-25` (seed=42 기준 동일 정렬 후 앞에서부터 N개 선택). `data_pipeline.py` 검증.

### 2.3 모델 (2개)
- Mask R-CNN (R-50 FPN, COCO pretrained, detectron2)
- Cascade Mask R-CNN (R-50 FPN, COCO pretrained, detectron2)

### 2.4 하이퍼파라미터 (**iter 기반 전면 전환**)

| 항목 | 값 | 근거 |
|------|:---:|------|
| batch_size | **12** (고정) | 공정 비교 원칙, A100 80GB 여유 |
| lr | **0.0015** | Linear Scaling Rule (Goyal 2017): `0.02 × 12/16 / 10` |
| optimizer | SGD (momentum=0.9, weight_decay=1e-4) | Detectron2/mmdet 표준 |
| **max_iters** | **20,000** | 상한, early-stop이 실제 종료 결정 |
| **warmup_iters** | **500 (linear, start_factor=0.001)** | 전체 2.5%, Detectron2 표준 |
| **lr schedule** | **WarmupCosineLR (detectron2) / CosineAnnealingLR by_epoch=False (mmdet)** | early-stop 친화, 조건별 종료 지점 편차 흡수 (SGDR, Loshchilov 2017) |
| **eval_period_iters** | **500** | 40회 평가 지점 |
| **early_stop patience** | **15 evals** (= 7,500 iter, 37.5%) | mmengine EarlyStoppingHook은 val 호출 횟수 단위 |
| checkpoint_period_iters | 2,000 | 최근 1개만 유지 |
| seed | **42 (기본)**; baseline·genai_125 는 `{42, 43, 44}` 3-seed | variance 관찰 |
| input_min_size | (640,672,704,736,768,800) | 기존 유지 |
| input_max_size | 1333 | 기존 유지 |

**핵심**: Step decay 삭제 (early-stop 전에 발동 안 해서 불공정). Cosine으로 진행률 기반 smooth 감쇠 → 조건별 종료 iter가 5k~14k로 달라도 공정.

### 2.5 평가

- **주 지표**: `segm_AP` @ **val_test (41장)** (COCO mAP, IoU 0.50:0.05:0.95)
- **부록**: `segm_AP` @ val_dev, `segm_AP` @ full val (Exp2_3cls 호환)
- 보조: AP50, AP75, per-class AP (Dirty/Inclusoes/impurities), bbox_AP
- **Leakage 완화**: 모델 선택은 val_dev 기준, 보고는 val_test → 독립성 확보

## 3. 담당자 분배

| 담당 | 조건 (×2 모델, seed=42 기본) | 3-seed 추가 | 총 학습 수 |
|------|-----|-----|:---:|
| **jjh** | baseline / genai_75 / genai_125 | baseline×2, genai_125×2 (seed 43,44, MRCNN/CMRCNN 각 1) | 6 + 4 = 10 |
| **yjw** | genai_25 / genai_50 / genai_100 | (없음) | 6 |

- ldy는 제외
- jjh 추가 부담은 variance 측정용 (baseline/genai_125만 3-seed)

## 4. 구현 계획 (코드 수정 세부)

### 4.1 `training/config.py` (L299-317 교체)

```python
DEFAULT_HYPERPARAMS = {
    # === iter 기반 (우선) ===
    "max_iters": 20000,
    "warmup_iters": 500,
    "eval_period_iters": 500,
    "checkpoint_period_iters": 2000,
    "early_stopping_patience": 15,      # eval 횟수 단위
    "early_stopping_metric": "segm/AP",
    # === 공통 ===
    "lr": 0.0015,
    "batch_size": 12,
    "seed": 42,
    "max_periodic_checkpoints": 1,
    "input_min_size": (640, 672, 704, 736, 768, 800),
    "input_max_size": 1333,
    "lr_scheduler": "cosine",           # "cosine" or "step"
    # === epoch 기반 (하위호환, 지정 시에만 사용) ===
    "max_epochs": None,
    "warmup_epochs": None,
    "eval_period_epochs": None,
}
```

**신규 EXPERIMENT 등록**:
```python
EXPERIMENTS["exp1_3cls"] = {
    "categories": ["Exp2_3cls"],
    "models": ["mask_rcnn", "cascade_mask_rcnn"],
    "conditions": {
        "baseline":  {"n_original_per_class": 20, "n_genai_per_class": 0,   "n_traditional_per_class": 0},
        "genai_25":  {"n_original_per_class": 20, "n_genai_per_class": 25,  "n_traditional_per_class": 0},
        "genai_50":  {"n_original_per_class": 20, "n_genai_per_class": 50,  "n_traditional_per_class": 0},
        "genai_75":  {"n_original_per_class": 20, "n_genai_per_class": 75,  "n_traditional_per_class": 0},
        "genai_100": {"n_original_per_class": 20, "n_genai_per_class": 100, "n_traditional_per_class": 0},
        "genai_125": {"n_original_per_class": 20, "n_genai_per_class": 125, "n_traditional_per_class": 0},
    },
}
```

### 4.2 `training/train.py` CLI 추가

```python
parser.add_argument('--max-iters', type=int, default=None)
parser.add_argument('--warmup-iters', type=int, default=None)
parser.add_argument('--eval-period-iters', type=int, default=None)
parser.add_argument('--lr-scheduler', type=str, default=None, choices=[None, 'cosine', 'step'])
```

### 4.3 `training/adapters/detectron2_adapter.py` (L284-325 수정)

- iter 기반 필드가 있으면 우선 사용, epoch→iter 환산 로직 우회
- `cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"` (lr_scheduler=="cosine"일 때)
- `cfg.SOLVER.MAX_ITER = max_iters`
- `cfg.SOLVER.WARMUP_ITERS = warmup_iters`
- `cfg.TEST.EVAL_PERIOD = eval_period_iters`
- EarlyStoppingHook 인자 `iters_per_epoch` 제거 (로그용만 유지)

### 4.4 `training/adapters/mmdet_adapter.py` (L117-147 완전 교체)

```python
cfg.train_cfg = dict(type="IterBasedTrainLoop",
                     max_iters=max_iters, val_interval=eval_period_iters)
cfg.param_scheduler = [
    dict(type="LinearLR", start_factor=0.001,
         by_epoch=False, begin=0, end=warmup_iters),
    dict(type="CosineAnnealingLR", by_epoch=False,
         begin=warmup_iters, end=max_iters,
         T_max=max_iters - warmup_iters),
]
cfg.default_hooks.checkpoint = dict(
    type="CheckpointHook", by_epoch=False,
    interval=ckpt_period_iters, max_keep_ckpts=3,
    save_best="coco/segm_mAP", rule="greater",
)
cfg.custom_hooks = [
    dict(type="EarlyStoppingHook", monitor="coco/segm_mAP",
         patience=patience, min_delta=0.0, rule="greater"),
]
# sampler: InfiniteSampler로 변경 (IterBased 필수)
cfg.train_dataloader.sampler = dict(type="InfiniteSampler", shuffle=True)
```

### 4.5 신규 스크립트

- `scripts/data_utils/stratify_exp1_3cls_val.py` — val 82장을 클래스 stratified 50/50 분할
- `scripts/data_utils/collect_annotation_stats.py` — 조건별 annotation 분포 출력
- `scripts/data_utils/freeze_exp1_3cls.sh` — git tag + sha256 manifest
- `scripts/data_utils/verify_exp1_3cls.sh` — 학습 전 해시 검증
- `scripts/run_exp1_3cls_jjh.sh`, `scripts/run_exp1_3cls_yjw.sh`

### 4.6 data_pipeline.py nested 샘플링 검증

- 현재 `_sample_per_class(rng.sample)` 인지 확인
- 있다면 **seed 고정 후 sort → head(N)** 로 변경하여 nested 보장

## 5. Smoke Test (본 실행 전 필수)

**구성**: `baseline` × Mask R-CNN × seed 42, `--max-iters 500 --warmup-iters 50 --eval-period-iters 100 --patience 3`.

**통과 기준 (7개 모두)**:
1. 학습이 정확히 500 iter에서 종료 (또는 early-stop)
2. 로그에 iter 기반 진행 표시 (`iter 100/500` 등)
3. val 평가가 100/200/300/400/500에서 5회 발동
4. Cosine schedule 정상 — step decay 미발동
5. `eval_results/results.json` 생성 + `segm_AP` 존재
6. mmdet smoke (Cascade R-CNN 등) 에서 `IterBasedTrainLoop` 로그 확인, `EpochBasedTrainLoop` 없음
7. 전체 smoke < 10분 (A100 기준)

**실패 시**: 본 실행 금지, 코드 수정 → 재smoke.

## 6. 재현성 / 팀 동기화

- **git tag**: 학습 시작 전 `exp1_3cls_frozen_YYYYMMDD` 태그. 두 담당자 동일 태그에서 checkout.
- **데이터 manifest**: `sha256sum` 로 `data/{Inclusoes,Dirty,impurities}` + `data_augmented/*/gen_ai` 전체 해시 → `docs/experiment_plans/exp1_3cls_v2_data_manifest.txt`.
- **verify 스크립트**: 학습 스크립트 진입 첫 줄에서 `bash scripts/data_utils/verify_exp1_3cls.sh` 호출, 불일치 시 abort.
- **commit hash + git status**: 학습 결과 디렉토리에 `run_meta.json` 으로 저장.
- **conda env**: `/home/jjh0709/.conda/envs/jjh` 공용.
- **GPU**: jjh=GPU 1, yjw=GPU 2 (충돌 방지).

## 7. 예상 시간

**iter 기반이라 조건별 시간 거의 동일** (max 20k, early-stop 5k~14k).

| 담당 | 학습 수 | 예상 시간 (A100 단일) |
|------|:---:|:---:|
| jjh | 10 (3seed 2조건 포함) | ~25-30h |
| yjw | 6 | ~15-18h |

## 8. 한계 및 수용 리스크

1. val_test 41장은 작음 → variance 여전히 존재. 3-seed(baseline/genai_125)로 보완.
2. annotation 수가 적어 per-class AP는 noise 큼.
3. 팀 간 GPU 세대/드라이버 차이 가능 (같은 서버라 낮음).

## 9. v2 자가 점검 체크리스트

- [x] iter 기반 필드 전면 교체 제시
- [x] config/train/어댑터 수정 포인트 파일:라인 명시
- [x] mmdet `IterBasedTrainLoop` 코드 스니펫 포함
- [x] val/test 층화 분리 설계 + 스크립트 경로
- [x] annotation 개수 테이블 구조 (TBD는 구현 단계 채움)
- [x] GenAI nested 샘플링 보장 방법 명시
- [x] git tag + sha256 manifest 프로토콜
- [x] smoke test 7개 기준
- [x] Cosine schedule 채택 + step 제거 이유
- [x] baseline/genai_125 3-seed 계획
- [x] 예상 시간 iter 기반 재산정

## 10. PASS 기준 (본 실행 승인 조건)

- reviewer agent PASS
- smoke test 7개 기준 전부 통과
- data manifest 해시 양 담당자 일치 확인
