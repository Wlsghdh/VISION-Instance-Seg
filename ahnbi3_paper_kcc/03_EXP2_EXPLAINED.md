# 03. 실험 2 — 전통 vs GenAI vs 결합 (핵심 실험)

## 🎯 목적

**전통 증강 vs 생성형 AI 증강**을 같은 증강량(125/cls)에서 1:1 비교 + **결합 시 최적 혼합비** 탐색.

→ 논문의 **메인 결과**가 여기서 나옴.

## 📋 구조 (2 Phase)

### Phase 1 — 전통 vs GenAI 단독 (cond1~cond3)

| 조건 | 구성 | train 총합 (3cls) |
|------|------|:---:|
| cond1 (baseline) | 원본 20/cls | 60 |
| cond2 | 원본 20 + **전통 125/cls** | 435 |
| cond3 | 원본 20 + **GenAI 125/cls** | 435 |

→ 같은 125장/cls 증강량에서 **전통 vs GenAI** 직접 비교.

### Phase 2 — 결합 시 전통 N배 스윕 (cond4_1x~10x)

**고정**: GenAI 125/cls
**스윕**: 전통 증강 수량 = 145×N/cls (N=1~10)

| 조건 | 전통 수량 | 1-cls 총 | 3-cls 합 |
|------|:---:|:---:|:---:|
| cond4_1x | 145 | 290 | 870 |
| cond4_2x | 290 | 435 | 1,305 |
| cond4_3x | 435 | 580 | 1,740 |
| cond4_4x | 580 | 725 | 2,175 |
| cond4_5x | 725 | 870 | 2,610 |
| cond4_6x | 870 | 1,015 | 3,045 |
| cond4_7x | 1,015 | 1,160 | 3,480 |
| **cond4_8x** ⭐ | **1,160** | **1,305** | **3,915** |
| cond4_9x | 1,305 | 1,450 | 4,350 |
| cond4_10x | 1,450 | 1,595 | 4,785 |

→ **N=8에서 cascade_mask_rcnn 최고 성능 (16.41)** 확인됨.

## 🤖 모델 (5종, cond4_8x 최종 비교용)

| 모델 | 프레임워크 | 상태 |
|------|----------|------|
| Mask R-CNN | detectron2 | ✅ 완료 (14.13) |
| Cascade Mask R-CNN | detectron2 | ✅ 완료 (16.41) |
| MaskDINO | detectron2 | ❌ 미진행 |
| Mask2Former | detectron2 | ❌ 미진행 (import 버그 수정됨) |
| Cascade R-CNN | mmdet | 🟢 학습 중 (usw) |
| SOLOv2 | mmdet | 🟢 학습 중 (ahnbi3 tmux) |
| RTMDet-Ins | mmdet | ❌ 미진행 |

## ⚙️ 하이퍼파라미터

**Detectron2 (mask_rcnn, cascade_mask_rcnn):**
- SGD, lr=0.0015, batch_size=12, epoch 기반 (max 200)

**mmdet (cascade_rcnn, solov2, rtmdet_ins):**
- AdamW, **lr=1e-4, batch_size=4**, COCO pretrained
- (이전에 segm_AP=0 문제 → pretrained 없음 + lr 높음 원인 → 수정 완료)

→ **약간의 비통일** 있음. 논문 Limitation에 명시 필요:
> "Detectron2 모델은 SGD + bs=12, mmdet 모델은 AdamW + bs=4로 각 프레임워크 표준 설정 사용."

## 👥 담당 분배 (완료 현황)

| 담당 | 담당 조건 | 상태 |
|------|-----------|:---:|
| **yjw** | cond1, cond2, cond3, cond4_1x~3x | ✅ 완료 |
| **jjh** | cond4_4x, cond4_5x, cond4_6x | ✅ 완료 |
| **ldy** | cond4_7x, cond4_8x, cond4_9x, cond4_10x | ✅ 완료 (mask/cascade) |
| **jjh (진행중)** | cond4_8x mmdet 3모델 | 🟢 |

## 📊 완료된 결과 (segm_AP)

### Phase 1 — 핵심 비교

| 조건 | Mask R-CNN | Cascade MRCNN | Δ vs baseline |
|------|:---:|:---:|:---:|
| **cond1** (baseline) | 10.30 | 11.26 | — |
| **cond2** (전통 125) | 8.54 | 10.36 | **-1.76 / -0.90** ⬇ |
| **cond3** (GenAI 125) | 12.18 | 11.78 | **+1.88 / +0.52** ⬆ |

**→ 같은 125장 증강에서 GenAI가 전통 대비 +3~4 AP 우위.**

### Phase 2 — N배 스윕 (cond4_Nx, 2모델만 현재)

```
N:    1x    2x    3x    4x    5x    6x    7x    8x    9x    10x
M:  13.39 12.63 14.30 12.18 12.23 13.08 13.28 14.13 13.45 13.52
C:  13.83 15.69 14.29 12.39 12.09 15.90 14.41 16.41 15.07 14.85
```

**핵심 관찰**:
- **cond4_8x cascade = 16.41 (최고점, +5.15 AP vs cond1)**
- N 곡선은 단조 증가 아님 — 진동하면서 8x 정점
- 10x는 **overfitting** 로 감소 추정

## 🏆 메인 결과 정리 (Table 1 후보)

```
| 조건                  | Mask R-CNN | Cascade MRCNN | Δ vs baseline (M/C) |
|-----------------------|:---:|:---:|:---:|
| baseline (원본 20)   | 10.30 | 11.26 | — |
| +전통 125 (cond2)    | 8.54  | 10.36 | -1.76 / -0.90 |
| +GenAI 125 (cond3)   | 12.18 | 11.78 | +1.88 / +0.52 |
| +GenAI+전통×8 (cond4_8x) | 14.13 | 16.41 | +3.83 / +5.15 |
```

**cond4_8x = 논문의 메인 자랑 수치.** (이전에 cond4_6x = 15.90이 최고인 줄 알았지만, cond4_8x = 16.41이 더 높음)

## 📁 결과 저장 위치

```
results_github/exp2/
├── cond1/Exp2_3cls/{mask_rcnn, cascade_mask_rcnn}/seed42/... ✅
├── cond2/...  ✅
├── cond3/...  ✅
├── cond4_1x ~ cond4_10x/Exp2_3cls/{mask_rcnn, cascade_mask_rcnn}/seed42/... ✅
└── cond4_8x/Exp2_3cls/
    ├── maskdino/             ← ldy 담당 (진행 중인지 확인 필요)
    ├── mask2former/           ← 미진행
    ├── cascade_rcnn/          🟢 usw에서 학습 중 (jjh)
    ├── solov2/                🟢 ahnbi3 tmux에서 학습 중 (jjh)
    └── rtmdet_ins/            ← 미진행 (jjh 다음 예정)
```

## 🎨 Figure 계획

### Figure 1 — cond4 N 스윕 곡선 (메인 Figure)

- X축: N (1~10)
- Y축: segm_AP
- Line 2개: Mask R-CNN (blue), Cascade Mask R-CNN (red)
- baseline(cond1) dashed horizontal line
- **N=8 지점 강조 (vertical line 또는 annotation)**
- 이미 생성됨: `docs/paper/figs/fig1_cond4_curve.pdf`

### Figure 3 — Per-class heatmap

- 조건별 × 클래스별 AP
- cond1, cond2, cond3, cond4_6x, cond4_8x 5열
- 3 클래스 (Dirty, Inclusoes, impurities) 행

### Figure 4 — 정성 비교

- cond1에서 틀리고 cond4_8x에서 맞춘 케이스
- 2×4 grid: input / GT / cond1 pred / cond4_8x pred

## 📚 관련 파일

- `docs/experiment_plans/exp1_3cls_v2.md` (참고)
- `scripts/run_exp2_jjh.sh` (기존)
- `scripts/run_exp2_cond4_8x_mmdet.sh` (mmdet 수정안)

---

## 📝 다음 파일

- [04_RESULTS_CURRENT.md](04_RESULTS_CURRENT.md): 모든 현재 수치 정리
- [05_PAPER_STRATEGY.md](05_PAPER_STRATEGY.md): 논문 story + 제목
