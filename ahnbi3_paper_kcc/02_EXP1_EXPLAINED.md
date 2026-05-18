# 02. 실험 1 — GenAI 증강 수량 스윕 (Exp1_3cls)

## 🎯 목적

원본 20장/클래스 기준으로 **GenAI 증강을 0부터 125장까지 스윕**했을 때 성능 변화 추이 관찰.

→ "GenAI 증강은 많을수록 좋은가? 포화점이 있는가?"를 검증.

## 📋 6개 조건

| 조건 | 원본 | +GenAI/cls | train 총합 (3cls) |
|------|:---:|:---:|:---:|
| baseline | 20 | 0 | 60 |
| genai_25 | 20 | 25 | 135 |
| genai_50 | 20 | 50 | 210 |
| genai_75 | 20 | 75 | 285 |
| genai_100 | 20 | 100 | 360 |
| genai_125 | 20 | 125 | 435 |

**Nested sampling**: `genai_N`의 샘플은 `genai_N-25` ⊂ `genai_N` 관계 (seed 고정).

## 🤖 모델 (2종)

- **Mask R-CNN** (R-50 FPN, COCO pretrained, detectron2)
- **Cascade Mask R-CNN** (R-50 FPN, COCO pretrained, detectron2)

## ⚙️ 하이퍼파라미터 (iter 기반 통일)

| 항목 | 값 |
|------|:---:|
| batch_size | 12 |
| lr | 0.0015 (SGD momentum=0.9, wd=1e-4) |
| max_iters | 20,000 |
| warmup_iters | 500 (linear) |
| lr_scheduler | WarmupCosineLR |
| eval_period_iters | 500 (40회 평가) |
| early_stop patience | 15 evals = 7,500 iter |
| seed | 42 (+일부 조건 43,44 추가) |

→ **공정성 원칙**: 조건(데이터량)만 변하고 나머지 전부 고정.

## 👥 담당 분배

| 담당 | 조건 (×2 모델) |
|------|------|
| **jjh** (완료) | baseline, genai_75, genai_125 |
| **yjw** (미진행) | genai_25, genai_50, genai_100 |

## 📊 현재 결과 (jjh 완료분)

### segm_AP (seed=42)

| 조건 | Mask R-CNN | Cascade Mask R-CNN |
|------|:---:|:---:|
| baseline | 8.51 | 9.34 |
| genai_75 | **11.60** | **12.32** |
| genai_125 | 11.66 | 12.25 |

### 3-seed variance (baseline / mask_rcnn 만)

| seed | segm_AP |
|:---:|:---:|
| 42 | 8.51 |
| 43 | 8.73 |
| 44 | 9.84 |
| **평균 ± 표준편차** | **9.03 ± 0.74** |

→ seed variance ~0.74 AP. 단일 seed 결과 해석 시 참고.

## 🔍 핵심 발견 (현재 데이터 기준)

1. **baseline → genai_75 큰 점프**
   - mask_rcnn: +3.09 AP (8.51 → 11.60)
   - cascade: +2.98 AP (9.34 → 12.32)

2. **genai_75 → genai_125 포화**
   - mask_rcnn: +0.06 (거의 무변화)
   - cascade: -0.07 (오히려 미세 감소)

3. **결론**: GenAI 75장/클래스에서 이미 saturate → **"효율적 GenAI 수량 = 75"** 주장 가능

### 미확보 (yjw 담당 대기)

- genai_25, genai_50, genai_100 × 2모델 = 6개
- → **완전한 스케일링 곡선**을 위해 필요

## 📁 결과 저장 위치

```
results_github/exp1_3cls/
├── baseline/Exp2_3cls/
│   ├── mask_rcnn/seed42/eval_results/results.json   ✅
│   ├── mask_rcnn/seed43/eval_results/results.json   ✅ (variance용)
│   ├── mask_rcnn/seed44/eval_results/results.json   ✅ (variance용)
│   └── cascade_mask_rcnn/seed42/eval_results/results.json ✅
├── genai_75/Exp2_3cls/{mask_rcnn, cascade_mask_rcnn}/seed42/... ✅
├── genai_125/Exp2_3cls/{mask_rcnn, cascade_mask_rcnn}/seed42/... ✅
├── genai_25/...   ❌ yjw 담당
├── genai_50/...   ❌ yjw 담당
└── genai_100/...  ❌ yjw 담당
```

## 🎨 Figure 계획

**Figure 2 — GenAI 스케일링 곡선** (논문용)

- X축: GenAI 수량 (0, 25, 50, 75, 100, 125)
- Y축: segm_AP
- Line 2개: Mask R-CNN, Cascade Mask R-CNN
- 스크립트: `docs/paper/scripts/fig2_genai_scaling.py` (아직 미작성)

**현재 그릴 수 있는 점 (jjh 완료분만)**:
```
x:  0    75   125
M: 8.51 11.60 11.66
C: 9.34 12.32 12.25
```
→ yjw 결과 추가되면 완전 6점 곡선.

## 📚 실험 계획서

- `docs/experiment_plans/exp1_3cls_v1.md` (FAIL된 v1)
- `docs/experiment_plans/exp1_3cls_v2.md` (PASS된 v2 — iter 기반 공정성)

## 🔗 관련 스크립트

- `scripts/run_exp1_3cls_jjh.sh` — jjh 실행용
- `scripts/run_exp1_3cls_yjw.sh` — yjw 실행용
- `training/train.py` — 통합 학습 CLI

---

## 📝 다음 파일

- [03_EXP2_EXPLAINED.md](03_EXP2_EXPLAINED.md): 실험 2 설명
