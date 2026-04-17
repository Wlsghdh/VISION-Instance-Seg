# 04. 현재까지 확보된 모든 수치

**데이터 기준**: 2026-04-17 18:00
**출처**: `results/evaluation/results.json` + `results_github/`
**평가**: Exp2_3cls val 82장, COCO segm AP (IoU 0.50:0.05:0.95)

---

## 1. Exp1_3cls (jjh 완료분 — iter 기반 공정 학습)

### 1.1 segm_AP (seed=42)

| 조건 | Mask R-CNN | Cascade Mask R-CNN |
|------|:---:|:---:|
| baseline | 8.51 | 9.34 |
| genai_75 | **11.60** | **12.32** |
| genai_125 | 11.66 | 12.25 |

### 1.2 Per-class (Cascade Mask R-CNN, cond4_6x 기준)

| 클래스 | AP |
|--------|:---:|
| Dirty | 14.60 |
| Inclusoes | 15.43 |
| impurities | 17.67 |

### 1.3 3-seed variance (baseline / mask_rcnn)

| seed | segm_AP |
|:---:|:---:|
| 42 | 8.51 |
| 43 | 8.73 |
| 44 | 9.84 |
| **평균** | **9.03** |
| **표준편차** | **0.74** |

→ 단일 seed 결과는 ±0.74 AP 불확실성 있음.

### 1.4 yjw 미진행 (필요)

- genai_25 / mask_rcnn, cascade
- genai_50 / mask_rcnn, cascade
- genai_100 / mask_rcnn, cascade

---

## 2. Exp2 (전체 팀)

### 2.1 Phase 1 — cond1~cond3 (전통 vs GenAI)

| 조건 | Mask R-CNN | Cascade Mask R-CNN |
|------|:---:|:---:|
| cond1 (원본 20) | 10.30 | 11.26 |
| cond2 (+전통 125) | 8.54 | 10.36 |
| cond3 (+GenAI 125) | 12.18 | 11.78 |

**비교**:
- cond2 - cond1: **-1.76 / -0.90** (전통 증강 단독 악화)
- cond3 - cond1: **+1.88 / +0.52** (GenAI 증강 단독 개선)
- cond3 - cond2: **+3.64 / +1.42** (같은 125장이라도 GenAI > 전통)

### 2.2 Phase 2 — cond4_Nx 스윕 (mask_rcnn / cascade_mask_rcnn)

| N | Mask R-CNN | Cascade MRCNN |
|:---:|:---:|:---:|
| 1x | 13.39 | 13.83 |
| 2x | 12.63 | 15.69 |
| 3x | 14.30 | 14.29 |
| 4x | 12.18 | 12.39 |
| 5x | 12.23 | 12.09 |
| 6x | 13.08 | 15.90 |
| 7x | 13.28 | 14.41 |
| **8x** ⭐ | **14.13** | **16.41** |
| 9x | 13.45 | 15.07 |
| 10x | 13.52 | 14.85 |

**통계**:
- Cascade MRCNN 평균 (1~10x): 14.46
- Cascade MRCNN 최고: **cond4_8x = 16.41 ⭐**
- Cascade MRCNN 최저: cond4_5x = 12.09
- 표준편차: ~1.36

### 2.3 cond4_8x 5모델 비교 (논문 Figure/Table 예정)

| 모델 | 프레임워크 | segm_AP | 상태 |
|------|:---:|:---:|:---:|
| Mask R-CNN | detectron2 | **14.13** | ✅ 완료 |
| Cascade Mask R-CNN | detectron2 | **16.41** | ✅ 완료 |
| MaskDINO | detectron2 | ? | ❌ 미진행 |
| Mask2Former | detectron2 | ? | ❌ 미진행 |
| Cascade R-CNN | mmdet | ? | 🟢 학습 중 (usw, epoch 25/200) |
| SOLOv2 | mmdet | ? | 🟢 학습 중 (ahnbi3 tmux) |
| RTMDet-Ins | mmdet | ? | ❌ 미진행 |

---

## 3. Per-class breakdown (Exp2)

### Mask R-CNN

| 조건 | Dirty | Inclusoes | impurities | mean |
|------|:---:|:---:|:---:|:---:|
| cond1 | 10.24 | 6.58 | 14.06 | 10.30 |
| cond2 | 4.21 | 9.60 | 11.81 | 8.54 |
| cond3 | 11.12 | 9.27 | 16.15 | 12.18 |
| cond4_4x | 11.20 | 10.56 | 14.78 | 12.18 |
| cond4_5x | 12.47 | 10.37 | 13.84 | 12.23 |
| cond4_6x | 6.70 | 12.34 | 20.21 | 13.08 |

### Cascade Mask R-CNN

| 조건 | Dirty | Inclusoes | impurities | mean |
|------|:---:|:---:|:---:|:---:|
| cond1 | 12.10 | 5.40 | 16.27 | 11.26 |
| cond2 | 7.57 | 9.08 | 14.44 | 10.36 |
| cond3 | 8.77 | 12.01 | 14.58 | 11.78 |
| cond4_4x | 9.31 | 14.45 | 13.42 | 12.39 |
| cond4_5x | 9.06 | 14.06 | 13.14 | 12.09 |
| cond4_6x | **14.60** | 15.43 | **17.67** | 15.90 |

**관찰**:
- **impurities** 가 전 조건 가장 높음 (가장 쉬운 결함)
- **Dirty** 가 cond2(전통 단독)에서 크게 무너짐 (Mask R-CNN: 10.24 → 4.21)
- **Inclusoes** 는 GenAI 조건에서 크게 상승 (5.40 → 15.43)

---

## 4. 주요 비교 수치 (논문 메시지용)

| 비교 | 수치 | 의미 |
|------|:---:|------|
| cond2 vs cond1 (Mask R-CNN) | **-1.76 AP** | 전통 증강 단독 악화 |
| cond3 vs cond1 (Mask R-CNN) | **+1.88 AP** | GenAI 증강 개선 |
| **cond4_8x vs cond1 (Cascade)** | **+5.15 AP** | **결합 시너지 (메인)** |
| cond4_8x vs cond4_10x (Cascade) | +1.56 AP | 8x가 10x보다 우수 (overfitting?) |
| Cascade 평균 vs Mask R-CNN 평균 (cond4) | +1.52 AP | Cascade 일관된 우위 |

---

## 5. 학습 시간 참고 (usw A100 기준)

| 모델/조건 | 학습 시간 | 비고 |
|-----------|:---:|------|
| mask_rcnn baseline | ~2h | iter 기반 |
| cascade_mask_rcnn baseline | ~3h | iter 기반 |
| cascade_rcnn cond4_8x | ~15-20h (예상) | epoch 기반, patience=10 |
| solov2 cond4_8x | ~5-7h (예상) | epoch 기반 |

→ 재학습 시 참고.

---

## 6. 데이터 정확 경로

```
results_github/
├── evaluation/results.json         ← 집계 (전체 결과)
├── exp1_3cls/
│   ├── baseline/Exp2_3cls/mask_rcnn/seed{42,43,44}/eval_results/results.json
│   ├── baseline/Exp2_3cls/cascade_mask_rcnn/seed42/eval_results/results.json
│   ├── genai_75/.../seed42/...
│   └── genai_125/.../seed42/...
└── exp2/
    ├── cond1/Exp2_3cls/{mask,cascade}/seed42/eval_results/results.json
    ├── cond2/...
    ├── cond3/...
    └── cond4_{1x..10x}/Exp2_3cls/{mask,cascade}/seed42/...
```

## 7. Python으로 데이터 로드하기

```python
import json
d = json.load(open('results/evaluation/results.json'))
# 또는
d = json.load(open('results_github/evaluation/results.json'))

exp2 = [r for r in d if r.get('experiment')=='exp2' and r.get('category')=='Exp2_3cls']
for r in exp2:
    ev = r.get('eval') or {}
    print(r['condition'], r['model'], ev.get('segm_AP'))
```

---

## 📝 다음 파일

- [05_PAPER_STRATEGY.md](05_PAPER_STRATEGY.md): 논문 story + 제목 + 구조
