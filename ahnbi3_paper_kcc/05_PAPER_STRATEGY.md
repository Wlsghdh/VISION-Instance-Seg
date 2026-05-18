# 05. 논문 Story & 전략

## 🎯 확정 제목

> **전통 증강의 한계와 생성형 AI 증강의 시너지 — 소규모 결함 데이터셋의 공정 비교 연구**

영문 부제 (선택): *Limits of Traditional Augmentation and Synergy of Generative AI: A Fair Comparison on Small-scale Industrial Defect Datasets*

### 왜 이 제목인가?

- **"한계"**: negative finding (전통 증강 악화)을 앞에 세워 임팩트 ↑
- **"시너지"**: positive finding (결합 효과)을 병치
- **"공정 비교"**: 방법론 기여(iter 기반 프로토콜) 강조
- **"소규모 결함"**: scope 명확화로 over-claim 방어

## 🔑 Story Line (3문장)

1. **Setup**: 산업 결함 Instance Segmentation에서 데이터는 늘 부족하며, 전통 증강과 생성형 AI 증강 중 어느 쪽이 유효한지는 공정 비교가 부재하다.

2. **Observation**: 소규모(클래스당 20장) 조건에서, 전통 증강 단독은 baseline보다 -1.76 AP로 악화되고, GenAI 증강 단독은 +1.88 AP로 개선된다.

3. **Solution**: 두 증강을 결합하고 전통 증강 수량을 N=8배로 최적화하면, Cascade Mask R-CNN에서 +5.15 AP의 시너지를 달성한다 (11.26 → 16.41).

## 🏆 3가지 기여 (Contribution)

### C1. Observation (Negative Finding)
**"소규모 산업 결함 데이터에서 전통 증강 단독은 성능을 악화시킨다"**
- 근거: cond2 vs cond1 = -1.76 AP (Mask R-CNN), -0.90 AP (Cascade)
- 의의: 일반적으로 "증강은 도움이 된다"는 통념에 **반례** 제시
- 방어: 소규모 조건 명확화 (클래스당 20장, 3 클래스)

### C2. Solution (Positive Finding)
**"GenAI 증강 + 전통 증강 결합의 시너지와 최적 혼합비 발견"**
- 근거: cond4_8x cascade = 16.41 (cond1 대비 +5.15 AP)
- N 스윕 곡선에서 **N=8 정점**
- 의의: 실무 가이드라인 제시 ("GenAI 125 + 전통 1160/cls 사용")

### C3. Benchmark (Methodology)
**"iter 기반 공정성 프로토콜 + 재현 가능한 코드·결과 공개"**
- iter 기반 통일 스케줄 (vs epoch 기반 불공정)
- COCO pretrained, cosine LR, fixed val, git tag + sha256 manifest
- 의의: 후속 연구의 재현성 표준 제시

## 📖 논문 구조 (5-6쪽, KCC)

| § | 제목 | 분량 | 핵심 |
|---|------|:---:|------|
| §1 | 서론 | 0.5쪽 | motivation: 증강 선택 딜레마 + 3 contribution |
| §2 | 관련 연구 | 0.75쪽 | Instance seg / 전통 / 생성 증강 / 공정 benchmark |
| §3 | 제안 방법 | 1쪽 | 공정성 프로토콜 + 증강 조합 설계 |
| §4 | 실험 설정 | 0.75쪽 | 데이터/모델/하이퍼파라미터 |
| §5 | 결과 | 2쪽 | Table 1, Figure 1/2/3/4 + per-class |
| §6 | 토의 | 0.5쪽 | 메커니즘 + 한계 |
| §7 | 결론 | 0.25쪽 | 가이드라인 |

## 🖼 Figure/Table 계획

### Table 1 — 핵심 결과 (cond1~3 + cond4_6x/8x)
```
조건 | Mask R-CNN | Cascade MRCNN | Δ vs baseline
baseline        10.30   11.26   —
+전통 125        8.54   10.36   -1.76 / -0.90
+GenAI 125      12.18   11.78   +1.88 / +0.52
+GenAI+전통×6   13.08   15.90   +2.78 / +4.64
+GenAI+전통×8   14.13   16.41   +3.83 / +5.15  ⭐
```

### Figure 1 — cond4 N 스윕 (메인)
- 1x~10x, 2모델, N=8 정점 강조
- 스크립트: `docs/paper/scripts/fig1_cond4_curve.py` ✅

### Figure 2 — Exp1_3cls 스케일링 (GenAI 수량)
- 0/25/50/75/100/125 × 2모델
- **yjw 결과 수집 후 완성**
- 스크립트: 미작성 (paper-visualizer로 생성)

### Figure 3 — Per-class heatmap
- 조건 × 3클래스, 모델별 2패널
- 클래스별 증강 적합도 시각화

### Figure 4 — 정성 비교 (선택)
- cond1 실패 / cond4_8x 성공 케이스 4장
- detectron2 inference 필요

### Table 2 — Per-class breakdown
- 이미 생성됨: `docs/paper/tables/table2_perclass.md`

### Table 3 — cond4_8x 5모델 비교 (**학습 완료 후**)
- Mask, Cascade, MaskDINO, Mask2Former, Cascade R-CNN, SOLOv2, RTMDet-Ins
- 7모델 or 5모델 비교 표

## 🛡 리뷰어 공격 예상 + 방어

| 공격 | 방어 논리 |
|------|----------|
| "단일 seed 아니냐?" | Exp1_3cls baseline 3-seed (9.03±0.74) 제시. 경향성은 variance 대비 충분히 큼. |
| "val=test 아니냐? leakage" | Limitation에 명시. val_dev/val_test 분리 설계는 있으나 현재 코드는 통합. |
| "3 클래스만?" | Scope 명확화: "소규모 결함"이 본 연구 조건. VISION-Datasets의 해당 3 defect 사용. |
| "GenAI 품질 지표 없음?" | Supplementary에 샘플 이미지 + 수작업 라벨링 프로토콜 명시. FID는 향후 과제. |
| "Mask R-CNN 계열만?" | cond4_8x 5-7모델 비교 (§5.3 or 부록). |
| "왜 cond4_6x → cond4_8x 점프?" | 단순 선형 아님. 실험적 관찰 결과. Discussion에서 논의. |
| "왜 cond4_10x가 더 낮나?" | Overfitting 가설. 전통 증강 비율이 높을수록 원본 분포 오염. |
| "Gemini 설정 공개?" | Supplementary: 프롬프트 템플릿, 생성 파라미터 기재. |

## 📋 Discussion (§6) 준비 포인트

### 왜 전통 증강이 소규모에서 해로운가?
- 가설: 소규모 원본 → 모델이 원본 패턴에 과적합 쉬움
- 전통 증강은 같은 패턴의 **pixel-level 변형**만 제공 → 변형된 over-fitting
- 정보 다양성 측면에서 entropy 부족

### 왜 GenAI가 도움이 되는가?
- Gemini 같은 대규모 모델은 **semantic-level diversity** 제공
- 원본에 없던 질감/각도/조명 조건 도입
- 클래스의 "concept" 일반화

### 왜 결합이 시너지를 내는가?
- GenAI: distribution 확장 (semantic)
- 전통 증강: robustness 보강 (pixel invariance)
- 상호 보완: 다양한 개념 + 변형 내성 동시 획득

### 왜 N=8이 최적인가?
- N < 8: 전통 증강 부족 → robustness 부족
- N > 8: 전통 증강 과다 → 원본 분포 왜곡
- Sweet spot: 원본:GenAI:전통 ≈ 1:6:58 비율

### 한계
1. 단일 seed (variance 보완 3-seed 있음)
2. val=test 구조 (별도 test 분리 미적용)
3. 3 클래스만 (generalization 범위 제한)
4. GenAI 품질 정량 평가 없음
5. 소규모 원본 (클래스당 20장) — 다른 규모에서 관찰 상이 가능

## 🎨 글쓰기 스타일

- **논문체 (학술 한국어)**: "본 연구는 ... 관찰한다", "~임을 실증한다"
- **Over-claim 금지**: "항상", "모든", "최고" 피함
- **수치 명시**: 주장에 수치 뒷받침 (±variance 동반)
- **Paragraph 구조**: 주장 → 근거 → 의미 순

### 피해야 할 표현
- "우리 방법은 최고다" → "본 연구 방법은 비교 조건에서 우월함을 보인다"
- "상당히 좋아졌다" → "+5.15 AP 향상되었다"
- "일반적으로" → "본 실험 조건에서"

## 🗓 작성 순서 (권장)

1. **§4 실험 설정** (사실 기반, 가장 쉬움)
2. **§5 결과** (Table/Figure + 1-2문장 해석)
3. **§3 방법** (§4~§5 쓰면서 정리)
4. **§6 토의** (결과 해석)
5. **§1 서론** (맨 나중 — 결과를 알아야 motivation 확정)
6. **§2 관련연구** (병렬)
7. **§7 결론** (마지막)

## 🔗 관련 파일

- `docs/paper/PAPER_PLAN.md` (중복 참고)
- `docs/result_summary_kcc_paper.md` (더 상세한 결과 요약)
- `docs/paper/draft/01_intro.md` (서론 bullet)
- `docs/paper/draft/02_related.md` (관련연구 bullet)

---

## 📝 다음 파일

- [06_DATA_AND_MODELS.md](06_DATA_AND_MODELS.md): 데이터 + 모델 상세
