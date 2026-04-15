# Exp1_3cls — GenAI 증강 수에 따른 성능 변화 (3 클래스 축소판)

**버전**: v1 (검토 대기)
**작성일**: 2026-04-15
**작성자**: jjh (Claude Code 보조)

---

## 1. 배경 및 목적

기존 Exp1(Unified 14 클래스)에서 baseline이 모든 GenAI 증강 조건보다 overall AP가 높게 나왔음. 그러나 이는 **쉬운 클래스(PistonMiss 100, Gap 65 등)가 평균을 끌어올려** 생긴 착시일 수 있음. 특히 GenAI 증강이 유효했던 클래스(Dirty / Inclusoes / impurities / Porosity 등)와 무관한 결함이 많이 섞여 있었음.

→ **Exp2_3cls와 동일한 3개 defect(Dirty, Inclusoes, impurities)에만 한정**하여 GenAI 증강 수 스윕 실험(exp1)을 재실행. Exp2_3cls의 baseline(cond1)과 직접 비교 가능한 결과 확보.

## 2. 실험 설정

### 2.1 데이터
- **카테고리**: Exp2_3cls (Dirty + Inclusoes + impurities)
- **원본**: 클래스당 20장 (총 60장)
- **Val**: `results/merged_datasets/_exp2_3cls_val/` 82장 / 113 annotations (전 조건 고정)
- **Train**: 조건별로 baseline + GenAI N장/cls

### 2.2 조건 (6개)

| 조건 | 원본 | +GenAI/cls | 총 train |
|------|:---:|:---:|:---:|
| baseline | 60 | 0 | 60 |
| genai_25 | 60 | 25×3 | 135 |
| genai_50 | 60 | 50×3 | 210 |
| genai_75 | 60 | 75×3 | 285 |
| genai_100 | 60 | 100×3 | 360 |
| genai_125 | 60 | 125×3 | 435 |

### 2.3 모델 (2개)
- Mask R-CNN (ResNet-50 FPN, COCO pretrained)
- Cascade Mask R-CNN (ResNet-50 FPN, COCO pretrained)

### 2.4 하이퍼파라미터 (전 조건 동일 / iter 기반 통일)

| 항목 | 값 | 근거 |
|------|:---:|------|
| batch_size | **12** (고정) | 공정 비교 원칙. A100 80GB에서 bs=16도 가능하나 12는 검증됨 |
| lr | **0.0015** (고정) | Linear Scaling Rule: `0.02 × (12/16) = 0.015`, fine-tune 1/10 |
| optimizer | SGD (momentum=0.9, wd=1e-4) | Detectron2/mmdet 표준 |
| max_iters | **20,000** | 상한. early-stop이 실제 종료 지점 결정 |
| warmup | **500 iter (linear)** | 전체의 2.5%, Detectron2 표준 |
| eval_period | **500 iter** | 총 40회 평가 지점 |
| early_stop_patience | **15 evals = 7,500 iter** | 전체의 37.5% 동안 미개선 시 종료 |
| lr_decay | **Step at (14000, 18000) = 70%/90%** | lr → lr/10 → lr/100 |
| seed | 42 | 단일 seed (variance 한계 명시) |
| input_size | (640~800) × 1333 max | 기존 설정 유지 |

### 2.5 평가
- **주 지표**: segm_AP (COCO mAP, IoU 0.50:0.05:0.95)
- **보조**: segm_AP50, AP75, per-class AP (Dirty/Inclusoes/impurities), bbox_AP
- val = test (분리 없음). 모델 선택과 보고 수치가 같은 셋 — leakage 한계 명시.

## 3. 담당자 분배

| 담당 | 조건 (×2 모델) | 총 학습 수 | 데이터량 합 |
|------|-----|:---:|:---:|
| **jjh** | baseline / genai_75 / genai_125 | 6회 | 780 |
| **yjw** | genai_25 / genai_50 / genai_100 | 6회 | 705 |

- ldy는 본 실험에서 제외 (다른 작업 병행 중)

## 4. 실행 계획

1. `training/config.py` 수정:
   - 새 실험 `exp1_3cls` 등록
   - `DEFAULT_HYPERPARAMS` 에 iter 기반 필드 추가
2. `training/train.py` 에 `--max-iters` CLI 옵션 추가 (epoch 대안)
3. `training/adapters/{detectron2,mmdet}_adapter.py` iter 기반 스케줄 분기 구현
4. `scripts/run_exp1_3cls_jjh.sh`, `scripts/run_exp1_3cls_yjw.sh` 작성
5. 로컬 smoke test (1개 조건 ×1 모델, max_iters=500)
6. 본 실행

## 5. 재현성 보장

- **commit hash**: 학습 직전 `git rev-parse HEAD` 를 결과 경로에 저장
- **데이터 해시**: 팀원들이 `md5sum data_augmented/{Inclusoes,Dirty,impurities}/gen_ai/annotations.json` 결과 공유해서 동일성 확인
- **conda env**: `/home/jjh0709/.conda/envs/jjh` 공용 경로 사용
- **GPU**: 동일 A100 80GB 사용

## 6. 예상 시간

| 조건 | 예상 종료 iter | 예상 시간 (A100, bs=12) |
|------|:---:|:---:|
| baseline | ~5,000 | ~1.0h |
| genai_25 | ~7,000 | ~1.5h |
| genai_50 | ~8,000 | ~2.0h |
| genai_75 | ~10,000 | ~2.5h |
| genai_100 | ~12,000 | ~3.0h |
| genai_125 | ~14,000 | ~3.5h |

- jjh (6회): ~14h
- yjw (6회): ~13h

## 7. 알려진 한계

1. **단일 seed**: 통계적 variance 측정 불가. 향후 seed 43/44 확장 여지.
2. **val = test**: 모델 선택과 보고 동일 셋. 논문용이면 분리 권장.
3. **클래스 3개만**: 결과의 일반화 범위 제한.
4. **팀원 간 데이터 sync**: md5 확인 외 자동화 없음.

## 8. 성공 판정

- 12회 학습 모두 정상 완료 + eval 결과 JSON 생성
- Exp2_3cls의 cond1~cond3 결과와 baseline/genai_125 비교 가능
- PASS 기준: `experiment-plan-reviewer` agent 통과 + 본 실행 12건 중 11건 이상 성공
