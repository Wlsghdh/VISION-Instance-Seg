# 실험2 Phase2 결과 요약 — 양진우 (cond4_1x ~ cond4_3x)

- **담당자**: 양진우 (GPU 0)
- **카테고리**: `Exp2_3cls` (Inclusoes / Dirty / impurities)
- **모델**: Mask R-CNN, Cascade Mask R-CNN (Phase2 = N배 탐색용 2모델)
- **학습 설정**: `--max-epochs 200 --patience 10`, seed=42
- **스크립트**: `scripts/run_exp2_yjw.sh` (로컬 실행 버전: `scripts/run_exp2_yjw_local.sh`)
- **원본 결과 경로**: `results/training/exp2/{cond}/Exp2_3cls/{model}/seed42/eval_results/results.json`
- **PR 사본 경로**: `results-push/exp2/{cond}/{model}/results.json`

---

## 메트릭 요약 (seed=42, 단일 시드)

| Condition | Model | segm_AP | segm_AP50 | bbox_AP | bbox_AP50 |
|-----------|-------|:-------:|:---------:|:-------:|:---------:|
| cond4_1x | mask_rcnn         | 13.39 | 28.90 | 15.69 | 29.12 |
| cond4_1x | cascade_mask_rcnn | 13.83 | 28.79 | 16.62 | 30.12 |
| cond4_2x | mask_rcnn         | 12.63 | 28.18 | 15.05 | 28.97 |
| **cond4_2x** | **cascade_mask_rcnn** | **15.69** ⭐ | **33.34** | **17.00** | **30.55** |
| cond4_3x | mask_rcnn         | 14.30 | 30.95 | 14.96 | 28.19 |
| cond4_3x | cascade_mask_rcnn | 14.29 | 29.66 | 17.33 ⭐ | 31.65 |

> 세부 수치(AP75/APs/APm/APl, 클래스별 AP)는 각 `results.json` 원본 참고.

---

## 클래스별 segm_AP (참고)

| Condition | Model | Inclusoes | Dirty | impurities |
|-----------|-------|:---------:|:-----:|:----------:|
| cond4_2x | cascade_mask_rcnn | 16.93 | 12.25 | 17.89 |
| cond4_3x | cascade_mask_rcnn | 14.11 | 11.41 | 17.35 |

---

## 관찰

1. **본 구간(1x~3x) 최적 조합**: `cond4_2x` + `cascade_mask_rcnn` (segm_AP **15.69**)
2. **단조 증가가 아님**: 1x → 2x 상승, 2x → 3x 하락 또는 정체 → 전통증강의 정보 다양성 포화 가능성
3. **Cascade가 전반적으로 우세**: 6쌍 중 5쌍에서 Cascade Mask R-CNN ≥ Mask R-CNN

---

## 다음 단계

- 정종현(cond4_4x~6x), 임대윤(cond4_7x~10x) 결과 합산 후 전체 best N 확정
- 그 N으로 **Phase3 (7모델 전체 비교)** 실행 → `scripts/run_exp2_phase3.sh`
