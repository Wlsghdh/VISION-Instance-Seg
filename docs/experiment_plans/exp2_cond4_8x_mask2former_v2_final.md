# Exp2 cond4_8x — Mask2Former 학습 결과

**일시**: 2026-04-19 ~ 04-20
**담당**: 양진우
**대상**: Exp2 / cond4_8x / mask2former / seed42
**서버**: usw GPU 0 (A100 80GB)

---

## 1. 결과

**segm_AP = 0.095** (55 epoch, early stop, 10.4 시간 소요)

| 메트릭 | 값 |
|---|---|
| segm_AP | **0.095** |
| segm_AP50 | 0.235 |
| segm_AP75 | 0.025 |
| segm_APs (small) | 1.173 |
| segm_APm (medium) | 0.336 |
| segm_APl (large) | 0.000 |
| bbox_AP | 0.000 |

### 클래스별

| 클래스 | segm_AP |
|---|---|
| Inclusoes | 0.001 |
| Dirty | 0.097 |
| impurities | 0.187 |

## 2. 학습 과정

### 2.1 시행 이력

| 시도 | 주요 변경 | 결과 |
|---|---|---:|
| v1-1 | MaskDINO mapper 재사용 + AMP + max_size=800 | AP=0.0 |
| v1-2 | 공식 COCOInstanceNewBaselineDatasetMapper + JIT patch | AP=0.0 |
| v1-3 | + OVERSAMPLE_RATIO=20, THRESHOLD 완화 | AP=0.0 (gradient 발산) |
| **v2 (성공)** | **COCO pretrained full ckpt + ldy recipe** | **AP=0.095** |

### 2.2 v2 최종 설정

**pretrained weight**
- Mask2Former R50 COCO instance seg full checkpoint (`model_final_3c8ec9.pkl`, 168MB)
- 경로: `checkpoints/mask2former/model_final_3c8ec9.pkl`

**하이퍼파라미터**
| 항목 | 값 |
|---|---|
| batch_size | 4 |
| lr | 5e-5 |
| lr_scheduler | WarmupCosineLR |
| warmup_iters | 4000 |
| AMP | False (FP32) |
| max_size_train | 1333 |
| min_size_train | (640, 672, 704, 736, 768, 800) |
| CLIP_TYPE | norm |
| CLIP_VALUE | 0.01 |
| max_epochs | 200 |
| patience | 10 evals |
| eval_period | 5 epoch |
| seed | 42 |

**코드 수정**
1. `training/config.py` — `MODELS["mask2former"]` 의 weight 경로 및 hyperparams 업데이트
2. `training/adapters/detectron2_adapter.py` — `_setup_mask2former()`:
   - JIT monkey-patch (`batch_dice_loss_jit` → non-JIT 함수로 교체, PyTorch 호환성)
   - 공식 `COCOInstanceNewBaselineDatasetMapper` 사용
   - `CLIP_VALUE=0.01, CLIP_TYPE=norm` 강제
   - AMP / scheduler / input_size 는 hyperparams로 일원 제어

### 2.3 Eval 추이

총 12번 eval 중 마지막만 비제로.

| Epoch | segm_AP |
|---|---|
| 5 ~ 50 (11 evals) | 0.000 |
| **55 (last)** | **0.095** |

## 3. Sanity check (본 학습 전)

3 epoch 짧은 run 으로 레시피 검증:
- `segm_AP = 0.114`
- 에러/OOM 없음
- 이 결과로 본 학습 진행 결정

## 4. 왜 성능이 낮게 나왔나

CNN 기반 모델 대비 현저히 낮음:

| 모델 | segm_AP |
|---|---:|
| Cascade Mask R-CNN | 16.41 |
| Mask R-CNN | 14.13 |
| RTMDet-Ins | 4.10 |
| **Mask2Former** | **0.095** |

**원인을 간단히 말하면**: 우리 결함 데이터는 객체(mask)가 너무 작아서 (이미지의 약 0.02%), Mask2Former가 이미지 전역에서 점을 뿌려 학습하는 방식이 작동하기 어려움. Mask R-CNN 같은 CNN 모델은 객체 영역을 먼저 찾고 그 안에서만 학습해서 작은 객체에도 잘 맞음.

→ **Transformer 기반 모델은 이런 작은 결함 데이터에 잘 맞지 않음.**

## 5. 파일 경로

| 항목 | 경로 |
|---|---|
| 결과 JSON | `results/training/exp2/cond4_8x/Exp2_3cls/mask2former/seed42/eval_results/results.json` |
| 학습 로그 | `results/logs/exp2_cond4_8x_mask2former.log` |
| 학습 스크립트 | `scripts/run_exp2_cond4_8x_mask2former_yjw.sh` |
| 저장된 config | `results/training/exp2/cond4_8x/Exp2_3cls/mask2former/seed42/config.yaml` |

## 6. 실행 명령 (재현용)

```bash
cd /project/ahnailab/yjw0619/seg/seg/VISION-Instance-Seg
bash scripts/run_exp2_cond4_8x_mask2former_yjw.sh 0
```
