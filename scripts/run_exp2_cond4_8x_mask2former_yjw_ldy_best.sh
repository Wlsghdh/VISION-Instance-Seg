#!/bin/bash
# ============================================================
# [양진우] Exp2 cond4_8x Mask2Former — ldy1118 best 설정 재현
#
#   ldy1118 repo의 m2f_lifeai_best.sh 설정 + results_github config.yaml 값 기반
#   peak segm_AP=1.17 달성한 설정 그대로 복제
#
#   데이터: Exp2_3cls cond4_8x
#     = casting_Inclusoes + console_Dirty + wood_impurities 각 원본 20장
#     + GenAI 125/class + 전통 증강 1160/class = 총 3915장
#
#   핵심 설정 (ldy best):
#     - Pretrained: Mask2Former R50 COCO instance seg full ckpt
#     - NUM_OBJECT_QUERIES: 50 (env var)
#     - BACKBONE.FREEZE_AT: 5 (env var, backbone 전체 freeze)
#     - BASE_LR: 2e-4
#     - IMS_PER_BATCH: 2
#     - MAX_ITER: 195,800 (≈ 200 epoch)
#     - WARMUP_ITERS: 4000
#     - LR_SCHEDULER: WarmupCosineLR
#     - CLIP_VALUE: 0.01 (norm)
#     - patience: 20 evals (eval_period_iters=1000 기준)
#
#   사용법: bash scripts/run_exp2_cond4_8x_mask2former_yjw_ldy_best.sh <GPU>
# ============================================================
set -e
cd /project/ahnailab/yjw0619/seg/seg/VISION-Instance-Seg
mkdir -p results/logs

GPU=${1:?사용법: $0 <GPU번호>}
PYTHON=/home/jjh0709/.conda/envs/jjh/bin/python
SUFFIX=ldy_best
OUT_DIR="results/training/exp2/cond4_8x/Exp2_3cls/mask2former/seed42_${SUFFIX}"
LOG="results/logs/exp2_cond4_8x_mask2former_${SUFFIX}.log"

# 이전 run 제거 (동일 suffix 재학습 시)
rm -rf "${OUT_DIR}"

# ── ldy1118 best 설정 env var ──
# q50 patched pkl: 원본 ckpt(q100)에서 class_embed/query_embed/query_feat 5개 제거 버전
export MASK2FORMER_WEIGHTS=/project/ahnailab/yjw0619/seg/seg/VISION-Instance-Seg/checkpoints/mask2former/model_final_3c8ec9_q50.pkl
export MASK2FORMER_FREEZE_BACKBONE=5
export MASK2FORMER_NUM_QUERIES=50

echo "===== [yjw] Mask2Former cond4_8x — ldy best 재현 $(date) ====="
echo "GPU: $GPU"
echo "FREEZE_BACKBONE=${MASK2FORMER_FREEZE_BACKBONE}, NUM_QUERIES=${MASK2FORMER_NUM_QUERIES}"

CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m training.train \
    --category Exp2_3cls --experiment exp2 --condition cond4_8x \
    --model mask2former --seed 42 --tag "${SUFFIX}" \
    --max-iters 195800 --warmup-iters 4000 \
    --eval-period-iters 1000 --checkpoint-period-iters 2000 \
    --patience 20 --lr 2e-4 --batch-size 2 --lr-scheduler cosine \
    2>&1 | tee "${LOG}"

echo "===== [yjw] Mask2Former cond4_8x ldy_best 완료 $(date) ====="
