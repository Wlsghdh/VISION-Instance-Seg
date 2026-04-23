#!/bin/bash
# ============================================================
# [양진우] Exp2 cond1 (baseline) Mask2Former — ldy best recipe
#
#   조건: cond1 (baseline) = 원본 20/cls × 3 cls = 60 장 (증강 없음)
#   모델: Mask2Former R50 + ldy1118 m2f_lifeai_best 설정
#
#   iter 조정 (cond4_8x 대비):
#     cond4_8x: 3,915 imgs × 200 ep ÷ batch 2 = 391,500 iters
#     cond1:        60 imgs × 200 ep ÷ batch 2 =   6,000 iters
#   → max-iters 6000, warmup 200, eval 30 (= 1 epoch), ckpt 100
#``
#   설정 (동일):
#     - q50 patched pkl
#     - BACKBONE.FREEZE_AT=5, NUM_OBJECT_QUERIES=50
#     - lr=2e-4, batch=2, cosine, CLIP_VALUE=0.01
#     - patience 20 evals (= 20 epoch no-improvement 시 stop)
#
#   사용법: bash scripts/run_exp2_cond1_mask2former_yjw_ldy_best.sh <GPU>
# ============================================================
set -e
cd /project/ahnailab/yjw0619/seg/seg/VISION-Instance-Seg
mkdir -p results/logs

GPU=${1:?사용법: $0 <GPU번호>}
PYTHON=/home/jjh0709/.conda/envs/jjh/bin/python
SUFFIX=ldy_best
LOG="results/logs/exp2_cond1_mask2former_${SUFFIX}.log"

# ── ldy1118 best 설정 env var (q50 patched ckpt) ──
export MASK2FORMER_WEIGHTS=/project/ahnailab/yjw0619/seg/seg/VISION-Instance-Seg/checkpoints/mask2former/model_final_3c8ec9_q50.pkl
export MASK2FORMER_FREEZE_BACKBONE=5
export MASK2FORMER_NUM_QUERIES=50

echo "===== [yjw] Mask2Former cond1 (baseline) — ldy best $(date) ====="
echo "GPU: $GPU"

CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m training.train \
    --category Exp2_3cls --experiment exp2 --condition cond1 \
    --model mask2former --seed 42 --tag "${SUFFIX}" \
    --max-iters 6000 --warmup-iters 200 \
    --eval-period-iters 30 --checkpoint-period-iters 100 \
    --patience 20 --lr 2e-4 --batch-size 2 --lr-scheduler cosine \
    2>&1 | tee "${LOG}"

echo "===== [yjw] cond1 ldy_best 완료 $(date) ====="
