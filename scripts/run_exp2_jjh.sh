#!/bin/bash
# ============================================================
# [주진호] 실험2: Phase1(cond1~3 × 2종) + cond4_4x~6x × 7종 모델
# 사용법: bash scripts/run_exp2_jjh.sh <GPU>
# ============================================================
set -e
cd /home/jjh0709/gitrepo/VISION-Instance-Seg
mkdir -p results/logs

GPU=${1:?사용법: $0 <GPU번호>}
CATEGORY="Exp2_3cls"

echo "===== [주진호] exp2 시작 $(date) ====="
echo "GPU: $GPU, Category: $CATEGORY"

# --- Phase 1: cond1~3 (2종 모델) ---
echo ""
echo "========== Phase 1: cond1~3 =========="
PHASE1_MODELS=("mask_rcnn" "cascade_mask_rcnn")
PHASE1_CONDS=("cond1" "cond2" "cond3")
P1_TOTAL=$((${#PHASE1_MODELS[@]} * ${#PHASE1_CONDS[@]}))
P1_COUNT=0

for cond in "${PHASE1_CONDS[@]}"; do
  for model in "${PHASE1_MODELS[@]}"; do
    P1_COUNT=$((P1_COUNT + 1))
    echo ""
    echo "===== [Phase1 $P1_COUNT/$P1_TOTAL] [$(date '+%H:%M:%S')] $cond / $model ====="
    CUDA_VISIBLE_DEVICES=$GPU conda run -n jjh python -m training.train \
      --category "$CATEGORY" --experiment exp2 --condition "$cond" --model "$model" \
      --max-epochs 200 --patience 10 2>&1 | tee "results/logs/exp2_${cond}_${model}.log"
    echo "[COOLDOWN] 5분 대기..."
    sleep 300
  done
done

# --- Phase 2: cond4_4x~6x (7종 모델) ---
echo ""
echo "========== Phase 2: cond4_4x~6x =========="
PHASE2_MODELS=("mask_rcnn" "cascade_mask_rcnn" "maskdino" "mask2former" "cascade_rcnn" "solov2" "rtmdet_ins")
PHASE2_CONDS=("cond4_4x" "cond4_5x" "cond4_6x")
P2_TOTAL=$((${#PHASE2_MODELS[@]} * ${#PHASE2_CONDS[@]}))
P2_COUNT=0

for cond in "${PHASE2_CONDS[@]}"; do
  for model in "${PHASE2_MODELS[@]}"; do
    P2_COUNT=$((P2_COUNT + 1))
    echo ""
    echo "===== [Phase2 $P2_COUNT/$P2_TOTAL] [$(date '+%H:%M:%S')] $cond / $model ====="
    CUDA_VISIBLE_DEVICES=$GPU conda run -n jjh python -m training.train \
      --category "$CATEGORY" --experiment exp2 --condition "$cond" --model "$model" \
      --max-epochs 200 --patience 10 2>&1 | tee "results/logs/exp2_${cond}_${model}.log"
    echo "[COOLDOWN] 5분 대기..."
    sleep 300
  done
done

echo "===== [주진호] exp2 완료 $(date) ====="
