#!/bin/bash
# ============================================================
# [양진우] 실험2 Phase2: cond4_1x~4x × mask_rcnn, maskdino, mask2former
# 3개 defect를 하나의 3클래스 모델로 학습
# 사용법: bash scripts/run_exp2_yjw.sh <GPU>
# ============================================================
set -e
cd /home/jjh0709/gitrepo/VISION-Instance-Seg
mkdir -p results/logs

GPU=${1:?사용법: $0 <GPU번호>}
CATEGORY="Exp2_3cls"  # Inclusoes + Dirty + impurities (3클래스)
MODELS=("mask_rcnn" "maskdino" "mask2former")
CONDITIONS=("cond4_1x" "cond4_2x" "cond4_3x" "cond4_4x")

TOTAL=$((${#MODELS[@]} * ${#CONDITIONS[@]}))
COUNT=0

echo "===== [양진우] exp2 시작 $(date) ====="
echo "GPU: $GPU, Category: $CATEGORY"
echo "Models: ${MODELS[*]}"
echo "Conditions: ${CONDITIONS[*]}"
echo "총 학습: $TOTAL 회"

for cond in "${CONDITIONS[@]}"; do
  for model in "${MODELS[@]}"; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "===== [$COUNT/$TOTAL] [$(date '+%H:%M:%S')] $cond / $model ====="
    CUDA_VISIBLE_DEVICES=$GPU conda run -n jjh python -m training.train \
      --category "$CATEGORY" --experiment exp2 --condition "$cond" --model "$model" \
      --max-epochs 200 --patience 10 2>&1 | tee "results/logs/exp2_${cond}_${model}.log"
    echo "[COOLDOWN] 5분 대기..."
    sleep 300
  done
done

echo "===== [양진우] exp2 완료 $(date) ====="
