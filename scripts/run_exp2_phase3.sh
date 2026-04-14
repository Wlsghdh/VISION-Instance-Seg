#!/bin/bash
# ============================================================
# [공용] 실험2 Phase3: 최적 N배 × 7종 모델 비교
#
# Phase1+2에서 2모델(mask_rcnn, cascade_mask_rcnn)로 cond4_1x~10x를
# 돌린 뒤, 가장 성능 좋은 N배를 골라서 7종 모델 전부로 비교.
#
# 사용법: bash scripts/run_exp2_phase3.sh <GPU> <BEST_COND>
# 예시:   bash scripts/run_exp2_phase3.sh 1 cond4_5x
# ============================================================
set -e
cd /home/jjh0709/gitrepo/VISION-Instance-Seg
mkdir -p results/logs

GPU=${1:?사용법: $0 <GPU번호> <최적조건, 예: cond4_5x>}
BEST_COND=${2:?사용법: $0 <GPU번호> <최적조건, 예: cond4_5x>}
CATEGORY="Exp2_3cls"
MODELS=("mask_rcnn" "cascade_mask_rcnn" "maskdino" "mask2former" "cascade_rcnn" "solov2" "rtmdet_ins")
COOLDOWN=30

TOTAL=${#MODELS[@]}
COUNT=0

echo "===== exp2 Phase3: ${BEST_COND} × 7모델 비교 $(date) ====="
echo "GPU: $GPU, Condition: $BEST_COND"

for model in "${MODELS[@]}"; do
  COUNT=$((COUNT + 1))
  RESULT="results/training/exp2/${BEST_COND}/${CATEGORY}/${model}/seed42/eval_results/results.json"
  if [ -f "$RESULT" ]; then
    echo "[SKIP] ${BEST_COND}/$model (이미 완료)"
    continue
  fi
  echo ""
  echo "===== [$COUNT/$TOTAL] [$(date '+%H:%M:%S')] $BEST_COND / $model ====="
  CUDA_VISIBLE_DEVICES=$GPU conda run -n jjh python -m training.train \
    --category "$CATEGORY" --experiment exp2 --condition "$BEST_COND" --model "$model" \
    --max-epochs 200 --patience 10 2>&1 | tee "results/logs/exp2_${BEST_COND}_${model}.log"
  echo "[COOLDOWN] ${COOLDOWN}초 대기..."
  sleep $COOLDOWN
done

echo ""
echo "===== Phase3 완료 $(date) ====="
