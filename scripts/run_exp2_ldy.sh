#!/bin/bash
# ============================================================
# [임대윤] 실험2: cond4_7x~10x × 2종 모델 (N배 탐색)
# 최적 N배가 정해지면 그 조건만 7종 모델로 재학습 (Phase3)
#
# 사용법: bash scripts/run_exp2_ldy.sh <GPU>
# ============================================================
set -e
cd /project/ahnailab/ldy1118/VISION-Instance-Seg
mkdir -p results/logs

GPU=${1:?사용법: $0 <GPU번호>}
CATEGORY="Exp2_3cls"
MODELS=("mask_rcnn" "cascade_mask_rcnn")
CONDITIONS=("cond4_7x" "cond4_8x" "cond4_9x" "cond4_10x")
COOLDOWN=30

TOTAL=$((${#MODELS[@]} * ${#CONDITIONS[@]}))
COUNT=0

echo "===== [임대윤] exp2 시작 $(date) ====="
echo "GPU: $GPU, Category: $CATEGORY"
echo "Models: ${MODELS[*]}"
echo "Conditions: ${CONDITIONS[*]}"
echo "총 학습: $TOTAL 회"

for cond in "${CONDITIONS[@]}"; do
  for model in "${MODELS[@]}"; do
    COUNT=$((COUNT + 1))
    RESULT="results/training/exp2/${cond}/${CATEGORY}/${model}/seed42/eval_results/results.json"
    if [ -f "$RESULT" ]; then
      echo "[SKIP] $cond/$model (이미 완료)"
      continue
    fi
    echo ""
    echo "===== [$COUNT/$TOTAL] [$(date '+%H:%M:%S')] $cond / $model ====="
    CUDA_VISIBLE_DEVICES=$GPU conda run -p /home/jjh0709/.conda/envs/jjh python -m training.train \
      --category "$CATEGORY" --experiment exp2 --condition "$cond" --model "$model" \
      --max-epochs 200 --patience 10 2>&1 | tee "results/logs/exp2_${cond}_${model}.log"
    echo "[COOLDOWN] ${COOLDOWN}초 대기..."
    sleep $COOLDOWN
  done
done

echo ""
echo "===== [임대윤] exp2 완료 $(date) ====="
echo ">>> 최적 N배 결정 후 Phase3(7모델) 실행 → run_exp2_phase3.sh"
