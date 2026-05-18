#!/bin/bash
# ============================================================
# [주진호] 실험2
#   Phase1: cond1~3 × 2종 모델 (완료)
#   Phase2: cond4_4x~6x × 2종 모델 (완료)
#   Phase3: 전체 결과 모인 후 최적 N배 × 7종 모델 (대기)
#
# 사용법: bash scripts/run_exp2_jjh.sh <GPU>
# ============================================================
set -e
cd /home/jjh0709/gitrepo/VISION-Instance-Seg
mkdir -p results/logs

GPU=${1:?사용법: $0 <GPU번호>}
CATEGORY="Exp2_3cls"
COOLDOWN=30

echo "===== [주진호] exp2 시작 $(date) ====="
echo "GPU: $GPU, Category: $CATEGORY"

# --- Phase 1: cond1~3 (2종 모델) ---
echo ""
echo "========== Phase 1: cond1~3 × 2모델 =========="
PHASE1_MODELS=("mask_rcnn" "cascade_mask_rcnn")
PHASE1_CONDS=("cond1" "cond2" "cond3")

for cond in "${PHASE1_CONDS[@]}"; do
  for model in "${PHASE1_MODELS[@]}"; do
    RESULT="results/training/exp2/${cond}/${CATEGORY}/${model}/seed42/eval_results/results.json"
    if [ -f "$RESULT" ]; then
      echo "[SKIP] $cond/$model (이미 완료)"
      continue
    fi
    echo ""
    echo "===== [$(date '+%H:%M:%S')] $cond / $model ====="
    CUDA_VISIBLE_DEVICES=$GPU conda run -n jjh python -m training.train \
      --category "$CATEGORY" --experiment exp2 --condition "$cond" --model "$model" \
      --max-epochs 200 --patience 10 2>&1 | tee "results/logs/exp2_${cond}_${model}.log"
    echo "[COOLDOWN] ${COOLDOWN}초 대기..."
    sleep $COOLDOWN
  done
done

# --- Phase 2: cond4_4x~6x (2종 모델만 — 최적 N배 탐색) ---
echo ""
echo "========== Phase 2: cond4_4x~6x × 2모델 (N배 탐색) =========="
PHASE2_MODELS=("mask_rcnn" "cascade_mask_rcnn")
PHASE2_CONDS=("cond4_4x" "cond4_5x" "cond4_6x")

for cond in "${PHASE2_CONDS[@]}"; do
  for model in "${PHASE2_MODELS[@]}"; do
    RESULT="results/training/exp2/${cond}/${CATEGORY}/${model}/seed42/eval_results/results.json"
    if [ -f "$RESULT" ]; then
      echo "[SKIP] $cond/$model (이미 완료)"
      continue
    fi
    echo ""
    echo "===== [$(date '+%H:%M:%S')] $cond / $model ====="
    CUDA_VISIBLE_DEVICES=$GPU conda run -n jjh python -m training.train \
      --category "$CATEGORY" --experiment exp2 --condition "$cond" --model "$model" \
      --max-epochs 200 --patience 10 2>&1 | tee "results/logs/exp2_${cond}_${model}.log"
    echo "[COOLDOWN] ${COOLDOWN}초 대기..."
    sleep $COOLDOWN
  done
done

echo ""
echo "===== [주진호] exp2 Phase1+2 완료 $(date) ====="
echo ""
echo ">>> Phase3(최적 N배 × 7모델)은 전체 2모델 결과 취합 후 실행"
echo ">>> run_exp2_phase3.sh 참조"
