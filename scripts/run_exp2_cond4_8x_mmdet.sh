#!/bin/bash
# ============================================================
# Exp2 cond4_8x — mmdet 3종 모델 학습
#   조건: 원본 20 + GenAI 125 + 전통 1,160 = 1,305/cls × 3cls = 3,915장
#   모델: cascade_rcnn, solov2, rtmdet_ins
#   학습 방식: epoch 기반 (기존 cond4_4x/5x/6x mmdet 결과와 직접 비교 가능하도록)
#
# 사용법: bash scripts/run_exp2_cond4_8x_mmdet.sh <GPU>
# ============================================================
set -e
cd /home/jjh0709/gitrepo/VISION-Instance-Seg
mkdir -p results/logs

GPU=${1:?사용법: $0 <GPU번호>}
EXPERIMENT="exp2"
CATEGORY="Exp2_3cls"
CONDITION="cond4_8x"
COOLDOWN=20

# mmdet 3종 (epoch 기반, 기존 cond4_4x~6x mmdet과 동일 설정)
MODELS=("cascade_rcnn" "solov2" "rtmdet_ins")

echo "===== cond4_8x mmdet 3종 학습 시작 $(date) ====="
echo "GPU: $GPU"
echo "Condition: $CONDITION (원본 20 + GenAI 125 + 전통 1,160 = 1,305/cls)"
echo "Models: ${MODELS[*]}"
echo ""

run_one() {
    local model=$1
    local logfile="results/logs/exp2_${CONDITION}_${model}.log"
    local result_dir="results/training/${EXPERIMENT}/${CONDITION}/${CATEGORY}/${model}/seed42/eval_results/results.json"

    if [ -f "$result_dir" ]; then
        echo "[SKIP] $model (이미 완료)"
        return
    fi

    echo ""
    echo "===== [$(date '+%H:%M:%S')] $CONDITION / $model ====="
    CUDA_VISIBLE_DEVICES=$GPU conda run -n jjh python -m training.train \
        --category "$CATEGORY" --experiment "$EXPERIMENT" \
        --condition "$CONDITION" --model "$model" --seed 42 \
        --max-epochs 200 --patience 10 \
        2>&1 | tee "$logfile"

    echo "[COOLDOWN] ${COOLDOWN}초 대기..."
    sleep $COOLDOWN
}

for model in "${MODELS[@]}"; do
    run_one "$model"
done

echo ""
echo "===== cond4_8x mmdet 학습 완료 $(date) ====="
echo ">>> sync: python scripts/sync_results_to_github.py --exp exp2"
