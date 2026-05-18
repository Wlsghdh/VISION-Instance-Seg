#!/bin/bash
# ============================================================
# [양진우] Exp1_3cls — iter 기반 공정 비교 (v2)
#   조건: genai_25 / genai_50 / genai_100 (×2 모델, seed=42 단일)
#   사용법: bash scripts/run_exp1_3cls_yjw.sh <GPU>
# ============================================================
set -e
cd /home/jjh0709/gitrepo/VISION-Instance-Seg
mkdir -p results/logs

GPU=${1:?사용법: $0 <GPU번호>}
EXPERIMENT="exp1_3cls"
CATEGORY="Exp2_3cls"
COOLDOWN=20

MAX_ITERS=20000
WARMUP_ITERS=500
EVAL_PERIOD_ITERS=500
PATIENCE=15
LR_SCHED=cosine

echo "===== [양진우] Exp1_3cls 시작 $(date) ====="
echo "GPU: $GPU"
echo ""

if [ -f docs/experiment_plans/exp1_3cls_v2_data_manifest.txt ]; then
    bash scripts/data_utils/verify_exp1_3cls.sh || { echo "❌ 데이터 검증 실패"; exit 1; }
else
    echo "⚠️  manifest 없음 — verify 스킵"
fi

run_one() {
    local cond=$1
    local model=$2
    local seed=$3
    local tag="${cond}_${model}_seed${seed}"
    local logfile="results/logs/exp1_3cls_${tag}.log"
    local result_dir="results/training/${EXPERIMENT}/${cond}/${CATEGORY}/${model}/seed${seed}/eval_results/results.json"

    if [ -f "$result_dir" ]; then
        echo "[SKIP] $tag (이미 완료)"
        return
    fi

    echo ""
    echo "===== [$(date '+%H:%M:%S')] $tag ====="
    CUDA_VISIBLE_DEVICES=$GPU conda run -n jjh python -m training.train \
        --category "$CATEGORY" --experiment "$EXPERIMENT" \
        --condition "$cond" --model "$model" --seed "$seed" \
        --max-iters "$MAX_ITERS" \
        --warmup-iters "$WARMUP_ITERS" \
        --eval-period-iters "$EVAL_PERIOD_ITERS" \
        --patience "$PATIENCE" \
        --lr-scheduler "$LR_SCHED" \
        2>&1 | tee "$logfile"

    echo "[COOLDOWN] ${COOLDOWN}초 대기..."
    sleep $COOLDOWN
}

for cond in genai_25 genai_50 genai_100; do
    for model in mask_rcnn cascade_mask_rcnn; do
        run_one "$cond" "$model" 42
    done
done

echo ""
echo "===== [양진우] Exp1_3cls 완료 $(date) ====="
