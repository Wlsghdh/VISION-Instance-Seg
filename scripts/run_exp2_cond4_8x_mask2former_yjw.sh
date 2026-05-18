#!/bin/bash
# ============================================================
# [양진우] Exp2 cond4_8x — Mask2Former 단일 학습
#   통일 환경: AdamW lr=1e-4, COCO pretrained, batch 자동(어댑터 내부 2로 축소)
#   사용법: bash scripts/run_exp2_cond4_8x_mask2former_yjw.sh <GPU>
# ============================================================
cd /project/ahnailab/yjw0619/seg/seg/VISION-Instance-Seg
mkdir -p results/logs

GPU=${1:?사용법: $0 <GPU번호>}
PYTHON=/home/jjh0709/.conda/envs/jjh/bin/python
EXPERIMENT="exp2"
CATEGORY="Exp2_3cls"
CONDITION="cond4_8x"
MODEL="mask2former"
SEED=42

RESULT="results/training/${EXPERIMENT}/${CONDITION}/${CATEGORY}/${MODEL}/seed${SEED}/eval_results/results.json"
TRAIN_DIR="results/training/${EXPERIMENT}/${CONDITION}/${CATEGORY}/${MODEL}/seed${SEED}"

# 이전 미완료/AP=0 결과 정리
if [ -f "$RESULT" ]; then
    PREV_AP=$(python3 -c "import json; print(json.load(open('$RESULT')).get('segm_AP',0))" 2>/dev/null)
    if [ "$PREV_AP" = "0" ] || [ "$PREV_AP" = "0.0" ]; then
        echo "[CLEAN] 이전 segm_AP=0 결과 제거"
        rm -rf "$TRAIN_DIR"
    else
        echo "[SKIP] 이미 완료 (segm_AP=$PREV_AP)"
        exit 0
    fi
elif [ -d "$TRAIN_DIR" ]; then
    echo "[CLEAN] 이전 미완료 디렉토리 제거"
    rm -rf "$TRAIN_DIR"
fi

echo "===== [양진우] cond4_8x / mask2former 시작 $(date) ====="
echo "GPU: $GPU, Condition: $CONDITION, Model: $MODEL"

CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m training.train \
    --category "$CATEGORY" --experiment "$EXPERIMENT" \
    --condition "$CONDITION" --model "$MODEL" --seed "$SEED" \
    --max-epochs 200 --patience 10 \
    2>&1 | tee "results/logs/exp2_${CONDITION}_${MODEL}.log"

echo "===== [양진우] cond4_8x / mask2former 완료 $(date) ====="
