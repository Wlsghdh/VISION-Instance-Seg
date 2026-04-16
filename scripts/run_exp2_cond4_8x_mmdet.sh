#!/bin/bash
# ============================================================
# Exp2 cond4_8x — mmdet 3종 모델 학습
#   조건: 원본 20 + GenAI 125 + 전통 1,160 = 1,305/cls × 3cls = 3,915장
#   모델: cascade_rcnn, solov2, rtmdet_ins
#   학습 방식: epoch 기반 (기존 cond4_4x/5x/6x mmdet 결과와 직접 비교 가능)
#   특이사항: cooldown 없음, 실패 시 다음 모델로 즉시 진행 (에러 로그는 보존)
#
# 사용법: bash scripts/run_exp2_cond4_8x_mmdet.sh <GPU>
# ============================================================
# set -e 제거 — 한 모델 실패해도 나머지 계속 진행
cd /home/jjh0709/gitrepo/VISION-Instance-Seg
mkdir -p results/logs

GPU=${1:?사용법: $0 <GPU번호>}
EXPERIMENT="exp2"
CATEGORY="Exp2_3cls"
CONDITION="cond4_8x"

# mmdet 3종 (epoch 기반, 기존 cond4_4x~6x mmdet과 동일 설정)
MODELS=("cascade_rcnn" "solov2" "rtmdet_ins")

FAILED=()
SUCCESS=()

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
        SUCCESS+=("$model")
        return 0
    fi

    echo ""
    echo "===== [$(date '+%H:%M:%S')] $CONDITION / $model ====="
    CUDA_VISIBLE_DEVICES=$GPU conda run -n jjh python -m training.train \
        --category "$CATEGORY" --experiment "$EXPERIMENT" \
        --condition "$CONDITION" --model "$model" --seed 42 \
        --max-epochs 200 --patience 10 \
        2>&1 | tee "$logfile"

    if [ -f "$result_dir" ]; then
        SUCCESS+=("$model")
        echo "[OK] $model 완료 — segm_AP=$(python3 -c "import json; print(f'{json.load(open(\"$result_dir\")).get(\"segm_AP\",0):.2f}')" 2>/dev/null)"
    else
        FAILED+=("$model")
        echo "[FAIL] $model — 로그 마지막 30줄:"
        tail -30 "$logfile"
        echo "[CONTINUE] 다음 모델로 진행"
    fi
}

for model in "${MODELS[@]}"; do
    run_one "$model"
done

echo ""
echo "===== cond4_8x mmdet 학습 완료 $(date) ====="
echo "성공: ${SUCCESS[*]:-(없음)}"
echo "실패: ${FAILED[*]:-(없음)}"
echo ""
echo ">>> sync: python scripts/sync_results_to_github.py --exp exp2"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo ">>> 실패 로그 위치:"
    for m in "${FAILED[@]}"; do
        echo "    results/logs/exp2_${CONDITION}_${m}.log"
    done
    exit 1
fi
