#!/bin/bash
# ============================================================
# Exp2 cond4_8x — mmdet 3종 모델 학습 (pretrained fix 적용)
#   조건: 원본 20 + GenAI 125 + 전통 1,160 = 1,305/cls × 3cls = 3,915장
#   모델: cascade_rcnn, solov2, rtmdet_ins
#   수정사항: COCO pretrained 로드 + AdamW lr=1e-4 + batch_size=4
#   쿨다운: 모델 간 5분 (GPU 메모리 정리)
#
# 사용법: bash scripts/run_exp2_cond4_8x_mmdet.sh <GPU>
# ============================================================
cd /home/jjh0709/gitrepo/VISION-Instance-Seg
mkdir -p results/logs

GPU=${1:?사용법: $0 <GPU번호>}
EXPERIMENT="exp2"
CATEGORY="Exp2_3cls"
CONDITION="cond4_8x"
COOLDOWN=300  # 5분

MODELS=("cascade_rcnn" "solov2" "rtmdet_ins")

FAILED=()
SUCCESS=()

echo "===== cond4_8x mmdet 3종 (pretrained fix) 시작 $(date) ====="
echo "GPU: $GPU"
echo "수정사항: COCO pretrained 로드 + AdamW lr=1e-4 + batch_size=4"
echo "Condition: $CONDITION (원본 20 + GenAI 125 + 전통 1,160 = 1,305/cls)"
echo "Models: ${MODELS[*]}"
echo "Cooldown: ${COOLDOWN}초 (5분)"
echo ""

run_one() {
    local model=$1
    local logfile="results/logs/exp2_${CONDITION}_${model}.log"
    local result_dir="results/training/${EXPERIMENT}/${CONDITION}/${CATEGORY}/${model}/seed42/eval_results/results.json"

    # 이전 실패 결과 디렉토리 정리 (segm_AP=0 결과 제거)
    local train_dir="results/training/${EXPERIMENT}/${CONDITION}/${CATEGORY}/${model}/seed42"
    if [ -d "$train_dir" ] && [ ! -f "$result_dir" ]; then
        echo "[CLEAN] 이전 실패 디렉토리 정리: $train_dir"
        rm -rf "$train_dir"
    fi
    if [ -f "$result_dir" ]; then
        local prev_ap=$(python3 -c "import json; print(json.load(open('$result_dir')).get('segm_AP',0))" 2>/dev/null)
        if [ "$prev_ap" = "0" ] || [ "$prev_ap" = "0.0" ]; then
            echo "[CLEAN] 이전 segm_AP=0 결과 제거: $train_dir"
            rm -rf "$train_dir"
        else
            echo "[SKIP] $model (이미 완료, segm_AP=$prev_ap)"
            SUCCESS+=("$model")
            return 0
        fi
    fi

    echo ""
    echo "===== [$(date '+%H:%M:%S')] $CONDITION / $model ====="
    CUDA_VISIBLE_DEVICES=$GPU conda run -n jjh python -m training.train \
        --category "$CATEGORY" --experiment "$EXPERIMENT" \
        --condition "$CONDITION" --model "$model" --seed 42 \
        --max-epochs 200 --patience 10 \
        2>&1 | tee "$logfile"

    if [ -f "$result_dir" ]; then
        local ap=$(python3 -c "import json; print(f'{json.load(open(\"$result_dir\")).get(\"segm_AP\",0):.2f}')" 2>/dev/null)
        SUCCESS+=("$model")
        echo "[OK] $model 완료 — segm_AP=$ap"
    else
        FAILED+=("$model")
        echo "[FAIL] $model — 로그 마지막 30줄:"
        tail -30 "$logfile"
        echo "[CONTINUE] 다음 모델로 진행"
    fi
}

for i in "${!MODELS[@]}"; do
    run_one "${MODELS[$i]}"
    # 마지막 모델 아니면 cooldown
    if [ $i -lt $((${#MODELS[@]}-1)) ]; then
        echo "[COOLDOWN] ${COOLDOWN}초 (5분) 대기 — GPU 메모리 정리..."
        sleep $COOLDOWN
    fi
done

echo ""
echo "===== cond4_8x mmdet 학습 완료 $(date) ====="
echo "성공: ${SUCCESS[*]:-(없음)}"
echo "실패: ${FAILED[*]:-(없음)}"
echo ""
echo ">>> sync: python scripts/sync_results_to_github.py --exp exp2"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo ">>> 실패 로그:"
    for m in "${FAILED[@]}"; do
        echo "    results/logs/exp2_${CONDITION}_${m}.log"
    done
    exit 1
fi
