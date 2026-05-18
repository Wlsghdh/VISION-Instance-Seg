#!/bin/bash
# ============================================================
# Exp2 cond4_8x — 5모델 통합 비교 (통일된 환경)
#   조건: 원본 20 + GenAI 125 + 전통 1,160 = 1,305/cls × 3cls = 3,915장
#   모델: maskdino, mask2former (detectron2) + cascade_rcnn, solov2, rtmdet_ins (mmdet)
#   통일 환경:
#     - 전 모델 COCO pretrained
#     - AdamW lr=1e-4 (각 모델 기본 optimizer 유지, lr 통일)
#     - batch_size=4 (single A100 안정)
#     - max_epochs=200, patience=10, eval_period=5
#   쿨다운: 모델 간 5분 (GPU 메모리 정리)
#
# 사용법: bash scripts/run_exp2_cond4_8x_5models.sh <GPU>
# ============================================================
cd /home/jjh0709/gitrepo/VISION-Instance-Seg
mkdir -p results/logs

GPU=${1:?사용법: $0 <GPU번호>}
EXPERIMENT="exp2"
CATEGORY="Exp2_3cls"
CONDITION="cond4_8x"
COOLDOWN=300  # 5분

# 5모델 순서: detectron2 먼저 (안정적), mmdet 나중
MODELS=("maskdino" "mask2former" "cascade_rcnn" "solov2" "rtmdet_ins")

FAILED=()
SUCCESS=()

echo "===== cond4_8x 5모델 통합 비교 시작 $(date) ====="
echo "GPU: $GPU"
echo "통일 환경: AdamW lr=1e-4, batch=4, COCO pretrained"
echo "Models: ${MODELS[*]}"
echo "Cooldown: ${COOLDOWN}초 (5분)"
echo ""

run_one() {
    local model=$1
    local logfile="results/logs/exp2_${CONDITION}_${model}.log"
    local result_dir="results/training/${EXPERIMENT}/${CONDITION}/${CATEGORY}/${model}/seed42/eval_results/results.json"
    local train_dir="results/training/${EXPERIMENT}/${CONDITION}/${CATEGORY}/${model}/seed42"

    # 이전 segm_AP=0 결과 자동 정리
    if [ -f "$result_dir" ]; then
        local prev_ap=$(python3 -c "import json; print(json.load(open('$result_dir')).get('segm_AP',0))" 2>/dev/null)
        if [ "$prev_ap" = "0" ] || [ "$prev_ap" = "0.0" ]; then
            echo "[CLEAN] 이전 segm_AP=0 결과 제거: $model"
            rm -rf "$train_dir"
        else
            echo "[SKIP] $model (이미 완료, segm_AP=$prev_ap)"
            SUCCESS+=("$model(=$prev_ap)")
            return 0
        fi
    elif [ -d "$train_dir" ]; then
        echo "[CLEAN] 이전 미완료 디렉토리 제거: $model"
        rm -rf "$train_dir"
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
        if [ "$ap" = "0.00" ]; then
            FAILED+=("$model(AP=0)")
            echo "[FAIL] $model — segm_AP=0 (학습 발산)"
            tail -30 "$logfile"
        else
            SUCCESS+=("$model(=$ap)")
            echo "[OK] $model — segm_AP=$ap"
        fi
    else
        FAILED+=("$model")
        echo "[FAIL] $model — eval 결과 없음. 로그 마지막 30줄:"
        tail -30 "$logfile"
    fi
    echo "[CONTINUE]"
}

for i in "${!MODELS[@]}"; do
    run_one "${MODELS[$i]}"
    # 마지막 모델 아니면 cooldown
    if [ $i -lt $((${#MODELS[@]}-1)) ]; then
        echo "[COOLDOWN] ${COOLDOWN}초 (5분) 대기..."
        sleep $COOLDOWN
    fi
done

echo ""
echo "============================================"
echo "  cond4_8x 5모델 통합 비교 완료 $(date)"
echo "============================================"
echo "성공: ${SUCCESS[*]:-(없음)}"
echo "실패: ${FAILED[*]:-(없음)}"
echo ""
echo ">>> 결과 동기화: python scripts/sync_results_to_github.py --exp exp2"
echo ">>> git add results_github && git commit && git push"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo ">>> 실패 로그:"
    for m in "${FAILED[@]}"; do
        echo "    results/logs/exp2_${CONDITION}_${m%%(*}.log"
    done
    exit 1
fi
