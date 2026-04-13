#!/bin/bash
# ============================================================
# 주진호 담당: exp3 N=5~7 × 2모델 (cascade_mask_rcnn, cascade_rcnn)
#            + exp2 Phase 1 전체
# 사용법: bash scripts/run_exp3_jjh.sh <GPU> <DEFECT1> <DEFECT2> <DEFECT3>
# 예시:   bash scripts/run_exp3_jjh.sh 1 Inclusoes Dirty impurities
# ============================================================
set -e
cd /home/jjh0709/gitrepo/VISION-Instance-Seg

GPU=${1:?사용법: $0 <GPU번호> <DEFECT1> <DEFECT2> <DEFECT3>}
D1=${2:?DEFECT1 필요}
D2=${3:?DEFECT2 필요}
D3=${4:?DEFECT3 필요}
DEFECTS=("$D1" "$D2" "$D3")

echo "===== [주진호] 시작 $(date) ====="
echo "GPU: $GPU, Defects: ${DEFECTS[*]}"

# --- Phase 1: exp2 cond1~3 ---
echo ""
echo "========== exp2 Phase 1 (cond1~3) =========="
for cat in "${DEFECTS[@]}"; do
  for model in mask_rcnn cascade_mask_rcnn; do
    echo ""
    echo "===== [$(date '+%H:%M:%S')] exp2 / $cat / $model ====="
    CUDA_VISIBLE_DEVICES=$GPU conda run -n jjh python -m training.train \
      --category "$cat" --experiment exp2 --condition all --model "$model" \
      --max-epochs 200 --patience 10 2>&1 | tee "results/logs/exp2_${cat}_${model}.log"
    sleep 10
  done
done

# --- Phase 2: exp3 N=5~7 ---
echo ""
echo "========== exp3 Phase 2 (N=5~7) =========="
MODELS=("cascade_mask_rcnn" "cascade_rcnn")
CONDITIONS=("trad_5x" "trad_6x" "trad_7x")

TOTAL=$((${#DEFECTS[@]} * ${#MODELS[@]} * ${#CONDITIONS[@]}))
COUNT=0

for cond in "${CONDITIONS[@]}"; do
  for cat in "${DEFECTS[@]}"; do
    for model in "${MODELS[@]}"; do
      COUNT=$((COUNT + 1))
      echo ""
      echo "===== [$COUNT/$TOTAL] [$(date '+%H:%M:%S')] $cat / $cond / $model ====="
      CUDA_VISIBLE_DEVICES=$GPU conda run -n jjh python -m training.train \
        --category "$cat" --experiment exp3 --condition "$cond" --model "$model" \
        --max-epochs 200 --patience 10 2>&1 | tee "results/logs/exp3_${cat}_${cond}_${model}.log"
      sleep 10
    done
  done
done

echo "===== [주진호] 완료 $(date) ====="
