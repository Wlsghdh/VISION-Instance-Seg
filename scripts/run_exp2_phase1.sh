#!/bin/bash
# ============================================================
# 실험 2 Phase 1: 전통 vs GenAI 비교 (cond1~3, 2종 모델)
# 사용법: bash scripts/run_exp2_phase1.sh <GPU> <DEFECT1> <DEFECT2> <DEFECT3>
# 예시:   bash scripts/run_exp2_phase1.sh 1 Inclusoes Dirty impurities
# ============================================================
set -e
cd /home/jjh0709/gitrepo/VISION-Instance-Seg

GPU=${1:?사용법: $0 <GPU번호> <DEFECT1> <DEFECT2> <DEFECT3>}
D1=${2:?DEFECT1 필요}
D2=${3:?DEFECT2 필요}
D3=${4:?DEFECT3 필요}
DEFECTS=("$D1" "$D2" "$D3")

echo "===== exp2 Phase 1 시작 $(date) ====="
echo "GPU: $GPU, Defects: ${DEFECTS[*]}"

for cat in "${DEFECTS[@]}"; do
  for model in mask_rcnn cascade_mask_rcnn; do
    echo ""
    echo "===== [$(date '+%H:%M:%S')] $cat / $model ====="
    CUDA_VISIBLE_DEVICES=$GPU conda run -n jjh python -m training.train \
      --category "$cat" --experiment exp2 --condition all --model "$model" \
      --max-epochs 200 --patience 10 2>&1 | tee "results/logs/exp2_${cat}_${model}.log"
    echo "[DONE] $cat / $model"
    echo "[COOLDOWN] 5분 대기..."
    sleep 300
  done
done

echo "===== exp2 Phase 1 완료 $(date) ====="
