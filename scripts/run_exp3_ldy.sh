#!/bin/bash
# ============================================================
# 임대윤 담당: exp3 N=8~10 × 2모델 (solov2, rtmdet_ins)
# 사용법: bash scripts/run_exp3_ldy.sh <GPU> <DEFECT1> <DEFECT2> <DEFECT3>
# 예시:   bash scripts/run_exp3_ldy.sh 3 Inclusoes Dirty impurities
# ============================================================
set -e
cd /home/jjh0709/gitrepo/VISION-Instance-Seg

GPU=${1:?사용법: $0 <GPU번호> <DEFECT1> <DEFECT2> <DEFECT3>}
D1=${2:?DEFECT1 필요}
D2=${3:?DEFECT2 필요}
D3=${4:?DEFECT3 필요}
DEFECTS=("$D1" "$D2" "$D3")
MODELS=("solov2" "rtmdet_ins")
CONDITIONS=("trad_8x" "trad_9x" "trad_10x")

TOTAL=$((${#DEFECTS[@]} * ${#MODELS[@]} * ${#CONDITIONS[@]}))
COUNT=0

echo "===== [임대윤] exp3 시작 $(date) ====="
echo "GPU: $GPU, Defects: ${DEFECTS[*]}"
echo "Models: ${MODELS[*]}"
echo "Conditions: ${CONDITIONS[*]}"
echo "총 학습: $TOTAL 회"

for cond in "${CONDITIONS[@]}"; do
  for cat in "${DEFECTS[@]}"; do
    for model in "${MODELS[@]}"; do
      COUNT=$((COUNT + 1))
      echo ""
      echo "===== [$COUNT/$TOTAL] [$(date '+%H:%M:%S')] $cat / $cond / $model ====="
      CUDA_VISIBLE_DEVICES=$GPU conda run -n jjh python -m training.train \
        --category "$cat" --experiment exp3 --condition "$cond" --model "$model" \
        --max-epochs 200 --patience 10 2>&1 | tee "results/logs/exp3_${cat}_${cond}_${model}.log"
      echo "[COOLDOWN] 5분 대기..."
      sleep 300
    done
  done
done

echo "===== [임대윤] exp3 완료 $(date) ====="
