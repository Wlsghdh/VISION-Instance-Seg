#!/usr/bin/env bash
# 실험1 (per-category, mask_rcnn) — GPU 1 전용
# 6 카테고리 × 6 conditions = 36 runs, 매 run 후 10분 쿨다운
#
# 사용:
#   nohup bash scripts/run_exp1_singlecat_gpu1.sh > results/logs/exp1_master_gpu1.log 2>&1 &

set -u  # -e는 사용 X (한 run 실패해도 다음 run 계속)

cd /home/jjh0709/gitrepo/VISION-Instance-Seg

export CUDA_VISIBLE_DEVICES=1

CATEGORIES=(Cable Casting Console Cylinder Wood Screw)
CONDITIONS=(baseline genai_25 genai_50 genai_75 genai_100 genai_125)
MODEL=mask_rcnn
COOLDOWN=600   # 10분

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="results/logs/exp1_singlecat_gpu1_${TS}"
SUMMARY_MD="results/reports/exp1_singlecat_gpu1_${TS}.md"
mkdir -p "$LOG_DIR" "$(dirname "$SUMMARY_MD")"

TOTAL=$(( ${#CATEGORIES[@]} * ${#CONDITIONS[@]} ))
IDX=0
START_TS=$(date '+%F %T')

echo "================================================================"
echo "  실험1 (single-category) GPU 1  시작: $START_TS"
echo "  Categories: ${CATEGORIES[*]}"
echo "  Conditions: ${CONDITIONS[*]}"
echo "  Model: $MODEL"
echo "  Total runs: $TOTAL  | Cooldown: ${COOLDOWN}s"
echo "  Log dir: $LOG_DIR"
echo "  Summary:  $SUMMARY_MD"
echo "================================================================"

for cat in "${CATEGORIES[@]}"; do
  for cond in "${CONDITIONS[@]}"; do
    IDX=$((IDX + 1))
    LOG="$LOG_DIR/${cat}_${cond}_${MODEL}.log"
    NOW=$(date '+%F %T')

    echo ""
    echo "----------------------------------------------------------------"
    echo "  [$IDX/$TOTAL] $NOW | $cat / $cond / $MODEL"
    echo "  Log: $LOG"
    echo "----------------------------------------------------------------"

    /home/jjh0709/.conda/envs/jjh/bin/python -m training.train \
        --category "$cat" \
        --experiment exp1 \
        --condition "$cond" \
        --model "$MODEL" \
        2>&1 | tee "$LOG"

    STATUS=${PIPESTATUS[0]}
    END_NOW=$(date '+%F %T')
    echo "  → [$IDX/$TOTAL] $cat/$cond exit=$STATUS  ($END_NOW)"

    # 매 run 후 요약 markdown 갱신 (실패해도 무시)
    /home/jjh0709/.conda/envs/jjh/bin/python scripts/summarize_exp1_singlecat.py --output "$SUMMARY_MD" || true

    if [ "$IDX" -lt "$TOTAL" ]; then
      echo "  Cooldown ${COOLDOWN}s ..."
      sleep "$COOLDOWN"
    fi
  done
done

# 최종 요약 한 번 더
/home/jjh0709/.conda/envs/jjh/bin/python scripts/summarize_exp1_singlecat.py --output "$SUMMARY_MD" || true

echo ""
echo "================================================================"
echo "  ALL DONE: $(date '+%F %T')  (started $START_TS)"
echo "  Summary: $SUMMARY_MD"
echo "================================================================"
