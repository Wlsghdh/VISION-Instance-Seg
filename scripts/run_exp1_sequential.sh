#!/bin/bash
# Cable, Casting, Console, Cylinder, Wood × mask_rcnn, cascade_mask_rcnn 순차 학습
# 각 (category, model)마다 쿨다운 + quota 체크

set -e
cd /home/jjh0709/gitrepo/VISION-Instance-Seg

CATEGORIES=("Cable" "Casting" "Console" "Cylinder" "Wood")
MODELS=("mask_rcnn" "cascade_mask_rcnn")
QUOTA_LIMIT_MB=82000  # 82 GB 초과 시 중단
COOLDOWN_SEC=30
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

check_quota() {
  # quota -s 출력: "/dev/md126p3  60423M  87891M  97657M  ..."
  # 두 번째 컬럼이 사용량 (M 단위)
  local used=$(quota -s 2>/dev/null | awk '/\/dev\// {print $2}' | sed 's/[^0-9]//g')
  if [ -z "$used" ]; then
    echo "[QUOTA] 파싱 실패 — skip"
    return 0
  fi
  echo "[QUOTA] used=${used}MB / limit=${QUOTA_LIMIT_MB}MB"
  if [ "$used" -gt "$QUOTA_LIMIT_MB" ]; then
    echo "[ABORT] Quota 초과 — 학습 중단"
    exit 1
  fi
}

echo "===== 시작 $(date) ====="
check_quota

for cat in "${CATEGORIES[@]}"; do
  for model in "${MODELS[@]}"; do
    LOG_FILE="$LOG_DIR/${cat}_${model}.log"
    echo ""
    echo "===== [$(date '+%H:%M:%S')] $cat / $model ====="

    CUDA_VISIBLE_DEVICES=1 conda run -n jjh python -m training.train \
      --category "$cat" --experiment exp1 --condition all --model "$model" \
      --max-epochs 200 --patience 10 2>&1 | tee "$LOG_FILE"

    echo ""
    echo "[DONE] $cat / $model — 결과:"
    grep "\[OK\]\|\[FAIL\]" "$LOG_FILE" | tail -10

    check_quota

    echo ""
    echo "[COOLDOWN] ${COOLDOWN_SEC}초 대기..."
    sleep "$COOLDOWN_SEC"
  done
done

echo ""
echo "===== 전체 완료 $(date) ====="
