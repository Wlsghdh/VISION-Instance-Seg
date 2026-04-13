#!/bin/bash
# ============================================================
# 쿼터 모니터 — 30분마다 체크, 80GB 초과 시 중간 체크포인트 자동 삭제
# 사용법: bash scripts/quota_monitor.sh &
# ============================================================
cd /home/jjh0709/gitrepo/VISION-Instance-Seg
LIMIT_MB=80000

while true; do
  USED=$(quota -s 2>/dev/null | awk '/\/dev\//{gsub(/[^0-9]/,"",$2); print $2}')
  if [ -z "$USED" ]; then
    USED=$(quota -s 2>/dev/null | awk '/\/dev\//{print $2}' | sed 's/M$//' | sed 's/\*$//')
  fi

  NOW=$(date '+%H:%M:%S')

  # 중간 체크포인트 개수 확인
  CKPT_COUNT=$(find results/training -name "model_*.pth" -not -name "model_final.pth" -not -name "model_best.pth" 2>/dev/null | wc -l)
  CKPT_SIZE=$(find results/training -name "model_*.pth" -not -name "model_final.pth" -not -name "model_best.pth" -printf "%s\n" 2>/dev/null | awk '{sum+=$1} END {printf "%.1f", sum/1024/1024/1024}')

  echo "[$NOW] QUOTA: ${USED}MB / 87891MB | 중간 체크포인트: ${CKPT_COUNT}개 (${CKPT_SIZE}GB)"

  if [ -n "$USED" ] && [ "$USED" -gt "$LIMIT_MB" ] 2>/dev/null; then
    echo "[$NOW] ⚠️  ${LIMIT_MB}MB 초과! 중간 체크포인트 삭제..."
    find results/training -name "model_*.pth" -not -name "model_final.pth" -not -name "model_best.pth" -delete
    echo "[$NOW] ✅ 삭제 완료"
  fi

  sleep 1800  # 30분
done
