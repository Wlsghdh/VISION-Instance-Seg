#!/bin/bash
# ============================================================
# [주진호] 자동 Phase3 실행기
#   1. 현재 학습(cond4_6x cascade_rcnn) 완료 대기
#   2. Phase1+2 전체 결과(cond4_1x~10x × 2모델 = 20건) 완료 체크
#   3. 최적 N배 자동 결정 (segm_AP 평균 최고)
#   4. Phase3 실행 (최적 N배 × 7모델)
#
# 사용법: bash scripts/run_phase3_auto.sh
# ============================================================
set -e
cd /home/jjh0709/gitrepo/VISION-Instance-Seg
mkdir -p results/logs

GPU=1
COOLDOWN=30
CATEGORY="Exp2_3cls"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# ============================================================
# 1. 현재 학습 완료 대기
# ============================================================
CURRENT_PID=44285
if ps -p $CURRENT_PID > /dev/null 2>&1; then
    log "현재 학습(PID=$CURRENT_PID, cond4_6x cascade_rcnn) 완료 대기..."
    while ps -p $CURRENT_PID > /dev/null 2>&1; do
        sleep 120
    done
    log "현재 학습 완료. 쿨다운 ${COOLDOWN}초..."
    sleep $COOLDOWN
fi

# ============================================================
# 2. Phase1+2 결과 준비 대기 (cond4_1x~10x × 2모델 = 20건)
# ============================================================
log "Phase1+2 결과 준비 대기 중 (yjw + ldy + jjh)..."

check_phase12_ready() {
    local count=0
    for cond in cond4_1x cond4_2x cond4_3x cond4_4x cond4_5x cond4_6x \
                cond4_7x cond4_8x cond4_9x cond4_10x; do
        for model in mask_rcnn cascade_mask_rcnn; do
            f="results/training/exp2/${cond}/${CATEGORY}/${model}/seed42/eval_results/results.json"
            [ -f "$f" ] && count=$((count+1))
        done
    done
    echo $count
}

while true; do
    done=$(check_phase12_ready)
    log "Phase1+2 진행: $done/20 완료"
    if [ "$done" -ge 20 ]; then
        log "Phase1+2 전체 완료! 최적 N배 분석 시작..."
        break
    fi
    sleep 1800  # 30분마다 체크
done

# ============================================================
# 3. 최적 N배 자동 결정
# ============================================================
BEST_COND=$(python3 -c "
import json, os

best_cond = None
best_avg = -1.0
for cond in ['cond4_1x','cond4_2x','cond4_3x','cond4_4x','cond4_5x',
             'cond4_6x','cond4_7x','cond4_8x','cond4_9x','cond4_10x']:
    aps = []
    for model in ['mask_rcnn', 'cascade_mask_rcnn']:
        f = f'results/training/exp2/{cond}/Exp2_3cls/{model}/seed42/eval_results/results.json'
        if os.path.exists(f):
            d = json.load(open(f))
            ap = d.get('segm_AP', d.get('segm/AP', -1))
            if ap > 0:
                aps.append(ap)
    if len(aps) == 2:
        avg = sum(aps) / 2
        if avg > best_avg:
            best_avg = avg
            best_cond = cond

print(best_cond)
")

if [ -z "$BEST_COND" ]; then
    log "ERROR: 최적 N배 결정 실패. 수동으로 실행 필요."
    exit 1
fi

log "최적 N배: $BEST_COND"

# 요약 출력
log "Phase1+2 결과 요약:"
python3 -c "
import json, os
print(f'  {\"조건\":>10} {\"mask_rcnn\":>12} {\"cascade_mask\":>14} {\"평균\":>10}')
for cond in ['cond4_1x','cond4_2x','cond4_3x','cond4_4x','cond4_5x',
             'cond4_6x','cond4_7x','cond4_8x','cond4_9x','cond4_10x']:
    aps = []
    for model in ['mask_rcnn', 'cascade_mask_rcnn']:
        f = f'results/training/exp2/{cond}/Exp2_3cls/{model}/seed42/eval_results/results.json'
        ap = -1
        if os.path.exists(f):
            d = json.load(open(f))
            ap = d.get('segm_AP', d.get('segm/AP', -1))
        aps.append(ap)
    avg = sum(aps) / 2 if all(a > 0 for a in aps) else -1
    marker = ' <= BEST' if cond == '$BEST_COND' else ''
    print(f'  {cond:>10} {aps[0]:>12.2f} {aps[1]:>14.2f} {avg:>10.2f}{marker}')
"

# ============================================================
# 4. Phase3 실행 (최적 N배 × 7모델)
# ============================================================
log "========== Phase3 시작: $BEST_COND × 7모델 =========="
bash scripts/run_exp2_phase3.sh $GPU $BEST_COND

log "============================================"
log "  모든 실험 완료!"
log "============================================"
