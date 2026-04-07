#!/bin/bash
# 실험 1 - Mask R-CNN 학습
# - 카테고리: Cable, Screw, Casting, Console, Cylinder (exp1_config.md 기준 5종)
# - 조건    : baseline ~ genai_125 (6개)
# - 시드    : seed=42 단일 (다중시드는 추후 별도 실행)
# - 총 학습 : 5 × 6 × 1 = 30회
#
# 사용법:
#   tmux new -d -s exp1_maskrcnn 'bash training/run_exp1_maskrcnn.sh'
#   tmux attach -t exp1_maskrcnn   # 확인
#   tmux ls                        # 세션 목록

set -u

# =============================================================
# 경로/환경 설정
# =============================================================
PROJECT_DIR=/project/ahnailab/yjw0619/seg/seg/VISION-Instance-Seg
PYTHON=/home/jjh0709/.conda/envs/jjh/bin/python

# 결과 저장 경로를 yjw0619 폴더로 분리 (config.py의 RESULTS_ROOT override)
export VISION_RESULTS_ROOT=$PROJECT_DIR

# GPU 3번 고정 (1인 1GPU 정책)
export CUDA_VISIBLE_DEVICES=3

# 학습 대상 카테고리 (exp1_config.md 5종)
CATEGORIES=(Cable Screw Casting Console Cylinder)
SEED=42

# 로그 디렉토리
LOG_DIR=$PROJECT_DIR/results/logs/exp1_maskrcnn
mkdir -p "$LOG_DIR"
LOG_FILE=$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log

cd "$PROJECT_DIR"

# =============================================================
# 실행 정보 기록
# =============================================================
{
    echo "============================================================"
    echo "  Experiment 1 - Mask R-CNN 학습 시작"
    echo "  시작 시각  : $(date)"
    echo "  사용자     : $(whoami)"
    echo "  작업 경로  : $PROJECT_DIR"
    echo "  결과 경로  : $VISION_RESULTS_ROOT/results/"
    echo "  GPU        : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "  Python     : $PYTHON"
    echo "  카테고리   : ${CATEGORIES[*]}"
    echo "  시드       : $SEED (단일)"
    echo "  총 학습 수 : ${#CATEGORIES[@]} cat × 6 cond × 1 seed = $((${#CATEGORIES[@]} * 6))회"
    echo "============================================================"
} | tee -a "$LOG_FILE"

# =============================================================
# 카테고리별 학습 실행 (실패 시 자동 재시도 최대 5회)
# =============================================================
MAX_RETRY=5

for CAT in "${CATEGORIES[@]}"; do
    {
        echo ""
        echo "############################################################"
        echo "##  카테고리: $CAT  /  $(date)"
        echo "############################################################"
    } | tee -a "$LOG_FILE"

    attempt=1
    while [ $attempt -le $MAX_RETRY ]; do
        echo ""                                                   | tee -a "$LOG_FILE"
        echo "[$CAT] Attempt $attempt/$MAX_RETRY  $(date)"         | tee -a "$LOG_FILE"
        echo "------------------------------------------------------------" | tee -a "$LOG_FILE"

        $PYTHON -m training.train \
            --experiment exp1 \
            --category "$CAT" \
            --condition all \
            --model mask_rcnn \
            --seed $SEED \
            2>&1 | tee -a "$LOG_FILE"

        exit_code=${PIPESTATUS[0]}

        if [ $exit_code -eq 0 ]; then
            echo "" | tee -a "$LOG_FILE"
            echo "[$CAT DONE] 정상 종료 ($(date))" | tee -a "$LOG_FILE"
            break
        fi

        echo "" | tee -a "$LOG_FILE"
        echo "[$CAT FAIL] exit=$exit_code. 60초 후 재시도..." | tee -a "$LOG_FILE"
        sleep 60
        attempt=$((attempt + 1))
    done
done

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "[END] 전체 종료 시각: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
