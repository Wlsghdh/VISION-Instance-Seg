#!/bin/bash
# MaskDINO Training Pipeline - GPU 1
# 사용법: CUDA_VISIBLE_DEVICES=1 bash run_train.sh

set -e

WORK_DIR="/data2/project/2026winter/jjh0709/AA_CV_R"
cd $WORK_DIR

echo "============================================================"
echo "MaskDINO Training Pipeline"
echo "Working Directory: $WORK_DIR"
echo "============================================================"

# 1. Original 데이터만 학습 (26장)
echo ""
echo "[STEP 1/4] Training with Original data only (26 images)..."
echo "============================================================"
mkdir -p output_original
python 1_train_original.py 2>&1 | tee output_original/train.log

# 2. Original 모델 시각화
echo ""
echo "[STEP 2/4] Visualizing Original model predictions..."
echo "============================================================"
python 3_visualize.py --model original --threshold 0.5 2>&1 | tee output_original/visualize.log

# 3. 전체 데이터 학습 (127장)
echo ""
echo "[STEP 3/4] Training with Full data (127 images)..."
echo "============================================================"
mkdir -p output_full
python 2_train_full.py 2>&1 | tee output_full/train.log

# 4. Full 모델 시각화
echo ""
echo "[STEP 4/4] Visualizing Full model predictions..."
echo "============================================================"
python 3_visualize.py --model full --threshold 0.5 2>&1 | tee output_full/visualize.log

# 결과 요약
echo ""
echo "============================================================"
echo "TRAINING COMPLETE!"
echo "============================================================"
echo ""
echo "Results:"
echo "  - Original model: $WORK_DIR/output_original/"
echo "  - Full model: $WORK_DIR/output_full/"
echo ""
if [ -f "$WORK_DIR/output_original/eval_results.json" ]; then
    echo "[Original] Evaluation:"
    cat $WORK_DIR/output_original/eval_results.json
fi
echo ""
if [ -f "$WORK_DIR/output_full/eval_results.json" ]; then
    echo "[Full] Evaluation:"
    cat $WORK_DIR/output_full/eval_results.json
fi
echo "============================================================"