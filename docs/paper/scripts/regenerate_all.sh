#!/bin/bash
# 모든 figure/table 재생성. 학습 결과 갱신 후 매번 실행 권장.
set -e
cd "$(dirname "$0")"

echo "===== Paper artifacts regeneration $(date) ====="

run() {
    echo ""
    echo "--- $1 ---"
    conda run -n jjh python "$1" "${@:2}" || echo "⚠️ $1 실패 (다른 파일은 계속)"
}

run fig1_cond4_curve.py
run table1_main.py
run table2_perclass.py

# (Phase 2 이후) 추가
# run fig2_genai_scaling.py
# run fig3_perclass_heatmap.py
# run fig4_qualitative.py

echo ""
echo "===== 완료 ====="
echo "생성물:"
ls ../figs/ 2>/dev/null | sed 's/^/  figs\//'
ls ../tables/ 2>/dev/null | sed 's/^/  tables\//'
