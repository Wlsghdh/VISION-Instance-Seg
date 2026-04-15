#!/bin/bash
# Exp1_3cls 데이터 무결성 검증.
# 학습 스크립트 진입 첫 줄에서 호출. manifest와 sha256 불일치 시 abort.
set -e
cd /home/jjh0709/gitrepo/VISION-Instance-Seg

MANIFEST="docs/experiment_plans/exp1_3cls_v2_data_manifest.txt"

if [ ! -f "$MANIFEST" ]; then
    echo "❌ Manifest 없음: $MANIFEST"
    echo "   먼저 'bash scripts/data_utils/freeze_exp1_3cls.sh' 실행 필요."
    exit 1
fi

echo "===== Exp1_3cls 데이터 검증 ====="
ERRORS=0
CHECKED=0
while IFS= read -r line; do
    # sha256sum 라인만 (64자리 hash + 두 칸 + path)
    if [[ "$line" =~ ^[a-f0-9]{64}\ \ (.+)$ ]]; then
        expected_hash="${line:0:64}"
        path="${BASH_REMATCH[1]}"
        if [ ! -f "$path" ]; then
            echo "  ❌ MISSING: $path"
            ERRORS=$((ERRORS+1))
            continue
        fi
        actual_hash=$(sha256sum "$path" | cut -d' ' -f1)
        if [ "$expected_hash" != "$actual_hash" ]; then
            echo "  ❌ MISMATCH: $path"
            echo "     expected: $expected_hash"
            echo "     actual:   $actual_hash"
            ERRORS=$((ERRORS+1))
        fi
        CHECKED=$((CHECKED+1))
    fi
done < "$MANIFEST"

echo ""
echo "Checked: $CHECKED files, Errors: $ERRORS"
if [ $ERRORS -gt 0 ]; then
    echo "❌ 검증 실패 — 학습 중단"
    exit 1
fi
echo "✅ 데이터 무결성 OK"
