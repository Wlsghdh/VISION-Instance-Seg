#!/bin/bash
# Exp1_3cls 데이터 동결: sha256 manifest + git tag.
# 학습 시작 전 1회 실행 (jjh가 호스트 측에서).
# 출력: docs/experiment_plans/exp1_3cls_v2_data_manifest.txt + git tag
set -e
cd /home/jjh0709/gitrepo/VISION-Instance-Seg

TAG="exp1_3cls_frozen_$(date +%Y%m%d)"
MANIFEST="docs/experiment_plans/exp1_3cls_v2_data_manifest.txt"

echo "===== Exp1_3cls 데이터 동결 ====="
echo "Tag: $TAG"
echo "Manifest: $MANIFEST"

# 1. sha256 manifest (원본 + GenAI for Dirty/Inclusoes/impurities + val split)
echo "" > "$MANIFEST"
echo "# Exp1_3cls v2 data manifest — generated $(date)" >> "$MANIFEST"
echo "# git commit: $(git rev-parse HEAD)" >> "$MANIFEST"
echo "" >> "$MANIFEST"

for cat in Dirty Inclusoes impurities; do
    echo "## $cat — original" >> "$MANIFEST"
    if [ -d "data/$cat" ]; then
        find "data/$cat" -type f \( -name "*.json" -o -name "*.jpg" -o -name "*.png" \) \
            -exec sha256sum {} \; | sort >> "$MANIFEST" 2>/dev/null || true
    fi
    echo "" >> "$MANIFEST"
    echo "## $cat — gen_ai" >> "$MANIFEST"
    if [ -d "data_augmented/$cat/gen_ai" ]; then
        find "data_augmented/$cat/gen_ai" -type f \( -name "*.json" -o -name "*.jpg" -o -name "*.png" \) \
            -exec sha256sum {} \; | sort >> "$MANIFEST" 2>/dev/null || true
    fi
    echo "" >> "$MANIFEST"
done

echo "## val split" >> "$MANIFEST"
sha256sum results/merged_datasets/_exp1_3cls_val_split.json >> "$MANIFEST" 2>/dev/null || true
sha256sum results/merged_datasets/_exp2_3cls_val/annotations.json >> "$MANIFEST" 2>/dev/null || true

echo ""
echo "✅ Manifest 작성 완료: $MANIFEST ($(wc -l < $MANIFEST) lines)"

# 2. git tag
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "⚠️  태그 $TAG 이미 존재. 스킵."
else
    git tag -a "$TAG" -m "Exp1_3cls v2 frozen — data + code snapshot for jjh/yjw"
    echo "✅ Git tag 생성: $TAG (push로 공유: git push origin $TAG)"
fi

echo ""
echo "===== 완료 ====="
echo "팀원 동기화 절차:"
echo "  1. git fetch --tags && git checkout $TAG"
echo "  2. bash scripts/data_utils/verify_exp1_3cls.sh   # 자동 검증"
