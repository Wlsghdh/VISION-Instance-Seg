"""
Exp1_3cls val 셋(82장)을 클래스 stratified 50/50 분할.
- val_dev: 모델 선택 / early-stop
- val_test: 최종 보고
seed=12345 고정으로 재현성 확보. 결과는 image_id 리스트 JSON.

사용법:
    python scripts/data_utils/stratify_exp1_3cls_val.py
출력:
    results/merged_datasets/_exp1_3cls_val_split.json
"""
import json
import random
from pathlib import Path
from collections import defaultdict

VAL_DIR = Path('/home/jjh0709/gitrepo/VISION-Instance-Seg/results/merged_datasets/_exp2_3cls_val')
VAL_ANN = VAL_DIR / 'annotations.json'
OUT_PATH = Path('/home/jjh0709/gitrepo/VISION-Instance-Seg/results/merged_datasets/_exp1_3cls_val_split.json')

SEED = 12345
DEV_RATIO = 0.5  # 50% dev / 50% test


def main():
    if not VAL_ANN.exists():
        raise FileNotFoundError(f"Val annotations not found: {VAL_ANN}")
    data = json.loads(VAL_ANN.read_text())
    images = data['images']
    annotations = data['annotations']

    # 이미지 ID → 등장 클래스 set
    img_classes = defaultdict(set)
    for ann in annotations:
        img_classes[ann['image_id']].add(ann['category_id'])

    # 이미지를 "주 클래스" 기준으로 그룹핑 (가장 작은 cat_id를 대표로)
    by_class = defaultdict(list)
    for img in images:
        cls = min(img_classes.get(img['id'], {-1}))  # -1 = no annotation
        by_class[cls].append(img['id'])

    rng = random.Random(SEED)
    dev_ids, test_ids = [], []
    for cls, ids in sorted(by_class.items()):
        ids_sorted = sorted(ids)  # 결정적 순서
        rng.shuffle(ids_sorted)
        n_dev = max(1, int(round(len(ids_sorted) * DEV_RATIO)))
        dev_ids.extend(ids_sorted[:n_dev])
        test_ids.extend(ids_sorted[n_dev:])
        print(f"  class={cls}: total={len(ids)}, dev={n_dev}, test={len(ids_sorted)-n_dev}")

    out = {
        "seed": SEED,
        "source": str(VAL_ANN),
        "total_images": len(images),
        "val_dev_ids": sorted(dev_ids),
        "val_test_ids": sorted(test_ids),
        "n_dev": len(dev_ids),
        "n_test": len(test_ids),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\n✅ saved: {OUT_PATH}")
    print(f"   dev={len(dev_ids)} / test={len(test_ids)} / total={len(images)}")


if __name__ == "__main__":
    main()
