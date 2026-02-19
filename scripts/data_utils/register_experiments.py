"""
AA_CV_R의 13개 실험 데이터셋을 Detectron2/MaskDINO에 등록하는 스크립트
"""

import os
from pathlib import Path
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

# 데이터셋 경로
BASE_DIR = Path("/data2/project/2026winter/jjh0709/AA_CV_R")
EXPERIMENTS_DIR = BASE_DIR / "experiments"
VAL_DIR = BASE_DIR / "val"

# 카테고리 정보 (단일 클래스)
THING_CLASSES = ["thunderbolt"]
THING_COLORS = [(255, 0, 0)]  # Red for visualization

# 13개 실험 리스트
EXPERIMENT_NAMES = [
    # 실험 1: GenAI 증강 데이터 양
    'exp_1_original26_genai50',
    'exp_1_original26_genai100',
    'exp_1_original26_genai150',
    'exp_1_original26_genai200',

    # 실험 2: 증강 방법 비교
    'exp_2_original26_only',  # Baseline
    'exp_2_original26_traditional50',
    'exp_2_original26_traditional100',
    'exp_2_original26_traditional150',
    'exp_2_original26_traditional200',
    'exp_2_original26_genai50_traditional',
    'exp_2_original26_genai100_traditional',
    'exp_2_original26_genai150_traditional',
    'exp_2_original26_genai200_traditional',
]


def register_experiment_dataset(exp_name):
    """단일 실험 데이터셋 등록"""

    # Train 데이터셋
    train_name = f"{exp_name}_train"
    train_img_dir = str(EXPERIMENTS_DIR / exp_name / "images")
    train_ann_file = str(EXPERIMENTS_DIR / exp_name / "annotations.json")

    # Test 데이터셋 (모든 실험이 동일한 test set 사용)
    test_name = f"{exp_name}_test"
    test_img_dir = str(VAL_DIR / "images")
    test_ann_file = str(VAL_DIR / "annotations.json")

    # 파일 존재 확인
    if not os.path.exists(train_ann_file):
        print(f"❌ Annotation file not found: {train_ann_file}")
        return False

    if not os.path.exists(train_img_dir):
        print(f"❌ Image directory not found: {train_img_dir}")
        return False

    # Train 데이터셋 등록
    if train_name in DatasetCatalog.list():
        DatasetCatalog.remove(train_name)
        MetadataCatalog.remove(train_name)

    DatasetCatalog.register(
        train_name,
        lambda: load_coco_json(train_ann_file, train_img_dir, train_name)
    )

    MetadataCatalog.get(train_name).set(
        thing_classes=THING_CLASSES,
        thing_colors=THING_COLORS,
        json_file=train_ann_file,
        image_root=train_img_dir,
        evaluator_type="coco",
    )

    # Test 데이터셋 등록
    if test_name in DatasetCatalog.list():
        DatasetCatalog.remove(test_name)
        MetadataCatalog.remove(test_name)

    DatasetCatalog.register(
        test_name,
        lambda: load_coco_json(test_ann_file, test_img_dir, test_name)
    )

    MetadataCatalog.get(test_name).set(
        thing_classes=THING_CLASSES,
        thing_colors=THING_COLORS,
        json_file=test_ann_file,
        image_root=test_img_dir,
        evaluator_type="coco",
    )

    # 데이터셋 정보 출력
    train_dataset = DatasetCatalog.get(train_name)
    test_dataset = DatasetCatalog.get(test_name)

    print(f"✓ Registered: {train_name}")
    print(f"    Train images: {len(train_dataset)}")
    print(f"    Test images: {len(test_dataset)}")

    return True


def register_all_experiments():
    """모든 실험 데이터셋 등록"""
    print("=" * 80)
    print("Registering AA_CV_R Experiment Datasets for MaskDINO")
    print("=" * 80)

    success_count = 0
    fail_count = 0

    for exp_name in EXPERIMENT_NAMES:
        print(f"\n[{exp_name}]")
        if register_experiment_dataset(exp_name):
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 80)
    print(f"✅ Registration Complete!")
    print(f"   Success: {success_count}/{len(EXPERIMENT_NAMES)}")
    if fail_count > 0:
        print(f"   Failed: {fail_count}/{len(EXPERIMENT_NAMES)}")
    print("=" * 80)

    return success_count, fail_count


def get_dataset_names(exp_name):
    """실험 이름에서 train/test 데이터셋 이름 반환"""
    return f"{exp_name}_train", f"{exp_name}_test"


def list_registered_datasets():
    """등록된 모든 데이터셋 목록 출력"""
    all_datasets = DatasetCatalog.list()
    experiment_datasets = [name for name in all_datasets if 'exp_' in name]

    print("\n" + "=" * 80)
    print("Registered Experiment Datasets")
    print("=" * 80)

    train_datasets = sorted([name for name in experiment_datasets if '_train' in name])
    test_datasets = sorted([name for name in experiment_datasets if '_test' in name])

    print(f"\nTrain Datasets ({len(train_datasets)}):")
    for name in train_datasets:
        dataset = DatasetCatalog.get(name)
        print(f"  - {name}: {len(dataset)} images")

    print(f"\nTest Datasets ({len(test_datasets)}):")
    for name in test_datasets:
        dataset = DatasetCatalog.get(name)
        print(f"  - {name}: {len(dataset)} images")

    print("=" * 80)


if __name__ == "__main__":
    # 모든 실험 데이터셋 등록
    register_all_experiments()

    # 등록된 데이터셋 목록 출력
    list_registered_datasets()
