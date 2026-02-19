"""
실험용 데이터셋 준비 스크립트
- GenAI 증강 데이터 샘플링
- 전통적 증강 (Flip/Rotation 포함, annotations 자동 변환)
- 각 실험별 데이터셋 생성
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict
import random
import numpy as np
from PIL import Image
import albumentations as A
from tqdm import tqdm

# Random seed 고정
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Paths
BASE_DIR = Path('/data2/project/2026winter/jjh0709/AA_CV_R')
TRAIN_IMAGES_DIR = BASE_DIR / 'train' / 'images'
TRAIN_ANNOTATIONS_FILE = BASE_DIR / 'train' / 'annotations.json'
EXPERIMENTS_DIR = BASE_DIR / 'experiments'
EXPERIMENTS_DIR.mkdir(exist_ok=True)

# GenAI 증강 데이터 양
GENAI_AMOUNTS = [50, 100, 150, 200]


def load_annotations(ann_file):
    """Load COCO format annotations"""
    with open(ann_file, 'r') as f:
        return json.load(f)


def classify_image_type(filename):
    """Classify image as original, Cable GenAI, or Thunderbolt GenAI"""
    if filename.startswith('Cable_'):
        return 'Cable_GenAI'
    elif filename.lower().startswith('thunderbolt_'):
        return 'Thunderbolt_GenAI'
    else:
        return 'Original'


def get_images_by_type(data):
    """Separate images by type"""
    images_by_type = defaultdict(list)

    for img in data['images']:
        img_type = classify_image_type(img['file_name'])
        images_by_type[img_type].append(img)

    return images_by_type


def sample_genai_images(images_by_type, amount):
    """
    Sample GenAI images (Cable + Thunderbolt)
    Cable과 Thunderbolt을 균등하게 샘플링
    """
    cable_images = images_by_type['Cable_GenAI']
    thunderbolt_images = images_by_type['Thunderbolt_GenAI']

    # 절반씩 샘플링 (홀수면 Cable에서 1개 더)
    cable_amount = (amount + 1) // 2
    thunderbolt_amount = amount // 2

    # 충분한 데이터가 있는지 확인
    cable_amount = min(cable_amount, len(cable_images))
    thunderbolt_amount = min(thunderbolt_amount, len(thunderbolt_images))

    sampled_cable = random.sample(cable_images, cable_amount)
    sampled_thunderbolt = random.sample(thunderbolt_images, thunderbolt_amount)

    return sampled_cable + sampled_thunderbolt


def create_augmentation_pipeline():
    """
    전통적 증강 파이프라인 (annotations 자동 변환 포함)
    """
    return A.Compose([
        # Geometric transformations (annotations 자동 변환)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),

        # Pixel-level transformations (annotations 불변)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
       keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def augment_image_with_annotations(img_path, annotations, aug_pipeline):
    """
    이미지와 annotations를 함께 증강

    Returns:
        augmented_image: PIL Image
        augmented_annotations: list of augmented annotation dicts
    """
    # Load image
    image = np.array(Image.open(img_path).convert('RGB'))

    # Prepare bboxes and keypoints
    bboxes = []
    category_ids = []
    keypoints_list = []

    for ann in annotations:
        # BBox: [x, y, width, height] (COCO format)
        bboxes.append(ann['bbox'])
        category_ids.append(ann['category_id'])

        # Segmentation to keypoints
        if 'segmentation' in ann and ann['segmentation']:
            # segmentation: [[x1, y1, x2, y2, ...]]
            seg = ann['segmentation'][0] if isinstance(ann['segmentation'], list) else ann['segmentation']
            keypoints = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
            keypoints_list.append(keypoints)
        else:
            keypoints_list.append([])

    # Apply augmentation
    try:
        # Flatten keypoints for albumentations
        all_keypoints = []
        keypoint_ann_indices = []
        for ann_idx, kps in enumerate(keypoints_list):
            for kp in kps:
                all_keypoints.append(kp)
                keypoint_ann_indices.append(ann_idx)

        if all_keypoints:
            transformed = aug_pipeline(
                image=image,
                bboxes=bboxes,
                category_ids=category_ids,
                keypoints=all_keypoints
            )
        else:
            # No segmentation, only bboxes
            transformed = aug_pipeline(
                image=image,
                bboxes=bboxes,
                category_ids=category_ids,
                keypoints=[]
            )

        aug_image = Image.fromarray(transformed['image'])
        aug_bboxes = transformed['bboxes']
        aug_keypoints = transformed['keypoints']

        # Reconstruct annotations
        aug_annotations = []

        # Reconstruct keypoints per annotation
        aug_keypoints_by_ann = defaultdict(list)
        for kp, ann_idx in zip(aug_keypoints, keypoint_ann_indices):
            aug_keypoints_by_ann[ann_idx].append(kp)

        for i, (bbox, cat_id) in enumerate(zip(aug_bboxes, category_ids)):
            ann = annotations[i].copy()
            ann['bbox'] = list(bbox)

            # Reconstruct segmentation
            if i in aug_keypoints_by_ann:
                kps = aug_keypoints_by_ann[i]
                segmentation = []
                for kp in kps:
                    segmentation.extend([float(kp[0]), float(kp[1])])
                ann['segmentation'] = [segmentation]

                # Recalculate area
                x, y, w, h = bbox
                ann['area'] = w * h

            aug_annotations.append(ann)

        return aug_image, aug_annotations

    except Exception as e:
        print(f"Augmentation failed: {e}")
        return None, None


def apply_traditional_augmentation(original_images, data, target_amount, output_dir):
    """
    원본 이미지에 전통적 증강을 적용하여 target_amount만큼 생성

    Args:
        original_images: list of original image dicts
        data: full COCO annotations data
        target_amount: 최종 생성할 총 이미지 수 (원본 포함)
        output_dir: 증강된 이미지를 저장할 디렉토리

    Returns:
        augmented_data: COCO format data with augmented images and annotations
    """
    aug_pipeline = create_augmentation_pipeline()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare annotation lookup
    img_id_to_anns = defaultdict(list)
    for ann in data['annotations']:
        img_id_to_anns[ann['image_id']].append(ann)

    # Calculate how many augmentations per image
    num_originals = len(original_images)
    num_augmentations_needed = target_amount - num_originals

    if num_augmentations_needed <= 0:
        print(f"Warning: target_amount ({target_amount}) <= num_originals ({num_originals})")
        return None

    augs_per_image = num_augmentations_needed // num_originals
    extra_augs = num_augmentations_needed % num_originals

    print(f"  Original images: {num_originals}")
    print(f"  Augmentations per image: {augs_per_image}")
    print(f"  Extra augmentations: {extra_augs}")

    # New data structure
    new_images = []
    new_annotations = []
    next_img_id = 0
    next_ann_id = 1

    # Copy original images first
    for orig_img in tqdm(original_images, desc="  Processing originals"):
        # Copy image file
        src_path = TRAIN_IMAGES_DIR / orig_img['file_name']
        dst_path = output_dir / orig_img['file_name']
        shutil.copy2(src_path, dst_path)

        # Add to new data
        new_img = orig_img.copy()
        new_img['id'] = next_img_id
        new_images.append(new_img)

        # Copy annotations
        orig_anns = img_id_to_anns.get(orig_img['id'], [])
        for ann in orig_anns:
            new_ann = ann.copy()
            new_ann['image_id'] = next_img_id
            new_ann['id'] = next_ann_id
            new_annotations.append(new_ann)
            next_ann_id += 1

        next_img_id += 1

    # Generate augmented images
    for idx, orig_img in enumerate(tqdm(original_images, desc="  Generating augmentations")):
        src_path = TRAIN_IMAGES_DIR / orig_img['file_name']
        orig_anns = img_id_to_anns.get(orig_img['id'], [])

        # Determine number of augmentations for this image
        num_augs = augs_per_image
        if idx < extra_augs:
            num_augs += 1

        # Generate augmentations
        for aug_idx in range(num_augs):
            aug_image, aug_anns = augment_image_with_annotations(
                src_path, orig_anns, aug_pipeline
            )

            if aug_image is None:
                continue

            # Save augmented image
            base_name = Path(orig_img['file_name']).stem
            ext = Path(orig_img['file_name']).suffix
            aug_filename = f"{base_name}_aug{aug_idx}{ext}"
            aug_path = output_dir / aug_filename
            aug_image.save(aug_path)

            # Add to new data
            new_img = {
                'id': next_img_id,
                'file_name': aug_filename,
                'height': aug_image.height,
                'width': aug_image.width,
            }
            new_images.append(new_img)

            # Add augmented annotations
            for ann in aug_anns:
                new_ann = ann.copy()
                new_ann['image_id'] = next_img_id
                new_ann['id'] = next_ann_id
                new_annotations.append(new_ann)
                next_ann_id += 1

            next_img_id += 1

    # Create new COCO data
    augmented_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': data['categories']
    }

    return augmented_data


def create_experiment_dataset(exp_name, images_list, data, output_dir):
    """
    Create experiment dataset by copying images and creating annotations
    """
    exp_dir = Path(output_dir)
    exp_images_dir = exp_dir / 'images'
    exp_images_dir.mkdir(parents=True, exist_ok=True)

    # Get image IDs
    img_ids = {img['id'] for img in images_list}

    # Copy images
    for img in images_list:
        src_path = TRAIN_IMAGES_DIR / img['file_name']
        dst_path = exp_images_dir / img['file_name']
        if src_path.exists():
            shutil.copy2(src_path, dst_path)

    # Filter annotations
    filtered_anns = [ann for ann in data['annotations'] if ann['image_id'] in img_ids]

    # Create new COCO data
    exp_data = {
        'images': images_list,
        'annotations': filtered_anns,
        'categories': data['categories']
    }

    # Save annotations
    ann_path = exp_dir / 'annotations.json'
    with open(ann_path, 'w') as f:
        json.dump(exp_data, f, indent=2)

    return exp_data


def main():
    print("=" * 80)
    print("실험용 데이터셋 준비")
    print("=" * 80)

    # Load training data
    print("\n1. Loading training data...")
    data = load_annotations(TRAIN_ANNOTATIONS_FILE)
    images_by_type = get_images_by_type(data)

    original_images = images_by_type['Original']
    cable_images = images_by_type['Cable_GenAI']
    thunderbolt_images = images_by_type['Thunderbolt_GenAI']

    print(f"   Original: {len(original_images)}")
    print(f"   Cable GenAI: {len(cable_images)}")
    print(f"   Thunderbolt GenAI: {len(thunderbolt_images)}")

    # ========================================================================
    # 실험 1: GenAI 증강 데이터 양에 따른 성능 변화
    # ========================================================================
    print("\n" + "=" * 80)
    print("실험 1: GenAI 증강 데이터 양에 따른 성능 변화")
    print("=" * 80)

    for amount in GENAI_AMOUNTS:
        exp_name = f"exp_1_original26_genai{amount}"
        exp_dir = EXPERIMENTS_DIR / exp_name

        print(f"\n[{exp_name}]")
        print(f"  Sampling {amount} GenAI images...")

        # Sample GenAI images
        sampled_genai = sample_genai_images(images_by_type, amount)

        # Combine with originals
        combined_images = original_images + sampled_genai

        print(f"  Total images: {len(combined_images)}")
        print(f"  Creating dataset...")

        # Create dataset
        create_experiment_dataset(exp_name, combined_images, data, exp_dir)

        print(f"  ✓ Saved to {exp_dir}")

    # ========================================================================
    # 실험 2: 증강 방법별 비교
    # ========================================================================
    print("\n" + "=" * 80)
    print("실험 2: 증강 방법별 비교")
    print("=" * 80)

    # Baseline: 원본 26장만
    print(f"\n[Baseline: Original 26 only]")
    exp_name = "exp_2_original26_only"
    exp_dir = EXPERIMENTS_DIR / exp_name
    create_experiment_dataset(exp_name, original_images, data, exp_dir)
    print(f"  ✓ Saved to {exp_dir}")

    # 각 데이터 양에 대해
    for amount in GENAI_AMOUNTS:
        print(f"\n{'='*80}")
        print(f"데이터 양: {amount}장")
        print(f"{'='*80}")

        # A: 원본 26장만 (이미 위에서 생성)
        # Skip

        # B: 원본 26장 + GenAI X장 (이미 실험 1에서 생성)
        # Skip

        # C: 원본 26장 → 전통적 증강으로 총 X장
        print(f"\n[C: Original 26 → Traditional augmentation → {amount} total]")
        exp_name = f"exp_2_original26_traditional{amount}"
        exp_dir = EXPERIMENTS_DIR / exp_name
        exp_images_dir = exp_dir / 'images'

        print(f"  Applying traditional augmentation...")
        aug_data = apply_traditional_augmentation(
            original_images, data, amount, exp_images_dir
        )

        if aug_data:
            # Save annotations
            ann_path = exp_dir / 'annotations.json'
            with open(ann_path, 'w') as f:
                json.dump(aug_data, f, indent=2)

            print(f"  Total images: {len(aug_data['images'])}")
            print(f"  Total annotations: {len(aug_data['annotations'])}")
            print(f"  ✓ Saved to {exp_dir}")

        # D: 원본 26장 + GenAI X장 + 원본에 전통적 증강
        print(f"\n[D: Original 26 + GenAI {amount} + Traditional augmentation on originals]")
        exp_name = f"exp_2_original26_genai{amount}_traditional"
        exp_dir = EXPERIMENTS_DIR / exp_name
        exp_images_dir = exp_dir / 'images'
        exp_images_dir.mkdir(parents=True, exist_ok=True)

        # Sample GenAI images (same as before)
        sampled_genai = sample_genai_images(images_by_type, amount)

        # Apply traditional augmentation to originals (same amount as originals = 26)
        print(f"  Applying traditional augmentation to originals...")
        aug_data_originals = apply_traditional_augmentation(
            original_images, data, 26 * 2, exp_images_dir  # 2x augmentation
        )

        if aug_data_originals:
            # Copy GenAI images
            print(f"  Copying {len(sampled_genai)} GenAI images...")
            genai_img_ids = {img['id'] for img in sampled_genai}
            genai_anns = [ann for ann in data['annotations'] if ann['image_id'] in genai_img_ids]

            next_img_id = len(aug_data_originals['images'])
            next_ann_id = max([ann['id'] for ann in aug_data_originals['annotations']]) + 1

            for img in sampled_genai:
                src_path = TRAIN_IMAGES_DIR / img['file_name']
                dst_path = exp_images_dir / img['file_name']
                shutil.copy2(src_path, dst_path)

                new_img = img.copy()
                new_img['id'] = next_img_id
                aug_data_originals['images'].append(new_img)

                # Copy annotations
                img_anns = [ann for ann in genai_anns if ann['image_id'] == img['id']]
                for ann in img_anns:
                    new_ann = ann.copy()
                    new_ann['image_id'] = next_img_id
                    new_ann['id'] = next_ann_id
                    aug_data_originals['annotations'].append(new_ann)
                    next_ann_id += 1

                next_img_id += 1

            # Save annotations
            ann_path = exp_dir / 'annotations.json'
            with open(ann_path, 'w') as f:
                json.dump(aug_data_originals, f, indent=2)

            print(f"  Total images: {len(aug_data_originals['images'])}")
            print(f"  Total annotations: {len(aug_data_originals['annotations'])}")
            print(f"  ✓ Saved to {exp_dir}")

    print("\n" + "=" * 80)
    print("✅ All experiments prepared!")
    print("=" * 80)
    print(f"\nExperiments directory: {EXPERIMENTS_DIR}")
    print("\nNext steps:")
    print("  1. Review generated datasets")
    print("  2. Configure model training pipeline")
    print("  3. Run experiments with same hyperparameters")
    print("  4. Compare results")


if __name__ == '__main__':
    main()
