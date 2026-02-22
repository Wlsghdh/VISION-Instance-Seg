"""
전통적 증강 스크립트 (Traditional Augmentation)
- Albumentations 기반 증강 (segmentation polygon 자동 변환 포함)
- 마스크 기반 증강으로 polygon 정확도 보장
- 기본 출력: data_augmented/{category}/traditional_aug/{images/, annotations.json}

사용법 (기본 - 카테고리 원본에서 증강):
    python scripts/augmentation/traditional_augment.py \
        --category Screw \
        --n_augment 2750 \
        --seed 42

사용법 (커스텀 입출력 - 실험 조건 5 등):
    python scripts/augmentation/traditional_augment.py \
        --category Screw \
        --input_dir /path/to/merged/images \
        --ann_file  /path/to/merged/annotations.json \
        --output_dir /path/to/output \
        --n_augment 2750 \
        --seed 42
"""

import argparse
import json
import os
import random
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
import albumentations as A
from tqdm import tqdm


# ============================================================
# 카테고리 설정
# ============================================================
CATEGORY_CONFIG = {
    "Cable":   {"keep_category": "thunderbolt", "keep_id": 1},
    "Screw":   {"keep_category": "defect",      "keep_id": 0},  # Screw defect = id 0
    "Casting": {"keep_category": None,          "keep_id": None},  # Inclusoes(0)+Rechupe(1) 전체 유지
}


# ============================================================
# Augmentation Pipeline
# ============================================================
def create_augmentation_pipeline():
    """
    Albumentations 2.x 호환 증강 파이프라인

    적용 증강 종류:
      - 상하좌우반전 (HorizontalFlip, VerticalFlip)
      - 회전 (Rotate, ShiftScaleRotate)
      - 색채도 변환 (HueSaturationValue, RandomBrightnessContrast)
      - 노이즈 (GaussNoise, GaussianBlur)
      - 회색마스킹 (CoarseDropout — 회색(128) 패치로 랜덤 영역 마스킹)
    """
    return A.Compose([
        # ── 상하좌우 반전 ─────────────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),

        # ── 회전 ──────────────────────────────────────────────
        A.Rotate(limit=20, p=0.6),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            fill=0,
            p=0.4
        ),

        # ── 색채도 변환 ───────────────────────────────────────
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),

        # ── 노이즈 ────────────────────────────────────────────
        A.GaussNoise(std_range=(0.02, 0.12), p=0.4),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),

        # ── 회색마스킹 (CoarseDropout) ────────────────────────
        # 이미지 일부 영역을 회색(128)으로 덮어 노출 강건성 향상
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(0.05, 0.15),
            hole_width_range=(0.05, 0.15),
            fill=128,
            p=0.4
        ),
    ])


# ============================================================
# Polygon <-> Mask 변환
# ============================================================
def polygon_to_mask(segmentation, height, width):
    """COCO polygon segmentation → binary mask"""
    mask = np.zeros((height, width), dtype=np.uint8)
    for seg in segmentation:
        pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
        pts = pts.astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_to_polygon(mask):
    """binary mask → COCO polygon segmentation [[x1,y1,...]]"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # 가장 큰 contour 사용
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1:
        return None
    # 단순화
    epsilon = 0.005 * cv2.arcLength(largest, True)
    simplified = cv2.approxPolyDP(largest, epsilon, True)
    flat = simplified.flatten().tolist()
    if len(flat) < 6:  # 최소 3점 필요
        return None
    return [flat]


def compute_bbox_from_mask(mask):
    """mask → COCO bbox [x, y, w, h]"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]


# ============================================================
# 단일 이미지 증강
# ============================================================
def augment_single(img_path, annotations, aug_pipeline):
    """
    하나의 이미지와 그 annotations를 증강.
    마스크 기반으로 segmentation polygon 자동 변환.

    Returns:
        (aug_pil_image, aug_annotations_list) or (None, None) on failure
    """
    try:
        image = np.array(Image.open(img_path).convert('RGB'))
        h, w = image.shape[:2]

        # 각 annotation별 마스크 생성
        masks = []
        valid_anns = []
        for ann in annotations:
            if 'segmentation' not in ann or not ann['segmentation']:
                continue
            m = polygon_to_mask(ann['segmentation'], h, w)
            if m.sum() == 0:
                continue
            masks.append(m)
            valid_anns.append(ann)

        if not masks:
            return None, None

        # albumentations: image + 다중 마스크
        aug_input = {'image': image}
        for i, m in enumerate(masks):
            aug_input[f'mask{i}'] = m

        # Additional targets 설정
        additional_targets = {f'mask{i}': 'mask' for i in range(len(masks))}
        pipeline = A.Compose(
            aug_pipeline.transforms,
            additional_targets=additional_targets
        )

        result = pipeline(**aug_input)

        aug_image = result['image']
        aug_h, aug_w = aug_image.shape[:2]

        # 증강된 마스크에서 polygon 복원
        aug_annotations = []
        for i, ann in enumerate(valid_anns):
            aug_mask = result[f'mask{i}']
            seg = mask_to_polygon(aug_mask)
            bbox = compute_bbox_from_mask(aug_mask)
            if seg is None or bbox is None:
                continue
            x, y, bw, bh = bbox
            aug_ann = {
                'category_id': ann['category_id'],
                'segmentation': seg,
                'bbox': bbox,
                'area': int(bw * bh),
                'iscrowd': 0,
            }
            aug_annotations.append(aug_ann)

        if not aug_annotations:
            return None, None

        aug_pil = Image.fromarray(aug_image)
        return aug_pil, aug_annotations

    except Exception as e:
        print(f"  [WARNING] Augmentation failed for {img_path}: {e}")
        return None, None


# ============================================================
# 메인 증강 함수
# ============================================================
def run_augmentation(category, n_augment, seed,
                     input_dir=None, ann_file=None, output_dir=None):
    """
    Args:
        category   : Cable / Screw / Casting (카테고리 필터 설정용)
        n_augment  : 생성할 이미지 수
        seed       : 랜덤 시드
        input_dir  : 이미지 디렉토리 (None → data/{category}/train/images)
        ann_file   : annotations.json 경로 (None → data/{category}/train/annotations.json)
        output_dir : 출력 디렉토리 (None → data_augmented/{category}/traditional_aug)
    """
    BASE = Path('/home/jjh0709/gitrepo/VISION-Instance-Seg')

    if input_dir is None:
        IMAGES_DIR = BASE / 'data' / category / 'train' / 'images'
    else:
        IMAGES_DIR = Path(input_dir)

    if ann_file is None:
        ANN_FILE = BASE / 'data' / category / 'train' / 'annotations.json'
    else:
        ANN_FILE = Path(ann_file)

    if output_dir is None:
        OUT_DIR = BASE / 'data_augmented' / category / 'traditional_aug'
    else:
        OUT_DIR = Path(output_dir)

    OUT_IMAGES = OUT_DIR / 'images'
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    # 시드 고정
    random.seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"전통적 증강: {category}")
    print(f"  이미지 입력: {IMAGES_DIR}")
    print(f"  Annotation : {ANN_FILE}")
    print(f"  출력: {OUT_DIR}")
    print(f"  생성 목표: {n_augment}장")
    print(f"  시드: {seed}")
    print(f"{'='*60}")

    # 어노테이션 로드
    with open(ANN_FILE) as f:
        data = json.load(f)

    # 카테고리 설정
    cat_cfg = CATEGORY_CONFIG.get(category, {"keep_category": None, "keep_id": None})
    keep_id = cat_cfg["keep_id"]

    # 이미지별 annotation 인덱싱
    img_id_to_anns = defaultdict(list)
    for ann in data['annotations']:
        if keep_id is None or ann['category_id'] == keep_id:
            img_id_to_anns[ann['image_id']].append(ann)

    # 유효 이미지 (annotation 있는 것만)
    valid_images = [
        img for img in data['images']
        if img_id_to_anns.get(img['id'])
    ]
    print(f"증강 소스 이미지: {len(valid_images)}장")

    if not valid_images:
        print("[ERROR] 증강할 이미지가 없습니다.")
        return

    aug_pipeline = create_augmentation_pipeline()

    # 이미지당 증강 횟수 계산
    n_base = n_augment // len(valid_images)
    n_extra = n_augment % len(valid_images)

    # 랜덤 셔플
    shuffled = valid_images[:]
    random.shuffle(shuffled)

    new_images = []
    new_annotations = []
    next_img_id = 0
    next_ann_id = 1

    total_generated = 0
    target = n_augment

    pbar = tqdm(total=target, desc="증강 생성")

    for img_idx, orig_img in enumerate(shuffled):
        img_path = Path(IMAGES_DIR) / orig_img['file_name']
        if not img_path.exists():
            print(f"  [SKIP] 파일 없음: {img_path}")
            continue

        anns = img_id_to_anns[orig_img['id']]
        n_this = n_base + (1 if img_idx < n_extra else 0)

        generated_this = 0
        max_attempts = n_this * 5  # 실패 대비 여유
        attempt = 0

        while generated_this < n_this and attempt < max_attempts:
            attempt += 1
            aug_img, aug_anns = augment_single(img_path, anns, aug_pipeline)
            if aug_img is None:
                continue

            # 파일명: {원본stem}_aug{idx}.jpg
            stem = Path(orig_img['file_name']).stem
            ext = '.jpg'
            aug_fname = f"{stem}_aug{generated_this:04d}{ext}"
            aug_path = OUT_IMAGES / aug_fname
            aug_img.save(str(aug_path), quality=95)

            new_images.append({
                'id': next_img_id,
                'file_name': aug_fname,
                'width': aug_img.width,
                'height': aug_img.height,
            })

            for aug_ann in aug_anns:
                aug_ann['id'] = next_ann_id
                aug_ann['image_id'] = next_img_id
                new_annotations.append(aug_ann)
                next_ann_id += 1

            next_img_id += 1
            generated_this += 1
            total_generated += 1
            pbar.update(1)

            if total_generated >= target:
                break

        if total_generated >= target:
            break

    pbar.close()

    # 부족분 처리 (rare case)
    if total_generated < target:
        print(f"  [WARNING] 목표 {target}장 중 {total_generated}장만 생성됨")

    # COCO annotations.json 저장
    output_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': data['categories'],
    }

    ann_out = OUT_DIR / 'annotations.json'
    with open(ann_out, 'w') as f:
        json.dump(output_data, f)

    print(f"\n결과:")
    print(f"  생성된 이미지: {len(new_images)}장")
    print(f"  생성된 annotation: {len(new_annotations)}개")
    print(f"  저장 경로: {OUT_DIR}")
    print(f"  annotations.json: {ann_out}")


# ============================================================
# Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='전통적 증강 스크립트 - Albumentations 기반',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 (원본 train 데이터에서 증강)
  python scripts/augmentation/traditional_augment.py \\
      --category Screw \\
      --n_augment 2750 \\
      --seed 42

  # 커스텀 입출력 (실험 조건 5: 원본+gen_ai 합친 데이터에 전통 증강)
  python scripts/augmentation/traditional_augment.py \\
      --category Screw \\
      --input_dir /path/to/merged/images \\
      --ann_file  /path/to/merged/annotations.json \\
      --output_dir /path/to/output_dir \\
      --n_augment 2750 \\
      --seed 42
        """
    )
    parser.add_argument('--category', type=str, required=True,
                        choices=['Cable', 'Screw', 'Casting'],
                        help='대상 카테고리 (Cable / Screw / Casting) — 카테고리 ID 필터에 사용')
    parser.add_argument('--n_augment', type=int, default=2750,
                        help='생성할 증강 이미지 수 (기본: 2750)')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드 (기본: 42)')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='이미지 입력 디렉토리 (기본: data/{category}/train/images)')
    parser.add_argument('--ann_file', type=str, default=None,
                        help='annotations.json 경로 (기본: data/{category}/train/annotations.json)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='출력 디렉토리 (기본: data_augmented/{category}/traditional_aug)')
    args = parser.parse_args()

    run_augmentation(
        args.category, args.n_augment, args.seed,
        input_dir=args.input_dir,
        ann_file=args.ann_file,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
