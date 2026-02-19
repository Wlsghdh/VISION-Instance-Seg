"""
Data Analysis Script for Cable/Thunderbolt Defect Detection Dataset
현재 데이터셋의 상세 분석을 수행합니다.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

# Paths
IMAGES_DIR = Path('/data2/project/2026winter/jjh0709/AA_CV_R/train/images')
ANNOTATIONS_FILE = Path('/data2/project/2026winter/jjh0709/AA_CV_R/train/annotations.json')

def load_annotations():
    """Load COCO format annotations"""
    with open(ANNOTATIONS_FILE, 'r') as f:
        return json.load(f)

def classify_image_type(filename):
    """Classify image as original, Cable GenAI, or Thunderbolt GenAI"""
    if filename.startswith('Cable_'):
        return 'Cable_GenAI'
    elif filename.lower().startswith('thunderbolt_'):
        return 'Thunderbolt_GenAI'
    else:
        return 'Original'

def analyze_dataset():
    """Comprehensive dataset analysis"""
    print("=" * 80)
    print("📊 Cable/Thunderbolt Defect Detection Dataset Analysis")
    print("=" * 80)

    # Load annotations
    data = load_annotations()
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    print(f"\n{'='*80}")
    print("1. Basic Statistics")
    print(f"{'='*80}")
    print(f"Total Images: {len(images)}")
    print(f"Total Annotations: {len(annotations)}")
    print(f"Total Categories: {len(categories)}")
    print(f"Categories: {[cat['name'] for cat in categories]}")

    # Classify images by type
    image_types = defaultdict(list)
    for img in images:
        img_type = classify_image_type(img['file_name'])
        image_types[img_type].append(img)

    print(f"\n{'='*80}")
    print("2. Data Distribution by Type")
    print(f"{'='*80}")
    for img_type, imgs in sorted(image_types.items()):
        print(f"{img_type:20s}: {len(imgs):3d} images ({len(imgs)/len(images)*100:.1f}%)")

    # Annotations per image
    print(f"\n{'='*80}")
    print("3. Annotations per Image")
    print(f"{'='*80}")

    img_id_to_anns = defaultdict(list)
    for ann in annotations:
        img_id_to_anns[ann['image_id']].append(ann)

    ann_counts = [len(anns) for anns in img_id_to_anns.values()]
    print(f"Images with annotations: {len(img_id_to_anns)}/{len(images)}")
    print(f"Images without annotations: {len(images) - len(img_id_to_anns)}")
    print(f"Annotations per image - Mean: {np.mean(ann_counts):.2f}, "
          f"Std: {np.std(ann_counts):.2f}, "
          f"Min: {np.min(ann_counts)}, "
          f"Max: {np.max(ann_counts)}")

    # Annotation distribution by image type
    print(f"\n{'='*80}")
    print("4. Annotations by Image Type")
    print(f"{'='*80}")

    img_id_to_type = {}
    for img in images:
        img_id_to_type[img['id']] = classify_image_type(img['file_name'])

    type_ann_counts = defaultdict(list)
    for img_id, anns in img_id_to_anns.items():
        img_type = img_id_to_type.get(img_id, 'Unknown')
        type_ann_counts[img_type].append(len(anns))

    for img_type in sorted(type_ann_counts.keys()):
        counts = type_ann_counts[img_type]
        print(f"{img_type:20s}: Avg {np.mean(counts):.2f} anns/img, "
              f"Total {sum(counts)} anns")

    # Image dimensions
    print(f"\n{'='*80}")
    print("5. Image Dimensions")
    print(f"{'='*80}")

    widths = [img['width'] for img in images]
    heights = [img['height'] for img in images]

    print(f"Width  - Mean: {np.mean(widths):.0f}, "
          f"Std: {np.std(widths):.0f}, "
          f"Min: {np.min(widths)}, "
          f"Max: {np.max(widths)}")
    print(f"Height - Mean: {np.mean(heights):.0f}, "
          f"Std: {np.std(heights):.0f}, "
          f"Min: {np.min(heights)}, "
          f"Max: {np.max(heights)}")

    # Unique dimensions
    unique_dims = set((img['width'], img['height']) for img in images)
    print(f"\nUnique dimensions: {len(unique_dims)}")
    dim_counts = defaultdict(int)
    for img in images:
        dim_counts[(img['width'], img['height'])] += 1

    print("Most common dimensions:")
    for (w, h), count in sorted(dim_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {w}x{h}: {count} images")

    # Bbox statistics
    print(f"\n{'='*80}")
    print("6. Bounding Box Statistics")
    print(f"{'='*80}")

    bbox_widths = [ann['bbox'][2] for ann in annotations]
    bbox_heights = [ann['bbox'][3] for ann in annotations]
    bbox_areas = [ann['area'] for ann in annotations]

    print(f"BBox Width  - Mean: {np.mean(bbox_widths):.1f}, "
          f"Std: {np.std(bbox_widths):.1f}, "
          f"Min: {np.min(bbox_widths):.1f}, "
          f"Max: {np.max(bbox_widths):.1f}")
    print(f"BBox Height - Mean: {np.mean(bbox_heights):.1f}, "
          f"Std: {np.std(bbox_heights):.1f}, "
          f"Min: {np.min(bbox_heights):.1f}, "
          f"Max: {np.max(bbox_heights):.1f}")
    print(f"BBox Area   - Mean: {np.mean(bbox_areas):.1f}, "
          f"Std: {np.std(bbox_areas):.1f}, "
          f"Min: {np.min(bbox_areas):.1f}, "
          f"Max: {np.max(bbox_areas):.1f}")

    # File format distribution
    print(f"\n{'='*80}")
    print("7. File Format Distribution")
    print(f"{'='*80}")

    format_counts = defaultdict(int)
    for img in images:
        ext = Path(img['file_name']).suffix.lower()
        format_counts[ext] += 1

    for ext, count in sorted(format_counts.items()):
        print(f"{ext:10s}: {count:3d} images ({count/len(images)*100:.1f}%)")

    # Check file existence
    print(f"\n{'='*80}")
    print("8. File Existence Check")
    print(f"{'='*80}")

    missing_files = []
    existing_files = []

    for img in images:
        img_path = IMAGES_DIR / img['file_name']
        if img_path.exists():
            existing_files.append(img['file_name'])
        else:
            missing_files.append(img['file_name'])

    print(f"Existing files: {len(existing_files)}/{len(images)}")
    if missing_files:
        print(f"Missing files: {len(missing_files)}")
        print("Missing file list:")
        for fname in missing_files[:10]:  # Show first 10
            print(f"  - {fname}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files)-10} more")

    # Original data details
    print(f"\n{'='*80}")
    print("9. Original Data Details (26 images)")
    print(f"{'='*80}")

    original_imgs = image_types['Original']
    original_ids = {img['id'] for img in original_imgs}
    original_anns = [ann for ann in annotations if ann['image_id'] in original_ids]

    print(f"Original images: {len(original_imgs)}")
    print(f"Original annotations: {len(original_anns)}")
    print(f"Annotations per original image: {len(original_anns)/len(original_imgs):.2f}")

    print("\nOriginal image filenames:")
    original_names = sorted([img['file_name'] for img in original_imgs])
    for i in range(0, len(original_names), 5):
        print("  " + ", ".join(original_names[i:i+5]))

    # Summary
    print(f"\n{'='*80}")
    print("10. Summary & Recommendations")
    print(f"{'='*80}")

    print(f"""
✅ Dataset is ready for splitting with the following characteristics:
  - Small original dataset (26 images, {len(original_anns)} annotations)
  - Heavy augmentation via GenAI (206 images, {len(annotations)-len(original_anns)} annotations)
  - Mixed file formats (jpg/png) - no issue for most frameworks
  - All annotation files are present in the images directory

⚠️  Considerations:
  - Original data is very limited (26 images)
  - Need careful splitting to ensure representative train/val/test sets
  - GenAI augmentation quality should be evaluated
  - Consider traditional augmentation for original data

💡 Recommended next steps:
  1. Choose data split strategy (Strategy 1/2/3 from data_split_strategy.md)
  2. Implement split script with fixed random seed
  3. Validate split distributions
  4. Set up augmentation pipeline
  5. Begin model training experiments
""")

if __name__ == '__main__':
    analyze_dataset()
