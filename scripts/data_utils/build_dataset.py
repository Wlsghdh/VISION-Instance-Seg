"""
Cable Thunderbolt 데이터셋 정리 및 통합
- 원본에서 thunderbolt만 추출
- 증강 데이터 추가
- train/val 구조로 정리
"""

import os
import json
import shutil
from pathlib import Path

# 경로 설정
ORIGINAL_DIR = "/data2/project/2026winter/jjh0709/AA_CV_O/Cable"
AUGMENTED_IMG_DIR = "/data2/project/2025summer/mym470/industrial/VISION-Datasets/Cable/cable_images_1/thunderbolts"
AUGMENTED_JSON = "/data2/project/2025summer/mym470/industrial/VISION-Datasets/Cable/cable_images_1/annotations/annotation_thunderbolt_000100.json"
OUTPUT_DIR = "/data2/project/2026winter/jjh0709/AA_CV_R"

def setup_directories():
    """출력 디렉토리 구조 생성"""
    for split in ["train", "val"]:
        os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
    print(f"✓ 디렉토리 생성: {OUTPUT_DIR}")

def load_original_annotations(split):
    """원본 annotation 로드"""
    ann_path = f"{ORIGINAL_DIR}/{split}/_annotations.coco.json"
    with open(ann_path, 'r') as f:
        return json.load(f)

def get_thunderbolt_category_id(coco_data):
    """thunderbolt category ID 찾기"""
    for cat in coco_data['categories']:
        cat_name = cat['name'].lower()
        if 'thunderbolt' in cat_name or 'thunder' in cat_name:
            print(f"  Found category: {cat['name']} (id={cat['id']})")
            return cat['id']
    return None

def filter_thunderbolt_data(coco_data, split):
    """thunderbolt만 필터링"""
    # Category 확인
    print(f"\n[{split}] Categories in original data:")
    for cat in coco_data['categories']:
        print(f"  - {cat['name']} (id={cat['id']})")
    
    thunderbolt_cat_id = get_thunderbolt_category_id(coco_data)
    
    if thunderbolt_cat_id is None:
        print(f"  WARNING: thunderbolt category not found, trying to find by inspection...")
        # annotation에서 category 분포 확인
        cat_counts = {}
        for ann in coco_data['annotations']:
            cat_id = ann['category_id']
            cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
        print(f"  Category distribution: {cat_counts}")
        
        # 첫 번째 카테고리가 아닌 것 선택 (보통 break가 먼저)
        for cat in coco_data['categories']:
            if 'break' not in cat['name'].lower():
                thunderbolt_cat_id = cat['id']
                print(f"  Using category: {cat['name']} (id={cat['id']})")
                break
    
    if thunderbolt_cat_id is None:
        print("  ERROR: Could not determine thunderbolt category")
        return None, [], []
    
    # thunderbolt annotation만 필터링
    filtered_anns = [ann for ann in coco_data['annotations'] if ann['category_id'] == thunderbolt_cat_id]
    
    # 해당 이미지 ID 수집
    image_ids_with_thunderbolt = set(ann['image_id'] for ann in filtered_anns)
    
    # 이미지 필터링
    filtered_images = [img for img in coco_data['images'] if img['id'] in image_ids_with_thunderbolt]
    
    print(f"  Filtered: {len(filtered_images)} images, {len(filtered_anns)} annotations")
    
    return thunderbolt_cat_id, filtered_images, filtered_anns

def process_original_data():
    """원본 데이터에서 thunderbolt만 추출"""
    results = {}
    
    for split in ["train", "val"]:
        print(f"\n{'='*50}")
        print(f"Processing original {split} data...")
        print(f"{'='*50}")
        
        coco_data = load_original_annotations(split)
        cat_id, filtered_images, filtered_anns = filter_thunderbolt_data(coco_data, split)
        
        if cat_id is None:
            print(f"  Skipping {split} due to category issue")
            continue
        
        # 이미지 복사
        copied_images = []
        for img in filtered_images:
            src = f"{ORIGINAL_DIR}/{split}/{img['file_name']}"
            dst = f"{OUTPUT_DIR}/{split}/images/{img['file_name']}"
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied_images.append(img)
            else:
                print(f"  WARNING: Image not found: {src}")
        
        print(f"  Copied {len(copied_images)} images to {OUTPUT_DIR}/{split}/images/")
        
        results[split] = {
            'images': copied_images,
            'annotations': filtered_anns,
            'category_id': cat_id,
            'categories': [cat for cat in coco_data['categories'] if cat['id'] == cat_id]
        }
    
    return results

def process_augmented_data(original_results):
    """증강 데이터 추가 (train only)"""
    print(f"\n{'='*50}")
    print("Processing augmented data...")
    print(f"{'='*50}")
    
    # 증강 JSON 로드
    with open(AUGMENTED_JSON, 'r') as f:
        aug_data = json.load(f)
    
    print(f"  Augmented JSON: {len(aug_data.get('images', []))} images, {len(aug_data.get('annotations', []))} annotations")
    print(f"  Categories: {aug_data.get('categories', [])}")
    
    # 증강 이미지 목록 (실제 존재하는 파일)
    aug_images_available = set(os.listdir(AUGMENTED_IMG_DIR))
    print(f"  Available augmented images: {len(aug_images_available)}")
    
    # 기존 train 데이터
    train_images = original_results.get('train', {}).get('images', [])
    train_anns = original_results.get('train', {}).get('annotations', [])
    
    # 새 ID 시작점 계산
    max_img_id = max([img['id'] for img in train_images], default=0)
    max_ann_id = max([ann['id'] for ann in train_anns], default=0)
    
    print(f"  Current max image_id: {max_img_id}, max annotation_id: {max_ann_id}")
    
    # 증강 데이터에서 thunderbolt 이미지만 필터링
    # thunderbolt_000001.png ~ thunderbolt_000100.png
    aug_img_id_map = {}  # old_id -> new_id
    added_images = []
    added_anns = []
    
    for img in aug_data.get('images', []):
        filename = img['file_name']
        # 파일명이 thunderbolt_XXXXXX.png 형식인지 확인
        if filename in aug_images_available:
            # 새 ID 할당
            max_img_id += 1
            new_img = img.copy()
            old_id = new_img['id']
            new_img['id'] = max_img_id
            aug_img_id_map[old_id] = max_img_id
            
            # 이미지 복사
            src = f"{AUGMENTED_IMG_DIR}/{filename}"
            dst = f"{OUTPUT_DIR}/train/images/{filename}"
            shutil.copy2(src, dst)
            
            added_images.append(new_img)
    
    print(f"  Added {len(added_images)} augmented images")
    
    # 해당 annotation 추가
    for ann in aug_data.get('annotations', []):
        if ann['image_id'] in aug_img_id_map:
            max_ann_id += 1
            new_ann = ann.copy()
            new_ann['id'] = max_ann_id
            new_ann['image_id'] = aug_img_id_map[ann['image_id']]
            # category_id는 1로 통일 (thunderbolt)
            new_ann['category_id'] = 1
            added_anns.append(new_ann)
    
    print(f"  Added {len(added_anns)} augmented annotations")
    
    return added_images, added_anns

def create_final_annotations(original_results, aug_images, aug_anns):
    """최종 annotation JSON 생성"""
    print(f"\n{'='*50}")
    print("Creating final annotations...")
    print(f"{'='*50}")
    
    # Category 정의 (thunderbolt만)
    categories = [{"id": 1, "name": "thunderbolt", "supercategory": "defect"}]
    
    for split in ["train", "val"]:
        print(f"\n[{split}]")
        
        images = original_results.get(split, {}).get('images', [])
        annotations = original_results.get(split, {}).get('annotations', [])
        
        # train에 증강 데이터 추가
        if split == "train":
            images = images + aug_images
            annotations = annotations + aug_anns
        
        # ID 재정렬
        img_id_map = {}
        final_images = []
        for new_id, img in enumerate(images, start=1):
            old_id = img['id']
            img_id_map[old_id] = new_id
            new_img = img.copy()
            new_img['id'] = new_id
            final_images.append(new_img)
        
        final_anns = []
        for new_id, ann in enumerate(annotations, start=1):
            new_ann = ann.copy()
            new_ann['id'] = new_id
            if ann['image_id'] in img_id_map:
                new_ann['image_id'] = img_id_map[ann['image_id']]
            new_ann['category_id'] = 1  # thunderbolt
            final_anns.append(new_ann)
        
        # JSON 저장
        final_coco = {
            "images": final_images,
            "annotations": final_anns,
            "categories": categories
        }
        
        output_path = f"{OUTPUT_DIR}/{split}/annotations.json"
        with open(output_path, 'w') as f:
            json.dump(final_coco, f, indent=2)
        
        print(f"  Saved: {output_path}")
        print(f"  Images: {len(final_images)}, Annotations: {len(final_anns)}")

def verify_dataset():
    """데이터셋 검증"""
    print(f"\n{'='*50}")
    print("Verifying dataset...")
    print(f"{'='*50}")
    
    for split in ["train", "val"]:
        img_dir = f"{OUTPUT_DIR}/{split}/images"
        ann_path = f"{OUTPUT_DIR}/{split}/annotations.json"
        
        # 이미지 수
        images = os.listdir(img_dir)
        
        # Annotation 로드
        with open(ann_path, 'r') as f:
            coco = json.load(f)
        
        # 검증
        ann_images = set(img['file_name'] for img in coco['images'])
        actual_images = set(images)
        
        missing = ann_images - actual_images
        extra = actual_images - ann_images
        
        print(f"\n[{split}]")
        print(f"  Images in folder: {len(actual_images)}")
        print(f"  Images in JSON: {len(ann_images)}")
        print(f"  Annotations: {len(coco['annotations'])}")
        print(f"  Categories: {coco['categories']}")
        
        if missing:
            print(f"  WARNING: Missing images: {list(missing)[:5]}...")
        if extra:
            print(f"  WARNING: Extra images not in JSON: {list(extra)[:5]}...")
        
        if not missing and not extra:
            print(f"  ✓ All images matched!")

def main():
    print("="*60)
    print("Cable Thunderbolt Dataset Preparation")
    print("="*60)
    
    # 1. 디렉토리 생성
    setup_directories()
    
    # 2. 원본 데이터 처리
    original_results = process_original_data()
    
    # 3. 증강 데이터 처리
    aug_images, aug_anns = process_augmented_data(original_results)
    
    # 4. 최종 annotation 생성
    create_final_annotations(original_results, aug_images, aug_anns)
    
    # 5. 검증
    verify_dataset()
    
    print("\n" + "="*60)
    print(f"✅ 완료! 데이터셋 위치: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()