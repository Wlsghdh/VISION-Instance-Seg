#!/usr/bin/env python3
"""
Annotation JSON의 image_id를 0부터 연속적으로 재정렬
Detectron2 COCO evaluation 호환성 문제 해결
"""

import json
import os
import shutil

BASE_DIR = "/data2/project/2026winter/jjh0709/AA_CV_R"

def fix_image_ids(json_path):
    """image_id를 0부터 연속적으로 재정렬"""
    
    # 백업
    backup_path = json_path + ".backup_imgid"
    if not os.path.exists(backup_path):
        shutil.copy(json_path, backup_path)
        print(f"[INFO] Backup created: {backup_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 기존 image_id → 새 image_id 매핑
    old_to_new = {}
    for new_id, img in enumerate(data['images']):
        old_id = img['id']
        old_to_new[old_id] = new_id
        img['id'] = new_id
    
    print(f"[INFO] Remapped {len(old_to_new)} image IDs")
    print(f"  Example: {list(old_to_new.items())[:3]}")
    
    # annotations의 image_id도 업데이트
    for anno in data['annotations']:
        old_img_id = anno['image_id']
        if old_img_id in old_to_new:
            anno['image_id'] = old_to_new[old_img_id]
        else:
            print(f"[WARNING] image_id {old_img_id} not found in images!")
    
    # 저장
    with open(json_path, 'w') as f:
        json.dump(data, f)
    
    print(f"[INFO] Saved: {json_path}")
    
    return old_to_new

if __name__ == "__main__":
    # Train annotations
    train_json = os.path.join(BASE_DIR, "train", "annotations.json")
    print(f"\n=== Fixing {train_json} ===")
    fix_image_ids(train_json)
    
    # Val annotations
    val_json = os.path.join(BASE_DIR, "val", "annotations.json")
    print(f"\n=== Fixing {val_json} ===")
    fix_image_ids(val_json)
    
    # temp_original (if exists)
    temp_json = os.path.join(BASE_DIR, "temp_original", "train", "annotations.json")
    if os.path.exists(temp_json):
        print(f"\n=== Fixing {temp_json} ===")
        fix_image_ids(temp_json)
    
    print("\n[DONE] All image IDs fixed!")
    print("Now run: python 4_evaluate.py")