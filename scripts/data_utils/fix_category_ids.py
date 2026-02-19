#!/usr/bin/env python3
"""
Annotation category_id 수정: 1 → 0
Detectron2는 category_id가 0부터 시작해야 함
"""

import json
import os
import shutil

BASE_DIR = "/data2/project/2026winter/jjh0709/AA_CV_R"

def fix_annotations(json_path):
    """category_id를 1에서 0으로 변경"""
    # 백업
    backup_path = json_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy(json_path, backup_path)
        print(f"[INFO] Backup created: {backup_path}")
    
    # 로드
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # categories 수정
    for cat in data['categories']:
        if cat['id'] == 1:
            cat['id'] = 0
            print(f"[INFO] Category id changed: 1 → 0")
    
    # annotations의 category_id 수정
    count = 0
    for anno in data['annotations']:
        if anno['category_id'] == 1:
            anno['category_id'] = 0
            count += 1
    
    print(f"[INFO] {count} annotations category_id changed: 1 → 0")
    
    # 저장
    with open(json_path, 'w') as f:
        json.dump(data, f)
    
    print(f"[INFO] Saved: {json_path}")

if __name__ == "__main__":
    # Train annotations
    train_json = os.path.join(BASE_DIR, "train", "annotations.json")
    print(f"\n=== Fixing {train_json} ===")
    fix_annotations(train_json)
    
    # Val annotations
    val_json = os.path.join(BASE_DIR, "val", "annotations.json")
    print(f"\n=== Fixing {val_json} ===")
    fix_annotations(val_json)
    
    # temp_original (if exists)
    temp_json = os.path.join(BASE_DIR, "temp_original", "train", "annotations.json")
    if os.path.exists(temp_json):
        print(f"\n=== Fixing {temp_json} ===")
        fix_annotations(temp_json)
    
    print("\n[DONE] All annotations fixed!")
    print("Now run: CUDA_VISIBLE_DEVICES=1 nohup bash run_train.sh > pipeline.log 2>&1 &")