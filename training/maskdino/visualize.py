#!/usr/bin/env python3
"""
MaskDINO Visualization Script
BBox + Segmentation 예측 시각화 (confidence >= threshold)
"""

import os
import sys
import json
import cv2
import numpy as np
import torch
import argparse

MASKDINO_PATH = "/data2/project/2026winter/jjh0709/MaskDINO"
sys.path.insert(0, MASKDINO_PATH)

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

from maskdino import add_maskdino_config

BASE_DIR = "/data2/project/2026winter/jjh0709/AA_CV_R"
CONFIDENCE_THRESHOLD = 0.5

def setup_cfg(model_type):
    if model_type == "original":
        output_dir = os.path.join(BASE_DIR, "output_original")
    else:
        output_dir = os.path.join(BASE_DIR, "output_full")
    
    cfg = get_cfg()
    add_maskdino_config(cfg)
    cfg.set_new_allowed(True)
    
    base_config = os.path.join(
        MASKDINO_PATH, 
        "configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml"
    )
    cfg.merge_from_file(base_config)
    
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = os.path.join(output_dir, "model_final.pth")
    cfg.OUTPUT_DIR = output_dir
    
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.INPUT.MAX_SIZE_TEST = 800
    
    return cfg, output_dir

def register_val_dataset():
    val_json = os.path.join(BASE_DIR, "val", "annotations.json")
    val_images = os.path.join(BASE_DIR, "val", "images")
    
    if "thunderbolt_val" in DatasetCatalog:
        DatasetCatalog.remove("thunderbolt_val")
    if "thunderbolt_val" in MetadataCatalog:
        MetadataCatalog.remove("thunderbolt_val")
    
    DatasetCatalog.register(
        "thunderbolt_val",
        lambda: load_coco_json(val_json, val_images)
    )
    MetadataCatalog.get("thunderbolt_val").set(
        json_file=val_json,
        image_root=val_images,
        thing_classes=["thunderbolt"],
        thing_colors=[(255, 0, 0)]
    )
    
    return val_images

def visualize_predictions(predictor, image_dir, output_dir, max_images=None):
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    metadata = MetadataCatalog.get("thunderbolt_val")
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\n[INFO] Visualizing {len(image_files)} images...")
    print(f"[INFO] Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    results_summary = []
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        
        if len(instances) > 0:
            scores = instances.scores
            keep = scores >= CONFIDENCE_THRESHOLD
            instances = instances[keep]
        
        num_predictions = len(instances)
        
        v = Visualizer(
            img[:, :, ::-1],
            metadata=metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE_BW
        )
        
        vis_output = v.draw_instance_predictions(instances)
        vis_img = vis_output.get_image()[:, :, ::-1]
        
        info_text = f"Predictions: {num_predictions} (conf >= {CONFIDENCE_THRESHOLD})"
        cv2.putText(vis_img, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        output_path = os.path.join(vis_dir, f"vis_{img_file}")
        cv2.imwrite(output_path, vis_img)
        
        scores_list = instances.scores.numpy().tolist() if num_predictions > 0 else []
        results_summary.append({
            'image': img_file,
            'predictions': num_predictions,
            'scores': scores_list
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(image_files)}")
    
    print(f"\n[INFO] Visualizations saved to: {vis_dir}")
    
    summary_path = os.path.join(vis_dir, "prediction_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    total_preds = sum(r['predictions'] for r in results_summary)
    images_with_preds = sum(1 for r in results_summary if r['predictions'] > 0)
    
    print(f"\n[SUMMARY]")
    print(f"  Total images: {len(results_summary)}")
    print(f"  Images with predictions: {images_with_preds}")
    print(f"  Total predictions: {total_preds}")
    if len(results_summary) > 0:
        print(f"  Avg predictions per image: {total_preds / len(results_summary):.2f}")

def visualize_comparison(predictor, image_dir, output_dir):
    compare_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(compare_dir, exist_ok=True)
    
    with open(os.path.join(BASE_DIR, "val", "annotations.json"), 'r') as f:
        gt_data = json.load(f)
    
    img_id_to_annos = {}
    for anno in gt_data['annotations']:
        img_id = anno['image_id']
        if img_id not in img_id_to_annos:
            img_id_to_annos[img_id] = []
        img_id_to_annos[img_id].append(anno)
    
    print("\n[INFO] Creating GT vs Prediction comparison...")
    
    for img_info in gt_data['images'][:20]:
        img_id = img_info['id']
        img_file = img_info['file_name']
        img_path = os.path.join(image_dir, img_file)
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # GT 그리기 (녹색)
        gt_img = img.copy()
        if img_id in img_id_to_annos:
            for anno in img_id_to_annos[img_id]:
                x, y, bw, bh = [int(v) for v in anno['bbox']]
                cv2.rectangle(gt_img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                
                if 'segmentation' in anno and anno['segmentation']:
                    for seg in anno['segmentation']:
                        pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.polylines(gt_img, [pts], True, (0, 255, 0), 2)
        
        cv2.putText(gt_img, "Ground Truth", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Prediction 그리기 (빨간색)
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        
        if len(instances) > 0:
            scores = instances.scores
            keep = scores >= CONFIDENCE_THRESHOLD
            instances = instances[keep]
        
        pred_img = img.copy()
        
        if len(instances) > 0:
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None
            
            for j in range(len(instances)):
                x1, y1, x2, y2 = boxes[j].astype(int)
                cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(pred_img, f"{scores[j]:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                if masks is not None:
                    mask = masks[j]
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(pred_img, contours, -1, (0, 0, 255), 2)
        
        cv2.putText(pred_img, f"Prediction ({len(instances)})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        comparison = np.hstack([gt_img, pred_img])
        output_path = os.path.join(compare_dir, f"compare_{img_file}")
        cv2.imwrite(output_path, comparison)
    
    print(f"[INFO] Comparison images saved to: {compare_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="original", choices=["original", "full"])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.threshold
    
    print("=" * 60)
    print(f"MaskDINO Visualization - {args.model.upper()} model")
    print("=" * 60)
    
    cfg, output_dir = setup_cfg(args.model)
    
    if not os.path.exists(cfg.MODEL.WEIGHTS):
        print(f"[ERROR] Model not found: {cfg.MODEL.WEIGHTS}")
        return
    
    val_images = register_val_dataset()
    predictor = DefaultPredictor(cfg)
    
    visualize_predictions(predictor, val_images, output_dir)
    visualize_comparison(predictor, val_images, output_dir)

if __name__ == "__main__":
    main()