#!/usr/bin/env python3
"""
MaskDINO Evaluation Script
학습된 모델의 mAP 평가
"""

import os
import sys
import json
import torch

MASKDINO_PATH = "/data2/project/2026winter/jjh0709/MaskDINO"
sys.path.insert(0, MASKDINO_PATH)

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from maskdino import add_maskdino_config

BASE_DIR = "/data2/project/2026winter/jjh0709/AA_CV_R"

def register_val_dataset():
    """Val 데이터셋 등록"""
    val_json = os.path.join(BASE_DIR, "val", "annotations.json")
    val_images = os.path.join(BASE_DIR, "val", "images")
    
    dataset_name = "thunderbolt_val_eval"
    
    if dataset_name in DatasetCatalog:
        DatasetCatalog.remove(dataset_name)
    if dataset_name in MetadataCatalog:
        MetadataCatalog.remove(dataset_name)
    
    DatasetCatalog.register(
        dataset_name,
        lambda: load_coco_json(val_json, val_images)
    )
    MetadataCatalog.get(dataset_name).set(
        json_file=val_json,
        image_root=val_images,
        thing_classes=["thunderbolt"],
        evaluator_type="coco"
    )
    
    with open(val_json, 'r') as f:
        val_data = json.load(f)
    
    print(f"[INFO] Val dataset: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")
    
    return dataset_name

def setup_cfg(model_path):
    """Config 설정"""
    cfg = get_cfg()
    add_maskdino_config(cfg)
    cfg.set_new_allowed(True)  # yaml 로드 전에 설정!
    
    config_file = os.path.join(
        MASKDINO_PATH, 
        "configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml"
    )
    cfg.merge_from_file(config_file)
    
    # 모델 설정
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.MaskDINO.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 테스트 설정
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.INPUT.MAX_SIZE_TEST = 800
    cfg.INPUT.MASK_FORMAT = "polygon"
    cfg.MODEL.MASK_ON = True
    
    # Gradient clipping (호환성)
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.01
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    
    cfg.freeze()
    return cfg

def evaluate_model(model_path, output_dir, model_name="model"):
    """모델 평가"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return None
    
    # 데이터셋 등록
    dataset_name = register_val_dataset()
    
    # Config 설정
    cfg = setup_cfg(model_path)
    
    # 모델 빌드
    model = build_model(cfg)
    model.eval()
    
    # 체크포인트 로드
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(model_path)
    
    # Evaluator 설정
    eval_output = os.path.join(output_dir, "eval_results")
    os.makedirs(eval_output, exist_ok=True)
    
    evaluator = COCOEvaluator(
        dataset_name,
        output_dir=eval_output,
        tasks=("bbox", "segm")
    )
    
    # Data loader
    val_loader = build_detection_test_loader(cfg, dataset_name)
    
    # Evaluation 실행
    print("\n[INFO] Running evaluation...")
    try:
        results = inference_on_dataset(model, val_loader, evaluator)
        
        print(f"\n{'='*60}")
        print(f"Results for {model_name}:")
        print(f"{'='*60}")
        
        if "bbox" in results:
            print("\n[BBox Detection]")
            for k, v in results["bbox"].items():
                print(f"  {k}: {v:.4f}")
        
        if "segm" in results:
            print("\n[Instance Segmentation]")
            for k, v in results["segm"].items():
                print(f"  {k}: {v:.4f}")
        
        # 결과 저장
        results_file = os.path.join(eval_output, f"{model_name}_results.json")
        with open(results_file, 'w') as f:
            # numpy/tensor를 float로 변환
            results_serializable = {}
            for task, metrics in results.items():
                results_serializable[task] = {k: float(v) for k, v in metrics.items()}
            json.dump(results_serializable, f, indent=2)
        print(f"\n[INFO] Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=" * 60)
    print("MaskDINO Model Evaluation")
    print("=" * 60)
    
    # GPU 확인
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    all_results = {}
    
    # 1. Original 모델 평가
    original_model = os.path.join(BASE_DIR, "output_original", "model_final.pth")
    if os.path.exists(original_model):
        results = evaluate_model(
            original_model,
            os.path.join(BASE_DIR, "output_original"),
            "Original (26 images)"
        )
        if results:
            all_results["original"] = results
    else:
        print(f"[SKIP] Original model not found")
    
    # 2. Full 모델 평가
    full_model = os.path.join(BASE_DIR, "output_full", "model_final.pth")
    if os.path.exists(full_model):
        results = evaluate_model(
            full_model,
            os.path.join(BASE_DIR, "output_full"),
            "Full (127 images)"
        )
        if results:
            all_results["full"] = results
    else:
        print(f"[SKIP] Full model not found")
    
    # 비교 요약
    if len(all_results) >= 2:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Metric':<20} {'Original':<15} {'Full':<15} {'Diff':<15}")
        print("-" * 60)
        
        for task in ["bbox", "segm"]:
            if task in all_results.get("original", {}) and task in all_results.get("full", {}):
                print(f"\n[{task.upper()}]")
                for metric in ["AP", "AP50", "AP75", "APs", "APm", "APl"]:
                    orig_val = all_results["original"][task].get(metric, 0)
                    full_val = all_results["full"][task].get(metric, 0)
                    diff = full_val - orig_val
                    diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
                    print(f"  {metric:<18} {orig_val:<15.4f} {full_val:<15.4f} {diff_str}")
    
    print("\n[DONE] Evaluation complete!")

if __name__ == "__main__":
    main()