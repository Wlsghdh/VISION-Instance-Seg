#!/usr/bin/env python3
"""
MaskDINO Training Script - Full Data (Fixed for BitMask)
Trains on all training images (127 images)
"""

import os
import sys
import json
import copy
import numpy as np
import torch

# Add MaskDINO to path
MASKDINO_PATH = "/data2/project/2026winter/jjh0709/MaskDINO"
sys.path.insert(0, MASKDINO_PATH)

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import Instances, Boxes
from detectron2.evaluation import COCOEvaluator
from detectron2.projects.deeplab import add_deeplab_config

# MaskDINO imports
from maskdino import add_maskdino_config

import pycocotools.mask as mask_util


# ============================================================
# Custom DatasetMapper that converts Polygon to BitMask
# ============================================================
class MaskDINODatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskDINO.
    
    Key difference: Converts PolygonMasks to BitMasks
    """
    
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.augmentations = self._build_augmentation(cfg, is_train)
        self.image_format = cfg.INPUT.FORMAT
        self.mask_format = cfg.INPUT.MASK_FORMAT
        
    def _build_augmentation(self, cfg, is_train):
        augs = []
        if is_train:
            augs.append(T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            ))
            if cfg.INPUT.CROP.ENABLED:
                augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            augs.append(T.RandomFlip())
        else:
            augs.append(T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TEST,
                cfg.INPUT.MAX_SIZE_TEST,
                "choice"
            ))
        return augs
    
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        
        # Read image
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        
        # Apply augmentations
        aug_input = T.AugInput(image)
        transforms = T.AugmentationList(self.augmentations)(aug_input)
        image = aug_input.image
        
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict
        
        # Process annotations
        if "annotations" in dataset_dict:
            annos = dataset_dict["annotations"]
            
            # Transform annotations
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in annos
                if obj.get("iscrowd", 0) == 0
            ]
            
            # Create Instances
            instances = Instances(image_shape)
            
            # Extract boxes
            boxes = [obj["bbox"] for obj in annos]
            boxes = torch.tensor(boxes, dtype=torch.float32)
            if len(boxes) > 0:
                instances.gt_boxes = Boxes(boxes)
            else:
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            
            # Extract classes
            classes = [obj["category_id"] for obj in annos]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            
            # Extract and convert masks to BitMasks
            # This is the key fix!
            if len(annos) > 0:
                masks = []
                for obj in annos:
                    segm = obj.get("segmentation", None)
                    if segm is None:
                        continue
                    
                    if isinstance(segm, list):
                        # Polygon format -> convert to RLE -> convert to binary mask
                        rles = mask_util.frPyObjects(segm, image_shape[0], image_shape[1])
                        rle = mask_util.merge(rles)
                        mask = mask_util.decode(rle)
                    elif isinstance(segm, dict):
                        # RLE format
                        mask = mask_util.decode(segm)
                    else:
                        raise ValueError(f"Unknown segmentation format: {type(segm)}")
                    
                    masks.append(mask)
                
                if len(masks) > 0:
                    masks = np.stack(masks, axis=0)
                    # MaskDINO expects raw tensor, not BitMasks object
                    instances.gt_masks = torch.tensor(masks, dtype=torch.bool)
                else:
                    instances.gt_masks = torch.zeros((0, *image_shape), dtype=torch.bool)
            else:
                instances.gt_masks = torch.zeros((0, *image_shape), dtype=torch.bool)
            
            dataset_dict["instances"] = instances
        
        return dataset_dict


# ============================================================
# Trainer with custom DatasetMapper
# ============================================================
class MaskDINOTrainer(DefaultTrainer):
    """
    Custom trainer that uses MaskDINODatasetMapper
    """
    
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = MaskDINODatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


# ============================================================
# Dataset registration
# ============================================================
def register_dataset(name, annotations_path, images_dir):
    """Register dataset in Detectron2 format"""
    
    def get_dicts():
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        
        # Build lookup tables
        image_lookup = {img['id']: img for img in data['images']}
        ann_by_image = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in ann_by_image:
                ann_by_image[img_id] = []
            ann_by_image[img_id].append(ann)
        
        # Create dataset dicts
        dataset_dicts = []
        for img in data['images']:
            record = {
                "file_name": os.path.join(images_dir, img['file_name']),
                "image_id": img['id'],
                "height": img['height'],
                "width": img['width'],
            }
            
            annotations = []
            for ann in ann_by_image.get(img['id'], []):
                obj = {
                    "bbox": ann['bbox'],  # COCO format: [x, y, width, height]
                    "bbox_mode": 1,  # BoxMode.XYWH_ABS
                    "segmentation": ann['segmentation'],
                    "category_id": 0,  # Single class (thunderbolt)
                    "iscrowd": ann.get('iscrowd', 0),
                }
                annotations.append(obj)
            
            record["annotations"] = annotations
            dataset_dicts.append(record)
        
        return dataset_dicts
    
    DatasetCatalog.register(name, get_dicts)
    MetadataCatalog.get(name).set(thing_classes=["thunderbolt"])


# ============================================================
# Main
# ============================================================
def setup_cfg():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    
    # Use base config
    base_config = os.path.join(
        MASKDINO_PATH, 
        "configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml"
    )
    cfg.merge_from_file(base_config)
    
    # Dataset settings
    cfg.DATASETS.TRAIN = ("thunderbolt_full_train",)
    cfg.DATASETS.TEST = ("thunderbolt_full_train",)
    
    # Model settings
    cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    
    # Input settings
    cfg.INPUT.MASK_FORMAT = "polygon"
    cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640)
    cfg.INPUT.MAX_SIZE_TRAIN = 800
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.INPUT.MAX_SIZE_TEST = 800
    
    # Training settings
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = (3500, 4500)
    cfg.SOLVER.WARMUP_ITERS = 200
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    
    cfg.TEST.EVAL_PERIOD = 500
    
    # Output
    cfg.OUTPUT_DIR = "/data2/project/2026winter/jjh0709/AA_CV_R/output_full"
    
    # Disable FP16 for stability
    cfg.SOLVER.AMP.ENABLED = False
    
    # Fix gradient clipping - must be valid type: "value", "norm", or "none"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"  # This is set by MaskDINO config
    # Override with valid detectron2 type
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.01
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    
    return cfg


def main():
    # Paths
    work_dir = "/data2/project/2026winter/jjh0709/AA_CV_R"
    train_annotations = os.path.join(work_dir, "train/annotations.json")
    train_images = os.path.join(work_dir, "train/images")
    
    # Count data
    with open(train_annotations, 'r') as f:
        data = json.load(f)
    num_images = len(data['images'])
    num_annotations = len(data['annotations'])
    
    # Register datasets
    register_dataset(
        "thunderbolt_full_train",
        train_annotations,
        train_images
    )
    
    # Setup config
    cfg = setup_cfg()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("MaskDINO Training - Full Data")
    print("="*60)
    print(f"Images: {num_images}")
    print(f"Annotations: {num_annotations}")
    print(f"Output: {cfg.OUTPUT_DIR}")
    print("="*60)
    
    # Train
    trainer = MaskDINOTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()