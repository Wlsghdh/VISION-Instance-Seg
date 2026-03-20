"""
MaskDINO용 커스텀 DatasetMapper
- Polygon → BitMask 변환
- 기존 training/maskdino/train_full.py에서 추출
"""

import copy
import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import Instances, Boxes
import pycocotools.mask as mask_util


class MaskDINODatasetMapper:
    """
    Detectron2 dataset dict → MaskDINO 입력 형식 변환.
    PolygonMasks → BitMasks 변환 포함.
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
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = T.AugmentationList(self.augmentations)(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict["annotations"]
                if obj.get("iscrowd", 0) == 0
            ]

            instances = Instances(image_shape)
            boxes = [obj["bbox"] for obj in annos]
            boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
            instances.gt_boxes = Boxes(boxes)
            instances.gt_classes = torch.tensor(
                [obj["category_id"] for obj in annos], dtype=torch.int64
            )

            if len(annos) > 0:
                masks = []
                for obj in annos:
                    segm = obj.get("segmentation")
                    if segm is None:
                        continue
                    if isinstance(segm, list):
                        rles = mask_util.frPyObjects(segm, image_shape[0], image_shape[1])
                        rle = mask_util.merge(rles)
                        mask = mask_util.decode(rle)
                    elif isinstance(segm, dict):
                        mask = mask_util.decode(segm)
                    else:
                        raise ValueError(f"Unknown segmentation format: {type(segm)}")
                    masks.append(mask)

                if masks:
                    instances.gt_masks = torch.tensor(
                        np.stack(masks, axis=0), dtype=torch.bool
                    )
                else:
                    instances.gt_masks = torch.zeros((0, *image_shape), dtype=torch.bool)
            else:
                instances.gt_masks = torch.zeros((0, *image_shape), dtype=torch.bool)

            dataset_dict["instances"] = instances
        return dataset_dict
