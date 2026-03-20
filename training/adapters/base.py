"""
ModelAdapter ABC — 모든 모델 어댑터의 기본 인터페이스
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class ModelAdapter(ABC):
    """
    모든 모델 어댑터가 구현해야 하는 인터페이스.

    사용법:
        adapter = SomeAdapter(model_name, num_classes, thing_classes)
        adapter.setup(train_data, val_data, output_dir, hyperparams)
        metrics = adapter.train()
        eval_results = adapter.evaluate()
    """

    def __init__(self, model_name: str, num_classes: int,
                 thing_classes: list, model_info: dict):
        self.model_name = model_name
        self.num_classes = num_classes
        self.thing_classes = thing_classes
        self.model_info = model_info
        self.output_dir: Optional[Path] = None

    @abstractmethod
    def setup(self, train_images_dir: Path, train_ann_path: Path,
              val_images_dir: Path, val_ann_path: Path,
              output_dir: Path, hyperparams: dict):
        """학습 환경 설정 (config 생성, 데이터 등록 등)"""
        ...

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """학습 실행. 로그/메트릭 반환."""
        ...

    @abstractmethod
    def evaluate(self, model_path: Optional[Path] = None) -> Dict[str, float]:
        """
        평가 실행. bbox_AP, segm_AP 등 반환.
        model_path 미지정 시 학습된 최종 모델 사용.
        """
        ...

    def get_final_model_path(self) -> Optional[Path]:
        """학습된 최종 모델 경로"""
        if self.output_dir is None:
            return None
        for name in ["model_final.pth", "best_coco_segm_mAP_epoch.pth",
                      "best_auto.pth", "epoch_12.pth"]:
            p = self.output_dir / name
            if p.exists():
                return p
        return None
