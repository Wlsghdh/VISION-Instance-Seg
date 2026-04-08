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
        """학습된 최종 모델 경로.

        우선순위:
        1. model_best.pth (early stopping이 best 갱신 시 저장한 스냅샷)
        2. model_final.pth (정상 종료 또는 early stop fallback 저장)
        3. mmdet 계열의 best/last 체크포인트
        4. detectron2의 last_checkpoint 텍스트 파일이 가리키는 파일
        5. 디렉토리 내 model_*.pth 중 가장 최근 (이름 기준)
        """
        if self.output_dir is None:
            return None
        for name in ["model_best.pth", "model_final.pth",
                     "best_coco_segm_mAP_epoch.pth", "best_auto.pth", "epoch_12.pth"]:
            p = self.output_dir / name
            if p.exists():
                return p
        # Fallback 1: detectron2가 유지하는 last_checkpoint 텍스트 파일
        last_ckpt_file = self.output_dir / "last_checkpoint"
        if last_ckpt_file.exists():
            try:
                last_name = last_ckpt_file.read_text().strip()
                # 절대 경로일 수도, 파일명만 있을 수도 있음
                p = Path(last_name)
                if not p.is_absolute():
                    p = self.output_dir / last_name
                if p.exists():
                    return p
            except Exception:
                pass
        # Fallback 2: model_*.pth 중 가장 최근 (이름 정렬 → iter 번호 기준)
        candidates = sorted(self.output_dir.glob("model_*.pth"))
        if candidates:
            return candidates[-1]
        return None
