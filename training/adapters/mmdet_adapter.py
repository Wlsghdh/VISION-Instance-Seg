"""
MMDetection 기반 모델 어댑터
- Cascade R-CNN (instance segmentation)
- SOLOv2
- RTMDet-Ins
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .base import ModelAdapter


class MMDetAdapter(ModelAdapter):
    """
    MMDetection(mmdet) 프레임워크 기반 모델 통합 어댑터.
    mmengine Runner API 사용.
    """

    def __init__(self, model_name: str, num_classes: int,
                 thing_classes: list, model_info: dict):
        super().__init__(model_name, num_classes, thing_classes, model_info)
        self.runner_cfg = None
        self.train_ann_path = None
        self.val_ann_path = None
        self.train_images_dir = None
        self.val_images_dir = None

    def _get_base_config(self) -> str:
        """mmdet 내장 config 경로 반환"""
        try:
            import mmdet
            mmdet_dir = Path(mmdet.__file__).parent.parent
            configs_dir = mmdet_dir / ".mim" / "configs"
            if not configs_dir.exists():
                # pip install로 설치된 경우
                import importlib.resources
                configs_dir = Path(mmdet.__file__).parent / ".mim" / "configs"
        except ImportError:
            raise ImportError("mmdet가 설치되지 않았습니다")

        config_map = {
            "cascade_rcnn": configs_dir / "cascade_rcnn" / "cascade-mask-rcnn_r50_fpn_1x_coco.py",
            "solov2": configs_dir / "solov2" / "solov2_r50_fpn_1x_coco.py",
            "rtmdet_ins": configs_dir / "rtmdet" / "rtmdet-ins_s_8xb32-300e_coco.py",
        }

        config_path = config_map.get(self.model_name)
        if config_path is None:
            raise ValueError(f"Unknown mmdet model: {self.model_name}")

        if not config_path.exists():
            # 대안 경로 시도
            alt_paths = list(configs_dir.rglob(f"*{self.model_info['config']}"))
            if alt_paths:
                config_path = alt_paths[0]
            else:
                raise FileNotFoundError(
                    f"mmdet config not found: {config_path}\n"
                    f"Available configs in {configs_dir}: {list(configs_dir.iterdir())[:10]}"
                )

        return str(config_path)

    def setup(self, train_images_dir: Path, train_ann_path: Path,
              val_images_dir: Path, val_ann_path: Path,
              output_dir: Path, hyperparams: dict):
        """학습 환경 설정"""
        from mmengine.config import Config

        self.output_dir = output_dir
        self.train_ann_path = train_ann_path
        self.val_ann_path = val_ann_path
        self.train_images_dir = train_images_dir
        self.val_images_dir = val_images_dir

        os.makedirs(str(output_dir), exist_ok=True)

        base_config_path = self._get_base_config()
        cfg = Config.fromfile(base_config_path)

        # 클래스 수 변경
        classes = tuple(self.thing_classes)
        metainfo = {"classes": classes}

        # 데이터 설정
        cfg.train_dataloader.dataset.ann_file = str(train_ann_path)
        cfg.train_dataloader.dataset.data_prefix = {"img": str(train_images_dir)}
        cfg.train_dataloader.dataset.metainfo = metainfo
        cfg.train_dataloader.batch_size = hyperparams.get("batch_size", 2)
        cfg.train_dataloader.num_workers = 2

        cfg.val_dataloader.dataset.ann_file = str(val_ann_path)
        cfg.val_dataloader.dataset.data_prefix = {"img": str(val_images_dir)}
        cfg.val_dataloader.dataset.metainfo = metainfo

        cfg.test_dataloader = cfg.val_dataloader.copy()

        # 평가 설정
        cfg.val_evaluator = dict(
            type="CocoMetric",
            ann_file=str(val_ann_path),
            metric=["bbox", "segm"],
            classwise=True,
        )
        cfg.test_evaluator = cfg.val_evaluator.copy()

        # 모델의 num_classes 업데이트
        self._update_num_classes(cfg)

        # 에폭 기반 학습 설정
        max_epochs = hyperparams.get("max_epochs", 300)
        lr = hyperparams.get("lr", 1e-4)
        eval_period_epochs = hyperparams.get("eval_period_epochs", 5)
        checkpoint_period_epochs = hyperparams.get("checkpoint_period_epochs", 10)
        warmup_epochs = hyperparams.get("warmup_epochs", 5)
        patience = hyperparams.get("early_stopping_patience", 15)

        cfg.train_cfg = dict(
            type="EpochBasedTrainLoop",
            max_epochs=max_epochs,
            val_interval=eval_period_epochs,
        )
        cfg.val_cfg = dict(type="ValLoop")
        cfg.test_cfg = dict(type="TestLoop")

        # optimizer
        cfg.optim_wrapper = dict(
            type="OptimWrapper",
            optimizer=dict(type="AdamW", lr=lr, weight_decay=0.05),
        )

        # LR scheduler (에폭 기반)
        cfg.param_scheduler = [
            dict(type="LinearLR", start_factor=0.001,
                 by_epoch=True, begin=0,
                 end=warmup_epochs),
            dict(type="CosineAnnealingLR", by_epoch=True,
                 begin=warmup_epochs,
                 end=max_epochs),
        ]

        # 체크포인트
        cfg.default_hooks = dict(
            timer=dict(type="IterTimerHook"),
            logger=dict(type="LoggerHook", interval=50),
            param_scheduler=dict(type="ParamSchedulerHook"),
            checkpoint=dict(
                type="CheckpointHook",
                by_epoch=True,
                interval=checkpoint_period_epochs,
                save_best="coco/segm_mAP",
                rule="greater",
            ),
            sampler_seed=dict(type="DistSamplerSeedHook"),
        )

        # Early stopping hook
        cfg.custom_hooks = [
            dict(
                type="EarlyStoppingHook",
                monitor="coco/segm_mAP",
                patience=patience,
                min_delta=0.0,
                rule="greater",
            ),
        ]

        self._max_epochs = max_epochs
        self._eval_period_epochs = eval_period_epochs
        self._patience = patience

        print(f"  Max epochs: {max_epochs}")
        print(f"  Eval every {eval_period_epochs} epochs")
        print(f"  Early stopping patience: {patience} evals ({patience * eval_period_epochs} epochs)")

        cfg.work_dir = str(output_dir)
        cfg.seed = hyperparams.get("seed", 42)
        cfg.load_from = None  # COCO pretrained는 mmdet가 자동 다운로드

        # GPU 설정
        cfg.gpu_ids = [0]

        self.runner_cfg = cfg

        # config 저장
        cfg.dump(str(output_dir / "config.py"))
        print(f"  Config saved: {output_dir / 'config.py'}")

    def _update_num_classes(self, cfg):
        """모델별 num_classes 업데이트"""
        nc = self.num_classes

        if self.model_name == "cascade_rcnn":
            # Cascade Mask R-CNN: 각 stage의 bbox_head + mask_head
            if hasattr(cfg.model, 'roi_head'):
                for head in cfg.model.roi_head.get('bbox_head', []):
                    if isinstance(head, dict):
                        head['num_classes'] = nc
                if hasattr(cfg.model.roi_head, 'mask_head'):
                    cfg.model.roi_head.mask_head['num_classes'] = nc

        elif self.model_name == "solov2":
            if hasattr(cfg.model, 'mask_head'):
                cfg.model.mask_head['num_classes'] = nc

        elif self.model_name == "rtmdet_ins":
            if hasattr(cfg.model, 'bbox_head'):
                cfg.model.bbox_head['num_classes'] = nc

    def train(self) -> Dict[str, Any]:
        """학습 실행"""
        if self.runner_cfg is None:
            raise RuntimeError("setup()을 먼저 호출하세요")

        from mmengine.runner import Runner

        print(f"\n{'='*60}")
        print(f"  학습 시작: {self.model_info['display_name']} (mmdet)")
        print(f"  MAX_EPOCHS: {self._max_epochs}")
        print(f"  OUTPUT: {self.runner_cfg.work_dir}")
        print(f"{'='*60}\n")

        # GPU 메모리 추적 시작
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        runner = Runner.from_cfg(self.runner_cfg)
        runner.train()

        # GPU 메모리 측정
        peak_memory_mb = None
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # 학습 종료 에폭 확인
        final_epoch = runner.epoch + 1  # 0-indexed → 1-indexed
        early_stopped = final_epoch < self._max_epochs

        metrics = {
            "model": self.model_name,
            "status": "completed",
            "total_epochs": final_epoch,
            "max_epochs": self._max_epochs,
            "peak_memory_mb": round(peak_memory_mb, 1) if peak_memory_mb else None,
            "early_stopped": early_stopped,
            "early_stop_epoch": final_epoch if early_stopped else None,
        }

        return metrics

    def evaluate(self, model_path: Optional[Path] = None) -> Dict[str, float]:
        """평가 실행"""
        if self.runner_cfg is None:
            raise RuntimeError("setup()을 먼저 호출하세요")

        from mmengine.runner import Runner

        if model_path is None:
            model_path = self.get_final_model_path()
        if model_path is None:
            # mmdet best checkpoint 탐색
            best_files = list(Path(self.output_dir).glob("best_*.pth"))
            if best_files:
                model_path = best_files[0]
            else:
                latest = Path(self.output_dir) / "iter_latest.pth"
                if latest.exists():
                    model_path = latest

        if model_path is None or not model_path.exists():
            raise FileNotFoundError(f"Model not found in {self.output_dir}")

        print(f"\n평가 시작: {model_path}")

        self.runner_cfg.load_from = str(model_path)
        runner = Runner.from_cfg(self.runner_cfg)
        metrics = runner.test()

        # 결과 정리
        flat_results = {}
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    flat_results[k] = float(v)

        # 저장
        eval_output = Path(self.output_dir) / "eval_results"
        eval_output.mkdir(parents=True, exist_ok=True)
        results_file = eval_output / "results.json"
        with open(results_file, 'w') as f:
            json.dump(flat_results, f, indent=2)

        print(f"  평가 결과 저장: {results_file}")
        return flat_results

    def get_final_model_path(self) -> Optional[Path]:
        """mmdet 모델 경로 탐색"""
        if self.output_dir is None:
            return None
        # best checkpoint 우선
        best_files = sorted(Path(self.output_dir).glob("best_*.pth"))
        if best_files:
            return best_files[-1]
        # latest
        latest = Path(self.output_dir) / "iter_latest.pth"
        if latest.exists():
            return latest
        return super().get_final_model_path()
