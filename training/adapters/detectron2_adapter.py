"""
Detectron2 기반 모델 어댑터
- Mask R-CNN
- Cascade Mask R-CNN
- MaskDINO
- Mask2Former
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .base import ModelAdapter


class EarlyStoppingHook:
    """
    Detectron2용 Early Stopping Hook.
    eval_period마다 metric을 체크하여 patience 횟수 동안 개선 없으면 학습 중단.
    """

    def __init__(self, eval_period: int, patience: int, metric_name: str = "segm/AP",
                 iters_per_epoch: int = 1):
        self.eval_period = eval_period
        self.patience = patience
        self.metric_name = metric_name
        self.iters_per_epoch = iters_per_epoch
        self.best_metric = -float("inf")
        self.best_iter = 0
        self.num_bad_evals = 0
        self.stopped_iter = None
        self.stopped_epoch = None
        self.trainer = None

    def before_train(self):
        pass

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self.eval_period != 0:
            return
        if next_iter <= 0:
            return

        # metrics.json에서 최신 eval 결과 읽기
        storage = self.trainer.storage
        try:
            metric_val = storage.latest().get(self.metric_name, None)
            if metric_val is not None:
                metric_val = metric_val[0]  # (value, iteration) 튜플
        except Exception:
            metric_val = None

        if metric_val is None:
            return

        current_epoch = next_iter / self.iters_per_epoch

        if metric_val > self.best_metric:
            self.best_metric = metric_val
            self.best_iter = next_iter
            self.num_bad_evals = 0
            print(f"  [EarlyStopping] New best {self.metric_name}={metric_val:.4f} "
                  f"at epoch {current_epoch:.1f} (iter {next_iter})")
        else:
            self.num_bad_evals += 1
            print(f"  [EarlyStopping] No improvement {self.num_bad_evals}/{self.patience} "
                  f"(best={self.best_metric:.4f} at iter {self.best_iter})")

        if self.num_bad_evals >= self.patience:
            self.stopped_iter = next_iter
            self.stopped_epoch = current_epoch
            print(f"\n  [EarlyStopping] STOP at epoch {current_epoch:.1f} (iter {next_iter}). "
                  f"Best {self.metric_name}={self.best_metric:.4f} at iter {self.best_iter}")
            # 학습 중단: max_iter를 현재 iter로 설정
            self.trainer.max_iter = next_iter

    def after_train(self):
        pass


class Detectron2Adapter(ModelAdapter):
    """
    Detectron2 프레임워크 기반 모델 통합 어댑터.
    model_name에 따라 적절한 config와 trainer를 사용.
    """

    def __init__(self, model_name: str, num_classes: int,
                 thing_classes: list, model_info: dict):
        super().__init__(model_name, num_classes, thing_classes, model_info)
        self.cfg = None
        self.train_dataset_name = None
        self.val_dataset_name = None
        self.iters_per_epoch = None
        self.early_stopping_hook = None

    def _import_detectron2(self):
        """Lazy import to avoid import errors when detectron2 is not installed"""
        import detectron2  # noqa: F401
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultTrainer
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader, build_detection_train_loader
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.modeling import build_model
        from detectron2 import model_zoo
        return {
            'get_cfg': get_cfg,
            'DefaultTrainer': DefaultTrainer,
            'COCOEvaluator': COCOEvaluator,
            'inference_on_dataset': inference_on_dataset,
            'build_detection_test_loader': build_detection_test_loader,
            'build_detection_train_loader': build_detection_train_loader,
            'DetectionCheckpointer': DetectionCheckpointer,
            'build_model': build_model,
            'model_zoo': model_zoo,
        }

    def _setup_maskdino(self, d2):
        """MaskDINO 전용 설정"""
        from training.config import MASKDINO_REPO

        if not MASKDINO_REPO.exists():
            raise FileNotFoundError(
                f"MaskDINO repo not found: {MASKDINO_REPO}\n"
                f"Clone: git clone https://github.com/IDEA-Research/MaskDINO.git {MASKDINO_REPO}"
            )

        sys.path.insert(0, str(MASKDINO_REPO))
        from detectron2.projects.deeplab import add_deeplab_config
        from maskdino import add_maskdino_config

        cfg = d2['get_cfg']()
        add_deeplab_config(cfg)
        add_maskdino_config(cfg)

        config_path = MASKDINO_REPO / "configs" / "coco" / "instance-segmentation" / self.model_info["config"]
        cfg.merge_from_file(str(config_path))

        cfg.MODEL.WEIGHTS = self.model_info["weights"]
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = self.num_classes

        # Gradient clipping fix
        cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
        cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
        cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.01
        cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

        # AMP 비활성화 (안정성)
        cfg.SOLVER.AMP.ENABLED = False

        return cfg

    def _setup_mask2former(self, d2):
        """Mask2Former 전용 설정"""
        from training.config import MASK2FORMER_REPO

        if not MASK2FORMER_REPO.exists():
            raise FileNotFoundError(
                f"Mask2Former repo not found: {MASK2FORMER_REPO}\n"
                f"Clone: git clone https://github.com/facebookresearch/Mask2Former.git {MASK2FORMER_REPO}"
            )

        sys.path.insert(0, str(MASK2FORMER_REPO))
        from detectron2.projects.deeplab import add_deeplab_config

        # Mask2Former __init__이 데이터셋 중복 등록 오류를 발생시킴
        # config와 모델만 직접 임포트
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mask2former_config",
            str(MASK2FORMER_REPO / "mask2former" / "config.py")
        )
        m2f_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m2f_config)
        add_maskformer2_config = m2f_config.add_maskformer2_config

        # 모델 등록을 위해 modeling 모듈도 임포트 (데이터셋 등록 안 함)
        spec2 = importlib.util.spec_from_file_location(
            "mask2former_model",
            str(MASK2FORMER_REPO / "mask2former" / "maskformer_model.py")
        )
        m2f_model = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(m2f_model)

        cfg = d2['get_cfg']()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)

        config_path = MASK2FORMER_REPO / "configs" / "coco" / "instance-segmentation" / "swin" / self.model_info["config"]
        if not config_path.exists():
            # fallback to R50 config
            config_path = MASK2FORMER_REPO / "configs" / "coco" / "instance-segmentation" / self.model_info["config"]
        cfg.merge_from_file(str(config_path))

        cfg.MODEL.WEIGHTS = self.model_info["weights"]
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = self.num_classes

        cfg.SOLVER.AMP.ENABLED = False

        return cfg

    def _setup_standard(self, d2):
        """표준 detectron2 모델 (Mask R-CNN, Cascade Mask R-CNN)"""
        cfg = d2['get_cfg']()
        cfg.merge_from_file(d2['model_zoo'].get_config_file(self.model_info["config"]))
        cfg.MODEL.WEIGHTS = self.model_info["weights"]
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        return cfg

    def setup(self, train_images_dir: Path, train_ann_path: Path,
              val_images_dir: Path, val_ann_path: Path,
              output_dir: Path, hyperparams: dict):
        """학습 환경 설정"""
        d2 = self._import_detectron2()

        # 데이터셋 등록
        from training.data_pipeline import register_for_detectron2

        self.train_dataset_name = f"train_{self.model_name}_{id(self)}"
        self.val_dataset_name = f"val_{self.model_name}_{id(self)}"

        register_for_detectron2(
            self.train_dataset_name, train_images_dir, train_ann_path, self.thing_classes
        )
        register_for_detectron2(
            self.val_dataset_name, val_images_dir, val_ann_path, self.thing_classes
        )

        # 모델별 config 설정
        if self.model_name == "maskdino":
            cfg = self._setup_maskdino(d2)
        elif self.model_name == "mask2former":
            cfg = self._setup_mask2former(d2)
        else:
            cfg = self._setup_standard(d2)

        # 공통 설정
        cfg.DATASETS.TRAIN = (self.train_dataset_name,)
        cfg.DATASETS.TEST = (self.val_dataset_name,)

        batch_size = hyperparams.get("batch_size", 2)
        cfg.SOLVER.IMS_PER_BATCH = batch_size
        cfg.SOLVER.BASE_LR = hyperparams.get("lr", 1e-4)

        # 에폭→이터레이션 변환
        from training.data_pipeline import load_coco
        train_data = load_coco(train_ann_path)
        n_train_images = len(train_data["images"])
        self.iters_per_epoch = max(1, math.ceil(n_train_images / batch_size))

        max_epochs = hyperparams.get("max_epochs", 300)
        max_iter = max_epochs * self.iters_per_epoch
        cfg.SOLVER.MAX_ITER = max_iter

        warmup_epochs = hyperparams.get("warmup_epochs", 5)
        cfg.SOLVER.WARMUP_ITERS = warmup_epochs * self.iters_per_epoch

        eval_period_epochs = hyperparams.get("eval_period_epochs", 5)
        eval_period = eval_period_epochs * self.iters_per_epoch
        cfg.TEST.EVAL_PERIOD = eval_period

        checkpoint_period_epochs = hyperparams.get("checkpoint_period_epochs", 10)
        cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period_epochs * self.iters_per_epoch

        # LR decay steps (마지막 30%, 10%)
        cfg.SOLVER.STEPS = (int(max_iter * 0.7), int(max_iter * 0.9))

        # Early stopping 설정 저장
        patience = hyperparams.get("early_stopping_patience", 15)
        es_metric = hyperparams.get("early_stopping_metric", "segm/AP")
        self.early_stopping_hook = EarlyStoppingHook(
            eval_period=eval_period,
            patience=patience,
            metric_name=es_metric,
            iters_per_epoch=self.iters_per_epoch,
        )

        print(f"  Train images: {n_train_images}, iters/epoch: {self.iters_per_epoch}")
        print(f"  Max epochs: {max_epochs} ({max_iter} iters)")
        print(f"  Eval every {eval_period_epochs} epochs ({eval_period} iters)")
        print(f"  Early stopping patience: {patience} evals ({patience * eval_period_epochs} epochs)")

        cfg.INPUT.MASK_FORMAT = "polygon"
        min_size = hyperparams.get("input_min_size", (480, 512, 544, 576, 608, 640))
        max_size = hyperparams.get("input_max_size", 800)
        cfg.INPUT.MIN_SIZE_TRAIN = min_size
        cfg.INPUT.MAX_SIZE_TRAIN = max_size
        cfg.INPUT.MIN_SIZE_TEST = min_size[-1] if isinstance(min_size, (list, tuple)) else min_size
        cfg.INPUT.MAX_SIZE_TEST = max_size

        cfg.OUTPUT_DIR = str(output_dir)
        cfg.SEED = hyperparams.get("seed", 42)

        self.output_dir = output_dir
        self.cfg = cfg
        os.makedirs(str(output_dir), exist_ok=True)

        # config 저장
        with open(output_dir / "config.yaml", "w") as f:
            f.write(cfg.dump())

        print(f"  Config saved: {output_dir / 'config.yaml'}")

    def _get_trainer_class(self):
        """모델별 Trainer 클래스 반환"""
        from detectron2.engine import DefaultTrainer
        from detectron2.data import build_detection_train_loader
        from detectron2.evaluation import COCOEvaluator

        if self.model_name == "maskdino":
            from training.utils.maskdino_mapper import MaskDINODatasetMapper

            class MaskDINOTrainer(DefaultTrainer):
                @classmethod
                def build_train_loader(cls, cfg):
                    mapper = MaskDINODatasetMapper(cfg, is_train=True)
                    return build_detection_train_loader(cfg, mapper=mapper)

                @classmethod
                def build_evaluator(cls, cfg, dataset_name, output_folder=None):
                    if output_folder is None:
                        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
                    return COCOEvaluator(dataset_name, output_dir=output_folder)

            return MaskDINOTrainer

        elif self.model_name == "mask2former":
            # Mask2Former도 커스텀 mapper가 필요할 수 있음
            class Mask2FormerTrainer(DefaultTrainer):
                @classmethod
                def build_evaluator(cls, cfg, dataset_name, output_folder=None):
                    if output_folder is None:
                        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
                    return COCOEvaluator(dataset_name, output_dir=output_folder)

            return Mask2FormerTrainer

        else:
            # Mask R-CNN, Cascade Mask R-CNN
            class StandardTrainer(DefaultTrainer):
                @classmethod
                def build_evaluator(cls, cfg, dataset_name, output_folder=None):
                    if output_folder is None:
                        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
                    return COCOEvaluator(dataset_name, output_dir=output_folder)

            return StandardTrainer

    def train(self) -> Dict[str, Any]:
        """학습 실행"""
        if self.cfg is None:
            raise RuntimeError("setup()을 먼저 호출하세요")

        TrainerClass = self._get_trainer_class()
        trainer = TrainerClass(self.cfg)
        trainer.resume_or_load(resume=False)

        # Early stopping hook 등록
        if self.early_stopping_hook is not None:
            self.early_stopping_hook.trainer = trainer
            trainer.register_hooks([self.early_stopping_hook])
            # hook을 eval hook 뒤로 이동 (eval 결과를 읽어야 하므로)
            # detectron2는 hooks[-1]이 가장 마지막에 실행됨
            hooks = trainer._hooks
            es_hook = hooks.pop()
            hooks.append(es_hook)

        # GPU 메모리 추적 시작
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        max_epochs = self.cfg.SOLVER.MAX_ITER / self.iters_per_epoch if self.iters_per_epoch else "?"

        print(f"\n{'='*60}")
        print(f"  학습 시작: {self.model_info['display_name']}")
        print(f"  MAX_EPOCHS: {max_epochs} ({self.cfg.SOLVER.MAX_ITER} iters)")
        print(f"  LR: {self.cfg.SOLVER.BASE_LR}")
        print(f"  BATCH: {self.cfg.SOLVER.IMS_PER_BATCH}")
        print(f"  OUTPUT: {self.cfg.OUTPUT_DIR}")
        print(f"{'='*60}\n")

        trainer.train()

        # GPU 메모리 측정
        peak_memory_mb = None
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # metrics 수집
        final_iter = trainer.iter
        final_epoch = final_iter / self.iters_per_epoch if self.iters_per_epoch else 0

        metrics = {
            "model": self.model_name,
            "status": "completed",
            "total_iters": final_iter,
            "total_epochs": round(final_epoch, 1),
            "iters_per_epoch": self.iters_per_epoch,
            "peak_memory_mb": round(peak_memory_mb, 1) if peak_memory_mb else None,
        }

        # Early stopping 정보
        if self.early_stopping_hook is not None:
            es = self.early_stopping_hook
            metrics["early_stopped"] = es.stopped_iter is not None
            metrics["early_stop_epoch"] = round(es.stopped_epoch, 1) if es.stopped_epoch else None
            metrics["early_stop_iter"] = es.stopped_iter
            metrics["best_metric_value"] = round(es.best_metric, 4) if es.best_metric > -float("inf") else None
            metrics["best_metric_iter"] = es.best_iter

        metrics_file = Path(self.cfg.OUTPUT_DIR) / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                for line in f:
                    pass
                try:
                    metrics["last_metrics"] = json.loads(line)
                except json.JSONDecodeError:
                    pass

        return metrics

    def evaluate(self, model_path: Optional[Path] = None) -> Dict[str, float]:
        """평가 실행"""
        if self.cfg is None:
            raise RuntimeError("setup()을 먼저 호출하세요")

        d2 = self._import_detectron2()

        if model_path is None:
            model_path = self.get_final_model_path()
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"\n평가 시작: {model_path}")

        model = d2['build_model'](self.cfg)
        model.eval()

        checkpointer = d2['DetectionCheckpointer'](model)
        checkpointer.load(str(model_path))

        eval_output = Path(self.cfg.OUTPUT_DIR) / "eval_results"
        eval_output.mkdir(parents=True, exist_ok=True)

        evaluator = d2['COCOEvaluator'](
            self.val_dataset_name,
            output_dir=str(eval_output),
            tasks=("bbox", "segm"),
        )

        val_loader = d2['build_detection_test_loader'](self.cfg, self.val_dataset_name)
        results = d2['inference_on_dataset'](model, val_loader, evaluator)

        # 결과 정리
        flat_results = {}
        for task in ["bbox", "segm"]:
            if task in results:
                for k, v in results[task].items():
                    flat_results[f"{task}_{k}"] = float(v)

        # 저장
        results_file = eval_output / "results.json"
        with open(results_file, 'w') as f:
            json.dump(flat_results, f, indent=2)

        print(f"  평가 결과 저장: {results_file}")
        for k, v in flat_results.items():
            if "AP" in k and "AP50" not in k and "AP75" not in k and "APs" not in k and "APm" not in k and "APl" not in k:
                print(f"  {k}: {v:.4f}")

        return flat_results
