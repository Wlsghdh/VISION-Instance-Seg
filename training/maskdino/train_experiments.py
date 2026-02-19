"""
MaskDINO 학습 스크립트 - AA_CV_R 13개 실험 자동 실행
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch

# MaskDINO 경로 추가
MASKDINO_DIR = Path("/data2/project/2026winter/jjh0709/MaskDINO")
sys.path.insert(0, str(MASKDINO_DIR))

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer

# MaskDINO imports
from maskdino import add_maskdino_config

# 실험 데이터셋 등록
from register_experiments import register_all_experiments, get_dataset_names, EXPERIMENT_NAMES

# Paths
AA_CV_R_DIR = Path("/data2/project/2026winter/jjh0709/AA_CV_R")
RESULTS_DIR = AA_CV_R_DIR / "results" / "maskdino"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Base config 경로
BASE_CONFIG = MASKDINO_DIR / "configs" / "coco" / "instance-segmentation" / "cable_thunderbolt.yaml"


class ExperimentTrainer(DefaultTrainer):
    """커스텀 Trainer with evaluation"""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def setup_cfg(exp_name, output_dir, args):
    """
    실험별 config 설정
    """
    cfg = get_cfg()
    add_maskdino_config(cfg)

    # Base config 로드
    cfg.merge_from_file(str(BASE_CONFIG))

    # 실험별 데이터셋 설정
    train_dataset, test_dataset = get_dataset_names(exp_name)
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = (test_dataset,)

    # Output 디렉토리
    cfg.OUTPUT_DIR = str(output_dir)

    # 하이퍼파라미터 설정
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = tuple([int(args.max_iter * 0.8), int(args.max_iter * 0.9)])
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.SOLVER.AMP.ENABLED = args.amp

    # Test 설정
    cfg.TEST.EVAL_PERIOD = args.eval_period

    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = args.num_workers

    # Random seed
    cfg.SEED = args.seed

    # Command line overrides
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    return cfg


def train_experiment(exp_name, args):
    """
    단일 실험 학습
    """
    print("\n" + "=" * 80)
    print(f"Training: {exp_name}")
    print("=" * 80)

    # Output 디렉토리
    output_dir = RESULTS_DIR / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config 설정
    cfg = setup_cfg(exp_name, output_dir, args)
    default_setup(cfg, args)

    # Logger 설정
    logger = setup_logger(output=str(output_dir / "train.log"), name=exp_name)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Config:\n{cfg}")

    # 학습
    trainer = ExperimentTrainer(cfg)

    # Resume or load pretrained model
    if args.resume:
        trainer.resume_or_load(resume=True)
    else:
        # Pretrained model 로드 (선택사항)
        if args.pretrained_model:
            checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
            checkpointer.load(args.pretrained_model)
            logger.info(f"Loaded pretrained model from {args.pretrained_model}")
        else:
            trainer.resume_or_load(resume=False)

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Final evaluation
    logger.info("Running final evaluation...")
    results = trainer.test(cfg, trainer.model)

    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_file}")
    logger.info(f"Training complete: {exp_name}")

    return results


def train_all_experiments(args):
    """
    모든 실험 순차 실행
    """
    print("\n" + "=" * 80)
    print(f"Training ALL experiments with MaskDINO")
    print(f"Total experiments: {len(EXPERIMENT_NAMES)}")
    print("=" * 80)

    all_results = {}

    for i, exp_name in enumerate(EXPERIMENT_NAMES, 1):
        print(f"\n[{i}/{len(EXPERIMENT_NAMES)}] {exp_name}")

        try:
            results = train_experiment(exp_name, args)
            all_results[exp_name] = results
        except Exception as e:
            print(f"❌ Error training {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp_name] = {"error": str(e)}

        print(f"\nProgress: {i}/{len(EXPERIMENT_NAMES)} completed")

    # 전체 결과 저장
    summary_file = RESULTS_DIR / "all_results.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ All experiments completed!")
    print(f"Summary saved to {summary_file}")
    print("=" * 80)

    # 결과 요약 출력
    print_results_summary(all_results)

    return all_results


def print_results_summary(all_results):
    """결과 요약 출력"""
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"{'Experiment':<45s} {'bbox/AP':>10s} {'bbox/AP50':>10s} {'segm/AP':>10s} {'segm/AP50':>10s}")
    print("-" * 90)

    for exp_name, results in all_results.items():
        if 'error' in results:
            print(f"{exp_name:<45s} {'ERROR':>10s}")
        elif 'bbox' in results and 'segm' in results:
            bbox_ap = results['bbox'].get('AP', 0.0)
            bbox_ap50 = results['bbox'].get('AP50', 0.0)
            segm_ap = results['segm'].get('AP', 0.0)
            segm_ap50 = results['segm'].get('AP50', 0.0)
            print(f"{exp_name:<45s} {bbox_ap:>10.2f} {bbox_ap50:>10.2f} {segm_ap:>10.2f} {segm_ap50:>10.2f}")
        else:
            print(f"{exp_name:<45s} {'N/A':>10s}")


def main():
    parser = argparse.ArgumentParser(description='Train MaskDINO on AA_CV_R experiments')

    # Experiment selection
    parser.add_argument('--exp', type=str, default='all',
                        help='Experiment name to run (or "all" for all experiments)')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size (default: 2)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate (default: 5e-5)')
    parser.add_argument('--max-iter', type=int, default=10000,
                        help='Maximum iterations (default: 10000)')
    parser.add_argument('--checkpoint-period', type=int, default=1000,
                        help='Checkpoint save period (default: 1000)')
    parser.add_argument('--eval-period', type=int, default=1000,
                        help='Evaluation period (default: 1000)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Enable automatic mixed precision (default: True)')

    # Model
    parser.add_argument('--pretrained-model', type=str, default=None,
                        help='Path to pretrained model (optional)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')

    # Detectron2 options
    parser.add_argument('--config-file', type=str, default='',
                        help='Path to config file (optional, overrides default)')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs')
    parser.add_argument('--num-machines', type=int, default=1,
                        help='Number of machines')
    parser.add_argument('--machine-rank', type=int, default=0,
                        help='Machine rank')
    parser.add_argument('--dist-url', type=str, default='auto',
                        help='Distributed training URL')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Modify config options using the command-line')

    args = parser.parse_args()

    # GPU 확인
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available, training will be very slow!")

    print(f"Using {args.num_gpus} GPU(s)")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # 데이터셋 등록
    print("\n" + "=" * 80)
    print("Registering datasets...")
    print("=" * 80)
    register_all_experiments()

    # 실험 실행
    if args.exp == 'all':
        # 모든 실험 실행
        train_all_experiments(args)
    else:
        # 단일 실험 실행
        if args.exp not in EXPERIMENT_NAMES:
            print(f"❌ Unknown experiment: {args.exp}")
            print(f"Available experiments: {EXPERIMENT_NAMES}")
            return

        train_experiment(args.exp, args)


if __name__ == '__main__':
    # 사용 예시:
    # python train_maskdino.py --exp all --batch-size 2 --max-iter 10000
    # python train_maskdino.py --exp exp_2_original26_only --batch-size 4
    # python train_maskdino.py --exp exp_1_original26_genai50 --pretrained-model path/to/model.pth

    main()
