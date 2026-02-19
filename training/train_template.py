"""
모델 학습 템플릿 스크립트
- 13개 실험을 순차적으로 또는 개별적으로 실행
- 동일한 하이퍼파라미터로 공정한 비교
- 결과 자동 저장 및 평가
"""

import os
import json
from pathlib import Path
import argparse

# ============================================================================
# 설정
# ============================================================================

# Paths
BASE_DIR = Path('/data2/project/2026winter/jjh0709/AA_CV_R')
EXPERIMENTS_DIR = BASE_DIR / 'experiments'
TEST_DIR = BASE_DIR / 'val'  # Test 데이터
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# 실험 리스트
EXPERIMENTS = [
    # 실험 1: GenAI 증강 데이터 양
    'exp_1_original26_genai50',
    'exp_1_original26_genai100',
    'exp_1_original26_genai150',
    'exp_1_original26_genai200',

    # 실험 2: 증강 방법 비교
    'exp_2_original26_only',  # Baseline
    'exp_2_original26_traditional50',
    'exp_2_original26_traditional100',
    'exp_2_original26_traditional150',
    'exp_2_original26_traditional200',
    'exp_2_original26_genai50_traditional',
    'exp_2_original26_genai100_traditional',
    'exp_2_original26_genai150_traditional',
    'exp_2_original26_genai200_traditional',
]

# 하이퍼파라미터 (모든 실험에 동일하게 적용)
HYPERPARAMS = {
    'epochs': 100,
    'batch_size': 8,  # GPU 메모리에 맞게 조정
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'optimizer': 'AdamW',
    'lr_scheduler': 'cosine',
    'warmup_epochs': 5,
    'img_size': 640,  # 또는 800, 1024
    'random_seed': 42,
    'early_stopping_patience': 15,
    'save_period': 10,  # 모델 저장 주기
}

# ============================================================================
# 모델별 학습 함수 (선택한 모델에 맞게 구현)
# ============================================================================

def train_yolo(exp_name, train_dir, test_dir, output_dir, hyperparams):
    """
    YOLOv8/v11 학습
    """
    from ultralytics import YOLO

    print(f"\n{'='*80}")
    print(f"Training YOLO on {exp_name}")
    print(f"{'='*80}")

    # COCO format을 YOLO format으로 변환 필요
    # TODO: COCO to YOLO conversion

    # 모델 초기화
    model = YOLO('yolov8n.pt')  # 또는 yolov8s.pt, yolov8m.pt

    # 학습
    results = model.train(
        data='data.yaml',  # TODO: 생성 필요
        epochs=hyperparams['epochs'],
        batch=hyperparams['batch_size'],
        imgsz=hyperparams['img_size'],
        lr0=hyperparams['learning_rate'],
        seed=hyperparams['random_seed'],
        project=str(output_dir),
        name=exp_name,
        exist_ok=True,
    )

    # 평가
    metrics = model.val()

    return {
        'mAP50': metrics.box.map50,
        'mAP75': metrics.box.map75,
        'mAP': metrics.box.map,
        'precision': metrics.box.p,
        'recall': metrics.box.r,
    }


def train_maskdino(exp_name, train_dir, test_dir, output_dir, hyperparams):
    """
    MaskDINO 학습
    """
    print(f"\n{'='*80}")
    print(f"Training MaskDINO on {exp_name}")
    print(f"{'='*80}")

    # MaskDINO 학습 코드
    # TODO: MaskDINO 학습 구현
    # - Config 파일 수정
    # - 데이터셋 경로 설정
    # - 학습 실행
    # - 평가 실행

    # Example structure:
    # from detectron2.config import get_cfg
    # from detectron2.engine import DefaultTrainer
    #
    # cfg = get_cfg()
    # cfg.merge_from_file("configs/maskdino_config.yaml")
    # cfg.DATASETS.TRAIN = (exp_name,)
    # cfg.DATASETS.TEST = ("test",)
    # cfg.SOLVER.MAX_ITER = ...
    # cfg.OUTPUT_DIR = str(output_dir)
    #
    # trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    # trainer.train()
    #
    # # Evaluate
    # from detectron2.evaluation import COCOEvaluator
    # evaluator = COCOEvaluator(...)
    # results = evaluator.evaluate()

    return {
        'mAP50': 0.0,  # TODO: 실제 결과로 대체
        'mAP75': 0.0,
        'mAP': 0.0,
        'precision': 0.0,
        'recall': 0.0,
    }


def train_faster_rcnn(exp_name, train_dir, test_dir, output_dir, hyperparams):
    """
    Faster R-CNN 학습 (Detectron2 사용)
    """
    print(f"\n{'='*80}")
    print(f"Training Faster R-CNN on {exp_name}")
    print(f"{'='*80}")

    # Detectron2 Faster R-CNN 학습 코드
    # TODO: Faster R-CNN 학습 구현

    return {
        'mAP50': 0.0,
        'mAP75': 0.0,
        'mAP': 0.0,
        'precision': 0.0,
        'recall': 0.0,
    }


# ============================================================================
# 실험 실행 함수
# ============================================================================

def run_experiment(exp_name, model_type, hyperparams):
    """
    단일 실험 실행
    """
    print(f"\n{'='*80}")
    print(f"Experiment: {exp_name}")
    print(f"Model: {model_type}")
    print(f"{'='*80}")

    # 경로 설정
    train_dir = EXPERIMENTS_DIR / exp_name
    test_dir = TEST_DIR
    output_dir = RESULTS_DIR / model_type / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 데이터셋 존재 확인
    if not train_dir.exists():
        print(f"❌ Train directory not found: {train_dir}")
        return None

    if not (train_dir / 'annotations.json').exists():
        print(f"❌ Annotations not found: {train_dir / 'annotations.json'}")
        return None

    # 모델 학습
    if model_type == 'yolo':
        results = train_yolo(exp_name, train_dir, test_dir, output_dir, hyperparams)
    elif model_type == 'maskdino':
        results = train_maskdino(exp_name, train_dir, test_dir, output_dir, hyperparams)
    elif model_type == 'faster_rcnn':
        results = train_faster_rcnn(exp_name, train_dir, test_dir, output_dir, hyperparams)
    else:
        print(f"❌ Unknown model type: {model_type}")
        return None

    # 결과 저장
    results['exp_name'] = exp_name
    results['model_type'] = model_type
    results['hyperparams'] = hyperparams

    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {results_file}")
    print(f"mAP@0.5: {results['mAP50']:.4f}")
    print(f"mAP@0.75: {results['mAP75']:.4f}")
    print(f"mAP@0.5:0.95: {results['mAP']:.4f}")

    return results


def run_all_experiments(model_type, hyperparams):
    """
    모든 실험 순차 실행
    """
    print(f"\n{'='*80}")
    print(f"Running ALL experiments with {model_type}")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"{'='*80}")

    all_results = []

    for i, exp_name in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] {exp_name}")

        results = run_experiment(exp_name, model_type, hyperparams)

        if results:
            all_results.append(results)

        print(f"\nProgress: {i}/{len(EXPERIMENTS)} completed")

    # 전체 결과 저장
    summary_file = RESULTS_DIR / model_type / 'all_results.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ All experiments completed!")
    print(f"Summary saved to {summary_file}")
    print(f"{'='*80}")

    # 결과 요약 출력
    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    print(f"{'Experiment':<45s} {'mAP@0.5':>10s} {'mAP@0.75':>10s} {'mAP':>10s}")
    print("-"*80)

    for res in all_results:
        print(f"{res['exp_name']:<45s} {res['mAP50']:>10.4f} {res['mAP75']:>10.4f} {res['mAP']:>10.4f}")

    return all_results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train models on experiments')
    parser.add_argument('--model', type=str, required=True,
                        choices=['yolo', 'maskdino', 'faster_rcnn'],
                        help='Model type to use')
    parser.add_argument('--exp', type=str, default='all',
                        help='Experiment name to run (or "all" for all experiments)')
    parser.add_argument('--epochs', type=int, default=HYPERPARAMS['epochs'],
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=HYPERPARAMS['batch_size'],
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=HYPERPARAMS['learning_rate'],
                        help='Learning rate')

    args = parser.parse_args()

    # 하이퍼파라미터 업데이트
    hyperparams = HYPERPARAMS.copy()
    hyperparams['epochs'] = args.epochs
    hyperparams['batch_size'] = args.batch_size
    hyperparams['learning_rate'] = args.lr

    # 실험 실행
    if args.exp == 'all':
        run_all_experiments(args.model, hyperparams)
    else:
        if args.exp not in EXPERIMENTS:
            print(f"❌ Unknown experiment: {args.exp}")
            print(f"Available experiments: {EXPERIMENTS}")
            return

        run_experiment(args.exp, args.model, hyperparams)


if __name__ == '__main__':
    # 사용 예시:
    # python train_template.py --model yolo --exp all
    # python train_template.py --model maskdino --exp exp_1_original26_genai50
    # python train_template.py --model faster_rcnn --exp all --epochs 50 --batch_size 16

    main()
