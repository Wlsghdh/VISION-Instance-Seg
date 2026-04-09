"""
독립 평가 스크립트

사용법:
    # 특정 모델 평가 (단일 시드, 기본 seed=42)
    python -m training.evaluate --category Cable --experiment exp2 --condition cond1 --model mask_rcnn

    # 다중 시드 일괄 평가 (seed42/43/44)
    python -m training.evaluate --category Cable --experiment exp2 --condition cond1 --model mask_rcnn --multi-seed

    # 특정 시드만 지정
    python -m training.evaluate --category Cable --experiment exp2 --condition cond1 --model mask_rcnn --seed 43

    # 모든 학습 결과 일괄 평가
    python -m training.evaluate --category Cable --experiment exp2 --condition all --model all --multi-seed
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .config import (
    CATEGORIES, MODELS, EXPERIMENTS, DEFAULT_HYPERPARAMS, MULTI_SEEDS,
    EVAL_DIR, TRAINING_DIR,
    get_experiment_info, get_output_dir,
)
from .train import create_adapter, save_results
from .data_pipeline import prepare_val_dataset


def evaluate_single(category: str, experiment: str, condition: str,
                    model_name: str, seed: int, model_path: Path = None) -> Dict:
    """단일 (모델, seed) 조합 평가"""
    output_dir = get_output_dir(experiment, condition, category, model_name, seed=seed)

    if not output_dir.exists():
        print(f"  [SKIP] 학습 결과 없음: {output_dir}")
        return None

    val_images_dir, val_ann_path = prepare_val_dataset(category)

    adapter = create_adapter(model_name, category)

    # 학습 시 사용된 config가 있으면 로드
    hyperparams = dict(DEFAULT_HYPERPARAMS)
    hyperparams["seed"] = seed

    # merged dataset 경로 (시드별 분리)
    from .config import get_merged_dir
    merged_dir = get_merged_dir(experiment, condition, category, seed=seed)
    train_images_dir = merged_dir / "images"
    train_ann_path = merged_dir / "annotations.json"

    if not train_ann_path.exists():
        print(f"  [WARN] 학습 데이터 없음, 더미 등록: {train_ann_path}")
        train_images_dir = val_images_dir
        train_ann_path = val_ann_path

    adapter.setup(
        train_images_dir=train_images_dir,
        train_ann_path=train_ann_path,
        val_images_dir=val_images_dir,
        val_ann_path=val_ann_path,
        output_dir=output_dir,
        hyperparams=hyperparams,
    )

    eval_results = adapter.evaluate(model_path)

    return {
        "category": category,
        "experiment": experiment,
        "condition": condition,
        "model": model_name,
        "seed": seed,
        "output_dir": str(output_dir),
        "eval": eval_results,
    }


def main():
    parser = argparse.ArgumentParser(description='독립 평가 스크립트')
    parser.add_argument('--category', type=str, required=True,
                        choices=list(CATEGORIES.keys()) + ['all'])
    parser.add_argument('--experiment', type=str, required=True,
                        choices=list(EXPERIMENTS.keys()))
    parser.add_argument('--condition', type=str, required=True)
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODELS.keys()) + ['all'])
    parser.add_argument('--model-path', type=str, default=None,
                        help='모델 경로 (미지정 시 자동 탐색)')
    parser.add_argument('--seed', type=int, default=None,
                        help='평가할 단일 시드 (default: DEFAULT_HYPERPARAMS의 seed)')
    parser.add_argument('--multi-seed', action='store_true',
                        help='MULTI_SEEDS의 모든 시드 일괄 평가')

    args = parser.parse_args()

    categories = list(CATEGORIES.keys()) if args.category == 'all' else [args.category]
    exp_info = get_experiment_info(args.experiment)
    conditions = list(exp_info["conditions"].keys()) if args.condition == 'all' else [args.condition]

    if args.model == 'all':
        models = list(MODELS.keys())
    else:
        models = [args.model]

    if args.multi_seed:
        seeds = list(MULTI_SEEDS)
    else:
        seeds = [args.seed if args.seed is not None else DEFAULT_HYPERPARAMS["seed"]]

    model_path = Path(args.model_path) if args.model_path else None

    all_results = []
    for cat in categories:
        for cond in conditions:
            for model in models:
                for seed in seeds:
                    try:
                        result = evaluate_single(cat, args.experiment, cond, model, seed, model_path)
                        if result:
                            all_results.append(result)
                    except Exception as e:
                        print(f"  [ERROR] {cat}/{cond}/{model}/seed{seed}: {e}")

    if all_results:
        save_results(all_results)

        print(f"\n{'='*70}")
        print(f"  평가 요약")
        print(f"{'='*70}")
        for r in all_results:
            if "eval" in r:
                segm_ap = r["eval"].get("segm_AP", r["eval"].get("coco/segm_mAP", "N/A"))
                bbox_ap = r["eval"].get("bbox_AP", r["eval"].get("coco/bbox_mAP", "N/A"))
                print(f"  {r['category']}/{r['condition']}/{r['model']}/seed{r['seed']}: "
                      f"bbox_AP={bbox_ap}, segm_AP={segm_ap}")


if __name__ == '__main__':
    main()
