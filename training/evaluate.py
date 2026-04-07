"""
독립 평가 스크립트

사용법:
    # 특정 모델 평가 (기본: DEFAULT_HYPERPARAMS['seeds'] 전체 시드 평가)
    python -m training.evaluate --category Cable --experiment exp2 --condition cond1 --model mask_rcnn

    # 단일 시드만 평가
    python -m training.evaluate --category Cable --experiment exp2 --condition cond1 --model mask_rcnn --seed 42

    # 다중 시드 명시
    python -m training.evaluate --category Cable --experiment exp2 --condition cond1 --model mask_rcnn --seeds 42 43 44

    # 모든 학습 결과 일괄 평가
    python -m training.evaluate --category Cable --experiment exp2 --condition all --model all
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .config import (
    CATEGORIES, MODELS, EXPERIMENTS, DEFAULT_HYPERPARAMS,
    EVAL_DIR, TRAINING_DIR,
    get_experiment_info, get_output_dir,
)
from .train import create_adapter, save_results
from .data_pipeline import prepare_val_dataset


def evaluate_single(category: str, experiment: str, condition: str,
                    model_name: str, seed: int = None,
                    model_path: Path = None) -> Dict:
    """단일 (카테고리, 조건, 모델, seed) 조합 평가"""
    output_dir = get_output_dir(experiment, condition, category, model_name, seed=seed)

    if not output_dir.exists():
        print(f"  [SKIP] 학습 결과 없음: {output_dir}")
        return None

    val_images_dir, val_ann_path = prepare_val_dataset(category)

    adapter = create_adapter(model_name, category)

    # 학습 시 사용된 config가 있으면 로드
    hyperparams = dict(DEFAULT_HYPERPARAMS)

    # merged dataset 경로
    from .config import get_merged_dir
    merged_dir = get_merged_dir(experiment, condition, category)
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
                        help='단일 시드 평가')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='다중 시드 평가 (예: --seeds 42 43 44). '
                             '생략 시 DEFAULT_HYPERPARAMS["seeds"] 전체 시드 자동 평가')

    args = parser.parse_args()

    categories = list(CATEGORIES.keys()) if args.category == 'all' else [args.category]
    exp_info = get_experiment_info(args.experiment)
    conditions = list(exp_info["conditions"].keys()) if args.condition == 'all' else [args.condition]

    if args.model == 'all':
        models = list(MODELS.keys())
    else:
        models = [args.model]

    # 시드 목록 결정 (train.py와 동일한 우선순위)
    if args.seeds is not None:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(DEFAULT_HYPERPARAMS.get("seeds", [DEFAULT_HYPERPARAMS["seed"]]))

    model_path = Path(args.model_path) if args.model_path else None

    total = len(categories) * len(conditions) * len(models) * len(seeds)
    print(f"\n{'='*70}")
    print(f"  평가 대상: {total}개 ({len(categories)} cat × {len(conditions)} cond × {len(models)} model × {len(seeds)} seed)")
    print(f"  시드: {seeds}")
    print(f"{'='*70}")

    all_results = []
    for cat in categories:
        for cond in conditions:
            for model in models:
                for seed in seeds:
                    try:
                        result = evaluate_single(cat, args.experiment, cond, model,
                                                 seed=seed, model_path=model_path)
                        if result:
                            all_results.append(result)
                    except Exception as e:
                        print(f"  [ERROR] {cat}/{cond}/{model}/seed_{seed}: {e}")

    if all_results:
        save_results(all_results)

        print(f"\n{'='*70}")
        print(f"  평가 요약")
        print(f"{'='*70}")
        for r in all_results:
            if "eval" in r:
                segm_ap = r["eval"].get("segm_AP", r["eval"].get("coco/segm_mAP", "N/A"))
                bbox_ap = r["eval"].get("bbox_AP", r["eval"].get("coco/bbox_mAP", "N/A"))
                seed_str = f"/seed_{r['seed']}" if r.get('seed') is not None else ""
                print(f"  {r['category']}/{r['condition']}/{r['model']}{seed_str}: "
                      f"bbox_AP={bbox_ap}, segm_AP={segm_ap}")


if __name__ == '__main__':
    main()
