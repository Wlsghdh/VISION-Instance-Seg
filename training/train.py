"""
통합 학습 CLI

사용법:
    # 단일 실행
    python -m training.train --category Cable --experiment exp2 --condition cond1 --model mask_rcnn

    # 전체 조건 실행
    python -m training.train --category Cable --experiment exp2 --condition all --model maskdino

    # 7모델 비교
    python -m training.train --category Cable --experiment exp3 --condition original_only --model all

    # 하이퍼파라미터 오버라이드
    python -m training.train --category Screw --experiment exp2 --condition cond3 --model maskdino --max-iter 15000 --lr 5e-5

    # 평가만
    python -m training.train --category Cable --experiment exp2 --condition cond1 --model mask_rcnn --eval-only
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from .config import (
    CATEGORIES, MODELS, EXPERIMENTS, DEFAULT_HYPERPARAMS,
    EVAL_DIR,
    get_category_info, get_model_info, get_experiment_info, get_output_dir,
)
from .data_pipeline import prepare_dataset, prepare_val_dataset


def create_adapter(model_name: str, category: str):
    """모델명에 맞는 Adapter 인스턴스 생성"""
    model_info = get_model_info(model_name)
    cat_info = get_category_info(category)

    if model_info["framework"] == "detectron2":
        from .adapters.detectron2_adapter import Detectron2Adapter
        return Detectron2Adapter(
            model_name=model_name,
            num_classes=cat_info["num_classes"],
            thing_classes=cat_info["classes"],
            model_info=model_info,
        )
    elif model_info["framework"] == "mmdet":
        from .adapters.mmdet_adapter import MMDetAdapter
        return MMDetAdapter(
            model_name=model_name,
            num_classes=cat_info["num_classes"],
            thing_classes=cat_info["classes"],
            model_info=model_info,
        )
    else:
        raise ValueError(f"Unknown framework: {model_info['framework']}")


def run_single(category: str, experiment: str, condition: str,
               model_name: str, hyperparams: dict,
               eval_only: bool = False) -> Dict:
    """단일 (카테고리, 실험, 조건, 모델) 조합 실행"""
    output_dir = get_output_dir(experiment, condition, category, model_name)
    model_info = get_model_info(model_name)

    print(f"\n{'='*70}")
    print(f"  [{model_info['display_name']}] {category} / {experiment} / {condition}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}")

    # 1. 데이터 준비
    merged_dir = prepare_dataset(experiment, condition, category, hyperparams.get("seed", 42))
    train_images_dir = merged_dir / "images"
    train_ann_path = merged_dir / "annotations.json"

    # 2. Val 데이터 준비
    val_images_dir, val_ann_path = prepare_val_dataset(category)

    # 3. Adapter 생성 + 설정
    adapter = create_adapter(model_name, category)
    adapter.setup(
        train_images_dir=train_images_dir,
        train_ann_path=train_ann_path,
        val_images_dir=val_images_dir,
        val_ann_path=val_ann_path,
        output_dir=output_dir,
        hyperparams=hyperparams,
    )

    result = {
        "category": category,
        "experiment": experiment,
        "condition": condition,
        "model": model_name,
        "output_dir": str(output_dir),
    }

    # 4. 학습 또는 평가
    if eval_only:
        eval_results = adapter.evaluate()
        result["eval"] = eval_results
    else:
        start_time = time.time()
        train_results = adapter.train()
        elapsed = time.time() - start_time
        result["train"] = train_results
        result["train_time_sec"] = elapsed
        print(f"\n  학습 완료: {elapsed:.1f}초")

        # 학습 후 자동 평가
        try:
            eval_results = adapter.evaluate()
            result["eval"] = eval_results
        except Exception as e:
            print(f"  [WARN] 평가 실패: {e}")
            result["eval_error"] = str(e)

    return result


def save_results(all_results: List[Dict]):
    """전체 결과를 마스터 파일에 저장"""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    results_file = EVAL_DIR / "results.json"

    # 기존 결과 로드
    existing = []
    if results_file.exists():
        with open(results_file) as f:
            existing = json.load(f)

    # 새 결과 추가 (중복 시 덮어쓰기)
    existing_keys = set()
    for r in existing:
        key = f"{r['category']}_{r['experiment']}_{r['condition']}_{r['model']}"
        existing_keys.add(key)

    for r in all_results:
        key = f"{r['category']}_{r['experiment']}_{r['condition']}_{r['model']}"
        if key in existing_keys:
            existing = [e for e in existing
                        if f"{e['category']}_{e['experiment']}_{e['condition']}_{e['model']}" != key]
        existing.append(r)

    with open(results_file, 'w') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"\n결과 저장: {results_file} ({len(existing)}개 항목)")


def main():
    parser = argparse.ArgumentParser(
        description='통합 학습 CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--category', type=str, required=True,
                        choices=list(CATEGORIES.keys()) + ['all'])
    parser.add_argument('--experiment', type=str, required=True,
                        choices=list(EXPERIMENTS.keys()))
    parser.add_argument('--condition', type=str, required=True,
                        help='실험 조건 (또는 "all")')
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODELS.keys()) + ['all'],
                        help='모델 (또는 "all")')
    parser.add_argument('--eval-only', action='store_true',
                        help='평가만 실행 (학습 스킵)')

    # 하이퍼파라미터 오버라이드
    parser.add_argument('--max-iter', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--eval-period', type=int, default=None)

    args = parser.parse_args()

    # 하이퍼파라미터 구성
    hyperparams = dict(DEFAULT_HYPERPARAMS)
    if args.max_iter is not None:
        hyperparams["max_iter"] = args.max_iter
    if args.lr is not None:
        hyperparams["lr"] = args.lr
    if args.batch_size is not None:
        hyperparams["batch_size"] = args.batch_size
    if args.seed is not None:
        hyperparams["seed"] = args.seed
    if args.eval_period is not None:
        hyperparams["eval_period"] = args.eval_period

    # 범위 결정
    categories = list(CATEGORIES.keys()) if args.category == 'all' else [args.category]
    exp_info = get_experiment_info(args.experiment)
    conditions = list(exp_info["conditions"].keys()) if args.condition == 'all' else [args.condition]

    if args.model == 'all':
        models = exp_info.get("models", list(MODELS.keys()))
    else:
        models = [args.model]

    # 실행 계획 출력
    total = len(categories) * len(conditions) * len(models)
    print(f"\n{'='*70}")
    print(f"  실험: {args.experiment}")
    print(f"  카테고리: {categories}")
    print(f"  조건: {conditions}")
    print(f"  모델: {models}")
    print(f"  총 실행 수: {total}")
    print(f"  {'평가만' if args.eval_only else '학습 + 평가'}")
    print(f"{'='*70}")

    # 실행
    all_results = []
    completed = 0

    for cat in categories:
        for cond in conditions:
            for model in models:
                completed += 1
                print(f"\n[{completed}/{total}]")
                try:
                    result = run_single(
                        category=cat,
                        experiment=args.experiment,
                        condition=cond,
                        model_name=model,
                        hyperparams=hyperparams,
                        eval_only=args.eval_only,
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"\n  [ERROR] {cat}/{cond}/{model}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        "category": cat, "experiment": args.experiment,
                        "condition": cond, "model": model,
                        "error": str(e),
                    })

    # 결과 저장
    save_results(all_results)

    # 요약 출력
    print(f"\n{'='*70}")
    print(f"  완료: {completed}/{total}")
    print(f"{'='*70}")

    for r in all_results:
        status = "OK" if "error" not in r else "FAIL"
        eval_info = ""
        if "eval" in r:
            segm_ap = r["eval"].get("segm_AP", r["eval"].get("coco/segm_mAP", "N/A"))
            eval_info = f" segm_AP={segm_ap}"
        print(f"  [{status}] {r['category']}/{r['condition']}/{r['model']}{eval_info}")


if __name__ == '__main__':
    main()
