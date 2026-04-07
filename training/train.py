"""
통합 학습 CLI (에폭 기반 + Early Stopping)

사용법:
    # 단일 실행
    python -m training.train --category Cable --experiment exp2 --condition cond1 --model mask_rcnn

    # 전체 조건 실행
    python -m training.train --category Cable --experiment exp2 --condition all --model maskdino

    # 7모델 비교
    python -m training.train --category Cable --experiment exp3 --condition original_only --model all

    # 하이퍼파라미터 오버라이드
    python -m training.train --category Screw --experiment exp2 --condition cond3 --model maskdino --max-epochs 500 --lr 5e-5

    # Early stopping patience 조절
    python -m training.train --category Cable --experiment exp1 --condition all --model mask_rcnn --patience 20

    # 평가만
    python -m training.train --category Cable --experiment exp2 --condition cond1 --model mask_rcnn --eval-only
"""

import argparse
import json
import math
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
    """단일 (카테고리, 실험, 조건, 모델, seed) 조합 실행"""
    seed = hyperparams.get("seed", 42)
    output_dir = get_output_dir(experiment, condition, category, model_name, seed=seed)
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
        result["train_time_sec"] = round(elapsed, 1)
        result["peak_memory_mb"] = train_results.get("peak_memory_mb")
        result["early_stopped"] = train_results.get("early_stopped", False)
        result["early_stop_epoch"] = train_results.get("early_stop_epoch")
        result["total_epochs"] = train_results.get("total_epochs")
        print(f"\n  학습 완료: {elapsed:.1f}초")
        if train_results.get("peak_memory_mb"):
            print(f"  GPU 피크 메모리: {train_results['peak_memory_mb']:.1f} MB")
        if train_results.get("early_stopped"):
            print(f"  Early stopped at epoch {train_results['early_stop_epoch']}")

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

    # 새 결과 추가 (중복 시 덮어쓰기) - seed 포함 키로 구분
    def _key(r):
        return f"{r['category']}_{r['experiment']}_{r['condition']}_{r['model']}_seed{r.get('seed', 'NA')}"

    existing_keys = {_key(r) for r in existing}

    for r in all_results:
        k = _key(r)
        if k in existing_keys:
            existing = [e for e in existing if _key(e) != k]
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
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None,
                        help='단일 시드 (단일 실행 시). --seeds와 함께 쓰지 말 것')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='다중 시드 반복 실행 (예: --seeds 42 43 44). '
                             '결과는 {output}/seed_{N}/ 하위에 저장')
    parser.add_argument('--eval-period-epochs', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None,
                        help='Early stopping patience (eval 횟수)')

    args = parser.parse_args()

    # 하이퍼파라미터 구성
    hyperparams = dict(DEFAULT_HYPERPARAMS)
    if args.max_epochs is not None:
        hyperparams["max_epochs"] = args.max_epochs
    if args.lr is not None:
        hyperparams["lr"] = args.lr
    if args.batch_size is not None:
        hyperparams["batch_size"] = args.batch_size
    if args.seed is not None:
        hyperparams["seed"] = args.seed
    if args.eval_period_epochs is not None:
        hyperparams["eval_period_epochs"] = args.eval_period_epochs
    if args.patience is not None:
        hyperparams["early_stopping_patience"] = args.patience

    # 범위 결정
    categories = list(CATEGORIES.keys()) if args.category == 'all' else [args.category]
    exp_info = get_experiment_info(args.experiment)
    conditions = list(exp_info["conditions"].keys()) if args.condition == 'all' else [args.condition]

    if args.model == 'all':
        models = exp_info.get("models", list(MODELS.keys()))
    else:
        models = [args.model]

    # 시드 목록 결정 (우선순위):
    #   1) --seeds 지정 → 해당 목록
    #   2) --seed 지정  → 단일 시드
    #   3) 아무것도 없음 → DEFAULT_HYPERPARAMS["seeds"] (기본 다중 시드, 예: [42,43,44])
    if args.seeds is not None:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(DEFAULT_HYPERPARAMS.get("seeds", [DEFAULT_HYPERPARAMS["seed"]]))

    # 실행 계획 출력
    total = len(categories) * len(conditions) * len(models) * len(seeds)
    print(f"\n{'='*70}")
    print(f"  실험: {args.experiment}")
    print(f"  카테고리: {categories}")
    print(f"  조건: {conditions}")
    print(f"  모델: {models}")
    print(f"  시드: {seeds}")
    print(f"  총 실행 수: {total}")
    print(f"  {'평가만' if args.eval_only else '학습 + 평가'}")
    print(f"{'='*70}")

    # 실행
    all_results = []
    completed = 0

    for cat in categories:
        for cond in conditions:
            for model in models:
                for seed in seeds:
                    completed += 1
                    print(f"\n[{completed}/{total}] seed={seed}")
                    # 시드별 hyperparams 사본 생성 (seed만 교체)
                    hp_run = dict(hyperparams)
                    hp_run["seed"] = seed
                    try:
                        result = run_single(
                            category=cat,
                            experiment=args.experiment,
                            condition=cond,
                            model_name=model,
                            hyperparams=hp_run,
                            eval_only=args.eval_only,
                        )
                        result["seed"] = seed
                        all_results.append(result)
                    except Exception as e:
                        print(f"\n  [ERROR] {cat}/{cond}/{model}/seed_{seed}: {e}")
                        import traceback
                        traceback.print_exc()
                        all_results.append({
                            "category": cat, "experiment": args.experiment,
                            "condition": cond, "model": model, "seed": seed,
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
        seed_str = f"/seed_{r.get('seed')}" if r.get('seed') is not None else ""
        parts = [f"[{status}] {r['category']}/{r['condition']}/{r['model']}{seed_str}"]
        if "eval" in r:
            segm_ap = r["eval"].get("segm_AP", r["eval"].get("coco/segm_mAP", "N/A"))
            parts.append(f"segm_AP={segm_ap}")
        if r.get("total_epochs"):
            parts.append(f"epochs={r['total_epochs']}")
        if r.get("early_stopped"):
            parts.append(f"early_stop@{r['early_stop_epoch']}")
        if r.get("peak_memory_mb"):
            parts.append(f"mem={r['peak_memory_mb']:.0f}MB")
        if r.get("train_time_sec"):
            parts.append(f"time={r['train_time_sec']:.0f}s")
        print(f"  {' | '.join(parts)}")


if __name__ == '__main__':
    main()
