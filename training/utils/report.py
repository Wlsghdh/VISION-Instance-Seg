"""
결과 시각화 및 리포트 생성

사용법:
    python -m training.utils.report
    python -m training.utils.report --experiment exp2 --metric segm_AP
    python -m training.utils.report --experiment exp1 --csv          # CSV (seed별 raw)
    python -m training.utils.report --experiment exp1 --raw          # 비교표도 seed별 raw
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from training.config import EVAL_DIR, REPORTS_DIR


def load_results() -> List[Dict]:
    """마스터 결과 파일 로드"""
    results_file = EVAL_DIR / "results.json"
    if not results_file.exists():
        print(f"결과 파일 없음: {results_file}")
        return []
    with open(results_file) as f:
        return json.load(f)


def _get_metric(eval_dict: Dict, metric: str) -> Optional[float]:
    """detectron2(segm_AP)와 mmdet(coco/segm_mAP) 키 형식 모두 지원"""
    val = eval_dict.get(metric)
    if val is not None:
        return float(val)
    # mmdet은 coco/{metric} 형식 — 단순 fallback
    mmdet_key = f"coco/{metric.replace('AP', 'mAP')}"
    val = eval_dict.get(mmdet_key)
    return float(val) if val is not None else None


def generate_csv(results: List[Dict], experiment: Optional[str] = None,
                 output_path: Optional[Path] = None) -> str:
    """결과를 CSV 형식으로 출력 (seed별 raw 값, 행 단위)"""
    if experiment:
        results = [r for r in results if r.get("experiment") == experiment]

    if not results:
        return "No results found."

    # 헤더 (seed 컬럼 포함)
    lines = ["category,experiment,condition,model,seed,bbox_AP,bbox_AP50,bbox_AP75,segm_AP,segm_AP50,segm_AP75,train_time_sec"]

    def _fmt(v):
        if v is None or v == "":
            return ""
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return ""

    for r in results:
        ev = r.get("eval", {})
        line = (
            f"{r.get('category', '')},{r.get('experiment', '')},"
            f"{r.get('condition', '')},{r.get('model', '')},"
            f"{r.get('seed', '')},"
            f"{_fmt(ev.get('bbox_AP', ev.get('coco/bbox_mAP')))},"
            f"{_fmt(ev.get('bbox_AP50', ev.get('coco/bbox_mAP_50')))},"
            f"{_fmt(ev.get('bbox_AP75', ev.get('coco/bbox_mAP_75')))},"
            f"{_fmt(ev.get('segm_AP', ev.get('coco/segm_mAP')))},"
            f"{_fmt(ev.get('segm_AP50', ev.get('coco/segm_mAP_50')))},"
            f"{_fmt(ev.get('segm_AP75', ev.get('coco/segm_mAP_75')))},"
            f"{r.get('train_time_sec', '')}"
        )
        lines.append(line)

    csv_text = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(csv_text)
        print(f"CSV 저장: {output_path}")

    return csv_text


def _aggregate_by_seed(matches: List[Dict], metric: str) -> Tuple[Optional[float], Optional[float], int]:
    """(cat, cond, model) 그룹 내 seed별 metric을 모아서 평균/표준편차 계산.

    Returns: (mean, stdev, n). 결과 없으면 (None, None, 0). seed가 1개면 stdev=None.
    """
    values = []
    for r in matches:
        v = _get_metric(r.get("eval", {}), metric)
        if v is not None:
            values.append(v)
    if not values:
        return None, None, 0
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else None
    return mean, stdev, len(values)


def generate_comparison_table(results: List[Dict], experiment: str,
                              metric: str = "segm_AP",
                              raw: bool = False) -> str:
    """조건 × 모델 비교 테이블 생성.

    raw=False (기본): 같은 (cat, cond, model)의 seed별 결과를 평균±표준편차로 집계
    raw=True: seed별로 별도 행 표시
    """
    results = [r for r in results if r.get("experiment") == experiment and "eval" in r]
    if not results:
        return "No results found."

    models = sorted(set(r["model"] for r in results))
    conditions = sorted(set(r["condition"] for r in results))
    categories = sorted(set(r["category"] for r in results))

    lines = []
    col_w = 22  # 평균±표준편차 표현용 컬럼 폭

    for cat in categories:
        lines.append(f"\n=== {cat} ===")

        if raw:
            # seed별 행
            seeds = sorted(set(r.get("seed") for r in results if r.get("seed") is not None))
            header = f"{'Condition':<14s}{'Seed':<8s}" + "".join(f"{m:<{col_w}s}" for m in models)
            lines.append(header)
            lines.append("-" * len(header))
            for cond in conditions:
                for seed in seeds:
                    row = f"{cond:<14s}{str(seed):<8s}"
                    for model in models:
                        match = next(
                            (r for r in results
                             if r["category"] == cat and r["condition"] == cond
                             and r["model"] == model and r.get("seed") == seed),
                            None,
                        )
                        if match:
                            v = _get_metric(match["eval"], metric)
                            row += f"{v:<{col_w}.4f}" if v is not None else f"{'N/A':<{col_w}s}"
                        else:
                            row += f"{'-':<{col_w}s}"
                    lines.append(row)
        else:
            # 집계 (mean ± stdev)
            header = f"{'Condition':<14s}" + "".join(f"{m:<{col_w}s}" for m in models)
            lines.append(header)
            lines.append("-" * len(header))
            for cond in conditions:
                row = f"{cond:<14s}"
                for model in models:
                    matches = [
                        r for r in results
                        if r["category"] == cat and r["condition"] == cond and r["model"] == model
                    ]
                    if not matches:
                        row += f"{'-':<{col_w}s}"
                        continue
                    mean, stdev, n = _aggregate_by_seed(matches, metric)
                    if mean is None:
                        row += f"{'N/A':<{col_w}s}"
                    elif stdev is None:
                        row += f"{f'{mean:.4f} (n=1)':<{col_w}s}"
                    else:
                        row += f"{f'{mean:.4f}±{stdev:.4f} (n={n})':<{col_w}s}"
                lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='결과 리포트 생성')
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--metric', type=str, default='segm_AP')
    parser.add_argument('--csv', action='store_true', help='CSV 출력 (seed별 raw)')
    parser.add_argument('--raw', action='store_true',
                        help='비교 표를 평균이 아닌 seed별 raw 값으로 출력')

    args = parser.parse_args()

    results = load_results()
    if not results:
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.csv:
        csv_path = REPORTS_DIR / f"results_{args.experiment or 'all'}.csv"
        csv_text = generate_csv(results, args.experiment, csv_path)
        print(csv_text)
    else:
        experiments = [args.experiment] if args.experiment else sorted(set(r.get("experiment", "") for r in results))
        for exp in experiments:
            table = generate_comparison_table(results, exp, args.metric, raw=args.raw)
            print(f"\n{'='*70}")
            mode = "raw (seed별)" if args.raw else "평균 ± 표준편차"
            print(f"  {exp} — {args.metric} ({mode})")
            print(f"{'='*70}")
            print(table)

            # 파일로도 저장
            suffix = "_raw" if args.raw else ""
            report_path = REPORTS_DIR / f"{exp}_{args.metric}{suffix}.txt"
            with open(report_path, 'w') as f:
                f.write(table)


if __name__ == '__main__':
    main()
