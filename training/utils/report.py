"""
결과 시각화 및 리포트 생성

사용법:
    python -m training.utils.report
    python -m training.utils.report --experiment exp2 --metric segm_AP
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from training.config import EVAL_DIR, REPORTS_DIR


def load_results() -> List[Dict]:
    """마스터 결과 파일 로드"""
    results_file = EVAL_DIR / "results.json"
    if not results_file.exists():
        print(f"결과 파일 없음: {results_file}")
        return []
    with open(results_file) as f:
        return json.load(f)


def generate_csv(results: List[Dict], experiment: Optional[str] = None,
                 output_path: Optional[Path] = None) -> str:
    """결과를 CSV 형식으로 출력"""
    if experiment:
        results = [r for r in results if r.get("experiment") == experiment]

    if not results:
        return "No results found."

    # 헤더
    lines = ["category,experiment,condition,model,bbox_AP,bbox_AP50,bbox_AP75,segm_AP,segm_AP50,segm_AP75,train_time_sec"]

    for r in results:
        ev = r.get("eval", {})
        line = (
            f"{r.get('category', '')},{r.get('experiment', '')},"
            f"{r.get('condition', '')},{r.get('model', '')},"
            f"{ev.get('bbox_AP', ev.get('coco/bbox_mAP', '')):.4f},"
            f"{ev.get('bbox_AP50', ev.get('coco/bbox_mAP_50', '')):.4f},"
            f"{ev.get('bbox_AP75', ev.get('coco/bbox_mAP_75', '')):.4f},"
            f"{ev.get('segm_AP', ev.get('coco/segm_mAP', '')):.4f},"
            f"{ev.get('segm_AP50', ev.get('coco/segm_mAP_50', '')):.4f},"
            f"{ev.get('segm_AP75', ev.get('coco/segm_mAP_75', '')):.4f},"
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


def generate_comparison_table(results: List[Dict], experiment: str,
                              metric: str = "segm_AP") -> str:
    """조건 × 모델 비교 테이블 생성"""
    results = [r for r in results if r.get("experiment") == experiment and "eval" in r]
    if not results:
        return "No results found."

    # 모델/조건 목록 추출
    models = sorted(set(r["model"] for r in results))
    conditions = sorted(set(r["condition"] for r in results))
    categories = sorted(set(r["category"] for r in results))

    lines = []
    for cat in categories:
        lines.append(f"\n=== {cat} ===")

        # 헤더
        header = f"{'Condition':<20s}" + "".join(f"{m:<18s}" for m in models)
        lines.append(header)
        lines.append("-" * len(header))

        for cond in conditions:
            row = f"{cond:<20s}"
            for model in models:
                matches = [
                    r for r in results
                    if r["category"] == cat and r["condition"] == cond and r["model"] == model
                ]
                if matches:
                    ev = matches[0]["eval"]
                    # detectron2와 mmdet 키 형식 모두 지원
                    val = ev.get(metric, ev.get(f"coco/{metric.replace('_', '_')}", None))
                    if val is not None:
                        row += f"{val:<18.4f}"
                    else:
                        row += f"{'N/A':<18s}"
                else:
                    row += f"{'-':<18s}"
            lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='결과 리포트 생성')
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--metric', type=str, default='segm_AP')
    parser.add_argument('--csv', action='store_true', help='CSV 출력')

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
            table = generate_comparison_table(results, exp, args.metric)
            print(f"\n{'='*70}")
            print(f"  {exp} — {args.metric}")
            print(f"{'='*70}")
            print(table)

            # 파일로도 저장
            report_path = REPORTS_DIR / f"{exp}_{args.metric}.txt"
            with open(report_path, 'w') as f:
                f.write(table)


if __name__ == '__main__':
    main()
