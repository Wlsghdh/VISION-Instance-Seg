"""
실험1 single-category 결과를 markdown으로 요약.
results/evaluation/results.json 을 읽어 필터링 후 .md 생성.

사용:
    python scripts/summarize_exp1_singlecat.py \
        --output results/reports/exp1_singlecat_<TS>.md \
        --since-ts <unix_ts>
"""

import argparse
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "results" / "evaluation" / "results.json"

CATEGORIES = ["Cable", "Casting", "Console", "Cylinder", "Wood", "Screw"]
CONDITIONS = ["baseline", "genai_25", "genai_50", "genai_75", "genai_100", "genai_125"]
EXPERIMENT = "exp1"
MODEL = "mask_rcnn"


def fmt_time(sec):
    if sec is None or sec == "":
        return "-"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def fmt_num(x, digits=4):
    if x is None or x == "":
        return "-"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def fmt_mem(mb):
    if mb is None or mb == "":
        return "-"
    try:
        return f"{float(mb):.0f} MB"
    except (TypeError, ValueError):
        return str(mb)


def get_eval_metric(ev, *keys):
    for k in keys:
        v = ev.get(k)
        if v is not None and v != "":
            return v
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--since-ts", type=float, default=0.0,
                   help="이 unix timestamp 이후 results.json 변경분만 (전체일 땐 0)")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not RESULTS_FILE.exists():
        out_path.write_text(f"# 실험1 (single-category) 요약\n\n결과 파일 없음: `{RESULTS_FILE}`\n")
        print(f"[summary] no results.json yet → {out_path}")
        return

    with open(RESULTS_FILE) as f:
        all_results = json.load(f)

    # 필터링: exp1, mask_rcnn, 지정 카테고리
    rows = [
        r for r in all_results
        if r.get("experiment") == EXPERIMENT
        and r.get("model") == MODEL
        and r.get("category") in CATEGORIES
    ]

    # 키: (category, condition)
    by_key = {(r["category"], r["condition"]): r for r in rows}

    lines = []
    lines.append(f"# 실험1 single-category 결과 요약 (mask_rcnn)")
    lines.append("")
    lines.append(f"- 생성 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 결과 원본: `{RESULTS_FILE.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '?')}")
    lines.append(f"- 모델: `{MODEL}` / 실험: `{EXPERIMENT}`")
    lines.append("")
    completed = sum(1 for r in rows if "error" not in r and "eval" in r)
    failed = sum(1 for r in rows if "error" in r)
    lines.append(f"**진행 현황: {completed} / {len(CATEGORIES) * len(CONDITIONS)} 완료, {failed} 실패**")
    lines.append("")

    # 카테고리별 테이블
    for cat in CATEGORIES:
        lines.append(f"## {cat}")
        lines.append("")
        lines.append("| Condition | segm_AP | segm_AP50 | segm_AP75 | bbox_AP | Epochs | EarlyStop | GPU Peak | Train Time | Status |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")

        for cond in CONDITIONS:
            r = by_key.get((cat, cond))
            if r is None:
                lines.append(f"| {cond} | - | - | - | - | - | - | - | - | _대기_ |")
                continue

            if "error" in r:
                lines.append(f"| {cond} | - | - | - | - | - | - | - | - | **FAIL**: {r['error'][:60]} |")
                continue

            ev = r.get("eval", {}) or {}
            segm_ap   = get_eval_metric(ev, "segm_AP",   "coco/segm_mAP")
            segm_ap50 = get_eval_metric(ev, "segm_AP50", "coco/segm_mAP_50")
            segm_ap75 = get_eval_metric(ev, "segm_AP75", "coco/segm_mAP_75")
            bbox_ap   = get_eval_metric(ev, "bbox_AP",   "coco/bbox_mAP")

            total_epochs = r.get("total_epochs")
            early_stopped = r.get("early_stopped")
            es_epoch = r.get("early_stop_epoch")
            if early_stopped:
                es_str = f"@ ep{es_epoch}"
            elif early_stopped is False:
                es_str = "no"
            else:
                es_str = "-"

            peak_mem = r.get("peak_memory_mb")
            train_time = r.get("train_time_sec")

            status = "OK" if "eval" in r else "_학습만_"

            lines.append(
                f"| {cond} "
                f"| {fmt_num(segm_ap)} "
                f"| {fmt_num(segm_ap50)} "
                f"| {fmt_num(segm_ap75)} "
                f"| {fmt_num(bbox_ap)} "
                f"| {total_epochs if total_epochs else '-'} "
                f"| {es_str} "
                f"| {fmt_mem(peak_mem)} "
                f"| {fmt_time(train_time)} "
                f"| {status} |"
            )
        lines.append("")

    # 전체 집계
    if rows:
        ok_rows = [r for r in rows if "eval" in r and "error" not in r]
        if ok_rows:
            total_train = sum(r.get("train_time_sec") or 0 for r in ok_rows)
            peaks = [r.get("peak_memory_mb") for r in ok_rows if r.get("peak_memory_mb")]
            es_count = sum(1 for r in ok_rows if r.get("early_stopped"))
            lines.append("## 집계")
            lines.append("")
            lines.append(f"- 완료된 학습 누적 시간: **{fmt_time(total_train)}**")
            if peaks:
                lines.append(f"- GPU 피크 메모리 (완료 run 중 최대): **{fmt_mem(max(peaks))}**")
            lines.append(f"- Early stopping 발동 횟수: **{es_count} / {len(ok_rows)}**")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[summary] {out_path} ({completed} done, {failed} failed)")


if __name__ == "__main__":
    main()
