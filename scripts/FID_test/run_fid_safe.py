"""
FID 평가 - 리소스 절약 버전
서버 관리자에게 kill 안 당하도록:
  - CPU/GPU 메모리 최소화
  - 비교를 순차적으로 하나씩 수행
  - 각 비교 후 메모리 해제 (gc + torch cache clear)
  - OMP/MKL 스레드 제한

실행 순서:
  1. original_image vs ai_generated   (ai_origin)
  2. original_image vs normal_00      (origin_normal)
  3. ai_generated  vs normal_00       (ai_normal)

사용법:
  cd /home/jjh0709/gitrepo/VISION-Instance-Seg/scripts/FID_test
  python run_fid_safe.py
  python run_fid_safe.py --classes casting_Inclusoes casting_Rechupe
"""

import argparse
import csv
import gc
import os
import time
from datetime import datetime
from pathlib import Path

# === 리소스 제한 (import 전에 설정) ===
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 1개만 사용

import torch
from cleanfid import fid

# ===== 설정 =====
BASE_DIR = Path(__file__).parent
CLASSES = [
    "cable", "casting", "console", "cylinder", "wood",
    "casting_Inclusoes", "casting_Rechupe",
    "Console_Collision", "Console_Dirty", "Console_Gap", "Console_Scratch",
    "Cylinder_Chip", "Cylinder_PistonMiss", "Cylinder_Porosity", "Cylinder_RCS",
    "screw_defect",
    "Wood_impurities", "Wood_pits",
]
MIN_IMAGES = 5

# 비교 순서: ai_origin → origin_normal → ai_normal
COMPARISONS = [
    ("original_vs_ai_generated", "original_image", "ai_generated"),
    ("original_vs_normal_00", "original_image", "normal_00"),
    ("ai_vs_normal_00", "ai_generated", "normal_00"),
]


def count_images(folder: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".PNG", ".JPG", ".JPEG"}
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir() if f.suffix in exts)


def clear_memory():
    """GPU + CPU 메모리 해제"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def compute_fid_safe(dir1: Path, dir2: Path) -> float:
    score = fid.compute_fid(str(dir1), str(dir2), mode="legacy_pytorch", verbose=False)
    clear_memory()
    return round(float(score), 4)


def main():
    parser = argparse.ArgumentParser(description="FID Evaluation (Resource-Safe)")
    parser.add_argument("--classes", nargs="+", default=CLASSES)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--delay", type=int, default=5,
                        help="비교 사이 대기 시간(초) (default: 5)")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else BASE_DIR / f"fid_results_safe_{ts}.csv"

    print("=" * 60)
    print("  FID Evaluation (Resource-Safe Mode)")
    print("=" * 60)
    print(f"  Classes : {len(args.classes)}개")
    print(f"  Output  : {output_path}")
    print(f"  Threads : OMP={os.environ.get('OMP_NUM_THREADS')}")
    print(f"  GPU     : CUDA {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"  Delay   : {args.delay}초 between comparisons")
    print(f"  순서    : ai_origin → origin_normal → ai_normal")
    print()

    results = []
    total_start = time.time()

    # 비교 종류별로 순차 실행 (모든 클래스 → 다음 비교)
    for comp_idx, (comp_name, dir_a_name, dir_b_name) in enumerate(COMPARISONS):
        print(f"\n{'='*60}")
        print(f"  Phase {comp_idx+1}/3: {comp_name}")
        print(f"{'='*60}")

        for cls in args.classes:
            class_dir = BASE_DIR / cls
            dir_a = class_dir / dir_a_name
            dir_b = class_dir / dir_b_name

            n_a = count_images(dir_a)
            n_b = count_images(dir_b)

            if n_a < MIN_IMAGES or n_b < MIN_IMAGES:
                if not dir_a.exists() or not dir_b.exists():
                    continue
                print(f"  [SKIP] {cls}/{comp_name}: images {n_a} vs {n_b} (need {MIN_IMAGES}+)")
                continue

            print(f"  [{cls}] {dir_a_name}({n_a}) vs {dir_b_name}({n_b}) ...", end=" ", flush=True)
            t0 = time.time()

            try:
                score = compute_fid_safe(dir_a, dir_b)
                elapsed = time.time() - t0
                print(f"FID={score:.4f} ({elapsed:.1f}s)")
                results.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "class": cls,
                    "comparison": comp_name,
                    "dir_A": dir_a_name,
                    "dir_B": dir_b_name,
                    "n_A": n_a,
                    "n_B": n_b,
                    "fid_score": score,
                    "hypothesis_check": "—",
                })
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "class": cls,
                    "comparison": comp_name,
                    "dir_A": dir_a_name,
                    "dir_B": dir_b_name,
                    "n_A": n_a,
                    "n_B": n_b,
                    "fid_score": "ERROR",
                    "hypothesis_check": "—",
                })

            # 각 클래스 비교 후 잠시 대기
            clear_memory()
            time.sleep(args.delay)

        # Phase 끝나면 중간 저장 (kill 당해도 결과 보존)
        if results:
            _save_csv(results, output_path)
            print(f"\n  [중간 저장] {len(results)}건 → {output_path}")

        # Phase 사이 추가 대기
        if comp_idx < len(COMPARISONS) - 1:
            print(f"\n  다음 Phase까지 {args.delay * 2}초 대기...")
            clear_memory()
            time.sleep(args.delay * 2)

    # 가설 검증
    annotate_hypothesis(results)

    # 최종 저장
    _save_csv(results, output_path)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  완료! 총 {total_elapsed/60:.1f}분, {len(results)}건")
    print(f"  CSV: {output_path}")
    print(f"{'='*60}")

    # Summary
    print_summary(results)


def annotate_hypothesis(results: list):
    class_scores = {}
    for row in results:
        cls = row["class"]
        if isinstance(row["fid_score"], (int, float)):
            if cls not in class_scores:
                class_scores[cls] = {}
            class_scores[cls][row["comparison"]] = row["fid_score"]

    for row in results:
        cls = row["class"]
        if cls not in class_scores:
            continue
        scores = class_scores[cls]
        fid_normal = scores.get("original_vs_normal_00")
        fid_ai = scores.get("original_vs_ai_generated")
        if fid_normal is None or fid_ai is None:
            continue

        if row["comparison"] == "original_vs_ai_generated":
            if fid_ai < fid_normal:
                row["hypothesis_check"] = f"PASS (AI {fid_ai:.2f} < Normal {fid_normal:.2f})"
            else:
                row["hypothesis_check"] = f"FAIL (AI {fid_ai:.2f} >= Normal {fid_normal:.2f})"
        elif row["comparison"] == "original_vs_normal_00":
            if fid_ai < fid_normal:
                row["hypothesis_check"] = "[ref] PASS"
            else:
                row["hypothesis_check"] = "[ref] FAIL"


def _save_csv(results: list, output_path: Path):
    fieldnames = [
        "timestamp", "class", "comparison",
        "dir_A", "dir_B", "n_A", "n_B",
        "fid_score", "hypothesis_check"
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: list):
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Class':<25}  {'Comparison':<28}  {'FID':>8}  {'Hypothesis'}")
    print(f"  {'-'*25}  {'-'*28}  {'-'*8}  {'-'*30}")
    for row in results:
        fid_str = f"{row['fid_score']:.4f}" if isinstance(row['fid_score'], float) else str(row['fid_score'])
        print(f"  {row['class']:<25}  {row['comparison']:<28}  {fid_str:>8}  {row['hypothesis_check']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
