"""
FID (Fréchet Inception Distance) 평가 스크립트

비교 구조:
  FID_test/
  ├── {class}/
  │   ├── original_image/   ← 원본 데이터셋 이미지 (기준)
  │   ├── normal_00/        ← 정상 이미지 (결함 없는 레퍼런스)
  │   └── ai_generated/     ← Gemini AI 생성 이미지

비교 조합 (클래스별):
  1. original_image  vs  normal_00      → FID_original_vs_normal
  2. original_image  vs  ai_generated   → FID_original_vs_ai

가설: FID(original, ai_generated) < FID(original, normal_00)
      → AI 생성 이미지가 normal_00보다 원본과 더 유사한 분포를 가짐

사용법:
  cd /home/jjh0709/gitrepo/VISION-Instance-Seg/scripts/FID_test
  python run_fid.py
  python run_fid.py --classes cable casting  # 특정 클래스만
"""

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

from cleanfid import fid

# ===== 설정 =====
BASE_DIR = Path(__file__).parent
CLASSES = ["cable", "casting", "console", "cylinder", "wood"]
MIN_IMAGES = 5  # FID 계산에 필요한 최소 이미지 수


def count_images(folder: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".PNG", ".JPG", ".JPEG"}
    return sum(1 for f in folder.iterdir() if f.suffix in exts)


def check_folder(folder: Path, label: str) -> bool:
    if not folder.exists():
        print(f"  [SKIP] {label}: 폴더 없음 ({folder})")
        return False
    n = count_images(folder)
    if n < MIN_IMAGES:
        print(f"  [SKIP] {label}: 이미지 {n}장 (최소 {MIN_IMAGES}장 필요)")
        return False
    print(f"  [OK]   {label}: {n}장")
    return True


def compute_fid(dir1: Path, dir2: Path) -> float:
    score = fid.compute_fid(str(dir1), str(dir2), mode="legacy_pytorch", verbose=False)
    return round(float(score), 4)


def run_fid_for_class(class_name: str, results: list):
    print(f"\n{'='*60}")
    print(f"  Class: {class_name.upper()}")
    print(f"{'='*60}")

    class_dir = BASE_DIR / class_name
    orig_dir = class_dir / "original_image"
    normal_dir = class_dir / "normal_00"
    ai_dir = class_dir / "ai_generated"

    # 폴더 및 이미지 수 확인
    orig_ok = check_folder(orig_dir, "original_image")
    normal_ok = check_folder(normal_dir, "normal_00")
    ai_ok = check_folder(ai_dir, "ai_generated")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1) original vs normal_00
    if orig_ok and normal_ok:
        print(f"\n  Computing FID: original_image vs normal_00 ...", flush=True)
        try:
            score = compute_fid(orig_dir, normal_dir)
            print(f"  → FID(original, normal_00) = {score:.4f}")
            results.append({
                "timestamp": timestamp,
                "class": class_name,
                "comparison": "original_vs_normal_00",
                "dir_A": str(orig_dir.relative_to(BASE_DIR)),
                "dir_B": str(normal_dir.relative_to(BASE_DIR)),
                "n_A": count_images(orig_dir),
                "n_B": count_images(normal_dir),
                "fid_score": score,
                "hypothesis_check": "—",
            })
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({
                "timestamp": timestamp,
                "class": class_name,
                "comparison": "original_vs_normal_00",
                "dir_A": str(orig_dir.relative_to(BASE_DIR)),
                "dir_B": str(normal_dir.relative_to(BASE_DIR)),
                "n_A": count_images(orig_dir) if orig_dir.exists() else 0,
                "n_B": count_images(normal_dir) if normal_dir.exists() else 0,
                "fid_score": "ERROR",
                "hypothesis_check": "—",
            })
    else:
        print(f"  [SKIP] original_vs_normal_00 (이미지 부족)")

    # 2) original vs ai_generated
    if orig_ok and ai_ok:
        print(f"\n  Computing FID: original_image vs ai_generated ...", flush=True)
        try:
            score = compute_fid(orig_dir, ai_dir)
            print(f"  → FID(original, ai_generated) = {score:.4f}")
            results.append({
                "timestamp": timestamp,
                "class": class_name,
                "comparison": "original_vs_ai_generated",
                "dir_A": str(orig_dir.relative_to(BASE_DIR)),
                "dir_B": str(ai_dir.relative_to(BASE_DIR)),
                "n_A": count_images(orig_dir),
                "n_B": count_images(ai_dir),
                "fid_score": score,
                "hypothesis_check": "—",
            })
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({
                "timestamp": timestamp,
                "class": class_name,
                "comparison": "original_vs_ai_generated",
                "dir_A": str(orig_dir.relative_to(BASE_DIR)),
                "dir_B": str(ai_dir.relative_to(BASE_DIR)),
                "n_A": count_images(orig_dir) if orig_dir.exists() else 0,
                "n_B": count_images(ai_dir) if ai_dir.exists() else 0,
                "fid_score": "ERROR",
                "hypothesis_check": "—",
            })
    else:
        print(f"  [SKIP] original_vs_ai_generated (이미지 부족)")


def annotate_hypothesis(results: list):
    """
    가설 검증: FID(original, ai) < FID(original, normal)
    → ai가 normal보다 낮으면 가설 성립 (PASS)
    """
    # class별로 두 FID 수치 짝 맞추기
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
            if fid_ai is not None:
                if fid_ai < fid_normal:
                    row["hypothesis_check"] = f"[ref] PASS"
                else:
                    row["hypothesis_check"] = f"[ref] FAIL"


def save_csv(results: list, output_path: Path):
    fieldnames = [
        "timestamp", "class", "comparison",
        "dir_A", "dir_B", "n_A", "n_B",
        "fid_score", "hypothesis_check"
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  CSV saved: {output_path}")


def print_summary(results: list):
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Class':<10}  {'Comparison':<30}  {'FID':>8}  {'Hypothesis'}")
    print(f"  {'-'*10}  {'-'*30}  {'-'*8}  {'-'*30}")
    for row in results:
        fid_str = f"{row['fid_score']:.4f}" if isinstance(row['fid_score'], float) else str(row['fid_score'])
        print(f"  {row['class']:<10}  {row['comparison']:<30}  {fid_str:>8}  {row['hypothesis_check']}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="FID Evaluation Script")
    parser.add_argument(
        "--classes", nargs="+", default=CLASSES,
        choices=CLASSES,
        help=f"Classes to evaluate (default: all). Choices: {CLASSES}"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="CSV output path (default: fid_results_TIMESTAMP.csv)"
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else BASE_DIR / f"fid_results_{ts}.csv"

    print("=" * 60)
    print("  FID Evaluation")
    print("=" * 60)
    print(f"  Classes : {args.classes}")
    print(f"  Output  : {output_path}")
    print(f"  Min imgs: {MIN_IMAGES} per folder")
    print(f"\n  가설: FID(original, ai_generated) < FID(original, normal_00)")
    print(f"       → AI 생성 이미지가 normal_00보다 원본 분포에 더 가까워야 함")

    results = []
    for cls in args.classes:
        run_fid_for_class(cls, results)

    if not results:
        print("\n[WARNING] 계산된 FID 결과 없음. 이미지를 각 폴더에 추가하세요.")
        return

    annotate_hypothesis(results)
    save_csv(results, output_path)
    print_summary(results)


if __name__ == "__main__":
    main()
