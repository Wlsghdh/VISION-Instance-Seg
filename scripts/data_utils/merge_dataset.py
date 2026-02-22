"""
실험용 데이터셋 병합 스크립트 (Experiment 2: 전통 증강 vs 생성AI 비교)

5가지 실험 조건:
  cond1: 원본 25장
  cond2: 원본 25 + 전통 증강 250  (원본에서 전통 증강)
  cond3: 원본 25 + 생성AI 250
  cond4: 원본 25 + 생성AI 250 + 전통 250  (원본에서 전통 증강)
  cond5: 원본 25 + 생성AI 250 + 전통 2,750  ★(원본+gen_ai 합친 275장에서 전통 증강)

사용법:
    # 조건 3: Screw, 원본 25 + gen_ai 250
    python scripts/data_utils/merge_dataset.py \\
        --category Screw --condition cond3

    # 조건 5: Screw (원본 25 + gen_ai 250 → 전통 증강 2750)
    python scripts/data_utils/merge_dataset.py \\
        --category Screw --condition cond5

    # 전체 카테고리 한번에
    python scripts/data_utils/merge_dataset.py \\
        --category all --condition cond5

출력 위치: results/experiment2/{category}/{condition}/
           ├── images/
           └── annotations.json
"""

import argparse
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

BASE = Path('/home/jjh0709/gitrepo/VISION-Instance-Seg')

# traditional_augment 임포트용 경로 추가
sys.path.insert(0, str(BASE / 'scripts' / 'augmentation'))
from traditional_augment import run_augmentation as _run_traditional_aug  # noqa: E402

CATEGORY_CLASSES = {
    "Cable":   [{"id": 1, "name": "thunderbolt", "supercategory": "thunderbolt"}],
    "Screw":   [{"id": 0, "name": "defect",      "supercategory": "defect"}],
    "Casting": [
        {"id": 0, "name": "Inclusoes", "supercategory": "defect"},
        {"id": 1, "name": "Rechupe",   "supercategory": "defect"},
    ],
}


# ============================================================
# 데이터 로드 헬퍼
# ============================================================
def load_coco(ann_path):
    with open(ann_path) as f:
        return json.load(f)


def sample_images(data, images_dir, n, seed):
    """
    annotation이 있는 이미지 중 n장 무작위 샘플.
    n >= 전체 이미지 수면 전체 반환.
    """
    img_id_with_ann = {a["image_id"] for a in data["annotations"]}
    valid_images = [
        img for img in data["images"]
        if img["id"] in img_id_with_ann
        and (Path(images_dir) / img["file_name"]).exists()
    ]

    rng = random.Random(seed)
    if n >= len(valid_images):
        sampled = valid_images
    else:
        sampled = rng.sample(valid_images, n)

    img_id_set = {img["id"] for img in sampled}
    anns = [a for a in data["annotations"] if a["image_id"] in img_id_set]
    return sampled, anns


def merge_sources(sources, out_dir, categories):
    """
    sources: list of (images_dir, images_list, annotations_list)
    병합 후 ID 재부여, out_dir/images/ 에 파일 복사, annotations.json 저장.
    """
    out_images_dir = Path(out_dir) / 'images'
    out_images_dir.mkdir(parents=True, exist_ok=True)

    all_images = []
    all_annotations = []
    next_img_id = 1
    next_ann_id = 1

    for images_dir, imgs, anns in sources:
        images_dir = Path(images_dir)
        img_id_map = {}  # old → new

        for img in imgs:
            src = images_dir / img["file_name"]
            dst = out_images_dir / img["file_name"]

            # 파일명 충돌 시 rename
            if dst.exists():
                stem = Path(img["file_name"]).stem
                ext  = Path(img["file_name"]).suffix
                dst  = out_images_dir / f"{stem}_{next_img_id:06d}{ext}"

            shutil.copy2(str(src), str(dst))

            old_id = img["id"]
            img_id_map[old_id] = next_img_id
            all_images.append({
                "id":        next_img_id,
                "file_name": dst.name,
                "width":     img["width"],
                "height":    img["height"],
            })
            next_img_id += 1

        for ann in anns:
            if ann["image_id"] not in img_id_map:
                continue
            all_annotations.append({
                "id":           next_ann_id,
                "image_id":     img_id_map[ann["image_id"]],
                "category_id":  ann["category_id"],
                "bbox":         ann["bbox"],
                "segmentation": ann.get("segmentation", []),
                "area":         ann.get("area", 0),
                "iscrowd":      ann.get("iscrowd", 0),
            })
            next_ann_id += 1

    coco_out = {
        "images": all_images,
        "annotations": all_annotations,
        "categories": categories,
    }
    with open(Path(out_dir) / "annotations.json", "w") as f:
        json.dump(coco_out, f)

    return len(all_images), len(all_annotations)


def write_coco(images, anns, categories, ann_path):
    """작은 COCO JSON 저장 헬퍼"""
    with open(ann_path, "w") as f:
        json.dump({"images": images, "annotations": anns, "categories": categories}, f)


# ============================================================
# 카테고리별 경로 정의
# ============================================================
def get_paths(category):
    return {
        "original_images": BASE / "data" / category / "train" / "images",
        "original_ann":    BASE / "data" / category / "train" / "annotations.json",
        "genai_images":    BASE / "data_augmented" / category / "gen_ai" / "images",
        "genai_ann":       BASE / "data_augmented" / category / "gen_ai" / "annotations.json",
        "trad_images":     BASE / "data_augmented" / category / "traditional_aug" / "images",
        "trad_ann":        BASE / "data_augmented" / category / "traditional_aug" / "annotations.json",
    }


# ============================================================
# 조건 5 전용: (원본 + gen_ai) → 전통 증강 2750
# ============================================================
def run_cond5_traditional(category, n_original, n_genai, n_traditional, seed, out_dir):
    """
    cond5 전용:
    1. 원본 25 + gen_ai 250 을 임시 디렉토리에 합침
    2. 그 합친 데이터를 소스로 전통 증강 2750장 생성
    3. (원본 25 + gen_ai 250 + 전통 2750) 최종 병합
    """
    paths = get_paths(category)
    cats  = CATEGORY_CLASSES[category]

    # ── step 1: 원본 + gen_ai 수집 ───────────────────────────
    if not paths["original_ann"].exists():
        print(f"  [ERROR] 원본 annotations.json 없음: {paths['original_ann']}")
        return
    if not paths["genai_ann"].exists():
        print(f"  [ERROR] gen_ai annotations.json 없음: {paths['genai_ann']}")
        return

    orig_data  = load_coco(paths["original_ann"])
    genai_data = load_coco(paths["genai_ann"])

    orig_imgs,  orig_anns  = sample_images(orig_data,  paths["original_images"], n_original, seed)
    genai_imgs, genai_anns = sample_images(genai_data, paths["genai_images"],    n_genai,    seed + 1)

    print(f"  원본: {len(orig_imgs)}장 (요청 {n_original})")
    print(f"  gen_ai: {len(genai_imgs)}장 (요청 {n_genai})")

    # ── step 2: 임시 디렉토리에 원본+gen_ai 합침 ─────────────
    tmp_dir = Path(out_dir) / "_tmp_cond5_src"
    tmp_imgs_dir = tmp_dir / "images"
    tmp_imgs_dir.mkdir(parents=True, exist_ok=True)

    combined_images = []
    combined_anns   = []
    next_img_id = 1
    next_ann_id = 1

    for src_imgs_dir, imgs, anns in [
        (paths["original_images"], orig_imgs,  orig_anns),
        (paths["genai_images"],    genai_imgs, genai_anns),
    ]:
        id_map = {}
        for img in imgs:
            src = Path(src_imgs_dir) / img["file_name"]
            dst = tmp_imgs_dir / img["file_name"]
            if dst.exists():
                stem = Path(img["file_name"]).stem
                ext  = Path(img["file_name"]).suffix
                dst  = tmp_imgs_dir / f"{stem}_{next_img_id:06d}{ext}"
            shutil.copy2(str(src), str(dst))
            id_map[img["id"]] = next_img_id
            combined_images.append({
                "id": next_img_id, "file_name": dst.name,
                "width": img["width"], "height": img["height"],
            })
            next_img_id += 1
        for ann in anns:
            if ann["image_id"] not in id_map:
                continue
            combined_anns.append({
                "id": next_ann_id, "image_id": id_map[ann["image_id"]],
                "category_id": ann["category_id"], "bbox": ann["bbox"],
                "segmentation": ann.get("segmentation", []),
                "area": ann.get("area", 0), "iscrowd": ann.get("iscrowd", 0),
            })
            next_ann_id += 1

    tmp_ann_path = tmp_dir / "annotations.json"
    write_coco(combined_images, combined_anns, cats, tmp_ann_path)
    print(f"  임시 소스: {len(combined_images)}장 → {tmp_dir}")

    # ── step 3: 임시 디렉토리에서 전통 증강 실행 ─────────────
    trad_out = Path(out_dir) / "_trad_aug"
    print(f"  전통 증강 시작: {n_traditional}장 생성 중...")
    _run_traditional_aug(
        category     = category,
        n_augment    = n_traditional,
        seed         = seed + 2,
        input_dir    = str(tmp_imgs_dir),
        ann_file     = str(tmp_ann_path),
        output_dir   = str(trad_out),
    )
    trad_ann_path = trad_out / "annotations.json"
    if not trad_ann_path.exists():
        print("  [ERROR] 전통 증강 실패")
        return
    trad_data = load_coco(trad_ann_path)
    print(f"  전통 증강 완료: {len(trad_data['images'])}장")

    # ── step 4: 최종 병합 (원본 + gen_ai + 전통) ─────────────
    sources = [
        (paths["original_images"], orig_imgs,  orig_anns),
        (paths["genai_images"],    genai_imgs, genai_anns),
        (trad_out / "images",      trad_data["images"], trad_data["annotations"]),
    ]
    n_imgs, n_anns = merge_sources(sources, out_dir, cats)
    print(f"\n  ✓ 병합 완료: {n_imgs}장 / {n_anns}개 annotations")
    print(f"  → {out_dir}")

    # ── step 5: 임시 파일 정리 ───────────────────────────────
    shutil.rmtree(str(tmp_dir), ignore_errors=True)
    shutil.rmtree(str(trad_out), ignore_errors=True)


# ============================================================
# 조건별 병합 실행 (cond1~cond4)
# ============================================================
def run_condition(category, condition, n_original, n_genai, n_traditional,
                  seed, output_base):
    paths = get_paths(category)
    cats  = CATEGORY_CLASSES[category]
    out_dir = Path(output_base) / category / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  카테고리: {category}  조건: {condition}")
    print(f"  출력: {out_dir}")
    print(f"{'='*60}")

    # ── cond5: 별도 처리 ──────────────────────────────────────
    if condition == "cond5":
        run_cond5_traditional(
            category=category,
            n_original=n_original,
            n_genai=n_genai,
            n_traditional=n_traditional,
            seed=seed,
            out_dir=out_dir,
        )
        return

    sources = []

    # ─── 원본 (모든 조건 공통) ───────────────────────────────
    if not paths["original_ann"].exists():
        print(f"  [ERROR] 원본 annotations.json 없음: {paths['original_ann']}")
        return
    orig_data = load_coco(paths["original_ann"])
    orig_imgs, orig_anns = sample_images(orig_data, paths["original_images"], n_original, seed)
    print(f"  원본: {len(orig_imgs)}장 (요청 {n_original})")
    sources.append((paths["original_images"], orig_imgs, orig_anns))

    # ─── gen_ai (cond3, cond4) ────────────────────────────────
    if condition in ("cond3", "cond4"):
        if not paths["genai_ann"].exists():
            print(f"  [ERROR] gen_ai annotations.json 없음: {paths['genai_ann']}")
            return
        genai_data = load_coco(paths["genai_ann"])
        genai_imgs, genai_anns = sample_images(genai_data, paths["genai_images"], n_genai, seed + 1)
        print(f"  gen_ai: {len(genai_imgs)}장 (요청 {n_genai})")
        sources.append((paths["genai_images"], genai_imgs, genai_anns))

    # ─── traditional_aug (cond2, cond4): 원본 데이터에서 증강된 것 ───
    if condition in ("cond2", "cond4"):
        if not paths["trad_ann"].exists():
            print(f"  [ERROR] traditional_aug annotations.json 없음: {paths['trad_ann']}")
            print(f"  먼저 traditional_augment.py 실행 필요")
            return
        trad_data = load_coco(paths["trad_ann"])
        trad_imgs, trad_anns = sample_images(trad_data, paths["trad_images"], n_traditional, seed + 2)
        print(f"  traditional_aug: {len(trad_imgs)}장 (요청 {n_traditional})")
        sources.append((paths["trad_images"], trad_imgs, trad_anns))

    # ─── 병합 ────────────────────────────────────────────────
    n_imgs, n_anns = merge_sources(sources, out_dir, cats)
    print(f"\n  ✓ 병합 완료: {n_imgs}장 / {n_anns}개 annotations")
    print(f"  → {out_dir}")


# ============================================================
# 조건별 기본 파라미터
# ============================================================
CONDITION_DEFAULTS = {
    "cond1": dict(n_original=25, n_genai=0,   n_traditional=0),
    "cond2": dict(n_original=25, n_genai=0,   n_traditional=250),
    "cond3": dict(n_original=25, n_genai=250, n_traditional=0),
    "cond4": dict(n_original=25, n_genai=250, n_traditional=250),
    "cond5": dict(n_original=25, n_genai=250, n_traditional=2750),
}


# ============================================================
# Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='실험용 데이터셋 병합 (Experiment 2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
조건 정의:
  cond1: 원본 25
  cond2: 원본 25 + 전통 250 (원본에서 전통 증강)
  cond3: 원본 25 + gen_ai 250
  cond4: 원본 25 + gen_ai 250 + 전통 250 (원본에서 전통 증강)
  cond5: 원본 25 + gen_ai 250 + 전통 2750 ★(원본+gen_ai 합쳐서 전통 증강)

예시:
  python scripts/data_utils/merge_dataset.py --category Screw --condition cond3
  python scripts/data_utils/merge_dataset.py --category all   --condition cond5
  python scripts/data_utils/merge_dataset.py --category Cable --condition cond2 --n_traditional 500
        """
    )
    parser.add_argument('--category', type=str, required=True,
                        choices=['Cable', 'Screw', 'Casting', 'all'],
                        help='카테고리 (all = 전체)')
    parser.add_argument('--condition', type=str, required=True,
                        choices=['cond1', 'cond2', 'cond3', 'cond4', 'cond5'],
                        help='실험 조건')
    parser.add_argument('--n_original', type=int, default=None,
                        help='원본 이미지 수 (기본: 조건별 기본값)')
    parser.add_argument('--n_genai', type=int, default=None,
                        help='gen_ai 이미지 수 (기본: 조건별 기본값)')
    parser.add_argument('--n_traditional', type=int, default=None,
                        help='전통 증강 이미지 수 (기본: 조건별 기본값)')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드 (기본: 42)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='출력 루트 디렉토리 (기본: results/experiment2)')

    args = parser.parse_args()

    defaults = CONDITION_DEFAULTS[args.condition]
    n_original    = args.n_original    if args.n_original    is not None else defaults["n_original"]
    n_genai       = args.n_genai       if args.n_genai       is not None else defaults["n_genai"]
    n_traditional = args.n_traditional if args.n_traditional is not None else defaults["n_traditional"]

    output_base = Path(args.output_dir) if args.output_dir else BASE / "results" / "experiment2"
    categories  = list(CATEGORY_CLASSES.keys()) if args.category == 'all' else [args.category]

    print(f"\n실험 데이터셋 병합 시작")
    print(f"  조건: {args.condition}")
    print(f"  원본: {n_original}장  |  gen_ai: {n_genai}장  |  전통 증강: {n_traditional}장")
    print(f"  시드: {args.seed}")
    print(f"  출력 루트: {output_base}")

    for cat in categories:
        run_condition(
            category=cat,
            condition=args.condition,
            n_original=n_original,
            n_genai=n_genai,
            n_traditional=n_traditional,
            seed=args.seed,
            output_base=output_base,
        )

    print(f"\n{'='*60}")
    print(f"전체 완료.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
