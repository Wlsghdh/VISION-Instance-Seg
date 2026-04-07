"""
데이터 파이프라인 — 데이터 병합 + 프레임워크별 등록

사용법:
    # 데이터 병합만 (학습 없이)
    python -m training.data_pipeline --category Cable --experiment exp2 --condition cond1
    python -m training.data_pipeline --category all --experiment exp2 --condition all
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import (
    PROJECT_ROOT, CATEGORIES, EXPERIMENTS, MERGED_DIR,
    get_category_info, get_experiment_info, get_merged_dir,
)

# merge_dataset.py의 핵심 함수 재사용
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'data_utils'))


# ============================================================
# COCO 데이터 유틸리티
# ============================================================
def load_coco(ann_path: Path) -> dict:
    with open(ann_path) as f:
        return json.load(f)


def save_coco(data: dict, ann_path: Path):
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ann_path, 'w') as f:
        json.dump(data, f)


def filter_coco_by_category(data: dict, keep_category_id: int,
                            category_remap: Optional[Dict[int, int]] = None) -> dict:
    """특정 카테고리만 필터링하고 선택적으로 ID 리매핑"""
    filtered_anns = [
        a for a in data["annotations"]
        if a["category_id"] == keep_category_id
    ]

    # 리매핑
    if category_remap:
        for ann in filtered_anns:
            if ann["category_id"] in category_remap:
                ann["category_id"] = category_remap[ann["category_id"]]

    # annotation이 있는 이미지만 유지
    img_ids_with_ann = {a["image_id"] for a in filtered_anns}
    filtered_imgs = [img for img in data["images"] if img["id"] in img_ids_with_ann]

    # 카테고리도 리매핑 반영
    if category_remap:
        new_cats = []
        for cat in data.get("categories", []):
            if cat["id"] == keep_category_id:
                new_cat = dict(cat)
                new_cat["id"] = category_remap.get(cat["id"], cat["id"])
                new_cats.append(new_cat)
    else:
        new_cats = [c for c in data.get("categories", []) if c["id"] == keep_category_id]

    return {
        "images": filtered_imgs,
        "annotations": filtered_anns,
        "categories": new_cats,
    }


# ============================================================
# 통합 카테고리 유틸리티
# ============================================================
def _remap_category_ids(anns: list, local_to_global: Dict[int, int]) -> list:
    """annotation의 category_id를 로컬→글로벌로 변환"""
    for ann in anns:
        ann["category_id"] = local_to_global[ann["category_id"]]
    return anns


def _prefix_filenames(imgs: list, prefix: str):
    """이미지 file_name에 카테고리명 접두사 추가 (충돌 방지). 원본은 _original_filename에 보존"""
    for img in imgs:
        img["_original_filename"] = img["file_name"]
        img["file_name"] = f"{prefix}_{img['file_name']}"


# ============================================================
# 데이터 병합
# ============================================================
def _sample_images(data: dict, images_dir: Path, n: int, seed: int):
    """annotation이 있는 이미지 중 n장 샘플 (n=-1이면 전체)"""
    import random

    img_id_with_ann = {a["image_id"] for a in data["annotations"]}
    valid_images = [
        img for img in data["images"]
        if img["id"] in img_id_with_ann
        and (images_dir / img["file_name"]).exists()
    ]

    if n < 0 or n >= len(valid_images):
        sampled = valid_images
    else:
        rng = random.Random(seed)
        sampled = rng.sample(valid_images, n)

    img_id_set = {img["id"] for img in sampled}
    anns = [a for a in data["annotations"] if a["image_id"] in img_id_set]
    return sampled, anns


def _sample_images_per_class(data: dict, images_dir: Path,
                             n_per_class: int, seed: int):
    """클래스별 균형 샘플링: 각 클래스에서 n_per_class장씩 샘플 (n_per_class=-1이면 전체)"""
    import random
    from collections import defaultdict

    # 이미지별 annotation의 category_id 집합
    img_categories = defaultdict(set)
    for ann in data["annotations"]:
        img_categories[ann["image_id"]].add(ann["category_id"])

    # 유효한 이미지만
    valid_images = {
        img["id"]: img for img in data["images"]
        if img["id"] in img_categories
        and (images_dir / img["file_name"]).exists()
    }

    # 카테고리별 이미지 분류 (주요 카테고리 기준)
    cat_ids = sorted({c["id"] for c in data.get("categories", [])})
    cat_to_images = defaultdict(list)
    for img_id, cats in img_categories.items():
        if img_id not in valid_images:
            continue
        # 이미지의 주요 카테고리 (가장 많은 annotation의 카테고리)
        primary_cat = max(cats, key=lambda c: sum(
            1 for a in data["annotations"]
            if a["image_id"] == img_id and a["category_id"] == c
        ))
        cat_to_images[primary_cat].append(valid_images[img_id])

    # 클래스별 샘플링
    rng = random.Random(seed)
    sampled_ids = set()
    for cat_id in cat_ids:
        images_for_cat = cat_to_images.get(cat_id, [])
        if n_per_class < 0 or n_per_class >= len(images_for_cat):
            selected = images_for_cat
        else:
            selected = rng.sample(images_for_cat, n_per_class)
        for img in selected:
            sampled_ids.add(img["id"])

    sampled = [valid_images[img_id] for img_id in sampled_ids]
    anns = [a for a in data["annotations"] if a["image_id"] in sampled_ids]
    return sampled, anns


def _merge_sources(sources: list, out_dir: Path, categories: list) -> Tuple[int, int]:
    """여러 소스를 하나로 병합 (ID 재부여, 파일 복사)"""
    out_images_dir = out_dir / 'images'
    out_images_dir.mkdir(parents=True, exist_ok=True)

    all_images = []
    all_annotations = []
    next_img_id = 1
    next_ann_id = 1

    for images_dir, imgs, anns in sources:
        images_dir = Path(images_dir)
        img_id_map = {}

        for img in imgs:
            src = images_dir / img.get("_original_filename", img["file_name"])
            dst = out_images_dir / img["file_name"]

            if dst.exists():
                stem = Path(img["file_name"]).stem
                ext = Path(img["file_name"]).suffix
                dst = out_images_dir / f"{stem}_{next_img_id:06d}{ext}"

            shutil.copy2(str(src), str(dst))
            img_id_map[img["id"]] = next_img_id
            all_images.append({
                "id": next_img_id,
                "file_name": dst.name,
                "width": img["width"],
                "height": img["height"],
            })
            next_img_id += 1

        for ann in anns:
            if ann["image_id"] not in img_id_map:
                continue
            all_annotations.append({
                "id": next_ann_id,
                "image_id": img_id_map[ann["image_id"]],
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "segmentation": ann.get("segmentation", []),
                "area": ann.get("area", 0),
                "iscrowd": ann.get("iscrowd", 0),
            })
            next_ann_id += 1

    coco_out = {
        "images": all_images,
        "annotations": all_annotations,
        "categories": categories,
    }
    save_coco(coco_out, out_dir / "annotations.json")
    return len(all_images), len(all_annotations)


def prepare_dataset(experiment: str, condition: str, category: str,
                    seed: int = 42, force: bool = False) -> Path:
    """
    실험 조건에 따라 데이터셋 병합.
    이미 존재하면 스킵 (force=True로 재생성).

    Returns: 병합된 데이터셋 디렉토리 경로
    """
    if category == "Unified":
        return prepare_unified_dataset(experiment, condition, seed, force)

    cat_info = get_category_info(category)
    exp_info = get_experiment_info(experiment)

    if condition not in exp_info["conditions"]:
        raise ValueError(
            f"Unknown condition '{condition}' for {experiment}. "
            f"Choose from {list(exp_info['conditions'].keys())}"
        )

    params = exp_info["conditions"][condition]
    out_dir = get_merged_dir(experiment, condition, category)

    # 이미 존재하면 스킵
    if not force and (out_dir / "annotations.json").exists():
        data = load_coco(out_dir / "annotations.json")
        print(f"  [SKIP] 이미 존재: {out_dir} ({len(data['images'])} images)")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    # 기존 파일 정리
    if (out_dir / "images").exists():
        shutil.rmtree(out_dir / "images")
    if (out_dir / "annotations.json").exists():
        (out_dir / "annotations.json").unlink()

    sources = []
    n_original = params["n_original"]
    n_genai_per_class = params["n_genai_per_class"]
    n_traditional = params["n_traditional"]

    # 원본 데이터
    if cat_info["train_ann"].exists():
        orig_data = load_coco(cat_info["train_ann"])
        # Cable의 경우 thunderbolt(id=1)만 사용, category_id를 0으로 리매핑
        if category == "Cable":
            orig_data = filter_coco_by_category(orig_data, 1, {1: 0})
        orig_imgs, orig_anns = _sample_images(orig_data, cat_info["train_images"], n_original, seed)
        print(f"  원본: {len(orig_imgs)}장")
        sources.append((cat_info["train_images"], orig_imgs, orig_anns))
    else:
        print(f"  [WARN] 원본 없음: {cat_info['train_ann']}")

    # GenAI 데이터 (클래스당 균형 샘플링)
    if n_genai_per_class > 0 and cat_info["genai_ann"].exists():
        genai_data = load_coco(cat_info["genai_ann"])
        genai_imgs, genai_anns = _sample_images_per_class(
            genai_data, cat_info["genai_images"], n_genai_per_class, seed + 1
        )
        print(f"  GenAI: {len(genai_imgs)}장 (클래스당 {n_genai_per_class}장 목표)")
        sources.append((cat_info["genai_images"], genai_imgs, genai_anns))
    elif n_genai_per_class > 0:
        print(f"  [WARN] GenAI 없음: {cat_info['genai_ann']}")

    # 전통 증강 데이터
    if n_traditional > 0 and cat_info["trad_ann"].exists():
        trad_data = load_coco(cat_info["trad_ann"])
        trad_imgs, trad_anns = _sample_images(trad_data, cat_info["trad_images"], n_traditional, seed + 2)
        print(f"  전통증강: {len(trad_imgs)}장")
        sources.append((cat_info["trad_images"], trad_imgs, trad_anns))
    elif n_traditional > 0:
        print(f"  [WARN] 전통증강 없음: {cat_info['trad_ann']}")

    # 병합
    # 학습용 카테고리는 0-indexed로 통일
    train_categories = [
        {"id": i, "name": cls, "supercategory": "defect"}
        for i, cls in enumerate(cat_info["classes"])
    ]
    n_imgs, n_anns = _merge_sources(sources, out_dir, train_categories)
    print(f"  병합 완료: {n_imgs}장, {n_anns}개 annotations → {out_dir}")
    return out_dir


def prepare_unified_dataset(experiment: str, condition: str,
                            seed: int = 42, force: bool = False) -> Path:
    """6개 카테고리를 하나의 14클래스 통합 데이터셋으로 병합"""
    unified_info = get_category_info("Unified")
    exp_info = get_experiment_info(experiment)

    if condition not in exp_info["conditions"]:
        raise ValueError(
            f"Unknown condition '{condition}' for {experiment}. "
            f"Choose from {list(exp_info['conditions'].keys())}"
        )

    params = exp_info["conditions"][condition]
    out_dir = get_merged_dir(experiment, condition, "Unified")

    if not force and (out_dir / "annotations.json").exists():
        data = load_coco(out_dir / "annotations.json")
        print(f"  [SKIP] 이미 존재: {out_dir} ({len(data['images'])} images)")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "images").exists():
        shutil.rmtree(out_dir / "images")
    if (out_dir / "annotations.json").exists():
        (out_dir / "annotations.json").unlink()

    sources = []
    global_id_offset = unified_info["global_id_offset"]
    n_original = params["n_original"]
    n_genai_per_class = params["n_genai_per_class"]
    n_traditional = params["n_traditional"]

    for subcat_name in unified_info["subcategories"]:
        subcat_info = get_category_info(subcat_name)
        offset = global_id_offset[subcat_name]
        local_to_global = {i: offset + i for i in range(subcat_info["num_classes"])}

        print(f"\n  [{subcat_name}] offset={offset}, classes={subcat_info['classes']}")

        # 원본 데이터
        if subcat_info["train_ann"] and subcat_info["train_ann"].exists():
            orig_data = load_coco(subcat_info["train_ann"])
            if subcat_name == "Cable":
                orig_data = filter_coco_by_category(orig_data, 1, {1: 0})
            orig_imgs, orig_anns = _sample_images(
                orig_data, subcat_info["train_images"], n_original, seed
            )
            _remap_category_ids(orig_anns, local_to_global)
            _prefix_filenames(orig_imgs, subcat_name)
            print(f"    원본: {len(orig_imgs)}장")
            sources.append((subcat_info["train_images"], orig_imgs, orig_anns))
        else:
            print(f"    [WARN] 원본 없음: {subcat_info['train_ann']}")

        # GenAI 데이터
        if n_genai_per_class > 0 and subcat_info["genai_ann"] and subcat_info["genai_ann"].exists():
            genai_data = load_coco(subcat_info["genai_ann"])
            genai_imgs, genai_anns = _sample_images_per_class(
                genai_data, subcat_info["genai_images"], n_genai_per_class, seed + 1
            )
            _remap_category_ids(genai_anns, local_to_global)
            _prefix_filenames(genai_imgs, subcat_name)
            print(f"    GenAI: {len(genai_imgs)}장 (클래스당 {n_genai_per_class}장 목표)")
            sources.append((subcat_info["genai_images"], genai_imgs, genai_anns))
        elif n_genai_per_class > 0:
            print(f"    [WARN] GenAI 없음: {subcat_info['genai_ann']}")

        # 전통 증강 데이터
        if n_traditional > 0 and subcat_info["trad_ann"] and subcat_info["trad_ann"].exists():
            trad_data = load_coco(subcat_info["trad_ann"])
            trad_imgs, trad_anns = _sample_images(
                trad_data, subcat_info["trad_images"], n_traditional, seed + 2
            )
            _remap_category_ids(trad_anns, local_to_global)
            _prefix_filenames(trad_imgs, subcat_name)
            print(f"    전통증강: {len(trad_imgs)}장")
            sources.append((subcat_info["trad_images"], trad_imgs, trad_anns))
        elif n_traditional > 0:
            print(f"    [WARN] 전통증강 없음: {subcat_info['trad_ann']}")

    # 통합 병합
    n_imgs, n_anns = _merge_sources(sources, out_dir, unified_info["coco_categories"])
    print(f"\n  통합 병합 완료: {n_imgs}장, {n_anns}개 annotations (14 classes) → {out_dir}")
    return out_dir


def prepare_unified_val_dataset() -> Tuple[Path, Path]:
    """6개 카테고리의 val 데이터를 14클래스 통합 val 데이터셋으로 병합"""
    unified_info = get_category_info("Unified")
    val_dir = MERGED_DIR / "_unified_val"
    val_ann_path = val_dir / "annotations.json"
    val_images_dir = val_dir / "images"

    if val_ann_path.exists():
        data = load_coco(val_ann_path)
        print(f"  [SKIP] 통합 val 이미 존재: {val_dir} ({len(data['images'])} images)")
        return val_images_dir, val_ann_path

    sources = []
    global_id_offset = unified_info["global_id_offset"]

    for subcat_name in unified_info["subcategories"]:
        subcat_info = get_category_info(subcat_name)
        offset = global_id_offset[subcat_name]
        local_to_global = {i: offset + i for i in range(subcat_info["num_classes"])}

        val_ann = subcat_info["val_ann"]
        val_images = subcat_info["val_images"]
        if not val_ann.exists():
            print(f"  [WARN] Val 없음: {subcat_name} ({val_ann})")
            continue

        data = load_coco(val_ann)

        # Cable: thunderbolt(id=1)만 필터, 로컬 0으로 리매핑
        if subcat_info["val_filter_category_id"] is not None:
            data = filter_coco_by_category(
                data,
                subcat_info["val_filter_category_id"],
                subcat_info["val_category_remap"],
            )

        _remap_category_ids(data["annotations"], local_to_global)
        _prefix_filenames(data["images"], subcat_name)
        print(f"  Val [{subcat_name}]: {len(data['images'])}장")
        sources.append((val_images, data["images"], data["annotations"]))

    n_imgs, n_anns = _merge_sources(sources, val_dir, unified_info["coco_categories"])
    print(f"  통합 val 완료: {n_imgs}장, {n_anns}개 annotations → {val_dir}")
    return val_images_dir, val_ann_path


# ============================================================
# Detectron2 데이터 등록
# ============================================================
def register_for_detectron2(name: str, images_dir: Path, ann_path: Path,
                            thing_classes: List[str]):
    """Detectron2 DatasetCatalog/MetadataCatalog에 등록"""
    try:
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.structures import BoxMode
    except ImportError:
        raise ImportError("detectron2가 설치되지 않았습니다")

    def get_dicts():
        data = load_coco(ann_path)
        dataset_dicts = []
        for img in data["images"]:
            record = {
                "file_name": str(images_dir / img["file_name"]),
                "image_id": img["id"],
                "height": img["height"],
                "width": img["width"],
            }
            anns = [a for a in data["annotations"] if a["image_id"] == img["id"]]
            objs = []
            for ann in anns:
                objs.append({
                    "bbox": ann["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": ann["segmentation"],
                    "category_id": ann["category_id"],
                    "iscrowd": ann.get("iscrowd", 0),
                })
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts

    # 이미 등록되어 있으면 제거
    if name in DatasetCatalog:
        DatasetCatalog.remove(name)
    if name in MetadataCatalog:
        MetadataCatalog.remove(name)

    DatasetCatalog.register(name, get_dicts)
    MetadataCatalog.get(name).set(
        thing_classes=thing_classes,
        json_file=str(ann_path),
        image_root=str(images_dir),
        evaluator_type="coco",
    )
    return name


def register_val_for_detectron2(category: str) -> str:
    """Val 데이터셋 등록 (필터링 포함)"""
    cat_info = get_category_info(category)
    val_ann = cat_info["val_ann"]
    val_images = cat_info["val_images"]

    if not val_ann.exists():
        raise FileNotFoundError(f"Val annotations not found: {val_ann}")

    name = f"{category.lower()}_val"

    try:
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.structures import BoxMode
    except ImportError:
        raise ImportError("detectron2가 설치되지 않았습니다")

    filter_cat_id = cat_info["val_filter_category_id"]
    cat_remap = cat_info["val_category_remap"]

    def get_dicts():
        data = load_coco(val_ann)

        # Cable: thunderbolt(id=1)만 필터링
        if filter_cat_id is not None:
            data = filter_coco_by_category(data, filter_cat_id, cat_remap)

        dataset_dicts = []
        ann_by_img = {}
        for a in data["annotations"]:
            ann_by_img.setdefault(a["image_id"], []).append(a)

        for img in data["images"]:
            record = {
                "file_name": str(val_images / img["file_name"]),
                "image_id": img["id"],
                "height": img["height"],
                "width": img["width"],
            }
            objs = []
            for ann in ann_by_img.get(img["id"], []):
                objs.append({
                    "bbox": ann["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": ann["segmentation"],
                    "category_id": ann["category_id"],
                    "iscrowd": ann.get("iscrowd", 0),
                })
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts

    if name in DatasetCatalog:
        DatasetCatalog.remove(name)
    if name in MetadataCatalog:
        MetadataCatalog.remove(name)

    DatasetCatalog.register(name, get_dicts)
    MetadataCatalog.get(name).set(
        thing_classes=cat_info["classes"],
        evaluator_type="coco",
    )
    return name


# ============================================================
# mmdet 데이터 설정
# ============================================================
def get_mmdet_data_config(images_dir: Path, ann_path: Path,
                          classes: List[str], val_images_dir: Path = None,
                          val_ann_path: Path = None) -> dict:
    """mmdet용 데이터 설정 dict 반환"""
    cfg = {
        "train_dataloader": {
            "dataset": {
                "type": "CocoDataset",
                "data_root": str(images_dir.parent),
                "ann_file": str(ann_path),
                "data_prefix": {"img": str(images_dir)},
                "metainfo": {"classes": tuple(classes)},
            },
        },
    }

    if val_images_dir and val_ann_path:
        cfg["val_dataloader"] = {
            "dataset": {
                "type": "CocoDataset",
                "data_root": str(val_images_dir.parent),
                "ann_file": str(val_ann_path),
                "data_prefix": {"img": str(val_images_dir)},
                "metainfo": {"classes": tuple(classes)},
            },
        }

    return cfg


# ============================================================
# Val 데이터 필터링 (파일로 저장)
# ============================================================
def prepare_val_dataset(category: str) -> Tuple[Path, Path]:
    """
    Val 데이터셋 준비 (필요 시 필터링 후 임시 저장).
    Returns: (val_images_dir, val_ann_path)
    """
    if category == "Unified":
        return prepare_unified_val_dataset()

    cat_info = get_category_info(category)
    val_ann = cat_info["val_ann"]
    val_images = cat_info["val_images"]

    if cat_info["val_filter_category_id"] is None:
        return val_images, val_ann

    # 필터링이 필요한 경우 (Cable) → 필터링된 annotation 저장
    filtered_ann_path = val_ann.parent / "_annotations_filtered.json"
    if filtered_ann_path.exists():
        return val_images, filtered_ann_path

    data = load_coco(val_ann)
    filtered = filter_coco_by_category(
        data,
        cat_info["val_filter_category_id"],
        cat_info["val_category_remap"],
    )
    save_coco(filtered, filtered_ann_path)
    print(f"  Val 필터링 완료: {len(filtered['images'])} images, {len(filtered['annotations'])} annotations")
    return val_images, filtered_ann_path


# ============================================================
# CLI
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description='데이터 파이프라인 — 병합 + 등록')
    parser.add_argument('--category', type=str, required=True,
                        choices=list(CATEGORIES.keys()) + ['all'])
    parser.add_argument('--experiment', type=str, required=True,
                        choices=list(EXPERIMENTS.keys()))
    parser.add_argument('--condition', type=str, required=True,
                        help='실험 조건 (또는 "all")')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true', help='기존 데이터 덮어쓰기')

    args = parser.parse_args()

    categories = [c for c in CATEGORIES if c != "Unified"] if args.category == 'all' else [args.category]
    exp_info = get_experiment_info(args.experiment)
    conditions = list(exp_info["conditions"].keys()) if args.condition == 'all' else [args.condition]

    for cat in categories:
        for cond in conditions:
            print(f"\n{'='*60}")
            print(f"  {args.experiment} / {cond} / {cat}")
            print(f"{'='*60}")
            prepare_dataset(args.experiment, cond, cat, args.seed, args.force)

    print(f"\n완료.")


if __name__ == '__main__':
    main()
