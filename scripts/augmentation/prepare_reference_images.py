"""
reference_images 폴더 구성 스크립트
Console, Cylinder, Wood, Lens 클래스의 defect별 폴더 생성 및
bbox가 그려진 reference 이미지 복사

사용법:
  cd /home/jjh0709/gitrepo/VISION-Instance-Seg/scripts/augmentation
  python prepare_reference_images.py
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw
import sys

# ===== 설정 =====
DATA_ROOT = Path("/home/jjh0709/gitrepo/VISION-Instance-Seg/data")
REF_ROOT = Path("/home/jjh0709/gitrepo/VISION-Instance-Seg/scripts/augmentation/reference_images")

# 각 defect 폴더당 목표 이미지 수
TARGET_IMAGES_PER_DEFECT = 9  # normal_00 제외, ref_01~ref_09

# bbox 스타일
BBOX_COLOR = (0, 100, 255)   # 파란색 (BGR → RGB로 PIL에서)
BBOX_WIDTH = 4               # 픽셀 (원본 크기 기준)

# 최대 출력 이미지 크기 (긴 변 기준 리사이즈)
MAX_LONG_SIDE = 1600

# ===== 클래스 설정 =====
CLASS_CONFIGS = {
    "Console": {
        "data_dir": DATA_ROOT / "Console" / "train",
        "annotation": DATA_ROOT / "Console" / "train" / "_annotations.coco.json",
        "defects": [
            {"id": 0, "name": "Collision"},
            {"id": 1, "name": "Dirty"},
            {"id": 2, "name": "Gap"},
            {"id": 3, "name": "Scratch"},
        ],
    },
    "Cylinder": {
        "data_dir": DATA_ROOT / "Cylinder" / "train",
        "annotation": DATA_ROOT / "Cylinder" / "train" / "_annotations.coco.json",
        "defects": [
            {"id": 0, "name": "Chip"},
            {"id": 1, "name": "PistonMiss"},
            {"id": 2, "name": "Porosity"},
            {"id": 3, "name": "RCS"},
        ],
    },
    "Wood": {
        "data_dir": DATA_ROOT / "Wood" / "train",
        "annotation": DATA_ROOT / "Wood" / "train" / "_annotations.coco.json",
        "defects": [
            {"id": 0, "name": "impurities"},
            {"id": 1, "name": "pits"},
        ],
    },
    "Lens": {
        "data_dir": DATA_ROOT / "Lens" / "train",
        "annotation": DATA_ROOT / "Lens" / "train" / "_annotations.coco.json",
        "defects": [
            {"id": 0, "name": "Fiber"},
            {"id": 1, "name": "FlashParticle"},
            {"id": 2, "name": "Hole"},
            {"id": 3, "name": "SurfaceDamage"},
            {"id": 4, "name": "Tear"},
        ],
    },
}


def load_coco(annotation_path):
    """COCO JSON 로드 및 이미지/어노테이션 맵핑 반환"""
    with open(annotation_path) as f:
        data = json.load(f)

    # image_id → file_name 매핑
    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    # category_id → image_id 목록 (이미지 중복 없이)
    cat_to_images = defaultdict(set)
    # image_id → annotations 목록
    img_to_anns = defaultdict(list)

    for ann in data["annotations"]:
        cat_to_images[ann["category_id"]].add(ann["image_id"])
        img_to_anns[ann["image_id"]].append(ann)

    return id_to_filename, cat_to_images, img_to_anns


def draw_bbox_on_image(img_path, annotations, max_long_side=MAX_LONG_SIDE):
    """이미지에 파란색 bbox를 그려서 반환 (PIL Image)"""
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    # 리사이즈 (긴 변 기준)
    long_side = max(orig_w, orig_h)
    if long_side > max_long_side:
        scale = max_long_side / long_side
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    else:
        scale = 1.0
        new_w, new_h = orig_w, orig_h

    draw = ImageDraw.Draw(img)

    # bbox 두께 (리사이즈 비율에 맞게 조정, 최소 2)
    line_width = max(2, int(BBOX_WIDTH * scale))

    for ann in annotations:
        x, y, w, h = ann["bbox"]
        # 스케일 적용
        x1 = int(x * scale)
        y1 = int(y * scale)
        x2 = int((x + w) * scale)
        y2 = int((y + h) * scale)
        draw.rectangle([x1, y1, x2, y2], outline=BBOX_COLOR, width=line_width)

    return img


def select_images_for_defect(cat_id, cat_to_images, img_to_anns, target_count):
    """
    특정 category의 이미지를 선택.
    - 해당 category의 annotation만 있는 이미지 우선 선택 (깔끔한 reference)
    - 부족하면 여러 category annotation이 있는 이미지도 포함
    - 최소 5장 보장
    """
    image_ids = list(cat_to_images[cat_id])

    # 해당 category만 있는 이미지 vs 여러 category 있는 이미지로 분류
    clean_images = []   # 해당 defect만 있는 이미지
    mixed_images = []   # 다른 defect도 같이 있는 이미지

    for img_id in image_ids:
        anns = img_to_anns[img_id]
        cat_ids_in_img = set(a["category_id"] for a in anns)
        if cat_ids_in_img == {cat_id}:
            clean_images.append(img_id)
        else:
            mixed_images.append(img_id)

    # 선택: clean 우선, 부족하면 mixed 추가
    selected = clean_images[:target_count]
    if len(selected) < target_count:
        needed = target_count - len(selected)
        selected += mixed_images[:needed]

    # 최소 5장 확인
    if len(selected) < 5:
        print(f"    WARNING: Only {len(selected)} images available (< 5)")

    return selected[:target_count]


def process_class(class_name, config):
    """단일 클래스의 모든 defect 폴더 생성"""
    print(f"\n{'='*60}")
    print(f"Processing: {class_name}")
    print(f"{'='*60}")

    # COCO 데이터 로드
    id_to_filename, cat_to_images, img_to_anns = load_coco(config["annotation"])
    data_dir = config["data_dir"]

    # 클래스별 00 폴더 생성 (사용자가 정상 이미지 넣을 폴더)
    class_00_dir = REF_ROOT / f"{class_name}" / f"{class_name}_00"
    class_00_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created: {class_name}/{class_name}_00/ (← 정상 이미지 직접 추가 필요)")

    # 각 defect 폴더 처리
    for defect in config["defects"]:
        cat_id = defect["id"]
        defect_name = defect["name"]
        folder_name = f"{class_name}_{defect_name}"

        out_dir = REF_ROOT / class_name / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{folder_name}]")

        # 이미지 선택
        selected_img_ids = select_images_for_defect(
            cat_id, cat_to_images, img_to_anns, TARGET_IMAGES_PER_DEFECT
        )
        print(f"    Selected {len(selected_img_ids)} images (target: {TARGET_IMAGES_PER_DEFECT})")

        for ref_idx, img_id in enumerate(selected_img_ids, start=1):
            filename = id_to_filename[img_id]
            img_path = data_dir / filename
            stem = Path(filename).stem  # e.g., "000001"

            # 해당 이미지의 특정 category annotation만 필터링
            all_anns = img_to_anns[img_id]
            target_anns = [a for a in all_anns if a["category_id"] == cat_id]

            # bbox 그린 이미지 생성
            annotated_img = draw_bbox_on_image(img_path, target_anns)

            # 저장
            out_filename = f"ref_{ref_idx:02d}_{stem}.jpg"
            out_path = out_dir / out_filename
            annotated_img.save(out_path, "JPEG", quality=95)

            print(f"    Saved: {out_filename} ({len(target_anns)} bbox)")

        print(f"    → {out_dir}")


def main():
    print("=" * 60)
    print("Reference Images 생성 스크립트")
    print("=" * 60)
    print(f"출력 경로: {REF_ROOT}")
    print(f"defect당 목표 이미지 수: {TARGET_IMAGES_PER_DEFECT}")

    for class_name, config in CLASS_CONFIGS.items():
        process_class(class_name, config)

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print("\n각 클래스의 _00 폴더에 정상 이미지(normal_00.jpg)를 추가해주세요:")
    for class_name in CLASS_CONFIGS:
        print(f"  {REF_ROOT / class_name / f'{class_name}_00'}/")


if __name__ == "__main__":
    main()
