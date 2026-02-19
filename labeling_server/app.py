"""
VISION Dataset AI-Assisted Annotation Tool v9.0
- argparse로 category / split 지정 (gen_ai, traditional_aug 등)
- 서버에 있는 기존 이미지 브라우징 및 annotation 로드/저장
- data_augmented/{category}/{split}/annotations.json 에 저장

사용법:
    python labeling_server/app.py --category Cable --split gen_ai
    python labeling_server/app.py --category Cable --split gen_ai --port 5200
"""

import json
import os
import argparse
import base64
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

# ============================================================
# 카테고리별 설정
# ============================================================
CATEGORY_CLASSES = {
    "Cable":   [{"id": 1, "name": "thunderbolt", "supercategory": "thunderbolt"}],
    "Screw":   [{"id": 1, "name": "defect",      "supercategory": "defect"}],
    "Casting": [
        {"id": 1, "name": "Inclusoes", "supercategory": "defect"},
        {"id": 2, "name": "Rechupe",   "supercategory": "defect"},
    ],
}

# 런타임 설정 (argparse에서 채워짐)
CONFIG = {
    "category":        "Cable",
    "split":           "gen_ai",
    "annotations_path": "",
    "images_dir":      "",
}


def get_categories_for_category(category):
    return {category: CATEGORY_CLASSES.get(category, [])}


def get_all_categories(category):
    return CATEGORY_CLASSES.get(category, [])


# ============================================================
# AI Segmentation (Fallback - Otsu thresholding)
# ============================================================
class FallbackSegmentation:
    def predict(self, image, threshold=0.5):
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {'mask': np.zeros((h, w), dtype=np.uint8), 'polygon': [], 'confidence': 0.0}
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        epsilon = 0.005 * cv2.arcLength(largest, True)
        simplified = cv2.approxPolyDP(largest, epsilon, True)
        polygon = simplified.flatten().tolist()
        return {'mask': mask, 'polygon': polygon, 'confidence': 0.7}


ai_model = FallbackSegmentation()


# ============================================================
# JSON 로드/저장
# ============================================================
def load_annotations():
    ann_path = CONFIG["annotations_path"]
    if os.path.exists(ann_path):
        with open(ann_path, 'r') as f:
            return json.load(f)
    cats = get_all_categories(CONFIG["category"])
    return {"images": [], "annotations": [], "categories": cats}


def save_annotations(data):
    ann_path = CONFIG["annotations_path"]
    # 백업
    if os.path.exists(ann_path):
        backup = ann_path + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(ann_path, 'r') as f:
            orig = f.read()
        with open(backup, 'w') as f:
            f.write(orig)
    with open(ann_path, 'w') as f:
        json.dump(data, f)


def get_next_ids(data):
    max_img_id = max((i["id"] for i in data["images"]), default=0)
    max_ann_id = max((a["id"] for a in data["annotations"]), default=0)
    return max_img_id + 1, max_ann_id + 1


def decode_base64_image(b64):
    if ',' in b64:
        b64 = b64.split(',')[1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)


# ============================================================
# Flask Routes
# ============================================================
@app.route('/')
def index():
    categories_by_domain = get_categories_for_category(CONFIG["category"])
    all_categories = get_all_categories(CONFIG["category"])
    server_config = {
        "category": CONFIG["category"],
        "split": CONFIG["split"],
        "images_dir": CONFIG["images_dir"],
        "annotations_path": CONFIG["annotations_path"],
    }
    return render_template(
        'annotation_template.html',
        CATEGORIES_JSON=json.dumps(categories_by_domain),
        ALL_CATEGORIES=json.dumps(all_categories),
        SERVER_CONFIG=json.dumps(server_config),
    )


@app.route('/info')
def info():
    data = load_annotations()
    domain_stats = {}
    for img in data["images"]:
        domain = img["file_name"].split("_")[0]
        domain_stats[domain] = domain_stats.get(domain, 0) + 1
    return jsonify({
        "num_images": len(data["images"]),
        "num_annotations": len(data["annotations"]),
        "domain_stats": domain_stats,
        "json_path": CONFIG["annotations_path"],
        "category": CONFIG["category"],
        "split": CONFIG["split"],
    })


@app.route('/images/list')
def images_list():
    """서버의 이미지 디렉토리 파일 목록 반환"""
    images_dir = Path(CONFIG["images_dir"])
    if not images_dir.exists():
        return jsonify({"files": [], "error": f"디렉토리 없음: {images_dir}"})

    supported = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    files = sorted(
        f.name for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in supported
    )

    # 이미 annotation 있는 파일 표시
    data = load_annotations()
    annotated = {img["file_name"] for img in data["images"]}

    result = [
        {"name": f, "annotated": f in annotated}
        for f in files
    ]
    return jsonify({"files": result, "total": len(files), "annotated": len(annotated)})


@app.route('/images/serve/<path:filename>')
def serve_image(filename):
    """이미지 파일 서빙"""
    images_dir = CONFIG["images_dir"]
    return send_from_directory(images_dir, filename)


@app.route('/annotations/for/<path:filename>')
def get_annotations_for(filename):
    """특정 이미지의 기존 annotations 반환"""
    data = load_annotations()
    img_entry = next((i for i in data["images"] if i["file_name"] == filename), None)
    if img_entry is None:
        return jsonify({"found": False, "image": None, "annotations": []})
    anns = [a for a in data["annotations"] if a["image_id"] == img_entry["id"]]
    return jsonify({"found": True, "image": img_entry, "annotations": anns})


@app.route('/save', methods=['POST'])
def save():
    """새 이미지 업로드 + annotation 저장"""
    try:
        domain = request.form['domain']
        width = int(request.form['width'])
        height = int(request.form['height'])
        annos = json.loads(request.form['annotations'])
        img_file = request.files['image']

        data = load_annotations()
        next_img_id, next_anno_id = get_next_ids(data)

        # 파일명 생성
        existing_names = {i["file_name"] for i in data["images"]}
        prefix = f"{domain}_"
        nums = []
        for fn in existing_names:
            if fn.startswith(prefix):
                try:
                    nums.append(int(fn[len(prefix):].split('.')[0]))
                except Exception:
                    pass
        next_num = max(nums, default=-1) + 1
        new_filename = f"{domain}_{next_num:06d}.jpg"
        img_path = os.path.join(CONFIG["images_dir"], new_filename)
        img_file.save(img_path)

        data["images"].append({
            "id": next_img_id,
            "file_name": new_filename,
            "width": width,
            "height": height,
        })

        for a in annos:
            seg = a["segmentation"]
            if isinstance(seg, list) and seg:
                seg = [[int(round(c)) for c in poly] for poly in seg]
            data["annotations"].append({
                "id": next_anno_id,
                "image_id": next_img_id,
                "category_id": int(a["category_id"]),
                "bbox": [int(round(b)) for b in a["bbox"]],
                "segmentation": seg,
                "area": int(round(a["area"])),
                "iscrowd": int(a.get("iscrowd", 0)),
            })
            next_anno_id += 1

        save_annotations(data)
        return jsonify({
            "success": True,
            "file_name": new_filename,
            "image_id": next_img_id,
            "total_images": len(data["images"]),
            "total_annotations": len(data["annotations"]),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route('/save/existing', methods=['POST'])
def save_existing():
    """서버에 이미 있는 이미지의 annotation 저장/갱신 (이미지 업로드 없음)"""
    try:
        req = request.get_json()
        filename = req['file_name']
        width = int(req['width'])
        height = int(req['height'])
        annos = req['annotations']

        data = load_annotations()

        # 이미 존재하는 이미지 엔트리 찾기
        existing_img = next((i for i in data["images"] if i["file_name"] == filename), None)

        if existing_img:
            img_id = existing_img["id"]
            # 기존 annotations 삭제
            data["annotations"] = [a for a in data["annotations"] if a["image_id"] != img_id]
            _, next_anno_id = get_next_ids(data)
            next_anno_id = max((a["id"] for a in data["annotations"]), default=0) + 1
        else:
            # 새로 추가 (이미지는 이미 디렉토리에 있음)
            next_img_id, next_anno_id = get_next_ids(data)
            img_id = next_img_id
            data["images"].append({
                "id": img_id,
                "file_name": filename,
                "width": width,
                "height": height,
            })

        for a in annos:
            seg = a.get("segmentation", [])
            if isinstance(seg, list) and seg:
                seg = [[int(round(c)) for c in poly] for poly in seg]
            data["annotations"].append({
                "id": next_anno_id,
                "image_id": img_id,
                "category_id": int(a["category_id"]),
                "bbox": [int(round(b)) for b in a["bbox"]],
                "segmentation": seg,
                "area": int(round(a.get("area", 0))),
                "iscrowd": int(a.get("iscrowd", 0)),
            })
            next_anno_id += 1

        save_annotations(data)
        return jsonify({
            "success": True,
            "file_name": filename,
            "image_id": img_id,
            "anno_count": len(annos),
            "total_images": len(data["images"]),
            "total_annotations": len(data["annotations"]),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route('/ai/segment', methods=['POST'])
def ai_segment():
    try:
        req_data = request.json
        image = decode_base64_image(req_data['image'])
        result = ai_model.predict(image, req_data.get('threshold', 0.5))
        _, buffer = cv2.imencode('.png', result['mask'])
        return jsonify({
            'polygon': result['polygon'],
            'confidence': result['confidence'],
            'mask_base64': base64.b64encode(buffer).decode('utf-8')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/delete', methods=['POST'])
def delete():
    try:
        req_data = request.json
        file_name = req_data['file_name']
        image_id = req_data['image_id']

        data = load_annotations()

        img_idx = next(
            (i for i, img in enumerate(data["images"])
             if img["file_name"] == file_name and img["id"] == image_id),
            None
        )
        if img_idx is None:
            return jsonify({"success": False, "error": "이미지를 찾을 수 없습니다"})

        data["annotations"] = [a for a in data["annotations"] if a["image_id"] != image_id]
        del data["images"][img_idx]

        # 파일 삭제 (gen_ai 이미지는 삭제하지 않음 - 원본 보존)
        img_path = os.path.join(CONFIG["images_dir"], file_name)
        if os.path.exists(img_path) and CONFIG["split"] not in ("gen_ai",):
            os.remove(img_path)

        save_annotations(data)
        return jsonify({
            "success": True,
            "deleted_file": file_name,
            "total_images": len(data["images"]),
            "total_annotations": len(data["annotations"]),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route('/stats')
def stats():
    data = load_annotations()
    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    anno_by_cat = {}
    for a in data["annotations"]:
        cat_name = cat_id_to_name.get(a["category_id"], "Unknown")
        anno_by_cat[cat_name] = anno_by_cat.get(cat_name, 0) + 1
    return jsonify({
        "total_images": len(data["images"]),
        "total_annotations": len(data["annotations"]),
        "annotations_by_category": anno_by_cat,
        "category": CONFIG["category"],
        "split": CONFIG["split"],
    })


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    BASE_DIR = Path(__file__).parent.parent  # VISION-Instance-Seg/

    parser = argparse.ArgumentParser(
        description='VISION AI Annotation Tool v9.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # gen_ai 이미지 라벨링 (Cable)
  python labeling_server/app.py --category Cable --split gen_ai

  # 커스텀 포트
  python labeling_server/app.py --category Screw --split gen_ai --port 5201
        """
    )
    parser.add_argument('--category', type=str, default='Cable',
                        choices=['Cable', 'Screw', 'Casting'],
                        help='대상 카테고리 (기본: Cable)')
    parser.add_argument('--split', type=str, default='gen_ai',
                        help='데이터 split (gen_ai / traditional_aug / 커스텀, 기본: gen_ai)')
    parser.add_argument('--port', type=int, default=5200, help='서버 포트 (기본: 5200)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='호스트 (기본: 0.0.0.0)')
    args = parser.parse_args()

    # 경로 설정
    data_aug_dir = BASE_DIR / 'data_augmented' / args.category / args.split
    CONFIG["category"] = args.category
    CONFIG["split"] = args.split
    CONFIG["annotations_path"] = str(data_aug_dir / 'annotations.json')
    CONFIG["images_dir"] = str(data_aug_dir / 'images')

    # 디렉토리 생성
    (data_aug_dir / 'images').mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("VISION AI Annotation Tool v9.0")
    print("=" * 60)
    print(f"  카테고리  : {args.category}")
    print(f"  Split     : {args.split}")
    print(f"  이미지    : {CONFIG['images_dir']}")
    print(f"  JSON      : {CONFIG['annotations_path']}")
    print(f"  URL       : http://localhost:{args.port}")
    print("=" * 60)

    data = load_annotations()
    print(f"  현재 상태: {len(data['images'])}장 이미지, {len(data['annotations'])}개 annotation")
    print()

    app.run(host=args.host, port=args.port, debug=False)
