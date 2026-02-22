"""
VISION Dataset AI-Assisted Annotation Tool v10.0
- --category all (기본): Cable + Screw + Casting 전체 동시 지원
- 카테고리별 data_augmented/{category}/{split}/ 에 자동 저장

사용법:
    # 전체 카테고리 (기본)
    python labeling_server/app.py --port 5198

    # 특정 카테고리만
    python labeling_server/app.py --category Cable   --port 5200
    python labeling_server/app.py --category Screw   --port 5201
    python labeling_server/app.py --category Casting --port 5202
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
    "Screw":   [{"id": 0, "name": "defect",      "supercategory": "defect"}],
    "Casting": [
        {"id": 0, "name": "Inclusoes", "supercategory": "defect"},
        {"id": 1, "name": "Rechupe",   "supercategory": "defect"},
    ],
}

# 런타임 설정 (argparse에서 채워짐)
CONFIG = {
    "category": None,   # None = 전체 카테고리
    "split":    "gen_ai",
    "base_dir": "",
}


def get_data_dir(category):
    """카테고리별 데이터 디렉토리"""
    return Path(CONFIG["base_dir"]) / "data_augmented" / category / CONFIG["split"]


def get_active_categories():
    """현재 활성 카테고리 목록"""
    if CONFIG["category"]:
        return [CONFIG["category"]]
    return list(CATEGORY_CLASSES.keys())


def get_categories_json():
    """템플릿에 전달할 categories dict"""
    return {c: CATEGORY_CLASSES[c] for c in get_active_categories()}


def guess_category_from_filename(filename):
    """파일명 앞부분으로 카테고리 추측 (Cable_000.jpg → Cable)"""
    lower = filename.lower()
    for cat in CATEGORY_CLASSES:
        if lower.startswith(cat.lower()):
            return cat
    return None


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
# JSON 로드/저장 (카테고리별)
# ============================================================
def load_annotations_for(category):
    ann_path = get_data_dir(category) / "annotations.json"
    if ann_path.exists():
        with open(ann_path, 'r') as f:
            return json.load(f)
    cats = CATEGORY_CLASSES.get(category, [])
    return {"images": [], "annotations": [], "categories": cats}


def save_annotations_for(data, category):
    ann_path = get_data_dir(category) / "annotations.json"
    if ann_path.exists():
        backup = str(ann_path) + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    cats_json = get_categories_json()
    all_cats = []
    for cat_list in cats_json.values():
        all_cats.extend(cat_list)
    server_config = {
        "category": CONFIG["category"] or "all",
        "split": CONFIG["split"],
    }
    return render_template(
        'annotation_template.html',
        CATEGORIES_JSON=json.dumps(cats_json),
        ALL_CATEGORIES=json.dumps(all_cats),
        SERVER_CONFIG=json.dumps(server_config),
    )


@app.route('/info')
def info():
    total_images = 0
    total_annotations = 0
    for cat in get_active_categories():
        data = load_annotations_for(cat)
        total_images += len(data["images"])
        total_annotations += len(data["annotations"])
    return jsonify({
        "num_images": total_images,
        "num_annotations": total_annotations,
        "category": CONFIG["category"] or "all",
        "split": CONFIG["split"],
    })


@app.route('/images/list')
def images_list():
    """모든 활성 카테고리의 이미지 목록 반환 (category 필드 포함)"""
    supported = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    result = []
    for cat in get_active_categories():
        images_dir = get_data_dir(cat) / 'images'
        if not images_dir.exists():
            continue
        data = load_annotations_for(cat)
        annotated = {img["file_name"] for img in data["images"]}
        files = sorted(
            f.name for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in supported
        )
        for f in files:
            result.append({"name": f, "category": cat, "annotated": f in annotated})
    total = len(result)
    annotated_count = sum(1 for f in result if f["annotated"])
    return jsonify({"files": result, "total": total, "annotated": annotated_count})


@app.route('/images/serve/<path:filename>')
def serve_image(filename):
    """모든 카테고리 이미지 디렉토리에서 파일 탐색 후 서빙"""
    for cat in get_active_categories():
        images_dir = get_data_dir(cat) / 'images'
        img_path = images_dir / filename
        if img_path.exists():
            return send_from_directory(str(images_dir), filename)
    return "Not found", 404


@app.route('/annotations/for/<path:filename>')
def get_annotations_for(filename):
    """모든 카테고리에서 해당 파일의 annotation 탐색"""
    for cat in get_active_categories():
        data = load_annotations_for(cat)
        img_entry = next((i for i in data["images"] if i["file_name"] == filename), None)
        if img_entry:
            anns = [a for a in data["annotations"] if a["image_id"] == img_entry["id"]]
            return jsonify({"found": True, "image": img_entry, "annotations": anns, "category": cat})
    return jsonify({"found": False, "image": None, "annotations": [], "category": None})


@app.route('/save', methods=['POST'])
def save():
    """새 이미지 업로드 + annotation 저장 (domain = category)"""
    try:
        domain = request.form['domain']   # Cable / Screw / Casting
        width = int(request.form['width'])
        height = int(request.form['height'])
        annos = json.loads(request.form['annotations'])
        img_file = request.files['image']

        images_dir = get_data_dir(domain) / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)

        data = load_annotations_for(domain)
        next_img_id, next_anno_id = get_next_ids(data)

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
        img_path = images_dir / new_filename
        img_file.save(str(img_path))

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

        save_annotations_for(data, domain)
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
    """서버에 이미 있는 이미지의 annotation 저장/갱신"""
    try:
        req = request.get_json()
        filename = req['file_name']
        width = int(req['width'])
        height = int(req['height'])
        annos = req['annotations']

        # category: 요청에서 받거나 파일명에서 추측
        category = req.get('category') or guess_category_from_filename(filename)
        if not category:
            category = get_active_categories()[0]

        data = load_annotations_for(category)
        existing_img = next((i for i in data["images"] if i["file_name"] == filename), None)

        if existing_img:
            img_id = existing_img["id"]
            data["annotations"] = [a for a in data["annotations"] if a["image_id"] != img_id]
            next_anno_id = max((a["id"] for a in data["annotations"]), default=0) + 1
        else:
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

        save_annotations_for(data, category)
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
        category = req_data.get('category') or guess_category_from_filename(file_name)

        categories_to_search = [category] if category else get_active_categories()

        for cat in categories_to_search:
            data = load_annotations_for(cat)
            img_idx = next(
                (i for i, img in enumerate(data["images"])
                 if img["file_name"] == file_name and img["id"] == image_id),
                None
            )
            if img_idx is not None:
                data["annotations"] = [a for a in data["annotations"] if a["image_id"] != image_id]
                del data["images"][img_idx]

                img_path = get_data_dir(cat) / 'images' / file_name
                if img_path.exists():
                    img_path.unlink()

                save_annotations_for(data, cat)
                return jsonify({
                    "success": True,
                    "deleted_file": file_name,
                    "total_images": len(data["images"]),
                    "total_annotations": len(data["annotations"]),
                })

        return jsonify({"success": False, "error": "이미지를 찾을 수 없습니다"})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route('/stats')
def stats():
    result = {}
    for cat in get_active_categories():
        data = load_annotations_for(cat)
        cat_id_to_name = {c["id"]: c["name"] for c in CATEGORY_CLASSES.get(cat, [])}
        anno_by_class = {}
        for a in data["annotations"]:
            name = cat_id_to_name.get(a["category_id"], "Unknown")
            anno_by_class[name] = anno_by_class.get(name, 0) + 1
        result[cat] = {
            "total_images": len(data["images"]),
            "total_annotations": len(data["annotations"]),
            "annotations_by_class": anno_by_class,
        }
    return jsonify({
        "category": CONFIG["category"] or "all",
        "split": CONFIG["split"],
        "stats": result,
    })


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    BASE_DIR_PATH = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description='VISION AI Annotation Tool v10.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 카테고리 (Cable + Screw + Casting)
  python labeling_server/app.py --port 5198

  # 특정 카테고리만
  python labeling_server/app.py --category Cable   --port 5200
  python labeling_server/app.py --category Screw   --port 5201
  python labeling_server/app.py --category Casting --port 5202
        """
    )
    parser.add_argument('--category', type=str, default='all',
                        choices=['all', 'Cable', 'Screw', 'Casting'],
                        help='대상 카테고리 (기본: all - 전체)')
    parser.add_argument('--split', type=str, default='gen_ai',
                        help='데이터 split (gen_ai / traditional_aug, 기본: gen_ai)')
    parser.add_argument('--port', type=int, default=5198, help='서버 포트 (기본: 5198)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='호스트')
    args = parser.parse_args()

    CONFIG["category"] = None if args.category == 'all' else args.category
    CONFIG["split"] = args.split
    CONFIG["base_dir"] = str(BASE_DIR_PATH)

    # 디렉토리 생성
    for cat in get_active_categories():
        (BASE_DIR_PATH / 'data_augmented' / cat / args.split / 'images').mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("VISION AI Annotation Tool v10.0")
    print("=" * 60)
    print(f"  카테고리  : {args.category}")
    print(f"  Split     : {args.split}")
    print(f"  URL       : http://localhost:{args.port}")
    print("=" * 60)
    for cat in get_active_categories():
        data = load_annotations_for(cat)
        images_dir = BASE_DIR_PATH / 'data_augmented' / cat / args.split / 'images'
        img_count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
        classes = [c["name"] for c in CATEGORY_CLASSES[cat]]
        print(f"  [{cat}] 클래스: {classes} | 이미지: {img_count}장 | annotation: {len(data['annotations'])}개")
        print(f"    → data_augmented/{cat}/{args.split}/annotations.json")
    print()

    app.run(host=args.host, port=args.port, debug=False)
