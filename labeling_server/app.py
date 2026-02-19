"""
VISION Dataset AI-Assisted Annotation Tool v8.0
증강 데이터를 annotations_all_merged.json에 추가하는 버전
"""

import json, os, argparse, base64
from datetime import datetime
import cv2
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# ============================================================
# 설정 - 경로를 실제 환경에 맞게 수정
# ============================================================
CONFIG = {
    # 기존 train JSON (여기에 새 데이터 추가됨)
    "annotations_path": "/data2/project/2026winter/jjh0709/AA_CV_R/train/annotations.json",
    
    # 이미지 저장 디렉토리
    "images_dir": "/data2/project/2026winter/jjh0709/AA_CV_R/train/images",
    
    # 44개 카테고리 (annotations_all_merged.json과 동일)
    "categories_by_domain": {
        "Cable": [{"id": 1, "name": "thunderbolt"}]
    }
}

# 전체 카테고리 리스트 생성
ALL_CATEGORIES = []
for domain, cats in CONFIG["categories_by_domain"].items():
    for c in cats:
        c["supercategory"] = domain
        ALL_CATEGORIES.append(c)

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
    """기존 annotations_all_merged.json 로드"""
    if os.path.exists(CONFIG["annotations_path"]):
        with open(CONFIG["annotations_path"], 'r') as f:
            return json.load(f)
    # 파일이 없으면 기본 구조 생성
    return {
        "images": [], 
        "annotations": [], 
        "categories": ALL_CATEGORIES
    }

def save_annotations(data):
    """JSON 저장 (백업 생성)"""
    # 백업 생성
    if os.path.exists(CONFIG["annotations_path"]):
        backup_path = CONFIG["annotations_path"] + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.system(f"cp '{CONFIG['annotations_path']}' '{backup_path}'")
        print(f"📦 백업 생성: {backup_path}")
    
    with open(CONFIG["annotations_path"], 'w') as f:
        json.dump(data, f)
    print(f"💾 저장 완료: {CONFIG['annotations_path']}")

def get_next_ids(data):
    """다음 image_id와 annotation_id 계산"""
    max_img_id = max([i["id"] for i in data["images"]], default=0)
    max_anno_id = max([a["id"] for a in data["annotations"]], default=0)
    return max_img_id + 1, max_anno_id + 1

def get_next_image_number(data, domain):
    """도메인별 다음 이미지 번호 계산 (예: Cable_000047.jpg, PCB_1_000047.jpg)"""
    nums = []
    prefix = f"{domain}_"
    
    for img in data["images"]:
        fname = img["file_name"]
        if fname.startswith(prefix):
            try:
                # Cable_000046.jpg -> 46
                # PCB_1_000046.jpg -> 46
                # 마지막 언더스코어 이후의 숫자 부분 추출
                remainder = fname[len(prefix):]  # "000046.jpg"
                num = int(remainder.split(".")[0])
                nums.append(num)
            except:
                pass
    return max(nums, default=-1) + 1

def decode_base64_image(b64):
    """Base64 -> OpenCV 이미지"""
    if ',' in b64:
        b64 = b64.split(',')[1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)

# ============================================================
# HTML 템플릿 로드
# ============================================================
HTML_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotation_template.html')

def get_html_template():
    if os.path.exists(HTML_FILE):
        with open(HTML_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    return DEFAULT_HTML

DEFAULT_HTML = '''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Error</title></head>
<body style="background:#0d1117;color:#fff;font-family:system-ui;padding:50px;text-align:center">
<h1>⚠️ 템플릿 파일을 찾을 수 없습니다</h1>
<p>annotation_template.html 파일이 필요합니다.</p>
<p>현재 경로: ''' + HTML_FILE + '''</p>
</body></html>'''

# ============================================================
# Flask Routes
# ============================================================
@app.route('/')
def index():
    html = get_html_template()
    html = html.replace('{{CATEGORIES_JSON}}', json.dumps(CONFIG["categories_by_domain"]))
    html = html.replace('{{ALL_CATEGORIES}}', json.dumps(ALL_CATEGORIES))
    return html

@app.route('/info')
def info():
    """서버 상태 정보"""
    data = load_annotations()
    
    # 도메인별 통계
    domain_stats = {}
    for img in data["images"]:
        domain = img["file_name"].split("_")[0]
        domain_stats[domain] = domain_stats.get(domain, 0) + 1
    
    return jsonify({
        "num_images": len(data["images"]),
        "num_annotations": len(data["annotations"]),
        "domain_stats": domain_stats,
        "json_path": CONFIG["annotations_path"]
    })

@app.route('/save', methods=['POST'])
def save():
    """이미지 + 어노테이션 저장"""
    try:
        domain = request.form['domain']
        width = int(request.form['width'])
        height = int(request.form['height'])
        annos = json.loads(request.form['annotations'])
        img_file = request.files['image']
        
        # 기존 데이터 로드
        data = load_annotations()
        
        # 다음 ID 계산
        next_img_id, next_anno_id = get_next_ids(data)
        next_num = get_next_image_number(data, domain)
        
        # 파일명 생성 (예: Cable_000047.jpg)
        new_filename = f"{domain}_{next_num:06d}.jpg"
        img_path = os.path.join(CONFIG["images_dir"], new_filename)
        
        # 이미지 저장
        img_file.save(img_path)
        print(f"🖼️ 이미지 저장: {img_path}")
        
        # images 배열에 추가
        data["images"].append({
            "id": next_img_id,
            "file_name": new_filename,
            "width": width,
            "height": height
        })
        
        # annotations 배열에 추가 (COCO 형식 보장)
        for a in annos:
            # segmentation 형식 검증: [[x1,y1,x2,y2,...]]
            seg = a["segmentation"]
            if isinstance(seg, list) and len(seg) > 0:
                # 모든 좌표를 정수로 변환
                seg = [[int(round(coord)) for coord in poly] for poly in seg]
            
            # bbox 정수 변환 [x, y, w, h]
            bbox = [int(round(b)) for b in a["bbox"]]
            
            data["annotations"].append({
                "id": next_anno_id,
                "image_id": next_img_id,
                "category_id": int(a["category_id"]),
                "bbox": bbox,
                "segmentation": seg,
                "area": int(round(a["area"])),
                "iscrowd": int(a.get("iscrowd", 0))
            })
            next_anno_id += 1
        
        # JSON 저장
        save_annotations(data)
        
        return jsonify({
            "success": True,
            "file_name": new_filename,
            "image_id": next_img_id,
            "total_images": len(data["images"]),
            "total_annotations": len(data["annotations"])
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})

@app.route('/ai/segment', methods=['POST'])
def ai_segment():
    """AI 자동 세그멘테이션"""
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
    """이미지 + 어노테이션 삭제"""
    try:
        req_data = request.json
        file_name = req_data['file_name']
        image_id = req_data['image_id']
        
        # 기존 데이터 로드
        data = load_annotations()
        
        # 이미지 찾기
        img_idx = None
        for i, img in enumerate(data["images"]):
            if img["file_name"] == file_name and img["id"] == image_id:
                img_idx = i
                break
        
        if img_idx is None:
            return jsonify({"success": False, "error": "이미지를 찾을 수 없습니다"})
        
        # 해당 이미지의 annotation 삭제
        data["annotations"] = [a for a in data["annotations"] if a["image_id"] != image_id]
        
        # 이미지 항목 삭제
        del data["images"][img_idx]
        
        # 이미지 파일 삭제
        img_path = os.path.join(CONFIG["images_dir"], file_name)
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"🗑️ 이미지 삭제: {img_path}")
        
        # JSON 저장
        save_annotations(data)
        
        return jsonify({
            "success": True,
            "deleted_file": file_name,
            "total_images": len(data["images"]),
            "total_annotations": len(data["annotations"])
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})

@app.route('/stats')
def stats():
    """상세 통계"""
    data = load_annotations()
    
    # 도메인별 이미지/어노테이션 수
    img_by_domain = {}
    for img in data["images"]:
        domain = img["file_name"].split("_")[0]
        img_by_domain[domain] = img_by_domain.get(domain, 0) + 1
    
    # 카테고리별 어노테이션 수
    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    anno_by_cat = {}
    for a in data["annotations"]:
        cat_name = cat_id_to_name.get(a["category_id"], "Unknown")
        anno_by_cat[cat_name] = anno_by_cat.get(cat_name, 0) + 1
    
    return jsonify({
        "images_by_domain": img_by_domain,
        "annotations_by_category": anno_by_cat,
        "total_images": len(data["images"]),
        "total_annotations": len(data["annotations"])
    })

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VISION AI Annotation Tool v8.0')
    parser.add_argument('--port', type=int, default=5200, help='서버 포트')
    parser.add_argument('--host', default='0.0.0.0', help='호스트')
    parser.add_argument('--json', type=str, help='JSON 파일 경로 (기본: annotations_all_merged.json)')
    parser.add_argument('--images', type=str, help='이미지 디렉토리 경로')
    args = parser.parse_args()
    
    # 커맨드라인 인자로 경로 오버라이드
    if args.json:
        CONFIG["annotations_path"] = args.json
    if args.images:
        CONFIG["images_dir"] = args.images
    
    print("\n" + "="*60)
    print("🤖 VISION AI Annotation Tool v8.0")
    print("="*60)
    print(f"📁 JSON 경로: {CONFIG['annotations_path']}")
    print(f"🖼️ 이미지 디렉토리: {CONFIG['images_dir']}")
    print(f"🌐 URL: http://ahnbi1.suwon.ac.kr:{args.port}")
    print("="*60 + "\n")
    
    # 초기 통계
    data = load_annotations()
    print(f"📊 현재 상태: {len(data['images'])}장 이미지, {len(data['annotations'])}개 어노테이션")
    print()
    
    app.run(host=args.host, port=args.port, debug=False)