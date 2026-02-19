# 라벨링 툴 분석: annotation_tool_v8.py

## 파일 개요

Flask 기반 웹 어노테이션 툴로, COCO format (instance segmentation)으로 이미지와 어노테이션을 저장하는 도구. AI 자동 세그멘테이션(Otsu thresholding fallback) 기능 포함.

**원본 경로 (ahnbi1):**
`/data2/project/2026winter/jjh0709/Resen/before/vision_ai_labeling/annotation_tool_v8.py`

---

## 사용 라이브러리

| 라이브러리 | 용도 |
|-----------|------|
| `flask` | 웹 서버 |
| `cv2` (OpenCV) | 이미지 처리, Otsu thresholding, 컨투어 추출 |
| `numpy` | 배열 연산 |
| `json`, `os` | 파일 I/O |
| `base64` | 이미지 인코딩/디코딩 |

---

## 전체 구조 (클래스/함수 목록과 역할)

### 전역 설정
```python
CONFIG = {
    "annotations_path": "/data2/project/2026winter/jjh0709/AA_CV_R/train/annotations.json",
    "images_dir": "/data2/project/2026winter/jjh0709/AA_CV_R/train/images",
    "categories_by_domain": {
        "Cable": [{"id": 1, "name": "thunderbolt"}]
    }
}
```
- 경로가 **하드코딩**되어 있음
- Cable 카테고리만 정의 (Screw, Casting 없음)

### 클래스

| 클래스 | 역할 |
|--------|------|
| `FallbackSegmentation` | AI 자동 세그멘테이션 (Otsu thresholding 기반) |

### 함수

| 함수 | 역할 |
|------|------|
| `load_annotations()` | annotations.json 로드 (없으면 기본 구조 생성) |
| `save_annotations(data)` | JSON 저장 + 백업 생성 (`*.backup_YYYYMMDD_HHMMSS`) |
| `get_next_ids(data)` | 다음 image_id, annotation_id 계산 |
| `get_next_image_number(data, domain)` | 도메인별 다음 이미지 번호 계산 |
| `decode_base64_image(b64)` | Base64 → OpenCV 이미지 변환 |
| `get_html_template()` | HTML 파일 로드 (`annotation_template.html`) |

### Flask Routes

| Route | 메서드 | 역할 |
|-------|--------|------|
| `/` | GET | 메인 어노테이션 UI |
| `/info` | GET | 서버 상태 (이미지/어노테이션 수, 도메인별 통계) |
| `/save` | POST | 이미지 + 어노테이션 저장 |
| `/ai/segment` | POST | AI 자동 세그멘테이션 |
| `/delete` | POST | 이미지 + 어노테이션 삭제 |
| `/stats` | GET | 상세 통계 (도메인별 이미지/카테고리별 어노테이션 수) |

---

## 핵심 로직 발췌

### 이미지 저장 부분 (`/save` 라우트)

```python
@app.route('/save', methods=['POST'])
def save():
    domain = request.form['domain']       # 웹 UI에서 선택한 도메인 (예: Cable)
    width = int(request.form['width'])
    height = int(request.form['height'])
    annos = json.loads(request.form['annotations'])
    img_file = request.files['image']

    data = load_annotations()
    next_img_id, next_anno_id = get_next_ids(data)
    next_num = get_next_image_number(data, domain)

    # 파일명 생성: Cable_000047.jpg
    new_filename = f"{domain}_{next_num:06d}.jpg"
    img_path = os.path.join(CONFIG["images_dir"], new_filename)

    img_file.save(img_path)   # 이미지 저장
    save_annotations(data)     # JSON 저장
```

**문제점**: 모든 카테고리(Cable, Screw, Casting)가 동일한 폴더(`AA_CV_R/train/images/`)에 저장됨.

### 어노테이션 저장 형식 (COCO format)

```python
# images 배열
data["images"].append({
    "id": next_img_id,
    "file_name": new_filename,
    "width": width,
    "height": height
})

# annotations 배열
data["annotations"].append({
    "id": next_anno_id,
    "image_id": next_img_id,
    "category_id": int(a["category_id"]),
    "bbox": [x, y, w, h],                       # COCO format [x, y, width, height]
    "segmentation": [[x1, y1, x2, y2, ...]],    # polygon 좌표
    "area": int(round(a["area"])),
    "iscrowd": int(a.get("iscrowd", 0))
})
```

**저장 형식**: 표준 COCO Instance Segmentation format.

### AI 세그멘테이션 (`FallbackSegmentation.predict`)

```python
def predict(self, image, threshold=0.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(largest, True)
    simplified = cv2.approxPolyDP(largest, epsilon, True)
    return {'mask': mask, 'polygon': polygon, 'confidence': 0.7}
```

### 웹 UI 구조

- HTML 파일: `annotation_template.html` (별도 파일, 레포에 미포함)
- UI에 도메인 선택 드롭다운 있음 → `categories_by_domain` 기준
- 현재 Cable만 정의되어 있음

---

## 현재 문제점

1. **카테고리 구분 없이 단일 폴더에 저장**: 모든 이미지가 `AA_CV_R/train/images/`에 섞임
2. **경로 하드코딩**: `CONFIG`에 절대 경로가 고정되어 있어 이식성 없음
3. **Cable만 지원**: `categories_by_domain`에 Cable/thunderbolt만 정의됨
4. **HTML 별도 파일**: `annotation_template.html`이 레포에 포함되지 않음
5. **단일 annotations.json**: 카테고리별 분리 없이 하나의 JSON에 모두 저장

---

## 리팩토링 시 수정 필요 포인트

리팩토링 목표 파일: `labeling_server/app.py`

1. **카테고리 선택 드롭다운 추가**: Cable / Screw / Casting
2. **데이터 타입 선택 드롭다운 추가**: gen_ai / traditional_aug
3. **저장 경로 자동 설정**:
   - gen_ai → `data_augmented/{Category}/gen_ai/images/` + `annotations.json`
   - traditional_aug → `data_augmented/{Category}/traditional_aug/images/` + `annotations.json`
4. **카테고리별 annotations.json 분리 관리**
5. **경로 환경변수 or argparse로 처리** (하드코딩 제거)
6. **카테고리별 category_id 매핑 정리** (COCO format 준수)
