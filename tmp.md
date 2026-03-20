# 생성형 AI 증강 이미지 만드는 방법 가이드

> 작성일: 2026-03-12
> 대상: 팀원 B (gemini_augment.py 담당)
> 작업 서버: lifeai (`/home/jjh0709/gitrepo/VISION-Instance-Seg`)

---

## 1. 전체 흐름 요약

```
① 레퍼런스 이미지 준비 (정상 1장 + 결함 예시 N장)
② 프롬프트 작성/수정 (DEFECT_CONFIGS 딕셔너리)
③ Gemini API로 결함 이미지 생성 (gemini_augment.py 실행)
④ 생성된 이미지를 data_augmented/ 로 이동
⑤ 라벨링 서버로 어노테이션 검수/작성
```

---

## 2. 파일 위치

| 파일 | 경로 | 역할 |
|------|------|------|
| **생성 스크립트** | `scripts/augmentation/gemini_augment.py` | Gemini API 호출 + 이미지 생성 |
| **레퍼런스 이미지 준비** | `scripts/augmentation/prepare_reference_images.py` | COCO annotation에서 bbox 그린 레퍼런스 자동 생성 |
| **레퍼런스 이미지 폴더** | `scripts/augmentation/reference_images/{카테고리}/` | 정상 + 결함 예시 이미지 |
| **생성 결과** | `scripts/augmentation/vision_ai_generated/{결함유형}/` | 생성된 이미지 저장 |
| **진행상황 파일** | `scripts/augmentation/progress_{결함유형}.json` | 중단/재시작 지원 |
| **프롬프트 텍스트 (미사용)** | `scripts/augmentation/prompts/*.txt` | 현재 비어있음, 프롬프트는 코드 내 하드코딩 |

---

## 3. 프롬프트 구조 이해 (핵심!)

### 3-1. 프롬프트가 어디에 있는가

프롬프트는 `gemini_augment.py` 파일 안의 **`DEFECT_CONFIGS` 딕셔너리**에 하드코딩되어 있다.
`scripts/augmentation/prompts/*.txt` 파일들은 현재 비어있고 실제로 사용되지 않는다.

### 3-2. 프롬프트 4단계 구조

각 결함 유형의 프롬프트는 4개 파트를 이어붙여 구성된다:

```python
prompt = prompt_base + prompt_key_instruction + prompt_variations[i % 10] + prompt_style
```

| 파트 | 역할 | 예시 |
|------|------|------|
| `prompt_base` | 결함 설명 + 레퍼런스 이미지 역할 안내 | "Generate a new image of a metal casting part with an inclusion defect..." |
| `prompt_key_instruction` | 생성 규칙 (필수 포함 결함, 크기 제약, 파란색 마킹 금지 등) | "MANDATORY: Output MUST contain exactly one clearly visible inclusion defect..." |
| `prompt_variations[i%10]` | 결함 위치 지정 (10가지 순환) | "Place the inclusion defect in the upper-left quadrant..." |
| `prompt_style` | 사진 스타일 + 일관성 유지 지시 | "Industrial inspection photography, even lighting, sharp focus..." |

### 3-3. 현재 등록된 결함 유형

**기존 방식 (bbox 레퍼런스)**:

| 결함 유형 키 | 카테고리 | 생성 목표 | 설명 |
|-------------|---------|----------|------|
| `casting_Inclusoes` | Casting | 150장 | 비금속 이물질 결함 |
| `casting_Rechupe` | Casting | 150장 | 수축/공동 결함 |
| `screw_defect` | Screw | 300장 | 나사 제조 결함 |

**새 방식 (bbox 없는 원본 이미지 방식)**:

| 결함 유형 키 | 카테고리 | 생성 목표 | 설명 |
|-------------|---------|----------|------|
| `Console_Collision` | Console | 75장 | 미세 충격 흠집 |
| `Console_Dirty` | Console | 75장 | 오염/지문 |
| `Console_Gap` | Console | 75장 | 부품 간 간격 |
| `Console_Scratch` | Console | 75장 | 긁힘 |
| `Cylinder_Chip` | Cylinder | 75장 | 하단 테두리 칩 |
| `Cylinder_PistonMiss` | Cylinder | 75장 | 단 경계 소실 |
| `Cylinder_Porosity` | Cylinder | 75장 | 기공/벗겨짐 |
| `Cylinder_RCS` | Cylinder | 75장 | 평행 긁힘 다수 |
| `Wood_impurities` | Wood | 150장 | 흰색 이물질 |
| `Wood_pits` | Wood | 150장 | 긁힘/패임 |

---

## 4. 실행 방법

### 4-1. 환경 준비

```bash
conda activate jjh
cd /home/jjh0709/gitrepo/VISION-Instance-Seg/scripts/augmentation

# Gemini API 키 설정 (필수!)
export GEMINI_API_KEY='your_api_key_here'
```

### 4-2. 실행 명령어

```bash
# 개별 결함 유형 실행
python gemini_augment.py casting_Inclusoes
python gemini_augment.py screw_defect

# 카테고리 그룹 실행 (하위 결함 유형 전부 순차 실행)
python gemini_augment.py Console    # → Collision, Dirty, Gap, Scratch 순차
python gemini_augment.py Cylinder   # → Chip, PistonMiss, Porosity, RCS 순차
python gemini_augment.py Wood       # → impurities, pits 순차

# 전체 실행
python gemini_augment.py all

# 테스트 (5장만 생성)
python gemini_augment.py screw_defect --count 5
```

### 4-3. 백그라운드 실행 (추천)

```bash
nohup python -u gemini_augment.py Console  > Console.log  2>&1 &
nohup python -u gemini_augment.py Cylinder > Cylinder.log 2>&1 &
nohup python -u gemini_augment.py Wood     > Wood.log     2>&1 &

# 진행 확인
tail -f Console.log
```

### 4-4. 중단 후 재시작

스크립트가 중간에 끊겨도 `progress_{결함유형}.json`에 진행상황이 저장되어 있어서, **같은 명령어로 다시 실행하면 이어서 생성**한다.

```bash
# 진행상황 확인
cat progress_screw_defect.json
# → "completed": [0,1,2,...49], "last_successful_index": 49
# 다시 실행하면 50번부터 이어서 생성
```

진행상황을 리셋하려면 해당 progress 파일을 삭제하면 된다.

---

## 5. 새로운 결함 유형 추가하는 방법

### 5-1. 레퍼런스 이미지 준비

레퍼런스 이미지는 Gemini에게 "이런 결함을 만들어줘"라고 보여주는 예시 이미지다.

**폴더 구조**:
```
reference_images/
└── {카테고리}/
    └── {카테고리}_{결함명}/
        ├── normal_00.png     ← 정상 이미지 1장 (필수)
        ├── ref_01_XXX.jpg    ← 결함 예시 이미지 1
        ├── ref_02_XXX.jpg    ← 결함 예시 이미지 2
        └── ...               ← 최대 9장 권장
```

**두 가지 방식**:

1. **bbox 방식 (기존: Casting, Screw, Cable)**: 결함 부위에 파란색 박스를 그린 이미지를 레퍼런스로 사용. `prepare_reference_images.py`로 자동 생성 가능.

2. **원본 이미지 방식 (새: Console, Cylinder, Wood)**: bbox 없이 원본 결함 이미지를 그대로 레퍼런스로 사용. `DEFECT_CONFIGS`에 `data_ref` 필드를 추가하면 COCO annotation에서 해당 카테고리 이미지를 자동으로 로드.

### 5-2. `prepare_reference_images.py`로 레퍼런스 자동 생성

이미 COCO annotation이 있는 카테고리의 경우:

```bash
cd /home/jjh0709/gitrepo/VISION-Instance-Seg/scripts/augmentation
python prepare_reference_images.py
```

이 스크립트는 `CLASS_CONFIGS` 딕셔너리에 정의된 카테고리의 COCO annotation을 읽어서, bbox를 파란색으로 그린 레퍼런스 이미지를 `reference_images/`에 자동 생성한다.

**새 카테고리를 추가하려면** `prepare_reference_images.py`의 `CLASS_CONFIGS`에 추가:

```python
CLASS_CONFIGS = {
    # ... 기존 ...
    "NewCategory": {
        "data_dir": DATA_ROOT / "NewCategory" / "train",
        "annotation": DATA_ROOT / "NewCategory" / "train" / "_annotations.coco.json",
        "defects": [
            {"id": 0, "name": "DefectA"},
            {"id": 1, "name": "DefectB"},
        ],
    },
}
```

생성 후 각 폴더에 **정상 이미지(`normal_00.jpg`)**를 직접 추가해야 한다.

### 5-3. `gemini_augment.py`에 새 결함 등록

`DEFECT_CONFIGS` 딕셔너리에 새 항목을 추가한다.

**bbox 방식 (레퍼런스 이미지에 bbox가 그려져 있는 경우)**:

```python
DEFECT_CONFIGS = {
    # ... 기존 ...

    "newcategory_defectname": {
        "total_images": 100,                    # 생성할 이미지 수
        "description": "결함에 대한 영어 설명",

        "prompt_base": (
            "Generate a new image of a [부품명] with a [결함명] defect. "
            "[결함이 무엇인지 구체적 설명]. "
            "The FIRST image is a NORMAL [부품명] — use it as the base appearance reference. "
            "The REMAINING images are DEFECTIVE [부품명]s "
            "(defect areas highlighted with BLUE BORDER — do NOT include in output). "
        ),

        "prompt_key_instruction": (
            "MANDATORY: Output MUST contain exactly one [결함명] defect. "
            "[결함의 핵심 특성 설명]. "
            "Place the defect at a DIFFERENT POSITION than shown in references. "
            "DEFECT SIZE: same as the blue boxes or SMALLER. "
            "Do NOT include blue markings. "
        ),

        "prompt_variations": [
            "Place the defect in the upper-left area.",
            "Place the defect slightly left of center.",
            "Place the defect in the lower-right area.",
            "Place the defect near the top-center.",
            "Place the defect in the upper-right area.",
            "Place the defect near the bottom-center.",
            "Place the defect on the left side.",
            "Place the defect in the lower-left area.",
            "Place the defect slightly right of center.",
            "Place the defect in the middle-right area.",
        ],

        "prompt_style": (
            "Industrial inspection photography, consistent lighting, sharp focus. "
            "Maintain exact same [부품명] shape, material, color as the FIRST (normal) reference. "
            "Only add one SMALL defect. No blue markings or annotation marks."
        ),
    },
}
```

**원본 이미지 방식 (bbox 없는 원본 data 이미지를 레퍼런스로 사용)**:

```python
"NewCategory_DefectA": {
    "total_images": 75,
    "ref_dir": "NewCategory/NewCategory_DefectA",    # normal_00 위치
    "out_dir": "NewCategory/NewCategory_DefectA",    # 출력 위치
    "data_ref": {                                     # 원본 데이터에서 결함 샘플 로드
        "data_dir": f"{DATA_ROOT}/NewCategory/train",
        "annotation": f"{DATA_ROOT}/NewCategory/train/_annotations.coco.json",
        "cat_id": 0,                                  # COCO category_id
        "n_samples": 9,                               # 로드할 결함 예시 수
    },
    "description": "...",
    "prompt_base": "...",
    "prompt_key_instruction": "...",
    "prompt_variations": [...],
    "prompt_style": "...",
},
```

카테고리 그룹으로 묶으려면 `CLASS_GROUPS`에도 추가:

```python
CLASS_GROUPS = {
    # ... 기존 ...
    "NewCategory": ["NewCategory_DefectA", "NewCategory_DefectB"],
}
```

---

## 6. 프롬프트 다듬는 팁

### 6-1. 현재 프롬프트의 공통 패턴 (잘 작동하는 것들)

1. **"MANDATORY" + "MUST"**: 결함을 반드시 포함하라는 강조 → 빈 이미지 생성 방지
2. **"Do NOT generate a defect-free image"**: 명시적 금지 → 정상 이미지 생성 방지
3. **"DEFECT SIZE: same as blue boxes or SMALLER"**: 크기 제한 → 비현실적으로 큰 결함 방지
4. **"Do NOT include blue markings"**: 레퍼런스의 파란색 bbox가 출력에 나오지 않도록
5. **"Maintain exact same ... shape, material, color"**: 배경/부품 일관성 유지
6. **10가지 위치 변형(prompt_variations)**: 결함 위치 다양성 확보

### 6-2. 품질이 안 좋을 때 개선 방법

| 문제 | 해결 방법 |
|------|----------|
| 결함이 생성되지 않음 | `prompt_key_instruction`에 "MANDATORY", "MUST", "Do NOT generate defect-free" 강화 |
| 결함이 너무 큼 | "DEFECT SIZE: TINY, SUBTLE" 또는 "same size as references or SMALLER" 추가 |
| 파란색 마킹이 출력에 나옴 | "ABSOLUTE RULE — NO BLUE MARKINGS" 등 더 강조, "No overlaid graphics" 추가 |
| 배경이 달라짐 | `prompt_style`에 "identical to FIRST reference" 강화 |
| 부품 모양이 달라짐 | "CRITICAL: Keep the same [부품] shape" 추가 |
| 결함 위치가 한 곳에 몰림 | `prompt_variations` 수를 10→20으로 늘리거나 더 구체적 위치 지정 |
| 같은 이미지만 반복 생성 | `temperature`를 0.3→0.5로 올리기 (코드 880줄 `temperature=0.3`) |

### 6-3. temperature 조절

`gemini_augment.py` 880줄:
```python
config=types.GenerateContentConfig(
    temperature=0.3,          # 낮을수록 일관적, 높을수록 다양함
    response_modalities=["Image"]
)
```

- `0.2~0.3`: 안정적이지만 다양성 부족
- `0.4~0.6`: 적당한 다양성 (권장)
- `0.7+`: 다양하지만 품질 불안정

### 6-4. 레퍼런스 이미지 수 조절

코드 869줄:
```python
MAX_DEFECT_REFS = min(4, n_defect)  # 한 번에 보여줄 결함 레퍼런스 수
```

- 레퍼런스가 많으면 Gemini가 결함 특성을 더 잘 이해
- 하지만 너무 많으면 API 토큰 소비 증가 + 프롬프트 혼란 가능
- 4장 정도가 적당

---

## 7. API 호출 구조 (이해용)

Gemini API에 보내는 `contents` 리스트 구성:

```
contents = [
    정상 이미지 (PNG bytes),          ← 항상 1장
    결함 예시 이미지 #1 (PNG bytes),   ← 최대 4장 (순환 선택)
    결함 예시 이미지 #2 (PNG bytes),
    결함 예시 이미지 #3 (PNG bytes),
    결함 예시 이미지 #4 (PNG bytes),
    텍스트 프롬프트 (문자열)            ← prompt_base + key_instruction + variation + style
]
```

- 모델: `gemini-2.5-flash-image`
- temperature: 0.3
- 이미지 간 딜레이: 10초
- Rate limit 시 자동 600초 대기 후 재시도
- 최대 3회 재시도 후 스킵

---

## 8. 생성 후 다음 단계

### 8-1. 생성된 이미지를 data_augmented/로 이동

현재 생성 이미지는 `scripts/augmentation/vision_ai_generated/` 하위에 저장된다. 실험에 사용하려면 `data_augmented/{카테고리}/gen_ai/images/`로 복사해야 한다.

```bash
# 예: Screw 생성 이미지 배치
cp scripts/augmentation/vision_ai_generated/screw_defect/*.png \
   data_augmented/Screw/gen_ai/images/
```

### 8-2. 어노테이션 작성

생성된 이미지에는 **어노테이션(annotation)이 없다**. 라벨링 서버로 수동 라벨링해야 한다.

```bash
cd /home/jjh0709/gitrepo/VISION-Instance-Seg
python labeling_server/app.py --category Screw --split gen_ai --port 5201
# → 브라우저에서 http://lifeai.suwon.ac.kr:5201 접속
# → "서버 이미지 목록"에서 이미지 선택 → bbox + segmentation 라벨링 → 저장
```

---

## 9. 현재 상태 & 남은 작업

### 이미 생성 완료된 것

| 결함 유형 | 생성 수 | 비고 |
|----------|---------|------|
| casting_Inclusoes | ~50장 | progress 파일 확인 |
| casting_Rechupe | ~50장 | progress 파일 확인 |
| screw_defect | ~100장 | progress 파일 확인 |
| Console 4종 | 각 ~20장 | progress 파일 확인 |
| Cylinder 4종 | 각 ~20장 | progress 파일 확인 |
| Wood 2종 | 각 ~50장 | progress 파일 확인 |

### 추가 생성이 필요한 것 (우선순위 높음)

| 작업 | 이유 | 필요량 |
|------|------|--------|
| Cable GenAI 추가 | 현재 104장, 실험에 250장 필요 | +146장 |
| Casting GenAI 추가 | 현재 193장, 실험에 250장 필요 | +57장 |

---

## 10. 빠른 참조 (복사해서 쓰세요)

```bash
# 환경 준비
conda activate jjh
cd /home/jjh0709/gitrepo/VISION-Instance-Seg/scripts/augmentation
export GEMINI_API_KEY='your_key'

# 테스트 (5장만)
python gemini_augment.py screw_defect --count 5

# 본격 생성 (백그라운드)
nohup python -u gemini_augment.py Console > Console.log 2>&1 &

# 진행 확인
tail -f Console.log
cat progress_Console_Collision.json

# 생성된 이미지 확인
ls vision_ai_generated/Console/Console_Collision/ | wc -l

# 이미지 데이터 폴더로 복사
cp vision_ai_generated/screw_defect/*.png \
   ../../data_augmented/Screw/gen_ai/images/

# 라벨링 서버 실행
cd /home/jjh0709/gitrepo/VISION-Instance-Seg
python labeling_server/app.py --category Screw --split gen_ai --port 5201
```
