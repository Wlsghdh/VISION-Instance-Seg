# VISION Instance Segmentation - 프로젝트 리포트

> 최종 업데이트: 2026-03-16

---

## 1. 프로젝트 개요

- **목적**: Gemini API로 불량 이미지 증강 → instance segmentation 성능 비교 연구
- **대상 부품**: 6개 카테고리 (Cable, Screw, Casting, Console, Cylinder, Wood)
- **핵심 질문**: 생성형 AI 증강이 전통적 증강 대비 얼마나 효과적인가?
- **평가지표**: mAP, mAR

---

## 2. 데이터 현황

### 2-1. 카테고리별 데이터

| 카테고리 | Train | Val | 결함 클래스 | GenAI (기존) | GenAI (추가 생성) | 전통 증강 |
|----------|:-----:|:---:|-----------|:----------:|:---------------:|:--------:|
| Cable | 26장 | 131장 | thunderbolt | 104장 | +100장 | 2,750장 |
| Screw | 57장 | 63장 | defect | 256장 | +44장 | 250장 |
| Casting | 54장 | 51장 | Inclusoes, Rechupe | 193장 | +107장 | 250장 |
| Console | 20장* | 64장 | Collision, Dirty, Gap, Scratch | 256장 | +44장 | - |
| Cylinder | 20장* | 76장 | Chip, PistonMiss, Porosity, RCS | 179장 | +121장 | - |
| Wood | 20장* | 34장 | impurities, pits | 0장 | +200장 | - |

> *Console/Cylinder/Wood: train annotation은 있으나 images/ 폴더 정리 상태 기준

### 2-2. GenAI 서브클래스별 상세 (추가 생성 필요량)

| 도메인 | 서브클래스 | 기존 | 목표 | 추가 생성 |
|--------|----------|:----:|:----:|:---------:|
| Cable | thunderbolt | 104 | 204 | **100장** |
| Casting | Inclusoes | 78 | 150 | **72장** |
| Casting | Rechupe | 115 | 150 | **35장** |
| Screw | defect | 256 | 300 | **44장** |
| Console | Collision | 50 | 75 | **25장** |
| Console | Dirty | 66 | 75 | **9장** |
| Console | Gap | 71 | 75 | **4장** |
| Console | Scratch | 69 | 75 | **6장** |
| Cylinder | Chip | 52 | 75 | **23장** |
| Cylinder | PistonMiss | 43 | 75 | **32장** |
| Cylinder | Porosity | 99 | 75 | 0 (초과) |
| Cylinder | RCS | 0 | 75 | **75장** |
| Wood | impurities | 0 | 100 | **100장** |
| Wood | pits | 0 | 100 | **100장** |
| **합계** | | | | **525장** |

---

## 3. Gemini API 증강 프롬프트 전략

### 3-1. 프롬프트 구조

4-part 구성으로 Gemini 2.5 Flash Image 모델에 전달:

```
prompt = prompt_base + prompt_key_instruction + prompt_variation + prompt_style
```

| 구성 요소 | 역할 |
|----------|------|
| `prompt_base` | 대상 부품과 결함 유형 설명, 정상/결함 레퍼런스 이미지 역할 지정 |
| `prompt_key_instruction` | 필수 생성 규칙 (크기 제한, 위치 변경, 결함 필수 포함) |
| `prompt_variation` | 10개 위치 변형 중 순환 선택 (upper-left, center, lower-right 등) |
| `prompt_style` | 촬영 스타일, 배경 유지, 출력 품질 지시 |

### 3-2. 레퍼런스 이미지 입력

| 방식 | 적용 대상 | 설명 |
|------|----------|------|
| **Blue bbox 방식** | Cable, Casting, Screw | 정상 이미지 1장 + 결함 부위에 파란 테두리 표시한 레퍼런스 이미지 |
| **원본 data_ref 방식** | Console, Cylinder, Wood | 정상 이미지 1장 + 원본 train 데이터에서 랜덤 샘플링한 결함 이미지 (bbox 없음) |

### 3-3. 토큰 절약 전략

| 항목 | 이전 | 현재 | 절약 효과 |
|------|:----:|:----:|:---------:|
| defect 레퍼런스 샘플 수 (data_ref) | 9장 | 3장 | 67% 감소 |
| 매 생성시 전송 이미지 수 | normal 1 + defect 4 = 5장 | normal 1 + defect 2 = 3장 | 40% 감소 |
| 레퍼런스 샘플링 방식 | 고정 순서 | **random.sample** | 다양성 증가 |

### 3-4. 핵심 프롬프트 개선 사항 (피드백 반영)

**문제**: 기존 생성 결과에서 defect가 너무 크거나 과장되는 경향

**해결 전략**:

1. **실제 defect 크기 비율 기반 제한**
   - 각 카테고리의 실제 annotation bbox 크기를 분석하여 프롬프트에 % 제한 명시
   - 예: Casting 평균 0.1% → "less than 0.5%", Cylinder 평균 6.6% → "less than 2-3%"

2. **심리적 억제 패턴**
   - "The defect must be MUCH SMALLER than you initially think"
   - "If in doubt, make it EVEN SMALLER"
   - "Make them a QUARTER as visible as you initially think"

3. **카테고리별 맞춤 지시**

| 카테고리 | 크기 제한 | 특수 지시 |
|----------|:--------:|----------|
| Cable thunderbolt | blue bbox 동일 크기 이하 | bbox 위치만 변경 |
| Casting Inclusoes | < 0.5% | 배경에 artifact 절대 금지, 배경 완전 동일 |
| Casting Rechupe | < 0.5% | 기본 사물 변형/왜곡 금지 |
| Screw defect | < 0.5% | blue marking 절대 금지 |
| Console Collision | 1-2% (중간 크기) | 배경에 묻히지 않게 contrast 확보 |
| Console Dirty | < 2% | 작고 자연스러운 smudge |
| Console Gap | < 1% | 부서진 게 아닌 **들린/분리된** 얇은 seam |
| Console Scratch | < 1% | 가늘지만 배경과 구분 가능한 contrast |
| Cylinder Chip | < 3% | 하단 rim/edge에만 배치 |
| Cylinder PistonMiss | < 2% | 자연스럽고 점진적, 인위적 삭제 금지 |
| Cylinder Porosity | < 2% | 최대 3개 pit, reference보다 살짝 작게 |
| Cylinder RCS | < 1% | 극도로 미세한 faint 스크래치 |
| Wood impurities | 1-3% | 자연스러운 irregular 형태, 점이 아닌 실제 얼룩 |
| Wood pits | 1-3% | 배경과 구분 가능한 contrast 확보 |

### 3-5. 생성 파라미터

```python
model = 'gemini-2.5-flash-image'
temperature = 0.3
response_modalities = ["Image"]
delay_between_images = 10초
max_retries = 3
rate_limit_backoff = 600초 (10분)
```

### 3-6. 프롬프트 예시 (Cylinder_RCS)

```
[Input Images]
  Image 1: normal_00.png (NORMAL cylinder)
  Image 2: 000001.jpg (DEFECT sample, random from train data)
  Image 3: 000007.jpg (DEFECT sample, random from train data)

[Prompt]
  Generate a new image of a precision-machined cylinder part with an RCS defect.
  The FIRST image is a NORMAL cylinder — use it as the base appearance reference.
  The REMAINING images are DEFECTIVE cylinders showing real RCS defects:
  multiple parallel linear scratches occurring simultaneously on the cylinder surface,
  like marks from a multi-point contact dragging across the surface at once.
  Study these examples to understand the pattern, spacing, and size of these scratches.

  MANDATORY: Output MUST contain an RCS defect —
  2–3 faint parallel scratches running in the same direction.
  CRITICAL SIZE CONSTRAINT: The scratch group must occupy less than 1% of the total image area.
  The scratches must be ALMOST INVISIBLE — extremely faint hairline marks, very short
  (less than 10% of image width).
  ...
  Place the parallel scratches diagonally in the lower-right area of the cylinder.

  Industrial inspection photography with slightly varied lighting.
  Maintain exact same cylinder shape, material, color as the FIRST (normal) image.
  The scratches must be EXTREMELY FAINT — barely visible surface scuffs, not prominent marks.
  Output must be a clean, realistic photo with no overlaid graphics or annotation marks.
```

---

## 4. 실험 계획

### 실험 1 (exp1): GenAI 증강 수에 따른 성능 변화

**목적**: GenAI 이미지를 몇 장 추가해야 성능이 올라가는지 확인

| 조건 | 원본 | GenAI | 합계 |
|------|:----:|:-----:|:----:|
| baseline | 25 | 0 | 25 |
| genai_50 | 25 | 50 | 75 |
| genai_100 | 25 | 100 | 125 |
| genai_150 | 25 | 150 | 175 |
| genai_200 | 25 | 200 | 225 |
| genai_250 | 25 | 250 | 275 |

**모델**: Mask R-CNN, Cascade Mask R-CNN

### 실험 2 (exp2): 전통 증강 vs GenAI 비교

**목적**: 같은 데이터량에서 증강 방식별 효과 비교

| 조건 | 원본 | GenAI | 전통 증강 | 합계 |
|------|:----:|:-----:|:--------:|:----:|
| cond1 | 25 | 0 | 0 | 25 |
| cond2 | 25 | 0 | 250 | 275 |
| cond3 | 25 | 250 | 0 | 275 |
| cond4 | 25 | 250 | 250 | 525 |
| cond5 | 25 | 250 | 2,750 | 3,025 |

**모델**: Mask R-CNN, Cascade Mask R-CNN, MaskDINO

### 실험 3 (exp3): 7종 모델 비교

**목적**: 동일 데이터 조건에서 모델 아키텍처별 성능 비교

| 조건 | 구성 |
|------|------|
| original_only | 원본 전체 |
| with_trad | 원본 + 전통 증강 3,000장 |
| with_genai_trad | 원본 + GenAI 250 + 전통 증강 2,750장 |

**모델** (7종):

| 모델 | 프레임워크 |
|------|-----------|
| Mask R-CNN (R50-FPN) | detectron2 |
| Cascade Mask R-CNN (R50-FPN) | detectron2 |
| MaskDINO (R50) | detectron2 + MaskDINO repo |
| Mask2Former (R50) | detectron2 + Mask2Former repo |
| Cascade R-CNN (R50-FPN) | mmdet |
| SOLOv2 (R50-FPN) | mmdet |
| RTMDet-Ins (S) | mmdet |

---

## 5. 학습 환경

| 항목 | 사양 |
|------|------|
| GPU | A100 80GB × 2 |
| CUDA | 12.2 |
| Python | 3.11 (conda env: jjh) |
| PyTorch | 2.5.1+cu121 |
| detectron2 | 0.6 |
| mmcv | 2.1.0 |
| mmdet | 3.3.0 |
| mmengine | 0.10.7 |

---

## 6. 통합 학습 CLI

```bash
# 학습
python -m training.train --category Cable --experiment exp2 --condition cond1 --model mask_rcnn

# 전체 조건 일괄
python -m training.train --category Screw --experiment exp2 --condition all --model maskdino

# 평가만
python -m training.train --category Cable --experiment exp2 --condition cond1 --model mask_rcnn --eval-only

# 데이터 준비 확인
python -m training.data_pipeline --category Cable --experiment exp2 --condition cond1

# 결과 리포트
python -m training.utils.report --experiment exp2 --csv
```

---

## 7. 실행 가능성 매트릭스 (현재 기준)

| | Cable | Screw | Casting | Console | Cylinder | Wood |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **exp1 전체** | genai 추가 생성 후 | 가능 | genai 추가 생성 후 | - | - | - |
| **exp2 전체** | genai 추가 생성 후 | 가능 | genai 추가 생성 후 | - | - | - |
| **exp3 original_only** | 가능 | 가능 | 가능 | 데이터 정리 후 | 데이터 정리 후 | 데이터 정리 후 |

---

## 8. 권장 실행 순서

1. **GenAI 추가 생성** (525장) — 현재 진행 중
2. **E2E 검증** — 짧은 학습으로 파이프라인 정상 작동 확인
3. **Screw exp2 전체** — 데이터 완비, 우선 실행
4. **3카테고리 × exp2 전체** — GenAI 생성 완료 후
5. **exp3 7모델 비교** — original_only부터 시작
6. **Console/Cylinder/Wood 확장** — GenAI 생성 + train 데이터 정리 후

---

## 9. 진행 상황

- [x] 기존 코드 분석 완료 (라벨링 툴, Gemini 증강, 전통 증강)
- [x] 라벨링 툴 리팩토링 (labeling_server/app.py v9)
- [x] Cable/Screw/Casting train 데이터 정리
- [x] 전통 증강 완료 (Cable 2750, Screw 250, Casting 250)
- [x] 통합 학습 환경 구축 (7종 모델 어댑터, CLI, 데이터 파이프라인)
- [x] Gemini 증강 프롬프트 전략 수립 및 튜닝 (2026-03-16)
- [x] Console/Cylinder/Wood GenAI 프롬프트 작성
- [ ] **GenAI 추가 생성 (525장)** ← 현재 진행 중
- [ ] E2E 학습 테스트
- [ ] exp1/exp2/exp3 본 학습
- [ ] 결과 분석 및 논문 작성
