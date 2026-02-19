# Gemini 증강 코드 분석: generate_defects.py

## 파일 개요

Google Gemini API (`gemini-2.5-flash-image`)를 사용하여 제조 결함 이미지를 생성하는 증강 스크립트. 레퍼런스 이미지(정상 + 결함 예시)를 입력으로 받아 새로운 결함 위치의 이미지를 생성함.

**원본 경로 (ahnbi1):**
`/data2/project/2026winter/jjh0709/Generated_AI/generate_defects.py`

**연관 파일:**
- `test_new_api.py` — API 키 연결/할당량 테스트 스크립트
- `test.py` — 간단한 Gemini API 테스트

---

## Gemini API 호출 방식

| 항목 | 내용 |
|------|------|
| 라이브러리 | `google.genai` (최신 SDK, `from google import genai`) |
| 모델 | `gemini-2.5-flash-image` |
| 온도 (temperature) | 0.3 |
| response_modalities | `["Image"]` |
| 이미지 간 지연 시간 | 35초 |
| 최대 재시도 횟수 | 3회 |
| Rate limit 대기 시간 | 600초 (10분) |
| 진행 상황 저장 파일 | `progress_{defect_type}.json` |

```python
client = genai.Client(api_key=GEMINI_API_KEY)

response = client.models.generate_content(
    model='gemini-2.5-flash-image',
    contents=contents,   # [normal_ref_image, defect_ref_image(s), text_prompt]
    config=types.GenerateContentConfig(
        temperature=0.3,
        response_modalities=["Image"]
    )
)
```

---

## 현재 지원 결함 유형 및 설정

| 결함 유형 | 생성 수 | 대상 부품 | 결함 설명 |
|-----------|--------|----------|-----------|
| `casting_Inclusoes` | 50장 | Casting | 비금속 이물질이 주조 표면에 박힌 결함 |
| `casting_Rechupe` | 50장 | Casting | 금속 응고 시 수축/공동으로 생긴 결함 |
| `screw_defect` | 100장 | Screw | 나사 제조 결함 (크랙, 버, 변형 등) |

**Cable 미지원**: Cable 결함 생성 로직 없음.

---

## 현재 사용 중인 프롬프트 전문

각 결함 유형의 프롬프트는 4개 파트를 이어붙여 구성:

```
prompt = prompt_base + prompt_key_instruction + prompt_variation[i % 10] + prompt_style
```

### casting_Inclusoes 프롬프트

**prompt_base:**
```
Generate a new image of a metal casting part with an inclusion defect.
An inclusion defect means non-metallic foreign material (such as sand, slag, or oxide)
is trapped inside the metal casting surface.
The first reference image shows a NORMAL casting without any defect.
The other reference images show DEFECTIVE castings with inclusion defects
(the defect areas are highlighted in blue in the references).
```

**prompt_key_instruction:**
```
Generate a realistic casting image WITH an inclusion defect, but place the defect
at a DIFFERENT POSITION than shown in the reference defect images.
The defect should look natural - a small dark spot, discoloration, or rough patch
where foreign material is embedded in the metal surface.
Do NOT include any blue markings or highlights in the generated image.
The defect should look like a real inclusion, not artificially marked.
```

**prompt_variations (10가지, index % 10 순환):**
```
1. Place the inclusion defect slightly to the upper-left area of the casting surface.
2. Place the inclusion defect near the center of the casting surface.
3. Place the inclusion defect slightly to the lower-right area of the casting surface.
4. Place the inclusion defect near the top edge of the casting surface.
5. Place the inclusion defect slightly to the upper-right area of the casting surface.
6. Place the inclusion defect near the bottom area of the casting surface.
7. Place the inclusion defect slightly to the left side of the casting surface.
8. Place the inclusion defect near the lower-left area of the casting surface.
9. Place the inclusion defect slightly off-center to the right.
10. Place the inclusion defect near the middle-left area of the casting surface.
```

**prompt_style:**
```
Industrial inspection photography, consistent lighting, sharp focus on the casting surface.
Maintain the same casting part type, material, and overall appearance as the references.
CRITICAL: Keep the same casting shape and material. Only change the defect position.
Do NOT add blue markings. The defect must look completely natural and realistic.
```

---

## 이미지 생성 → 저장 파이프라인

```
reference_images/{defect_type}/
    ├── 0_normal.jpg          ← 정상 이미지 (첫 번째 파일)
    ├── 1_defect.jpg          ← 결함 예시 이미지 #1
    └── ...                   ← 추가 결함 예시 이미지들

↓ load_reference_images()      PIL로 로드 → bytes 변환

↓ generate_prompt(config, i)   variation을 (i % 10)으로 순환 선택

↓ API 요청 contents 구성:
    [
        types.Part.from_bytes(normal_ref),           # 정상 레퍼런스
        types.Part.from_bytes(defect_refs[i%N]),     # 결함 레퍼런스 (순환)
        types.Part.from_bytes(defect_refs[(i+1)%N]), # 추가 결함 레퍼런스 (있을 때)
        text_prompt                                   # 텍스트 프롬프트
    ]

↓ client.models.generate_content() 호출

↓ response.candidates[0].content.parts → part.inline_data.data

↓ 저장: vision_ai_generated/{defect_type}/{defect_type}_{i:03d}.png

↓ progress_{defect_type}.json 업데이트 (재시작 지원)
```

---

## 진행 상황 관리 (중단/재시작 지원)

```json
{
    "completed": [0, 1, 2, ...],
    "failed": [],
    "last_successful_index": 49,
    "start_time": 1707000000.0
}
```

- 스크립트 재실행 시 `last_successful_index + 1`부터 재개
- `completed` 목록에 있는 인덱스는 자동으로 스킵

---

## 현재 문제점

1. **Cable 미지원**: Casting과 Screw만 지원. Cable 결함 생성 로직 없음
2. **어노테이션 자동 생성 없음**: 생성된 이미지에 대한 COCO format 어노테이션은 수동 라벨링 필요
3. **경로 상대 경로**: `reference_images/`, `vision_ai_generated/`가 실행 위치에 의존
4. **API 키 소스코드 하드코딩**: 보안 취약점
5. **prompt_variations 10가지 고정**: 다양성 제한 (더 많은 변형 필요)
6. **다중 카테고리 처리 구조 없음**: 새 결함 유형 추가 시 `DEFECT_CONFIGS` 딕셔너리 직접 수정 필요

---

## 리팩토링 시 수정 필요 포인트

리팩토링 목표 파일: `scripts/augmentation/gemini_augment.py`

1. **argparse 추가**: `--category (Cable/Screw/Casting)`, `--num_images`, `--output_dir`
2. **프롬프트 외부화**: `scripts/augmentation/prompts/{category}_prompt.txt`에서 로드
3. **절대 경로 처리**: 레퍼런스 및 출력 경로를 인수로 받거나 프로젝트 루트 기준으로 처리
4. **Cable 프롬프트 신규 작성**: `scripts/augmentation/prompts/cable_prompt.txt`
5. **API 키 환경변수화**: `os.environ["GEMINI_API_KEY"]` 사용
6. **출력 경로 통일**: `data_augmented/{Category}/gen_ai/images/`로 저장
7. **생성 시간 로깅**: 이미지당 소요 시간, 총 소요 시간 기록
