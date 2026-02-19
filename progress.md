# 진행 상황 (progress.md)

> 최종 업데이트: 2026-02-19
> 작업 서버: lifeai (`/home/jjh0709/gitrepo/VISION-Instance-Seg`)

---

## 전체 진행 현황

| 단계 | 항목 | 상태 | 비고 |
|------|------|------|------|
| 데이터 | Cable train 정리 | ✅ 완료 | break 제거, images/ 구조, 26장 |
| 데이터 | Cable gen_ai 배치 | ✅ 완료 | cable_transfer → 105장 |
| 데이터 | Cable traditional_aug 생성 | ✅ 완료 | 2750장, seed=42 |
| 데이터 | Screw train 구조 정리 | ✅ 완료 | images/ 구조, 57장 |
| 데이터 | Casting train 구조 정리 | ✅ 완료 | images/ 구조, 54장 |
| 데이터 | Screw gen_ai 배치 | ⏳ 대기 | gen_ai 데이터 미입수 |
| 데이터 | Casting gen_ai 배치 | ⏳ 대기 | gen_ai 데이터 미입수 |
| 데이터 | Screw traditional_aug 생성 | ⏳ 대기 | gen_ai 배치 후 실행 |
| 데이터 | Casting traditional_aug 생성 | ⏳ 대기 | gen_ai 배치 후 실행 |
| 스크립트 | traditional_augment.py | ✅ 완료 | albumentations 2.x, 마스크 기반 |
| 스크립트 | labeling_server/app.py v9 | ✅ 완료 | argparse, gen_ai 브라우징 |
| 스크립트 | gemini_augment.py | ⏳ 보류 | Step 2에서 진행 |
| 스크립트 | merge_dataset.py | ❌ 미작성 | Step 4 |
| 실험 | 학습/평가 스크립트 | ❌ 미작성 | Step 5 |

---

## 현재 데이터 통계

### data/ (원본)

| 카테고리 | Split | 이미지 수 | 어노테이션 수 | 카테고리(id) | 구조 |
|----------|-------|----------|-------------|------------|------|
| Cable | train | 26장 | 34개 | thunderbolt(1) | images/ ✅ |
| Cable | val | 131장 | — | break(0)+thunderbolt(1) | flat (수정 금지) |
| Cable | inference | 1146장 | 0개 | — | flat (수정 금지) |
| Screw | train | 57장 | 75개 | defect(0) | images/ ✅ |
| Screw | val | — | — | — | flat (수정 금지) |
| Casting | train | 54장 | 59개 | Inclusoes(0)+Rechupe(1) | images/ ✅ |
| Casting | val | — | — | — | flat (수정 금지) |

### data_augmented/ (증강)

| 카테고리 | Split | 이미지 수 | 어노테이션 수 | 상태 |
|----------|-------|----------|-------------|------|
| Cable | gen_ai | 105장 | 105개 | ✅ 완료 |
| Cable | traditional_aug | 2750장 | 3559개 | ✅ 완료 (seed=42) |
| Screw | gen_ai | — | — | ⏳ 데이터 대기 |
| Screw | traditional_aug | — | — | ⏳ 대기 |
| Casting | gen_ai | — | — | ⏳ 데이터 대기 |
| Casting | traditional_aug | — | — | ⏳ 대기 |

---

## 상세 작업 기록

### [완료] Cable train 데이터 정리

**문제**: `break`(id=0), `thunderbolt`(id=1) 혼재, 이미지가 flat 구조

**처리**:
- break-only 이미지 15장 삭제 (`000001`, `000005~007`, `000011~014`, `000016`, `000022`, `000028`, `000033`, `000036`, `000038`, `000040`)
- 혼재 이미지 5장: break annotation만 제거, thunderbolt 유지
- `images/` 서브디렉토리 생성 후 26장 이동
- `annotations.json` 생성 (categories: thunderbolt id=1)
- `_annotations.coco.json` — 원본 백업으로 유지

---

### [완료] Cable gen_ai 데이터 배치 (cable_transfer)

**cable_transfer 패키지 구성**:
```
cable_transfer/
├── Cable_000000.jpg ~ Cable_000104.jpg  (105장) → gen_ai 이미지
├── thunderbolt_000001.png ~ ~000100.png (100장) → 원본 thunderbolt (제거)
├── 000000.jpg 등 26장                           → 원본 train (이미 존재, 무시)
└── annotations.json                             → Cable_*: cat_id=1 / thunderbolt_*: cat_id=0
```

**처리**:
- `Cable_XXXXXX.jpg` 105장만 `data_augmented/Cable/gen_ai/images/`로 복사
- `annotations.json` 생성 (Cable_ 이미지만 포함, thunderbolt id=1, 105개 annotation)
- `thunderbolt_*.png`, 원본 `000XXX.jpg` 항목 제거

---

### [완료] Cable traditional_aug 생성

```bash
python scripts/augmentation/traditional_augment.py \
    --category Cable --n_augment 2750 --seed 42
```
- 소스: `data/Cable/train/images/` 26장
- 출력: `data_augmented/Cable/traditional_aug/images/` 2750장
- 이미지당 약 105회 증강
- 파일명 형식: `000000_aug0000.jpg` ~ `000039_aug0105.jpg`

---

### [완료] Screw/Casting train 구조 정리

- 이미지를 `images/` 서브디렉토리로 이동
- `_annotations.coco.json` → `annotations.json` 동일 내용으로 복사
- 카테고리 필터링 없음 (Screw: defect만 있음 / Casting: 두 카테고리 모두 유지)

---

### [완료] labeling_server/app.py v9 리팩토링

**변경 내용**:
- v8 하드코딩 경로 → argparse `--category`, `--split`으로 동적 결정
- 새 엔드포인트 4개 추가 (서버 이미지 목록, 서빙, annotation 조회, 기존 저장)
- HTML: 서버 이미지 목록 모달 + 기존 annotation 자동 로드
- 저장 경로: `data_augmented/{category}/{split}/annotations.json`

---

### [완료] .gitignore 보완

추가된 패턴:
- `work_dirs/` — mmdetection 학습 출력
- `*.bin`, `*.safetensors` — HuggingFace 모델 가중치
- `*.log.json` — mmdetection 로그
- `mlruns/` — MLflow
- `*.backup_*` — annotation 자동 백업 파일

---

## 다음 작업 순서

```
1. Screw/Casting gen_ai 데이터 입수 후:
   - cable_transfer와 동일한 방식으로 data_augmented/ 배치
   - labeling_server로 annotation 검수/수정

2. Screw/Casting traditional_aug 실행:
   python scripts/augmentation/traditional_augment.py --category Screw --n_augment 2750
   python scripts/augmentation/traditional_augment.py --category Casting --n_augment 2750

3. merge_dataset.py 작성 (실험 조건별 데이터 조합)

4. 실험 학습 스크립트 작성 (mmdetection 기반)

5. 실험 1 시작: Cable Baseline (Mask R-CNN, 원본 25장)
```
