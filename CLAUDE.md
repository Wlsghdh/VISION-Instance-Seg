# CLAUDE.md
# Claude Code가 이 파일을 자동으로 읽습니다.
# 프로젝트 컨텍스트를 파악하는 데 사용됩니다.

## 프로젝트 개요

- **프로젝트명**: VISION Instance Segmentation
- **목적**: Gemini API로 불량 이미지 증강 → instance segmentation 성능 비교 연구
- **GitHub**: https://github.com/Wlsghdh/VISION-Instance-Seg
- **작업 서버**: lifeai (`/home/jjh0709/gitrepo/VISION-Instance-Seg`)
- **대상 부품**: 14개 카테고리 (Cable, Screw, Casting, Console, Cylinder, Capacitor, Electronics, Groove, Hemisphere, Lens, PCB_1, PCB_2, Ring, Wood)
- **즉시 실험 가능**: Cable, Screw, Casting (train 이미지 + 증강 데이터 완비)
- **확장 후보**: Console (GenAI 187장 있음), Cylinder (GenAI 26장 있음)
- **평가지표**: mAP, mAR

---

## 데이터 구조

원본:
```
data/{Category}/train/{images/, _annotations.coco.json}
data/{Category}/val/{이미지 직접 저장, _annotations.coco.json}   ← val = test용
```

증강:
```
data_augmented/{Category}/gen_ai/{images/, annotations.json}
data_augmented/{Category}/traditional_aug/{images/, annotations.json}
```

주요 데이터 현황 (즉시 실험 가능 3종):

| 카테고리 | Train | Val | GenAI | Trad Aug | 결함 클래스 |
|----------|:-----:|:---:|:-----:|:--------:|-----------|
| Cable | 26장 | 131장 | 104장 | 2,750장 | thunderbolt |
| Screw | 57장 | 63장 | 256장 | 250장 | defect |
| Casting | 54장 | 51장 | 193장 | 250장 | Inclusoes, Rechupe |

- val/ 디렉토리는 images/ 하위 폴더 없이 이미지 직접 저장
- Cable val: break(id=0) + thunderbolt(id=1) 혼재 → **thunderbolt만 평가에 사용**
- 실험 시 `training/data_pipeline.py`로 필요한 조합 병합 (기존 `merge_dataset.py` 대체)
- **val 데이터는 절대 변경 안 함**

---

## 레포 디렉토리 구조

```
VISION-Instance-Seg/
├── CLAUDE.md                        ← Claude Code 자동 인식 (프로젝트 컨텍스트)
├── README.md
├── RULE.md
├── requirements.txt
├── .gitignore
│
├── docs/
│   ├── experiment_plan.md
│   ├── data_spec.md
│   └── legacy/                      ← ahnbi1 기존 코드 분석 결과
│       ├── annotation_tool_analysis.md
│       ├── gemini_augment_analysis.md
│       └── traditional_aug_analysis.md
│
├── configs/                         # 모델 학습 설정
│   ├── mask_rcnn/
│   ├── cascade_rcnn/
│   ├── cascade_mask_rcnn/
│   ├── solov2/
│   └── mask_dino/
│
├── data/                            # ⛔ .gitignore (직접 서버에 배치)
│   ├── Cable/{train,test}/{images/,annotations.json}
│   ├── Screw/...
│   └── Casting/...
│
├── data_augmented/                  # ⛔ .gitignore (직접 서버에 배치)
│   ├── Cable/{gen_ai,traditional_aug}/{images/,annotations.json}
│   ├── Screw/...
│   └── Casting/...
│
├── scripts/
│   ├── augmentation/
│   │   ├── gemini_augment.py
│   │   ├── traditional_augment.py
│   │   └── prompts/
│   │       ├── cable_prompt.txt
│   │       ├── screw_prompt.txt
│   │       └── casting_prompt.txt
│   ├── data_utils/
│   │   ├── merge_dataset.py
│   │   ├── rename_val_to_test.py
│   │   ├── convert_format.py
│   │   └── validate_annotations.py
│   └── evaluation/
│       └── eval_metrics.py
│
├── labeling_server/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── training/                        # 통합 학습 환경 (2026-03-08 구축)
│   ├── config.py                    # 중앙 설정 (카테고리, 모델 7종, 실험 3종)
│   ├── data_pipeline.py             # 데이터 병합 + 프레임워크별 등록
│   ├── train.py                     # 통합 CLI (--category/--model/--experiment)
│   ├── evaluate.py                  # 독립 평가
│   ├── adapters/                    # 모델 어댑터 패턴
│   │   ├── base.py                  # ModelAdapter ABC
│   │   ├── detectron2_adapter.py    # Mask R-CNN, Cascade Mask R-CNN, MaskDINO, Mask2Former
│   │   └── mmdet_adapter.py         # Cascade R-CNN, SOLOv2, RTMDet-Ins
│   ├── utils/
│   │   ├── maskdino_mapper.py       # Polygon→BitMask 변환
│   │   └── report.py               # 결과 CSV/비교 테이블
│   └── maskdino/                    # 기존 MaskDINO 학습 코드 (reference)
│
└── results/                         # ⛔ .gitignore
    ├── experiment1/
    ├── experiment2/
    └── experiment3/
```

---

## 기존 코드 분석 (상세 → docs/legacy/ 참조)

| 분석 대상 | 원본 경로 (ahnbi1) | 분석 문서 |
|----------|------------------|----------|
| 라벨링 툴 | `/data2/project/2026winter/jjh0709/Resen/before/vision_ai_labeling/annotation_tool_v8.py` | `docs/legacy/annotation_tool_analysis.md` |
| Gemini 증강 | `/data2/project/2026winter/jjh0709/Generated_AI/generate_defects.py` | `docs/legacy/gemini_augment_analysis.md` |
| 전통 증강 | `/data2/project/2026winter/jjh0709/AA_CV_R/prepare_experiments.py` | `docs/legacy/traditional_aug_analysis.md` |

---

## 실험 계획 요약

### 실험 1: 생성AI 증강 수에 따른 성능 변화

| 조건 | 원본 | 생성AI | 합계 |
|------|------|--------|------|
| Baseline | 25장 | 0 | 25 |
| +50 | 25장 | 50 | 75 |
| +100 | 25장 | 100 | 125 |
| +150 | 25장 | 150 | 175 |
| +200 | 25장 | 200 | 225 |
| +250 | 25장 | 250 | 275 |

모델: Mask R-CNN, Cascade Mask R-CNN

### 실험 2: 전통적 증강 vs 생성형 AI 증강 비교 (5가지 조건)

| # | 구성 | 총 데이터 |
|---|------|---------|
| 1 | 원본 25장 | 25 |
| 2 | 원본 25 + 전통 250 | 275 |
| 3 | 원본 25 + 생성AI 250 | 275 |
| 4 | 원본 25 + 생성AI 250 + 전통 250 | 525 |
| 5 | 원본 25 + 생성AI 250 + 전통 2,750 | 3,025 |

### 실험 3: 7종 모델 비교

모델: Mask R-CNN, Cascade R-CNN, Cascade Mask R-CNN, SOLOv2, Mask DINO, +최신 2종

데이터 조건:
1. 원본 전체
2. 원본 전체 + 전통 증강 3,000장
3. 원본 전체 + 생성AI 250장 + 전통 증강 2,750장

---

## 설치된 프레임워크

| 패키지 | 버전 | 용도 |
|--------|------|------|
| detectron2 | 0.6 | Mask R-CNN, Cascade Mask R-CNN, MaskDINO, Mask2Former |
| mmcv | 2.1.0 | mmdet 의존성 |
| mmdet | 3.3.0 | Cascade R-CNN, SOLOv2, RTMDet-Ins |
| mmengine | 0.10.7 | mmdet Runner |
| MaskDINO repo | `/home/jjh0709/gitrepo/MaskDINO/` | + deformable attention CUDA ops |
| Mask2Former repo | `/home/jjh0709/gitrepo/Mask2Former/` | + deformable attention CUDA ops |

## 통합 학습 CLI 사용법

```bash
cd /home/jjh0709/gitrepo/VISION-Instance-Seg

# 학습
python -m training.train --category Cable --experiment exp2 --condition cond1 --model mask_rcnn
python -m training.train --category Screw --experiment exp2 --condition all --model maskdino

# 평가만
python -m training.train --category Cable --experiment exp2 --condition cond1 --model mask_rcnn --eval-only

# 데이터 준비만
python -m training.data_pipeline --category Cable --experiment exp2 --condition cond1

# 결과 리포트
python -m training.utils.report --experiment exp2 --csv
```

## 주의사항

- `data/`, `data_augmented/`는 Git 추적 안 함 (서버에 직접 배치)
- 어노테이션 형식: **COCO format** (instance segmentation)
- Python 3.11, conda env `jjh`, CUDA 12.2, A100 80GB x2
- Cable val의 break(id=0) → 평가에서 제외, thunderbolt(id=1)만 사용
- Mask2Former import 시 `mask2former.config`만 선택적 임포트 (데이터셋 중복 등록 방지)
- **작업 완료 시마다 이 파일의 [현재 진행 상황]을 업데이트할 것**

---

## 현재 진행 상황

- [x] ahnbi1 기존 코드 분석 완료
- [x] CLAUDE.md 및 docs/legacy/*.md 생성
- [x] 라벨링 툴 리팩토링 (labeling_server/app.py v9)
- [x] Cable train 데이터 정리 (thunderbolt 26장)
- [x] gen_ai 데이터 설정 (Cable 104장, Screw 256장, Casting 193장, Console 187장, Cylinder 26장)
- [x] 전통 증강 (Cable 2750장, Screw 250장, Casting 250장)
- [x] merge_dataset.py 작성
- [x] **통합 학습 환경 구축** (2026-03-08)
  - detectron2, mmcv, mmdet 설치
  - MaskDINO, Mask2Former repo 클론 + CUDA ops 빌드
  - 7종 모델 어댑터, 통합 CLI, 데이터 파이프라인, 평가/리포트 스크립트
  - 데이터 파이프라인 검증 완료 (Cable exp2 cond1 → 25장 병합 OK)
- [ ] Cable GenAI 추가 생성 (+146장 → 250장 필요)
- [ ] Casting GenAI 추가 생성 (+57장 → 250장 필요)
- [ ] Screw/Casting 전통 증강 추가 (250→2,750장)
- [ ] 실제 학습 테스트 실행 (E2E 검증)
- [ ] exp1/exp2/exp3 본 학습
- [ ] gemini_augment.py 리팩토링
- [ ] 나머지 카테고리 데이터 정리 (Console, Cylinder 등)
