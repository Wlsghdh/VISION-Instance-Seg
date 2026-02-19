# CLAUDE.md
# Claude Code가 이 파일을 자동으로 읽습니다.
# 프로젝트 컨텍스트를 파악하는 데 사용됩니다.

## 프로젝트 개요

- **프로젝트명**: VISION Instance Segmentation
- **목적**: Gemini API로 불량 이미지 증강 → instance segmentation 성능 비교 연구
- **GitHub**: https://github.com/Wlsghdh/VISION-Instance-Seg
- **작업 서버**: lifeai (`/home/jjh0709/gitrepo/VISION-Instance-Seg`)
- **대상 부품**: Cable, Screw, Casting
- **평가지표**: mAP, mAR

---

## 데이터 구조

원본 (VISION 데이터셋, val→test 이름 변경):
```
data/{Category}/train/{images/, annotations.json}
data/{Category}/test/{images/, annotations.json}
```

증강 (원본과 분리):
```
data_augmented/{Category}/gen_ai/{images/, annotations.json}
data_augmented/{Category}/traditional_aug/{images/, annotations.json}
```

- 라벨링 툴에서 gen_ai 이미지 라벨링 시 → `data_augmented/{Category}/gen_ai/`에 저장
- 전통 증강 실행 시 → `data_augmented/{Category}/traditional_aug/`에 저장
- 실험 시 `merge_dataset.py`로 필요한 조합만 병합
- **test 데이터는 절대 변경 안 함**

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
├── training/
│   ├── train.py
│   ├── test.py
│   └── run_experiments.sh
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

## 주의사항

- `data/`, `data_augmented/`는 Git 추적 안 함 (서버에 직접 배치)
- 어노테이션 형식: **COCO format** (instance segmentation)
- Python 3.9+, mmdetection 기반
- **작업 완료 시마다 이 파일의 [현재 진행 상황]을 업데이트할 것**

---

## 현재 진행 상황

- [x] ahnbi1 기존 코드 분석 완료
- [x] CLAUDE.md 및 docs/legacy/*.md 생성
- [ ] 레포 구조 세팅 (폴더, README, RULE.md, .gitignore)
- [ ] 라벨링 툴 리팩토링 (카테고리별 분리 저장)
- [ ] 증강 스크립트 정리 (다중 카테고리 지원)
- [ ] 전통 증강 스크립트 작성
- [ ] merge_dataset.py 작성
- [ ] 실험/평가 스크립트 작성
- [ ] Gemini 프롬프트 고도화
