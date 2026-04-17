# 01. 프로젝트 개요

## 📌 연구 목적

**산업 제품 결함 Instance Segmentation**에서, 클래스당 20장 수준의 극소규모 데이터로 시작할 때 어떤 데이터 증강이 유효한가?

- **대상 결함**: Dirty (이물질 오염), Inclusoes (포함물), impurities (불순물) — 3종
- **데이터 기반**: VISION-Datasets (Casting/Wood 등) 산업 결함 벤치마크에서 발췌
- **비교 대상**: 전통 증강 (flip/rotate/color jitter) vs 생성형 AI 증강 (Gemini)

## 🎯 최종 논문 제목 (B 확정)

> **전통 증강의 한계와 생성형 AI 증강의 시너지 — 소규모 결함 데이터셋의 공정 비교 연구**

(영문 부제 후보: "Limits of Traditional Augmentation and Synergy of Generative AI: A Fair Comparison on Small-scale Industrial Defect Datasets")

## 🏆 3가지 기여 (Contribution)

1. **Observation (Negative Finding)**
   소규모 산업 결함 데이터(클래스당 20장)에서 **전통 증강 단독은 baseline보다 성능 악화** 시킴을 실증. (cond2: -1.76 AP)

2. **Solution (Positive Finding)**
   **GenAI 증강 (Gemini) + 전통 증강 결합**이 시너지를 내며, **최적 혼합비 N=8** (GenAI 125 + 전통 1160/cls)에서 baseline 대비 **+5.15 AP** 달성. (Cascade Mask R-CNN: 11.26 → 16.41)

3. **Benchmark (Methodology)**
   **iter 기반 공정성 프로토콜** (통일 스케줄, cosine LR, COCO pretrained, fixed val) + 재현 가능한 코드·결과 공개.

## 📊 투고 대상

- **학회**: 한국정보과학회 2026 (KCC 2026)
- **분량**: 5-6쪽 (KCC 논문지 표준)
- **언어**: 한국어 (초록·제목은 영문 병기)

## 🗓 일정 (추정)

| 단계 | 기간 | 담당 |
|------|------|------|
| 추가 학습 (cond4_8x 5모델) | ~4/18 | usw + ahnbi3 |
| Figure/Table 최종화 | 4/19 | paper-visualizer |
| §1~§7 본문 작성 | 4/19~4/22 | jjh (+ paper-doctor 리뷰) |
| VISION-Datasets 인용 정리 | 4/20~4/22 | paper-references-manager |
| 논문박사 최종 리뷰 | 4/23 | paper-doctor |
| 투고 | 4/24~ | — |

## 👥 팀

- **jjh0709** (본 논문 대표 작성자, 본 폴더의 주인)
- **yjw** (exp1_3cls genai_25/50/100 담당 예정)
- **ldy** (exp2 cond4_7x~10x 담당, 완료)

## 📍 서버 환경

- **usw (lifeai)** [현재 주 작업]
  - GPU: A100 80GB × 2
  - 경로: `/home/jjh0709/gitrepo/VISION-Instance-Seg`
  - 용도: Exp2 cond4_4x~6x, cond4_8x cascade_rcnn 학습
- **ahnbi3 (수원대 서버)** [논문 작성용]
  - GPU: V100 32GB × 8
  - 경로: `/home/jjh0709/gitrepo/VISION-Instance-Seg`
  - 용도: cond4_8x solov2/rtmdet_ins 병렬 학습 + 논문 본문 작성
- **Github**: https://github.com/Wlsghdh/VISION-Instance-Seg (dev 브랜치)

## 🛠 사용 프레임워크

| 패키지 | 버전 | 용도 |
|--------|------|------|
| detectron2 | 0.6 | Mask R-CNN, Cascade, MaskDINO, Mask2Former |
| mmdet | 3.3.0 | Cascade R-CNN, SOLOv2, RTMDet-Ins |
| mmcv | 2.1.0 | mmdet 의존성 |
| mmengine | 0.10.7 | mmdet Runner |
| torch | 2.5.1 (usw) / 2.1.2 (ahnbi3) | deep learning |
| numpy | <2 (ahnbi3 호환성) | — |

## 📝 다음 파일

- [02_EXP1_EXPLAINED.md](02_EXP1_EXPLAINED.md): 실험 1 설명
- [03_EXP2_EXPLAINED.md](03_EXP2_EXPLAINED.md): 실험 2 설명
