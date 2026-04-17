# §2. 관련 연구 (Draft)

**상태**: 초안 대기.
**분량 목표**: 0.75쪽

---

## 2.1 Instance Segmentation

- **Mask R-CNN** (He et al. 2017, `he_2017_maskrcnn`): Faster R-CNN + mask head, 2-stage 표준
- **Cascade Mask R-CNN** (Cai & Vasconcelos 2018, `cai_2018_cascade`): 3-stage cascade로 품질 향상
- **Transformer 계열**: MaskDINO (`li_2023_maskdino`), Mask2Former (`cheng_2022_mask2former`)
- **Anchor-free**: SOLOv2 (`wang_2020_solov2`), RTMDet-Ins (`lyu_2022_rtmdet`)

→ 본 연구는 2-stage (Mask R-CNN, Cascade) + 1-stage (SOLOv2, RTMDet) + Transformer (MaskDINO, Mask2Former) 다양 모델에서 관찰 일관성 검증.

## 2.2 전통 데이터 증강

- Geometric: flip, rotate, crop, scale
- Photometric: color jitter, brightness, contrast
- Advanced: Mixup (`zhang_2018_mixup`), CutMix (`yun_2019_cutmix`), AutoAugment (`cubuk_2019_autoaugment`), RandAugment (`cubuk_2020_randaugment`)

→ 모두 **pixel-level perturbation**. 소규모 데이터에서는 **같은 이미지의 변형**만 제공 → feature diversity 부족.

## 2.3 생성형 AI 기반 증강

- **GAN 기반**: DatasetGAN (`zhang_2021_datasetgan`) — labeled data 생성
- **Diffusion 기반**: DiffAug (`zhao_2020_diffaug`), LDM (`rombach_2022_ldm`), Stable Diffusion
- **대규모 언어-비전 모델**: Gemini (`gemini_2023`), DALL-E — 텍스트 프롬프트로 결함 이미지 생성

→ 본 연구는 **Gemini** (multimodal) 기반 생성 + 수작업 라벨링 파이프라인 사용.

## 2.4 산업 결함 벤치마크

- **MVTec AD** (Bergmann 2019, `bergmann_2019_mvtec`): 15 카테고리 산업 결함, anomaly detection 표준
- **VISION-Datasets** (원 논문, **TODO 인용**): 14개 산업 부품 category, instance segmentation용
- PaDiM, PatchCore: embedding 기반 anomaly detection

→ **VISION-Datasets**가 본 연구의 실험 데이터 근원 (3 defect 클래스만 선별: Dirty, Inclusoes, impurities).

## 2.5 공정 benchmark / 재현성

- **Linear Scaling Rule** (Goyal 2017, `goyal_2017_largeminibatch`): lr ∝ batch
- **Cosine annealing** (Loshchilov 2017, `loshchilov_2017_sgdr`): early-stop 친화
- **Bag of Tricks** (He 2019, `he_2019_bagtricks`): ImageNet fine-tune 권장 실천

→ 본 연구는 이들을 준수하고, **iter 기반 스케줄 통일**로 데이터량 차이에도 공정한 gradient update 보장.

---

## 차별점 (본 논문이 다른 점)

1. **소규모 (클래스당 20장) + 산업 결함 + 증강 비교**의 3요소 조합은 기존에 없음
2. **전통 증강 단독이 해로움**을 실증한 연구는 드물다 (대부분 긍정 효과만 보고)
3. **GenAI + 전통 결합의 시너지**와 **최적 혼합비 N 스윕**은 신규 기여

---

## 레퍼런스 점검 지시

paper-references-manager agent에 제출:
```
subagent_type: paper-references-manager
prompt: docs/paper/draft/02_related.md 에 언급된 모든 인용 키가 refs.bib에 있는지 확인하고,
누락된 것과 VISION-Datasets 원 논문 placeholder 정리 현황 알려줘.
```
