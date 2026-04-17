# 06. 데이터 & 모델 상세

## 📊 데이터

### 원본 데이터 출처

**VISION-Datasets**에서 발췌한 3개 결함 클래스:

| 클래스 | 원본 카테고리 | 설명 |
|--------|:---:|------|
| **Dirty** | Console | 표면 이물질 오염 |
| **Inclusoes** | Casting | 주물 포함물 |
| **impurities** | Wood | 목재 불순물 |

→ Exp2_3cls 통합 카테고리로 묶어서 **3-class instance segmentation** 학습.

### 데이터 수량

| 분할 | 클래스당 | 총 이미지 | annotations |
|------|:---:|:---:|:---:|
| Train (원본) | 20 | 60 | ? |
| Val | 평균 27 | 82 | 113 |

→ **Val은 모든 조건에서 고정** (_exp2_3cls_val)

### 증강 데이터

**전통 증강 (traditional_aug)**:
- Flip, rotate, color jitter, brightness, contrast, cutout
- Albumentations 기반
- 클래스당 최대 2,750장 보유

**생성형 AI 증강 (gen_ai)**:
- **Gemini API** (Google) 생성 → 수작업 라벨링
- 클래스당: Dirty 187장, Inclusoes 193장, impurities ~ (여유 있음)
- Prompt 템플릿: `scripts/augmentation/prompts/*.txt`

### 디렉토리 구조

```
VISION-Instance-Seg/
├── data/                   # ⛔ gitignore
│   ├── Dirty/
│   │   ├── train/images/ + _annotations.coco.json
│   │   └── val/ + _annotations.coco.json
│   ├── Inclusoes/
│   └── impurities/
└── data_augmented/         # ⛔ gitignore
    ├── Dirty/
    │   ├── gen_ai/images/ + annotations.json
    │   └── traditional_aug/images/ + annotations.json
    ├── Inclusoes/
    └── impurities/
```

(※ 실제로는 data/ 와 data_augmented/ 에 Cable, Casting, Console 등 상위 카테고리별로 저장되어 있고, Dirty/Inclusoes/impurities는 그 안의 클래스로 존재. `training/config.py`의 `_cat_single_defect`가 필터링해서 추출.)

### 데이터 파이프라인

- `training/data_pipeline.py`가 조건별 병합:
  1. 원본 20장/클래스 샘플링 (`_sample_images_per_class`)
  2. GenAI N장 샘플링 (nested: N-25 ⊂ N)
  3. 전통 증강 M장 샘플링
  4. COCO JSON 생성 → `results/merged_datasets/{exp}/{cond}/{cat}/seed42/`
- Val 셋은 공통: `results/merged_datasets/_exp2_3cls_val/`

## 🤖 모델 (5~7종, cond4_8x 비교용)

### Detectron2 계열 (4종)

#### 1. Mask R-CNN
- **논문**: He et al. 2017
- **backbone**: ResNet-50 FPN
- **pretrained**: `detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl`
- **학습 설정**: SGD lr=0.0015, bs=12
- **특징**: 2-stage, ROI pooling + mask head

#### 2. Cascade Mask R-CNN
- **논문**: Cai & Vasconcelos 2018
- **backbone**: ResNet-50 FPN
- **pretrained**: `detectron2://Misc/cascade_mask_rcnn_R_50_FPN_3x/144998488/model_final_480dd8.pkl`
- **학습 설정**: SGD lr=0.0015, bs=12
- **특징**: 3-stage cascade refinement

#### 3. MaskDINO
- **논문**: Li et al. 2023
- **backbone**: ResNet-50 (config: `maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml`)
- **pretrained**: ImageNet R-50
- **학습 설정**: AdamW lr=1e-4, bs=4
- **특징**: DETR 기반 query 방식
- **특이사항**: 초기 loss가 수천 단위 (Hungarian matching 합산)

#### 4. Mask2Former
- **논문**: Cheng et al. 2022
- **backbone**: ResNet-50 (config: `maskformer2_R50_bs16_50ep.yaml`)
- **pretrained**: ImageNet R-50
- **학습 설정**: AdamW lr=1e-4, bs=4
- **특이사항**: import 버그 있었음 → `training/adapters/detectron2_adapter.py` 수정 완료 (`08efdbd`)

### mmdet 계열 (3종)

#### 5. Cascade R-CNN (mask head 포함 — cascade_mask_rcnn config)
- **논문**: Cai & Vasconcelos 2018
- **config**: `cascade-mask-rcnn_r50_fpn_1x_coco.py`
- **pretrained (수정 후)**: `https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth`
- **학습 설정**: AdamW lr=1e-4, bs=4
- **이전 이슈**: `cfg.load_from=None`이라 pretrained 미로드 → segm_AP=0. 수정됨.

#### 6. SOLOv2
- **논문**: Wang et al. 2020
- **config**: `solov2_r50_fpn_1x_coco.py`
- **pretrained (수정 후)**: `https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_1x_coco/solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth`
- **학습 설정**: AdamW lr=1e-4, bs=4
- **특이사항**: Mask-only 모델 (bbox 없음). evaluator에서 `metric=["segm"]`만 사용.

#### 7. RTMDet-Ins
- **논문**: Lyu et al. 2022
- **config**: `rtmdet-ins_s_8xb32-300e_coco.py`
- **pretrained (수정 후)**: `https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_s_8xb32-300e_coco/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth`
- **학습 설정**: AdamW lr=1e-4, bs=4

## ⚙️ 하이퍼파라미터 테이블 (cond4_8x 5모델 비교)

| 항목 | Mask/Cascade MRCNN | MaskDINO/Mask2Former | Cascade R-CNN/SOLOv2/RTMDet (mmdet) |
|------|:---:|:---:|:---:|
| optimizer | SGD | AdamW | AdamW |
| lr | 0.0015 | 1e-4 | 1e-4 |
| batch_size | 12 | 4 | 4 |
| max_epochs | 200 | 200 | 200 |
| patience | 10 | 10 | 10 |
| eval_period | 5 epoch | 5 epoch | 5 epoch |
| pretrained | COCO | ImageNet R-50 | COCO |

→ **약간의 비통일** 있음. 논문 Limitation 또는 §4 "각 프레임워크 표준 설정 사용" 명시.

### 공통 설정
- input_min_size: (640, 672, 704, 736, 768, 800)
- input_max_size: 1333
- seed: 42 (기본)
- **hardware**: A100 80GB (usw) / V100 32GB (ahnbi3)

## 🔍 mmdet 수정 히스토리 (중요)

### 이전 실패
- `cfg.load_from=None` → pretrained 미로드
- lr=0.0015 + AdamW → 학습 발산
- batch_size=12 + single GPU → 메모리 부하

### 수정 (commit 964d81c, 08efdbd)
- `training/config.py`: mmdet 3종에 `weights` + `hyperparams` 추가
- `training/adapters/mmdet_adapter.py`: `cfg.load_from = model_info["weights"]`
- 결과: COCO pretrained 로드됨, AdamW lr=1e-4 + bs=4로 안정

### Mask2Former import 수정
- 이전: `importlib.spec_from_file_location` (relative import 실패)
- 수정: `from mask2former.config import ...` + `import mask2former.modeling`
- data 모듈 건너뛰어 데이터셋 중복 등록 방지

## 📦 환경 설정 요약

### usw (기존 작업 환경)
```bash
conda activate jjh
python -c "import torch, mmdet, detectron2; print(torch.__version__)"
# → 2.5.1
```

### ahnbi3 (논문 작업 환경)
```bash
conda activate jjh
python -c "import torch, mmdet, mmcv, numpy; print('torch:', torch.__version__, 'numpy:', numpy.__version__)"
# → torch: 2.1.2+cu118, numpy: 1.x (2.x 아님)
```

### ahnbi3 추가 설치
- `pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118`
- `pip install "numpy<2"`
- `conda install nodejs -y` (Claude Code용)

## 🛠 Claude Code 설치 (ahnbi3)

```bash
conda activate jjh
conda install nodejs -y
npm install -g @anthropic-ai/claude-code --prefix ~/.npm-global
export PATH=~/.npm-global/bin:$PATH
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
claude --version
```

---

## 📝 다음 파일

- [07_AGENTS_GUIDE.md](07_AGENTS_GUIDE.md): Agent 활용법
