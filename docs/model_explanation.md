# Detectron2 & 모델 설명

## 1. Detectron2란?

Facebook(Meta)이 만든 **객체 탐지/세그멘테이션 라이브러리**이다. PyTorch 기반이며, pip으로 설치한다.

```bash
pip install detectron2
```

라이브러리 안에 **모델 구조, 학습 루프, 평가, 데이터 로딩**이 전부 구현되어 있다.
사용자는 **config(설정)만 바꿔서** 원하는 모델을 돌릴 수 있다.

### 핵심 구성요소

```
detectron2/
├── modeling/          ← 모델 구현 (Mask R-CNN, Cascade 등)
│   ├── meta_arch/     ← 모델 전체 구조 (forward 포함)
│   ├── backbone/      ← ResNet, FPN
│   ├── proposal_generator/  ← RPN
│   └── roi_heads/     ← ROI Head (bbox + mask)
├── engine/            ← DefaultTrainer (학습 루프)
├── data/              ← 데이터 로딩, augmentation
├── evaluation/        ← COCO 평가 (mAP 계산)
├── config/            ← CfgNode (config 관리)
└── model_zoo/         ← 사전학습 모델 + yaml config 모음
```

### 작동 방식

```
yaml config 작성 (또는 수정)
       ↓
DefaultTrainer(cfg)   ← config를 넘기면
       ↓
내부에서 자동으로:
  build_model(cfg)         → config 보고 모델 생성
  build_optimizer(cfg)     → config 보고 옵티마이저 생성
  build_train_loader(cfg)  → config 보고 데이터 로딩
       ↓
trainer.train()       ← 학습 시작
```

**사용자가 모델 코드를 직접 작성할 필요 없다.** config만 수정하면 된다.

---

## 2. Config 시스템

detectron2는 **yaml 파일**로 모델 설정을 관리한다.

### Mask R-CNN의 실제 config (Base-RCNN-FPN.yaml)

```yaml
MODEL:
  META_ARCHITECTURE: "GeneralizedRCNN"        # 모델 전체 구조
  BACKBONE:
    NAME: "build_resnet_fpn_backbone"          # ResNet + FPN
  RESNETS:
    DEPTH: 50                                  # ResNet-50
    OUT_FEATURES: ["res2", "res3", "res4", "res5"]
  FPN:
    IN_FEATURES: ["res2", "res3", "res4", "res5"]
  ANCHOR_GENERATOR:
    SIZES: [[32], [64], [128], [256], [512]]
    ASPECT_RATIOS: [[0.5, 1.0, 2.0]]
  RPN:
    IN_FEATURES: ["p2", "p3", "p4", "p5", "p6"]
  ROI_HEADS:
    NAME: "StandardROIHeads"                   # Mask R-CNN용
    IN_FEATURES: ["p2", "p3", "p4", "p5"]
  ROI_BOX_HEAD:
    NAME: "FastRCNNConvFCHead"
    NUM_FC: 2
    POOLER_RESOLUTION: 7
  ROI_MASK_HEAD:
    NAME: "MaskRCNNConvUpsampleHead"
    NUM_CONV: 4
    POOLER_RESOLUTION: 14
  MASK_ON: True                                # 마스크 예측 활성화
  WEIGHTS: "detectron2://...R-50.pkl"          # COCO 사전학습 가중치

SOLVER:
  IMS_PER_BATCH: 16                            # batch size
  BASE_LR: 0.02                                # learning rate
  MAX_ITER: 270000

INPUT:
  MIN_SIZE_TRAIN: (640, 672, 704, 736, 768, 800)
```

detectron2는 이 yaml을 읽고, `MODEL.META_ARCHITECTURE`가 `"GeneralizedRCNN"`이니까 해당 클래스를 찾아서 모델을 만든다. `BACKBONE.NAME`이 `"build_resnet_fpn_backbone"`이니까 ResNet50 + FPN backbone을 만든다. 전부 **문자열 → 클래스 매핑**으로 동작한다.

### Cascade Mask R-CNN의 config 차이

```yaml
# Mask R-CNN은:
ROI_HEADS:
  NAME: "StandardROIHeads"    # 1-stage ROI Head

# Cascade Mask R-CNN은:
ROI_HEADS:
  NAME: "CascadeROIHeads"     # 3-stage Cascade ROI Head
ROI_BOX_HEAD:
  CLS_AGNOSTIC_BBOX_REG: True
```

**ROI_HEADS.NAME 하나만 바꾸면** Mask R-CNN이 Cascade Mask R-CNN이 된다.

---

## 3. Mask R-CNN 구조

이미지에서 **객체의 위치(bbox) + 클래스 + 마스크(픽셀 단위 윤곽)**를 예측하는 모델이다.

### 전체 파이프라인

```
입력 이미지 (예: 640x480)
       ↓
┌─────────────────────────┐
│  Backbone (ResNet-50)    │  이미지 → 특성 맵(feature map) 추출
│  + FPN (Feature Pyramid) │  다양한 크기의 특성 맵 생성
└─────────────────────────┘
       ↓
  [P2, P3, P4, P5, P6]   ← 5단계 특성 맵 (작은 물체 ~ 큰 물체)
       ↓
┌─────────────────────────┐
│  RPN (Region Proposal)   │  "여기에 물체가 있을 것 같다" 후보 영역 ~1000개 제안
└─────────────────────────┘
       ↓
  Proposals (후보 bbox ~1000개)
       ↓
┌─────────────────────────┐
│  ROI Head                │
│  ├── Box Head            │  후보마다: 이게 뭔지(클래스) + 정확한 위치(bbox)
│  └── Mask Head           │  후보마다: 픽셀 단위 마스크 (28x28 → 원본 크기로 복원)
└─────────────────────────┘
       ↓
  최종 출력: [{bbox, class, score, mask}, ...]
```

### 각 단계 설명

#### (1) Backbone: ResNet-50

이미지를 **특성 맵(feature map)**으로 변환한다.

```
입력: [3, 640, 480]  (RGB 이미지)
  ↓ conv1 (7x7, stride 2)
  ↓ res2 (3개 블록) → 출력: [256, 160, 120]
  ↓ res3 (4개 블록) → 출력: [512, 80, 60]
  ↓ res4 (6개 블록) → 출력: [1024, 40, 30]
  ↓ res5 (3개 블록) → 출력: [2048, 20, 15]
```

각 stage가 점점 작지만 의미적으로 풍부한 특성 맵을 만든다.

#### (2) FPN (Feature Pyramid Network)

ResNet의 각 stage 출력을 **동일한 채널 수(256)**로 맞추고, 위→아래로 합쳐서 **다중 스케일 특성 맵**을 만든다.

```
res5 [2048, 20, 15] → 1x1 conv → P5 [256, 20, 15]
                                      ↓ upsample + 합침
res4 [1024, 40, 30] → 1x1 conv → P4 [256, 40, 30]
                                      ↓ upsample + 합침
res3 [512, 80, 60]  → 1x1 conv → P3 [256, 80, 60]
                                      ↓ upsample + 합침
res2 [256, 160, 120] → 1x1 conv → P2 [256, 160, 120]

P5 → max pool → P6 [256, 10, 8]   (추가 레벨)
```

이렇게 하면 **작은 물체(P2)부터 큰 물체(P5)**까지 다양한 크기를 잡을 수 있다.

#### (3) RPN (Region Proposal Network)

각 특성 맵의 모든 위치에서 **"여기에 물체가 있나?"**를 판단한다.

```
P2~P6의 각 위치에서:
  - anchor 생성 (크기: 32/64/128/256/512, 비율: 0.5/1.0/2.0)
  - 각 anchor마다: 물체 있음/없음 (2-class classification)
  - 각 anchor마다: bbox 보정값 (dx, dy, dw, dh)
  ↓
  NMS (Non-Maximum Suppression)로 겹치는 것 제거
  ↓
  상위 1000개 proposal 선택
```

#### (4) ROI Head — Box Head

각 proposal에 대해 **정확한 분류 + bbox 보정**을 한다.

```
proposal bbox [1000개]
  ↓
ROI Align: 특성 맵에서 해당 영역을 7x7로 잘라냄
  ↓
FC layer x 2 (1024 차원)
  ↓
├── cls_score: [num_classes + 1] (클래스 분류, +1은 배경)
└── bbox_pred: [num_classes x 4] (bbox 보정)
  ↓
NMS + score threshold로 최종 detection 선택
```

#### (5) ROI Head — Mask Head

Box Head에서 통과한 detection에 대해 **픽셀 단위 마스크**를 예측한다.

```
detection bbox [N개]
  ↓
ROI Align: 특성 맵에서 해당 영역을 14x14로 잘라냄
  ↓
Conv layer x 4 (256 채널)
  ↓
ConvTranspose (14x14 → 28x28 upsample)
  ↓
1x1 Conv → [num_classes, 28, 28] (클래스별 마스크)
  ↓
해당 클래스의 28x28 마스크를 원본 크기로 resize
```

### 실제 모델 구조 (detectron2 출력)

```
GeneralizedRCNN(
  (backbone): FPN(
    (bottom_up): ResNet(
      (stem): BasicStem(conv1: 3→64)
      (res2): 3x BottleneckBlock (64→256)
      (res3): 4x BottleneckBlock (128→512)
      (res4): 6x BottleneckBlock (256→1024)
      (res5): 3x BottleneckBlock (512→2048)
    )
    (fpn_lateral2~5): 1x1 Conv
    (fpn_output2~5): 3x3 Conv
  )
  (proposal_generator): RPN(
    (rpn_head): Conv + cls_logits + bbox_pred
  )
  (roi_heads): StandardROIHeads(
    (box_pooler): ROIPooler(7x7)
    (box_head): FastRCNNConvFCHead(fc1: 1024, fc2: 1024)
    (box_predictor): FastRCNNOutputLayers(cls + bbox)
    (mask_pooler): ROIPooler(14x14)
    (mask_head): MaskRCNNConvUpsampleHead(4x Conv256 + ConvTranspose)
  )
)
```

### Loss 함수

학습 시 4개의 loss를 합산한다:

```
Total Loss = loss_rpn_cls        (RPN: 물체 있나 없나)
           + loss_rpn_loc        (RPN: anchor bbox 보정)
           + loss_box_cls        (ROI: 클래스 분류)
           + loss_box_reg        (ROI: bbox 보정)
           + loss_mask           (ROI: 마스크 예측)
```

---

## 4. Cascade Mask R-CNN 구조

Mask R-CNN과 거의 같지만 **ROI Head를 3번 반복(cascade)**한다.

### Mask R-CNN vs Cascade Mask R-CNN

```
Mask R-CNN:
  proposals → [ROI Head (1회)] → detection + mask

Cascade Mask R-CNN:
  proposals → [ROI Head Stage 1] → 보정된 bbox
                    ↓
              [ROI Head Stage 2] → 더 보정된 bbox
                    ↓
              [ROI Head Stage 3] → 최종 detection + mask
```

### 왜 cascade가 좋은가?

- Stage 1: IoU threshold 0.5 (느슨하게 선별)
- Stage 2: IoU threshold 0.6 (좀 더 엄격)
- Stage 3: IoU threshold 0.7 (가장 엄격)

**점진적으로 정밀도를 높여간다.** bbox가 반복될수록 정확해지므로, 최종 mask도 더 정확해진다.

### config 차이 (이것만 다름)

```yaml
# Mask R-CNN
ROI_HEADS:
  NAME: "StandardROIHeads"     # 1-stage

# Cascade Mask R-CNN
ROI_HEADS:
  NAME: "CascadeROIHeads"      # 3-stage cascade
ROI_BOX_HEAD:
  CLS_AGNOSTIC_BBOX_REG: True  # 클래스 무관 bbox regression
```

---

## 5. 우리 프로젝트에서의 사용

### 파일 구조

```
training/
├── config.py                     ← 하이퍼파라미터 (lr, batch_size, epochs)
├── train.py                      ← CLI 진입점
├── data_pipeline.py              ← 데이터 병합 + detectron2 데이터 등록
└── adapters/
    └── detectron2_adapter.py     ← detectron2 config 생성 + Trainer 실행
```

### 실제 코드 흐름

```python
# 1. config.py에서 하이퍼파라미터 읽기
DEFAULT_HYPERPARAMS = {
    "max_epochs": 300,
    "lr": 1e-4,
    "batch_size": 2,
    ...
}

# 2. detectron2_adapter.py → setup()에서 config 생성
cfg = get_cfg()

# 2-1. detectron2 model_zoo에서 기본 config 로드 (yaml)
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
))

# 2-2. 사전학습 가중치 (COCO로 학습된 것)
cfg.MODEL.WEIGHTS = "detectron2://...model_final_f10217.pkl"

# 2-3. 우리 데이터에 맞게 수정
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1           # thunderbolt 1개
cfg.DATASETS.TRAIN = ("train_mask_rcnn_...",)  # 우리 데이터셋
cfg.DATASETS.TEST = ("val_mask_rcnn_...",)

# 2-4. 하이퍼파라미터 적용
cfg.SOLVER.IMS_PER_BATCH = 2                  # batch_size
cfg.SOLVER.BASE_LR = 1e-4                     # lr
cfg.SOLVER.MAX_ITER = 3900                    # 300 epochs x 13 iters/epoch

# 3. detectron2_adapter.py → train()에서 학습 실행
trainer = DefaultTrainer(cfg)     # config 넘기면 모델+옵티마이저+데이터 자동 구성
trainer.resume_or_load(resume=False)  # 사전학습 가중치 로드
trainer.train()                   # 학습 시작

# 4. 내부에서 일어나는 일 (detectron2 라이브러리)
# 매 iteration:
#   이미지 2장 로드 → augmentation → forward → loss 계산 → backward → optimizer.step()
# 매 5 epoch:
#   val 데이터로 mAP 평가 → early stopping 체크
```

### 사용자가 바꿀 수 있는 것

| 항목 | 어디서 | 예시 |
|------|--------|------|
| lr, batch_size, epochs | `config.py` 또는 CLI | `--lr 5e-5 --batch-size 4` |
| 모델 선택 | CLI `--model` | `mask_rcnn` 또는 `cascade_mask_rcnn` |
| 데이터 조합 | CLI `--experiment --condition` | `--experiment exp1 --condition genai_50` |
| backbone, head 등 모델 구조 | `detectron2_adapter.py`의 `setup()`에서 `cfg.MODEL.*` | 거의 안 바꿈 |

### 사용자가 바꿀 필요 없는 것

- 모델 코드 (detectron2 라이브러리 안에 있음)
- 학습 루프 (DefaultTrainer가 처리)
- 데이터 로딩 (data_pipeline.py가 자동 등록)
- 평가 (COCOEvaluator가 자동 계산)
