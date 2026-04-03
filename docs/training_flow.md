# 학습 전체 흐름 — 실제 코드 추적

이 문서는 아래 명령어 하나가 실행될 때 내부적으로 무슨 일이 벌어지는지,
**실제 코드의 어느 파일 몇 번째 줄**에서 무엇을 하는지 전부 추적한다.

```bash
python -m training.train --category Cable --experiment exp1 --condition genai_50 --model mask_rcnn
```

---

## 전체 흐름 요약

```
[사용자]  CLI 명령어 입력
    ↓
[train.py]  인자 파싱 → 하이퍼파라미터 구성 → run_single() 호출
    ↓
[data_pipeline.py]  COCO json 읽기 → 데이터 병합 → detectron2에 데이터 등록
    ↓
[detectron2_adapter.py]  yaml 원본 읽기 → config 덮어쓰기 → Trainer 생성 → 학습 실행
    ↓
[detectron2 라이브러리]  모델 생성 → 데이터 로딩 → forward/backward → 평가
    ↓
[train.py]  결과 저장 → results.json
```

---

## STEP 0: 시작점 — yaml 원본은 이렇게 생겼다

detectron2 패키지 안에 있는 yaml 파일. 우리가 만든 게 아니라 **detectron2를 설치하면 따라오는 것**이다.

경로: `~/.conda/envs/jjh/lib/python3.11/site-packages/detectron2/model_zoo/configs/`

### Base-RCNN-FPN.yaml (부모 파일 — 모든 R-CNN 계열이 상속)

```yaml
MODEL:
  META_ARCHITECTURE: "GeneralizedRCNN"              # 모델 전체 클래스명
  BACKBONE:
    NAME: "build_resnet_fpn_backbone"                # ResNet + FPN backbone
  RESNETS:
    OUT_FEATURES: ["res2", "res3", "res4", "res5"]   # ResNet 출력 4단계
  FPN:
    IN_FEATURES: ["res2", "res3", "res4", "res5"]    # FPN 입력
  ANCHOR_GENERATOR:
    SIZES: [[32], [64], [128], [256], [512]]          # RPN anchor 크기
    ASPECT_RATIOS: [[0.5, 1.0, 2.0]]                 # anchor 가로세로 비율
  RPN:
    IN_FEATURES: ["p2", "p3", "p4", "p5", "p6"]      # FPN 출력 5단계
    POST_NMS_TOPK_TRAIN: 1000                         # NMS 후 proposal 수
  ROI_HEADS:
    NAME: "StandardROIHeads"                          # ROI Head 종류
    IN_FEATURES: ["p2", "p3", "p4", "p5"]
  ROI_BOX_HEAD:
    NAME: "FastRCNNConvFCHead"                        # bbox head
    NUM_FC: 2                                         # FC layer 2개
    POOLER_RESOLUTION: 7                              # ROI Align 7x7
  ROI_MASK_HEAD:
    NAME: "MaskRCNNConvUpsampleHead"                  # mask head
    NUM_CONV: 4                                       # Conv layer 4개
    POOLER_RESOLUTION: 14                             # ROI Align 14x14
DATASETS:
  TRAIN: ("coco_2017_train",)                         # COCO 학습 데이터
  TEST: ("coco_2017_val",)                            # COCO 검증 데이터
SOLVER:
  IMS_PER_BATCH: 16                                   # batch size 16
  BASE_LR: 0.02                                       # learning rate
  STEPS: (60000, 80000)                               # LR decay 시점
  MAX_ITER: 90000                                     # 총 iteration
INPUT:
  MIN_SIZE_TRAIN: (640, 672, 704, 736, 768, 800)      # 학습 이미지 리사이즈
```

### mask_rcnn_R_50_FPN_3x.yaml (자식 파일 — Mask R-CNN 전용)

```yaml
_BASE_: "../Base-RCNN-FPN.yaml"                       # 위 파일 전부 상속
MODEL:
  WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-50.pkl"  # 사전학습 가중치
  MASK_ON: True                                       # mask head 활성화
  RESNETS:
    DEPTH: 50                                         # ResNet-50
SOLVER:
  STEPS: (210000, 250000)                             # LR decay 시점 (3x 스케줄)
  MAX_ITER: 270000                                    # 3x 스케줄
```

### cascade_mask_rcnn_R_50_FPN_3x.yaml (Cascade 전용 — 다른 점만)

```yaml
_BASE_: "../Base-RCNN-FPN.yaml"
MODEL:
  ROI_HEADS:
    NAME: "CascadeROIHeads"                           # ← 여기만 다름 (3-stage)
  ROI_BOX_HEAD:
    CLS_AGNOSTIC_BBOX_REG: True                       # 클래스 무관 bbox
  RPN:
    POST_NMS_TOPK_TRAIN: 2000                         # proposal 더 많이
```

**이 yaml 파일들은 절대 수정하지 않는다. 읽기만 한다.**

---

## STEP 1: train.py — CLI 진입 + 하이퍼파라미터 구성

### train.py line 164~190: 인자 파싱

```python
def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument('--category', ...)     # Cable
    parser.add_argument('--experiment', ...)    # exp1
    parser.add_argument('--condition', ...)     # genai_50
    parser.add_argument('--model', ...)         # mask_rcnn
    parser.add_argument('--max-epochs', ...)    # None (기본값 사용)
    parser.add_argument('--lr', ...)            # None (기본값 사용)
    parser.add_argument('--batch-size', ...)    # None (기본값 사용)
    parser.add_argument('--patience', ...)      # None (기본값 사용)
    args = parser.parse_args()
```

### train.py line 193: 하이퍼파라미터 구성

```python
hyperparams = dict(DEFAULT_HYPERPARAMS)
```

여기서 `DEFAULT_HYPERPARAMS`는 `config.py` line 171~185에서 온다:

```python
# config.py
DEFAULT_HYPERPARAMS = {
    "max_epochs": 300,
    "lr": 1e-4,
    "batch_size": 2,
    "seed": 42,
    "warmup_epochs": 5,
    "eval_period_epochs": 5,
    "checkpoint_period_epochs": 10,
    "input_min_size": (480, 512, 544, 576, 608, 640),
    "input_max_size": 800,
    "early_stopping_patience": 15,
    "early_stopping_metric": "segm/AP",
}
```

CLI에서 `--lr 5e-5`처럼 오버라이드하면 여기서 덮어씌운다 (train.py line 194~205).

### train.py line 207~215: 실행 범위 결정

```python
categories = ["Cable"]
conditions = ["genai_50"]
models = ["mask_rcnn"]
```

### train.py line 238: run_single() 호출

```python
result = run_single(
    category="Cable",
    experiment="exp1",
    condition="genai_50",
    model_name="mask_rcnn",
    hyperparams=hyperparams,  # {max_epochs:300, lr:1e-4, batch_size:2, ...}
)
```

---

## STEP 2: run_single() — 데이터 병합

### train.py line 77: prepare_dataset() 호출

```python
merged_dir = prepare_dataset("exp1", "genai_50", "Cable", seed=42)
```

### data_pipeline.py line 204~212: 실험 조건 읽기

```python
# config.py의 EXPERIMENTS에서 조건 가져옴
params = EXPERIMENTS["exp1"]["conditions"]["genai_50"]
# → {"n_original": -1, "n_genai_per_class": 50, "n_traditional": 0}

out_dir = get_merged_dir("exp1", "genai_50", "Cable")
# → results/merged_datasets/exp1/genai_50/Cable/
```

### data_pipeline.py line 237~250: 원본 데이터 로드

```python
n_original = -1            # 전체 사용
n_genai_per_class = 50     # 클래스당 50장
n_traditional = 0          # 전통증강 안 씀

# 원본 annotation 읽기
orig_data = load_coco("data/Cable/train/annotations.json")
# → {"images": [{id:1, file_name:"img1.jpg", ...}, ...],
#    "annotations": [{image_id:1, category_id:1, bbox:[...], segmentation:[...]}, ...]}

# Cable: thunderbolt(id=1)만 필터링, id를 0으로 리매핑
orig_data = filter_coco_by_category(orig_data, 1, {1: 0})

# n_original=-1 → 전체 사용
orig_imgs, orig_anns = _sample_images(orig_data, "data/Cable/train/images", n=-1, seed=42)
# → 26장 전체
```

### data_pipeline.py line 254~261: GenAI 데이터 로드 (클래스별 균형 샘플링)

```python
genai_data = load_coco("data_augmented/Cable/gen_ai/annotations.json")

genai_imgs, genai_anns = _sample_images_per_class(
    genai_data,
    "data_augmented/Cable/gen_ai/images",
    n_per_class=50,    # thunderbolt 클래스에서 50장
    seed=43
)
# → 50장
```

### data_pipeline.py line 274~281: 병합

```python
# 카테고리를 0-indexed로 통일
train_categories = [{"id": 0, "name": "thunderbolt", "supercategory": "defect"}]

# 원본 26장 + GenAI 50장 = 76장
# 이미지 파일을 results/merged_datasets/.../images/로 복사
# annotation의 image_id, annotation_id를 1부터 재부여
n_imgs, n_anns = _merge_sources(sources, out_dir, train_categories)
# → results/merged_datasets/exp1/genai_50/Cable/
#     ├── images/        (76장)
#     └── annotations.json  (COCO format, id 재부여됨)
```

---

## STEP 3: run_single() — 어댑터 생성

### train.py line 39~51: 모델에 맞는 어댑터 선택

```python
def create_adapter(model_name, category):
    model_info = get_model_info("mask_rcnn")
    # → {"framework": "detectron2",
    #    "display_name": "Mask R-CNN",
    #    "config": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    #    "weights": "detectron2://...model_final_f10217.pkl"}

    cat_info = get_category_info("Cable")
    # → {"classes": ["thunderbolt"], "num_classes": 1, ...}

    # framework가 "detectron2"이므로:
    return Detectron2Adapter(
        model_name="mask_rcnn",
        num_classes=1,
        thing_classes=["thunderbolt"],
        model_info=model_info,
    )
```

---

## STEP 4: adapter.setup() — yaml 읽기 + config 덮어쓰기

이 단계가 **yaml 원본을 메모리에서 수정하는 핵심**이다.

### detectron2_adapter.py line 220: detectron2 import

```python
d2 = self._import_detectron2()
# → get_cfg, DefaultTrainer, model_zoo 등 detectron2 함수들을 가져옴
```

### detectron2_adapter.py line 228~233: 데이터를 detectron2에 등록

```python
self.train_dataset_name = "train_mask_rcnn_140234567890"

register_for_detectron2(
    "train_mask_rcnn_140234567890",
    "results/merged_datasets/exp1/genai_50/Cable/images/",   # 이미지 경로
    "results/merged_datasets/exp1/genai_50/Cable/annotations.json",  # annotation 경로
    ["thunderbolt"],  # 클래스명
)
```

이 함수 안에서 (data_pipeline.py line 288~327):

```python
def register_for_detectron2(name, images_dir, ann_path, thing_classes):

    def get_dicts():
        # COCO json을 읽어서 detectron2가 원하는 dict 리스트로 변환
        data = load_coco(ann_path)  # annotations.json 읽기
        dataset_dicts = []
        for img in data["images"]:
            record = {
                "file_name": str(images_dir / img["file_name"]),
                # → "/풀경로/.../images/img001.jpg"
                "image_id": img["id"],       # 1
                "height": img["height"],     # 480
                "width": img["width"],       # 640
            }
            anns = [a for a in data["annotations"] if a["image_id"] == img["id"]]
            objs = []
            for ann in anns:
                objs.append({
                    "bbox": ann["bbox"],                    # [x, y, w, h]
                    "bbox_mode": BoxMode.XYWH_ABS,          # COCO format 명시
                    "segmentation": ann["segmentation"],     # [[x1,y1,x2,y2,...]] polygon
                    "category_id": ann["category_id"],       # 0 (thunderbolt)
                    "iscrowd": ann.get("iscrowd", 0),
                })
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts

    # detectron2의 전역 카탈로그에 등록
    DatasetCatalog.register(name, get_dicts)
    # → detectron2가 "train_mask_rcnn_140234567890" 이름으로 데이터를 찾을 수 있게 됨

    MetadataCatalog.get(name).set(
        thing_classes=["thunderbolt"],  # 클래스명 메타데이터
        evaluator_type="coco",          # 평가 방식
    )
```

### detectron2_adapter.py line 236~241: 모델별 config 분기

```python
if self.model_name == "maskdino":
    cfg = self._setup_maskdino(d2)
elif self.model_name == "mask2former":
    cfg = self._setup_mask2former(d2)
else:
    cfg = self._setup_standard(d2)    # ← mask_rcnn, cascade_mask_rcnn은 여기
```

### detectron2_adapter.py line 208~214: _setup_standard() — yaml 읽기

```python
def _setup_standard(self, d2):
    # ① 빈 config 객체 생성
    cfg = d2['get_cfg']()

    # ② yaml 파일을 메모리로 읽어옴
    cfg.merge_from_file(
        d2['model_zoo'].get_config_file(self.model_info["config"])
        # → "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        # 이 yaml이 _BASE_로 Base-RCNN-FPN.yaml도 함께 로드함
    )
    # 이 시점에서 cfg 상태 (yaml 원본 그대로):
    #   cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    #   cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    #   cfg.SOLVER.IMS_PER_BATCH = 16
    #   cfg.SOLVER.BASE_LR = 0.02
    #   cfg.SOLVER.MAX_ITER = 270000
    #   cfg.DATASETS.TRAIN = ("coco_2017_train",)

    # ③ 사전학습 가중치 URL 설정
    cfg.MODEL.WEIGHTS = self.model_info["weights"]
    # → "detectron2://COCO-InstanceSegmentation/.../model_final_f10217.pkl"
    # (COCO 데이터로 이미 학습된 모델 가중치. 자동 다운로드됨)

    # ④ 클래스 수 변경
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
    # 80 → 1 (thunderbolt만)

    return cfg
```

**만약 `--model cascade_mask_rcnn`이었다면:**
같은 `_setup_standard()`을 타지만, yaml 파일이 다르므로:
- `cfg.MODEL.ROI_HEADS.NAME = "CascadeROIHeads"` (3-stage)
- 나머지 흐름은 동일

### detectron2_adapter.py line 244~245: 데이터셋 이름을 config에 등록

```python
cfg.DATASETS.TRAIN = (self.train_dataset_name,)
# ("coco_2017_train",) → ("train_mask_rcnn_140234567890",)
# detectron2가 학습할 때 이 이름으로 데이터를 찾음

cfg.DATASETS.TEST = (self.val_dataset_name,)
# ("coco_2017_val",) → ("val_mask_rcnn_140234567890",)
```

### detectron2_adapter.py line 247~249: batch size, lr 변경

```python
batch_size = hyperparams.get("batch_size", 2)
cfg.SOLVER.IMS_PER_BATCH = batch_size
# 16 → 2

cfg.SOLVER.BASE_LR = hyperparams.get("lr", 1e-4)
# 0.02 → 0.0001
```

### detectron2_adapter.py line 251~259: 에폭 → 이터레이션 변환

detectron2는 내부적으로 **iteration 단위**로 동작한다. 우리는 epoch으로 지정했으므로 변환이 필요하다.

```python
# 학습 데이터 개수 확인
train_data = load_coco(train_ann_path)
n_train_images = len(train_data["images"])
# → 76장 (원본 26 + GenAI 50)

# 1 에폭 = 몇 iteration?
self.iters_per_epoch = max(1, math.ceil(n_train_images / batch_size))
# math.ceil(76 / 2) = 38 iters/epoch
# 즉, 76장을 2장씩 처리하면 38번 반복해야 1 에폭

# 총 iteration
max_epochs = hyperparams.get("max_epochs", 300)
max_iter = max_epochs * self.iters_per_epoch
# 300 × 38 = 11400
cfg.SOLVER.MAX_ITER = max_iter
# 270000 → 11400
```

### detectron2_adapter.py line 261~269: warmup, eval, checkpoint 주기

```python
# warmup: 학습 초반에 lr을 천천히 올리는 구간
warmup_epochs = 5
cfg.SOLVER.WARMUP_ITERS = 5 * 38  # = 190 iters

# 평가 주기: 5 에폭마다
eval_period = 5 * 38  # = 190 iters
cfg.TEST.EVAL_PERIOD = 190

# 체크포인트 저장 주기: 10 에폭마다
cfg.SOLVER.CHECKPOINT_PERIOD = 10 * 38  # = 380 iters
```

### detectron2_adapter.py line 271~272: LR decay

```python
cfg.SOLVER.STEPS = (int(11400 * 0.7), int(11400 * 0.9))
# (210000, 250000) → (7980, 10260)
# 학습의 70% 지점(7980 iter)에서 lr을 1/10로,
# 90% 지점(10260 iter)에서 다시 1/10로 줄임
```

### detectron2_adapter.py line 274~282: Early stopping hook 생성

```python
self.early_stopping_hook = EarlyStoppingHook(
    eval_period=190,         # 190 iter마다 (= 5 에폭마다) 체크
    patience=15,             # 15번 연속 개선 없으면 중단 (= 75 에폭)
    metric_name="segm/AP",   # instance segmentation mAP 기준
    iters_per_epoch=38,
)
```

### detectron2_adapter.py line 289~298: 이미지 크기, 출력 경로, 시드

```python
cfg.INPUT.MASK_FORMAT = "polygon"
cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640)
cfg.INPUT.MAX_SIZE_TRAIN = 800
cfg.INPUT.MIN_SIZE_TEST = 640
cfg.INPUT.MAX_SIZE_TEST = 800

cfg.OUTPUT_DIR = "results/training/exp1/genai_50/Cable/mask_rcnn"
cfg.SEED = 42
```

### detectron2_adapter.py line 304~306: 최종 config 저장

```python
with open(output_dir / "config.yaml", "w") as f:
    f.write(cfg.dump())
# → results/training/exp1/genai_50/Cable/mask_rcnn/config.yaml
# 이 파일을 열면 실제 사용된 모든 설정값을 확인할 수 있음
```

### setup() 완료 시점의 cfg 상태 (yaml 원본 → 최종)

| 항목 | yaml 원본 | 최종값 | 바꾼 위치 (line) |
|------|-----------|--------|------------------|
| `MODEL.ROI_HEADS.NUM_CLASSES` | 80 | **1** | 213 |
| `MODEL.WEIGHTS` | ImageNet R-50 | **COCO pretrained** | 212 |
| `MODEL.ROI_HEADS.NAME` | StandardROIHeads | StandardROIHeads (변경 없음) | - |
| `MODEL.BACKBONE.NAME` | build_resnet_fpn_backbone | (변경 없음) | - |
| `MODEL.RESNETS.DEPTH` | 50 | (변경 없음) | - |
| `DATASETS.TRAIN` | coco_2017_train | **train_mask_rcnn_...** | 244 |
| `DATASETS.TEST` | coco_2017_val | **val_mask_rcnn_...** | 245 |
| `SOLVER.IMS_PER_BATCH` | 16 | **2** | 248 |
| `SOLVER.BASE_LR` | 0.02 | **0.0001** | 249 |
| `SOLVER.MAX_ITER` | 270000 | **11400** | 259 |
| `SOLVER.WARMUP_ITERS` | (없음) | **190** | 262 |
| `SOLVER.STEPS` | (210000, 250000) | **(7980, 10260)** | 272 |
| `SOLVER.CHECKPOINT_PERIOD` | (없음) | **380** | 269 |
| `TEST.EVAL_PERIOD` | (없음) | **190** | 266 |
| `INPUT.MIN_SIZE_TRAIN` | (640~800) | **(480~640)** | 292 |
| `OUTPUT_DIR` | (없음) | **results/training/...** | 297 |
| `SEED` | (없음) | **42** | 298 |

---

## STEP 5: adapter.train() — 학습 실행

### detectron2_adapter.py line 355~362: Trainer 생성

```python
TrainerClass = self._get_trainer_class()
# → StandardTrainer (Mask R-CNN용)
# StandardTrainer는 DefaultTrainer를 상속하며,
# build_evaluator()만 COCOEvaluator로 오버라이드 (line 346~351)

trainer = TrainerClass(self.cfg)
# DefaultTrainer(cfg) 내부에서 일어나는 일 (detectron2 라이브러리):
#   1. build_model(cfg)
#      → cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
#      → cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
#      → cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
#      → 이 문자열들을 보고 해당 클래스를 찾아서 모델 객체 생성
#      → ResNet50 + FPN + RPN + StandardROIHeads(BoxHead + MaskHead)
#
#   2. build_optimizer(cfg)
#      → cfg.SOLVER.BASE_LR = 0.0001
#      → SGD or AdamW 옵티마이저 생성
#
#   3. build_train_loader(cfg)
#      → cfg.DATASETS.TRAIN = ("train_mask_rcnn_...",)
#      → DatasetCatalog에서 이 이름으로 등록된 get_dicts() 함수 호출
#      → 76장의 dict 리스트 반환
#      → batch_size=2로 DataLoader 생성

trainer.resume_or_load(resume=False)
# → cfg.MODEL.WEIGHTS의 COCO pretrained 가중치를 다운로드하여 모델에 로드
# → 마지막 레이어(NUM_CLASSES)가 80→1로 바뀌었으므로 해당 레이어만 랜덤 초기화
```

### detectron2_adapter.py line 364~372: Early stopping hook 등록

```python
self.early_stopping_hook.trainer = trainer    # hook에 trainer 참조 연결
trainer.register_hooks([self.early_stopping_hook])
# → trainer._hooks 리스트에 추가됨
# → 매 iteration 후 after_step()이 자동 호출됨
```

### detectron2_adapter.py line 374~376: GPU 메모리 추적 시작

```python
torch.cuda.reset_peak_memory_stats()
# → 이 시점부터 GPU 메모리 사용량 최대치를 추적
```

### detectron2_adapter.py line 388: 학습 루프 시작

```python
trainer.train()
```

**이 한 줄 안에서 detectron2 라이브러리가 하는 일:**

```
for iter in range(0, 11400):    # MAX_ITER = 11400

    # 1. 데이터 로딩
    data = next(data_loader)    # batch_size=2, 이미지 2장 + annotation

    # 2. Forward pass
    loss_dict = model(data)     # GeneralizedRCNN.forward()
    # loss_dict = {
    #   "loss_rpn_cls": 0.12,    (RPN: 물체 있나 없나)
    #   "loss_rpn_loc": 0.08,    (RPN: anchor bbox 보정)
    #   "loss_cls": 0.15,        (ROI: 클래스 분류)
    #   "loss_box_reg": 0.10,    (ROI: bbox 보정)
    #   "loss_mask": 0.20,       (ROI: 마스크 예측)
    # }

    # 3. Loss 합산 + Backward
    total_loss = sum(loss_dict.values())    # 0.65
    total_loss.backward()

    # 4. Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # 5. LR scheduler step
    # iter < 190: warmup (lr을 0에서 0.0001까지 서서히 올림)
    # iter 7980: lr을 0.0001 → 0.00001로 줄임
    # iter 10260: lr을 0.00001 → 0.000001로 줄임

    # 6. Hook 실행 (매 iteration)
    for hook in hooks:
        hook.after_step()

    # 7. 평가 (매 190 iter = 5 에폭마다)
    if (iter + 1) % 190 == 0:
        # val 데이터 131장으로 평가
        # → segm/AP (mAP) 계산
        # → EarlyStoppingHook.after_step()에서 체크:
        #    - 개선됨 → best 갱신, patience 리셋
        #    - 안 됨 → patience 카운트 +1
        #    - patience 15회 초과 → trainer.max_iter = 현재 iter → 학습 중단
```

### EarlyStoppingHook 동작 (detectron2_adapter.py line 43~81)

```python
def after_step(self):
    next_iter = self.trainer.iter + 1

    # 190 iter마다만 실행 (5 에폭마다)
    if next_iter % self.eval_period != 0:
        return

    # detectron2의 EventStorage에서 최신 segm/AP 읽기
    metric_val = storage.latest().get("segm/AP", None)

    if metric_val > self.best_metric:
        # 개선됨!
        self.best_metric = metric_val
        self.num_bad_evals = 0
        # 출력: "[EarlyStopping] New best segm/AP=0.4512 at epoch 25.0"
    else:
        # 개선 안 됨
        self.num_bad_evals += 1
        # 출력: "[EarlyStopping] No improvement 3/15"

    if self.num_bad_evals >= 15:
        # 75 에폭 동안 개선 없음 → 중단
        self.trainer.max_iter = next_iter
        # 출력: "[EarlyStopping] STOP at epoch 85.0"
```

### detectron2_adapter.py line 390~427: 학습 완료 후 메트릭 수집

```python
# GPU 피크 메모리
peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
# → 예: 12345.6 MB

# 최종 결과
metrics = {
    "model": "mask_rcnn",
    "status": "completed",
    "total_iters": 3230,           # early stopping으로 여기서 멈춤
    "total_epochs": 85.0,          # 3230 / 38 = 85 에폭
    "iters_per_epoch": 38,
    "peak_memory_mb": 12345.6,
    "early_stopped": True,
    "early_stop_epoch": 85.0,
    "best_metric_value": 0.4512,   # 최고 segm/AP
    "best_metric_iter": 950,       # 25 에폭 시점
}
```

---

## STEP 6: adapter.evaluate() — 평가

### train.py line 124~126: 학습 후 자동 평가

```python
eval_results = adapter.evaluate()
```

### detectron2_adapter.py line 429~478: 평가 실행

```python
# 학습된 모델 로드
model = build_model(self.cfg)           # 모델 구조 생성
model.eval()                             # 평가 모드
checkpointer = DetectionCheckpointer(model)
checkpointer.load("results/.../model_final.pth")  # 학습된 가중치 로드

# 평가기 생성
evaluator = COCOEvaluator(
    self.val_dataset_name,               # val 데이터 131장
    tasks=("bbox", "segm"),              # bbox + mask 둘 다 평가
)

# val 데이터로 추론 + 평가
val_loader = build_detection_test_loader(self.cfg, self.val_dataset_name)
results = inference_on_dataset(model, val_loader, evaluator)
# → 131장 각각에 대해:
#    model(image) → predictions (bbox + class + score + mask)
#    predictions vs ground_truth 비교 → COCO mAP 계산

# 결과 정리
flat_results = {
    "bbox_AP": 48.789,        # bbox mAP
    "bbox_AP50": 75.123,      # bbox AP@IoU=0.50
    "bbox_AP75": 52.456,      # bbox AP@IoU=0.75
    "segm_AP": 45.123,        # mask mAP ← 핵심 지표
    "segm_AP50": 72.456,
    "segm_AP75": 48.901,
}

# 저장
# → results/training/exp1/genai_50/Cable/mask_rcnn/eval_results/results.json
```

---

## STEP 7: 결과 저장

### train.py line 108~121: 결과 수집

```python
result = {
    "category": "Cable",
    "experiment": "exp1",
    "condition": "genai_50",
    "model": "mask_rcnn",
    "output_dir": "results/training/exp1/genai_50/Cable/mask_rcnn",
    "train_time_sec": 1234.5,
    "peak_memory_mb": 12345.6,
    "early_stopped": True,
    "early_stop_epoch": 85.0,
    "total_epochs": 85.0,
    "eval": {
        "segm_AP": 45.123,
        "segm_AP50": 72.456,
        "bbox_AP": 48.789,
        ...
    }
}
```

### train.py line 134~161: 마스터 결과 파일에 저장

```python
# → results/evaluation/results.json에 추가
# 같은 (category, experiment, condition, model) 조합이 이미 있으면 덮어쓰기
```

---

## 최종 output 디렉토리 구조

```
results/training/exp1/genai_50/Cable/mask_rcnn/
├── config.yaml            ← 실제 사용된 detectron2 config (수정된 값 전부 포함)
├── model_final.pth        ← 학습된 모델 가중치
├── model_0000379.pth      ← 체크포인트 (380 iter = 10 에폭마다)
├── metrics.json           ← 학습 중 loss, lr 기록 (매 iteration)
├── eval_results/
│   └── results.json       ← 평가 결과 (mAP 등)
└── inference/
    └── coco_instances_results.json  ← 추론 결과 (각 이미지별 prediction)
```

---

## 실험 조건을 바꾸고 싶으면?

### 데이터 조합 변경

`training/config.py`의 `EXPERIMENTS` 수정:

```python
EXPERIMENTS = {
    "exp1": {
        "conditions": {
            "genai_50": {"n_original": -1, "n_genai_per_class": 50, "n_traditional": 0},
            #                                ↑ 이 숫자를 바꾸면 GenAI 데이터 양이 바뀜
        },
    },
}
```

### 하이퍼파라미터 변경

방법 1 — `config.py`의 `DEFAULT_HYPERPARAMS` 수정:
```python
DEFAULT_HYPERPARAMS = {
    "lr": 5e-5,          # ← 여기서 바꾸면 모든 실험에 적용
    "batch_size": 4,
}
```

방법 2 — CLI에서 오버라이드 (해당 실행에만 적용):
```bash
python -m training.train ... --lr 5e-5 --batch-size 4
```

### 모델 변경

CLI `--model`만 바꾸면 됨:
```bash
--model mask_rcnn           # ROI_HEADS.NAME = "StandardROIHeads" (1-stage)
--model cascade_mask_rcnn   # ROI_HEADS.NAME = "CascadeROIHeads" (3-stage)
```

이 둘의 차이는 **yaml 파일이 다른 것**이고, `_setup_standard()`에서 읽는 yaml만 달라진다.
나머지 흐름(데이터 등록, config 덮어쓰기, Trainer 실행)은 완전히 동일하다.
