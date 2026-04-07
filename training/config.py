"""
통합 학습 환경 중앙 설정
- 경로, 카테고리, 모델, 실험 조건 정의
"""

from pathlib import Path

# ============================================================
# 경로 설정
# ============================================================
# 데이터(읽기 전용)는 jjh0709 원본 레포지토리를 공유 사용.
# 결과(쓰기)는 사용자별 작업 디렉토리로 분리 저장.
#   - 환경변수 VISION_RESULTS_ROOT 로 override 가능 (다중 사용자 병렬 실행 지원)
#   - 기본값: PROJECT_ROOT (jjh0709) → 단독 사용 시 기존 동작과 동일
import os

PROJECT_ROOT = Path('/home/jjh0709/gitrepo/VISION-Instance-Seg')              # 데이터/코드 원본(읽기)
RESULTS_ROOT = Path(os.environ.get('VISION_RESULTS_ROOT', str(PROJECT_ROOT)))  # 결과 저장 위치(쓰기)

DATA_DIR = PROJECT_ROOT / 'data'
DATA_AUG_DIR = PROJECT_ROOT / 'data_augmented'
RESULTS_DIR = RESULTS_ROOT / 'results'
MERGED_DIR = RESULTS_DIR / 'merged_datasets'
TRAINING_DIR = RESULTS_DIR / 'training'
EVAL_DIR = RESULTS_DIR / 'evaluation'
REPORTS_DIR = RESULTS_DIR / 'reports'

# 외부 레포지토리 경로
MASKDINO_REPO = PROJECT_ROOT.parent / 'MaskDINO'
MASK2FORMER_REPO = PROJECT_ROOT.parent / 'Mask2Former'

# ============================================================
# 카테고리 정의
# ============================================================
def _cat(name, classes, train_images_subdir="images", train_ann_name="annotations.json",
         val_filter_category_id=None, val_category_remap=None):
    """카테고리 설정 헬퍼"""
    return {
        "classes": classes,
        "coco_categories": [
            {"id": i, "name": c, "supercategory": "defect"} for i, c in enumerate(classes)
        ],
        "train_images": DATA_DIR / name / "train" / train_images_subdir if train_images_subdir else DATA_DIR / name / "train",
        "train_ann": DATA_DIR / name / "train" / train_ann_name,
        "val_images": DATA_DIR / name / "val",
        "val_ann": DATA_DIR / name / "val" / "_annotations.coco.json",
        "genai_images": DATA_AUG_DIR / name / "gen_ai" / "images",
        "genai_ann": DATA_AUG_DIR / name / "gen_ai" / "annotations.json",
        "trad_images": DATA_AUG_DIR / name / "traditional_aug" / "images",
        "trad_ann": DATA_AUG_DIR / name / "traditional_aug" / "annotations.json",
        "val_filter_category_id": val_filter_category_id,
        "val_category_remap": val_category_remap,
        "num_classes": len(classes),
    }


CATEGORIES = {
    # === 기존 3종 (train/images/ + annotations.json) ===
    "Cable": _cat("Cable", ["thunderbolt"],
                  val_filter_category_id=1, val_category_remap={1: 0}),
    "Screw": _cat("Screw", ["defect"]),
    "Casting": _cat("Casting", ["Inclusoes", "Rechupe"]),

    # === 나머지 11종 (train/ 직접 + _annotations.coco.json) ===
    "Capacitor": _cat("Capacitor", ["0"],
                      train_images_subdir="", train_ann_name="_annotations.coco.json"),
    "Console": _cat("Console", ["Collision", "Dirty", "Gap", "Scratch"],
                    train_images_subdir="", train_ann_name="_annotations.coco.json"),
    "Cylinder": _cat("Cylinder", ["Chip", "PistonMiss", "Porosity", "RCS"],
                     train_images_subdir="", train_ann_name="_annotations.coco.json"),
    "Electronics": _cat("Electronics", ["damage"],
                        train_images_subdir="", train_ann_name="_annotations.coco.json"),
    "Groove": _cat("Groove", ["s_burr", "s_scratch"],
                   train_images_subdir="", train_ann_name="_annotations.coco.json"),
    "Hemisphere": _cat("Hemisphere", ["Defect-A", "Defect-B", "Defect-C", "Defect-D"],
                       train_images_subdir="", train_ann_name="_annotations.coco.json"),
    "Lens": _cat("Lens", ["Fiber", "Flash Particle", "Hole", "Surface Damage", "Tear"],
                 train_images_subdir="", train_ann_name="_annotations.coco.json"),
    "PCB_1": _cat("PCB_1", ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"],
                  train_images_subdir="", train_ann_name="_annotations.coco.json"),
    "PCB_2": _cat("PCB_2", ["defect1", "defect2", "defect3", "defect4", "defect5", "defect6", "defect7"],
                  train_images_subdir="", train_ann_name="_annotations.coco.json"),
    "Ring": _cat("Ring", ["t_contamination", "t_scratch", "unfinished_surface"],
                 train_images_subdir="", train_ann_name="_annotations.coco.json"),
    "Wood": _cat("Wood", ["impurities", "pits"],
                 train_images_subdir="", train_ann_name="_annotations.coco.json"),
}

# ============================================================
# 모델 정의
# ============================================================
MODELS = {
    "mask_rcnn": {
        "framework": "detectron2",
        "display_name": "Mask R-CNN",
        "config": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "weights": "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl",
    },
    "cascade_mask_rcnn": {
        "framework": "detectron2",
        "display_name": "Cascade Mask R-CNN",
        "config": "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
        "weights": "detectron2://Misc/cascade_mask_rcnn_R_50_FPN_3x/144998488/model_final_480dd8.pkl",
    },
    "maskdino": {
        "framework": "detectron2",
        "display_name": "MaskDINO",
        "config": "maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml",
        "weights": "detectron2://ImageNetPretrained/torchvision/R-50.pkl",
        "requires_repo": "MaskDINO",
    },
    "mask2former": {
        "framework": "detectron2",
        "display_name": "Mask2Former",
        "config": "maskformer2_R50_bs16_50ep.yaml",
        "weights": "detectron2://ImageNetPretrained/torchvision/R-50.pkl",
        "requires_repo": "Mask2Former",
    },
    "cascade_rcnn": {
        "framework": "mmdet",
        "display_name": "Cascade R-CNN",
        "config": "cascade-rcnn_r50_fpn_1x_coco.py",
    },
    "solov2": {
        "framework": "mmdet",
        "display_name": "SOLOv2",
        "config": "solov2_r50_fpn_1x_coco.py",
    },
    "rtmdet_ins": {
        "framework": "mmdet",
        "display_name": "RTMDet-Ins",
        "config": "rtmdet-ins_s_8xb32-300e_coco.py",
    },
}

# ============================================================
# 실험 정의
# ============================================================
EXPERIMENTS = {
    "exp1": {
        "description": "생성AI 증강 수에 따른 성능 변화 (클래스당 0~125장)",
        "models": ["mask_rcnn", "cascade_mask_rcnn"],
        "conditions": {
            "baseline":   {"n_original": -1, "n_genai_per_class": 0,   "n_traditional": 0},
            "genai_25":   {"n_original": -1, "n_genai_per_class": 25,  "n_traditional": 0},
            "genai_50":   {"n_original": -1, "n_genai_per_class": 50,  "n_traditional": 0},
            "genai_75":   {"n_original": -1, "n_genai_per_class": 75,  "n_traditional": 0},
            "genai_100":  {"n_original": -1, "n_genai_per_class": 100, "n_traditional": 0},
            "genai_125":  {"n_original": -1, "n_genai_per_class": 125, "n_traditional": 0},
        },
    },
    "exp2": {
        "description": "전통적 증강 vs 생성형 AI 증강 비교",
        "models": ["mask_rcnn", "cascade_mask_rcnn", "maskdino"],
        "conditions": {
            "cond1": {"n_original": -1, "n_genai_per_class": 0,   "n_traditional": 0},
            "cond2": {"n_original": -1, "n_genai_per_class": 0,   "n_traditional": 250},
            "cond3": {"n_original": -1, "n_genai_per_class": 125, "n_traditional": 0},
            "cond4": {"n_original": -1, "n_genai_per_class": 125, "n_traditional": 250},
            "cond5": {"n_original": -1, "n_genai_per_class": 125, "n_traditional": 2750},
        },
    },
    "exp3": {
        "description": "7종 모델 비교",
        "models": list(MODELS.keys()),
        "conditions": {
            "original_only":   {"n_original": -1, "n_genai_per_class": 0,   "n_traditional": 0},      # 원본 전체
            "with_trad":       {"n_original": -1, "n_genai_per_class": 0,   "n_traditional": 3000},    # 원본 + 전통 3000
            "with_genai_trad": {"n_original": -1, "n_genai_per_class": 125, "n_traditional": 2750},    # 원본 + genai 125/cls + 전통 2750
        },
    },
}

# ============================================================
# 기본 하이퍼파라미터
# ============================================================
DEFAULT_HYPERPARAMS = {
    # 실험 1 통일 하이퍼파라미터 (exp1_config.md 기준)
    # 에폭 기반 학습 (early stopping이 자동 종료)
    "max_epochs": 1000,                    # early stopping에 맡기고 넉넉히
    "lr": 0.0015,                          # Linear Scaling Rule(0.02*12/16=0.015) + fine-tuning 1/10 보정
    "batch_size": 12,                      # A100 80GB GPU 활용률 개선 + gradient 안정
    "seed": 42,                            # 단일 시드 실행 시 기본값
    "seeds": [42, 43, 44],                 # 다중 시드 반복 실행 시 기본 목록 (exp1_config.md 기준)
    "warmup_epochs": 5,                    # 학습 초반 lr 점진 상승
    "eval_period_epochs": 5,               # 5 에폭마다 평가
    "checkpoint_period_epochs": 50,        # 디스크 절약 (체크포인트 1개 ≈ 548MB)
    "input_min_size": (640, 672, 704, 736, 768, 800),  # detectron2 기본값
    "input_max_size": 1333,                # 고해상도 유지, 작은 결함 보존
    # Early stopping
    "early_stopping_patience": 15,         # 15 * 5 = 75 에폭 동안 개선 없으면 중단
    "early_stopping_metric": "segm/AP",    # instance segmentation mAP
    # LR Decay (Step LR, 학습 길이의 70%/90% 지점)
    "lr_decay_steps": (0.7, 0.9),          # lr → lr/10 → lr/100
}

# ============================================================
# 헬퍼 함수
# ============================================================
def get_output_dir(experiment: str, condition: str, category: str, model: str, seed: int = None) -> Path:
    """학습 결과 출력 디렉토리

    seed가 주어지면 model 하위에 seed_{N}/ 디렉토리를 추가한다.
    (다중 시드 반복 실험에서 결과가 덮어쓰이지 않도록)
    """
    base = TRAINING_DIR / experiment / condition / category / model
    if seed is not None:
        return base / f"seed_{seed}"
    return base


def get_merged_dir(experiment: str, condition: str, category: str) -> Path:
    """병합 데이터셋 디렉토리"""
    return MERGED_DIR / experiment / condition / category


def get_category_info(category: str) -> dict:
    """카테고리 정보 반환"""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}. Choose from {list(CATEGORIES.keys())}")
    return CATEGORIES[category]


def get_model_info(model: str) -> dict:
    """모델 정보 반환"""
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}. Choose from {list(MODELS.keys())}")
    return MODELS[model]


def get_experiment_info(experiment: str) -> dict:
    """실험 정보 반환"""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}. Choose from {list(EXPERIMENTS.keys())}")
    return EXPERIMENTS[experiment]
