"""
통합 학습 환경 중앙 설정
- 경로, 카테고리, 모델, 실험 조건 정의
"""

from pathlib import Path

# ============================================================
# 경로 설정
# ============================================================
PROJECT_ROOT = Path('/home/jjh0709/gitrepo/VISION-Instance-Seg')
DATA_DIR = PROJECT_ROOT / 'data'
DATA_AUG_DIR = PROJECT_ROOT / 'data_augmented'
RESULTS_DIR = PROJECT_ROOT / 'results'
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
CATEGORIES = {
    "Cable": {
        "classes": ["thunderbolt"],
        "coco_categories": [
            {"id": 1, "name": "thunderbolt", "supercategory": "defect"},
        ],
        "train_images": DATA_DIR / "Cable" / "train" / "images",
        "train_ann": DATA_DIR / "Cable" / "train" / "annotations.json",
        "val_images": DATA_DIR / "Cable" / "val",  # 이미지가 직접 저장됨 (images/ 하위 없음)
        "val_ann": DATA_DIR / "Cable" / "val" / "_annotations.coco.json",
        "genai_images": DATA_AUG_DIR / "Cable" / "gen_ai" / "images",
        "genai_ann": DATA_AUG_DIR / "Cable" / "gen_ai" / "annotations.json",
        "trad_images": DATA_AUG_DIR / "Cable" / "traditional_aug" / "images",
        "trad_ann": DATA_AUG_DIR / "Cable" / "traditional_aug" / "annotations.json",
        # val에 break(id=0) + thunderbolt(id=1) 혼재 → thunderbolt(id=1)만 평가
        "val_filter_category_id": 1,
        "val_category_remap": {1: 0},  # val의 category_id 1 → 학습용 0으로 리매핑
        "num_classes": 1,
    },
    "Screw": {
        "classes": ["defect"],
        "coco_categories": [
            {"id": 0, "name": "defect", "supercategory": "defect"},
        ],
        "train_images": DATA_DIR / "Screw" / "train" / "images",
        "train_ann": DATA_DIR / "Screw" / "train" / "annotations.json",
        "val_images": DATA_DIR / "Screw" / "val",
        "val_ann": DATA_DIR / "Screw" / "val" / "_annotations.coco.json",
        "genai_images": DATA_AUG_DIR / "Screw" / "gen_ai" / "images",
        "genai_ann": DATA_AUG_DIR / "Screw" / "gen_ai" / "annotations.json",
        "trad_images": DATA_AUG_DIR / "Screw" / "traditional_aug" / "images",
        "trad_ann": DATA_AUG_DIR / "Screw" / "traditional_aug" / "annotations.json",
        "val_filter_category_id": None,  # 필터링 불필요
        "val_category_remap": None,
        "num_classes": 1,
    },
    "Casting": {
        "classes": ["Inclusoes", "Rechupe"],
        "coco_categories": [
            {"id": 0, "name": "Inclusoes", "supercategory": "defect"},
            {"id": 1, "name": "Rechupe", "supercategory": "defect"},
        ],
        "train_images": DATA_DIR / "Casting" / "train" / "images",
        "train_ann": DATA_DIR / "Casting" / "train" / "annotations.json",
        "val_images": DATA_DIR / "Casting" / "val",
        "val_ann": DATA_DIR / "Casting" / "val" / "_annotations.coco.json",
        "genai_images": DATA_AUG_DIR / "Casting" / "gen_ai" / "images",
        "genai_ann": DATA_AUG_DIR / "Casting" / "gen_ai" / "annotations.json",
        "trad_images": DATA_AUG_DIR / "Casting" / "traditional_aug" / "images",
        "trad_ann": DATA_AUG_DIR / "Casting" / "traditional_aug" / "annotations.json",
        "val_filter_category_id": None,
        "val_category_remap": None,
        "num_classes": 2,
    },
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
        "description": "생성AI 증강 수에 따른 성능 변화",
        "models": ["mask_rcnn", "cascade_mask_rcnn"],
        "conditions": {
            "baseline":   {"n_original": 25, "n_genai": 0,   "n_traditional": 0},
            "genai_50":   {"n_original": 25, "n_genai": 50,  "n_traditional": 0},
            "genai_100":  {"n_original": 25, "n_genai": 100, "n_traditional": 0},
            "genai_150":  {"n_original": 25, "n_genai": 150, "n_traditional": 0},
            "genai_200":  {"n_original": 25, "n_genai": 200, "n_traditional": 0},
            "genai_250":  {"n_original": 25, "n_genai": 250, "n_traditional": 0},
        },
    },
    "exp2": {
        "description": "전통적 증강 vs 생성형 AI 증강 비교",
        "models": ["mask_rcnn", "cascade_mask_rcnn", "maskdino"],
        "conditions": {
            "cond1": {"n_original": 25, "n_genai": 0,   "n_traditional": 0},
            "cond2": {"n_original": 25, "n_genai": 0,   "n_traditional": 250},
            "cond3": {"n_original": 25, "n_genai": 250, "n_traditional": 0},
            "cond4": {"n_original": 25, "n_genai": 250, "n_traditional": 250},
            "cond5": {"n_original": 25, "n_genai": 250, "n_traditional": 2750},
        },
    },
    "exp3": {
        "description": "7종 모델 비교",
        "models": list(MODELS.keys()),
        "conditions": {
            "original_only":  {"n_original": -1, "n_genai": 0,    "n_traditional": 0},     # 원본 전체
            "with_trad":      {"n_original": -1, "n_genai": 0,    "n_traditional": 3000},   # 원본 + 전통 3000
            "with_genai_trad": {"n_original": -1, "n_genai": 250, "n_traditional": 2750},   # 원본 + genai 250 + 전통 2750
        },
    },
}

# ============================================================
# 기본 하이퍼파라미터
# ============================================================
DEFAULT_HYPERPARAMS = {
    "max_iter": 10000,
    "lr": 1e-4,
    "batch_size": 2,
    "seed": 42,
    "warmup_iters": 200,
    "eval_period": 500,
    "checkpoint_period": 1000,
    "input_min_size": (480, 512, 544, 576, 608, 640),
    "input_max_size": 800,
}

# ============================================================
# 헬퍼 함수
# ============================================================
def get_output_dir(experiment: str, condition: str, category: str, model: str) -> Path:
    """학습 결과 출력 디렉토리"""
    return TRAINING_DIR / experiment / condition / category / model


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
