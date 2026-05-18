"""
통합 학습 환경 중앙 설정
- 경로, 카테고리, 모델, 실험 조건 정의
"""

from pathlib import Path

# ============================================================
# 경로 설정
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # training/ 상위 = 프로젝트 루트
DATA_ROOT = Path('/home/jjh0709/gitrepo/VISION-Instance-Seg')  # 데이터 원본 경로
DATA_DIR = DATA_ROOT / 'data'
DATA_AUG_DIR = DATA_ROOT / 'data_augmented'
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
# 통합 카테고리 (6개 카테고리 × 14개 결함 클래스)
# ============================================================
UNIFIED_SUBCATEGORIES = ["Cable", "Screw", "Casting", "Console", "Cylinder", "Wood"]


def _build_unified_category():
    """6개 카테고리의 클래스를 합쳐 통합 카테고리 생성"""
    all_classes = []
    global_id_offset = {}
    for subcat in UNIFIED_SUBCATEGORIES:
        global_id_offset[subcat] = len(all_classes)
        all_classes.extend(CATEGORIES[subcat]["classes"])
    return {
        "classes": all_classes,
        "coco_categories": [
            {"id": i, "name": c, "supercategory": "defect"}
            for i, c in enumerate(all_classes)
        ],
        "num_classes": len(all_classes),
        "subcategories": UNIFIED_SUBCATEGORIES,
        "global_id_offset": global_id_offset,
        "train_images": None,
        "train_ann": None,
        "val_images": None,
        "val_ann": None,
        "genai_images": None,
        "genai_ann": None,
        "trad_images": None,
        "trad_ann": None,
        "val_filter_category_id": None,
        "val_category_remap": None,
    }


CATEGORIES["Unified"] = _build_unified_category()


# 실험 2용 3클래스 통합 (Inclusoes + Dirty + impurities)
EXP2_SUBCATEGORIES = ["Inclusoes", "Dirty", "impurities"]


def _build_exp2_3cls():
    """실험 2 대상 3개 defect를 합쳐 3클래스 카테고리 생성"""
    all_classes = []
    global_id_offset = {}
    for subcat in EXP2_SUBCATEGORIES:
        global_id_offset[subcat] = len(all_classes)
        all_classes.extend(CATEGORIES[subcat]["classes"])
    return {
        "classes": all_classes,   # ["Inclusoes", "Dirty", "impurities"]
        "coco_categories": [
            {"id": i, "name": c, "supercategory": "defect"}
            for i, c in enumerate(all_classes)
        ],
        "num_classes": len(all_classes),
        "subcategories": EXP2_SUBCATEGORIES,
        "global_id_offset": global_id_offset,
        "train_images": None,
        "train_ann": None,
        "val_images": None,
        "val_ann": None,
        "genai_images": None,
        "genai_ann": None,
        "trad_images": None,
        "trad_ann": None,
        "val_filter_category_id": None,
        "val_category_remap": None,
    }


# ============================================================
# 개별 Defect 카테고리 (단일 클래스 학습용)
# 부모 카테고리의 데이터를 공유하되, 특정 defect만 필터링
# ============================================================
def _cat_single_defect(parent_name, defect_name, filter_category_id):
    """부모 카테고리에서 특정 defect만 추출한 단일 클래스 카테고리"""
    parent = CATEGORIES[parent_name]
    return {
        "classes": [defect_name],
        "coco_categories": [{"id": 0, "name": defect_name, "supercategory": "defect"}],
        "train_images": parent["train_images"],
        "train_ann": parent["train_ann"],
        "val_images": parent["val_images"],
        "val_ann": parent["val_ann"],
        "genai_images": parent["genai_images"],
        "genai_ann": parent["genai_ann"],
        "trad_images": parent["trad_images"],
        "trad_ann": parent["trad_ann"],
        "val_filter_category_id": filter_category_id,
        "val_category_remap": {filter_category_id: 0},
        "train_filter_category_id": filter_category_id,
        "train_category_remap": {filter_category_id: 0},
        "num_classes": 1,
        "parent_category": parent_name,
    }


# Casting 개별 defect
CATEGORIES["Inclusoes"] = _cat_single_defect("Casting", "Inclusoes", 0)
CATEGORIES["Rechupe"] = _cat_single_defect("Casting", "Rechupe", 1)

# Console 개별 defect
CATEGORIES["Collision"] = _cat_single_defect("Console", "Collision", 0)
CATEGORIES["Dirty"] = _cat_single_defect("Console", "Dirty", 1)
CATEGORIES["Gap"] = _cat_single_defect("Console", "Gap", 2)
CATEGORIES["Scratch"] = _cat_single_defect("Console", "Scratch", 3)

# Cylinder 개별 defect
CATEGORIES["Chip"] = _cat_single_defect("Cylinder", "Chip", 0)
CATEGORIES["PistonMiss"] = _cat_single_defect("Cylinder", "PistonMiss", 1)
CATEGORIES["Porosity"] = _cat_single_defect("Cylinder", "Porosity", 2)
CATEGORIES["RCS"] = _cat_single_defect("Cylinder", "RCS", 3)

# Wood 개별 defect
CATEGORIES["impurities"] = _cat_single_defect("Wood", "impurities", 0)
CATEGORIES["pits"] = _cat_single_defect("Wood", "pits", 1)

# Cable/Screw는 원래 1클래스이므로 그대로 사용 가능
# CATEGORIES["thunderbolt"] = Cable, CATEGORIES["defect"] = Screw

# 실험 2용 3클래스 통합 등록 (개별 defect 정의 이후에 호출)
CATEGORIES["Exp2_3cls"] = _build_exp2_3cls()

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
        "hyperparams": {"lr": 1e-4, "batch_size": 4},  # AdamW fine-tuning (MaskDINO 기본값)
    },
    "mask2former": {
        "framework": "detectron2",
        "display_name": "Mask2Former",
        "config": "maskformer2_R50_bs16_50ep.yaml",
        # Mask2Former COCO pretrained 전체 ckpt (R50 instance seg, 43.7 mask AP)
        # ImageNet backbone만으론 3-class fine-tuning 수렴 불가(v1~v3 실패 입증)
        "weights": "/project/ahnailab/yjw0619/seg/seg/VISION-Instance-Seg/checkpoints/mask2former/model_final_3c8ec9.pkl",
        "requires_repo": "Mask2Former",
        # ldy1118 MaskDINO v3-lifeai 레시피 준용:
        # lr=5e-5, cosine, AMP off, warmup_iters=4000, grad clip 0.01 (어댑터 내부)
        "hyperparams": {
            "lr": 5e-5,
            "batch_size": 4,
            "lr_scheduler": "cosine",
            "amp_enabled": False,
            "warmup_iters": 4000,
        },
    },
    "cascade_rcnn": {
        "framework": "mmdet",
        "display_name": "Cascade R-CNN",
        "config": "cascade-rcnn_r50_fpn_1x_coco.py",
        "weights": "https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth",
        "hyperparams": {"lr": 1e-4, "batch_size": 4},  # AdamW fine-tuning
    },
    "solov2": {
        "framework": "mmdet",
        "display_name": "SOLOv2",
        "config": "solov2_r50_fpn_1x_coco.py",
        "weights": "https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_1x_coco/solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth",
        "hyperparams": {"lr": 1e-4, "batch_size": 4},
    },
    "rtmdet_ins": {
        "framework": "mmdet",
        "display_name": "RTMDet-Ins",
        "config": "rtmdet-ins_s_8xb32-300e_coco.py",
        "weights": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_s_8xb32-300e_coco/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth",
        "hyperparams": {"lr": 1e-4, "batch_size": 4},
    },
}

# ============================================================
# 실험 정의
# ============================================================
# 학습용 원본 이미지 수 (클래스당). _sample_images_per_class()로 클래스별 균형 샘플링.
# 클래스의 가용 이미지가 이 값보다 적으면 자동으로 전체 사용 (data_pipeline._sample_images_per_class).
N_ORIGINAL_TRAIN_PER_CLASS = 20

EXPERIMENTS = {
    "exp1": {
        "description": "생성AI 증강 수에 따른 성능 변화 (클래스당 0~125장)",
        "models": ["mask_rcnn", "cascade_mask_rcnn"],
        "conditions": {
            "baseline":   {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 0,   "n_traditional_per_class": 0},
            "genai_25":   {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 25,  "n_traditional_per_class": 0},
            "genai_50":   {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 50,  "n_traditional_per_class": 0},
            "genai_75":   {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 75,  "n_traditional_per_class": 0},
            "genai_100":  {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 100, "n_traditional_per_class": 0},
            "genai_125":  {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 125, "n_traditional_per_class": 0},
        },
    },
    "exp1_3cls": {
        "description": "Exp1 3클래스 축소판 (Dirty/Inclusoes/impurities) — iter 기반 공정 비교",
        "models": ["mask_rcnn", "cascade_mask_rcnn"],
        "categories": ["Exp2_3cls"],
        "conditions": {
            "baseline":  {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 0,   "n_traditional_per_class": 0},
            "genai_25":  {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 25,  "n_traditional_per_class": 0},
            "genai_50":  {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 50,  "n_traditional_per_class": 0},
            "genai_75":  {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 75,  "n_traditional_per_class": 0},
            "genai_100": {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 100, "n_traditional_per_class": 0},
            "genai_125": {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 125, "n_traditional_per_class": 0},
        },
    },
    "exp2": {
        "description": "전통 vs GenAI 비교 + (원본+GenAI)+전통 N배 탐색",
        "models": list(MODELS.keys()),  # Phase1은 2종, Phase2(cond4)는 7종 전부
        "conditions": {
            # Phase 1: 전통 vs GenAI (2종 모델)
            "cond1": {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 0,   "n_traditional_per_class": 0},       # baseline
            "cond2": {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 0,   "n_traditional_per_class": 125},     # 전통 125/cls
            "cond3": {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 125, "n_traditional_per_class": 0},       # GenAI 125/cls
            # Phase 2: cond4 (원본20+GenAI125)+전통N배 — (20+125)×N = 145×N per class (7종 모델)
            "cond4_1x":  {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 125, "n_traditional_per_class": 145},
            "cond4_2x":  {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 125, "n_traditional_per_class": 290},
            "cond4_3x":  {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 125, "n_traditional_per_class": 435},
            "cond4_4x":  {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 125, "n_traditional_per_class": 580},
            "cond4_5x":  {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 125, "n_traditional_per_class": 725},
            "cond4_6x":  {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 125, "n_traditional_per_class": 870},
            "cond4_7x":  {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 125, "n_traditional_per_class": 1015},
            "cond4_8x":  {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 125, "n_traditional_per_class": 1160},
            "cond4_9x":  {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 125, "n_traditional_per_class": 1305},
            "cond4_10x": {"n_original_per_class": N_ORIGINAL_TRAIN_PER_CLASS, "n_genai_per_class": 125, "n_traditional_per_class": 1450},
        },
    },
}

# ============================================================
# 기본 하이퍼파라미터
# ============================================================
DEFAULT_HYPERPARAMS = {
    # === iter 기반 (우선 사용 — exp1_3cls 등 공정 비교 실험) ===
    # iter 필드가 None이 아니면 epoch 필드보다 우선 적용됨 (detectron2_adapter)
    "max_iters": None,                     # 예: 20000. None이면 epoch 모드
    "warmup_iters": None,                  # 예: 500
    "eval_period_iters": None,             # 예: 500
    "checkpoint_period_iters": None,       # 예: 2000
    "lr_scheduler": "step",                # "step" or "cosine" (cosine은 WarmupCosineLR)

    # === epoch 기반 (기존 — 하위호환) ===
    "max_epochs": 1000,                    # early stopping에 맡기고 넉넉히
    "warmup_epochs": 5,                    # 학습 초반 lr 점진 상승
    "eval_period_epochs": 5,               # 5 에폭마다 평가
    "checkpoint_period_epochs": 50,        # 디스크 절약 (체크포인트 1개 ≈ 548MB)

    # === 공통 ===
    "lr": 0.0015,                          # Linear Scaling Rule(0.02*12/16=0.015) + fine-tuning 1/10 보정
    "batch_size": 12,                      # A100 80GB GPU 활용률 개선 + gradient 안정
    "seed": 42,                            # 3회 반복 시 42, 43, 44 사용
    "max_periodic_checkpoints": 1,         # 주기 체크포인트 rotation (최신 N개만 유지)
    "input_min_size": (640, 672, 704, 736, 768, 800),  # detectron2 기본값
    "input_max_size": 1333,                # 고해상도 유지, 작은 결함 보존
    # Early stopping (단위: eval 횟수. iter 모드에선 patience × eval_period_iters 동안 미개선)
    "early_stopping_patience": 15,         # epoch 모드: 15*5=75 epoch / iter 모드: 15*500=7500 iter
    "early_stopping_metric": "segm/AP",    # instance segmentation mAP
    # LR Decay (Step LR, 학습 길이의 70%/90% 지점) — step scheduler 전용
    "lr_decay_steps": (0.7, 0.9),          # lr → lr/10 → lr/100
}

# 다중 시드 학습 시 사용할 시드 목록 (--multi-seed 옵션 사용 시)
MULTI_SEEDS = [42, 43, 44]

# ============================================================
# 헬퍼 함수
# ============================================================
def get_output_dir(experiment: str, condition: str, category: str, model: str,
                   seed: int = None) -> Path:
    """학습 결과 출력 디렉토리. seed가 주어지면 seed별 하위 디렉토리 생성"""
    base = TRAINING_DIR / experiment / condition / category / model
    if seed is not None:
        return base / f"seed{seed}"
    return base


def get_merged_dir(experiment: str, condition: str, category: str,
                   seed: int = None) -> Path:
    """병합 데이터셋 디렉토리.

    seed가 주어지면 시드별 하위 폴더 생성하여 진짜 독립 replication 보장
    (각 시드가 자체 데이터 샘플링을 하도록).
    하드 링크로 원본을 공유하므로 시드별 추가 디스크 비용은 거의 없음.
    """
    base = MERGED_DIR / experiment / condition / category
    if seed is not None:
        return base / f"seed{seed}"
    return base


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
