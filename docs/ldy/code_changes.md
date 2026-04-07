# 코드 수정 내역

ldy1118이 수정한 코드 변경 사항 정리.

---

## 1. training/config.py

### 1-1. 경로 분리
- `PROJECT_ROOT`: 하드코딩(`/home/jjh0709/...`) → 동적 경로(`Path(__file__).resolve().parent.parent`)
- `DATA_ROOT` 추가: 데이터 원본 경로는 `/home/jjh0709/gitrepo/VISION-Instance-Seg`로 분리
- **왜**: 기존 코드는 jjh0709 계정 경로가 하드코딩되어 있어서 ldy1118 계정에서 실행하면 결과 저장 시 권한 에러(PermissionError) 발생. 데이터 읽기 경로와 결과 저장 경로를 분리하여 각자 계정에서 독립적으로 실행 가능하도록 함.

### 1-2. Unified 카테고리 추가 (14클래스 통합)
- `UNIFIED_SUBCATEGORIES` 상수: 6개 카테고리 목록 (Cable, Screw, Casting, Console, Cylinder, Wood)
- `_build_unified_category()` 함수: 6개 카테고리의 클래스를 합쳐 14개 글로벌 클래스 매핑 생성
- `CATEGORIES["Unified"]` 항목 추가
- **왜**: 기존 코드는 카테고리별 독립 학습만 지원. 실험 목적상 6개 카테고리 14개 결함 클래스를 하나의 모델로 동시에 학습/평가해야 함. 각 카테고리의 category_id가 0부터 시작하므로 글로벌 ID 재매핑이 필요.

글로벌 클래스 매핑:
| Global ID | 카테고리 | 클래스명 |
|:---------:|---------|---------|
| 0 | Cable | thunderbolt |
| 1 | Screw | defect |
| 2 | Casting | Inclusoes |
| 3 | Casting | Rechupe |
| 4 | Console | Collision |
| 5 | Console | Dirty |
| 6 | Console | Gap |
| 7 | Console | Scratch |
| 8 | Cylinder | Chip |
| 9 | Cylinder | PistonMiss |
| 10 | Cylinder | Porosity |
| 11 | Cylinder | RCS |
| 12 | Wood | impurities |
| 13 | Wood | pits |

### 1-3. (deprecated) Cascade Mask R-CNN 모델별 하이퍼파라미터
- 한때 `MODELS["cascade_mask_rcnn"]["hyperparams"]`로 모델별 hyperparams를 분리했으나, exp1_config.md에서 Mask R-CNN과 Cascade Mask R-CNN을 동일 하이퍼파라미터로 통일하기로 결정 → DEFAULT_HYPERPARAMS만 사용하도록 되돌림
- train.py의 모델별 hyperparams 병합 로직(`get_model_info(model).get("hyperparams", {})`)은 그대로 유지 (다른 모델에서 필요할 때 사용 가능)

### 1-4. MULTI_SEEDS 추가
- `MULTI_SEEDS = [42, 43, 44]` 추가
- **왜**: 실험 1에서 3회 반복 실험(다중 시드)을 통해 평균 ± 표준편차를 보고하기 위함. seed가 영향을 주는 모델 가중치 초기화, 데이터 셔플 순서, 학습 augmentation 랜덤성을 통제하여 학습 안정성 평가.

### 1-5. get_output_dir에 seed 옵션 추가
- `get_output_dir(experiment, condition, category, model, seed=None)` 시그니처 확장
- seed가 주어지면 `model/seed{N}/` 하위 폴더에 결과 저장
- **왜**: 다중 시드 학습 시 같은 모델/조건이라도 시드별로 결과가 분리 저장되어야 덮어쓰기 방지.

### 1-6. DEFAULT_HYPERPARAMS 변경
- `input_min_size`: (480~640) → (640~800) / `input_max_size`: 800 → 1333
  - **왜**: 원본 이미지가 대부분 1000px 이상 고해상도인데, max=800으로 과도하게 축소하면 작은 결함이 소실됨. detectron2 기본값(640~800, max=1333)으로 복원하여 결함 검출 성능 향상.
- `checkpoint_period_epochs`: 10 → 50
  - **왜**: 체크포인트 1개가 548MB. 10 에폭마다 저장하면 30개 × 548MB = ~17GB로 디스크 쿼타(88GB) 초과. 50 에폭으로 줄여 디스크 사용량 절약.
- `max_epochs`: 300 → 1000
  - **왜**: early stopping에 맡기고 넉넉히 설정. baseline(데이터 적음)에서도 200 에폭까지 학습되는 경우가 있어 데이터가 많은 조건에서는 300으로 부족할 수 있음.
- `early_stopping_patience`: 15 (유지)
  - **참고**: 한때 10으로 줄였다가 다시 15로 복원. exp1_config.md 기준은 15. patience를 줄여도 미세 개선으로 카운터가 리셋되는 본질적 문제는 해결되지 않으며, min_delta는 자의적 기준이 되어 연구 공정성을 해칠 수 있어 도입하지 않음.
- `lr_decay_steps`: (0.7, 0.9) 추가
  - **왜**: Step LR decay. 학습 길이의 70%, 90% 지점에서 lr을 1/10씩 감소.

---

## 2. training/data_pipeline.py

### 2-1. import 추가
- `MERGED_DIR` import 추가
- **왜**: `prepare_unified_val_dataset()`에서 val 데이터 저장 경로로 `MERGED_DIR` 직접 사용 필요.

### 2-2. 통합 카테고리 유틸리티 함수 추가
- `_remap_category_ids(anns, local_to_global)`: annotation의 category_id를 로컬→글로벌로 변환
- `_prefix_filenames(imgs, prefix)`: 이미지 file_name에 카테고리명 접두사 추가. 원본은 `_original_filename`에 보존
- **왜**: 6개 카테고리를 하나로 합치려면 (1) category_id 충돌 해결 (각 카테고리가 0부터 시작)과 (2) 파일명 충돌 해결 (다른 카테고리에 같은 파일명 존재 가능)이 필요.

### 2-3. _merge_sources() 수정 (1줄)
- `src = images_dir / img["file_name"]` → `src = images_dir / img.get("_original_filename", img["file_name"])`
- **왜**: `_prefix_filenames()`로 file_name에 접두사를 붙이면 원본 파일 경로를 찾을 수 없음. `_original_filename`에 원본 경로를 보존하고 복사 시 사용. 기존 호출에는 `_original_filename`이 없으므로 동작 변화 없음 (하위 호환).

### 2-4. prepare_unified_dataset() 추가
- 6개 카테고리를 순회하며 원본/GenAI/전통증강 데이터 로드 → 글로벌 ID 재매핑 → 파일명 접두사 → 통합 병합
- 출력: `results/merged_datasets/{exp}/{cond}/Unified/`
- **왜**: 14클래스 통합 학습 데이터셋을 생성하는 핵심 함수. 기존 `prepare_dataset()`은 카테고리 1개만 처리 가능.

### 2-5. prepare_unified_val_dataset() 추가
- 6개 카테고리의 val 데이터를 동일 방식으로 통합
- Cable val의 thunderbolt 필터링 포함
- 출력: `results/merged_datasets/_unified_val/`
- **왜**: 학습 데이터와 동일한 14클래스 매핑으로 val 데이터를 통합해야 평가 가능. val은 실험 조건에 무관하므로 한 번만 생성하여 재사용.

### 2-6. 기존 함수에 분기 추가
- `prepare_dataset()`: `if category == "Unified": return prepare_unified_dataset(...)`
- `prepare_val_dataset()`: `if category == "Unified": return prepare_unified_val_dataset()`
- **왜**: 기존 코드를 최소한으로 수정하면서 Unified 지원. category가 "Unified"일 때만 통합 로직을 타고, 기존 카테고리별 로직은 그대로 유지.

### 2-7. CLI --category all에서 Unified 제외
- `all` 선택 시 Unified 포함하지 않음 (명시적으로만 사용)
- **왜**: `--category all`은 보통 카테고리별 독립 학습 용도. Unified가 포함되면 의도치 않게 14클래스 통합 학습이 함께 실행되어 시간/디스크 낭비.

---

## 3. training/train.py

### 3-1. --tag 옵션 추가
- 결과 저장 경로에 태그 추가 (예: `--tag bs4_lr5e-4`)
- **왜**: 하이퍼파라미터 비교 실험 시 같은 모델/조건이라도 다른 설정으로 돌리면 결과가 덮어써짐. tag로 경로를 분리하여 결과 보존. (예: `cascade_mask_rcnn_bs4_lr5e-4/`)

### 3-2. 모델별 하이퍼파라미터 병합 로직
- 기존: `DEFAULT_HYPERPARAMS` → CLI 오버라이드
- 변경: `DEFAULT_HYPERPARAMS` → 모델별 설정(`MODELS[model]["hyperparams"]`) → CLI 오버라이드
- 우선순위: DEFAULT < 모델별 < CLI
- **왜**: 모델별로 다른 하이퍼파라미터를 사용하려면 매번 긴 CLI 명령어를 입력해야 함. config.py에 모델별 설정을 정의하면 `--model cascade_mask_rcnn`만으로 해당 모델에 맞는 설정이 자동 적용.

### 3-3. run_single() 수정
- `base_model_name` 파라미터 추가
- tag가 붙은 경우 결과 경로는 tag 포함, adapter 생성은 원래 모델명 사용
- **왜**: tag가 붙으면 모델명이 `cascade_mask_rcnn_bs4_lr5e-4`로 바뀌는데, 이걸로 MODELS dict를 조회하면 KeyError. adapter 생성에는 원래 모델명(`cascade_mask_rcnn`)을 사용하도록 분리.

### 3-4. --category all에서 Unified 제외
- `categories = [c for c in CATEGORIES if c != "Unified"] if args.category == 'all' else [args.category]`
- **왜**: `--category all`은 카테고리별 독립 학습 용도. Unified는 명시적으로만 실행되어야 함.

### 3-5. --multi-seed 옵션 추가
- `--multi-seed` 플래그 추가: 사용 시 `MULTI_SEEDS=[42, 43, 44]`로 각 (카테고리, 조건, 모델) 조합을 3회 반복 실행
- 단일 시드(기본): 결과 경로 `model/`
- 다중 시드: 결과 경로 `model/seed42/`, `model/seed43/`, `model/seed44/`
- 결과 JSON에 `seed` 필드 추가, save_results의 중복 키 판단도 seed 포함
- **왜**: 단일 시드만으로는 학습 결과의 통계적 신뢰성이 부족. 3회 반복하여 평균±표준편차로 보고해야 GenAI 증강 효과를 통계적으로 의미있게 비교 가능. 단일/다중 모드를 토글할 수 있어 빠른 테스트와 정식 실험을 모두 지원.

### 3-6. run_single에 seed_subdir 파라미터 추가
- `seed_subdir`이 주어지면 출력 경로에 `seed{N}` 하위 폴더 생성
- **왜**: 다중 시드 모드에서 같은 모델/조건이라도 시드별 결과가 분리 저장되도록 지원.

---

## 4. training/adapters/detectron2_adapter.py

### 4-1. EarlyStoppingHook 버그 수정
- `class EarlyStoppingHook:` → `class EarlyStoppingHook(HookBase):`
- `from detectron2.engine.train_loop import HookBase` import 추가
- **왜**: detectron2의 `trainer.register_hooks()`는 `HookBase` 인스턴스만 받음. 상속 없이 등록하면 `AssertionError`가 발생하여 학습 자체가 불가능했음. 기존 코드의 버그.

### 4-2. GPU 메모리 기록 개선
- 기존: `torch.cuda.max_memory_allocated()` (텐서 할당량만, ~4.6GB)
- 변경: `max_memory_allocated()` + `max_memory_reserved()` 둘 다 기록
- `peak_memory_mb`: 모델 실 사용량 (allocated)
- `peak_memory_reserved_mb`: GPU 점유량 (reserved, nvidia-smi 기준)
- **왜**: `max_memory_allocated()`는 4.6GB로 표시되지만 실제 nvidia-smi에서는 13GB 사용. PyTorch가 내부 캐시용으로 추가 예약하는 메모리가 빠져있었음. 두 값을 모두 기록하여 모델 효율성(allocated)과 실제 GPU 요구량(reserved) 모두 파악 가능.

### 4-3. Early Stopping 학습 중단 버그 수정 (2건)

**버그 1: 학습 루프가 멈추지 않음**
- 기존: `self.trainer.max_iter = next_iter`로 중단 시도
- 변경: `EarlyStopException` 커스텀 예외를 raise하여 강제 중단, `train()` 호출부에서 `try-except`로 안전하게 처리
- **왜**: detectron2의 학습 루프가 `for self.iter in range(start_iter, max_iter)`로 동작하는데, `range()`는 루프 시작 시 이미 생성되므로 중간에 `max_iter`를 바꿔도 루프가 멈추지 않음. 실제로 STOP 로그는 찍혔지만 학습이 300 에폭까지 계속 돌아감.

**버그 2: EarlyStopException이 detectron2에서 에러로 처리됨**
- 기존: `EarlyStopException(Exception)` → detectron2의 `except Exception`에 잡혀서 에러로 기록, 이후 조건들도 전부 실패
- 변경: `EarlyStopException(BaseException)` → detectron2의 `except Exception`에 안 잡히고, 우리 `try-except`에서만 잡혀서 정상 종료 처리
- **왜**: detectron2 train_loop.py가 `except Exception: raise`로 모든 Exception을 에러로 기록 후 재발생시킴. BaseException을 상속하면 이 catch에 안 잡히면서도 `finally: self.after_train()`은 정상 실행됨.

### 4-4. GPU 활용률 피크 기록 추가
- 백그라운드 스레드로 60초마다 `nvidia-smi`에서 GPU utilization 측정
- 학습 전체에서 최대값을 `gpu_utilization_peak_pct`로 JSON에 저장
- **왜**: GPU 메모리 사용량만으로는 GPU를 얼마나 효율적으로 쓰고 있는지 알 수 없음. 활용률(%)을 함께 기록하면 모델별 GPU 효율성 비교 가능. 학습 종료 시점의 순간값이 아닌 피크값을 저장하여 학습 중 최대 부하를 파악.

### 4-5. 학습 전 GPU 상태 기록 추가
- 학습 시작 전 GPU 상태를 `pre_train_gpu` dict로 JSON에 저장
- 기록 항목: device_id, 학습 전 GPU 메모리 사용량(nvidia-smi), GPU 총 메모리, lr, batch_size
- **왜**: 다른 사용자의 프로세스가 같은 GPU에서 돌고 있으면 메모리/성능에 영향을 줄 수 있음. 학습 전 GPU 상태를 기록해두면 결과 해석 시 외부 요인 영향을 확인 가능. lr, batch_size도 함께 저장하여 어떤 하이퍼파라미터로 학습했는지 결과 파일만 보고 파악 가능.

---

## 5. scripts/run_hp_search.sh (신규 생성)

- Cascade Mask R-CNN 하이퍼파라미터 비교 스크립트
- 3가지 config (bs2/lr2.5e-4, bs4/lr5e-4, bs8/lr1e-3) 순차 실행
- 학습 완료 후 중간 체크포인트 자동 삭제 (디스크 절약)
- tmux 백그라운드 실행용
- **왜**: 최적 하이퍼파라미터를 찾기 위해 여러 config를 비교해야 함. 수동 실행하면 각 config가 끝날 때까지 기다려야 하므로, 순차 자동 실행 스크립트로 자리 비운 사이에 완료되도록 함.

---

## 6. scripts/run_exp1.sh (신규 생성)

- 실험 1 본 실행 스크립트 (Cascade Mask R-CNN, 6개 조건 순차)
- baseline → genai_25 → ... → genai_125 자동 진행
- 각 조건 학습 후 중간 체크포인트 자동 정리
- **왜**: 실험 1의 6개 조건을 일관되게 실행하기 위함. 디스크 누적을 막기 위해 조건마다 정리 단계 포함.

---

## 7. docs/exp1_config.md 수정

기존 dev에 있는 exp1_config.md에서 다음 항목 수정:
- **카테고리 5개 → 6개** (Wood 추가)
- **총 학습 횟수**: `5 카테고리 × 6 × 2 × 3 = 180회` → `1 (Unified) × 6 × 2 × 3 = 36회`
- **카테고리 테이블**에 Wood (impurities, pits) 추가
- **데이터 설명**: "5개 카테고리, baseline 26~138장" → "6개 카테고리 통합(Unified, 14클래스), baseline 421장"
- **왜**: dev 버전은 카테고리별 독립 학습 가정. 우리는 Unified로 14클래스 통합 학습하므로 데이터셋 단위가 1개. Wood도 정상 학습 대상 카테고리.

---

## 8. docs/학습_guideline.md 통째 재작성

기존 가이드는 옛날 정보(`--max-iter`, lr=1e-4, bs=2, 카테고리 3개만 등)로 되어 있어서 새 코드와 안 맞음. 다음 내용으로 통합 재작성:
- 사전 조건 (test env, jjh env 둘 다 지원)
- 새 CLI 옵션 (`--multi-seed`, `--tag`, `--max-epochs` 등)
- Unified 카테고리 (14클래스 통합) 설명 + 글로벌 클래스 매핑
- 단일/다중 시드 학습 예시
- 백그라운드 실행 (tmux) 가이드
- 결과 JSON 항목 정리
- 디스크 관리, 디버깅 섹션
- **왜**: 진우와 공유할 공식 사용법 문서. 새 기능(Unified, multi-seed, GPU 추적 등)이 모두 반영되어야 함.

---

## 9. 환경 설정

- conda env `test` (Python 3.12)에 detectron2 0.6, torchvision 0.20.1 설치
- PyTorch 2.5.1+cu121 (서버 CUDA 드라이버 12.2 호환)
- **왜**: 기존 환경(jjh conda env)은 jjh0709 계정에만 있어서 ldy1118 계정에서 사용 불가. 독립적인 실행 환경 구축 필요. PyTorch는 서버 CUDA 드라이버(12.2)에 맞는 버전으로 설치해야 GPU 사용 가능 (12.8 버전은 cuDNN 초기화 에러 발생).
