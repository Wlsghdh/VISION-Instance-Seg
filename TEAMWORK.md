# TEAMWORK.md — 팀 협업 가이드

> 프로젝트: VISION Instance Segmentation
> 인원: 3명 (본인 jjh0709 + 팀원 A + 팀원 B)
> 목표: 7종 모델 구현 · 실험 · 평가지표 비교

---

## 1. 역할 및 업무 분담

### 공통 작업 (전원)
| 작업 | 내용 | 비고 |
|------|------|------|
| **Annotation 검수** | Cable / Screw / Casting gen_ai 라벨링 | labeling_server로 공동 작업 |
| **실험 설계 회의** | 실험 조건 결정, 하이퍼파라미터 통일 | 주기적 동기화 필요 |
| **결과 기록** | `results/` 하위에 실험 결과 정리 | 형식 통일 (아래 참고) |

---

### 개인 담당 모델 (7종 분배)

| 담당자 | 모델 | Branch명 |
|--------|------|----------|
| **본인 (jjh0709)** | Mask DINO | `feat/model/mask-dino` |
| **본인 (jjh0709)** | 최신 모델 1종 (예: Mask2Former) | `feat/model/mask2former` |
| **팀원 A** | Mask R-CNN | `feat/model/mask-rcnn` |
| **팀원 A** | Cascade R-CNN | `feat/model/cascade-rcnn` |
| **팀원 A** | Cascade Mask R-CNN | `feat/model/cascade-mask-rcnn` |
| **팀원 B** | SOLOv2 | `feat/model/solov2` |
| **팀원 B** | 최신 모델 1종 (예: QueryInst) | `feat/model/queryinst` |

> 최신 2종은 팀 회의 후 확정. 후보: Mask2Former, QueryInst, SparseInst, SAM2

---

### 추가 개인 담당 (인프라 / 스크립트)

| 담당자 | 작업 | Branch명 |
|--------|------|----------|
| **본인 (jjh0709)** | 프로젝트 리드, merge 관리, merge_dataset.py | `dev` 직접 |
| **팀원 A** | 평가지표 스크립트 (`eval_metrics.py`) | `feat/eval` |
| **팀원 B** | gemini_augment.py 리팩토링 | `feat/augment/gemini` |

---

## 2. Branch 전략

### Branch 구조

```
main
 └── dev                          ← 통합 브랜치 (PR → dev → main)
      ├── feat/annotation          ← 어노테이션 공통 작업
      ├── feat/model/mask-rcnn     ← 팀원 A
      ├── feat/model/cascade-rcnn  ← 팀원 A
      ├── feat/model/cascade-mask-rcnn ← 팀원 A
      ├── feat/model/solov2        ← 팀원 B
      ├── feat/model/queryinst     ← 팀원 B
      ├── feat/model/mask-dino     ← 본인
      ├── feat/model/mask2former   ← 본인
      ├── feat/eval                ← 팀원 A
      └── feat/augment/gemini      ← 팀원 B
```

### 규칙

| 규칙 | 내용 |
|------|------|
| **직접 push 금지** | `main`, `dev`에 직접 push하지 않는다 |
| **PR → dev** | 모든 작업은 feature branch에서 PR로 `dev`에 merge |
| **PR → main** | dev 안정화 확인 후 리더(본인)가 `main`에 merge |
| **충돌 방지** | 담당 모델 디렉토리(configs/, training/ 하위) 외 파일은 PR 전에 dev pull 후 rebase |
| **브랜치 정리** | merge 완료된 feature branch는 삭제 |

### 브랜치 생성 방법

```bash
# dev 기준으로 feature 브랜치 생성
git checkout dev
git pull origin dev
git checkout -b feat/model/mask-rcnn

# 작업 완료 후
git add <files>
git commit -m "[feat] Mask R-CNN config 및 학습 스크립트 추가"
git push origin feat/model/mask-rcnn
# → GitHub에서 PR: feat/model/mask-rcnn → dev
```

### Commit 메시지 컨벤션

```
[feat]   새 기능 추가
[fix]    버그 수정
[docs]   문서 수정
[config] 설정 파일 변경
[exp]    실험 실행/결과 기록
[refactor] 코드 리팩토링
```

예시:
```
[feat] Mask R-CNN mmdetection config 추가 (Cable baseline)
[exp] 실험1-Cable Mask R-CNN baseline 결과 기록
[fix] SOLOv2 데이터 로더 경로 버그 수정
```

---

## 3. 디렉토리 구조 (전체)

```
VISION-Instance-Seg/
│
├── CLAUDE.md                    ← Claude Code 자동 인식
├── TEAMWORK.md                  ← 이 파일 (팀 협업 가이드)
├── README.md                    ← 프로젝트 소개
├── RULE.md                      ← 세부 규칙 (미작성 시 추가 예정)
├── requirements.txt
├── .gitignore
│
├── docs/
│   ├── experiment_plan.md       ← 실험 설계 문서
│   ├── data_spec.md             ← 데이터 명세
│   └── legacy/                  ← 기존 코드 분석 결과
│       ├── annotation_tool_analysis.md
│       ├── gemini_augment_analysis.md
│       └── traditional_aug_analysis.md
│
├── configs/                     ← 모델별 학습 설정 (담당자가 관리)
│   ├── mask_rcnn/               ← 팀원 A 담당
│   │   ├── cable_baseline.py
│   │   ├── cable_trad250.py
│   │   └── ...
│   ├── cascade_rcnn/            ← 팀원 A 담당
│   ├── cascade_mask_rcnn/       ← 팀원 A 담당
│   ├── solov2/                  ← 팀원 B 담당
│   ├── queryinst/               ← 팀원 B 담당
│   ├── mask_dino/               ← 본인 담당
│   └── mask2former/             ← 본인 담당
│
├── data/                        ← ⛔ .gitignore (서버에 직접 배치)
│   ├── Cable/{train,test}/{images/, annotations.json}
│   ├── Screw/{train,test}/{images/, annotations.json}
│   └── Casting/{train,test}/{images/, annotations.json}
│
├── data_augmented/              ← ⛔ .gitignore (서버에 직접 배치)
│   ├── Cable/{gen_ai,traditional_aug}/{images/, annotations.json}
│   ├── Screw/{gen_ai,traditional_aug}/{images/, annotations.json}
│   └── Casting/{gen_ai,traditional_aug}/{images/, annotations.json}
│
├── data_merged/                 ← ⛔ .gitignore (merge_dataset.py 출력)
│   ├── exp1_cable_baseline/
│   ├── exp1_cable_+50/
│   └── ...
│
├── scripts/
│   ├── augmentation/
│   │   ├── gemini_augment.py    ← 팀원 B 담당
│   │   ├── traditional_augment.py
│   │   └── prompts/
│   │       ├── cable_prompt.txt
│   │       ├── screw_prompt.txt
│   │       └── casting_prompt.txt
│   ├── data_utils/
│   │   ├── merge_dataset.py     ← 본인 담당
│   │   ├── rename_val_to_test.py
│   │   ├── convert_format.py
│   │   └── validate_annotations.py
│   └── evaluation/
│       └── eval_metrics.py      ← 팀원 A 담당
│
├── labeling_server/             ← 어노테이션 공통 작업
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── training/
│   ├── train.py
│   ├── test.py
│   └── run_experiments.sh
│
└── results/                     ← ⛔ .gitignore (서버에만 존재)
    ├── exp1_genai_scale/        ← 실험 1 결과
    │   ├── cable_baseline/
    │   ├── cable_+50/
    │   └── summary.csv
    ├── exp2_aug_comparison/     ← 실험 2 결과
    └── exp3_model_comparison/   ← 실험 3 결과 (7종 모델)
        ├── mask_rcnn/
        ├── cascade_rcnn/
        ├── cascade_mask_rcnn/
        ├── solov2/
        ├── queryinst/
        ├── mask_dino/
        └── mask2former/
```

---

## 4. 데이터 경로 규칙

### 절대 경로 (서버 기준)
```
/home/jjh0709/gitrepo/VISION-Instance-Seg/
```

### data / data_augmented — Git 추적 안 함
- 서버에 직접 배치, Git push 하지 않음
- 팀원이 작업할 경우 **scp 또는 서버 직접 접속**으로 데이터 공유
- 데이터 변경 시 `progress.md` 업데이트

### configs/ — 모델별 경로 패턴
```python
# config 파일 내 data_root 설정 예시
data_root = '/home/jjh0709/gitrepo/VISION-Instance-Seg/data_merged/exp3_cond1/'
```
- 절대 경로 대신 환경변수 또는 상대 경로 사용 권장:
```python
import os
BASE = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(BASE, '../../data_merged/exp3_cond1/')
```

### results/ — 실험 결과 기록 형식
각 실험 디렉토리 안에 `result.md` 작성:
```markdown
# 실험명: exp3 Mask R-CNN / Cond1 (원본 전체)

- 날짜: 2026-03-XX
- 담당: 팀원 A
- 모델: Mask R-CNN
- 데이터 조건: 원본 전체 (Cable 26장, Screw 57장, Casting 54장)
- Epoch: 50
- mAP@0.5: 0.XXX
- mAP@0.5:0.95: 0.XXX
- mAR: 0.XXX
```

---

## 5. 앞으로 할 일 (단계별)

### Phase 1 — Annotation 완료 (전원 공동)
- [ ] Screw gen_ai 데이터 입수 → `data_augmented/Screw/gen_ai/` 배치
- [ ] Casting gen_ai 데이터 입수 → `data_augmented/Casting/gen_ai/` 배치
- [ ] labeling_server로 Screw / Casting gen_ai annotation 검수·수정
- [ ] Screw / Casting traditional_aug 생성
  ```bash
  python scripts/augmentation/traditional_augment.py --category Screw --n_augment 2750
  python scripts/augmentation/traditional_augment.py --category Casting --n_augment 2750
  ```

### Phase 2 — 인프라 준비 (담당자)
- [ ] `merge_dataset.py` 작성 (본인) → 실험 조건별 데이터 병합
- [ ] `eval_metrics.py` 완성 (팀원 A) → mAP, mAR 자동 출력
- [ ] `gemini_augment.py` 리팩토링 (팀원 B)

### Phase 3 — 모델 Config 작성 (각 담당자)
- [ ] 각 담당 모델 mmdetection config 작성 (configs/ 하위)
- [ ] `training/train.py` / `run_experiments.sh` 에 모델 연동

### Phase 4 — 실험 1 (생성AI 증강 규모)
- Cable 대상, Mask R-CNN + Cascade Mask R-CNN 사용
- 조건: Baseline / +50 / +100 / +150 / +200 / +250

### Phase 5 — 실험 2 (증강 방법 비교)
- 5가지 데이터 조건 비교
- 사용 모델: 팀 결정

### Phase 6 — 실험 3 (7종 모델 비교)
- 각 담당자가 본인 모델 실험 실행
- 동일한 데이터 조건 3가지로 진행
- `results/exp3_model_comparison/` 에 결과 통일 형식으로 기록

---

## 6. 팀원 온보딩 체크리스트

팀원이 처음 합류할 때 확인할 사항:

```bash
# 1. 레포 clone
git clone https://github.com/Wlsghdh/VISION-Instance-Seg.git
cd VISION-Instance-Seg
git checkout dev

# 2. 본인 branch 생성 (예: 팀원 A)
git checkout -b feat/model/mask-rcnn

# 3. 필수 문서 읽기
#    - CLAUDE.md (프로젝트 컨텍스트)
#    - TEAMWORK.md (이 파일)
#    - docs/experiment_plan.md (실험 계획)
#    - progress.md (현재 진행 상황)

# 4. 환경 설정
pip install -r requirements.txt
# mmdetection 설치는 guide.md 참고

# 5. 데이터 확인 (서버 직접 접속)
ls /home/jjh0709/gitrepo/VISION-Instance-Seg/data/
ls /home/jjh0709/gitrepo/VISION-Instance-Seg/data_augmented/
```

---

## 7. 주의사항 요약

| ⛔ 하지 말 것 | ✅ 해야 할 것 |
|--------------|-------------|
| `main`, `dev`에 직접 push | feature branch → PR → dev |
| `data/`, `data_augmented/` Git add | .gitignore 확인 후 커밋 |
| test 데이터 변경 | test 데이터는 절대 수정 금지 |
| 결과 파일 커밋 (`results/`) | results는 서버에만 보관 |
| 모델 가중치 커밋 (`*.pth`, `*.bin`) | .gitignore에 포함됨 |
| 다른 사람 담당 config 수정 | 본인 담당 모델 디렉토리만 수정 |

---

> 마지막 업데이트: 2026-02-26
> 담당: jjh0709
