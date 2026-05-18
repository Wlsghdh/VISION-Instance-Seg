# 실험 2~3 담당자 분배 및 실행 가이드

---

## 1. 전체 구조

```
실험 2 (Phase 1): 전통 vs GenAI 비교    →  18회 (주진호 담당)
실험 3 (Phase 2): (원본+GenAI)+전통 N배  → 210회 (3명 분배)
                                         ─────────
                                   합계   228회
```

---

## 2. 실험 조건 요약

### 실험 2: cond1~3 (2종 모델)

| 조건 | 원본/cls | GenAI/cls | 전통/cls |
|------|:---:|:---:|:---:|
| cond1 | 20 | 0 | 0 |
| cond2 | 20 | 0 | 125 |
| cond3 | 20 | 125 | 0 |

모델: Mask R-CNN, Cascade Mask R-CNN

### 실험 3: trad_1x~10x (7종 모델)

| N | 원본/cls | GenAI/cls | 전통/cls | 총/cls |
|:-:|:---:|:---:|:---:|:---:|
| 1x | 20 | 125 | 145 | 290 |
| 2x | 20 | 125 | 290 | 435 |
| 3x | 20 | 125 | 435 | 580 |
| 4x | 20 | 125 | 580 | 725 |
| 5x | 20 | 125 | 725 | 870 |
| 6x | 20 | 125 | 870 | 1015 |
| 7x | 20 | 125 | 1015 | 1160 |
| 8x | 20 | 125 | 1160 | 1305 |
| 9x | 20 | 125 | 1305 | 1450 |
| 10x | 20 | 125 | 1450 | 1595 |

모델 7종: mask_rcnn, cascade_mask_rcnn, maskdino, mask2former, cascade_rcnn, solov2, rtmdet_ins

---

## 3. 담당자 분배

### 양진우 — GPU 0

| 실험 | N 범위 | 모델 | 학습 횟수 | 예상 시간 |
|------|:------:|------|:--------:|:---------:|
| exp3 | N=1~4 | mask_rcnn, maskdino, mask2former | **36회** | ~9h |

```bash
# tmux에서 실행
tmux new -s yjw
bash scripts/run_exp3_yjw.sh 0 Inclusoes Dirty impurities
```

### 주진호 — GPU 1

| 실험 | N 범위 | 모델 | 학습 횟수 | 예상 시간 |
|------|:------:|------|:--------:|:---------:|
| exp2 | Phase 1 전체 | mask_rcnn, cascade_mask_rcnn | **18회** | ~3h |
| exp3 | N=5~7 | cascade_mask_rcnn, cascade_rcnn | **18회** | ~6h |
| | | **소계** | **36회** | **~9h** |

```bash
# tmux에서 실행
tmux new -s jjh
bash scripts/run_exp3_jjh.sh 1 Inclusoes Dirty impurities
```

### 임대윤 — GPU 3

| 실험 | N 범위 | 모델 | 학습 횟수 | 예상 시간 |
|------|:------:|------|:--------:|:---------:|
| exp3 | N=8~10 | solov2, rtmdet_ins | **18회** | ~6h |

```bash
# tmux에서 실행
tmux new -s ldy
bash scripts/run_exp3_ldy.sh 3 Inclusoes Dirty impurities
```

---

## 4. 남은 모델 (2차 분배)

위 1차 분배로 7종 × 10N × 3defect = 210회 중 **72회만 커버**됩니다.
남은 138회는 1차가 끝난 사람부터 추가 배정:

### 미배정 조합

| N 범위 | 미배정 모델 |
|:------:|-----------|
| N=1~4 | cascade_mask_rcnn, cascade_rcnn, solov2, rtmdet_ins |
| N=5~7 | mask_rcnn, maskdino, mask2former, solov2, rtmdet_ins |
| N=8~10 | mask_rcnn, cascade_mask_rcnn, maskdino, mask2former, cascade_rcnn |

**1차 끝나면 알려주세요 → 2차 스크립트 생성해드립니다.**

또는 각자 직접 실행:
```bash
# 예: 양진우가 1차 끝나고 N=1~4 cascade_mask_rcnn 추가
CUDA_VISIBLE_DEVICES=0 conda run -n jjh python -m training.train \
  --category Inclusoes --experiment exp3 --condition trad_1x --model cascade_mask_rcnn \
  --max-epochs 200 --patience 10
```

---

## 5. 사전 준비 (실행 전 필수)

### 5-1. dev 브랜치 pull

```bash
cd /home/jjh0709/gitrepo/VISION-Instance-Seg
git checkout dev
git pull origin dev
```

### 5-2. 3개 defect 확정

스크립트의 인자로 넘깁니다. 예시 (변경 가능):
```
Inclusoes  — Casting에서 추출, GenAI +128%
Dirty      — Console에서 추출, GenAI +151%
impurities — Wood에서 추출, GenAI +7%
```

다른 defect로 바꾸려면 인자만 변경:
```bash
bash scripts/run_exp3_yjw.sh 0 thunderbolt Porosity Dirty
```

사용 가능한 개별 defect 이름:
```
Inclusoes, Rechupe, Collision, Dirty, Gap, Scratch,
Chip, PistonMiss, Porosity, RCS, impurities, pits
(+ Cable, Screw는 원래 1클래스라 그대로 사용)
```

### 5-3. 전통 증강 데이터 확인

N=10 (최대) 기준 1,450장/cls 필요:
```bash
# 확인 명령어
python -m training.data_pipeline --category Inclusoes --experiment exp3 --condition trad_10x
```

부족하면 `traditional_augment.py`로 생성 필요.

### 5-4. tmux 사용법

```bash
tmux new -s 이름         # 세션 생성
# ... 스크립트 실행 ...
Ctrl+B, D               # 세션 분리 (백그라운드 유지)
tmux attach -t 이름      # 다시 연결
tmux ls                  # 세션 목록
```

---

## 6. 체크리스트

| # | 항목 | 상태 |
|:-:|------|:----:|
| 1 | dev pull | ⬜ |
| 2 | 3개 defect 확정 | ⬜ |
| 3 | 전통 증강 데이터 확인/생성 | ⬜ |
| 4 | mmdet 모델 smoke test (임대윤) | ⬜ |
| 5 | 양진우 exp3 N=1~4 실행 | ⬜ |
| 6 | 주진호 exp2 + exp3 N=5~7 실행 | ⬜ |
| 7 | 임대윤 exp3 N=8~10 실행 | ⬜ |
| 8 | 2차 분배 실행 | ⬜ |
| 9 | 결과 취합 + md 작성 | ⬜ |

---

## 7. 결과 확인

```bash
# 전체 결과 확인
cat results/evaluation/results.json | python -m json.tool | grep segm_AP

# 특정 defect 결과
grep "Inclusoes" results/logs/exp3_*.log | grep "\[OK\]"

# quota 확인
quota -s
```

---

## 8. 문제 발생 시

| 문제 | 해결 |
|------|------|
| CUDA OOM | `--batch-size 8` 또는 `4`로 줄이기 |
| mmdet import 에러 | `conda activate jjh` 확인 |
| MaskDINO/Mask2Former 에러 | CUDA ops 빌드 확인 |
| 디스크 quota | `find results/training -name "model_*.pth" -not -name "model_final.pth" -delete` |
| 전통 증강 부족 | `python scripts/augmentation/traditional_augment.py` 실행 |
