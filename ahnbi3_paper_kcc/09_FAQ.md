# 09. FAQ — 자주 막히는 점 + 해결

## Q1. git pull 안 돼요

**A.** ahnbi3에 git이 없을 수 있음. conda로 설치:
```bash
conda activate jjh
conda install git -y
```

또는 수동 pull 대안:
```bash
cd ~/gitrepo/VISION-Instance-Seg
git init
git remote add origin https://github.com/Wlsghdh/VISION-Instance-Seg.git
git fetch origin dev
git checkout -f origin/dev -- docs/ ahnbi3_paper_kcc/ .claude/ scripts/ training/ CLAUDE.md
```

---

## Q2. Claude Code가 없어요 (ahnbi3)

**A.** Node.js + npm으로 설치:
```bash
conda activate jjh
conda install nodejs -y
npm install -g @anthropic-ai/claude-code --prefix ~/.npm-global
export PATH=~/.npm-global/bin:$PATH
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
claude --version
```

첫 실행 시 Anthropic 로그인 (브라우저 인증).

---

## Q3. Figure 생성 시 `No module named 'matplotlib'`

**A.** conda env 활성화 확인:
```bash
conda activate jjh
which python
# → /home/jjh0709/.conda/envs/jjh/bin/python 이어야 함
```

또는 `bash regenerate_all.sh` 에서 `conda run -n jjh python ...` 사용:
```bash
# docs/paper/scripts/regenerate_all.sh 내용 확인
cat docs/paper/scripts/regenerate_all.sh
```

---

## Q4. mmdet 모델이 segm_AP=0 나와요

**A.** 이미 수정됨 (commit 964d81c, 08efdbd). 최신 코드 pull하세요.

원인:
1. `cfg.load_from = None` → COCO pretrained 미로드
2. AdamW lr=0.0015 → 너무 높음

수정 후:
- `training/config.py` mmdet 3종에 `weights` URL + `hyperparams` (lr=1e-4, bs=4) 추가
- `training/adapters/mmdet_adapter.py` pretrained 자동 로드

기존 segm_AP=0 결과 디렉토리는 자동 삭제 후 재학습:
```bash
bash scripts/run_exp2_cond4_8x_mmdet.sh 0
```

---

## Q5. Mask2Former import 에러

**A.** 이미 수정됨 (commit 08efdbd). `detectron2_adapter.py`에서 `importlib.spec_from_file_location` 대신 패키지 임포트로 교체:
```python
from mask2former.config import add_maskformer2_config
import mask2former.modeling
from mask2former.maskformer_model import MaskFormer
```

→ `from . import data` 건너뛰어 데이터셋 중복 등록 방지.

---

## Q6. 학습 도중 "Killed" 떠서 죽었어요

**A.** 두 가지 원인 가능:
1. **SSH 세션 종료로 SIGHUP** → tmux 사용 (`tmux new -s solov2`)
2. **OOM** → `free -h` 로 메모리 확인. 드물게 시스템 RAM 부족

tmux 쓰면 세션 끊어도 학습 계속:
```bash
tmux new -s 세션이름
# 학습 명령 실행
# Ctrl+B, D 로 detach
tmux attach -t 세션이름 # 다시 들어가기
```

---

## Q7. 터미널이 긴 명령어를 자꾸 쪼개요

**A.** bash 라인 wrap 이슈. 해결:

1. **짧은 명령어 분할**:
```bash
cd ~/gitrepo/VISION-Instance-Seg
CUDA_VISIBLE_DEVICES=0 python -m training.train ...
```

2. **스크립트 파일로 저장**:
```bash
echo 'CUDA_VISIBLE_DEVICES=0 python ...' > run.sh
bash run.sh
```

3. **bracketed paste 모드** 확인 (아예 잘 붙는 터미널 사용)

---

## Q8. Agent 호출 시 "agent type not found"

**A.** 새 세션 시작해야 함. Claude Code 재시작:
```bash
# /exit 으로 종료 후
claude
```

`.claude/agents/*.md` 는 세션 시작 시에만 로드됨.

---

## Q9. val=test 리뷰어 공격 어떻게?

**A.** Limitation 섹션에 명시:
> "본 연구는 val 82장을 모델 선택 및 최종 평가에 공통 사용하여 leakage 위험이 존재한다. 다만 조건 간 상대 비교는 동일 val셋에서 이루어져 상대 순위는 신뢰할 수 있다."

향후 과제: val_dev/val_test 분리 (설계는 `stratify_exp1_3cls_val.py`에 있음)

---

## Q10. 단일 seed 리뷰어 공격?

**A.** 이미 준비: Exp1_3cls baseline mask_rcnn 3-seed 결과
- seed 42: 8.51
- seed 43: 8.73
- seed 44: 9.84
- 평균 9.03 ± 0.74

→ "seed variance는 ±0.74 AP 수준이며, 본 연구의 핵심 비교 (예: cond4_8x cascade = +5.15 AP)는 이 variance를 충분히 넘어선다"

---

## Q11. 전통 증강이 왜 해로운가 Discussion은?

**A.** 가설 3가지 (`05_PAPER_STRATEGY.md` 참조):

1. **Overfitting 가속**: 원본 20장 × 증강 7배 = 여전히 20개 패턴의 변형 → 같은 패턴에 더 과적합
2. **Label noise**: 증강 과정에서 mask 경계 불일치 발생 가능 (예: rotate 후 ROI 벗어남)
3. **Domain shift**: 전통 증강된 이미지가 원본 val 분포와 괴리 (color jitter가 과할 때)

---

## Q12. VISION-Datasets 원 논문을 아직 못 받았어요

**A.** 임시로 placeholder로 두고 진행:
```
docs/references/refs.bib:
% TODO: VISION-Datasets 원 논문 — 사용자에게 PDF/arXiv ID 받으면 paper-references-manager가 채움
```

본문에서 `\cite{vision_datasets_2023}` 로 인용 위치만 잡아두기. 나중에 키 확정되면 일괄 교체.

---

## Q13. Claude가 agent를 잘못 쓰고 있어요

**A.** prompt에 구체성 부족할 때 그럼. 아래처럼 명시:
- 입력 파일 경로
- 원하는 출력 파일 경로
- 분량/스타일 제약
- 참고할 문서

예:
❌ "§1 써줘"
✅ "docs/paper/draft/01_intro.md의 bullet point 4개를 KCC 논문체 한국어 본문 0.5쪽 분량으로 확장. 제목 B 기준, Motivation은 VISION-Datasets 소규모 결함 benchmark 문제. paper-doctor로 리뷰 후 반영."

---

## Q14. 논문 전체 draft를 한 파일로 합치고 싶어요

**A.** 간단 스크립트:
```bash
cd docs/paper/draft
cat 01_intro.md 02_related.md 03_method.md 04_setup.md 05_results.md 06_discussion.md 07_conclusion.md > ../full_draft.md
```

또는 pandoc으로 PDF:
```bash
pandoc 01_intro.md 02_related.md ... -o paper.pdf
```

---

## Q15. 결과가 바뀌었어요 (새 학습 완료) — Figure 갱신?

**A.** 재생성 스크립트 실행:
```bash
# 1. results_github 갱신
python scripts/sync_results_to_github.py --exp exp2

# 2. Figure/Table 갱신
cd docs/paper/scripts
bash regenerate_all.sh

# 3. 커밋 & push
cd ~/gitrepo/VISION-Instance-Seg
git add -A
git commit -m "[update] Figure/Table 재생성"
git push origin dev
```

---

## Q16. paper-doctor가 너무 critical 해서 draft가 자꾸 퇴짜 맞아요

**A.** paper-doctor는 의도적으로 엄격. 그러나 실제 구현 시 다음 균형:
- **High/Critical 지적**: 반드시 수정
- **Medium**: 가능하면 반영
- **Low/Nice-to-have**: 선택

전부 반영하면 완벽주의에 빠져서 작성이 안 됨. 주요 지적만 반영하고 넘어가도 OK.

---

## 🆘 그래도 막힐 때

1. **이 폴더 다 읽기**: `00_START_HERE.md` ~ `09_FAQ.md`
2. **`CLAUDE.md` 읽기**: 프로젝트 규칙
3. **`docs/result_summary_kcc_paper.md`**: 더 상세한 결과
4. **Claude에게 직접 질문**: 구체적인 파일 경로와 함께
5. **usw 서버 확인**: jjh의 원본 작업 서버에서 로그 확인 가능

---

## 📝 마지막 팁

- **커밋 자주**: 작업 단위마다 git add + commit + push
- **tmux 활용**: 긴 학습은 반드시 tmux 안에서
- **결과 sync**: 학습 후 `scripts/sync_results_to_github.py --exp exp2`
- **agent 활용**: 혼자 고민하지 말고 paper-doctor/visualizer/references-manager 적극 사용

논문 화이팅! 🎉
