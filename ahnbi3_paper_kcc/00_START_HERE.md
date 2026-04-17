# 🚀 시작 가이드 — ahnbi3에서 KCC 논문 작성 이어하기

**작성일**: 2026-04-17
**대상**: jjh0709 (ahnbi3 서버)
**최종 목표**: 한국정보과학회 2026 (KCC) 투고

---

## 📖 이 폴더 읽기 순서

```
00_START_HERE.md         ← 지금 여기 (전체 맵)
01_OVERVIEW.md           ← 프로젝트 개요 & 최종 논문 제목
02_EXP1_EXPLAINED.md     ← 실험 1 (GenAI 수량 스윕)
03_EXP2_EXPLAINED.md     ← 실험 2 (전통 vs GenAI vs 결합)
04_RESULTS_CURRENT.md    ← 지금까지 확보한 모든 수치
05_PAPER_STRATEGY.md     ← 논문 story + 제목 B + 리뷰어 대응
06_DATA_AND_MODELS.md    ← 데이터 구조 + 5개 모델 정보
07_AGENTS_GUIDE.md       ← Claude agent 5종 활용법
08_NEXT_TASKS.md         ← ahnbi3에서 할 작업 체크리스트
09_FAQ.md                ← 자주 막히는 점 + 해결
```

---

## ⚡ 가장 빠르게 논문 작업 시작하는 법

### 1. 코드·문서 최신화
```bash
cd ~/gitrepo/VISION-Instance-Seg
git pull origin dev
# 또는 (git 없으면)
git fetch origin dev && git checkout -f origin/dev -- docs/ ahnbi3_paper_kcc/ .claude/ CLAUDE.md
```

### 2. Claude Code 실행 & 첫 프롬프트
```bash
claude
```

Claude에게 아래를 통째로 복붙:

```
KCC 논문 작성 이어하겠습니다. ahnbi3_paper_kcc/00_START_HERE.md 부터
09_FAQ.md 까지 전부 읽어서 현재 상태 파악해줘. 그리고 논문 제목 B
("전통 증강의 한계와 생성형 AI 증강의 시너지 — 소규모 결함 데이터셋의
공정 비교 연구") 기준으로 §1 서론 초안을 docs/paper/draft/01_intro.md
bullet point를 문단으로 확장해서 작성해줘. 완성 후 paper-doctor agent로
리뷰 의뢰.
```

---

## 🎯 현재 상황 한 줄 요약

- ✅ **Exp2 cond1~cond4_10x 전체 완료** (핵심 결과 확보)
- ✅ **Exp1_3cls Phase A 완료** (baseline + genai_75 + genai_125)
- 🟢 **cond4_8x cascade_rcnn** usw에서 학습 중 (~8h 남음)
- 🟢 **cond4_8x solov2** ahnbi3 tmux 에서 학습 중 (~5h 남음)
- ❌ **cond4_8x rtmdet_ins / maskdino / mask2former** 아직 안 돌림
- 📝 **논문 본문 작성 대기** (draft skeleton만 있음)

### 핵심 수치 (pres기반)
| 구분 | Mask R-CNN | Cascade Mask R-CNN |
|------|:---:|:---:|
| baseline (원본만) | 10.30 | 11.26 |
| +전통 125 | 8.54 ⬇ | 10.36 |
| +GenAI 125 | 12.18 ⬆ | 11.78 |
| **cond4_8x** (최적) | **14.13** | **16.41** ⭐ |

→ **+5.15 AP (11.26 → 16.41)**이 논문 메인 수치.

---

## 📂 프로젝트 전체 구조

```
VISION-Instance-Seg/
├── ahnbi3_paper_kcc/       ← 👀 이 폴더 (KCC 논문용 모든 정보)
├── CLAUDE.md               ← 프로젝트 규칙 (자동 로드됨)
├── .claude/agents/         ← 5개 agent
├── docs/
│   ├── paper/              ← 논문 파이프라인 (figure/table/draft)
│   ├── references/refs.bib ← BibTeX (18개 시드)
│   ├── result_summary_kcc_paper.md
│   └── experiment_plans/   ← v2 계획서 (검증 통과)
├── training/               ← 학습 코드
├── scripts/                ← 실행 스크립트
├── data/, data_augmented/  ← 데이터 (gitignore)
└── results_github/         ← 팀 공유 결과 (git 추적)
    ├── exp1_3cls/          ← jjh 본 실험
    └── exp2/               ← cond1~cond4_10x 전체
```

---

## 🔑 제일 중요한 3가지

1. **논문 제목 B 확정**:
   > 전통 증강의 한계와 생성형 AI 증강의 시너지 — 소규모 결함 데이터셋의 공정 비교 연구

2. **Story line (3문장)**:
   - 소규모 데이터에서 전통 증강 단독은 **해로움**
   - GenAI 증강이 **소량으로도 효과**
   - 결합 시 **시너지** (+5.15 AP)

3. **다음 작업**: `08_NEXT_TASKS.md` 체크리스트 보고 하나씩 처리

---

궁금한 거 생기면 `09_FAQ.md` 확인. 그래도 해결 안 되면 Claude에게 질문.
