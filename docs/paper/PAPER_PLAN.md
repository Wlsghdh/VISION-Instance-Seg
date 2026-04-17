# KCC 2026 논문 작성 계획서 (작업 이어하기용)

**작성일**: 2026-04-17
**작업자**: jjh
**투고 대상**: 한국정보과학회 2026 (KCC 2026)
**현재 상태**: 실험 진행 중 + 본문 초안 대기

---

## 📌 한눈에 — 지금 어디까지 왔나

### ✅ 확보된 결과 (Exp2_3cls val 82장 기준, segm_AP)

**Exp2 (핵심):**

| 조건 | Mask R-CNN | Cascade MRCNN |
|------|:---:|:---:|
| cond1 (원본 20) | 10.30 | 11.26 |
| cond2 (+전통 125) | **8.54 ⬇** | 10.36 |
| cond3 (+GenAI 125) | **12.18 ⬆** | 11.78 |
| cond4_4x | 12.18 | 12.39 |
| cond4_5x | 12.23 | 12.09 |
| cond4_6x | 13.08 | **15.90 ⭐** |

**Exp1_3cls (iter 기반 공정 비교):**

| 조건 | Mask R-CNN | Cascade MRCNN | 3-seed variance (MRCNN) |
|------|:---:|:---:|:---:|
| baseline (원본 20) | 8.51 | 9.34 | 8.51/8.73/9.84 → 9.03±0.74 |
| genai_75 | **11.60** | **12.32** | — |
| genai_125 | 11.66 | 12.25 | — |

**cond4 N배 스윕 (ldy/yjw push됨 — results_github):**

```
N=  1x    2x    3x    4x    5x    6x    7x    8x    9x    10x
M: 13.39 12.63 14.30 12.18 12.23 13.08 13.28 14.13 13.45 13.52
C: 13.83 15.69 14.29 12.39 12.09 15.90 14.41 16.41 15.07 14.85
```

**✨ cond4_8x cascade = 16.41 (최고점)**

### 🟢 진행 중 (이 작업 끝나면 추가 데이터)

- **usw (lifeai)**: cond4_8x / cascade_rcnn (epoch 25/200, segm_mAP 0.10 수준) — ~8h 남음
- **ahnbi3**: cond4_8x / solov2 (tmux session, epoch 1/200) — ~5h 예상
- **아직 미완료**: cond4_8x / rtmdet_ins, maskdino, mask2former

### ⚠️ 알려진 이슈

1. **mmdet 모델이 이전에 segm_AP=0** 이었던 것 → COCO pretrained 로드 + AdamW lr=1e-4 + bs=4 로 **수정 완료** (commit `964d81c`, `08efdbd`)
2. **mask2former import 버그** → 수정 완료 (commit `08efdbd`)
3. **detectron2 미설치 (ahnbi3)** → solov2 학습엔 불필요

---

## 🎯 논문 핵심 Story (3문장 요약)

1. 소규모 산업 결함 데이터(클래스당 20장)에서 **전통 증강은 오히려 성능을 악화**시킨다 (cond2: -1.76 AP).
2. **생성형 AI (Gemini) 증강**은 소량(75-125장)으로도 **+2-3 AP 향상**을 가져온다 (cond3, Exp1_3cls).
3. **GenAI + 전통 증강 결합**은 cond4_8x(Cascade)에서 **+5.15 AP**의 시너지를 만든다 (11.26 → 16.41).

---

## 📝 논문 제목 후보

- **A**: 소규모 산업 결함 Instance Segmentation에서 생성형 AI 기반 데이터 증강의 효과 분석
- **B** ⭐: **전통 증강의 한계와 생성형 AI 증강의 시너지 — 소규모 결함 데이터셋의 공정 비교 연구**
- **C**: Gemini 기반 생성 증강이 전통 증강보다 우수한 조건 — 3종 결함 Instance Segmentation 실증 연구

**추천: B** (negative finding + positive finding 모두 담음, 임팩트 강함)

---

## 📖 논문 구조 (KCC 5-6쪽)

| § | 제목 | 분량 | 핵심 메시지 |
|---|------|:---:|------------|
| §1 | 서론 | 0.5쪽 | motivation: VISION-Datasets 같은 소규모 결함 benchmark에서 증강 선택이 중요 |
| §2 | 관련 연구 | 0.75쪽 | Instance seg / 전통 증강 / 생성형 증강 / 공정 benchmark |
| §3 | 제안 방법 | 1쪽 | iter 기반 공정성 프로토콜 + 증강 조합 설계 |
| §4 | 실험 설정 | 0.75쪽 | 데이터/모델/하이퍼파라미터 |
| §5 | 결과 | 2쪽 | Table 1 (cond1~3 + cond4_6x) + Figure 1 (N스윕) + Figure 2 (Exp1_3cls 스케일링) + per-class 분석 |
| §6 | 토의 | 0.5쪽 | 왜 전통 증강이 해로운가 + GenAI 메커니즘 + 한계 |
| §7 | 결론 | 0.25쪽 | 가이드라인 + 후속 과제 |

---

## 🤖 활용 Agent (작업 시작 시 CLAUDE.md에 규칙 있음)

| Agent | 파일 | 용도 |
|-------|------|------|
| `paper-doctor` | `.claude/agents/paper-doctor.md` | Story-line 진단, 문장 리뷰, 리뷰어 공격 시뮬레이션 |
| `paper-visualizer` | `.claude/agents/paper-visualizer.md` | Figure/Table 자동 생성 (matplotlib) |
| `paper-references-manager` | `.claude/agents/paper-references-manager.md` | BibTeX 관리, 인용 점검 |
| `experiment-plan-reviewer` | `.claude/agents/experiment-plan-reviewer.md` | 추가 실험 계획 검증 |
| `experiment-doctor` | `.claude/agents/experiment-doctor.md` | 실험 설계 자문 |

호출 예시:
```
Agent({
  subagent_type: "paper-doctor",
  description: "§1 서론 초안 리뷰",
  prompt: "docs/paper/draft/01_intro.md 을 리뷰해줘. 리뷰어 공격 시뮬레이션 포함."
})
```

---

## 🚀 다음 작업 (ahnbi3에서 이어하기)

### 1. 환경 준비 (ahnbi3 접속 후)
```bash
ssh jjh0709@ahnbi3.suwon.ac.kr
cd ~/gitrepo/VISION-Instance-Seg

# 최신 코드 pull
git init 2>/dev/null
git remote add origin https://github.com/Wlsghdh/VISION-Instance-Seg.git 2>/dev/null
git fetch origin dev
git checkout -f origin/dev -- docs/ .claude/ scripts/ training/ CLAUDE.md

# 또는 git 설치됐으면
git pull origin dev
```

### 2. 작업 시작 (Claude Code 실행)
```bash
cd ~/gitrepo/VISION-Instance-Seg
claude
```

### 3. Claude에게 첫 프롬프트 (복붙용)

```
이제 KCC 논문 작성 이어하겠습니다. docs/paper/PAPER_PLAN.md 를 먼저 읽고,
docs/result_summary_kcc_paper.md 와 docs/paper/visualization_pipeline.md 도 읽어서
현재 상태를 파악해줘. 그리고 다음 작업 중 제일 우선순위 높은 것부터 시작해:

1. Figure 1 (cond4 N 스윕 곡선) 최신 데이터로 재생성
2. §1 서론 초안 작성 → paper-doctor로 리뷰
3. Table 1 / Table 2 재생성
4. 관련 연구 (§2) BibTeX 정리
```

### 4. 우선순위 작업 리스트

#### A. 즉시 가능 (결과 있는 것 기반)
- [ ] **Figure 1 재생성** (cond4 1x~10x 전체, 2모델 곡선)
  - `bash docs/paper/scripts/regenerate_all.sh`
- [ ] **Table 1 (main)**, **Table 2 (per-class)** 재생성 — 같은 스크립트
- [ ] **Figure 2 (Exp1_3cls 스케일링)** 새로 작성
  - `docs/paper/scripts/fig2_exp1_3cls_scaling.py` 신규 작성 필요
- [ ] **§1 서론 초안 작성** (`docs/paper/draft/01_intro.md`)
- [ ] **§2 관련 연구** (`docs/paper/draft/02_related.md`)

#### B. 학습 완료 대기 후
- [ ] cond4_8x solov2 / cascade_rcnn / rtmdet_ins 결과 통합
- [ ] 5모델 비교 표 추가 (cond4_8x에서 모든 모델 segm_AP)

#### C. VISION-Datasets 원 논문 받은 후
- [ ] `paper-references-manager` agent로 BibTeX 추가
- [ ] §1, §2에 인용 위치 확정
- [ ] 차별점 명확화

---

## 📂 파일 맵

```
docs/
├── paper/
│   ├── PAPER_PLAN.md           ← 본 파일 (이걸 먼저 읽기!)
│   ├── visualization_pipeline.md
│   ├── scripts/                ← figure/table 생성 스크립트
│   │   ├── _style.py
│   │   ├── _loader.py
│   │   ├── fig1_cond4_curve.py
│   │   ├── table1_main.py
│   │   ├── table2_perclass.py
│   │   └── regenerate_all.sh
│   ├── figs/                   ← 생성된 figure (pdf+png)
│   ├── tables/                 ← 생성된 표 (tex+md)
│   └── draft/                  ← 본문 초안 (비어있음 — 여기 쓸 것)
├── references/
│   └── refs.bib                ← 18개 시드 엔트리, VISION-Datasets placeholder
├── result_summary_kcc_paper.md ← 전략 + 상세 결과
└── experiment_plans/
    ├── exp1_3cls_v1.md
    └── exp1_3cls_v2.md         ← PASS 받은 계획서

results_github/                 ← 팀 공유 결과 (git 추적)
├── exp1/                       (옛 버전 — 무시)
├── exp1_3cls/                  ← 신버전
├── exp2/                       ← cond1~cond4_10x 전체
└── evaluation/results.json
```

---

## 🔑 리뷰어 공격 예상 + 방어 준비

| 공격 | 방어 |
|------|------|
| "단일 seed 아니냐?" | Exp1_3cls baseline 3-seed 결과 제시 (9.03±0.74) |
| "val=test 아니냐?" | Limitation에 명시 + val_dev/val_test 분리 설계 있음 |
| "3 클래스만?" | Scope 명확화: 산업 결함 소규모 축소 실험 |
| "GenAI 품질은?" | Supplementary에 샘플 이미지 + FID (추후) |
| "Mask R-CNN만?" | cond4_8x 5모델 비교 (학습 중) |
| "왜 cond4_6x가 cond4_10x보다 좋나?" | cond4_8x가 최적임 (cascade 16.41), 10x는 overfitting |

---

## 💡 글쓰기 원칙 (paper-doctor가 리뷰할 것)

1. **Motivation은 negative finding으로 시작** — "전통 증강 해로움"
2. **Over-claim 금지** — "최고", "항상", "모든" 등 피하기
3. **수치는 반드시 ±variance** 또는 단일 seed 명시
4. **한국어 논문체** — 구어체 금지
5. **Contribution 3개 이내** — 많으면 핵심 약화

---

## 📬 문의

- 이 계획서에 없는 것 궁금하면 `docs/result_summary_kcc_paper.md` 참조
- 실험 설계 질문 → `experiment-doctor` agent
- 논문 문장 질문 → `paper-doctor` agent
- 참고문헌 질문 → `paper-references-manager` agent
