# 07. Claude Agent 활용 가이드

## 🤖 5개 프로젝트 전용 Agent

프로젝트 `.claude/agents/` 에 있는 5개 agent. Claude Code 실행 시 자동 인식됨.

| Agent 이름 | 파일 | 용도 | 언제 쓰나 |
|-----------|------|------|-----------|
| `paper-doctor` | `paper-doctor.md` | 논문 글쓰기 총괄 | §1~§7 초안 리뷰, 리뷰어 공격 시뮬레이션 |
| `paper-visualizer` | `paper-visualizer.md` | Figure/Table 자동 생성 | Figure 1~4, Table 1~3 생성/갱신 |
| `paper-references-manager` | `paper-references-manager.md` | BibTeX 관리 | 인용 추가/점검, VISION-Datasets 원 논문 추가 |
| `experiment-plan-reviewer` | `experiment-plan-reviewer.md` | 실험 계획 검증 | 새 실험 시작 전 필수 (CLAUDE.md 규칙) |
| `experiment-doctor` | `experiment-doctor.md` | 실험 설계 자문 | plan-reviewer가 FAIL 냈을 때 |

## 📝 paper-doctor (논문박사)

### 언제 쓰나
- 섹션 초안 완성 → 리뷰 의뢰
- "이 주장이 방어 가능한가?" 질문
- 전체 story-line 진단
- 리뷰어 공격 시뮬레이션

### 호출 예시

**§1 서론 리뷰:**
```
Agent({
  subagent_type: "paper-doctor",
  description: "§1 서론 초안 리뷰",
  prompt: "docs/paper/draft/01_intro.md 읽고 story-line 진단 + 리뷰어 공격 시뮬레이션. 특히 motivation이 충분히 좁혀져 있는지, 3개 contribution이 독립적인지 확인. 수정 문장 제안 포함."
})
```

**전체 draft 리뷰:**
```
Agent({
  subagent_type: "paper-doctor",
  description: "전체 논문 draft 진단",
  prompt: "docs/paper/draft/*.md 전부 읽고 논문 story-map 그려줘. 섹션 간 연결 부족한 곳, 분량 불균형, 누락된 내용 지적. KCC 5-6쪽 기준 분량 진단 포함."
})
```

**Claim 방어 준비:**
```
Agent({
  subagent_type: "paper-doctor",
  description: "주요 claim 방어 자료 생성",
  prompt: "논문 메인 claim은 'cond4_8x cascade = +5.15 AP'. 리뷰어 공격 5가지와 방어 논리 준비해줘. ahnbi3_paper_kcc/05_PAPER_STRATEGY.md 참고."
})
```

## 🎨 paper-visualizer

### 언제 쓰나
- Figure 1~4 생성/갱신
- Table 1~3 생성
- 결과 JSON → publication-quality 시각물 변환

### 호출 예시

**Figure 1 재생성 (데이터 갱신 반영):**
```
Agent({
  subagent_type: "paper-visualizer",
  description: "Figure 1 cond4 N 스윕 재생성",
  prompt: "docs/paper/scripts/fig1_cond4_curve.py 실행해서 figs/fig1_cond4_curve.{pdf,png} 갱신. results_github/exp2/cond4_*x/ 전체 데이터 반영. N=8 정점을 annotation으로 강조."
})
```

**Figure 2 새로 만들기:**
```
Agent({
  subagent_type: "paper-visualizer",
  description: "Figure 2 Exp1_3cls 스케일링",
  prompt: "docs/paper/scripts/fig2_exp1_3cls_scaling.py 새로 작성. X축: GenAI 수량 (0, 25, 50, 75, 100, 125), Y축: segm_AP, Line 2개 (mask_rcnn/cascade). 현재 jjh 결과(0/75/125)만 있고 yjw 결과(25/50/100)는 추후 추가. 미확보 점은 별도 스타일로 표시."
})
```

**Table 3 새로 만들기 (5모델 비교):**
```
Agent({
  subagent_type: "paper-visualizer",
  description: "Table 3 cond4_8x 5모델 비교",
  prompt: "cond4_8x에서 5~7모델 (mask_rcnn, cascade_mask_rcnn, maskdino, mask2former, cascade_rcnn, solov2, rtmdet_ins) 의 segm_AP, AP50, AP75, bbox_AP 비교 표. 미학습 모델은 '—' 표시. LaTeX + Markdown 동시 생성."
})
```

## 📚 paper-references-manager

### 언제 쓰나
- BibTeX 새 엔트리 추가
- 본문 인용 점검 (누락/미사용)
- VISION-Datasets 원 논문 정리

### 호출 예시

**VISION-Datasets 원 논문 추가:**
```
Agent({
  subagent_type: "paper-references-manager",
  description: "VISION-Datasets 원 논문 등록",
  prompt: "VISION-Datasets 원 논문 BibTeX 추가해줘. arXiv ID: [받으면 제공]. docs/references/refs.bib에 `vision_datasets_2023` 키로 추가 + §1 서론과 §2 관련연구에 인용 위치 제안."
})
```

**전체 인용 점검:**
```
Agent({
  subagent_type: "paper-references-manager",
  description: "본문 인용 점검",
  prompt: "docs/paper/draft/*.md 전체 읽고 인용된 refs.bib 키 목록 추출. 누락된 것 (본문에 있지만 bib 없음), 미사용 (bib에 있지만 본문 인용 없음) 리스트. 도메인 관점에서 필수 인용 빠진 것 있으면 지적."
})
```

## 🔬 experiment-plan-reviewer

### 언제 쓰나
- 새 실험(또는 재학습) 시작 전 필수 (CLAUDE.md 규칙)
- 팀원에게 계획서 전달 전 사전 검증

### 호출 예시

```
Agent({
  subagent_type: "experiment-plan-reviewer",
  description: "추가 실험 계획 검증",
  prompt: "cond4_8x 5모델 전체 재학습 계획서 점검해줘. 하이퍼파라미터: [여기 명시]. 공정성, 재현성, 통계적 유효성 기준 PASS/FAIL 판정."
})
```

## 👨‍⚕️ experiment-doctor (실험박사)

### 언제 쓰나
- plan-reviewer가 FAIL 냈을 때 수정안 자문

### 호출 예시
(이전 Exp1_3cls v1 → v2 과정에서 사용됨. 현재는 거의 활용 불필요.)

## 🔁 Agent 워크플로우 (논문 작성)

```
[1] 섹션 초안 작성 (jjh 직접 or Claude에게 의뢰)
     ↓
[2] paper-doctor로 1차 리뷰
     ↓
[3] 수정 반영
     ↓
[4] paper-references-manager로 인용 점검
     ↓
[5] Figure/Table 필요하면 paper-visualizer
     ↓
[6] paper-doctor로 2차 리뷰 (완성본)
     ↓
[7] 다음 섹션으로
```

## 💡 팁

1. **Agent는 독립 세션**이라 대화 맥락이 없다. **prompt에 필요한 파일 경로·context 명시** 필수.
2. **요청은 구체적으로**: "좋게 해줘" ❌, "§1 motivation을 VISION-Datasets 데이터 부족 문제로 좁혀서 3문장 제안" ✅
3. **Agent 응답은 tool result로만 전달됨** (사용자는 못 봄) — Claude 메인이 요약해서 보여줘야.
4. **여러 agent 병렬 호출**: 독립적인 작업은 한 메시지에 여러 Agent tool call 동시 실행 가능.

## 🔗 파일 위치

- `/home/jjh0709/gitrepo/VISION-Instance-Seg/.claude/agents/*.md`
- 각 agent의 역할·책임·출력 포맷 상세 정의되어 있음

---

## 📝 다음 파일

- [08_NEXT_TASKS.md](08_NEXT_TASKS.md): 해야 할 일 체크리스트
