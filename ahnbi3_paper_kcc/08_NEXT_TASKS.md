# 08. 해야 할 일 (To-Do 체크리스트)

**우선순위 순서**로 정리. 위에서부터 하나씩 처리.

---

## 🔥 A. 즉시 가능 (데이터 있음)

### A1. Figure/Table 재생성 (5분)
```bash
cd ~/gitrepo/VISION-Instance-Seg/docs/paper/scripts
bash regenerate_all.sh
```
- [ ] Figure 1 (cond4 N 스윕) PDF/PNG 생성
- [ ] Table 1 (main) tex/md 생성
- [ ] Table 2 (per-class) md 생성

**갱신된 데이터 포함**: cond4_1x~10x 전체 (ldy/yjw 완료분). N=8 정점 확인.

### A2. Figure 2 작성 (Exp1_3cls 스케일링)
현재 미작성. paper-visualizer agent에 의뢰:

```
Agent(subagent_type="paper-visualizer", prompt="
docs/paper/scripts/fig2_exp1_3cls_scaling.py 새로 작성해줘.
X축: GenAI 수량 (0, 25, 50, 75, 100, 125), Y축: segm_AP
- mask_rcnn: blue, cascade_mask_rcnn: red
- 현재 jjh 결과 (0, 75, 125)만 실제 점, (25, 50, 100)은 점선으로 '예상' 표시 또는 생략
- baseline 3-seed variance bar 추가 (mask_rcnn 8.51/8.73/9.84)
- figs/fig2_exp1_3cls_scaling.{pdf,png} 저장
")
```

### A3. §4 실험 설정 본문 작성
가장 쉬운 섹션 (사실 기술). `docs/paper/draft/04_setup.md` 신규 작성.

```
Agent(subagent_type="paper-doctor", prompt="
docs/paper/draft/04_setup.md 신규 작성해줘. 포함 내용:
- Dataset (Exp2_3cls: Dirty/Inclusoes/impurities, 클래스당 train 20장 + val 82장)
- 증강 데이터 (Gemini 생성 + 전통 Albumentations)
- 모델 2종 (Mask R-CNN, Cascade Mask R-CNN) — 5모델 버전 있으면 확장
- 하이퍼파라미터 표 (batch, lr, optimizer, schedule, patience)
- Hardware (A100 80GB 또는 V100 32GB)
- 평가 지표 (COCO segm AP)
한국어 논문체, 0.75쪽 분량, paper-doctor 스타일로.
")
```

### A4. §5 결과 본문 작성
Table 1, Figure 1, per-class 해석. `docs/paper/draft/05_results.md`.

```
Agent(subagent_type="paper-doctor", prompt="
docs/paper/draft/05_results.md 작성. 포함:
§5.1 전통 vs GenAI (cond1~3, Table 1 참조) - 1문단
§5.2 결합 시너지 (cond4 N 스윕, Figure 1 참조) - 1문단
§5.3 Per-class 분석 (Table 2 참조) - 1문단
§5.4 (추후) 5모델 비교 (Table 3)
메인 수치: cond4_8x cascade = 16.41 (+5.15 AP vs baseline)
분량: 2쪽
")
```

---

## 📖 B. 본문 작성 (순차)

### B1. §1 서론 (`docs/paper/draft/01_intro.md` bullet → 본문)
```
Agent(subagent_type="paper-doctor", prompt="
docs/paper/draft/01_intro.md 의 bullet point 4개 문단을 논문 본문으로 확장.
제목 B 기준: '전통 증강의 한계와 생성형 AI 증강의 시너지'
Motivation: VISION-Datasets 같은 소규모 benchmark에서 증강 선택이 중요
Contribution 3개 (Observation/Solution/Benchmark) 명시
분량: 0.5쪽, KCC 논문체
")
```

### B2. §2 관련 연구 (`02_related.md` bullet → 본문)
```
Agent(subagent_type="paper-doctor", prompt="
docs/paper/draft/02_related.md 를 본문으로 작성. 5개 subsection (Instance Seg,
전통 증강, 생성 증강, 산업 benchmark, 공정 benchmark). 각 subsection 2-3문장.
docs/references/refs.bib 키로 인용. 마지막에 '차별점' 문단 추가.
분량: 0.75쪽
")
```

### B3. §3 제안 방법 (`03_method.md`)
```
Agent(subagent_type="paper-doctor", prompt="
docs/paper/draft/03_method.md 신규 작성.
§3.1 공정성 프로토콜: iter 기반 통일 스케줄, cosine LR, COCO pretrained, fixed val
§3.2 증강 조합 설계: baseline, +전통 125, +GenAI 125, +GenAI+전통 Nx
§3.3 재현성: git tag, sha256 manifest
분량: 1쪽
")
```

### B4. §6 토의 (`06_discussion.md`)
```
Agent(subagent_type="paper-doctor", prompt="
docs/paper/draft/06_discussion.md 작성. 4문단:
1. 왜 전통 증강이 소규모에서 해로운가 (overfitting)
2. 왜 GenAI가 도움이 되는가 (semantic diversity)
3. 왜 결합이 시너지 (pixel + semantic 보완)
4. 왜 N=8이 최적 (sweet spot)
한계 명시: 단일 seed, val=test, 3 클래스만
분량: 0.5쪽
")
```

### B5. §7 결론 (`07_conclusion.md`)
```
Agent(subagent_type="paper-doctor", prompt="
docs/paper/draft/07_conclusion.md 작성. 3문단:
1. 3 contribution 요약
2. 실무 가이드라인 ('Cascade Mask R-CNN + GenAI 125 + 전통 1160/cls')
3. 향후 과제 (다른 클래스, 7모델 비교, seed 반복)
분량: 0.25쪽
")
```

---

## 📚 C. 인용 정리

### C1. VISION-Datasets 원 논문 등록 (PDF 받은 후)
```
Agent(subagent_type="paper-references-manager", prompt="
[PDF 경로 or arXiv ID] 를 docs/references/refs.bib 에 vision_datasets_XXXX 키로 추가.
§1 서론 데이터 언급 위치, §2 산업 benchmark subsection 에 인용 제안.
")
```

### C2. 전체 인용 점검 (초안 완성 후)
```
Agent(subagent_type="paper-references-manager", prompt="
docs/paper/draft/*.md 전체 인용 점검. 누락/미사용 리스트.
도메인 관점 필수 인용 빠진 것 지적 (예: Bag of Tricks, SGDR 등).
")
```

---

## 🎨 D. 고급 Figure

### D1. Figure 3 Per-class heatmap
```
Agent(subagent_type="paper-visualizer", prompt="
docs/paper/scripts/fig3_perclass_heatmap.py 작성.
X축: 조건 (cond1, cond2, cond3, cond4_6x, cond4_8x)
Y축: 클래스 (Dirty, Inclusoes, impurities)
색상: viridis, AP 값 annotation
subplot 2개 (mask_rcnn, cascade_mask_rcnn)
")
```

### D2. Figure 4 정성 비교 (선택, 시간 여유 있으면)
- cond1 best 체크포인트 + cond4_8x best 체크포인트로 inference
- val 셋에서 cond1이 틀리고 cond4_8x가 맞춘 4장 자동 선별
- 2×4 grid: input / GT / cond1 pred / cond4_8x pred

---

## 🧪 E. 추가 학습 (여유 있으면)

### E1. cond4_8x 5모델 전체 완성
완료: mask_rcnn, cascade_mask_rcnn
진행중: cascade_rcnn (usw), solov2 (ahnbi3)
미진행: maskdino, mask2former, rtmdet_ins

### E2. yjw 결과 수령
- exp1_3cls genai_25, genai_50, genai_100 × 2 모델 = 6개
- 받으면 Figure 2 완전한 스케일링 곡선 완성

### E3. Exp1_3cls 3-seed 확장 (선택)
- 현재 baseline/mask_rcnn만 3-seed
- genai_125도 3-seed 돌리면 variance 측정 완성 → 논문 robustness ↑

---

## 📄 F. 논문 통합 & 투고

### F1. 전체 draft 통합
- `docs/paper/draft/*.md` 합쳐서 한 파일로
- 또는 KCC hwp 템플릿에 붙여넣기

### F2. 최종 리뷰
```
Agent(subagent_type="paper-doctor", prompt="
전체 논문 draft 최종 리뷰. 모든 섹션 story 연결, 분량 균형, 수치 일관성,
인용 형식 확인. 투고 직전 체크리스트 생성.
")
```

### F3. KCC 템플릿 변환
- hwp 또는 LaTeX 템플릿에 본문 이식
- Figure/Table 배치

### F4. 투고
- 저자 정보 (jjh0709, yjw, ldy, 지도교수)
- 초록 (200단어)

---

## ⚡ 빠른 시작 (권장 순서)

작업 시간 있으면 이 순서:

1. **A1** (Figure/Table 재생성, 5분)
2. **A3** (§4 실험 설정, 30분)
3. **A4** (§5 결과, 1시간)
4. **B3** (§3 방법, 1시간)
5. **B1** (§1 서론, 1시간)
6. **B2** (§2 관련연구, 1시간)
7. **B4** (§6 토의, 30분)
8. **B5** (§7 결론, 20분)
9. **C2** (인용 점검, 30분)
10. **F2** (최종 리뷰, 30분)

합계: **약 6-7시간**이면 초안 완성.

---

## 📝 다음 파일

- [09_FAQ.md](09_FAQ.md): 자주 막히는 점 + 해결
