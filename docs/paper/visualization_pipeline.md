# 논문 시각화 & 파이프라인 전략

**목적**: 실험 결과(JSON) → 논문-quality figure/table을 **자동 재생성** 가능하게 만든다.
새 결과가 들어오면 스크립트 하나만 돌리면 논문에 쓰이는 모든 시각물이 최신화됨.

---

## 1. 디렉토리 구조 (신규 생성)

```
docs/paper/
├── figs/                    # 생성된 figure (PDF + PNG)
│   ├── fig1_cond4_curve.pdf
│   ├── fig1_cond4_curve.png
│   ├── fig2_genai_scaling.pdf
│   ├── fig3_perclass_heatmap.pdf
│   └── fig4_qualitative.pdf
├── tables/                  # 표 (LaTeX + Markdown)
│   ├── table1_main.tex
│   ├── table1_main.md
│   ├── table2_perclass.tex
│   └── table2_perclass.md
├── scripts/                 # 재생성 Python 스크립트
│   ├── _style.py           # matplotlib 공통 설정
│   ├── _loader.py          # results.json 로더 (공통)
│   ├── fig1_cond4_curve.py
│   ├── fig2_genai_scaling.py
│   ├── fig3_perclass_heatmap.py
│   ├── fig4_qualitative.py
│   ├── table1_main.py
│   ├── table2_perclass.py
│   └── regenerate_all.sh   # 한 번에 재생성
├── captions.md              # 모든 figure/table 캡션 draft
├── visualization_pipeline.md  # 본 문서
└── draft/                   # 논문 본문 draft (섹션별)
    ├── 01_intro.md
    ├── 02_related.md
    ├── 03_method.md
    ├── 04_setup.md
    ├── 05_results.md
    ├── 06_discussion.md
    └── 07_conclusion.md
```

---

## 2. 파이프라인 흐름

```
[1] 학습 완료
   ↓
[2] python scripts/sync_results_to_github.py       # eval JSON 공유
   ↓
[3] cd docs/paper && bash scripts/regenerate_all.sh
   ├─ fig1_cond4_curve.py     → figs/fig1_*.{pdf,png}
   ├─ fig2_genai_scaling.py   → figs/fig2_*.{pdf,png}
   ├─ fig3_perclass_heatmap.py→ figs/fig3_*.{pdf,png}
   ├─ table1_main.py          → tables/table1_main.{tex,md}
   └─ table2_perclass.py      → tables/table2_perclass.{tex,md}
   ↓
[4] 논문박사(paper-doctor) agent 에 draft 리뷰 요청
   ↓
[5] 참고문헌 매니저(paper-references-manager) agent 에 인용 점검
   ↓
[6] 최종 투고 (LaTeX/hwp 컨버전)
```

---

## 3. 공통 유틸 설계

### `_style.py` — matplotlib 통일
- Publication-quality: DPI 300 PDF, 단일 컬럼 5.5×3.5 inch
- 컬러 팔레트: Mask R-CNN=blue, Cascade=red (흑백 인쇄에도 구분되도록 선 스타일 병행)
- 한글 폰트 지원: NanumGothic 자동 감지

### `_loader.py` — 결과 통합 조회
```python
def load_all_results():
    """results/evaluation/results.json 파싱, (exp, cond, cat, model, seed) → metric dict"""
def filter_exp(df, exp):  ...
def filter_category(df, cat='Exp2_3cls'):  ...
def get_metric(df, metric='segm_AP'):  ...
```
→ figure 스크립트들은 이걸로 공통 접근, 포맷 변경에 강건.

### `regenerate_all.sh`
- `set -e`, `set -x` 로 에러 즉시 중단 + 진행 출력
- 각 스크립트 실행 후 **성공 여부 + 생성 파일 리스트** 출력
- 실패 시에도 다른 figure는 계속 생성 (`|| echo "⚠️ fig1 실패"`)

---

## 4. Figure/Table 상세 스펙

### Figure 1. cond4 N 스윕 곡선 (핵심!)
- **X축**: 전통 증강 배수 N (1~10) — x값은 실제 전통 이미지 수 (145N)로도 보조 표기
- **Y축**: segm_AP
- **Line**: mask_rcnn (blue, circle), cascade_mask_rcnn (red, square)
- **Annotation**: cond1 baseline 값을 dashed horizontal line으로 표시 (참조선)
- **Caption**: "GenAI 125 + 전통 N×145/class에서 N 스윕. **N=6에서 Cascade +4.64 AP 도약**, 단조 증가 추세."

### Figure 2. Exp1_3cls GenAI 단독 스케일링
- **X축**: GenAI 수량 (0, 25, 50, 75, 100, 125)
- **Y축**: segm_AP
- **비교**: 기존 Unified 14cls 의 3클래스 평균 vs 현재 Exp1_3cls (선택적 이중 line)
- **Caption**: "원본 20장/class 고정, GenAI 0→125 스윕. **3클래스 분리 학습이 14클래스 통합 대비 높은 AP**를 보임."

### Figure 3. Per-class AP heatmap
- **2 subplot**: mask_rcnn / cascade_mask_rcnn
- **X축**: 조건 (cond1, cond2, cond3, cond4_4x, 5x, 6x)
- **Y축**: 클래스 (Dirty, Inclusoes, impurities)
- **색**: viridis, 숫자 annotation
- **Caption**: "클래스별 증강 효과 차이. impurities는 전 조건에서 우위, Dirty는 cond2에서 큰 하락 → 전통 증강이 Dirty 특성에 해롭다는 증거."

### Figure 4. 정성 비교 (qualitative)
- **4개 샘플**: cond1에서 틀림 + cond4_6x에서 맞춤
- **Grid 2×4**: [input | GT mask | cond1 predict | cond4_6x predict]
- **요구 작업**: detectron2로 2회 inference (cond1 best ckpt, cond4_6x best ckpt)
- **Caption**: "cond4_6x가 baseline 대비 복구한 케이스. 공통 패턴: 작은 결함/불균일 조명/클래스간 혼동."

### Table 1. Main result
```markdown
| 조건 | 구성 | Mask R-CNN ↑ | Cascade MRCNN ↑ | Δ vs baseline |
|------|------|:---:|:---:|:---:|
| baseline | 원본 20/cls | 10.30 | 11.26 | — |
| +전통 125 | 20+trad125 | 8.54 | 10.36 | −1.76 / −0.90 |
| +GenAI 125 | 20+gen125 | 12.18 | 11.78 | +1.88 / +0.52 |
| +GenAI+전통6x | (20+125)×6 | **13.08** | **15.90** | +2.78 / **+4.64** |
```

### Table 2. Per-class breakdown
- 조건별 3 클래스 AP + 평균
- 모델별 2 패널 (한 페이지에 다 들어가도록)

### Table 3. Hyperparameters (static)
- batch/lr/max_iters/warmup/scheduler 고정값 참조용

---

## 5. 실행 예시

```bash
# 전체 재생성
cd docs/paper && bash scripts/regenerate_all.sh

# 특정 figure만
python scripts/fig1_cond4_curve.py

# 특정 figure + 한글
python scripts/fig1_cond4_curve.py --lang=kr

# 한글/영문 동시 생성
python scripts/fig1_cond4_curve.py --lang=both
```

---

## 6. 점진적 구현 로드맵

**Phase 1 (이번 커밋)**: 파이프라인 skeleton + 현재 데이터로 Figure 1, Table 1, Table 2 생성
**Phase 2 (Exp1_3cls 완주 후, ~4/16)**: Figure 2 완성
**Phase 3 (cond4_7x~10x 수집 후)**: Figure 1 10 point 확장
**Phase 4 (정성 Figure용 inference 후)**: Figure 4
**Phase 5**: paper-doctor agent 리뷰 → 본문 draft 반복

---

## 7. 품질 체크리스트 (투고 직전)

- [ ] 모든 figure는 흑백 인쇄해도 구분 가능 (마커 + 선 스타일)
- [ ] 모든 축 레이블 존재, 단위 명시 (`segm AP (%)`)
- [ ] 범례 위치 데이터와 겹치지 않음
- [ ] Font size ≥ 8pt (축 tick), ≥10pt (캡션/제목)
- [ ] 모든 Table 헤더에 `↑` / `↓` 표기 (높을수록/낮을수록 좋은지)
- [ ] 캡션이 self-contained (본문 안 읽어도 figure 이해 가능)
- [ ] PDF 열어봐서 벡터(확대해도 안 깨짐) 확인
- [ ] 색맹 친화 색상 (blue/red/green 대신 blue/orange 권장)

---

## 8. Agent 활용 안내

이 파이프라인을 운영하면서 **3개 agent**를 적극 활용:

1. **paper-visualizer** (`.claude/agents/paper-visualizer.md`)
   - "Figure 1 재생성해줘"
   - "per-class heatmap 한국어 버전"
   - "Table 1 LaTeX로 변환"

2. **paper-references-manager** (`.claude/agents/paper-references-manager.md`)
   - "Mask R-CNN, Cascade 인용 추가"
   - "VISION-Datasets 논문 BibTeX 만들어줘" (원본 논문 경로 주면)
   - "§2 관련연구 누락 인용 점검"

3. **paper-doctor** (`.claude/agents/paper-doctor.md`)
   - "§1 서론 draft 봐줘"
   - "전체 논문 story-line 진단"
   - "리뷰어 공격 시뮬레이션"

---

## 9. 전략적 강조점 (논문 story 관점)

본 시각화 파이프라인은 단순한 "결과 그림"이 아니라 **논문의 서사 도구**다:

- **Figure 1 (cond4 N 스윕)** → "결합의 시너지, 단조 증가" 주장 **시각적 증거**
- **Figure 2 (GenAI 스케일링)** → "GenAI 단독으로도 증가" 주장 뒷받침
- **Figure 3 (heatmap)** → "클래스별로 다르다" 라는 **Discussion의 nuance** 뒷받침
- **Figure 4 (정성)** → 리뷰어가 "정말 눈에 보이는 차이가 있나?" 공격할 때 **직접 반박**
- **Table 1 (main)** → Abstract·Conclusion의 수치 근거
- **Table 2 (per-class)** → "일반화 가능한 주장인가?" 검증

→ 각 figure/table은 **contribution 하나씩**을 뒷받침하도록 설계되어야 한다.
