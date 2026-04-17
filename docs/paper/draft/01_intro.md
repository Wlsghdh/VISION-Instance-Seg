# §1. 서론 (Draft)

**상태**: 초안 대기. 아래 bullet point를 본문 문단으로 확장해야 함.
**참고**: `docs/paper/PAPER_PLAN.md` story-line 확인.

---

## 핵심 전개 (3-4 문단)

### Para 1 — 문제 제기 (산업 결함 데이터 부족)
- 산업 제품 결함 검출에서 Instance Segmentation은 픽셀 단위 localization으로 품질 관리에 필수
- 그러나 **결함 샘플 확보가 어렵다** — 생산라인에서 불량은 드물고, 라벨링 비용 높음
- 대표 벤치마크 (VISION-Datasets [TODO 인용], MVTec AD) 도 클래스당 수십 장 수준
- → **데이터 증강**이 사실상 필수 선택지

### Para 2 — 증강 선택의 딜레마
- 전통 증강 (flip, rotate, color jitter): pixel-level perturbation, 저비용, 널리 사용
- 생성형 AI 증강 (GAN, Diffusion, Gemini 등): semantic-level diversity, 고비용, 상대적 신규
- **어느 쪽이 소규모 산업 결함에 유효한가?** 공정 비교 부재

### Para 3 — 관찰과 기여
- 본 연구 관찰: 클래스당 20장 극소규모에서 **전통 증강 단독은 baseline보다 성능 악화** (cond2: -1.76 AP)
- GenAI 증강(Gemini)은 소량(75-125장)으로 **+2-3 AP 개선** (cond3)
- 두 증강을 결합하면 **시너지** (cond4_6x: +4.64 AP, cond4_8x: +5.15 AP)

### Para 4 — Contribution 정리 (3가지)
1. **Observation**: 소규모 산업 결함에서 전통 증강 단독이 해로움을 실증
2. **Solution**: GenAI 증강 + 전통 증강 결합의 시너지와 최적 혼합비 (N=8) 발견
3. **Benchmark**: iter 기반 공정성 프로토콜 + 재현 가능한 결과·코드 공개

---

## 작성 시 주의

- **첫 문장**: "산업 자동화의 핵심인 ... 소규모 결함 데이터의 증강 전략은 ..." 같은 연구 가치 강조
- **VISION-Datasets 원 논문 반드시 인용** (받으면 추가)
- **Gemini**: Google 2023 논문 인용
- **논문체**: "우리는 ~함을 발견하였다" 보다 "본 연구는 ~임을 관찰한다" 선호
- **분량**: 0.5쪽 이내 (KCC 기준 ~250단어 한국어)

---

## 키 레퍼런스 (refs.bib 키)

- `he_2017_maskrcnn` — Mask R-CNN
- `cai_2018_cascade` — Cascade Mask R-CNN
- `bergmann_2019_mvtec` — MVTec AD
- `cubuk_2019_autoaugment` — 전통 증강
- `zhang_2021_datasetgan` — GAN 기반 증강
- `rombach_2022_ldm` — Diffusion 증강
- `gemini_2023` — Gemini
- `goyal_2017_largeminibatch` — Linear Scaling Rule (§3 방법론에서)
- TODO: `vision_datasets_XXXX` — VISION-Datasets 원 논문

---

## 작성 후 리뷰

paper-doctor agent에 제출:
```
subagent_type: paper-doctor
prompt: docs/paper/draft/01_intro.md 을 읽고 story-line 진단 + 리뷰어 공격 시뮬레이션 해줘.
특히 motivation이 충분히 좁혀져 있는지, contribution 3개가 독립적인지 확인.
```
