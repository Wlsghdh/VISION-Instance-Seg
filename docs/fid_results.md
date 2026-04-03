# FID 평가 결과

- 실행일: 2026-04-03
- 라이브러리: cleanfid (legacy_pytorch mode)
- 모델: InceptionV3

---

## 가설

> FID(원본, AI생성) < FID(원본, 정상이미지)

AI 생성 이미지가 정상 이미지보다 원본 결함 이미지 분포에 더 가까운지 검증한다.
FID 점수가 낮을수록 두 이미지 분포가 유사하다.

---

## 결과 요약

**13개 클래스 전부 PASS** - AI 생성 이미지가 모든 클래스에서 원본 결함 분포에 더 가깝다.

| 클래스 | FID(원본 vs AI) | FID(원본 vs 정상) | 차이 | 결과 |
|--------|:--------------:|:----------------:|:----:|:----:|
| casting_Inclusoes | **81.74** | 273.37 | -191.63 | PASS |
| casting_Rechupe | **55.27** | 164.94 | -109.67 | PASS |
| Console_Collision | **184.83** | 338.76 | -153.93 | PASS |
| Console_Dirty | **215.78** | 283.64 | -67.86 | PASS |
| Console_Gap | **139.62** | 195.53 | -55.91 | PASS |
| Console_Scratch | **210.31** | 315.85 | -105.54 | PASS |
| Cylinder_Chip | **102.76** | 152.84 | -50.08 | PASS |
| Cylinder_PistonMiss | **248.83** | 258.97 | -10.14 | PASS |
| Cylinder_Porosity | **112.72** | 169.82 | -57.10 | PASS |
| Cylinder_RCS | **153.81** | 234.58 | -80.77 | PASS |
| screw_defect | **80.32** | 136.08 | -55.76 | PASS |
| Wood_impurities | **170.38** | 242.71 | -72.33 | PASS |
| Wood_pits | **260.25** | 315.91 | -55.66 | PASS |

---

## 카테고리별 분석

### Casting (2클래스)

| 비교 | Inclusoes | Rechupe |
|------|:---------:|:-------:|
| 원본 vs AI | 81.74 | 55.27 |
| 원본 vs 정상 | 273.37 | 164.94 |
| AI vs 정상 | 278.76 | 158.43 |

- Rechupe가 가장 낮은 FID(55.27) → AI 생성 품질이 가장 좋음
- 두 클래스 모두 AI 이미지가 원본에 매우 가깝게 생성됨

### Console (4클래스)

| 비교 | Collision | Dirty | Gap | Scratch |
|------|:---------:|:-----:|:---:|:-------:|
| 원본 vs AI | 184.83 | 215.78 | 139.62 | 210.31 |
| 원본 vs 정상 | 338.76 | 283.64 | 195.53 | 315.85 |
| AI vs 정상 | 181.16 | 185.12 | 119.89 | 125.10 |

- Gap이 가장 좋은 FID(139.62)
- Dirty가 상대적으로 높지만 여전히 PASS

### Cylinder (4클래스)

| 비교 | Chip | PistonMiss | Porosity | RCS |
|------|:----:|:----------:|:--------:|:---:|
| 원본 vs AI | 102.76 | 248.83 | 112.72 | 153.81 |
| 원본 vs 정상 | 152.84 | 258.97 | 169.82 | 234.58 |
| AI vs 정상 | 142.33 | 83.29 | 160.43 | 146.17 |

- Chip(102.76)과 Porosity(112.72)가 우수
- PistonMiss는 차이가 가장 적음 (248.83 vs 258.97, 차이 10.14) → AI 생성이 상대적으로 어려운 클래스

### Screw (1클래스)

| 비교 | defect |
|------|:------:|
| 원본 vs AI | 80.32 |
| 원본 vs 정상 | 136.08 |
| AI vs 정상 | 106.38 |

- FID 80.32로 양호

### Wood (2클래스)

| 비교 | impurities | pits |
|------|:----------:|:----:|
| 원본 vs AI | 170.38 | 260.25 |
| 원본 vs 정상 | 242.71 | 315.91 |
| AI vs 정상 | 186.93 | 274.47 |

- pits가 상대적으로 FID가 높음 (260.25) → 생성 난이도가 높은 클래스

---

## FID 점수 랭킹 (원본 vs AI, 낮을수록 좋음)

| 순위 | 클래스 | FID |
|:----:|--------|:---:|
| 1 | casting_Rechupe | 55.27 |
| 2 | screw_defect | 80.32 |
| 3 | casting_Inclusoes | 81.74 |
| 4 | Cylinder_Chip | 102.76 |
| 5 | Cylinder_Porosity | 112.72 |
| 6 | Console_Gap | 139.62 |
| 7 | Cylinder_RCS | 153.81 |
| 8 | Wood_impurities | 170.38 |
| 9 | Console_Collision | 184.83 |
| 10 | Console_Scratch | 210.31 |
| 11 | Console_Dirty | 215.78 |
| 12 | Cylinder_PistonMiss | 248.83 |
| 13 | Wood_pits | 260.25 |

---

## 데이터 현황

| 클래스 | 원본 이미지 | AI 생성 이미지 | 정상 이미지 |
|--------|:----------:|:-------------:|:----------:|
| casting_Inclusoes | 9장 | 100장 | 100장 |
| casting_Rechupe | 9장 | 100장 | 100장 |
| Console_Collision | 9장 | 75장 | 100장 |
| Console_Dirty | 9장 | 75장 | 100장 |
| Console_Gap | 9장 | 75장 | 100장 |
| Console_Scratch | 9장 | 75장 | 100장 |
| Cylinder_Chip | 9장 | 75장 | 100장 |
| Cylinder_PistonMiss | 9장 | 75장 | 100장 |
| Cylinder_Porosity | 9장 | 75장 | 100장 |
| Cylinder_RCS | 9장 | 75장 | 100장 |
| screw_defect | 9장 | 100장 | 100장 |
| Wood_impurities | 9장 | 100장 | 100장 |
| Wood_pits | 9장 | 100장 | 100장 |

---

## 비교 설명

| 비교 | 의미 |
|------|------|
| 원본 vs AI | 원본 결함 이미지와 AI 생성 결함 이미지 간 분포 거리 |
| 원본 vs 정상 | 원본 결함 이미지와 정상(무결함) 이미지 간 분포 거리 |
| AI vs 정상 | AI 생성 결함 이미지와 정상 이미지 간 분포 거리 |

**가설 검증**: FID(원본 vs AI) < FID(원본 vs 정상)이면 PASS
→ AI가 생성한 결함 이미지가 정상 이미지보다 실제 결함 분포에 더 가깝다는 의미

---

## 결론

- **13개 클래스 전부 가설 통과** (PASS rate: 100%)
- Gemini API로 생성한 결함 이미지가 모든 카테고리에서 원본 결함 분포를 잘 반영함
- 특히 casting_Rechupe(55.27), screw_defect(80.32), casting_Inclusoes(81.74)에서 우수한 결과
- Cylinder_PistonMiss(차이 10.14)와 Wood_pits(260.25)는 상대적으로 생성 난이도가 높은 클래스
