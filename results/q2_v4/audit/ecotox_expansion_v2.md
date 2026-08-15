# ECOTOX 확장 v2 — endpoint 필터 P-유도 교정 + 전량 재계산 + 공정성 + 제거건 소재. 학습 없음.

기존 STEP0/stepA/feasibility/verify 파일 보존. 산출 = 이 파일 + `ecotox_expansion_v2.json`.

## 1. ★ P에서 유도한 필터 (규칙 신설 아님). P 68,129행 raw 조인(2,541행 raw 부재 → 유도 제외).

- **endpoint (11값)**: LC50 51,751 · EC50 11,425 · LC50* 2,518 · LC50/ 1,874 · EC50* 294 · EC50/ 139 · LD50/ 52 · LC50*/ 38 · EC50*/ 33 · IC50 4 · EC0 1.
- **habitat (1값)**: **Water 68,129 (100%)** → 기존 `organism_habitat="Water"` **정확·불변**.
- **effect (6값)**: MOR 56,724 · ITX 5,420 · POP 4,877 · GRO 465 · PHY 407 · ~MOR 121. (기존 E-full은 effect 무필터였음 → 이제 이 6값으로 제한.)
- **media**: FW 49,577 · SW 10,763 · FW/ 6,297 · SW/ 1,370 · CUL/ 122 … (담수/해수 변형).
- **교정 E-full = endpoint∈위11 ∧ effect∈위6 ∧ habitat=Water ∧ media∈P집합**, 전 분류군(taxa 제한 없음), species.txt 조인.

## 2. 전량 재계산 (이전 → 교정 → delta)

### STEP A (a=원시종 / b=레코드가중)
| 지표 | 이전 | 교정 | delta |
|---|---|---|---|
| E-full 종 / 레코드 / 화합물 | 3,179 / 156,897 / 5,747 | **3,268 / 160,869 / 6,053** | +89 / +3,972 / +306 |
| E-full phylo a / b | .183 / .553 | .182 / **.560** | −.001 / +.007 |
| E-full DEB a / b | .078 / .504 | .076 / .506 | ~0 |
| E-full taxonomy class / genus | .988 / .972 | .988 / .971 | ~0 |
| B1-old 종 / 레코드 | 2,594 / 83,580 | 2,651 / 82,763 | +57 / −817 |
| B1-new 종 / 레코드 | 493 / 10,141 | 498 / 9,977 | +5 / −164 |
| P / A203 | (불변) | 1,267 / 203 | 0 |

⟹ endpoint 플래그판 추가(+) 와 effect/media 제한(−)이 상쇄, **커버리지 백분율 거의 불변**(최대 +0.7%p). 이전 STEP A 결론 견고.

### STEP B 회수 퍼널 (result_id; 이전→교정)
| 단계 | all 이전→교정 | old 교정 | new 교정 |
|---|---|---|---|
| 0 B1 원시 | 93,721→**92,740** | 82,763 이하 | — |
| 2 dur≤96h | 72,242→72,121 | | |
| 4 SMILES | 46,751→45,612 | | |
| 6 최종 회수 N | **46,733→45,596** (Δ−1,137) | **39,322** | **6,274** |
- 신규성: 신규 종 1,950→**2,004** · 신규 화합물 2,517→2,763. 비용 76.8→**74.9h**.

### 확인 2 (분모 = 회수 45,596; 분자/분모)
| 지표 | 이전(46,733) | 교정(45,596) |
|---|---|---|
| test_id 행 | 861 (1.84%) | 746/45,596 (**1.64%**) |
| test_id 고유 | 447/41,136 (1.09%) | 410/40,808 (1.00%) |
| reference 행 | 19,474 (41.67%) | 19,848/45,596 (**43.53%**) |
| reference 고유 | 1,492/4,175 (35.74%) | 1,503/4,150 (36.22%) |
| ★ 정밀 (ref,종,CAS) | 7,959 (17.03%) | 7,824/45,596 (**17.16%**) |
| ★ 정밀 (+duration) | 4,169 (8.92%) | 4,069/45,596 (8.93%) |

### 소항목 4 밀도비 (커버종 종당평균레코드 ÷ 미커버종; 이전→교정)
| 집합 | phylo | DEB |
|---|---|---|
| E-full | 5.52→**5.71×** | 12.00→12.46× |
| B1-old | 3.75→3.72× | 7.22→7.21× |
| B1-new | 2.84→2.84× | 4.81→4.81× |
| 신규 1,950(→2,004) | 0.58→0.62× | 0.25→0.26× |

## 3. ★ 분류군 내부 커버리지 (공정성) — phylo_a / DEB_a (종수)
tax_group 매핑: fish=class∈{Actinopterygii/Actinopteri/Chondrichthyes…}, crusta=subphylum Crustacea 또는 class∈{Malacostraca/Branchiopoda…}, algae=phylum∈{Chlorophyta/Bacillariophyta/Cyanobacteria…} 또는 class∈{Chlorophyceae…}, 그 외=other.

| 집합 | fish | crusta | algae | other |
|---|---|---|---|---|
| E-full | **.66/.32 (640)** | .15/.07 (653) | .17/.00 (385) | .01/.00 (1590) |
| 신규(2,004) | **.32/.09 (119)** | .08/.00 (150) | .07/.00 (182) | .00/.00 (1553) |
| 기존 779 | .76/.40 (444) | .22/.11 (317) | .11/.00 (9) | .00/.00 (9) |
| P | .74/.37 (520) | .17/.09 (503) | .25/.00 (201) | .14/.00 (43) |
| A203 | 1.00/1.00 (140) | 1.00/1.00 (17) | 1.00/.00 (42) | 1.00/.00 (4) |

⟹ **전체 평균과 크게 다름(명시)**: E-full 전체 phylo 18%는 **other 1,590종(phylo 1%)이 끌어내린 값**. **어류만 66%, 신규 어류만 32%**(전체 3.2% 아님). DEB는 어류도 ≤40%, 조류는 전 집합 0%.

**자원의 분류군 한정성(수치)**:
- 계통거리 행렬 853종 = **fish 458 · crusta 109 · algae 82 · other 204** (어류 편중).
- AmP(DEB) 수록 284종 = **fish 223 · crusta 51 · other 10 · algae 0** (조류 전무).
- E-full "other" 상위 phylum: Arthropoda 566(곤충 등) · Mollusca 362 · Chordata 155(양서류 등) · Annelida 84 · Ciliophora 71 · Magnoliophyta 67(식물) · Rotifera 37 · Nematoda 37 · Cnidaria 35 · Platyhelminthes 30. ⟹ other = 세 대상 분류군 밖(자원 미구축 대상).

## 4. 제거된 2,541건 소재 (세기만; 재판정 안 함)
split 파일 strata키(smiles,species,endpoint,duration) 매칭. ⚠ split 파일은 train/test만; GNN valid=train 내부 val_frac 0.1 carve(파일 부재).

| | discovery | replication |
|---|---|---|
| train | 940 | 1,207 |
| **test** | **145** | **249** |

- **테스트 세트 합 = 394건** (discovery 145 + replication 249).
- 상위 종: Daphnia_magna 312 · Pimephales_promelas 311 · Lepomis_macrochirus 220 · Oncorhynchus_mykiss 138 · Fundulus_heteroclitus 126 · Navicula_seminulum 84 · Ceriodaphnia_dubia 75 · Daphnia_pulex 73. (전 20 JSON.)
- pLC50 분포: 제거건 중앙값 **4.75** (Q1 3.57 / Q3 5.76) vs 전체 P 중앙값 5.04 (3.93/6.29) → 제거건이 약간 낮은 독성 쪽, 분포 형태 유사.

## 5. 단계 표기
- STEP A E-full = "전 레코드" 단계 · B1-old/new = "B1 원시" 단계 · 확인3 within-taxon(신규 2,004) = "B1 원시" 단계 · 779 = discovery 전체 · 회수 N = "최종 회수(6단계)" 단계. 각 표에 단계 명시.
- (이전 확인3의 "1,950종/43,188레코드"는 B1-원시 단계였음. 교정 후 신규=2,004종.)

## 가정·매핑
endpoint/effect/habitat/media = **P 실측 집합**(신설 아님). 단위환산 mass 직접/molar는 chem_mw. dur h/d/mi/wk/mo→시간 ≤96h. D16 질량하한 0.1 ng/L 프록시. tax_group 매핑 위 명시. 밀도비 정의 = GAP_LOG:99.
