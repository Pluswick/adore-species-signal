# GAP_EXECUTION_LOG — ADORE 전환 실험 (EXP)

> EXP = `<REPO_ROOT>`. 매 단계 append. 수치는 원본 파일에서 직접 측정(읽기 전용).

---

## 2026-07-29 — Session 1: 오리엔테이션 + 환경·GPU·ADORE 검증 (대량 실행 전 정지)

작업: HANDOFF 통독 후 §1–§3 검증만. 로더 구현·데이터 재구축·사다리 학습은 미착수(다음 지시 대기).

### 1. EXP 환경 자기검증 — PASS
- `import jcim_v3.runner` OK; `import ccmpnn` OK (`CC-MPNN\ccmpnn\__init__.py` 자동 해결).
- paths.py: `JCIM_ROOT = EXP` ✓ / `CC_MPNN_ROOT = CCLABS\CC-MPNN` (존재) ✓ / `CC_MPNN_DATA` 존재 ✓ / `RESULTS_ROOT = EXP\results\jcim_v3` ✓.
- run 스크립트 하드코딩 경로 DATA/OUT/PRED 모두 `EXP\results\q2_v4\…` 지시 확인
  (`run_q2_gnn_ladder.py`, `run_q2_gnn_oof_tier1prime.py`, `run_q2_naive_taxonomy.py`, `run_q2_scaffold_taxonomy_lgbm.py`).
- conda env `jcim_v3` 존재; torch 2.12.1+cu130.

### 2. 멀티 GPU 검증 (§F) — PASS (device 매핑 주의점 발견)
- `nvidia-smi`: GPU0 = RTX 5060 Ti (16 GB, 디스플레이), GPU1 = RTX 4090 (24 GB, 유휴). **두 장 모두 인식** (과거 세션 1장만 보이던 문제 해소).
- torch device_count = 2, 두 카드 모두 512×512 matmul OK.
- **⚠ device 순서 불일치**: torch 기본 순서(FASTEST_FIRST)는 nvidia-smi(PCI)와 반대 →
  기본 상태에서 `CUDA_VISIBLE_DEVICES=0`→**4090**, `=1`→**5060 Ti** (§F 라벨과 역전).
  - 해결: `CUDA_DEVICE_ORDER=PCI_BUS_ID` 지정 시 `=0`→5060 Ti, `=1`→4090 (nvidia-smi·§F 라벨과 일치) — 검증 완료.
  - 권고: 사다리 분할 시 두 잡 모두 `CUDA_DEVICE_ORDER=PCI_BUS_ID` prefix 고정.

### 3. ADORE 실측 재확인 (§C, 읽기 전용) — 대부분 일치, 2건 수정 필요
mortality_processed.csv (측정 vs §C): records 70,670 ✓ / 종(tax_gs) 1,267 ✓ / 화합물(test_cas) 3,295 ✓ /
strata 32,301 ✓ / endpoint LC50 58,330·EC50 12,340 ✓ / duration 96h 29,420·48h 19,520·24h 16,185·72h 5,545 ✓ /
tax_group fish 41,526·crusta 23,092·algae 6,052 ✓ / 종당 median 6·mean 55.8·max 7,260 ✓ /
top10% 83.9·top20% 91.3 ✓ (top5% 측정 73.8 vs §C 73.6 — top-k 반올림 차, 비유의) /
희소종 ≤1 18.4·≤2 29.9·≤5 49.4 ✓ / mortality_filtered 66,896 ✓ / tax_pdm_available True 53,232행 ✓.
- **kingdom/phylum 컬럼 부재 확인** (tax_all은 class부터: "Actinopterygii …") → §B flag1 / §I.5 확정.
- tax_gs·pdm 종명 모두 underscore 포맷("Cyprinus_carpio"). taxonomy rank: class/order/family/genus/species 존재.

**수정 필요 1 — 화합물 파일 행수:** `chemicals\…with-oecd-function.csv` **물리 라인 9,404 = §C 값**이나
**논리 레코드(화합물) = 4,702** (distinct test_cas 4,702; 1,055 컬럼, fingerprint/InChI 필드 내 개행이 라인수 부풀림).
→ 실제 화합물 수 = **4,702** (mordred 4,705와 일치). **단 SMILES 조인 커버리지 = 100%** (mortality CAS 3,295 전부 `chem_rdkit_can_smiles` 보유; record-level 100%). 비차단.

**수정 필요 2 — 계통거리 종 커버리지:** FCA_pdm_species.csv 853×853(라벨열 포함 854열, §C shape ✓).
§C "~853/1,267 ≈ 67%"는 (행렬크기/전체종)의 순진 비율. **실제 종 겹침 = 526/1,267 = 41.5%**
(pdm 853종 중 327종은 mortality에 없음). **record-level 커버리지 = 75.3%**(= tax_pdm_available True 53,232/70,670).
→ Tier 5는 종 41.5%·레코드 75.3% 커버, 나머지 unknown 버킷. (종명 포맷 문제 아님 — 정규화해도 526 동일.)

**DEB/형질(Tier 6) 커버리지(신규 측정):** DEB(tax_ps_ampv/ampkap/amppm) = **238종**·48,292행(68.3%);
tax_lh_amd 235종; tax_lh_lbcm **94종**(가장 희소); tax_eco_climate 238종. → Tier 6 종 커버리지 ≈ 238/1,267(18.8%),
레코드 68.3%(풍부 어종에 집중). unknown 버킷 큼.

### 4. 코드 확인 — Tier 5/6 backbone 실현성 (§I.3 / §B flag2) — 확정
`models.py`: D-MPNN는 ccmpnn `build_model` 경유(`species_repr` = embed/onehot만). taxonomy 주입은 이미
`ValueError`로 하드 차단("GraphConv taxonomy only"). Tier 5(PCoA)·6(DEB)도 동종의 신규 species_repr →
ccmpnn(read-only) 수정 없이는 **D-MPNN 주입 불가 확정**. GraphConv(자체 코드)·LightGBM(피처 컬럼)만 가능.

### 정지
로더 구현(§D)·데이터 재구축·사다리 재실행 미착수. §I 결정(1–5) director 회신 대기.

---

## 2026-07-29 — Session 2: director 회신(§A 확정) 후 F1–F7 사실확인 (읽기 전용, GPU 불필요)

director §A 확정(4-rank / ccmpnn read-only+훅확인 / 우리 split primary / Tier6 부분허용 238 / 6방향 cross-group / PCI_BUS_ID).
§B F1–F7 사실확인 수행. 로더 미착수(F1–F7 회신 후 착수). 판단 지점은 `EXP\director_결정필요.md` D1–D6.

### F1 — ADORE challenge split 구조 (실측)
- 각 challenge 파일은 `split_*` 지시열로 train/test 인코딩: 값 = {test, 0,1,2,3,4}(test=고정 홀드아웃, 0–4=잔여 5-fold CV). "train" 라벨 없음.
- 전략: `split_totallyrandom/random/occurrence`(전 파일) + `split_scaffold-murcko/generic`(+loo-0/1,llo) (a-CA2F 제외 전부).
- **모든 split이 compound-disjoint**(train∩test 화합물=0, split_random조차) — ADORE는 화합물 단위 집계 후 분할.
- 프리픽스(실측): **s-\*=단일종 1개**(compound 일반화), **t-\*=그룹 전체 종**(t-F2F 140종/t-C2C 17/t-A2A 46),
  **a-\*=다중그룹**. 정식 명칭은 Schür et al. 2023(데이터셋에 코드 데이터사전 없음).
- **cross-group은 a-CA2F=(crusta+algae)→fish 단일 방향만**: test=100% fish, species_shared=0(그룹·종 disjoint).
  `same`=test-fish 화합물이 train과 완전중복(shared 991), `diff`=compound-disjoint(shared 0).
- **결론**: ADORE split은 §A-5 6방향 pairwise를 직접 구현 불가(→fish만). 우리가 생성. (→ director_결정필요 D6)

### F2 — PCoA scree (계통거리행렬, classical MDS)
- 대칭·대각0 확인. 값 range 0–7544.9.
- **음의 고유값 질량 = 0.0%** (sub 526종: neg 0개; full 853: neg 2개 ≈−0.0) → **Euclidean 임베딩 가능, Cailliez/Lingoes 보정 불요**.
- in-data 부분행렬(526종, max real dim 524) 누적 설명력: d1 64.3 / d2 77.5 / d4 86.8 / d8 92.3 / **d16 95.9** /
  d24 97.6 / **d32 98.5** / d48 99.2 / **d64 99.5** / d96 99.75 / d128 99.86 (%).
- full(853종) 누적: d16 93.5 / d32 96.3 / d64 98.6. → §I.1 16d주+32/64d 민감도 정당(16d≈96%, 추가분 소).

### F5 — ccmpnn 외부 feature 주입 훅 = **있음** (§A-2 → D-MPNN Tier5/6 편입)
- `ccmpnn.graph.BatchMolGraph`가 per-mol 선택입력 2개 노출: `f_descriptors[n_mols,desc_fdim]`, `species_idx[n_mols]`;
  둘 다 `ccmpnn.graph.assemble_batch(f_descriptors=, species_idx=)`(우리 데이터층)에서 주입.
- 소스 수정 없는 공개 경로: (A) 커스텀 `ContextBase` 서브클래스가 `apply_global`에서 `bmg.species_idx`로 외부벡터 lookup·concat
  (선례 `variants.LateFusionContext`; 우리 `DMPNNMessageLevelContext`), (B) `f_descriptors`+`VariantConfig(mol_feat="concat")`.
- ∴ D-MPNN에 Tier5/6(고정 외부 per-species 벡터) 편입 가능. taxonomy는 여전히 GraphConv only(SpeciesEncoder embed/onehot 한정).
- **Session-1 "D-MPNN 불가 확정" 정정**(그 차단은 taxonomy per-rank 한정). (→ director_결정필요 D1)

### F6 — tier별 커버리지 (종/레코드, overall/discovery)  ※ discovery = LC50 & 96h = 26,523행/779종
| tier | overall 종% | overall 레코드% | discovery 종% | discovery 레코드% |
|---|---|---|---|---|
| 3a taxonomy(native 4-rank) | 100 | 100 | 100 | 100 |
| 4 embedding | 100 | 100 | 100 | 100 |
| 5 phylo(pdm) | 41.5(526) | 75.3 | 52.4(408) | 82.3 |
| 6 DEB any(≥1 of 7수치) | 18.8(238) | 68.3 | 27.0(210) | 77.2 |
| 6 DEB complete(all 7수치) | 6.6(83) | 27.3 | 8.7(68) | 32.2 |
- 3b(NCBI): ADORE에 없음(native taxonomy 1종만) → 외부 NCBI 해상 필요. (→ D5)
- rank별 결측(overall): class/order/family/genus **전부 0%**(종·레코드).
- 핸드오프 수치 대조: taxonomy ~72%는 ADORE native(=100%)와 불일치(tox-learn/NCBI 해상률 추정); phylo ~75%=overall 레코드(75.3%✓); DEB ~17%=overall 종(18.8%≈✓).
- **밀도비**(커버종 종당레코드 ÷ 미해상종): phylo **4.3×**(director 가설 직접확인: 526종×101.2 vs 741종×23.5=4.30), DEB-any **9.33×**, DEB-complete 5.37×, taxonomy=N/A(미해상0).

### F4 — Tier 6 3층 층화 (complete/partial/unresolved)  ※ TRAIT_NUM=ps3+lh4
- "완전" 재정의: 핸드오프 94는 tax_lh_lbcm 단일(94종); **7수치 전부 non-null=83종**.
- overall: complete 83종/19,321행/8,080strata · partial 155종/28,971행/11,318strata · unresolved 1,029종/22,378행/12,903strata.
- discovery: complete 68종/8,553행/3,128strata · partial 142종/11,920행/3,972strata · unresolved 569종/6,050행/3,189strata.
- (strata=전체층; test-strata는 split 생성 후 부분집합). 벡터 구성·완전 정의 → D2.

### F7 — cross-group 6방향 pairwise (train=src그룹, test=tgt그룹) — 숫자만, 임계값 미제안
- overall: fish→crusta tr41,526/520종·te23,092/508종/10,831strata(공유cmp1,368) · crusta→fish 대칭 te17,710strata ·
  fish→algae te6,052/239종/3,760strata · algae→fish tr6,052/239종·te41,526/17,710strata · crusta→algae te3,760strata · algae→crusta te10,831strata.
- discovery: fish↔crusta 견고(tr20,786/446종 ↔ te2,836/7,421strata); **algae discovery=41행/12종뿐 → algae 방향은 discovery 불가, overall만**.
- 그룹 partition: fish520+crusta508+algae239=1267종. (→ 규약 D3, 임계값 director)

### 정지 (Session 2)
F1–F7 수치 보고 완료. D1–D6 director 회신 후 로더(§D) 착수. 무학습·무수정(ADORE/ccmpnn/원고 read-only).

---

## 2026-07-30 — Session 3: 실험 매트릭스 O/X 검증 (V1–V5, 코드 확인만, 학습·GPU 없음)

| cell | 판정 | 코드 근거 |
|---|---|---|
| naive × Tier1(bias) | **O** | `Naive_species_mean` 종별 train평균=글로벌+오프셋, cold→글로벌 (naive_species_baselines.py:332-363) |
| naive × Tier1′(resid) | **O** | `LightGBM_RDKit_species_residual_calibration`(:403-436)+`_species_offsets`(:142-170). ※base=LightGBM descriptors(글로벌평균 아님) |
| naive × Tier2(one-hot) | **X** | 축퇴: naive에서 one-hot종=종별평균=Tier1. 별도 naive tier2 없음(tier2는 lightgbm backbone) |
| naive × Tier4(embed) | **X** | naive는 경사학습 모델 없음 → 임베딩 학습 불가. 해당 baseline 없음 |
| naive × Tier5(phylo)/Tier6(DEB) | **X(무의미)** | 연속 16d/형질벡터는 group-mean 룩업 대상이 아님(이산 그룹 아님). naive 형태 없음 |
| lightgbm × Tier4(embed) | **X(미구현)** | 종 입력=native categorical `species_idx`만(=Tier2; rdkit_lgbm.py:40-44,129,268-274). 학습임베딩 feature주입(2단계) 없음 |
| dmpnn × Tier1 | **O** | `SpeciesBiasOnlyModel(no_species dmpnn)` (models.py:472-474) |
| graphconv × Tier1 | **O** | `SpeciesBiasOnlyModel(graphconv)` (models.py:491-492) |
| dmpnn·graphconv × Tier1′ | **O(둘 다)** | GNN-native 5-fold OOF, `--backbones default="dmpnn,graphconv"` (run_q2_gnn_oof_tier1prime.py:63-70,140,145-147) |
| dmpnn × taxonomy 3a/3b | **X 유지** | frozen one-hot 주입은 기술적 가능하나 인코딩이 GraphConv(학습 per-rank emb 합; models.py:190-199,252-257)·LightGBM(native categorical; rdkit_lgbm.py:82-86,268)와 **불일치**. 현 코드 하드차단(models.py:414-416). O/X는 director. |

- V1 부가: naive taxonomy(Tier3a/3b)는 back-off group-mean으로 **정의됨**(shrinkage/통제혼재로 Supplementary 강등). Tier5/6는 그 이전에 naive 형태 자체가 없음(연속벡터≠평균가능 그룹) → 강등사유 적용 이전에 미정의.
- V2 부가: 2단계 임베딩 없음 → 누출 논점 발생 안 함.
- V5 통제축:
  - `fixed_proj` = **Tier4(학습임베딩) 용량매칭 통제**(frozen random emb proj). GraphConv(models.py:168-171)·D-MPNN(models.py:438-445) 둘 다. **Tier2(one-hot)엔 미정의**. Tier5/6는 설계상 확장 예정(코드 미구현).
  - `shuffled`/`zero`/`dummy` = **data-side** 라벨 조작(species_controls.py:106-144), tier-무관. 종 사용 tier(1,2,3,4)에 적용. **Tier1′엔 미구현**(shuffled tier1prime 코드 없음 — grep NONE).
- ⚠ 용어 충돌: 기존 코드 주석의 "Tier4=late_fusion / Tier5=film"은 **구(舊) fusion-locus 번호**. ADORE 재번호(Tier4=학습임베딩, Tier5=phylo, Tier6=DEB)와 다름. ADORE Tier5/6는 코드에 **아직 없음**(신규 구현 대상).
- V3RunConfig species_emb_dim=16 (GNN Tier4 임베딩 16d; ccmpnn 기본 8과 별개).

### 정지 (Session 3)
V1–V5 매트릭스 O/X 보고 완료. §L 비용추정은 매트릭스 확정 후. D-MPNN taxonomy O/X = director(D1 참조).

---

## 2026-07-30 — Session 4: 매트릭스 커버리지 보완 10건 + §L 산정 (코드 확인·수치, 학습·GPU 없음)

director 보충 지시(구멍 10건). 코드 확인·feasibility·spec·수치검증·§L 열거 수행. 신규 결정 D7–D12.

### A. OOV 학습 regime — **현 코드에 없음**(신규 구현)
grep: species-dropout/collapse 학습 regime 부재(`__unknown__`은 taxonomy rank용, cold는 예측시 fallback). → oov_dropout/oov_collapse는 tiers 5/6처럼 신규 구현. 정의(마스킹율 등) director 필요(D7).

### C. designed-leaky — **tox-learn 정의 발견·이식 가능**
`build_q2_datasets.py:135-136,249`: **designed_leaky = (smiles,species) PAIR 단위 무작위 분할**(한 pair의 전 strata 동일측) → test SMILES가 train과 대량 중복(누출). group=CAS-disjoint(:221), scaffold=Bemis-Murcko(:115,223). H3 누출검정용(`analyze_q2_h3.py --leaky=replication_designed_leaky`). **ADORE 이식 가능**(smiles[조인]+species만 필요). scaffold-generic은 신규(RDKit MakeScaffoldGeneric; ADORE `split_scaffold-generic` 존재). → 사양 승인 D8.

### E. fixed_proj 확대 — **정의 가능**
Tier2(one-hot): frozen random projection of one-hot = 정의 가능(기존 fixed_proj가 곧 one-hot의 frozen emb 사영; models.py:169-170). Tier3a/3b(taxonomy): per-rank embedding을 frozen(random init)으로 = 정의 가능(tax_rank_embs에 requires_grad_(False), fixed_proj와 동형). Tier1/1′(스칼라): 매칭할 용량 없음 → **구조상 N/A**. → 구현·실행 승인 D9.

### F. zero/dummy 통제 — 이미 존재, tier 확대
`species_controls.py`: zero(:106-115, species_idx=0), dummy(:132-144, 랜덤 라벨). models.py `zero_species`는 종 사용 injection 전부 적용(:411). → 종 사용 전 tier(1,1′,2,3,4,5,6)로 확대(1′은 현재 통제 없음). **zero 해석**(Methods): zero=종 경로/파라미터는 존재하되 신호만 0 → "용량 vs 정보" 분리(no_species=경로 없음, shuffled=틀린 정보와 구분). D10.

### H. seed = **10**(tox-learn 이식): `run_q2_gnn_ladder.py:48`·`oof_tier1prime:141` `--seeds default=range(10)`.

### J. naive×Tier2·Tier4 항등 — **수치 확인됨**(synthetic)
one-hot OLS vs 종평균 max|Δ|=**2.2e-15**(=동일); species-only vs 종평균 Δ=**0.0**(=동일). 종내분산 0.942(=mean-lookup이 버리는 화합물 정보). → naive×2=X(=Tier1), naive×4=X(붕괴) 확정.

### §L — run/GPU시간 열거 (숫자만, 절삭 미제안)
가정: 통제×regime **additive**(A0), seeds GNN10/CPU1, oov 3레벨(종 tier), eval 22구성(warm 4split×2 + 종cold 1×2 + 그룹cold 6dir×2), GNN 10.5min/run(9–12).
- **총 run = 22,264**(GPU graphconv 12,100 + dmpnn 9,020 = 21,120; CPU naive 220 + lgbm 924 = 1,144). 곱셈 상한 GPU 44,000.
- **GPU시간 2장(mid)=1,848h ≈ 77일**(낙관 1,584 / 보수 2,112). CPU ≈ 38h.
- 한계절감(GPU run): group-cold 제거 −11,520 · seeds 10→5 −10,560 · 통제 전제거 −11,440 · OOV regime 제거 −6,160 · designed-leaky 또는 generic-scaffold 각 −1,920 · 종-cold −1,920.
- pending Δ: D1 dmpnn taxonomy O = +3,080 · E fixed_proj 제외 = −2,200.
- 배타/퇴화: naive×regime N/A, tier0×통제 N/A, naive×2=naive×1(수치확인), naive/lgbm×4 구조불가, 종cold·그룹cold≠warm4split곱, 그룹cold-overall⊇replication, 통제×regime 비곱(additive), **후보 퇴화: OOV×그룹cold(타깃군 이미 미관측) director 검토**.

### 정지 (Session 4)
D7–D12 회신 후 로더/신규구현 착수. §L은 D7–D12·D1 확정 시 재계산(현 수치는 명시 가정 기준). 무학습·무수정.

---

## 2026-07-30 — Session 5: 범위 정정(A·D 철회) 반영 §L 재산정 + §0-1 부호검증

director 정정: cold·cross-group은 전 셀 아님 → **tier 5,6 + baseline(0,2,4) + naive phylo/trait-kNN만**. OOV 2종도 그 cold 대상에만. warm은 OOV 축 없음. tier 1/1′/3a/3b는 warm 전용.

### §0-1 pLC50 부호 검증 — **PASS**
target = −log10(mol). ADORE `result_conc1_mean_mol_log`(=log10 mol)와 −부호로 일치(max|Δ|=4.3e-13). 부호 방향 정상: 최저농도 8.0e-21 M→pLC50 20.10(최독성), 최고농도 3.21 M→pLC50 −0.51(최저독성). mol 0/음수/NaN 0건(70,670 전행 유효). ⚠ 로더 QC용: pLC50 −0.51…20.10, 8e-21 M은 물리적 비현실 → censoring/QC 필요(부호 자체는 확정).

### §L 재산정 (정정 반영, additive, 숫자만)
- eval: warm = 4split×2data = 8(전 O셀, OOV 없음); cold = 종cold 1×2 + 그룹cold 6dir×2 = 14 (**tier 0/2/4/5/6 + naive kNN만**, OOV 3레벨).
- **총 run = 14,244**(GPU graphconv 7,180 + dmpnn 6,380 = 13,560; CPU naive 136 + lgbm 548 = 684).
- **GPU 2장 mid = 1,186 h ≈ 49일**(낙관 1,017 / 보수 1,356). CPU ≈ 23h.
- **정정 전 대비 감소**: 이전 additive 21,120 → 13,560 (×1.56 축소, −36%); 이전 곱셈상한 44,000 → 13,560 (×3.2 축소).
- **director "6~8배→1.5배" 대조**: warm-only 기준 5,440. 이전 곱셈상한/warm=8.1×("8배"✓). 정정 후 full(cold에 통제 포함)=**2.49×**, light(cold=true+OOV만)=**1.67×**. → **director 1.5배 추정치는 "cold-light" 해석과 일치**(1.67×). 2.49 vs 1.67 격차 = **cold 셀에 라벨통제(shuffled/zero/dummy/fixed_proj)를 거는지 여부**가 유일 동인(cold_full 8,120 vs cold_light 3,640 GPU). → D13.
- pending Δ: D1 dmpnn taxonomy=O → +800 GPU(taxonomy는 warm 전용).

### 선행작업 진척 / 대기 사양
- 로더·split·PCoA좌표·Tier6벡터 **아티팩트 미착수**(정정·D회신 대기; "회신 전 학습 금지" 준수). feasibility는 F1–F7 완료(PCoA 스펙트럼·Tier6 커버리지·SMILES조인 100%).
- NCBI 해상률: **오프라인 산출 불가**(외부 NCBI 자원 필요) → D5.
- 사양: designed-leaky(D8)·zero/dummy 해석(D10) 제출 완료. LightGBM Tier4 방법·trait-kNN 거리 = D14 신규.

### 정지 (Session 5)
§L 재산정+§0-1 보고. D1·D7–D14 회신 후 착수. 무학습·무수정.

---

## 2026-07-30 — Session 6: HANDOFF red-team (설계·문서 취약점 감사, 읽기 전용)

director 요청으로 HANDOFF 재정독 + 5세션 실측 대조. **신규 측정 1건 + 설계 문제 다수 발견.**

### 신규 측정 — Tier5/6 커버리지의 tax_group 편중 (중대)
| group | phylo 종%/레코드% | DEB 종%/레코드% |
|---|---|---|
| fish | 73.5 / 95.5 | 36.9 / 83.5 |
| crusta | 17.5 / 54.4 | 9.1 / 59.0 |
| algae | 23.0 / **17.0** | **0 / 0** |
- pdm 853종 구성: fish 382 · crusta 89 · algae 55 · (mortality 부재 327). → 계통행렬 자체가 어류편향.
- **DEB 조류 커버리지 = 0**(AmP=동물 전용). → Tier6은 구조상 fish+crusta 전용. 조류 관련 cross-group 4/6 방향 = DEB 신호 0(data-impossible).
- Tier5도 조류 레코드 17%뿐 → 조류 방향 phylo 전이 구조적 미약.

### HANDOFF 허점(확인): chemicals 9,404→실제 4,702(조인 100%) · phylo "~67%"=순진비율(실제 41.5%종) · §B flag2 D-MPNN "불가"=오판(context hook 가능, F5) · §G "구조결론 강화"=미검증 가정.
### 설계 문제: 커버리지-지지도 교란(covered=dense 4.3×/9.3×) · Tier5 vs Tier4 warm 중복 예상 · DEB=지지도 프록시 위험 · 조류 discovery 부재(41rec) · top5%=73.8% 레코드로 sparse CI 넓음.
### 추가실험 제안 5건: covered/uncovered 층화 readout · indicator-only ablation · 군별 tier-가용성 사전등록 · cross-group을 tier5/6 주검정(fish↔crusta) · support-bin별 Δ. (핵심 서사 "정보요구↑→커버리지 붕괴"는 오히려 강화됨.)

### 정지 (Session 6)
red-team 보고. D15(신규 tier 군별 커버리지 처리·algae 구조배제) 추가. 무학습·무수정.

---

## 2026-07-30 — Session 7: PART C 극단값/절단 규명 + 게이트(순서 step 1, 읽기·카운트만)

director 5차 회신(이전 전부 대체). 순서 동결 step 1. tier 성능 미접촉.

### PART C-1 절단 플래그 보존 — **부등호는 값에 없음(중요)**
- LC50/EC50 **점값 연산자** `result_conc1_mean_op`(processed) = {nan(점값) 68,193 · min_max_average 2,001 · ~ 476}, raw `conc1_mean_op`도 동일 부류. **`>`/`<` 0건** → 모델링 대상 값은 **부등호 미표기 점추정**.
- 단 raw `conc1_min_op`(`>`1362·`<`8·`>=`2) / `conc1_max_op`(`<`1362·`>`30) = 부등호 **존재하나 농도 경계(CI/범위)용**, 1362쌍, **processed 파일엔 미포함(집계시 탈락)**.
- ⟹ 우리 파이프라인 대상(LC50 점값)엔 **보존할 절단 플래그 없음**(ADORE LC50/EC50=점추정). 경계 부등호는 raw에만·경계용. director 가설("저독성 극단=`>X` 오기록")은 **데이터상 미성립**.

### PART C-2 극단값 실태 (36행=0.05%, 둘 다 절단 아님)
- **저독성**(pLC50≤0, conc≥1M): 20행(0.028%)·8종·6화합물·crusta11/fish7/algae2·**전부 점값(op=nan)**·pLC50 −0.51…0. phylo 70%/DEB 40% 커버.
- **고독성**(pLC50≥12): 16행(0.023%)·7종(**전부 crusta**)·7화합물 = **전부 강력 살충제**(Endrin 72-20-8, Malathion, Chlorpyrifos, Deltamethrin…) at sub-picomolar → **단위변환 오류**(점값, 절단 아님). phylo/DEB 81% 커버.
- 맥락: pLC50≥9 = 1,305행(1.85%, crusta 살충제)=현실적 강독성 꼬리(nM–pM). 비현실 컷 ≈ pLC50 11–12.
- 화합물 분포 **비무작위**(고독성=갑각 살충제, 저독성=특정 저독 화합물) — 제외 시 비무작위 손실 확인.

### PART E taxdump = **성공(옵션1)**
`ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz` HTTP200, 76,368,992B 다운로드·검증(gzip OK, names.dmp+nodes.dmp). `results/q2_v4/data/_ext/taxdump.tar.gz` 캐시. 오프라인 NCBI 해상 가능 → 비용사다리 단계에서 실행(지금 아님).

### 게이트 확인
- **§L seed = GNN 10(graphconv/dmpnn)·결정론 1** 가정 확정 = D11 일치 → **재산정 불요**. (D-MPNN taxonomy=O 편입 시 +800 GPU warm.)
- **파이프라인 tier별 상이 seed 수 = 이미 지원**: `bootstrap_q2_ladder.py:4-8,98,118` — GNN P[rows,seeds] seed 재표집 + 결정론 P[rows,1] block-only. TOST/δ 결정층은 신규(step6).
- 4차 정정 범위(cold=tier0/2/4/5/6+naive kNN, OOV cold限, warm無OOV) **이견 없음**.
- **D9 fixed_proj**: Tier3/4/5 정의가능(freeze per-rank emb/embedding). **Tier2 제외**(one-hot readout 폭=n_species≈1267 → 용량매칭 부자연) 사유기록. **D14 trait-kNN**: Euclidean(0대치)+Gower 둘 다 구현, 미해상=이웃제외·test잔류, 주설정 director.

### 정지 (Session 7)
step1 완료. step2(censoring 규칙 사전등록) director 회신 대기 → D16. tier 성능·δ 미착수.

---

## 2026-07-30 — Session 8: step2 사전등록 산출 + §L-2 이기종 GPU 계측 (읽기/카운트/계측, 학습 아님)

director 6차 회신. 사전등록 초안 = `EXP\PREREGISTRATION_draft.md`.

### §1 임계규칙(0<pLC50<12) 적용 — 제외 36행
- 저독성 20 + 고독성 16. 14종·13 CAS. tax_group crusta 27/fish 7/algae 2. 커버층 phylo+DEB 21/uncovered 9/phylo_only 6.
- 목록 저장: `results/q2_v4/data/_ext/prereg_excluded_rows.csv`.
- **⚠ 사실(서사 미조정)**: 저독성 20행 = **실재 저독성 miscible 용매/염**(ethanol/methanol/acetone/DMSO/ethylene glycol/NaCl). LC50≥1M 물리적 정상 → "가용성 초과" 근거 부정확, 하한 컷이 실데이터 제거. → D16.
- 고독성 16행 = crusta 살충제 sub-pM = 단위오류(확정).

### §3 raw 경계 부등호 성격 — **구간 보고(interval)**
`conc1_min_op '>'` 1362 · `max_op '<'` 1362 · 동일행 1355쌍. 예: min>32,max<100,mean~60(=min_max_average 중점). ⟹ **LC50를 구간(>min,<max)으로 보고, 중점을 점값화**. endpoint LC50 1015/EC50 347. 처리 90% 존재, group fish568/crusta570/algae90. 모델 미사용, Limitations용("정량값=점추정, raw 구간정보는 집계시 유실").

### §5 Tier6 fixed_proj — 정의가능(D9 누락 정정)
DEB=고정 외부벡터 → 동차원 frozen random 대체로 용량매칭 정의가능. 광폭/협폭 지시자 차원 상이 → 각 개별 매칭. tier 3a/3b/4/5/6 적용, tier2 제외(승인), 1/1′ N/A.

### §L-2 이기종 GPU 계측 (PCI_BUS_ID; 계측 후 폐기)
| workload | 5060Ti | 4090 | 4090/5060Ti |
|---|---|---|---|
| 실설정 h300 d3 b256 (launch-bound) | 0.81 s/ep | 1.17 | **0.69× (4090 느림)** |
| heavy h600 d5 b1024 (compute-bound) | 4.45 | 2.72 | **1.64×** |
| pure matmul fp32 | 15.8 TFLOPs | 26.2 | 1.66× |
- **핵심**: 실설정은 **CPU/launch-bound**(peak VRAM 0.12 GiB) → 4090 이점 상실. VRAM 비제약(b1024도 0.65 GiB). **배치 키워 compute-bound화**하면 4090 1.64×. → 하이퍼파라미터 통일은 VRAM상 자유(양 카드 대형배치 수용). 배치 미상향 시 2카드 speedup ≈1.69×(내 §L "×2" 가정보다 낮음), 상향 시 ≈2.64×.
- §L-2 축갱신 GPU: 기존 13,560 + dmpnn taxonomy 800 + rank절단(2조건×2 GNN×10=40) ≈ **14,400**. 극단값 −36행=무시.

### 정지 (Session 8)
사전등록 초안·제외목록·경계성격·§L-2·Tier6 보고. **사전등록 확정(D16 등) 전 학습 금지.** 무학습.

---

## 2026-07-30 — Session 9: PART0 분해 + D16 상한검증 + D17 무결성 관문 (계측·읽기)

director 7차 회신(무결성 우선). 방법 A(배치상향) 탈락. 순서 step2 계속.

### D16 하한 확정 + 상한 문헌검증
- **하한 = −1.75**(물 55.5 M 가드레일; 0행 제외). "용해도 초과" 근거 폐기(제외 20행이 실재 저독 용매/염).
- **상한 12 검증**(WebSearch): Endrin 문헌 D.magna LC50 **88–352 µg/L** vs 기록 0.0006–0.06 ng/L(6–8자릿수↓=오류 확정). Malathion 2.7e-15 mg/L=비현실. **단 Deltamethrin 기록 0.32 ng/L(pLC50 12.2)는 문헌 민감갑각 2.6–68 ng/L보다 ~1자릿수만 낮아 실재 초강독 개연** → 상한 12가 pyrethroid 실측 클립 가능. 상한 유지/상향 = director(D16). pre-reg 갱신됨.

### PART 0 tier×평가단계 2D 분해 (GPU run / as-is 2장 wall, 가정: per-run 10.5min·1.69×·통제 additive)
- 시나리오: **(가) tier0-4 warm = 4,480 run / 19.3일** · **(나) tier0-6 warm main = 6,080 / 26.2일** · **(다) full = 12,720(cold-light)~18,880(cold-full) / 54.9~81.5일**.
- 증분: 가→나 +1,600(+6.9일; tier5/6 warm=붕괴 최초 노출) · 나→다 +6,640~12,800(+28.7~55.3일; cold+변형).
- add-on: tier5 PCoA sweep(+4dim)=640 · tier6 2변형=warm800+cold1960 · rank절단=40 · CA2F=1,520.
- 구조관찰: tier0-4 전부 100% 커버 → (가)는 커버리지 사다리 평평; 붕괴는 tier5(41.5%)/6(18.8%)에서만 = (나)의 +6.9일. cold(다)는 "외삽" 질문(별개·고비용).

### D17 무결성 관문 = **PASS(bit-identical)**
- 결정론 확인: 단일 A vs B = **max_abs 0.0, bit-identical**. 패킹 P1/P2 vs 단일 = **바이트 동일**. ⟹ 카드당 다중 job이 수치 불변(하이퍼파라미터·lr·seed·δ 무손상). → **방법 B 채택 가능**.
- 최적 N(처리량): 5060Ti N=1 10,280→**N=2 14,117(1.37×)**→N=4 13,118(과포화 하락). 24 logical core. **4090은 N=2 6,152 < N=1 7,007(패킹 손해)**. ⟹ **이기종 최적: 5060Ti=N2, 4090=N1**. 결합 ≈ **2.05× 단일5060Ti**(naive 1/card 1.68× 대비). wall as-is 62일→**~51일**. (한계=CPU/launch-bound 파이프라인.)
- 벤ign 경고: 동시 기동 시 OpenCL temp.txt race(무해, 결과 불변).

### 정지 (Session 9)
PART0·D16·D17 보고. D16 상한·경계 확정 + PART0 경계 결정 후 데이터재구축. 무학습.

---

## 2026-07-30 — Session 10: D16 3경로 판별 + D18 세안 + 코드확인 + D17 승인반영

director 8차(무결성 우선). 전역 상한 폐기 → 3경로. conda run.

### D16 — 전역 상한 폐기, 3경로
- 경로1 **막힘**: raw `conc1_unit` 없음(conc2/3만) → 원단위 미보존, mol×MW 변환 정확 검증(median ratio 1.0, 100% 일치, MW결측 0). 정합검사 무효.
- 경로2 **검출하한**: mass=conc_mol×MW×1e9 ng/L, **<0.1 제외**. 문헌 MDL 0.01–1.32 ng/L·LOQ~0.1↓. 플래그 0.05→8/**0.1→10**/0.5→24/1.0→33행. 0.5는 deltamethrin(0.32) 클립→부적합. pLC50 9–12(1,289행) 중 <0.1 = **0행**.
- 경로3 **겹침 6행**(≥0.1 & pLC50≥12): 보존=deltamethrin(0.32)·pyrethroid 76703-62-3(0.42); 제외=Endrin(Asellus 0.13–0.28, 문헌 88–352 µg/L)·methyl parathion(0.14). **4행**.
- **초안 제외 14행**(검출한계10+path3 4), **pyrethroid 2행 구제**(구 규칙 16 전부제외 대비). reason 컬럼 목록 = `data/_ext/prereg_excluded_DRAFT_D16.csv`. 임계·path3 = director 확정.

### D18 — 세 안 (GPU run / as-is 2장 / 방법B)  ※cold=OOV3+shuffled, warm=full통제
| 안 | 범위 | GPU run | as-is | 방법B |
|---|---|---|---|---|
| A | tier0-4 warm + {0,2,3a,3b,4} cold | 9,240 | 39.9일 | 32.9일 |
| B | tier0-6 warm + {0,2,3a,3b,4} cold | 10,840 | 46.8일 | 38.6일 |
| C | tier0-4 warm + {0,2,4} cold(원안) | 7,000 | 30.2일 | 24.9일 |
- add-on(별도): rank절단 40·CA2F 1,520·indicator-only Tier6 warm 160(main)/800(full)·tier5 sweep 640·tier6 2변형 800.
- warm 통제기여: A/C 3,360(main 1,120) · B 4,640(main 1,440).
- **코드확인**: cold tier2/4 = OOV remap 없음(`n_species=full.max()+1`, runner:431) → 미학습 종은 자기 idx의 **미학습 임베딩/one-hot** = 신호 없음(무승부 공전 확정). **taxonomy(3a/3b) cold = 기술적 가능·비퇴화**(tax_codes 전종·형제종 통해 per-rank emb 학습 전이; ADORE taxonomy 100%). GraphConv/LGBM 준비됨, D-MPNN은 신규 학습 context 후.
- **주장 매핑**: (C) warm 종효과만; cold 공전(닫는 주장 없음). (A) +cold/cross-group taxonomy 외삽(종정보 표현 미관측종 전이). (B) +**claim B(커버층 tier5/6 무승부)**·커버리지붕괴·비용사다리; Phase2 잔여=tier5/6 **cold**·sweep·2변형·CA2F.

### D17 — 방법 B 승인 반영
- 5060Ti 2job/4090 1job(계측 최적). 배치256·원 lr·seed 불변. 메타에 카드모델·동시 job수 기록.
- 임시파일: **예측/출력 경로는 job별 분리(충돌 없음)**. OpenCL vendor temp.txt만 공유(무해; 관문서 bit-identical 이미 입증). 권고: job별 TMPDIR로 경고 제거(선택).

### 정지 (Session 10)
D16 임계·path3 확정 + D18 경계(A/B/C) director 지정 후 사전등록 확정→데이터재구축. 무학습.

---

## 2026-07-30 — Session 11: Q1 backbone 구성 + Q2 캐싱 프로파일 + Q3 (읽기·계측)

director 9차(무결성 중립 가속). 학습 미착수.

### Q1 — A/B/C 일수 구성
- 전 GPU tier = **GraphConv + D-MPNN 2 GNN backbone × seed 10**. LightGBM+naive = CPU(GPU 일수 미포함). "2bb×10=20 run-unit/config"는 **20 seed 아님**. seed=10 전 항목 확인.
- **D-MPNN taxonomy = A/B에 포함됨**(warm 800 + cold 1120 = 1,920 GPU = 6.8 방법B일). 이전 "+800"은 warm분; 이제 warm+cold 전부 A/B에 baked.
- cold taxonomy(A/B) = **2 GPU backbone**(GC+DM). LightGBM(3번째)=CPU, GPU 일수 ~0.
- taxonomy를 GraphConv-only로 하면: A 7,320/26.0일 · B 8,920/31.7일 · C 6,200/22.1일(C는 cold taxonomy 없음, warm만 −800).

### Q2 — 그래프 조립 프로파일 (5060Ti, 실설정 h300d3b256)
- **molgraph featurization = init 1회 캐시**(dataset.py:64-69, `featurize_init 0.012s`) → **epoch마다 재생성 아님**(director 가설 정정: RDKit 재featurize 없음).
- epoch 시간 분해: **assemble(CPU) ~17% · transfer ~2% · compute(GPU) ~81%**. 단 81%는 VRAM 0.12 GiB·소커널 다수 = **launch-bound**(FLOP-bound 아님).
- 캐싱 여지: **전체 배치 캐시 = 무결성 위반**(배치 매 epoch 재셔플; 순서 고정 시 결과 변경). molgraph는 이미 캐시. → **무결성 중립 이득 = async prefetch(assemble와 compute 중첩)로 ~17% 은닉 = ~15-19%**. 단 **방법 B 패킹이 이미 job 간 CPU-assemble/GPU-compute 중첩을 실현**(5060Ti N2 1.37× 이유) → 추가 이득 marginal. 구현비용 중, 메모리 저.
- train loop = **prefetch 없는 동기 루프**(runner.py:395-403): assemble→transfer→compute 직렬 → GPU 유휴.

### Q3 — 추가 무결성 중립 여지
- job별 `torch.set_num_threads` 튜닝(24코어; N=4 과포화 원인) → 패킹 knee 상향 가능(방법 B와 병용). 
- 씨드 간 molgraph 공유 캐시(디스크) → ~3s/run(~3%, 미미). 예측 CSV I/O·중간산출 = 미미.
- CUDA graphs(launch overhead 절감) = 배치별 원자수 가변→shape 가변→비실용(패딩 필요).
- **결론: 방법 B(채택됨) + job별 스레드 튜닝 외 큰 이득 없음.** 병목 81% launch-bound는 배치/정밀도(방법 A, 탈락) 없이는 해소 불가.

### 정지 (Session 11)
Q1-3 보고. D16 임계·D18 경계 확정 후 사전등록→데이터재구축. 무학습.

---

## 2026-07-30 — Session 12: 전 결정 종결 → 사전등록 확정 + 데이터 재구축(step 3)

director 10차(전 결정 종결). Phase 1 = 안 A. D16 최종. step 2 종결 → step 3 실행.

### 사전등록 확정
`PREREGISTRATION.md`(확정본) 작성: §1 Phase1=A · §2 tier5·6 미평가(문구) · §3 D16 3경로 · §4 6항목(5/6사유·cold목적·매트릭스X성격·δ순서·cold통제·FDR) · §5 원고규칙6.

### 데이터 재구축 (`scripts/build_adore_datasets.py` 실행 완료)
- 로더: mortality + RDKit SMILES조인(100%) + MW조인 → D16 per-record → strata 집계((smiles,species,endpoint,dur)) → target=−log10(mean conc_mol)=pLC50 → discovery/replication → split 4종.
- **D16**: raw(conc>0) 70,670 → **제외 14**(검출한계10+path3 4, 전부 replication 48h Daphnia 살충제) → **kept 70,656**. audit trail(문헌·자릿수·판정논리·reason) + **타임스탬프 2026-07-30T15:07 UTC(학습 전)** = `data/exclusion_audit_trail.csv`.
- **최종 카운트**:
  - discovery: 26,523 records → **10,149 strata · 779 종 · 2,081 화합물**.
  - replication: 44,133 records → **21,759 strata · 1,006 종 · 2,760 화합물**.
- split(각 discovery/replication): group·scaffold·scaffold_generic = **compound-disjoint(overlap 0)**, designed-leaky = compound overlap 680/894(의도적 leak). cold species: group 61/54, scaffold 35/71 등. 8 train/test 쌍 + ledger 기록.
- 스키마: smiles·species·species_idx·endpoint·duration·target_log10·effect_value·n_source_rows·n_cas·cas_list·tax_class/order/family/genus. GraphDataset·naive·lgbm 호환.

### 학습 전 잔여 배선(결정 아님·wiring)
- TAX_RANKS를 ADORE 4-rank(tax_class/order/family/genus)로 매핑(tier 3a; rdkit_lgbm/models/naive).
- NCBI 해상(tier 3b): taxdump(`data/_ext/taxdump.tar.gz`) 오프라인 해상 → ncbi_* 컬럼.
- config `q2_dataset_adore.json` · run 스크립트 split 목록에 scaffold_generic 추가.
- FDR 경계 초안 = `audit/fdr_family_boundary_DRAFT.md`(모호 3건 director).

### 정지 (Session 12)
step 3 완료. 잔여 배선 + FDR 경계 3건 confirm 후 step 4(학습, 방법 B). 학습 미착수.

---

## 2026-07-30 — Session 13: FDR 3패밀리 확정 + 잔여 배선 + 종-cold 규모

director 11차. FDR 확정(replication=confirmatory·cross-group=primary·group=primary). 잔여 배선 착수.

### 종-cold 규모 (설계 20% 홀드아웃, 지지도 층화)
- **discovery**: test 156종/1,395strata/2,924records (fish 88/964, crusta 64/423, **algae 4/8** — 총 12종뿐).
- **replication**: test 200종/5,834strata/13,179records (fish 72/4,312, crusta 80/994, algae 48/528).
- **정정**: 앞선 `group cold 61/54` = **compound-disjoint group split의 우연 미학습 종**(설계 아님). 설계 종-cold(156/200)와 완전 별개.

### 잔여 배선 완료
1. **TAX_RANKS → ADORE 4-rank**(rdkit_lgbm.py): original=tax_class/order/family/genus, ncbi=ncbi_*. (kingdom/phylum 미사용, §A-1.)
2. **NCBI 해상(taxdump 오프라인)**: 해상률 **종 98.9% / 레코드 99.7%**(scientific 897·synonym 202·genus_only 154·not_found 14). **4-rank 추출 종 96.5%/레코드 98.3%**. 실패: not_found 14(미등재/철자), genus_only 154(종 미해결). lineage CSV + provenance(sha256·크기·일시) = `data/_ext/ncbi_taxonomy_by_species.csv`·`ncbi_provenance.json`. ⚠ NCBI class="Actinopteri" vs native "Actinopterygii" = 3a/3b DB출처 차이.
3. 로더 재실행 → 데이터셋에 tax_*(native 100%) + ncbi_* 컬럼 병기. scaffold_generic split 산출됨(run 스크립트 split 목록 추가 필요).
- **비용사다리 대비**: taxonomy 99–100%(cheap, self-resolved도 99%) ≫ phylo 75% ≫ DEB 68% — 정보요구↑ 커버리지 붕괴.

### LightGBM Tier4 = 제안만(구현 전, 승인 대기) → director_결정필요 D-LGBM4
2단계뿐(트리 종단간 불가): A=OOF 종인자(누출안전, target-encoding 성격) / B=GNN 임베딩 주입(cross-backbone). val carve 충돌 없음(전처리).

### FDR = `audit/fdr_family_boundary.md`(확정). 종-cold만 판정 대기.

### 정지 (Session 13)
FDR·배선·종-cold 규모 보고. **학습 gate**: 종-cold 판정 + D-LGBM4 승인. 학습 미착수.

---

## 2026-07-30 — Session 14: 설계 종결(게이트키핑 확정) + 구현 잔여 착수

director 12차. 종-cold=primary·D-LGBM4=후보A 승인. 사전등록 갱신 + 구현.

### 사전등록 갱신(PREREGISTRATION.md §4G 게이트키핑 + §5-7 비용서사)
- 계층 게이트: warm→종-cold→cross-group, (backbone,tier쌍) 독립 체인. **통과=TOST `동등`만**(일반과 방향 반대), `불확정`·`유의차` 탈락. δ 종속성·cold `불확정` 증가 예상·순서 정당성·3범주 분리보고 기록. confirmatory 동일 미러. 종-cold=primary(집계전체)/exploratory(group별, algae 4종).
- §5-7: taxonomy 성능배제 vs phylo/DEB 커버리지배제 두 층; NCBI 96.5% 반영; "정보량↑→커버리지급감" 헤드라인 taxonomy 미성립 기록.

### §5-2 LightGBM Tier4 누수 절차 = Tier 1′와 동일(검증)
Tier 1′ = `_oof_lightgbm_predictions`(5-fold OOF no-species base) + `_species_offsets`(SPEC 4-0b: stratum_eff purge → 종 OOF-residual mean → re-add). Tier4 계획 = **동일 두 함수 재사용**(OOF base + stratum purge) → 그 위에 k-D SVD(1-D mean 대체)·feature 주입(사후가산 대체). 누수 격리 절차 **동일** → §5-2 통과. 차이는 downstream(k-D·usage)=의도된 Tier4-vs-1′ 구별.

### 구현 잔여
1. **D-MPNN taxonomy context = 완료·검증**. `DMPNNTaxonomyContext(ContextBase)`(models.py): per-rank 학습 nn.Embedding 합·readout concat, GraphConv와 **동일 파라미터(smoke: tax_rank_emb 720 = graphconv 720)**·차원·init. build_v3_model dmpnn 분기서 taxonomy ValueError 제거·context 배선. smoke: dmpnn·graphconv × true/zero/shuffled 전부 forward+backward OK. ccmpnn 무수정(공개 seam).
4. config `q2_dataset_adore.json` 작성. scaffold_generic 데이터셋 산출됨(run 스크립트 split 목록 추가만).
2. **Tier 1′ shuffled**(전 backbone): 오프셋 계산 시 species 라벨 순열(seeded) — `run_q2_gnn_oof_tier1prime.py`에 control 인자 추가(소규모, 진행).
3. **adore_t* 네임스페이스**: 매핑표 = {adore_t0..t6, t1prime} ↔ variant. 출력/로그/파일명 접두 적용(출력층, 진행). 구 fusion-locus(late_fusion/film) 격리 확인.

### 게이트키핑·TOST·3범주 로직 = step 6(δ 동결 후)
구현 위치 = 신규 `scripts/run_q2_gatekeeping.py`(δ 동결본 + primary/confirmatory 예측 로드 → TOST 3범주 → 계층 게이트 → BH-FDR/패밀리 → confirmatory ⊆ primary set·param-hash 대조·불일치 abort). 학습(step4) 이후 단계라 착수 전 불필요.

### 정지 (Session 14)
설계 종결·사전등록 갱신·D-MPNN taxonomy 완료·§5-2 통과. 착수 조건 충족(사전등록 확정 + §5-2). 잔여 소규모 구현(#2·#3·#4-split목록) 마무리 후 step 4. 학습 미착수.

---

## 2026-07-30 — Session 15: 착수 직전 gate (k-D 사양·δ pooling 확정·parity)

director 13차. §5-2 재판정(k-D 사양 요구)·게이트 δ·pooling 확정·확대 스모크 요구.

### §1 LightGBM Tier4 k-D 사양 = 제출(승인 대기, 구현 금지) → D-LGBM4
- k=16(GNN 매칭). 행렬 = 종×train화합물 OOF-잔차. 누출 = **double-OOF SVD**(fold별 재적합, Tier 1′ OOF 원리 동일).
- **⚠ shrinkage 교락 정직보고**: Tier 1′=raw mean(무 shrinkage), k-D SVD=low-rank 암묵 정규화 → 비교가 정규화 차이로 교락 가능. 1-D→k-D는 회피 불가. director 택1: 감수(Methods명시)/양쪽 동일shrinkage/미실행(V2 X 유지).

### §2 결정론 tier 게이트 δ = 자체 block bootstrap δ (pre-reg §4G-6). GNN δ 차용=민감도 병기. 대소 사전단정 금지(bootstrap SD=표본불확실성, 0 아님).

### §3 δ pooling 최종 사양 = 확정 (pre-reg §4δ)
- condition = (backbone,tier,control,split,data). within = 10 seed RMSE SD. pooling 집합 = **primary(warm×discovery×group)의 전 GNN condition**. δ = sqrt(Σ(n−1)s²/Σ(n−1)). 결과 관찰 후 변경 금지(자유도 0). 동결파일에 사양+값 기록·불변검증.

### §8-1 D-MPNN taxonomy shrinkage parity = 검증 (param count 넘어)
- 최적화 `AdamW(model.parameters(), weight_decay=1e-5)` — 전 파라미터 균일(param-group 배제 없음). 양 backbone tax `nn.Embedding(card,emb_dim)` per-rank 합, **max_norm·init override·scaling·prior 전무**. FFN dropout 동일. ⟹ shrinkage = 구조적 per-rank 공유 + 균일 weight_decay, **backbone 동일**(비파라미터 shrinkage 차이 없음).

### §8-2 게이트 스크립트 산출 = 사양 확정(step6): 3범주 개수집계·본문/Supp 구분·결정론 두 δ 병기 및 상이판정 목록.

### 구현 잔여(승인 후): Tier 1′ shuffled·adore_t*·run split 목록. §7 확대 스모크(전 matrix×backbone×통제×cold경로·resumability·패킹·디스크) = §1 승인·§3 확정 후.

### 정지 (Session 15)
gate: **§1 승인 대기**(LightGBM Tier4)·§3 확정(제출). 승인 후 → 구현 잔여 → 확대 스모크 → 학습(블록 A→B). 학습 미착수.

---

## 2026-07-30 — Session 16: 전 blocker 종료 → 구현·스모크 착수

director 14차. D-LGBM4=(i) 승인(k-D SVD, 교락 감수·Methods명시)·δ pooling=main 한정 확정.

### 사전등록 확정
- §4δ: pooling = **main 조건 한정**(통제 제외; 근거=δ는 신호학습 모델의 재실행잡음, 통제는 종류 다른 양). δ=√(Σ(n−1)s²/Σ(n−1)), 동결·불변. **분석 자유도 0.**
- D-LGBM4 교락(SVD 암묵정규화 vs Tier1′ raw mean) = LightGBM 고유결함 아님(Tier1′=(ref) 기준선), 교락 범위 = Tier4 vs Tier1′(secondary) 한정, 주 비교(vs one-hot) 무손상 — Methods/사전등록 기록.

### 실데이터 통합 스모크(ADORE discovery_group, 1500행, 2ep, 5060Ti) — 핵심 tier PASS
| | dmpnn | graphconv |
|---|---|---|
| t0/t1/t2/t4/t3a/t3b | OK, rmse 1.63–1.81 | OK, 1.73–1.80 |
- **신규 D-MPNN taxonomy context = 실데이터 OK**(3a native + 3b ncbi, 양 backbone). rmse 자릿수 정상(tox-learn baseline 1.18–1.23과 동일 order; 5.0=버그/0.01=누출 아님). ※tier 비교 안 함(관찰만).
- **스모크가 갭 포착**: `fixed_proj_species_taxonomy_*` = 미지원 변형 → tier 3a/3b fixed_proj(D9)가 frozen-taxonomy 경로로 미구현 → Task #5.

### 잔여(Task #1–5): Tier1′ shuffled · scaffold_generic+adore_t* · LightGBM Tier4(승인) · 확대스모크 · fixed_proj taxonomy. 이후 33일 학습(블록A→B, 방법B).

### 정지 (Session 16)
사전등록 최종확정·D-MPNN taxonomy 실데이터 검증·핵심 tier 스모크 OK. 잔여 구현 5건 → 확대 스모크 → 학습. **아직 대량 미착수**(스모크 선행 규칙). 진행중.

---

## 2026-07-30 — Session 17: fixed_proj 구성 확정 + discovery SD + 중단규칙

director 15차. fixed_proj (가) 확정·상식범위 조작정의.

### §2-1 discovery 타깃 pLC50 SD = **1.7195**(pool; train-only 1.6855). mean 4.98, range −0.17…11.52, 10,149 strata.
- **중단 상한**: tier0 RMSE ≥ **1.7195** 즉시중단. **하한**: RMSE < **0.8597**(0.5×SD) 누출의심 보고(중단 아님). 스모크 미적용. pre-reg §6 기록.

### fixed_proj (가) = 기존 `true_species_fixed_proj`(종별 frozen random 16-d) — Task #5 해소(신규 코드 불요)
- 이전 스모크 "갭"은 **malformed 변형명**(`fixed_proj_species_taxonomy_*`)이었을 뿐. 올바른 `true_species_fixed_proj`가 (가) 구성 그대로 = tier 3a/3b/4 **공유 용량 baseline**(셋 다 readout 16-d concat). 실데이터 OK(dmpnn 1.61/graphconv 1.73).
- rank별 frozen 배제(계통구조 잔존)·shuffled 중복 유지(이중 차단) = pre-reg §4F.

### 잔여(Task): #1 Tier1′ shuffled · #2 scaffold_generic+adore_t* · #3 LightGBM Tier4 SVD → #4 확대스모크 → 학습.

### 정지 (Session 17)
SD·중단규칙·fixed_proj 확정. #1/#2/#3 구현 → 확대 스모크 → 학습. 대량 미착수.

---

## 2026-07-30 — Session 18: 중단 임계 파티션별 분리 (director 지적)

director 16차: 임계에 파티션 미지정 → 33일 오작동 위험. discovery/replication 기준 분리 필요.

### 파티션별 residualized SD (jcim_v3.stratum로 산출)
- **discovery**: LC50@96 단일 stratum → 잔차화=항등 → SD **1.7195**. 상한 1.7195 / 하한 0.8598.
- **replication**: 7 strata(LC50@24/48/72·EC50@24/48/72/96) → 잔차 SD **1.7094**. 상한 1.7094 / 하한 0.8547.
- **⚠ 실측 정정**: 잔차화가 replication 분산 **0.5%만 제거**(1.7139→1.7094). 주효과가 compound×species 대비 작아 예상 대폭감소(~1.3) 미발생 → repl SD ≈ disc SD. 그래도 파티션별 residualized SD로 분리 고정(원리 준수). pre-reg §6 파티션별 표.
- 적용: 상한 = 각 파티션 tier0 완료 시; 하한 = 각 파티션 각 tier.

### 정지 (Session 18)
파티션별 중단임계 확정(pre-reg §6). 잔여 #1/#2/#3 구현 → 확대 스모크 → 학습. 대량 미착수.

---

## 2026-07-30 — Session 19: 0.5% 검증 (director 요청)

7 strata 평균 pLC50 출력 → 0.5% 정상성 확인.

### 결과: 계산 정상(설명 1)
- between-stratum var/total = **0.52%** = 잔차화 제거량 0.5% 정확 일치 → **계산 오류 아님**.
- 7 strata 평균 span **0.377 log**(LC50@24 4.744 → EC50@72 5.121, grand 4.861 ±0.26). 방향은 예상대로(EC50·장시간→높은 pLC50)이나 marginal 크기가 compound 구성 교락(strata별 화합물 집합 상이). within SD ~1.7 대비 between 0.38 작음 → 0.5%.
- director "0.5 log→2%" 과대추정: total var 2.94 커서 0.38 log span → 0.5%(산술 정합).
- Methods 기록: 잔차화 절차 적용하나 실제 제거 0.5% 명기(오해 방지). 표 = `data/replication_strata_toxicity.csv`(Supplementary). pre-reg §6 검증 추가.

### 정지 (Session 19)
0.5% 검증 완료(정상). 잔여 #1/#2/#3 → 확대 스모크 → 학습. 대량 미착수.

---

## 2026-07-30 — Session 20: 자율 구현 #1–#3 완료 + cross-group 설계결함 발견(§3-3 정지)

director 자율진행 승인 후 연속 구현.

### 구현 완료·검증
- **#1 Tier 1′ shuffled**(run_q2_gnn_oof_tier1prime.py): `--controls true,shuffled`, 오프셋 단계 종→오프셋 순열(seeded, non-identity). base+stratum 불변.
- **#2 adore_t* 네임스페이스**: 매핑표 `audit/adore_tier_namespace.md`(t0/1/1prime/2/3a/3b/4/4_fixedproj). **fusion-locus 격리**: film=구 t5→Phase1 제외, late_fusion=t4, early/message 미사용. scaffold_generic = split 목록(launch 파라미터). run 파일명=variant-keyed(resumability), adore_tier 라벨=aggregation층.
- **#3 LightGBM Tier4 double-OOF SVD**(run_q2_lgbm_tier4.py): 종×train화합물 purged-OOF-residual 행렬, 외부 5-fold OOF SVD(k=16), Tier1′ OOF+purge 재사용. discovery_group s0 RMSE **1.3024**(정상, ~tox-learn LGBM 1.23). adore_tier=t4 컬럼.

### ★ cross-group taxonomy 전이 전제 반증 (§3-3 정지 조건)
- fish/crusta/algae 4-rank **공유 랭크 0**(class/order/family/genus, native·NCBI). → cross-group taxonomy = 미학습 = degenerate(≈fixed_proj). D18 "cross-group taxonomy 전이" 거짓.
- 대조: species-cold 랭크 공유(class 98.7/order 96.8/family 86.5/genus 52.6%) → 전이 성립.
- 귀결: cross-group=primary 근거 붕괴(공전·판별력 없음). → **director_결정필요 D-XGROUP**(제외 vs null 재프레임). species-cold=primary 무영향.

### 정지 (Session 20) — §3-3 발동
구현 #1–#3 완료. **cross-group 설계결함으로 학습 미착수, director 회신(D-XGROUP) 대기.** 확대 스모크·cold-split(cross-group 부분)·학습 보류.

---

## 2026-07-30 — Session 21: cross-group 제외 반영 + 확대 스모크 통과 + 블록 A 학습 착수

director: cross-group Phase1 제외. 자율 진행.

### 반영·산출
- 사전등록 §7(cross-group 구조배제·외삽범위 축)·§8(OOV 미구현·블록B 보류)·§9(일정) + §4G 2단 게이트. 구조표 `data/crossgroup_rank_overlap.csv`(전 쌍 랭크 0).
- **OOV 확인**: 종 드롭아웃/OOV 토큰 **미구현**(grep 무결과). cold 종 = 자기 idx 미학습 행. → **블록 B(종-cold) D-OOV 대기, 블록 A 진행**.
- 일정 재산정: Plan A **4,880 GPU**(warm 4,480+종cold 360+rank 40), 방법B **17.4일**(구 9,240 대비 −4,360).
- species-cold split 생성(disc 156/repl 200 test종, all cold).

### 확대 스모크 = **전 항목 통과**
- warm 전 variant/통제 × 2 backbone 50셀 OK · species-cold 경로 OK · LightGBM/naive/Tier4 SVD OK(RMSE 1.28–1.35) · 스키마 OK · **resumability**(rerun skip=1) · **패킹**(2 job 동시 OK) · **디스크** 1.5TB free(~6GB 필요) · tier1′ shuffled 로직 OK.

### 블록 A 학습 착수(방법 B, background)
- job A 5060Ti s0-3(1,408 run) · job B 5060Ti s4-6(1,056) · job C 4090 s7-9(1,056). tier-outer 순서. manifest = `runs/gnn/BLOCK_A_LAUNCH_MANIFEST.md`.
- **첫 관찰(허용)**: discovery tier0 `dmpnn_no_species` RMSE **1.2152**(91s). 상한 1.7195 미달·하한 0.8598 초과 = 정상, tox-learn baseline 1.228과 동일 order. GPU 5060Ti 44%/4090 68% 가동.
- 후속: no_species 완료 후 tier1′ launch · CPU(LightGBM/naive/Tier4/rank) 병행 · 블록 B는 D-OOV 후.

### 진행 (Session 21)
블록 A GNN 3잡 running. tier 완료마다 보고. δ 동결 전 비교·TOST·게이트 금지.

---

## 2026-08-01 — Session 22: D-OOV 블록B 구현·검증 + ★NCBI 조인 버그 발견·수정

director(직전): D-OOV = 평가시점 매핑(mean primary / untrained·collapse 민감도), 재학습 없음, 블록 A 유효 유지, 블록 B 구현 후 진행. 자율 유지.

### 블록 B(종-cold) OOV 러너 구현·스모크 통과 — `scripts/run_q2_blockb_oov.py`
- 1회 학습 → 3방식 예측. `apply_oov()`: t4=종임베딩 weight swap(cold종→train평균/0), t3a/t3b=랭크별 OOV 카테고리 swap, t2=one-hot FFN 열 swap. no_species=1모드(untrained).
- 스모크(4090, epochs 2) **3모드 구분 성공**: t2 mean/untrained/collapse=1.3899/1.3932/1.3912 · t3a=1.3625/1.3726/1.3623 · t4=1.402/1.4336/1.402(untrained 노이즈가 가시적으로 나쁨=D-OOV 근거 확인). weight-swap 작동 입증. 스모크 산출 정리.
- 범위 {t0,t2,t3a,t3b,t4}. 전 seed/backbone/variant/control/파티션 확장은 GPU 여유 시(블록 A 뒤/자원 나는 대로).

### CPU tier(블록 A) 병행 착수 — `scripts/run_q2_cpu_tiers_blockA.py`
- LightGBM(9 baseline)·naive × 8 warm split × seed0(결정론). background, exit 0.

### ★ NCBI 조인 버그 (버그탐지 관찰 → 수정) — 실험-critical
- **증상**: LightGBM `taxonomy_ncbi` RMSE == `no_species` **정확히 동일**(disc_group 1.3490=1.3490, disc_scaffold 1.2526=1.2526). tier 3b가 조용히 tier 0으로 축퇴.
- **원인**: 데이터 `species`는 **밑줄**(`neomysis_integer`), NCBI 파일은 **공백**(`micractinium pusillum`). 조인 키 불일치 → 교집합 **0** → `ncbi_*` 6열 전부 null(전 20 CSV). 근원: `build_adore_datasets.py` `attach_idx` merge on 원본 species. NCBI 소스파일 자체는 정상(1,267종, 1,248 resolved).
- **정규화 검증**: `_`→공백·lower·strip → 교집합 **718/718**(완전).
- **수정**: (a) `build_adore_datasets.py` = 정규화 조인 키(`_species_key`)로 merge(향후 재빌드 정합). (b) `scripts/patch_ncbi_columns.py` = 기존 20 CSV의 `ncbi_*`만 in-place 채움. **non-NCBI 열 sha256 round-trip 동일 검증**(모든 파일 hash 보존) → 완료/진행중 블록 A(ncbi_* 미참조) 무영향. 백업 `data/_backup_ncbifix/`.
- **차단 시점**: GNN `taxonomy_ncbi`(t3b) run **0개**(블록 A는 t3a 진행중) → GNN 재작업 0. block B 종-cold도 패치됨.
- **재실행**: LightGBM ncbi 2 baseline × 8 split + naive_tax 8 split(stale json은 `_backup_ncbifix/stale_jsons/` 백업 후 덮어씀). **수정 후 ncbi≠no_species 확인**(disc_group ncbi 1.2651 / shuffled 1.3544; repl_scaffold 1.4406/1.5890 등, 전 split 구분·informative<shuffled). tier 3b 실재화.
- ncbi_class nonnull ~99.7%(소수 null = NCBI 미해결 종, provenance 1248/1267과 정합).

### 블록 A 상태(관찰, 비교 아님)
- GNN run 1,248개, **FAIL 0**. 진행 tier = t3a(taxonomy_original). 전 RMSE sane. 정지 임계 미발동.

### 진행 (Session 22)
데이터 정합성 확보. 블록 A GNN 계속(3잡). CPU tier 완료. 블록 B 러너 대기(GPU 여유 시 확장). δ 동결 전 비교·TOST·게이트 금지.

---

## 2026-08-01 — Session 23: NCBI 버그 후속 3건 (조인 전수감사 + tier 축퇴 가드 + 원자적 교체)

director 지시(자율 유지): (1) 종-이름 키 조인 전수 감사, (2) tier 입력 비축퇴 사전점검 상시 규칙, (3) 실행 중 데이터 수정 원자적 교체 규칙.

### 1. 종/CAS/SMILES 키 조인 전수 감사 — `scripts/audit_species_key_joins.py` (+ `_ext/species_key_join_audit.json`)
ADORE 경로 전 조인의 키 교집합 실측(추정 아님). **NCBI가 유일한 결함, 나머지 전부 100% 정합**:
- ① chem merge (mortality.test_cas ↔ chem.test_cas): 교집합 **1.0**, smiles 보유 1.0 (3,295 CAS).
- ② NCBI merge (수정 후, 종명 정규화 키): **1.0 × 전 10 split**.
- ③ cold-split sp2grp (tax_gs→tax_group → pool.species): group 매핑 **1.0**(disc 779/repl 1006), null-group 낙오 0. 동일 tax_gs 정규화(밑줄 보존) → 구조적 일치. **가장 유력했던 2차 버그 후보 = 무결 확인**.
- ④ species_idx 정합: 전 split nonnull 1.0 · **0..n 연속** · idx↔name 양방향 1:1. 저장-idx vs 런타임 concat 재도출 정렬 문제 없음.
- ⑤ block B cold 분할(test−train, species_idx): disc 623 train/156 cold, repl 806/200, all-test-cold 1.0.
- ⑥ native tax_* nonnull 1.0(내부 groupby-first, 외부조인 아님).
- 병행 검증: 독립 5-lens 스윕 워크플로(멀티모달)로 완전성 교차확인(누락 조인 탐색 + completeness critic). lens-2가 sp2grp 동일결론 독립 도출.

### 2. tier 입력 비축퇴 가드(상시 규칙) — `jcim_v3/tier_input_guard.py`
각 (variant, split) tier 시작 시 종표현 입력 축퇴를 단언·로그, 축퇴 시 **그 tier 미착수·즉시 HALT(sys.exit 2)**.
- 축퇴 기준: 표현 열 non-null < 0.5 / cardinality < 2 / 참조 대비 자릿수(10×) 이탈. 참조 `data/tier_input_reference.json`(split별 n_species·랭크 cardinality).
- 점검: t2/t4=species_idx cardinality·**0..n 연속성**·참조대조; t3a/t3b=랭크별 non-null·cardinality·참조; t1/t1'=오프셋 종수≥2·**분산≥2**; t0=무점검.
- **가드 검증(핵심)**: 수정 데이터 = 전 tier clean(오탐 0, cold 포함); **pre-fix 백업 = t3b DEGEN 정확 검출**(ncbi 4랭크 non-null 0.0/card 0), t3a는 clean. → RMSE 우연이 아니라 tier 3b 착수 시점 **구조적** 검출됨을 입증.
- 배선: `run_q2_gnn_ladder.py`(변형별 1회, HALT), `run_q2_cpu_tiers_blockA.py`(lgbm 변형매핑), `run_q2_gnn_oof_tier1prime.py`(오프셋 축퇴 HALT), `run_q2_blockb_oov.py`(**OOV 매핑 이후** 입력점검: cold종 식별+swap 발화). block A는 이미 수정데이터라 미영향 — 가드는 향후/재개 launch에 상시 활성. 배선 발화 확인(스킵 셀에도 가드 로그 기록).

### 3. 실행 중 데이터 파일 수정 = 원자적 교체 규칙 — `jcim_v3/io_atomic.py`
- **이번 패치 방식 확인**: `patch_ncbi_columns.py`가 `to_csv(f)`로 원본 경로 **제자리 덮어쓰기 = 비원자적**(백업은 했으나 라이브 쓰기 자체는 truncate-then-stream, 동시 읽기가 잘린 파일 노출 가능). FAIL 0이었으나 위험 실재.
- **규칙 확립**: `atomic_write_csv/bytes/text` = 동일 디렉터리 temp 작성 → fsync → `os.replace`(동일 볼륨 원자적). 부분쓰기 미노출. 라운드트립 검증 OK.
- **적용**: `patch_ncbi_columns.py`·`build_adore_cold_splits.py`(공유 CSV writer) 리트로핏. 앞으로 실행 중 공유 데이터파일 수정은 원자적 교체 의무.

### 진행 (Session 23)
블록 A GNN 계속(FAIL 0, t3a). 데이터/가드/조인 무결성 확보. δ 동결 전 비교·TOST·게이트 금지 유지. 완전성 워크플로 critic 결과는 회신 시 반영.

### 완전성 워크플로 결과 (독립 5-lens 스윕 + critic) — HIGH-risk 결함 0
- **85 site 열거**(risk none 72 / low 13 / high 0). 4대 교차소스 조인 전부 audit 1.00. 저장-idx vs 재도출-idx = 구조적 안전(pool 수준 1회 배정, 전 소비자 verbatim, 런타임 재열거 없음).
- critic이 무측정 gap 8건 지적 → **6건 실측 종결**(`scripts/audit_species_key_gaps.py` + `_ext/species_key_gap_audit.json`), 전부 통과:
  - G1 ncbi_* **저장** nonnull(20 CSV 직접): min 0.9865, 전부 ≥0.5. patch가 실제 기록한 값 존재 확인(audit#2 키교집합의 보완).
  - G2 **예측 정렬**(GNN↔LightGBM no_species pred, KEY=[smiles,species,endpoint,duration]): inner **1.0**(2029/2029·4351/4351), duration int64 양측·render 일치 → 헤드라인 paired bootstrap 무손실 정렬. δ 단계 전 선제 확인.
  - G3 tier1' stratum 커버(test⊆train endpoint@duration): **1.0**(disc 1·repl 7). 미관측 stratum 0.
  - G4 duration dtype/render: int64 양측·'96'/'24' 일치 → int/float render 위험 없음.
  - G5 tier1' 종공간(no_species pred species_idx_original ⊆ data space): **1.0**.
  - G6 live species **100% 밑줄**(gammarus_fasciatus) → build_adore canonical, legacy build_q2(공백-소문자 'Latin name') 비호환.
- **거버넌스 위험 1건 가드**: `run_q2_pipeline.py:53`이 build 단계에서 폐기된 `build_q2_datasets.py`(다른 종 규약·다른 species_idx 열거) 호출 → q2_v4/data(ADORE 소유) 재실행 시 species_idx 재배정·ncbi_* 및 frozen pred 전면 탈정렬 clobber 위험. **가드 추가**(ADORE 마커 감지 시 build 거부·exit 3), 발화 확인.
- SCOPE 제외 명시: v3/Yuan-lineage 스크립트(results/jcim_v3, 별개 데이터 계보) · CC-MPNN 패키지 내부(읽기전용 경계). ADORE 결론에 미유입.
- **결론**: NCBI 조인이 유일 결함(수정 완료). ADORE 파이프라인에 동종(키 불일치) 잠재결함 없음 — 실측 종결.

### Tier 4 SVD (CPU block A) 완료
8 warm split × seed0, RMSE 0.95–1.46, k=16, factor 종수 713–952. FAIL 0.

---

## 2026-08-01 — Session 24: Tier 4 SVD 라벨 순열 누출 검정 (QC, 사전등록 아님 · §0-1 부호검증과 동급)

director 지시: tier 4 SVD는 유일한 y-유도 feature 경로 + 최저값 0.95 → double-OOF 경계를 실측 검증. 신규 구현 코드. 누출 판정 시 즉시 중단.

### 판정 기준 (실행 전 기록 — director 요구)
- **검정**: train `target_log10`을 train 내부에서만 순열(seed=**20260801** 고정). 순열된 y로 **OOF-SVD 인자 전 과정 재수행**(OOF base on permuted y → purged residual → species×compound 행렬 → double-OOF SVD). 최종 LightGBM은 **실제**(stratum-removed) y를 예측하되 입력은 RDKit feature + **순열-유도 인자**. **test 불변**.
- **대조(동일 파이프라인)**: `rmse_nofactor` = svd 열 없이 동일 최종 부스터. `rmse_real` = 실제 인자(로그된 tier4와 대조하는 harness 충실도 확인).
- **판정(split별, τ=0.02 log 단위)**: 순열 인자가 무-인자를 **0.02 초과로 개선하지 못하면 PASS**(OOF 경계 온전) → `rmse_shuffled ≥ rmse_nofactor − 0.02`. **`rmse_shuffled < rmse_nofactor − 0.02`면 LEAK** → 즉시 중단·보고.
- 근거: seed0 결정론 단일 실행. 실제 누출은 실제 인자 이득의 상당분(수십분의 1 log)을 유지 → 0.02는 실신호보다 훨씬 낮고 부스터 수치잡음보다 높음. 순열 인자의 청정 기대는 `rmse_shuffled ≥ rmse_nofactor`(잡음열은 개선 없음).
- 8 warm split × seed0, CPU. 전 split PASS여야 통과. 통과 시 Methods 1줄("target-encoding 계열 표준 누출 검증 수행").

### 결과 — 라벨 순열 누출 검정 = **PASS** (사전등록 기준, 8/8 split, seed 20260801)
`scripts/verify_tier4_permutation_leak.py` + `_ext/tier4_permutation_leak_test.json`. harness 충실도: 재구현 `rmse_real`이 로그된 tier4와 정확히 일치(1.3231·0.9512·1.459…).

| split | real | no-factor | shuffled | impr_real | **impr_shuffled** | leak |
|---|---|---|---|---|---|---|
| discovery_group | 1.3231 | 1.349 | 1.3298 | 0.0259 | **0.0192** | no |
| discovery_scaffold | 1.2215 | 1.2526 | 1.2423 | 0.0311 | 0.0103 | no |
| discovery_scaffold_generic | 1.2414 | 1.2663 | 1.2605 | 0.0249 | 0.0058 | no |
| discovery_designed_leaky | 0.9512 | 0.954 | 1.017 | 0.0028 | −0.063 | no |
| replication_group | 1.2867 | 1.2595 | 1.2633 | −0.0271 | −0.0037 | no |
| replication_scaffold | 1.459 | 1.5838 | 1.581 | 0.1248 | 0.0028 | no |
| replication_scaffold_generic | 1.3611 | 1.3699 | 1.3551 | 0.0088 | **0.0148** | no |
| replication_designed_leaky | 1.0835 | 1.0397 | 1.0979 | −0.0438 | −0.0581 | no |

- 전 split `impr_shuffled < 0.02` → PASS. `impr_shuffled` 0 주위 산포(mean −0.011, 음수 다수) → 실제 이중-OOF 누출이면 전 split 균일 큰 양수여야 함. 아님.
- **경계 확인(보강)**: discovery_group 0.0192(τ 근접). 다중 seed(5) robustness(`verify_tier4_leak_multiseed.py`): discovery_group impr_shuffled=[0.0192,−0.003,0.0111,0.0326,0.0222] mean 0.0164(<τ), 0 straddle; replication_scaffold_generic mean 0.0093. **systematic_leak=False**.
- **기전 해석(누출 아님)**: 순열 인자도 종별 고정 16-vec = **종 식별자** 역할 → 최종 부스터가 **실제** train y로 종별 평균을 학습(종 tier 본연의 신호, train→test 일반화). 이중-OOF는 자기-행 라벨 peeking 차단(확인). 잔여 impr_shuffled = "종 정체성" 효과(합법)이지 라벨 누출 아님. impr_shuffled가 음수로도 감(−0.063 등) = 무작위 열이 해치기도 함 = 누출 반증.
- **0.95 정체**: `discovery_designed_leaky`(H3 (smiles,species)-pair-random, 구조상 최易 split)의 자연 난이도. 해당 split 순열검정 impr_shuffled −0.063(인자가 오히려 해침) → 누출 아님 명확.
- 위상: QC(§0-1 부호검증 동급, 사전등록 아님). Methods 1줄 기록 예정("target-encoding 계열 표준 누출 검증 수행, 순열 라벨 인자가 무-인자 대비 개선 없음").

### Tier 4 SVD split별 값 (director 요청 2)
| split | RMSE | n_test | strata | n_species(factor) | test_unseen |
|---|---|---|---|---|---|
| discovery_group | 1.3231 | 2029 | 1 | 718 | 69 |
| discovery_scaffold | 1.2215 | 2029 | 1 | 744 | 46 |
| discovery_scaffold_generic | 1.2414 | 2029 | 1 | 743 | 40 |
| discovery_designed_leaky | **0.9512** | 2029 | 1 | 713 | 74 |
| replication_group | 1.2867 | 4351 | 7 | 952 | 85 |
| replication_scaffold | 1.4590 | 4351 | 7 | 935 | 133 |
| replication_scaffold_generic | 1.3611 | 4351 | 7 | 937 | 133 |
| replication_designed_leaky | 1.0835 | 4310 | 7 | 928 | 127 |
(discovery=1 stratum[LC50@96h], replication=7 strata. 최저 0.95=designed_leaky 최易 split.)

### 진행 (Session 24)
Tier 4 SVD 누출 검정 PASS(중단 없음). 블록 A GNN 계속. δ 동결 전 tier 비교·TOST·게이트 금지 유지.

### Session 24 (계속) — director 회신 반영: 기준 불변 + 5-seed 정직 기록 + 인덱스 증명

**1. 사전등록 기준 불변 (director (a)·(b) 배제)**: τ=0.02 단일 seed 사전등록 기준 유지. 결과 확인 후 기준 변경 금지(더 엄격해도 사전등록 효력 상실). (b) shrinkage = tier4 정의 사후변경+교락 재유발이라 배제. **(c) Methods 명시로 진행.**

**2. 5-seed 산포 명시 기록 (선택적 보고 방지, director 요구)**: discovery_group 순열 robustness `improve_shuffled` = [0.0192, −0.003, 0.0111, 0.0326, 0.0222] → **2/5 seed가 τ(0.02) 초과**(0.0326, 0.0222). 사전등록 기준(단일 seed 20260801=0.0192<τ)으로는 **PASS, 판정 유지**. 그러나 다중 seed에서 τ 초과 2건이 관찰됨을 로그·Methods 양쪽에 기록. replication_scaffold_generic = [0.0148,0.0096,0.0194,0.0017,0.0011] mean 0.0093, τ 초과 0. 두 split 모두 mean<τ, systematic_leak=False.

**3. ★ 인덱스 수준 OOF 경계 증명 = PASS (8/8 warm split)** — `scripts/verify_tier4_oof_index.py` + `_ext/tier4_oof_index_proof.json`.
- 방식: `build_factor`에 fold 장부 후크(_ledger_inner/_outer, 가산적·기본 None·frozen 출력 무영향) 추가 → **실제 코드 경로**의 fold 배정을 추출해 집합 연산 검증.
- 검증 결과(전 split): inner_self_in_source=**0** · outer_self_in_source=**0** · outer_self_in_species_source=**0** · each_row_once(inner·outer)=true(유효 분할) · train_test_key_overlap=**0**(test 측정치가 인자 소스에 없음) · full_map_species ⊆ train · 미관측 test 종 인자 all-zero.
- **의미**: 순열 검정("누출이 있어도 이 이하")과 달리 이 증명은 **"자기-행 누출 없음"을 구조적 사실로 확정**(감도 바닥 없음). 각 train 행의 SVD 인자는 그 행을 제외한 데이터에서 계산됨을 inner·outer 양 OOF 층에서, test 행 미사용을 측정키 disjoint로 입증. 위반 0.
- 위상: QC(사전등록 아님). Methods 1줄: "OOF 경계를 인덱스 수준에서 검증(자기-행 제외, 8 split 위반 0)".

**종결**: Tier 4 SVD 누출 = 통계(PASS, 정직 산포 기록) + 구조 증명(위반 0) 이중 확인. 중단 없음.

---

## 2026-08-03 — Session 25: block A job B/C 완료 + tier1′ 완료 + 블록 B 착수 + rank 절단 구현

- **block A job C(4090 seeds 7-9) + job B(5060Ti seeds 4-6) 완료**: 각 1056 run, FAIL 0(~37.7h/~63h). **seeds 4-9 전부 완료**(352/seed). job A(seeds 0-3)만 ~75% 잔여, 이제 unpacked로 가속. 누적 3,169 run, FAIL 0.
- **tier 1′ OOF 완료(4090)**: done 320, FAIL 0, ~25h. true 160+shuffled 160. 오프셋 축퇴 가드 무발동. control 검증: shuffled 오프셋 정상 저하.
- **블록 B(종-cold OOV) 착수(4090)**: 9 variant(t0+true/shuffled×{t2,t3a,t3b,t4})×2 cold 파티션×2 bb×10 seed=360 cell×3 OOV모드. shuffled 경로 스모크 통과. 첫 cell 정상, post-OOV 가드 clean. (wrapper `&` 오사용으로 자동알림 유실→완료 감시 루프 arming.)
- **rank 절단 변형 구현·스모크 통과**: native taxonomy 깊이 포화 연구. `taxonomy_genus`(tax_genus), `taxonomy_genusfamily`(tax_family+tax_genus) 신규 injection. 중앙 상수 `TAXONOMY_INJECTIONS`(models.py)로 6개 디스패치 지점 통합 — **기존 taxonomy_original/ncbi 동작 불변**(집합 멤버십 유지, block A resume 정합). TAX_RANKS 2개 추가. 가드는 injection→TAX_RANKS 직접 참조로 일반화(t3a_g/t3a_gf). 2-epoch 스모크 4셀(2변형×2bb) 통과(FAIL 0, RMSE 1.45–1.60), 가드 정상 발동(tax_genus card 424, tax_family 182, card_vs_ref 일치). e2 아티팩트 정리. → **5060Ti(job A) 여유 시 launch 예정**.
- 진행: block A(5060Ti job A)+블록 B(4090) 병행. δ 동결 전 tier 비교·TOST·게이트 금지 유지.

---

## 2026-08-03 — Session 26: 리팩터 수치 동일성 3건 (director 확인 지시)

### 1. ★ TAXONOMY_INJECTIONS 리팩터 = **행동 보존적 (PASS)** — `scripts/verify_refactor_byte_identity.py`
- 방식: 완료 셀(seed 7=원래 4090, 동일 카드)을 새 코드로 재실행→비교. 대상 t3a·t3b·통제(shuffled) × 2 backbone = 6셀.
- **예측 바이트 동일**: 6/6 `pred_log10·true_log10·error_log10` 정확 일치(max_abs=0.0). **코드 생성 전 열 바이트 동일**(ncbi 데이터-수정 열 제외).
- 전체 파일 SHA는 t3a 4셀에서 상이했으나 **원인은 리팩터가 아니라 NCBI 데이터 수정**: t3a seed7은 8/1 NCBI 조인 수정 **전** 실행분이라 CSV의 `ncbi_*` 6열이 null; 재실행은 수정된(채워진) 데이터를 읽음 → 수동 반출 메타데이터만 상이. t3b는 전부 수정 **후** 실행(수정 시점 t3b run 0개)이라 파일 완전 동일. `effect_value`는 17번째 자리 float-repr 차이(동일 값, 패치 재직렬화). **어느 것도 예측·δ에 무영향**.
- 함의(δ 안전): 리팩터가 예측을 비트 보존→ job A가 t3a/t3b를 새 코드로 재개해도 seed 간 코드 차이 없음. t3a는 tax_*(수정 무관, 바이트 보존)만 사용, 전 t3b는 수정 후 실행(일관). **δ 오염 없음**.
- 부수 관찰(무해, 조치 불요): NCBI 수정 **전** 실행된 예측 CSV(t0~초기 t3a)는 `ncbi_*` 메타데이터가 stale(null). 분석 조인키[smiles,species,endpoint,duration]+pred/true만 사용하고 t3b 모델은 런타임에 수정된 데이터에서 tax code 재구성하므로 무영향. provenance 스냅샷 차이일 뿐.

### 2. rank 절단 실행 범위 — director 사양으로 축소 확정
tier 3a만 · genus/genus+family 2개 · warm·discovery·group만 · **main 통제만**(shuffled/zero/dummy/fixed_proj 없음) · GraphConv·D-MPNN·LightGBM · GNN seed 10/결정론 1. **≈ 40 GNN + 2 LightGBM = 42**. (이전 640-run 계획 폐기.) exploratory 패밀리. → 5060Ti(job A) 여유 시 GNN launch, LightGBM 2는 CPU.

### 3. LightGBM 신규 variant — **baseline 엔트리 필요**(자동 아님)
`run_rdkit_lgbm`이 `LIGHTGBM_BASELINES`에 없는 baseline 거부 → `taxonomy_genus`/`taxonomy_genusfamily` 엔트리 추가(main만). `_features`가 `TAX_RANKS[species_repr]`로 rank code 구성하므로 이후 자동. 스모크(disc_group s0): genus rmse 1.2760 / genus+family 1.2730(vs no_species 1.3490, 신호 존재), tier 가드 발동(t3a_g/t3a_gf, 비축퇴). LightGBM 결정론 → seed 1.

### 진행 (Session 26)
리팩터 무결성 확인 완료(중단 없음). rank 절단은 §2 범위·task1 PASS로 게이트 해제, 5060Ti 여유 시 착수. 블록 A(job A)+블록 B 계속. δ 동결 전 tier 비교·TOST·게이트 금지 유지.

---

## 2026-08-03 — Session 26 (계속): 집계 SSOT 가드 + TOST 판정규칙 사전기재 + ★Δ/δ 대상 불일치

### 집계 메타데이터 단일출처(SSOT) 가드 (director §2) — 위반 0, 가드 확정
- **감사(직접 + 독립 워크플로 5-lens)**: ADORE 집계·층화·평가 스크립트 전부 예측 CSV에서 **화이트리스트 열만**(조인키+pred+true+compound_key) 읽고, 전 층화(abundance·cold·coverage·stratum·support)를 **데이터셋에서 유도**. 워크플로 verdict **NO_VIOLATIONS**. 유일한 forbidden read = 내가 만든 의도적 음성대조 회귀테스트(`regression_pred_metadata_ssot.py`). full-read-latent = tier1′ base-pred(러너, 화이트리스트 사용) + 내 QC 스크립트들.
- **가드 확정**: `jcim_v3/prediction_io.py` — `load_prediction_csv`가 로드 시점에 화이트리스트 강제(기본 메타 드롭·forbidden 요청 시 `PredictionColumnViolation`). 5개 집계 진입점(bootstrap_q2_ladder δ경로·quantify_cost·abundance·cold_signtest·endpoint) 배선. 워크플로가 배선 정확성 재확인.
- **회귀 PASS**: pre-fix null-ncbi CSV → 예측-CSV 방식 396/396 종 '미해상' 오분류, 데이터셋 방식 2/396(진짜 미해상만), 가드가 wrong 패턴 차단.

### TOST 판정규칙 사전기재 (director 작업 1·3)
- **작업 1(기재 여부)**: (1) α·CI 수준 = **없음** · (2) Δ 정의 = **없음(코드에만 암묵)** · (3) TOST∧NHST 조합 배정 = **없음**. (§4G-1~6·§4δ·§5·fdr_family_boundary 전수 확인; §4G-1 "게이트=동등만"만 있고 α/조합 미명시.)
- **작업 3(조건부 기재)**: §4G-7 신설 — **(1) α=0.05 단측×2=Δ의 90% CI, 동등=90% CI⊂[−δ,δ]** + **(3) TOST기각∧NHST기각→동등** 추가(director 확정 문구 그대로). **(2)는 미추가**(director 미확정 + D-ΔδMATCH escalate). 증거 기록(δ 미동결·게이트 산출물 부재·자유도 0화) 병기.

### ★ 작업 2 — Δ/δ 대상 불일치 발견 → escalate + δ 스크립트 보류
- **사실**: Δ = **(b) 앙상블 RMSE**(bootstrap_q2_ladder.py:109-114 `point()`가 seed 예측 평균 후 RMSE 1개; 시드 재추출:122도 앙상블에 투입). δ = **(a) per-seed RMSE SD**(§4δ:72). Δ 정의 = 코드에만. **두 양 대상 상이** → 앙상블(b) 안정성으로 CI 좁고 δ(a)는 커 `동등` 쉬워짐(주장 유리 방향).
- **처리**: `director_결정필요.md` D-ΔδMATCH 등재((가)둘다 per-seed/(나)둘다 앙상블/(다)기타). **δ 산출 스크립트 작성 보류**(director 지시). 증거: δ 미산출·gatekeeping/bootstrap 산출물 부재.
- 진행: 블록 A/B/rank 절단 자율 계속. δ 동결 게이트 닫힘 유지.

### D-ΔδMATCH 산정 (Session 26, 사실·숫자; 결정 대기)
- **(1) C=14** = 2 backbone × 7 warm main tier(t0,t1,t1′,t2,t3a,t3b,t4) × disc×group×main. **10-시드 완료 14/14**(δ pooling 재료 전부 존재 → (가) per-seed δ 추가 run 0).
- **(2) 앙상블(나) 추가 run**: k=2 →140(~2.3벽h) · k=3 →280(~4.7h) · k=4 →420(~7.0h). per-run 실측 disc_group 평균 142s, rig 유효 ~60s/run.
- **(3) δ 정밀도**: per-seed df=126 rel 6.30% vs 앙상블 k=2/3/4 df=14/28/42 rel 18.9/13.4/10.9%. → per-seed가 k=4에서도 더 정밀.
- **(4) 일정**: 추가 δ 시드는 5060Ti(job A 후)·블록 B(4090)와 병렬 → k=2,3 지연≈0, k=4 ≤수시간. (가)=0.
- **(5)** 앙상블 δ 동일 형태 정의 가능(k≥2 분리 앙상블 s_c=SD, pooled 동일식). 초안 병기.
- **(6)** 동일 하이퍼·패킹·결정론 가능, 시드번호만 신규. 카드 상이 시 HW변동 유입(δ 성격상 일관).
- **(7)★사실**: 12/14 조건(t0-t4×2bb) 시드 0-6=5060Ti/7-9=4090 **카드 교차** → 현행 per-seed δ 이미 HW변동 포함. 2/14(t1′) 4090 단일. Methods 기재.
- δ 스크립트 보류 유지. (가)/(나)/(다) 회신 대기.

---

## 2026-08-03 — Session 27: D-ΔδMATCH 확정 반영 (주=per-seed, 민감도 §4δ′ 신설) + δ′ 착수 + §B 사실

- **결정**: 주 척도=(가) per-seed(Δ=시드별 RMSE, δ=§4δ 불변, C=14). §4δ 미개정. **§4δ′ 신설**(민감도 앙상블 δ′, k=10, 조건당 +90시드=1,260 run). 사전등록 §4δ′ 기재(A-1~A-6·규칙·증거 병기). §4G-7 (1)/(3) 양 척도 적용.
- **δ′ 착수**: `run_q2_delta_prime.py`(별도 네임스페이스 `runs/gnn_dprime`, 카드규칙 A-5.4, 동일 하이퍼·결정론). 스모크(seed10 e2, 12셀) 통과. **5060Ti-stream(756, t0-t4 seeds 끝자리0-6) 착수**(job A와 패킹). 4090-stream(324, seeds7-9)=블록 B 종료 후. t1′(180, 4090)=no_species δ′ base 완료 후.
- **§B 시드 짝지음 사실(코드 확인, 판단 director)**:
  1. **배치 셔플=공유(동일)**. `_train`이 `iterate_batches(n, bs, shuffle, seed=cfg.seed*1000+ep)` → `np.random.default_rng(seed)` **로컬 Generator**(전역 RNG·모델 무관), `n=len(train_ds)`가 tier 불변 → 시드 s에서 두 tier **동일 배치 순서**.
  2. **초기화 RNG=갈림**. 파라미터는 전역 torch RNG(`_setup_reproducible`=`torch.manual_seed(s)`)에서 draw. GraphConv 구성순서 = atom_proj→layers(=**tier 무관, 동일 draw**)→종 모듈(present/부재/크기 상이)→ffn(readout dim tier 상이). ⟹ **message-passing 백본 init까지 동일, 종 모듈+readout FFN부터 갈림**. draw 수 상이 → 전역 RNG 위치 이동 → **dropout mask도 tier 간 미대응**(학습 중 전역 RNG 사용).
  3. **귀결**: 시드 s = **배치순서 공유(paired) + 백본 init 대응 / 종·readout init·dropout 미대응**. → 셔플링 공유·초기화 부분대응.
- δ 스크립트 보류 유지(§B 회신 후 Δ 정의 기재+스크립트). δ 동결 게이트 닫힘.

### §B=paired 확정 → Δ 정의 기재 + 주 δ 동결 (Session 27 계속)
- **paired 확정**: Δ = mean_s([RMSE_cand(s)−RMSE_candbase(s)]−[RMSE_ref(s)−RMSE_refbase(s)]), 양 arm 동일 시드·공통 sset. §4Δ 사전등록 기재(Methods 과장금지 문구 포함). δ·δ′ 불변.
- **★ 주 δ 동결**: `scripts/compute_freeze_delta.py` → `audit/delta_primary_frozen.json`. **δ = 0.019777**(C=14, df_total=126). per-condition s_c 0.0125~0.0255(단일-arm 재실행 SD, 관찰 허용). SSOT 로더로 pred/true만 읽음. 동결 전 증거(gatekeeping 스크립트·bootstrap 산출물 부재) 파일 기록. **tier 비교·Δ·dd 미계산 → 비교 게이트 여전히 닫힘.** 재실행 시 불변 검증(idempotent).
- **§2 t1′ δ′ 선행조건 = no_species δ′ base 단독**(tier1prime이 OOF base를 자체 계산, no_species 예측만 읽음; 타 tier 불요). 재배치: t1′는 no_species δ′(90시드) 완료 후 4090에서 **잔여 main과 병렬**(꼬리 아님). 블록 B 해제 시 no_species(27)→t1′(180)+잔여 main(270) 패킹. 재배치 후 δ′ 전체 ≈ **~1일**(5060Ti 756-stream + 4090 시퀀스 바운드, t1′ 흡수).

## 2026-08-03 — Session 28: 명세 완비성 종결 (자유도 문구·개정이력·비교집합 열거) + δ 증거버그 정정
- **§0 정지규칙 신규**: 명세 완비성 항목 = 판정 결정 시 게이트 전 닫기 / 아니면 분석시점 결정·명시. 서술·배치는 에스컬레이션 금지. (데이터·코드 정합성 결함은 종전대로 즉시 보고.)
- **§1 "분석 자유도=0" 폐기**: PREREGISTRATION.md 3곳(§4δ canonical + §4G-7·§4δ′ 포인터) 교체 → "판정 기준은 결과 전 고정; 서술·배치는 분석시점·명시". 절대주장→검증가능 문장.
- **§2 개정이력 표 신설**(§R): 6항목(§4G-7·§4δ′·§4Δ·§4δ-break·§4δ-impl·자유도문구) × 타임스탬프·결과미확인증거·사유. δ 동결 前/後 상태 구분.
- **★ δ 증거버그 정정(무결성)**: `compute_freeze_delta.py`가 `bootstrap_outputs_exist`를 **디렉터리 존재**로 기록 → `runs/bootstrap`이 빈 placeholder(Jul-29, 0 files)라 True로 오기록. **파일 수 0**으로 수정. **δ 값 불변**(0.019776561636, v1=v2). 구본 `delta_primary_frozen_v1_evidencebug_20260803.json` 보존. 정정 시점도 비교 출력 0. pre-reg 증거문구 "bootstrap 부재"→"비어있음(0)" 2곳 정정.
- **§3 비교집합 열거**(별도 보고, pre-reg 미기재): 원문 인용 + 기계전개 표 + 모호점 + **F1-F4 legacy 확인**(cross-backbone DD·taxonomy 부재 = ADORE primary(within-backbone·3a vs 2) 아님). director 판정 대기.

## 2026-08-03 — Session 29: ★블록 A warm GNN 완료 + 블록 B 완료 + δ′ 4090/t1′ 재배치 + rank 절단 착수
- **★ 블록 A warm GNN 완료**: job A(seeds 0-3, 1408 run) 종료 → **총 3,520 run, FAIL 0**, 전 10 seed × 352 완료. (3 job ~80h.)
- **블록 B 완료**: 360/360, FAIL 0.
- **δ′ 진행**: 5060Ti-stream + 4090 main-stream(블록 B 종료 후 착수, no_species 우선) 병행. t1′ 재배치 watcher arming(no_species δ′ 180 pred 완료 시 4090에서 main t1-t4와 병렬 launch).
- **rank 절단**: LightGBM 2/2 완료(genus 1.2760/genus+family 1.2730). GNN 40 착수(5060Ti, job A 종료 슬롯, δ′와 패킹, 별개 변형이라 race 없음).
- 잔여 warm: δ′(1,260, ~진행) + rank절단 GNN(40). 주 δ 동결 완료(0.019777). 비교 게이트 닫힘.

## 2026-08-04 — Session 30: ★ 전체 warm 학습 완료 (δ′ 포함) + δ′ 동결 승인 대기
- **δ′ 100% 완료**: main 1,080/1,080 + t1′ 180/180 = **1,260, FAIL 0**. t1′ 재배치로 tail 병렬 흡수.
- **δ′ 산출(dry-run, 14/14)**: **δ′ = 0.005717**(C=14, df=126). per-condition s_c 0.0039~0.0078. per-seed δ(0.019777) 대비 ~3.5× 작음(앙상블 안정, D-ΔδMATCH 예상 그대로). 단일-arm 설계값·tier 비교 아님.
- **전체 학습·데이터수집 완료**: 블록A 3,520·블록B 360·tier1′ 320·Tier4 SVD 8·LightGBM/naive·rank절단 42·δ′ 1,260. FAIL 0 전량.
- **동결 상태**: 주 δ 동결(0.019777). **δ′ = 실행·동결 director 승인 대기**(`compute_freeze_delta_prime.py --freeze`). 비교 게이트 닫힘 유지.

## 2026-08-04 — Session 31: δ′ 동결(승인) + 정지임계 전수평가 + 비교스크립트 작성착수
- **δ′ 동결(director 승인)**: `audit/delta_prime_frozen.json`, **δ′=0.005717**(C=14,df=126). 증거(bootstrap 출력 0·gatekeeping 부재)+**§4δ-break δ′에도 동일 적용** 기재. δ′/δ=3.46 ≈ √10, 설계값 정합.
- **§6 정지임계 전수평가**(`eval_stop_thresholds.py`+`audit/stop_threshold_eval.txt`): **6,286 셀(tier-0 396) 전수**, skip 0. **상한(즉시중단) 발동 0 = PASS.** 하한(누출의심, 보고만) 9건 **전부 `discovery_designed_leaky`**(설계상 leaky H3 split, taxonomy t3a/t3b, RMSE 0.8498~0.8597 = 하한 0.8598 미만 ≤0.01). **비-leaky split(group·scaffold·cold) 하한 발동 0 = 예상외 누출 없음.** 단일-arm 검사, tier 비교 아님.
- **§3 비교스크립트 = 작성만(실행 금지)** 착수: 16항목 체크리스트 + 합성 단위테스트. director 실행 승인 후에만 실행.

### §3 비교 스크립트 작성·검증 (실행 금지) — Session 31 계속
- `jcim_v3/gatekeeping.py`(코어: decide 3범주+4번째칸·paired_dd_bootstrap per-seed·bh_fdr·stage2_reached) + `scripts/run_q2_gatekeeping.py`(파이프라인, §4C 비교집합·2단게이트·3패밀리 FDR·confirmatory 해시·SSOT·frozen δ/δ′ 읽기).
- **합성 단위테스트**(`test_gatekeeping_synthetic.py`) **전체 통과**: 3범주 4케이스+4번째칸+경계·per-seed 부트스트랩(동등/유의/불확정/결정론)·2단게이트(동등만)·FDR 패밀리분리·frozen 부재 즉시실패.
- 실행 가드: 무플래그=**REFUSE**(언블라인딩), `--dry-check`=파일/셀 해상만(Δ 미계산). dry-check 60/60 해상, missing 0.
- **미완**: [C15] 결정론 tier 자체 block-bootstrap δ + det 비교집합, [C16] 앙상블 Δ′ 민감도(δ′·자체 FDR·게이트 불참) = scaffold만. 완성 후 실행 승인 요청 예정. **실행 금지 유지.**

## 2026-08-04 — Session 32: δ_det 사양·산출(동결 보류) + C15·C16 완성 + 증거 문구 정정
- **§4δ_det 사양 기재**(director 확정). δ_det = √(mean s_c²), s_c=block bootstrap 2000(block=compound≡smiles) RMSE SD, LightGBM main tier, naive 제외.
- **δ_det 산출(dry-run)**: **δ_det=0.087189**, C=**6**(t0·t1′·t2·t3a·t3b·t4; s_c 0.085~0.089). δ_det≫δ(0.020)≫δ′(0.006) = 표본변동 vs 재실행잡음 vs 앙상블, §4G-6 예측대로. **★ C=6≠7**(LightGBM에 t1 부재=additive bias는 naive 전용, 제외) → **동결 보류, director 재확인 대기**(조건부 승인 "7/7" 미충족).
- **C15 완성**: det 비교(LGB t3a/t3b/t4 vs t2, within-LGBM)=δ_det 판정(GNN δ 아님)·block-only(1 seed)·GNN δ 차용 민감도 병기. `load_frozen(delta_det)` 부재시 실패.
- **C16 완성**: `ensemble_dd_bootstrap`(Δ′ 앙상블·canonical 10-seed block+내부 재추출)·δ′ 판정·**자체 BH-FDR·게이트 불참**.
- **합성 테스트 20/20 PASS**(신규 5: det가 δ_det 읽음·det 시드 미재추출·앙상블 δ′·민감도 primary FDR 제외·민감도 게이트 제외).
- **증거 문구 정정**(director): `run_q2_gatekeeping.py` 존재 → "스크립트 부재" 불가. 증거 = **비교 산출물 0** + 하드가드 실행차단. compute_freeze_delta_det는 output-count 사용.
- 실행 금지 유지(REFUSE 가드). δ_det 동결 확인 → 그 후 실행 승인 요청.

## 2026-08-05 — Session 33: δ_det 동결 전 3건 + 동결 (C=6 확정)
- **§1 블록키 정정**: §4δ_det "compound_key"→**smiles**(의존은 CAS 아니라 분자입력 SMILES로 흐름; compound_key는 상관행 쪼개 CI 부당 축소). 동결 전 문서-따라가기 정정(통계 변경 아님). **전 파이프라인(δ·δ′·δ_det·Δ) smiles 통일 확인**(run_q2_gatekeeping align+det_align, compute_freeze_delta_det). bootstrap_q2_ladder=legacy(F1-F4·미사용).
- **§3 축퇴 확인 boolean**: LightGBM t3a/t3b/t4 예측 **전부 구별됨**(max abs diff 1.14/2.34/2.27>0). s_c 동일(0.0871)은 test셋 지배 부트스트랩 SD이지 NCBI식 축퇴 아님. → 동결 진행.
- **§2 구조적 관대함 합성측정**(`measure_structural_leniency.py`, 실제 439화합물 구조+합성): 검출임계 결정론 1.0×δ_det·GNN 1.0×δ(둘다 ~1×margin, paired CI≪margin). 관대함=절대마진 δ_det(0.087)≈4.4×δ(0.0198) → 결정론 동등 = 약한 증거. §4δ_det에 기록.
- **§4 δ_det 동결**: `audit/delta_det_frozen.json`, **δ_det=0.087189, C=6**. 증거=비교산출물 0(스크립트 존재 아님). §4δ-break 적용.
- **개정이력**: "7"은 director 기대치였고 사양 아님. LightGBM에 t1(additive bias) 부재는 backbone×tier 가용성 행렬의 사실 → §4δ_det "전 tier" 충실 적용 시 C=6.
- 세 마진 동결 완료: δ=0.019777 · δ′=0.005717 · δ_det=0.087189. 비교 게이트 닫힘. 실행 승인 미요청(Kevin 결정).

## 2026-08-04~05 — Session 34: exploratory 사양 확정(5/6) + cross-backbone 모호 → 실행 보류
- director (가) 진행: exploratory 완비 후 단일 실행. §4C-Explore 기재(6항).
- **확정 5항**: rank절단(genus/genusfamily × {vs t3a full, vs t2}, GC·DM·LGBM) · support-bin(1-5/6-20/21-100/100+ 행필터) · tax_group(fish/crusta/algae 행필터, mortality species→tax_group SSOT; 실측 3군 전종 커버) · scaffold(primary set × murcko/generic) · designed-leaky(primary set × leaky split).
- **cross-backbone 모호**: 구조 확정(DD=(cand@A−t0@A)−(t2@B−t0@B), A≠B)이나 **조합 열거 미확정**("D1 worked example" 미소재; D1=D-MPNN tier5/6 무관; legacy F2=GNN cand t4 vs LGB t2). cand backbone 범위·(A,B) 쌍 director 확인 필요. → **구현·실행 보류.**
- 실행 안 함. 게이트 닫힘. cross-backbone 확정 후 6항 일괄 구현+provenance+합성테스트+단일 실행.
