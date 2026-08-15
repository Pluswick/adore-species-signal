# Q2 v4 — HANDOFF: ADORE 전환 실험 (새 세션 승계용)

> 이 문서 **하나만 읽고** ADORE 전환 실험을 이어갈 수 있도록 자족적으로 작성됨.
> 작성 근거: 저장소·코드·로그·provenance·ADORE 파일 직접 감사(읽기 전용). 추정은 "추정"으로 명시.
> 확정 설계(§B)는 director(claude.ai) 결정 — 그대로 기록하되, 실현 가능성 의문점은 **⚠director 확인 필요**로 flag.

경로 표기 (셋업 완료):
- **`EXP = <REPO_ROOT>`** — 이 실험의 **자족적 작업 공간**. 코드·스크립트·config·이 HANDOFF가 여기 복사됨. **새 세션은 여기서 작업.**
- `ADORE = <ECOTOX_DATA_DIR>` — 원본 데이터(외부, 읽기 전용).
- `CC-MPNN = <USER_HOME>\Desktop\CCLABS\CC-MPNN` — read-only 인코더(외부; `EXP\jcim_v3\paths.py`가 형제 경로로 자동 해결).
- `TOXLEARN_REF = <USER_HOME>\Desktop\CCLABS\JCIM\results\q2_v4` — 기존 tox-learn 산출물·원고(비교 참조용, **읽기 전용·수정 금지**). 원고 working 사본은 `<USER_HOME>\Desktop\CCLABS\Q2\working\`.

> **아래 §D–§H에서 "JC\scripts…", "JC\jcim_v3…"로 표기된 재사용 코드는 이제 `EXP\scripts\`, `EXP\jcim_v3\`에 있음**(복사·경로 이동 완료). 실행 경로만 EXP로 읽으면 됨.

---

## §0. 작업 공간 상태 (이미 셋업됨 — 검증 완료)

- `EXP\jcim_v3\`(15 py)·`EXP\scripts\`·`EXP\configs\`(14) 복사 완료. 스크립트의 하드코딩 절대경로를 **JCIM → adore_experiment로 이동**(JC 원본 무변경), stale `__pycache__` 정리.
- **검증됨**: `import jcim_v3.runner` OK → **ccmpnn 자동 해결**(`CC_MPNN_ROOT = CCLABS\CC-MPNN`, 존재 확인). `paths.py` RESULTS_ROOT = `EXP\results\jcim_v3`. run 스크립트 DATA/OUT/PRED = `EXP\results\q2_v4\…`.
- `EXP\results\q2_v4\{data, runs\gnn\predictions, runs\replication\predictions, runs\bootstrap, audit\logs}` **빈 골격 생성됨** — ADORE 산출물 저장 위치.
- 실행 규약: `cd EXP` 후 `conda run --no-capture-output -n jcim_v3 python scripts\<name>.py` (스크립트가 `sys.path.insert`로 `EXP\jcim_v3`를 잡음).
- **아직 없는 것(정상 — §D/§H에서 새 세션이 생성)**: ADORE 로더(`build_adore_datasets.py`), `configs\q2_dataset_adore.json`, Tier 5/6 주입, 그리고 `EXP\results\q2_v4\data\*`(ADORE 재구축 전엔 비어 있음). tox-learn `vendor/`·`data/`·기존 `runs/`는 **의도적으로 미포함**(ADORE는 새로 빌드).

---

## A. 역할·규칙 (기존 HANDOFF.md 승계)

- **Claude Code** = 실행·파일시스템·원본 수치 검증. **director(claude.ai)** = 전략·판단·프롬프트. 결과를 서사에 맞추지 말 것(원본 CSV 인용).
- **`conda run --no-capture-output -n jcim_v3 python …` 필수** — 직접 `python.exe`는 Windows BLAS DLL 로딩 실패로 세그폴트. `conda run python -c`는 인자에 개행 있으면 실패 → 스크립트를 파일로 쓰고 실행.
- **CC-MPNN/ccmpnn 인코더 read-only** → D-MPNN taxonomy/신규-tier 주입은 불가할 수 있음(§B/§E에서 확인). GraphConv·LightGBM 위주.
- **무삭제·resumable**(완료 `runs/<id>.json` 또는 예측 파일 있으면 skip)·**백업 후 편집**. 장시간 잡은 harness-tracked 백그라운드(`run_in_background:true`) — nohup 체인은 셸 teardown에서 죽음.
- **단일 GPU 직렬** 전제였음 → 이제 2 GPU(RTX 5060 Ti + 4090) 병렬화가 첫 과제(§F).
- 대량 실행 전 **1-seed 스모크 필수**(이번 프로젝트에서 버그 2건 잡음: data path, species_idx). split-name glob `_group`/`_scaffold` 중복 버그 주의. 매 단계 `GAP_EXECUTION_LOG.md` append.

---

## B. 확정된 전환 설계 (director 확정 — 그대로 기록)

- **주 데이터 tox-learn → ADORE** (CC-BY 4.0, `ADORE\`). tox-learn 라이선스(data availability) 회피가 목적. ADORE는 독립 큐레이션 + SMILES·계통·형질 내장.
- **사다리 tier**: 기존 0(no-species)/1(bias)/1′(residual calib)/2(categorical one-hot)/3a(taxonomy original)/3b(taxonomy ncbi)/4(learned embedding) + **신규 5(계통거리 PCoA 16d, 고차원 민감도 병기)** / **6(DEB·형질 벡터)**.
- **미해상 종 = unknown 버킷 + 전 tier 동일 test set**(행 제외 금지). 신규 tier에도 control(shuffled/dummy/zero) + **fixed_proj(용량 매칭)** 적용. 평가 = warm + cold + **cross-group 외삽**.
- **discovery = LC50@96h / replication = 나머지**. 값 = **−log10 mol/L (pLC50)**. split = group / scaffold / designed-leaky.
- **통계·backbone 규약 기존과 동일**: strata 집계 (smiles,species,endpoint,duration), endpoint/duration residualization(train 추정 가산 주효과 제거→예측 시 복원), **block bootstrap 2000**(블록=compound_key), GNN=seed×block 이중·결정론=block-only, **global BH-FDR**, `strength()` 4단계(robust/moderate/directional/not_supported), **val carve 금지(LightGBM)·GNN은 val_frac 0.1+early stopping**, **D-MPNN taxonomy N/A(ccmpnn read-only)**.

**⚠director 확인 필요 (감사에서 드러난 실현 가능성 의문점):**
1. **kingdom/phylum 부재.** ADORE `tax_all`은 **class부터** 시작(예: "Actinopterygii Cypriniformes Cyprinidae Cyprinus carpio"). kingdom/phylum 없음 → 설계상 "kingdom/phylum from tax_all"은 **성립 안 함**. taxonomy tier(3a/3b)는 ADORE에선 **class/order/family/genus(4 rank)** 기준이 됨. 5-rank(kingdom..family)를 유지하려면 tax_group→kingdom/phylum 매핑을 별도 작성해야 함.
2. **신규 tier 5/6를 어느 backbone에.** ccmpnn read-only라 D-MPNN 신규 주입 불가 가능성 높음 → GraphConv+LightGBM만? (§E, §I).
3. **cross-group 외삽 정식 결과 범위**(fish→crusta 등) — 현재 사다리에 cross-group split이 없음(신규 필요).

---

## C. ADORE 실측 현황 (감사 결과 — 파일 경로 명시)

주 데이터: **`ADORE\processed\ecotox_mortality_processed.csv`** (70,670행) / `…_filtered.csv`(66,896행).

| 항목 | 값(실측) | tox-learn 대조 |
|---|---|---|
| 레코드 | **70,670** | 33,259(repl) |
| distinct 종(`tax_gs`) | **1,267** | 1,750 |
| distinct 화합물(`test_cas`) | 3,295 | — |
| strata (cas,종,endpoint,dur) | **32,301** | ~33k |
| endpoint | LC50 58,330 · EC50 12,340 | LC50/EC50 |
| duration(`result_obs_duration_mean`) | 4종: 96h(29,420)·48h(19,520)·24h(16,185)·72h(5,545) | 다중 |
| tax_group | fish 41,526 · crusta 23,092 · algae 6,052 | fish/crusta/algae |
| 종당 record median | **6** (mean 55.8, max 7,260) | ~6 |
| 상위 5% 집중 | **73.6%** (10%=83.9·20%=91.3) | 66.1% |
| 희소종 ≤1/≤2/≤5건 | 18.4% / 29.9% / **49.4%** | 유사 |

→ tox-learn보다 **레코드 2×·더 집중적·희소종 풍부** ⇒ 임베딩 tier·support 층화·cold 분석 전부 유의미(감당 **가능**).

**내장 자산(파일·조인 키):**
- **SMILES**: `ADORE\chemicals\ecotox_properties_with-oecd-function.csv` (9,404 화합물) — 컬럼 `chem_rdkit_can_smiles`(RDKit canonical) / `chem_pcp_can_smiles`(PubChem). **조인 키 = `test_cas`**. (SDF `chemicals\ecotox_properties.sdf` 6.8M, mordred `chemicals\mordred-descriptors.csv` 4,705행도 有 — mordred는 우리 파이프라인에서 미사용, RDKit-6 재계산.)
- **원본 taxonomy**: mortality 파일 내 `tax_class/tax_order/tax_family/tax_genus/tax_species/tax_gs`, 전체 lineage `tax_all`(class부터).
- **계통거리(Tier 5)**: `ADORE\taxonomy\FCA_pdm_species.csv` = **853×854 종×종 거리행렬**(row `Unnamed: 0`=종, col=종명). 계통트리 `ADORE\taxonomy\FCA_species.nwk`. 커버리지 ~853/1,267종(≈67%; `tax_pdm_available` True=53,232행).
- **DEB·형질(Tier 6)**: mortality 내 `tax_ps_ampv/ampkap/amppm`(DEB), `tax_lh_amd/lbcm`(생활사), `tax_eco_*`(생태). 소스 `ADORE\taxonomy\FC_amp_pseudodata.csv`(`latin_name,species_number,amp_v,amp_kap,amp_p_M`), `FC_amp_lifehistory.csv`, `FC_amp_ecology.csv`.
- **값**: `result_conc1_mean_mol`(molar; 예 0.00503 mol/L). **target = −log10(mol)** (예 pLC50 2.298). 낮은 농도=고독성=높은 pLC50.
- **사전정의 split**(ADORE 자체): `processed\{a,s,t}-{F2F,C2C,A2A,…}_mortality.csv` (예 `s-F2F-1_mortality.csv` 9,282행). ADORE의 challenge 시나리오 — 재사용 여부는 §I.

---

## D. 새 ADORE 로더 요구사항 (구현 전 명세)

**README의 "새 데이터셋=config 교체"는 ADORE엔 완전히 성립하지 않음** — 스키마가 tox-learn과 달라 **새 로더 필요**. config로 되는 부분(집계 키·필터·matched budget)은 유지, 아래는 코드:

| 요구 | 소스 | 조인 키 / 변환 |
|---|---|---|
| 컬럼 매핑 → 우리 스키마(smiles,species,endpoint,duration,target_log10) | mortality_processed.csv | `test_cas`→cas, `tax_gs`→species, `result_endpoint`→endpoint, `result_obs_duration_mean`→duration |
| CAS→SMILES 조인 | chemicals/…with-oecd-function.csv | `test_cas` 기준 left join → `chem_rdkit_can_smiles` |
| 값 변환 | `result_conc1_mean_mol` | **target_log10 = −log10(mol)** (pLC50); 0/결측 처리 |
| discovery/replication 필터 | endpoint+duration | discovery = LC50 & 96h / replication = 나머지 |
| taxonomy rank 파생 | `tax_all`/`tax_class..genus` | ⚠ **kingdom/phylum 없음** → class/order/family/genus 사용 또는 tax_group→상위 rank 매핑 별도 |
| 계통거리행렬 로드(Tier 5) | taxonomy/FCA_pdm_species.csv | 종명 매칭 → **PCoA 16d**(미해상 종=unknown 벡터) |
| DEB 형질 로드(Tier 6) | mortality `tax_ps_*`/`tax_lh_*` 또는 FC_amp_*.csv | species 기준; 결측 종=unknown 버킷 |

이후 **사다리 러너·control·fixed_proj·bootstrap은 데이터 무관하게 재사용**(스키마만 맞으면 동작). 스모크: ADORE 1-seed로 로더→집계→split→1개 tier 학습이 크래시 없이 도는지 먼저 확인.

---

## E. 재사용 vs 신규 구현 코드 (파일 경로·수정 지점)

**그대로 재사용:**
- `JC\scripts\run_q2_gnn_ladder.py` — GNN 사다리 러너(`--splits --backbones --variants --seeds --epochs`, resumable).
- `JC\scripts\run_q2_gnn_oof_tier1prime.py` — GNN-native Tier 1′ 5-fold OOF. **SPEC 4-0b stratum purge/re-add 수정 반영됨**(lgbm `_species_offsets` 미러링; `_stratum_key` import). 버그 재도입 금지.
- `JC\scripts\run_q2_replication_ladder.py` — naive+LightGBM Tier 0/1′/2 + control.
- `JC\scripts\bootstrap_q2_ladder.py` — `load_gnn/load_lgb`, `dd_bootstrap`, `bh_fdr`, `strength()`, N_BOOT=2000.
- `JC\jcim_v3\species_controls.py`(shuffled/dummy/zero), `stratum.py`(residualization fit/remove/restore), `naive_species_baselines.py`(naive + `run_naive_taxonomy_baselines` 계층 backoff), `rdkit_lgbm.py`(RDKit-6 descriptor from SMILES; `TAX_RANKS`).
- `JC\jcim_v3\models.py` — 기존 주입 모드(late_fusion/categorical/fixed_proj/taxonomy_original/taxonomy_ncbi), `build_v3_model`, `count_trainable_params`.
- `JC\jcim_v3\runner.py` — `run_v3_smoke`(featurize→scale→build→train→predict), `_train`, `_setup_reproducible`.
- `JC\results\q2_v4\audit\run_final_analysis_v3.py`(통합 122-대비 FDR 골격), `build_execution_matrix.py`.

**신규 구현:**
- **ADORE 로더**(§D) — `build_q2_datasets.py`의 `load_partition`(tox-learn USECOLS 전용)를 대체할 ADORE 버전. 신규 파일 권장(`build_adore_datasets.py`), config `configs\q2_dataset_adore.json`.
- **Tier 5 (계통거리 PCoA 인코더)** + **Tier 6 (DEB 형질)** 주입: `models.py`의 `_INJECTION_SUFFIXES` + `build_v3_model` + `forward`에 신규 모드 추가(taxonomy 주입 패턴 참고). LightGBM은 `rdkit_lgbm.py`에 phylo-PCoA/DEB feature 컬럼 추가. **⚠ D-MPNN은 ccmpnn read-only → 신규 주입 불가 가능성 확인 필수**(GraphConv+LightGBM만일 수 있음).
- PCoA 계산(계통거리행렬→16d, 고차원 민감도용 32/64d 병기) — 로더 또는 runner에서 종별 벡터 준비.

---

## F. 멀티 GPU 확인 사항 (실험 채팅 첫 과제)

**현재 코드**: `runner.py:_setup_reproducible`가 `torch.device("cuda" if torch.cuda.is_available() else "cpu")` — **항상 기본 GPU(cuda:0)**. device 인덱스·`CUDA_VISIBLE_DEVICES` 처리 **없음**. 러너/스케줄러도 없음(잡은 독립적).

**결론(감사): 러너 코드 수정 없이 병렬 가능.** 각 run은 독립 학습이므로 **env-var 핀 + run 목록 분할**이 최단:
- GPU0(5060 Ti): `CUDA_VISIBLE_DEVICES=0 conda run … run_q2_gnn_ladder.py --seeds 0 1 2 3 4 …`
- GPU1(4090): `CUDA_VISIBLE_DEVICES=1 conda run … run_q2_gnn_ladder.py --seeds 5 6 7 8 9 …`
- 두 프로세스 각각 harness-tracked 백그라운드. 예측 파일명이 seed로 갈려 충돌 없음(resumable). variant/split 축으로 분할해도 됨.

**첫 과제 절차**: ① `nvidia-smi`로 4090 실제 존재·CUDA 가시성 확인(과거 세션은 GPU 1장만 보였음 — **⚠ 두 번째 GPU 인식 확인 필요**). ② 위 env-var 분할로 1-seed 스모크를 두 GPU에 각각 띄워 둘 다 학습되는지 확인. ③ 확인되면 사다리 run 목록을 절반씩 분배. (선택: `runner`에 `--device` 인자를 추가해 세밀 배정 — 실제 구현은 director 승인 후.)

---

## G. 시간 추정 (실측 로그 기반 — 추정)

**단일 GPU(5060 Ti) 실측**〔audit/logs, GAP_EXECUTION_LOG〕: scaffold GNN 360 run = **36.7h**(~6.1분/run) · item6(categorical+fixed_proj) 200 run = **10.4h**(~3.1분/run) · Task A taxonomy 80 run = **5.5h**(~4.1분/run) · Phase4 OOF(5-fold) group 200 training = **9.8h** / scaffold **9.4h** · 최종 GNN 예측 1,280개·LightGBM/naive 815개. tox-learn 전체 GNN ≈ **~100h 단일 GPU**(추정).

**ADORE 조정**: 레코드 ~2× → GNN run당 **~1.5–2× 소요**. 신규 tier 5/6(+control) → variant 수 증가. run 수는 tier 추가분만큼 늘어남.

| 단계 | 낙관 | 현실 | 보수 |
|---|---|---|---|
| 데이터 재구축(로더·조인·split·taxonomy·PCoA·DEB·descriptor) | ~4h | ~1일 | ~2일 |
| GNN 재학습 **단일 GPU** | ~150h | ~220h | ~300h(신규 tier·고차원 민감도) |
| GNN 재학습 **2-GPU 병렬**(≈½) | ~75h | ~110h | ~150h |
| LightGBM/naive(CPU, ~2× 큼) | ~4h | ~6h | ~8h |
| 부트스트랩 + 통합 FDR | ~2h | ~3h | ~5h |
| **총 벽시계 (2-GPU 기준)** | **≈ 4–5일** | **≈ 1주** | **≈ 2주** |

*(단일 GPU면 GNN 부분이 2× → 총 ~2–3주. 2-GPU 병렬이 사실상 필수인 이유.)*

**원고 영향(한 문단):** 데이터 전면 교체 = 모든 수치·표·그림 재도출(방금 재프레임한 F2·within dd·baseline·커버리지·66.1% 등 전부). 단 **구조적 결론은 견고할 가능성 높고**(편중 73.6%로 더 강함, 커버리지 이미 ADORE 기반) **오히려 강화**됨 — 성능·커버리지가 단일 CC-BY 데이터셋에서 일관, **DEB·계통거리를 성능 tier로 편입**(tox-learn은 DEB 부재로 불가했던 확장).

---

## H. 진행 순서 (실험 채팅용)

1. **멀티 GPU 확인**(§F) — `nvidia-smi`로 4090 인식, env-var 분할 스모크.
2. **ADORE 로더 구축·검증**(§D) — `build_adore_datasets.py` + `configs\q2_dataset_adore.json`. **1-seed 스모크**(로더→집계→split→1 tier 학습 크래시 없음).
3. **데이터 재구축** — group/scaffold/designed-leaky split, discovery/replication, taxonomy(class–genus), 계통거리 PCoA, DEB 벡터, RDKit descriptor. 산출 검증(행수·종수·strata·pdm 커버리지).
4. **사다리 재실행** — 기존 tier(0/1/1′/2/3a/3b/4) → 신규 tier(5/6). backbone별(GraphConv·LightGBM; D-MPNN는 taxonomy·신규-tier N/A 확인). 2-GPU 분할. 각 backbone 1-seed 스모크 선행.
5. **통제·cold·cross-group** — shuffled/dummy/zero + fixed_proj(신규 tier 포함), cold(미관측 종), cross-group 외삽.
6. **부트스트랩·통합 FDR** — block 2000, global BH-FDR, strength(). `run_final_analysis_v3.py` 골격 확장.
7. **director 보고** — 수치만(해석·헤드라인은 director).

**원칙**: 매 단계 스모크 선행, **무검증 대량 실행 금지**, `GAP_EXECUTION_LOG.md` 지속 갱신, 실패는 해당 항목만 중단·계속, 이상/스모크 실패 시에만 정지·보고.

---

## I. 열린 판단 지점 (director 결정 대기)

1. **PCoA 민감도 차원** — 16d 주 + 고차원(32/64d?) 병기 범위.
2. **cross-group 외삽 정식 결과 범위** — fish↔crusta↔algae 어느 쌍을, warm/cold 조합 어디까지.
3. **계통거리(Tier 5)·DEB(Tier 6) tier를 어느 backbone에** — GraphConv+LightGBM 확정? D-MPNN 불가 확인 후.
4. **ADORE 자체 split 재사용 여부** — `{a,s,t}-*_mortality.csv`(challenge split)를 쓸지, 우리 group/scaffold를 새로 생성할지(사다리 설계엔 compound-disjoint group이 자연스러움).
5. **kingdom/phylum 처리**(§B flag 1) — 4-rank(class–genus)로 갈지, tax_group→상위 rank 매핑을 만들지.

---

## 참고 — 기존 산출물 위치 (승계)
- `results\q2_v4\audit\` : `unified_fdr_v3.csv`(122 대비), `FINAL_SUMMARY.md`, `execution_matrix{,_summary}`, `GAP_EXECUTION_LOG.md`, 분석 스크립트(`run_final_analysis_v3.py`, `run_taskB_analysis.py`, `run_phase1_scaffold_within.py`, `validate_oof_fix.py` 등).
- `results\q2_v4\HANDOFF.md` : tox-learn 실험 승계 문서(이번 문서의 전신).
- `results\q2_v4\provenance\source_provenance.md` : tox-learn 출처·SHA·재현성 R-2(파생 split 예치 권고).
- 원고: `Q2\working\manuscript\Manuscript_master.{docx,md}`(tox-learn 기반 재프레임 완료본; ADORE 전환 시 전면 재도출). **원고는 director 지시 없이 수정 금지.**

*(이 문서는 읽기 전용 감사로 작성 — 실행·학습·다른 파일 수정 없음. 모든 경로·수치는 파일시스템에서 재확인. conda run -n jcim_v3.)*
