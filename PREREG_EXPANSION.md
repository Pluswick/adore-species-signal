# ADORE Q2 확장 실험 — 사전등록 (B1 단독 재현)

> **동결 시점 = 성능 수치 확인 전.** 증거(실측): `results/q2_v4/runs_b1/` 학습 산출물 0, 진행 원장 `runs_b1/_status/progress.jsonl` 부재/공백, gatekeeping 출력 0. 타임스탬프 **2026-08-06**.
> 근거 데이터 = `data_b1/`(B1_final, P와 result_id·정밀중복 분리). 방법 = Phase 1 `PREREGISTRATION.md` 전면 준용.

## §0. 원칙 (director 최종 사양)

**데이터(B1_final) 외 모든 것을 Phase 1과 동일하게 한다. 데이터 때문에 물리적으로 불가능한 것만 예외로 하고, 예외는 본 사양에 명시한다.** 분류: **동일 / 불가피(절차 동일, 데이터라 값만 다름) / 판단필요(즉시 중단·보고)**.

## §1. 동일성 대조 (§0 게이트)

전 항목 대조표 = `results/q2_v4/audit/expansion_identity_check.md`. **판정 요약: 판단필요(설계) 불일치 = 없음.** 하이퍼파라미터·optimizer·loss·scheduler·early stop·valid carve·tier 구현·shuffled 통제·잔차화·앙상블 k·Δ/TOST·seed 목록 = **전부 동일(코드 원문 근거)**. split 절차·species_idx 맵·native/NCBI taxonomy·n_species·warm OOV 처리 = **불가피(새 데이터, 절차·seed 동일)**.

- ⚠ 불가피 특기: B1 group split은 test 종 ~**20% OOV**(Phase 1 warm ≈0). 처리는 Phase 1 코드와 동일(미학습 임베딩 그대로). prereg 사전기록대로 **불확정 증가 가능 = 실패 아님**.
- 하이퍼파라미터 출처 = **Phase 1 코드 경로 리터럴**(`run_q2_gnn_ladder.py`, `runner.py`). config JSON 로드 안 함(director 정정 반영).

## §2. Pre-flight 검사 (통과 못하면 중단)

- **§2.3 tier-input 비축퇴** — **PASS**. 전 rank 카디널리티 >1, t3a≠t3b, 통제 distinct. (감사: 관련 스크립트 출력.)
- **§2.4 t4 누출 재검증 (순열 검정 + OOF 경계 증명)** — 아래 §8에서 **오탐으로 규명·처분**. 결론: **누출 없음, 통과.**
- **§2.5 종-키 조인 sanity** — **PASS** (fails=0). species↔species_idx 전역 bijection(1975종·0–1974 연속), native tax 파일간 일관, embedded ncbi_*==SSOT(0 불일치), 4 split 전부 train/test key-disjoint, species-cold all-cold(overlap 0). (`data_b1/_ext/join_sanity_b1.json`)
- **가드**: tier_input_guard · prediction_io · io_atomic 활성 + 전 arm 입력 축퇴 검사.

## §3. 실행 규모 (census, 동결 = 3,794; director 확정)

| 블록 | run 수 | 구성 |
|---|---|---|
| gnn_warm | **1,920** | 24 variant × 2 backbone(D-MPNN·GraphConv) × 10 seed × 4 split(b1_group/scaffold/scaffold_generic/designed_leaky) |
| species_cold | **500** | 종-cold OOV (§8-OOV 평가시점 매핑 준용) |
| rank | **40** | rank 절단 taxonomy(genus / genus+family), t3a·warm·main, 3 backbone |
| deterministic | **74** | LightGBM 38 + naive 32 + SVD(t4) 4 |
| dprime | **1,260** | δ′ 앙상블 민감도(+90 seed/condition) |
| **합계** | **3,794** | |

- variant 목록·블록별 run_id 규약 = Phase 1 ladder와 동일(부록 A에 실측 열거; 인프라 매핑으로 확정).
- **GPU (director 정정 2026-08-06, "4090 단독" → 4090 주력 + 5060 Ti 보조):** 최초 "4090 단독" 사양을, 처리량 확인(순수 GNN ~22 run/h → 래더 ~80h) 후 director가 **5060 Ti 보조 투입**으로 변경 승인. **핵심 제약 — GPU를 seed에 고정 배정(변이·비교와 무관)**: seed **0–6 → 4090**(cuda:1), **7–9 → 5060 Ti**(cuda:0). 이유: Δ가 per-seed paired이므로 같은 seed의 cand/base가 같은 GPU에 있어야 GPU 효과가 차이에서 상쇄됨. δ(seed SD)는 교차-GPU 변동을 포함하나 이는 **Phase 1이 실제로 두 GPU(§9 "5060Ti 2 job/4090 1 job")를 쓴 것과 동일 성격** → 오히려 더 충실. 결정론 env(PCI_BUS_ID·CUBLAS_WORKSPACE_CONFIG=:4096:8) 양 GPU 동일. batch 등 학습 하이퍼파라미터 고정(변경 금지).

## §4. 세 마진 재도출 (상수 재사용 금지)

Phase 1의 동결값(δ=0.019777·δ′=0.005717·δ_det=0.087189)을 **재사용하지 않는다.** B1 산출물에서 **동일 수식으로 재산출**한다.

- **δ (per-seed)** = √(Σ_c (n_c−1)·s_c² / Σ_c (n_c−1)), condition = (backbone, tier, main, split, B1), s_c = 10-seed RMSE SD. pooling = primary 사다리(warm × group split × 전 GNN backbone × 전 tier, main만). (§4δ 준용)
- **δ_det** = √(mean(s_c²)), block bootstrap 2000회, 블록=**smiles**, LightGBM main, naive 제외. (§4δ_det 준용)
- **δ′** = √(Σ_c (k−1)·s_c² / Σ_c (k−1)), 분리 10-seed 앙상블 k=10. (§4δ′ 준용)
- **순서**: 전 run 완료 → 해당 부분집합에서 마진 산출 → **값+사양+타임스탬프 동결** → **그 다음** tier 비교·TOST. §4δ-break 파기 규약 공통 적용. **비교 전 동결 불변.**

## §5. 게이트키핑 (2단; Phase 1 §4G 준용)

- **Stage 1 warm**(전 primary 가설 BH-FDR) → **Stage 2 종-cold**(Stage 1 통과분). 게이트 단위 = (backbone, tier쌍) 독립 체인.
- **통과 조건 = TOST `동등`만**(4G-1; `불확정`·`유의차이`는 통과 불가). α = 0.05 단측×2 = 90% CI, `동등` = 90% CI ⊂ [−δ,+δ]. 판정 3범주 분리 보고. (§4G-5·4G-7 준용)
- **FDR 패밀리 = primary + exploratory.** (confirmatory 부재 = §6.)

## §6. Confirmatory 부재 (구조적 예외 — 명시)

B1에는 **replication 파티션이 없다**(B1_final = 단일 코퍼스; endpoint/effect/habitat/media 필터 후 disc/repl 분할 없음). 따라서 Phase 1의 confirmatory 미러(replication 2단)는 **구조적으로 성립 불가**. → **confirmatory 패밀리 없음.** ⚠ **대체물 조작 금지**(임의 홀드아웃을 confirmatory로 위장하지 않음). 이는 데이터 유래 불가피 예외이며 원고에 그대로 명시.

## §7. Phase 1 대비 = 병렬 표기만 (신규 통계검정 없음)

B1 결과를 Phase 1 결과와 **나란히 표로 대조**한다. **Phase1-vs-B1 신규 통계검정을 수행하지 않는다**(두 실험은 독립 코퍼스·독립 마진; 교차검정은 사전등록에 없음). 대조는 판정 3범주 개수·방향 일치 여부의 **서술적 병기**에 한정.

## §8. §2.4 게이트 처분 — t4 순열 flag = 특성 규명된 오탐 (director 승인 2026-08-06)

**flag 사실**: b1_scaffold에서 순열 검정(shuffled-label factor가 no-factor를 improve>TAU=0.02로 이김) = LEAK 신호. 문자 그대로면 중단.

**규명 3단(성능 보기 전, 결정론/구조 증명):**
1. **① OOF 경계 증명(결정론)** — B1 4 split 전부 PASS. 어떤 train 행의 SVD factor도 자기 라벨로 미계산(inner·outer OOF), test 미참여, unseen→0. → **구축 누출 없음.** (`data_b1/_ext/tier4_oof_index_proof_b1.json`)
2. **② Phase 1 8 split × 20 perm** — 전부 mean<TAU(최대 +0.0127). → **t4 공통 아티팩트 아님.** (`data/_ext/tier4_permutation_leak_robust_phase1.json`)
3. **③ 결정적 대조(용량바닥+종오배정, 30 perm)** — b1_scaffold flag(+0.052)의 **약 79% = 순수 노이즈 16컬럼 용량바닥(+0.039)**; 바닥 초과분 +0.010(<TAU)이며 종 매칭과 무관(A≈C, A>C=0.50). → **순열 검정 귀무(랜덤라벨 improve≈0)가 이 split에서 위배된 오탐.** (`data_b1/tier4_decisive_control_b1.json`)

**처분(director 승인):** b1_scaffold flag = **오탐으로 기록**, **t4 on b1_scaffold 포함해 진행.** Phase 1 게이트 기준(순열+OOF)은 그대로 유지(동일성). 상세 = `audit/tier4_leak_mechanism_b1_vs_phase1.md`.

**⚠ 하위 caveat (director 승인 처리 = 본 실험 후 삽입):** b1_scaffold는 용량바닥 +0.039 실재 → 피처 추가 tier가 바닥 이득 업음. **본 실험은 계획대로 진행**하고, **gatekeeping 단계에서 노이즈-바닥 기준선을 tier 비교에 삽입**해 "용량 vs 종 신호"를 분리(Phase 1은 바닥≈0라 무관). 코드 결함 아님(①), 데이터 유래 성질.

## §9. 중단 조건 (director 확정 — 이 둘만)

1. **scope/claim-range 변경**, 또는
2. **data/code 무결성 결함**(실제). 
그 외(게이트 발화 포함)는 **결정론/구조 증명으로 (c) 특성 규명된 데이터 성질**로 판명되면 진행+문서화. 진행 판단은 **①처럼 결정론·성능전·결과독립인 기전 증명**을 요구.

## §10. 산출물 위치

- 데이터: `data_b1/` (splits·cold·NCBI·tier_input_reference).
- run 산출물: `results/q2_v4/runs_b1/`, 진행 원장 `runs_b1/_status/progress.jsonl`.
- 마진 동결: `runs_b1/_status/` 또는 `audit/`(delta_*_b1_frozen.json).
- 감사: `audit/tier4_leak_mechanism_b1_vs_phase1.md`, `audit/expansion_identity_check.md`.

## §R. 동결 증거 (결과 확인 전)

- 본 문서 작성 시점 `runs_b1/` 학습 산출물 **0**, 진행 원장 공백, gatekeeping 출력 **0**.
- §8 규명 3단은 전부 **성능(tier RMSE) 확인 전** 수행된 **누출-QC 진단**(shuffled/noise factor 대상 — 실험의 tier 성능 아님).
- 타임스탬프 **2026-08-06**. 이후 §4 마진은 run 완료 후·비교 전 동결.

---
## 부록 A. variant·run_id 규약 (Phase 1 실측 인벤토리 확정 2026-08-06)

Phase 1 `runs/gnn/predictions`(4,880) 실측 분해로 확정: 22 ladder + 2 t1′ = **24 warm variant**(rank 2·cold 별도). "24"는 22 ladder + **t1′ 2종**이며 film/추가 fixed_proj 아님(별도 스크립트 `run_q2_gnn_oof_tier1prime.py`라 ladder 스크립트엔 22만 보임).

**gnn_warm 24 variant** (× 2 backbone × 10 seed × 4 b1 split = 1,920):
- t0: `no_species` (1)
- t1: `species_bias_only` · `shuffled_/zero_/dummy_species_bias_only` (4)
- t2: `true_/shuffled_/zero_/dummy_species_categorical` (4)
- t3a: `true_/shuffled_/zero_/dummy_species_taxonomy_original` (4)
- 용량통제: `true_species_fixed_proj` (1)
- t3b: `true_/shuffled_/zero_/dummy_species_taxonomy_ncbi` (4)
- t4: `true_/shuffled_/zero_/dummy_species_late_fusion` (4)
- t1′: `tier1prime_oof` · `shuffled_tier1prime_oof` (2, main+shuffled만 — zero/dummy 없음, Phase 1 동일)

**rank** (b1_group만): `true_species_taxonomy_genus` · `true_species_taxonomy_genusfamily` × 2 GNN bb × 10 seed = 40(GNN). (+LightGBM rank는 deterministic 블록 집계에 포함.)

**run_id 규약**: GNN `{backbone}_{variant}_{split}_s{seed}_e{epochs}_nfull` (`runner.py:546`); LGBM `{baseline}_{split}_s{seed}`; SVD t4 `LightGBM_RDKit_species_svd_factor_{split}_s{seed}`. b1 split명이라 Phase 1 run_id와 자연 분리.

**실행 구조**: 코어(`jcim_v3/`) 무편집·데이터경로 파라미터화 완비 → B1 드라이버가 동일 코어 함수를 `data_dir=data_b1`·`out_root=runs_b1/…`로 호출 + 각 run 후 원장 append + skip-if-exists. GPU 핀 = `CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=1`(4090) + `CUBLAS_WORKSPACE_CONFIG=:4096:8`(Phase 1 결정론 재현).
