# ADORE Q2 — 사전등록 확정본 (Phase 1)

> step 2 종결. tier 성능을 보기 전에 규칙을 고정한다. 데이터 재구축(step 3) → 학습(step 4)은 이 문서 확정 후.
> 근거 수치 = `results/q2_v4/audit/GAP_EXECUTION_LOG.md` Session 1–11. conda run·순서 동결·무결성 우선.

## §1. Phase 1 = 안 A (≈ 32.9일, 방법 B)

| 항목 | 내용 |
|---|---|
| tier | 0 / 1 / 1′ / 2 / 3a / 3b / 4 |
| warm | 전 tier, 전 통제(main·shuffled·zero·dummy·fixed_proj) |
| cold·cross-group | {0, 2, 3a, 3b, 4}, 통제 = **shuffled만** |
| backbone | GraphConv·D-MPNN(GPU; taxonomy = 학습 per-rank embedding context 신규구현) + LightGBM·naive(CPU) |
| seed | GNN 10 / 결정론 1 |
| split | group(CAS) · scaffold(murcko·generic) · designed-leaky((smiles,species) pair) |
| 데이터 | discovery(LC50@96h) + replication(LC50 24/48/72h + EC50) |
| rank 절단 | 포함 (§2-rank) |
| 실행 | 방법 B — 5060 Ti 2 job / 4090 1 job, batch 256·원 lr·seed 불변(δ 무손상) |
| fixed_proj | Tier 3a·3b·4 (Tier 2 제외[readout 폭=종수]·Tier 1·1′ 구조상 N/A·cold 미적용) |
| naive 열 | tier 0·1·1′·3a·3b (phylo/trait-kNN 제외) |

**§2-rank. taxonomy rank 절단(포함)**: 조건 3 = `genus만`/`genus+family`/`4-rank 전부`(=tier 3a 재사용) → 신규 2조건 ≈ 40 run. tier 3a only, warm·discovery·group, 통제 main만, 3 backbone(GC·DM·LGBM), seed 10/1. tier 3b 미적용. **근거**: Phase 1의 "taxonomy=one-hot 무승부" 관찰의 유일한 기전 설명(genus만으로 유지되면 taxonomy 예측력 = genus 수준 종 identity).

## §2. tier 5·6 성능 미평가 (커버리지 사다리 限)

계통거리(5)·DEB(6)는 **커버리지 사다리에서만** 다루고 **성능 실험 전면 제외**(Phase 2/후속). 함께 제외: PCoA 차원 스윕·tier6 광폭/협폭·naive phylo/trait-kNN·tier5·6 fixed_proj.

> **[사전등록 문구]** cross-group 조건에서 계통거리·형질 표현의 커버리지가 종 기준 41.5%/18.8%, 조류 0%로 붕괴한다. 평가 표본이 부족해 동등성 판정이 성립하지 않거나, 성립하더라도 성능 차이가 커버리지 차이와 분리되지 않는다.

- **indicator-only ablation 부재 = tier 6 미평가의 자동 귀결**(판단 아님). tier 6을 평가하지 않으므로 그 ablation이 성립하지 않는다.

## §3. D16 값 유효 규칙 (확정)

- **하한**: `pLC50 > −1.75` — 순수 물 55.5 M, 수용액 초과 불가. 현재 **0행 제외**(가드레일). "용해도 초과" 근거 폐기(제외 대상이 miscible 용매/염 ethanol·methanol·acetone·DMSO·ethylene glycol·NaCl → LC50≥1M 정상).
- **상한**: 전역 pLC50 임계 **폐기**(Endrin 오류 12.0–14.8 ∩ deltamethrin 실재 12.2). 대신 **질량농도 검출하한 `mass = conc_mol×MW×1e9 < 0.1 ng/L` 제외**. 근거 = 수중 농약 LC-MS/MS MDL 0.01–1.32 ng/L·LOQ 하한 ~0.1 ng/L. **임계 0.1 선정**: 0.5는 실재 pyrethroid(deltamethrin 0.32) 오클립, 0.05는 확정오류 잔존 — 양방향 부적합 실측 확인.
- **path 3(≥0.1 & pLC50≥12, 6행) 개별판정**: deltamethrin(52918-63-5,0.32)·pyrethroid(76703-62-3,0.42) **보존**; Endrin(72-20-8)×3·Methyl parathion(298-00-0,0.14) **제외**.
- **최종 제외 = 14행**(검출하한 10 + path3 4). 실재 pyrethroid 2행은 구 규칙(pLC50<12)이 오제거하던 값. 훈련·평가 양쪽 적용.
- **⚠ 주 분석 데이터셋 무손상**: 제외 14행은 **전부 replication**(48h Daphnia 살충제) → **discovery(주 분석, LC50@96h)는 D16 규칙의 영향을 0행 받음**. discovery 26,523 records 불변. (주 결과가 데이터 큐레이션 판단의 영향을 받지 않는다는 사실 자체를 기록.)
- **감사추적**: `exclusion_reason`(below_detection_limit/source_conversion_error/retained_plausible) + 문헌출처·자릿수차·판정논리 + **타임스탬프(학습 전)**. 목록 = `results/q2_v4/data/exclusion_audit_trail.csv`.

## §4. 확정 6개 항목

**4-1. tier 5·6 미평가 사유** = §2 문구(회피 아닌 검정력·교락). indicator-only 부재 = 자동 귀결.

**4-2. cold 목적 변경**: 원래(핸드오프 §3-2) cold = tier 5·6 외삽 시험 → tier 5·6 제외로 이제 **"무승부가 외삽 조건에서도 유지되는가"**. taxonomy를 cold에 포함해 성립 — cold에서 tier 2·4는 미학습 임베딩/one-hot으로 퇴화(코드확인: OOV remap 없음, `n_species=full.max()+1`)하나 taxonomy는 형제 종 통해 전이(tax_codes 전종·per-rank emb 학습).

**4-3. 매트릭스 X 성격 구분**:
- **범위 결정**(가능하나 안 함): tier 5·6 전 행.
- **수학적 축퇴**(구조상 불가): naive×tier2(=tier1 항등, 수치확인 Δ=2.2e-15)·naive×tier4(경사학습 모델 부재, Δ=0).

**4-4. δ 순서 강제**: censoring 적용 → 데이터 재구축 → 학습 → **최종 지표에서 δ 산출** → 동결(값·근거·타임스탬프 파일) → tier 비교·TOST. **처리 전 δ 산출 금지.** 동결 후 불변 검증.

**4-5. cold 통제 축소 사유**: zero/dummy 제외 = no-species와 실질 중복. shuffled 유지 = 계통·종 신호 진위 판별 필수 통제이며 **차원 보존 순열이라 용량 매칭 동시 충족**(cold fixed_proj 불요, 용량 반박 차단).

**4-6. FDR 패밀리 경계**: primary/exploratory 경계를 **실행 전 확정·기록**. 확정 = `results/q2_v4/audit/fdr_family_boundary.md`(3패밀리: primary/confirmatory/exploratory; 각 BH-FDR 독립 + global 병기).

## §4G. 계층적 게이트키핑 (primary/confirmatory 내부, 결과 전 확정)

- **구조(2단, cross-group 제외)**: primary 패밀리 *내부*를 평가체제로 계층화. Stage 1 **warm**(전 primary 가설 BH-FDR) → Stage 2 **종-cold**(Stage 1 통과분). confirmatory(replication)도 **동일 2단 미러링**. exploratory는 게이트 없음. (~~Stage 3 cross-group~~ = §7로 제외.)
- **게이트 단위 = (backbone, tier쌍) 각 비교의 독립 체인**("warm 전체 통과"가 아님). 어떤 비교가 warm에서 실패하면 그 체인만 종료.
- **4G-1 게이트 통과 조건 = TOST `동등`만.** ⚠ 일반 게이트키핑("유의하면 다음")과 **방향 반대**. `불확정`·`유의한 차이`는 통과 불가. (미명시 시 리뷰어 오독 방지 위해 명기.)
- **4G-2 게이트 건전성 = δ 종속.** δ 과대 시 넓은 CI도 `동등` 판정 → 게이트가 필터 기능 상실. δ = pooled within-condition seed SD(데이터 기반·임의 상수 없음)로 통제. `불확정`은 검정력 부족을 반영하며 통과 불가 → 검정력 부족이 가짜 통과 안 만듦.
- **4G-3 cold 단계 `불확정` 증가 예상(사전기록).** δ는 warm·discovery·group에서 산출; cold는 test strata 적어 CI 넓어짐 → 같은 δ 적용 시 `불확정` 증가 예상. **이는 무승부의 증거 아님** — "불확정 더미"를 "차이 없음"으로 읽지 않기 위해 미리 기록.
- **4G-4 순서 정당성.** warm ⊃ 종-cold ⊃ cross-group = 처방 적용 범위 확장 순서, 데이터 무관 결정.
- **4G-5 판정 3범주 분리 보고(강제)**: `동등`(무승부 포함) / `유의한 차이`(무승부 예외로 **본문 명시**, Supp 매장 금지) / `불확정`(미결로 **본문 명시**). 세 범주 개수 각각 집계. `불확정`을 어느 쪽과도 섞지 않음. 게이트는 어느 결과에 FDR 통제 주장을 붙일지만 정하고, 실행 조건 전량 Supp 수록 원칙 유지.
- **종-cold = primary**(집계 전체; disc 1,395 strata ≥ 하한 1,000). **tax_group별 분해 = exploratory**(algae disc 4종/8strata = 판정불가 규모, primary 넣으면 `불확정` 양산). 승격 근거: 종-cold는 taxonomy 최대 유리 조건(형제 종 전이 최대) — 최대이점에도 못 이기면 "계통 정보 미사용"의 최강 증거. confirmatory 배제: warm/cold는 평가체제 축, discovery/replication은 파티션 축(직교); 종-cold는 warm과 다른 객체 검정(cold의 tier2·4 = 미학습 표현).

- **4G-6 결정론 tier 게이트 δ(확정)**: 결정론 tier(LightGBM·naive)의 게이트 판정은 **자체 block bootstrap δ**로 내린다. within-backbone이 primary이므로 비교 기준도 해당 backbone 자신의 것. GNN pooled δ 차용은 **민감도로 병기**하고 두 δ가 상이 판정을 준 비교는 상이함을 명시. 두 δ는 성격이 다른 양(재실행 잡음 vs 표본 불확실성)이라 **대소를 사전에 가정하지 않고** 산출값을 함께 보고. (⚠ 결정론 tier가 쓰는 것은 seed SD가 아니라 **bootstrap SD**이며 표본 불확실성이라 0에 가깝지 않고 GNN δ보다 클 수도 있음 — 사전 단정 금지.)

- **4G-7 TOST 판정 규칙 명시 (2026-08-03 추가 · 자유도 축소이지 신규 생성 아님).** §4G/§4δ 검토 중 판정 규칙에 미명시로 확인된 지점을 **δ 동결·어떤 tier 비교 수행 전에** 사전에 못 박음(결과 관찰 후 채우면 방어력 소멸).
  - **(1) α·CI 수준 [director 확정 2026-08-03]**: TOST α = **0.05 단측 × 2 = Δ의 90% CI**. `동등` 판정 = **Δ의 90% CI ⊂ [−δ, +δ]**. (TOST 표준; 종전 미명시=없음.)
  - **(3) TOST 기각 ∧ NHST 기각 = `동등` [director 확정 2026-08-03]**: Δ의 CI가 0을 포함하지 않으면서(NHST 기각) 동시에 [−δ, +δ]에 완전히 들어가는(TOST 기각) 조합은 **`동등`으로 배정**한다. 사유: 본 연구 주장은 "차이가 시드 변동 범위 내"라는 **실질적 동등성**이지 통계적 검출 가능성이 아니다 → 검출되더라도 δ 내면 주장 대상이며 게이트 통과. 4G-5 3범주 표에 이 조합을 명시하여 판정 시점 재량을 제거(종전 미배정=없음).
  - **(2) Δ 대표 RMSE 정의 = 별도 미결(D-ΔδMATCH, director_결정필요.md).** Δ가 앙상블 RMSE(b) 위·δ가 per-seed RMSE SD(a) 위로 **대상 불일치** 확인 → escalate, δ 산출 보류. **본 4G-7은 (2)를 확정하지 않으며**, (1)·(3)은 Δ 대상 선택과 무관한 α/CI·범주배정 자유도만 닫는다.
  - **증거 기록(사후 수정 아님)**: 추가 시각 **2026-08-03**. **δ 미산출·미동결**(동결 산출물 부재 확인). **tier 비교·TOST·dd·게이트 산출물 전무**(`run_q2_gatekeeping.py` 부재, `results/q2_v4/runs/bootstrap` 비어 있음(산출물 파일 0)). 개정 성격 = **판정 기준을 결과 확인 전에 고정**하는 것이지 새 자유도 생성이 아님(§4δ "판정 자유도 문구" 참조).

## §4C. 비교 집합 확정 (director 배정 2026-08-03; §3 열거 모호점 해소, 결과 확인 전)

- **Primary 동등성 TOST (within-backbone) = {t3a vs t2, t3b vs t2, t4 vs t2} × {dmpnn, graphconv} = 6.** warm × discovery × group.
  - **t3b 포함 근거**: t3a와 같은 구조물(taxonomy)의 다른 출처(NCBI) → 등록 집합의 완성.
  - **t1·t1′ = TOST primary 미포함**(처방 후보가 아닌 참조 tier). t1·t1′ 관련 TOST는 **exploratory**.
- **Primary 우월성** = 기존 문서(`fdr_family_boundary_DRAFT.md` A1) 따름: 종 tier {t1,t1′,t2,t3a,t3b,t4} vs {t0(no_species), shuffled_self} × backbone, warm×discovery×group, 일측 우월성.
- **나머지 소속 = 기존 문서 따름**: cross-backbone(DRAFT A2 "같은 backbone 내" ⟹ primary 아님) · naive(§1 표) · replication(confirmatory 미러) · split(group=primary) · **arm 구조 = §4Δ 4-arm DD**(within-backbone은 candbase=refbase=동일 t0 → 직접차 `t_cand − t2`로 환원).
- **F1~F4 legacy 불사용**: ADORE 비교 스크립트는 신규 작성(비교 게이트 개방 후 구현).
- **증거(결과 확인 전)**: 2026-08-03. δ 동결됨(값 불변) · `run_q2_gatekeeping.py` 부재 · `runs/bootstrap` 산출물 파일 0 · dd/TOST/게이트 출력 0.

### §4C-Explore. Exploratory 패밀리 내용 확정 (director 2026-08-04; 언블라인딩 이전 추가, 결과 확인 전)

전부 **exploratory 패밀리 · 자체 BH-FDR · 게이트 불참**. 증거 = 비교 산출물 0 + 하드가드 REFUSE.

| 항목 | 사양 |
|---|---|
| **rank 절단** | genus / genus+family 각각을 **① full 4-rank t3a 대비**(절단 손실=기전) **② t2 대비**(사다리 일관성) — 둘 다. tier 3a·warm·discovery·group·main · GraphConv·D-MPNN·LightGBM |
| **support-bin** | 종당 데이터량 **1–5 / 6–20 / 21–100 / 100+**. primary TOST 쌍(t3a·t3b·t4 vs t2)을 구간별 분해(행 필터=종의 train count 구간) |
| **tax_group** | primary TOST와 **동일 쌍**을 **fish / crusta / algae**로 분해(행 필터=종의 tax_group; 실측 3군 전종 커버, mortality 원본 species→tax_group SSOT 유도). 새 쌍 없음 |
| **scaffold** | **primary 비교집합 그대로**를 scaffold(murcko)·scaffold(generic) split에서 |
| **designed-leaky** | **primary 비교집합 그대로**를 designed_leaky split(H3 통제) |
| **cross-backbone** | primary TOST 쌍의 ref를 **다른 backbone**에서 취하고 **각 backbone이 자기 t0를 base**로(진짜 4항 DD). **조합 = (a) 6개 확정**(director 2026-08-04): ref = **LightGBM t2 고정**, cand = **{t3a, t3b, t4} × {dmpnn, graphconv}** = 6. 각 arm은 **자기 backbone의 t0**를 base(GNN cand base=GNN t0, LightGBM ref base=LightGBM t0). warm·discovery·group·main. 마진 = **δ**(per-seed; GNN 시드 변동이 지배, LightGBM은 결정론 상수 → 시드 축 tile, block 변동만). 자체 BH-FDR·게이트 불참 |

**cross-backbone 조합 확정(director 2026-08-04, 언블라인딩 이전·비교 산출물 0):** (a) 6개만 채택. **(b) 전 backbone쌍·(c) 역방향 제외** — "GraphConv t3a vs D-MPNN t2"류는 실무자가 묻지 않는 질문이고, 12~18조합으로 늘리면 exploratory BH-FDR가 희석되어 rank 절단 검정력이 깎임. 6항 전체 확정.

## §4δ. δ pooling 범위 — 최종 사양 (확정·변경 불가, 마지막 자유도)

- **condition 단위** = (backbone, tier, control=**main**, split, data)의 한 셀. within-condition 변동 = 그 셀의 **10 seed RMSE의 SD** `s_c`(df `n_c−1 = 9`).
- **pooling 집합** = **primary 사다리 = warm × discovery × group split × 전 GNN backbone × 전 tier, main 조건만**. **통제 조건(shuffled·zero·dummy·fixed_proj) 제외.** cold·replication·scaffold·leaky도 미포함(δ는 여기서만 산출, 타 체제엔 적용만).
  - **제외 근거(정의상)**: δ = **신호를 학습하는 모델의 재실행 잡음**. 통제 조건은 종 신호가 파괴·대체된 모델이라 δ가 재려는 양과 **종류가 다름** → 크기와 무관하게 정의에서 배제. ⚠ "통제가 δ를 부풀린다"는 경험적 예측을 근거로 쓰지 않음(결과로 반증 가능); 분산 동질성 보고도 근거에 결합 안 함 — 결과가 말하게 둠.
- **pooled SD 수식**: δ = √( Σ_c (n_c−1)·s_c² / Σ_c (n_c−1) ) — df-가중 pooled within-condition SD. RMSE는 stratum 집계·endpoint/duration residualization 후 최종 지표에서 산출.
- **적용**: 전 TOST/게이트에 이 단일 δ(결정론 tier는 §4G-6 자체 bootstrap δ). 산출 시 **사양+값+타임스탬프**를 동결 파일에 기록, 이후 불변 검증. **결과 관찰 후 pooling 변경 금지**(중간 관찰이 증명적으로 무해).
- **판정 자유도 문구(확정)**: 판정을 결정하는 기준(마진 δ · 유의수준 · 판정 3범주 · 비교 집합 · 다중검정 구조 · 게이트 조건)은 전부 결과 확인 전에 고정됐다. 서술·배치 등 판정에 영향을 주지 않는 선택은 분석 시점에 이뤄지며, 그 사실을 명시한다. (⚠ "남은 분석 자유도 = 0"류 절대표현 폐기 — 서술·그림 선택은 항상 남으므로 반증 가능. director 확정 2026-08-03.)

## §4δ′. 앙상블 척도 δ′ — 민감도 전용 (신설, §4δ 불변; D-ΔδMATCH 확정 2026-08-03)

> **§4δ는 한 글자도 개정하지 않는다.** 주 척도 = **(가) per-seed**(Δ=시드별 RMSE, δ=§4δ 그대로, C=14, 추가 run 0). 본 §4δ′는 **민감도 전용** 신규 사양이다.

- **주 척도 확정(A-1)**: Δ = **per-seed RMSE** 기준(대표 앙상블 RMSE 아님). δ = §4δ 그대로(C=14, t0·t1′ 포함). 보조 스케일(tier0↔tier1 간격)도 per-seed. 근거: 처방("종 표현에 자원 쓰지 마라")이 적용되는 장면 = 단일 학습 실행.
- **결과 표(A-2)**: 앙상블 RMSE(대표 지표, 유지) + **per-seed 평균±SD**(검정 대상)를 **같은 표 병기**, **판정은 per-seed 열에 붙음을 각주 명시**. 앙상블 Δ는 기술통계로 병기하되 주 척도에서 판정 안 붙임.
- **§4δ′ 정의(A-3)**: condition 단위 = §4δ와 동일(C=14). 각 셀에서 **분리(disjoint) 10-시드 앙상블 k=10개**의 앙상블-RMSE 산출, `s_c` = 그 **10 값의 SD**(df 9). **δ′ = √(Σ_c (k−1)·s_c² / Σ_c (k−1))** = §4δ와 동일 형태(pooled within-condition SD).
- **Δ′(A-3)** = 앙상블 RMSE 차이(현행 `_dd_core` dd 방식). **CI = canonical 10-시드 앙상블에 대한 block bootstrap + 그 10 시드 내 재추출**. **추가 90 시드는 δ′ 추정 전용, Δ′ 계산 미투입.**
- **규모**: 조건당 추가 90 시드 = **1,260 run**. k=10 사유 = per-seed δ와 정밀도(df 9/condition, pooled 126) 일치 → 두 척도 갈림이 "δ′ 부정확"으로 설명되지 않게.
- **민감도 위상(A-4)**: **primary 패밀리 미포함**(같은 가설 다른 척도 재검정 → primary 넣으면 가설 2배·주 검정력 감소). **민감도 세트 자체 BH-FDR**. **게이트키핑 미투입**(게이트=primary 전용). 판정 3범주·α(90% CI)·§4G-7 네 번째 칸 규칙은 **양 척도 동일 적용**.
- **실행 제약(A-5, 전부 필수·확정)**:
  1. canonical 10-시드(0–9) = 대표 앙상블 정본 **불변**. 100-시드 앙상블 만들지 않음.
  2. **시드 오프셋 규칙**: δ′ 추가 시드 = **10–99**(조건별). rank 절단(seeds 0–9, 별개 tier t3a_g/t3a_gf)과 **미충돌**(δ′ 추가는 10부터).
  3. **앙상블 묶음 규칙(사전 고정)**: 시드 순 **연속 10개**씩 = 앙상블 j(1..9) = seeds [10j..10j+9]; canonical = 앙상블 0 = [0..9]. 총 k=10.
  4. **카드 배정(현행 구성 일치)**: 12/14 조건(t0-t4×2bb) = 앙상블당 seeds 끝자리 0–6(7개) **5060Ti**, 7–9(3개) **4090**(현행 canonical과 동일 7:3). 2/14 조건(t1′×2bb) = **4090 단독**(현행대로). 스케줄러 임의 배정 금지.
  5. **별도 네임스페이스**: δ′ 산출 = `results/q2_v4/runs/gnn_dprime/`(대표 앙상블과 물리 분리).
  6. 하이퍼파라미터·패킹·결정론 **완전 동일**, 변수 = 시드 번호뿐.
- **증거 기록(A-6, 사후 아님)**: 추가 시각 **2026-08-03**. **δ·δ′ 미산출·미동결**(동결 산출물 부재). **tier 비교·TOST·dd·게이트 산출물 전무**(`run_q2_gatekeeping.py` 부재, `results/q2_v4/runs/bootstrap` 비어 있음(산출물 파일 0)). 성격 = 판정 기준을 결과 확인 전 고정(§4δ "판정 자유도 문구" 참조).

## §4Δ. Δ 검정 통계량 정의 — per-seed paired (D-ΔδMATCH 확정 2026-08-03)

- **Δ = mean_s( [RMSE_cand(s) − RMSE_candbase(s)] − [RMSE_ref(s) − RMSE_refbase(s)] )** (2-arm 일반형: `mean_s(RMSE_A(s) − RMSE_B(s))`). **같은 시드 집합을 양 arm에 적용(paired).** `RMSE_arm(s)` = arm의 **시드-s 예측**의 RMSE(stratum 집계·endpoint/duration 잔차화 후 최종 지표). ⚠ 대표 앙상블 RMSE(§4δ′ 병기용) 위가 **아니다**.
- **부트스트랩**: block(compound_key) 재추출 + 시드 재추출을 **양 arm 공통 `sset`**(짝지은 구조; §B 확인 = 시드가 배치순서·백본 init 공유). CI = 90%(§4G-7 (1)). δ·δ′는 단일-arm 양이라 짝지음 무관·불변.
- **Methods 문구(과장 금지, 확정)**: "시드 `s`는 tier 간 배치 순서와 message-passing 백본 초기화를 공유하나, 종 표현 모듈·readout FFN 초기화와 dropout mask는 대응되지 않는다. 따라서 짝지은 분석은 시드 유래 공통 변동의 **일부만** 제거하며, 대응 정도는 종 모듈 크기에 따라 tier 쌍마다 다르다."
- **보조 스케일(tier0↔tier1 간격)**: **per-seed 척도**로 산출(앙상블 아님). 비교 단계 산출물(δ 동결·게이트 개방 후 산출; 지금은 사양만).
- 근거: 처방("종 표현에 자원 쓰지 마라") 적용 장면 = 단일 학습 실행 → 마진·검정이 per-seed 위.

## §4δ-break. δ 동결 파기 규약 (director 확정 2026-08-03, 동결 시점 사전 기록)

> **δ 동결 파기 규약.** δ pooling 집합에 속한 run이 사후에 무효로 판정되면: (a) δ를 재산출한다. (b) **구 δ 값과 신 δ 값을 모두 보존**하고 어느 것도 삭제하지 않는다. (c) 파기 사유·시점·무효 run 목록·영향 범위를 감사 추적에 기록한다. (d) 구 δ로 이미 산출된 판정이 있으면 **전량 재실행**하고, 구·신 판정을 함께 보고한다. (e) 이 규약은 δ 동결 시점에 사전 기록된 것이며 사건 후에 만들어지지 않았음을 타임스탬프로 증명한다.

(기록 시각 **2026-08-03**, 주 δ 동결(`audit/delta_primary_frozen.json`)과 동시. 파기 사건 발생 전 사전 기록.)

## §4δ-impl. 동결된 δ의 함의 (director 확정 2026-08-03, 결과 확인 전 기록)

> **동결된 δ의 함의(결과 확인 전 기록).** δ = 0.019777은 관측 RMSE 규모 대비 약 1.5~2%로, 사전등록 §8이 기록한 위험("δ 과대 시 게이트가 필터 기능을 상실")은 **해소**된다. 반대로 마진이 좁으므로 CI가 [−δ, +δ]에 들어가기 어려워 **`불확정` 판정 비중이 높을 수 있으며, 이는 warm 단계에서도 그러할 수 있다.** 이 예상은 어떤 tier 비교도 수행되기 전에 기록되며, 결과 확인 후 `불확정` 다수를 "차이가 없었다"로 읽지 않기 위한 것이다. δ는 결과가 아니라 동결된 설계 파라미터이므로 그 함의를 사전 기술하는 것은 순환이 아니다.

## §4δ_det. 결정론 tier 동등성 마진 (director 확정 2026-08-04)

> **§4δ_det — 결정론 tier 동등성 마진.** condition = (LightGBM, tier, control=main, group split, discovery, warm)의 한 셀. 각 셀에서 **block bootstrap 2000회, 블록 = `smiles`**로 최종 지표 RMSE의 표준편차 `s_c`를 산출한다. pooling 집합 = warm × discovery × group split × **LightGBM** × 전 tier, **main 조건만**. naive는 검정 대상이 아니므로 제외한다. 부트스트랩 SD는 복제 수가 동일하므로 §4δ의 df 가중이 등가중으로 축약되며, **δ_det = √( mean(s_c²) )**. 산출·동결 순서와 §4δ-break 파기 규약은 δ·δ′와 동일하게 적용한다.
>
> **블록 키 = `smiles` (director 정정 2026-08-04, 동결 전; 통계량 변경 아니라 문서가 코드를 따라가는 정정)**: 의존 구조는 CAS(등록 식별자)가 아니라 **모델이 보는 분자 입력(SMILES)**을 통해 흐른다. 같은 SMILES 행은 같은 입력이라 오차가 상관되며, 그것이 블록을 묶는 이유다. 한 SMILES가 여러 CAS로 갈린 경우 `compound_key` 블록은 상관된 행을 쪼개 CI를 부당하게 좁힌다. **전 비교 파이프라인(δ·δ′·δ_det·Δ, §4Δ의 C3 포함)이 동일하게 `smiles`로 블록**한다(비교마다 블록이 달라지지 않도록 통일).
>
> **성격 명시(Methods)**: δ_det는 **재실행 잡음이 아니라 표본 변동**을 재는 양이므로 GNN의 δ와 종류가 다르다. 따라서 backbone 간 동등성 판정을 직접 비교할 수 없다. GNN δ를 결정론 tier에 차용한 결과는 민감도로 병기한다.

**§4δ_det 구조적 관대함 (합성 측정·결과 확인 전 기록, 2026-08-04).** δ_det는 단일 arm의 블록 부트스트랩 SD인 반면, Δ의 신뢰구간은 **동일한 화합물 표본을 공유하는 두 arm의 차이**에서 산출된다. 표본 추출 충격이 두 arm에 공통으로 작용해 차이에서 상쇄되므로 CI 폭은 δ_det보다 구조적으로 작다. 이는 δ_det의 특정 값이 아니라 "마진은 한 arm, CI는 상관된 두 arm"이라는 **구조**에서 비롯된다. **합성 측정**(실제 discovery_group strata 구조 2,029행·439화합물 + 합성 pred/true, 상관 구조; `scripts/measure_structural_leniency.py`) 결과: 두 경로 모두 검출 임계 ≈ **1.0×margin**(결정론 = **1.0×δ_det**, GNN = **1.0×δ**) — 상관된 paired CI가 margin보다 훨씬 좁아(예 k=0.5: 결정론 CI폭 ~0.02 ≪ δ_det 0.087; GNN CI폭 ~0.001 ≪ δ 0.0198) 전환이 dd≈margin에서 급격히 일어난다. 관대함의 실체는 **절대 마진 크기**: δ_det(0.087) ≈ **4.4×** δ(0.0198)이므로 결정론 tier의 `동등`은 최대 ~4.4× 큰 실제 차이까지 허용한다. **따라서 결정론 tier의 `동등`은 GNN 경로의 `동등`보다 약한 증거이며, 두 판정을 같은 강도로 합산하지 않는다.** GNN δ를 차용한 민감도가 결정론 tier의 실질적 점검이 된다. 이 기록은 **어떤 tier 비교도 수행되기 전**에 작성됐다.

**⚠ 실측 주석(pooling 집합 크기, 2026-08-04)**: LightGBM 사다리의 main tier는 **{t0, t1′, t2, t3a, t3b, t4} = 6개**(t1=additive bias는 LightGBM에 없음 — naive·GNN 전용이고 naive는 제외). rank절단(genus/genusfamily)은 exploratory이므로 미포함. ⟹ pooling C=**6**(§1 표의 7-tier 중 t1 부재). director 승인의 "7/7"과 상이 → **동결 director 재확인 대기**(값·조건별 s_c는 산출·보고).

## §4F. fixed_proj 용량 통제 구성 (확정)
- **구성(가)**: **종별 고정 랜덤 벡터(dim=species_emb_dim=16)** — Tier 4 fixed_proj와 **동일 구성**(frozen random 테이블 종-인덱스 조회). 변형 `true_species_fixed_proj`(기존)가 그대로 tier 3a·3b·4의 **공유 용량 baseline**(셋 다 readout에 16-d concat). 실데이터 스모크 OK(dmpnn 1.61/graphconv 1.73).
- **배제**: rank별 고정 랜덤 임베딩 — 같은 family 종이 같은 벡터 → **계통 구조 잔존**("frozen taxonomy"이지 용량 통제 아님). fixed_proj는 "차원 vs 학습 내용"을 가르는 통제이므로 구조 보존 구성 부적합.
- **shuffled 중복 유지**: taxonomy에서 fixed_proj⊃shuffled 겹치나 유지 — shuffled=구조 주변분포 유지·배정 파괴, fixed_proj=구조 자체 없음. **두 독립 각도 일치 시 용량 반박 이중 차단**(상호 보강). 범위 = 3a·3b·4(tier2 제외·1/1′ N/A·cold 미적용).

## §6. 실행 중단 규칙 (상식 범위 조작적 정의) — **파티션별**

RMSE는 restore 후 지표라 mean-prediction ≈ **residualized SD**가 기준. 파티션별로 고정(실행 중 변경 금지):

| 파티션 | residualized 타깃 SD | 상한(즉시중단): tier0 RMSE ≥ | 하한(누출의심 보고·중단아님): RMSE < |
|---|---|---|---|
| **discovery** (LC50@96 단일 stratum, 잔차화=항등) | **1.7195** | **1.7195** | **0.8598** |
| **replication** (7 strata; 잔차화 활성) | **1.7094** | **1.7094** | **0.8547** |

- **적용 대상 명시**: 상한은 **각 파티션의 tier 0** 완료 시 그 파티션 기준으로 적용. 하한은 각 파티션 각 tier 완료 시 그 파티션 기준.
- ⚠ **실측 정정(서사 미조정) + 검증**: endpoint/duration 잔차화가 replication 분산을 **0.5%만 제거**(원 1.7139 → 잔차 1.7094). **계산 검증됨**: between-stratum var/total = **0.52%** = 제거량과 정확히 일치(계산 오류 아님). 7 strata 평균 pLC50는 4.744~5.121(**span 0.377 log**, grand 4.861 대비 ±0.26 이내) — 방향은 예상대로(EC50·장시간 → 높은 pLC50)이나 marginal 크기가 compound 구성 차이에 교락(strata별 화합물 집합 상이, n_compounds 468~1,518). within-stratum SD ~1.7 대비 between span 0.38이 작아 0.5%.
  - **Methods 확정 문구**: "endpoint/duration 잔차화는 각 stratum의 **주변(marginal) 오프셋**을 제거한다. strata별 화합물 수가 크게 다르므로(468–1,518) 관측된 marginal 차(span 0.377 log)는 화합물 구성과 교락되어 있고, 화합물 보정 후 진짜 endpoint/duration 효과는 이보다 클 수 있다. 우리 절차는 보정되지 않은 주변 오프셋만 제거하므로 잔여 endpoint/duration 효과가 남을 수 있으나, 이는 **전 tier·전 backbone에 동일하게 적용**되어 공유되므로 tier 간 비교에 편향을 주지 않는다." (⚠ "endpoint/duration 효과를 제거했다"고만 쓰면 과장 → 위 문구로.)
  - Supplementary 표 = `data/replication_strata_toxicity.csv`: 방향성(EC50·장시간 → 고독성)이 생물학적으로 일관 → **데이터 자체 sanity check**. 기준은 파티션별 residualized SD로 분리 고정(원리 준수).
- 스모크엔 미적용(과소학습 정상; baseline 대조는 본 학습에서만).

## §5. 원고 규칙 (수치 갱신 시)

- **5-1** 핸드오프 §2 "상방(계통거리·DEB를 성능 tier로 → 가장 비싼 표현조차 무승부 직접 증명)" 서술 **삭제**(tier 5·6 제외로 무효). ADORE 전환 정당화 = 라이선스·data availability 하나로 좁힘.
- **5-2** Limitations 하이브리드 논점: 커버 종에서 계통·형질이 더 나은지 미평가 → 가용 종만 그 표현·나머지 대체하는 하이브리드가 one-hot보다 우수할 가능성 배제 안 됨. 단 하이브리드도 종 58.5%(phylo)·81.2%(trait)에서 대체 의존, 조류 형질 전무.
- **5-3** "재현" 금지 → **"endpoint 일반화"**(discovery/replication 화합물·종 공유; 중복률 병기).
- **5-4** "커버리지 = 표현 × 자원 속성"(ADORE taxonomy 100% → 정보성 표현 best case, 여기서조차 tier5·6 붕괴가 논지).
- **5-5** 커버리지 표 = 종·레코드 수준 병기 + 밀도비 정식 열(phylo 4.3×/DEB-any 9.33×), 비용사다리와 별개.
- **5-6** 인용 금지: "taxonomy ~72%"(ADORE native 100%)·"완전벡터 94종"(실제 83)·"화합물 9,404"(실제 4,702)·"계통거리 67%"(실제 종 41.5%).
- **5-8 OOV Limitations**: 평균 임베딩 매핑은 **학습된 OOV fallback이 아님**. 종 드롭아웃으로 OOV 토큰을 훈련하면 모델이 "미관측 종" 상태 자체를 표현하나, 본 매핑은 훈련 표현들의 중심만 사용. 따라서 미관측 종에 대한 one-hot·학습 임베딩 성능은 OOV 토큰 명시 학습 대비 **보수적으로 추정**될 수 있음.
- **5-7 비용 사다리 서사(NCBI 96.5% 반영)**: taxonomy는 커버리지·비용 장벽 실질 없음(ADORE 내장 100%, 자력 해상 **4-rank 96.5%종/98.3%레코드**; 비용 = 종 3.5% 손실 + 파이프라인 부담). ⟹ **처방 두 층**: taxonomy = **성능 무승부**(쓸 수 있으나 이득 없음), 계통거리·DEB = **커버리지 붕괴**(41.5%/18.8%, 조류 0%; 쓰고 싶어도 못 씀). tier 5·6 성능제외 결정과 정합(배제 사유가 커버리지 vs 성능). ⚠ **핸드오프 헤드라인 "정보량↑→커버리지 급감"은 ADORE taxonomy엔 미성립**(급감은 phylo/DEB만) — 헤드라인 확정은 성능 결과 후 원고 단계, 지금은 이 사실만 기록. NCBI class `Actinopteri` vs native `Actinopterygii` = 3a/3b DB출처 순수 대비 예시(Methods).

## §7. cross-group Phase 1 제외 (구조적 판별 불가) + 외삽 범위 축

- **결정**: cross-group(그룹-cold)을 **Phase 1 실행 제외**(Phase 2 이월). primary = warm 우월성 + warm TOST + **종-cold**. 블록 B = 종-cold만(`t0→2→3a→3b→4`).
- **근거(사전등록 문구)**: cross-group을 Phase 1에서 평가하지 않는 것은 **구조적 판별 불가**이지 결과 회피가 아니다. 4-rank taxonomy·종 인덱스 표현(one-hot·학습 임베딩)은 모두 훈련 라벨 공간 밖에서 정의되지 않으므로, cross-group에서는 비교 대상 전부가 동일하게 미학습이 된다. tier 5·6을 커버리지 붕괴로 제외한 것과 같은 성격.
  1. **실측(GPU 0)**: fish/crusta/algae **어느 쌍도 4-rank 공유 랭크 0**(class/order/family/genus, native·NCBI 모두) — `data/crossgroup_rank_overlap.csv`(표1). §A-1에서 kingdom/phylum 제외(tax_group 3값 축퇴) → 세 군 잇는 공통 라벨 없음.
  2. **대조(표2)**: 종-cold held-out 종의 랭크가 train에 존재 = class 98.7/order 96.8/family 86.5/genus 52.6% → **같은 군 내 전이, 군 넘으면 불가**.
  3. tier 2·4도 미학습(OOV remap 없음) → cross-group은 전 표현 동등 무력, 판별력 0. 핸드오프 §3-2가 cross-group을 tier 5·6(연속 계통거리·형질 = 라벨 공간 밖에서도 정의) 전용으로 한 것이 옳았음.
- **kingdom/phylum 재도입 안 함**(절편 하나, 정보 0; 결과 위해 설계 되돌리기 배제; §A-1 유지).
- **원고**: "4-rank taxonomy는 어느 쌍도 공유 랭크 0 → 이산 분류 표현은 라벨 공간 밖으로 외삽 안 됨. 성능 검정 아닌 구조적 성질, 무학습 데이터 확인." **처방 세 번째 축 = 외삽 범위**(커버리지·비용에 추가; taxonomy는 커버리지 100%·저비용이나 라벨 공간 못 벗어남).

## §8. OOV = 평가 시점 매핑 (D-OOV 확정, 훈련 불변)

- **종-cold 평가에서 미관측 종(및 미학습 taxonomy 랭크)의 표현은 평가 시점에 훈련 종·훈련 카테고리의 임베딩 평균으로 매핑한다(주 설정).** 미학습 행 그대로·no-species 붕괴를 민감도로 병기. **세 처리 모두 평가 시점 매핑 = 훈련 불변**(블록 A/warm 무손상, 게이트 "같은 비교" 유지).

| 처리 | 내용 | 위상 | 라벨 |
|---|---|---|---|
| **평균 임베딩** | cold 종 → 그 run **train split에 등장한 종들의 임베딩 평균** | **주** | `oov=mean` |
| 미학습 행 | 초기화·미학습 자기 행(현행) | 민감도 | `oov=untrained` |
| 붕괴 | 종 경로 기여 0 | 민감도 | `oov=collapse` |

- **taxonomy에도 동일**(§1-2 대칭): 미학습 랭크(종-cold genus 47% OOV: class 98.7/order 96.8/family 86.5/**genus 52.6%**) → 그 rank에서 **train 등장 카테고리 임베딩 평균**. 미적용 시 tier 2·4만 깨끗한 fallback·taxonomy는 잡음 = 부당 비대칭(원칙 위반).
- **구현**: 평균 = **각 run train split의 실제 등장 종/카테고리** 기준(전체데이터 아님). 재적합 아닌 평가시점 고정 계산. tier 2(one-hot) 평균 = 학습 종 가중치 벡터 평균(tier 4와 동일 논리). 블록 B 범위 {t0,t2,t3a,t3b,t4}(tier 1·1′ 해당 없음).
- **근거(기록)**: 무작위 벡터 = 정보부재 아닌 잡음 주입(구현 세부가 결과 결정); "반박 tier에 최선 조건"(tier5·6 원칙) 대칭; 세 처리 민감도가 매핑 임의성 반박 차단. 배제: OOV 토큰 학습(훈련변경→warm 오염·블록A 폐기, 정합성 이유로 열등)·최근접(순환)·영벡터(원점≠중심).
- ⚠ **결과 관찰 후 매핑 변경 금지**(주=평균 임베딩 지금 고정).

## §9. 일정 재산정 (cross-group 제외)
- Plan A GPU run = warm 4,480 + 종-cold 360 + rank절단 40 = **4,880**(구 9,240 대비 −4,360). wall **as-is 21.1일 / 방법 B 17.4일**. LightGBM·naive = CPU 별도. 파티션 임계 불변. OOV 구현 시 종-cold 소폭 증가(pending).

*(확정 2026-07-30. cross-group Phase 1 제외. 블록 A 학습 착수 순.)*

---

## §R. 개정 이력 (명세 완비성 추가분; 전부 결과 확인 이전)

> 목적: 모든 명세 시점을 한 곳에 기록하고 각각이 **결과 확인 이전**임을 증명. "결과 미확인 증거" = 각 시점 실측 사실(추정 아님).

| 항목 | 추가·변경 내용 요약 | 타임스탬프 | 결과 미확인 증거 | 사유 |
|---|---|---|---|---|
| §4G-7 | TOST α=0.05단측×2=Δ 90% CI · TOST기각∧NHST기각→`동등` · (2)Δ정의 미확정 escalate | 2026-08-03 (δ 동결 前) | δ 미동결(`delta_primary_frozen.json` 부재) · `run_q2_gatekeeping.py` 부재 · `runs/bootstrap` 산출물 파일 0 · dd/TOST/게이트 출력 0 | 판정 규칙 미명시 지점을 결과 전 고정 |
| §4δ′ | 앙상블 척도 δ′(민감도 전용, k=10, +90시드/조건=1,260 run) · 주 척도=per-seed 확정 | 2026-08-03 (δ 동결 前) | 〃 (δ·δ′ 미동결 · 비교 출력 0) | D-ΔδMATCH: 주=per-seed, 앙상블=민감도 |
| §4Δ | Δ per-seed paired 정의 · 부트스트랩 공통 sset · Methods 문구 | 2026-08-03 (δ 동결 直前) | δ 미동결 · 비교 출력 0 | §B=paired 확정 반영 |
| §4δ-break | δ 동결 파기 규약 (a~e) | 2026-08-03 (δ 동결과 동시) | δ 동결됨(값 0.019777) · `run_q2_gatekeeping.py` 부재 · `runs/bootstrap` 산출물 0 · dd/TOST/게이트 출력 0 | 무효화 사건 前 규칙 고정 |
| §4δ-impl | 동결 δ=0.019777 함의(§8 위험 해소 · `불확정` 다수 가능·warm 포함) | 2026-08-03 (δ 동결 後) | δ 동결됨 · 비교 출력 0 | 결과 前 함의 기록(순환 아님) |
| §4δ "판정 자유도 문구" | "남은 분석 자유도=0" 절대표현 폐기 → 검증가능 문장 교체(pre-reg 3곳: §4δ·§4G-7·§4δ′) | 2026-08-03 (δ 동결 後) | δ 동결됨 · 비교 출력 0 | 절대주장은 반증가능 → 참·검증가능 문장으로 |

**주 δ 동결 기록**: `audit/delta_primary_frozen.json`, **δ = 0.019777**(C=14, df=126), frozen 2026-08-03. ⚠ 동결 직후 증거-필드 기록 버그 정정(`bootstrap_outputs_exist`가 빈 placeholder 디렉터리 존재를 True로 기록 → **파일 수 0**으로 수정). **δ 값 불변**(0.019776561636, v1=v2 확인), 구본 `delta_primary_frozen_v1_evidencebug_20260803.json` 보존. 정정 시점에도 비교 출력 0(사건 전). **정정 방향 = 방어에 유리**: 오기록(exists=True)이 오히려 "비교 산출물 존재"로 오독될 여지였고, 정정본(파일 0)은 "동결 전 비교 미수행"을 **더 강하게** 입증. 구본 보존 + 빈 디렉터리 실측이 그 방어 근거.
- **§4C 비교 집합 확정**(2026-08-03, δ 동결 後): primary TOST = {t3a·t3b·t4 vs t2} within-backbone. 증거 = δ 동결·비교출력 0. 사유 = §3 모호점 director 배정.

**§R 추가 (2026-08-04~05, 전부 결과 확인 전; 증거 = 비교 산출물 0 + 하드가드 REFUSE, 스크립트 존재 아님):**

| 항목 | 요약 | 타임스탬프 | 사유 |
|---|---|---|---|
| §4δ_det | 결정론 tier 마진 δ_det=√(mean s_c²), block bootstrap SD, LightGBM main, naive 제외 | 2026-08-04 | 결정론 tier 게이트에 GNN δ 부적합(표본변동 vs 재실행잡음) |
| §4δ_det 블록키 | "compound_key"→**smiles**(분자입력 상관; 전 파이프라인 통일) | 2026-08-04 | 동결 전 문서-따라가기 정정, 통계 변경 아님 |
| §4δ_det 관대함 | 합성측정: 검출임계 ~1×margin 양경로, δ_det≈4.4×δ → 결정론 동등=약한 증거 | 2026-08-04 | 구조적 관대함 결과 전 기록 |
| §4δ_det C=6 확정 | LightGBM main tier 6개(t1=additive bias 부재, naive 전용). **"7"은 director 기대치였고 사양 아님** | 2026-08-05 | backbone×tier 가용성 사실; "전 tier" 충실 적용 |
| δ_det 동결 | δ_det=0.087189, C=6 | 2026-08-05 | §4δ_det 사양대로 |

**세 마진 동결 완료**: δ=0.019777(per-seed) · δ′=0.005717(앙상블 민감도) · δ_det=0.087189(결정론). 전부 비교 산출물 0 시점 동결, §4δ-break 파기규약 공통 적용.

---

## §R-posthoc. 사후 분석 지정 (⚠ 언블라인딩 이후 · 결과 확인 **후**)

> ⚠ **위상 경고**: 이 항목은 위 모든 §R 항목과 근본적으로 다르다. 위 항목은 전부 **결과 확인 이전**(비교 산출물 0)이나, 본 항목은 `gatekeeping_results.json` 언블라인딩(2026-08-05) **이후** director가 지정한 **사후 분석**이다. 사전등록 판정과 **동일 지위 아님**. 원고에서 **"post-hoc"으로 라벨링**하며 사전등록 검정과 같은 근거력으로 쓰지 않는다.

| 항목 | 요약 | 타임스탬프 | 위상 |
|---|---|---|---|
| post-hoc t2 vs t1 | cand=t2(one-hot) vs ref=t1(additive bias) within-backbone DD, **4건**(discovery/replication × dmpnn/graphconv). per-seed paired · block=smiles · 2000 boot · seed×block 공통 sset · 90% CI · 동결 δ=0.019777(파일 read) · 3범주+네 번째 칸. **BH-FDR = 이 4건 내부만**(기존 패밀리 미혼합). | 2026-08-05 (언블라인딩 **後**) | **사후 분석 (post-hoc)** |

- **언제·왜 지정**: primary·confirmatory **우월성 결과(dd)를 본 뒤** director가 t2 vs t1 쌍을 지정(사전등록 §4C 비교집합에 없던 쌍). 사유 = 우월성 값에서 유도한 t2−t1 간격(discovery 0.008·0.018, replication 0.087·0.080 = `dd(t2>t0)−dd(t1>t0)`)은 점추정뿐 불확실성이 없어, 그 간격에 **직접 CI·검정을 붙이기 위해**.
- **격리(director §0)**: 기존 패밀리(primary·confirmatory·exploratory·sensitivity_ensemble) 어디에도 미투입. **별도 post-hoc 세트**. 기존 산출물 `gatekeeping_results.json`·`REPORT_2-0_to_2-9.txt` **불변**. 신규 파일에만 기록: `runs/bootstrap/posthoc_t2_vs_t1_results.json` · `runs/bootstrap/POSTHOC_t2_vs_t1_REPORT.txt`. 스크립트 `scripts/run_posthoc_t2_vs_t1.py`(동결 파이프라인의 load_arm/run_comparison/δ 로더 재사용 → 절차 동일).
- **산술 정합성(director §3)**: 직접 t2-vs-t1 DD = 유도값 `dd(t2>t0)−dd(t1>t0)`와 **4건 전부 일치**(차이 ≤ 1.39e-17, 기계정밀도). 구조: 우월성은 4항 DD 일반형에서 candbase=ref=refbase=t0로 2항(RMSE(cand)−RMSE(t0))으로 축약되어 있고, 두 우월성 dd의 차에서 공유 t0가 상쇄(mean_s는 선형) → mean_s[RMSE(t2,s)−RMSE(t1,s)] = 직접 4항 DD. t0·t1·t2 행집합 동일(n=2029 disc / 4351 rep)이라 점추정 일치. **차이는 CI에만 있음**(유도값은 CI 없음, 직접계산이 부여).
