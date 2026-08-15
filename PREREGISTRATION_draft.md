# ADORE Q2 — 사전등록 초안 (DRAFT, director 확정 대기)

> 순서 동결 step 2. 이 문서는 **tier 성능을 보기 전** 규칙을 고정한다. 확정 후 데이터 재구축(step 3)에 적용.
> 근거 수치는 `results/q2_v4/audit/GAP_EXECUTION_LOG.md` Session 1–7. 확정 전 학습 착수 금지.

## §1. 값 유효 범위 임계 규칙 (데이터셋 수준, 훈련·평가 공통)

**규칙(D16 3단 경로 — 전역 상한 폐기)**: 전역 `pLC50<12`는 값 구간 겹침(Endrin 오류 12.0–14.8 vs deltamethrin 실재 12.2)으로 분리 불가 → **폐기**. 아래 순서로 행별 판별, 행마다 `exclusion_reason` 기록.

- **하한(확정) = 물 몰농도 가드레일**: pLC50 > **−1.75** (순수 물 55.5 M; 수용액 초과 불가). 판단 무개입. 현재 **0행 제외**. ※"용해도 초과" 근거 폐기(제외 20행이 miscible 용매/염 ethanol·methanol·acetone·DMSO·ethylene glycol·NaCl → LC50≥1M 실재 정상; 보존).
- **경로 1(원본 단위 정합) = 막힘**: raw에 `conc1_unit` 없음(conc2/3만) → 원 단위 미보존, 변환식(mol×MW) 정확 검증됨 → 정합검사 무효. 경로 2로.
- **경로 2(질량농도 검출하한 — 상한 대체)**: mass = conc_mol×MW×1e9 [ng/L]. **mass < 0.1 ng/L 제외**. 근거: 수중 농약 LC-MS/MS MDL **0.01–1.32 ng/L**, LOQ ~0.1 ng/L↓(문헌) → 이하 LC50 측정불가. 현재 **10행 제외**(전부 crusta), deltamethrin 0.32 **보존**. 후보 플래그: 0.05→8 / **0.1→10** / 0.5→24 / 1.0→33. (0.5는 deltamethrin 클립 → 부적합.)
- **경로 3(겹침 개별검토)**: ≥0.1이나 pLC50≥12인 6행 문헌 대조 — **보존**: deltamethrin(0.32)·pyrethroid 76703-62-3(0.42)(문헌 2.6–68 ng/L, ~1자릿수 내). **제외(source_error)**: Endrin(72-20-8, 0.13–0.28; 문헌 88–352 µg/L)·Methyl parathion(298-00-0, 0.14). **4행**.
- **결과**: 초안 제외 **14행**(검출한계10 + path3 4), **pyrethroid 2행 보존**(구 규칙 16행 전부 제외 대비 실재 2행 구제). 목록 = `results/q2_v4/data/_ext/prereg_excluded_DRAFT_D16.csv`(reason 컬럼). **임계 0.1·path3 판정 = director 확정 대기.** pLC50 9–12(1,289행) 중 <0.1 ng/L = 0행(강독 꼬리 무손상).

**적용·기록**
- 제외 = 훈련·평가 양쪽(데이터셋 수준).
- 제외 목록(행·종·CAS·tax_group·커버층) → `results/q2_v4/data/_ext/prereg_excluded_rows.csv` (36행: 저독성 20 + 고독성 16; crusta 27/fish 7/algae 2; 13 CAS). Supplementary·리뷰 검증용.
- **비무작위 손실 Limitations 명기**: 제외가 갑각류·살충제(고독성)와 저독성 용매/염(저독성)에 몰림.

## §2. 파생·근사값 플래그 민감도 (평가 단계, run 0)

- ADORE `result_conc1_mean_op`: **3.5%가 직접 점추정 아님** — `min_max_average` 2,001 + `~` 476 (직접=nan 68,193).
- **사전지정 민감도**: `result_conc1_mean_op == nan`(직접 측정) 부분집합에서 전 지표를 **test 집계만 분리 재산출**(같은 모델·같은 훈련, 추가 run 0). 훈련에선 미제외.
- 주 결과=전체, 이 부분집합=품질 민감도로 병기. 목적: 값 출처 품질(직접 vs min-max midpoint vs 근사)이 결론에 영향 주는지.

## §4. taxonomy rank 절단 실험 (tier 3a, 채택)

- **목적**: taxonomy=one-hot 무승부의 *기전* 규명. genus만으로 유지되면 "taxonomy 예측력 = genus 수준 종 identity"로 관찰→설명 격상.
- **조건 3**: `genus만` / `genus+family` / `4-rank 전부`(=기존 tier 3a 재사용) → **신규 2조건**.
- **범위**: tier 3a **only**, **warm·discovery·group split** 한정(cold/cross-group/replication/scaffold/leaky 미적용).
- **통제**: `main`만(정보-vs-용량 아님 → 통제축 미곱).
- **backbone**: 3종 전부(D-MPNN·GraphConv·LightGBM). 비용은 §L-2 별도 항목.
- **tier 3b(NCBI)**: NCBI 해상률 산출 후 별도 결정(미적용).
- seed: GNN 10 / 결정론 1.

## §5. fixed_proj 용량매칭 통제 범위

- **적용**: Tier **3a·3b·4·5·6**(전 벡터 tier). 각 tier의 학습/외부 벡터를 frozen random 동차원으로 대체 → "용량 vs 정보" 분리.
  - **Tier 6(DEB)**: 광폭/협폭 변형은 지시자 포함 최종 차원이 다르므로 **각 변형별 개별 매칭**(D9 누락 정정 — 포함 확정).
- **제외**: Tier **2(one-hot)** — readout 폭 = 종 수(≈1,267)라 매칭 대상 용량이 emb_dim이 아님(구조상 부자연). **director 승인됨**.
- **제외**: Tier **1·1′**(스칼라 오프셋 — 매칭할 용량 없음, 구조상 N/A).
- **cold에서는 미적용**(4차 정정 범위).

## §0. 확증편향 방지 (사전 약속)

- ADORE에서 구조적 결론(무승부·커버리지·편중)이 **재현 실패해도 그대로 보고**한다.
- "오히려 강화될 여지" 같은 **검정 전 선취 서술을 결과 판정에 쓰지 않는다.**
- discovery/replication은 화합물·종을 공유 → **"재현" 아님**; 라벨·표·로그에서 **"endpoint 일반화"**로 표기, 중복률 병기.
- tier 5·6 성능은 **커버층/미해상층 분리** 출력(커버층=성능 결론, 전체 희석=커버리지 결론).
- Tier 6 커버층에 조류 0종 → "정보 있는 종에서조차 무승부" 주장은 **어류·갑각 한정** 명기.

*(DRAFT — §1 하한 근거 및 D16 회신 후 확정.)*
