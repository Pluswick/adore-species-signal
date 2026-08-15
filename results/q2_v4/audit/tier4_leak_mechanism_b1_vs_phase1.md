# t4 순열 신호 기전 규명 — ① B1 OOF 경계 증명 + ② Phase 1 8 split × 20 perm

목적: b1_scaffold에서 순열 검정이 flag한 t4 신호(improve_shuffled > TAU=0.02)의 **기전**을 가름.
director 지정 2건. 성능 판정 아님 — QC/무결성 진단. 두 검증은 동일 비용(각 20 perm)·결정론.

## ① B1 OOF 경계 증명 (결정론적, 지표-수준 구조 증명) — data_b1 4 split
스크립트: scratchpad/b1_oof_index.py (= verify_tier4_oof_index.py 로직, data_b1로 지정).
`build_factor`의 실제 fold ledger(_ledger_inner/_ledger_outer)를 추출해 집합 연산으로 검증.

| split | inner self_in_source | outer self_in_source | outer self_in_species_source | 각 행 1회(inner/outer) | train/test key 겹침 | unseen→0 factor | ok |
|---|---|---|---|---|---|---|---|
| b1_group | 0 | 0 | 0 | ✓/✓ | 0 | ✓ | PASS |
| b1_scaffold | 0 | 0 | 0 | ✓/✓ | 0 | ✓ | PASS |
| b1_scaffold_generic | 0 | 0 | 0 | ✓/✓ | 0 | ✓ | PASS |
| b1_designed_leaky | 0 | 0 | 0 | ✓/✓ | 0 | ✓ | PASS |

**결론 ①: 지표-수준 구축 누출 없음.** 어떤 train 행의 SVD factor도 자기 라벨로 계산되지 않음(inner 잔차 OOF·outer SVD OOF 양쪽), test 행은 factor 구축에 미참여, unseen 종은 0 factor. → "빌드 결함(fold 배정/인덱스 누출/species_idx 무결성)" 가지 배제.
출력: results/q2_v4/data_b1/_ext/tier4_oof_index_proof_b1.json (verdict=PASS).

## ② Phase 1 8 split × 20 perm 강건성 — results/q2_v4/data
스크립트: scratchpad/ph1_leak_robust.py. PERM base=20260801, NPERM=20, TAU=0.02.
improve_shuffled = rmse_nofactor − rmse_shuffled_factor. mean>TAU면 flag.

| Phase 1 split | mean | std | median | [min, max] | frac>0.02 | mean>TAU |
|---|---|---|---|---|---|---|
| discovery_group | +0.0115 | 0.0107 | +0.0132 | [−0.0073, +0.0326] | 0.25 | False |
| discovery_scaffold | +0.0055 | 0.0117 | +0.0072 | [−0.0122, +0.0303] | 0.05 | False |
| discovery_scaffold_generic | +0.0127 | 0.0158 | +0.0133 | [−0.0307, +0.0398] | 0.30 | False |
| discovery_designed_leaky | −0.0618 | 0.0104 | −0.0624 | [−0.0788, −0.0419] | 0.00 | False |
| replication_group | −0.0046 | 0.0061 | −0.0040 | [−0.0162, +0.0055] | 0.00 | False |
| replication_scaffold | +0.0111 | 0.0173 | +0.0142 | [−0.0259, +0.0345] | 0.35 | False |
| replication_scaffold_generic | +0.0107 | 0.0074 | +0.0122 | [−0.0067, +0.0213] | 0.05 | False |
| replication_designed_leaky | −0.0443 | 0.0097 | −0.0441 | [−0.0638, −0.0271] | 0.00 | False |

**VERDICT (mean-based): PASS** — 8 split 전부 mean < TAU. 최대 mean = +0.0127(disc scaffold_generic).
출력: results/q2_v4/data/_ext/tier4_permutation_leak_robust_phase1.json.

## B1 20-perm(기존) 대조
| B1 split | mean | std | [min,max] | frac>0.02 | mean>TAU |
|---|---|---|---|---|---|
| b1_group | +0.0187 | 0.0083 | [+0.0038,+0.0349] | 0.40 | False |
| **b1_scaffold** | **+0.0520** | 0.0065 | [+0.0415,+0.0635] | **1.00** | **True** |
| b1_scaffold_generic | +0.0186 | 0.0105 | [+0.0008,+0.0359] | 0.50 | False |
| b1_designed_leaky | −0.0390 | 0.0062 | [−0.0512,−0.0293] | 0.00 | False |

## scaffold 직접 대조 (핵심)
- Phase 1 scaffold: disc **+0.0055** / repl **+0.0111** (frac 0.05 / 0.35) — 둘 다 PASS.
- B1 scaffold: **+0.0520** (frac **1.00**) — Phase 1 scaffold의 **약 5–9배**, 20개 순열 전부 TAU 초과.

## 종합 판정
1. **① 구축 무결(no build defect)** — B1 4 split 전부 OOF 경계 PASS.
2. **② t4 공통 아님** — Phase 1 t4 factor는 20 perm에서 8 split 전부 clean(최대 mean +0.013). 순열이 no-factor를 이기는 성질은 t4 보편 현상 아님.
3. ①·②를 합치면: b1_scaffold의 +0.052는 (a) 누출도 아니고 (b) t4 보편도 아님 → **B1-scaffold 분할 고유의 데이터 성질**.

## 기전 해석 (증명 ①·②로 좁혀진 후보)
순열 검정은 train **y만 행단위 치환**하고 species_idx·smiles는 불변 → SVD 행렬
`groupby([sp,cp])[r].mean().unstack(fill_value=0.0)`의 **0-fill 희소/공기(共起) 패턴**은 실제 factor와 동일,
관측 셀의 값만 무작위화. 따라서 라벨-치환 factor도 **종-정체성(측정 패턴·행 support·공유 화합물 블록)**을
여전히 인코딩. b1_scaffold(Murcko)에서 이 종-정체성이 test 독성과 강하게 상관 → 라벨-치환 factor도
no-factor를 이김. 이는 OOF 증명 docstring이 명시한 순열 검정의 **감도 하한(species-identity effect)** —
**누출이 아니라 데이터 성질**. (b1_scaffold_generic은 +0.019로 낮음 → generic이 아닌 Murcko 분할에서만 spike.)

미확정(선택적 추가 진단): 위 "희소-패턴 채널" 가설을 결정론 1회로 직접 확증하려면, b1_scaffold에서
y와 함께 species_idx도 치환(또는 종→factor 배정을 파괴)해 improve가 ~0으로 떨어지는지 확인 가능.
director 미지시 항목이므로 실행 보류(범위 확장 방지).

## ③ 결정적 대조 (director 교정 반영) — 용량 바닥 + test-side 종 오배정, 30 perm
director 지적: "붕괴 바닥은 0이 아니다 — 16 컬럼 추가 자체의 용량 효과가 바닥이고, 붕괴는 재현적이어야 한다."
b1_scaffold(flag) + b1_group(양성 대조). Arm B=순수노이즈 16컬럼(용량 바닥), A=shuffled-correct, C=shuffled+test오배정(derangement; support-size 매칭까지 파괴).

| split | B 용량바닥 | A shuffled-correct | C shuffled-misassigned | A−B(바닥초과) | A−C(종매칭분) | A>C |
|---|---|---|---|---|---|---|
| **b1_scaffold** | **+0.0391**±0.0064 | +0.0493±0.0103 | +0.0467±0.0084 | **+0.0102** | +0.0026 | **0.50** |
| b1_group (대조) | −0.0021±0.0057 | +0.0182±0.0089 | +0.0132±0.0085 | +0.0203 | +0.0050 | 0.767 |

**결론 ③ (director 가설 확증):** b1_scaffold의 flag(+0.052)은 **약 79%가 순수 용량 바닥(+0.039)** — 즉 이 split에선 *아무* 16 컬럼을 더해도 test RMSE가 +0.039 개선됨. 바닥 초과분은 +0.010(<TAU)뿐이고, 그마저 **종 매칭과 무관**(A≈C, A>C=0.50=동전). 순열 검정의 암묵적 귀무(랜덤라벨 factor improve≈0)가 **b1_scaffold에서 위배**됨. → flag는 **오탐(귀무 오설정/용량 바닥)**, 누출 아님.

양성 대조 b1_group: 바닥≈0(노이즈 무익, 기계 정상), A가 바닥 위 +0.020(3.6σ)이고 종매칭 존재(A>C=0.77) → 방법이 "종매칭 있으면 검출/바닥 정확측정"함을 검증.

**바닥 보정 순열값**(shuffled vs 용량바닥): b1_scaffold **+0.0102 (<TAU → PASS)** / b1_group +0.0203(경계, 양성 종-정체성). 귀무를 바르게 잡으면 b1_scaffold는 통과.
출력: results/q2_v4/data_b1/tier4_decisive_control_b1.json.

## §2.4 최종 판정 (①+②+③)
- ① 구축 무결(누출 아님) · ② t4 공통 아님 · ③ flag의 79%=용량바닥, 종매칭 무의미 → **b1_scaffold flag = 특성 규명된 오탐. t4 on b1_scaffold 누출 없음 → 진행.**
- Phase 1 게이트 기준은 그대로 유지(동일성). 이 flag의 처분만 기전 증명과 함께 기록.

## ⚠ 하위 caveat (중단 아님, 분석단계 반영 필요) — director 판단 요청
b1_scaffold는 **용량 바닥 +0.039**가 존재 → 이 split에선 피처를 더하는 *모든* tier(t1~t4)가 바닥 이득을 업음.
따라서 b1_scaffold의 "종 신호" 해석은 **바닥 보정**이 필요(Phase 1은 바닥≈0라 무관했음). 원인 추정: scaffold 분할의 화합물 피처 과적합을 여분 컬럼이 완화(정규화). 제안: gatekeeping/분석 단계에서 b1_scaffold(및 필요 split)에 **노이즈-바닥 기준선**을 tier 비교에 포함. 코드 결함 아님(①), 데이터 유래 성질 — 진행하되 분석에서 계정.

## §2.4 게이트에 대한 함의 — director 판단 사항
- 사전등록 순열 기준(literal): rmse_shuffled < rmse_nofactor − 0.02 ⟹ LEAK ⟹ 중단. b1_scaffold는 **flag**.
- 그러나 ①이 그 기준이 겨냥한 실패양식(자기 행 peeking/구축 누출)의 **부재를 증명**. 검정이 자기 confound(종-정체성 감도 하한)에서 발화.
- 두 읽기: (A) 규칙 문자대로 b1_scaffold t4 중단/제외 vs (B) 검정 목적(구축 누출 탐지)은 ①로 충족, >TAU는 데이터 성질이므로 t4 유효·오탐. → **director 결정**.
- §10 중단조건 대비: ①은 코드 결함 없음을 증명 → "data/code 무결성 결함"에 해당하지 않을 소지. 단 사전등록 문자 기준과 충돌하므로 스스로 진행하지 않고 보고·대기.
