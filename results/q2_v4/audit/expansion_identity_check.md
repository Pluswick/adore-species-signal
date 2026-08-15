# 확장 실험 §0 동일성 대조표 (코드 원문 근거). 성능 수치 보기 전 작성.

Phase 1 실제 설정을 코드에서 읽어 확장(B1 단독)과 항목별 대조. **데이터(B1_final) 외 차이는 불일치**로 본다.
분류: **동일** / **불가피**(데이터가 달라 물리적으로 같게 못하나 절차는 동일) / **판단필요**(즉시 중단·보고).

| 항목 | Phase 1 (파일:줄) | 확장 실험 | 판정 |
|---|---|---|---|
| hidden / depth / dropout / emb / val_frac | 300 / 3 / 0.1 / 16 / 0.1 (`run_q2_gnn_ladder.py:85-88`, `runner.py:36-46`) | 동일 값 재사용 | **동일** |
| batch / lr / weight_decay / epochs | 256 / 5e-4 / 1e-5 / 100 (`run_q2_gnn_ladder.py:85`) | 동일 | **동일** |
| optimizer / loss | AdamW (`runner.py:388`) / MSELoss (`:389`) | 동일 | **동일** |
| scheduler | 없음 (`runner.py` LR 스케줄러 부재) | 없음 | **동일** |
| early stopping | patience **15**, min_delta **1e-4**, best-state 복원 (`runner.py:393,407,415`) | 동일 | **동일** |
| valid carve | 무작위 permutation, val_frac 0.1, seed 기반 (`runner.py:58-65,436`) | 동일 | **동일** |
| tier 구현 t0/t1/t1′/t2/t3a/t3b/t4 | `models.py` `model_spec_from_variant`/`build_v3_model` (읽음) | 동일 코드 | **동일** |
| shuffled 통제 | species_idx **행별 permutation**(marginal 보존·쌍 파괴), seed=cfg.seed+101/202/303, `_ensure_not_identity` (`species_controls.py:117-130`, `runner.py:439-459`) | 동일 코드·동일 seed 규칙 | **동일** |
| endpoint/duration 잔차화 | train 추정 가산 주효과 제거→예측 복원 (`runner.py:466-478,535`); 단일 stratum=항등 | 동일 코드 (B1 혼합→활성, LC50@96h 부분집합→항등) | **동일** |
| 앙상블 k=10 | canonical seed 0–9 (prereg §4δ′) | 동일 | **동일** |
| Δ (4-arm DD, per-seed paired) | `jcim_v3/gatekeeping.py` `paired_dd_bootstrap`/`_dd_point` (읽음) | 동일 | **동일** |
| TOST α·CI·판정 | α 0.05단측×2=90% CI, 3범주+4번째칸 (`gatekeeping.py:15-32`) · δ=0.019777 동결 | 동일 (δ 동일 재사용) | **동일** |
| seed 목록 | `range(10)` (`run_q2_gnn_ladder.py:51`) | 동일 | **동일** |
| **split 절차** | compound(CAS)-disjoint group · Bemis-Murcko scaffold · generic scaffold (`build_adore_datasets.py`, 고정 seed) | **B1 데이터로 동일 절차·동일 seed로 재생성** | **불가피**(새 데이터) |
| **species_idx 맵** | 전역 vocab(train+test) → idx (`runner.py:431,250-273`) | B1 1,975종으로 동일 방식 재생성 | **불가피** |
| **t3a native taxonomy** | ADORE native 4랭크 class/order/family/genus (`tier_input_reference.json`) | ECOTOX `species.txt` 동일 4랭크로 구성 (일치 확인 예정) | **불가피**(새 종) |
| **t3b NCBI taxonomy** | NCBI 해상 ncbi_class/order/family/genus (taxdump) | B1 종 대상 **동일 NCBI 해상기 재실행**; 미해상=`__unknown__` 버킷(`runner.py:504`)+tier-input guard | **불가피**(새 종 재해상) |
| **n_species (one-hot 차원)** | 779(disc)/1006(repl) | 1,975(B1) | **불가피**(director 명시) |
| **warm OOV 종 처리** | 전역 species_idx로 인덱스 유효, **미학습 임베딩 그대로 사용**(제외·매핑 없음) (`runner.py:431`; §8 매핑은 종-cold 전용) | 동일 코드·동일 처리. 단 B1 group split은 test 종 ~20% OOV(Phase1 warm ≈0) | **불가피**(처리 동일, 유병률만 데이터 차이) |

## 판정 요약
- **판단필요(설계) 불일치 = 없음.** 전 항목 동일 또는 불가피(데이터 유래·절차 동일).
- 불가피 항목은 전부 "새 데이터라 재생성하되 Phase 1 절차·seed 그대로"에 해당.
- ⚠ 특기(불가피): B1 warm은 test 종 ~**20% OOV**(밀도확인 seed평균 19.9%). 처리는 Phase 1 코드와 동일(미학습 임베딩 그대로)이나 t2/t4 warm 평가가 미학습 표현에 더 노출됨 → **불확정 증가 가능**(prereg 사전기록대로 실패 아님).

## 실행 전 필요한 빌드 (Phase 1 절차 이식; 성능 무관)
1. B1_final(41,527행)을 파이프라인 포맷으로: smiles(CAS→SMILES)·species·species_idx·endpoint·duration·target_log10(−log10 mol/L)·native/NCBI taxonomy.
2. species_idx 맵(1,975종) · native tax(species.txt) · **NCBI 재해상**(taxdump) · 3 split 생성(동일 seed).
3. **t4 누출 재검증**(순열 검정 + OOF 경계 증명) B1에서 재실행 — 미통과 시 중단.
4. 가드: tier_input_guard·prediction_io·io_atomic 활성 + 13 arm 입력 축퇴 검사.

## 실행 규모 (사실)
- GNN 13 arm × 2 backbone × 10 seed × 3 split = **780 run** + 결정론 78 = **858 run**.
- 4090 단독. batch 등 학습 하이퍼파라미터 고정(변경 금지); 속도는 **동시 run 수로만** 조정.
- 소요: GNN ~수 분/run × 780 → 병렬도에 따라 수십 시간(다수 시간~1–2일). GPU(4090) 인식 확인 필요.

## 제외(명시) — primary 패밀리 한정
designed-leaky split · 종-cold arm · δ′ 90-seed 민감도 · exploratory 6종 **제외**. 본 재현 = **primary 패밀리 한정**.
