# FDR 패밀리 경계 — 확정 (Phase 1)

> director 확정(3패밀리). 각 패밀리에 BH-FDR **독립** 적용 + global BH-FDR도 산출해 Supplementary 병기(주 결과 = 분리 패밀리). 확정 2026-07-30(학습 전).

## 3패밀리

| 패밀리 | 내용 |
|---|---|
| **Primary** | warm 우월성(종 vs no-species/shuffled) + warm 동등성 TOST(taxonomy vs one-hot, embedding vs one-hot) + **cross-group** — group split · discovery |
| **Confirmatory** | replication(24/48/72h + EC50)에서 **동일 가설·동일 절차·동일 방향**의 검정 (제약 아래) |
| **Exploratory** | rank 절단 · designed-leaky H3 · CA2F · support-bin · scaffold(murcko·generic) 재현 · 종-cold(판정 대기) |

## Confirmatory 제약 (사전등록 문구)

> confirmatory 패밀리는 discovery 패밀리가 검정한 것과 **동일한 가설 집합**을, **동일한 검정 절차·동일한 방향**으로만 검정한다. 가설 추가, 검정 절차(단측/양측·지표·집계 단위·δ 정의) 변경, 방향 변경은 불허. 그러한 검정은 confirmatory가 아니라 exploratory에 귀속.

### 파이프라인 검증 (자동 대조·불일치 시 중단)
- confirmatory 가설 목록이 primary의 **부분집합**인지 (가설 키 = (backbone, tier_A, tier_B, test_type, direction)) set-⊆ 검사.
- 검정 파라미터 동일성: {test(one/two-sided), metric(RMSE), agg(strata), δ_def(pooled within-condition SD), tost_bound} 를 primary/confirmatory에서 해시 비교.
- 불일치 → **abort**(로그에 diff 출력). 통과 시에만 confirmatory FDR 진행.

## 배정 근거 (감사)
- **group = primary**(핸드오프 §3-3 승계, CAS primary/scaffold secondary). 재논의 없음.
- **cross-group = primary**: 검정력 아닌 **처방 경계 검정**(one-hot·임베딩이 미학습 퇴화, taxonomy만 형제 종 전이 → 처방이 가장 취약한 지점; taxonomy 승리 시 처방에 "미학습 분류군 예측 시" 범위 한정어).
- **replication = confirmatory**(primary 편입 시 패밀리 2배로 discovery 검출력↓ 자기모순; exploratory 편입 시 확증 무게 소실 → 독립 오류통제).
- **scaffold = exploratory**(재현 확인). **종-cold = 판정 대기**(director; 처방 경계 "미학습 종" 함의 정리 별도 보고).

## 통제 깊이 (Methods 명시, 패밀리 분리 아님)
- cold 계열 통제 = **shuffled만**(warm 전 통제보다 얕음).
- 사유: zero/dummy는 cold에서 no-species와 실질 중복; shuffled = **차원 보존 순열 → 용량 매칭 동시 충족**(cold fixed_proj 불요, 용량 반박 차단).

## 비교 집합 확정 (director 2026-08-03, 결과 확인 전)
- **Primary 동등성 TOST = {t3a vs t2, t3b vs t2, t4 vs t2}, within-backbone × {dmpnn, graphconv}.** (DRAFT A2 "핵심 3a vs 2·4 vs 2" → **t3b vs t2 추가**로 확정: t3a와 같은 구조물의 다른 출처(NCBI)이므로 등록 집합 완성.)
- **t1·t1′는 TOST primary 미포함**(참조 tier, 처방 후보 아님); t1·t1′ 관련 TOST = exploratory.
- Primary 우월성·나머지 소속(cross-backbone=within만/naive/replication=confirmatory/split=group/arm=§4Δ 4-arm DD) = 기존 문서 유지. 상세 = `PREREGISTRATION.md §4C`.
