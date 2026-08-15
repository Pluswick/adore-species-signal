# §6 가드

> 사실 보고. 해석·서사·결론 없음(판정은 director). 전 수치는 산출물에서 재확인. Phase 1은 원 산출물 재독.

- **tier-input 축퇴 검사**: 225 check records, 축퇴 flagged = **0**
- **variant 구별성(학습 후 예측)**: b1_group seed0 — dmpnn 24 variant / 276 pair 중 identical=0; graphconv identical=0
- **t4 누출 최종**: OOF 경계 증명 B1 = **PASS**; permutation flag = b1_scaffold flagged (single+20-perm robust; other splits not flagged on mean); 결정적 대조(b1_scaffold) noise-floor=0.0391 shuffled-correct=0.0493 shuffled-misassigned=0.0467; 처분=**characterized false positive (capacity floor ~79% of flag; OOF construction clean; not t4-common). director-approved.**; **t4 arm 포함=True**; caveat=b1_scaffold noise-floor baseline to be added at analysis (director-approved deferral)
- **누출 탐지선(정지 임계) 발동**: (B1 §6 임계 미사전등록; Phase1-analog 0.5×잔차SD 참조)

| split | 잔차 target SD | 하한(0.5×SD) | min RMSE | n runs | 하한 미만 |
|---|---|---|---|---|---|
| b1_group | 1.7586 | 0.8793 | 1.268 | 480 | 0 |
| b1_scaffold | 1.7101 | 0.8551 | 1.374 | 440 | 0 |
| b1_scaffold_generic | 1.7296 | 0.8648 | 1.4153 | 440 | 0 |
| b1_designed_leaky | 1.729 | 0.8645 | 1.0911 | 440 | 0 |

- **조인 성공률**: join_sanity = **PASS** (n_species=1975); species<->species_idx bijection, native-tax consistent, embedded ncbi==SSOT (0 mismatch), train/test key-disjoint, species-cold all-cold (from join_sanity_b1.json)