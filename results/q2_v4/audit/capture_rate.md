# capture_rate (거친 계산 — 산출물 고정)

> 정의: capture_rate(tier,backbone) = (RMSE_t0² − RMSE_tier²) / (species-axis pairwise SD)²
> ⚠ 거친 계산: 분모(종축 pairwise SD)²는 측정/복제 잡음을 포함하므로 상한이 1이 아님.

## 입력·출처
- RMSE = per-seed 평균, group·warm·main (Phase1 discovery_group / B1 b1_group; PART3 = `expansion_results.json s4_4`).
- species-axis SD: Phase1 discovery = **0.960** (PART6 `mechanism_facts_compute.txt`); B1 = **1.0792** (동일 방법, 본 파일에서 산출).
- 방법 검증(discovery 재현): SD=0.9597, n_units=1146, n_pairs=157441 → PART6(0.960/1146/157441) 일치=True.

## capture_rate (%) — 전 tier × 양 backbone × 양 실험
| tier | P1 dmpnn | P1 graphconv | B1 dmpnn | B1 graphconv |
|---|---|---|---|---|
| t0 | 0.0% | 0.0% | 0.0% | 0.0% |
| t1 | 17.3% | 12.6% | 21.6% | 29.5% |
| t1p | 13.2% | 13.1% | 26.5% | 26.0% |
| t2 | 19.2% | 16.9% | 35.9% | 47.1% |
| t3a | 23.4% | 24.9% | 37.7% | 45.0% |
| t3b | 23.2% | 27.1% | 35.5% | 39.7% |
| t4 | 16.4% | 20.6% | 36.0% | 37.0% |

## director 손계산 4값 대조
- t2 D-MPNN: 기재 ~19% vs 산출 **19.2%**
- t2 GraphConv: 기재 ~17% vs 산출 **16.9%**
- t3b D-MPNN: 기재 ~23% vs 산출 **23.2%**
- t3b GraphConv: 기재 ~27% vs 산출 **27.1%**