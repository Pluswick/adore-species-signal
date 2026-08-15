# ADORE tier 번호 네임스페이스 + fusion-locus 격리 (Phase 1)

> 출력·로그·결과 컬럼에 `adore_t*` 명시 ID. run 파일명은 variant-keyed 유지(resumability·충돌 없음), `adore_tier` 라벨은 매핑으로 aggregation/분석 층에서 부여.

## variant ↔ adore_t* 매핑

| adore_tier | variant (prefix = 통제) | 비고 |
|---|---|---|
| **adore_t0** | `no_species` | Tier 0 baseline |
| **adore_t1** | `{ctrl}_species_bias_only` | Tier 1 additive bias |
| **adore_t1prime** | `{tier1prime_oof, shuffled_tier1prime_oof}` | Tier 1′ residual calib (GNN-native OOF) |
| **adore_t2** | `{ctrl}_species_categorical` | Tier 2 one-hot |
| **adore_t3a** | `{ctrl}_species_taxonomy_original` | Tier 3a native taxonomy (tax_*) |
| **adore_t3b** | `{ctrl}_species_taxonomy_ncbi` | Tier 3b NCBI taxonomy (ncbi_*) |
| **adore_t4** | `{ctrl}_species_late_fusion` | Tier 4 learned embedding |
| **adore_t4_fixedproj** | `{ctrl}_species_fixed_proj` | fixed_proj 용량통제 (3a·3b·4 공유 baseline) |

ctrl ∈ {true, shuffled, zero, dummy}. cold(블록 B) 통제 = shuffled만.

## ⚠ 구 fusion-locus 격리 (확인)

기존 코드의 fusion-locus 번호는 ADORE tier와 다름 — 혼동 방지:
- `late_fusion` = **ADORE t4**(학습 임베딩). (구 주석 "Tier 4=late_fusion"는 우연히 일치.)
- **`film` = 구 fusion-locus tier 5 → Phase 1에서 사용 안 함**(ADORE t5=phylo는 성능 실험 제외). 학습 variant 목록에서 **제외**.
- `early_injection`/`message_level` = 구 fusion-locus ablation → Phase 1 미사용.
- ⟹ Phase 1 학습 variant는 위 표의 것만. film·early·message는 launch 목록에 포함하지 않음.

## split 목록 (warm)

`{partition}_{split}` where partition ∈ {discovery, replication}, split ∈ **{group, scaffold, scaffold_generic, designed_leaky}**. scaffold_generic 포함(신규). group = primary.
- cold(블록 B): 종-cold·cross-group split 계열(별도 생성, group/scaffold와 비곱).
