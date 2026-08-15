# B1 확장 실험 결과 (사실 보고)

> 사실 보고. 해석·서사·결론 없음(판정은 director). 전 수치는 산출물에서 재확인. Phase 1은 원 산출물 재독.

## §2 실행 무결성
- 계획 vs 완료: **3794/3794**, 실패 **0**, 재시도 0. 블록별 전부 일치.
- 이상: NaN=0 inf=0 미파싱=0 (run JSON 2954건); 조기중단=정상 early stopping(비정상 종료 0).
- prereg 최종수정 2026-08-06 16:37:30; 초기작성 후 수정: §2.5 join-sanity result (PASS) recorded; appendix A: 24-warm-variant enumeration (22 ladder + 2 t1') from empirical Phase-1 run inventory; §3 GPU deviation: '4090-only' -> '4090 primary (seed0-4) + 5060Ti auxiliary (seed5-9)', director-approved. per-edit timestamps not individually recorded (no VCS); file last-write 2026-08-06 16:37. The §3 GPU-deviation amendment post-dates the first run artifact (2026-08-06 15:00). All amendments predate margin derivation and any tier-performance number.
- 자원: RTX 4090 (cuda:1) + RTX 5060 Ti (cuda:0); CPU for deterministic block; seed 0-4 -> 4090, seed 5-9 -> 5060Ti; dprime seed 10-54 -> 4090, seed 55-99 -> 5060Ti; 8 GNN shards (4 per GPU) during ladder; per-block shard layouts thereafter; deterministic on CPU concurrent with ladder; wall first run artifact 2026-08-06 15:00 -> sequencer COMPLETE 2026-08-10 16:07 (~97h).
- GPU 기록: run JSON config does NOT record device; GPU assignment confirmed via launch env (CUDA_VISIBLE_DEVICES per shard) + nvidia-smi observations (both GPUs utilized throughout, 77-98% util observed)
- 하이퍼파라미터: epochs 100, batch 256, lr 5e-4, wd 1e-5, hidden 300, depth 3, dropout 0.1, species_emb_dim 16, val_frac 0.1 (frozen, unchanged from Phase 1 code path)

## §3 마진 (상세 = expansion_margins.md)
δ 0.019777→0.035641 · δ_det 0.087189→0.074082 · δ′ 0.005717→0.015957. 판정=B1 마진.

## §4 주 결과 (비교 수 Phase1=225 / B1=189)
### §4-1 판정 census [동등/유의/불확정]
| 패밀리 | Phase 1 | B1 |
|---|---|---|
| primary 우월성 | 0 / 24 / 0 | 0 / 23 / 1 |
| primary TOST | 0 / 1 / 5 | 0 / 1 / 5 |
| confirmatory 우월성 | 0 / 20 / 4 | ABSENT (structural: no replication partition) |
| confirmatory TOST | 0 / 2 / 4 | ABSENT (structural) |
| deterministic TOST | 3 / 0 / 0 | 2 / 1 / 0 |
| exploratory rank | 0 / 1 / 11 | 2 / 4 / 6 |
| exploratory support-bin | 0 / 6 / 18 | 0 / 3 / 21 |
| exploratory tax_group | 0 / 4 / 14 | 0 / 1 / 17 |
| exploratory scaffold | 0 / 42 / 18 | 1 / 51 / 8 |
| exploratory designed-leaky | 1 / 28 / 1 | 0 / 28 / 2 |
| exploratory cross-backbone | 0 / 1 / 5 | 0 / 0 / 6 |
| sensitivity_ensemble | 0 / 5 / 7 | 0 / 1 / 5 |
| 게이트 Stage2 도달 | 0 | 0 |

### §4-2 효과크기 dd [90% CI] (block=smiles, N_BOOT 2000)

**group**
| 비교 | Phase1 dm / gc | B1 dm / gc |
|---|---|---|
| t0->t2 | -0.0761[-0.124,-0.025](sig) / -0.0674[-0.114,-0.018](sig) | -0.1502[-0.211,-0.092](sig) / -0.1946[-0.262,-0.128](sig) |
| t2->t3a | -0.0175[-0.045,+0.009](ind) / -0.0332[-0.070,+0.002](ind) | -0.0078[-0.067,+0.047](ind) / +0.0090[-0.043,+0.054](ind) |
| t2->t3b | -0.0166[-0.046,+0.014](ind) / -0.0426[-0.079,-0.009](sig) | +0.0018[-0.055,+0.053](ind) / +0.0321[-0.017,+0.078](ind) |
| t2->t4 | +0.0115[-0.013,+0.034](ind) / -0.0151[-0.048,+0.016](ind) | -0.0003[-0.048,+0.040](ind) / +0.0438[+0.009,+0.080](sig) |

**scaffold**
| 비교 | Phase1 dm / gc | B1 dm / gc |
|---|---|---|
| t0->t2 | -0.0642[-0.114,-0.014](sig) / -0.0573[-0.106,-0.008](sig) | -0.1291[-0.184,-0.065](sig) / -0.1577[-0.216,-0.091](sig) |
| t2->t3a | -0.0057[-0.051,+0.049](ind) / -0.0193[-0.055,+0.017](ind) | +0.0267[-0.015,+0.068](ind) / +0.0108[-0.029,+0.048](ind) |
| t2->t3b | -0.0138[-0.061,+0.042](ind) / -0.0154[-0.056,+0.027](ind) | +0.0124[-0.032,+0.055](ind) / -0.0066[-0.042,+0.026](ind) |
| t2->t4 | -0.0126[-0.039,+0.012](ind) / +0.0159[-0.024,+0.054](ind) | +0.0218[-0.013,+0.055](ind) / +0.0432[+0.013,+0.071](sig) |

**scaffold_generic**
| 비교 | Phase1 dm / gc | B1 dm / gc |
|---|---|---|
| t0->t2 | -0.0311[-0.062,+0.002](ind) / -0.0451[-0.088,-0.003](sig) | -0.1499[-0.211,-0.086](sig) / -0.0988[-0.160,-0.037](sig) |
| t2->t3a | +0.0072[-0.014,+0.029](ind) / +0.0006[-0.026,+0.027](ind) | +0.0058[-0.024,+0.036](ind) / -0.0516[-0.089,-0.015](sig) |
| t2->t3b | +0.0013[-0.018,+0.022](ind) / -0.0016[-0.033,+0.028](ind) | +0.0015[-0.033,+0.035](equ) / -0.0436[-0.076,-0.009](sig) |
| t2->t4 | -0.0076[-0.027,+0.010](ind) / +0.0005[-0.024,+0.023](ind) | +0.0197[-0.005,+0.042](ind) / -0.0246[-0.059,+0.005](ind) |

**designed_leaky**
| 비교 | Phase1 dm / gc | B1 dm / gc |
|---|---|---|
| t0->t2 | -0.0638[-0.089,-0.036](sig) / -0.0687[-0.093,-0.042](sig) | -0.1108[-0.147,-0.076](sig) / -0.0908[-0.129,-0.055](sig) |
| t2->t3a | -0.0403[-0.061,-0.019](sig) / -0.0446[-0.067,-0.022](sig) | -0.0208[-0.051,+0.006](ind) / -0.0449[-0.075,-0.017](sig) |
| t2->t3b | -0.0336[-0.053,-0.014](sig) / -0.0438[-0.066,-0.021](sig) | -0.0114[-0.039,+0.015](ind) / -0.0353[-0.061,-0.010](sig) |
| t2->t4 | +0.0082[-0.008,+0.025](ind) / -0.0030[-0.019,+0.013](equ) | +0.0600[+0.043,+0.077](sig) / +0.0395[+0.024,+0.055](sig) |

### §4-3 primary TOST q_family (BH-FDR)
| | Phase 1 | B1 |
|---|---|---|
| n | 6 | 6 |
| min q | 0.06 | 0.045 |
| q<0.05 통과 | 0 | 1 |

### §4-4 절대 RMSE (per-seed mean±sd / ensemble) — group split (전 split = json s4_4_absolute_rmse)
| tier | Phase1 dm | Phase1 gc | B1 dm | B1 gc |
|---|---|---|---|---|
| t0 | 1.1983±0.0189/1.18 | 1.1913±0.0243/1.1667 | 1.4678±0.0326/1.4448 | 1.5062±0.0552/1.4747 |
| t1 | 1.1299±0.021/1.1103 | 1.1417±0.016/1.1203 | 1.3794±0.0226/1.3609 | 1.3873±0.0492/1.3574 |
| t1p | 1.1463±0.0191/1.1264 | 1.1396±0.0255/1.1133 | 1.3586±0.0367/1.331 | 1.4022±0.0578/1.3652 |
| t2 | 1.1222±0.0178/1.0933 | 1.1239±0.016/1.0929 | 1.3175±0.0258/1.2784 | 1.3116±0.0329/1.2642 |
| t3a | 1.1046±0.0246/1.0695 | 1.0908±0.0183/1.0507 | 1.3097±0.0173/1.2314 | 1.3206±0.0298/1.2365 |
| t3b | 1.1055±0.0131/1.0708 | 1.0814±0.0125/1.0436 | 1.3193±0.0214/1.2499 | 1.3438±0.0391/1.263 |
| t4 | 1.1337±0.0215/1.0867 | 1.1088±0.0226/1.0607 | 1.3172±0.0156/1.2205 | 1.3554±0.0299/1.2556 |

### §4-5 우월성 24 (category 요약; 상세=json)
- Phase1: significant (n=24); 유의=24
- B1: indeterminate, significant (n=24); 유의=23; 비유의 항목: dmpnn/t1>shuf

### 종-cold ungated TOST (oov=mean; Stage2 도달 0이라 서술용, 판정 아님)
| 비교 | Phase1 dm/gc | B1 dm/gc |
|---|---|---|
| t2→t3a | -0.1712(sig)/-0.1849(sig) | -0.0990(sig)/-0.0651(sig) |
| t2→t3b | -0.1782(sig)/-0.1837(sig) | -0.1027(sig)/-0.0790(sig) |
| t2→t4 | +0.0055(equ)/+0.0054(ind) | +0.0174(ind)/+0.0681(sig) |

## §5 사전 지정 층화
### §5-1 support-bin (bins 1-4/5-9/10-19/20-49/50+) — B1 구간별 종수/행수
| bin | n_species | n_test_rows |
|---|---|---|
| 1-4 | 1126 | 823 |
| 5-9 | 296 | 456 |
| 10-19 | 195 | 556 |
| 20-49 | 111 | 694 |
| 50+ | 60 | 2032 |

B1 t2→{t3a,t3b,t4} dd[CI] per bin (dmpnn) = json s5_1_support_bins.b1.dd; Phase1 = .phase1.dd.

### §5-2 종 수 / one-hot
- one-hot: Phase1 discovery **779** / replication **1006**; B1 **1975**
- Phase1 discovery n_train_species: group=718, scaffold=744, scaffold_generic=743, designed_leaky=713
- B1 n_train_species: group=1788, scaffold=1846, scaffold_generic=1842, designed_leaky=1797

### §5-3 OOV (warm test)
| split | Phase1 종/행 | B1 종/행 |
|---|---|---|
| group | 61/69 | 187/341 |
| scaffold | 35/46 | 129/245 |
| scaffold_generic | 36/40 | 133/257 |
| designed_leaky | 66/74 | 178/317 |

- OOV-제거 재집계 dd (b1_group, 비-OOV종만; 참고, 판정 아님): dmpnn/t3a~t2=-0.0203, dmpnn/t3b~t2=-0.0102, dmpnn/t4~t2=-0.0157, graphconv/t3a~t2=-0.01, graphconv/t3b~t2=0.0065, graphconv/t4~t2=0.0177

## §6 가드 (상세 = expansion_guards.md)
축퇴 flagged=0; variant identical pairs=0; OOF 증명=PASS, t4 포함; 누출 탐지선 발동=0(전 split); 조인=PASS.

## §7 Phase 1 대비 차이 전수
| 항목 | Phase 1 | B1 |
|---|---|---|
| confirmatory family | present (replication_group, 30 comparisons) | ABSENT — no replication/confirmatory partition (structural exception; no substitute fabricated) |
| GPU seed->device map | dual-GPU: seed 0-6 -> 5060Ti, 7-9 -> 4090 (BLOCK_A_LAUNCH_MANIFEST) | dual-GPU: seed 0-4 -> 4090, 5-9 -> 5060Ti (director-approved; both experiments dual-GPU, maps differ) |
| three margins | delta 0.019777 / delta_det 0.087189 / delta' 0.005717 | RE-DERIVED (no reuse): delta 0.035641 / delta_det 0.074082 / delta' 0.015957 |
| NCBI taxonomy resolver | resolved once for Phase-1 species | SAME resolver re-run on B1's 1975 species (validated: 100% agreement reproducing Phase-1 committed ncbi_taxonomy output) |
| t4 leak handling | permutation test clean (all splits) | b1_scaffold flagged -> diagnosed (OOF-proof PASS, Phase1 20-perm clean, decisive control: ~79% capacity floor) -> characterized false positive -> t4 arm INCLUDED; capacity-floor baseline deferred to analysis (director-approved) |
| tax_group source coverage | mortality map covers Phase-1 species | same MORT map covers only 453/1788 b1_group train species (B1's new species absent from Phase-1 mortality file) -> tax_group exploratory sub-family reduced coverage |
| one-hot dimension | 779 (discovery) / 1006 (replication) | 1975 (single global vocab) |
| comparison count | 225 (30 primary + 30 confirmatory + 3 det + 150 exploratory + 12 sensitivity) | 189 (30 primary + 0 confirmatory + 3 det + 150 exploratory + 6 sensitivity) |
| prereg freeze integrity | n/a | PREREG_EXPANSION.md amended after initial creation incl. GPU deviation post-dating first run (15:00); all amendments predate margin derivation/performance |
| support-bin edges (§5-1 director-specified) | gatekeeping default 1-5/6-20/21-100/100+ | director §5-1 bins 1-4/5-9/10-19/20-49/50+ computed separately (both retained) |
| exploratory replication | full exploratory set | exploratory computed (rank/support-bin/tax_group/scaffold/designed-leaky/cross-backbone); tax_group reduced coverage as above; otherwise same structure |
| dataset rows | discovery+replication ECOTOX (Phase-1 curation) | B1_final (2026 ECOTOX pull, P-derived filters, disjoint from Phase-1 training P by result_id + precise dup) |

## §8 크게 다른 항목 (부호 상이 OR 크기 ≥2×; 사실만)
| 항목 | Phase 1 | B1 | 부호상이 | B1/P1 |
|---|---|---|---|---|
| margin delta_prime | 0.005717 | 0.015957 |  | 2.79 |
| one-hot dim | 779/1006 | 1975 |  | >=2x vs discovery(779) |
| warm OOV test rows (group) | 69 | 341 |  | 4.94 |
| warm OOV test species (group) | 61 | 187 |  | 3.07 |
| dd group/t0->t2/graphconv | -0.0674 | -0.1946 |  | 2.89 |
| dd group/t2->t3a/dmpnn | -0.0175 | -0.0078 |  | 0.45 |
| dd group/t2->t3a/graphconv | -0.0332 | 0.009 | 예 | 0.27 |
| dd group/t2->t3b/dmpnn | -0.0166 | 0.0018 |  | 0.11 |
| dd group/t2->t3b/graphconv | -0.0426 | 0.0321 | 예 | 0.75 |
| dd group/t2->t4/dmpnn | 0.0115 | -0.0003 |  | 0.03 |
| dd group/t2->t4/graphconv | -0.0151 | 0.0438 | 예 | 2.9 |
| dd scaffold/t0->t2/dmpnn | -0.0642 | -0.1291 |  | 2.01 |
| dd scaffold/t0->t2/graphconv | -0.0573 | -0.1577 |  | 2.75 |
| dd scaffold/t2->t3a/dmpnn | -0.0057 | 0.0267 | 예 | 4.68 |
| dd scaffold/t2->t3a/graphconv | -0.0193 | 0.0108 | 예 | 0.56 |
| dd scaffold/t2->t3b/dmpnn | -0.0138 | 0.0124 | 예 | 0.9 |
| dd scaffold/t2->t3b/graphconv | -0.0154 | -0.0066 |  | 0.43 |
| dd scaffold/t2->t4/dmpnn | -0.0126 | 0.0218 | 예 | 1.73 |
| dd scaffold/t2->t4/graphconv | 0.0159 | 0.0432 |  | 2.72 |
| dd scaffold_generic/t0->t2/dmpnn | -0.0311 | -0.1499 |  | 4.82 |
| dd scaffold_generic/t0->t2/graphconv | -0.0451 | -0.0988 |  | 2.19 |
| dd scaffold_generic/t2->t3a/graphconv | 0.0006 | -0.0516 |  | 86.0 |
| dd scaffold_generic/t2->t3b/graphconv | -0.0016 | -0.0436 |  | 27.25 |
| dd scaffold_generic/t2->t4/dmpnn | -0.0076 | 0.0197 | 예 | 2.59 |
| dd scaffold_generic/t2->t4/graphconv | 0.0005 | -0.0246 |  | 49.2 |
| dd designed_leaky/t2->t3b/dmpnn | -0.0336 | -0.0114 |  | 0.34 |
| dd designed_leaky/t2->t4/dmpnn | 0.0082 | 0.06 |  | 7.32 |
| dd designed_leaky/t2->t4/graphconv | -0.003 | 0.0395 |  | 13.17 |

---
전 수치 기계판독 = `audit/expansion_results.json`. 장치진단 = `audit/expansion_device_diagnostic.json`. 마진 동결 = `audit/delta_*_frozen_b1.json`. gatekeeping 원본 = `runs_b1/bootstrap/gatekeeping_results.json`. 예측 CSV 전량 = `runs_b1/**/predictions/`.