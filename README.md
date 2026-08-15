# ADORE Q2 — Where the species signal saturates across a representation tier ladder

Pre-registered study of **how much of the species-related variance in aquatic acute
toxicity prediction is captured as the species representation is made progressively
richer** (a tier ladder t0 → t1 → t1′ → t2 → t3a → t3b → t4), on D-MPNN and GraphConv
backbones (plus LightGBM/naive deterministic baselines), with an **independent replication
(B1)** on a fresh, disjoint ECOTOX pull.

This repository contains the **code, pre-registration, and audit artifacts** needed to
verify the claims and reproduce the analysis. Large run outputs and raw data are **not**
included (see *Data & outputs* below).

> Paths in the code that were absolute on the author's machine have been replaced with
> repo-relative form (`.`) or placeholders (`<DATA_ROOT>`, `<USER_HOME>`). Run scripts from
> the repository root. No personal identifiers remain in the tree.

## Tier ladder

| tier | representation of species |
|---|---|
| t0 | none (no_species) |
| t1 | additive per-species bias |
| t1′ | out-of-fold species-mean residual calibration |
| t2 | one-hot / categorical embedding |
| t3a | native 4-rank taxonomy |
| t3b | NCBI 4-rank taxonomy |
| t4 | learned late-fusion species embedding (GNN) / OOF-SVD species factor (LightGBM) |

## Repository layout

```
PREREGISTRATION.md            Phase-1 pre-registration (frozen before results)
PREREG_EXPANSION.md           B1 replication pre-registration
HANDOFF_ADORE_experiment.md   design/handoff notes
jcim_v3/                      core compute (backbone-agnostic; UNCHANGED across Phase 1 / B1)
  runner.py, models.py          training + tier/variant construction
  gatekeeping.py                dd / TOST / BH-FDR decision logic
  species_controls.py           shuffled/zero/dummy controls
  tier_input_guard.py           input non-degeneracy guards
  rdkit_lgbm.py, naive_species_baselines.py, stratum.py, prediction_io.py, paths.py
scripts/                      launchers + analysis (hardcoded paths only; core is in jcim_v3)
  build_adore_datasets.py        Phase-1 dataset build (ECOTOX -> aggregated tiers)
  run_q2_gnn_ladder.py           GNN warm ladder (Phase 1)
  run_q2_gnn_oof_tier1prime.py   t1' OOF
  run_q2_blockb_oov.py           species-cold (OOV) block
  run_q2_lgbm_tier4.py           LightGBM t4 double-OOF SVD factor
  run_q2_cpu_tiers_blockA.py     LightGBM + naive deterministic tiers
  run_q2_delta_prime.py          delta' ensemble-sensitivity seeds
  compute_freeze_delta.py        primary margin delta  (freeze)
  compute_freeze_delta_det.py    deterministic margin delta_det (freeze)
  compute_freeze_delta_prime.py  ensemble margin delta' (freeze)
  run_q2_gatekeeping.py          Phase-1 gatekeeping (dd/TOST/FDR)
  ncbi_resolve.py                NCBI taxonomy resolver (taxdump)
  run_b1_*.py                    B1 replication launchers (ladder, t1', cold, dprime, deterministic)
  run_b1_sequencer.sh            B1 GPU pipeline sequencer
  compute_freeze_delta*_b1.py    B1 margin freezes
  run_q2_gatekeeping_b1.py       B1 gatekeeping (confirmatory family removed; B1 has no replication partition)
  verify_tier4_*.py              t4 leak tests (permutation, OOF-index proof)
  analysis/                      scripts that produced the B1 expansion report numbers
    parse_gate.py, s4_4_rmse.py, s5_bins.py, s5_cold_oov.py, s6_guards.py,
    s7_s8.py, device_diag.py, capture_rate.py, assemble_md.py
results/q2_v4/audit/          audit artifacts = the SOURCE of the paper's numbers
  delta_primary_frozen.json / _b1.json    frozen margins (value + spec + timestamp + pre-freeze evidence)
  delta_det_frozen*.json, delta_prime_frozen*.json
  expansion_results.{md,json}             B1 report (execution integrity, census, dd+CI, q, RMSE, stratification)
  expansion_margins.md, expansion_guards.md
  expansion_device_diagnostic.json        between-device variance component of the margins
  capture_rate.{md,json}                  species-variance capture rate (rough calc; see caveat in file)
  mechanism_facts_compute.txt             mechanism analysis (RMSE, species-axis SD, cross-class rho)
  GAP_EXECUTION_LOG.md, ...               provenance / audit trail
results/q2_v4/runs/bootstrap/         Phase-1 gatekeeping_results.json + human report (small)
results/q2_v4/runs_b1/bootstrap/      B1 gatekeeping_results.json (small)
```

## Margins (frozen, re-derived independently for B1)

| margin | Phase 1 | B1 | definition |
|---|---|---|---|
| δ (per-seed) | 0.019777 | 0.035641 | df-weighted pooled within-condition SD of per-seed RMSE (C=14) |
| δ_det | 0.087189 | 0.074082 | LightGBM block-bootstrap (block=smiles) RMSE SD (C=6) |
| δ′ (ensemble) | 0.005717 | 0.015957 | pooled SD of k=10 disjoint 10-seed ensemble RMSEs (C=14) |

Frozen files record value + spec + timestamp + pre-freeze evidence (no comparison output existed
at freeze time). Gatekeeping reads margins from the frozen files and never recomputes them.

## Reproduce

Environment: `conda` env with RDKit, LightGBM, PyTorch, pandas, numpy, scikit-learn
(the code invokes `conda run -n jcim_v3 python ...`).

1. Obtain the ECOTOX-derived source data (see *Data availability*) and set `<DATA_ROOT>`.
2. Build datasets: `python scripts/build_adore_datasets.py` (Phase 1); B1 build is a documented
   variant (see *B1 dataset build* note).
3. Train: `python scripts/run_q2_gnn_ladder.py --splits ...` etc. (GPU); deterministic tiers on CPU.
4. Freeze margins **before** any comparison: `python scripts/compute_freeze_delta*.py`.
5. Gatekeeping: `python scripts/run_q2_gatekeeping.py --execute` (B1: `run_q2_gatekeeping_b1.py`).
6. Report tables: `scripts/analysis/*.py`.

## Data & outputs (NOT in this repo)

- **Run outputs** (per-seed prediction CSVs, run JSONs) are ~16 GB and are **excluded**
  (`.gitignore`). They are regenerable from the run scripts. Only the small aggregated
  gatekeeping results are included.
- **Raw ECOTOX data** is **not redistributed** here (licensing). The build reads a processed
  ECOTOX mortality/effects export; obtain ECOTOX from the U.S. EPA (https://cfpub.epa.gov/ecotox/)
  and set `<DATA_ROOT>`.
- `results/q2_v4/data*/` aggregated dataset CSVs are excluded (data licensing + size); the
  aggregation is fully specified in `build_adore_datasets.py` and the audit docs.

### B1 dataset build (note)

`scripts/build_b1.py` and `scripts/ncbi_resolve.py` are **reconstructions**: the original working
scripts were developed in a temporary directory that was later cleared. `build_b1.py` is a faithful
re-implementation of the documented recipe (`results/q2_v4/audit/ecotox_expansion_v2.md` §1) on top of
the Phase-1 build template (`build_adore_datasets.py`, whose D16/aggregation/split functions it imports
UNCHANGED); it self-validates its split counts against `results/q2_v4/data_b1/data_provenance_ledger.csv`.
A few author-specific inputs (the 2026 ECOTOX ASCII source columns, the exact `MEDIA_P` set, and the P
`result_id` / precise-key sets used for the disjointness subtraction) must be supplied from the author's
environment — see the header of `build_b1.py`. `ncbi_resolve.py` reproduces the Phase-1 NCBI resolution
and validates against the committed `ncbi_taxonomy_by_species.csv`.

## License

Code is released under the **Apache License 2.0** (see `LICENSE`; attribution and the data notice
are in `NOTICE`). Copyright 2026 Kim Seonghan, Sangmyung University CCLABS (Creative Contents Labs).
ECOTOX-derived data is **not** licensed by this repository (see `NOTICE` and *Data availability*).

## Citation

To be added on publication.
