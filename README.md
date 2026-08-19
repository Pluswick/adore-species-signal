# Comparison for species representation in aquatic toxicity prediction

Code, pre-registration, and audit artifacts for the paper
*"Data availability, not predictive performance, constrains the choice of species representation
in multi-species aquatic toxicity prediction."*

The study pre-registers a ladder of seven ways to represent the species in an acute-toxicity model,
trains it across backbones, seeds and splits, and judges the differences by equivalence testing
against a frozen margin — on two datasets (the ADORE benchmark and a second, ECOTOX-external set).
This repository lets you verify the reported numbers and re-run the pipeline. Raw data and large run
outputs are not included (see **Data** below).

## Tier ladder

| tier | species representation |
|---|---|
| t0 | none |
| t1 | additive per-species offset |
| t1' | out-of-fold residual calibration |
| t2 | one-hot identity |
| t3a | taxonomy (source authority) |
| t3b | taxonomy (NCBI) |
| t4 | learned embedding |

## What's here

- `PREREGISTRATION.md`, `PREREG_EXPANSION.md` — the frozen pre-registrations.
- `jcim_v3/` — core compute (training, species controls, gatekeeping); identical across both datasets.
- `scripts/` — dataset build, training launchers, margin freezes, gatekeeping, and analysis.
- `results/q2_v4/audit/` — the audit artifacts that are the source of the paper's numbers
  (frozen margins, the full gatekeeping enumerations, the expansion report).

## Requirements

A conda environment named `jcim_v3` with RDKit, LightGBM, PyTorch, pandas, numpy and scikit-learn.
Run scripts from the repository root.

## Reproduce

1. Obtain the ECOTOX source data (not redistributed here — see **Data**) and set `<DATA_ROOT>`.
2. Build the datasets: `python scripts/build_adore_datasets.py` (ADORE) and
   `python scripts/build_b1.py` (second dataset; reconstructs the documented recipe in
   `results/q2_v4/audit/ecotox_expansion_v2.md` — see the script header for the inputs it needs).
3. Train the ladder with the `run_q2_*` / `run_b1_*` scripts (GNN on GPU, deterministic tiers on CPU).
4. Freeze the margins **before** any comparison: `python scripts/compute_freeze_delta*.py`.
5. Run gatekeeping: `python scripts/run_q2_gatekeeping.py` and `python scripts/run_q2_gatekeeping_b1.py`.

## Data

Raw ECOTOX records and the aggregated modelling tables are **not** redistributed (licensing and size).
Obtain ECOTOX from the U.S. EPA (https://cfpub.epa.gov/ecotox/); the second dataset uses the June 2026
release. Per-seed run outputs (~16 GB) are excluded and are regenerable from the run scripts. The build
and aggregation are fully specified in the build scripts and the audit docs.

## License

Code is released under the Apache License 2.0 (`LICENSE`, `NOTICE`). ECOTOX-derived data is not
licensed by this repository.

## Citation

To be added on publication.
