# Species representation in multi-species aquatic toxicity prediction

Code, pre-registration, and audit artefacts for the paper

> **The choice of species representation in multi-species aquatic toxicity prediction is
> constrained by data availability rather than by predictive accuracy.**

The study compares seven species representations under an identical training procedure across
two datasets, four backbones, ten seeds, and four split schemes, with every criterion that
determines a verdict fixed in a written specification before any comparison was computed.
Differences are judged by two one-sided tests against a margin derived per dataset from the
models' own re-run noise.

This repository contains the analysis pipeline, the frozen pre-registration, the frozen
equivalence margins, and the full gatekeeping enumerations. Raw source data and per-seed run
outputs are not included; see **Data** below.

---

## Species representations

The seven representations divide into two families by the information each requires per species.
One-hot identity is the reference throughout. The identifiers in the first column are those used
in the frozen pre-registration, in the code, and in the result files; they are given here so that
the comparisons reported in the paper can be matched to the records in this repository.

| repo id | name in the paper | family | per-species d.o.f. |
|---|---|---|---|
| `t0` | no species information | identifier-based | 0 |
| `t1` | species mean offset | identifier-based | 1 |
| `t1'` | out-of-fold residual calibration | identifier-based | post-hoc |
| `t2` | one-hot identity | identifier-based | unconstrained |
| `t4` | learned embedding | identifier-based | 16 |
| `t3a` | source-authority taxonomy | externally structured | rank membership |
| `t3b` | NCBI taxonomy | externally structured | rank membership |
| `t4_fixedproj` | fixed-projection capacity control | control | 16 (frozen) |

Every species-using representation is accompanied by permuted, zeroed, and dummy label controls
(`ctrl ∈ {true, shuffled, zero, dummy}` in the variant names) and by the capacity-matched frozen
random projection.

## Other names used in this repository

Several internal names predate the terminology settled in the paper. The full mapping is in
[`docs/NAMING.md`](docs/NAMING.md); the ones a reader meets first are:

| in this repository | in the paper |
|---|---|
| `b1`, "expansion", "second dataset" | the ECOTOX-external set |
| `discovery` | the single-condition partition (LC50, 96 h) |
| `replication` | the seven-condition partition (LC50/EC50, 24–96 h) |
| `Phase 1` | the ADORE data and the analyses run on it |
| `q2`, `v3`, `v4`, `src` | internal project and revision labels; no scientific meaning |

## What's here

- `PREREGISTRATION.md` — the frozen pre-registration. `PREREGISTRATION_draft.md` is the earlier
  draft, retained because it timestamps which family-assignment questions were still open before
  training began; `PREREG_EXPANSION.md` extends the same specification to the ECOTOX-external set.
- `src/` — core compute: dataset assembly, model construction, species controls, bootstrap,
  gatekeeping. Identical across both datasets.
- `scripts/` — dataset builds, training launchers, margin freezes, gatekeeping, and analysis.
- `results/q2_v4/audit/` — the frozen margins, the construction record for the ECOTOX-external
  set, the guard reports, and the supporting analyses cited in the paper.
- `results/q2_v4/runs/`, `results/q2_v4/runs_b1/` — the gatekeeping enumerations and bootstrap
  outputs for the two datasets.
- `docs/FREEZE_HISTORY.md` — the chronology of the margin freeze, including the one correction
  that was made to the freeze-evidence record.
- `docs/NAMING.md` — internal name to paper terminology mapping.
- `HANDOFF_experiment.md`, `results/q2_v4/audit/GAP_EXECUTION_LOG.md` — the working
  execution record, retained for provenance. Written in Korean; not required to use the pipeline.

> **Note on language.** `PREREGISTRATION.md` and several audit documents are written in Korean.
> An English rendering is provided in [`docs/`] where one exists. See **Known limitations**.

## Requirements

Python 3.11, with:

| package | version |
|---|---|
| PyTorch | 2.12.1 (CUDA 13.0) |
| RDKit | 2026.3.3 |
| LightGBM | 4.6.0 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |

`environment.yml` and `requirements.txt` reproduce the environment. Both scaffold splits
(Bemis–Murcko and generic) are computed by RDKit, so the RDKit version affects split assignment
and must be matched to reproduce the reported partitions.

Run all scripts from the repository root.

### The bundled `ccmpnn` package

`ccmpnn/` supplies the molecular graph representation (`MolGraph`, `assemble_batch`), the
directed message-passing backbone (`DMPNNModel`, `MPNEncoder`), the late-fusion species context,
and the evaluation metrics. `src` builds both backbones on top of it, so neither network runs
without it.

The package name comes from an earlier, discontinued project of the same author
("Categorical-Context MPNN"). **It is not a separately published library and there is no
accompanying paper.** The work reported here uses only its standard directed message-passing
core and late fusion; the mechanism the package is named after is not used. See
[`docs/NAMING.md`](docs/NAMING.md) for the full list of components that are and are not used.

It is licensed under the Apache License 2.0, as is the rest of this repository.

## Reproduce

1. **Obtain the source tables.**
   - ADORE: the intermediate processed table (`processed/ecotox_mortality_processed.csv` and
     `chemicals/ecotox_properties_with-oecd-function.csv`) is distributed with the benchmark
     under CC-BY 4.0. Set the path at the top of `scripts/build_adore_datasets.py`.
   - ECOTOX-external: the aggregated table is deposited under CC0 1.0 at
     [DOI]. Rebuilding it from source instead requires
     the June 2026 ECOTOX ASCII export; see `scripts/build_b1.py` and
     `results/q2_v4/audit/ecotox_expansion_v2.md`.
2. **Build the datasets:** `python scripts/build_adore_datasets.py`, and
   `python scripts/make_p_keys.py` followed by `python scripts/build_b1.py` if rebuilding the
   ECOTOX-external set from source.
3. **Train:** the `run_q2_*` scripts (ADORE) and `run_b1_*` scripts (ECOTOX-external). Networks on
   GPU, deterministic models on CPU.
4. **Freeze the margins before any comparison:** `python scripts/compute_freeze_delta*.py`.
5. **Run gatekeeping:** `python scripts/run_q2_gatekeeping.py` and
   `python scripts/run_q2_gatekeeping_b1.py`.

## Data

**Not redistributed here.** The ADORE benchmark is available under CC-BY 4.0 and ECOTOX is
distributed by the US EPA as a public ASCII export that can be downloaded without registration
(<https://cfpub.epa.gov/ecotox/>).

Because the EPA export is replaced quarterly and superseded releases are not served from the
ECOTOX site, the aggregated **ECOTOX-external table is deposited under CC0 1.0** at [DOI], with checksums, so that it need not be rebuilt from the
June 2026 release.

Per-seed run outputs (approximately 16 GB, 6,286 prediction files) are excluded and are
regenerable from the run scripts.

## What can be checked without re-training

The frozen margins, the comparison-set enumerations, and the gatekeeping verdicts in
`results/q2_v4/` are the direct source of the tables in the paper and can be inspected as they
stand. Reproducing the effect sizes themselves requires re-running the training pipeline, since
the per-seed predictions are not committed.

## Known limitations

- **The pre-registration and much of the audit record are in Korean.** They are the primary
  evidence for the pre-registration claims made in the paper and should be readable by
  reviewers who do not read Korean.
- **Several scripts contain author-specific paths.** Those that leaked local usernames have been
  replaced with environment variables; others are marked in their headers.

## License

Code is released under the Apache License 2.0 (`LICENSE`, `NOTICE`). ECOTOX-derived data is not
licensed by this repository; the deposited ECOTOX-external table is released under CC0 1.0.

## Citation

Dataset:

> ECOTOX-external set: aggregated aquatic acute toxicity strata disjoint from the ADORE processed
> table. Zenodo. [DOI]

Paper: to be added on publication.
