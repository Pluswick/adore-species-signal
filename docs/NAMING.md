# Internal names and the terminology used in the paper

Development-stage names survive in file names, variant keys, and result columns in this
repository. Renaming them would break the correspondence with the frozen pre-registration and
with the committed result files, so they are retained and mapped here instead.

## Species representations

| repo id | variant key | name in the paper |
|---|---|---|
| `t0` | `no_species` | no species information |
| `t1` | `{ctrl}_species_bias_only` | species mean offset |
| `t1'` | `tier1prime_oof`, `shuffled_tier1prime_oof` | out-of-fold residual calibration |
| `t2` | `{ctrl}_species_categorical` | one-hot identity |
| `t3a` | `{ctrl}_species_taxonomy_original` | source-authority taxonomy |
| `t3b` | `{ctrl}_species_taxonomy_ncbi` | NCBI taxonomy |
| `t4` | `{ctrl}_species_late_fusion` | learned embedding |
| `t4_fixedproj` | `{ctrl}_species_fixed_proj` | fixed-projection capacity control |

`ctrl ∈ {true, shuffled, zero, dummy}` maps to the label controls described in the paper:
`true` is the representation itself, `shuffled` is the permuted (permute-and-relearn) control,
`zero` is the zeroed control, and `dummy` is the constant-placeholder control.

The paper groups these into two families by information requirement — identifier-based
(`t0`, `t1`, `t1'`, `t2`, `t4`) and externally structured (`t3a`, `t3b`) — with one-hot identity
as the reference. The numeric ordering of the identifiers reflects the order in which they were
registered, not a ranking or a ladder.

`late_fusion` names the fusion point of the learned embedding and is not related to the
fusion-locus numbering used in earlier code; `film`, `early_injection`, and `message_level`
belong to that earlier numbering and were not used in this study.

## Datasets and partitions

| in this repository | in the paper |
|---|---|
| `b1`, `runs_b1`, `data_b1`, `build_b1.py` | the ECOTOX-external set |
| "expansion", `PREREG_EXPANSION.md`, `ecotox_expansion_*` | the construction of the ECOTOX-external set and the analyses on it |
| `Phase 1`, `P` | the ADORE data and the analyses run on it |
| `discovery` | the single-condition partition (LC50 at 96 h; 779 species) |
| `replication` | the seven-condition partition (LC50/EC50 at 24–96 h; 1,006 species) |
| `E-full`, `corrected E-full` | the June 2026 ECOTOX pull before subtraction of the ADORE records |

## Splits

| in this repository | in the paper |
|---|---|
| `group` | grouping by compound registry number (CASRN) |
| `scaffold`, `scaffold_murcko` | Bemis–Murcko scaffold split |
| `scaffold_generic` | generic scaffold split |
| `designed_leaky` | the deliberately leaky split retained as a positive control |
| `warm` | the warm regime |
| `cold`, `block B` | the species-cold regime |
| `cross-group` | cross-group evaluation, removed before unblinding |

## Statistics

| in this repository | in the paper |
|---|---|
| `δ`, `delta_primary` | the per-seed margin |
| `δ′`, `delta_prime` | the ensemble margin |
| `δ_det`, `delta_det` | the margin for deterministic models |
| `dd`, `DD`, `4-arm DD` | the per-seed paired difference-in-differences |
| `D16` | the screening rule fixed before training (Section 3.1) |
| `q_family` | the Benjamini–Hochberg adjusted p-value within a family |

## Project labels with no scientific meaning

`q2`, `v3`, `v4`, `src`, `s4_*`, `s5_*`, `s6_*`, `s7_s8`, `GAP`, `blockA`, `blockB`,
`director`. These are internal project, revision, and section labels. `src` in particular is
a package name inherited from an earlier submission target and does not indicate the venue of
this work.

## Components of the bundled `ccmpnn` package that this study does not use

`ccmpnn/` is retained in full so that the released code matches the code that was executed.
Several of its components belong to an earlier, discontinued line of work and play no part in
the results reported in the paper.

| component | status in this study |
|---|---|
| `DMPNNModel`, `MPNEncoder` | **used** — the directed message-passing backbone |
| `MolGraph`, `assemble_batch` | **used** — molecular graph representation, shared by both backbones |
| `LateFusionContext`, `SpeciesEncoder`, `ContextBase` | **used** — the learned-embedding species path |
| `perf_metrics`, `species_binned_rmse` | **used** — evaluation metrics |
| `CCMPNNContext` | **not used** — message-level conditioning; the mechanism the package is named after |
| `FiLMContext` | **not used** — corresponds to the `film` injection |
| `experiment`, `train`, `gbdt`, `split`, `data` modules | **not used** — `src` has its own runner and splits |

The corresponding injection variants `film`, `early_injection`, and `message_level` are likewise
present in `src/models.py` but were not trained; the seven representations evaluated in the
paper are the ones listed at the top of this document.
