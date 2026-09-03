# Chronology of the pre-registration and the margin freeze

This document exists because the repository contains more than one version of two
verdict-determining records. Both are retained deliberately. This is what each one is.

## Family assignment

`results/q2_v4/audit/fdr_family_boundary_DRAFT.md` — written 2026-07-30, **before training**.
It fixes the primary and exploratory families provisionally and explicitly lists three questions
it does not resolve: whether the seven-condition partition belongs to the primary or the
exploratory family, whether the species-cold and cross-group stages should be promoted to
primary, and which split scheme is primary. The draft records that these were not to be assigned
without an explicit decision.

`results/q2_v4/audit/fdr_family_boundary.md` — the resolution of those three questions, recorded
2026-07-30 (before training), with the comparison set for the primary equivalence family
finalised 2026-08-03, before any comparison output existed. The one substantive change relative
to the draft is the addition of NCBI taxonomy versus one-hot identity to the primary equivalence
set, so that both taxonomy sources are registered rather than only one.

The draft is retained because it is the timestamped evidence that these questions were open
before training and were closed before results were seen. Removing it would remove that evidence.

`PREREGISTRATION_draft.md` stands in the same relation to `PREREGISTRATION.md`.

## The margin freeze

The primary margin was frozen on 2026-08-03 at δ = 0.019776561636 (14 cells, 126 degrees of
freedom).

`results/q2_v4/audit/delta_primary_frozen_v1_evidencebug_20260803.json` (frozen 05:05:03 UTC) and
`results/q2_v4/audit/delta_primary_frozen.json` (re-issued 06:23:17 UTC) are **identical in the
margin and in every per-cell value**. The margin was not re-derived and no comparison output
existed at either time.

What changed is the freeze-evidence block. The first version recorded
`bootstrap_outputs_exist: true` because `compute_freeze_delta.py` tested for the existence of the
`runs/bootstrap` directory rather than for its contents; the directory existed as an empty
placeholder created on 29 July and held zero files. The re-issued version records
`bootstrap_output_file_count: 0` together with the (empty) file listing.

The correction therefore strengthens rather than weakens the pre-freeze evidence: the original
field could be misread as indicating that comparison outputs existed, and the corrected field
shows directly that none did. The superseded file is kept so that this can be verified rather
than taken on trust.

## Analyses specified after unblinding

The paper lists these in its Limitations: the rank correlation, the attenuation control, the
species-axis variance-capture estimates, the concentration-fold interpretation, the comparison of
one-hot identity with the species mean offset (`run_posthoc_t2_vs_t1.py`,
`POSTHOC_t2_vs_t1_REPORT.txt`), the benchmark coverage analysis, the solubility check
(`mechanism_b5_solubility.json`), and the decomposition of per-seed variance by processor
(`expansion_device_diagnostic.json`). They do not carry the status of the pre-registered verdicts.
