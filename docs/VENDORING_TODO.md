# Remaining steps before release

The repository as committed here is complete except for the `ccmpnn` package itself, which
must be copied in. Everything else has been prepared for that layout.

## 1. Copy the package

Copy **`ccmpnn/` and `tests/` only** from the CC-MPNN working directory to the root of this
repository. Exclude: `JMGM_*.docx`, `실험_데이터_결과*.docx`, `HANDOFF.md`, `claude/`,
`claude.md`, `codex.md`, `project_decisions*.md`, `scripts/`, `data/`, and all `__pycache__/`.

Resulting layout:

```
species-representation-toxicity/
  ccmpnn/          <- copied
  tests/           <- copied
  src/
  scripts/
  results/
  docs/
```

`src/paths.py` already resolves `ccmpnn` at the repository root and falls back to a sibling
`../CC-MPNN` checkout for development installs. It raises a clear error if neither is present.

## 2. Add the licence

Add `ccmpnn/LICENSE` (Apache License 2.0, same as the parent repository) and record authorship
in `ccmpnn/NOTICE`. The package currently carries no licence, which means it cannot legally be
redistributed as it stands.

Record the upstream commit hash of the copied state in `ccmpnn/NOTICE` so that the bundled copy
can be identified later.

## 3. Recover the missing status document

`src/models.py` refers twice to `results/q2_v4/runs/replication/TASK_D_STATUS.md`, which is
excluded by `.gitignore`. It documents why `GNN_STRATUM_EXPOSURE` is disabled — a decision that
supports the paper's claim that the two backbones are configured identically. Either commit it
(preferably, under `results/q2_v4/audit/`) or remove the references.

## 4. Smoke test the released copy

Run the pipeline end to end from a fresh clone of **this** repository, not from the working
directory. The released tree and the working tree have diverged before: several scripts were
released with placeholder paths (`<DATA_ROOT>`, `<ECOTOX_DATA_DIR>`, `<USER_HOME>`) that would
have failed at run time. Those are now environment variables, but only a clean-clone run
confirms the repository is self-contained.

Environment variables the pipeline reads:

| variable | purpose |
|---|---|
| `ADORE_ROOT` | root of the ADORE benchmark distribution (`processed/`, `chemicals/`) |
| `ADORE_PROCESSED` | path to `processed/ecotox_mortality_processed.csv`, for `make_p_keys.py` |
| `TOX_LEARN_ROOT` | optional; unused by the reported runs |
| `SCRATCH_DIR` | optional scratch directory for `verify_refactor_byte_identity.py` |

## 5. Fill in the deposit DOI

`README.md` and `NOTICE` contain `[DOI]` placeholders for the Zenodo deposit of the
ECOTOX-external table. Fill them in after publishing the deposit.

## 6. Tag the submitted state

Create a tag (for example `v1.0-submission`) and cite that tag's URL in the manuscript rather
than a bare `main` URL, so that the state corresponding to the paper is fixed.

---

Delete this file once the steps above are done.
