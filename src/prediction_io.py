"""Single-source-of-truth enforcement for per-run prediction CSVs (director rule, Session 26).

Stratum assignment and ALL aggregation metadata must be DERIVED FROM THE DATASET (the canonical
{split}_train.csv / {split}_test.csv). A per-run prediction CSV may be read ONLY for join keys,
predictions, truths, and run identifiers — NEVER for metadata columns (taxonomy, ncbi_*, tax_group,
traits, coverage counts, effect_value, ...). Prediction CSVs written before the Aug-1 NCBI join fix
carry NULL ncbi_* columns; stratifying on a prediction-CSV metadata column would silently misclassify
those species into an 'unresolved' stratum for pre-fix cells — the SAME failure shape as the NCBI bug.

`load_prediction_csv` enforces the whitelist at load time: requesting any non-whitelist column raises
PredictionColumnViolation. Loud failure beats a silent, plausible-looking wrong result.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

# join / compound-identity keys (all deterministic from smiles/species/endpoint/duration; stale-immune)
PRED_JOIN_KEYS = frozenset({
    "smiles", "species", "endpoint", "duration",
    "species_idx", "species_idx_original", "source_row_id", "compound_key", "scaffold_key",
})
PRED_VALUES = frozenset({"pred_log10", "true_log10", "target_log10", "error_log10"})
PRED_RUN_IDS = frozenset({
    "model_name", "backbone", "variant", "species_mode",
    "injection_location", "species_control", "split", "seed",
})
PREDICTION_WHITELIST = PRED_JOIN_KEYS | PRED_VALUES | PRED_RUN_IDS

# Columns that MUST come from the dataset, never a prediction CSV (explicit for clear error messages).
FORBIDDEN_METADATA = frozenset({
    "effect_value", "n_source_rows", "n_cas", "cas_list",
    "tax_class", "tax_order", "tax_family", "tax_genus", "tax_group",
    "ncbi_taxid", "ncbi_resolved", "ncbi_class", "ncbi_order", "ncbi_family", "ncbi_genus",
    "species_for_model", "input_species", "species_control_type",
    "is_shuffled", "is_zero_species", "is_dummy_species", "scaffold",
})


class PredictionColumnViolation(RuntimeError):
    """Raised when aggregation tries to read a non-whitelist column from a prediction CSV."""


def assert_prediction_columns(columns, *, source: str = "prediction CSV") -> None:
    """Fail loudly if `columns` contains anything outside the prediction whitelist."""
    bad = [c for c in columns if c not in PREDICTION_WHITELIST]
    if bad:
        raise PredictionColumnViolation(
            f"aggregation must not read {sorted(bad)} from {source}; stratum/metadata must be joined "
            f"from the dataset ({{split}}_train/test.csv). Whitelist = join keys + pred/true + run ids.")


def load_prediction_csv(path: str | Path, columns=None) -> pd.DataFrame:
    """Read a per-run prediction CSV, enforcing the whitelist.

    columns=None  -> load only the whitelisted columns present in the file (forbidden cols dropped).
    columns=[...] -> load exactly those; raises PredictionColumnViolation if any is non-whitelist.
    """
    path = Path(path)
    if columns is not None:
        assert_prediction_columns(columns, source=f"prediction CSV {path.name}")
        return pd.read_csv(path, usecols=list(columns))
    header = pd.read_csv(path, nrows=0).columns
    keep = [c for c in header if c in PREDICTION_WHITELIST]
    return pd.read_csv(path, usecols=keep)
