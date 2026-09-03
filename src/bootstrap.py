from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.stats import (
    approximate_sign_p_value,
    improvement_probability,
    metric_value,
    practical_effect_category,
)


PAIR_MERGE_KEYS = [
    "smiles",
    "species",
    "compound_key",
    "scaffold_key",
    "true_log10",
    "split",
    "seed",
]


@dataclass(frozen=True)
class PairResult:
    paired: pd.DataFrame | None
    warning: str | None


def align_predictions(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    merge_keys: list[str] | None = None,
) -> PairResult:
    merge_keys = merge_keys or PAIR_MERGE_KEYS
    missing = [
        key
        for key in merge_keys
        if key not in candidate.columns or key not in reference.columns
    ]
    if missing:
        return PairResult(None, f"missing merge keys: {missing}")

    if candidate.duplicated(merge_keys).any():
        return PairResult(None, "candidate has duplicate merge keys")
    if reference.duplicated(merge_keys).any():
        return PairResult(None, "reference has duplicate merge keys")

    true_in_merge_keys = "true_log10" in merge_keys
    cand_cols = merge_keys + ["pred_log10"]
    ref_cols = merge_keys + ["pred_log10"]
    if not true_in_merge_keys and "true_log10" in candidate.columns and "true_log10" in reference.columns:
        cand_cols.append("true_log10")
        ref_cols.append("true_log10")
    cand_rename = {"pred_log10": "pred_log10_candidate"}
    ref_rename = {"pred_log10": "pred_log10_reference"}
    if not true_in_merge_keys:
        cand_rename["true_log10"] = "true_log10_candidate"
        ref_rename["true_log10"] = "true_log10_reference"
    cand = candidate[cand_cols].rename(columns=cand_rename)
    ref = reference[ref_cols].rename(columns=ref_rename)
    paired = cand.merge(ref, on=merge_keys, how="inner", validate="one_to_one")
    if len(paired) != len(candidate) or len(paired) != len(reference):
        return PairResult(
            None,
            f"paired row mismatch: candidate={len(candidate)}, reference={len(reference)}, merged={len(paired)}",
        )
    if not true_in_merge_keys:
        if "true_log10_candidate" not in paired.columns or "true_log10_reference" not in paired.columns:
            return PairResult(None, "true_log10 missing after merge")
        max_abs_target_diff = float(
            np.max(np.abs(paired["true_log10_candidate"] - paired["true_log10_reference"]))
        )
        if max_abs_target_diff > 1e-6:
            return PairResult(None, f"true_log10 mismatch after merge: max_abs_diff={max_abs_target_diff}")
        paired["true_log10"] = paired["true_log10_candidate"]
    return PairResult(paired, None)


def block_bootstrap_metric_delta(
    paired: pd.DataFrame,
    *,
    block_key: str,
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> dict:
    if block_key not in paired.columns:
        raise ValueError(f"block key missing from paired dataframe: {block_key!r}")
    if paired[block_key].isna().any():
        raise ValueError(f"block key contains null values: {block_key!r}")
    if len(paired) == 0:
        raise ValueError("paired dataframe is empty")

    y_true = paired["true_log10"].to_numpy(np.float64)
    pred_candidate = paired["pred_log10_candidate"].to_numpy(np.float64)
    pred_reference = paired["pred_log10_reference"].to_numpy(np.float64)
    candidate_metric = metric_value(y_true, pred_candidate, metric)
    reference_metric = metric_value(y_true, pred_reference, metric)
    delta = candidate_metric - reference_metric

    grouped_indices = [
        idx.to_numpy(dtype=np.int64)
        for _, idx in paired.groupby(block_key, sort=False).groups.items()
    ]
    n_blocks = len(grouped_indices)
    rng = np.random.default_rng(seed)
    boot_deltas = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sampled_blocks = rng.integers(0, n_blocks, size=n_blocks)
        sampled_idx = np.concatenate([grouped_indices[j] for j in sampled_blocks])
        sample_true = y_true[sampled_idx]
        sample_candidate = pred_candidate[sampled_idx]
        sample_reference = pred_reference[sampled_idx]
        boot_deltas[i] = metric_value(sample_true, sample_candidate, metric) - metric_value(
            sample_true,
            sample_reference,
            metric,
        )

    ci_low, ci_high = np.percentile(boot_deltas, [2.5, 97.5])
    return {
        "metric": metric,
        "candidate_metric": candidate_metric,
        "reference_metric": reference_metric,
        "delta": float(delta),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_blocks": int(n_blocks),
        "n_rows": int(len(paired)),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_probability_improvement": improvement_probability(boot_deltas, metric),
        "p_value_approx": approximate_sign_p_value(boot_deltas),
        "practical_effect_category": practical_effect_category(metric, float(delta)),
    }
