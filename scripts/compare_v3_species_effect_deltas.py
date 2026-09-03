from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.stats import approximate_sign_p_value, bh_fdr, metric_value


MERGE_KEYS = ["smiles", "species", "compound_key", "scaffold_key", "split", "seed"]
METRICS = ["rmse", "mae"]


def _load_prediction(prediction_dir: Path, *, backbone: str, species_mode: str, split: str, seed: int) -> pd.DataFrame:
    matches = []
    for path in prediction_dir.glob("*.csv"):
        df0 = pd.read_csv(path, nrows=1)
        if df0.empty:
            continue
        row = df0.iloc[0]
        if (
            str(row.get("backbone")) == backbone
            and str(row.get("species_mode")) == species_mode
            and str(row.get("split")) == split
            and int(row.get("seed")) == int(seed)
        ):
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one prediction for backbone={backbone}, species_mode={species_mode}, "
            f"split={split}, seed={seed}; found {len(matches)}"
        )
    return pd.read_csv(matches[0])


def _merge_four(
    *,
    gnn_candidate: pd.DataFrame,
    gnn_reference: pd.DataFrame,
    calib_candidate: pd.DataFrame,
    calib_reference: pd.DataFrame,
) -> pd.DataFrame:
    base = gnn_candidate[MERGE_KEYS + ["true_log10", "pred_log10"]].rename(
        columns={"true_log10": "true_gnn_candidate", "pred_log10": "pred_gnn_candidate"}
    )
    frames = [
        (
            gnn_reference[MERGE_KEYS + ["true_log10", "pred_log10"]].rename(
                columns={"true_log10": "true_gnn_reference", "pred_log10": "pred_gnn_reference"}
            ),
            "gnn_reference",
        ),
        (
            calib_candidate[MERGE_KEYS + ["true_log10", "pred_log10"]].rename(
                columns={"true_log10": "true_calib_candidate", "pred_log10": "pred_calib_candidate"}
            ),
            "calib_candidate",
        ),
        (
            calib_reference[MERGE_KEYS + ["true_log10", "pred_log10"]].rename(
                columns={"true_log10": "true_calib_reference", "pred_log10": "pred_calib_reference"}
            ),
            "calib_reference",
        ),
    ]
    merged = base
    for frame, name in frames:
        if frame.duplicated(MERGE_KEYS).any():
            raise ValueError(f"{name} has duplicate merge keys")
        merged = merged.merge(frame, on=MERGE_KEYS, how="inner", validate="one_to_one")
    expected = len(gnn_candidate)
    if len(merged) != expected:
        raise ValueError(f"merged row mismatch: expected={expected}, merged={len(merged)}")
    true_cols = [
        "true_gnn_candidate",
        "true_gnn_reference",
        "true_calib_candidate",
        "true_calib_reference",
    ]
    true = merged[true_cols[0]].to_numpy(np.float64)
    max_diff = max(float(np.max(np.abs(merged[col].to_numpy(np.float64) - true))) for col in true_cols[1:])
    if max_diff > 1e-6:
        raise ValueError(f"true_log10 mismatch after merge: max_abs_diff={max_diff}")
    merged["true_log10"] = true
    return merged


def _delta_of_deltas(paired: pd.DataFrame, *, metric: str) -> tuple[float, float, float]:
    y = paired["true_log10"].to_numpy(np.float64)
    gnn_delta = metric_value(y, paired["pred_gnn_candidate"].to_numpy(np.float64), metric) - metric_value(
        y, paired["pred_gnn_reference"].to_numpy(np.float64), metric
    )
    calib_delta = metric_value(y, paired["pred_calib_candidate"].to_numpy(np.float64), metric) - metric_value(
        y, paired["pred_calib_reference"].to_numpy(np.float64), metric
    )
    return float(gnn_delta), float(calib_delta), float(gnn_delta - calib_delta)


def _bootstrap(
    paired: pd.DataFrame,
    *,
    block_key: str,
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> dict:
    gnn_delta, calib_delta, diff = _delta_of_deltas(paired, metric=metric)
    grouped_indices = [
        idx.to_numpy(dtype=np.int64)
        for _, idx in paired.groupby(block_key, sort=False).groups.items()
    ]
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sampled_blocks = rng.integers(0, len(grouped_indices), size=len(grouped_indices))
        sampled_idx = np.concatenate([grouped_indices[j] for j in sampled_blocks])
        sample = paired.iloc[sampled_idx]
        _, _, samples[i] = _delta_of_deltas(sample, metric=metric)
    ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
    return {
        "metric": metric,
        "gnn_species_effect_delta": gnn_delta,
        "naive_calibration_effect_delta": calib_delta,
        "delta_of_deltas": diff,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value_approx": approximate_sign_p_value(samples),
        "n_blocks": int(len(grouped_indices)),
        "n_rows": int(len(paired)),
        "n_bootstrap": int(n_bootstrap),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", default="results/src/compound_random_core_full/predictions")
    parser.add_argument("--out-dir", default="results/src/compound_random_core_full/summary_tables")
    parser.add_argument("--split", default="compound_random")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--block-key", default="compound_key")
    args = parser.parse_args()

    prediction_dir = Path(args.prediction_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds or [0, 1, 2, 3, 4]
    rows = []
    warnings = []
    for seed in seeds:
        for backbone in ["dmpnn", "graphconv"]:
            try:
                paired = _merge_four(
                    gnn_candidate=_load_prediction(
                        prediction_dir,
                        backbone=backbone,
                        species_mode="species_bias_only",
                        split=args.split,
                        seed=seed,
                    ),
                    gnn_reference=_load_prediction(
                        prediction_dir,
                        backbone=backbone,
                        species_mode="no_species",
                        split=args.split,
                        seed=seed,
                    ),
                    calib_candidate=_load_prediction(
                        prediction_dir,
                        backbone="lightgbm_rdkit",
                        species_mode="species_residual_calibration",
                        split=args.split,
                        seed=seed,
                    ),
                    calib_reference=_load_prediction(
                        prediction_dir,
                        backbone="lightgbm_rdkit",
                        species_mode="no_species_oof_base",
                        split=args.split,
                        seed=seed,
                    ),
                )
            except Exception as exc:
                warnings.append({"seed": seed, "backbone": backbone, "warning": str(exc)})
                continue
            for metric in METRICS:
                rows.append(
                    {
                        "backbone": backbone,
                        "split": args.split,
                        "seed": seed,
                        "comparison_family": "species_effect_delta_vs_naive_calibration",
                        **_bootstrap(
                            paired,
                            block_key=args.block_key,
                            metric=metric,
                            n_bootstrap=args.n_bootstrap,
                            seed=int(seed) + 7919 * (1 + len(rows)),
                        ),
                    }
                )
    raw = pd.DataFrame(rows)
    if len(raw):
        raw["q_value_bh_fdr"] = bh_fdr(raw["p_value_approx"])
        raw["significant_fdr_0_05"] = raw["q_value_bh_fdr"] <= 0.05
    raw_path = out_dir / "gnn_species_effect_delta_vs_naive_calibration_raw.csv"
    summary_path = out_dir / "gnn_species_effect_delta_vs_naive_calibration_summary.csv"
    warnings_path = out_dir / "gnn_species_effect_delta_vs_naive_calibration_warnings.json"
    raw.to_csv(raw_path, index=False)
    summary = raw.copy()
    if len(summary):
        summary["favorable_gnn_effect"] = summary["delta_of_deltas"] < 0
        summary = (
            summary.groupby(["backbone", "metric"], as_index=False)
            .agg(
                median_delta_of_deltas=("delta_of_deltas", "median"),
                mean_delta_of_deltas=("delta_of_deltas", "mean"),
                median_gnn_effect_delta=("gnn_species_effect_delta", "median"),
                median_naive_calibration_effect_delta=("naive_calibration_effect_delta", "median"),
                favorable_seeds=("favorable_gnn_effect", "sum"),
                fdr_sig_seeds=("significant_fdr_0_05", "sum"),
                ci_low_median=("ci_low", "median"),
                ci_high_median=("ci_high", "median"),
                n_seeds=("delta_of_deltas", "size"),
            )
        )
    summary.to_csv(summary_path, index=False)
    with open(warnings_path, "w", encoding="utf-8") as f:
        json.dump({"warnings": warnings, "n_warnings": len(warnings)}, f, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                "raw": str(raw_path),
                "summary": str(summary_path),
                "warnings": str(warnings_path),
                "rows": int(len(raw)),
                "warnings_count": len(warnings),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
