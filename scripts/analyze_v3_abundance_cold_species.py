from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bootstrap import align_predictions, block_bootstrap_metric_delta
from src.stats import bh_fdr, prediction_metrics


MERGE_KEYS = ["smiles", "species", "compound_key", "scaffold_key", "split", "seed"]
METRICS = ["rmse", "mae"]

ABUNDANCE_BINS = [
    ("cold", 0, 0),
    ("very_rare", 1, 4),
    ("rare", 5, 9),
    ("low_support", 10, 49),
    ("moderate_support", 50, 199),
    ("high_support", 200, None),
]

COMPARISONS = [
    {
        "comparison_family": "gnn_bias_vs_no_species",
        "candidate_backbone": "dmpnn",
        "candidate_species_mode": "species_bias_only",
        "reference_backbone": "dmpnn",
        "reference_species_mode": "no_species",
    },
    {
        "comparison_family": "gnn_bias_vs_no_species",
        "candidate_backbone": "graphconv",
        "candidate_species_mode": "species_bias_only",
        "reference_backbone": "graphconv",
        "reference_species_mode": "no_species",
    },
    {
        "comparison_family": "late_fusion_vs_no_species",
        "candidate_backbone": "dmpnn",
        "candidate_species_mode": "true_species_late_fusion",
        "reference_backbone": "dmpnn",
        "reference_species_mode": "no_species",
    },
    {
        "comparison_family": "late_fusion_vs_no_species",
        "candidate_backbone": "graphconv",
        "candidate_species_mode": "true_species_late_fusion",
        "reference_backbone": "graphconv",
        "reference_species_mode": "no_species",
    },
    {
        "comparison_family": "residual_calibration_vs_base",
        "candidate_backbone": "lightgbm_rdkit",
        "candidate_species_mode": "species_residual_calibration",
        "reference_backbone": "lightgbm_rdkit",
        "reference_species_mode": "no_species_oof_base",
    },
    {
        "comparison_family": "species_mean_vs_global_mean",
        "candidate_backbone": "naive_species",
        "candidate_species_mode": "species_mean",
        "reference_backbone": "naive_species",
        "reference_species_mode": "global_mean",
    },
    {
        "comparison_family": "lightgbm_categorical_vs_no_species",
        "candidate_backbone": "lightgbm_rdkit",
        "candidate_species_mode": "species_categorical",
        "reference_backbone": "lightgbm_rdkit",
        "reference_species_mode": "no_species",
    },
]


def _bin_label(count: int) -> str:
    count = int(count)
    for label, lo, hi in ABUNDANCE_BINS:
        if count >= lo and (hi is None or count <= hi):
            return label
    raise ValueError(f"unhandled species count: {count}")


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _train_counts(data_dir: Path, split: str) -> dict[int, int]:
    train = pd.read_csv(data_dir / f"{split}_train.csv")
    return train["species_idx"].astype(int).value_counts().to_dict()


def _add_abundance(df: pd.DataFrame, counts: dict[int, int]) -> pd.DataFrame:
    species_col = "species_idx_original" if "species_idx_original" in df.columns else "species_idx"
    out = df.copy()
    out["species_train_count"] = out[species_col].astype(int).map(counts).fillna(0).astype(int)
    out["abundance_bin"] = out["species_train_count"].map(_bin_label)
    out["is_cold_species"] = out["species_train_count"].eq(0)
    return out


def _prediction_index(prediction_dir: Path, *, split: str) -> dict[tuple[str, str, int], pd.DataFrame]:
    out: dict[tuple[str, str, int], pd.DataFrame] = {}
    for path in sorted(prediction_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0]
        if str(row["split"]) != split:
            continue
        key = (str(row["backbone"]), str(row["species_mode"]), int(row["seed"]))
        out[key] = df.copy()
    return out


def _metric_rows(indexed: dict[tuple[str, str, int], pd.DataFrame], counts: dict[int, int]) -> list[dict]:
    rows = []
    for (backbone, species_mode, seed), df in sorted(indexed.items()):
        enriched = _add_abundance(df, counts)
        row0 = enriched.iloc[0]
        for label, _, _ in ABUNDANCE_BINS:
            part = enriched[enriched["abundance_bin"].eq(label)]
            if part.empty:
                continue
            rows.append(
                {
                    "backbone": backbone,
                    "species_mode": species_mode,
                    "model_name": str(row0.get("model_name", "")),
                    "split": str(row0["split"]),
                    "seed": int(seed),
                    "abundance_bin": label,
                    "n_rows": int(len(part)),
                    "n_species": int(part["species_idx_original"].nunique())
                    if "species_idx_original" in part.columns
                    else int(part["species_idx"].nunique()),
                    "n_compounds": int(part["compound_key"].nunique()),
                    **prediction_metrics(part),
                }
            )
    return rows


def _bin_counts_from_test(data_dir: Path, split: str, counts: dict[int, int]) -> pd.DataFrame:
    test = pd.read_csv(data_dir / f"{split}_test.csv")
    enriched = _add_abundance(test, counts)
    rows = []
    for label, lo, hi in ABUNDANCE_BINS:
        part = enriched[enriched["abundance_bin"].eq(label)]
        rows.append(
            {
                "abundance_bin": label,
                "train_count_min": lo,
                "train_count_max": "inf" if hi is None else hi,
                "n_test_rows": int(len(part)),
                "n_test_species": int(part["species_idx"].nunique()) if len(part) else 0,
                "n_test_compounds": int(part["smiles"].nunique()) if len(part) else 0,
            }
        )
    return pd.DataFrame(rows)


def _comparison_rows(
    indexed: dict[tuple[str, str, int], pd.DataFrame],
    counts: dict[int, int],
    *,
    split: str,
    seeds: list[int],
    n_bootstrap: int,
    block_key: str,
) -> tuple[list[dict], list[dict]]:
    rows = []
    warnings = []
    n_completed = 0
    for seed in seeds:
        for comparison in COMPARISONS:
            cand_key = (
                comparison["candidate_backbone"],
                comparison["candidate_species_mode"],
                int(seed),
            )
            ref_key = (
                comparison["reference_backbone"],
                comparison["reference_species_mode"],
                int(seed),
            )
            cand = indexed.get(cand_key)
            ref = indexed.get(ref_key)
            if cand is None or ref is None:
                warnings.append({"seed": seed, **comparison, "warning": "missing prediction file"})
                continue
            cand = _add_abundance(cand, counts)
            ref = _add_abundance(ref, counts)
            for label, _, _ in ABUNDANCE_BINS:
                cand_part = cand[cand["abundance_bin"].eq(label)].copy()
                ref_part = ref[ref["abundance_bin"].eq(label)].copy()
                if cand_part.empty or ref_part.empty:
                    warnings.append({"seed": seed, "abundance_bin": label, **comparison, "warning": "empty bin"})
                    continue
                pair = align_predictions(cand_part, ref_part, merge_keys=MERGE_KEYS)
                if pair.warning:
                    warnings.append({"seed": seed, "abundance_bin": label, **comparison, "warning": pair.warning})
                    continue
                assert pair.paired is not None
                if pair.paired[block_key].nunique() < 2:
                    warnings.append(
                        {
                            "seed": seed,
                            "abundance_bin": label,
                            **comparison,
                            "warning": f"too few bootstrap blocks: {pair.paired[block_key].nunique()}",
                        }
                    )
                    continue
                n_completed += 1
                for metric in METRICS:
                    result = block_bootstrap_metric_delta(
                        pair.paired,
                        block_key=block_key,
                        metric=metric,
                        n_bootstrap=n_bootstrap,
                        seed=int(seed) + 1291 * n_completed + 37 * METRICS.index(metric),
                    )
                    rows.append(
                        {
                            "split": split,
                            "seed": int(seed),
                            "abundance_bin": label,
                            **comparison,
                            **result,
                        }
                    )
    return rows, warnings


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    data = raw.copy()
    data["favorable"] = data["delta"] < 0
    return (
        data.groupby(
            [
                "comparison_family",
                "candidate_backbone",
                "candidate_species_mode",
                "reference_backbone",
                "reference_species_mode",
                "abundance_bin",
                "metric",
            ],
            as_index=False,
        )
        .agg(
            median_delta=("delta", "median"),
            mean_delta=("delta", "mean"),
            median_candidate_metric=("candidate_metric", "median"),
            median_reference_metric=("reference_metric", "median"),
            favorable_seeds=("favorable", "sum"),
            fdr_sig_seeds=("significant_fdr_0_05", "sum"),
            ci_low_median=("ci_low", "median"),
            ci_high_median=("ci_high", "median"),
            n_seeds=("delta", "size"),
            n_rows_median=("n_rows", "median"),
            n_blocks_median=("n_blocks", "median"),
        )
        .sort_values(["comparison_family", "candidate_backbone", "metric", "abundance_bin"])
    )


def _write_markdown(path: Path, bin_counts: pd.DataFrame, comparison_summary: pd.DataFrame) -> None:
    cold = comparison_summary[comparison_summary["abundance_bin"].eq("cold")]
    lines = [
        "# Abundance And Cold-Species Confirmatory Analysis",
        "",
        "This analysis uses pre-specified train-species-count bins; bins were not selected after inspecting model results.",
        "",
        "Expected direction before looking at results: if the species effect is observed-species calibration, improvement should concentrate in higher-support species and collapse for cold species.",
        "",
        "## Pre-Specified Bins",
        "",
        "```csv",
        bin_counts.to_csv(index=False).strip(),
        "```",
        "",
        "## Comparison Summary",
        "",
        "Deltas are candidate minus reference; negative favors the species-aware or calibrated candidate.",
        "",
        "```csv",
        comparison_summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Cold-Species Rows Only",
        "",
        "Cold species have zero training rows. Species-mean and residual-calibration baselines use their documented fallback behavior here, so cold-species improvement should be interpreted as a diagnostic for leakage or unintended generalization.",
        "",
        "```csv",
        cold.to_csv(index=False).strip(),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--prediction-dir", default="results/src/compound_random_core_full/predictions")
    parser.add_argument("--data-dir", default="results/src/clean_splits")
    parser.add_argument("--out-dir", default="results/src/compound_random_core_full/summary_tables")
    parser.add_argument("--split", default="compound_random")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--block-key", default="compound_key")
    args = parser.parse_args()

    config = _load_config(Path(args.config) if args.config else None)
    prediction_dir = Path(config.get("prediction_root", args.prediction_dir))
    data_dir = Path(config.get("dataset_path", args.data_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = args.split
    seeds = args.seeds or [0, 1, 2, 3, 4]
    counts = _train_counts(data_dir, split)
    indexed = _prediction_index(prediction_dir, split=split)

    bin_counts = _bin_counts_from_test(data_dir, split, counts)
    metrics = pd.DataFrame(_metric_rows(indexed, counts))
    comparison_rows, warnings = _comparison_rows(
        indexed,
        counts,
        split=split,
        seeds=seeds,
        n_bootstrap=int(args.n_bootstrap),
        block_key=args.block_key,
    )
    raw = pd.DataFrame(comparison_rows)
    if len(raw):
        raw["q_value_bh_fdr"] = bh_fdr(raw["p_value_approx"])
        raw["significant_fdr_0_05"] = raw["q_value_bh_fdr"] <= 0.05
    summary = _summarize(raw)
    cold = summary[summary["abundance_bin"].eq("cold")].copy() if len(summary) else summary

    paths = {
        "bin_counts": out_dir / "abundance_bin_counts.csv",
        "metrics": out_dir / "abundance_bin_metrics.csv",
        "raw": out_dir / "abundance_bin_comparisons_raw.csv",
        "fdr": out_dir / "abundance_bin_comparisons_fdr.csv",
        "summary": out_dir / "abundance_bin_comparison_summary.csv",
        "cold": out_dir / "cold_species_subset_summary.csv",
        "warnings": out_dir / "abundance_cold_species_warnings.json",
        "markdown": out_dir / "abundance_cold_species_summary.md",
    }
    bin_counts.to_csv(paths["bin_counts"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    raw.to_csv(paths["raw"], index=False)
    raw.to_csv(paths["fdr"], index=False)
    summary.to_csv(paths["summary"], index=False)
    cold.to_csv(paths["cold"], index=False)
    with open(paths["warnings"], "w", encoding="utf-8") as f:
        json.dump({"warnings": warnings, "n_warnings": len(warnings)}, f, ensure_ascii=False, indent=2)
    _write_markdown(paths["markdown"], bin_counts, summary)
    print(
        json.dumps(
            {
                "outputs": {key: str(value) for key, value in paths.items()},
                "n_metric_rows": int(len(metrics)),
                "n_comparison_rows": int(len(raw)),
                "n_summary_rows": int(len(summary)),
                "n_warnings": len(warnings),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
