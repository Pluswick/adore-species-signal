from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("rmse", "mae", "within_2fold", "within_3fold")
LOWER_IS_BETTER = {"rmse", "mae"}
HIGHER_IS_BETTER = {"within_2fold", "within_3fold"}


def _metric(df: pd.DataFrame, metric: str, pred_col: str = "pred_log10") -> float:
    if len(df) == 0:
        return float("nan")
    err = df[pred_col].astype(float) - df["true_log10"].astype(float)
    if metric == "rmse":
        return float(math.sqrt((err**2).mean()))
    if metric == "mae":
        return float(err.abs().mean())
    if metric == "within_2fold":
        return float((err.abs() <= math.log10(2.0)).mean())
    if metric == "within_3fold":
        return float((err.abs() <= math.log10(3.0)).mean())
    raise ValueError(metric)


def _metrics(df: pd.DataFrame, pred_col: str = "pred_log10") -> dict:
    return {metric: _metric(df, metric, pred_col=pred_col) for metric in METRICS}


def _label(row: pd.Series) -> str:
    baseline = str(row.get("baseline", "") or "")
    if baseline and baseline != "nan":
        return baseline
    return str(row["species_mode"])


def _read_prediction(path: Path, train_smiles: set[str], train_cas: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["compound_overlap_smiles"] = df["smiles"].dropna().astype(str).isin(train_smiles).reindex(df.index, fill_value=False)
    df["compound_overlap_cas"] = df["CAS"].dropna().astype(str).isin(train_cas).reindex(df.index, fill_value=False)
    df["leakage_subset"] = df["compound_overlap_smiles"].map({True: "train_smiles_overlap", False: "train_smiles_disjoint"})
    if "source_row_id" not in df.columns:
        df["source_row_id"] = range(len(df))
    return df


def _load_index(prediction_dir: Path, data_dir: Path) -> dict[tuple[str, int, str, str], pd.DataFrame]:
    index: dict[tuple[str, int, str, str], pd.DataFrame] = {}
    train_cache: dict[str, tuple[set[str], set[str]]] = {}
    for path in sorted(prediction_dir.glob("*.csv")):
        sample = pd.read_csv(path, nrows=1)
        if sample.empty:
            continue
        split = str(sample["split"].iloc[0])
        seed = int(sample["seed"].iloc[0])
        backbone = str(sample["backbone"].iloc[0])
        model_label = _label(sample.iloc[0])
        if split not in train_cache:
            train = pd.read_csv(data_dir / f"{split}_train.csv")
            train_cache[split] = (
                set(train["smiles"].dropna().astype(str)),
                set(train["CAS"].dropna().astype(str)),
            )
        train_smiles, train_cas = train_cache[split]
        df = _read_prediction(path, train_smiles, train_cas)
        df["model_label"] = model_label
        df["prediction_file"] = path.name
        index[(split, seed, backbone, model_label)] = df
    return index


def _metric_rows(index: dict[tuple[str, int, str, str], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for (split, seed, backbone, model_label), df in index.items():
        subsets = [("all", df)]
        for subset, group in df.groupby("leakage_subset", dropna=False):
            subsets.append((str(subset), group))
        for subset, group in subsets:
            row = {
                "split": split,
                "seed": seed,
                "backbone": backbone,
                "model_label": model_label,
                "leakage_subset": subset,
                "n": int(len(group)),
                "n_overlap_smiles": int(group["compound_overlap_smiles"].sum()),
                "n_overlap_cas": int(group["compound_overlap_cas"].sum()),
                "prediction_file": str(group["prediction_file"].iloc[0]) if len(group) else "",
            }
            row.update(_metrics(group))
            rows.append(row)
    return pd.DataFrame(rows)


def _comparison_pairs() -> list[tuple[str, str]]:
    return [
        ("species_bias_only", "no_species"),
        ("true_species_late_fusion", "no_species"),
        ("true_species_early_injection", "no_species"),
        ("true_species_message_level", "no_species"),
        ("true_species_film", "no_species"),
        ("LightGBM_RDKit_species_categorical", "LightGBM_RDKit_no_species"),
    ]


def _paired_frame(
    index: dict[tuple[str, int, str, str], pd.DataFrame],
    cand_key: tuple[str, int, str, str],
    ref_key: tuple[str, int, str, str],
) -> pd.DataFrame:
    cand = index[cand_key].copy()
    ref = index[ref_key].copy()
    merge_cols = ["source_row_id", "smiles", "species", "true_log10", "compound_key"]
    paired = cand[
        merge_cols
        + ["pred_log10", "compound_overlap_smiles", "compound_overlap_cas", "leakage_subset"]
    ].merge(
        ref[merge_cols + ["pred_log10"]],
        on=merge_cols,
        how="inner",
        suffixes=("_candidate", "_reference"),
    )
    return paired.rename(columns={"pred_log10_candidate": "pred_candidate", "pred_log10_reference": "pred_reference"})


def _paired_delta_rows(index: dict[tuple[str, int, str, str], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for split, seed, backbone, candidate in sorted(index.keys()):
        for candidate_label, reference_label in _comparison_pairs():
            if candidate != candidate_label:
                continue
            ref_key = (split, seed, backbone, reference_label)
            cand_key = (split, seed, backbone, candidate_label)
            if ref_key not in index:
                continue
            paired = _paired_frame(index, cand_key, ref_key)
            subsets = [("all", paired)]
            for subset, group in paired.groupby("leakage_subset", dropna=False):
                subsets.append((str(subset), group))
            for subset, group in subsets:
                row = {
                    "split": split,
                    "seed": seed,
                    "backbone": backbone,
                    "candidate": candidate_label,
                    "reference": reference_label,
                    "leakage_subset": subset,
                    "n": int(len(group)),
                    "n_overlap_smiles": int(group["compound_overlap_smiles"].sum()) if len(group) else 0,
                    "n_overlap_cas": int(group["compound_overlap_cas"].sum()) if len(group) else 0,
                }
                for metric in METRICS:
                    cand_value = _metric(group, metric, pred_col="pred_candidate")
                    ref_value = _metric(group, metric, pred_col="pred_reference")
                    delta = cand_value - ref_value
                    row[f"{metric}_candidate"] = cand_value
                    row[f"{metric}_reference"] = ref_value
                    row[f"delta_{metric}"] = delta
                    if metric in LOWER_IS_BETTER:
                        row[f"{metric}_favorable"] = bool(delta < 0)
                    else:
                        row[f"{metric}_favorable"] = bool(delta > 0)
                rows.append(row)
    return pd.DataFrame(rows)


def _sample_blocks(frame: pd.DataFrame, rng: np.random.Generator, block_col: str = "compound_key") -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    blocks = frame[block_col].fillna("__NA__").astype(str)
    groups = {block: idx.to_numpy() for block, idx in blocks.groupby(blocks).groups.items()}
    keys = np.array(list(groups.keys()), dtype=object)
    sampled_keys = rng.choice(keys, size=len(keys), replace=True)
    sampled_idx = np.concatenate([groups[key] for key in sampled_keys])
    return frame.iloc[sampled_idx]


def _block_aggregates(frame: pd.DataFrame, block_col: str = "compound_key") -> pd.DataFrame:
    tmp = pd.DataFrame(
        {
            "block": frame[block_col].fillna("__NA__").astype(str),
            "err_candidate": frame["pred_candidate"].astype(float) - frame["true_log10"].astype(float),
            "err_reference": frame["pred_reference"].astype(float) - frame["true_log10"].astype(float),
        }
    )
    tmp["n"] = 1
    tmp["sq_candidate"] = tmp["err_candidate"] ** 2
    tmp["sq_reference"] = tmp["err_reference"] ** 2
    tmp["abs_candidate"] = tmp["err_candidate"].abs()
    tmp["abs_reference"] = tmp["err_reference"].abs()
    tmp["within2_candidate"] = (tmp["abs_candidate"] <= math.log10(2.0)).astype(int)
    tmp["within2_reference"] = (tmp["abs_reference"] <= math.log10(2.0)).astype(int)
    tmp["within3_candidate"] = (tmp["abs_candidate"] <= math.log10(3.0)).astype(int)
    tmp["within3_reference"] = (tmp["abs_reference"] <= math.log10(3.0)).astype(int)
    return tmp.groupby("block", sort=False).sum(numeric_only=True).reset_index(drop=True)


def _metric_delta_from_aggregate(agg: pd.DataFrame, metric: str) -> float:
    n = float(agg["n"].sum())
    if n == 0:
        return float("nan")
    if metric == "rmse":
        cand = math.sqrt(float(agg["sq_candidate"].sum()) / n)
        ref = math.sqrt(float(agg["sq_reference"].sum()) / n)
        return cand - ref
    if metric == "mae":
        cand = float(agg["abs_candidate"].sum()) / n
        ref = float(agg["abs_reference"].sum()) / n
        return cand - ref
    if metric == "within_2fold":
        cand = float(agg["within2_candidate"].sum()) / n
        ref = float(agg["within2_reference"].sum()) / n
        return cand - ref
    if metric == "within_3fold":
        cand = float(agg["within3_candidate"].sum()) / n
        ref = float(agg["within3_reference"].sum()) / n
        return cand - ref
    raise ValueError(metric)


def _bootstrap_metric_deltas(
    frame: pd.DataFrame,
    metric: str,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    agg = _block_aggregates(frame)
    n_blocks = len(agg)
    sample_idx = rng.integers(0, n_blocks, size=(n_bootstrap, n_blocks))
    n = agg["n"].to_numpy(float)[sample_idx].sum(axis=1)
    if metric == "rmse":
        cand = np.sqrt(agg["sq_candidate"].to_numpy(float)[sample_idx].sum(axis=1) / n)
        ref = np.sqrt(agg["sq_reference"].to_numpy(float)[sample_idx].sum(axis=1) / n)
        return cand - ref
    if metric == "mae":
        cand = agg["abs_candidate"].to_numpy(float)[sample_idx].sum(axis=1) / n
        ref = agg["abs_reference"].to_numpy(float)[sample_idx].sum(axis=1) / n
        return cand - ref
    if metric == "within_2fold":
        cand = agg["within2_candidate"].to_numpy(float)[sample_idx].sum(axis=1) / n
        ref = agg["within2_reference"].to_numpy(float)[sample_idx].sum(axis=1) / n
        return cand - ref
    if metric == "within_3fold":
        cand = agg["within3_candidate"].to_numpy(float)[sample_idx].sum(axis=1) / n
        ref = agg["within3_reference"].to_numpy(float)[sample_idx].sum(axis=1) / n
        return cand - ref
    raise ValueError(metric)


def _bootstrap_interaction_rows(
    index: dict[tuple[str, int, str, str], pd.DataFrame],
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    for split, run_seed, backbone, candidate in sorted(index.keys()):
        if split != "random":
            continue
        for candidate_label, reference_label in _comparison_pairs():
            if candidate != candidate_label:
                continue
            cand_key = (split, run_seed, backbone, candidate_label)
            ref_key = (split, run_seed, backbone, reference_label)
            if ref_key not in index:
                continue
            paired = _paired_frame(index, cand_key, ref_key)
            overlap = paired[paired["leakage_subset"].eq("train_smiles_overlap")].copy()
            disjoint = paired[paired["leakage_subset"].eq("train_smiles_disjoint")].copy()
            if overlap.empty or disjoint.empty:
                continue
            overlap_agg = _block_aggregates(overlap)
            disjoint_agg = _block_aggregates(disjoint)
            for metric in METRICS:
                observed_overlap_delta = _metric_delta_from_aggregate(overlap_agg, metric)
                observed_disjoint_delta = _metric_delta_from_aggregate(disjoint_agg, metric)
                observed_interaction = observed_overlap_delta - observed_disjoint_delta
                overlap_samples = _bootstrap_metric_deltas(overlap, metric, n_bootstrap=n_bootstrap, rng=rng)
                disjoint_samples = _bootstrap_metric_deltas(disjoint, metric, n_bootstrap=n_bootstrap, rng=rng)
                arr = overlap_samples - disjoint_samples
                p_low = float(np.mean(arr <= 0.0))
                p_high = float(np.mean(arr >= 0.0))
                rows.append(
                    {
                        "split": split,
                        "seed": int(run_seed),
                        "backbone": backbone,
                        "candidate": candidate_label,
                        "reference": reference_label,
                        "metric": metric,
                        "n_overlap_rows": int(len(overlap)),
                        "n_disjoint_rows": int(len(disjoint)),
                        "n_bootstrap": int(n_bootstrap),
                        "overlap_delta": float(observed_overlap_delta),
                        "disjoint_delta": float(observed_disjoint_delta),
                        "interaction_overlap_minus_disjoint": float(observed_interaction),
                        "interaction_ci_low": float(np.quantile(arr, 0.025)),
                        "interaction_ci_high": float(np.quantile(arr, 0.975)),
                        "interaction_p_sign_approx": min(1.0, 2.0 * min(p_low, p_high)),
                    }
                )
    return pd.DataFrame(rows)


def _aggregate_deltas(delta: pd.DataFrame) -> pd.DataFrame:
    if delta.empty:
        return pd.DataFrame()
    agg_map = {
        "n": "mean",
        "delta_rmse": ["mean", "median", "std"],
        "delta_mae": ["mean", "median", "std"],
        "delta_within_2fold": ["mean", "median", "std"],
        "delta_within_3fold": ["mean", "median", "std"],
        "rmse_favorable": "sum",
        "mae_favorable": "sum",
        "within_2fold_favorable": "sum",
        "within_3fold_favorable": "sum",
        "seed": "count",
    }
    out = delta.groupby(["split", "backbone", "candidate", "reference", "leakage_subset"], dropna=False).agg(agg_map)
    out.columns = ["_".join(col).rstrip("_") for col in out.columns.to_flat_index()]
    out = out.reset_index().rename(columns={"seed_count": "n_seed_rows", "n_mean": "mean_n_rows"})
    return out.sort_values(["split", "backbone", "candidate", "leakage_subset"])


def _aggregate_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    if interactions.empty:
        return pd.DataFrame()
    out = (
        interactions.groupby(["backbone", "candidate", "reference", "metric"], dropna=False)
        .agg(
            n_seed_rows=("seed", "count"),
            mean_interaction=("interaction_overlap_minus_disjoint", "mean"),
            median_interaction=("interaction_overlap_minus_disjoint", "median"),
            mean_ci_low=("interaction_ci_low", "mean"),
            mean_ci_high=("interaction_ci_high", "mean"),
            n_ci_excludes_zero=(
                "interaction_ci_low",
                lambda s: int(
                    (
                        (s.to_numpy() > 0)
                        | (
                            interactions.loc[s.index, "interaction_ci_high"].to_numpy()
                            < 0
                        )
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )
    return out.sort_values(["backbone", "candidate", "metric"])


def _write_summary_md(path: Path, agg: pd.DataFrame, interaction_summary: pd.DataFrame) -> None:
    core = agg[
        agg["candidate"].isin(["species_bias_only", "true_species_late_fusion", "LightGBM_RDKit_species_categorical"])
        & agg["leakage_subset"].isin(["train_smiles_overlap", "train_smiles_disjoint", "all"])
    ].copy()
    cols = [
        "split",
        "backbone",
        "candidate",
        "reference",
        "leakage_subset",
        "mean_n_rows",
        "delta_rmse_mean",
        "delta_mae_mean",
        "rmse_favorable_sum",
        "n_seed_rows",
    ]
    interaction_core = interaction_summary[
        interaction_summary["candidate"].isin(
            ["species_bias_only", "true_species_late_fusion", "LightGBM_RDKit_species_categorical"]
        )
        & interaction_summary["metric"].isin(["rmse", "mae"])
    ].copy()
    lines = [
        "# JCIM v3 Leakage Effect Subset Analysis",
        "",
        "This is a post-hoc analysis over existing full prediction CSVs. It does not retrain models.",
        "",
        "Negative `delta_rmse` / `delta_mae` means the candidate is better than the reference.",
        "Random split rows are divided by whether their SMILES appears in the corresponding train split.",
        "",
        "## Core Comparison Summary",
        "",
        core[cols].to_markdown(index=False) if not core.empty else "No core rows found.",
        "",
        "## Difference-Of-Deltas Bootstrap Summary",
        "",
        "`interaction = overlap_delta - disjoint_delta`. For RMSE/MAE, a negative interaction means the candidate's benefit is larger in train-overlapping rows.",
        "",
        interaction_core.to_markdown(index=False) if not interaction_core.empty else "No interaction rows found.",
        "",
        "## Interpretation",
        "",
        "- If species-related gains concentrate in `train_smiles_overlap`, the random-split effect is consistent with leakage-assisted within-compound calibration.",
        "- If gains persist in `train_smiles_disjoint`, a cleaner species-context signal may remain even within the existing random split.",
        "- This analysis does not replace a compound-disjoint random/group split; it is a cheaper diagnostic bridge.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="<USER_HOME>/Desktop/CCLABS/CC-MPNN/data")
    parser.add_argument("--prediction-dir", default="results/jcim_v3/full/predictions")
    parser.add_argument("--out-dir", default="results/jcim_v3/full/paper_results/leakage_effect")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260713)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    prediction_dir = Path(args.prediction_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = _load_index(prediction_dir, data_dir)
    metrics = _metric_rows(index)
    deltas = _paired_delta_rows(index)
    agg = _aggregate_deltas(deltas)
    interactions = _bootstrap_interaction_rows(index, n_bootstrap=args.n_bootstrap, seed=args.bootstrap_seed)
    interaction_summary = _aggregate_interactions(interactions)

    metrics.to_csv(out_dir / "leakage_subset_metrics.csv", index=False, encoding="utf-8")
    deltas.to_csv(out_dir / "leakage_subset_pairwise_deltas.csv", index=False, encoding="utf-8")
    agg.to_csv(out_dir / "leakage_subset_pairwise_delta_summary.csv", index=False, encoding="utf-8")
    interactions.to_csv(out_dir / "leakage_interaction_bootstrap.csv", index=False, encoding="utf-8")
    interaction_summary.to_csv(out_dir / "leakage_interaction_bootstrap_summary.csv", index=False, encoding="utf-8")
    payload = {
        "n_prediction_files": int(len(index)),
        "n_metric_rows": int(len(metrics)),
        "n_pairwise_rows": int(len(deltas)),
        "n_summary_rows": int(len(agg)),
        "n_interaction_rows": int(len(interactions)),
        "n_interaction_summary_rows": int(len(interaction_summary)),
        "n_bootstrap": int(args.n_bootstrap),
        "output_dir": str(out_dir),
    }
    (out_dir / "leakage_effect_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_summary_md(out_dir / "leakage_effect_summary.md", agg, interaction_summary)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
