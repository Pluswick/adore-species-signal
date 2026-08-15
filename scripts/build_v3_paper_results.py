from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "results" / "jcim_v3" / "full"
DEFAULT_OUT = DEFAULT_ROOT / "paper_results"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "jcim_v3" / ".matplotlib_cache"))

import matplotlib.pyplot as plt

LOWER_IS_BETTER = {"rmse", "mae"}
HIGHER_IS_BETTER = {"within_2fold", "within_3fold"}
TRUE_MODES = [
    "true_species_late_fusion",
    "true_species_early_injection",
    "true_species_message_level",
    "true_species_film",
]
MAIN_DISPLAY_MODES = [
    "LightGBM_RDKit_no_species",
    "LightGBM_RDKit_species_categorical",
    "no_species",
    "species_bias_only",
    "true_species_late_fusion",
    "true_species_early_injection",
    "true_species_message_level",
    "true_species_film",
]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _delta_is_favorable(metric: str, values: pd.Series) -> pd.Series:
    if metric in LOWER_IS_BETTER:
        return values < 0
    if metric in HIGHER_IS_BETTER:
        return values > 0
    return pd.Series(False, index=values.index)


def _load(root: Path) -> dict[str, pd.DataFrame]:
    return {
        "metrics": pd.read_csv(root / "metrics" / "aggregated_metrics.csv"),
        "metric_summary": pd.read_csv(root / "summary_tables" / "metric_summary_by_model.csv"),
        "bootstrap": pd.read_csv(root / "bootstrap" / "bootstrap_comparisons_fdr.csv"),
        "parameters": pd.read_csv(root / "parameter_counts" / "parameter_counts.csv"),
        "scaffold": pd.read_csv(root / "embedding_analysis" / "scaffold_improvement_summary.csv"),
        "embedding_corr": pd.read_csv(root / "embedding_analysis" / "embedding_sensitivity_correlation.csv"),
    }


def _main_performance_table(metric_summary: pd.DataFrame) -> pd.DataFrame:
    out = metric_summary.copy()
    out["model_label"] = np.where(
        out["backbone"].eq("lightgbm_rdkit"),
        out["species_mode"].map(
            {
                "no_species": "LightGBM_RDKit_no_species",
                "species_categorical": "LightGBM_RDKit_species_categorical",
            }
        ),
        out["species_mode"],
    )
    keep = out["model_label"].isin(MAIN_DISPLAY_MODES)
    out = out.loc[keep].copy()
    order = {name: i for i, name in enumerate(MAIN_DISPLAY_MODES)}
    out["model_order"] = out["model_label"].map(order)
    out = out.sort_values(["split", "backbone", "model_order"])
    cols = [
        "split",
        "backbone",
        "model_label",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "within_2fold_mean",
        "within_2fold_std",
        "within_3fold_mean",
        "within_3fold_std",
        "rmse_count",
    ]
    return out[cols].rename(columns={"rmse_count": "n_seeds"})


def _claim_audit(bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "comparison_family",
        "backbone",
        "split",
        "candidate_species_mode",
        "reference_species_mode",
        "metric",
    ]
    for key, g in bootstrap.groupby(group_cols, dropna=False):
        fam, backbone, split, cand, ref, metric = key
        favorable = _delta_is_favorable(metric, g["delta"])
        significant = g["significant_fdr_0_05"].astype(bool)
        sig_favorable = favorable & significant
        n = int(len(g))
        n_sig_fav = int(sig_favorable.sum())
        n_fav = int(favorable.sum())
        q_max_sig_fav = float(g.loc[sig_favorable, "q_value_bh_fdr"].max()) if n_sig_fav else np.nan
        median_delta = float(g["delta"].median())
        mean_delta = float(g["delta"].mean())
        ci_low_median = float(g["ci_low"].median())
        ci_high_median = float(g["ci_high"].median())
        if n_sig_fav == n and n_fav == n:
            strength = "robust_candidate"
        elif n_sig_fav >= max(3, int(np.ceil(0.6 * n))) and n_fav == n:
            strength = "moderate_candidate"
        elif n_fav == n and n_sig_fav > 0:
            strength = "directional_support"
        elif n_fav >= max(3, int(np.ceil(0.6 * n))):
            strength = "directional_only"
        else:
            strength = "not_supported"
        rows.append(
            {
                "comparison_family": fam,
                "backbone": backbone,
                "split": split,
                "candidate_species_mode": cand,
                "reference_species_mode": ref,
                "metric": metric,
                "n_seeds": n,
                "n_favorable": n_fav,
                "n_significant_favorable_fdr_0_05": n_sig_fav,
                "median_delta": median_delta,
                "mean_delta": mean_delta,
                "median_ci_low": ci_low_median,
                "median_ci_high": ci_high_median,
                "max_q_among_significant_favorable": q_max_sig_fav,
                "claim_strength": strength,
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "comparison_family",
            "split",
            "backbone",
            "metric",
            "claim_strength",
            "candidate_species_mode",
            "reference_species_mode",
        ]
    )


def _primary_claims(claim_audit: pd.DataFrame) -> pd.DataFrame:
    primary = claim_audit[
        claim_audit["comparison_family"].isin(["species_information", "control", "injection_position"])
        & claim_audit["metric"].isin(["rmse", "mae"])
    ].copy()
    priority = {
        "robust_candidate": 0,
        "moderate_candidate": 1,
        "directional_support": 2,
        "directional_only": 3,
        "not_supported": 4,
    }
    primary["claim_priority"] = primary["claim_strength"].map(priority)
    return primary.sort_values(
        [
            "claim_priority",
            "comparison_family",
            "split",
            "backbone",
            "metric",
            "median_delta",
        ]
    )


def _parameter_match_audit(parameters: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    param = (
        parameters.groupby(["backbone", "split", "seed", "species_mode"], dropna=False)["trainable_params"]
        .first()
        .reset_index()
    )
    pairs = bootstrap[
        ["backbone", "split", "seed", "candidate_species_mode", "reference_species_mode", "comparison_family"]
    ].drop_duplicates()
    pairs = pairs.merge(
        param.rename(columns={"species_mode": "candidate_species_mode", "trainable_params": "candidate_params"}),
        on=["backbone", "split", "seed", "candidate_species_mode"],
        how="left",
    )
    pairs = pairs.merge(
        param.rename(columns={"species_mode": "reference_species_mode", "trainable_params": "reference_params"}),
        on=["backbone", "split", "seed", "reference_species_mode"],
        how="left",
    )
    pairs["absolute_param_diff"] = (pairs["candidate_params"] - pairs["reference_params"]).abs()
    pairs["relative_param_diff"] = pairs["absolute_param_diff"] / pairs["reference_params"].replace(0, np.nan)
    pairs["within_5pct_parameter_match"] = pairs["relative_param_diff"] <= 0.05
    pairs["within_1pct_parameter_match"] = pairs["relative_param_diff"] <= 0.01
    return pairs.sort_values(["comparison_family", "backbone", "split", "seed", "candidate_species_mode"])


def _best_model_summary(metric_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, backbone), g in metric_summary.groupby(["split", "backbone"], dropna=False):
        best = g.sort_values("rmse_mean").iloc[0]
        baseline_rows = g[g["species_mode"].eq("no_species")]
        baseline_rmse = float(baseline_rows["rmse_mean"].iloc[0]) if not baseline_rows.empty else np.nan
        rows.append(
            {
                "split": split,
                "backbone": backbone,
                "best_species_mode_by_rmse": best["species_mode"],
                "best_rmse_mean": float(best["rmse_mean"]),
                "best_mae_mean": float(best["mae_mean"]),
                "no_species_rmse_mean": baseline_rmse,
                "delta_rmse_vs_no_species": float(best["rmse_mean"] - baseline_rmse)
                if np.isfinite(baseline_rmse)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "backbone"])


def _scaffold_summary(scaffold: pd.DataFrame) -> pd.DataFrame:
    out = (
        scaffold.groupby(["backbone", "split", "candidate_species_mode", "reference_species_mode"], dropna=False)
        .agg(
            n_scaffold_rows=("scaffold_key", "count"),
            median_delta_rmse=("delta_rmse", "median"),
            mean_delta_rmse=("delta_rmse", "mean"),
            fraction_scaffolds_improved_rmse=("delta_rmse", lambda s: float((s < 0).mean())),
            median_delta_mae=("delta_mae", "median"),
            fraction_scaffolds_improved_mae=("delta_mae", lambda s: float((s < 0).mean())),
        )
        .reset_index()
    )
    return out.sort_values(["split", "backbone", "candidate_species_mode", "reference_species_mode"])


def _embedding_summary(embedding_corr: pd.DataFrame) -> pd.DataFrame:
    keep = embedding_corr[
        embedding_corr["sensitivity_variable"].isin(["species_bias", "mean_true_log10_test", "rmse_by_species"])
    ].copy()
    out = (
        keep.groupby(["backbone", "species_mode", "split", "sensitivity_variable"], dropna=False)
        .agg(
            n_seed_rows=("seed", "count"),
            median_spearman_r=("spearman_r", "median"),
            mean_spearman_r=("spearman_r", "mean"),
            median_pearson_r=("pearson_r", "median"),
        )
        .reset_index()
    )
    return out.sort_values(["split", "backbone", "species_mode", "sensitivity_variable"])


def _write_figures(out_dir: Path, main_perf: pd.DataFrame, claim_audit: pd.DataFrame, parameters: pd.DataFrame) -> list[dict]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    g = main_perf[main_perf["backbone"].isin(["dmpnn", "graphconv"])].copy()
    g = g[g["model_label"].isin(["no_species", "species_bias_only", *TRUE_MODES])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, split in zip(axes, ["random", "scaffold"]):
        sub = g[g["split"].eq(split)].copy()
        labels = [f"{b}\n{m.replace('true_species_', 'true_')}" for b, m in zip(sub["backbone"], sub["model_label"])]
        ax.bar(range(len(sub)), sub["rmse_mean"], yerr=sub["rmse_std"], color="#4C78A8", alpha=0.85)
        ax.set_title(f"{split} split")
        ax.set_ylabel("RMSE (log10 LC50)" if split == "random" else "")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = fig_dir / "main_rmse_by_backbone_species_mode.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    manifest.append({"figure": path.name, "purpose": "Main RMSE comparison across no-species, species-bias, and true-species GNN variants."})

    heat = claim_audit[
        claim_audit["comparison_family"].eq("species_information")
        & claim_audit["metric"].eq("rmse")
        & claim_audit["candidate_species_mode"].isin([*TRUE_MODES, "species_bias_only"])
    ].copy()
    heat["panel"] = heat["split"] + " / " + heat["backbone"]
    pivot = heat.pivot_table(
        index="candidate_species_mode",
        columns="panel",
        values="median_delta",
        aggfunc="first",
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Median delta RMSE vs no_species (negative is better)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Delta RMSE")
    fig.tight_layout()
    path = fig_dir / "species_information_delta_rmse_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    manifest.append({"figure": path.name, "purpose": "Species information effect size heatmap against no_species."})

    param = parameters[parameters["run_type"].eq("gnn")].copy()
    param_summary = (
        param.groupby(["backbone", "species_mode"], dropna=False)["trainable_params"].first().reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    param_summary = param_summary.sort_values(["backbone", "trainable_params", "species_mode"])
    labels = [f"{b}\n{m.replace('true_species_', 'true_')}" for b, m in zip(param_summary["backbone"], param_summary["species_mode"])]
    ax.bar(range(len(param_summary)), param_summary["trainable_params"], color="#59A14F", alpha=0.85)
    ax.set_ylabel("Trainable parameters")
    ax.set_title("GNN trainable parameter counts by variant")
    ax.set_xticks(range(len(param_summary)))
    ax.set_xticklabels(labels, rotation=80, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = fig_dir / "parameter_counts_by_variant.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    manifest.append({"figure": path.name, "purpose": "Parameter count audit for parameter-matched control interpretation."})
    return manifest


def _write_markdown(path: Path, payload: dict, table_paths: dict[str, str], figure_manifest: list[dict]) -> None:
    lines = [
        "# JCIM v3 Full Result Interpretation Pack",
        "",
        f"- Generated at UTC: `{payload['generated_at_utc']}`",
        f"- Full result root: `{payload['full_root']}`",
        f"- Output root: `{payload['output_root']}`",
        "",
        "## Scope",
        "",
        "This pack separates table/figure candidates from scientific claims. A row marked `robust_candidate` is a claim candidate, not final manuscript wording.",
        "Primary statistical evidence uses the existing global BH-FDR corrected bootstrap outputs.",
        "",
        "## Validation Snapshot",
        "",
    ]
    for key, value in payload["validation_snapshot"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Claim Strength Rule", ""])
    for key, value in payload["claim_strength_rule"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Generated Tables", ""])
    for name, rel in table_paths.items():
        lines.append(f"- `{name}`: `{rel}`")
    lines.extend(["", "## Generated Figures", ""])
    for row in figure_manifest:
        lines.append(f"- `{row['figure']}`: {row['purpose']}")
    lines.extend(
        [
            "",
            "## Immediate Interpretation Guardrails",
            "",
            "- Do not cite pilot or mini results for scientific claims.",
            "- Treat UMAP as not available in this run; embedding visualization is PCA-based.",
            "- Use control comparisons to distinguish species identity information from parameter count or species-frequency effects.",
            "- Use `parameter_match_audit.csv` before calling a comparison parameter-matched.",
            "- Treat current random-split results as non-compound-disjoint calibration diagnostics, not chemical generalization evidence.",
            "- Do not collapse all random-split species effects into one leakage explanation: `species_bias_only` is leakage-sensitive, while `true_species_late_fusion` is a smaller leakage-insensitive learned-interaction candidate whose clean-random evidence is weak and backbone-dependent.",
            "- Treat scaffold split as the primary current chemical-generalization split.",
            "- Treat the five seeds as training/control seeds on fixed split files, not independent split realizations.",
            "- Treat `robust_candidate` / `moderate_candidate` labels as claim-screening heuristics; report effect sizes and confidence intervals directly.",
            "",
            "## Manuscript Planning Documents",
            "",
            "- `CLAIM_SUMMARY.md`: result-derived claim candidates and limitations.",
            "- `CLAIM_REVISION_GUIDE_KO.md`: Korean guide for revising the original claim.",
            "- `MANUSCRIPT_POSITIONING_KO.md`: consolidated manuscript framing, novelty, and next-analysis plan.",
            "- `REVIEWER_RISK_AUDIT_KO.md`: reviewer-facing structural risks and required responses.",
            "",
            "## Manuscript Framing",
            "",
            "Use this result pack to support a controlled empirical study. Clean compound-disjoint random evaluation, naive species-intercept baselines, and abundance/cold-species follow-up analyses are now available. The central claim should remain limited to observed-species calibration and should not be extended to unseen-species generalization:",
            "",
            "> Species effects in multi-species aquatic LC50 96h regression decompose into multiple components. The strongest and most interpretable component is observed-species calibration: GNN `species_bias_only` remains beneficial under compound-disjoint random evaluation, a naive species-mean baseline improves over the global mean, and post-hoc LightGBM species residual calibration improves over its matched no-species molecular base. RDKit LightGBM `species_categorical` validates species label association versus shuffled/dummy categorical controls, but it shows no significant additional benefit over the naive residual-calibrated baseline. Learned true-species late fusion is smaller, backbone-dependent, and not consistently supported. Cold-species analysis shows no robust transfer of this calibration component to species absent from training.",
            "",
            "## Remaining Manuscript Work",
            "",
            "1. Build the main split/control/mechanism figures from the completed result tables.",
            "2. Report abundance/cold-species results as a boundary condition: cold species collapse, while observed non-cold bins do not form a simple monotonic high-support gradient.",
            "3. Add the Bio-QSAR/prior-work contrast table.",
            "",
            "Current clean-random configs include late-fusion zero/shuffled/dummy GNN controls, LightGBM zero/shuffled/dummy categorical controls, naive species mean/residual-calibration baselines, and GNN-vs-naive species-effect delta checks. Pilot 14 runs validated schema/execution, and the current clean-random core contains 105 prediction files as the clean-random scientific evidence block.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_claim_summary(
    path: Path,
    claim_audit: pd.DataFrame,
    best_models: pd.DataFrame,
    param_audit: pd.DataFrame,
) -> None:
    species_primary = claim_audit[
        claim_audit["comparison_family"].eq("species_information")
        & claim_audit["metric"].isin(["rmse", "mae"])
    ].copy()
    species_supported = species_primary[
        species_primary["claim_strength"].isin(["robust_candidate", "moderate_candidate"])
    ].sort_values(["split", "backbone", "metric", "median_delta"])
    species_not_supported = species_primary[
        species_primary["claim_strength"].eq("not_supported")
    ].sort_values(["split", "backbone", "candidate_species_mode", "metric"])
    control_primary = claim_audit[
        claim_audit["comparison_family"].eq("control")
        & claim_audit["metric"].isin(["rmse", "mae"])
    ]
    injection_primary = claim_audit[
        claim_audit["comparison_family"].eq("injection_position")
        & claim_audit["metric"].isin(["rmse", "mae"])
    ]
    control_counts = (
        control_primary.groupby(["split", "backbone", "claim_strength"]).size().unstack(fill_value=0).reset_index()
    )
    injection_counts = (
        injection_primary.groupby(["split", "backbone", "claim_strength"]).size().unstack(fill_value=0).reset_index()
    )
    param_counts = (
        param_audit.groupby("comparison_family")["within_5pct_parameter_match"]
        .agg(["sum", "count"])
        .reset_index()
    )
    lines = [
        "# JCIM v3 Claim Summary",
        "",
        "This summary is generated from the full 380-run result set and the global BH-FDR corrected bootstrap outputs.",
        "It is intended for manuscript planning, not as final wording.",
        "",
        "## Manuscript Claim Lock",
        "",
        "The manuscript should be positioned as a controlled empirical study of split artifacts and species-context effects in multi-species aquatic LC50 96h regression, not as a new species-embedding method paper.",
        "",
            "The central mechanism claim is supported by naive species-intercept baselines and abundance/cold-species follow-up analysis. It should be framed as observed-species calibration, not as unseen-species generalization or biological mechanism evidence.",
        "",
        "Recommended central claim:",
        "",
        "> Species effects in multi-species aquatic toxicity regression decompose into multiple components. The strongest and most interpretable component is observed-species calibration: GNN `species_bias_only` remains beneficial under compound-disjoint random evaluation, a naive species-mean baseline improves over the global mean, and post-hoc LightGBM species residual calibration improves over its matched no-species molecular base. RDKit LightGBM `species_categorical` validates species label association versus shuffled/dummy categorical controls, but it shows no significant additional benefit over the naive residual-calibrated baseline. Learned true-species late fusion is smaller, backbone-dependent, and not consistently supported. Cold-species analysis shows that this calibration component does not robustly transfer to species absent from training.",
        "",
        "## Critical Split Audit",
        "",
        "The current split leakage audit shows that the random split is not compound-disjoint:",
        "",
        "| Split | Key | Overlapping Unique Keys | Test Rows With Train Key | Test Row Overlap Fraction |",
        "|---|---|---:|---:|---:|",
        "| `random` | `smiles` | 1,151 | 2,788 / 3,445 | 0.8093 |",
        "| `random` | `CAS` | 1,078 | 2,873 / 3,445 | 0.8340 |",
        "| `random` | `smiles+species` | 0 | 0 / 3,445 | 0.0000 |",
        "| `scaffold` | `smiles` | 0 | 0 / 1,936 | 0.0000 |",
        "| `scaffold` | `CAS` | 4 | 28 / 1,936 | 0.0145 |",
        "",
        "This means random-split results are diagnostic evidence, while scaffold-split results carry the current chemical-generalization burden. Within the random diagnostic, `species_bias_only` and `true_species_late_fusion` must be interpreted separately because only `species_bias_only` shows clear leakage-sensitive behavior in the overlap-vs-disjoint subset analysis.",
        "",
        "## Candidate Claims With Stronger Support",
        "",
        "1. Full execution and analysis completed: 380 prediction files, 380 run JSON files, 380 parameter-count rows, and full bootstrap/FDR outputs are available.",
        "2. Species information effects are numerically clearest on the current random split, but that split is compound-overlapping. Treat random-split results as diagnostic evidence unless supported by scaffold or compound-disjoint analyses.",
        "3. Species-bias-only is a robust candidate versus no-species for both D-MPNN and GraphConv on RMSE and MAE in the current random split; leakage-effect diagnostics show this component is strongly overlap-sensitive and is best interpreted as species-level intercept/calibration.",
        "4. True-species late fusion is a moderate candidate versus no-species on the current random split. Its overlap-vs-disjoint interaction CIs include zero, so it should be treated as a separate learned-interaction candidate rather than collapsed into the leakage-assisted calibration story.",
        "5. True-species controls are strongest against dummy and shuffled species controls in the random split. This supports the need for explicit controls, but it should not be rewritten as a universal improvement claim for every injection location.",
        "6. Late fusion is the best-supported injection position on the random split, especially versus early injection and message-level injection.",
        "7. Abundance/cold-species analysis supports the boundary condition: additive species calibration collapses for cold species, while observed non-cold species can retain calibration gains across several support bins.",
        "",
        "## Claims That Need Limitation",
        "",
        "1. Do not claim universal species-embedding improvement across all splits. Scaffold split evidence is mostly directional and not robust after global BH-FDR.",
        "2. Do not claim early injection or message-level species injection improves over no-species. These are frequently not_supported versus no-species.",
        "3. Do not treat UMAP as an available interpretation result. The run generated PCA-based embedding outputs; UMAP was disabled.",
        "4. Do not call species_information comparisons against no_species parameter-matched unless `parameter_match_audit.csv` confirms the pair. The true-species versus no-species comparisons are not parameter-matched in this design.",
        "5. Do not claim FiLM is a key performance-improving mechanism. FiLM was evaluated, but FiLM versus no-species comparisons are mostly directional and not robust.",
        "6. Do not call the five seeds independent split replicates. They reuse the same fixed train/test partitions and mainly vary initialization, validation carve-out, training order, and control randomization.",
        "7. Do not present the robust/favorable seed-count rule as a formal meta-analysis. It is a screening heuristic unless backed by across-seed effect-size intervals or a mixed/meta-analytic summary.",
        "",
        "## Required Manuscript Additions Before Submission",
        "",
        "1. Add a prior-work contrast table for Bio-QSAR-like studies.",
        "2. Report the completed species-abundance and cold-species analysis. State that cold species show no robust species-calibration benefit, while observed non-cold species do not follow a simple monotonic abundance gradient.",
        "3. Integrate the LightGBM categorical, residual-calibration, and GNN-vs-naive species-effect delta results explicitly; categorical species labels validate label association, but they show no significant additional benefit over a simple residual-calibrated baseline, and GNN species-bias effects do not show significant species-effect delta advantage over naive residual calibration.",
        "",
        "## Best RMSE Model By Split And Backbone",
        "",
        "Important table note: every `random` row below comes from the compound-overlapping random split and should be interpreted as diagnostic evidence, not chemical generalization evidence. Within that diagnostic setting, `species_bias_only` and `true_species_late_fusion` should be interpreted separately because only the former showed clear leakage-sensitive behavior.",
        "",
        "| Split | Backbone | Best Species Mode | Best RMSE | No-Species RMSE | Delta RMSE |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in best_models.itertuples(index=False):
        lines.append(
            f"| `{row.split}` | `{row.backbone}` | `{row.best_species_mode_by_rmse}` | {row.best_rmse_mean:.4f} | {row.no_species_rmse_mean:.4f} | {row.delta_rmse_vs_no_species:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Species Information Comparisons Supported By RMSE/MAE",
            "",
            "| Split | Backbone | Candidate | Reference | Metric | Seeds Favorable | Seeds Significant+Favorable | Median Delta | Strength |",
            "|---|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in species_supported.itertuples(index=False):
        lines.append(
            f"| `{row.split}` | `{row.backbone}` | `{row.candidate_species_mode}` | `{row.reference_species_mode}` | `{row.metric}` | {row.n_favorable} | {row.n_significant_favorable_fdr_0_05} | {row.median_delta:.4f} | `{row.claim_strength}` |"
        )
    lines.extend(
        [
            "",
            "## Species Information Comparisons Not Supported",
            "",
            "| Split | Backbone | Candidate | Metric | Median Delta |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in species_not_supported.itertuples(index=False):
        lines.append(
            f"| `{row.split}` | `{row.backbone}` | `{row.candidate_species_mode}` | `{row.metric}` | {row.median_delta:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Control Comparison Strength Counts",
            "",
            "| Split | Backbone | Robust | Moderate | Directional Support | Directional Only | Not Supported |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in control_counts.itertuples(index=False):
        lines.append(
            f"| `{row.split}` | `{row.backbone}` | {getattr(row, 'robust_candidate', 0)} | {getattr(row, 'moderate_candidate', 0)} | {getattr(row, 'directional_support', 0)} | {getattr(row, 'directional_only', 0)} | {getattr(row, 'not_supported', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Injection Position Comparison Strength Counts",
            "",
            "| Split | Backbone | Robust | Moderate | Directional Support | Directional Only | Not Supported |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in injection_counts.itertuples(index=False):
        lines.append(
            f"| `{row.split}` | `{row.backbone}` | {getattr(row, 'robust_candidate', 0)} | {getattr(row, 'moderate_candidate', 0)} | {getattr(row, 'directional_support', 0)} | {getattr(row, 'directional_only', 0)} | {getattr(row, 'not_supported', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Parameter-Match Audit Summary",
            "",
            "| Comparison Family | Within 5% Count | Total Pairs |",
            "|---|---:|---:|",
        ]
    )
    for row in param_counts.itertuples(index=False):
        lines.append(f"| `{row.comparison_family}` | {int(row.sum)} | {int(row.count)} |")
    lines.extend(
        [
            "",
            "Use `parameter_match_audit.csv` for the exact pair-level parameter-count evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _load(root)

    main_perf = _main_performance_table(data["metric_summary"])
    claim_audit = _claim_audit(data["bootstrap"])
    primary_claims = _primary_claims(claim_audit)
    param_audit = _parameter_match_audit(data["parameters"], data["bootstrap"])
    best_models = _best_model_summary(data["metric_summary"])
    scaffold_summary = _scaffold_summary(data["scaffold"])
    embedding_summary = _embedding_summary(data["embedding_corr"])
    figure_manifest = _write_figures(out_dir, main_perf, claim_audit, data["parameters"])

    tables = {
        "main_performance_table": main_perf,
        "claim_audit": claim_audit,
        "primary_claim_candidates": primary_claims,
        "parameter_match_audit": param_audit,
        "best_model_by_split_backbone": best_models,
        "scaffold_improvement_aggregate": scaffold_summary,
        "embedding_correlation_summary": embedding_summary,
        "figure_candidate_manifest": pd.DataFrame(figure_manifest),
    }
    table_paths = {}
    for name, df in tables.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        table_paths[name] = str(path.relative_to(out_dir))

    validation = {
        "aggregated_metric_rows": int(len(data["metrics"])),
        "bootstrap_fdr_rows": int(len(data["bootstrap"])),
        "main_performance_rows": int(len(main_perf)),
        "claim_audit_rows": int(len(claim_audit)),
        "primary_claim_candidate_rows": int(len(primary_claims)),
        "all_bootstrap_q_values_in_0_1": bool(
            ((data["bootstrap"]["q_value_bh_fdr"] >= 0) & (data["bootstrap"]["q_value_bh_fdr"] <= 1)).all()
        ),
        "n_bootstrap_values": sorted(int(x) for x in data["bootstrap"]["n_bootstrap"].dropna().unique()),
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_root": str(root),
        "output_root": str(out_dir),
        "validation_snapshot": validation,
        "claim_strength_rule": {
            "robust_candidate": "all 5 seeds are favorable and globally FDR-significant for the metric",
            "moderate_candidate": "at least 3 of 5 seeds are favorable and globally FDR-significant, with all 5 seeds favorable",
            "directional_support": "all 5 seeds are favorable, but only 1-2 seeds are globally FDR-significant",
            "directional_only": "at least 3 of 5 seeds are favorable, without FDR-significant support",
            "not_supported": "does not meet the above criteria",
        },
        "generated_tables": table_paths,
        "generated_figures": figure_manifest,
        "notes": [
            "No model training or bootstrap recomputation is performed by this script.",
            "UMAP was disabled in the full analysis; PCA-based embedding outputs are available.",
        ],
    }
    _write_json(out_dir / "paper_results_summary.json", payload)
    _write_markdown(out_dir / "RESULT_INTERPRETATION_GUIDE.md", payload, table_paths, figure_manifest)
    _write_claim_summary(out_dir / "CLAIM_SUMMARY.md", claim_audit, best_models, param_audit)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
