from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.bootstrap import PAIR_MERGE_KEYS, align_predictions, block_bootstrap_metric_delta
from src.paths import RESULTS_ROOT
from src.stats import bh_fdr


DEFAULT_COMPARISONS = [
    ("true_species_late_fusion", "no_species", "species_information"),
    ("true_species_early_injection", "no_species", "species_information"),
    ("true_species_message_level", "no_species", "species_information"),
    ("true_species_film", "no_species", "species_information"),
    ("species_bias_only", "no_species", "species_information"),
    ("true_species_late_fusion", "zero_species_late_fusion", "control"),
    ("true_species_late_fusion", "shuffled_species_late_fusion", "control"),
    ("true_species_late_fusion", "dummy_species_late_fusion", "control"),
    ("true_species_early_injection", "zero_species_early_injection", "control"),
    ("true_species_early_injection", "shuffled_species_early_injection", "control"),
    ("true_species_early_injection", "dummy_species_early_injection", "control"),
    ("true_species_message_level", "zero_species_message_level", "control"),
    ("true_species_message_level", "shuffled_species_message_level", "control"),
    ("true_species_message_level", "dummy_species_message_level", "control"),
    ("true_species_film", "zero_species_film", "control"),
    ("true_species_film", "shuffled_species_film", "control"),
    ("true_species_film", "dummy_species_film", "control"),
    ("true_species_late_fusion", "true_species_early_injection", "injection_position"),
    ("true_species_late_fusion", "true_species_message_level", "injection_position"),
    ("true_species_late_fusion", "true_species_film", "injection_position"),
    ("true_species_late_fusion", "species_bias_only", "injection_position"),
    ("true_species_early_injection", "true_species_message_level", "injection_position"),
    ("true_species_early_injection", "true_species_film", "injection_position"),
    ("true_species_message_level", "true_species_film", "injection_position"),
]


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _prediction_index(prediction_dir: Path, *, split: str, seed: int) -> dict[tuple[str, str], pd.DataFrame]:
    out = {}
    for path in sorted(prediction_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0]
        if str(row["split"]) != split or int(row["seed"]) != seed:
            continue
        key = (str(row["backbone"]), str(row["species_mode"]))
        df = df.copy()
        df["_prediction_file"] = str(path)
        out[key] = df
    return out


def _model_meta(df: pd.DataFrame) -> dict:
    row = df.iloc[0]
    return {
        "model_name": str(row["model_name"]),
        "species_mode": str(row["species_mode"]),
        "injection_location": str(row.get("injection_location", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-dir",
        default=str(RESULTS_ROOT / "smoke" / "src_env" / "injection_positions" / "predictions"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(RESULTS_ROOT / "smoke" / "src_env" / "stats_smoke"),
    )
    parser.add_argument("--config")
    parser.add_argument("--split", default="scaffold")
    parser.add_argument("--block-key", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        default=[],
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    prediction_dir = Path(config.get("prediction_root", args.prediction_dir))
    out_dir = Path(config.get("output_root", args.out_dir))
    split = str(config.get("split", args.split))
    seed = int(config.get("seed", args.seed))
    split_values = list(config.get("splits") or config.get("split_types") or [split])
    seed_values = [int(value) for value in (config.get("seeds") or [seed])]
    n_bootstrap = int(config.get("n_bootstrap", args.n_bootstrap))
    metrics = list(config.get("metrics", args.metrics or [])) or [
        "rmse",
        "mae",
        "within_2fold",
        "within_3fold",
    ]
    block_key = config.get("block_key") or args.block_key
    block_keys = config.get("block_keys") or config.get(
        "primary_block_keys",
        {"scaffold": "scaffold_key", "random": "compound_key"},
    )
    backbones = config.get("backbones") or ["dmpnn", "graphconv"]
    comparisons = config.get("comparisons") or [
        {
            "candidate_species_mode": cand,
            "reference_species_mode": ref,
            "comparison_family": family,
        }
        for cand, ref, family in DEFAULT_COMPARISONS
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    warnings = []
    n_attempted = 0
    n_completed = 0
    for current_split in split_values:
        current_block_key = block_key or block_keys.get(
            current_split,
            "scaffold_key" if current_split == "scaffold" else "compound_key",
        )
        for current_seed in seed_values:
            indexed = _prediction_index(prediction_dir, split=str(current_split), seed=int(current_seed))
            for backbone in backbones:
                for comparison in comparisons:
                    explicit_candidate_backbone = comparison.get("candidate_backbone")
                    explicit_reference_backbone = comparison.get("reference_backbone")
                    if explicit_candidate_backbone or explicit_reference_backbone:
                        cand_backbone = str(explicit_candidate_backbone or backbone)
                        ref_backbone = str(explicit_reference_backbone or backbone)
                        if backbone != cand_backbone:
                            continue
                        comparison_backbone = comparison.get(
                            "comparison_backbone",
                            cand_backbone
                            if cand_backbone == ref_backbone
                            else f"{cand_backbone}_vs_{ref_backbone}",
                        )
                    else:
                        allowed_backbones = comparison.get("backbones")
                        if allowed_backbones and backbone not in allowed_backbones:
                            continue
                        cand_backbone = backbone
                        ref_backbone = backbone
                        comparison_backbone = backbone
                    n_attempted += 1
                    cand_mode = comparison["candidate_species_mode"]
                    ref_mode = comparison["reference_species_mode"]
                    cand_df = indexed.get((cand_backbone, cand_mode))
                    ref_df = indexed.get((ref_backbone, ref_mode))
                    if cand_df is None or ref_df is None:
                        warnings.append(
                            {
                                "backbone": comparison_backbone,
                                "candidate_backbone": cand_backbone,
                                "reference_backbone": ref_backbone,
                                "split": current_split,
                                "seed": current_seed,
                                "candidate_species_mode": cand_mode,
                                "reference_species_mode": ref_mode,
                                "warning": "missing prediction file for comparison",
                            }
                        )
                        continue
                    comparison_merge_keys = comparison.get("merge_keys") or PAIR_MERGE_KEYS
                    pair = align_predictions(cand_df, ref_df, merge_keys=comparison_merge_keys)
                    if pair.warning:
                        warnings.append(
                            {
                                "backbone": comparison_backbone,
                                "candidate_backbone": cand_backbone,
                                "reference_backbone": ref_backbone,
                                "split": current_split,
                                "seed": current_seed,
                                "candidate_species_mode": cand_mode,
                                "reference_species_mode": ref_mode,
                                "warning": pair.warning,
                            }
                        )
                        continue
                    assert pair.paired is not None
                    if pair.paired[current_block_key].isna().any():
                        warnings.append(
                            {
                                "backbone": comparison_backbone,
                                "candidate_backbone": cand_backbone,
                                "reference_backbone": ref_backbone,
                                "split": current_split,
                                "seed": current_seed,
                                "candidate_species_mode": cand_mode,
                                "reference_species_mode": ref_mode,
                                "warning": f"block key contains null values: {current_block_key}",
                            }
                        )
                        continue
                    cand_meta = _model_meta(cand_df)
                    ref_meta = _model_meta(ref_df)
                    n_completed += 1
                    for metric in metrics:
                        result = block_bootstrap_metric_delta(
                            pair.paired,
                            block_key=current_block_key,
                            metric=metric,
                            n_bootstrap=n_bootstrap,
                            seed=int(current_seed) + 1009 * n_completed + 37 * metrics.index(metric),
                        )
                        rows.append(
                            {
                                "backbone": comparison_backbone,
                                "candidate_backbone": cand_backbone,
                                "reference_backbone": ref_backbone,
                                "split": current_split,
                                "seed": current_seed,
                                "comparison_family": comparison.get("comparison_family"),
                                "candidate_model": cand_meta["model_name"],
                                "reference_model": ref_meta["model_name"],
                                "candidate_species_mode": cand_mode,
                                "reference_species_mode": ref_mode,
                                "candidate_injection_location": cand_meta["injection_location"],
                                "reference_injection_location": ref_meta["injection_location"],
                                "block_key": current_block_key,
                                **result,
                            }
                        )

    raw = pd.DataFrame(rows)
    raw_path = out_dir / "bootstrap_comparisons_raw.csv"
    fdr_path = out_dir / "bootstrap_comparisons_fdr.csv"
    warnings_path = out_dir / "bootstrap_warnings.json"
    raw.to_csv(raw_path, index=False, encoding="utf-8")
    fdr = raw.copy()
    if len(fdr):
        fdr["q_value_bh_fdr"] = bh_fdr(fdr["p_value_approx"].to_numpy())
        fdr["significant_fdr_0_05"] = fdr["q_value_bh_fdr"] <= 0.05
    else:
        fdr["q_value_bh_fdr"] = []
        fdr["significant_fdr_0_05"] = []
    fdr.to_csv(fdr_path, index=False, encoding="utf-8")
    with open(warnings_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_attempted_comparisons": n_attempted,
                "n_completed_comparisons": n_completed,
                "n_result_rows": len(raw),
                "warnings": warnings,
                "note": "Approximate bootstrap sign-based p-values are used for paired model-comparison screening; manuscript claims should report effect sizes, confidence intervals, q-values, and practical significance together.",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(
        json.dumps(
            {
                "raw": str(raw_path),
                "fdr": str(fdr_path),
                "warnings": str(warnings_path),
                "n_attempted_comparisons": n_attempted,
                "n_completed_comparisons": n_completed,
                "n_result_rows": len(raw),
                "n_warnings": len(warnings),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
