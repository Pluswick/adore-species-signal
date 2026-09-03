from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.paths import RESULTS_ROOT
from src.runner import V3RunConfig, run_v3_smoke


SPECIES_MODES = [
    "no_species",
    "true_species_late_fusion",
    "zero_species_late_fusion",
    "shuffled_species_late_fusion",
    "dummy_species_late_fusion",
    "species_bias_only",
]

REQUIRED_PREDICTION_COLUMNS = [
    "smiles",
    "compound_key",
    "scaffold",
    "scaffold_key",
    "species",
    "true_log10",
    "pred_log10",
    "error_log10",
    "split",
    "model_name",
    "backbone",
    "species_mode",
    "seed",
]

RECOMMENDED_CONTROL_COLUMNS = [
    "input_species",
    "species_for_model",
    "species_control_type",
    "is_shuffled",
    "is_zero_species",
    "is_dummy_species",
]

PARAMETER_MATCHED_MODES = {
    "zero_species_late_fusion",
    "shuffled_species_late_fusion",
    "dummy_species_late_fusion",
}


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _base_config(raw: dict) -> V3RunConfig:
    aliases = {
        "dataset_path": "data_dir",
        "max_epochs": "epochs",
        "train_subset_size": "limit_train",
        "test_subset_size": "limit_test",
    }
    normalized = dict(raw)
    for src, dst in aliases.items():
        if src in normalized and dst not in normalized:
            normalized[dst] = normalized[src]

    valid = {field.name for field in fields(V3RunConfig)}
    kwargs = {key: value for key, value in normalized.items() if key in valid}
    kwargs.setdefault("backbone", "dmpnn")
    kwargs.setdefault("variant", "no_species")
    return V3RunConfig(**kwargs)


def _prediction_checks(prediction_file: str, expected_rows: int | None) -> dict:
    df = pd.read_csv(prediction_file)
    missing = [col for col in REQUIRED_PREDICTION_COLUMNS if col not in df.columns]
    missing_control = [col for col in RECOMMENDED_CONTROL_COLUMNS if col not in df.columns]
    return {
        "prediction_file": prediction_file,
        "rows": int(len(df)),
        "missing_required_columns": missing,
        "missing_recommended_control_columns": missing_control,
        "schema_complete": not missing,
        "recommended_control_columns_complete": not missing_control,
        "expected_rows": expected_rows,
        "row_count_matches": expected_rows is None or int(len(df)) == int(expected_rows),
        "has_nan_prediction": bool(df["pred_log10"].isna().any()) if "pred_log10" in df else True,
    }


def _parameter_counts(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    refs = (
        frame[frame["species_mode"] == "true_species_late_fusion"]
        .set_index("backbone")["trainable_params"]
        .to_dict()
    )
    out_rows = []
    for row in rows:
        ref = refs.get(row["backbone"])
        delta = None if ref is None else int(row["trainable_params"]) - int(ref)
        rel = None if ref in (None, 0) else abs(delta) / int(ref)
        within = None if rel is None else bool(rel <= 0.05)
        mode = row["species_mode"]
        if mode == "true_species_late_fusion":
            status = "reference"
        elif mode in PARAMETER_MATCHED_MODES:
            status = "parameter_matched_within_5pct" if within else "not_parameter_matched"
        else:
            status = "not_a_parameter_matched_control"
        out_rows.append(
            {
                **row,
                "reference_species_mode": "true_species_late_fusion",
                "reference_trainable_params": ref,
                "delta_vs_reference": delta,
                "relative_delta_vs_reference": rel,
                "within_5pct": within,
                "parameter_match_status": status,
            }
        )
    return pd.DataFrame(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v3_controls_smoke.json")
    args = parser.parse_args()

    raw = _load_config(Path(args.config))
    base = _base_config(raw)
    backbones = raw.get("backbone_list") or raw.get("backbones") or ["dmpnn", "graphconv"]
    species_modes = raw.get("species_mode_list") or raw.get("species_modes") or SPECIES_MODES
    out_root = Path(base.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "runs").mkdir(parents=True, exist_ok=True)
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)

    config_snapshot = out_root / "control_smoke_config_snapshot.json"
    with open(config_snapshot, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    smoke_rows = []
    param_rows = []
    sanity_rows = []
    prediction_checks = []
    species_bias_files = []
    t0 = time.time()

    for backbone in backbones:
        for species_mode in species_modes:
            cfg = replace(base, backbone=backbone, variant=species_mode, out_root=str(out_root))
            result = run_v3_smoke(cfg)
            run_config = result["config"]
            metrics = result["A"]
            details = result["D"]
            smoke_rows.append(
                {
                    "run_id": run_config["run_id"],
                    "backbone": backbone,
                    "species_mode": species_mode,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "within_2fold": metrics["within_2fold"],
                    "within_3fold": metrics["within_3fold"],
                    "trainable_params": details["trainable_params"],
                    "species_trainable_params": details["species_trainable_params"],
                    "epochs": details["epochs"],
                    "elapsed_sec": details["train_sec"],
                    "zero_species_vector_all_zero": details["zero_species_vector_all_zero"],
                    "prediction_file": run_config["prediction_file"],
                    "run_json": str(out_root / "runs" / f"{run_config['run_id']}.json"),
                    "species_bias_file": run_config["species_bias_file"],
                }
            )
            param_rows.append(
                {
                    "run_id": run_config["run_id"],
                    "backbone": backbone,
                    "species_mode": species_mode,
                    "trainable_params": details["trainable_params"],
                    "species_trainable_params": details["species_trainable_params"],
                }
            )
            for row in result["E"]["control_sanity"]:
                sanity_rows.append({"run_id": run_config["run_id"], **row})
            prediction_checks.append(
                _prediction_checks(run_config["prediction_file"], expected_rows=base.limit_test)
            )
            if run_config["species_bias_file"]:
                species_bias_files.append(run_config["species_bias_file"])
            print("done", run_config["run_id"])

    smoke_df = pd.DataFrame(smoke_rows)
    smoke_df.to_csv(out_root / "control_smoke_summary.csv", index=False, encoding="utf-8")

    param_df = _parameter_counts(param_rows)
    param_df.to_csv(out_root / "control_parameter_counts.csv", index=False, encoding="utf-8")

    sanity_df = pd.DataFrame(sanity_rows)
    sanity_df.to_csv(out_root / "control_sanity_rows.csv", index=False, encoding="utf-8")

    shuffled_df = sanity_df[sanity_df["species_mode"] == "shuffled_species_late_fusion"].copy()
    audit_dir = RESULTS_ROOT / "data_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    shuffled_df.to_csv(audit_dir / "shuffled_species_check.csv", index=False, encoding="utf-8")

    zero_rows = sanity_df[sanity_df["species_mode"] == "zero_species_late_fusion"]
    dummy_rows = sanity_df[sanity_df["species_mode"] == "dummy_species_late_fusion"]
    shuffled_rows = sanity_df[sanity_df["species_mode"] == "shuffled_species_late_fusion"]
    bias_rows = smoke_df[smoke_df["species_mode"] == "species_bias_only"]

    checks = {
        "all_12_runs_completed": len(smoke_df) == 12,
        "all_prediction_schemas_complete": all(row["schema_complete"] for row in prediction_checks),
        "all_recommended_control_columns_complete": all(
            row["recommended_control_columns_complete"] for row in prediction_checks
        ),
        "all_prediction_row_counts_match": all(row["row_count_matches"] for row in prediction_checks),
        "no_nan_predictions": not any(row["has_nan_prediction"] for row in prediction_checks),
        "zero_species_idx_all_zero": bool(zero_rows["all_zero_species_idx"].all()),
        "zero_species_vector_all_zero": bool(
            smoke_df[smoke_df["species_mode"] == "zero_species_late_fusion"][
                "zero_species_vector_all_zero"
            ].all()
        ),
        "shuffled_marginal_preserved": bool(shuffled_rows["marginal_preserved"].all()),
        "shuffled_not_identical": bool((~shuffled_rows["identical_to_original"]).all()),
        "dummy_not_identical": bool((~dummy_rows["identical_to_original"]).all()),
        "dummy_has_multiple_categories": bool((dummy_rows["n_unique_for_model"] > 1).all()),
        "dummy_parameter_matched_within_5pct": bool(
            param_df[
                param_df["species_mode"] == "dummy_species_late_fusion"
            ]["within_5pct"].all()
        ),
        "species_bias_files_saved": len(species_bias_files) == len(bias_rows),
        "species_bias_has_parameters": bool((bias_rows["species_trainable_params"] > 0).all()),
    }

    summary = {
        "config": raw,
        "config_snapshot": str(config_snapshot),
        "out_root": str(out_root),
        "elapsed_total_sec": round(time.time() - t0, 2),
        "prediction_checks": prediction_checks,
        "parameter_count_file": str(out_root / "control_parameter_counts.csv"),
        "smoke_summary_file": str(out_root / "control_smoke_summary.csv"),
        "shuffled_species_check_file": str(audit_dir / "shuffled_species_check.csv"),
        "species_bias_files": species_bias_files,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    with open(out_root / "control_sanity_checks.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
