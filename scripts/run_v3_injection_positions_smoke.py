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

from src.models import model_spec_from_variant
from src.runner import V3RunConfig, run_v3_smoke


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
    "injection_location",
    "input_species",
    "species_for_model",
    "species_control_type",
    "is_shuffled",
    "is_zero_species",
    "is_dummy_species",
]

PARAMETER_CONTROL_TYPES = {"zero", "shuffled", "dummy"}


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
        frame[frame["species_control_type"] == "true"]
        .set_index(["backbone", "injection_location"])["trainable_params"]
        .to_dict()
    )
    out_rows = []
    for row in rows:
        key = (row["backbone"], row["injection_location"])
        ref = refs.get(key)
        if row["injection_location"] in {"none", "output_bias"}:
            ref = None
        delta = None if ref is None else int(row["trainable_params"]) - int(ref)
        pct = None if ref in (None, 0) else abs(delta) / int(ref) * 100.0
        within = None if pct is None else bool(pct <= 5.0)
        if row["species_control_type"] == "true":
            status = "reference"
        elif row["species_control_type"] in PARAMETER_CONTROL_TYPES and row["injection_location"] not in {
            "none",
            "output_bias",
        }:
            status = "parameter_matched_within_5pct" if within else "not_parameter_matched"
        else:
            status = "not_a_parameter_matched_control"
        out_rows.append(
            {
                **row,
                "reference_true_species_params": ref,
                "param_diff_percent": pct,
                "within_5_percent": within,
                "parameter_match_status": status,
            }
        )
    return pd.DataFrame(out_rows)


def _shape_checks_pass(smoke_df: pd.DataFrame) -> dict:
    film = smoke_df[smoke_df["injection_location"] == "film"]
    message = smoke_df[smoke_df["injection_location"] == "message_level"]
    early = smoke_df[smoke_df["injection_location"] == "early_injection"]
    return {
        "film_gamma_beta_shape_ok": bool(film["film_gamma_beta_shape_ok"].all()),
        "message_level_hidden_shape_ok": bool(message["message_level_hidden_shape_ok"].all()),
        "early_injection_shape_ok": bool(early["early_injection_shape_ok"].all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v3_injection_positions_smoke.json")
    args = parser.parse_args()

    raw = _load_config(Path(args.config))
    base = _base_config(raw)
    backbones = raw.get("backbone_list") or raw.get("backbones") or ["dmpnn", "graphconv"]
    species_modes = raw.get("species_mode_list") or raw.get("species_modes")
    if not species_modes:
        raise ValueError("species_mode_list is required")
    out_root = Path(base.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "runs").mkdir(parents=True, exist_ok=True)
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)

    config_snapshot = out_root / "injection_smoke_config_snapshot.json"
    with open(config_snapshot, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    smoke_rows = []
    param_rows = []
    sanity_rows = []
    prediction_checks = []
    warnings = []
    t0 = time.time()

    for backbone in backbones:
        for species_mode in species_modes:
            spec = model_spec_from_variant(backbone, species_mode)
            cfg = replace(base, backbone=backbone, variant=species_mode, out_root=str(out_root))
            result = run_v3_smoke(cfg)
            run_config = result["config"]
            details = result["D"]
            metrics = result["A"]
            shape = details["injection_shape_sanity"]
            smoke_rows.append(
                {
                    "run_id": run_config["run_id"],
                    "backbone": backbone,
                    "species_mode": species_mode,
                    "species_control_type": spec.species_control,
                    "injection_location": spec.injection,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "within_2fold": metrics["within_2fold"],
                    "within_3fold": metrics["within_3fold"],
                    "trainable_params": details["trainable_params"],
                    "species_params": details["species_trainable_params"],
                    "epochs": details["epochs"],
                    "elapsed_sec": details["train_sec"],
                    "zero_species_vector_all_zero": details["zero_species_vector_all_zero"],
                    "film_gamma_beta_shape_ok": shape["film_gamma_beta_shape_ok"],
                    "message_level_hidden_shape_ok": shape["message_level_hidden_shape_ok"],
                    "early_injection_shape_ok": shape["early_injection_shape_ok"],
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
                    "species_control_type": spec.species_control,
                    "injection_location": spec.injection,
                    "trainable_params": details["trainable_params"],
                    "species_params": details["species_trainable_params"],
                }
            )
            for row in result["E"]["control_sanity"]:
                sanity_rows.append(
                    {
                        "run_id": run_config["run_id"],
                        "injection_location": spec.injection,
                        **row,
                    }
                )
            prediction_checks.append(
                _prediction_checks(run_config["prediction_file"], expected_rows=base.limit_test)
            )
            print("done", run_config["run_id"])

    smoke_df = pd.DataFrame(smoke_rows)
    smoke_df.to_csv(out_root / "injection_smoke_summary.csv", index=False, encoding="utf-8")

    param_df = _parameter_counts(param_rows)
    param_df.to_csv(out_root / "injection_parameter_counts.csv", index=False, encoding="utf-8")
    bad_dummy = param_df[
        (param_df["species_control_type"] == "dummy")
        & (param_df["injection_location"].isin(["late_fusion", "early_injection", "message_level", "film"]))
        & (~param_df["within_5_percent"].fillna(False).astype(bool))
    ]
    if len(bad_dummy):
        warnings.append(
            "Some dummy controls exceed the +/-5% parameter matching threshold; "
            "do not call them parameter-matched."
        )

    sanity_df = pd.DataFrame(sanity_rows)
    sanity_df.to_csv(out_root / "injection_sanity_rows.csv", index=False, encoding="utf-8")

    zero_rows = sanity_df[sanity_df["mode"] == "zero"]
    shuffled_rows = sanity_df[sanity_df["mode"] == "shuffled"]
    dummy_rows = sanity_df[sanity_df["mode"] == "dummy"]
    shape_checks = _shape_checks_pass(smoke_df)
    checks = {
        "all_runs_completed": len(smoke_df) == len(backbones) * len(species_modes),
        "all_prediction_schemas_complete": all(row["schema_complete"] for row in prediction_checks),
        "all_recommended_control_columns_complete": all(
            row["recommended_control_columns_complete"] for row in prediction_checks
        ),
        "all_prediction_row_counts_match": all(row["row_count_matches"] for row in prediction_checks),
        "no_nan_predictions": not any(row["has_nan_prediction"] for row in prediction_checks),
        "zero_species_idx_all_zero": bool(zero_rows["all_zero_species_idx"].all()),
        "zero_species_vector_all_zero": bool(
            smoke_df[smoke_df["species_control_type"] == "zero"][
                "zero_species_vector_all_zero"
            ].all()
        ),
        "shuffled_marginal_preserved": bool(shuffled_rows["marginal_preserved"].all()),
        "shuffled_not_identical": bool((~shuffled_rows["identical_to_original"]).all()),
        "dummy_not_identical": bool((~dummy_rows["identical_to_original"]).all()),
        "dummy_has_multiple_categories": bool((dummy_rows["n_unique_for_model"] > 1).all()),
        "dummy_parameter_matched_within_5pct": bool(
            param_df[
                (param_df["species_control_type"] == "dummy")
                & (
                    param_df["injection_location"].isin(
                        ["late_fusion", "early_injection", "message_level", "film"]
                    )
                )
            ]["within_5_percent"].all()
        ),
        **shape_checks,
        "same_species_mode_names_for_backbones": bool(
            set(smoke_df[smoke_df["backbone"] == "dmpnn"]["species_mode"])
            == set(smoke_df[smoke_df["backbone"] == "graphconv"]["species_mode"])
        ),
    }

    summary = {
        "config": raw,
        "config_snapshot": str(config_snapshot),
        "out_root": str(out_root),
        "elapsed_total_sec": round(time.time() - t0, 2),
        "n_runs": int(len(smoke_df)),
        "prediction_checks": prediction_checks,
        "parameter_count_file": str(out_root / "injection_parameter_counts.csv"),
        "smoke_summary_file": str(out_root / "injection_smoke_summary.csv"),
        "sanity_rows_file": str(out_root / "injection_sanity_rows.csv"),
        "warnings": warnings,
        "checks": checks,
        "all_checks_passed": all(checks.values()) and not warnings,
    }
    with open(out_root / "injection_sanity_checks.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
