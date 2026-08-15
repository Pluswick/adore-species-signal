from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "results" / "jcim_v3" / "full"
DEFAULT_CONFIG = ROOT / "configs" / "v3_full_experiment.json"
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
    "injection_location",
]


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dedupe(items: list[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _run_id(row: dict, config: dict) -> str:
    if row["run_type"] == "lightgbm_rdkit":
        return f"{row['baseline']}_{row['split']}_s{row['seed']}"
    epochs = int(config.get("max_epochs", config.get("epochs", 100)))
    limit_train = config.get("train_subset_size", config.get("limit_train"))
    n_part = limit_train if limit_train is not None else "full"
    return f"{row['backbone']}_{row['species_mode']}_{row['split']}_s{row['seed']}_e{epochs}_n{n_part}"


def _expected_rows(config: dict, split: str, seed: int) -> list[dict]:
    backbones = list(config.get("backbones") or ["dmpnn", "graphconv"])
    species_modes = _dedupe(list(config.get("main_species_modes", [])) + list(config.get("control_species_modes", [])))
    baselines = list(config.get("lightgbm_baselines", []))
    rows = []
    for backbone in backbones:
        for species_mode in species_modes:
            row = {
                "run_type": "gnn",
                "split": split,
                "seed": seed,
                "backbone": backbone,
                "species_mode": species_mode,
                "baseline": "",
            }
            row["run_id"] = _run_id(row, config)
            rows.append(row)
    for baseline in baselines:
        row = {
            "run_type": "lightgbm_rdkit",
            "split": split,
            "seed": seed,
            "backbone": "lightgbm_rdkit",
            "species_mode": "species_categorical" if baseline.endswith("species_categorical") else "no_species",
            "baseline": baseline,
        }
        row["run_id"] = _run_id(row, config)
        rows.append(row)
    return rows


def _prediction_schema(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "missing_required_columns": REQUIRED_PREDICTION_COLUMNS, "schema_complete": False}
    df = pd.read_csv(path, nrows=20)
    missing = [col for col in REQUIRED_PREDICTION_COLUMNS if col not in df.columns]
    return {"exists": True, "missing_required_columns": missing, "schema_complete": not missing}


def _prediction_has_nan(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        df = pd.read_csv(path, usecols=["true_log10", "pred_log10", "error_log10"])
    except (ValueError, EmptyDataError):
        return True
    return bool(df.replace([np.inf, -np.inf], np.nan).isna().any().any())


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        f"# JCIM v3 Full Batch Validation: {payload['split']} seed {payload['seed']}",
        "",
        f"- Generated at UTC: `{payload['generated_at_utc']}`",
        f"- Root: `{payload['root']}`",
        f"- Overall passed: `{payload['overall_passed']}`",
        f"- Planned runs: `{payload['counts']['planned_runs']}`",
        f"- Completed rows: `{payload['counts']['completed_rows']}`",
        f"- Failed rows: `{payload['counts']['failed_rows']}`",
        f"- Prediction CSV files: `{payload['counts']['prediction_files']}`",
        f"- Metric JSON files: `{payload['counts']['metric_json_files']}`",
        f"- Parameter count rows: `{payload['counts']['parameter_count_rows']}`",
        f"- Embedding CSV files: `{payload['counts']['embedding_csv_files']}`",
        f"- Species bias CSV files: `{payload['counts']['species_bias_csv_files']}`",
        "",
        "## Checks",
        "",
    ]
    for name, passed in payload["checks"].items():
        lines.append(f"- `{name}`: `{passed}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--warning-log")
    args = parser.parse_args()

    root = Path(args.root)
    config = _read_json(Path(args.config))
    expected = _expected_rows(config, args.split, args.seed)
    expected_ids = {row["run_id"] for row in expected}
    expected_gnn = [row for row in expected if row["run_type"] == "gnn"]
    expected_embedding_rows = [
        row
        for row in expected_gnn
        if row["species_mode"] not in {"no_species", "species_bias_only"}
    ]
    expected_bias_rows = [row for row in expected_gnn if row["species_mode"] == "species_bias_only"]

    completed_rows = [
        row
        for row in _read_csv(root / "completed_runs.csv")
        if row.get("run_id") in expected_ids
    ]
    failed_rows = [
        row
        for row in _read_csv(root / "failed_runs.csv")
        if row.get("run_id") in expected_ids
    ]
    parameter_rows = [
        row
        for row in _read_csv(root / "parameter_counts" / "parameter_counts.csv")
        if row.get("run_id") in expected_ids
    ]
    prediction_paths = {run_id: root / "predictions" / f"{run_id}.csv" for run_id in sorted(expected_ids)}
    metric_paths = {run_id: root / "metrics" / f"{run_id}.json" for run_id in sorted(expected_ids)}
    run_json_paths = {run_id: root / "runs" / f"{run_id}.json" for run_id in sorted(expected_ids)}

    schema = {run_id: _prediction_schema(path) for run_id, path in prediction_paths.items()}
    nan_predictions = [run_id for run_id, path in prediction_paths.items() if _prediction_has_nan(path)]
    embedding_csv_files = list((root / "embeddings").glob(f"*__{args.split}__seed{args.seed}__species_embeddings.csv"))
    embedding_metadata_files = list((root / "embeddings").glob(f"*__{args.split}__seed{args.seed}__species_embedding_metadata.json"))
    species_bias_csv_files = list((root / "embeddings").glob(f"*_{args.split}_s{args.seed}_*species_bias.csv"))
    if not species_bias_csv_files:
        species_bias_csv_files = list((root / "embeddings").glob(f"*__{args.split}__seed{args.seed}__species_bias.csv"))

    warning_candidates = []
    if args.warning_log:
        warning_candidates.append(Path(args.warning_log))
    warning_candidates.extend(
        [
            root / "environment_warnings.log",
            root / "batch_validation" / f"{args.split}_seed{args.seed}_environment_warnings.log",
        ]
    )
    warning_log = next((path for path in warning_candidates if path.exists()), None)

    counts = {
        "planned_runs": len(expected),
        "completed_rows": len(completed_rows),
        "failed_rows": len(failed_rows),
        "prediction_files": sum(path.exists() for path in prediction_paths.values()),
        "metric_json_files": sum(path.exists() for path in metric_paths.values()),
        "run_json_files": sum(path.exists() for path in run_json_paths.values()),
        "parameter_count_rows": len(parameter_rows),
        "embedding_csv_files": len(embedding_csv_files),
        "embedding_metadata_files": len(embedding_metadata_files),
        "species_bias_csv_files": len(species_bias_csv_files),
    }
    checks = {
        "planned_run_count_is_38": counts["planned_runs"] == 38,
        "completed_row_count_is_38": counts["completed_rows"] == 38,
        "failed_row_count_is_0": counts["failed_rows"] == 0,
        "prediction_count_is_38": counts["prediction_files"] == 38,
        "metric_json_count_is_38": counts["metric_json_files"] == 38,
        "run_json_count_is_38": counts["run_json_files"] == 38,
        "parameter_count_rows_is_38": counts["parameter_count_rows"] == 38,
        "embedding_csv_count_is_32": counts["embedding_csv_files"] == len(expected_embedding_rows),
        "embedding_metadata_count_is_36": counts["embedding_metadata_files"] == len(expected_gnn),
        "species_bias_csv_count_is_2": counts["species_bias_csv_files"] == len(expected_bias_rows),
        "prediction_schema_complete": bool(schema) and all(row["schema_complete"] for row in schema.values()),
        "prediction_values_have_no_nan": not nan_predictions,
        "environment_warning_log_recorded": warning_log is not None,
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "split": args.split,
        "seed": args.seed,
        "counts": counts,
        "checks": checks,
        "overall_passed": all(checks.values()),
        "missing_predictions": [run_id for run_id, path in prediction_paths.items() if not path.exists()],
        "missing_metrics": [run_id for run_id, path in metric_paths.items() if not path.exists()],
        "missing_run_json": [run_id for run_id, path in run_json_paths.items() if not path.exists()],
        "schema": schema,
        "nan_prediction_run_ids": nan_predictions,
        "failed_runs": failed_rows,
        "environment_warning_log": str(warning_log) if warning_log else None,
        "training_was_executed": False,
    }
    out_dir = root / "batch_validation"
    json_path = out_dir / f"{args.split}_seed{args.seed}_summary.json"
    md_path = out_dir / f"{args.split}_seed{args.seed}_summary.md"
    _write_json(json_path, payload)
    _write_markdown(md_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
