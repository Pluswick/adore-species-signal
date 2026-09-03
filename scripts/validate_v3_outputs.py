from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.paths import CC_MPNN_DATA, RESULTS_ROOT


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


def _csv_info(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "rows": None, "columns": []}
    df = pd.read_csv(path)
    return {"exists": True, "rows": int(len(df)), "columns": list(df.columns)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(CC_MPNN_DATA))
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--smoke-root", default=str(RESULTS_ROOT / "smoke" / "src_env"))
    parser.add_argument(
        "--out",
        default=str(RESULTS_ROOT / "data_audit" / "src_revalidation_summary.json"),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_root = Path(args.results_root)
    smoke_root = Path(args.smoke_root)
    audit_dir = results_root / "data_audit"
    feature_dir = results_root / "features"

    consolidated = pd.read_csv(data_dir / "lc50_96_consolidated.csv")
    feature_summary_path = feature_dir / "feature_cache_summary.json"
    censored_summary_path = audit_dir / "censored_summary.json"

    with open(feature_summary_path, encoding="utf-8") as f:
        feature_summary = json.load(f)
    with open(censored_summary_path, encoding="utf-8") as f:
        censored_summary = json.load(f)

    audit_files = {
        "rdkit_descriptor_failures": _csv_info(audit_dir / "rdkit_descriptor_failures.csv"),
        "scaffold_failures": _csv_info(audit_dir / "scaffold_failures.csv"),
        "species_count_summary": _csv_info(audit_dir / "species_count_summary.csv"),
        "compound_count_summary": _csv_info(audit_dir / "compound_count_summary.csv"),
        "split_summary": _csv_info(audit_dir / "split_summary.csv"),
    }

    prediction_files = sorted((smoke_root / "predictions").glob("*.csv"))
    prediction_schema = {}
    for path in prediction_files:
        df = pd.read_csv(path, nrows=5)
        missing = [col for col in REQUIRED_PREDICTION_COLUMNS if col not in df.columns]
        prediction_schema[path.name] = {
            "path": str(path),
            "missing_required_columns": missing,
            "columns": list(df.columns),
            "rows": int(sum(1 for _ in open(path, encoding="utf-8")) - 1),
        }

    checks = {
        "main_dataset_rows_is_17229": int(len(consolidated)) == 17229,
        "unique_smiles_is_4954": int(consolidated["smiles"].nunique()) == 4954,
        "species_count_is_1481": int(consolidated["species"].nunique()) == 1481,
        "censored_lc50_96_is_zero": int(
            censored_summary["lc50_96_censored_records_excluded_from_main"]
        )
        == 0,
        "rdkit_descriptor_failures_is_zero": int(
            feature_summary["rdkit_descriptor_failures"]
        )
        == 0,
        "scaffold_failures_is_zero": int(feature_summary["scaffold_failures"]) == 0,
        "six_smoke_prediction_files_present": len(prediction_files) == 6,
        "prediction_schema_complete": all(
            not info["missing_required_columns"] for info in prediction_schema.values()
        )
        and len(prediction_files) == 6,
    }

    summary = {
        "data_dir": str(data_dir),
        "results_root": str(results_root),
        "smoke_root": str(smoke_root),
        "consolidated": {
            "rows": int(len(consolidated)),
            "unique_smiles": int(consolidated["smiles"].nunique()),
            "unique_species": int(consolidated["species"].nunique()),
        },
        "feature_summary": {
            "generated_with": feature_summary.get("generated_with"),
            "n_unique_smiles": feature_summary["n_unique_smiles"],
            "rdkit_descriptor_success": feature_summary["rdkit_descriptor_success"],
            "rdkit_descriptor_failures": feature_summary["rdkit_descriptor_failures"],
            "scaffold_success": feature_summary["scaffold_success"],
            "scaffold_failures": feature_summary["scaffold_failures"],
        },
        "censored_summary": censored_summary,
        "audit_files": audit_files,
        "prediction_schema": prediction_schema,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"json: {out}")


if __name__ == "__main__":
    main()
