from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.bootstrap import PAIR_MERGE_KEYS
from src.paths import RESULTS_ROOT


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


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _file_info(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
    }


def _prediction_schema(prediction_dir: Path, block_key: str) -> dict:
    files = sorted(prediction_dir.glob("*.csv"))
    out = {}
    for path in files:
        df = pd.read_csv(path, nrows=20)
        missing = [col for col in REQUIRED_PREDICTION_COLUMNS if col not in df.columns]
        out[path.name] = {
            "missing_required_columns": missing,
            "block_key": block_key,
            "block_key_missing": block_key not in df.columns,
            "block_key_null_in_sample": bool(df[block_key].isna().any()) if block_key in df else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v3_stats_smoke.json")
    parser.add_argument(
        "--out",
        default=str(
            RESULTS_ROOT / "smoke" / "src_env" / "stats_smoke" / "stats_sanity_checks.json"
        ),
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    prediction_dir = Path(
        config.get(
            "prediction_root",
            RESULTS_ROOT / "smoke" / "src_env" / "injection_positions" / "predictions",
        )
    )
    out_dir = Path(config.get("output_root", RESULTS_ROOT / "smoke" / "src_env" / "stats_smoke"))
    block_key = str(config.get("block_key", "scaffold_key"))

    files = {
        "aggregated_metrics_csv": out_dir / "aggregated_metrics.csv",
        "aggregated_metrics_json": out_dir / "aggregated_metrics.json",
        "species_bin_metrics_csv": out_dir / "species_bin_metrics.csv",
        "bootstrap_raw_csv": out_dir / "bootstrap_comparisons_raw.csv",
        "bootstrap_fdr_csv": out_dir / "bootstrap_comparisons_fdr.csv",
        "bootstrap_warnings_json": out_dir / "bootstrap_warnings.json",
    }
    file_info = {name: _file_info(path) for name, path in files.items()}

    prediction_schema = _prediction_schema(prediction_dir, block_key)
    schema_complete = all(not item["missing_required_columns"] for item in prediction_schema.values())
    block_key_present = all(not item["block_key_missing"] for item in prediction_schema.values())

    raw = pd.read_csv(files["bootstrap_raw_csv"]) if files["bootstrap_raw_csv"].exists() else pd.DataFrame()
    fdr = pd.read_csv(files["bootstrap_fdr_csv"]) if files["bootstrap_fdr_csv"].exists() else pd.DataFrame()
    agg = pd.read_csv(files["aggregated_metrics_csv"]) if files["aggregated_metrics_csv"].exists() else pd.DataFrame()
    warnings = {}
    if files["bootstrap_warnings_json"].exists():
        with open(files["bootstrap_warnings_json"], encoding="utf-8") as f:
            warnings = json.load(f)

    ci_contains = (
        (raw["ci_low"] <= raw["delta"]) & (raw["delta"] <= raw["ci_high"])
        if len(raw)
        else pd.Series(dtype=bool)
    )
    ci_contains_rate = float(ci_contains.mean()) if len(ci_contains) else 0.0
    checks = {
        "all_required_output_files_exist": all(info["exists"] for info in file_info.values()),
        "prediction_schema_complete": schema_complete,
        "block_key_present_in_predictions": block_key_present,
        "bootstrap_raw_has_rows": len(raw) > 0,
        "bootstrap_fdr_has_same_rows_as_raw": len(raw) == len(fdr) and len(raw) > 0,
        "aggregated_metrics_has_prediction_file_rows": len(agg) == len(prediction_schema),
        "bootstrap_results_have_no_nan": not raw.isna().any().any() if len(raw) else False,
        "fdr_results_have_no_nan": not fdr.isna().any().any() if len(fdr) else False,
        "ci_contains_delta_rate_at_least_0_90": ci_contains_rate >= 0.90,
        "q_values_in_0_1": bool(
            ((fdr["q_value_bh_fdr"] >= 0.0) & (fdr["q_value_bh_fdr"] <= 1.0)).all()
        )
        if len(fdr) and "q_value_bh_fdr" in fdr
        else False,
        "paired_comparisons_not_skipped": warnings.get("n_completed_comparisons")
        == warnings.get("n_attempted_comparisons"),
        "merge_warnings_absent": len(warnings.get("warnings", [])) == 0,
    }

    summary = {
        "config": config,
        "prediction_dir": str(prediction_dir),
        "out_dir": str(out_dir),
        "block_key": block_key,
        "pair_merge_keys": PAIR_MERGE_KEYS,
        "files": file_info,
        "prediction_schema": prediction_schema,
        "n_prediction_files": len(prediction_schema),
        "n_aggregated_rows": int(len(agg)),
        "n_bootstrap_raw_rows": int(len(raw)),
        "n_bootstrap_fdr_rows": int(len(fdr)),
        "ci_contains_delta_count": int(ci_contains.sum()) if len(ci_contains) else 0,
        "ci_contains_delta_rate": ci_contains_rate,
        "bootstrap_warnings": warnings,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
