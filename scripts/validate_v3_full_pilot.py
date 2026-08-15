from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "results" / "jcim_v3" / "full_pilot_scaffold_seed0"
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_csv(path)))
    except EmptyDataError:
        return 0


def _file_info(path: Path) -> dict:
    return {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}


def _has_nan(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return True
    if df.empty:
        return True
    return bool(df.replace([np.inf, -np.inf], np.nan).isna().any().any())


def _q_values_in_range(path: Path) -> bool:
    if not path.exists():
        return False
    df = pd.read_csv(path)
    if df.empty or "q_value_bh_fdr" not in df:
        return False
    return bool(((df["q_value_bh_fdr"] >= 0.0) & (df["q_value_bh_fdr"] <= 1.0)).all())


def _prediction_schema(prediction_dir: Path) -> dict:
    out = {}
    for path in sorted(prediction_dir.glob("*.csv")):
        df = pd.read_csv(path, nrows=20)
        missing = [col for col in REQUIRED_PREDICTION_COLUMNS if col not in df.columns]
        out[path.name] = {
            "path": str(path),
            "missing_required_columns": missing,
            "schema_complete": len(missing) == 0,
        }
    return out


def _warning_summary(path: Path) -> dict:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    warning_lines = [
        line
        for line in text.splitlines()
        if any(token in line.lower() for token in ["warning", "access is denied", "could not find", "dll", "opencl"])
    ]
    return {
        "path": str(path),
        "exists": path.exists(),
        "warning_line_count": len(warning_lines),
        "warning_lines": warning_lines[:80],
    }


def _write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# JCIM v3 Full-Scale Pilot Summary",
        "",
        f"- Generated at UTC: `{summary['generated_at_utc']}`",
        f"- Output root: `{summary['output_root']}`",
        f"- Overall sanity passed: `{summary['overall_passed']}`",
        f"- Planned runs: `{summary['counts']['planned_runs']}`",
        f"- Completed/skipped runs: `{summary['counts']['completed_runs']}`",
        f"- Failed runs: `{summary['counts']['failed_runs']}`",
        f"- Prediction CSV files: `{summary['counts']['prediction_csv_files']}`",
        f"- Metric JSON files: `{summary['counts']['metric_json_files']}`",
        f"- Embedding CSV files: `{summary['counts']['embedding_csv_files']}`",
        f"- Embedding metadata files: `{summary['counts']['embedding_metadata_files']}`",
        f"- Species bias CSV files: `{summary['counts']['species_bias_csv_files']}`",
        "",
        "Pilot results are used only to validate full-scale execution and analysis compatibility. They are not used for scientific claims.",
        "",
        "## Checks",
        "",
    ]
    for name, passed in summary["checks"].items():
        lines.append(f"- `{name}`: `{passed}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()

    root = Path(args.root)
    manifest = root / "run_manifest.csv"
    completed = root / "completed_runs.csv"
    failed = root / "failed_runs.csv"
    parameter_counts = root / "parameter_counts" / "parameter_counts.csv"
    files = {
        "aggregated_metrics": root / "metrics" / "aggregated_metrics.csv",
        "species_bin_metrics": root / "metrics" / "species_bin_metrics.csv",
        "bootstrap_raw": root / "bootstrap" / "bootstrap_comparisons_raw.csv",
        "bootstrap_fdr": root / "bootstrap" / "bootstrap_comparisons_fdr.csv",
        "embedding_correlation": root / "embedding_analysis" / "embedding_sensitivity_correlation.csv",
        "embedding_pca": root / "embedding_analysis" / "embedding_pca_coordinates.csv",
        "embedding_pca_plot": root / "embedding_analysis" / "embedding_pca_plot.png",
        "embedding_vs_bias": root / "embedding_analysis" / "embedding_vs_species_bias.csv",
        "scaffold_improvement": root / "embedding_analysis" / "scaffold_improvement_summary.csv",
        "summary_metric_by_model": root / "summary_tables" / "metric_summary_by_model.csv",
        "summary_bootstrap_fdr": root / "summary_tables" / "bootstrap_fdr_summary.csv",
    }
    prediction_schema = _prediction_schema(root / "predictions")
    bootstrap_warnings = _read_json(root / "bootstrap" / "bootstrap_warnings.json")
    analysis_summary = _read_json(root / "full_analysis_summary.json")
    counts = {
        "planned_runs": _csv_rows(manifest),
        "completed_runs": _csv_rows(completed),
        "failed_runs": _csv_rows(failed),
        "parameter_count_rows": _csv_rows(parameter_counts),
        "prediction_csv_files": len(list((root / "predictions").glob("*.csv"))),
        "metric_json_files": len([p for p in (root / "metrics").glob("*.json") if p.name != "aggregated_metrics.json"]),
        "run_json_files": len(list((root / "runs").glob("*.json"))),
        "embedding_csv_files": len(list((root / "embeddings").glob("*__species_embeddings.csv"))),
        "embedding_metadata_files": len(list((root / "embeddings").glob("*__species_embedding_metadata.json"))),
        "species_bias_csv_files": len(list((root / "embeddings").glob("*__species_bias.csv"))),
        "aggregated_metrics_rows": _csv_rows(files["aggregated_metrics"]),
        "species_bin_metric_rows": _csv_rows(files["species_bin_metrics"]),
        "bootstrap_raw_rows": _csv_rows(files["bootstrap_raw"]),
        "bootstrap_fdr_rows": _csv_rows(files["bootstrap_fdr"]),
        "scaffold_improvement_rows": _csv_rows(files["scaffold_improvement"]),
    }
    schema_complete = bool(prediction_schema) and all(row["schema_complete"] for row in prediction_schema.values())
    paired_complete = bootstrap_warnings.get("n_attempted_comparisons") == bootstrap_warnings.get("n_completed_comparisons")
    checks = {
        "manifest_exists": manifest.exists(),
        "planned_run_count_is_38": counts["planned_runs"] == 38,
        "completed_or_skipped_run_count_is_38": counts["completed_runs"] == 38,
        "failed_run_count_is_0": counts["failed_runs"] == 0,
        "prediction_count_is_38": counts["prediction_csv_files"] == 38,
        "metric_json_count_is_38": counts["metric_json_files"] == 38,
        "run_json_count_is_38": counts["run_json_files"] == 38,
        "parameter_count_rows_is_38": counts["parameter_count_rows"] == 38,
        "embedding_csv_count_is_32": counts["embedding_csv_files"] == 32,
        "embedding_metadata_count_is_36": counts["embedding_metadata_files"] == 36,
        "species_bias_csv_count_is_2": counts["species_bias_csv_files"] == 2,
        "prediction_schema_complete": schema_complete,
        "aggregated_metrics_exists_nonempty": files["aggregated_metrics"].exists() and counts["aggregated_metrics_rows"] > 0,
        "species_bin_metrics_exists_nonempty": files["species_bin_metrics"].exists() and counts["species_bin_metric_rows"] > 0,
        "bootstrap_raw_exists_nonempty": files["bootstrap_raw"].exists() and counts["bootstrap_raw_rows"] > 0,
        "bootstrap_fdr_exists_nonempty": files["bootstrap_fdr"].exists() and counts["bootstrap_fdr_rows"] > 0,
        "bootstrap_fdr_rows_match_raw": counts["bootstrap_raw_rows"] == counts["bootstrap_fdr_rows"] and counts["bootstrap_raw_rows"] > 0,
        "paired_comparisons_not_skipped": paired_complete,
        "bootstrap_warnings_absent": len(bootstrap_warnings.get("warnings", [])) == 0,
        "fdr_q_values_in_range": _q_values_in_range(files["bootstrap_fdr"]),
        "embedding_analysis_outputs_exist": all(
            files[name].exists()
            for name in ["embedding_correlation", "embedding_pca", "embedding_pca_plot", "embedding_vs_bias"]
        ),
        "scaffold_improvement_exists_nonempty": files["scaffold_improvement"].exists() and counts["scaffold_improvement_rows"] > 0,
        "summary_tables_exist": files["summary_metric_by_model"].exists() and files["summary_bootstrap_fdr"].exists(),
        "analysis_runner_passed": bool(analysis_summary.get("overall_passed", False)),
        "key_metric_outputs_no_nan": not any(_has_nan(files[name]) for name in ["aggregated_metrics", "bootstrap_raw", "bootstrap_fdr"]),
        "environment_warning_log_exists": (root / "environment_warnings.log").exists(),
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(root),
        "counts": counts,
        "files": {name: _file_info(path) for name, path in files.items()},
        "prediction_schema": prediction_schema,
        "bootstrap_warnings": bootstrap_warnings,
        "environment_warnings": _warning_summary(root / "environment_warnings.log"),
        "checks": checks,
        "overall_passed": all(checks.values()),
        "interpretation_limit": "Pilot results are used only to validate full-scale execution and analysis compatibility. They are not used for scientific claims.",
    }
    _write_json(root / "pilot_sanity_checks.json", payload)
    _write_json(root / "pilot_summary.json", payload)
    _write_markdown(root / "pilot_summary.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
