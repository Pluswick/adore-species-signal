from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.paths import CC_MPNN_DATA, RESULTS_ROOT


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _embedding_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("embedding_dim_")]


def _dedupe_warnings(warnings: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for warning in warnings:
        key = json.dumps(warning, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(warning)
    return out


def _add_check(checks: list[dict], name: str, passed: bool, **details: object) -> None:
    row = {"check": name, "passed": bool(passed)}
    row.update(details)
    checks.append(row)


def _validate_embedding_file(
    path: Path,
    *,
    expected_species: int,
    expected_dim: int,
    checks: list[dict],
    warnings: list[dict],
) -> None:
    if not path.exists():
        _add_check(checks, f"embedding_exists::{path.name}", False, path=str(path))
        return
    frame = pd.read_csv(path)
    dim_cols = _embedding_columns(frame)
    required = {"species", "species_idx", "train_count", "test_count"}
    _add_check(
        checks,
        f"embedding_required_columns::{path.name}",
        required.issubset(frame.columns),
        missing=sorted(required.difference(frame.columns)),
    )
    _add_check(
        checks,
        f"embedding_row_count::{path.name}",
        len(frame) == expected_species,
        observed=int(len(frame)),
        expected=int(expected_species),
    )
    _add_check(
        checks,
        f"embedding_dim::{path.name}",
        len(dim_cols) == expected_dim,
        observed=int(len(dim_cols)),
        expected=int(expected_dim),
    )
    _add_check(
        checks,
        f"embedding_species_idx_unique::{path.name}",
        frame["species_idx"].is_unique if "species_idx" in frame else False,
    )
    species_present = "species" in frame and int(frame["species"].isna().sum()) == 0
    _add_check(checks, f"embedding_species_names_present::{path.name}", species_present)
    if dim_cols:
        finite = np.isfinite(frame[dim_cols].to_numpy(np.float64)).all()
        _add_check(checks, f"embedding_values_finite::{path.name}", bool(finite))
    else:
        warnings.append({"file": str(path), "warning": "no embedding_dim_* columns found"})


def _validate_bias_file(
    path: Path,
    *,
    expected_species: int,
    checks: list[dict],
) -> None:
    if not path.exists():
        _add_check(checks, f"species_bias_exists::{path.name}", False, path=str(path))
        return
    frame = pd.read_csv(path)
    required = {
        "species",
        "species_idx",
        "species_bias_model_output_space",
        "species_bias_log10_space",
    }
    _add_check(
        checks,
        f"species_bias_required_columns::{path.name}",
        required.issubset(frame.columns),
        missing=sorted(required.difference(frame.columns)),
    )
    _add_check(
        checks,
        f"species_bias_row_count::{path.name}",
        len(frame) == expected_species,
        observed=int(len(frame)),
        expected=int(expected_species),
    )
    _add_check(
        checks,
        f"species_bias_species_idx_unique::{path.name}",
        frame["species_idx"].is_unique if "species_idx" in frame else False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v3_embedding_smoke.json")
    parser.add_argument(
        "--out-root",
        default=str(RESULTS_ROOT / "smoke" / "src_env" / "embedding_smoke"),
    )
    parser.add_argument("--checks-out")
    parser.add_argument("--warnings-out")
    args = parser.parse_args()

    config = _load_config(args.config)
    out_root = Path(config.get("out_root", args.out_root))
    data_dir = Path(config.get("dataset_path", str(CC_MPNN_DATA)))
    species_index = pd.read_csv(data_dir / "species_index.csv")
    expected_species = int(len(species_index))
    expected_dim = int(config.get("species_emb_dim", 16))
    backbones = config.get("backbone_list", ["dmpnn", "graphconv"])
    species_modes = config.get("species_mode_list", [])

    checks: list[dict] = []
    validation_warnings: list[dict] = []
    embeddings_dir = out_root / "embeddings"

    summary_path = out_root / "embedding_smoke_summary.csv"
    _add_check(checks, "embedding_smoke_summary_exists", summary_path.exists(), path=str(summary_path))
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    expected_runs = int(len(backbones) * len(species_modes))
    _add_check(
        checks,
        "embedding_smoke_summary_run_count",
        len(summary) == expected_runs,
        observed=int(len(summary)),
        expected=expected_runs,
    )

    metadata_files = sorted(embeddings_dir.glob("*__species_embedding_metadata.json"))
    _add_check(
        checks,
        "species_embedding_metadata_count",
        len(metadata_files) == expected_runs,
        observed=int(len(metadata_files)),
        expected=expected_runs,
    )

    for _, row in summary.iterrows():
        mode = str(row.get("species_mode"))
        embedding_path = row.get("species_embeddings")
        metadata_path = row.get("species_embedding_metadata")
        bias_path = row.get("species_bias")
        if isinstance(metadata_path, str) and metadata_path:
            meta = _read_json(Path(metadata_path))
            _add_check(
                checks,
                f"metadata_exists::{Path(metadata_path).name}",
                bool(meta),
                path=metadata_path,
            )
            _add_check(
                checks,
                f"metadata_species_dim::{Path(metadata_path).name}",
                int(meta.get("species_emb_dim", expected_dim)) == expected_dim,
                observed=meta.get("species_emb_dim"),
                expected=expected_dim,
            )
        else:
            _add_check(checks, f"metadata_path_present::{mode}", False)

        if mode == "species_bias_only":
            if isinstance(bias_path, str) and bias_path:
                _validate_bias_file(Path(bias_path), expected_species=expected_species, checks=checks)
            else:
                _add_check(checks, f"species_bias_path_present::{mode}", False)
            continue

        if isinstance(embedding_path, str) and embedding_path:
            _validate_embedding_file(
                Path(embedding_path),
                expected_species=expected_species,
                expected_dim=expected_dim,
                checks=checks,
                warnings=validation_warnings,
            )
        else:
            _add_check(checks, f"embedding_path_present::{mode}", False)

    species_summary_path = out_root / "species_summary.csv"
    _add_check(checks, "species_summary_exists", species_summary_path.exists(), path=str(species_summary_path))
    if species_summary_path.exists():
        species_summary = pd.read_csv(species_summary_path)
        required_species_summary = {
            "species",
            "species_idx",
            "train_count",
            "test_count",
            "total_count",
            "train_mean_true",
            "test_mean_true",
            "mean_pred",
            "mean_error",
            "mean_abs_error",
            "rmse_by_species",
            "species_count_bin",
            "has_enough_test_rows",
            "n_test_rows_for_summary",
        }
        _add_check(
            checks,
            "species_summary_required_columns",
            required_species_summary.issubset(species_summary.columns),
            missing=sorted(required_species_summary.difference(species_summary.columns)),
        )
        _add_check(
            checks,
            "species_summary_has_rows",
            len(species_summary) > 0,
            rows=int(len(species_summary)),
        )

    corr_path = out_root / "embedding_sensitivity_correlation.csv"
    _add_check(checks, "embedding_sensitivity_correlation_exists", corr_path.exists(), path=str(corr_path))
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        required_corr = {
            "backbone",
            "species_mode",
            "embedding_distance_type",
            "sensitivity_variable",
            "n_species",
            "n_pairs",
            "spearman_r",
            "spearman_p",
            "pearson_r",
            "pearson_p",
        }
        _add_check(
            checks,
            "embedding_sensitivity_correlation_required_columns",
            required_corr.issubset(corr.columns),
            missing=sorted(required_corr.difference(corr.columns)),
        )
        _add_check(checks, "embedding_sensitivity_correlation_has_rows", len(corr) > 0, rows=int(len(corr)))

    pca_path = out_root / "embedding_pca_coordinates.csv"
    _add_check(checks, "embedding_pca_coordinates_exists", pca_path.exists(), path=str(pca_path))
    if pca_path.exists():
        pca = pd.read_csv(pca_path)
        required_pca = {"species", "species_idx", "backbone", "species_mode", "pc1", "pc2"}
        _add_check(
            checks,
            "embedding_pca_required_columns",
            required_pca.issubset(pca.columns),
            missing=sorted(required_pca.difference(pca.columns)),
        )
        finite_pca = False
        if {"pc1", "pc2"}.issubset(pca.columns):
            finite_pca = np.isfinite(pca[["pc1", "pc2"]].to_numpy(np.float64)).all()
        _add_check(checks, "embedding_pca_no_nan", bool(finite_pca))

    pca_plot = out_root / "embedding_pca_plot.png"
    _add_check(checks, "embedding_pca_plot_exists", pca_plot.exists(), path=str(pca_plot))

    vs_bias_path = out_root / "embedding_vs_species_bias.csv"
    _add_check(checks, "embedding_vs_species_bias_exists", vs_bias_path.exists(), path=str(vs_bias_path))
    if vs_bias_path.exists():
        vs_bias = pd.read_csv(vs_bias_path)
        _add_check(checks, "embedding_vs_species_bias_has_rows", len(vs_bias) > 0, rows=int(len(vs_bias)))

    scaffold_path = out_root / "scaffold_improvement_summary.csv"
    _add_check(checks, "scaffold_improvement_summary_exists", scaffold_path.exists(), path=str(scaffold_path))
    if scaffold_path.exists():
        scaffold = pd.read_csv(scaffold_path)
        _add_check(checks, "scaffold_improvement_summary_has_rows", len(scaffold) > 0, rows=int(len(scaffold)))

    warnings_path = Path(args.warnings_out) if args.warnings_out else out_root / "embedding_warnings.json"
    source_warnings_path = out_root / "embedding_warnings.json"
    existing_warnings = _read_json(source_warnings_path).get("warnings", [])
    if args.warnings_out and warnings_path.exists():
        existing_warnings.extend(_read_json(warnings_path).get("warnings", []))
    if not config.get("enable_umap", False):
        has_umap_warning = any(
            "umap" in str(warning).lower() for warning in existing_warnings
        )
        _add_check(checks, "umap_disabled_warning_recorded", has_umap_warning)
        if not has_umap_warning:
            validation_warnings.append(
                {
                    "analysis": "umap",
                    "warning": "UMAP is disabled in config, but no UMAP warning was found.",
                }
            )

    combined_warnings = _dedupe_warnings(existing_warnings + validation_warnings)
    _write_json(warnings_path, {"warnings": combined_warnings})

    all_passed = all(row["passed"] for row in checks)
    checks_payload = {
        "overall_passed": bool(all_passed),
        "out_root": str(out_root),
        "expected_species": expected_species,
        "expected_embedding_dim": expected_dim,
        "checks": checks,
    }
    checks_path = Path(args.checks_out) if args.checks_out else out_root / "embedding_sanity_checks.json"
    _write_json(checks_path, checks_payload)
    print(json.dumps(checks_payload, ensure_ascii=False, indent=2))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
