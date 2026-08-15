from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULL_CONFIG = ROOT / "configs" / "v3_full_experiment.json"
DEFAULT_PILOT_CONFIG = ROOT / "configs" / "v3_full_pilot_scaffold_seed0.json"
DEFAULT_PILOT_ROOT = ROOT / "results" / "jcim_v3" / "full_pilot_scaffold_seed0"
DEFAULT_OUT_DIR = ROOT / "results" / "jcim_v3" / "full_run_plan"

TRAINING_FIELDS = [
    "official_env",
    "target",
    "dataset_path",
    "max_epochs",
    "early_stopping_patience",
    "batch_size",
    "lr",
    "weight_decay",
    "dropout",
    "hidden",
    "depth",
    "species_emb_dim",
    "val_frac",
    "train_subset_size",
    "test_subset_size",
    "backbones",
    "main_species_modes",
    "control_species_modes",
    "lightgbm_baselines",
]

IMPLEMENTATION_FILES = [
    ROOT / "scripts" / "run_v3_full_experiment.py",
    ROOT / "jcim_v3" / "runner.py",
    ROOT / "jcim_v3" / "models.py",
    ROOT / "jcim_v3" / "dataset.py",
    ROOT / "jcim_v3" / "featurizer.py",
    ROOT / "jcim_v3" / "species_controls.py",
    ROOT / "jcim_v3" / "rdkit_lgbm.py",
    ROOT / "jcim_v3" / "rdkit_features.py",
]


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_meta(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        if path.exists()
        else None,
        "sha256": _sha256(path),
    }


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        lines = sum(1 for _ in f)
    return max(0, lines - 1)


def _compare_field(full: dict, pilot: dict, field: str) -> dict:
    return {
        "full_value": full.get(field),
        "pilot_value": pilot.get(field),
        "matches": full.get(field) == pilot.get(field),
    }


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# JCIM v3 Pilot Reuse Check",
        "",
        f"- Generated at UTC: `{payload['generated_at_utc']}`",
        f"- Full config: `{payload['full_config']}`",
        f"- Pilot config: `{payload['pilot_config']}`",
        f"- Reusable: `{payload['reusable']}`",
        f"- Reason: `{payload['reuse_decision']}`",
        "",
        "## Scope",
        "",
        f"- Full split contains scaffold: `{payload['scope_checks']['full_contains_scaffold']}`",
        f"- Full seed contains 0: `{payload['scope_checks']['full_contains_seed0']}`",
        f"- Pilot split exactly scaffold: `{payload['scope_checks']['pilot_is_scaffold_only']}`",
        f"- Pilot seed exactly 0: `{payload['scope_checks']['pilot_is_seed0_only']}`",
        "",
        "## Training Field Comparison",
        "",
        "| Field | Matches | Full | Pilot |",
        "|---|---:|---|---|",
    ]
    for field, row in payload["field_comparison"].items():
        lines.append(
            f"| `{field}` | `{row['matches']}` | `{json.dumps(row['full_value'], ensure_ascii=False)}` | `{json.dumps(row['pilot_value'], ensure_ascii=False)}` |"
        )
    lines.extend(
        [
            "",
            "## Non-Gating Differences",
            "",
            f"- Full output root: `{payload['non_gating_differences']['full_output_root']}`",
            f"- Pilot output root: `{payload['non_gating_differences']['pilot_output_root']}`",
            f"- Full analysis n_bootstrap: `{payload['non_gating_differences']['full_analysis_n_bootstrap']}`",
            f"- Pilot analysis n_bootstrap: `{payload['non_gating_differences']['pilot_analysis_n_bootstrap']}`",
            "",
            "## Pilot Completion Evidence",
            "",
            f"- Pilot sanity overall passed: `{payload['pilot_completion']['pilot_sanity_overall_passed']}`",
            f"- Pilot completed runs: `{payload['pilot_completion']['completed_runs']}`",
            f"- Pilot failed runs: `{payload['pilot_completion']['failed_runs']}`",
            f"- Pilot prediction CSV files: `{payload['pilot_completion']['prediction_csv_files']}`",
            "",
            "This file only validates whether the scaffold seed 0 pilot can be reused. It does not copy files or execute training.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-config", default=str(DEFAULT_FULL_CONFIG))
    parser.add_argument("--pilot-config", default=str(DEFAULT_PILOT_CONFIG))
    parser.add_argument("--pilot-root", default=str(DEFAULT_PILOT_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    full_config_path = Path(args.full_config)
    pilot_config_path = Path(args.pilot_config)
    pilot_root = Path(args.pilot_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full = _read_json(full_config_path)
    pilot = _read_json(pilot_config_path)
    pilot_summary_path = pilot_root / "pilot_sanity_checks.json"
    pilot_summary = _read_json(pilot_summary_path) if pilot_summary_path.exists() else {}

    field_comparison = {field: _compare_field(full, pilot, field) for field in TRAINING_FIELDS}
    scope_checks = {
        "full_contains_scaffold": "scaffold" in list(full.get("split_types", [])),
        "full_contains_seed0": 0 in [int(seed) for seed in full.get("seeds", [])],
        "pilot_is_scaffold_only": list(pilot.get("split_types", [])) == ["scaffold"],
        "pilot_is_seed0_only": [int(seed) for seed in pilot.get("seeds", [])] == [0],
    }

    feature_dir = ROOT / "results" / "jcim_v3" / "features"
    feature_paths = {
        "rdkit_descriptor_cache": feature_dir / "rdkit6_by_smiles.csv",
        "scaffold_key_cache": feature_dir / "scaffold_by_smiles.csv",
    }
    feature_cache = {name: _file_meta(path) for name, path in feature_paths.items()}
    implementation = {path.relative_to(ROOT).as_posix(): _file_meta(path) for path in IMPLEMENTATION_FILES}

    pilot_completion = {
        "pilot_sanity_overall_passed": bool(pilot_summary.get("overall_passed", False)),
        "completed_runs": _csv_row_count(pilot_root / "completed_runs.csv"),
        "failed_runs": _csv_row_count(pilot_root / "failed_runs.csv"),
        "prediction_csv_files": len(list((pilot_root / "predictions").glob("*.csv"))),
    }

    gating_checks = {
        "all_training_fields_match": all(row["matches"] for row in field_comparison.values()),
        "scope_matches_full_scaffold_seed0": all(scope_checks.values()),
        "feature_caches_exist": all(row["exists"] for row in feature_cache.values()),
        "implementation_files_exist": all(row["exists"] for row in implementation.values()),
        "pilot_sanity_passed": pilot_completion["pilot_sanity_overall_passed"],
        "pilot_has_38_completed_runs": pilot_completion["completed_runs"] == 38,
        "pilot_has_0_failed_runs": pilot_completion["failed_runs"] == 0,
        "pilot_has_38_predictions": pilot_completion["prediction_csv_files"] == 38,
    }
    reusable = all(gating_checks.values())
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_config": str(full_config_path),
        "pilot_config": str(pilot_config_path),
        "pilot_root": str(pilot_root),
        "reusable": reusable,
        "reuse_decision": "scaffold seed 0 pilot is reusable for the matching full batch"
        if reusable
        else "do not reuse pilot until all gating checks pass",
        "field_comparison": field_comparison,
        "scope_checks": scope_checks,
        "feature_cache": feature_cache,
        "random_shuffle_seed_handling": {
            "summary": "The same runner and species control implementation files are used for pilot and full. Per-run seed comes from the config seed value.",
            "relevant_files": [
                "scripts/run_v3_full_experiment.py",
                "jcim_v3/runner.py",
                "jcim_v3/species_controls.py",
                "jcim_v3/rdkit_lgbm.py",
            ],
        },
        "implementation_fingerprints": implementation,
        "pilot_completion": pilot_completion,
        "non_gating_differences": {
            "full_output_root": full.get("output_root"),
            "pilot_output_root": pilot.get("output_root"),
            "full_analysis_n_bootstrap": full.get("analysis", {}).get("n_bootstrap"),
            "pilot_analysis_n_bootstrap": pilot.get("analysis", {}).get("n_bootstrap"),
            "note": "Output roots and analysis bootstrap size differ by design and are not training reuse gates.",
        },
        "gating_checks": gating_checks,
        "training_was_executed": False,
    }
    json_path = out_dir / "pilot_reuse_check.json"
    md_path = out_dir / "pilot_reuse_check.md"
    _write_json(json_path, payload)
    _write_markdown(md_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
