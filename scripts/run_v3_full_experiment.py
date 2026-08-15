from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from jcim_v3.rdkit_lgbm import RDKitLGBMConfig, run_rdkit_lgbm
from jcim_v3.runner import V3RunConfig, run_v3_smoke


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v3_full_experiment.json"

LIGHTGBM_SPECIES_MODES = {
    "LightGBM_RDKit_no_species": "no_species",
    "LightGBM_RDKit_species_categorical": "species_categorical",
    "LightGBM_RDKit_zero_species_categorical": "zero_species_categorical",
    "LightGBM_RDKit_shuffled_species_categorical": "shuffled_species_categorical",
    "LightGBM_RDKit_dummy_species_categorical": "dummy_species_categorical",
}


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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


def _output_root(config: dict) -> Path:
    return Path(config.get("output_root") or config.get("out_root") or ROOT / "results" / "jcim_v3" / "full")


def _plan_runs(config: dict) -> list[dict]:
    splits = list(config.get("split_types") or config.get("splits") or [config.get("split", "scaffold")])
    seeds = [int(value) for value in config.get("seeds", [config.get("seed", 0)])]
    backbones = list(config.get("backbones") or config.get("backbone_list") or ["dmpnn", "graphconv"])
    species_modes = _dedupe(list(config.get("main_species_modes", [])) + list(config.get("control_species_modes", [])))
    baselines = list(config.get("lightgbm_baselines", []))
    rows = []
    for split in splits:
        for seed in seeds:
            for backbone in backbones:
                for species_mode in species_modes:
                    rows.append(
                        {
                            "run_type": "gnn",
                            "split": split,
                            "seed": seed,
                            "backbone": backbone,
                            "species_mode": species_mode,
                            "baseline": "",
                        }
                    )
            for baseline in baselines:
                rows.append(
                    {
                        "run_type": "lightgbm_rdkit",
                        "split": split,
                        "seed": seed,
                        "backbone": "lightgbm_rdkit",
                        "species_mode": LIGHTGBM_SPECIES_MODES.get(baseline, "species_categorical"),
                        "baseline": baseline,
                    }
                )
    return rows


def _filter_plan(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    out = rows
    if args.split:
        out = [row for row in out if row["split"] == args.split]
    if args.seed is not None:
        out = [row for row in out if int(row["seed"]) == int(args.seed)]
    if args.backbone:
        out = [row for row in out if row["backbone"] == args.backbone]
    if args.species_mode:
        out = [row for row in out if row["species_mode"] == args.species_mode]
    if args.limit_runs is not None:
        out = out[: int(args.limit_runs)]
    return out


def _run_id(row: dict, config: dict) -> str:
    if row["run_type"] == "lightgbm_rdkit":
        return f"{row['baseline']}_{row['split']}_s{row['seed']}"
    epochs = int(config.get("max_epochs", config.get("epochs", 100)))
    limit_train = config.get("train_subset_size", config.get("limit_train"))
    n_part = limit_train if limit_train is not None else "full"
    return f"{row['backbone']}_{row['species_mode']}_{row['split']}_s{row['seed']}_e{epochs}_n{n_part}"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rows_by_run_id(rows: list[dict]) -> dict[str, dict]:
    return {str(row["run_id"]): dict(row) for row in rows if row.get("run_id")}


def _gnn_config(row: dict, config: dict, out_root: Path) -> V3RunConfig:
    return V3RunConfig(
        backbone=row["backbone"],
        variant=row["species_mode"],
        split=row["split"],
        seed=int(row["seed"]),
        epochs=int(config.get("max_epochs", config.get("epochs", 100))),
        batch_size=int(config.get("batch_size", 256)),
        lr=float(config.get("lr", 5e-4)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
        hidden=int(config.get("hidden", 300)),
        depth=int(config.get("depth", 3)),
        dropout=float(config.get("dropout", 0.1)),
        species_emb_dim=int(config.get("species_emb_dim", 16)),
        val_frac=float(config.get("val_frac", 0.1)),
        limit_train=config.get("train_subset_size", config.get("limit_train")),
        limit_test=config.get("test_subset_size", config.get("limit_test")),
        data_dir=str(config.get("dataset_path")),
        out_root=str(out_root),
        save_species_artifacts=bool(config.get("save_embeddings", True)),
    )


def _lgbm_config(row: dict, config: dict, out_root: Path) -> RDKitLGBMConfig:
    return RDKitLGBMConfig(
        baseline=row["baseline"],
        split=row["split"],
        seed=int(row["seed"]),
        limit_train=config.get("train_subset_size", config.get("limit_train")),
        limit_test=config.get("test_subset_size", config.get("limit_test")),
        data_dir=str(config.get("dataset_path")),
        out_root=str(out_root),
    )


def _completed(run_id: str, out_root: Path) -> bool:
    return (out_root / "runs" / f"{run_id}.json").exists() and (
        out_root / "predictions" / f"{run_id}.csv"
    ).exists()


def _persist_result_artifacts(run_id: str, row: dict, result: dict, out_root: Path) -> dict:
    metrics_payload = {
        "run_id": run_id,
        "run_type": row["run_type"],
        "config": result.get("config", {}),
        "metrics": result.get("A", {}),
        "species_bin_metrics": result.get("B", {}),
        "training_or_model_summary": result.get("D", {}),
    }
    metrics_path = out_root / "metrics" / f"{run_id}.json"
    _write_json(metrics_path, metrics_payload)
    param_payload = {
        "run_id": run_id,
        "run_type": row["run_type"],
        "split": row["split"],
        "seed": row["seed"],
        "backbone": row["backbone"],
        "species_mode": row["species_mode"],
        "baseline": row["baseline"],
        "trainable_params": result.get("D", {}).get("trainable_params", 0),
        "species_trainable_params": result.get("D", {}).get("species_trainable_params", 0),
        "n_features": result.get("D", {}).get("n_features"),
        "n_trees": result.get("D", {}).get("n_trees"),
    }
    return {
        "metrics_file": str(metrics_path),
        **param_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--backbone")
    parser.add_argument("--species-mode")
    parser.add_argument("--limit-runs", type=int)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_config(config_path)
    out_root = _output_root(config)
    for subdir in [
        "predictions",
        "metrics",
        "runs",
        "parameter_counts",
        "embeddings",
        "bootstrap",
        "embedding_analysis",
        "figures",
        "summary_tables",
    ]:
        (out_root / subdir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, out_root / "full_experiment_config_snapshot.json")

    full_plan = _plan_runs(config)
    plan = _filter_plan(full_plan, args)
    for row in plan:
        row["run_id"] = _run_id(row, config)
        row["planned_at_utc"] = datetime.now(timezone.utc).isoformat()
    plan_meta = {
        "config": str(config_path),
        "output_root": str(out_root),
        "n_config_runs": len(full_plan),
        "n_planned_runs": len(plan),
        "filters": {
            "split": args.split,
            "seed": args.seed,
            "backbone": args.backbone,
            "species_mode": args.species_mode,
            "limit_runs": args.limit_runs,
        },
    }
    _write_csv(out_root / "run_manifest.csv", plan)
    _write_json(out_root / "run_manifest_meta.json", plan_meta)

    if args.dry_run:
        print(json.dumps({"dry_run": True, **plan_meta, "run_manifest": str(out_root / "run_manifest.csv")}, indent=2))
        return

    completed_path = out_root / "completed_runs.csv"
    failed_path = out_root / "failed_runs.csv"
    parameter_path = out_root / "parameter_counts" / "parameter_counts.csv"
    completed_by_id = _rows_by_run_id(_read_csv(completed_path))
    failed_by_id = _rows_by_run_id(_read_csv(failed_path))
    parameter_by_id = _rows_by_run_id(_read_csv(parameter_path))
    current_completed_rows = []
    current_failed_rows = []
    for row in plan:
        run_id = row["run_id"]
        if args.resume and _completed(run_id, out_root):
            completed_row = {**row, "status": "skipped_existing"}
            completed_by_id[run_id] = completed_row
            failed_by_id.pop(run_id, None)
            current_completed_rows.append(completed_row)
            continue
        try:
            if row["run_type"] == "gnn":
                result = run_v3_smoke(_gnn_config(row, config, out_root))
            else:
                result = run_rdkit_lgbm(_lgbm_config(row, config, out_root))
            parameter_by_id[run_id] = _persist_result_artifacts(run_id, row, result, out_root)
            completed_row = {**row, "status": "completed"}
            completed_by_id[run_id] = completed_row
            failed_by_id.pop(run_id, None)
            current_completed_rows.append(completed_row)
            _write_csv(completed_path, list(completed_by_id.values()))
            _write_csv(parameter_path, list(parameter_by_id.values()))
            _write_csv(failed_path, list(failed_by_id.values()))
        except Exception as exc:
            failed_row = {
                **row,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            failed_by_id[run_id] = failed_row
            completed_by_id.pop(run_id, None)
            parameter_by_id.pop(run_id, None)
            current_failed_rows.append(failed_row)
            _write_csv(completed_path, list(completed_by_id.values()))
            _write_csv(failed_path, list(failed_by_id.values()))
            _write_csv(parameter_path, list(parameter_by_id.values()))
            continue

    _write_csv(completed_path, list(completed_by_id.values()))
    _write_csv(failed_path, list(failed_by_id.values()))
    _write_csv(parameter_path, list(parameter_by_id.values()))
    print(
        json.dumps(
            {
                "out_root": str(out_root),
                "planned_runs": len(plan),
                "completed_or_skipped": len(current_completed_rows),
                "failed": len(current_failed_rows),
                "cumulative_completed_or_skipped": len(completed_by_id),
                "cumulative_failed": len(failed_by_id),
            },
            indent=2,
        )
    )
    if current_failed_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
