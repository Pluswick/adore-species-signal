from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.naive_species_baselines import NaiveSpeciesBaselineConfig, run_naive_species_baselines


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v3_compound_random_core_full.json"


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _output_root(config: dict) -> Path:
    return Path(config.get("output_root") or config.get("out_root") or ROOT / "results" / "src" / "full")


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _upsert(path: Path, rows: list[dict], key: str = "run_id") -> None:
    existing = {str(row.get(key)): row for row in _read_csv(path) if row.get(key)}
    for row in rows:
        existing[str(row[key])] = row
    _write_csv(path, list(existing.values()))


def _run_ids(split: str, seed: int) -> list[str]:
    return [
        f"Naive_global_mean_{split}_s{seed}",
        f"Naive_species_mean_{split}_s{seed}",
        f"LightGBM_RDKit_no_species_oof_base_{split}_s{seed}",
        f"LightGBM_RDKit_species_residual_calibration_{split}_s{seed}",
    ]


def _completed(out_root: Path, split: str, seed: int) -> bool:
    return all(
        (out_root / "predictions" / f"{run_id}.csv").exists()
        and (out_root / "runs" / f"{run_id}.json").exists()
        for run_id in _run_ids(split, seed)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-test", type=int)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_config(config_path)
    out_root = _output_root(config)
    splits = list(config.get("split_types") or [config.get("split", "compound_random")])
    seeds = [int(value) for value in config.get("seeds", [config.get("seed", 0)])]
    if args.split:
        splits = [split for split in splits if split == args.split]
    if args.seed is not None:
        seeds = [seed for seed in seeds if seed == int(args.seed)]

    plan = []
    for split in splits:
        for seed in seeds:
            for run_id in _run_ids(split, seed):
                plan.append({"split": split, "seed": seed, "run_id": run_id})
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "config": str(config_path),
                    "out_root": str(out_root),
                    "planned_runs": len(plan),
                    "planned_groups": len(splits) * len(seeds),
                    "run_ids": [row["run_id"] for row in plan],
                },
                indent=2,
            )
        )
        return

    all_outputs = []
    skipped = []
    for split in splits:
        for seed in seeds:
            if args.resume and _completed(out_root, split, seed):
                skipped.extend(_run_ids(split, seed))
                continue
            cfg = NaiveSpeciesBaselineConfig(
                split=split,
                seed=seed,
                data_dir=str(config.get("dataset_path")),
                out_root=str(out_root),
                limit_train=args.limit_train if args.limit_train is not None else config.get("train_subset_size"),
                limit_test=args.limit_test if args.limit_test is not None else config.get("test_subset_size"),
            )
            all_outputs.extend(run_naive_species_baselines(cfg))

    manifest_rows = [
        {
            "run_id": row["run_id"],
            "run_type": row["run_type"],
            "split": row["split"],
            "seed": row["seed"],
            "prediction_file": row["prediction_file"],
            "metrics_file": row["metrics_file"],
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        for row in all_outputs
    ]
    summary_path = out_root / "summary_tables" / "naive_species_baseline_summary.csv"
    _upsert(summary_path, all_outputs)

    param_rows = []
    for row in all_outputs:
        param_rows.append(
            {
                "run_id": row["run_id"],
                "run_type": row["run_type"],
                "split": row["split"],
                "seed": row["seed"],
                "backbone": row["backbone"],
                "species_mode": row["species_mode"],
                "baseline": row["model_name"],
                "trainable_params": 0,
                "species_trainable_params": 0,
                "n_features": row.get("n_features"),
                "n_trees": row.get("n_trees"),
            }
        )
    _upsert(out_root / "parameter_counts" / "parameter_counts.csv", param_rows)
    _upsert(out_root / "naive_baseline_completed_runs.csv", manifest_rows)

    print(
        json.dumps(
            {
                "out_root": str(out_root),
                "planned_runs": len(plan),
                "completed": len(all_outputs),
                "skipped_existing": len(skipped),
                "summary": str(summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
