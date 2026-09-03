from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runner import V3RunConfig, run_v3_smoke

SPECIES_MODES = [
    "no_species",
    "true_species_late_fusion",
    "zero_species_late_fusion",
    "shuffled_species_late_fusion",
    "dummy_species_late_fusion",
    "true_species_early_injection",
    "zero_species_early_injection",
    "shuffled_species_early_injection",
    "dummy_species_early_injection",
    "true_species_message_level",
    "zero_species_message_level",
    "shuffled_species_message_level",
    "dummy_species_message_level",
    "true_species_film",
    "zero_species_film",
    "shuffled_species_film",
    "dummy_species_film",
    "species_bias_only",
]


def _load_defaults(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/v3_smoke.json")
    p.add_argument("--backbone", choices=["dmpnn", "graphconv", "all"], default="all")
    p.add_argument(
        "--variant",
        choices=[*SPECIES_MODES, "all"],
        default="all",
    )
    p.add_argument("--split", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--out-root", default=None)
    args = p.parse_args()

    defaults = _load_defaults(args.config)
    base = V3RunConfig(**defaults)
    overrides = {}
    for key, value in {
        "split": args.split,
        "seed": args.seed,
        "epochs": args.epochs,
        "limit_train": args.limit_train,
        "limit_test": args.limit_test,
        "out_root": args.out_root,
    }.items():
        if value is not None:
            overrides[key] = value
    base = replace(base, **overrides)

    backbones = ["dmpnn", "graphconv"] if args.backbone == "all" else [args.backbone]
    variants = SPECIES_MODES if args.variant == "all" else [args.variant]

    rows = []
    for backbone in backbones:
        for variant in variants:
            cfg = replace(base, backbone=backbone, variant=variant)
            result = run_v3_smoke(cfg)
            rows.append(
                {
                    "run_id": result["config"]["run_id"],
                    "backbone": backbone,
                    "variant": variant,
                    "rmse": result["A"]["rmse"],
                    "mae": result["A"]["mae"],
                    "within_2fold": result["A"]["within_2fold"],
                    "within_3fold": result["A"]["within_3fold"],
                    "trainable_params": result["D"]["trainable_params"],
                    "species_trainable_params": result["D"]["species_trainable_params"],
                    "epochs": result["D"]["epochs"],
                    "best_val_rmse": result["D"]["best_val_rmse"],
                    "prediction_file": result["config"]["prediction_file"],
                }
            )
            print("done", rows[-1])

    out_root = Path(base.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = out_root / "smoke_summary.csv"
    with open(summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
