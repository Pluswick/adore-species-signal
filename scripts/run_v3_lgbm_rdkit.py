from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jcim_v3.rdkit_lgbm import RDKitLGBMConfig, run_rdkit_lgbm


BASELINES = [
    "LightGBM_RDKit_no_species",
    "LightGBM_RDKit_species_categorical",
    "LightGBM_RDKit_zero_species_categorical",
    "LightGBM_RDKit_shuffled_species_categorical",
    "LightGBM_RDKit_dummy_species_categorical",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--baseline",
        choices=[*BASELINES, "all"],
        default="all",
    )
    p.add_argument("--split", default="scaffold")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-train", type=int)
    p.add_argument("--limit-test", type=int)
    p.add_argument("--out-root")
    args = p.parse_args()

    baselines = BASELINES if args.baseline == "all" else [args.baseline]
    base = RDKitLGBMConfig(
        baseline="LightGBM_RDKit_no_species",
        split=args.split,
        seed=args.seed,
    )
    updates = {}
    if args.limit_train is not None:
        updates["limit_train"] = args.limit_train
    if args.limit_test is not None:
        updates["limit_test"] = args.limit_test
    if args.out_root is not None:
        updates["out_root"] = args.out_root
    base = replace(base, **updates)

    for baseline in baselines:
        result = run_rdkit_lgbm(replace(base, baseline=baseline))
        rows = [{
            "run_id": result["config"]["run_id"],
            "baseline": baseline,
            "split": result["config"]["split"],
            "seed": result["config"]["seed"],
            "rmse": result["A"]["rmse"],
            "mae": result["A"]["mae"],
            "within_2fold": result["A"]["within_2fold"],
            "within_3fold": result["A"]["within_3fold"],
            "n_features": result["D"]["n_features"],
            "n_trees": result["D"]["n_trees"],
            "prediction_file": result["config"]["prediction_file"],
        }]
        out_root = Path(result["config"]["out_root"])
        out_root.mkdir(parents=True, exist_ok=True)
        summary = out_root / "rdkit_lgbm_smoke_summary.csv"
        exists = summary.exists()
        with open(summary, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
        print("done", result["config"]["run_id"], result["A"])


if __name__ == "__main__":
    main()
