"""Q2 v4 Task D driver — runnable ladder tiers on the new q2 datasets.

Runs the tiers that already exist in the pipeline, pointed at
results/q2_v4/data via data_dir override:
  Tier 0  no_species        : naive oof_base + lightgbm no_species
  Tier 1' residual_calib     : naive species_residual_calibration (REFERENCE)
  Tier 2  species_categorical: lightgbm + zero/shuffled/dummy controls
  (+ naive global_mean / species_mean)

NOT run here (need new code / heavy compute; see configs/q2_replication_ladder.json):
  Tier 1 (gnn bias), Tier 3a/3b (taxonomy features), Tier 4/5 (gnn fusion/film).

Env: src.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.naive_species_baselines import NaiveSpeciesBaselineConfig, run_naive_species_baselines
from src.rdkit_lgbm import RDKitLGBMConfig, run_rdkit_lgbm

LGBM = ["LightGBM_RDKit_no_species", "LightGBM_RDKit_species_categorical",
        "LightGBM_RDKit_zero_species_categorical", "LightGBM_RDKit_shuffled_species_categorical",
        "LightGBM_RDKit_dummy_species_categorical"]


def _rmse_from_A(A) -> float | None:
    if isinstance(A, dict):
        for k in ("rmse", "RMSE", "rmse_log10", "root_mean_squared_error"):
            if k in A and isinstance(A[k], (int, float)):
                return float(A[k])
        for v in A.values():
            if isinstance(v, dict) and "rmse" in v:
                return float(v["rmse"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "q2_replication_ladder.json"))
    ap.add_argument("--splits", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    data_dir, out_root = cfg["dataset_path"], cfg["output_root"]
    splits = args.splits or cfg["split_types"]
    seeds = args.seeds or cfg["seeds"]

    for split in splits:
        for seed in seeds:
            run_naive_species_baselines(NaiveSpeciesBaselineConfig(
                split=split, seed=seed, data_dir=data_dir, out_root=out_root))
            for b in LGBM:
                run_rdkit_lgbm(RDKitLGBMConfig(
                    baseline=b, split=split, seed=seed, data_dir=data_dir, out_root=out_root))
            print(f"[done] {split} seed={seed}", flush=True)

    # aggregate rmse per (run stem) over seeds from runs/*.json
    import pandas as pd
    rows = []
    for jf in glob.glob(str(Path(out_root) / "runs" / "*.json")):
        d = json.loads(Path(jf).read_text(encoding="utf-8"))
        conf = d.get("config", {})
        rid = conf.get("run_id", Path(jf).stem)
        split = conf.get("split")
        seed = conf.get("seed")
        # strip split/seed suffix -> model stem
        stem = rid
        if split and f"_{split}_" in rid:
            stem = rid.split(f"_{split}_")[0]
        rows.append({"stem": stem, "split": split, "seed": seed,
                     "rmse": _rmse_from_A(d.get("A"))})
    df = pd.DataFrame(rows)
    if not df.empty:
        summ = (df.dropna(subset=["rmse"]).groupby(["split", "stem"])
                .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), n=("rmse", "size"))
                .reset_index().sort_values(["split", "rmse_mean"]))
        outp = Path(out_root) / "runnable_ladder_summary.csv"
        summ.to_csv(outp, index=False, encoding="utf-8")
        print("\n=== runnable ladder summary (rmse, mean over seeds) ===", flush=True)
        print(summ.to_string(index=False), flush=True)
        print(f"\nwrote {outp}", flush=True)


if __name__ == "__main__":
    main()
