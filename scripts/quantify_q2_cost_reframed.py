"""Q2 v4 — cost reframed: experimental acquisition cost, not GPU cost.

A. species-level coverage of the >=3 measurements/species entry condition
B. new-species (cold) view
C. method freedom: measured lgbm(one-hot) vs GNN(embedding) training cost
D. failure mode: embedding collapse in cold vs one-hot

Env: conda run -n src.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(r".\results\q2_v4\data")
GNN = Path(r".\results\q2_v4\runs\gnn")
LGB = Path(r".\results\q2_v4\runs\replication")
OUT = Path(r".\results\q2_v4\cost")
SPLITS = ["replication_group", "discovery_group"]
BREAKEVEN = 3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, dist_rows, cold_rows = [], [], []

    for split in SPLITS:
        tr = pd.read_csv(DATA / f"{split}_train.csv", usecols=["species"])
        te = pd.read_csv(DATA / f"{split}_test.csv", usecols=["species"])
        cnt = tr.groupby("species").size()

        # ---- A. species-level distribution over ALL species in the dataset ----
        all_sp = sorted(set(tr["species"]) | set(te["species"]))
        support = pd.Series({s: int(cnt.get(s, 0)) for s in all_sp})
        n_all = len(support)
        for k in [0, 1, 2, 3, 5, 10, 20, 50, 100]:
            below = int((support < k).sum())
            rows.append({"split": split, "scope": "all_species", "threshold": k,
                         "n_species_below": below, "pct_species_below": round(below / n_all * 100, 1),
                         "n_species_at_or_above": n_all - below,
                         "pct_species_at_or_above": round((n_all - below) / n_all * 100, 1),
                         "n_species_total": n_all})
        # exact counts at 0,1,2
        for k in (0, 1, 2, 3):
            dist_rows.append({"split": split, "scope": "all_species",
                              "measurements_per_species": k if k < 3 else "3+",
                              "n_species": int((support == k).sum()) if k < 3 else int((support >= 3).sum()),
                              "pct": round((int((support == k).sum()) if k < 3 else int((support >= 3).sum())) / n_all * 100, 1)})

        # same, restricted to TEST species (the ones we must predict)
        te_sp = sorted(set(te["species"]))
        sup_te = pd.Series({s: int(cnt.get(s, 0)) for s in te_sp})
        n_te = len(sup_te)
        below3 = int((sup_te < BREAKEVEN).sum())
        rows.append({"split": split, "scope": "test_species", "threshold": BREAKEVEN,
                     "n_species_below": below3, "pct_species_below": round(below3 / n_te * 100, 1),
                     "n_species_at_or_above": n_te - below3,
                     "pct_species_at_or_above": round((n_te - below3) / n_te * 100, 1),
                     "n_species_total": n_te})

        # ---- B. cold / new species ----
        cold = [s for s in te_sp if cnt.get(s, 0) == 0]
        te_rows = te["species"]
        cold_rows.append({
            "split": split, "n_test_species": n_te, "n_cold_species": len(cold),
            "pct_cold_species": round(len(cold) / n_te * 100, 1),
            "n_cold_test_rows": int(te_rows.isin(cold).sum()),
            "pct_cold_test_rows": round(int(te_rows.isin(cold).sum()) / len(te_rows) * 100, 1),
            "median_test_rows_per_cold_species": float(
                te_rows[te_rows.isin(cold)].value_counts().median()) if cold else np.nan,
            "test_species_below_breakeven_pct": round(below3 / n_te * 100, 1),
        })

    pd.DataFrame(rows).to_csv(OUT / "species_coverage_reframed.csv", index=False, encoding="utf-8")
    pd.DataFrame(dist_rows).to_csv(OUT / "species_support_distribution.csv", index=False, encoding="utf-8")
    pd.DataFrame(cold_rows).to_csv(OUT / "cold_species_view.csv", index=False, encoding="utf-8")

    print("=== A. species-level: how many species fail the >=3 entry condition ===")
    print(pd.DataFrame(dist_rows).to_string(index=False))
    print()
    print(pd.DataFrame(rows).query("threshold in [1,3,10,50]").to_string(index=False))
    print("\n=== B. cold / new species ===")
    print(pd.DataFrame(cold_rows).to_string(index=False))

    # ---- C. measured lgbm one-hot cost (fills the NaN) ----
    print("\n=== C. measured training cost: one-hot on lgbm vs embedding on GNN ===")
    import sys
    sys.path.insert(0, r".")
    from src.rdkit_lgbm import RDKitLGBMConfig, run_rdkit_lgbm
    timings = []
    for baseline in ("LightGBM_RDKit_no_species", "LightGBM_RDKit_species_categorical"):
        t0 = time.time()
        run_rdkit_lgbm(RDKitLGBMConfig(
            baseline=baseline, split="replication_group", seed=0,
            data_dir=str(DATA), out_root=str(OUT / "_timing_tmp")))
        timings.append({"tier": baseline, "wall_sec": round(time.time() - t0, 2), "device": "cpu"})
    # GNN reference from run JSONs
    for bb in ("dmpnn", "graphconv"):
        secs = []
        for s in range(10):
            f = GNN / "runs" / f"{bb}_true_species_late_fusion_replication_group_s{s}_e100_nfull.json"
            if f.exists():
                d = json.loads(f.read_text(encoding="utf-8")).get("D", {})
                if d.get("train_sec"):
                    secs.append(float(d["train_sec"]))
        if secs:
            timings.append({"tier": f"{bb} late_fusion (embedding)", "wall_sec": round(np.mean(secs), 1),
                            "device": "cuda"})
    tdf = pd.DataFrame(timings)
    lg = tdf[tdf.tier.str.contains("categorical")]["wall_sec"].iloc[0]
    for _, r in tdf[tdf.tier.str.contains("late_fusion")].iterrows():
        tdf.loc[tdf.tier == r.tier, "ratio_vs_lgbm_onehot"] = round(r.wall_sec / lg, 1)
    tdf.to_csv(OUT / "measured_training_cost.csv", index=False, encoding="utf-8")
    print(tdf.to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
