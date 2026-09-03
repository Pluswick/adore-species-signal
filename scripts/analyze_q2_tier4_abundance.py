"""Is the Tier 4 (late_fusion / embedding) advantage support-dependent?

Decomposes delta-of-deltas by abundance bin (SPEC 5: {0 cold, 1, 2-9, 10-49, 50+}).

  dd(T4 vs T1') = [T4_gnn - T0_gnn] - [T1'_lgbm - T0oof_lgbm]
  dd(T4) - dd(T2) = [T4_gnn - T0_gnn] - [T2_lgbm - T0_lgbm]

Per-bin RMSE is computed inside each prediction file (same test set across models),
so no cross-file row matching is needed.

Efficiency question: if the Tier 4 edge concentrates in warm species and vanishes in
rare/cold, the embedding advantage is realised only where a species is data-rich.

Env: conda run -n src.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prediction_io import load_prediction_csv  # pred CSV = keys/pred/true only; strata from dataset

GNN = Path(r".\results\q2_v4\runs\gnn\predictions")
LGB = Path(r".\results\q2_v4\runs\replication\predictions")
DATA = Path(r".\results\q2_v4\data")

ORDER = ["0 cold", "1", "2-9", "10-49", "50+"]


def binlab(n):
    if n == 0:
        return "0 cold"
    if n == 1:
        return "1"
    if n <= 9:
        return "2-9"
    if n <= 49:
        return "10-49"
    return "50+"


def load_gnn(bb, var, split, seed, ep=100):
    f = GNN / f"{bb}_{var}_{split}_s{seed}_e{ep}_nfull.csv"
    return None if not f.exists() else load_prediction_csv(f, columns=["species", "pred_log10", "true_log10"])


def load_lgb(stem, split, seed):
    f = LGB / f"{stem}_{split}_s{seed}.csv"
    return None if not f.exists() else load_prediction_csv(f, columns=["species", "pred_log10", "true_log10"])


def per_bin_rmse(df, binmap):
    d = df.copy()
    d["bin"] = d["species"].map(binmap).fillna("0 cold")
    g = d.groupby("bin").apply(
        lambda x: np.sqrt(np.mean((x["pred_log10"] - x["true_log10"]) ** 2)), include_groups=False)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="replication_group")
    ap.add_argument("--backbone", default="dmpnn")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    a = ap.parse_args()

    tr = pd.read_csv(DATA / f"{a.split}_train.csv", usecols=["species"])
    counts = tr.groupby("species").size()
    binmap = counts.map(binlab)

    te = pd.read_csv(DATA / f"{a.split}_test.csv", usecols=["species"])
    te["bin"] = te["species"].map(binmap).fillna("0 cold")
    nrows = te["bin"].value_counts().reindex(ORDER).fillna(0).astype(int)

    recs = []
    for s in a.seeds:
        g0 = load_gnn(a.backbone, "no_species", a.split, s)
        g1 = load_gnn(a.backbone, "species_bias_only", a.split, s)
        g4 = load_gnn(a.backbone, "true_species_late_fusion", a.split, s)
        l0 = load_lgb("LightGBM_RDKit_no_species", a.split, s)
        loof = load_lgb("LightGBM_RDKit_no_species_oof_base", a.split, s)
        l1p = load_lgb("LightGBM_RDKit_species_residual_calibration", a.split, s)
        l2 = load_lgb("LightGBM_RDKit_species_categorical", a.split, s)
        if any(x is None for x in (g0, g1, g4, l0, loof, l1p, l2)):
            continue
        r = {k: per_bin_rmse(v, binmap) for k, v in
             {"g0": g0, "g1": g1, "g4": g4, "l0": l0, "loof": loof, "l1p": l1p, "l2": l2}.items()}
        for b in ORDER:
            if b not in r["g0"].index:
                continue
            d_t4 = r["g4"][b] - r["g0"][b]
            d_t1g = r["g1"][b] - r["g0"][b]
            d_t1p = r["l1p"][b] - r["loof"][b]
            d_t2 = r["l2"][b] - r["l0"][b]
            recs.append({
                "bin": b, "seed": s,
                "d_T1gnn": d_t1g, "d_T1p": d_t1p, "d_T2": d_t2, "d_T4": d_t4,
                "dd_T4_vs_T1p": d_t4 - d_t1p,
                "dd_T4_minus_dd_T2": d_t4 - d_t2,
                "dd_T4_vs_T1gnn": d_t4 - d_t1g,
            })
    df = pd.DataFrame(recs)
    if df.empty:
        print("no runs found.")
        return

    print(f"===== Tier4 x abundance : {a.backbone} / {a.split} =====")
    print("(negative delta = tier improves over its own no-species baseline)\n")
    agg = df.groupby("bin").agg(
        d_T1p=("d_T1p", "mean"), d_T2=("d_T2", "mean"),
        d_T1gnn=("d_T1gnn", "mean"), d_T4=("d_T4", "mean"),
        dd_T4_vs_T1p=("dd_T4_vs_T1p", "mean"),
        dd_T4_minus_dd_T2=("dd_T4_minus_dd_T2", "mean"),
    ).reindex(ORDER)
    agg.insert(0, "n_test", nrows)
    fav_t1p = df[df.dd_T4_vs_T1p < 0].groupby("bin").size().reindex(ORDER).fillna(0).astype(int)
    fav_t2 = df[df.dd_T4_minus_dd_T2 < 0].groupby("bin").size().reindex(ORDER).fillna(0).astype(int)
    nseed = df.groupby("bin").size().reindex(ORDER).fillna(0).astype(int)
    agg["fav_vs_T1'"] = [f"{a}/{b}" for a, b in zip(fav_t1p, nseed)]
    agg["fav_vs_T2"] = [f"{a}/{b}" for a, b in zip(fav_t2, nseed)]
    print(agg.to_string(float_format=lambda x: f"{x:8.4f}"))

    print("\n-- gradient check: does the Tier4 edge grow with species support? --")
    for col, lab in [("dd_T4_vs_T1p", "dd(T4 vs T1')"), ("dd_T4_minus_dd_T2", "dd(T4)-dd(T2)")]:
        vals = agg[col].to_numpy(float)
        ok = np.all(np.diff(vals[~np.isnan(vals)]) <= 0)
        print(f"  {lab:18s} cold->warm: {[f'{v:+.4f}' for v in vals]}  "
              f"{'monotone toward warm' if ok else 'not monotone'}")


if __name__ == "__main__":
    main()
