"""Cold-species sign test (exploratory, post-hoc): T2 (categorical) vs T4 (embedding).

Cold species have ~1 row each, so a magnitude test is impossible. But each cold
species yields ONE sign: which model predicts it better. If the sign is lopsided,
a two-sided sign test (exact binomial vs p=0.5) gives significance on DIRECTION only.

This is EXPLORATORY and post-hoc. It is NOT a pre-registered confirmatory test; it
only supports the confirmatory warm/rare results. Per species, errors are averaged
over seeds first (to denoise the GNN), then the per-species sign is taken.

Env: conda run -n src.
"""
from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prediction_io import load_prediction_csv  # pred CSV = keys/pred/true only; strata from dataset

GNN = Path(r".\results\q2_v4\runs\gnn\predictions")
LGB = Path(r".\results\q2_v4\runs\replication\predictions")
DATA = Path(r".\results\q2_v4\data")


def load_gnn(bb, var, split, seed, ep=100):
    f = GNN / f"{bb}_{var}_{split}_s{seed}_e{ep}_nfull.csv"
    return None if not f.exists() else load_prediction_csv(f, columns=["species", "pred_log10", "true_log10"])


def load_lgb(stem, split, seed):
    f = LGB / f"{stem}_{split}_s{seed}.csv"
    return None if not f.exists() else load_prediction_csv(f, columns=["species", "pred_log10", "true_log10"])


def sign_test_two_sided(k, n):
    """Exact two-sided binomial p vs 0.5 (k = successes out of n)."""
    if n == 0:
        return float("nan")
    kk = min(k, n - k)
    tail = sum(comb(n, i) for i in range(kk + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def per_species_abs_err(frames):
    """Mean |error| per species, averaged over seeds (rows aligned per file)."""
    acc = None
    for df in frames:
        e = df.copy()
        e["ae"] = (e["pred_log10"] - e["true_log10"]).abs()
        s = e.groupby("species")["ae"].mean()
        acc = s if acc is None else acc.add(s, fill_value=0)
    return acc / len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="replication_group")
    ap.add_argument("--backbone", default="dmpnn")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    a = ap.parse_args()

    tr = pd.read_csv(DATA / f"{a.split}_train.csv", usecols=["species"])
    train_species = set(tr["species"].unique())
    te = pd.read_csv(DATA / f"{a.split}_test.csv", usecols=["species"])
    cold = set(te["species"].unique()) - train_species

    t2 = [load_lgb("LightGBM_RDKit_species_categorical", a.split, s) for s in a.seeds]
    t4 = [load_gnn(a.backbone, "true_species_late_fusion", a.split, s) for s in a.seeds]
    t1p = [load_lgb("LightGBM_RDKit_species_residual_calibration", a.split, s) for s in a.seeds]
    if any(x is None for x in t2 + t4 + t1p):
        print("missing prediction files (graphconv may still be running).")
        return

    ae_t2 = per_species_abs_err(t2)
    ae_t4 = per_species_abs_err(t4)
    ae_t1p = per_species_abs_err(t1p)

    def report(name, a_err, b_err, label_a, label_b):
        idx = [s for s in cold if s in a_err.index and s in b_err.index]
        d = (a_err.loc[idx] - b_err.loc[idx])  # <0 => a better
        d = d[d.abs() > 1e-12]  # drop exact ties
        n = len(d)
        k = int((d < 0).sum())  # a better
        p = sign_test_two_sided(k, n)
        print(f"  {name}")
        print(f"    cold species compared: {n}  (ties dropped)")
        print(f"    {label_a} better: {k}/{n} ({k/n*100:.0f}%)   {label_b} better: {n-k}/{n}")
        print(f"    two-sided sign test p = {p:.2e}")

    print(f"===== cold sign test (EXPLORATORY, post-hoc): {a.backbone} / {a.split} =====")
    print(f"cold species (test-only): {len(cold)}\n")
    report("T2 categorical  vs  T4 embedding", ae_t2, ae_t4, "T2", "T4")
    print()
    report("T2 categorical  vs  T1' residual_calib", ae_t2, ae_t1p, "T2", "T1'")


if __name__ == "__main__":
    main()
