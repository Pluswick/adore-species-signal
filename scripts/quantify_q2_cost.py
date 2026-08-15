"""Q2 v4 — cost quantification: when does embedding beat one-hot?

(1) Break-even support: dd(T4)-dd(T2) on FINE support bins, with block-bootstrap CI,
    for both backbones and both datasets. Find where the sign flips.
(2) Coverage: how many species / measurements sit above the break-even.
(3) Model complexity: trainable params, species params, growth vs n_species.
(4) Compute cost: train_sec per tier (from run JSONs), device.

Bootstrap here resamples COMPOUND blocks within each support bin (GNN seeds averaged;
per-bin n is small so the CI is the honest limiter -- reported explicitly).

Env: conda run -n jcim_v3.
"""
from __future__ import annotations

import json
from pathlib import Path

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jcim_v3.prediction_io import load_prediction_csv  # pred CSV = keys/pred/true only; strata from dataset

GNN = Path(r".\results\q2_v4\runs\gnn")
LGB = Path(r".\results\q2_v4\runs\replication")
DATA = Path(r".\results\q2_v4\data")
OUT = Path(r".\results\q2_v4\cost")
KEY = ["smiles", "species", "endpoint", "duration"]
SEEDS = list(range(10))
N_BOOT = 2000
RNG = np.random.default_rng(20260724)

# fine-grained support bins (train measurements per species)
FINE = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 10, "6-10"),
        (11, 20, "11-20"), (21, 50, "21-50"), (51, 100, "51-100"), (101, 10**9, "100+")]


def binlab(n):
    for lo, hi, lab in FINE:
        if lo <= n <= hi:
            return lab
    return "100+"


def gnn_pred(bb, var, split):
    frames = []
    for s in SEEDS:
        f = GNN / "predictions" / f"{bb}_{var}_{split}_s{s}_e100_nfull.csv"
        if not f.exists():
            return None
        frames.append(load_prediction_csv(f, columns=KEY + ["pred_log10", "true_log10", "compound_key"]).set_index(KEY))
    base = frames[0]
    p = np.mean([fr.loc[base.index, "pred_log10"].to_numpy(float) for fr in frames], axis=0)
    out = base[["true_log10", "compound_key"]].copy()
    out["pred"] = p
    return out.reset_index()


def lgb_pred(stem, split):
    f = LGB / "predictions" / f"{stem}_{split}_s0.csv"
    if not f.exists():
        return None
    d = load_prediction_csv(f, columns=KEY + ["pred_log10", "true_log10", "compound_key"])
    return d.rename(columns={"pred_log10": "pred"})


def rmse(t, p, idx):
    d = p[idx] - t[idx]
    return float(np.sqrt(np.mean(d * d)))


def dd_ci(m):
    """m has true, compound_key, c(cand T4), cb(T0 gnn), r(T2), rb(T0 lgbm)."""
    t = m["true_log10"].to_numpy(float)
    c, cb, r, rb = (m[x].to_numpy(float) for x in ("c", "cb", "r", "rb"))
    full = np.arange(len(m))
    point = (rmse(t, c, full) - rmse(t, cb, full)) - (rmse(t, r, full) - rmse(t, rb, full))
    blocks = [np.asarray(v, np.int64) for v in
              m.reset_index(drop=True).groupby("compound_key").indices.values()]
    nb = len(blocks)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = np.concatenate([blocks[j] for j in RNG.integers(0, nb, nb)])
        boot[i] = (rmse(t, c, idx) - rmse(t, cb, idx)) - (rmse(t, r, idx) - rmse(t, rb, idx))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi), nb, float((boot < 0).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    coverage = []

    for split in ("replication_group", "discovery_group"):
        tr = pd.read_csv(DATA / f"{split}_train.csv", usecols=["species"])
        cnt = tr.groupby("species").size()
        te = pd.read_csv(DATA / f"{split}_test.csv", usecols=["species"])
        te_sp = te["species"]
        supp = te_sp.map(cnt).fillna(0).astype(int)

        t2, t0l = lgb_pred("LightGBM_RDKit_species_categorical", split), lgb_pred("LightGBM_RDKit_no_species", split)
        for bb in ("dmpnn", "graphconv"):
            g4, g0 = gnn_pred(bb, "true_species_late_fusion", split), gnn_pred(bb, "no_species", split)
            if any(x is None for x in (g4, g0, t2, t0l)):
                continue
            m = (g4.rename(columns={"pred": "c"})
                 .merge(g0[KEY + ["pred"]].rename(columns={"pred": "cb"}), on=KEY)
                 .merge(t2[KEY + ["pred"]].rename(columns={"pred": "r"}), on=KEY)
                 .merge(t0l[KEY + ["pred"]].rename(columns={"pred": "rb"}), on=KEY))
            m["support"] = m["species"].map(cnt).fillna(0).astype(int)
            m["bin"] = m["support"].map(binlab)
            for _, _, lab in FINE:
                sub = m[m["bin"] == lab]
                if len(sub) < 5:
                    rows.append({"split": split, "backbone": bb, "bin": lab, "n_rows": len(sub),
                                 "n_species": sub["species"].nunique(), "dd": np.nan,
                                 "ci_low": np.nan, "ci_high": np.nan, "n_blocks": 0, "p_favorable": np.nan})
                    continue
                pt, lo, hi, nb, pf = dd_ci(sub)
                rows.append({"split": split, "backbone": bb, "bin": lab, "n_rows": len(sub),
                             "n_species": sub["species"].nunique(), "dd": pt,
                             "ci_low": lo, "ci_high": hi, "n_blocks": nb, "p_favorable": pf})

        # coverage at candidate thresholds
        for thr in (1, 2, 3, 6, 11, 21, 51, 101):
            sp_ok = set(cnt[cnt >= thr].index) & set(te_sp.unique())
            n_rows_ok = int(te_sp.isin(sp_ok).sum())
            coverage.append({"split": split, "threshold_measurements_per_species": thr,
                             "n_species_at_or_above": len(sp_ok),
                             "pct_species": round(len(sp_ok) / te_sp.nunique() * 100, 1),
                             "n_test_rows": n_rows_ok,
                             "pct_test_rows": round(n_rows_ok / len(te_sp) * 100, 1)})

    df = pd.DataFrame(rows)
    cov = pd.DataFrame(coverage)
    df.to_csv(OUT / "breakeven_by_support.csv", index=False, encoding="utf-8")
    cov.to_csv(OUT / "coverage_by_threshold.csv", index=False, encoding="utf-8")

    # ---- model complexity + compute cost from run JSONs ----
    comp = []
    for bb in ("dmpnn", "graphconv"):
        for var in ("no_species", "species_bias_only", "true_species_late_fusion"):
            secs, tp, sp_, dev = [], None, None, None
            for s in SEEDS:
                f = GNN / "runs" / f"{bb}_{var}_replication_group_s{s}_e100_nfull.json"
                if not f.exists():
                    continue
                d = json.loads(f.read_text(encoding="utf-8")).get("D", {})
                tp, sp_, dev = d.get("trainable_params"), d.get("species_trainable_params"), d.get("device")
                if d.get("train_sec"):
                    secs.append(float(d["train_sec"]))
            if tp is not None:
                comp.append({"backbone": bb, "tier": var, "trainable_params": tp,
                             "species_params": sp_,
                             "species_param_share_pct": round(100 * (sp_ or 0) / tp, 2),
                             "train_sec_mean": round(np.mean(secs), 1) if secs else None,
                             "n_runs": len(secs), "device": dev})
    # lgbm/naive tiers
    for stem, tier in [("LightGBM_RDKit_no_species", "lgbm_no_species"),
                       ("LightGBM_RDKit_species_categorical", "lgbm_species_categorical"),
                       ("LightGBM_RDKit_species_residual_calibration", "naive_residual_calibration")]:
        f = LGB / "runs" / f"{stem}_replication_group_s0.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            cfgd = d.get("config", {})
            comp.append({"backbone": "lightgbm/naive", "tier": tier,
                         "trainable_params": cfgd.get("n_trees") or cfgd.get("final_num_boost_round"),
                         "species_params": None, "species_param_share_pct": None,
                         "train_sec_mean": cfgd.get("train_sec") or cfgd.get("train_sec_total"),
                         "n_runs": 1, "device": "cpu"})
    cdf = pd.DataFrame(comp)
    cdf.to_csv(OUT / "model_and_compute_cost.csv", index=False, encoding="utf-8")

    pd.set_option("display.width", 200)
    print("=== (1-1) dd(T4)-dd(T2) by fine support bin  [negative = embedding wins] ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print("\n=== (1-2) coverage at thresholds ===")
    print(cov.to_string(index=False))
    print("\n=== (2,3) model complexity + compute ===")
    print(cdf.to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
