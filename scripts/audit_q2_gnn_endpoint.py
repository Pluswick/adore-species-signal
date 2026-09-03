"""Re-check 1: is the GNN Tier4/5 advantage riding on endpoint after all?

Three independent probes:
  (a) confirm the (A) stratum main effect is real and non-degenerate on this split
  (b) WITHIN-STRATUM ladder: RMSE per (endpoint,duration). If the ladder ordering holds
      inside a single endpoint x duration cell, endpoint MIXING cannot explain it.
  (c) per-species gain (T4-T0) vs that species' EC50 fraction -- the same diagnostic that
      exposed the Tier 1' offset confound (corr was -0.251 there).

Env: conda run -n src.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.stratum import fit_stratum_effect  # noqa: E402
from src.prediction_io import load_prediction_csv  # noqa: E402  (pred CSV = keys/pred/true only)

PRED = Path(r".\results\q2_v4\runs\gnn\predictions")
DATA = Path(r".\results\q2_v4\data")
TIERS = {
    "T0": "no_species",
    "T1": "species_bias_only",
    "T4": "true_species_late_fusion",
    "T5": "true_species_film",
}


def load(backbone, variant, split, seed, epochs=100):
    f = PRED / f"{backbone}_{variant}_{split}_s{seed}_e{epochs}_nfull.csv"
    if not f.exists():
        return None
    return load_prediction_csv(f, columns=["smiles", "species", "endpoint", "duration",
                                           "pred_log10", "true_log10"])


def rmse(d):
    return float(np.sqrt(np.mean((d["pred_log10"] - d["true_log10"]) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="replication_group")
    ap.add_argument("--backbone", default="dmpnn")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    a = ap.parse_args()

    # (a) stratum main effect actually present?
    tr = pd.read_csv(DATA / f"{a.split}_train.csv", usecols=["endpoint", "duration", "target_log10"])
    eff = fit_stratum_effect(tr, tr["target_log10"].to_numpy(np.float64))
    strata = {k: v for k, v in eff.items() if k != "__grand__"}
    print("=== (a) (A) stratum main effect on this split ===")
    print(f"  n strata = {len(strata)}   grand mean = {eff['__grand__']:+.4f}")
    for k, v in sorted(strata.items()):
        print(f"    {k:12s} effect = {v:+.4f}")
    print(f"  spread = {max(strata.values()) - min(strata.values()):.4f}"
          f"   ({'DEGENERATE (identity)' if len(strata) <= 1 else 'non-degenerate'})")

    # (b) within-stratum ladder
    print("\n=== (b) WITHIN-STRATUM ladder RMSE (mean over seeds) ===")
    rows = []
    for seed in a.seeds:
        preds = {k: load(a.backbone, v, a.split, seed) for k, v in TIERS.items()}
        if any(p is None for p in preds.values()):
            continue
        for key in preds["T0"].groupby(["endpoint", "duration"]).groups:
            rec = {"endpoint": key[0], "duration": key[1], "seed": seed}
            for t, p in preds.items():
                sub = p[(p["endpoint"] == key[0]) & (p["duration"] == key[1])]
                rec[t] = rmse(sub) if len(sub) else np.nan
                rec["n"] = len(sub)
            rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        print("  no prediction files found.")
        return
    agg = df.groupby(["endpoint", "duration"]).agg(
        n=("n", "first"), T0=("T0", "mean"), T1=("T1", "mean"),
        T4=("T4", "mean"), T5=("T5", "mean")).reset_index()
    agg["T4-T0"] = agg["T4"] - agg["T0"]
    agg["T4-T1"] = agg["T4"] - agg["T1"]
    agg["T5-T4"] = agg["T5"] - agg["T4"]
    print(agg.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    ok = int((agg["T4-T1"] < 0).sum())
    print(f"\n  strata where T4 < T1 (interaction still wins inside one endpoint x duration cell): "
          f"{ok}/{len(agg)}")
    ok5 = int((agg["T5-T4"] < 0).sum())
    print(f"  strata where T5 < T4: {ok5}/{len(agg)}")

    # (c) per-species gain vs EC50 fraction
    print("\n=== (c) per-species gain (T4-T0) vs species EC50 fraction ===")
    trs = pd.read_csv(DATA / f"{a.split}_train.csv", usecols=["species", "endpoint", "n_source_rows"])
    trs["ec"] = (trs["endpoint"].astype(str).str.upper() == "EC50") * trs["n_source_rows"]
    g = trs.groupby("species")
    frac = (g["ec"].sum() / g["n_source_rows"].sum()).rename("ec50_frac")
    gains = []
    for seed in a.seeds:
        p0, p4 = load(a.backbone, TIERS["T0"], a.split, seed), load(a.backbone, TIERS["T4"], a.split, seed)
        if p0 is None or p4 is None:
            continue
        m = p0[["species"]].copy()
        m["e0"] = (p0["pred_log10"] - p0["true_log10"]).abs()
        m["e4"] = (p4["pred_log10"] - p4["true_log10"]).abs()
        s = m.groupby("species").agg(g0=("e0", "mean"), g4=("e4", "mean"))
        s["gain"] = s["g4"] - s["g0"]
        gains.append(s["gain"])
    gm = pd.concat(gains, axis=1).mean(axis=1).rename("gain")
    j = pd.concat([gm, frac], axis=1).dropna()
    x, y = j["gain"].to_numpy(float), j["ec50_frac"].to_numpy(float)
    r = float(np.corrcoef(x, y)[0, 1])
    rs = float(np.corrcoef(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())[0, 1])
    print(f"  n species = {len(j)}   corr(gain, ec50_frac): Pearson={r:+.3f}  Spearman={rs:+.3f}")
    print(f"  [Tier 1' offset confound was -0.251 before the fix, -0.072 after]")
    pl, pe = j[j["ec50_frac"] == 0]["gain"], j[j["ec50_frac"] == 1]["gain"]
    if len(pl) > 1 and len(pe) > 1:
        print(f"  pure-LC50 species n={len(pl)} mean gain={pl.mean():+.4f} | "
              f"pure-EC50 n={len(pe)} mean gain={pe.mean():+.4f} | diff={pe.mean()-pl.mean():+.4f}")


if __name__ == "__main__":
    main()
