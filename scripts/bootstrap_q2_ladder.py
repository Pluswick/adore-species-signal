"""Q2 v4 — formal block bootstrap (2000) + global BH-FDR for F1-F4.

Two-layer uncertainty (SPEC 4-0a), done rigorously:
  - lgbm/naive tiers: deterministic (seed sd ~= 0) -> one prediction vector,
    uncertainty from the COMPOUND-block bootstrap only.
  - GNN tiers: each bootstrap iteration resamples SEEDS (with replacement, paired
    across tiers of the same split) AND compound blocks, so BOTH seed variance and
    row variance enter the CI. Point = mean over all seeds.

Contrasts:
  F1 dd(T2 vs T1')            (all deterministic)             both datasets
  F2 dd(T4) - dd(T2)          (GNN T4 vs lgbm T2)             both datasets x both backbones
  F3 dd(T4) - dd(T2) | warm   and | cold                     both datasets x both backbones
  F4 selectivity = [RMSE(T1,leaky)-RMSE(T4,leaky)]
                 - [RMSE(T1,grp) -RMSE(T4,grp)]  (<0 => leakage erodes T4)  both x both

All aligned on KEY; block = compound_key. CI = pct(2.5,97.5); two-sided p from the
bootstrap sign. Global BH-FDR across every 'ok' contrast.

Env: conda run -n jcim_v3.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jcim_v3.prediction_io import load_prediction_csv  # single-source-of-truth: pred CSV = keys/pred/true only

GNN = Path(r".\results\q2_v4\runs\gnn\predictions")
LGB = Path(r".\results\q2_v4\runs\replication\predictions")
DATA = Path(r".\results\q2_v4\data")
OUT = Path(r".\results\q2_v4\runs\bootstrap")
KEY = ["smiles", "species", "endpoint", "duration"]
N_BOOT = 2000
RNG = np.random.default_rng(20260723)


def load_gnn(bb, var, split, seeds):
    """Aligned frame: KEY, true_log10, compound_key, P=[n_rows, n_seeds]."""
    ref = None
    cols = []
    for s in seeds:
        f = GNN / f"{bb}_{var}_{split}_s{s}_e100_nfull.csv"
        if not f.exists():
            return None
        d = load_prediction_csv(f, columns=KEY + ["pred_log10", "true_log10", "compound_key"]).set_index(KEY)
        if ref is None:
            ref = d[["true_log10", "compound_key"]].copy()
            order = ref.index
        cols.append(d.loc[order, "pred_log10"].to_numpy(np.float64))
    ref = ref.reset_index()
    return {"key": ref, "P": np.stack(cols, axis=1)}


def load_lgb(stem, split, seed0):
    f = LGB / f"{stem}_{split}_s{seed0}.csv"
    if not f.exists():
        return None
    d = load_prediction_csv(f, columns=KEY + ["pred_log10", "true_log10", "compound_key"])
    return {"key": d[KEY + ["true_log10", "compound_key"]].reset_index(drop=True),
            "P": d["pred_log10"].to_numpy(np.float64)[:, None]}  # single "seed"


def _align(arms):
    """Inner-join all arms on KEY; return true, compound_key, and per-arm P matrices."""
    base = arms[0]["key"][KEY + ["true_log10", "compound_key"]].copy()
    base["_row"] = np.arange(len(base))
    merged = base
    idxs = []
    for arm in arms:
        k = arm["key"][KEY].copy()
        k["_ai"] = np.arange(len(k))
        merged = merged.merge(k, on=KEY, how="inner")
        idxs.append("_ai")
        merged = merged.rename(columns={"_ai": f"_ai{len(idxs)}"})
    true = merged["true_log10"].to_numpy(np.float64)
    blk = merged["compound_key"].to_numpy()
    Ps = [arms[i]["P"][merged[f"_ai{i+1}"].to_numpy(np.int64)] for i in range(len(arms))]
    return true, blk, Ps


def _rmse(t, p, idx):
    d = p[idx] - t[idx]
    return np.sqrt(np.mean(d * d))


def dd_bootstrap(cand, candbase, ref, refbase, seeds, *, tag=""):
    """dd = (RMSE_cand - RMSE_candbase) - (RMSE_ref - RMSE_refbase), aligned rows."""
    arms = [cand, candbase, ref, refbase]
    if any(a is None for a in arms):
        return {"tag": tag, "status": "missing"}
    true, blk, Ps = _align(arms)
    return _dd_core(true, blk, Ps, tag=tag)


def _dd_core(true, blk, Ps, *, tag):
    nseed = [P.shape[1] for P in Ps]
    blocks = {}
    for i, b in enumerate(blk):
        blocks.setdefault(b, []).append(i)
    block_arr = [np.asarray(v, np.int64) for v in blocks.values()]
    nb = len(block_arr)

    def point(idx, seed_sets):
        vals = []
        for P, sset in zip(Ps, seed_sets):
            vals.append(P[:, sset].mean(axis=1))
        c, cb, r, rb = vals
        return (_rmse(true, c, idx) - _rmse(true, cb, idx)) - (_rmse(true, r, idx) - _rmse(true, rb, idx))

    full = np.arange(len(true))
    all_seeds = [np.arange(n) for n in nseed]
    dd_point = point(full, all_seeds)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        rows = np.concatenate([block_arr[j] for j in RNG.integers(0, nb, nb)])
        sset = [RNG.integers(0, n, n) if n > 1 else np.array([0]) for n in nseed]
        boot[i] = point(rows, sset)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot >= 0).mean(), (boot <= 0).mean())
    return {"tag": tag, "status": "ok", "n_rows": int(len(true)), "n_blocks": int(nb),
            "dd": float(dd_point), "ci_low": float(lo), "ci_high": float(hi), "p": float(p)}


def dd_subset(cand, candbase, ref, refbase, seeds, species_set, tag):
    """Same as dd but restricted to a species subset (align carries species)."""
    arms = [cand, candbase, ref, refbase]
    if any(a is None for a in arms):
        return {"tag": tag, "status": "missing"}
    base = arms[0]["key"][KEY + ["true_log10", "compound_key"]].copy()
    merged = base
    for i, arm in enumerate(arms):
        k = arm["key"][KEY].copy()
        k[f"_ai{i}"] = np.arange(len(k))
        merged = merged.merge(k, on=KEY, how="inner")
    sel = merged["species"].isin(species_set).to_numpy()
    merged = merged[sel].reset_index(drop=True)
    if len(merged) == 0:
        return {"tag": tag, "status": "empty"}
    true = merged["true_log10"].to_numpy(np.float64)
    blk = merged["compound_key"].to_numpy()
    Ps = [arms[i]["P"][merged[f"_ai{i}"].to_numpy(np.int64)] for i in range(4)]
    return _dd_core(true, blk, Ps, tag=tag)


def selectivity_bootstrap(t1g, t4g, t1l, t4l, seeds, tag):
    """selectivity = [RMSE(T1,leaky)-RMSE(T4,leaky)] - [RMSE(T1,grp)-RMSE(T4,grp)].
    Two independent splits; resample group-blocks and leaky-blocks; paired seeds."""
    if any(a is None for a in (t1g, t4g, t1l, t4l)):
        return {"tag": tag, "status": "missing"}

    def prep(a1, a4):
        tr, bk, (P1, P4) = _align([a1, a4])
        blocks = {}
        for i, b in enumerate(bk):
            blocks.setdefault(b, []).append(i)
        return tr, [np.asarray(v, np.int64) for v in blocks.values()], P1, P4

    tg, bg, P1g, P4g = prep(t1g, t4g)
    tl, bl, P1l, P4l = prep(t1l, t4l)
    ns = P1g.shape[1]

    def sel(t, P1, P4, idx, sset):
        return _rmse(t, P1[:, sset].mean(1), idx) - _rmse(t, P4[:, sset].mean(1), idx)

    alls = np.arange(ns)
    point = sel(tl, P1l, P4l, np.arange(len(tl)), alls) - sel(tg, P1g, P4g, np.arange(len(tg)), alls)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sset = RNG.integers(0, ns, ns)
        rg = np.concatenate([bg[j] for j in RNG.integers(0, len(bg), len(bg))])
        rl = np.concatenate([bl[j] for j in RNG.integers(0, len(bl), len(bl))])
        boot[i] = sel(tl, P1l, P4l, rl, sset) - sel(tg, P1g, P4g, rg, sset)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot >= 0).mean(), (boot <= 0).mean())
    return {"tag": tag, "status": "ok", "n_rows": int(len(tg) + len(tl)),
            "n_blocks": int(len(bg) + len(bl)), "dd": float(point),
            "ci_low": float(lo), "ci_high": float(hi), "p": float(p)}


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return q


def strength(r):
    if r.get("status") != "ok":
        return "n/a"
    sig = r["q"] < 0.05 and (r["ci_low"] * r["ci_high"] > 0)
    fav = r["dd"] < 0
    if sig and fav:
        return "robust"
    if fav and r["p"] < 0.10:
        return "moderate"
    if fav:
        return "directional"
    return "not_supported"


def cold_warm(split):
    c = pd.read_csv(DATA / f"{split}_train.csv", usecols=["species"]).groupby("species").size()
    sp = set(pd.read_csv(DATA / f"{split}_test.csv", usecols=["species"])["species"].unique())
    return {s for s in sp if s not in c.index}, {s for s in sp if s in c.index and c[s] >= 50}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    a = ap.parse_args()
    S = a.seeds
    OUT.mkdir(parents=True, exist_ok=True)
    RG, DG = "replication_group", "discovery_group"
    res = []

    for split in (RG, DG):
        res.append(dd_bootstrap(load_lgb("LightGBM_RDKit_species_categorical", split, S[0]),
                                load_lgb("LightGBM_RDKit_no_species", split, S[0]),
                                load_lgb("LightGBM_RDKit_species_residual_calibration", split, S[0]),
                                load_lgb("LightGBM_RDKit_no_species_oof_base", split, S[0]),
                                S, tag=f"F1 T2>T1' [{split}]"))

    for split in (RG, DG):
        for bb in ("dmpnn", "graphconv"):
            res.append(dd_bootstrap(load_gnn(bb, "true_species_late_fusion", split, S),
                                    load_gnn(bb, "no_species", split, S),
                                    load_lgb("LightGBM_RDKit_species_categorical", split, S[0]),
                                    load_lgb("LightGBM_RDKit_no_species", split, S[0]),
                                    S, tag=f"F2 T4>T2 [{split}/{bb}]"))

    for split in (RG, DG):
        cold, warm = cold_warm(split)
        for bb in ("dmpnn", "graphconv"):
            g4, g0 = load_gnn(bb, "true_species_late_fusion", split, S), load_gnn(bb, "no_species", split, S)
            t2, t0 = load_lgb("LightGBM_RDKit_species_categorical", split, S[0]), load_lgb("LightGBM_RDKit_no_species", split, S[0])
            res.append(dd_subset(g4, g0, t2, t0, S, warm, f"F3 warm T4>T2 [{split}/{bb}]"))
            res.append(dd_subset(g4, g0, t2, t0, S, cold, f"F3 cold T4>T2 [{split}/{bb}]"))

    for base, leak, lab in [(RG, "replication_designed_leaky", "rep"), (DG, "discovery_designed_leaky", "disc")]:
        for bb in ("dmpnn", "graphconv"):
            res.append(selectivity_bootstrap(
                load_gnn(bb, "species_bias_only", base, S), load_gnn(bb, "true_species_late_fusion", base, S),
                load_gnn(bb, "species_bias_only", leak, S), load_gnn(bb, "true_species_late_fusion", leak, S),
                S, tag=f"F4 selectivity(T1-T4 leak) [{lab}/{bb}]"))

    ok = [r for r in res if r.get("status") == "ok"]
    for r, qq in zip(ok, bh_fdr([r["p"] for r in ok])):
        r["q"] = float(qq)
        r["strength"] = strength(r)
    pd.DataFrame(res).to_csv(OUT / "bootstrap_ladder_fdr.csv", index=False, encoding="utf-8")
    print(f"n_bootstrap={N_BOOT}  contrasts={len(res)}  ok={len(ok)}\n")
    for r in res:
        if r.get("status") != "ok":
            print(f"  {r['tag']:46s} {r.get('status')}")
        else:
            print(f"  {r['tag']:46s} dd={r['dd']:+.4f} CI[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] "
                  f"p={r['p']:.1e} q={r['q']:.1e} -> {r['strength']}  (n={r['n_rows']},blk={r['n_blocks']})")
    print(f"\nwrote {OUT / 'bootstrap_ladder_fdr.csv'}")


if __name__ == "__main__":
    main()
