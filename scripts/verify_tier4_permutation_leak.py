"""Tier 4 SVD label-permutation leakage test (Session 24, QC — director-mandated).

Permute train y (train-only, fixed seed), rebuild the ENTIRE OOF-SVD factor from the permuted
labels, then predict the REAL (stratum-removed) test y using RDKit features + that shuffled-derived
factor. A clean double-OOF factor built from random labels must NOT beat the no-factor baseline;
if it does, the factor is peeking at its own rows => LEAK => stop.

Pre-registered criterion (see GAP_EXECUTION_LOG Session 24): per split, PASS iff
rmse_shuffled >= rmse_nofactor - 0.02 ; LEAK iff rmse_shuffled < rmse_nofactor - 0.02.
Env: conda run -n jcim_v3.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, r".")
sys.path.insert(0, r".\scripts")
# reuse the EXACT tier4 factor pipeline
from run_q2_lgbm_tier4 import build_factor, K, DATA
from jcim_v3.rdkit_lgbm import _features, RDKitLGBMConfig
from jcim_v3.naive_species_baselines import _train_booster
from jcim_v3.stratum import fit_stratum_effect
from jcim_v3.stratum import remove as stratum_remove
from jcim_v3.stratum import restore as stratum_restore
from jcim_v3.paths import add_ccmpnn_to_path
add_ccmpnn_to_path()
from ccmpnn.metrics import perf_metrics  # noqa: E402

PERM_SEED = 20260801           # fixed & recorded
TAU = 0.02                     # pre-registered tolerance (log units)
SPLITS = [f"{p}_{s}" for p in ("discovery", "replication")
          for s in ("group", "scaffold", "scaffold_generic", "designed_leaky")]


def _final_rmse(train, test, cfg, train_factor, full_map):
    """Final booster on the REAL (stratum-removed) y; factor may be None (no-factor baseline)."""
    stratum_eff = fit_stratum_effect(train, train["target_log10"].to_numpy(np.float64))
    y_adj = stratum_remove(train, train["target_log10"].to_numpy(np.float64), stratum_eff)
    X_tr, _ = _features(train, include_species=False)
    X_te, _ = _features(test, include_species=False)
    if train_factor is not None:
        test_factor = np.array([full_map.get(int(s), np.zeros(K)) for s in test["species_idx"].astype(int)])
        for j in range(K):
            X_tr[f"svd{j}"] = train_factor[:, j]
            X_te[f"svd{j}"] = test_factor[:, j]
    booster = _train_booster(X_tr, y_adj, cfg=cfg)
    pred_adj = booster.predict(X_te, num_iteration=cfg.n_estimators)
    pred = stratum_restore(test, np.asarray(pred_adj, np.float64), stratum_eff)
    return float(perf_metrics(pred, test["target_log10"].to_numpy(np.float64))["rmse"])


def run_split(split):
    cfg = RDKitLGBMConfig(baseline="LightGBM_RDKit_no_species", split=split, seed=0,
                          data_dir=str(DATA), out_root="")
    train = pd.read_csv(DATA / f"{split}_train.csv")
    test = pd.read_csv(DATA / f"{split}_test.csv")

    # 1) real factor (harness fidelity vs logged tier4)
    tf_real, map_real = build_factor(train, cfg, K)
    rmse_real = _final_rmse(train, test, cfg, tf_real, map_real)
    # 2) no-factor baseline (identical pipeline, factor omitted)
    rmse_nofactor = _final_rmse(train, test, cfg, None, None)
    # 3) SHUFFLED factor: permute train y (train-only), rebuild the WHOLE factor from permuted labels
    rng = np.random.RandomState(PERM_SEED)
    train_perm = train.copy()
    train_perm["target_log10"] = train["target_log10"].to_numpy(np.float64)[rng.permutation(len(train))]
    tf_shuf, map_shuf = build_factor(train_perm, cfg, K)      # OOF base + residual + SVD all on permuted y
    rmse_shuffled = _final_rmse(train, test, cfg, tf_shuf, map_shuf)   # predict REAL y

    improve_shuffled = rmse_nofactor - rmse_shuffled   # >0 means shuffled factor helped (suspicious)
    improve_real = rmse_nofactor - rmse_real
    leak = improve_shuffled > TAU
    return {"split": split, "n_test": len(test),
            "rmse_real_factor": round(rmse_real, 4), "rmse_nofactor": round(rmse_nofactor, 4),
            "rmse_shuffled_factor": round(rmse_shuffled, 4),
            "improve_real": round(improve_real, 4), "improve_shuffled": round(improve_shuffled, 4),
            "leak": bool(leak)}


if __name__ == "__main__":
    rows = []
    for split in SPLITS:
        r = run_split(split)
        rows.append(r)
        tag = "LEAK!!!" if r["leak"] else "clean"
        print(json.dumps({**r, "verdict": tag}, ensure_ascii=False), flush=True)
    any_leak = any(r["leak"] for r in rows)
    summary = {"perm_seed": PERM_SEED, "tau": TAU, "n_splits": len(rows),
               "any_leak": any_leak, "verdict": "LEAK" if any_leak else "PASS", "rows": rows}
    outp = DATA / "_ext" / "tier4_permutation_leak_test.json"
    outp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== TIER4 PERMUTATION LEAK TEST: " + summary["verdict"] +
          f" (any_leak={any_leak}) === -> {outp}", flush=True)
    sys.exit(2 if any_leak else 0)
