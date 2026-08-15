"""LightGBM Tier 4 (adore_t4) — double-OOF SVD species factor (k=16). Approved §1 spec.

Species x train-compound OOF-residual matrix; double-OOF 5-fold SVD -> k-D per-species factor
injected as LightGBM features. Leakage isolation = Tier 1' procedure (5-fold OOF base + SPEC 4-0b
stratum purge/re-add); the SVD is fit OUT-OF-FOLD so a train row's own label never enters its factor.
Methods: OOF-fit low-dim species factor (NOT GNN end-to-end embedding) -> no cross-backbone Tier 4
comparison. Confound (SVD rank-16 implicit shrinkage vs Tier 1' raw mean) recorded in pre-reg.

Env: conda run -n jcim_v3.
"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, r".")
from jcim_v3.rdkit_lgbm import _features, RDKitLGBMConfig
from jcim_v3.naive_species_baselines import _stratum_key, _train_booster
from jcim_v3.stratum import fit_stratum_effect
from jcim_v3.stratum import remove as stratum_remove
from jcim_v3.stratum import restore as stratum_restore
from jcim_v3.paths import add_ccmpnn_to_path
add_ccmpnn_to_path()
from ccmpnn.metrics import perf_metrics  # noqa: E402

DATA = Path(r".\results\q2_v4\data")
OUT = Path(r".\results\q2_v4\runs\replication")
K = 16
N_FOLDS = 5


def _svd_species_factors(dfp: pd.DataFrame, k: int) -> dict[int, np.ndarray]:
    """species x compound mean-purged-residual matrix (0-fill) -> {species: k-vec} = U[:, :k] * S[:k]."""
    piv = dfp.groupby(["sp", "cp"])["r"].mean().unstack(fill_value=0.0)
    M = piv.to_numpy(np.float64)
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    kk = min(k, S.shape[0])
    fac = U[:, :kk] * S[:kk]
    if kk < k:
        fac = np.pad(fac, ((0, 0), (0, k - kk)))
    return {int(sp): fac[i] for i, sp in enumerate(piv.index)}


def _oof_base_purged_resid(train: pd.DataFrame, cfg: RDKitLGBMConfig, _ledger=None) -> np.ndarray:
    """5-fold OOF no-species base -> residual, SPEC 4-0b stratum-purged (same as Tier 1').
    _ledger (opt): list to append (tr_idx, va_idx) per fold for index-level OOF verification."""
    X, y = _features(train, include_species=False)
    n = len(train)
    oof = np.empty(n, np.float64)
    kf = KFold(N_FOLDS, shuffle=True, random_state=cfg.seed)
    for tr_idx, va_idx in kf.split(np.arange(n)):
        if _ledger is not None:
            _ledger.append((np.asarray(tr_idx), np.asarray(va_idx)))
        b = _train_booster(X.iloc[tr_idx].reset_index(drop=True), y[tr_idx], cfg=cfg)
        oof[va_idx] = b.predict(X.iloc[va_idx], num_iteration=cfg.n_estimators)
    resid = y - oof
    strat = pd.Series(_stratum_key(train))
    seff = resid - pd.Series(resid).groupby(strat).transform("mean").to_numpy()  # purged
    return seff


def build_factor(train: pd.DataFrame, cfg: RDKitLGBMConfig, k: int, _ledger_inner=None, _ledger_outer=None):
    """Return (train_factor [n,k] via double-OOF SVD, full_map {species:k-vec} for test).
    _ledger_inner/_ledger_outer (opt): lists to append per-fold (tr_idx, va_idx) for the inner
    (base residual) and outer (SVD factor) OOF, enabling index-level self-exclusion verification."""
    purged = _oof_base_purged_resid(train, cfg, _ledger=_ledger_inner)
    sp = train["species_idx"].astype(int).to_numpy()
    cp = train["smiles"].astype(str).to_numpy()
    base = pd.DataFrame({"sp": sp, "cp": cp, "r": purged})
    n = len(train)
    train_factor = np.zeros((n, k), np.float64)
    kf = KFold(N_FOLDS, shuffle=True, random_state=cfg.seed + 1)          # OUTER OOF (leakage-safe)
    for tr_idx, va_idx in kf.split(np.arange(n)):
        if _ledger_outer is not None:
            _ledger_outer.append((np.asarray(tr_idx), np.asarray(va_idx)))
        fac_map = _svd_species_factors(base.iloc[tr_idx], k)              # SVD on other folds only
        for i in va_idx:
            train_factor[i] = fac_map.get(int(sp[i]), np.zeros(k))        # unseen species -> 0 (Tier1' cold)
    full_map = _svd_species_factors(base, k)                              # full-train SVD -> test
    return train_factor, full_map


def run_one(backbone_split_seed):
    split, seed = backbone_split_seed
    run_id = f"LightGBM_RDKit_species_svd_factor_{split}_s{seed}"
    out_csv = OUT / "predictions" / f"{run_id}.csv"
    if out_csv.exists():
        return {"skipped": True}
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    cfg = RDKitLGBMConfig(baseline="LightGBM_RDKit_no_species", split=split, seed=seed,
                          data_dir=str(DATA), out_root=str(OUT))
    t0 = time.time()
    train = pd.read_csv(DATA / f"{split}_train.csv")
    test = pd.read_csv(DATA / f"{split}_test.csv")
    train_factor, full_map = build_factor(train, cfg, K)

    stratum_eff = fit_stratum_effect(train, train["target_log10"].to_numpy(np.float64))
    y_adj = stratum_remove(train, train["target_log10"].to_numpy(np.float64), stratum_eff)
    X_tr, _ = _features(train, include_species=False)
    X_te, _ = _features(test, include_species=False)
    for j in range(K):
        X_tr[f"svd{j}"] = train_factor[:, j]
    test_factor = np.array([full_map.get(int(s), np.zeros(K)) for s in test["species_idx"].astype(int)])
    for j in range(K):
        X_te[f"svd{j}"] = test_factor[:, j]

    booster = _train_booster(X_tr, y_adj, cfg=cfg)
    pred_adj = booster.predict(X_te, num_iteration=cfg.n_estimators)
    pred = stratum_restore(test, np.asarray(pred_adj, np.float64), stratum_eff)
    yte = test["target_log10"].to_numpy(np.float64)

    pf = test.copy()
    pf["pred_log10"] = pred; pf["true_log10"] = yte; pf["error_log10"] = pred - yte
    pf["model_name"] = "LightGBM_RDKit_species_svd_factor"; pf["backbone"] = "lightgbm_rdkit"
    pf["variant"] = "species_svd_factor"; pf["species_idx_original"] = test["species_idx"].astype(int)
    pf["adore_tier"] = "adore_t4"
    pf.to_csv(out_csv, index=False, encoding="utf-8")
    rmse = float(perf_metrics(pred, yte)["rmse"])
    meta = {"run_id": run_id, "k": K, "n_train": len(train), "n_test": len(test),
            "n_species_in_factor": len(full_map), "n_test_unseen": int((~test["species_idx"].astype(int)
                .isin(full_map)).sum()), "rmse": round(rmse, 4), "sec": round(time.time() - t0, 1),
            "method": "double_OOF_SVD species x compound purged-OOF-residual, k=16"}
    json.dump(meta, open(OUT / "runs" / f"{run_id}.json", "w"), ensure_ascii=False, indent=2)
    return {"skipped": False, **meta}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="discovery_group")
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    for split in a.splits.split(","):
        for sd in [int(x) for x in a.seeds.split(",")]:
            r = run_one((split, sd))
            print(json.dumps(r, ensure_ascii=False), flush=True)
