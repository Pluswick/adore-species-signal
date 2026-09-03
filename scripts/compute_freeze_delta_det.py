"""Compute (and with --freeze, FREEZE) the deterministic-tier margin δ_det (§4δ_det).

Per LightGBM main-tier condition: block bootstrap 2000 (block=compound_key) of the final-metric RMSE
-> s_c = SD of the bootstrap RMSEs. δ_det = √(mean(s_c²)) (equal replicates -> equal weight). naive
EXCLUDED. SSOT loader (pred/true/compound_key). DEFAULT = dry-run; --freeze needs director approval
AND a resolved pooling-set count. Env: conda run -n src.
"""
from __future__ import annotations
import sys, json, argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
sys.path.insert(0, r".")
from src.prediction_io import load_prediction_csv

R = Path(r".\results\q2_v4")
LGB = R / "runs" / "replication" / "lgbm" / "predictions"
NAIVE = R / "runs" / "replication" / "naive" / "predictions"
REP = R / "runs" / "replication" / "predictions"
FROZEN = R / "audit" / "delta_det_frozen.json"
SPLIT = "discovery_group"; N_BOOT = 2000
# LightGBM main-tier ladder (naive excluded; rank-truncation excluded). t1 (additive bias) has NO
# LightGBM baseline -> not present.
TIERS = [("t0", LGB / f"LightGBM_RDKit_no_species_{SPLIT}_s0.csv"),
         ("t1p", NAIVE / f"LightGBM_RDKit_species_residual_calibration_{SPLIT}_s0.csv"),
         ("t2", LGB / f"LightGBM_RDKit_species_categorical_{SPLIT}_s0.csv"),
         ("t3a", LGB / f"LightGBM_RDKit_taxonomy_original_{SPLIT}_s0.csv"),
         ("t3b", LGB / f"LightGBM_RDKit_taxonomy_ncbi_{SPLIT}_s0.csv"),
         ("t4", REP / f"LightGBM_RDKit_species_svd_factor_{SPLIT}_s0.csv")]


def block_boot_rmse_sd(path, seed=20260723):
    # block = compound identity. compound_key ≡ smiles-grouping; use smiles (present in every pred CSV;
    # the t4 SVD CSV lacks a compound_key column). Compound-level block bootstrap either way.
    d = load_prediction_csv(path, columns=["pred_log10", "true_log10", "smiles"])
    pred = d["pred_log10"].to_numpy(np.float64); true = d["true_log10"].to_numpy(np.float64)
    blk = d["smiles"].to_numpy()
    blocks = {}
    for i, b in enumerate(blk):
        blocks.setdefault(b, []).append(i)
    barr = [np.asarray(v, np.int64) for v in blocks.values()]; nb = len(barr)
    rng = np.random.default_rng(seed); rmses = np.empty(N_BOOT)
    for i in range(N_BOOT):
        rows = np.concatenate([barr[j] for j in rng.integers(0, nb, nb)])
        e = pred[rows] - true[rows]; rmses[i] = np.sqrt(np.mean(e * e))
    return float(np.std(rmses, ddof=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--freeze", action="store_true"); a = ap.parse_args()
    per = {}; missing = []
    for tier, p in TIERS:
        if not p.exists():
            missing.append((tier, str(p))); continue
        per[tier] = block_boot_rmse_sd(p)
    delta_det = float(np.sqrt(np.mean([v ** 2 for v in per.values()]))) if per else float("nan")
    print(f"δ_det = {delta_det:.6f}  (C={len(per)} LightGBM main tiers)")
    print("  per-condition s_c: " + ", ".join(f"{k}={v:.4f}" for k, v in per.items()))
    if missing:
        print(f"  MISSING/undefined: {missing}")
    # C=6 director-confirmed 2026-08-04 (LightGBM has no t1; "7" was expectation, not spec). Freeze if all
    # 6 defined tiers resolved (no missing files).
    complete = (not missing and len(per) == 6)
    if not a.freeze:
        print(f"[dry-run] C={len(per)} (director-confirmed 6). {'freeze-eligible' if complete else 'HOLD — missing tier'}")
        sys.exit(0)
    if not complete:
        print(f"REFUSED to freeze: missing tier(s) {missing} or C!={6}. Report."); sys.exit(3)
    bs = R / "runs" / "bootstrap"
    record = {"spec": "§4δ_det = √(mean s_c²); s_c = block-bootstrap(2000, block=smiles) SD of RMSE; "
                      "LightGBM main tiers, discovery×group×warm; naive excluded.",
              "scale": "sample-variation (bootstrap SD; NOT re-run noise)", "C": len(per),
              "delta_det": delta_det, "per_condition": per, "frozen_utc": datetime.now(timezone.utc).isoformat(),
              "delta_break_protocol": "§4δ-break applies equally to δ_det.",
              "pre_freeze_evidence": {"comparison_output_file_count": len(list((bs).rglob("*"))) if bs.exists() else 0,
                                      "note": "frozen before any comparison output exists."}}
    FROZEN.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[FROZEN] δ_det -> {FROZEN}")
