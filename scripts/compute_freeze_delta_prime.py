"""Compute (and, with --freeze, FREEZE) the sensitivity δ′ (§4δ′, ensemble-scale).

δ′ = √(Σ_c (k−1) s_c² / Σ_c (k−1)), s_c = SD over k=10 disjoint 10-seed ENSEMBLE-RMSEs of condition c.
Ensemble j (j=0..9) = seeds [10j..10j+9]; ensemble-RMSE = RMSE of the 10-seed-AVERAGED prediction
(mirrors Δ′ / _dd_core). Canonical ensemble #0 (seeds 0-9) from runs/gnn; ensembles #1-9 (seeds 10-99)
from runs/gnn_dprime. C=14 warm main conditions (discovery×group). SSOT loader (pred/true only).

DEFAULT = dry-run (compute + print, NO freeze). Pass --freeze to write the immutable frozen file
(execution·freeze requires director approval per the Session-26 process note). Env: conda run -n src.
"""
from __future__ import annotations
import sys, json, argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
sys.path.insert(0, r".")
from src.prediction_io import load_prediction_csv

CANON = Path(r".\results\q2_v4\runs\gnn\predictions")
DPRIME = Path(r".\results\q2_v4\runs\gnn_dprime\predictions")
FROZEN = Path(r".\results\q2_v4\audit\delta_prime_frozen.json")
SPLIT = "discovery_group"; KEY = ["smiles", "species", "endpoint", "duration"]
TIERS = [("t0", "no_species"), ("t1", "species_bias_only"), ("t1p", "tier1prime_oof"),
         ("t2", "true_species_categorical"), ("t3a", "true_species_taxonomy_original"),
         ("t3b", "true_species_taxonomy_ncbi"), ("t4", "true_species_late_fusion")]
BB = ["dmpnn", "graphconv"]
K = 10  # ensembles per condition (§4δ′)


def _path(bb, tier, var, seed):
    d = CANON if seed <= 9 else DPRIME  # canonical 0-9 vs δ′ 10-99
    stem = f"{bb}_tier1prime_oof_{SPLIT}_s{seed}_e100_nfull" if tier == "t1p" \
        else f"{bb}_{var}_{SPLIT}_s{seed}_e100_nfull"
    return d / f"{stem}.csv"


def ensemble_rmse(bb, tier, var, seeds):
    frames = []
    for s in seeds:
        p = _path(bb, tier, var, s)
        if not p.exists():
            raise FileNotFoundError(str(p))
        frames.append(load_prediction_csv(p, columns=KEY + ["pred_log10", "true_log10"]).set_index(KEY))
    order = frames[0].index
    preds = np.mean([f.loc[order, "pred_log10"].to_numpy(np.float64) for f in frames], axis=0)
    true = frames[0]["true_log10"].to_numpy(np.float64)
    return float(np.sqrt(np.mean((preds - true) ** 2)))


def compute():
    per = {}; num = den = 0.0; missing = []
    for tier, var in TIERS:
        for bb in BB:
            cond = f"{bb}/{tier}"
            ens_rmse = []
            for j in range(K):
                seeds = list(range(10 * j, 10 * j + 10))
                try:
                    ens_rmse.append(ensemble_rmse(bb, tier, var, seeds))
                except FileNotFoundError as e:
                    missing.append(f"{cond} ens{j}: {e}")
            if len(ens_rmse) == K:
                s_c = float(np.std(ens_rmse, ddof=1))
                per[cond] = {"k": K, "s_c": s_c, "ensemble_rmses": [round(r, 6) for r in ens_rmse]}
                num += (K - 1) * s_c ** 2; den += (K - 1)
    delta_prime = float(np.sqrt(num / den)) if den else float("nan")
    return delta_prime, per, den, missing


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--freeze", action="store_true"); a = ap.parse_args()
    dp, per, df_total, missing = compute()
    if missing:
        print(f"[INCOMPLETE] {len(missing)} ensembles missing (δ′ runs not done) — cannot freeze yet:")
        for m in missing[:6]:
            print("   ", m)
        print(f"   conditions complete: {len(per)}/14")
        if not a.freeze:
            print(f"[dry-run] partial δ′ (complete conds only) = {dp:.6f} over {len(per)} conds")
        sys.exit(1)
    print(f"δ′ = {dp:.6f}  (C={len(per)}, df_total={int(df_total)})")
    print("  per-condition s_c: " + ", ".join(f"{k}={v['s_c']:.4f}" for k, v in per.items()))
    if not a.freeze:
        print("[dry-run] all 14 conditions complete. Pass --freeze to write the frozen file (needs approval).")
        sys.exit(0)
    bs = Path("results/q2_v4/runs/bootstrap")
    bs_files = [str(p) for p in bs.rglob("*") if p.is_file()] if bs.exists() else []
    record = {"spec": "§4δ′ sensitivity δ′ = pooled within-condition SD of k=10 disjoint 10-seed "
                      "ensemble-RMSEs; C=14; k=10; df_c=9.", "scale": "ensemble (§4δ′)",
              "C": len(per), "df_total": int(df_total), "delta_prime": dp, "per_condition": per,
              "frozen_utc": datetime.now(timezone.utc).isoformat(),
              "primary_delta_ref": "audit/delta_primary_frozen.json (per-seed δ=0.019777)",
              "delta_break_protocol": "§4δ-break (δ 동결 파기 규약) applies EQUALLY to δ′: on invalidation of "
                      "any δ′-pooling run, recompute + preserve old&new + audit + re-run any judgments.",
              "pre_freeze_evidence": {
                  "gatekeeping_script_exists": Path("scripts/run_q2_gatekeeping.py").exists(),
                  "bootstrap_dir_exists": bs.exists(), "bootstrap_output_file_count": len(bs_files),
                  "note": "δ′ frozen BEFORE any tier comparison/TOST/gate output (runs/bootstrap = 0 output files)."}}
    FROZEN.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[FROZEN] δ′ written -> {FROZEN}")
