"""Compute and FREEZE the primary δ (§4δ, per-seed pooled within-condition RMSE SD).

δ = √( Σ_c (n_c−1)·s_c² / Σ_c (n_c−1) ), s_c = SD of the 10 canonical per-seed RMSEs of condition c
(C=14 warm main conditions, discovery×group). Single-arm quantity (each condition's own RMSEs) — NOT
a tier comparison, so computing/freezing it is the gate-OPENER prerequisite, not a gated comparison.
Reads predictions via the SSOT whitelist loader (pred/true only). First run freezes; later runs verify
the frozen value is unchanged (§4δ '동결 후 불변 검증'). Env: conda run -n jcim_v3.
"""
from __future__ import annotations
import sys, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
sys.path.insert(0, r".")
from jcim_v3.prediction_io import load_prediction_csv

PRED = Path(r".\results\q2_v4\runs\gnn\predictions")
FROZEN = Path(r".\results\q2_v4\audit\delta_primary_frozen.json")
SPLIT = "discovery_group"; SEEDS = list(range(10))
TIERS = [("t0", "no_species"), ("t1", "species_bias_only"), ("t1p", "tier1prime_oof"),
         ("t2", "true_species_categorical"), ("t3a", "true_species_taxonomy_original"),
         ("t3b", "true_species_taxonomy_ncbi"), ("t4", "true_species_late_fusion")]
BB = ["dmpnn", "graphconv"]


def rmse_of(bb, tier, var, seed):
    stem = f"{bb}_tier1prime_oof_{SPLIT}_s{seed}_e100_nfull" if tier == "t1p" \
        else f"{bb}_{var}_{SPLIT}_s{seed}_e100_nfull"
    d = load_prediction_csv(PRED / f"{stem}.csv", columns=["pred_log10", "true_log10"])
    e = d["pred_log10"].to_numpy(np.float64) - d["true_log10"].to_numpy(np.float64)
    return float(np.sqrt(np.mean(e * e)))


def _pre_freeze_evidence():
    """Accurate results-not-observed evidence: count actual comparison OUTPUT FILES, not dir existence.
    (runs/bootstrap is an empty Jul-29 placeholder dir; an empty dir is NOT a comparison output.)"""
    bs = Path("results/q2_v4/runs/bootstrap")
    bs_files = [str(p) for p in bs.rglob("*") if p.is_file()] if bs.exists() else []
    return {
        "gatekeeping_script_exists": Path("scripts/run_q2_gatekeeping.py").exists(),
        "bootstrap_dir_exists": bs.exists(),
        "bootstrap_output_file_count": len(bs_files),
        "bootstrap_output_files": bs_files,
        "note": "δ frozen BEFORE any tier comparison/TOST/gate output exists. runs/bootstrap exists as an "
                "empty placeholder (0 output files) -> no comparison ran; gatekeeping script absent.",
    }


def compute():
    per = {}
    num = den = 0.0
    for tier, var in TIERS:
        for bb in BB:
            cond = f"{bb}/{tier}"
            rmses = [rmse_of(bb, tier, var, s) for s in SEEDS]
            s_c = float(np.std(rmses, ddof=1))
            per[cond] = {"n": len(rmses), "s_c": s_c, "rmses": [round(r, 6) for r in rmses]}
            num += (len(rmses) - 1) * s_c ** 2
            den += (len(rmses) - 1)
    delta = float(np.sqrt(num / den))
    return delta, per, den


if __name__ == "__main__":
    delta, per, df_total = compute()
    record = {
        "spec": "§4δ primary δ = df-weighted pooled within-condition SD of per-seed RMSE; "
                "condition=(backbone,tier,main,group,discovery,warm); C=14; n_c=10; df_c=9.",
        "scale": "per-seed (D-ΔδMATCH A-1)", "C": len(per), "df_total": int(df_total),
        "delta": delta, "per_condition": per,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "pre_freeze_evidence": _pre_freeze_evidence(),
        "delta_prime_status": "separate §4δ′ sensitivity; frozen independently after gnn_dprime 1,260 runs complete.",
    }
    if FROZEN.exists():
        old = json.loads(FROZEN.read_text(encoding="utf-8"))
        match = abs(old["delta"] - delta) < 1e-12
        print(f"[VERIFY] frozen δ={old['delta']:.6f} vs recomputed {delta:.6f} -> {'MATCH' if match else 'MISMATCH!!!'}")
        sys.exit(0 if match else 2)
    FROZEN.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[FROZEN] primary δ = {delta:.6f}  (C={len(per)}, df_total={int(df_total)})")
    print(f"  per-condition s_c: " + ", ".join(f"{k}={v['s_c']:.4f}" for k, v in per.items()))
    print(f"  written -> {FROZEN}")
