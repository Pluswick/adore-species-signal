"""§6 stop-threshold full evaluation over ALL completed cells (director task, pre-gate integrity).

Upper (immediate-stop) = tier-0 (no_species) cells only; Lower (leak-suspicion, report-not-stop) = ALL tiers.
Partition-specific: discovery upper 1.7195 / lower 0.8598 ; replication upper 1.7094 / lower 0.8547.
Single-arm check (each cell's own RMSE) — NO tier comparison. RMSE from prediction CSVs via SSOT loader
(pred/true only). Reports triggers + near-lower cells + eval scope. Env: conda run -n jcim_v3.
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np
sys.path.insert(0, r".")
from jcim_v3.prediction_io import load_prediction_csv, PredictionColumnViolation

ROOT = Path(r".\results\q2_v4\runs")
DIRS = [ROOT / "gnn" / "predictions", ROOT / "gnn_dprime" / "predictions",
        ROOT / "replication" / "lgbm" / "predictions", ROOT / "replication" / "naive" / "predictions",
        ROOT / "replication" / "predictions"]
TH = {"discovery": {"upper": 1.7195, "lower": 0.8598}, "replication": {"upper": 1.7094, "lower": 0.8547}}
NEAR = 0.10  # flag cells within 10% above the lower bound


def rmse_of(path):
    # RMSE from pred/true (always present, whitelist). Partition + tier-0 from the FILENAME so that
    # block B OOV CSVs (which lack a 'split' column) are also covered.
    d = load_prediction_csv(path, columns=["pred_log10", "true_log10"])
    if "pred_log10" not in d or len(d) == 0:
        return None
    e = d["pred_log10"].to_numpy(np.float64) - d["true_log10"].to_numpy(np.float64)
    name = Path(path).name
    part = "discovery" if "discovery" in name else ("replication" if "replication" in name else None)
    if part is None:
        return None
    is_t0 = "_no_species_" in name
    return name, part, is_t0, float(np.sqrt(np.mean(e * e)))


def main():
    files = []
    for dr in DIRS:
        files += glob.glob(str(dr / "*.csv"))
    upper_trig, lower_trig, near_lower = [], [], []
    n_eval = n_tier0 = 0; skipped = 0
    for f in files:
        try:
            r = rmse_of(f)
        except (PredictionColumnViolation, Exception):
            skipped += 1; continue
        if r is None:
            skipped += 1; continue
        name, part, is_t0, rmse = r
        th = TH[part]
        n_eval += 1; n_tier0 += int(is_t0)
        if is_t0 and rmse >= th["upper"]:
            upper_trig.append((name, part, round(rmse, 4)))
        if rmse < th["lower"]:
            lower_trig.append((name, part, round(rmse, 4)))
        elif rmse < th["lower"] * (1 + NEAR):
            near_lower.append((name, part, round(rmse, 4)))
    print(f"=== §6 STOP-THRESHOLD EVAL ===")
    print(f"eval scope: {n_eval} cells ({n_tier0} tier-0), skipped {skipped}")
    print(f"\nUPPER (immediate-stop, tier-0 only) triggers: {len(upper_trig)}")
    for t in upper_trig: print("   !!", t)
    print(f"\nLOWER (leak-suspicion, all tiers) triggers: {len(lower_trig)}")
    for t in sorted(lower_trig, key=lambda x: x[2]): print("   !", t)
    print(f"\nNEAR-LOWER (within {int(NEAR*100)}% above lower, context): {len(near_lower)}")
    for t in sorted(near_lower, key=lambda x: x[2])[:20]: print("   ~", t)
    verdict = "PASS (no upper trips)" if not upper_trig else "UPPER-BOUND TRIP — STOP"
    print(f"\n=== VERDICT: {verdict}; lower-bound flags={len(lower_trig)} (report-only) ===")
    sys.exit(2 if upper_trig else 0)


if __name__ == "__main__":
    main()
