"""Structural-leniency synthetic measurement (§2, director). REAL strata structure (smiles blocks,
row counts of discovery_group) + SYNTHETIC pred/true. Two arms SHARE the compound sample with
correlated errors; inject a true difference of k×margin (k=0,.25,.5,1,2) and record the decision.
Detection threshold = smallest k whose decision leaves `동등`. Done for BOTH the deterministic path
(δ_det, 1 seed, block-only) and the GNN path (δ, 10 seeds, seed×block). Split structure is design,
not result -> NOT unblinding (no real pred/true). Env: conda run -n jcim_v3.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, r".")
from jcim_v3.prediction_io import load_prediction_csv
from jcim_v3.gatekeeping import paired_dd_bootstrap, decide

LGB = Path(r".\results\q2_v4\runs\replication\lgbm\predictions")
DELTA = json.loads(Path("results/q2_v4/audit/delta_primary_frozen.json").read_text())["delta"]
DELTA_DET = 0.087189
KS = [0.0, 0.25, 0.5, 1.0, 2.0]
RNG = np.random.default_rng(20260804)

# REAL structure: smiles blocks + row count from a real discovery_group test set (values NOT used)
real = load_prediction_csv(LGB / "LightGBM_RDKit_no_species_discovery_group_s0.csv", columns=["smiles"])
smiles = real["smiles"].to_numpy(); n = len(smiles)
uniq, comp_id = np.unique(smiles, return_inverse=True)
print(f"real structure: {n} rows, {len(uniq)} compounds (blocks)")


def arms(n_seeds, k, margin, sig_shared=0.9, sig_indep=0.2):
    """cand & ref SHARE the SAME per-seed error realization e (correlation); cand = ref with e scaled
    by (1+f) so dd ≈ margin·k. Sharing e means the sampling shock cancels in the difference (narrow CI
    — the structural point). base cancels in the DD."""
    true = RNG.normal(0, 1, n)
    rmse_ref = np.sqrt(sig_shared ** 2 + sig_indep ** 2)
    f = (k * margin) / rmse_ref
    ref_cols, cand_cols, base_cols = [], [], []
    for _ in range(n_seeds):
        e = sig_shared * RNG.normal(0, 1, len(uniq))[comp_id] + RNG.normal(0, sig_indep, n)   # ONE realization
        ref_cols.append(true + e)
        cand_cols.append(true + e * (1 + f))                                                  # SAME e, scaled
        base_cols.append(true + RNG.normal(0, 0.5, n))
    S = lambda c: np.stack(c, axis=1)
    return true, smiles, [S(cand_cols), S(base_cols), S(ref_cols), S(base_cols)]


def path(name, n_seeds, margin):
    print(f"\n[{name}] margin={margin:.6f}, seeds={n_seeds}")
    detection_k = None
    for k in KS:
        true, blk, Ps = arms(n_seeds, k, margin)
        bs = paired_dd_bootstrap(true, blk, Ps, n_boot=800)
        cat = decide(bs["ci_lo"], bs["ci_hi"], margin)
        print(f"   k={k:<4} dd={bs['dd']:+.4f} CI=[{bs['ci_lo']:+.4f},{bs['ci_hi']:+.4f}] -> {cat}")
        if detection_k is None and cat != "equivalent":
            detection_k = k
    print(f"   detection threshold = {detection_k}×margin" if detection_k is not None else "   detection threshold > 2×margin")
    return detection_k


det_k = path("deterministic (δ_det, block-only)", 1, DELTA_DET)
gnn_k = path("GNN (δ, seed×block)", 10, DELTA)
print(f"\n=== detection thresholds: deterministic = {det_k}×δ_det ; GNN = {gnn_k}×δ ===")
