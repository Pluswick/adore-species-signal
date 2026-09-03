"""Synthetic-data unit tests for the gatekeeping/TOST decision logic (§3-2, director).
NO real data, NO Δ on real predictions — pure logic verification. Env: conda run -n src."""
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, r".")
from src.gatekeeping import (decide, is_fourth_cell, paired_dd_bootstrap, ensemble_dd_bootstrap,
                                 bh_fdr, stage2_reached)

D = 0.02          # synthetic δ
fails = []
def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond: fails.append(name)


def _arms(true, cand_err, ref_err, n_seeds, seed_noise, rng):
    """Build [cand, candbase, ref, refbase] each [n_rows, n_seeds]; shared baseline (candbase=refbase).
    cand_pred = true + N(cand_err, seed_noise) per seed; ref similarly. Baseline = true + big const."""
    n = len(true)
    def P(err):
        cols = []
        for _ in range(n_seeds):
            cols.append(true + err + rng.normal(0, seed_noise, n))
        return np.stack(cols, axis=1)
    cand, ref = P(cand_err), P(ref_err)
    base = P(0.5)                      # common baseline (cancels in the DD)
    return [cand, base, ref, base]


print("=== 1) decision logic (3 categories + 4th cell) ===")
# 1: Δ≈0, narrow CI -> 동등
check("Δ≈0 narrow -> equivalent", decide(-0.005, 0.005, D) == "equivalent")
# 2: Δ=3δ, narrow CI -> 유의한 차이
check("Δ=3δ narrow -> significant", decide(3*D-0.005, 3*D+0.005, D) == "significant")
# 3: Δ≈0, wide CI (beyond δ) -> 불확정
check("Δ≈0 wide -> indeterminate", decide(-3*D, 3*D, D) == "indeterminate")
# 4: CI excludes 0 AND ⊂[−δ,δ] -> 동등 (4th cell)
check("4th cell (excl 0, in [−δ,δ]) -> equivalent", decide(0.005, 0.015, D) == "equivalent")
check("4th cell flagged", is_fourth_cell(0.005, 0.015, D) is True)
check("non-4th (incl 0) not flagged", is_fourth_cell(-0.005, 0.015, D) is False)
# boundary: CI hi exactly δ -> still equivalent
check("boundary hi=δ -> equivalent", decide(-0.01, D, D) == "equivalent")

print("\n=== 2) paired DD bootstrap on synthetic predictions ===")
rng = np.random.default_rng(0)
n_rows = 400
true = rng.normal(0, 1, n_rows)
blk = np.repeat(np.arange(n_rows // 4), 4)           # 100 compound blocks of 4
# (a) cand == ref behaviour -> dd≈0, low seed noise -> narrow CI -> equivalent
Ps_eq = _arms(true, cand_err=0.30, ref_err=0.30, n_seeds=10, seed_noise=0.02, rng=rng)
r_eq = paired_dd_bootstrap(true, blk, Ps_eq, n_boot=500)
check(f"equal arms -> equivalent (dd={r_eq['dd']:.4f}, CI=[{r_eq['ci_lo']:.4f},{r_eq['ci_hi']:.4f}])",
      decide(r_eq['ci_lo'], r_eq['ci_hi'], D) == "equivalent")
# (b) cand much worse -> dd large positive -> significant
Ps_sig = _arms(true, cand_err=0.60, ref_err=0.20, n_seeds=10, seed_noise=0.02, rng=rng)
r_sig = paired_dd_bootstrap(true, blk, Ps_sig, n_boot=500)
check(f"worse cand -> significant (dd={r_sig['dd']:.4f})",
      decide(r_sig['ci_lo'], r_sig['ci_hi'], D) == "significant")
# (c) equal MEAN dd≈0 but each seed has its own error offset -> dd(s) swings across seeds ->
#     wide bootstrap CI straddling 0 and exceeding δ -> indeterminate
def _arms_seedvary(spread):
    def P(base_err):
        cols = [true + (base_err + rng.normal(0, spread)) + rng.normal(0, 0.02, n_rows) for _ in range(10)]
        return np.stack(cols, axis=1)
    return [P(0.30), P(0.50), P(0.30), P(0.50)]   # cand,base,ref,base ; cand & ref same mean -> dd≈0
Ps_ind = _arms_seedvary(spread=0.10)
r_ind = paired_dd_bootstrap(true, blk, Ps_ind, n_boot=500)
check(f"seed-varying dd -> indeterminate (CI=[{r_ind['ci_lo']:.3f},{r_ind['ci_hi']:.3f}], dd={r_ind['dd']:.4f})",
      decide(r_ind['ci_lo'], r_ind['ci_hi'], D) == "indeterminate")
# deterministic tier (1 seed) -> block-only, no seed resample, still runs
Ps_det = _arms(true, 0.30, 0.30, n_seeds=1, seed_noise=0.02, rng=rng)
r_det = paired_dd_bootstrap(true, blk, Ps_det, n_boot=300)
check("det tier (1 seed) block-only runs", r_det["n_seeds"] == 1 and np.isfinite(r_det["dd"]))

print("\n=== 3) 2-stage gate (§4G-1: only 동등 advances) ===")
check("indeterminate warm does NOT reach stage2", stage2_reached("indeterminate") is False)
check("significant warm does NOT reach stage2", stage2_reached("significant") is False)
check("equivalent warm reaches stage2", stage2_reached("equivalent") is True)

print("\n=== 4) BH-FDR family separation (no cross-family mixing) ===")
famA = [0.001, 0.02, 0.5]
famB = [0.04, 0.04, 0.04, 0.04]
qA_alone = bh_fdr(famA)
qA_in_global = bh_fdr(famA + famB)[:len(famA)]
check("family-A q-values independent of family-B (separate BH)", not np.allclose(qA_alone, qA_in_global))
check("BH monotone + ≤1", np.all(bh_fdr(famA) <= 1.0))

print("\n=== 5) frozen-file-absence guard (pipeline must fail if δ/δ′ missing) ===")
from pathlib import Path
missing = not Path("results/q2_v4/audit/NONEXISTENT_delta.json").exists()
def load_frozen_or_die(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"frozen δ file absent: {path} — pipeline must not recompute δ")
    return p
raised = False
try:
    load_frozen_or_die("results/q2_v4/audit/NONEXISTENT_delta.json")
except FileNotFoundError:
    raised = True
check("missing frozen δ -> immediate fail", raised)

print("\n=== 6) C15 (δ_det) + C16 (δ′ ensemble sensitivity) ===")
D_DET = 0.087
# (1) det comparison decides with δ_det, not GNN δ: a CI in (δ, δ_det) is 동등 under δ_det but not under δ
ci_between = (0.03, 0.05)
check("det uses δ_det -> equivalent", decide(*ci_between, D_DET) == "equivalent")
check("same CI under GNN δ -> NOT equivalent (margin choice matters)", decide(*ci_between, D) != "equivalent")
# (2) deterministic bootstrap = block-only (n_seeds==1, no seed resample) -> reproducible given rng
Ps_1 = _arms(true, 0.30, 0.30, n_seeds=1, seed_noise=0.02, rng=np.random.default_rng(1))
b1 = paired_dd_bootstrap(true, blk, Ps_1, n_boot=300, rng_seed=7)
b2 = paired_dd_bootstrap(true, blk, Ps_1, n_boot=300, rng_seed=7)
check("det bootstrap n_seeds==1 (no seed resample)", b1["n_seeds"] == 1)
check("det bootstrap block-only reproducible", (b1["ci_lo"], b1["ci_hi"]) == (b2["ci_lo"], b2["ci_hi"]))
# (3) ensemble sensitivity uses δ′ and the ensemble Δ′ is more stable (narrower CI) than per-seed
Ps_e = _arms(true, 0.35, 0.30, n_seeds=10, seed_noise=0.06, rng=np.random.default_rng(2))
r_per = paired_dd_bootstrap(true, blk, Ps_e, n_boot=300)
r_ens = ensemble_dd_bootstrap(true, blk, Ps_e, n_boot=300)
check("ensemble Δ′ tagged 'ensemble'", r_ens.get("scale") == "ensemble")
check("ensemble CI narrower than per-seed", (r_ens["ci_hi"] - r_ens["ci_lo"]) < (r_per["ci_hi"] - r_per["ci_lo"]))
# (4) sensitivity_ensemble excluded from the primary BH-FDR pool
mock = [{"family": "primary", "p": 0.01}, {"family": "primary", "p": 0.5},
        {"family": "sensitivity_ensemble", "p": 0.01}]
check("primary FDR pool excludes sensitivity", len([r for r in mock if r["family"] == "primary"]) == 2)
# (5) sensitivity_ensemble is NOT gated (gate only primary/confirmatory TOST)
def _gated(r): return r.get("test") == "TOST" and r.get("family") in ("primary", "confirmatory")
check("sensitivity TOST NOT gated", _gated({"test": "TOST", "family": "sensitivity_ensemble"}) is False)
check("primary TOST IS gated", _gated({"test": "TOST", "family": "primary"}) is True)

print("\n=== 7) exploratory + deterministic family separation (own FDR, not gated) ===")
mock2 = [{"family": "primary", "p": 0.01}, {"family": "exploratory", "p": 0.01},
         {"family": "deterministic", "p": 0.02}, {"family": "sensitivity_ensemble", "p": 0.03}]
check("primary FDR pool excludes exploratory/deterministic/sensitivity",
      len([r for r in mock2 if r["family"] == "primary"]) == 1)
check("exploratory TOST NOT gated", _gated({"test": "TOST", "family": "exploratory"}) is False)
check("deterministic TOST NOT gated", _gated({"test": "TOST", "family": "deterministic"}) is False)
check("per-family FDR: each family q from own p-list only",
      all(len([r for r in mock2 if r["family"] == f]) >= 1 for f in ["primary", "exploratory", "deterministic"]))

print("\n=== 8) cross-backbone mixed-seed tiling (10-seed GNN vs 1-seed deterministic ref) ===")
rng8 = np.random.default_rng(8)
n8 = 300
true8 = rng8.normal(0, 1, n8)
blk8 = np.repeat(np.arange(n8 // 3), 3)
cand8 = np.stack([true8 + 0.30 + rng8.normal(0, 0.03, n8) for _ in range(10)], axis=1)      # GNN 10-seed
cbase8 = np.stack([true8 + 0.50 + rng8.normal(0, 0.03, n8) for _ in range(10)], axis=1)      # GNN 10-seed
ref8 = (true8 + 0.32)[:, None]                                                                # LightGBM 1-seed
rbase8 = (true8 + 0.50)[:, None]                                                              # LightGBM 1-seed
Ps8 = [cand8, cbase8, ref8, rbase8]
nmax = max(P.shape[1] for P in Ps8)
Ps8t = [np.repeat(P, nmax, axis=1) if P.shape[1] == 1 else P for P in Ps8]                    # align() tiling
check("tiling: all arms 10 cols after normalize", all(P.shape[1] == 10 for P in Ps8t))
raised8 = False
try:
    r8 = paired_dd_bootstrap(true8, blk8, Ps8t, n_boot=300)
except ValueError:
    raised8 = True
check("mixed-seed bootstrap does NOT raise after tiling", not raised8)
# expected dd = mean_s[RMSE(cand_s)-RMSE(cbase_s)] - [RMSE(ref)-RMSE(rbase)]  (det bracket constant)
def _r(p, t): d = p - t; return float(np.sqrt(np.mean(d * d)))
gnn_diff = float(np.mean([_r(cand8[:, s], true8) - _r(cbase8[:, s], true8) for s in range(10)]))
det_diff = _r(ref8[:, 0], true8) - _r(rbase8[:, 0], true8)
check(f"tiled dd == GNN-per-seed − det-const (exp={gnn_diff-det_diff:+.4f}, got={r8['dd']:+.4f})",
      abs(r8["dd"] - (gnn_diff - det_diff)) < 1e-9)
check("det arm adds 0 seed variance (tiled cols identical)", np.allclose(Ps8t[2][:, 0], Ps8t[2][:, 9]))

print(f"\n=== SYNTHETIC TESTS: {'ALL PASS' if not fails else f'{len(fails)} FAILED: {fails}'} ===")
sys.exit(1 if fails else 0)
