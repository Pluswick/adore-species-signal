"""Core TOST/gatekeeping logic (§4Δ·§4G·§4G-7). WRITE-ONLY module — the pipeline that consumes it
(scripts/run_q2_gatekeeping.py) must NOT be run on real data before director execution approval
(running = unblinding). This module holds the pure, synthetically-testable pieces.

Δ = per-seed PAIRED difference-in-differences (4-arm general form; shared baseline -> reduces to a
direct per-seed difference). Bootstrap = block(compound_key) + (GNN) common seed sset across arms;
deterministic tier = block-only. Decision = 3 categories with the §4G-7(3) 4th cell (TOST∧NHST both
reject -> 동등). All quantities in log-RMSE units; δ/δ′ are READ from frozen files, never recomputed.
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------- decision (3-cat + 4th cell)
def decide(ci_lo: float, ci_hi: float, delta: float) -> str:
    """§4G-5 + §4G-7. `동등` if the CI ⊂ [−δ,δ] (this ALREADY covers the 4th cell: TOST-reject ∧
    NHST-reject, since a CI inside [−δ,δ] that also excludes 0 is still `동등`). Else `유의한 차이`
    if the CI excludes 0; else `불확정`."""
    if delta <= 0:
        raise ValueError("delta must be > 0 (read from frozen file)")
    tost_reject = (ci_lo >= -delta) and (ci_hi <= delta)      # CI fully inside [−δ, δ]
    nhst_reject = (ci_lo * ci_hi) > 0                          # CI excludes 0
    if tost_reject:
        return "equivalent"        # 동등 (includes 4th cell: tost_reject ∧ nhst_reject)
    if nhst_reject:
        return "significant"       # 유의한 차이
    return "indeterminate"         # 불확정


def is_fourth_cell(ci_lo, ci_hi, delta) -> bool:
    """True iff both TOST rejects (CI⊂[−δ,δ]) AND NHST rejects (CI excludes 0) — the §4G-7(3) cell."""
    return ((ci_lo >= -delta) and (ci_hi <= delta)) and ((ci_lo * ci_hi) > 0)


# ---------------------------------------------------------------- per-seed paired DD bootstrap
def _rmse(true, pred, rows):
    d = pred[rows] - true[rows]
    return np.sqrt(np.mean(d * d))


def _dd_point(true, Ps, rows, sset):
    """Per-seed paired dd averaged over seed indices `sset` (common across arms), on row subset `rows`.
    Ps = [cand, candbase, ref, refbase], each [n_rows, n_seeds]. 4-arm general form; when
    candbase and refbase are the same column (shared baseline) the baseline cancels per seed."""
    vals = []
    for s in sset:
        c = _rmse(true, Ps[0][:, s], rows); cb = _rmse(true, Ps[1][:, s], rows)
        r = _rmse(true, Ps[2][:, s], rows); rb = _rmse(true, Ps[3][:, s], rows)
        vals.append((c - cb) - (r - rb))
    return float(np.mean(vals))


def paired_dd_bootstrap(true, blk, Ps, *, n_boot=2000, rng_seed=20260723):
    """Return {dd, ci_lo, ci_hi, p} at 90% CI. Block bootstrap over compound_key; for GNN
    (n_seeds>1) also resample seeds with a COMMON sset applied to all 4 arms (paired); deterministic
    tier (n_seeds==1) is block-only. Point estimate uses all rows + all seeds."""
    true = np.asarray(true, np.float64)
    n_seeds = Ps[0].shape[1]
    for P in Ps:
        if P.shape[1] != n_seeds:
            raise ValueError("all arms must share the same seed count (paired)")
    # block index -> row list
    blocks = {}
    for i, b in enumerate(blk):
        blocks.setdefault(b, []).append(i)
    block_arr = [np.asarray(v, np.int64) for v in blocks.values()]
    nb = len(block_arr)
    all_rows = np.arange(len(true))
    all_seeds = np.arange(n_seeds)
    dd = _dd_point(true, Ps, all_rows, all_seeds)
    rng = np.random.default_rng(rng_seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        rows = np.concatenate([block_arr[j] for j in rng.integers(0, nb, nb)])
        sset = rng.integers(0, n_seeds, n_seeds) if n_seeds > 1 else np.array([0])  # common across arms
        boot[i] = _dd_point(true, Ps, rows, sset)
    lo, hi = np.percentile(boot, [5, 95])                     # 90% CI (α=0.05 one-sided ×2)
    p = 2.0 * min((boot >= 0).mean(), (boot <= 0).mean())
    return {"dd": dd, "ci_lo": float(lo), "ci_hi": float(hi), "p": float(min(p, 1.0)), "n_seeds": n_seeds}


# ---------------------------------------------------------------- ensemble Δ′ bootstrap (§4δ′ sensitivity)
def _ensemble_dd(true, Ps, rows, sset):
    """Δ′ ensemble form: AVERAGE the per-seed predictions over sset, then one RMSE per arm (legacy
    _dd_core style). Contrast with _dd_point (per-seed then average)."""
    def ens_rmse(P):
        pe = P[np.ix_(rows, sset)].mean(axis=1)
        d = pe - true[rows]
        return np.sqrt(np.mean(d * d))
    return (ens_rmse(Ps[0]) - ens_rmse(Ps[1])) - (ens_rmse(Ps[2]) - ens_rmse(Ps[3]))


def ensemble_dd_bootstrap(true, blk, Ps, *, n_boot=2000, rng_seed=20260724):
    """§4δ′ sensitivity Δ′: point + CI on the CANONICAL 10-seed ensemble. Block bootstrap over
    compound + resample seeds WITHIN the 10 (common sset). Compared against δ′, NOT δ."""
    true = np.asarray(true, np.float64)
    n_seeds = Ps[0].shape[1]
    blocks = {}
    for i, b in enumerate(blk):
        blocks.setdefault(b, []).append(i)
    barr = [np.asarray(v, np.int64) for v in blocks.values()]; nb = len(barr)
    dd = _ensemble_dd(true, Ps, np.arange(len(true)), np.arange(n_seeds))
    rng = np.random.default_rng(rng_seed); boot = np.empty(n_boot)
    for i in range(n_boot):
        rows = np.concatenate([barr[j] for j in rng.integers(0, nb, nb)])
        sset = rng.integers(0, n_seeds, n_seeds)                  # resample within the 10 seeds
        boot[i] = _ensemble_dd(true, Ps, rows, sset)
    lo, hi = np.percentile(boot, [5, 95])
    p = 2.0 * min((boot >= 0).mean(), (boot <= 0).mean())
    return {"dd": float(dd), "ci_lo": float(lo), "ci_hi": float(hi), "p": float(min(p, 1.0)), "scale": "ensemble"}


# ---------------------------------------------------------------- 2-stage gate (§4G-1)
def stage2_reached(warm_decision: str) -> bool:
    """§4G-1: a (backbone, tier-pair) chain advances warm -> 종-cold ONLY if warm == `동등`.
    `불확정` and `유의한 차이` terminate the chain at stage 1 (do NOT reach stage 2)."""
    return warm_decision == "equivalent"


# ---------------------------------------------------------------- BH-FDR (per family + global)
def bh_fdr(pvals):
    """Benjamini-Hochberg q-values, order-preserving."""
    p = np.asarray(pvals, np.float64)
    n = len(p)
    if n == 0:
        return np.array([])
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        prev = min(prev, p[idx] * n / (rank + 1))
        q[idx] = prev
    return q
