"""Tier 4 SVD OOF-boundary PROOF at the index level (Session 24, QC — director §3).

Not a statistical test (which has a sensitivity floor at the species-identity effect) but a
STRUCTURAL proof: extract the actual fold ledger from build_factor and verify by set operations
that every train row's assigned SVD factor was computed from data EXCLUDING that row, at BOTH the
inner (base-residual) and outer (SVD-factor) OOF layers; and that no test row is ever used in
factor construction. Any violation -> immediate stop. 8 warm splits. Env: conda run -n jcim_v3.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, r".")
sys.path.insert(0, r".\scripts")
from run_q2_lgbm_tier4 import build_factor, K, DATA
from jcim_v3.rdkit_lgbm import RDKitLGBMConfig

SPLITS = [f"{p}_{s}" for p in ("discovery", "replication")
          for s in ("group", "scaffold", "scaffold_generic", "designed_leaky")]
AGG = ["smiles", "species_idx", "endpoint", "duration"]


def _check_folds(ledger, n, sp):
    """Assert valid partition + self-exclusion. Returns (violations, per-row source sizes)."""
    covered = np.zeros(n, dtype=int)
    self_in_source = 0            # rows whose factor SVD pool contains themselves
    self_in_species_source = 0    # rows whose SAME-SPECIES factor contributors contain themselves
    va_all = []
    for tr_idx, va_idx in ledger:
        tr_set = set(int(x) for x in tr_idx)
        if not tr_set.isdisjoint(set(int(x) for x in va_idx)):
            return {"fatal": "tr/va overlap in a fold"}
        covered[va_idx] += 1
        va_all.append(np.asarray(va_idx))
        # species-level contributors for each held-out row
        sp_tr = sp[np.asarray(tr_idx)]
        for i in va_idx:
            if int(i) in tr_set:
                self_in_source += 1
            # same-species contributors = tr rows with species == sp[i]; i is not among them iff i not in tr_set
            if int(i) in tr_set and sp[int(i)] in set(sp_tr):
                self_in_species_source += 1
    return {"fatal": None, "each_row_assigned_once": bool((covered == 1).all()),
            "n_assigned_times_max": int(covered.max()), "n_assigned_times_min": int(covered.min()),
            "self_in_source": self_in_source, "self_in_species_source": self_in_species_source}


def run_split(split):
    cfg = RDKitLGBMConfig(baseline="LightGBM_RDKit_no_species", split=split, seed=0,
                          data_dir=str(DATA), out_root="")
    train = pd.read_csv(DATA / f"{split}_train.csv")
    test = pd.read_csv(DATA / f"{split}_test.csv")
    n = len(train)
    sp = train["species_idx"].astype(int).to_numpy()
    li, lo = [], []
    tf, full_map = build_factor(train, cfg, K, _ledger_inner=li, _ledger_outer=lo)  # REAL code path

    inner = _check_folds(li, n, sp)
    outer = _check_folds(lo, n, sp)

    # TEST exclusion: factor universe is exactly the n train rows; full_map species subset of train;
    # train/test measurement keys disjoint (no test row can be in the factor source).
    train_keys = set(map(tuple, train[AGG].astype(str).to_numpy()))
    test_keys = set(map(tuple, test[AGG].astype(str).to_numpy()))
    test_species = set(int(x) for x in test["species_idx"].astype(int).unique())
    train_species = set(int(x) for x in sp)
    test_unseen = test_species - train_species
    # every test row's factor is full_map.get(sp) (unseen -> zeros); check unseen species really map to 0
    test_factor = np.array([full_map.get(int(s), np.zeros(K)) for s in test["species_idx"].astype(int)])
    unseen_rows = test["species_idx"].astype(int).isin(list(test_unseen)).to_numpy()
    unseen_all_zero = bool(np.all(test_factor[unseen_rows] == 0.0)) if unseen_rows.any() else True

    violations = []
    if inner.get("fatal") or outer.get("fatal"):
        violations.append(f"fold overlap inner={inner.get('fatal')} outer={outer.get('fatal')}")
    if not inner.get("each_row_assigned_once"): violations.append("inner: a row assigned !=1 times")
    if not outer.get("each_row_assigned_once"): violations.append("outer: a row assigned !=1 times")
    if inner.get("self_in_source", 1) != 0: violations.append(f"inner self_in_source={inner['self_in_source']}")
    if outer.get("self_in_source", 1) != 0: violations.append(f"outer self_in_source={outer['self_in_source']}")
    if outer.get("self_in_species_source", 1) != 0: violations.append(f"outer self_in_species_source={outer['self_in_species_source']}")
    if len(train_keys & test_keys) != 0: violations.append(f"train/test key overlap={len(train_keys & test_keys)}")
    if len(full_map) and not set(full_map.keys()).issubset(train_species):
        violations.append("full_map has non-train species")
    if not unseen_all_zero: violations.append("unseen test species got nonzero factor")

    return {"split": split, "n_train": n, "n_test": len(test),
            "inner_self_in_source": inner.get("self_in_source"), "outer_self_in_source": outer.get("self_in_source"),
            "outer_self_in_species_source": outer.get("self_in_species_source"),
            "each_row_once_inner": inner.get("each_row_assigned_once"),
            "each_row_once_outer": outer.get("each_row_assigned_once"),
            "train_test_key_overlap": len(train_keys & test_keys),
            "full_map_species_subset_of_train": bool(set(full_map.keys()).issubset(train_species)),
            "test_unseen_species": len(test_unseen), "unseen_factor_all_zero": unseen_all_zero,
            "violations": violations, "ok": len(violations) == 0}


if __name__ == "__main__":
    rows = []
    for split in SPLITS:
        r = run_split(split)
        rows.append(r)
        print(json.dumps({**r}, ensure_ascii=False), flush=True)
    all_ok = all(r["ok"] for r in rows)
    summary = {"n_splits": len(rows), "all_ok": all_ok, "verdict": "PASS" if all_ok else "VIOLATION", "rows": rows}
    (DATA / "_ext" / "tier4_oof_index_proof.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== TIER4 OOF INDEX PROOF: {summary['verdict']} (all_ok={all_ok}) ===", flush=True)
    sys.exit(2 if not all_ok else 0)
