"""Block A CPU tiers (LightGBM + naive + Tier4 SVD) over warm splits. Deterministic -> seed 0 (D11).
Runs on CPU in parallel with the GPU ladder. Resumable (skip if run json exists)."""
from __future__ import annotations
import sys, json, time
from pathlib import Path
sys.path.insert(0, r".")
from src.rdkit_lgbm import run_rdkit_lgbm, RDKitLGBMConfig
from src.naive_species_baselines import (run_naive_species_baselines, run_naive_taxonomy_baselines,
                                             NaiveSpeciesBaselineConfig)
from src.tier_input_guard import assert_tier_input, TierInputDegenerate

def _lgb_variant(base):  # map a LightGBM baseline name to a guard variant (representation input)
    if "taxonomy_ncbi" in base: return "true_species_taxonomy_ncbi"
    if "taxonomy_original" in base: return "true_species_taxonomy_original"
    if "categorical" in base: return "true_species_categorical"
    return "no_species"

DATA = r".\results\q2_v4\data"
LGB_OUT = r".\results\q2_v4\runs\replication\lgbm"
NAIVE_OUT = r".\results\q2_v4\runs\replication\naive"
SPLITS = [f"{p}_{s}" for p in ("discovery", "replication")
          for s in ("group", "scaffold", "scaffold_generic", "designed_leaky")]
LGB = ["LightGBM_RDKit_no_species",
       "LightGBM_RDKit_species_categorical", "LightGBM_RDKit_zero_species_categorical",
       "LightGBM_RDKit_shuffled_species_categorical", "LightGBM_RDKit_dummy_species_categorical",
       "LightGBM_RDKit_taxonomy_original", "LightGBM_RDKit_shuffled_taxonomy_original",
       "LightGBM_RDKit_taxonomy_ncbi", "LightGBM_RDKit_shuffled_taxonomy_ncbi"]

GUARD_LOG = str(Path(LGB_OUT) / "tier_input_guard.jsonl")
REF = str(Path(DATA) / "tier_input_reference.json")
done = skip = fail = 0; t0 = time.time()
for split in SPLITS:
    guarded = set()
    for base in LGB:
        v = _lgb_variant(base)
        if v not in guarded:
            guarded.add(v)
            try:
                assert_tier_input(v, split, DATA, GUARD_LOG, REF)
            except TierInputDegenerate as deg:
                print(f"### HALT — tier input degenerate ({base}/{split}) ###\n{deg}", flush=True); sys.exit(2)
        rid = f"{base}_{split}_s0"
        if (Path(LGB_OUT) / "runs" / f"{rid}.json").exists():
            skip += 1; continue
        try:
            r = run_rdkit_lgbm(RDKitLGBMConfig(baseline=base, split=split, seed=0, data_dir=DATA, out_root=LGB_OUT))
            done += 1; print(f"[LGB done] {rid} rmse={r['A']['rmse']:.4f}", flush=True)
        except Exception as e:
            fail += 1; print(f"[LGB FAIL] {rid}: {type(e).__name__}: {e}", flush=True)
    # naive species mean + residual-calibration (Tier 0/1/1') and taxon backoff (3a/3b)
    for fn, tag in [(run_naive_species_baselines, "naive_sp"), (run_naive_taxonomy_baselines, "naive_tax")]:
        try:
            nr = fn(NaiveSpeciesBaselineConfig(split=split, seed=0, data_dir=DATA, out_root=NAIVE_OUT))
            done += 1; print(f"[{tag} done] {split} n={len(nr)}", flush=True)
        except Exception as e:
            fail += 1; print(f"[{tag} FAIL] {split}: {type(e).__name__}: {e}", flush=True)
print(f"\n=== CPU tiers: done={done} skip={skip} fail={fail} time={(time.time()-t0)/60:.1f}m ===", flush=True)
