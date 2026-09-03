"""B1 deterministic block (CPU) — LightGBM baselines + rank LightGBM + naive species/taxonomy +
SVD Tier4. Byte-identical compute to Phase 1 (run_q2_cpu_tiers_blockA.py + run_q2_lgbm_tier4.py);
only data path (data_b1), output root (runs_b1), split names (b1_*), + ledger append differ.
Deterministic -> seed 0. Runs on CPU concurrently with the GPU ladder. Resumable.
Env: conda run -n src.  (no GPU env needed)
"""
from __future__ import annotations
import sys, json, time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, r".")
sys.path.insert(0, r".\scripts")
from src.rdkit_lgbm import run_rdkit_lgbm, RDKitLGBMConfig
from src.naive_species_baselines import (run_naive_species_baselines, run_naive_taxonomy_baselines,
                                             NaiveSpeciesBaselineConfig)
from src.tier_input_guard import assert_tier_input, TierInputDegenerate

ROOT = Path(r".\results\q2_v4")
DATA = str(ROOT / "data_b1")
LGB_OUT = str(ROOT / "runs_b1" / "lgbm")
NAIVE_OUT = str(ROOT / "runs_b1" / "naive")
GUARD_LOG = str(Path(LGB_OUT) / "tier_input_guard.jsonl")
REF = str(Path(DATA) / "tier_input_reference.json")
LEDGER = ROOT / "runs_b1" / "_status" / "progress_deterministic.jsonl"
BLOCK = "deterministic"

SPLITS = ["b1_group", "b1_scaffold", "b1_scaffold_generic", "b1_designed_leaky"]
LGB = ["LightGBM_RDKit_no_species",
       "LightGBM_RDKit_species_categorical", "LightGBM_RDKit_zero_species_categorical",
       "LightGBM_RDKit_shuffled_species_categorical", "LightGBM_RDKit_dummy_species_categorical",
       "LightGBM_RDKit_taxonomy_original", "LightGBM_RDKit_shuffled_taxonomy_original",
       "LightGBM_RDKit_taxonomy_ncbi", "LightGBM_RDKit_shuffled_taxonomy_ncbi"]
LGB_RANK = ["LightGBM_RDKit_taxonomy_genus", "LightGBM_RDKit_taxonomy_genusfamily"]  # b1_group only


def ledger(rid, status, error=None):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    rec = {"run_id": rid, "block": BLOCK, "status": status, "ts": datetime.now().isoformat(timespec="seconds")}
    if error is not None:
        rec["error"] = str(error)[:200]
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _lgb_variant(base):
    if "taxonomy_ncbi" in base: return "true_species_taxonomy_ncbi"
    if "taxonomy_genusfamily" in base: return "true_species_taxonomy_genusfamily"
    if "taxonomy_genus" in base: return "true_species_taxonomy_genus"
    if "taxonomy_original" in base: return "true_species_taxonomy_original"
    if "categorical" in base: return "true_species_categorical"
    return "no_species"


def do_lgb(base, split):
    rid = f"{base}_{split}_s0"
    if (Path(LGB_OUT) / "runs" / f"{rid}.json").exists():
        return "skip"
    try:
        run_rdkit_lgbm(RDKitLGBMConfig(baseline=base, split=split, seed=0, data_dir=DATA, out_root=LGB_OUT))
        ledger(rid, "ok"); return "ok"
    except Exception as e:
        ledger(rid, "fail", e); print(f"[LGB FAIL] {rid}: {type(e).__name__}: {e}", flush=True); return "fail"


def main():
    t0 = time.time(); nok = nfail = nskip = 0
    # --- LightGBM standard baselines (+ guard per representation) over 4 splits ---
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
            r = do_lgb(base, split); nok += r == "ok"; nfail += r == "fail"; nskip += r == "skip"
            if r == "ok": print(f"[LGB ok] {base}_{split}", flush=True)
    # --- rank LightGBM (b1_group only) ---
    for base in LGB_RANK:
        r = do_lgb(base, "b1_group"); nok += r == "ok"; nfail += r == "fail"; nskip += r == "skip"
        if r == "ok": print(f"[LGB-rank ok] {base}_b1_group", flush=True)
    # --- naive species + taxonomy (per split; log per returned model) ---
    for split in SPLITS:
        for fn, tag in [(run_naive_species_baselines, "naive_sp"), (run_naive_taxonomy_baselines, "naive_tax")]:
            try:
                nr = fn(NaiveSpeciesBaselineConfig(split=split, seed=0, data_dir=DATA, out_root=NAIVE_OUT))
                items = nr if isinstance(nr, (list, tuple)) else [nr]
                for i, it in enumerate(items):
                    rid = None
                    if isinstance(it, dict):
                        rid = it.get("run_id") or it.get("model_name")
                    rid = rid or f"{tag}_{split}_m{i}"
                    ledger(rid, "ok"); nok += 1
                print(f"[{tag} ok] {split} n={len(items)}", flush=True)
            except Exception as e:
                ledger(f"{tag}_{split}", "fail", e); nfail += 1
                print(f"[{tag} FAIL] {split}: {type(e).__name__}: {e}", flush=True)
    # --- SVD Tier4 (monkeypatch module paths, reuse exact run_one) ---
    import run_q2_lgbm_tier4 as t4
    t4.DATA = Path(DATA); t4.OUT = Path(LGB_OUT)
    for split in SPLITS:
        rid = f"LightGBM_RDKit_species_svd_factor_{split}_s0"
        if (Path(LGB_OUT) / "predictions" / f"{rid}.csv").exists():
            nskip += 1; continue
        try:
            t4.run_one((split, 0)); ledger(rid, "ok"); nok += 1; print(f"[SVD ok] {split}", flush=True)
        except Exception as e:
            ledger(rid, "fail", e); nfail += 1; print(f"[SVD FAIL] {split}: {type(e).__name__}: {e}", flush=True)

    print(f"\n=== B1 deterministic: ok={nok} fail={nfail} skip={nskip} time={(time.time()-t0)/60:.1f}m ===", flush=True)


if __name__ == "__main__":
    main()
