"""Build the tier-input reference JSON (expected n_species + per-rank cardinality per split)
from the CURRENT fixed data, then VERIFY the guard: (a) passes on fixed data for all tiers,
(b) HALTS on the pre-fix backup data for a taxonomy_ncbi tier (proves it would have caught the
NCBI-join bug structurally, not by RMSE luck)."""
from __future__ import annotations
import sys, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, r".")
from jcim_v3.tier_input_guard import check_tier_input, TierInputDegenerate, _rank_stats, _idx_stats
from jcim_v3.rdkit_lgbm import TAX_RANKS

DATA = Path(r".\results\q2_v4\data")
BK = DATA / "_backup_ncbifix"
REF = DATA / "tier_input_reference.json"
SPLITS = [f"{p}_{s}" for p in ("discovery", "replication")
          for s in ("group", "scaffold", "scaffold_generic", "designed_leaky", "species_cold")]

# ---- build reference from fixed data ----
ref = {}
for split in SPLITS:
    tr = pd.read_csv(DATA / f"{split}_train.csv"); te = pd.read_csv(DATA / f"{split}_test.csv")
    full = pd.concat([tr, te], ignore_index=True)
    idx = _idx_stats(full, tr)
    ranks = {}
    for inj in ("taxonomy_original", "taxonomy_ncbi"):
        ranks.update(_rank_stats(full, TAX_RANKS[inj]))
    ref[split] = {"n_species": idx["n_species"], "ranks": {k: {"cardinality": v["cardinality"]} for k, v in ranks.items()}}
REF.write_text(json.dumps(ref, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[reference] wrote {REF} for {len(ref)} splits")

# ---- (a) fixed data must PASS for representative variants across tiers ----
VARIANTS = ["no_species", "species_bias_only", "true_species_categorical",
            "true_species_taxonomy_original", "true_species_taxonomy_ncbi", "true_species_late_fusion"]
print("\n=== (a) guard on FIXED data (expect all ok) ===")
fail = 0
for split in ["discovery_group", "replication_scaffold", "discovery_species_cold"]:
    tr = pd.read_csv(DATA / f"{split}_train.csv"); te = pd.read_csv(DATA / f"{split}_test.csv")
    full = pd.concat([tr, te], ignore_index=True)
    for v in VARIANTS:
        rec = check_tier_input(v, split, tr, full, ref[split])
        flag = "OK " if not rec["degenerate"] else "DEGEN"
        if rec["degenerate"]:
            fail += 1
        print(f"  [{flag}] {split:26s} {v:32s} {rec['tier']:4s} " +
              (";".join(rec["reasons"]) if rec["degenerate"] else "clean"))
print(f"  fixed-data unexpected-degenerate = {fail} (must be 0)")

# ---- (b) pre-fix BACKUP data must HALT tier 3b ----
print("\n=== (b) guard on PRE-FIX backup data (expect t3b DEGEN, t3a OK) ===")
tr = pd.read_csv(BK / "discovery_group_train.csv"); te = pd.read_csv(BK / "discovery_group_test.csv")
full = pd.concat([tr, te], ignore_index=True)
for v in ["true_species_taxonomy_original", "true_species_taxonomy_ncbi"]:
    rec = check_tier_input(v, "discovery_group", tr, full, ref["discovery_group"])
    print(f"  [{'DEGEN' if rec['degenerate'] else 'OK '}] {v:32s} {rec['tier']:4s} " +
          (";".join(rec["reasons"]) if rec["degenerate"] else "clean"))
    if v.endswith("taxonomy_ncbi") and not rec["degenerate"]:
        print("  !!! FAIL: guard did NOT catch the pre-fix NCBI degeneracy"); sys.exit(1)
    if v.endswith("taxonomy_original") and rec["degenerate"]:
        print("  !!! FAIL: guard false-positive on healthy native taxonomy"); sys.exit(1)
print("\n=== guard verification PASSED ===")
