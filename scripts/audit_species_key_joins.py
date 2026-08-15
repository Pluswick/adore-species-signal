"""Central key-join audit (director task 1): measure the key-intersection ratio of every
cross-source / alignment-critical join in the ADORE pipeline, to prove no other join has the
same underscore-vs-space (or any key-mismatch) defect the NCBI join had. READ-ONLY.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, r".")
from jcim_v3.rdkit_lgbm import TAX_RANKS

ADORE = Path(r"<ECOTOX_DATA_DIR>")
MORT = ADORE / "processed" / "ecotox_mortality_processed.csv"
CHEM = ADORE / "chemicals" / "ecotox_properties_with-oecd-function.csv"
DATA = Path(r".\results\q2_v4\data")
NCBI = DATA / "_ext" / "ncbi_taxonomy_by_species.csv"
def spkey(s): return s.astype(str).str.strip().str.lower().str.replace("_", " ", regex=False)
WARM = [f"{p}_{s}" for p in ("discovery", "replication")
        for s in ("group", "scaffold", "scaffold_generic", "designed_leaky")]
COLD = ["discovery_species_cold", "replication_species_cold"]
res = {}

# 1. chem merge on test_cas (build_adore_datasets.py:46) — CAS key, cross-file.
m = pd.read_csv(MORT, usecols=["test_cas"], dtype={"test_cas": str}, low_memory=False)
chem = pd.read_csv(CHEM, usecols=["test_cas", "chem_rdkit_can_smiles"], dtype={"test_cas": str}, low_memory=False)
mcas = set(m["test_cas"].astype(str).str.strip()); ccas = set(chem["test_cas"].astype(str).str.strip())
chem_smiles = set(chem.dropna(subset=["chem_rdkit_can_smiles"])["test_cas"].astype(str).str.strip())
res["1_chem_merge_on_test_cas"] = {
    "key": "mortality.test_cas vs chem.test_cas", "left_n": len(mcas),
    "intersection_ratio": round(len(mcas & ccas) / len(mcas), 4),
    "with_nonnull_smiles_ratio": round(len(mcas & chem_smiles) / len(mcas), 4),
    "note": "left=distinct mortality CAS; some CAS legitimately lack an RDKit SMILES (dropped by D16)."}

# 2. NCBI merge on species (build_adore_datasets.py:186, FIXED) — normalized name key, per split.
ncbi = pd.read_csv(NCBI, dtype=str); nkeys = set(spkey(ncbi["species"]))
res["2_ncbi_merge_on_species"] = {}
for split in WARM + COLD:
    sp = set(pd.read_csv(DATA / f"{split}_train.csv", usecols=["species"])["species"].unique()) | \
         set(pd.read_csv(DATA / f"{split}_test.csv", usecols=["species"])["species"].unique())
    sk = set(spkey(pd.Series(list(sp))))
    res["2_ncbi_merge_on_species"][split] = {"data_species": len(sk),
        "intersection_ratio": round(len(sk & nkeys) / len(sk), 4)}

# 3. cold-split sp2grp mapping (build_adore_cold_splits.py:13,19) — tax_gs->tax_group onto pool.species.
mm = pd.read_csv(MORT, usecols=["tax_gs", "tax_group"], low_memory=False).drop_duplicates("tax_gs")
mm["species"] = mm["tax_gs"].astype(str).str.strip().str.lower()
sp2grp = dict(zip(mm["species"], mm["tax_group"]))
res["3_coldsplit_sp2grp"] = {}
for part in ("discovery", "replication"):
    pool = pd.concat([pd.read_csv(DATA / f"{part}_group_train.csv", usecols=["species"]),
                      pd.read_csv(DATA / f"{part}_group_test.csv", usecols=["species"])], ignore_index=True)
    pool_sp = set(pool["species"].unique())
    mapped = {s for s in pool_sp if s in sp2grp and pd.notna(sp2grp[s])}
    res["3_coldsplit_sp2grp"][part] = {"pool_species": len(pool_sp),
        "group_mapped_ratio": round(len(mapped) / len(pool_sp), 4),
        "note": "unmapped species would be silently dropped from stratified holdout; must be 1.0."}

# 4. species_idx integrity per split (build_adore_datasets.py:187 sidx map + runtime concat re-derivation).
res["4_species_idx_integrity"] = {}
for split in WARM + COLD:
    tr = pd.read_csv(DATA / f"{split}_train.csv", usecols=["species", "species_idx"])
    te = pd.read_csv(DATA / f"{split}_test.csv", usecols=["species", "species_idx"])
    full = pd.concat([tr, te], ignore_index=True)
    idx = full["species_idx"]; n = int(idx.max()) + 1
    uniq = set(int(x) for x in idx.dropna().unique())
    # each species_idx maps to exactly one species name and vice versa?
    one_to_one = full.dropna(subset=["species_idx"]).groupby("species_idx")["species"].nunique().max()
    name_to_idx = full.dropna(subset=["species_idx"]).groupby("species")["species_idx"].nunique().max()
    res["4_species_idx_integrity"][split] = {
        "n_species": n, "nonnull_ratio": round(float(idx.notna().mean()), 4),
        "contiguous_0_to_n": bool(uniq == set(range(n))),
        "idx_to_name_one_to_one": bool(one_to_one == 1), "name_to_idx_one_to_one": bool(name_to_idx == 1)}

# 5. block B train/cold species partition (run_q2_blockb_oov.py) — cold = test - train, on species_idx.
res["5_blockb_cold_partition"] = {}
for split in COLD:
    tr = pd.read_csv(DATA / f"{split}_train.csv", usecols=["species_idx"])
    te = pd.read_csv(DATA / f"{split}_test.csv", usecols=["species_idx"])
    trs = set(int(x) for x in tr["species_idx"].dropna().unique())
    tes = set(int(x) for x in te["species_idx"].dropna().unique())
    cold = tes - trs
    res["5_blockb_cold_partition"][split] = {"train_species": len(trs), "test_species": len(tes),
        "cold_species": len(cold), "all_test_cold_ratio": round(len(cold) / len(tes), 4)}

# 6. native tax_* provenance (build_adore_datasets.py: groupby-first from same mortality rows — internal).
res["6_native_tax_nonnull"] = {}
for split in ["discovery_group", "replication_group"]:
    full = pd.concat([pd.read_csv(DATA / f"{split}_train.csv"), pd.read_csv(DATA / f"{split}_test.csv")], ignore_index=True)
    res["6_native_tax_nonnull"][split] = {rk: round(float(full[rk].notna().mean()), 4)
                                          for rk in TAX_RANKS["taxonomy_original"]}

print(json.dumps(res, ensure_ascii=False, indent=2))
out = DATA / "_ext" / "species_key_join_audit.json"
out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[written] {out}")
