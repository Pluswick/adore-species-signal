"""B1 dataset build (independent replication corpus).

⚠ RECONSTRUCTED, not the original. The working build_b1.py was developed in a temporary
directory that was later cleared. This is a faithful re-implementation of the DOCUMENTED recipe
(results/q2_v4/audit/ecotox_expansion_v2.md §1) on top of the Phase-1 build template
(build_adore_datasets.py: D16 exclusion, aggregation, splits, index/NCBI attach), which are
imported and reused UNCHANGED so the tier-input construction is identical to Phase 1.

Recipe (ecotox_expansion_v2.md §1):
  corrected E-full = 2026 ECOTOX pull, rows with
      endpoint ∈ ENDPOINTS_11  ∧  effect ∈ EFFECTS_6  ∧  habitat = "Water"  ∧  media ∈ MEDIA_P
      (all taxa; species.txt join for taxonomy), then duration ≤ 96 h, conc_mol > 0, valid SMILES.
  B1_final = corrected E-full  MINUS P  (by result_id)  MINUS precise (reference,species,CAS,duration)
             duplicates of P.   -> ~41,523 kept records -> 22,809 strata after aggregation.
  Then: global species_idx over B1 species (1,975), attach NCBI (ncbi_resolve.py output), and the
  SAME four splits as Phase 1 but on a SINGLE corpus (no discovery/replication axis):
      b1_group (compound-disjoint) / b1_scaffold (Murcko) / b1_scaffold_generic / b1_designed_leaky.

TO VERIFY before running in your environment (author-specific, not recoverable from the repo):
  * RAW_ECOTOX  : path + column names of the 2026 ECOTOX ASCII pull (ecotox_ascii_06_11_2026).
  * MEDIA_P     : the exact media-code set observed in P (below is the documented top set).
  * P_RESULT_IDS / P_PRECISE_KEYS : the result_id set and (reference,species,CAS,duration) key set
                  of the Phase-1 training corpus P, used for the disjointness subtraction.
Self-validation at the end asserts the split-count table matches data_b1/data_provenance_ledger.csv.

Env: jcim_v3 (conda run). No training.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
# reuse Phase-1 verified building blocks UNCHANGED
from build_adore_datasets import (apply_d16, aggregate, valid_smiles,
                                  group_split, scaffold_split, pairrandom_split, TAX4, SEED, TEST_SIZE)

ROOT = Path(r".\results\q2_v4")
OUT = ROOT / "data_b1"; OUT.mkdir(parents=True, exist_ok=True)

# ---- documented P-derived filter sets (ecotox_expansion_v2.md §1) ----
ENDPOINTS_11 = {"LC50", "EC50", "LC50*", "LC50/", "EC50*", "EC50/", "LD50/", "LC50*/", "EC50*/", "IC50", "EC0"}
EFFECTS_6 = {"MOR", "ITX", "POP", "GRO", "PHY", "~MOR"}
HABITAT = "Water"
MEDIA_P = {"FW", "SW", "FW/", "SW/", "CUL/"}  # documented top set; = P's observed media codes (verify)
MAX_DURATION_H = 96.0

# ---- author-specific inputs (see header) ----
RAW_ECOTOX = Path(r"<DATA_ROOT>\ecotox_ascii_06_11_2026")   # 2026 ECOTOX ASCII pull
CHEM = Path(r"<DATA_ROOT>\adore_dataset\chemicals\ecotox_properties_with-oecd-function.csv")
NCBI_B1 = OUT / "ncbi_taxonomy_by_species_b1.csv"           # produced by ncbi_resolve.py


def load_corrected_efull():
    """Load 2026 ECOTOX, apply the P-derived filters + duration + SMILES + conc.
    Column names below follow ecotox_mortality_processed.csv; adapt to the raw ASCII pull if needed."""
    # NOTE: adapt reader to the actual ECOTOX ASCII layout (pipe-delimited tests.txt/results.txt +
    # species.txt join). Columns used: result_id, reference_number, test_cas, tax_gs, result_endpoint,
    # result_effect, result_obs_duration_mean, result_conc1_mean_mol, test_media_type, organism_habitat.
    m = pd.read_csv(RAW_ECOTOX / "ecotox_processed.csv", dtype={"test_cas": str}, low_memory=False)
    chem = pd.read_csv(CHEM, usecols=["test_cas", "chem_rdkit_can_smiles", "chem_mw"],
                       dtype={"test_cas": str}, low_memory=False).drop_duplicates("test_cas")
    m = m.merge(chem, on="test_cas", how="left")
    m["endpoint"] = m["result_endpoint"].astype(str).str.strip()
    m["effect"] = m["result_effect"].astype(str).str.strip()
    m["habitat"] = m["organism_habitat"].astype(str).str.strip()
    m["media"] = m["test_media_type"].astype(str).str.strip()
    m["duration"] = pd.to_numeric(m["result_obs_duration_mean"], errors="coerce")
    m["conc_mol"] = pd.to_numeric(m["result_conc1_mean_mol"], errors="coerce")
    m["mw"] = pd.to_numeric(m["chem_mw"], errors="coerce")
    m["cas"] = m["test_cas"].astype(str).str.strip()
    m["smiles"] = m["chem_rdkit_can_smiles"].astype(str).str.strip()
    m["species"] = m["tax_gs"].astype(str).str.strip().str.lower()
    for c in TAX4:
        if c not in m.columns:  # taxonomy from species.txt join in the raw pull
            m[c] = m.get(c)
    # P-derived filters
    m = m[m["endpoint"].isin(ENDPOINTS_11) & m["effect"].isin(EFFECTS_6)
          & m["habitat"].eq(HABITAT) & m["media"].isin(MEDIA_P)]
    m = m[(m["conc_mol"] > 0) & (m["duration"] <= MAX_DURATION_H)].copy()
    m["pLC50"] = -np.log10(m["conc_mol"].to_numpy(float))
    m["mass_ng_L"] = m["conc_mol"].to_numpy(float) * m["mw"].to_numpy(float) * 1e9
    return m


def subtract_P(m):
    """B1_final = corrected E-full MINUS P (by result_id) MINUS precise (reference,species,CAS,duration)
    P-duplicates. P_RESULT_IDS / P_PRECISE_KEYS come from the Phase-1 corpus P (author-provided)."""
    p_ids_path = OUT / "_ext" / "P_result_ids.txt"
    p_keys_path = OUT / "_ext" / "P_precise_keys.txt"
    P_RESULT_IDS = set(p_ids_path.read_text().split()) if p_ids_path.exists() else set()
    P_PRECISE_KEYS = set(p_keys_path.read_text().splitlines()) if p_keys_path.exists() else set()
    if P_RESULT_IDS:
        m = m[~m["result_id"].astype(str).isin(P_RESULT_IDS)]
    pk = (m["reference_number"].astype(str) + "|" + m["species"] + "|" + m["cas"] + "|"
          + m["duration"].astype(str))
    if P_PRECISE_KEYS:
        m = m[~pk.isin(P_PRECISE_KEYS)]
    return m.copy()


def main():
    m = subtract_P(load_corrected_efull())
    kept, _excl = apply_d16(m)
    print(f"B1 corrected E-full (post P-subtraction, conc>0): {len(m)}  D16-kept: {len(kept)}", flush=True)

    pool = aggregate(valid_smiles(kept))
    sidx = {sp: i for i, sp in enumerate(sorted(pool["species"].unique()))}  # global species_idx
    pool["species_idx"] = pool["species"].map(sidx).astype("Int64")
    ncbi = pd.read_csv(NCBI_B1, dtype=str) if NCBI_B1.exists() else None
    if ncbi is not None:
        n = ncbi.drop(columns=["species"]).copy()
        n["__k"] = ncbi["species"].astype(str).str.strip().str.lower().str.replace("_", " ", regex=False)
        pool["__k"] = pool["species"].astype(str).str.strip().str.lower().str.replace("_", " ", regex=False)
        pool = pool.merge(n, on="__k", how="left").drop(columns="__k")
    front = ["smiles", "species", "species_idx", "endpoint", "duration", "target_log10",
             "effect_value", "n_source_rows", "n_cas", "cas_list"] + TAX4
    pool = pool[front + [c for c in pool.columns if c not in front]]

    ledger = []
    for stem, (tr, te) in {
        "b1_group": group_split(pool),
        "b1_scaffold": scaffold_split(pool, generic=False),
        "b1_scaffold_generic": scaffold_split(pool, generic=True),
        "b1_designed_leaky": pairrandom_split(pool),
    }.items():
        tr.to_csv(OUT / f"{stem}_train.csv", index=False, encoding="utf-8")
        te.to_csv(OUT / f"{stem}_test.csv", index=False, encoding="utf-8")
        cold = set(te["species"]) - set(tr["species"])
        ledger.append({"split": stem, "train_rows": len(tr), "test_rows": len(te),
                       "train_species": tr["species"].nunique(), "test_species": te["species"].nunique(),
                       "train_compounds": tr["smiles"].nunique(), "test_compounds": te["smiles"].nunique(),
                       "compound_overlap": len(set(tr["smiles"]) & set(te["smiles"])),
                       "cold_species": len(cold), "cold_rows": int(te["species"].isin(cold).sum())})
    led = pd.DataFrame(ledger); print(led.to_string(index=False), flush=True)

    # ---- self-validation vs committed provenance ledger ----
    ref_path = OUT / "data_provenance_ledger.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        merged = led.merge(ref, on="split", suffixes=("_new", "_ref"))
        ok = (merged["train_rows_new"].eq(merged["train_rows_ref"]).all()
              and merged["test_rows_new"].eq(merged["test_rows_ref"]).all())
        print(f"\n[self-validation] split counts match committed data_provenance_ledger.csv: {ok}")
        if not ok:
            print("  -> mismatch: verify RAW_ECOTOX columns / MEDIA_P / P_RESULT_IDS / P_PRECISE_KEYS.")


if __name__ == "__main__":
    main()
