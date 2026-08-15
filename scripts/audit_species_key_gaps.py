"""Close the completeness-critic's 6 unmeasured gaps (Session 23). READ-ONLY measurement.
Every item is EXPECTED to pass; a failure would reveal a residual key/render mismatch."""
from __future__ import annotations
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
DATA = Path(r".\results\q2_v4\data")
GNN = Path(r".\results\q2_v4\runs\gnn\predictions")
LGB = Path(r".\results\q2_v4\runs\replication\lgbm\predictions")
KEY = ["smiles", "species", "endpoint", "duration"]
out = {}

# G1. ncbi_* stored nonnull directly in each of the 20 data CSVs (audit#2 measured key-intersection
#     vs the live NCBI file; this proves the values patch_ncbi_columns actually WROTE are present).
g1 = {}
for f in sorted(list(DATA.glob("*_train.csv")) + list(DATA.glob("*_test.csv"))):
    df = pd.read_csv(f, usecols=lambda c: c in ("ncbi_class", "ncbi_genus"))
    g1[f.name] = {"ncbi_class_nonnull": round(float(df["ncbi_class"].notna().mean()), 4),
                  "ncbi_genus_nonnull": round(float(df["ncbi_genus"].notna().mean()), 4)}
out["G1_ncbi_stored_nonnull"] = {"min_class_nonnull": round(min(v["ncbi_class_nonnull"] for v in g1.values()), 4),
                                 "all_ge_0.5": all(v["ncbi_class_nonnull"] >= 0.5 for v in g1.values()),
                                 "per_file": g1}

# G2. prediction-alignment inner-merge: GNN vs LightGBM no_species prediction CSVs on KEY, plus
#     the duration int/float render hazard. Expected inner ratio 1.00, identical duration dtype.
def _find(base, patt):
    hits = glob.glob(str(base / patt))
    return hits[0] if hits else None
g2 = {}
for split in ["discovery_group", "replication_group"]:
    gp = _find(GNN, f"dmpnn_no_species_{split}_s0_*nfull.csv")
    lp = _find(LGB, f"LightGBM_RDKit_no_species_{split}_s0.csv")
    if not gp or not lp:
        g2[split] = {"note": f"missing pred CSV (gnn={bool(gp)} lgb={bool(lp)})"}; continue
    G = pd.read_csv(gp); L = pd.read_csv(lp)
    keyG = [k for k in KEY if k in G.columns]; keyL = [k for k in KEY if k in L.columns]
    common = [k for k in KEY if k in G.columns and k in L.columns]
    merged = G[common].drop_duplicates().merge(L[common].drop_duplicates(), on=common, how="inner")
    g2[split] = {"gnn_cols_present": keyG, "lgb_cols_present": keyL, "merge_key": common,
                 "gnn_rows": len(G), "lgb_rows": len(L),
                 "inner_ratio_vs_min": round(len(merged) / max(1, min(G[common].drop_duplicates().shape[0],
                                                                       L[common].drop_duplicates().shape[0])), 4),
                 "duration_dtype_gnn": str(G["duration"].dtype) if "duration" in G else "absent",
                 "duration_dtype_lgb": str(L["duration"].dtype) if "duration" in L else "absent",
                 "duration_render_match": (str(G["duration"].dropna().iloc[0]) == str(L["duration"].dropna().iloc[0]))
                                          if ("duration" in G and "duration" in L and len(G) and len(L)) else None}
out["G2_prediction_alignment"] = g2

# G3. tier1' stratum coverage: every TEST endpoint@duration stratum present in TRAIN (else silent 0 effect).
def _strat(df): return (df["endpoint"].astype(str) + "@" + df["duration"].astype(str))
g3 = {}
for split in ["discovery_group", "replication_group", "replication_scaffold"]:
    tr = pd.read_csv(DATA / f"{split}_train.csv", usecols=["endpoint", "duration"])
    te = pd.read_csv(DATA / f"{split}_test.csv", usecols=["endpoint", "duration"])
    strain = set(_strat(tr)); stest = set(_strat(te))
    g3[split] = {"n_train_strata": len(strain), "n_test_strata": len(stest),
                 "test_subset_of_train_ratio": round(len(stest & strain) / max(1, len(stest)), 4)}
out["G3_tier1prime_stratum_coverage"] = g3

# G4. duration dtype/render consistency across train/test data CSVs (the stratum-key render class).
g4 = {}
for split in ["discovery_group", "replication_group"]:
    tr = pd.read_csv(DATA / f"{split}_train.csv", usecols=["duration"])
    te = pd.read_csv(DATA / f"{split}_test.csv", usecols=["duration"])
    g4[split] = {"dtype_train": str(tr["duration"].dtype), "dtype_test": str(te["duration"].dtype),
                 "dtype_match": str(tr["duration"].dtype) == str(te["duration"].dtype),
                 "sample_render_train": str(tr["duration"].dropna().iloc[0]),
                 "sample_render_test": str(te["duration"].dropna().iloc[0])}
out["G4_duration_dtype_render"] = g4

# G5. tier1' offset species-space: no_species GNN pred species_idx_original ⊆ data species_idx space.
g5 = {}
for split in ["discovery_group", "replication_group"]:
    gp = _find(GNN, f"dmpnn_no_species_{split}_s0_*nfull.csv")
    if not gp: g5[split] = {"note": "missing pred"}; continue
    P = pd.read_csv(gp)
    full = pd.concat([pd.read_csv(DATA / f"{split}_train.csv", usecols=["species_idx"]),
                      pd.read_csv(DATA / f"{split}_test.csv", usecols=["species_idx"])], ignore_index=True)
    space = set(int(x) for x in full["species_idx"].dropna().unique())
    col = "species_idx_original" if "species_idx_original" in P.columns else ("species_idx" if "species_idx" in P.columns else None)
    if col is None: g5[split] = {"note": "no species_idx col in pred"}; continue
    pids = set(int(x) for x in P[col].dropna().unique())
    g5[split] = {"pred_col": col, "pred_species": len(pids), "subset_of_data_space_ratio": round(len(pids & space) / max(1, len(pids)), 4)}
out["G5_tier1prime_species_space"] = g5

# G6. builder convention: live data species is underscored (=> build_adore canonical; legacy build_q2 is space-lc).
sp = pd.read_csv(DATA / "discovery_group_train.csv", usecols=["species"])["species"].dropna().astype(str)
out["G6_builder_convention"] = {"live_species_underscored_ratio": round(float(sp.str.contains("_").mean()), 4),
    "sample": sp.iloc[0], "note": "legacy build_q2_datasets uses space-lowercase 'Latin name' -> incompatible key+sort; pipeline build-stage now guarded."}

print(json.dumps(out, ensure_ascii=False, indent=2))
(DATA / "_ext" / "species_key_gap_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
