"""ADORE Q2 — dataset rebuild (step 3). STRATA aggregation, D16 exclusion, splits.

Loads ADORE mortality_processed, joins RDKit SMILES + MW, applies the D16 rule
(pre-registered: lower guardrail pLC50>-1.75 ; detection-limit mass<0.1 ng/L ;
path-3 per-compound), aggregates by (smiles, species, endpoint, duration), and
writes discovery (LC50@96h) / replication (LC50 24/48/72h + EC50) x
{group(compound-disjoint) / scaffold-murcko / scaffold-generic / designed-leaky}.

target_log10 = -log10(mean conc_mol) = pLC50 (sign verified §0-1).
Env: jcim_v3 (conda run). No training.
"""
from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# Root of the ADORE benchmark distribution (CC-BY 4.0), containing processed/ and
# chemicals/. Supply it via the ADORE_ROOT environment variable.
import os
ADORE = Path(os.environ.get("ADORE_ROOT", ""))
if not ADORE.name:
    raise SystemExit("Set ADORE_ROOT to the root of the ADORE benchmark distribution.")
MORT = ADORE / "processed" / "ecotox_mortality_processed.csv"
CHEM = ADORE / "chemicals" / "ecotox_properties_with-oecd-function.csv"
OUT = Path(r".\results\q2_v4\data")
OUT.mkdir(parents=True, exist_ok=True)

TAX4 = ["tax_class", "tax_order", "tax_family", "tax_genus"]
AGG_KEYS = ["smiles", "species", "endpoint", "duration"]
SEED, TEST_SIZE = 42, 0.2
DL_NG_L = 0.1                     # detection-limit floor (pre-reg §3)
PATH3_EXCLUDE = {"72-20-8", "298-00-0"}   # Endrin, Methyl parathion (source error)
PATH3_RETAIN = {"52918-63-5", "76703-62-3"}  # deltamethrin, pyrethroid (real)


# ---------------------------------------------------------------- load / D16
def load_records() -> pd.DataFrame:
    m = pd.read_csv(MORT, usecols=["test_cas", "tax_gs", "result_endpoint",
        "result_obs_duration_mean", "result_conc1_mean_mol"] + TAX4,
        dtype={"test_cas": str}, low_memory=False)
    chem = pd.read_csv(CHEM, usecols=["test_cas", "chem_rdkit_can_smiles", "chem_mw"],
        dtype={"test_cas": str}, low_memory=False).drop_duplicates("test_cas")
    m = m.merge(chem, on="test_cas", how="left")
    m["cas"] = m["test_cas"].astype(str).str.strip()
    m["smiles"] = m["chem_rdkit_can_smiles"].astype(str).str.strip()
    m["species"] = m["tax_gs"].astype(str).str.strip().str.lower()
    m["endpoint"] = m["result_endpoint"].astype(str).str.strip()
    m["duration"] = pd.to_numeric(m["result_obs_duration_mean"], errors="coerce")
    m["conc_mol"] = pd.to_numeric(m["result_conc1_mean_mol"], errors="coerce")
    m["mw"] = pd.to_numeric(m["chem_mw"], errors="coerce")
    m = m[m["conc_mol"] > 0].copy()
    m["pLC50"] = -np.log10(m["conc_mol"].to_numpy(float))
    m["mass_ng_L"] = m["conc_mol"].to_numpy(float) * m["mw"].to_numpy(float) * 1e9
    return m


def apply_d16(m: pd.DataFrame):
    """Return (kept, audit_excluded). Pre-registered rule."""
    def reason(r):
        if r.pLC50 <= -1.75: return "excluded:below_water_molarity"
        if r.mass_ng_L < DL_NG_L: return "excluded:below_detection_limit"
        if r.pLC50 >= 12 and r.cas in PATH3_EXCLUDE: return "excluded:source_conversion_error"
        return "retained"
    m = m.copy()
    m["exclusion_reason"] = [reason(r) for r in m.itertuples(index=False)]
    kept = m[m["exclusion_reason"] == "retained"].copy()
    excl = m[m["exclusion_reason"].str.startswith("excluded")].copy()
    return kept, excl


LIT = {  # path-3 / audit literature (per-compound)
 "72-20-8": ("Endrin", "88-352 ug/L (D. magna)", "waterquality.gov.au/.../endrin-2000", "~6",
             "organochlorine; recorded 0.13-0.28 ng/L vs literature 88-352 ug/L (~6 orders) -> conversion error"),
 "298-00-0": ("Methyl parathion", "~ug/L (OP)", "ECOTOX/lit", "~4",
             "organophosphate; recorded 0.14 ng/L vs literature ~ug/L (~4 orders) -> conversion error"),
 "52918-63-5": ("Deltamethrin", "2.6-68 ng/L (sensitive crustacea)", "PMC9996817", "~1",
             "pyrethroid; recorded 0.32 ng/L ~1 order below sensitive-crustacean LC50 -> plausible real, retained"),
 "76703-62-3": ("pyrethroid(cypermethrin-class)", "single-tens ng/L", "PMC9996817", "~1",
             "pyrethroid; recorded 0.42 ng/L within ~1 order of sensitive-crustacean LC50 -> plausible real, retained"),
}


def write_audit(excl: pd.DataFrame):
    rows = []
    for r in excl.itertuples(index=False):
        name, litrange, src, orders, logic = LIT.get(r.cas, ("", "", "", "", "below analytical detection limit ~0.1 ng/L (MDL 0.01-1.32 ng/L, molecules27061872) -> unmeasurable"))
        rows.append({"CAS": r.cas, "compound": name, "species": r.species,
            "mass_ng_L": round(float(r.mass_ng_L), 5), "pLC50": round(float(r.pLC50), 3),
            "literature_range": litrange, "literature_source": src, "orders_of_magnitude_diff": orders,
            "exclusion_reason": r.exclusion_reason.split(":")[1], "verdict_logic": logic})
    a = pd.DataFrame(rows).sort_values("pLC50")
    stamp = datetime.now(timezone.utc).isoformat()
    a.to_csv(OUT / "exclusion_audit_trail.csv", index=False, encoding="utf-8")
    (OUT / "exclusion_audit_trail_TIMESTAMP.txt").write_text(
        f"D16 exclusion applied (pre-training) at UTC {stamp}\n"
        f"rule: pLC50>-1.75 (guardrail) ; mass<{DL_NG_L} ng/L detection-limit ; path3 per-compound\n"
        f"excluded rows={len(a)} (below_detection_limit + source_conversion_error)\n", encoding="utf-8")
    return len(a), stamp


# ------------------------------------------------------------------- aggregate
def valid_smiles(df):
    ok = {s: (Chem.MolFromSmiles(s) is not None) for s in df["smiles"].unique()}
    return df[df["smiles"].map(ok)].copy()


def aggregate(df):
    g = df.groupby(AGG_KEYS, sort=True)
    out = g.agg(effect_value=("conc_mol", "mean"), n_source_rows=("conc_mol", "size"),
                n_cas=("cas", "nunique"),
                cas_list=("cas", lambda s: ";".join(sorted(set(map(str, s))))),
                tax_class=("tax_class", "first"), tax_order=("tax_order", "first"),
                tax_family=("tax_family", "first"), tax_genus=("tax_genus", "first")).reset_index()
    out["duration"] = out["duration"].astype(float).astype("Int64")
    out["target_log10"] = -np.log10(out["effect_value"].astype(float))   # pLC50
    return out


def discovery_mask(df): return df["endpoint"].eq("LC50") & df["duration"].eq(96.0)
def replication_mask(df):
    return (df["endpoint"].eq("LC50") & df["duration"].isin([24.0, 48.0, 72.0])) | df["endpoint"].eq("EC50")


# ---------------------------------------------------------------------- splits
_SC = {}
def scaffold_of(smi, generic=False):
    key = (smi, generic)
    if key in _SC: return _SC[key]
    mol = Chem.MolFromSmiles(smi); s = None
    if mol is not None:
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            if generic: scaf = MurckoScaffold.MakeScaffoldGeneric(scaf)
            s = Chem.MolToSmiles(scaf, isomericSmiles=(not generic))
        except Exception:
            s = None
    _SC[key] = s
    return s


def _greedy_group_split(df, keyseries):
    df = df.reset_index(drop=True)
    groups = defaultdict(list)
    for i, k in enumerate(keyseries):
        groups[k].append(i)
    sets = list(groups.values())
    rng = np.random.RandomState(SEED); rng.shuffle(sets)
    cutoff, cur, test_idx = int(len(df) * TEST_SIZE), 0, []
    train_idx = []
    for st in sets:
        if cur + len(st) <= cutoff:
            test_idx += st; cur += len(st)
        else:
            train_idx += st
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def group_split(df):        # compound-disjoint by SMILES (aggregation key)
    return _greedy_group_split(df, df["smiles"].to_numpy())
def scaffold_split(df, generic=False):
    return _greedy_group_split(df, df["smiles"].map(lambda s: scaffold_of(s, generic)).to_numpy())
def pairrandom_split(df):   # designed-leaky: (smiles,species) pair random
    df = df.reset_index(drop=True)
    pk = df["smiles"].astype(str) + "\x00" + df["species"].astype(str)
    pairs = pk.drop_duplicates().to_numpy()
    rng = np.random.RandomState(SEED); idx = rng.permutation(len(pairs))
    test_pairs = set(pairs[idx[:int(len(pairs) * TEST_SIZE)]])
    is_test = pk.isin(test_pairs).to_numpy()
    return df[~is_test].copy(), df[is_test].copy()


NCBI_COLS = ["ncbi_class", "ncbi_order", "ncbi_family", "ncbi_genus", "ncbi_resolved", "ncbi_taxid"]
def load_ncbi():
    p = OUT / "_ext" / "ncbi_taxonomy_by_species.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, dtype=str)

def _species_key(s):  # normalize join key: NCBI file uses spaces, our species uses underscores
    return s.astype(str).str.strip().str.lower().str.replace("_", " ", regex=False)

def attach_idx(df, sidx, ncbi):
    df = df.copy()
    df["species_idx"] = df["species"].map(sidx).astype("Int64")
    if ncbi is not None:
        n = ncbi.drop(columns=["species"]).copy()
        n["__spkey"] = _species_key(ncbi["species"])
        df["__spkey"] = _species_key(df["species"])
        df = df.merge(n, on="__spkey", how="left").drop(columns="__spkey")
    front = ["smiles", "species", "species_idx", "endpoint", "duration", "target_log10",
             "effect_value", "n_source_rows", "n_cas", "cas_list"] + TAX4
    return df[front + [c for c in df.columns if c not in front]]


def counts(df, name):
    return {"split": name, "rows": int(len(df)), "species": int(df["species"].nunique()),
            "compounds": int(df["smiles"].nunique()), "strata": int(len(df))}


def main():
    m = load_records()
    n_raw = len(m)
    kept, excl = apply_d16(m)
    n_excl, stamp = write_audit(excl)
    print(f"records: raw(conc>0)={n_raw}  excluded(D16)={n_excl}  kept={len(kept)}  ts={stamp}", flush=True)

    ncbi = load_ncbi()
    ledger = []
    for dataset, mask in [("discovery", discovery_mask), ("replication", replication_mask)]:
        pool = aggregate(valid_smiles(kept[mask(kept)]))
        sidx = {sp: i for i, sp in enumerate(sorted(pool["species"].unique()))}
        pool = attach_idx(pool, sidx, ncbi)
        ledger.append({**counts(pool, f"{dataset}_POOL"),
                       "records_used": int(kept[mask(kept)].shape[0])})
        for stem, (tr, te) in {
            f"{dataset}_group": group_split(pool),
            f"{dataset}_scaffold": scaffold_split(pool, generic=False),
            f"{dataset}_scaffold_generic": scaffold_split(pool, generic=True),
            f"{dataset}_designed_leaky": pairrandom_split(pool),
        }.items():
            tr.to_csv(OUT / f"{stem}_train.csv", index=False, encoding="utf-8")
            te.to_csv(OUT / f"{stem}_test.csv", index=False, encoding="utf-8")
            cold = set(te["species"]) - set(tr["species"])
            ledger.append({"split": stem, "train_rows": len(tr), "test_rows": len(te),
                "train_species": tr["species"].nunique(), "test_species": te["species"].nunique(),
                "train_compounds": tr["smiles"].nunique(), "test_compounds": te["smiles"].nunique(),
                "compound_overlap": len(set(tr["smiles"]) & set(te["smiles"])),
                "cold_species": len(cold),
                "cold_rows": int(te["species"].isin(cold).sum())})
    led = pd.DataFrame(ledger)
    led.to_csv(OUT / "data_provenance_ledger.csv", index=False, encoding="utf-8")
    print(led.to_string(index=False), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
