"""Q2 v4 Task A — partition-preserving dataset rebuild (STRATA aggregation).

Rebuilds discovery (LC50 96h) and replication (LC50 24/48/72h + mortality EC50)
from the vendored tox-learn group-split CSVs WITHOUT merging the two Yuan
partitions. Aggregation is (canonical SMILES, species, endpoint, duration) --
strata-preserving (SPEC 6) so the species term cannot absorb endpoint identity.
For discovery (single LC50/96h stratum) this is identical to (smiles, species).

Carries original taxonomy (as-is, no backfill) and joins the self-generated
ncbi_filled taxonomy by species. Produces Yuan-group, scaffold, and
designed-leaky splits + a provenance ledger.

Env: jcim_v3 (run via `conda run -n jcim_v3` -- direct invocation crashes in BLAS).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(r".\results\q2_v4")
VENDOR = ROOT / "vendor" / "toxlearn"
DERIVED = VENDOR / "derived"
OUT = ROOT / "data"
OUT.mkdir(parents=True, exist_ok=True)

ORIG_TAX = ["Taxonomic kingdom", "Taxonomic phylum or division", "Taxonomic subphylum",
            "Taxonomic class", "Taxonomic order", "Taxonomic family", "Taxonomic superclass"]
NCBI_RANKS = ["Taxonomic kingdom", "Taxonomic phylum or division", "Taxonomic subphylum",
              "Taxonomic class", "Taxonomic order", "Taxonomic family"]
USECOLS = ["CAS", "Latin name", "Duration (hours)", "Effect value", "Test statistic",
           "Canonical SMILES", "Original CAS"] + ORIG_TAX
AGG_KEYS = ["smiles", "species", "endpoint", "duration"]
SEED, TEST_SIZE = 42, 0.2


# ---------------------------------------------------------------- load / filter
def load_partition(name: str) -> pd.DataFrame:
    df = pd.read_csv(VENDOR / name, usecols=USECOLS, dtype=str)
    df["CAS"] = df["CAS"].astype(str).str.strip()
    df["smiles"] = df["Canonical SMILES"].astype(str).str.strip()
    df["species"] = df["Latin name"].astype(str).str.strip().str.lower()
    df["endpoint"] = df["Test statistic"].astype(str).str.strip()
    df["duration"] = pd.to_numeric(df["Duration (hours)"], errors="coerce")
    df["effect"] = pd.to_numeric(df["Effect value"], errors="coerce")
    return df


def discovery_mask(df):
    return df["endpoint"].eq("LC50") & df["duration"].eq(96.0) & df["effect"].gt(0)


def replication_mask(df):
    lc = df["endpoint"].eq("LC50") & df["duration"].isin([24.0, 48.0, 72.0])
    ec = df["endpoint"].eq("EC50")
    return (lc | ec) & df["effect"].gt(0)


def valid_smiles(df: pd.DataFrame) -> pd.DataFrame:
    uniq = df["smiles"].unique()
    ok = {s: (Chem.MolFromSmiles(s) is not None) for s in uniq}
    return df[df["smiles"].map(ok)].copy()


# ------------------------------------------------------------------- aggregate
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Within-partition STRATA aggregate by (smiles, species, endpoint, duration)."""
    g = df.groupby(AGG_KEYS, sort=True)
    out = g.agg(
        effect_value=("effect", "mean"),
        n_source_rows=("effect", "size"),
        n_cas=("CAS", "nunique"),
        cas_list=("CAS", lambda s: ";".join(sorted(set(map(str, s))))),
    ).reset_index()
    out["duration"] = out["duration"].astype(float).astype("Int64")
    out["target_log10"] = np.log10(out["effect_value"].astype(float))
    return out


def species_original_taxonomy(raw_all: pd.DataFrame) -> pd.DataFrame:
    t = raw_all[["species"] + ORIG_TAX].copy()
    for c in ORIG_TAX:
        t[c] = t[c].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return t.groupby("species", sort=True).first().reset_index()


def load_ncbi() -> pd.DataFrame:
    n = pd.read_csv(DERIVED / "taxonomy_ncbi_filled.csv")
    n["species"] = n["Latin name"].astype(str).str.strip().str.lower()
    ren = {r: f"ncbi_{r.split()[-1].lower()}" for r in NCBI_RANKS}
    n = n.rename(columns=ren)
    return n[["species", "ncbi_resolved", "ncbi_taxid"] + list(ren.values())]


# ---------------------------------------------------------------------- splits
_SCAF_CACHE: dict[str, str | None] = {}


def scaffold_of(smiles: str):
    if smiles in _SCAF_CACHE:
        return _SCAF_CACHE[smiles]
    mol = Chem.MolFromSmiles(smiles)
    s = None if mol is None else Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=True)
    _SCAF_CACHE[smiles] = s
    return s


def scaffold_split(df: pd.DataFrame):
    """Bemis-Murcko greedy; all strata of a SMILES share a scaffold -> same side."""
    df = df.reset_index(drop=True)
    scafs = df["smiles"].map(scaffold_of)
    groups = defaultdict(list)
    for i, s in enumerate(scafs):
        groups[s].append(i)
    sets = list(groups.values())
    rng = np.random.RandomState(SEED)
    rng.shuffle(sets)
    cutoff, cur, test_idx, train_idx = int(len(df) * TEST_SIZE), 0, [], []
    for st in sets:
        if cur + len(st) <= cutoff:
            test_idx += st
            cur += len(st)
        else:
            train_idx += st
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def pairrandom_split(df: pd.DataFrame):
    """designed-leaky: random split at the (smiles, species) PAIR level (SPEC 6).

    All strata of a pair stay on one side, so the injected leak stays
    COMPOUND-level (same SMILES via a different species). Splitting strata of the
    same pair across train/test would be a far stronger pair-level leak and would
    change what H3 measures.
    """
    df = df.reset_index(drop=True)
    pk = df["smiles"].astype(str) + "\x00" + df["species"].astype(str)
    pairs = pk.drop_duplicates().to_numpy()
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(pairs))
    test_pairs = set(pairs[idx[:int(len(pairs) * TEST_SIZE)]])
    is_test = pk.isin(test_pairs).to_numpy()
    return df[~is_test].copy(), df[is_test].copy()


# --------------------------------------------------------------------- metrics
def cas_set(frame: pd.DataFrame) -> set:
    out = set()
    for cl in frame["cas_list"].astype(str):
        out.update(x for x in cl.split(";") if x and x.lower() != "nan")
    return out


def split_metrics(name: str, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    tr_sp, te_sp = set(train["species"]), set(test["species"])
    cold = te_sp - tr_sp
    cold_rows = int(test["species"].isin(cold).sum())
    tr_sm, te_sm = set(train["smiles"]), set(test["smiles"])
    tr_pair = set(map(tuple, train[["smiles", "species"]].values))
    te_pair = set(map(tuple, test[["smiles", "species"]].values))
    te_res = test["ncbi_resolved"].fillna(False).astype(bool)
    cold_mask = test["species"].isin(cold)
    trc = train.groupby("species").size()
    b = test["species"].map(trc).fillna(0).astype(int)
    rare = (b >= 2) & (b <= 9)
    return {
        "split": name,
        "train_rows": len(train), "test_rows": len(test),
        "train_pairs": len(tr_pair), "test_pairs": len(te_pair),
        "train_compounds": train["smiles"].nunique(), "test_compounds": test["smiles"].nunique(),
        "train_species": len(tr_sp), "test_species": len(te_sp), "shared_species": len(tr_sp & te_sp),
        "cold_species": len(cold), "cold_rows": cold_rows,
        "rare2_9_rows": int(rare.sum()), "rare2_9_species": int(test.loc[rare, "species"].nunique()),
        "test_train_ratio": round(len(test) / max(len(train), 1), 4),
        "CAS_overlap": len(cas_set(train) & cas_set(test)),
        "SMILES_overlap": len(tr_sm & te_sm),
        "pair_overlap": len(tr_pair & te_pair),
        "test_ncbi_resolved_species": int(test.loc[te_res, "species"].nunique()),
        "test_ncbi_resolved_rows": int(te_res.sum()),
        "cold_ncbi_resolved_rows": int((cold_mask & te_res).sum()),
    }


def attach(df: pd.DataFrame, sidx: dict, otax: pd.DataFrame, ntax: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(otax, on="species", how="left").merge(ntax, on="species", how="left")
    df["species_idx"] = df["species"].map(sidx).astype("Int64")
    df["ncbi_resolved"] = df["ncbi_resolved"].fillna(False)
    front = ["smiles", "species", "species_idx", "endpoint", "duration", "target_log10",
             "effect_value", "n_source_rows", "n_cas", "cas_list"]
    return df[front + [c for c in df.columns if c not in front]]


def write(train, test, stem):
    train.to_csv(OUT / f"{stem}_train.csv", index=False, encoding="utf-8")
    test.to_csv(OUT / f"{stem}_test.csv", index=False, encoding="utf-8")


# ------------------------------------------------------------------------ main
def main():
    print("loading vendored partitions...", flush=True)
    raw_tr, raw_te = load_partition("groupsplit_train.csv"), load_partition("groupsplit_test.csv")
    raw_all = pd.concat([raw_tr, raw_te], ignore_index=True)
    otax = species_original_taxonomy(raw_all)
    ntax = load_ncbi()

    ledger, dropped_report = [], {}
    for dataset, mask in [("discovery", discovery_mask), ("replication", replication_mask)]:
        tr = aggregate(valid_smiles(raw_tr[mask(raw_tr)]))
        te = aggregate(valid_smiles(raw_te[mask(raw_te)]))
        pool = aggregate(valid_smiles(raw_all[mask(raw_all)]))
        sidx = {sp: i for i, sp in enumerate(sorted(pool["species"].unique()))}
        tr, te, pool = (attach(x, sidx, otax, ntax) for x in (tr, te, pool))

        write(tr, te, f"{dataset}_group")
        s_tr, s_te = scaffold_split(pool)
        write(s_tr, s_te, f"{dataset}_scaffold")
        l_tr, l_te = pairrandom_split(pool)
        write(l_tr, l_te, f"{dataset}_designed_leaky")

        ledger.append(split_metrics(f"{dataset}_group", tr, te))
        ledger.append(split_metrics(f"{dataset}_scaffold", s_tr, s_te))
        ledger.append(split_metrics(f"{dataset}_designed_leaky", l_tr, l_te))

        raw_cas = set(raw_all["CAS"]) - {"", "nan"}
        kept_cas = cas_set(pool)
        dropped_report[dataset] = {"raw_cas_groups": len(raw_cas), "kept_cas_groups": len(kept_cas),
                                   "dropped_cas_groups": len(raw_cas - kept_cas)}

    led = pd.DataFrame(ledger)
    led.to_csv(OUT / "data_provenance_ledger.csv", index=False, encoding="utf-8")

    def df_to_md(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        head = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
        return "\n".join([head, sep] + body)

    lines = ["# Q2 v4 — data_provenance_ledger (STRATA aggregation)", "",
             "생성: build_q2_datasets.py (env jcim_v3). 집계 단위 = (smiles, species, endpoint, duration).",
             "discovery는 단일 stratum이라 (smiles, species) 집계와 항등. raw 무변경.", "",
             "designed_leaky = (smiles,species) pair 단위 무작위 분할(한 pair의 모든 strata는 같은 쪽).", "",
             "## Split metrics", "", df_to_md(led), "", "## CAS groups dropped by filter", ""]
    for d, r in dropped_report.items():
        lines.append(f"- **{d}**: raw CAS={r['raw_cas_groups']} · kept={r['kept_cas_groups']} · "
                     f"dropped={r['dropped_cas_groups']}")
    (OUT / "data_provenance_ledger.md").write_text("\n".join(lines), encoding="utf-8")

    print(led.to_string(index=False), flush=True)
    print("\nDropped CAS groups:", json.dumps(dropped_report, ensure_ascii=False), flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
