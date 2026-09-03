"""Species-cold split (Phase 1 block B): 20% species holdout per tax_group, support-stratified.
cross-group EXCLUDED (§7 structural non-discriminability). Writes {partition}_species_cold_{train,test}.csv."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, r".")
from src.io_atomic import atomic_write_csv  # atomic replace for shared data CSVs
DATA = Path(r".\results\q2_v4\data")
ADORE = r"<ECOTOX_DATA_DIR>\processed\ecotox_mortality_processed.csv"
SEED = 42

mm = pd.read_csv(ADORE, usecols=["tax_gs", "tax_group"], low_memory=False).drop_duplicates("tax_gs")
mm["species"] = mm["tax_gs"].astype(str).str.strip().str.lower()
sp2grp = dict(zip(mm["species"], mm["tax_group"]))

out = {}
for part in ["discovery", "replication"]:
    pool = pd.concat([pd.read_csv(DATA / f"{part}_group_train.csv"),
                      pd.read_csv(DATA / f"{part}_group_test.csv")], ignore_index=True)
    pool["_grp"] = pool["species"].map(sp2grp)
    sup = pool.groupby("species").agg(n=("species", "size"), grp=("_grp", "first")).reset_index()
    rng = np.random.RandomState(SEED)
    test_sp = set()
    for g, sub in sup.groupby("grp"):
        s = sub.sort_values("n")
        for b in np.array_split(s.index.to_numpy(), min(4, len(s))):
            k = max(1, int(round(0.2 * len(b))))
            test_sp.update(sup.loc[rng.choice(b, min(k, len(b)), replace=False), "species"])
    te = pool[pool["species"].isin(test_sp)].drop(columns="_grp")
    tr = pool[~pool["species"].isin(test_sp)].drop(columns="_grp")
    atomic_write_csv(tr, DATA / f"{part}_species_cold_train.csv", encoding="utf-8")
    atomic_write_csv(te, DATA / f"{part}_species_cold_test.csv", encoding="utf-8")
    out[part] = {"train_rows": len(tr), "test_rows": len(te), "test_species": int(te["species"].nunique()),
                 "train_species": int(tr["species"].nunique()),
                 "test_species_all_cold": bool(len(set(te["species"]) & set(tr["species"])) == 0)}
print(json.dumps(out, ensure_ascii=False, indent=2))
