"""NCBI taxonomy resolution for B1 from the SAME taxdump Phase 1 used. Method-identity guarantee =
reproduce Phase 1's committed ncbi_taxonomy_by_species.csv. No training.

(Restored from the working copy; absolute paths replaced with repo-relative form. Run from repo
root. Requires the NCBI taxdump at ./results/q2_v4/data/_ext/taxdump.tar.gz and the Phase-1
committed ncbi_taxonomy_by_species.csv for the validation step.)"""
from __future__ import annotations
import tarfile, io, re
from pathlib import Path
import pandas as pd
D = Path(r".\results\q2_v4")
TAR = D / "data" / "_ext" / "taxdump.tar.gz"
RANKS4 = ["class", "order", "family", "genus"]

print("parse taxdump...")
sci = {}; syn = {}; name_of = {}
with tarfile.open(TAR, "r:gz") as t:
    for raw in io.TextIOWrapper(t.extractfile("names.dmp"), encoding="latin-1"):
        p = [x.strip() for x in raw.split("\t|")]
        taxid, name, ncls = p[0], p[1], p[3]
        if ncls == "scientific name":
            sci.setdefault(name.lower(), taxid); name_of[taxid] = name
        elif ncls in ("synonym", "equivalent name", "genbank synonym", "includes"):
            syn.setdefault(name.lower(), taxid)
    parent = {}; rank = {}
    for raw in io.TextIOWrapper(t.extractfile("nodes.dmp"), encoding="latin-1"):
        p = [x.strip() for x in raw.split("\t|")]
        parent[p[0]] = p[1]; rank[p[0]] = p[2]
print(f"  sci={len(sci)} syn={len(syn)} nodes={len(parent)}")


def resolve(name):
    key = re.sub(r"\s+", " ", str(name).strip().lower())
    tid = sci.get(key) or syn.get(key)
    if tid is None and " " in key:
        k2 = " ".join(key.split(" ")[:2]); tid = sci.get(k2) or syn.get(k2)
    if tid is None:
        return None
    out = {"ncbi_taxid": tid, "ncbi_resolved": True}
    cur = tid; seen = set()
    while cur and cur not in seen and cur != "1":
        seen.add(cur); r = rank.get(cur)
        if r in RANKS4 and f"ncbi_{r}" not in out:
            out[f"ncbi_{r}"] = name_of.get(cur)
        cur = parent.get(cur)
    for r in RANKS4:
        out.setdefault(f"ncbi_{r}", None)
    return out


# validate vs Phase 1 committed resolution (method-identity check)
ph = pd.read_csv(D / "data" / "_ext" / "ncbi_taxonomy_by_species.csv", dtype=str)
res_agree = res_tot = 0; agree = {r: 0 for r in RANKS4}; present = {r: 0 for r in RANKS4}; mism = []
for _, row in ph.iterrows():
    r = resolve(row["species"]); was = (row.get("ncbi_resolved") == "True")
    if was:
        res_tot += 1; res_agree += (1 if r else 0)
    if r and was:
        for rk in RANKS4:
            a = str(row.get(f"ncbi_{rk}") or ""); bb = str(r.get(f"ncbi_{rk}") or "")
            if a:
                present[rk] += 1; agree[rk] += (1 if a == bb else 0)
                if a != bb and len(mism) < 10:
                    mism.append((row["species"], rk, a, bb))
print("\n=== VALIDATION vs Phase 1 committed ncbi_taxonomy ===")
print(f"resolved-status agreement: {res_agree}/{res_tot} ({100*res_agree/max(res_tot,1):.1f}%)")
for rk in RANKS4:
    print(f"  {rk}: name agreement {agree[rk]}/{present[rk]} ({100*agree[rk]/max(present[rk],1):.1f}%)")
print("sample mismatches:", mism[:8])

# resolve B1 species
b = pd.concat([pd.read_csv(D / "data_b1" / f"b1_group_{s}.csv", usecols=["species"]) for s in ["train", "test"]])
bsp = sorted(set(b["species"])); rows = []
for sp in bsp:
    r = resolve(sp.replace("_", " "))
    rows.append({"species": sp, **(r if r else {"ncbi_taxid": None, "ncbi_resolved": False,
                                                 **{f"ncbi_{k}": None for k in RANKS4}})})
out = pd.DataFrame(rows); nres = int(out["ncbi_resolved"].sum())
print(f"\n=== B1 resolution: {nres}/{len(out)} species ({100*nres/len(out):.1f}%) ===")
for rk in RANKS4:
    nn = int(out[f"ncbi_{rk}"].notna().sum()); print(f"  ncbi_{rk} nonnull: {nn}/{len(out)} ({100*nn/len(out):.1f}%)")
out.to_csv(D / "data_b1" / "ncbi_taxonomy_by_species_b1.csv", index=False, encoding="utf-8")
print("[written] data_b1/ncbi_taxonomy_by_species_b1.csv")
