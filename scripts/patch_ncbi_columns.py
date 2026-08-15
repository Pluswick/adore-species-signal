"""Surgical fix for the NCBI-join key bug: fill the all-null ncbi_* columns in the
20 existing data CSVs IN PLACE. Non-NCBI columns are verified byte-identical (round-trip
safe) so completed/in-flight block A runs (which never read ncbi_*) are unaffected.
Backup first. NCBI file uses spaces; our species uses underscores -> normalized join key."""
from __future__ import annotations
import sys, shutil, hashlib
from pathlib import Path
import pandas as pd
sys.path.insert(0, r".")
from jcim_v3.io_atomic import atomic_write_csv  # in-flight data writes must be atomic (temp+fsync+rename)

D = Path(r".\results\q2_v4\data")
BK = D / "_backup_ncbifix"
NCBI_COLS = ["ncbi_class", "ncbi_order", "ncbi_family", "ncbi_genus", "ncbi_resolved", "ncbi_taxid"]

def key(s): return s.astype(str).str.strip().str.lower().str.replace("_", " ", regex=False)

ncbi = pd.read_csv(D / "_ext" / "ncbi_taxonomy_by_species.csv", dtype=str)
n = ncbi.drop(columns=["species"]).copy(); n["__spkey"] = key(ncbi["species"])

BK.mkdir(exist_ok=True)
files = sorted(list(D.glob("*_train.csv")) + list(D.glob("*_test.csv")))
def col_hash(df, cols):
    return hashlib.sha256(df[cols].to_csv(index=False).encode()).hexdigest()[:16]

n_ok = n_fail = 0
for f in files:
    df = pd.read_csv(f)
    non_ncbi = [c for c in df.columns if c not in NCBI_COLS]
    order = list(df.columns)
    h_before = col_hash(df, non_ncbi)
    # backup raw file
    shutil.copy2(f, BK / f.name)
    merged = df[non_ncbi].copy()
    merged["__spkey"] = key(merged["species"])
    merged = merged.merge(n, on="__spkey", how="left").drop(columns="__spkey")
    merged = merged[order]  # restore exact original column order
    h_after = col_hash(merged, non_ncbi)
    if h_before != h_after:
        print(f"[ABORT] {f.name}: non-ncbi columns changed! {h_before}!={h_after}", flush=True)
        n_fail += 1; continue
    nn = merged["ncbi_class"].notna().sum()
    atomic_write_csv(merged, f, encoding="utf-8")   # atomic replace (no partial-write exposure)
    print(f"[ok] {f.name:44s} rows={len(merged)} ncbi_class_nonnull={nn}/{len(merged)} non_ncbi_hash={h_before}", flush=True)
    n_ok += 1
print(f"\n=== patched ok={n_ok} fail={n_fail}; backups in {BK} ===", flush=True)
