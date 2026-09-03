"""Regression: single-source-of-truth for aggregation metadata (director task 4, Session 26).

Take a prediction CSV written BEFORE the Aug-1 NCBI fix (carries NULL ncbi_*) and show:
  (WRONG, old pattern) stratifying species by ncbi_class READ FROM the prediction CSV -> every
      species collapses to an 'unresolved' stratum (silent misclassification).
  (RIGHT, new rule)     load pred via load_prediction_csv (whitelist), join ncbi_class from the
      DATASET by species -> correct taxonomy stratum for each species.
  (GUARD)               load_prediction_csv(..., columns=[...,'ncbi_class']) RAISES, so the WRONG
      pattern is now structurally impossible.
Env: conda run -n src.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, r".")
from src.prediction_io import load_prediction_csv, PredictionColumnViolation

DATA = Path(r".\results\q2_v4\data")
PRED = Path(r".\results\q2_v4\runs\gnn\predictions")
SPLIT = "discovery_group"
# a cell that ran before the NCBI fix -> its frozen CSV still carries null ncbi_*
CELL = PRED / f"dmpnn_true_species_taxonomy_original_{SPLIT}_s7_e100_nfull.csv"

raw = pd.read_csv(CELL)
assert "ncbi_class" in raw.columns, "expected ncbi_class column in the prediction CSV"
pre_fix = raw["ncbi_class"].isna().all()
print(f"chosen pred CSV pre-fix (ncbi_class all-null): {pre_fix}")
if not pre_fix:
    print("!! chosen cell is not pre-fix; pick another. (Test still runs but is less illustrative.)")

# ---- WRONG: stratify by ncbi_class read from the prediction CSV ----
wrong = raw[["species", "ncbi_class"]].drop_duplicates("species").copy()
wrong["stratum"] = wrong["ncbi_class"].fillna("__unresolved__")
n_unresolved_wrong = int((wrong["stratum"] == "__unresolved__").sum())
n_species = len(wrong)

# ---- RIGHT: pred via whitelist loader, taxonomy joined from the dataset by species ----
pred = load_prediction_csv(CELL, columns=["species", "pred_log10", "true_log10"])
ds = pd.concat([pd.read_csv(DATA / f"{SPLIT}_train.csv", usecols=["species", "ncbi_class"]),
                pd.read_csv(DATA / f"{SPLIT}_test.csv", usecols=["species", "ncbi_class"])],
               ignore_index=True).drop_duplicates("species")
right = pred[["species"]].drop_duplicates("species").merge(ds, on="species", how="left")
right["stratum"] = right["ncbi_class"].fillna("__unresolved__")
n_unresolved_right = int((right["stratum"] == "__unresolved__").sum())

# ---- GUARD: the wrong pattern is now impossible ----
guard_raises = False
try:
    load_prediction_csv(CELL, columns=["species", "ncbi_class"])
except PredictionColumnViolation:
    guard_raises = True

print(f"\nspecies in cell: {n_species}")
print(f"WRONG (ncbi_class from pred CSV): unresolved = {n_unresolved_wrong}/{n_species}")
print(f"RIGHT (ncbi_class from dataset) : unresolved = {n_unresolved_right}/{n_species}")
print(f"GUARD raises on reading ncbi_class from pred CSV: {guard_raises}")

# Regression assertions
ok = True
if pre_fix and n_unresolved_wrong != n_species:
    print("FAIL: pre-fix WRONG path did not misclassify all species"); ok = False
if n_unresolved_right >= n_unresolved_wrong and pre_fix:
    print("FAIL: RIGHT path did not fix the misclassification"); ok = False
if not guard_raises:
    print("FAIL: guard did not block reading ncbi_class from the prediction CSV"); ok = False
misclassified_fixed = n_unresolved_wrong - n_unresolved_right
print(f"\n=== REGRESSION {'PASS' if ok else 'FAIL'}: dataset-sourced path recovers {misclassified_fixed} "
      f"species that the pred-CSV path misclassified as unresolved; guard blocks the wrong pattern ===")
sys.exit(0 if ok else 2)
