"""Coverage across candidate entry thresholds, ALL-SPECIES basis as the primary denominator.

Primary denominator = all species in the dataset (train u test). Rationale: the practical
question is "does the species I want to predict have enough training data", and that
population is the whole database, not the evaluation subset. Test-species basis is
reported alongside as a footnote figure, always labelled.

Thresholds reported: >=1, >=2, >=3, >=6, >=10, >=50.
  - 3 = lowest bin where dd is negative in 4/4 (sign-based lower bound)
  - 6 = lowest bin where the bootstrap CI excludes 0 in 4/4 (CI-based lower bound)

Env: conda run -n src.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(r".\results\q2_v4\data")
OUT = Path(r".\results\q2_v4\cost")
THRESHOLDS = [1, 2, 3, 6, 10, 50]

rows = []
for split in ("replication_group", "discovery_group"):
    tr = pd.read_csv(DATA / f"{split}_train.csv", usecols=["species"])
    te = pd.read_csv(DATA / f"{split}_test.csv", usecols=["species"])
    cnt = tr.groupby("species").size()

    all_sp = sorted(set(tr["species"]) | set(te["species"]))
    sup_all = pd.Series({s: int(cnt.get(s, 0)) for s in all_sp})
    n_all = len(sup_all)
    total_train = int(sup_all.sum())

    te_sp = te["species"]
    n_te_sp = te_sp.nunique()
    n_te_rows = len(te_sp)
    sup_te_rows = te_sp.map(cnt).fillna(0).astype(int)

    for thr in THRESHOLDS:
        ok_all = sup_all >= thr
        # share of TRAIN measurements held by qualifying species
        train_share = sup_all[ok_all].sum() / total_train * 100
        ok_te_species = {s for s in te_sp.unique() if cnt.get(s, 0) >= thr}
        rows.append({
            "split": split,
            "threshold": thr,
            # PRIMARY: all-species basis
            "n_species_total": n_all,
            "n_species_below": int((~ok_all).sum()),
            "pct_species_below_PRIMARY": round((~ok_all).mean() * 100, 1),
            "pct_species_at_or_above": round(ok_all.mean() * 100, 1),
            "pct_train_measurements_held": round(train_share, 1),
            # secondary: test-species basis
            "n_test_species": n_te_sp,
            "pct_test_species_below": round((1 - len(ok_te_species) / n_te_sp) * 100, 1),
            "pct_test_rows_at_or_above": round((sup_te_rows >= thr).mean() * 100, 1),
        })

df = pd.DataFrame(rows)
df.to_csv(OUT / "threshold_coverage_unified.csv", index=False, encoding="utf-8")

pd.set_option("display.width", 250)
print("PRIMARY denominator = ALL species (train u test). '_PRIMARY' column is the headline number.\n")
for split in ("replication_group", "discovery_group"):
    s = df[df.split == split]
    print(f"===== {split}  (all species = {s.n_species_total.iloc[0]}, "
          f"test species = {s.n_test_species.iloc[0]}) =====")
    print(s[["threshold", "n_species_below", "pct_species_below_PRIMARY",
             "pct_species_at_or_above", "pct_train_measurements_held",
             "pct_test_species_below", "pct_test_rows_at_or_above"]].to_string(index=False))
    print()
print(f"wrote {OUT / 'threshold_coverage_unified.csv'}")
