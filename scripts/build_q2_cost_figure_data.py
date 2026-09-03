"""Figure data: cumulative coverage by support threshold, species-basis vs row-basis.

The gap between the two curves visualises the concentration structure (a few
data-rich species carry most measurements). Marks the observed lower bound (3).

Env: conda run -n src.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(r".\results\q2_v4\data")
OUT = Path(r".\results\q2_v4\cost")
LOWER_BOUND = 3

rows = []
for split in ("replication_group", "discovery_group"):
    tr = pd.read_csv(DATA / f"{split}_train.csv", usecols=["species"])
    te = pd.read_csv(DATA / f"{split}_test.csv", usecols=["species"])
    cnt = tr.groupby("species").size()
    all_sp = sorted(set(tr["species"]) | set(te["species"]))
    sup_all = pd.Series({s: int(cnt.get(s, 0)) for s in all_sp})
    te_sp = te["species"]
    sup_rows = te_sp.map(cnt).fillna(0).astype(int)

    for thr in list(range(0, 21)) + [25, 30, 40, 50, 75, 100, 150, 200]:
        rows.append({
            "split": split,
            "threshold": thr,
            "pct_species_at_or_above": round((sup_all >= thr).mean() * 100, 2),
            "pct_test_rows_at_or_above": round((sup_rows >= thr).mean() * 100, 2),
            "n_species_at_or_above": int((sup_all >= thr).sum()),
            "n_test_rows_at_or_above": int((sup_rows >= thr).sum()),
            "is_lower_bound": thr == LOWER_BOUND,
        })

df = pd.DataFrame(rows)
df.to_csv(OUT / "figure_support_cumulative.csv", index=False, encoding="utf-8")
print(df[df.threshold.isin([0, 1, 2, 3, 5, 10, 50, 100])].to_string(index=False))
print(f"\nwrote {OUT / 'figure_support_cumulative.csv'}")
