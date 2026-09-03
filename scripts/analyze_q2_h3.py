"""H3 — leakage component selectivity (GNN, formal).

Within-split species gain (absolute RMSE across splits is forbidden -- the leaky and
group test sets differ). For each tier and split:
    delta_split(tier) = RMSE(tier, split) - RMSE(no_species, split)
Leakage effect on a tier:
    leak_shift(tier) = delta_leaky(tier) - delta_group(tier)
    (negative = leakage makes the species gain LARGER; positive = leakage ERODES it)

H3 (original prediction): leakage amplifies the additive intercept (Tier1) but not the
interaction (Tier4)  ->  leak_shift(T1) more negative than leak_shift(T4).
lgbm proxy found the OPPOSITE (intercept unchanged, categorical/interaction eroded).

Env: conda run -n src.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = Path(r".\results\q2_v4\runs\gnn\runs")
TIERS = {
    "T0": "no_species",
    "T1": "species_bias_only",
    "T4": "true_species_late_fusion",
}


def rmse(bb, var, split, seed, ep=100):
    f = RUNS / f"{bb}_{var}_{split}_s{seed}_e{ep}_nfull.json"
    if not f.exists():
        return None
    return float((json.loads(f.read_text(encoding="utf-8")).get("A") or {})["rmse"])


def stats(v):
    n = len(v)
    if n == 0:
        return None
    m = sum(v) / n
    sd = (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, sd, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="replication_group")
    ap.add_argument("--leaky", default="replication_designed_leaky")
    ap.add_argument("--backbones", nargs="+", default=["dmpnn", "graphconv"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    a = ap.parse_args()

    for bb in a.backbones:
        shift_t1, shift_t4, sel = [], [], []
        dg1, dl1, dg4, dl4 = [], [], [], []
        for s in a.seeds:
            g = {k: rmse(bb, v, a.group, s) for k, v in TIERS.items()}
            l = {k: rmse(bb, v, a.leaky, s) for k, v in TIERS.items()}
            if any(x is None for x in list(g.values()) + list(l.values())):
                continue
            d_g1 = g["T1"] - g["T0"]
            d_l1 = l["T1"] - l["T0"]
            d_g4 = g["T4"] - g["T0"]
            d_l4 = l["T4"] - l["T0"]
            dg1.append(d_g1); dl1.append(d_l1); dg4.append(d_g4); dl4.append(d_l4)
            shift_t1.append(d_l1 - d_g1)
            shift_t4.append(d_l4 - d_g4)
            sel.append((d_l1 - d_g1) - (d_l4 - d_g4))  # H3 selectivity contrast

        st1 = stats(shift_t1)
        if st1 is None:
            print(f"\n== {bb}: incomplete (group and/or leaky runs missing) ==")
            continue
        print(f"\n===== H3 GNN: {bb}  (n={st1[2]} seeds) =====")
        print(f"  delta_group(T1)={stats(dg1)[0]:+.4f}  delta_leaky(T1)={stats(dl1)[0]:+.4f}")
        print(f"  delta_group(T4)={stats(dg4)[0]:+.4f}  delta_leaky(T4)={stats(dl4)[0]:+.4f}")
        m1, sd1, n1 = st1
        m4, sd4, _ = stats(shift_t4)
        print(f"  leak_shift(T1 intercept)  = {m1:+.5f} sd={sd1:.5f}  "
              f"({'amplified' if m1 < 0 else 'eroded'}, {sum(1 for x in shift_t1 if x<0)}/{n1} amplified)")
        print(f"  leak_shift(T4 interaction)= {m4:+.5f} sd={sd4:.5f}  "
              f"({'amplified' if m4 < 0 else 'eroded'}, {sum(1 for x in shift_t4 if x<0)}/{n1} amplified)")
        ms, sds, _ = stats(sel)
        favH3 = sum(1 for x in sel if x < 0)  # original H3: T1 more amplified than T4
        print(f"  H3 selectivity [leak_shift(T1)-leak_shift(T4)] = {ms:+.5f} sd={sds:.5f}")
        print(f"    original-H3 direction (T1 amplified more than T4): {favH3}/{n1}")
        verdict = ("supports original H3 (intercept selectively amplified)" if ms < 0
                   else "OPPOSITE of original H3 (leakage erodes interaction, not intercept)")
        print(f"    -> {verdict}")


if __name__ == "__main__":
    main()
