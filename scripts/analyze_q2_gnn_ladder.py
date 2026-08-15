"""Q2 v4 — early A/B signal for the GNN ladder (point estimates + seed direction).

Reference is Tier 1' (naive residual calibration, lgbm family) per SPEC 4-1.
Bootstrap + FDR is a separate, later step; this script reports point estimates,
per-seed favorable counts, and the pre-registered A/B retreat criteria.

Retreat criteria (SPEC-fixed, judged on QUALITY not direction):
  1. GNN unstable      : Tier4 favorable roughly 6-7/10 (direction unresolved)
  2. H3 inconsistent   : GNN H3 disagrees with the lgbm proxy (needs designed_leaky)
  3. Tier4/5 borderline: Tier4 vs Tier2 too close to fix a saturation point

Env: run via `conda run -n jcim_v3`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(r".\results\q2_v4\runs")
GNN = ROOT / "gnn" / "runs"
LGBM = ROOT / "replication" / "runs"

GNN_TIERS = {
    "T0_gnn": "no_species",
    "T1_bias": "species_bias_only",
    "T1_bias_shuf": "shuffled_species_bias_only",
    "T4_late": "true_species_late_fusion",
    "T4_shuf": "shuffled_species_late_fusion",
    "T5_film": "true_species_film",
    "T5_shuf": "shuffled_species_film",
}
LGBM_TIERS = {
    "T0": "LightGBM_RDKit_no_species",
    "T0oof": "LightGBM_RDKit_no_species_oof_base",
    "T1p": "LightGBM_RDKit_species_residual_calibration",
    "T2": "LightGBM_RDKit_species_categorical",
}


def _rmse(path: Path):
    if not path.exists():
        return None
    A = json.loads(path.read_text(encoding="utf-8")).get("A") or {}
    return float(A["rmse"]) if "rmse" in A else None


def gnn_rmse(backbone, variant, split, seed, epochs=100):
    return _rmse(GNN / f"{backbone}_{variant}_{split}_s{seed}_e{epochs}_nfull.json")


def lgbm_rmse(stem, split, seed):
    return _rmse(LGBM / f"{stem}_{split}_s{seed}.json")


def stats(v):
    n = len(v)
    if n == 0:
        return None, None, 0
    m = sum(v) / n
    sd = (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, sd, n


def line(label, vals, better_is_negative=True):
    m, sd, n = stats(vals)
    if n == 0:
        print(f"  {label:44s} (no runs)")
        return None
    fav = sum(1 for x in vals if x < 0) if better_is_negative else sum(1 for x in vals if x > 0)
    print(f"  {label:44s} mean={m:+.5f} sd={sd:.5f}  favorable {fav}/{n}")
    return m, sd, n, fav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="replication_group")
    ap.add_argument("--backbone", default="dmpnn")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=100)
    a = ap.parse_args()

    print(f"===== GNN ladder: {a.backbone} / {a.split} =====")
    g = {k: [] for k in GNN_TIERS}
    l = {k: [] for k in LGBM_TIERS}
    usable = []
    for s in a.seeds:
        gv = {k: gnn_rmse(a.backbone, v, a.split, s, a.epochs) for k, v in GNN_TIERS.items()}
        lv = {k: lgbm_rmse(v, a.split, s) for k, v in LGBM_TIERS.items()}
        if gv["T0_gnn"] is None:
            continue
        usable.append(s)
        for k in GNN_TIERS:
            g[k].append(gv[k])
        for k in LGBM_TIERS:
            l[k].append(lv[k])
    if not usable:
        print("no completed GNN runs yet.")
        return
    print(f"seeds with runs: {usable}\n")

    print("  -- tier RMSE (mean over seeds) --")
    for k in GNN_TIERS:
        vals = [x for x in g[k] if x is not None]
        if vals:
            m, sd, n = stats(vals)
            print(f"  {k:16s} ({GNN_TIERS[k][:28]:28s}) {m:8.4f}  sd={sd:.5f}  n={n}")

    def pair(a_key, b_key, src_a=g, src_b=g):
        out = []
        for i in range(len(usable)):
            x, y = src_a[a_key][i], src_b[b_key][i]
            if x is not None and y is not None:
                out.append(x - y)
        return out

    print("\n  -- species-signal contrasts (negative = first better) --")
    line("T1_bias vs T0_gnn        [Tier1 gain]", pair("T1_bias", "T0_gnn"))
    line("T1_bias vs T1_bias_shuf  [CONTROL]", pair("T1_bias", "T1_bias_shuf"))
    line("T4_late vs T0_gnn        [Tier4 gain]", pair("T4_late", "T0_gnn"))
    line("T4_late vs T4_shuf       [CONTROL]", pair("T4_late", "T4_shuf"))
    line("T5_film vs T0_gnn        [Tier5 gain]", pair("T5_film", "T0_gnn"))
    line("T5_film vs T5_shuf       [CONTROL]", pair("T5_film", "T5_shuf"))
    line("T4_late vs T1_bias       [injection]", pair("T4_late", "T1_bias"))
    line("T5_film vs T4_late       [FiLM extra]", pair("T5_film", "T4_late"))

    print("\n  -- delta-of-deltas vs Tier 1' reference (SPEC 4-1) --")
    def dd(tier_key):
        out = []
        for i in range(len(usable)):
            tk, t0 = g[tier_key][i], g["T0_gnn"][i]
            t1p, t0o = l["T1p"][i], l["T0oof"][i]
            if None in (tk, t0, t1p, t0o):
                continue
            out.append((tk - t0) - (t1p - t0o))
        return out
    r_t1 = line("dd(Tier1 GNN bias vs Tier1')", dd("T1_bias"))
    r_t4 = line("dd(Tier4 late_fusion vs Tier1')", dd("T4_late"))
    r_t5 = line("dd(Tier5 film      vs Tier1')", dd("T5_film"))

    # Tier4 vs Tier2 (cross-backbone -> SPEC 4-0a' asymmetry note)
    print("\n  -- saturation point: Tier4 (GNN) vs Tier2 (lgbm categorical) --")
    print("     NOTE SPEC 4-0a': lgbm=deterministic, GNN=stochastic; CI widths differ structurally.")
    t4_vs_t2 = []
    for i in range(len(usable)):
        t4, t0, t2, t0l = g["T4_late"][i], g["T0_gnn"][i], l["T2"][i], l["T0"][i]
        if None in (t4, t0, t2, t0l):
            continue
        t4_vs_t2.append((t4 - t0) - (t2 - t0l))
    r_sat = line("dd(Tier4) - dd(Tier2)  [<0 = Tier4 wins]", t4_vs_t2)

    print("\n===== pre-registered A/B check =====")
    if r_t4:
        _, _, n4, fav4 = r_t4
        c1 = 5 <= fav4 <= 8 and n4 >= 8
        print(f"  (1) GNN unstable      : Tier4 favorable {fav4}/{n4} -> {'RETREAT' if c1 else 'ok'}")
    if r_sat:
        m_s, sd_s, n_s, fav_s = r_sat
        c3 = abs(m_s) < 0.005 or (2 <= fav_s <= n_s - 2)
        print(f"  (3) Tier4/5 borderline: dd(T4)-dd(T2) mean={m_s:+.5f} favorable {fav_s}/{n_s} "
              f"-> {'RETREAT' if c3 else 'ok'}")
    print("  (2) H3 inconsistency  : requires designed_leaky runs (pending)")


if __name__ == "__main__":
    main()
