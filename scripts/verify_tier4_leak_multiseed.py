"""Supplementary robustness for the Tier 4 leak test: the pre-registered single-seed run left
discovery_group at improve_shuffled=0.0192 (just under tau=0.02). If that were real label leakage
it would be a STABLE positive across permutation seeds; if noise/support-structure it straddles 0.
Run the two positive-margin splits across several permutation seeds. Does NOT change the
pre-registered verdict — pure robustness. Env: conda run -n jcim_v3.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, r".")
sys.path.insert(0, r".\scripts")
from run_q2_lgbm_tier4 import build_factor, K, DATA
from verify_tier4_permutation_leak import _final_rmse
from jcim_v3.rdkit_lgbm import RDKitLGBMConfig

SPLITS = ["discovery_group", "replication_scaffold_generic"]
SEEDS = [20260801, 20260802, 20260803, 20260804, 20260805]
TAU = 0.02

out = {}
for split in SPLITS:
    cfg = RDKitLGBMConfig(baseline="LightGBM_RDKit_no_species", split=split, seed=0,
                          data_dir=str(DATA), out_root="")
    train = pd.read_csv(DATA / f"{split}_train.csv"); test = pd.read_csv(DATA / f"{split}_test.csv")
    rmse_nofactor = _final_rmse(train, test, cfg, None, None)
    imps = []
    for ps in SEEDS:
        rng = np.random.RandomState(ps)
        tp = train.copy(); tp["target_log10"] = train["target_log10"].to_numpy(np.float64)[rng.permutation(len(train))]
        tf, mp = build_factor(tp, cfg, K)
        rs = _final_rmse(train, test, cfg, tf, mp)
        imps.append(round(rmse_nofactor - rs, 4))
    arr = np.array(imps)
    out[split] = {"rmse_nofactor": round(rmse_nofactor, 4), "improve_shuffled_by_seed": imps,
                  "mean": round(float(arr.mean()), 4), "max": round(float(arr.max()), 4),
                  "min": round(float(arr.min()), 4), "n_exceed_tau": int((arr > TAU).sum()),
                  "systematic_leak": bool(arr.mean() > TAU)}
    print(json.dumps({split: out[split]}, ensure_ascii=False), flush=True)

any_sys = any(v["systematic_leak"] for v in out.values())
(DATA / "_ext" / "tier4_leak_multiseed.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n=== multiseed robustness: systematic_leak={any_sys} (mean>tau on any split) ===", flush=True)
