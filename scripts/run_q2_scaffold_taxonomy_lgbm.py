"""Step 4 Phase 3 (LightGBM) — Tier 3a/3b taxonomy on the SCAFFOLD split.
Reuses the exact item-7 LightGBM taxonomy protocol (run_rdkit_lgbm; 5 core ranks, unknown bucket,
rows never dropped, full train, fixed rounds, endpoint/duration residualization). shuffled control
included. Resumable: skips a (baseline,split,seed) whose prediction file exists.
"""
import sys, os, argparse
sys.path.insert(0, r".")
from jcim_v3.rdkit_lgbm import RDKitLGBMConfig, run_rdkit_lgbm

DATA = r".\results\q2_v4\data"
OUT = r".\results\q2_v4\runs\replication"
BASELINES = ["LightGBM_RDKit_taxonomy_original", "LightGBM_RDKit_shuffled_taxonomy_original",
             "LightGBM_RDKit_taxonomy_ncbi", "LightGBM_RDKit_shuffled_taxonomy_ncbi"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="discovery_scaffold,replication_scaffold")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    a = ap.parse_args()
    splits = a.splits.split(","); seeds = [int(x) for x in a.seeds.split(",")]
    pred = os.path.join(OUT, "predictions")
    done = skip = 0
    for split in splits:
        for b in BASELINES:
            for seed in seeds:
                out = os.path.join(pred, f"{b}_{split}_s{seed}.csv")
                if os.path.exists(out):
                    skip += 1; continue
                try:
                    run_rdkit_lgbm(RDKitLGBMConfig(baseline=b, split=split, seed=seed, data_dir=DATA, out_root=OUT))
                    done += 1; print(f"[DONE] {b} {split} s{seed}", flush=True)
                except Exception as e:
                    print(f"[FAIL] {b} {split} s{seed}: {type(e).__name__}: {e}", flush=True)
    print(f"\n=== scaffold taxonomy LGBM: done={done} skipped={skip} ===", flush=True)
