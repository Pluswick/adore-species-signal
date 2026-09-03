"""Step 4 Phase 2 — naive taxon-group mean baselines (Tier 3a/3b, naive backbone).
Runs run_naive_taxonomy_baselines over splits x seeds. Resumable (skips a (split,seed) whose
4 prediction files already exist). LightGBM/naive family: full train, fixed protocol, no GPU.
"""
import sys, argparse, os
sys.path.insert(0, r".")
from src.naive_species_baselines import NaiveSpeciesBaselineConfig, run_naive_taxonomy_baselines

DATA = r".\results\q2_v4\data"
OUT = r".\results\q2_v4\runs\replication"
MODELS = ["Naive_taxon_mean_original", "Naive_shuffled_taxon_mean_original",
          "Naive_taxon_mean_ncbi", "Naive_shuffled_taxon_mean_ncbi"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="discovery_group,replication_group,discovery_scaffold,replication_scaffold")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    a = ap.parse_args()
    splits = a.splits.split(","); seeds = [int(x) for x in a.seeds.split(",")]
    pred_dir = os.path.join(OUT, "predictions")
    done = skip = 0
    for split in splits:
        for seed in seeds:
            outs = [os.path.join(pred_dir, f"{m}_{split}_s{seed}.csv") for m in MODELS]
            if all(os.path.exists(o) for o in outs):
                skip += 1; print(f"[SKIP] {split} s{seed}", flush=True); continue
            try:
                res = run_naive_taxonomy_baselines(NaiveSpeciesBaselineConfig(
                    split=split, seed=seed, data_dir=DATA, out_root=OUT))
                done += 1
                rmses = {r["model_name"]: round(r.get("rmse", float("nan")), 4) for r in res}
                print(f"[DONE] {split} s{seed}  {rmses}", flush=True)
            except Exception as e:
                print(f"[FAIL] {split} s{seed}: {type(e).__name__}: {e}", flush=True)
    print(f"\n=== naive taxonomy: done={done} skipped={skip} ===", flush=True)
