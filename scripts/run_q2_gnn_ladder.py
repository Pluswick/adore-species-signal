"""Q2 v4 Task D — GNN ladder runs (Tier 0/1/4/5 + shuffled controls).

Matched budget = discovery full config (v3_compound_random_core_full.json):
  epochs 100, patience 15, batch 256, lr 5e-4, wd 1e-5, dropout 0.1,
  hidden 300, depth 3, species_emb_dim 16, val_frac 0.1.

(A) endpoint/duration residualization is applied inside runner.py (y preprocessing);
GNN_STRATUM_EXPOSURE stays False (ccmpnn mol_feat axis is unusable for our fusions).
Resumable: a run whose runs/<id>.json already exists is skipped.

Env: run via `conda run -n src`.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runner import V3RunConfig, run_v3_smoke
from src.tier_input_guard import assert_tier_input, TierInputDegenerate

DATA = r".\results\q2_v4\data"
OUT = r".\results\q2_v4\runs\gnn"
GUARD_LOG = str(Path(OUT) / "tier_input_guard.jsonl")
REF = str(Path(DATA) / "tier_input_reference.json")

# Tier 0 baseline + Tier 1/4/5 + shuffled controls
VARIANTS = [
    "no_species",                    # Tier 0 (delta baseline)
    "species_bias_only",             # Tier 1
    "true_species_late_fusion",      # Tier 4
    "true_species_film",             # Tier 5
    "shuffled_species_late_fusion",  # control for Tier 4
    "shuffled_species_film",         # control for Tier 5
]


def run_id(backbone, variant, split, seed, epochs):
    return f"{backbone}_{variant}_{split}_s{seed}_e{epochs}_nfull"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--backbones", nargs="+", default=["dmpnn", "graphconv"])
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    out_root = Path(OUT)
    (out_root / "runs").mkdir(parents=True, exist_ok=True)

    # tier(variant)-outer order: complete each tier fully before the next (aligns execution
    # with the gate order t0->t1->...->t4 so "tier complete" reporting is well-defined).
    plan = [(sp, bb, va, sd)
            for va in args.variants
            for sp in args.splits for bb in args.backbones for sd in args.seeds]
    print(f"planned runs: {len(plan)}", flush=True)

    done = failed = skipped = 0
    guarded: set[tuple[str, str]] = set()
    t_start = time.time()
    for i, (split, backbone, variant, seed) in enumerate(plan, 1):
        # Tier-input non-degeneracy guard: assert once per (variant, split) BEFORE its cells run.
        # A degenerate species representation (e.g. an all-null taxonomy rank) halts the tier.
        key = (variant, split)
        if key not in guarded:
            guarded.add(key)
            try:
                assert_tier_input(variant, split, DATA, GUARD_LOG, REF)
            except TierInputDegenerate as deg:
                print(f"\n### HALT — tier input degenerate, not starting this tier ###\n{deg}", flush=True)
                sys.exit(2)
        rid = run_id(backbone, variant, split, seed, args.epochs)
        if (out_root / "runs" / f"{rid}.json").exists():
            skipped += 1
            continue
        cfg = V3RunConfig(
            backbone=backbone, variant=variant, split=split, seed=seed,
            epochs=args.epochs, batch_size=256, lr=5e-4, weight_decay=1e-5,
            hidden=300, depth=3, dropout=0.1, species_emb_dim=16, val_frac=0.1,
            limit_train=None, limit_test=None,
            data_dir=DATA, out_root=OUT,
        )
        t0 = time.time()
        try:
            res = run_v3_smoke(cfg)
            rmse = (res.get("A") or {}).get("rmse")
            done += 1
            print(f"[{i}/{len(plan)}] OK   {rid}  rmse={rmse:.4f}  "
                  f"({time.time()-t0:.0f}s, elapsed {(time.time()-t_start)/60:.1f}m)", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[{i}/{len(plan)}] FAIL {rid}: {exc!r}", flush=True)
            traceback.print_exc()

    print(f"\ndone={done} skipped={skipped} failed={failed} "
          f"total_time={(time.time()-t_start)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
