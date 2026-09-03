"""B1 GNN warm ladder — 22 ladder variants. Byte-identical compute to Phase 1
(`run_q2_gnn_ladder.py`): same imports, same V3RunConfig hyperparameters, same guard, same
run_v3_smoke. ONLY the data path (data_b1), output root (runs_b1/gnn), and split names (b1_*)
differ, plus a progress-ledger append. Resumable (skip if runs/<id>.json exists).

t1' (tier1prime_oof / shuffled_tier1prime_oof) is the other 2 of the 24 warm variants and runs
separately (run_b1_gnn_t1prime.py) because it needs the no_species base CSVs this script writes.

GPU: launch with CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 (4090) CUBLAS_WORKSPACE_CONFIG=:4096:8.
Env: conda run -n src.
"""
from __future__ import annotations
import argparse, sys, time, json, traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runner import V3RunConfig, run_v3_smoke
from src.tier_input_guard import assert_tier_input, TierInputDegenerate

ROOT = Path(r".\results\q2_v4")
DATA = str(ROOT / "data_b1")
OUT = str(ROOT / "runs_b1" / "gnn")
REF = str(Path(DATA) / "tier_input_reference.json")
BLOCK = "gnn_warm"
# per-shard ledger + guard-log paths (set from --tag in main) so concurrent shards never race.
GUARD_LOG = str(Path(OUT) / "tier_input_guard.jsonl")
LEDGER = ROOT / "runs_b1" / "_status" / "progress.jsonl"

# 22 ladder variants (the other 2 of the 24 warm = t1' via run_b1_gnn_t1prime.py)
VARIANTS = [
    "no_species",                                                                        # t0
    "species_bias_only", "shuffled_species_bias_only", "zero_species_bias_only", "dummy_species_bias_only",       # t1
    "true_species_categorical", "shuffled_species_categorical", "zero_species_categorical", "dummy_species_categorical",  # t2
    "true_species_taxonomy_original", "shuffled_species_taxonomy_original", "zero_species_taxonomy_original", "dummy_species_taxonomy_original",  # t3a
    "true_species_fixed_proj",                                                           # capacity control
    "true_species_taxonomy_ncbi", "shuffled_species_taxonomy_ncbi", "zero_species_taxonomy_ncbi", "dummy_species_taxonomy_ncbi",  # t3b
    "true_species_late_fusion", "shuffled_species_late_fusion", "zero_species_late_fusion", "dummy_species_late_fusion",  # t4
]
SPLITS = ["b1_group", "b1_scaffold", "b1_scaffold_generic", "b1_designed_leaky"]


def ledger(rid, status, error=None):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    rec = {"run_id": rid, "block": BLOCK, "status": status, "ts": datetime.now().isoformat(timespec="seconds")}
    if error is not None:
        rec["error"] = str(error)[:200]
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_id(backbone, variant, split, seed, epochs):
    return f"{backbone}_{variant}_{split}_s{seed}_e{epochs}_nfull"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--backbones", nargs="+", default=["dmpnn", "graphconv"])
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--tag", default=None, help="shard tag; suffixes ledger + guard-log to avoid concurrent races")
    ap.add_argument("--block", default="gnn_warm", help="ledger block label (gnn_warm | rank)")
    args = ap.parse_args()

    global GUARD_LOG, LEDGER, BLOCK
    BLOCK = args.block
    if args.tag:
        GUARD_LOG = str(Path(OUT) / f"tier_input_guard_{args.tag}.jsonl")
        LEDGER = ROOT / "runs_b1" / "_status" / f"progress_{args.tag}.jsonl"

    out_root = Path(OUT)
    (out_root / "runs").mkdir(parents=True, exist_ok=True)
    plan = [(sp, bb, va, sd)
            for va in args.variants
            for sp in args.splits for bb in args.backbones for sd in args.seeds]
    print(f"planned runs: {len(plan)}  (data={DATA})", flush=True)

    done = failed = skipped = 0
    guarded: set[tuple[str, str]] = set()
    t_start = time.time()
    for i, (split, backbone, variant, seed) in enumerate(plan, 1):
        key = (variant, split)
        if key not in guarded:
            guarded.add(key)
            try:
                assert_tier_input(variant, split, DATA, GUARD_LOG, REF)
            except TierInputDegenerate as deg:
                print(f"\n### HALT — tier input degenerate ###\n{deg}", flush=True)
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
            done += 1
            ledger(rid, "ok")
            print(f"[{i}/{len(plan)}] OK   {rid}  ({time.time()-t0:.0f}s, elapsed {(time.time()-t_start)/60:.1f}m)", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            ledger(rid, "fail", exc)
            print(f"[{i}/{len(plan)}] FAIL {rid}: {exc!r}", flush=True)
            traceback.print_exc()

    print(f"\ndone={done} skipped={skipped} failed={failed} total_time={(time.time()-t_start)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
