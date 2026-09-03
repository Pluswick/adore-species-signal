"""B1 δ′ ensemble-scale sensitivity seeds (§4δ′). SEPARATE namespace runs_b1/gnn_dprime.
Byte-identical compute to Phase 1 (run_q2_delta_prime.py): same V3RunConfig hyperparameters, same
tier1prime OOF; only data path (data_b1), out_root (runs_b1/gnn_dprime), split (b1_group) differ,
+ ledger. Seeds 10-99 (90/condition). main = 6 var x 2 bb; tier1prime = 2 bb.

GPU assignment by seed (balanced 50/50 for B1's ~equal GPUs; Phase 1's 7:3 card rule assumed a
slower 5060Ti — for B1 the sequencer passes explicit balanced --seeds per card). δ′ = within-condition
seed SD includes cross-GPU variance, same character as Phase 1. Resumable. Env: conda run -n src.
"""
from __future__ import annotations
import argparse, sys, time, json
from datetime import datetime
from pathlib import Path
sys.path.insert(0, r".")
sys.path.insert(0, r".\scripts")
from src.runner import V3RunConfig, run_v3_smoke

ROOT = Path(r".\results\q2_v4")
DATA = str(ROOT / "data_b1")
OUT = str(ROOT / "runs_b1" / "gnn_dprime")
SPLIT = "b1_group"
VARIANTS_MAIN = ["no_species", "species_bias_only", "true_species_categorical",
                 "true_species_taxonomy_original", "true_species_taxonomy_ncbi", "true_species_late_fusion"]
BACKBONES = ["dmpnn", "graphconv"]
BLOCK = "dprime"


def make_log(tag):
    ledger = ROOT / "runs_b1" / "_status" / f"progress_{tag}.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    def log(rid, status, error=None):
        rec = {"run_id": rid, "block": BLOCK, "status": status, "ts": datetime.now().isoformat(timespec="seconds")}
        if error is not None:
            rec["error"] = str(error)[:200]
        with open(ledger, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return log


def run_main(seeds, epochs, log, backbones=BACKBONES):
    out_root = Path(OUT); (out_root / "runs").mkdir(parents=True, exist_ok=True)
    done = skip = fail = 0; t0 = time.time()
    for var in VARIANTS_MAIN:
        for bb in backbones:
            for sd in seeds:
                rid = f"{bb}_{var}_{SPLIT}_s{sd}_e{epochs}_nfull"
                if (out_root / "runs" / f"{rid}.json").exists():
                    skip += 1; continue
                cfg = V3RunConfig(backbone=bb, variant=var, split=SPLIT, seed=sd, epochs=epochs,
                                  batch_size=256, lr=5e-4, weight_decay=1e-5, hidden=300, depth=3,
                                  dropout=0.1, species_emb_dim=16, val_frac=0.1,
                                  limit_train=None, limit_test=None, data_dir=DATA, out_root=OUT)
                try:
                    run_v3_smoke(cfg); done += 1; log(rid, "ok")
                    print(f"[dprime OK] {rid} ({(time.time()-t0)/60:.1f}m)", flush=True)
                except Exception as e:
                    fail += 1; log(rid, "fail", e); print(f"[dprime FAIL] {rid}: {e!r}", flush=True)
    print(f"\n=== dprime main: done={done} skip={skip} fail={fail} time={(time.time()-t0)/60:.1f}m ===", flush=True)


def run_tier1prime(seeds, epochs, log, backbones=BACKBONES):
    import run_q2_gnn_oof_tier1prime as t1p
    t1p.DATA = Path(DATA)
    t1p.PRED = Path(OUT) / "predictions"; t1p.PRED.mkdir(parents=True, exist_ok=True)
    done = skip = fail = 0; t0 = time.time()
    for bb in backbones:
        for sd in seeds:
            rid = f"{bb}_tier1prime_oof_{SPLIT}_s{sd}_e{epochs}_nfull"
            try:
                r = t1p.run_one(bb, SPLIT, sd, "true")
                if r.get("skipped"): skip += 1
                else: done += 1; log(rid, "ok"); print(f"[dprime-t1' OK] {bb} s{sd} ({(time.time()-t0)/60:.1f}m)", flush=True)
            except Exception as e:
                fail += 1; log(rid, "fail", e); print(f"[dprime-t1' FAIL] {bb} s{sd}: {e!r}", flush=True)
    print(f"\n=== dprime tier1prime: done={done} skip={skip} fail={fail} time={(time.time()-t0)/60:.1f}m ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", choices=["main", "tier1prime"], default="main")
    ap.add_argument("--seeds", required=True, help="comma-sep seed list (10-99 space)")
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--tag", default="dprime")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    log = make_log(a.tag)
    print(f"δ′ B1: tiers={a.tiers} bb={a.backbones} n_seeds={len(seeds)} seeds[:6]={seeds[:6]}", flush=True)
    if a.tiers == "tier1prime":
        run_tier1prime(seeds, a.epochs, log, a.backbones)
    else:
        run_main(seeds, a.epochs, log, a.backbones)
