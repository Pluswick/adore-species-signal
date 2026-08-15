"""δ′ ensemble-scale sensitivity seeds (D-ΔδMATCH, §4δ′). SEPARATE namespace: results/q2_v4/runs/gnn_dprime.

Adds 90 seeds/condition (seeds 10-99) for the C=14 warm main conditions (discovery x group), grouped
into 9 disjoint ensembles of consecutive 10s (§4δ′ A-5.3). Card rule (A-5.4): within each decade,
seeds ending 0-6 -> 5060Ti, 7-9 -> 4090 (matches canonical 7:3); t1' entirely on 4090. Identical
hyperparameters/packing/determinism as canonical — only the seed number differs (A-5.6). Resumable.
Canonical seeds 0-9 (runs/gnn) are the immutable representative ensemble and are NEVER run here.
Env: conda run -n jcim_v3 with CUDA_VISIBLE_DEVICES set per card.
"""
from __future__ import annotations
import sys, argparse, time, traceback
from pathlib import Path
sys.path.insert(0, r".")
from jcim_v3.runner import V3RunConfig, run_v3_smoke

DATA = r".\results\q2_v4\data"
OUT = r".\results\q2_v4\runs\gnn_dprime"
SPLIT = "discovery_group"
VARIANTS_MAIN = ["no_species", "species_bias_only", "true_species_categorical",
                 "true_species_taxonomy_original", "true_species_taxonomy_ncbi", "true_species_late_fusion"]
BACKBONES = ["dmpnn", "graphconv"]


def card_seeds(card):
    """Deterministic §4δ′ A-5.4 rule over δ′ seeds 10-99 (9 decades)."""
    five, nine = [], []
    for dec in range(1, 10):
        b = dec * 10
        five += [b + i for i in range(0, 7)]    # ...0-6 -> 5060Ti
        nine += [b + i for i in range(7, 10)]   # ...7-9 -> 4090
    return five if card == "5060ti" else nine


def run_main(seeds, epochs):
    out_root = Path(OUT); (out_root / "runs").mkdir(parents=True, exist_ok=True)
    done = skip = fail = 0; t0 = time.time()
    for var in VARIANTS_MAIN:
        for bb in BACKBONES:
            for sd in seeds:
                rid = f"{bb}_{var}_{SPLIT}_s{sd}_e{epochs}_nfull"
                if (out_root / "runs" / f"{rid}.json").exists():
                    skip += 1; continue
                cfg = V3RunConfig(backbone=bb, variant=var, split=SPLIT, seed=sd, epochs=epochs,
                                  batch_size=256, lr=5e-4, weight_decay=1e-5, hidden=300, depth=3,
                                  dropout=0.1, species_emb_dim=16, val_frac=0.1,
                                  limit_train=None, limit_test=None, data_dir=DATA, out_root=OUT)
                try:
                    r = run_v3_smoke(cfg); rmse = (r.get("A") or {}).get("rmse")
                    done += 1; print(f"[dprime OK] {rid} rmse={rmse:.4f} ({(time.time()-t0)/60:.1f}m)", flush=True)
                except Exception as e:
                    fail += 1; print(f"[dprime FAIL] {rid}: {e!r}", flush=True); traceback.print_exc()
    print(f"\n=== dprime main: done={done} skip={skip} fail={fail} time={(time.time()-t0)/60:.1f}m ===", flush=True)


def run_tier1prime(seeds, epochs):
    # t1' delta-prime: reuse the tier1prime runner but redirect its PRED to the dprime namespace,
    # so it reads the no_species dprime base and writes the tier1' dprime prediction there.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_q2_gnn_oof_tier1prime as t1p
    t1p.PRED = Path(OUT) / "predictions"
    t1p.PRED.mkdir(parents=True, exist_ok=True)
    done = skip = fail = 0; t0 = time.time()
    for bb in BACKBONES:
        for sd in seeds:
            try:
                r = t1p.run_one(bb, SPLIT, sd, "true")
                if r.get("skipped"): skip += 1
                else: done += 1; print(f"[dprime-t1' OK] {bb} s{sd} RMSE T1'={r['rmse_tier1prime']:.4f}", flush=True)
            except Exception as e:
                fail += 1; print(f"[dprime-t1' FAIL] {bb} s{sd}: {e!r}", flush=True)
    print(f"\n=== dprime tier1prime: done={done} skip={skip} fail={fail} time={(time.time()-t0)/60:.1f}m ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", choices=["5060ti", "4090"], required=True)
    ap.add_argument("--tiers", choices=["main", "tier1prime"], default="main")
    ap.add_argument("--seeds", default="", help="override seed list (comma sep); else card rule")
    ap.add_argument("--epochs", type=int, default=100)
    a = ap.parse_args()
    if a.seeds:
        seeds = [int(x) for x in a.seeds.split(",")]
    elif a.tiers == "tier1prime":
        seeds = list(range(10, 100))            # t1' entirely on 4090 (A-5.4)
    else:
        seeds = card_seeds(a.card)
    print(f"δ′ launch: card={a.card} tiers={a.tiers} n_seeds={len(seeds)} seeds[:6]={seeds[:6]}...", flush=True)
    (run_tier1prime if a.tiers == "tier1prime" else run_main)(seeds, a.epochs)
