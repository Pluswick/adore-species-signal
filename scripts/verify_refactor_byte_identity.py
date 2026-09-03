"""Byte-identity check for the TAXONOMY_INJECTIONS refactor (Session 25, director task 1).

Re-run already-completed t3a/t3b (+ a control) cells with the CURRENT (refactored) code into a
scratch dir and compare against the frozen block-A predictions. Training is deterministic, so if
the refactor is behavior-preserving the predictions must be byte-identical. MUST run on the SAME
GPU the originals used (GPU determinism is per-architecture): seeds 7-9 -> 4090 (VD=1). Any
mismatch -> exit 2 (stop). Env: conda run -n jcim_v3, CUDA_VISIBLE_DEVICES=1.
"""
from __future__ import annotations
import sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, r".")
from jcim_v3.runner import V3RunConfig, run_v3_smoke

DATA = r".\results\q2_v4\data"
REAL = Path(r".\results\q2_v4\runs\gnn\predictions")
import os, tempfile
SCRATCH = Path(os.environ.get("SCRATCH_DIR", Path(tempfile.gettempdir()) / "refactor_verify"))
SPLIT, SEED = "discovery_group", 7          # seed 7 originally ran on the 4090 (job C)
VARIANTS = ["true_species_taxonomy_original", "true_species_taxonomy_ncbi",
            "shuffled_species_taxonomy_original"]   # t3a, t3b, control (same dispatch path)
BACKBONES = ["dmpnn", "graphconv"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def run_one(bb, variant):
    rid = f"{bb}_{variant}_{SPLIT}_s{SEED}_e100_nfull"
    old = REAL / f"{rid}.csv"
    if not old.exists():
        return {"cell": rid, "status": "OLD_MISSING"}
    cfg = V3RunConfig(backbone=bb, variant=variant, split=SPLIT, seed=SEED, epochs=100,
                      batch_size=256, lr=5e-4, weight_decay=1e-5, hidden=300, depth=3, dropout=0.1,
                      species_emb_dim=16, val_frac=0.1, limit_train=None, limit_test=None,
                      data_dir=DATA, out_root=str(SCRATCH))
    run_v3_smoke(cfg)
    new = SCRATCH / "predictions" / f"{rid}.csv"
    o = pd.read_csv(old); n = pd.read_csv(new)
    pred_equal = bool(len(o) == len(n) and np.array_equal(o["pred_log10"].to_numpy(), n["pred_log10"].to_numpy()))
    max_abs = float(np.max(np.abs(o["pred_log10"].to_numpy() - n["pred_log10"].to_numpy()))) if len(o) == len(n) else float("nan")
    file_equal = (sha(old) == sha(new))
    return {"cell": rid, "status": "OK" if (pred_equal and file_equal) else "MISMATCH",
            "pred_array_equal": pred_equal, "file_bytes_equal": file_equal,
            "max_abs_pred_diff": max_abs, "sha_old": sha(old), "sha_new": sha(new)}


if __name__ == "__main__":
    SCRATCH.mkdir(parents=True, exist_ok=True)
    rows, bad = [], 0
    for variant in VARIANTS:
        for bb in BACKBONES:
            r = run_one(bb, variant)
            rows.append(r)
            if r["status"] != "OK":
                bad += 1
            print(f"[{r['status']:8s}] {r['cell']:58s} pred_equal={r.get('pred_array_equal')} "
                  f"file_equal={r.get('file_bytes_equal')} max_abs={r.get('max_abs_pred_diff')} "
                  f"sha {r.get('sha_old')}->{r.get('sha_new')}", flush=True)
    verdict = "PASS" if bad == 0 else "MISMATCH"
    print(f"\n=== REFACTOR BYTE-IDENTITY: {verdict} (mismatches={bad}/{len(rows)}) ===", flush=True)
    sys.exit(2 if bad else 0)
