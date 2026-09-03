"""B1 GNN Tier 1' (OOF residual calibration) — 2 controls (true, shuffled). Byte-identical compute
to Phase 1 (run_q2_gnn_oof_tier1prime.py); monkeypatch its DATA/PRED module constants to B1 paths and
call its run_one. Needs the no_species base CSVs the ladder writes. Counts toward gnn_warm (24 warm
variants = 22 ladder + 2 t1'). Ledger + skip (run_one skips if output CSV exists).

Dual-GPU: launch with CUDA_VISIBLE_DEVICES=1 --seeds 0 1 2 3 4  (4090) and =0 --seeds 5 6 7 8 9 (5060Ti).
Env: conda run -n src.
"""
from __future__ import annotations
import argparse, sys, time, json
from datetime import datetime
from pathlib import Path
sys.path.insert(0, r".")
sys.path.insert(0, r".\scripts")
import run_q2_gnn_oof_tier1prime as t1p
from src.tier_input_guard import TierInputDegenerate

ROOT = Path(r".\results\q2_v4")
t1p.DATA = ROOT / "data_b1"
t1p.PRED = ROOT / "runs_b1" / "gnn" / "predictions"
BLOCK = "gnn_warm"
SPLITS = ["b1_group", "b1_scaffold", "b1_scaffold_generic", "b1_designed_leaky"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--backbones", nargs="+", default=["dmpnn", "graphconv"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--controls", nargs="+", default=["true", "shuffled"])
    ap.add_argument("--tag", default="t1prime")
    args = ap.parse_args()
    ledger = ROOT / "runs_b1" / "_status" / f"progress_{args.tag}.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)

    def log(rid, status, error=None):
        rec = {"run_id": rid, "block": BLOCK, "status": status, "ts": datetime.now().isoformat(timespec="seconds")}
        if error is not None:
            rec["error"] = str(error)[:200]
        with open(ledger, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    plan = [(sp, bb, sd, ctrl) for sp in args.splits for bb in args.backbones
            for sd in args.seeds for ctrl in args.controls]
    print(f"t1' planned: {len(plan)}  (data={t1p.DATA})", flush=True)
    done = fail = skip = 0; t0 = time.time()
    for i, (sp, bb, sd, ctrl) in enumerate(plan, 1):
        ovar = "tier1prime_oof" if ctrl == "true" else "shuffled_tier1prime_oof"
        rid = f"{bb}_{ovar}_{sp}_s{sd}_e100_nfull"
        try:
            r = t1p.run_one(bb, sp, sd, ctrl)
            if r.get("skipped"):
                skip += 1
            else:
                done += 1; log(rid, "ok")
                print(f"[{i}/{len(plan)}] OK {rid} ({time.time()-t0:.0f}s)", flush=True)
        except TierInputDegenerate as deg:
            print(f"### HALT — t1' input degenerate ###\n{deg}", flush=True); sys.exit(2)
        except Exception as e:
            fail += 1; log(rid, "fail", e)
            print(f"[{i}/{len(plan)}] FAIL {rid}: {type(e).__name__}: {e}", flush=True)
    print(f"\nt1' done={done} skip={skip} fail={fail} time={(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
