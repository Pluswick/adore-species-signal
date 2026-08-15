"""B1 species-cold (block B, eval-time OOV mapping). Byte-identical compute to Phase 1
(run_q2_blockb_oov.py); monkeypatch its DATA/PRED to B1 and call run_one. Range {t0,t2,t3a,t3b,t4}
main + shuffled (tier 1/1' N/A). Each species variant -> 3 oov CSVs (mean/untrained/collapse),
no_species -> 1. Ledger per oov-file (block species_cold -> ~500). Skip if all oov files exist.

Dual-GPU: CUDA_VISIBLE_DEVICES=1 --seeds 0 1 2 3 4 (4090) / =0 --seeds 5 6 7 8 9 (5060Ti).
Env: conda run -n jcim_v3.
"""
from __future__ import annotations
import argparse, sys, time, json
from datetime import datetime
from pathlib import Path
sys.path.insert(0, r".")
sys.path.insert(0, r".\scripts")
import run_q2_blockb_oov as bb
from jcim_v3.tier_input_guard import TierInputDegenerate

ROOT = Path(r".\results\q2_v4")
bb.DATA = ROOT / "data_b1"
bb.PRED = ROOT / "runs_b1" / "gnn" / "predictions"
BLOCK = "species_cold"
SPLIT = "b1_species_cold"
VARIANTS = ["no_species",
            "true_species_categorical", "true_species_taxonomy_original",
            "true_species_taxonomy_ncbi", "true_species_late_fusion",
            "shuffled_species_categorical", "shuffled_species_taxonomy_original",
            "shuffled_species_taxonomy_ncbi", "shuffled_species_late_fusion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["dmpnn", "graphconv"])
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--tag", default="cold")
    args = ap.parse_args()
    ledger = ROOT / "runs_b1" / "_status" / f"progress_{args.tag}.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)

    def log(rid, status, error=None):
        rec = {"run_id": rid, "block": BLOCK, "status": status, "ts": datetime.now().isoformat(timespec="seconds")}
        if error is not None:
            rec["error"] = str(error)[:200]
        with open(ledger, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    plan = [(bkb, va, sd) for va in args.variants for bkb in args.backbones for sd in args.seeds]
    print(f"cold planned: {len(plan)} runs (data={bb.DATA}, split={SPLIT})", flush=True)
    done = fail = skip = 0; t0 = time.time()
    for i, (bkb, va, sd) in enumerate(plan, 1):
        try:
            r = bb.run_one(bkb, va, SPLIT, sd, "true", args.epochs)
            if r.get("skipped"):
                skip += 1; continue
            for mode in r.get("rmse_by_oov", {}):
                log(f"{bkb}_{va}_{SPLIT}_s{sd}_e{args.epochs}_oov-{mode}", "ok")
                done += 1
            print(f"[{i}/{len(plan)}] OK {bkb}/{va}/s{sd} modes={list(r.get('rmse_by_oov',{}))} ({time.time()-t0:.0f}s)", flush=True)
        except TierInputDegenerate as deg:
            print(f"### HALT — block B degenerate ###\n{deg}", flush=True); sys.exit(2)
        except Exception as e:
            fail += 1; log(f"{bkb}_{va}_{SPLIT}_s{sd}", "fail", e)
            print(f"[{i}/{len(plan)}] FAIL {bkb}/{va}/s{sd}: {type(e).__name__}: {e}", flush=True)
    print(f"\ncold done(oov-files)={done} skip(runs)={skip} fail={fail} time={(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
