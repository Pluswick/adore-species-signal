"""B1 re-experiment progress reporter (hourly scheduler target).

Reads the run ledger + phase marker and prints a COMPACT status block: progress %, ok/fail run
counts per block, recent failures, GPU state, timestamp. NO performance numbers (director rule:
progress reports contain counts/status only; performance judgment only after all runs complete).

Contract (all under results/q2_v4/runs_b1/_status/):
  census.json    : {"total": N, "blocks": {name: {"expected": k, "desc": ...}}}  (frozen plan)
  phase.txt      : single-line human-readable current phase (maintained by the operator)
  progress.jsonl : append-only, one JSON per run attempt written by the runner:
                   {"run_id","block","status":"ok"|"fail","error"?,"ts"}
                   Retries allowed: a run_id's LATEST line wins.
Pure stdlib. Run: conda run -n jcim_v3 python scripts/b1_status.py
"""
from __future__ import annotations
import json, os, subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(r".\results\q2_v4\runs_b1\_status")
CENSUS = ROOT / "census.json"
PHASE = ROOT / "phase.txt"
LEDGER = ROOT / "progress.jsonl"

DEFAULT_CENSUS = {"total": 3794, "blocks": {
    "gnn_warm": {"expected": 1920}, "species_cold": {"expected": 500},
    "rank": {"expected": 40}, "deterministic": {"expected": 74}, "dprime": {"expected": 1260}}}


def load_census():
    try:
        return json.loads(CENSUS.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_CENSUS


def read_ledger():
    """Return {run_id: {...}} keeping the LAST entry per run_id, aggregated across ALL shard
    ledgers (progress.jsonl + progress_<tag>.jsonl written by concurrent shards)."""
    latest = {}
    ledgers = sorted(ROOT.glob("progress*.jsonl"))
    for lf in ledgers:
        try:
            text = lf.read_text(encoding="utf-8")
        except Exception:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            rid = r.get("run_id")
            if rid is None:
                continue
            latest[rid] = r
    return latest


def gpu_line():
    try:
        q = "utilization.gpu,memory.used,memory.total,name"
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        if out.returncode != 0 or not out.stdout.strip():
            return "GPU: nvidia-smi no response"
        lines = [l.strip() for l in out.stdout.strip().splitlines() if l.strip()]
        parts = []
        for i, l in enumerate(lines):
            f = [x.strip() for x in l.split(",")]
            if len(f) >= 4:
                parts.append(f"[{i}] {f[3]} util={f[0]}% mem={f[1]}/{f[2]}MiB")
        return "GPU: " + " | ".join(parts) if parts else "GPU: (parse failed)"
    except FileNotFoundError:
        return "GPU: nvidia-smi not found"
    except Exception as e:
        return f"GPU: query failed ({type(e).__name__})"


def main():
    census = load_census()
    total = census.get("total", 3794)
    blocks = census.get("blocks", {})
    phase = PHASE.read_text(encoding="utf-8").strip() if PHASE.exists() else "(phase 미기록)"
    latest = read_ledger()

    per = {b: {"ok": 0, "fail": 0} for b in blocks}
    fails = []
    for rid, r in latest.items():
        b = r.get("block", "?")
        st = r.get("status", "?")
        per.setdefault(b, {"ok": 0, "fail": 0})
        if st == "ok":
            per[b]["ok"] += 1
        elif st == "fail":
            per[b]["fail"] += 1
            fails.append((rid, b, str(r.get("error", ""))[:80]))

    ok = sum(v["ok"] for v in per.values())
    fail = sum(v["fail"] for v in per.values())
    attempted = ok + fail
    pending = max(total - attempted, 0)
    pct_ok = 100.0 * ok / total if total else 0.0
    pct_att = 100.0 * attempted / total if total else 0.0

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"=== B1 progress @ {ts} ===")
    print(f"phase: {phase}")
    print(f"progress: ok {ok}/{total} ({pct_ok:.1f}%) | attempted {attempted}/{total} ({pct_att:.1f}%) | "
          f"pending {pending} | failed {fail}")
    print("by block (ok/fail/expected):")
    for b, meta in blocks.items():
        exp = meta.get("expected", 0)
        c = per.get(b, {"ok": 0, "fail": 0})
        done = c["ok"] + c["fail"]
        bar = f"{c['ok']}/{c['fail']}/{exp}"
        pc = f"{100.0*c['ok']/exp:.0f}%" if exp else "-"
        flag = "  <-- not started" if done == 0 else ("  <-- done" if c["ok"] >= exp else "")
        print(f"  {b:<14} ok/fail/exp={bar:<14} ({pc}){flag}")
    # any blocks in ledger not in census
    for b in per:
        if b not in blocks and b != "?":
            print(f"  [unplanned block] {b}: ok={per[b]['ok']} fail={per[b]['fail']}")

    if fail:
        print(f"failed runs (last {min(len(fails),10)}):")
        for rid, b, err in fails[-10:]:
            print(f"  - {rid} [{b}] {err}")
    else:
        print("failed runs: none")

    print(gpu_line())

    if total and attempted >= total and fail == 0:
        print("STATUS: COMPLETE (all runs ok)")
    elif total and attempted >= total and fail > 0:
        print(f"STATUS: ATTEMPTED-ALL, {fail} failure(s) remain -- retry/investigate")
    else:
        print("STATUS: IN-PROGRESS" if attempted > 0 else "STATUS: NOT-STARTED (pre-flight)")


if __name__ == "__main__":
    main()
