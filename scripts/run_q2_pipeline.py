"""Q2 v4 — single entry point. Runs the whole pipeline for one dataset config.

    conda run -n src python scripts/run_q2_pipeline.py --stage all
    conda run -n src python scripts/run_q2_pipeline.py --stage lgbm --seeds 0-9

Stages (each resumable / independently runnable):
  build   : build_q2_datasets.py           -> results/<ws>/data/*.csv + ledger
  lgbm    : run_q2_replication_ladder.py    -> Tier 0/1'/2 + controls (naive+lgbm), all splits
  gnn     : run_q2_gnn_ladder.py            -> Tier 0/1/4 + controls (dmpnn+graphconv)
  analyze : ladder / abundance / H3 / cold sign test summaries

A new dataset = copy configs/q2_dataset_toxlearn.json, edit vendor_raw + filters, then
`--config <new.json>` (or set Q2_DATASET_CONFIG). Nothing else hardcoded. See README_Q2.md.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _seeds(spec: str) -> list[str]:
    if "-" in spec:
        a, b = spec.split("-")
        return [str(i) for i in range(int(a), int(b) + 1)]
    return spec.split(",")


def run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="dataset config (else Q2_DATASET_CONFIG env or toxlearn default)")
    ap.add_argument("--stage", choices=["build", "lgbm", "gnn", "analyze", "all"], default="all")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--splits", nargs="+",
                    default=["replication_group", "discovery_group", "replication_designed_leaky"])
    a = ap.parse_args()
    seeds = _seeds(a.seeds)
    cfg = ["--config", a.config] if a.config else []

    def script(name):
        return [PY, str(ROOT / "scripts" / name)]

    if a.stage in ("build", "all"):
        # GUARD (Session 23, NCBI-bug follow-up): results/q2_v4/data is owned by the ADORE
        # pipeline (build_adore_datasets.py: species = underscored tax_gs). The legacy
        # build_q2_datasets.py uses a DIFFERENT species convention (space-lowercase 'Latin name')
        # and a DIFFERENT species_idx enumeration; running it here would silently reassign every
        # species_idx and de-align the patched ncbi_* columns + all frozen predictions. Refuse.
        adore_data = ROOT / "results" / "q2_v4" / "data"
        if (adore_data / "exclusion_audit_trail.csv").exists() and \
           (adore_data / "_ext" / "ncbi_taxonomy_by_species.csv").exists():
            print("### REFUSING build stage — results/q2_v4/data is ADORE-owned (build_adore_datasets.py).\n"
                  "    Running legacy build_q2_datasets.py would clobber species_idx / ncbi_* / frozen predictions.\n"
                  "    To (re)build ADORE data, run scripts/build_adore_datasets.py explicitly.", flush=True)
            sys.exit(3)
        run(script("build_q2_datasets.py"))

    if a.stage in ("lgbm", "all"):
        for sp in a.splits:
            run(script("run_q2_replication_ladder.py") + ["--splits", sp, "--seeds", *seeds])

    if a.stage in ("gnn", "all"):
        gnn_vars = ["no_species", "species_bias_only", "shuffled_species_bias_only",
                    "true_species_late_fusion", "shuffled_species_late_fusion"]
        for sp in a.splits:
            run(script("run_q2_gnn_ladder.py") + ["--splits", sp, "--backbones", "dmpnn", "graphconv",
                                                  "--variants", *gnn_vars, "--seeds", *seeds])

    if a.stage in ("analyze", "all"):
        for sp in ("replication_group", "discovery_group"):
            for bb in ("dmpnn", "graphconv"):
                run(script("analyze_q2_gnn_ladder.py") + ["--split", sp, "--backbone", bb, "--seeds", *seeds])
                run(script("analyze_q2_tier4_abundance.py") + ["--split", sp, "--backbone", bb, "--seeds", *seeds])
        run(script("analyze_q2_h3.py") + ["--seeds", *seeds])

    print("\n[run_q2_pipeline] done.", flush=True)


if __name__ == "__main__":
    main()
