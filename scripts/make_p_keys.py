"""Derive the Phase-1 corpus P disjointness keys consumed by build_b1.py.

build_b1.py subtracts the Phase-1 training corpus P from the corrected 2026 E-full
two ways: by ECOTOX result_id, and by a precise (reference_number, species, CAS, duration)
key.  P == the ADORE intermediate `processed` mortality table (70,670 records, 1,267 species;
the same file build_adore_datasets.py ingests for Phase 1).  This script materialises:

    results/q2_v4/data_b1/_ext/P_result_ids.txt    -- one result_id per line
    results/q2_v4/data_b1/_ext/P_precise_keys.txt  -- one "ref|species|cas|duration" per line

The key transforms are byte-identical to build_b1.subtract_P / load_corrected_efull:
    species  = tax_gs.strip().lower()
    cas      = test_cas.strip()
    duration = str(pd.to_numeric(result_obs_duration_mean))    # e.g. "48.0"
    ref      = str(reference_number)

INPUT (not redistributed here; see repo README, "Data"):
    ADORE_PROCESSED = the ADORE `processed/ecotox_mortality_processed.csv` (ECOTOX 2022-09),
    distributed with the ADORE benchmark under CC-BY 4.0. Supply it via the ADORE_PROCESSED
    environment variable or as the first command-line argument.

The keys are the P side only; build_b1.py's self-validation (split counts vs the committed
data_provenance_ledger.csv) is the end-to-end check that P and the 2026 E-full align.
Env: jcim_v3 (conda run). No training.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

# ---- input: the ADORE intermediate processed mortality table (CC-BY 4.0) ----
# Set ADORE_PROCESSED to its path, or pass it as the first command-line argument.
import os
ADORE_PROCESSED = Path(
    sys.argv[1] if len(sys.argv) > 1
    else os.environ.get("ADORE_PROCESSED", "")
)
if not ADORE_PROCESSED.name:
    raise SystemExit(
        "Set the ADORE_PROCESSED environment variable to "
        "<ADORE>/processed/ecotox_mortality_processed.csv, "
        "or pass the path as the first argument."
    )

OUT = (Path(__file__).resolve().parent.parent / "results" / "q2_v4" / "data_b1" / "_ext")
OUT.mkdir(parents=True, exist_ok=True)

USECOLS = ["result_id", "reference_number", "test_cas", "tax_gs", "result_obs_duration_mean"]


def main():
    df = pd.read_csv(ADORE_PROCESSED, usecols=USECOLS, low_memory=False)
    print(f"P source: {ADORE_PROCESSED.name}  rows={len(df)}")

    ref = df["reference_number"].astype(str)
    species = df["tax_gs"].astype(str).str.strip().str.lower()
    cas = df["test_cas"].astype(str).str.strip()
    duration = pd.to_numeric(df["result_obs_duration_mean"], errors="coerce").astype(str)

    result_ids = sorted(set(df["result_id"].astype(str).str.strip()))
    precise = sorted(set(ref + "|" + species + "|" + cas + "|" + duration))

    (OUT / "P_result_ids.txt").write_text("\n".join(result_ids) + "\n", encoding="utf-8")
    (OUT / "P_precise_keys.txt").write_text("\n".join(precise) + "\n", encoding="utf-8")

    print(f"wrote P_result_ids.txt   : {len(result_ids)} unique result_ids  e.g. {result_ids[:3]}")
    print(f"wrote P_precise_keys.txt : {len(precise)} unique keys")
    print(f"  sample key: {precise[0]!r}")
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
