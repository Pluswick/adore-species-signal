from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.paths import CC_MPNN_DATA, RAW_TOX_LEARN, RESULTS_ROOT


CENSOR_RE = re.compile(r"^\s*(<=|>=|<|>)\s*[-+]?\d")
META_COLS = [
    "CAS",
    "Latin name",
    "Duration (hours)",
    "Effect value",
    "Effect value std",
    "Test statistic",
    "Canonical SMILES",
]


def _operator(value: object) -> str | None:
    s = "" if pd.isna(value) else str(value).strip()
    m = CENSOR_RE.match(s)
    return m.group(1) if m else None


def _write_csv(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    if df.empty:
        pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8")
    else:
        df.to_csv(path, index=False, encoding="utf-8")


def generate_censored_report(
    *,
    raw_dir: str | Path = RAW_TOX_LEARN,
    out_dir: str | Path = RESULTS_ROOT / "data_audit",
    data_dir: str | Path = CC_MPNN_DATA,
) -> dict:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    data_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for name in ("groupsplit_train.csv", "groupsplit_test.csv"):
        path = raw_dir / name
        cols = pd.read_csv(path, nrows=0).columns
        usecols = [c for c in META_COLS if c in cols]
        frame = pd.read_csv(path, usecols=usecols, dtype=str)
        frame["source_file"] = name
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw["censor_operator"] = raw["Effect value"].map(_operator)
    raw["duration_numeric"] = pd.to_numeric(raw["Duration (hours)"], errors="coerce")
    raw["effect_numeric"] = pd.to_numeric(raw["Effect value"], errors="coerce")
    raw["is_censored"] = raw["censor_operator"].notna()
    raw["is_exact_numeric"] = raw["effect_numeric"].notna() & ~raw["is_censored"]

    lc50_96 = raw[(raw["Test statistic"] == "LC50") & (raw["duration_numeric"] == 96.0)].copy()
    censored = raw[raw["is_censored"]].copy()
    lc50_96_censored = lc50_96[lc50_96["is_censored"]].copy()
    lc50_96_exact = lc50_96[lc50_96["is_exact_numeric"]].copy()

    by_endpoint_duration = (
        censored.groupby(["Test statistic", "Duration (hours)", "censor_operator"], dropna=False)
        .size()
        .reset_index(name="n_records")
        .sort_values("n_records", ascending=False)
    )
    by_operator = (
        lc50_96_censored.groupby("censor_operator", dropna=False)
        .size()
        .reset_index(name="n_records")
        .sort_values("n_records", ascending=False)
    )
    by_species = (
        lc50_96_censored.groupby("Latin name", dropna=False)
        .size()
        .reset_index(name="n_records")
        .sort_values("n_records", ascending=False)
    )
    by_compound = (
        lc50_96_censored.groupby(["CAS", "Canonical SMILES"], dropna=False)
        .size()
        .reset_index(name="n_records")
        .sort_values("n_records", ascending=False)
    )

    consolidated_path = data_dir / "lc50_96_consolidated.csv"
    consolidated_rows = None
    if consolidated_path.exists():
        consolidated_rows = int(len(pd.read_csv(consolidated_path, usecols=["smiles"])))

    summary = {
        "raw_files": [str(raw_dir / "groupsplit_train.csv"), str(raw_dir / "groupsplit_test.csv")],
        "effect_value_column": "Effect value",
        "censor_patterns": [">", "<", ">=", "<="],
        "total_raw_records": int(len(raw)),
        "total_censored_records_any_endpoint_duration": int(raw["is_censored"].sum()),
        "lc50_96_raw_records": int(len(lc50_96)),
        "lc50_96_exact_numeric_raw_records": int(len(lc50_96_exact)),
        "lc50_96_censored_records_excluded_from_main": int(len(lc50_96_censored)),
        "lc50_96_censored_species": int(lc50_96_censored["Latin name"].nunique(dropna=True)),
        "lc50_96_censored_compounds_cas": int(lc50_96_censored["CAS"].nunique(dropna=True)),
        "lc50_96_exact_numeric_species": int(lc50_96_exact["Latin name"].nunique(dropna=True)),
        "lc50_96_exact_numeric_compounds_cas": int(lc50_96_exact["CAS"].nunique(dropna=True)),
        "main_consolidated_rows_after_smiles_species_aggregation": consolidated_rows,
        "main_analysis_policy": "Use exact numeric LC50 96h records only; censored records are excluded.",
    }

    _write_csv(
        by_endpoint_duration,
        out_dir / "censored_by_endpoint_duration.csv",
        ["Test statistic", "Duration (hours)", "censor_operator", "n_records"],
    )
    _write_csv(
        by_operator,
        out_dir / "lc50_96_censored_by_operator.csv",
        ["censor_operator", "n_records"],
    )
    _write_csv(
        by_species,
        out_dir / "lc50_96_censored_by_species.csv",
        ["Latin name", "n_records"],
    )
    _write_csv(
        by_compound,
        out_dir / "lc50_96_censored_by_compound.csv",
        ["CAS", "Canonical SMILES", "n_records"],
    )
    with open(out_dir / "censored_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

