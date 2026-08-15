from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RAW_COLUMNS = [
    "CAS",
    "Latin name",
    "Duration (hours)",
    "Effect value",
    "Test statistic",
    "Canonical SMILES",
]


def _load_raw(raw_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for partition, name in [("train", "groupsplit_train.csv"), ("test", "groupsplit_test.csv")]:
        path = raw_dir / name
        df = pd.read_csv(path, usecols=RAW_COLUMNS, dtype=str)
        df["yuan_partition"] = partition
        df["yuan_source_file"] = name
        df["yuan_source_row_id"] = range(len(df))
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["CAS"] = raw["CAS"].astype(str).str.strip()
    raw["Canonical SMILES"] = raw["Canonical SMILES"].astype(str).str.strip()
    raw["Latin name"] = raw["Latin name"].astype(str).str.strip()
    raw["Test statistic"] = raw["Test statistic"].astype(str).str.strip()
    raw["duration_numeric"] = pd.to_numeric(raw["Duration (hours)"], errors="coerce")
    raw["effect_value_numeric"] = pd.to_numeric(raw["Effect value"], errors="coerce")
    return raw


def _valid_smiles_mask(smiles: pd.Series) -> tuple[pd.Series, str | None]:
    try:
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception as exc:  # pragma: no cover - environment dependent
        return pd.Series([True] * len(smiles), index=smiles.index), repr(exc)

    def ok(value: object) -> bool:
        if pd.isna(value):
            return False
        try:
            return Chem.MolFromSmiles(str(value)) is not None
        except Exception:
            return False

    return smiles.map(ok), None


def _partition_label(values: pd.Series) -> str:
    parts = sorted(set(values.dropna().astype(str)))
    if parts == ["test"]:
        return "test"
    if parts == ["train"]:
        return "train"
    if parts == ["test", "train"]:
        return "both"
    return "|".join(parts)


def _join_values(values: pd.Series, limit: int = 50) -> str:
    vals = sorted(v for v in set(values.dropna().astype(str)) if v and v.lower() != "nan")
    shown = vals[:limit]
    suffix = "" if len(vals) <= limit else f";...(+{len(vals) - limit})"
    return ";".join(shown) + suffix


def _stage_counts(raw: pd.DataFrame, valid_mask: pd.Series, expected_total: int | None) -> pd.DataFrame:
    masks = {
        "raw": pd.Series([True] * len(raw), index=raw.index),
        "lc50": raw["Test statistic"].eq("LC50"),
        "lc50_96h": raw["Test statistic"].eq("LC50") & raw["duration_numeric"].eq(96.0),
        "lc50_96h_numeric_effect": raw["Test statistic"].eq("LC50")
        & raw["duration_numeric"].eq(96.0)
        & raw["effect_value_numeric"].notna(),
        "lc50_96h_positive_effect": raw["Test statistic"].eq("LC50")
        & raw["duration_numeric"].eq(96.0)
        & raw["effect_value_numeric"].gt(0),
        "lc50_96h_positive_valid_smiles": raw["Test statistic"].eq("LC50")
        & raw["duration_numeric"].eq(96.0)
        & raw["effect_value_numeric"].gt(0)
        & valid_mask,
    }
    rows: list[dict] = []
    for stage, mask in masks.items():
        sub = raw.loc[mask].copy()
        rows.append(
            {
                "stage": stage,
                "partition": "all",
                "n_rows": int(len(sub)),
                "n_cas": int(sub["CAS"].nunique(dropna=True)),
                "n_smiles": int(sub["Canonical SMILES"].nunique(dropna=True)),
                "n_species": int(sub["Latin name"].nunique(dropna=True)),
                "expected_total_raw_records": expected_total if stage == "raw" else None,
                "delta_vs_expected_total": int(len(sub) - expected_total)
                if stage == "raw" and expected_total is not None
                else None,
            }
        )
        for partition in ["train", "test"]:
            part = sub[sub["yuan_partition"].eq(partition)]
            rows.append(
                {
                    "stage": stage,
                    "partition": partition,
                    "n_rows": int(len(part)),
                    "n_cas": int(part["CAS"].nunique(dropna=True)),
                    "n_smiles": int(part["Canonical SMILES"].nunique(dropna=True)),
                    "n_species": int(part["Latin name"].nunique(dropna=True)),
                    "expected_total_raw_records": None,
                    "delta_vs_expected_total": None,
                }
            )
    return pd.DataFrame(rows)


def _aggregated_membership(filtered: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (smiles, species), group in filtered.groupby(["Canonical SMILES", "Latin name"], dropna=False):
        rows.append(
            {
                "smiles": smiles,
                "species": species,
                "source_n_rows": int(len(group)),
                "source_n_cas": int(group["CAS"].nunique(dropna=True)),
                "source_partitions": _partition_label(group["yuan_partition"]),
                "source_crosses_yuan_partition": bool(group["yuan_partition"].nunique(dropna=True) > 1),
                "source_cas_all": _join_values(group["CAS"]),
                "source_cas_train": _join_values(group.loc[group["yuan_partition"].eq("train"), "CAS"]),
                "source_cas_test": _join_values(group.loc[group["yuan_partition"].eq("test"), "CAS"]),
            }
        )
    return pd.DataFrame(rows)


def _smiles_cas_conflicts(filtered: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for smiles, group in filtered.groupby("Canonical SMILES", dropna=False):
        n_cas = int(group["CAS"].nunique(dropna=True))
        if n_cas < 2:
            continue
        train = group[group["yuan_partition"].eq("train")]
        test = group[group["yuan_partition"].eq("test")]
        rows.append(
            {
                "smiles": smiles,
                "n_rows": int(len(group)),
                "n_cas": n_cas,
                "n_species": int(group["Latin name"].nunique(dropna=True)),
                "n_smiles_species_pairs": int(
                    group[["Canonical SMILES", "Latin name"]].drop_duplicates().shape[0]
                ),
                "source_partitions": _partition_label(group["yuan_partition"]),
                "crosses_yuan_partition": bool(group["yuan_partition"].nunique(dropna=True) > 1),
                "n_train_rows": int(len(train)),
                "n_test_rows": int(len(test)),
                "train_cas": _join_values(train["CAS"]),
                "test_cas": _join_values(test["CAS"]),
                "all_cas": _join_values(group["CAS"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["crosses_yuan_partition", "n_cas", "n_rows"], ascending=[False, False, False]
    )


def _pair_crossings(filtered: pd.DataFrame) -> pd.DataFrame:
    membership = _aggregated_membership(filtered)
    return membership[membership["source_crosses_yuan_partition"]].sort_values(
        ["source_n_cas", "source_n_rows"], ascending=[False, False]
    )


def _current_consolidated_membership(consolidated_path: Path, membership: pd.DataFrame) -> pd.DataFrame:
    current = pd.read_csv(consolidated_path, dtype={"CAS": str})
    merged = current.merge(membership, on=["smiles", "species"], how="left", validate="one_to_one")
    return merged[
        [
            "smiles",
            "species",
            "species_idx",
            "target_log10",
            "effect_value",
            "CAS",
            "source_n_rows",
            "source_n_cas",
            "source_partitions",
            "source_crosses_yuan_partition",
            "source_cas_all",
            "source_cas_train",
            "source_cas_test",
        ]
    ]


def _overlap(values_a: set[str], values_b: set[str]) -> dict:
    overlap = values_a & values_b
    return {
        "n_train_unique": int(len(values_a)),
        "n_test_unique": int(len(values_b)),
        "n_overlap_unique": int(len(overlap)),
        "overlap_values_preview": sorted(overlap)[:50],
    }


def _clean_split_cas_audit(clean_dir: Path, membership: pd.DataFrame) -> dict | None:
    train_path = clean_dir / "compound_random_train.csv"
    test_path = clean_dir / "compound_random_test.csv"
    if not train_path.exists() or not test_path.exists():
        return None

    train = pd.read_csv(train_path, dtype={"CAS": str})
    test = pd.read_csv(test_path, dtype={"CAS": str})
    current_cas = _overlap(
        set(train["CAS"].dropna().astype(str)),
        set(test["CAS"].dropna().astype(str)),
    )

    source_map = {
        (row.smiles, row.species): set(str(row.source_cas_all).split(";"))
        for row in membership.itertuples(index=False)
    }

    def source_cas_for(frame: pd.DataFrame) -> set[str]:
        out: set[str] = set()
        for row in frame[["smiles", "species"]].itertuples(index=False):
            for cas in source_map.get((row.smiles, row.species), set()):
                if cas and cas.lower() != "nan":
                    out.add(cas)
        return out

    source_cas = _overlap(source_cas_for(train), source_cas_for(test))
    smiles = _overlap(set(train["smiles"].astype(str)), set(test["smiles"].astype(str)))
    return {
        "clean_split_dir": str(clean_dir),
        "smiles_overlap": smiles,
        "current_consolidated_CAS_overlap": current_cas,
        "source_derived_CAS_overlap": source_cas,
    }


def _write_markdown(
    path: Path,
    *,
    raw_dir: Path,
    consolidated_path: Path,
    counts: pd.DataFrame,
    conflicts: pd.DataFrame,
    pair_crossings: pd.DataFrame,
    current_membership: pd.DataFrame,
    clean_audit: dict | None,
    rdkit_error: str | None,
) -> None:
    raw_all = counts[(counts["stage"].eq("raw")) & (counts["partition"].eq("all"))].iloc[0]
    lc50_96 = counts[(counts["stage"].eq("lc50_96h")) & (counts["partition"].eq("all"))].iloc[0]
    valid = counts[(counts["stage"].eq("lc50_96h_positive_valid_smiles")) & (counts["partition"].eq("all"))].iloc[0]
    affected = current_membership[current_membership["source_crosses_yuan_partition"].fillna(False)]
    lines = [
        "# Yuan tox-learn Source Partition Audit",
        "",
        f"- Raw tox-learn directory: `{raw_dir}`",
        f"- Current consolidated file: `{consolidated_path}`",
        f"- RDKit parse check: `{'skipped: ' + rdkit_error if rdkit_error else 'available'}`",
        "",
        "## Key Counts",
        "",
        f"- Raw rows observed: `{int(raw_all.n_rows)}`",
        f"- LC50 96h rows before positivity/SMILES checks: `{int(lc50_96.n_rows)}`",
        f"- LC50 96h positive valid-SMILES rows: `{int(valid.n_rows)}`",
        f"- Aggregated current rows: `{len(current_membership)}`",
        f"- Canonical SMILES with >=2 CAS after LC50 96h filtering: `{len(conflicts)}`",
        f"- Canonical SMILES with CAS crossing Yuan train/test partitions: `{int(conflicts['crosses_yuan_partition'].sum()) if not conflicts.empty else 0}`",
        f"- `(SMILES, species)` groups crossing Yuan train/test partitions: `{len(pair_crossings)}`",
        f"- Current consolidated rows sourced from both Yuan partitions: `{len(affected)}`",
        "",
        "## Interpretation",
        "",
        "- LC50/96h filtering alone would not break Yuan's CAS-group partition.",
        "- Any `(SMILES, species)` group with `source_partitions = both` was created by merging source rows across Yuan train/test partitions during local consolidation.",
        "- Any SMILES with train-only CAS and test-only CAS indicates that SMILES-based compound identity can collapse multiple Yuan CAS groups.",
        "- These rows should be handled by a Yuan-partition-preserving reconstruction before using Yuan group split as evidence.",
        "",
        "## Clean Compound-Random Split CAS Check",
        "",
    ]
    if clean_audit is None:
        lines.append("- Clean split files were not found; CAS overlap audit skipped.")
    else:
        lines.extend(
            [
                f"- SMILES overlap: `{clean_audit['smiles_overlap']['n_overlap_unique']}`",
                f"- Current consolidated CAS overlap: `{clean_audit['current_consolidated_CAS_overlap']['n_overlap_unique']}`",
                f"- Source-derived CAS overlap: `{clean_audit['source_derived_CAS_overlap']['n_overlap_unique']}`",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="<DATA_ROOT>/tox-learn")
    parser.add_argument(
        "--consolidated",
        default="<USER_HOME>/Desktop/CCLABS/CC-MPNN/data/lc50_96_consolidated.csv",
    )
    parser.add_argument("--clean-dir", default="results/jcim_v3/clean_splits")
    parser.add_argument("--out-dir", default="results/jcim_v3/data_audit")
    parser.add_argument("--expected-raw-total", type=int, default=50603)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    consolidated_path = Path(args.consolidated)
    clean_dir = Path(args.clean_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_raw(raw_dir)
    valid_mask, rdkit_error = _valid_smiles_mask(raw["Canonical SMILES"])
    counts = _stage_counts(raw, valid_mask, args.expected_raw_total)

    filtered_mask = (
        raw["Test statistic"].eq("LC50")
        & raw["duration_numeric"].eq(96.0)
        & raw["effect_value_numeric"].gt(0)
        & valid_mask
    )
    filtered = raw.loc[filtered_mask].copy()

    membership = _aggregated_membership(filtered)
    conflicts = _smiles_cas_conflicts(filtered)
    crossing_conflicts = conflicts[conflicts["crosses_yuan_partition"]].copy()
    pair_crossings = _pair_crossings(filtered)
    current_membership = _current_consolidated_membership(consolidated_path, membership)
    affected_current = current_membership[
        current_membership["source_crosses_yuan_partition"].fillna(False)
    ].copy()
    clean_audit = _clean_split_cas_audit(clean_dir, membership)

    paths = {
        "filter_counts_csv": out_dir / "yuan_source_filter_counts.csv",
        "cas_smiles_conflicts_csv": out_dir / "yuan_cas_smiles_conflicts.csv",
        "cas_smiles_partition_crossings_csv": out_dir / "yuan_cas_smiles_partition_crossings.csv",
        "smiles_species_partition_crossings_csv": out_dir / "yuan_smiles_species_partition_crossings.csv",
        "consolidated_membership_csv": out_dir / "yuan_consolidated_partition_membership.csv",
        "boundary_affected_consolidated_rows_csv": out_dir
        / "yuan_boundary_affected_consolidated_rows.csv",
        "clean_split_cas_audit_json": out_dir / "yuan_clean_split_cas_overlap_audit.json",
        "summary_json": out_dir / "yuan_source_partition_audit_summary.json",
        "summary_md": out_dir / "yuan_source_partition_audit_summary.md",
    }

    counts.to_csv(paths["filter_counts_csv"], index=False, encoding="utf-8")
    conflicts.to_csv(paths["cas_smiles_conflicts_csv"], index=False, encoding="utf-8")
    crossing_conflicts.to_csv(
        paths["cas_smiles_partition_crossings_csv"], index=False, encoding="utf-8"
    )
    pair_crossings.to_csv(
        paths["smiles_species_partition_crossings_csv"], index=False, encoding="utf-8"
    )
    current_membership.to_csv(paths["consolidated_membership_csv"], index=False, encoding="utf-8")
    affected_current.to_csv(
        paths["boundary_affected_consolidated_rows_csv"], index=False, encoding="utf-8"
    )
    paths["clean_split_cas_audit_json"].write_text(
        json.dumps(clean_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "raw_dir": str(raw_dir),
        "consolidated": str(consolidated_path),
        "rdkit_smiles_check_error": rdkit_error,
        "expected_raw_total": args.expected_raw_total,
        "observed_raw_total": int(len(raw)),
        "delta_vs_expected_raw_total": int(len(raw) - args.expected_raw_total),
        "lc50_96_positive_valid_rows": int(len(filtered)),
        "current_consolidated_rows": int(len(current_membership)),
        "current_consolidated_rows_missing_source_membership": int(
            current_membership["source_n_rows"].isna().sum()
        ),
        "n_smiles_with_multiple_cas": int(len(conflicts)),
        "n_smiles_with_cas_crossing_yuan_partitions": int(len(crossing_conflicts)),
        "n_smiles_species_groups_crossing_yuan_partitions": int(len(pair_crossings)),
        "n_current_consolidated_rows_crossing_yuan_partitions": int(len(affected_current)),
        "clean_split_cas_audit": clean_audit,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(
        paths["summary_md"],
        raw_dir=raw_dir,
        consolidated_path=consolidated_path,
        counts=counts,
        conflicts=conflicts,
        pair_crossings=pair_crossings,
        current_membership=current_membership,
        clean_audit=clean_audit,
        rdkit_error=rdkit_error,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
