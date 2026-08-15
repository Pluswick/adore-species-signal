from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

DESCRIPTOR_NAMES = (
    "LogP",
    "TPSA",
    "MolWt",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
)


def rdkit6_from_smiles(smiles: str) -> tuple[dict[str, float] | None, str | None]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        return None, "invalid_or_empty_mol"
    try:
        return (
            {
                "LogP": float(Descriptors.MolLogP(mol)),
                "TPSA": float(Descriptors.TPSA(mol)),
                "MolWt": float(Descriptors.MolWt(mol)),
                "NumHDonors": float(Descriptors.NumHDonors(mol)),
                "NumHAcceptors": float(Descriptors.NumHAcceptors(mol)),
                "NumRotatableBonds": float(Descriptors.NumRotatableBonds(mol)),
            },
            None,
        )
    except Exception as exc:
        return None, type(exc).__name__ + ": " + str(exc)


def scaffold_from_smiles(smiles: str) -> tuple[str | None, str | None]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        return None, "invalid_or_empty_mol"
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=True), None
    except Exception as exc:
        return None, type(exc).__name__ + ": " + str(exc)


def build_feature_cache(consolidated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    smiles_values = sorted(consolidated["smiles"].dropna().astype(str).unique())
    desc_rows = []
    desc_failures = []
    scaf_rows = []
    scaf_failures = []

    for smiles in smiles_values:
        desc, desc_error = rdkit6_from_smiles(smiles)
        if desc is None:
            desc_failures.append({"smiles": smiles, "reason": desc_error})
        else:
            desc_rows.append({"smiles": smiles, **desc})

        scaffold, scaffold_error = scaffold_from_smiles(smiles)
        if scaffold is None:
            scaf_failures.append({"smiles": smiles, "reason": scaffold_error})
        else:
            scaf_rows.append({"smiles": smiles, "scaffold": scaffold})

    desc_df = pd.DataFrame(desc_rows, columns=["smiles", *DESCRIPTOR_NAMES])
    desc_fail_df = pd.DataFrame(desc_failures, columns=["smiles", "reason"])
    scaf_df = pd.DataFrame(scaf_rows, columns=["smiles", "scaffold"])
    scaf_fail_df = pd.DataFrame(scaf_failures, columns=["smiles", "reason"])
    return desc_df, desc_fail_df, scaf_df, scaf_fail_df


def write_standardized_split_features(
    *,
    data_dir: Path,
    features_dir: Path,
    descriptor_cache: pd.DataFrame,
    split: str,
) -> dict:
    train = pd.read_csv(data_dir / f"{split}_train.csv")
    test = pd.read_csv(data_dir / f"{split}_test.csv")
    train_m = train[["smiles"]].merge(descriptor_cache, on="smiles", how="left")
    test_m = test[["smiles"]].merge(descriptor_cache, on="smiles", how="left")
    descriptor_cols = list(DESCRIPTOR_NAMES)
    if train_m[descriptor_cols].isna().any().any():
        missing = int(train_m[descriptor_cols].isna().any(axis=1).sum())
        raise ValueError(f"{split} train has {missing} rows with missing RDKit descriptors")
    if test_m[descriptor_cols].isna().any().any():
        missing = int(test_m[descriptor_cols].isna().any(axis=1).sum())
        raise ValueError(f"{split} test has {missing} rows with missing RDKit descriptors")

    mean = train_m[descriptor_cols].mean()
    std = train_m[descriptor_cols].std(ddof=0).replace(0.0, 1.0)
    stats = {
        "split": split,
        "fit_partition": "train",
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "mean": {k: float(v) for k, v in mean.items()},
        "std": {k: float(v) for k, v in std.items()},
    }
    with open(features_dir / f"rdkit6_standardization_{split}.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    for partition, frame, merged in (("train", train, train_m), ("test", test, test_m)):
        out = frame[["smiles", "species", "species_idx", "target_log10", "CAS"]].copy()
        raw = merged[descriptor_cols].reset_index(drop=True)
        z = ((raw - mean) / std).add_suffix("_z")
        raw = raw.add_suffix("_raw")
        out = pd.concat([out.reset_index(drop=True), raw, z], axis=1)
        out.to_csv(features_dir / f"rdkit6_{split}_{partition}_standardized.csv", index=False, encoding="utf-8")
    return stats


def write_data_audit_reports(*, data_dir: Path, audit_dir: Path, scaffold_cache: pd.DataFrame) -> None:
    consolidated = pd.read_csv(data_dir / "lc50_96_consolidated.csv")
    species_index = pd.read_csv(data_dir / "species_index.csv")
    split_frames = {}
    for split in ("random", "scaffold"):
        for partition in ("train", "test"):
            key = f"{split}_{partition}"
            split_frames[key] = pd.read_csv(data_dir / f"{key}.csv")

    species = species_index.copy()
    species["n_total"] = species["species_idx"].map(consolidated["species_idx"].value_counts()).fillna(0).astype(int)
    for key, frame in split_frames.items():
        species[f"n_{key}"] = species["species_idx"].map(frame["species_idx"].value_counts()).fillna(0).astype(int)
    species.to_csv(audit_dir / "species_count_summary.csv", index=False, encoding="utf-8")

    compounds = pd.DataFrame({"smiles": sorted(consolidated["smiles"].astype(str).unique())})
    compounds["n_total"] = compounds["smiles"].map(consolidated["smiles"].value_counts()).fillna(0).astype(int)
    compounds["n_species_total"] = compounds["smiles"].map(consolidated.groupby("smiles")["species_idx"].nunique()).fillna(0).astype(int)
    compounds = compounds.merge(scaffold_cache, on="smiles", how="left")
    for key, frame in split_frames.items():
        compounds[f"n_{key}"] = compounds["smiles"].map(frame["smiles"].value_counts()).fillna(0).astype(int)
    compounds.to_csv(audit_dir / "compound_count_summary.csv", index=False, encoding="utf-8")

    split_rows = []
    for split in ("random", "scaffold"):
        train = split_frames[f"{split}_train"]
        test = split_frames[f"{split}_test"]
        train_species = set(train["species_idx"])
        for partition, frame in (("train", train), ("test", test)):
            merged = frame[["smiles"]].merge(scaffold_cache, on="smiles", how="left")
            cold_species = set(frame["species_idx"]) - train_species if partition == "test" else set()
            split_rows.append(
                {
                    "split": split,
                    "partition": partition,
                    "n_records": len(frame),
                    "n_compounds": frame["smiles"].nunique(),
                    "n_species": frame["species_idx"].nunique(),
                    "n_scaffolds": merged["scaffold"].nunique(dropna=True),
                    "n_cold_species_rows": int(frame["species_idx"].isin(cold_species).sum()),
                    "n_cold_species": len(cold_species),
                    "target_log10_min": float(frame["target_log10"].min()),
                    "target_log10_max": float(frame["target_log10"].max()),
                }
            )
    pd.DataFrame(split_rows).to_csv(audit_dir / "split_summary.csv", index=False, encoding="utf-8")
