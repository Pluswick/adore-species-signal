from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from jcim_v3.paths import CC_MPNN_DATA, RESULTS_ROOT
from jcim_v3.rdkit_features import (
    build_feature_cache,
    write_data_audit_reports,
    write_standardized_split_features,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_info(path: Path) -> dict:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(CC_MPNN_DATA))
    parser.add_argument("--out-root", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    features_dir = out_root / "features"
    audit_dir = out_root / "data_audit"
    features_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    consolidated = pd.read_csv(data_dir / "lc50_96_consolidated.csv")
    desc, desc_fail, scaffolds, scaffold_fail = build_feature_cache(consolidated)

    desc.to_csv(features_dir / "rdkit6_by_smiles.csv", index=False, encoding="utf-8")
    scaffolds.to_csv(features_dir / "scaffold_by_smiles.csv", index=False, encoding="utf-8")
    desc_fail.to_csv(audit_dir / "rdkit_descriptor_failures.csv", index=False, encoding="utf-8")
    scaffold_fail.to_csv(audit_dir / "scaffold_failures.csv", index=False, encoding="utf-8")

    standardization = {}
    for split in ("random", "scaffold"):
        standardization[split] = write_standardized_split_features(
            data_dir=data_dir,
            features_dir=features_dir,
            descriptor_cache=desc,
            split=split,
        )

    write_data_audit_reports(data_dir=data_dir, audit_dir=audit_dir, scaffold_cache=scaffolds)

    generated_files = [
        features_dir / "rdkit6_by_smiles.csv",
        features_dir / "scaffold_by_smiles.csv",
        features_dir / "rdkit6_standardization_random.json",
        features_dir / "rdkit6_standardization_scaffold.json",
        features_dir / "rdkit6_random_train_standardized.csv",
        features_dir / "rdkit6_random_test_standardized.csv",
        features_dir / "rdkit6_scaffold_train_standardized.csv",
        features_dir / "rdkit6_scaffold_test_standardized.csv",
    ]

    summary = {
        "generated_with": {
            "python_executable": sys.executable,
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "n_unique_smiles": int(consolidated["smiles"].nunique()),
        "rdkit_descriptor_success": int(len(desc)),
        "rdkit_descriptor_failures": int(len(desc_fail)),
        "scaffold_success": int(len(scaffolds)),
        "scaffold_failures": int(len(scaffold_fail)),
        "descriptor_cache": str(features_dir / "rdkit6_by_smiles.csv"),
        "scaffold_cache": str(features_dir / "scaffold_by_smiles.csv"),
        "standardization": standardization,
        "generated_files": {path.name: _file_info(path) for path in generated_files},
    }
    with open(features_dir / "feature_cache_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
