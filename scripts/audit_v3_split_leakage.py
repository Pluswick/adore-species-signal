from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _overlap_summary(train: pd.DataFrame, test: pd.DataFrame, split: str, key: str) -> dict:
    train_values = set(train[key].dropna().astype(str))
    test_values = set(test[key].dropna().astype(str))
    overlap = train_values & test_values
    test_key = test[key].dropna().astype(str)
    test_overlap_mask = test_key.isin(overlap)
    return {
        "split": split,
        "key": key,
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_unique": int(len(train_values)),
        "n_test_unique": int(len(test_values)),
        "n_overlap_unique": int(len(overlap)),
        "test_unique_overlap_fraction": float(len(overlap) / len(test_values)) if test_values else None,
        "train_unique_overlap_fraction": float(len(overlap) / len(train_values)) if train_values else None,
        "test_rows_with_train_key": int(test_overlap_mask.sum()),
        "test_row_overlap_fraction": float(test_overlap_mask.mean()) if len(test_key) else None,
    }


def _pair_overlap_summary(train: pd.DataFrame, test: pd.DataFrame, split: str, cols: list[str]) -> dict:
    train_pairs = set(map(tuple, train[cols].astype(str).values.tolist()))
    test_pairs = set(map(tuple, test[cols].astype(str).values.tolist()))
    overlap = train_pairs & test_pairs
    test_rows_with_overlap = sum(tuple(row) in overlap for row in test[cols].astype(str).values.tolist())
    return {
        "split": split,
        "key": "+".join(cols),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_unique": int(len(train_pairs)),
        "n_test_unique": int(len(test_pairs)),
        "n_overlap_unique": int(len(overlap)),
        "test_unique_overlap_fraction": float(len(overlap) / len(test_pairs)) if test_pairs else None,
        "train_unique_overlap_fraction": float(len(overlap) / len(train_pairs)) if train_pairs else None,
        "test_rows_with_train_key": int(test_rows_with_overlap),
        "test_row_overlap_fraction": float(test_rows_with_overlap / len(test)) if len(test) else None,
    }


def _examples(train: pd.DataFrame, test: pd.DataFrame, split: str, key: str, limit: int) -> list[dict]:
    overlap = sorted(set(train[key].dropna().astype(str)) & set(test[key].dropna().astype(str)))
    return [{"split": split, "key": key, "value": value} for value in overlap[:limit]]


def _write_markdown(path: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# JCIM v3 Split Leakage Diagnostics",
        "",
        "This audit checks whether train and test partitions share compound identifiers.",
        "It does not modify raw data or model outputs.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Interpretation Rules",
        "",
        "- `smiles` overlap means the split is not compound-disjoint at the molecular graph level.",
        "- `CAS` overlap means source compound identifiers overlap, even when canonical SMILES differ.",
        "- `smiles+species` overlap checks exact compound-species record duplication.",
        "- Random-split performance with high `smiles` overlap should not be interpreted as chemical-space generalization.",
        "- Scaffold-split performance is the primary current evidence for chemical generalization.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="<USER_HOME>/Desktop/CCLABS/CC-MPNN/data")
    parser.add_argument("--out-dir", default="results/jcim_v3/data_audit")
    parser.add_argument("--example-limit", type=int, default=20)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    example_rows: list[dict] = []
    for split in ["random", "scaffold"]:
        train = pd.read_csv(data_dir / f"{split}_train.csv")
        test = pd.read_csv(data_dir / f"{split}_test.csv")
        for key in ["smiles", "CAS"]:
            rows.append(_overlap_summary(train, test, split, key))
            example_rows.extend(_examples(train, test, split, key, args.example_limit))
        rows.append(_pair_overlap_summary(train, test, split, ["smiles", "species"]))

    summary = pd.DataFrame(rows)
    examples = pd.DataFrame(example_rows)
    summary_path = out_dir / "split_leakage_diagnostics.csv"
    examples_path = out_dir / "split_leakage_overlap_examples.csv"
    json_path = out_dir / "split_leakage_diagnostics.json"
    md_path = out_dir / "split_leakage_diagnostics.md"

    summary.to_csv(summary_path, index=False, encoding="utf-8")
    examples.to_csv(examples_path, index=False, encoding="utf-8")
    json_path.write_text(
        json.dumps({"rows": rows, "examples": example_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_markdown(md_path, summary)

    print(summary.to_string(index=False))
    print(f"wrote {summary_path}")
    print(f"wrote {examples_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
