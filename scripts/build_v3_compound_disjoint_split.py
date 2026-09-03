from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _choose_test_keys(df: pd.DataFrame, key: str, test_frac: float, seed: int) -> set[str]:
    counts = df.groupby(key, dropna=False).size().reset_index(name="n_rows")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(counts))
    target_rows = int(round(len(df) * test_frac))
    selected: set[str] = set()
    n_selected_rows = 0
    for idx in order:
        row = counts.iloc[int(idx)]
        selected.add(str(row[key]))
        n_selected_rows += int(row["n_rows"])
        if n_selected_rows >= target_rows:
            break
    return selected


def _summary(train: pd.DataFrame, test: pd.DataFrame, key: str) -> dict:
    train_keys = set(train[key].dropna().astype(str))
    test_keys = set(test[key].dropna().astype(str))
    overlap = train_keys & test_keys
    return {
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_total_rows": int(len(train) + len(test)),
        "n_train_compounds": int(len(train_keys)),
        "n_test_compounds": int(len(test_keys)),
        "n_total_compounds": int(len(train_keys | test_keys)),
        "n_overlap_compounds": int(len(overlap)),
        "test_row_fraction": float(len(test) / (len(train) + len(test))) if len(train) + len(test) else None,
        "test_compound_fraction": float(len(test_keys) / len(train_keys | test_keys)) if train_keys or test_keys else None,
    }


def _species_index(df: pd.DataFrame) -> pd.DataFrame:
    required = {"species", "species_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing species index columns: {sorted(missing)}")

    species = df[["species", "species_idx"]].drop_duplicates().copy()
    species["species_idx"] = species["species_idx"].astype(int)
    duplicated_idx = species[species["species_idx"].duplicated(keep=False)].sort_values("species_idx")
    if not duplicated_idx.empty:
        raise ValueError(
            "species_idx maps to multiple species names: "
            + duplicated_idx.head(10).to_dict(orient="records").__repr__()
        )
    return species.sort_values("species_idx").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="<USER_HOME>/Desktop/CCLABS/CC-MPNN/data/lc50_96_consolidated.csv")
    parser.add_argument("--out-dir", default="results/src/clean_splits")
    parser.add_argument("--split-name", default="compound_random")
    parser.add_argument("--key", default="smiles")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source)
    if args.key not in df.columns:
        raise ValueError(f"Missing split key column: {args.key}")
    if df[args.key].isna().any():
        raise ValueError(f"Split key has missing values: {args.key}")

    key_as_str = df[args.key].astype(str)
    test_keys = _choose_test_keys(df.assign(_split_key=key_as_str), "_split_key", args.test_frac, args.seed)
    is_test = key_as_str.isin(test_keys)
    train = df.loc[~is_test].copy()
    test = df.loc[is_test].copy()

    train_path = out_dir / f"{args.split_name}_train.csv"
    test_path = out_dir / f"{args.split_name}_test.csv"
    train.to_csv(train_path, index=False, encoding="utf-8")
    test.to_csv(test_path, index=False, encoding="utf-8")
    species_index_path = out_dir / "species_index.csv"
    _species_index(df).to_csv(species_index_path, index=False, encoding="utf-8")

    payload = {
        "source": str(source),
        "out_dir": str(out_dir),
        "split_name": args.split_name,
        "key": args.key,
        "test_frac_requested": args.test_frac,
        "seed": args.seed,
        "summary": _summary(train, test, args.key),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "species_index_path": str(species_index_path),
    }
    summary_json = out_dir / f"{args.split_name}_summary.json"
    summary_md = out_dir / f"{args.split_name}_summary.md"
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_md.write_text(
        "\n".join(
            [
                f"# {args.split_name} Split Summary",
                "",
                f"- Source: `{source}`",
                f"- Split key: `{args.key}`",
                f"- Seed: `{args.seed}`",
                f"- Requested test fraction: `{args.test_frac}`",
                f"- Train path: `{train_path}`",
                f"- Test path: `{test_path}`",
                f"- Species index path: `{species_index_path}`",
                "",
                "## Summary",
                "",
                pd.DataFrame([payload["summary"]]).to_markdown(index=False),
                "",
                "This split is compound-disjoint by the selected key. It is intended for clean-random follow-up experiments.",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
