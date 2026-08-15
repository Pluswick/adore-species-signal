from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default="results/jcim_v3/clean_splits")
    parser.add_argument("--split-name", default="compound_random")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    train_path = split_dir / f"{args.split_name}_train.csv"
    test_path = split_dir / f"{args.split_name}_test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train_species = set(train["species"].astype(str))
    test_species = set(test["species"].astype(str))
    cold_species = test_species - train_species
    test_species_series = test["species"].astype(str)
    train_counts = train["species"].astype(str).value_counts()
    test_counts = test_species_series.value_counts()
    train_count_for_test = test_species_series.map(train_counts).fillna(0).astype(int)

    bins = pd.cut(
        train_count_for_test,
        bins=[-1, 0, 1, 4, 20, float("inf")],
        labels=["cold(0)", "singleton(1)", "few(2-4)", "mid(5-20)", "rich(>20)"],
    )
    bin_summary = (
        pd.DataFrame({"species_train_count_bin": bins})
        .value_counts("species_train_count_bin", sort=False)
        .reset_index(name="n_test_rows")
    )
    bin_summary["fraction_test_rows"] = bin_summary["n_test_rows"] / len(test)

    species_summary = pd.DataFrame(
        {
            "species": sorted(test_species),
            "n_train": [int(train_counts.get(species, 0)) for species in sorted(test_species)],
            "n_test": [int(test_counts.get(species, 0)) for species in sorted(test_species)],
        }
    )
    species_summary["is_cold_species"] = species_summary["n_train"].eq(0)
    species_summary = species_summary.sort_values(["is_cold_species", "n_test", "species"], ascending=[False, False, True])

    payload = {
        "split_name": args.split_name,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_species": int(len(train_species)),
        "n_test_species": int(len(test_species)),
        "n_cold_species": int(len(cold_species)),
        "n_cold_species_test_rows": int(test_species_series.isin(cold_species).sum()),
        "cold_species_test_row_fraction": float(test_species_series.isin(cold_species).mean()),
        "n_shared_species": int(len(train_species & test_species)),
    }

    summary_json = split_dir / f"{args.split_name}_species_coverage_summary.json"
    summary_md = split_dir / f"{args.split_name}_species_coverage_summary.md"
    species_csv = split_dir / f"{args.split_name}_species_coverage_by_species.csv"
    bins_csv = split_dir / f"{args.split_name}_species_coverage_bins.csv"
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    species_summary.to_csv(species_csv, index=False, encoding="utf-8")
    bin_summary.to_csv(bins_csv, index=False, encoding="utf-8")
    summary_md.write_text(
        "\n".join(
            [
                f"# {args.split_name} Species Coverage Summary",
                "",
                f"- Train rows: `{payload['n_train_rows']}`",
                f"- Test rows: `{payload['n_test_rows']}`",
                f"- Train species: `{payload['n_train_species']}`",
                f"- Test species: `{payload['n_test_species']}`",
                f"- Shared species: `{payload['n_shared_species']}`",
                f"- Cold test species: `{payload['n_cold_species']}`",
                f"- Cold-species test rows: `{payload['n_cold_species_test_rows']}`",
                f"- Cold-species test row fraction: `{payload['cold_species_test_row_fraction']:.4f}`",
                "",
                "## Test Rows By Train Species Count Bin",
                "",
                bin_summary.to_markdown(index=False),
                "",
                "## Interpretation",
                "",
                "- Cold species rows should be reported separately for species-aware variants.",
                "- Species-bias-only cannot learn species-specific intercepts for cold species and must fall back to its unseen-species handling.",
                "- This split is compound-disjoint but not species-closed.",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
