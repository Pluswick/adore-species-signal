from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.paths import CC_MPNN_DATA, RESULTS_ROOT
from src.stats import prediction_metrics


GROUP_COLUMNS = [
    "model_name",
    "backbone",
    "species_mode",
    "injection_location",
    "seed",
    "split",
    "species_control_type",
]

BIN_ORDER = ["cold", "few", "mid", "rich"]


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _metadata(df: pd.DataFrame, path: Path) -> dict:
    row = df.iloc[0]
    out = {"prediction_file": str(path)}
    for col in GROUP_COLUMNS:
        value = row[col] if col in df.columns else None
        if hasattr(value, "item"):
            value = value.item()
        out[col] = value
    return out


def _bin_species_count(count: int) -> str:
    if count == 0:
        return "cold"
    if 1 <= count <= 4:
        return "few"
    if 5 <= count <= 20:
        return "mid"
    return "rich"


def _species_train_counts(data_dir: Path, split: str) -> dict[int, int]:
    train_path = data_dir / f"{split}_train.csv"
    train = pd.read_csv(train_path)
    return train["species_idx"].astype(int).value_counts().to_dict()


def _with_species_bins(df: pd.DataFrame, data_dir: Path, split: str) -> pd.DataFrame:
    counts = _species_train_counts(data_dir, split)
    species_col = "species_idx_original" if "species_idx_original" in df.columns else "species_idx"
    out = df.copy()
    out["species_train_count"] = out[species_col].astype(int).map(counts).fillna(0).astype(int)
    out["species_count_bin"] = out["species_train_count"].map(_bin_species_count)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-dir",
        default=str(RESULTS_ROOT / "smoke" / "src_env" / "injection_positions" / "predictions"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(RESULTS_ROOT / "smoke" / "src_env" / "stats_smoke"),
    )
    parser.add_argument("--config")
    parser.add_argument("--data-dir", default=str(CC_MPNN_DATA))
    args = parser.parse_args()

    config = _load_config(args.config)
    prediction_dir = Path(config.get("prediction_root", args.prediction_dir))
    out_dir = Path(config.get("output_root", args.out_dir))
    data_dir = Path(config.get("dataset_path", args.data_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    bin_records = []
    warnings = []
    prediction_files = sorted(prediction_dir.glob("*.csv"))
    for path in prediction_files:
        df = pd.read_csv(path)
        meta = _metadata(df, path)
        metrics = prediction_metrics(df)
        records.append({**meta, **metrics})

        split = str(meta["split"])
        binned = _with_species_bins(df, data_dir, split)
        present_bins = set(binned["species_count_bin"])
        missing_bins = [name for name in BIN_ORDER if name not in present_bins]
        if missing_bins:
            warnings.append(
                {
                    "prediction_file": str(path),
                    "missing_species_count_bins": missing_bins,
                }
            )
        for bin_name in BIN_ORDER:
            part = binned[binned["species_count_bin"] == bin_name]
            if len(part) == 0:
                continue
            bin_records.append(
                {
                    **meta,
                    "species_count_bin": bin_name,
                    **prediction_metrics(part),
                }
            )

    metrics_df = pd.DataFrame(records)
    bin_df = pd.DataFrame(bin_records)
    metrics_path = out_dir / "aggregated_metrics.csv"
    bins_path = out_dir / "species_bin_metrics.csv"
    json_path = out_dir / "aggregated_metrics.json"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
    bin_df.to_csv(bins_path, index=False, encoding="utf-8")
    payload = {
        "prediction_dir": str(prediction_dir),
        "out_dir": str(out_dir),
        "n_prediction_files": len(prediction_files),
        "aggregated_metrics_file": str(metrics_path),
        "species_bin_metrics_file": str(bins_path),
        "warnings": warnings,
        "records": records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: payload[k] for k in payload if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
