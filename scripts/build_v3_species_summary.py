from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from jcim_v3.paths import CC_MPNN_DATA, RESULTS_ROOT


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _bin_species_count(count: int) -> str:
    if count == 0:
        return "cold"
    if 1 <= count <= 4:
        return "few"
    if 5 <= count <= 20:
        return "mid"
    return "rich"


def _rmse(values: pd.Series) -> float:
    arr = values.to_numpy(np.float64)
    return float(math.sqrt(np.mean(arr**2))) if len(arr) else float("nan")


def _base_species_summary(data_dir: Path, split: str) -> pd.DataFrame:
    species_index = pd.read_csv(data_dir / "species_index.csv")
    train = pd.read_csv(data_dir / f"{split}_train.csv")
    test = pd.read_csv(data_dir / f"{split}_test.csv")
    consolidated_path = data_dir / "lc50_96_consolidated.csv"
    if consolidated_path.exists():
        consolidated = pd.read_csv(consolidated_path)
    else:
        consolidated = pd.concat([train, test], ignore_index=True)

    base = species_index.copy()
    base["species_idx"] = base["species_idx"].astype(int)
    base["train_count"] = base["species_idx"].map(train["species_idx"].value_counts()).fillna(0).astype(int)
    base["test_count"] = base["species_idx"].map(test["species_idx"].value_counts()).fillna(0).astype(int)
    base["total_count"] = base["species_idx"].map(consolidated["species_idx"].value_counts()).fillna(0).astype(int)
    base["mean_true_log10_train"] = base["species_idx"].map(train.groupby("species_idx")["target_log10"].mean())
    base["mean_true_log10_test"] = base["species_idx"].map(test.groupby("species_idx")["target_log10"].mean())
    base["species_count_bin"] = base["train_count"].map(_bin_species_count)
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v3_embedding_smoke.json")
    parser.add_argument("--data-dir", default=str(CC_MPNN_DATA))
    parser.add_argument(
        "--prediction-dir",
        default=str(RESULTS_ROOT / "smoke" / "jcim_v3_env" / "embedding_smoke" / "predictions"),
    )
    parser.add_argument(
        "--out",
        default=str(RESULTS_ROOT / "smoke" / "jcim_v3_env" / "embedding_smoke" / "species_summary.csv"),
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    data_dir = Path(config.get("dataset_path", args.data_dir))
    prediction_dir = Path(config.get("prediction_root", args.prediction_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    default_split = str(config.get("split", "scaffold"))
    base_by_split: dict[str, pd.DataFrame] = {}

    rows = []
    for path in sorted(prediction_dir.glob("*.csv")):
        pred = pd.read_csv(path)
        if pred.empty:
            continue
        meta = pred.iloc[0]
        split = str(meta.get("split", default_split))
        if split not in base_by_split:
            base_by_split[split] = _base_species_summary(data_dir, split)
        base = base_by_split[split]
        grouped = pred.groupby("species_idx_original" if "species_idx_original" in pred.columns else "species_idx")
        summary = grouped.agg(
            mean_pred_log10=("pred_log10", "mean"),
            mean_error_log10=("error_log10", "mean"),
            mean_abs_error_log10=("error_log10", lambda s: float(np.mean(np.abs(s)))),
            rmse_by_species=("error_log10", _rmse),
            n_test_rows_for_summary=("error_log10", "size"),
        ).reset_index(names="species_idx")
        frame = base.merge(summary, on="species_idx", how="left")
        frame["backbone"] = meta.get("backbone")
        frame["species_mode"] = meta.get("species_mode")
        frame["injection_location"] = meta.get("injection_location")
        frame["model_name"] = meta.get("model_name")
        frame["seed"] = int(meta.get("seed"))
        frame["split"] = split
        frame["prediction_file"] = str(path)
        frame["n_test_rows_for_summary"] = frame["n_test_rows_for_summary"].fillna(0).astype(int)
        frame["has_enough_test_rows"] = frame["n_test_rows_for_summary"] >= 3
        frame["train_mean_true"] = frame["mean_true_log10_train"]
        frame["test_mean_true"] = frame["mean_true_log10_test"]
        frame["mean_pred"] = frame["mean_pred_log10"]
        frame["mean_error"] = frame["mean_error_log10"]
        frame["mean_abs_error"] = frame["mean_abs_error_log10"]
        rows.append(frame)

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    result.to_csv(out, index=False, encoding="utf-8")
    print(json.dumps({"out": str(out), "rows": int(len(result)), "prediction_files": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
