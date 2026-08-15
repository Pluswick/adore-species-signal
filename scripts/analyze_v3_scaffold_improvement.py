from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from jcim_v3.bootstrap import align_predictions
from jcim_v3.paths import RESULTS_ROOT


MERGE_KEYS = [
    "smiles",
    "species",
    "compound_key",
    "scaffold_key",
    "true_log10",
    "split",
    "seed",
]


def _find_prediction(primary_dir: Path, fallback_dir: Path | None, filename: str) -> Path | None:
    primary = primary_dir / filename
    if primary.exists():
        return primary
    if fallback_dir is not None:
        fallback = fallback_dir / filename
        if fallback.exists():
            return fallback
    return None


def _prediction_index(prediction_dirs: list[Path]) -> dict[tuple[str, str, str, int], Path]:
    out: dict[tuple[str, str, str, int], Path] = {}
    for prediction_dir in prediction_dirs:
        if not prediction_dir.exists():
            continue
        for path in sorted(prediction_dir.glob("*.csv")):
            try:
                row = pd.read_csv(path, nrows=1).iloc[0]
            except Exception:
                continue
            if "backbone" not in row or "species_mode" not in row:
                continue
            key = (
                str(row["backbone"]),
                str(row["species_mode"]),
                str(row.get("split", "scaffold")),
                int(row.get("seed", 0)),
            )
            out.setdefault(key, path)
    return out


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _rmse(arr: pd.Series) -> float:
    values = arr.to_numpy(np.float64)
    return float(math.sqrt(np.mean(values**2))) if len(values) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v3_embedding_smoke.json")
    parser.add_argument(
        "--prediction-dir",
        default=str(RESULTS_ROOT / "smoke" / "jcim_v3_env" / "embedding_smoke" / "predictions"),
    )
    parser.add_argument(
        "--reference-prediction-dir",
        default=str(RESULTS_ROOT / "smoke" / "jcim_v3_env" / "injection_positions" / "predictions"),
    )
    parser.add_argument(
        "--out",
        default=str(
            RESULTS_ROOT
            / "smoke"
            / "jcim_v3_env"
            / "embedding_smoke"
            / "scaffold_improvement_summary.csv"
        ),
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    prediction_dir = Path(config.get("prediction_root", args.prediction_dir))
    reference_prediction_dir = Path(
        config.get("reference_prediction_root", args.reference_prediction_dir)
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    primary_index = _prediction_index([prediction_dir])
    reference_index = _prediction_index([prediction_dir, reference_prediction_dir])
    candidate_items = sorted(
        (key, path)
        for key, path in primary_index.items()
        if key[1] == "true_species_late_fusion"
    )
    if not candidate_items:
        for backbone in config.get("backbone_list", ["dmpnn", "graphconv"]):
            cand_path = _find_prediction(
                prediction_dir,
                None,
                f"{backbone}_true_species_late_fusion_scaffold_s0_e1_n512.csv",
            )
            if cand_path is not None:
                candidate_items.append(((backbone, "true_species_late_fusion", "scaffold", 0), cand_path))

    rows = []
    warnings = []
    for (backbone, _, split, seed), cand_path in candidate_items:
        ref_path = reference_index.get((backbone, "no_species", split, seed))
        if ref_path is None:
            ref_path = _find_prediction(
                prediction_dir,
                reference_prediction_dir,
                f"{backbone}_no_species_{split}_s{seed}_e1_n512.csv",
            )
        if cand_path is None or ref_path is None:
            warnings.append(
                {
                    "backbone": backbone,
                    "split": split,
                    "seed": seed,
                    "warning": "candidate or no_species prediction file missing",
                    "candidate_dir": str(prediction_dir),
                    "reference_dir": str(reference_prediction_dir),
                }
            )
            continue
        candidate = pd.read_csv(cand_path)
        reference = pd.read_csv(ref_path)
        pair = align_predictions(candidate, reference, merge_keys=MERGE_KEYS)
        if pair.warning or pair.paired is None:
            warnings.append({"backbone": backbone, "split": split, "seed": seed, "warning": pair.warning})
            continue
        paired = pair.paired.copy()
        paired["error_candidate"] = paired["pred_log10_candidate"] - paired["true_log10"]
        paired["error_reference"] = paired["pred_log10_reference"] - paired["true_log10"]
        grouped = paired.groupby("scaffold_key")
        for scaffold, group in grouped:
            rmse_ref = _rmse(group["error_reference"])
            rmse_cand = _rmse(group["error_candidate"])
            mae_ref = float(np.mean(np.abs(group["error_reference"])))
            mae_cand = float(np.mean(np.abs(group["error_candidate"])))
            rows.append(
                {
                    "backbone": backbone,
                    "split": split,
                    "seed": seed,
                    "candidate_species_mode": "true_species_late_fusion",
                    "reference_species_mode": "no_species",
                    "scaffold_key": scaffold,
                    "n_rows": int(len(group)),
                    "rmse_reference": rmse_ref,
                    "rmse_candidate": rmse_cand,
                    "delta_rmse": rmse_cand - rmse_ref,
                    "mean_abs_error_reference": mae_ref,
                    "mean_abs_error_candidate": mae_cand,
                    "delta_mae": mae_cand - mae_ref,
                    "species_count": int(group["species"].nunique()),
                    "compound_count": int(group["compound_key"].nunique()),
                }
            )
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    warnings_path = out.with_name("scaffold_improvement_warnings.json")
    with open(warnings_path, "w", encoding="utf-8") as f:
        json.dump({"warnings": warnings}, f, ensure_ascii=False, indent=2)
    print(json.dumps({"out": str(out), "rows": len(rows), "warnings": warnings}, indent=2))


if __name__ == "__main__":
    main()
