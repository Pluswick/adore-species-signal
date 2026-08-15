from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v3_full_experiment.json"
DEFAULT_OUT = ROOT / "results" / "jcim_v3" / "full_run_plan"


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dedupe(items: list[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _smoke_runtime_reference() -> dict:
    roots = [
        ROOT / "results" / "jcim_v3" / "smoke" / "jcim_v3_env" / "controls" / "control_smoke_summary.csv",
        ROOT / "results" / "jcim_v3" / "smoke" / "jcim_v3_env" / "injection_positions" / "injection_smoke_summary.csv",
        ROOT / "results" / "jcim_v3" / "smoke" / "jcim_v3_env" / "embedding_smoke" / "embedding_smoke_summary.csv",
    ]
    seconds = []
    for path in roots:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for col in ["train_sec", "elapsed_sec"]:
            if col in df:
                seconds.extend(float(v) for v in df[col].dropna().tolist())
    if not seconds:
        return {"available": False}
    return {
        "available": True,
        "mean_train_sec_per_smoke_run": round(sum(seconds) / len(seconds), 3),
        "n_smoke_runtime_rows": len(seconds),
    }


def _estimate_storage(prediction_files: int, embedding_files: int, metadata_files: int, bias_files: int) -> dict:
    return {
        "prediction_csv_count": prediction_files,
        "embedding_csv_count": embedding_files,
        "embedding_metadata_json_count": metadata_files,
        "species_bias_csv_count": bias_files,
        "rough_note": "Storage depends on full test-set size and CSV precision; use this as a file-count estimate, not a byte-accurate estimate.",
    }


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# JCIM v3 Full Run Estimate",
        "",
        f"- Config: `{payload['config_path']}`",
        f"- Generated at UTC: `{payload['generated_at_utc']}`",
        f"- GNN runs: `{payload['gnn_runs']}`",
        f"- LightGBM runs: `{payload['lightgbm_runs']}`",
        f"- Total runs: `{payload['total_runs']}`",
        f"- Expected prediction CSV files: `{payload['expected_prediction_csv_files']}`",
        f"- Expected embedding CSV files: `{payload['expected_embedding_csv_files']}`",
        f"- Expected species-bias CSV files: `{payload['expected_species_bias_csv_files']}`",
        "",
        "## Dimensions",
        "",
        f"- Splits: `{payload['dimensions']['splits']}`",
        f"- Seeds: `{payload['dimensions']['seeds']}`",
        f"- Backbones: `{payload['dimensions']['backbones']}`",
        f"- GNN species modes: `{payload['dimensions']['gnn_species_modes']}`",
        f"- LightGBM baselines: `{payload['dimensions']['lightgbm_baselines']}`",
        "",
        "This is a planning artifact only. It does not execute training.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)

    splits = list(config.get("split_types") or config.get("splits") or [config.get("split", "scaffold")])
    seeds = [int(value) for value in config.get("seeds", [config.get("seed", 0)])]
    backbones = list(config.get("backbones") or config.get("backbone_list") or ["dmpnn", "graphconv"])
    main_modes = list(config.get("main_species_modes", []))
    control_modes = list(config.get("control_species_modes", []))
    species_modes = _dedupe(main_modes + control_modes)
    baselines = list(config.get("lightgbm_baselines", []))

    gnn_runs = len(splits) * len(seeds) * len(backbones) * len(species_modes)
    lightgbm_runs = len(splits) * len(seeds) * len(baselines)
    total_runs = gnn_runs + lightgbm_runs
    modes_with_embeddings = [
        mode for mode in species_modes if mode not in {"no_species", "species_bias_only"}
    ]
    embedding_files = len(splits) * len(seeds) * len(backbones) * len(modes_with_embeddings)
    bias_files = len(splits) * len(seeds) * len(backbones) * (1 if "species_bias_only" in species_modes else 0)
    metadata_files = gnn_runs
    runtime_ref = _smoke_runtime_reference()
    runtime_estimate = {"available": False}
    if runtime_ref.get("available"):
        runtime_estimate = {
            "available": True,
            "smoke_reference": runtime_ref,
            "linear_smoke_runtime_sec_lower_bound": round(
                runtime_ref["mean_train_sec_per_smoke_run"] * gnn_runs,
                2,
            ),
            "note": "Full runtime will be much larger than this lower bound because full data and up to 100 epochs are used.",
        }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "dimensions": {
            "splits": splits,
            "seeds": seeds,
            "backbones": backbones,
            "gnn_species_modes": species_modes,
            "lightgbm_baselines": baselines,
        },
        "gnn_runs": gnn_runs,
        "lightgbm_runs": lightgbm_runs,
        "total_runs": total_runs,
        "expected_prediction_csv_files": total_runs,
        "expected_embedding_csv_files": embedding_files,
        "expected_embedding_metadata_json_files": metadata_files,
        "expected_species_bias_csv_files": bias_files,
        "storage_file_count_estimate": _estimate_storage(total_runs, embedding_files, metadata_files, bias_files),
        "runtime_estimate": runtime_estimate,
        "note": "No training was executed.",
    }
    json_path = out_dir / "full_run_estimate.json"
    md_path = out_dir / "full_run_estimate.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_markdown(md_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
