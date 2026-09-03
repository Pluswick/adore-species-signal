from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields, replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.runner import V3RunConfig, run_v3_smoke


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _base_config(raw: dict) -> V3RunConfig:
    aliases = {
        "dataset_path": "data_dir",
        "max_epochs": "epochs",
        "train_subset_size": "limit_train",
        "test_subset_size": "limit_test",
    }
    normalized = dict(raw)
    for src, dst in aliases.items():
        if src in normalized and dst not in normalized:
            normalized[dst] = normalized[src]
    valid = {field.name for field in fields(V3RunConfig)}
    kwargs = {key: value for key, value in normalized.items() if key in valid}
    kwargs.setdefault("backbone", "dmpnn")
    kwargs.setdefault("variant", "true_species_late_fusion")
    kwargs["save_species_artifacts"] = bool(raw.get("save_species_artifacts", True))
    return V3RunConfig(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v3_embedding_smoke.json")
    args = parser.parse_args()

    raw = _load_config(Path(args.config))
    base = _base_config(raw)
    backbones = raw.get("backbone_list") or raw.get("backbones") or ["dmpnn", "graphconv"]
    species_modes = raw.get("species_mode_list") or raw.get("species_modes")
    if not species_modes:
        raise ValueError("species_mode_list is required")

    out_root = Path(base.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "embedding_smoke_config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    rows = []
    for backbone in backbones:
        for species_mode in species_modes:
            cfg = replace(base, backbone=backbone, variant=species_mode, out_root=str(out_root))
            result = run_v3_smoke(cfg)
            run_cfg = result["config"]
            artifacts = run_cfg.get("species_artifact_files") or {}
            rows.append(
                {
                    "run_id": run_cfg["run_id"],
                    "backbone": backbone,
                    "species_mode": species_mode,
                    "prediction_file": run_cfg["prediction_file"],
                    "run_json": str(out_root / "runs" / f"{run_cfg['run_id']}.json"),
                    "species_embeddings": artifacts.get("species_embeddings"),
                    "species_embedding_metadata": artifacts.get("species_embedding_metadata"),
                    "species_bias": artifacts.get("species_bias"),
                    "trainable_params": result["D"]["trainable_params"],
                    "species_trainable_params": result["D"]["species_trainable_params"],
                    "epochs": result["D"]["epochs"],
                }
            )
            print("done", run_cfg["run_id"])

    summary = pd.DataFrame(rows)
    path = out_root / "embedding_smoke_summary.csv"
    summary.to_csv(path, index=False, encoding="utf-8")
    print(f"summary: {path}")


if __name__ == "__main__":
    main()
