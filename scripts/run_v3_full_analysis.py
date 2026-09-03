from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v3_full_experiment.json"


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _output_root(config: dict) -> Path:
    return Path(config.get("output_root") or config.get("out_root") or ROOT / "results" / "src" / "full")


def _analysis_config(config: dict, out_root: Path) -> tuple[Path, Path]:
    analysis = config.get("analysis", {})
    metrics_root = Path(analysis.get("metrics_root", out_root / "metrics"))
    bootstrap_root = Path(analysis.get("bootstrap_root", out_root / "bootstrap"))
    embedding_root = Path(analysis.get("embedding_analysis_root", out_root / "embedding_analysis"))
    stats_config = {
        "official_env": config.get("official_env", "src"),
        "dataset_path": config.get("dataset_path"),
        "prediction_root": str(out_root / "predictions"),
        "output_root": str(bootstrap_root),
        "split_types": config.get("split_types", ["random", "scaffold"]),
        "seeds": config.get("seeds", [0, 1, 2, 3, 4]),
        "n_bootstrap": int(analysis.get("n_bootstrap", 2000)),
        "backbones": analysis.get("backbones", config.get("backbones", ["dmpnn", "graphconv"])),
        "metrics": analysis.get("metrics", ["rmse", "mae", "within_2fold", "within_3fold"]),
        "primary_block_keys": analysis.get(
            "primary_block_keys",
            {"random": "compound_key", "scaffold": "scaffold_key"},
        ),
    }
    comparisons = analysis.get("comparisons") or config.get("comparisons")
    if comparisons:
        stats_config["comparisons"] = comparisons
    embedding_config = {
        "official_env": config.get("official_env", "src"),
        "dataset_path": config.get("dataset_path"),
        "prediction_root": str(out_root / "predictions"),
        "reference_prediction_root": str(out_root / "predictions"),
        "out_root": str(embedding_root),
        "embedding_root": str(out_root / "embeddings"),
        "split_types": config.get("split_types", ["random", "scaffold"]),
        "seeds": config.get("seeds", [0, 1, 2, 3, 4]),
        "enable_umap": bool(analysis.get("enable_umap", False)),
    }
    metrics_root.mkdir(parents=True, exist_ok=True)
    bootstrap_root.mkdir(parents=True, exist_ok=True)
    embedding_root.mkdir(parents=True, exist_ok=True)
    stats_path = out_root / "full_analysis_stats_config.json"
    embedding_path = out_root / "full_analysis_embedding_config.json"
    _write_json(stats_path, stats_config)
    _write_json(embedding_path, embedding_config)
    return stats_path, embedding_path


def _run(label: str, args: list[str], log_root: Path) -> dict:
    t0 = time.time()
    proc = subprocess.run([sys.executable, *args], cwd=str(ROOT), text=True, capture_output=True)
    stdout = log_root / f"{label}_stdout.txt"
    stderr = log_root / f"{label}_stderr.txt"
    stdout.write_text(proc.stdout, encoding="utf-8")
    stderr.write_text(proc.stderr, encoding="utf-8")
    return {
        "label": label,
        "command": "python " + " ".join(args),
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "elapsed_sec": round(time.time() - t0, 3),
        "stdout_file": str(stdout),
        "stderr_file": str(stderr),
        "error_message": proc.stderr.strip() if proc.returncode else "",
    }


def _make_summary_tables(out_root: Path, config: dict) -> dict:
    analysis = config.get("analysis", {})
    metrics_root = Path(analysis.get("metrics_root", out_root / "metrics"))
    bootstrap_root = Path(analysis.get("bootstrap_root", out_root / "bootstrap"))
    summary_root = Path(analysis.get("summary_tables_root", out_root / "summary_tables"))
    summary_root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    agg_path = metrics_root / "aggregated_metrics.csv"
    if agg_path.exists():
        agg = pd.read_csv(agg_path)
        group_cols = [col for col in ["backbone", "species_mode", "split"] if col in agg.columns]
        metric_cols = [col for col in ["rmse", "mae", "within_2fold", "within_3fold"] if col in agg.columns]
        if group_cols and metric_cols:
            table = agg.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "count"])
            table.columns = ["_".join(col).strip("_") for col in table.columns.to_flat_index()]
            table = table.reset_index()
            path = summary_root / "metric_summary_by_model.csv"
            table.to_csv(path, index=False, encoding="utf-8")
            outputs["metric_summary_by_model"] = str(path)
    fdr_path = bootstrap_root / "bootstrap_comparisons_fdr.csv"
    if fdr_path.exists():
        fdr = pd.read_csv(fdr_path)
        keep = [
            col
            for col in [
                "backbone",
                "split",
                "seed",
                "comparison_family",
                "candidate_species_mode",
                "reference_species_mode",
                "metric",
                "delta",
                "ci_low",
                "ci_high",
                "p_value_approx",
                "q_value_bh_fdr",
                "significant_fdr_0_05",
            ]
            if col in fdr.columns
        ]
        path = summary_root / "bootstrap_fdr_summary.csv"
        fdr[keep].to_csv(path, index=False, encoding="utf-8")
        outputs["bootstrap_fdr_summary"] = str(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_config(config_path)
    out_root = _output_root(config)
    analysis = config.get("analysis", {})
    metrics_root = Path(analysis.get("metrics_root", out_root / "metrics"))
    embedding_root = Path(analysis.get("embedding_analysis_root", out_root / "embedding_analysis"))
    log_root = out_root / "analysis_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stats_config, embedding_config = _analysis_config(config, out_root)

    commands = [
        [
            "aggregate_metrics",
            [
                "scripts/aggregate_v3_predictions.py",
                "--prediction-dir",
                str(out_root / "predictions"),
                "--out-dir",
                str(metrics_root),
                "--data-dir",
                str(config.get("dataset_path")),
            ],
        ],
        ["block_bootstrap_and_fdr", ["scripts/bootstrap_v3_predictions.py", "--config", str(stats_config)]],
        [
            "species_summary",
            [
                "scripts/build_v3_species_summary.py",
                "--config",
                str(embedding_config),
                "--out",
                str(embedding_root / "species_summary.csv"),
            ],
        ],
        ["embedding_interpretation", ["scripts/analyze_v3_species_embedding.py", "--config", str(embedding_config)]],
        [
            "scaffold_improvement",
            [
                "scripts/analyze_v3_scaffold_improvement.py",
                "--config",
                str(embedding_config),
                "--out",
                str(embedding_root / "scaffold_improvement_summary.csv"),
            ],
        ],
    ]

    if args.dry_run:
        payload = {
            "dry_run": True,
            "commands": [{"label": label, "command": "python " + " ".join(cmd)} for label, cmd in commands],
            "stats_config": str(stats_config),
            "embedding_config": str(embedding_config),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    steps = []
    for label, cmd in commands:
        result = _run(label, cmd, log_root)
        steps.append(result)
        if not result["passed"]:
            break
    summary_outputs = _make_summary_tables(out_root, config) if all(step["passed"] for step in steps) else {}
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "out_root": str(out_root),
        "steps": steps,
        "summary_tables": summary_outputs,
        "overall_passed": all(step["passed"] for step in steps),
    }
    _write_json(out_root / "full_analysis_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
