from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "v3_full_experiment.json"
DEFAULT_OUT_DIR = ROOT / "results" / "jcim_v3" / "full_run_plan"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _dedupe(items: list[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _batch_name(split: str, seed: int) -> str:
    return f"{split}_seed{seed}"


def _ps_path(path: Path) -> str:
    try:
        display_path = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        display_path = path
    return display_path.as_posix().replace("/", "\\")


def _run_command(config_path: Path, split: str, seed: int, action: str) -> str:
    flag = "--dry-run" if action == "dry-run" else "--resume"
    return f"python scripts\\run_v3_full_experiment.py --config {_ps_path(config_path)} --split {split} --seed {seed} {flag}"


def _validation_command(split: str, seed: int) -> str:
    return f"python scripts\\validate_v3_full_batch.py --root results\\jcim_v3\\full --split {split} --seed {seed}"


def _write_batch_plan_md(path: Path, rows: list[dict], generated_at: str) -> None:
    lines = [
        "# JCIM v3 Full Batch Plan",
        "",
        f"- Generated at UTC: `{generated_at}`",
        f"- Total batches: `{len(rows)}`",
        f"- Runs per batch: `38`",
        "",
        "| Batch | Split | Seed | Runs | Status | Reused From Pilot |",
        "|---|---|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['batch_name']}` | `{row['split']}` | `{row['seed']}` | `{row['planned_runs']}` | `{row['status']}` | `{row['reused_from_pilot']}` |"
        )
    lines.extend(["", "This is a planning artifact only. It does not execute training.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_commands_md(path: Path, rows: list[dict], config_path: Path, scaffold_seed0_reused: bool) -> None:
    executable_rows = [row for row in rows if row["status"] == "planned"]
    lines = [
        "# JCIM v3 Full Batch Commands",
        "",
        "Run these commands in Windows PowerShell from the JCIM repository root.",
        "",
        "## A. Common Pre-Run Checks",
        "",
        "```powershell",
        "conda activate jcim_v3",
        "python scripts\\run_v3_clean_preflight.py",
        f"python scripts\\run_v3_full_experiment.py --config {_ps_path(config_path)} --dry-run",
        "```",
        "",
        "If PowerShell prints OpenCL, DLL, or conda activation warnings during a real batch, append the warning text to `results\\jcim_v3\\full\\environment_warnings.log` before running batch validation.",
        "",
    ]
    if scaffold_seed0_reused:
        lines.extend(
            [
                "## Optional: Register Scaffold Seed 0 Pilot",
                "",
                "Run this before full batches if you decide to reuse the validated scaffold seed 0 pilot artifacts.",
                "",
                "```powershell",
                "python scripts\\check_v3_pilot_reuse.py",
                "python scripts\\register_v3_pilot_as_full.py --execute",
                "python scripts\\validate_v3_full_batch.py --root results\\jcim_v3\\full --split scaffold --seed 0",
                "```",
                "",
            ]
        )
    lines.extend(["## B. Batch Dry-Runs", ""])
    for row in rows:
        lines.extend(["```powershell", row["dry_run_command"], "```", ""])
    lines.extend(["## C. Batch Execution Commands", ""])
    for row in executable_rows:
        lines.extend(["```powershell", row["run_command"], "```", ""])
    if scaffold_seed0_reused:
        lines.extend(
            [
                "`scaffold_seed0` is excluded from the execution list because the pilot is marked reusable. Validate/register it instead of retraining.",
                "",
            ]
        )
    lines.extend(["## D. Batch Validation Commands", ""])
    for row in rows:
        lines.extend(["```powershell", row["validation_command"], "```", ""])
    lines.extend(
        [
            "## E. Recommended Execution Order",
            "",
            "1. `random_seed0`",
            "2. `random_seed1`",
            "3. `random_seed2`",
            "4. `random_seed3`",
            "5. `random_seed4`",
        ]
    )
    if scaffold_seed0_reused:
        lines.append("6. Register and validate `scaffold_seed0` from pilot artifacts")
        offset = 7
    else:
        lines.append("6. `scaffold_seed0`")
        offset = 7
    for idx, seed in enumerate([1, 2, 3, 4], start=offset):
        lines.append(f"{idx}. `scaffold_seed{seed}`")
    lines.extend(
        [
            "",
            "Do not run the full 380-run config as one uninterrupted job unless you intentionally accept the runtime risk.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_analysis_commands_md(path: Path, config_path: Path) -> None:
    lines = [
        "# JCIM v3 Full Analysis Commands",
        "",
        "Run this only after all full training batches and batch validations have passed.",
        "",
        "```powershell",
        "conda activate jcim_v3",
        f"python scripts\\run_v3_full_analysis.py --config {_ps_path(config_path)}",
        "```",
        "",
        "This command runs the full analysis settings from `configs/v3_full_experiment.json`, including `n_bootstrap=2000`. Do not run it before full training is complete.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = _read_json(config_path)
    reuse_check = _read_json(out_dir / "pilot_reuse_check.json")

    splits = list(config.get("split_types", []))
    seeds = [int(seed) for seed in config.get("seeds", [])]
    species_modes = _dedupe(list(config.get("main_species_modes", [])) + list(config.get("control_species_modes", [])))
    backbones = list(config.get("backbones", []))
    baselines = list(config.get("lightgbm_baselines", []))
    gnn_runs = len(backbones) * len(species_modes)
    lgbm_runs = len(baselines)
    planned_runs = gnn_runs + lgbm_runs

    rows = []
    scaffold_seed0_reused = bool(reuse_check.get("reusable", False))
    for split in splits:
        for seed in seeds:
            reused = split == "scaffold" and seed == 0 and scaffold_seed0_reused
            status = "reused_from_pilot" if reused else "planned"
            rows.append(
                {
                    "batch_name": _batch_name(split, seed),
                    "split": split,
                    "seed": seed,
                    "planned_runs": planned_runs,
                    "gnn_runs": gnn_runs,
                    "lgbm_runs": lgbm_runs,
                    "status": status,
                    "reused_from_pilot": reused,
                    "estimated_runtime_optional": "use pilot runtime as lower-bound reference; validate locally before long runs",
                    "dry_run_command": _run_command(config_path, split, seed, "dry-run"),
                    "run_command": "SKIP: register scaffold_seed0 from pilot artifacts" if reused else _run_command(config_path, split, seed, "run"),
                    "validation_command": _validation_command(split, seed),
                }
            )

    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at_utc": generated_at,
        "config": str(config_path),
        "batch_count": len(rows),
        "runs_per_batch": planned_runs,
        "total_planned_runs": sum(int(row["planned_runs"]) for row in rows),
        "scaffold_seed0_reused_from_pilot": scaffold_seed0_reused,
        "training_was_executed": False,
    }
    _write_csv(out_dir / "full_batch_plan.csv", rows)
    _write_batch_plan_md(out_dir / "full_batch_plan.md", rows, generated_at)
    _write_commands_md(out_dir / "full_batch_commands.md", rows, config_path, scaffold_seed0_reused)
    _write_analysis_commands_md(out_dir / "full_analysis_commands.md", config_path)
    _write_json(out_dir / "full_batch_plan_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
