from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULL_CONFIG = ROOT / "configs" / "v3_full_experiment.json"
DEFAULT_PILOT_ROOT = ROOT / "results" / "jcim_v3" / "full_pilot_scaffold_seed0"
DEFAULT_FULL_ROOT = ROOT / "results" / "jcim_v3" / "full"
DEFAULT_PLAN_DIR = ROOT / "results" / "jcim_v3" / "full_run_plan"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
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


def _run_id(row: dict, config: dict) -> str:
    if row["run_type"] == "lightgbm_rdkit":
        return f"{row['baseline']}_{row['split']}_s{row['seed']}"
    epochs = int(config.get("max_epochs", config.get("epochs", 100)))
    limit_train = config.get("train_subset_size", config.get("limit_train"))
    n_part = limit_train if limit_train is not None else "full"
    return f"{row['backbone']}_{row['species_mode']}_{row['split']}_s{row['seed']}_e{epochs}_n{n_part}"


def _full_manifest_rows(config: dict) -> list[dict]:
    splits = list(config.get("split_types") or config.get("splits") or [config.get("split", "scaffold")])
    seeds = [int(value) for value in config.get("seeds", [config.get("seed", 0)])]
    backbones = list(config.get("backbones") or ["dmpnn", "graphconv"])
    species_modes = _dedupe(list(config.get("main_species_modes", [])) + list(config.get("control_species_modes", [])))
    baselines = list(config.get("lightgbm_baselines", []))
    rows = []
    for split in splits:
        for seed in seeds:
            for backbone in backbones:
                for species_mode in species_modes:
                    row = {
                        "run_type": "gnn",
                        "split": split,
                        "seed": seed,
                        "backbone": backbone,
                        "species_mode": species_mode,
                        "baseline": "",
                    }
                    row["run_id"] = _run_id(row, config)
                    rows.append(row)
            for baseline in baselines:
                row = {
                    "run_type": "lightgbm_rdkit",
                    "split": split,
                    "seed": seed,
                    "backbone": "lightgbm_rdkit",
                    "species_mode": "species_categorical" if baseline.endswith("species_categorical") else "no_species",
                    "baseline": baseline,
                }
                row["run_id"] = _run_id(row, config)
                rows.append(row)
    return rows


def _copy_file(src: Path, dst: Path, execute: bool, overwrite: bool) -> str:
    if not src.exists():
        return "missing_source"
    if dst.exists() and not overwrite:
        return "exists_skipped"
    if execute:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return "copied"
    return "planned"


def _merge_by_run_id(existing: list[dict], incoming: list[dict], status: str | None = None) -> list[dict]:
    out = {row.get("run_id"): dict(row) for row in existing if row.get("run_id")}
    for row in incoming:
        next_row = dict(row)
        if status is not None:
            next_row["status"] = status
        out[next_row.get("run_id")] = next_row
    return list(out.values())


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Register Scaffold Seed 0 Pilot As Full",
        "",
        f"- Generated at UTC: `{payload['generated_at_utc']}`",
        f"- Execute mode: `{payload['execute']}`",
        f"- Reuse check passed: `{payload['reuse_check_passed']}`",
        f"- Pilot root: `{payload['pilot_root']}`",
        f"- Full root: `{payload['full_root']}`",
        f"- Planned/copied run rows: `{payload['run_rows']}`",
        f"- Planned/copied artifact rows: `{payload['artifact_rows']}`",
        "",
        "This script does not train models. It only copies/registers already completed scaffold seed 0 pilot artifacts when `--execute` is explicitly provided.",
        "",
    ]
    if not payload["execute"]:
        lines.extend(
            [
                "## Execute Command",
                "",
                "```powershell",
                "conda activate jcim_v3",
                "python scripts\\register_v3_pilot_as_full.py --execute",
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-config", default=str(DEFAULT_FULL_CONFIG))
    parser.add_argument("--pilot-root", default=str(DEFAULT_PILOT_ROOT))
    parser.add_argument("--full-root", default=str(DEFAULT_FULL_ROOT))
    parser.add_argument("--plan-dir", default=str(DEFAULT_PLAN_DIR))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    full_config = _read_json(Path(args.full_config))
    pilot_root = Path(args.pilot_root)
    full_root = Path(args.full_root)
    plan_dir = Path(args.plan_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)

    reuse_check = _read_json(plan_dir / "pilot_reuse_check.json")
    reuse_check_passed = bool(reuse_check.get("reusable", False))
    if args.execute and not reuse_check_passed:
        raise SystemExit("Refusing to register pilot because pilot_reuse_check.json does not have reusable=true")

    pilot_manifest = _read_csv(pilot_root / "run_manifest.csv")
    pilot_rows = [
        row
        for row in pilot_manifest
        if row.get("split") == "scaffold" and str(row.get("seed")) == "0"
    ]
    expected_run_ids = {row.get("run_id") for row in pilot_rows}
    completed_rows = [row for row in _read_csv(pilot_root / "completed_runs.csv") if row.get("run_id") in expected_run_ids]
    parameter_rows = [row for row in _read_csv(pilot_root / "parameter_counts" / "parameter_counts.csv") if row.get("run_id") in expected_run_ids]

    artifact_rows = []
    for row in pilot_rows:
        run_id = row["run_id"]
        for subdir, suffix in [("predictions", ".csv"), ("metrics", ".json"), ("runs", ".json")]:
            src = pilot_root / subdir / f"{run_id}{suffix}"
            dst = full_root / subdir / f"{run_id}{suffix}"
            status = _copy_file(src, dst, args.execute, args.overwrite)
            artifact_rows.append(
                {
                    "run_id": run_id,
                    "artifact_type": subdir,
                    "source_path": str(src),
                    "dest_path": str(dst),
                    "status": status,
                }
            )

    for src in sorted((pilot_root / "embeddings").glob("*")):
        if not src.is_file():
            continue
        dst = full_root / "embeddings" / src.name
        status = _copy_file(src, dst, args.execute, args.overwrite)
        artifact_rows.append(
            {
                "run_id": "",
                "artifact_type": "embeddings",
                "source_path": str(src),
                "dest_path": str(dst),
                "status": status,
            }
        )

    full_manifest = _full_manifest_rows(full_config)
    for row in full_manifest:
        if row["split"] == "scaffold" and int(row["seed"]) == 0:
            row["registration_status"] = "reused_from_pilot"
        else:
            row["registration_status"] = "planned"

    if args.execute:
        _write_csv(full_root / "run_manifest.csv", full_manifest)
        _write_csv(full_root / "reused_from_pilot_manifest.csv", artifact_rows)
        merged_completed = _merge_by_run_id(_read_csv(full_root / "completed_runs.csv"), completed_rows, "reused_from_pilot")
        _write_csv(full_root / "completed_runs.csv", merged_completed)
        merged_parameters = _merge_by_run_id(_read_csv(full_root / "parameter_counts" / "parameter_counts.csv"), parameter_rows)
        _write_csv(full_root / "parameter_counts" / "parameter_counts.csv", merged_parameters)
    else:
        _write_csv(plan_dir / "reused_from_pilot_manifest_preview.csv", artifact_rows)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execute": bool(args.execute),
        "reuse_check_passed": reuse_check_passed,
        "pilot_root": str(pilot_root),
        "full_root": str(full_root),
        "run_rows": len(pilot_rows),
        "completed_rows": len(completed_rows),
        "parameter_rows": len(parameter_rows),
        "artifact_rows": len(artifact_rows),
        "artifact_status_counts": {
            status: sum(1 for row in artifact_rows if row["status"] == status)
            for status in sorted({row["status"] for row in artifact_rows})
        },
        "training_was_executed": False,
    }
    _write_json(plan_dir / "register_pilot_preview.json", payload)
    _write_markdown(plan_dir / "register_pilot_commands.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
