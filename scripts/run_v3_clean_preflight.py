from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "jcim_v3" / "preflight"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _run_command(label: str, args: list[str], out_dir: Path) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    stdout_path = out_dir / f"{label}_stdout.txt"
    stderr_path = out_dir / f"{label}_stderr.txt"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    return {
        "label": label,
        "command": "python " + " ".join(args),
        "started_at_utc": started,
        "elapsed_sec": round(time.time() - t0, 3),
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout_file": str(stdout_path),
        "stderr_file": str(stderr_path),
        "error_message": proc.stderr.strip() if proc.returncode else "",
    }


def _json_file_check(label: str, path: Path, pass_key: str) -> dict:
    payload = _read_json(path)
    passed = bool(payload.get(pass_key, False))
    return {
        "label": label,
        "command": f"read {path}",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": 0.0,
        "returncode": 0 if passed else 1,
        "passed": passed,
        "json_file": str(path),
        "error_message": "" if passed else f"{pass_key} is not true or file is missing",
    }


def _extract_env_summary(env_payload: dict) -> dict:
    rdkit_rows = env_payload.get("rdkit", [])
    torch_result = env_payload.get("torch_cuda", {})
    torch_payload = {}
    if torch_result.get("stdout"):
        try:
            torch_payload = json.loads(torch_result["stdout"].splitlines()[-1])
        except Exception:
            torch_payload = {"parse_error": torch_result.get("stdout")}
    return {
        "python_executable": env_payload.get("python_executable", sys.executable),
        "conda_env_name": env_payload.get("conda_default_env") or os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": env_payload.get("conda_prefix") or os.environ.get("CONDA_PREFIX"),
        "rdkit_imports_passed": all(row.get("success") for row in rdkit_rows),
        "rdkit_imports": rdkit_rows,
        "torch_cuda_check_passed": bool(torch_result.get("success")),
        "torch_cuda": torch_payload,
    }


def _write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# JCIM v3 Clean Preflight Summary",
        "",
        f"- Generated at UTC: `{summary['generated_at_utc']}`",
        f"- Python executable: `{summary['python_executable']}`",
        f"- Conda env: `{summary.get('conda_env_name')}`",
        f"- Overall passed: `{summary['overall_passed']}`",
        "",
        "## Steps",
        "",
    ]
    for step in summary["steps"]:
        status = "PASS" if step["passed"] else "FAIL"
        lines.extend(
            [
                f"### {step['label']}",
                "",
                f"- Status: `{status}`",
                f"- Command: `{step['command']}`",
                f"- Elapsed seconds: `{step['elapsed_sec']}`",
                f"- Return code: `{step['returncode']}`",
            ]
        )
        if step.get("error_message"):
            lines.append(f"- Error: `{step['error_message']}`")
        lines.append("")
    env = summary["environment"]
    lines.extend(
        [
            "## Environment",
            "",
            f"- RDKit imports passed: `{env['rdkit_imports_passed']}`",
            f"- Torch CUDA check passed: `{env['torch_cuda_check_passed']}`",
            f"- CUDA available: `{env.get('torch_cuda', {}).get('cuda_available')}`",
            f"- GPU: `{env.get('torch_cuda', {}).get('gpu_name')}`",
            "",
            "Smoke outputs are validated as preflight inputs only. Smoke performance values are not used as full experiment evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    steps: list[dict] = []

    env_out = out_dir / "jcim_v3_import_check.json"
    steps.append(_run_command("01_env_check", ["scripts/check_jcim_v3_env.py", "--out", str(env_out)], out_dir))

    feature_out = out_dir / "feature_cache_validation.json"
    steps.append(_run_command("02_feature_cache_validation", ["scripts/validate_v3_outputs.py", "--out", str(feature_out)], out_dir))

    steps.append(
        _json_file_check(
            "03_control_smoke_validation",
            ROOT / "results" / "jcim_v3" / "smoke" / "jcim_v3_env" / "controls" / "control_sanity_checks.json",
            "all_checks_passed",
        )
    )
    steps.append(
        _json_file_check(
            "04_injection_position_smoke_validation",
            ROOT
            / "results"
            / "jcim_v3"
            / "smoke"
            / "jcim_v3_env"
            / "injection_positions"
            / "injection_sanity_checks.json",
            "all_checks_passed",
        )
    )

    stats_out = out_dir / "stats_sanity_checks.json"
    steps.append(
        _run_command(
            "05_stats_smoke_validation",
            ["scripts/validate_v3_stats_smoke.py", "--config", "configs/v3_stats_smoke.json", "--out", str(stats_out)],
            out_dir,
        )
    )

    embedding_checks = out_dir / "embedding_sanity_checks.json"
    embedding_warnings = out_dir / "embedding_warnings.json"
    steps.append(
        _run_command(
            "06_embedding_smoke_validation",
            [
                "scripts/validate_v3_embedding_smoke.py",
                "--config",
                "configs/v3_embedding_smoke.json",
                "--checks-out",
                str(embedding_checks),
                "--warnings-out",
                str(embedding_warnings),
            ],
            out_dir,
        )
    )

    env_summary = _extract_env_summary(_read_json(env_out))
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at,
        "python_executable": sys.executable,
        "conda_env_name": env_summary.get("conda_env_name"),
        "steps": steps,
        "environment": env_summary,
        "overall_passed": all(step["passed"] for step in steps),
        "note": "This preflight validates existing smoke artifacts and environment state. It does not run full training or full bootstrap.",
    }
    _write_json(out_dir / "preflight_summary.json", summary)
    _write_markdown(out_dir / "preflight_summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
