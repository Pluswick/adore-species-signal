from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "jcim_v3" / "env_diagnostics"
DEFAULT_ENV_ROOT = Path(r"<USER_HOME>\anaconda3\envs")

TESTS = {
    "rdkit_Chem": "from rdkit import Chem; print('ok')",
    "rdkit_Descriptors": "from rdkit.Chem import Descriptors; print('ok')",
    "rdkit_MurckoScaffold": "from rdkit.Chem.Scaffolds import MurckoScaffold; print('ok')",
    "rdkit_AllChem": "from rdkit.Chem import AllChem; print('ok')",
}


def run_test(python_exe: Path, stmt: str) -> tuple[bool, int | None, str, str]:
    if not python_exe.exists():
        return False, None, "", "python.exe missing"
    proc = subprocess.run(
        [str(python_exe), "-c", stmt],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=60,
    )
    return (
        proc.returncode == 0,
        proc.returncode,
        proc.stdout.strip(),
        proc.stderr.strip().replace("\n", " | "),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", action="append", dest="envs", default=[])
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    envs = args.envs or ["ccmpnn", "Kp", "mordred_env"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for env in envs:
        python_exe = DEFAULT_ENV_ROOT / env / "python.exe"
        for test_name, stmt in TESTS.items():
            success, returncode, stdout, stderr = run_test(python_exe, stmt)
            rows.append(
                {
                    "env": env,
                    "python": str(python_exe),
                    "test": test_name,
                    "success": success,
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                }
            )

    csv_path = out_dir / "rdkit_import_diagnostics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for env in envs:
        env_rows = [row for row in rows if row["env"] == env]
        summary[env] = {
            "all_rdkit_tests_passed": all(row["success"] for row in env_rows),
            "passed": [row["test"] for row in env_rows if row["success"]],
            "failed": [
                {"test": row["test"], "stderr": row["stderr"]}
                for row in env_rows
                if not row["success"]
            ],
        }
    json_path = out_dir / "rdkit_import_diagnostics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"csv: {csv_path}")
    print(f"json: {json_path}")


if __name__ == "__main__":
    main()

