from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "src" / "env_diagnostics" / "src_import_check.json"

TESTS = {
    "rdkit_Chem": "from rdkit import Chem; print('ok')",
    "rdkit_Descriptors": "from rdkit.Chem import Descriptors; print('ok')",
    "rdkit_MurckoScaffold": "from rdkit.Chem.Scaffolds import MurckoScaffold; print('ok')",
    "rdkit_AllChem": "from rdkit.Chem import AllChem; print('ok')",
}

TORCH_CHECK = """
import json
import torch
payload = {
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "device_count": torch.cuda.device_count(),
}
print(json.dumps(payload))
"""


def run_stmt(name: str, stmt: str, timeout: int) -> dict:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", stmt],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "name": name,
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip().replace("\n", " | "),
            "timeout_seconds": timeout,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "success": False,
            "returncode": None,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": f"timeout after {timeout} seconds",
            "timeout_seconds": timeout,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--include-umap", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rdkit_results = [run_stmt(name, stmt, timeout=60) for name, stmt in TESTS.items()]
    torch_result = run_stmt("torch_cuda", TORCH_CHECK, timeout=60)
    optional_results = []
    if args.include_umap:
        optional_results.append(run_stmt("umap_import", "import umap; print('ok')", timeout=120))

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "rdkit": rdkit_results,
        "torch_cuda": torch_result,
        "optional": optional_results,
        "all_required_imports_passed": all(row["success"] for row in rdkit_results)
        and torch_result["success"],
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"json: {out}")


if __name__ == "__main__":
    main()
