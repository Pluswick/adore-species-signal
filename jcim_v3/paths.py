from __future__ import annotations

import os
import sys
from pathlib import Path


JCIM_ROOT = Path(__file__).resolve().parents[1]

# The `ccmpnn` package is bundled at the repository root. It supplies the molecular
# graph representation, the directed message-passing backbone, and the evaluation
# metrics. A sibling checkout is accepted as a fallback for development installs.
CC_MPNN_ROOT = JCIM_ROOT if (JCIM_ROOT / "ccmpnn").is_dir() else JCIM_ROOT.parent / "CC-MPNN"
CC_MPNN_DATA = CC_MPNN_ROOT / "data"

RESULTS_ROOT = JCIM_ROOT / "results" / "q2_v4"

# Optional external corpus; unused by the reported runs. Set TOX_LEARN_ROOT if needed.
RAW_TOX_LEARN = Path(os.environ.get("TOX_LEARN_ROOT", "")) if os.environ.get("TOX_LEARN_ROOT") else None

_DLL_HANDLES = []


def add_conda_dll_directories() -> None:
    """Register conda DLL folders for direct python.exe invocation on Windows."""

    if os.name != "nt":
        return
    env_root = Path(sys.executable).resolve().parent
    candidates = [
        env_root,
        env_root / "DLLs",
        env_root / "Library" / "bin",
        env_root / "Library" / "usr" / "bin",
    ]
    for path in candidates:
        if not path.exists():
            continue
        path_str = str(path)
        if path_str not in os.environ.get("PATH", ""):
            os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            try:
                _DLL_HANDLES.append(add_dll_directory(path_str))
            except OSError:
                pass


def add_ccmpnn_to_path() -> None:
    add_conda_dll_directories()
    if not (CC_MPNN_ROOT / "ccmpnn").is_dir():
        raise ModuleNotFoundError(
            "The `ccmpnn` package was not found. It is expected at "
            f"{CC_MPNN_ROOT / 'ccmpnn'}. See README.md, 'Requirements'."
        )
    path = str(CC_MPNN_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def ensure_results_root() -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULTS_ROOT
