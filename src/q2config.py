"""Single source of truth for Q2 v4 dataset paths / hyperparameters.

Every q2 script resolves paths through here so a new dataset is a one-file swap:
copy configs/q2_dataset_toxlearn.json, point vendor_raw + filters at the new source,
and set Q2_DATASET_CONFIG (env var) or pass --config. No hardcoded paths in scripts.

    from src.q2config import load_q2_config
    C = load_q2_config()          # env Q2_DATASET_CONFIG or the toxlearn default
    C.data_dir / "replication_group_train.csv"
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parents[1] / "configs" / "q2_dataset_toxlearn.json"


@dataclass(frozen=True)
class Q2Config:
    raw: dict
    path: Path

    @property
    def workspace(self) -> Path:
        return Path(self.raw["workspace_root"])

    @property
    def data_dir(self) -> Path:
        return self.workspace / "data"

    @property
    def output_root(self) -> Path:
        return Path(self.raw["output_root"])

    @property
    def gnn_runs(self) -> Path:
        return self.output_root / "gnn" / "runs"

    @property
    def gnn_preds(self) -> Path:
        return self.output_root / "gnn" / "predictions"

    @property
    def lgbm_preds(self) -> Path:
        return self.output_root / "replication" / "predictions"

    @property
    def lgbm_runs(self) -> Path:
        return self.output_root / "replication" / "runs"

    @property
    def vendor_dir(self) -> Path:
        return Path(self.raw["vendor_raw"]["dir"])

    @property
    def taxonomy_ncbi_filled(self) -> Path:
        return Path(self.raw["taxonomy_ncbi_filled"])

    def abundance_bin(self, train_count: int) -> str:
        for name, (lo, hi) in self.raw["abundance_bins"].items():
            if train_count >= lo and (hi is None or train_count <= hi):
                return name
        return "warm"


def load_q2_config(path: str | os.PathLike | None = None) -> Q2Config:
    p = Path(path or os.environ.get("Q2_DATASET_CONFIG") or _DEFAULT)
    return Q2Config(raw=json.loads(p.read_text(encoding="utf-8")), path=p)
