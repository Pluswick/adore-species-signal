from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from jcim_v3.featurizer import ATOM_FDIM, BOND_FDIM, mol_to_molgraph, smiles_to_mol
from jcim_v3.paths import add_ccmpnn_to_path

add_ccmpnn_to_path()

from ccmpnn.graph import assemble_batch  # noqa: E402


class StandardScaler:
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_

    def inverse(self, X):
        return np.asarray(X, dtype=np.float64) * self.std_ + self.mean_


def stratum_descriptors(df) -> np.ndarray | None:
    """SPEC 4-0b: endpoint/duration exposed identically to EVERY GNN tier.

    Returns [n_rows, 2] = (endpoint_is_ec50, duration_hours), or None when the
    frame has no stratum columns (single-stratum / pre-strata data), in which
    case the GNN runs exactly as before (mol_feat=off).
    """
    if "endpoint" not in df.columns or "duration" not in df.columns:
        return None
    ep = (df["endpoint"].astype(str).str.upper() == "EC50").to_numpy(np.float64)
    du = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    return np.stack([ep, du], axis=1)


class GraphDataset:
    """DataFrame to BatchMolGraph without RDKit descriptor calculation.

    Carries the endpoint/duration stratum vector as `f_descriptors` so every tier
    (including Tier 0) sees endpoint identically.
    """

    def __init__(self, df, target_scaler=None, desc_scaler=None):
        self.smiles = df["smiles"].tolist()
        self.species_idx = df["species_idx"].to_numpy(np.int64)
        self.target = df["target_log10"].to_numpy(np.float32)
        self.target_scaler = target_scaler
        self.raw_desc = stratum_descriptors(df)
        self.desc_scaler = desc_scaler
        if self.raw_desc is None:
            self.desc = None
        elif desc_scaler is not None:
            self.desc = desc_scaler.transform(self.raw_desc).astype(np.float32)
        else:
            self.desc = self.raw_desc.astype(np.float32)
        self._mg = {}
        for smi in set(self.smiles):
            mol = smiles_to_mol(smi)
            if mol is None or mol.GetNumAtoms() == 0:
                raise ValueError("Invalid or empty SMILES: {!r}".format(smi))
            self._mg[smi] = mol_to_molgraph(mol)

    @property
    def desc_fdim(self) -> int:
        return 0 if self.desc is None else int(self.desc.shape[1])

    def __len__(self):
        return len(self.smiles)

    def batch(self, idx):
        mgs = [self._mg[self.smiles[i]] for i in idx]
        bmg = assemble_batch(
            mgs,
            ATOM_FDIM,
            BOND_FDIM,
            f_descriptors=None if self.desc is None else self.desc[idx],
            species_idx=self.species_idx[idx],
        )
        y = self.target[idx].astype(np.float32).reshape(-1, 1)
        if self.target_scaler is not None:
            y = self.target_scaler.transform(y).astype(np.float32)
        return bmg, torch.from_numpy(y)


def iterate_batches(n, batch_size, shuffle, seed):
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, n, batch_size):
        yield idx[i : i + batch_size]
