from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger

from src.paths import add_ccmpnn_to_path

add_ccmpnn_to_path()

from ccmpnn.graph import MolGraph  # noqa: E402

RDLogger.DisableLog("rdApp.*")

MAX_ATOMIC_NUM = 100
_ATOM_CHOICES = {
    "atomic_num": list(range(MAX_ATOMIC_NUM)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-1, -2, 1, 2, 0],
    "chiral_tag": [0, 1, 2, 3],
    "num_Hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2,
    ],
}

ATOM_FDIM = sum(len(v) + 1 for v in _ATOM_CHOICES.values()) + 2
BOND_FDIM = 14
DESC_FDIM = 6


def _onek_unk(value, choices):
    enc = [0.0] * (len(choices) + 1)
    enc[choices.index(value) if value in choices else -1] = 1.0
    return enc


def atom_features(atom):
    f = (
        _onek_unk(atom.GetAtomicNum() - 1, _ATOM_CHOICES["atomic_num"])
        + _onek_unk(atom.GetTotalDegree(), _ATOM_CHOICES["degree"])
        + _onek_unk(atom.GetFormalCharge(), _ATOM_CHOICES["formal_charge"])
        + _onek_unk(int(atom.GetChiralTag()), _ATOM_CHOICES["chiral_tag"])
        + _onek_unk(int(atom.GetTotalNumHs()), _ATOM_CHOICES["num_Hs"])
        + _onek_unk(atom.GetHybridization(), _ATOM_CHOICES["hybridization"])
        + [1.0 if atom.GetIsAromatic() else 0.0]
        + [atom.GetMass() * 0.01]
    )
    return np.asarray(f, dtype=np.float32)


def bond_features(bond):
    bt = bond.GetBondType()
    f = [
        0.0,
        float(bt == Chem.BondType.SINGLE),
        float(bt == Chem.BondType.DOUBLE),
        float(bt == Chem.BondType.TRIPLE),
        float(bt == Chem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ] + _onek_unk(int(bond.GetStereo()), list(range(6)))
    return np.asarray(f, dtype=np.float32)


def mol_to_molgraph(mol):
    f_atoms = np.stack([atom_features(a) for a in mol.GetAtoms()])
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_features(b)) for b in mol.GetBonds()]
    return MolGraph(f_atoms=f_atoms, edges=edges)


def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(str(smiles))


def bemis_murcko_scaffold(smiles):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return ""
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold

        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=True)
    except Exception:
        return ""
