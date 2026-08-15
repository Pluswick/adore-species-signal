from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SUPPORTED_SPECIES_CONTROLS = {
    "none",
    "true",
    "zero",
    "shuffled",
    "dummy",
    "bias_only",
}


@dataclass(frozen=True)
class SpeciesControlResult:
    frame: pd.DataFrame
    mode: str
    preserves_marginal: bool
    breaks_compound_species_pairing: bool
    sanity: dict


def _lookup_species(species_idx: int, species_lookup: dict[int, str] | None) -> str:
    if species_lookup is None:
        return f"species_idx:{species_idx}"
    return str(species_lookup.get(int(species_idx), f"species_idx:{species_idx}"))


def _ensure_not_identity(values: np.ndarray, original: np.ndarray, n_species: int) -> np.ndarray:
    if len(values) > 1 and np.array_equal(values, original):
        return np.roll(values, 1)
    if len(values) == 1 and n_species > 1 and values[0] == original[0]:
        return np.asarray([(int(values[0]) + 1) % n_species], dtype=int)
    return values


def _sanity(original: np.ndarray, controlled: np.ndarray, mode: str) -> dict:
    original_counts = pd.Series(original).value_counts().sort_index()
    controlled_counts = pd.Series(controlled).value_counts().sort_index()
    marginal_preserved = original_counts.equals(controlled_counts)
    identical = bool(np.array_equal(original, controlled))
    same_fraction = float((original == controlled).mean()) if len(original) else 0.0
    return {
        "mode": mode,
        "n_rows": int(len(original)),
        "n_unique_original": int(pd.Series(original).nunique()),
        "n_unique_for_model": int(pd.Series(controlled).nunique()),
        "marginal_preserved": bool(marginal_preserved),
        "identical_to_original": identical,
        "same_fraction": same_fraction,
        "all_zero_species_idx": bool((controlled == 0).all()) if len(controlled) else False,
        "has_nan_species_idx": bool(pd.Series(controlled).isna().any()),
    }


def apply_species_control(
    df: pd.DataFrame,
    *,
    mode: str,
    seed: int,
    n_species: int,
    species_lookup: dict[int, str] | None = None,
) -> SpeciesControlResult:
    """Return a copy whose species_idx column is the model-visible label.

    The original label is preserved in species_idx_original. For shuffled mode,
    permutation is row-wise within the supplied frame, so the species marginal
    distribution is exactly preserved while row-to-species pairing is broken.
    """

    if mode not in SUPPORTED_SPECIES_CONTROLS:
        raise ValueError(f"Unsupported species control: {mode!r}")
    if "species_idx" not in df.columns:
        raise ValueError("df must include species_idx")

    out = df.copy()
    out["species_idx_original"] = out["species_idx"].astype(int)
    original = out["species_idx_original"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)

    if mode == "none":
        out["species_for_model"] = -1
        out["input_species"] = "NO_SPECIES"
        out["species_control_type"] = mode
        out["is_shuffled"] = False
        out["is_zero_species"] = False
        out["is_dummy_species"] = False
        return SpeciesControlResult(out, mode, True, False, _sanity(original, original, mode))

    if mode in ("true", "bias_only"):
        out["species_for_model"] = out["species_idx"].astype(int)
        out["input_species"] = out["species"].astype(str) if "species" in out else [
            _lookup_species(idx, species_lookup) for idx in out["species_idx"]
        ]
        out["species_control_type"] = mode
        out["is_shuffled"] = False
        out["is_zero_species"] = False
        out["is_dummy_species"] = False
        return SpeciesControlResult(out, mode, True, False, _sanity(original, original, mode))

    if mode == "zero":
        out["species_idx"] = 0
        out["species_for_model"] = 0
        out["input_species"] = "ZERO_SPECIES"
        out["species_control_type"] = mode
        out["is_shuffled"] = False
        out["is_zero_species"] = True
        out["is_dummy_species"] = False
        controlled = out["species_idx"].to_numpy(dtype=int)
        return SpeciesControlResult(out, mode, False, True, _sanity(original, controlled, mode))

    if mode == "shuffled":
        shuffled = rng.permutation(out["species_idx"].to_numpy(dtype=int))
        shuffled = _ensure_not_identity(shuffled, original, n_species)
        out["species_idx"] = shuffled
        out["species_for_model"] = out["species_idx"].astype(int)
        out["input_species"] = [
            _lookup_species(idx, species_lookup) for idx in out["species_idx"].to_numpy(dtype=int)
        ]
        out["species_control_type"] = mode
        out["is_shuffled"] = True
        out["is_zero_species"] = False
        out["is_dummy_species"] = False
        controlled = out["species_idx"].to_numpy(dtype=int)
        return SpeciesControlResult(out, mode, True, True, _sanity(original, controlled, mode))

    if n_species <= 0:
        raise ValueError("dummy species control requires n_species > 0")
    dummy = rng.integers(0, n_species, size=len(out))
    dummy = _ensure_not_identity(dummy.astype(int), original, n_species)
    out["species_idx"] = dummy
    out["species_for_model"] = out["species_idx"].astype(int)
    out["input_species"] = [f"DUMMY_{idx}" for idx in out["species_idx"].to_numpy(dtype=int)]
    out["species_control_type"] = mode
    out["is_shuffled"] = False
    out["is_zero_species"] = False
    out["is_dummy_species"] = True
    controlled = out["species_idx"].to_numpy(dtype=int)
    return SpeciesControlResult(out, mode, False, True, _sanity(original, controlled, mode))
