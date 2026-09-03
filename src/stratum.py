"""SPEC 4-0b (A): endpoint/duration additive main-effect control by target residualization.

The SAME operation is applied to every tier and every backbone (naive / lightgbm /
dmpnn / graphconv): the train-estimated additive stratum main effect is removed from
the training target and restored at prediction time. No model architecture changes,
so tiers stay exactly comparable and the ladder measures species signal only.

The endpoint x structure interaction is deliberately OUT of scope: the ladder asks
where the SPECIES signal saturates after endpoint main effects are removed.

On single-stratum data (discovery: LC50 96h only) the effect is a single constant,
so removal-then-restoration is the identity -- discovery is unchanged by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GRAND = "__grand__"


def stratum_labels(df: pd.DataFrame) -> np.ndarray:
    """endpoint x duration label; constant when the frame has no stratum columns."""
    n = len(df)
    ep = df["endpoint"].astype(str).to_numpy() if "endpoint" in df.columns else np.full(n, "NA")
    du = df["duration"].astype(str).to_numpy() if "duration" in df.columns else np.full(n, "NA")
    return np.array([f"{a}@{b}" for a, b in zip(ep, du)], dtype=object)


def fit_stratum_effect(train: pd.DataFrame, y_train: np.ndarray) -> dict[str, float]:
    """Additive main effect per stratum, centered on the grand mean (train only)."""
    y = np.asarray(y_train, dtype=np.float64)
    lab = stratum_labels(train)
    grand = float(np.mean(y))
    means = pd.Series(y).groupby(pd.Series(lab)).mean()
    eff = {str(k): float(v) - grand for k, v in means.items()}
    eff[GRAND] = grand
    return eff


def remove(df: pd.DataFrame, y: np.ndarray, eff: dict[str, float]) -> np.ndarray:
    """y_adjusted = y - stratum_effect(row). Unseen strata get 0 (no adjustment)."""
    lab = stratum_labels(df)
    adj = np.array([eff.get(str(k), 0.0) for k in lab], dtype=np.float64)
    return np.asarray(y, dtype=np.float64) - adj


def restore(df: pd.DataFrame, pred: np.ndarray, eff: dict[str, float]) -> np.ndarray:
    """Add the stratum effect back onto predictions made in adjusted space."""
    lab = stratum_labels(df)
    adj = np.array([eff.get(str(k), 0.0) for k in lab], dtype=np.float64)
    return np.asarray(pred, dtype=np.float64) + adj


def residualize_frames(
    train: pd.DataFrame, others: list[pd.DataFrame], target_col: str = "target_log10"
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, float]]:
    """Return copies whose target column is stratum-adjusted, plus the effect map."""
    eff = fit_stratum_effect(train, train[target_col].to_numpy(np.float64))
    tr = train.copy()
    tr[target_col] = remove(train, train[target_col].to_numpy(np.float64), eff)
    outs = []
    for df in others:
        d = df.copy()
        d[target_col] = remove(df, df[target_col].to_numpy(np.float64), eff)
        outs.append(d)
    return tr, outs, eff
