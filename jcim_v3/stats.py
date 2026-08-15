from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


LOWER_IS_BETTER = {"rmse", "mae", "mean_abs_error", "median_abs_error"}
HIGHER_IS_BETTER = {"within_2fold", "within_3fold"}


def metric_value(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    if len(err) == 0:
        return float("nan")
    if metric == "rmse":
        return float(math.sqrt(np.mean(err**2)))
    if metric == "mae":
        return float(np.mean(np.abs(err)))
    if metric == "within_2fold":
        return float(np.mean(np.abs(err) <= math.log10(2.0)))
    if metric == "within_3fold":
        return float(np.mean(np.abs(err) <= math.log10(3.0)))
    if metric == "mean_error":
        return float(np.mean(err))
    if metric == "median_abs_error":
        return float(np.median(np.abs(err)))
    raise ValueError(f"Unsupported metric: {metric!r}")


def prediction_metrics(df: pd.DataFrame) -> dict:
    y_true = df["true_log10"].to_numpy(np.float64)
    y_pred = df["pred_log10"].to_numpy(np.float64)
    err = y_pred - y_true
    return {
        "n": int(len(df)),
        "rmse": metric_value(y_true, y_pred, "rmse"),
        "mae": metric_value(y_true, y_pred, "mae"),
        "within_2fold": metric_value(y_true, y_pred, "within_2fold"),
        "within_3fold": metric_value(y_true, y_pred, "within_3fold"),
        "mean_error": float(np.mean(err)) if len(err) else float("nan"),
        "median_abs_error": float(np.median(np.abs(err))) if len(err) else float("nan"),
    }


def improvement_probability(delta_samples: np.ndarray, metric: str) -> float:
    delta_samples = np.asarray(delta_samples, dtype=np.float64)
    if metric in HIGHER_IS_BETTER:
        return float(np.mean(delta_samples >= 0.0))
    return float(np.mean(delta_samples <= 0.0))


def approximate_sign_p_value(delta_samples: np.ndarray) -> float:
    delta_samples = np.asarray(delta_samples, dtype=np.float64)
    if len(delta_samples) == 0:
        return float("nan")
    p_low = float(np.mean(delta_samples <= 0.0))
    p_high = float(np.mean(delta_samples >= 0.0))
    return min(1.0, 2.0 * min(p_low, p_high))


def practical_effect_category(metric: str, delta: float) -> str:
    if metric not in {"rmse", "mae"}:
        return "not_assigned_for_metric"
    abs_delta = abs(float(delta))
    if abs_delta >= 0.10:
        return "strong"
    if abs_delta >= 0.05:
        return "modest"
    return "weak_or_negligible"


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=np.float64)
    q = np.full_like(p, np.nan, dtype=np.float64)
    valid = np.isfinite(p)
    if not valid.any():
        return q

    valid_idx = np.where(valid)[0]
    valid_p = p[valid]
    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    m = len(ranked_p)
    ranked_q = ranked_p * m / np.arange(1, m + 1)
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0.0, 1.0)
    q[valid_idx[order]] = ranked_q
    return q
