"""Metriche di errore tra mappa stimata e ground-truth."""

from __future__ import annotations

import numpy as np


def mae(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean Absolute Error.

    Args:
        pred: mappa stimata.
        true: mappa ground-truth.
        mask: maschera booleana opzionale (valuta solo i pixel True).
    """
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    diff = np.abs(pred - true)
    if mask is not None:
        diff = diff[mask]
    return float(diff.mean())


def rmse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Root Mean Squared Error."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    diff = (pred - true) ** 2
    if mask is not None:
        diff = diff[mask]
    return float(np.sqrt(diff.mean()))


def mape(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean Absolute Percentage Error (%). Ignora pixel con true ~ 0."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    valid = np.abs(true) > 1e-6
    if mask is not None:
        valid = valid & mask
    rel = np.abs(pred[valid] - true[valid]) / np.abs(true[valid])
    return float(100.0 * rel.mean())


def all_metrics(
    pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, float]:
    """Restituisce MAE, RMSE e MAPE in un dizionario."""
    return {
        "mae": mae(pred, true, mask),
        "rmse": rmse(pred, true, mask),
        "mape": mape(pred, true, mask),
    }
