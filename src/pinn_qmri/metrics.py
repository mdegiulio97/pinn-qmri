"""Error metrics between an estimated map and the ground truth."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def mae(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean Absolute Error.

    Args:
        pred: estimated map.
        true: ground-truth map.
        mask: optional boolean mask (only True pixels are evaluated).
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
    """Mean Absolute Percentage Error (%). Ignores pixels where true ~ 0."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    valid = np.abs(true) > 1e-6
    if mask is not None:
        valid = valid & mask
    rel = np.abs(pred[valid] - true[valid]) / np.abs(true[valid])
    return float(100.0 * rel.mean())


def all_metrics(
    pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, float]:
    """Return MAE, RMSE and MAPE in a dictionary."""
    return {
        "mae": mae(pred, true, mask),
        "rmse": rmse(pred, true, mask),
        "mape": mape(pred, true, mask),
    }


def rank_models(
    metrics: Mapping[str, Mapping[str, float]],
    key: str = "rmse",
    lower_is_better: bool = True,
) -> list[tuple[str, float]]:
    """Sort models by a metric, from best to worst.

    Args:
        metrics: mapping model_name -> dictionary of metrics
            (for example the output of :func:`all_metrics` for each model).
        key: name of the metric to sort on (default "rmse").
        lower_is_better: if True, lower values are better (errors such as
            MAE/RMSE); if False, higher values are better (for example R^2).

    Returns:
        List of ``(name, value)`` pairs sorted from best to worst.
        Returns ``[]`` for an empty mapping. Ties keep insertion order
        (stable sort).
    """
    pairs = [(name, float(values[key])) for name, values in metrics.items()]
    return sorted(pairs, key=lambda item: item[1], reverse=not lower_is_better)


def best_model(
    metrics: Mapping[str, Mapping[str, float]],
    key: str = "rmse",
    lower_is_better: bool = True,
) -> str:
    """Name of the best model according to a metric.

    Args:
        metrics: mapping model_name -> dictionary of metrics.
        key: name of the metric to compare on (default "rmse").
        lower_is_better: if True, lower values are better.

    Returns:
        The name of the best model.

    Raises:
        ValueError: if ``metrics`` is empty (no models to compare).
    """
    ranking = rank_models(metrics, key=key, lower_is_better=lower_is_better)
    if not ranking:
        raise ValueError("No models to compare: the metrics mapping is empty.")
    return ranking[0][0]
