"""Metriche di errore tra mappa stimata e ground-truth."""

from __future__ import annotations

from collections.abc import Mapping

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


def rank_models(
    metrics: Mapping[str, Mapping[str, float]],
    key: str = "rmse",
    lower_is_better: bool = True,
) -> list[tuple[str, float]]:
    """Ordina i modelli per una metrica, dal migliore al peggiore.

    Args:
        metrics: mappa nome_modello -> dizionario di metriche
            (ad esempio l'output di :func:`all_metrics` per ciascun modello).
        key: nome della metrica su cui ordinare (default "rmse").
        lower_is_better: se True valori piu' bassi sono migliori (errori come
            MAE/RMSE); se False valori piu' alti sono migliori (ad esempio R^2).

    Returns:
        Lista di coppie ``(nome, valore)`` ordinata dal migliore al peggiore.
        Su mappa vuota restituisce ``[]``. A parita' di valore l'ordine di
        inserimento e' preservato (ordinamento stabile).
    """
    pairs = [(name, float(values[key])) for name, values in metrics.items()]
    return sorted(pairs, key=lambda item: item[1], reverse=not lower_is_better)


def best_model(
    metrics: Mapping[str, Mapping[str, float]],
    key: str = "rmse",
    lower_is_better: bool = True,
) -> str:
    """Nome del modello migliore secondo una metrica.

    Args:
        metrics: mappa nome_modello -> dizionario di metriche.
        key: nome della metrica su cui confrontare (default "rmse").
        lower_is_better: se True valori piu' bassi sono migliori.

    Returns:
        Il nome del modello migliore.

    Raises:
        ValueError: se ``metrics`` e' vuoto (nessun modello da confrontare).
    """
    ranking = rank_models(metrics, key=key, lower_is_better=lower_is_better)
    if not ranking:
        raise ValueError("Nessun modello da confrontare: la mappa delle metriche e' vuota.")
    return ranking[0][0]
