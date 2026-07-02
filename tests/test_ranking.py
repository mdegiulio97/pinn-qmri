"""Test delle funzioni di ranking/selezione dei modelli (offline, senza rete)."""

import pytest

from pinn_qmri.metrics import best_model, rank_models

# Metriche fittizie: due modelli con errori (RMSE/MAE) e uno score R^2.
METRICS = {
    "pinn": {"rmse": 5.5, "mae": 2.5, "r2": 0.90},
    "ls": {"rmse": 1.8, "mae": 1.3, "r2": 0.95},
    "mean": {"rmse": 9.0, "mae": 7.0, "r2": 0.40},
}


def test_rank_models_lower_is_better():
    ranking = rank_models(METRICS, key="rmse", lower_is_better=True)
    assert [name for name, _ in ranking] == ["ls", "pinn", "mean"]
    assert ranking[0] == ("ls", 1.8)


def test_rank_models_higher_is_better():
    ranking = rank_models(METRICS, key="r2", lower_is_better=False)
    assert [name for name, _ in ranking] == ["ls", "pinn", "mean"]
    assert ranking[0][0] == "ls"


def test_rank_models_empty_returns_empty_list():
    assert rank_models({}) == []


def test_rank_models_stable_on_ties():
    tied = {"a": {"rmse": 2.0}, "b": {"rmse": 2.0}, "c": {"rmse": 1.0}}
    ranking = rank_models(tied, key="rmse", lower_is_better=True)
    # 'c' e' il migliore; a parita' l'ordine di inserimento e' preservato.
    assert [name for name, _ in ranking] == ["c", "a", "b"]


def test_best_model_default_key():
    assert best_model(METRICS) == "ls"


def test_best_model_higher_is_better():
    assert best_model(METRICS, key="r2", lower_is_better=False) == "ls"


def test_best_model_empty_raises():
    with pytest.raises(ValueError):
        best_model({})
