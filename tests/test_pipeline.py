"""Test end-to-end della pipeline su immagini piccole e poche epoche (offline, cpu)."""

import numpy as np

from pinn_qmri.baseline import fit_image, fit_pixel
from pinn_qmri.metrics import all_metrics, mae
from pinn_qmri.physics import default_te, signal_numpy
from pinn_qmri.synth import generate
from pinn_qmri.train import TrainConfig, train_pinn


def test_synth_shapes_and_groundtruth():
    data = generate(height=16, width=16, num_echoes=6, noise_sigma=0.0, seed=1)
    assert data.signals.shape == (16, 16, 6)
    assert data.s0_true.shape == (16, 16)
    assert data.t2_true.shape == (16, 16)
    assert np.all(data.t2_true > 0)


def test_baseline_recovers_t2_noiseless():
    # Senza rumore il fitting LS deve recuperare T2 quasi esattamente.
    te = default_te(8)
    y = signal_numpy(te, np.array(100.0), np.array(55.0))
    s0, t2 = fit_pixel(te, y)
    assert np.isclose(t2, 55.0, rtol=1e-2)
    assert np.isclose(s0, 100.0, rtol=1e-2)


def test_baseline_image_low_error_noiseless():
    data = generate(height=12, width=12, num_echoes=8, noise_sigma=0.0, seed=2)
    _, t2_ls = fit_image(data.te, data.signals)
    assert mae(t2_ls, data.t2_true) < 1.0  # ms


def test_pinn_trains_and_reduces_loss():
    data = generate(height=16, width=16, num_echoes=8, noise_sigma=1.0, seed=3)
    cfg = TrainConfig(epochs=150, seed=3, hidden=32, depth=2)
    _, _, t2_pinn, history = train_pinn(data.signals, data.te, cfg, device=None)
    # La loss deve diminuire e la mappa T2 avere forma corretta e valori positivi.
    assert history[-1] < history[0]
    assert t2_pinn.shape == data.t2_true.shape
    assert np.all(t2_pinn > 0)
    # Errore ragionevole su immagine piccola e poche epoche (soglia generosa).
    assert mae(t2_pinn, data.t2_true) < 40.0


def test_metrics_dict():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = a + 1.0
    m = all_metrics(a, b)
    assert np.isclose(m["mae"], 1.0)
    assert np.isclose(m["rmse"], 1.0)
    assert set(m) == {"mae", "rmse", "mape"}
