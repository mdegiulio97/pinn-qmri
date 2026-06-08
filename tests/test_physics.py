"""Test del forward model fisico e della coerenza NumPy/PyTorch."""

import numpy as np
import torch

from pinn_qmri.physics import default_te, signal_numpy, signal_torch


def test_signal_decays_monotonically():
    te = default_te(8)
    s = signal_numpy(te, np.array(100.0), np.array(50.0))
    assert s.shape == (8,)
    # Decadimento monotono e segnale a TE=0 ~ S0.
    assert np.all(np.diff(s) < 0)
    s0_est = signal_numpy(np.array([0.0]), np.array(100.0), np.array(50.0))[0]
    assert np.isclose(s0_est, 100.0)


def test_known_value():
    # A TE = T2 il segnale vale S0/e.
    s = signal_numpy(np.array([50.0]), np.array(200.0), np.array(50.0))[0]
    assert np.isclose(s, 200.0 / np.e, rtol=1e-6)


def test_numpy_torch_consistency():
    te = default_te(6)
    s0 = np.array([100.0, 80.0, 120.0])
    t2 = np.array([40.0, 60.0, 90.0])
    s_np = signal_numpy(te, s0, t2)  # (3, 6)
    s_t = signal_torch(
        torch.tensor(te, dtype=torch.float64),
        torch.tensor(s0),
        torch.tensor(t2),
    ).numpy()
    assert np.allclose(s_np, s_t, rtol=1e-6)
