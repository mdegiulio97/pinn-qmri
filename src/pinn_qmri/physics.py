"""Physical forward model of T2 relaxometry and signal generation.

The multi-echo signal in a T2 sequence follows a mono-exponential decay:

    S(TE) = S0 * exp(-TE / T2)

where:
- TE  : echo time (ms)
- S0  : signal amplitude at TE -> 0 (proton-density weighted)
- T2  : transverse relaxation time (ms)

This module provides the forward model both in NumPy (for data generation and
the least-squares baseline) and in PyTorch (for the PINN physics loss).
"""

from __future__ import annotations

import numpy as np

try:  # torch is optional for the numpy-only functions, but always present in the project
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def signal_numpy(te: np.ndarray, s0: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """NumPy forward model: S(TE) = S0 * exp(-TE/T2).

    Args:
        te:  echo times, shape (K,) in ms.
        s0:  amplitude, scalar or array broadcastable with t2.
        t2:  relaxation time (ms), same shape as s0.

    Returns:
        Signals with shape (..., K), where ... is the shape of s0/t2.
    """
    te = np.asarray(te, dtype=np.float64)
    s0 = np.asarray(s0, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64)
    # Avoid division by zero: clip T2 to a positive minimum.
    t2_safe = np.clip(t2, 1e-6, None)
    # broadcast: (...,1) * exp(-(K,)/(...,1)) -> (...,K)
    return s0[..., None] * np.exp(-te / t2_safe[..., None])


def signal_torch(te: "torch.Tensor", s0: "torch.Tensor", t2: "torch.Tensor") -> "torch.Tensor":
    """PyTorch forward model (differentiable) for the physics loss.

    Args:
        te:  (K,) tensor of echo times (ms).
        s0:  (N,) tensor of predicted amplitudes.
        t2:  (N,) tensor of predicted T2 values (ms).

    Returns:
        (N, K) tensor of reconstructed signals.
    """
    t2_safe = torch.clamp(t2, min=1e-6)
    return s0[:, None] * torch.exp(-te[None, :] / t2_safe[:, None])


def default_te(num_echoes: int = 8, te_min: float = 10.0, te_max: float = 80.0) -> np.ndarray:
    """Default equally spaced echo times (ms).

    Args:
        num_echoes: number of echoes K.
        te_min:     first TE (ms).
        te_max:     last TE (ms).
    """
    return np.linspace(te_min, te_max, num_echoes, dtype=np.float64)
