"""Classical baseline: per-pixel nonlinear least-squares fitting.

For each pixel, (S0, T2) is estimated with scipy.optimize.curve_fit on the model
S(TE)=S0*exp(-TE/T2). It is the standard reference the PINN is compared against.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def _model(te: np.ndarray, s0: float, t2: float) -> np.ndarray:
    """Exponential model for curve_fit."""
    return s0 * np.exp(-te / np.maximum(t2, 1e-6))


def fit_pixel(te: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Estimate (S0, T2) for a single pixel with least squares.

    Args:
        te: (K,) echo times (ms).
        y:  (K,) measured signals of the pixel.

    Returns:
        (s0, t2). If the fit does not converge, a log-linear estimate is used.
    """
    s0_init = float(max(y[0], 1e-3))
    # Initial T2 estimate from the log-linear slope (robust).
    t2_init = _loglin_t2(te, y)
    try:
        popt, _ = curve_fit(
            _model,
            te,
            y,
            p0=[s0_init, t2_init],
            bounds=([0.0, 1e-3], [np.inf, 1e4]),
            maxfev=5000,
        )
        return float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        return s0_init, t2_init


def _loglin_t2(te: np.ndarray, y: np.ndarray) -> float:
    """Initial T2 estimate from a log-linear regression of ln(S) on TE."""
    yc = np.clip(y, 1e-6, None)
    ln = np.log(yc)
    # ln(S) = ln(S0) - TE/T2  ->  slope = -1/T2
    a = np.polyfit(te, ln, 1)
    slope = a[0]
    if slope >= -1e-9:  # non-negative slope: T2 undefined, fall back
        return 80.0
    return float(-1.0 / slope)


def fit_image(te: np.ndarray, signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit over the whole image, pixel by pixel.

    Args:
        te:      (K,) echo times (ms).
        signals: (H, W, K) multi-echo images.

    Returns:
        (s0_map (H,W), t2_map (H,W)).
    """
    h, w, _ = signals.shape
    s0_map = np.zeros((h, w), dtype=np.float64)
    t2_map = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            s0_map[i, j], t2_map[i, j] = fit_pixel(te, signals[i, j])
    return s0_map, t2_map
