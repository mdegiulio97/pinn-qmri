"""Baseline classica: fitting non lineare ai minimi quadrati per-pixel.

Per ogni pixel si stima (S0, T2) con scipy.optimize.curve_fit sul modello
S(TE)=S0*exp(-TE/T2). E' il riferimento standard contro cui confrontare la PINN.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def _model(te: np.ndarray, s0: float, t2: float) -> np.ndarray:
    """Modello esponenziale per curve_fit."""
    return s0 * np.exp(-te / np.maximum(t2, 1e-6))


def fit_pixel(te: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Stima (S0, T2) su un singolo pixel con least-squares.

    Args:
        te: (K,) tempi di eco (ms).
        y:  (K,) segnali misurati del pixel.

    Returns:
        (s0, t2). In caso di mancata convergenza usa una stima log-lineare.
    """
    s0_init = float(max(y[0], 1e-3))
    # Stima iniziale di T2 via pendenza log-lineare (robusta).
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
    """Stima iniziale di T2 tramite regressione log-lineare su ln(S) vs TE."""
    yc = np.clip(y, 1e-6, None)
    ln = np.log(yc)
    # ln(S) = ln(S0) - TE/T2  ->  pendenza = -1/T2
    a = np.polyfit(te, ln, 1)
    slope = a[0]
    if slope >= -1e-9:  # pendenza non negativa: T2 indefinito, fallback
        return 80.0
    return float(-1.0 / slope)


def fit_image(te: np.ndarray, signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fitting least-squares su tutta l'immagine, pixel per pixel.

    Args:
        te:      (K,) tempi di eco (ms).
        signals: (H, W, K) immagini multi-eco.

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
