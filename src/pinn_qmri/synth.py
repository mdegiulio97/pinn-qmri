"""Generation of synthetic images with ground-truth S0, T2 maps and noise.

A 2D map is built in which every pixel has spatially varying (S0, T2) values
and distinct regions (background + a few "lesions" / tissues with different T2).
From these maps, K images at different echo times TE are simulated with the
physical forward model, and Gaussian or Rician noise is added.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import default_te, signal_numpy


@dataclass
class SynthData:
    """Container for the synthetic dataset.

    Attributes:
        te:        (K,) echo times (ms).
        signals:   (H, W, K) noisy multi-echo images.
        s0_true:   (H, W) ground-truth S0 map.
        t2_true:   (H, W) ground-truth T2 map (ms).
        noise_sigma: standard deviation of the noise used.
        noise_type:  "gaussian" or "rician".
    """

    te: np.ndarray
    signals: np.ndarray
    s0_true: np.ndarray
    t2_true: np.ndarray
    noise_sigma: float
    noise_type: str


def _disk(h: int, w: int, cy: float, cx: float, r: float) -> np.ndarray:
    """Boolean mask of a disk of radius r centred at (cy, cx)."""
    yy, xx = np.mgrid[0:h, 0:w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def make_ground_truth(height: int = 64, width: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Create spatially varying ground-truth (S0, T2) maps.

    The background has a smooth T2 gradient; disks with distinct T2 values
    (e.g. tissues/lesions) are overlaid. S0 has an independent gradient.

    Returns:
        (s0_true, t2_true), both with shape (height, width). T2 in ms.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    yn = yy / max(height - 1, 1)
    xn = xx / max(width - 1, 1)

    # Base T2: smooth gradient 40..90 ms
    t2 = 40.0 + 50.0 * (0.5 * yn + 0.5 * xn)

    # Regions with distinct T2 (centre/radii chosen relative to the image size)
    t2[_disk(height, width, height * 0.30, width * 0.35, min(height, width) * 0.14)] = 120.0
    t2[_disk(height, width, height * 0.68, width * 0.65, min(height, width) * 0.11)] = 25.0
    t2[_disk(height, width, height * 0.55, width * 0.25, min(height, width) * 0.08)] = 160.0

    # S0: independent gradient 80..120 (arbitrary units)
    s0 = 80.0 + 40.0 * (1.0 - yn) * (0.5 + 0.5 * xn)

    return s0.astype(np.float64), t2.astype(np.float64)


def add_noise(
    clean: np.ndarray,
    sigma: float,
    noise_type: str = "gaussian",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add noise to the clean signals.

    Args:
        clean:      noise-free signals, any shape.
        sigma:      noise standard deviation (same units as the signal).
        noise_type: "gaussian" or "rician".
        rng:        optional numpy generator (for reproducibility).

    Returns:
        Noisy signals, same shape as clean.
    """
    if rng is None:
        rng = np.random.default_rng()
    if sigma <= 0:
        return clean.copy()
    if noise_type == "gaussian":
        return clean + rng.normal(0.0, sigma, size=clean.shape)
    if noise_type == "rician":
        # Rician noise: magnitude of two independent Gaussian channels.
        real = clean + rng.normal(0.0, sigma, size=clean.shape)
        imag = rng.normal(0.0, sigma, size=clean.shape)
        return np.sqrt(real**2 + imag**2)
    raise ValueError(f"unknown noise_type: {noise_type!r}")


def generate(
    height: int = 64,
    width: int = 64,
    num_echoes: int = 8,
    te_min: float = 10.0,
    te_max: float = 80.0,
    noise_sigma: float = 2.0,
    noise_type: str = "gaussian",
    seed: int = 0,
) -> SynthData:
    """Generate a complete, reproducible synthetic dataset.

    Args:
        height, width: image size.
        num_echoes:    number K of echo times.
        te_min, te_max: TE range limits (ms).
        noise_sigma:   noise sigma.
        noise_type:    "gaussian" or "rician".
        seed:          seed for reproducibility.

    Returns:
        A SynthData object.
    """
    rng = np.random.default_rng(seed)
    te = default_te(num_echoes, te_min, te_max)
    s0_true, t2_true = make_ground_truth(height, width)

    clean = signal_numpy(te, s0_true, t2_true)  # (H, W, K)
    signals = add_noise(clean, noise_sigma, noise_type, rng)

    return SynthData(
        te=te,
        signals=signals.astype(np.float64),
        s0_true=s0_true,
        t2_true=t2_true,
        noise_sigma=float(noise_sigma),
        noise_type=noise_type,
    )
