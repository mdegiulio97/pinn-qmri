"""Generazione di immagini sintetiche con mappe S0, T2 ground-truth e rumore.

Si costruisce una mappa 2D in cui ogni pixel ha valori (S0, T2) spazialmente
variabili e regioni distinte (sfondo + alcune "lesioni" / tessuti con T2 diverso).
Da queste mappe si simulano K immagini a tempi di eco TE diversi tramite il
forward model fisico, aggiungendo rumore gaussiano oppure riciano.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import default_te, signal_numpy


@dataclass
class SynthData:
    """Contenitore del dataset sintetico.

    Attributi:
        te:        (K,) tempi di eco (ms).
        signals:   (H, W, K) immagini multi-eco rumorose.
        s0_true:   (H, W) mappa ground-truth di S0.
        t2_true:   (H, W) mappa ground-truth di T2 (ms).
        noise_sigma: deviazione standard del rumore usata.
        noise_type:  "gaussian" oppure "rician".
    """

    te: np.ndarray
    signals: np.ndarray
    s0_true: np.ndarray
    t2_true: np.ndarray
    noise_sigma: float
    noise_type: str


def _disk(h: int, w: int, cy: float, cx: float, r: float) -> np.ndarray:
    """Maschera booleana di un disco di raggio r centrato in (cy, cx)."""
    yy, xx = np.mgrid[0:h, 0:w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def make_ground_truth(height: int = 64, width: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Crea mappe ground-truth (S0, T2) spazialmente variabili.

    Lo sfondo ha un gradiente dolce di T2; vengono sovrapposti dischi con T2
    distinti (es. tessuti/lesioni). S0 ha un gradiente indipendente.

    Returns:
        (s0_true, t2_true) entrambi con shape (height, width). T2 in ms.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    yn = yy / max(height - 1, 1)
    xn = xx / max(width - 1, 1)

    # T2 di base: gradiente dolce 40..90 ms
    t2 = 40.0 + 50.0 * (0.5 * yn + 0.5 * xn)

    # Regioni con T2 distinti (centro/raggi scelti relativamente alle dimensioni)
    t2[_disk(height, width, height * 0.30, width * 0.35, min(height, width) * 0.14)] = 120.0
    t2[_disk(height, width, height * 0.68, width * 0.65, min(height, width) * 0.11)] = 25.0
    t2[_disk(height, width, height * 0.55, width * 0.25, min(height, width) * 0.08)] = 160.0

    # S0: gradiente indipendente 80..120 (unita' arbitrarie)
    s0 = 80.0 + 40.0 * (1.0 - yn) * (0.5 + 0.5 * xn)

    return s0.astype(np.float64), t2.astype(np.float64)


def add_noise(
    clean: np.ndarray,
    sigma: float,
    noise_type: str = "gaussian",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Aggiunge rumore ai segnali puliti.

    Args:
        clean:      segnali senza rumore, shape qualsiasi.
        sigma:      deviazione standard del rumore (stesse unita' del segnale).
        noise_type: "gaussian" o "rician".
        rng:        generatore numpy opzionale (per riproducibilita').

    Returns:
        Segnali rumorosi, stessa shape di clean.
    """
    if rng is None:
        rng = np.random.default_rng()
    if sigma <= 0:
        return clean.copy()
    if noise_type == "gaussian":
        return clean + rng.normal(0.0, sigma, size=clean.shape)
    if noise_type == "rician":
        # Rumore riciano: magnitudine di due canali gaussiani indipendenti.
        real = clean + rng.normal(0.0, sigma, size=clean.shape)
        imag = rng.normal(0.0, sigma, size=clean.shape)
        return np.sqrt(real**2 + imag**2)
    raise ValueError(f"noise_type sconosciuto: {noise_type!r}")


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
    """Genera un dataset sintetico completo e riproducibile.

    Args:
        height, width: dimensioni dell'immagine.
        num_echoes:    numero K di tempi di eco.
        te_min, te_max: estremi dei TE (ms).
        noise_sigma:   sigma del rumore.
        noise_type:    "gaussian" o "rician".
        seed:          seme per la riproducibilita'.

    Returns:
        Un oggetto SynthData.
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
