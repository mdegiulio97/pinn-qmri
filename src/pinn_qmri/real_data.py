"""I/O e preparazione di dati multi-echo GRE REALI per il T2* mapping.

A differenza del track sintetico, qui non esiste ground truth. Questo modulo si
limita a: leggere i tempi di eco dai sidecar BIDS, caricare i NIfTI magnitude dei
singoli echi, estrarre una slice rappresentativa con una brain mask, e fornire la
held-out echo cross-validation (validazione GT-free) usata da `evaluate_real.py`.

Il forward model fisico e' lo stesso del track sintetico (mono-esponenziale
`S(TE)=S0*exp(-TE/T2*)`): nel contesto GRE il parametro stimato e' T2*, non T2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np

from . import physics

# Una funzione di fitting prende (te (K,), signals (H,W,K)) e ritorna (s0_map, t2_map).
FitFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class MEGRE:
    """Una slice multi-echo GRE pronta per l'inversione.

    Attributi:
        signals:     (H, W, K) segnali magnitude della slice ai K echi.
        te_ms:       (K,) tempi di eco in millisecondi.
        mask:        (H, W) brain mask booleana.
        slice_index: indice della slice estratta lungo l'asse assiale.
        volume_shape: forma (X, Y, Z) del volume 3D originale.
    """

    signals: np.ndarray
    te_ms: np.ndarray
    mask: np.ndarray
    slice_index: int
    volume_shape: tuple[int, int, int]


def read_te_from_sidecars(json_paths: list[Path]) -> np.ndarray:
    """Legge ``EchoTime`` (in secondi, BIDS) dai sidecar e converte in ms.

    Args:
        json_paths: percorsi dei JSON BIDS, uno per eco, nell'ordine degli echi.

    Returns:
        (K,) tempi di eco in millisecondi.
    """
    te_ms: list[float] = []
    for p in json_paths:
        meta = json.loads(Path(p).read_text())
        if "EchoTime" not in meta:
            raise KeyError(f"'EchoTime' assente nel sidecar {p}")
        te_ms.append(float(meta["EchoTime"]) * 1000.0)
    return np.asarray(te_ms, dtype=np.float64)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Soglia di Otsu (massimizza la varianza inter-classe) su un array 1D/2D."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0 or finite.min() == finite.max():
        return float(finite.min()) if finite.size else 0.0
    hist, edges = np.histogram(finite, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight = np.cumsum(hist)
    total = weight[-1]
    weight_bg = weight
    weight_fg = total - weight
    cum = np.cumsum(hist * centers)
    mean_total = cum[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_bg = cum / np.maximum(weight_bg, 1)
        mean_fg = (mean_total - cum) / np.maximum(weight_fg, 1)
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    idx = int(np.nanargmax(between))
    return float(centers[idx])


def otsu_mask(echo_slice: np.ndarray) -> np.ndarray:
    """Brain mask booleana da una slice magnitude via soglia di Otsu."""
    thr = otsu_threshold(echo_slice)
    return np.asarray(echo_slice, dtype=np.float64) > thr


def select_slice(volume_4d: np.ndarray, axis: int = 2) -> int:
    """Indice della slice a massima energia del primo eco lungo ``axis``.

    Args:
        volume_4d: (X, Y, Z, K) volume multi-eco.
        axis:      asse lungo cui scegliere la slice (default Z=2).
    """
    echo0 = volume_4d[..., 0]
    sum_axes = tuple(a for a in range(echo0.ndim) if a != axis)
    energy = echo0.sum(axis=sum_axes)
    return int(np.argmax(energy))


def load_megre(
    nifti_paths: list[Path],
    json_paths: list[Path] | None = None,
    te_ms: np.ndarray | None = None,
    slice_index: int | None = None,
) -> MEGRE:
    """Carica i NIfTI magnitude dei singoli echi ed estrae una slice + mask.

    Args:
        nifti_paths: percorsi dei NIfTI 3D, uno per eco, in ordine crescente di TE.
        json_paths:  sidecar BIDS per leggere i TE (alternativo a ``te_ms``).
        te_ms:       TE in ms espliciti (ha precedenza se forniti).
        slice_index: slice assiale da estrarre; se None si sceglie automaticamente.

    Returns:
        Un oggetto :class:`MEGRE`.
    """
    if te_ms is None:
        if json_paths is None:
            raise ValueError("Fornire json_paths oppure te_ms.")
        te_ms = read_te_from_sidecars(json_paths)
    te_ms = np.asarray(te_ms, dtype=np.float64)

    echoes = [np.asanyarray(nib.load(str(p)).dataobj, dtype=np.float64) for p in nifti_paths]
    volume_4d = np.stack(echoes, axis=-1)  # (X, Y, Z, K)
    vx, vy, vz, _ = volume_4d.shape

    if slice_index is None:
        slice_index = select_slice(volume_4d, axis=2)
    signals = volume_4d[:, :, slice_index, :]  # (H, W, K)
    mask = otsu_mask(signals[:, :, 0])
    return MEGRE(
        signals=signals,
        te_ms=te_ms,
        mask=mask,
        slice_index=slice_index,
        volume_shape=(vx, vy, vz),
    )


def nrmse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    """RMSE normalizzato (per la media del segnale vero) entro la mask."""
    m = np.asarray(mask, dtype=bool)
    if m.sum() == 0:
        return float("nan")
    diff = (np.asarray(pred) - np.asarray(true))[m]
    denom = np.abs(np.asarray(true)[m]).mean()
    rmse = float(np.sqrt(np.mean(diff**2)))
    return rmse / denom if denom > 0 else float("nan")


def heldout_echo_cv(
    signals: np.ndarray,
    te_ms: np.ndarray,
    fit_fn: FitFn,
    mask: np.ndarray,
) -> dict:
    """Leave-one-echo-out cross-validation (validazione senza ground truth).

    Per ogni eco ``j`` si stima ``(S0, T2*)`` dagli altri ``K-1`` echi e si predice
    l'eco escluso; il residuo (NRMSE entro la mask) misura la coerenza fisica
    dell'inversione senza alcuna ground truth.

    Args:
        signals: (H, W, K) segnali della slice.
        te_ms:   (K,) tempi di eco in ms.
        fit_fn:  funzione (te, signals) -> (s0_map, t2_map).
        mask:    (H, W) brain mask booleana.

    Returns:
        ``{"per_fold": [nrmse_0, ...], "mean": float}``.
    """
    te_ms = np.asarray(te_ms, dtype=np.float64)
    k = signals.shape[-1]
    per_fold: list[float] = []
    for j in range(k):
        keep = [i for i in range(k) if i != j]
        s0_map, t2_map = fit_fn(te_ms[keep], signals[:, :, keep])
        pred_j = physics.signal_numpy(te_ms[j : j + 1], s0_map, t2_map)[..., 0]
        per_fold.append(nrmse(pred_j, signals[:, :, j], mask))
    return {"per_fold": per_fold, "mean": float(np.nanmean(per_fold))}
