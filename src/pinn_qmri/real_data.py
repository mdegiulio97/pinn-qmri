"""I/O and preparation of REAL multi-echo GRE data for T2* mapping.

Unlike the synthetic track, there is no ground truth here. This module only:
reads echo times from the BIDS sidecars, loads the magnitude NIfTI files of the
individual echoes, extracts a representative slice with a brain mask, and provides
the held-out echo cross-validation (GT-free validation) used by `evaluate_real.py`.

The physical forward model is the same as in the synthetic track (mono-exponential
`S(TE)=S0*exp(-TE/T2*)`): in a GRE setting the estimated parameter is T2*, not T2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np

from . import physics

# A fitting function takes (te (K,), signals (H,W,K)) and returns (s0_map, t2_map).
FitFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class MEGRE:
    """A multi-echo GRE slice ready for the inversion.

    Attributes:
        signals:     (H, W, K) magnitude signals of the slice at the K echoes.
        te_ms:       (K,) echo times in milliseconds.
        mask:        (H, W) boolean brain mask.
        slice_index: index of the extracted slice along the axial axis.
        volume_shape: shape (X, Y, Z) of the original 3D volume.
    """

    signals: np.ndarray
    te_ms: np.ndarray
    mask: np.ndarray
    slice_index: int
    volume_shape: tuple[int, int, int]


def read_te_from_sidecars(json_paths: list[Path]) -> np.ndarray:
    """Read ``EchoTime`` (in seconds, BIDS) from the sidecars and convert to ms.

    Args:
        json_paths: paths of the BIDS JSON files, one per echo, in echo order.

    Returns:
        (K,) echo times in milliseconds.
    """
    te_ms: list[float] = []
    for p in json_paths:
        meta = json.loads(Path(p).read_text())
        if "EchoTime" not in meta:
            raise KeyError(f"'EchoTime' missing from sidecar {p}")
        te_ms.append(float(meta["EchoTime"]) * 1000.0)
    return np.asarray(te_ms, dtype=np.float64)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu threshold (maximises the between-class variance) on a 1D/2D array."""
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
    """Boolean brain mask from a magnitude slice via Otsu thresholding."""
    thr = otsu_threshold(echo_slice)
    return np.asarray(echo_slice, dtype=np.float64) > thr


def select_slice(volume_4d: np.ndarray, axis: int = 2) -> int:
    """Index of the slice with the highest first-echo energy along ``axis``.

    Args:
        volume_4d: (X, Y, Z, K) multi-echo volume.
        axis:      axis along which the slice is chosen (default Z=2).
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
    """Load the magnitude NIfTI files of the individual echoes and extract a slice + mask.

    Args:
        nifti_paths: paths of the 3D NIfTI files, one per echo, in increasing TE order.
        json_paths:  BIDS sidecars to read the TEs from (alternative to ``te_ms``).
        te_ms:       explicit TEs in ms (take precedence if given).
        slice_index: axial slice to extract; if None it is chosen automatically.

    Returns:
        A :class:`MEGRE` object.
    """
    if te_ms is None:
        if json_paths is None:
            raise ValueError("Provide either json_paths or te_ms.")
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
    """RMSE normalised by the mean of the true signal, within the mask."""
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
    """Leave-one-echo-out cross-validation (validation without ground truth).

    For each echo ``j``, ``(S0, T2*)`` is estimated from the other ``K-1`` echoes
    and the left-out echo is predicted; the residual (NRMSE within the mask)
    measures the physical consistency of the inversion without any ground truth.

    Args:
        signals: (H, W, K) signals of the slice.
        te_ms:   (K,) echo times in ms.
        fit_fn:  function (te, signals) -> (s0_map, t2_map).
        mask:    (H, W) boolean brain mask.

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
