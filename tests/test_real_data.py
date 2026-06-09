"""Test offline del track dati reali (nessun download).

Si costruisce un piccolo NIfTI multi-eco SINTETICO con (S0, T2*) noti, scritto in
una directory temporanea, e si verifica che: il parsing dei TE dai sidecar BIDS
funzioni, il loader restituisca forme coerenti e una mask non vuota, e la
held-out echo cross-validation su un modello esatto dia NRMSE piccolo.
"""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np

from pinn_qmri.baseline import fit_image
from pinn_qmri.real_data import (
    heldout_echo_cv,
    load_megre,
    nrmse,
    otsu_mask,
    read_te_from_sidecars,
)

TE_MS = [7.5, 15.0, 22.5, 30.0]


def _write_synthetic_megre(tmp_path, te_ms=TE_MS, shape=(10, 10, 3)):
    """Scrive K NIfTI 3D (un eco ciascuno) + sidecar JSON con EchoTime (s)."""
    rng = np.random.default_rng(0)
    s0 = rng.uniform(80.0, 120.0, size=shape)
    t2 = rng.uniform(30.0, 60.0, size=shape)
    # Bordo a zero per simulare il background (separabile da Otsu).
    s0[0, :, :] = s0[-1, :, :] = s0[:, 0, :] = s0[:, -1, :] = 0.0
    te = np.asarray(te_ms, dtype=np.float64)
    signal = s0[..., None] * np.exp(-te / np.maximum(t2, 1e-6)[..., None])  # (X,Y,Z,K)

    nii_paths, json_paths = [], []
    affine = np.eye(4)
    for k, t in enumerate(te_ms):
        npath = tmp_path / f"echo-{k + 1}_part-mag_MEGRE.nii.gz"
        nib.save(nib.Nifti1Image(signal[..., k].astype(np.float32), affine), npath)
        jpath = tmp_path / f"echo-{k + 1}_part-mag_MEGRE.json"
        jpath.write_text(json.dumps({"EchoTime": t / 1000.0}))  # BIDS: secondi
        nii_paths.append(npath)
        json_paths.append(jpath)
    return nii_paths, json_paths


def test_read_te_from_sidecars_converts_seconds_to_ms(tmp_path):
    _, json_paths = _write_synthetic_megre(tmp_path)
    te = read_te_from_sidecars(json_paths)
    assert np.allclose(te, TE_MS)


def test_load_megre_shapes_and_mask(tmp_path):
    nii_paths, json_paths = _write_synthetic_megre(tmp_path)
    megre = load_megre(nii_paths, json_paths)
    h, w, k = megre.signals.shape
    assert k == len(TE_MS)
    assert megre.signals.ndim == 3  # slice 2D x K echi
    assert megre.mask.shape == (h, w)
    assert megre.mask.sum() > 0  # mask non vuota
    assert np.allclose(megre.te_ms, TE_MS)


def test_otsu_mask_separates_foreground(tmp_path):
    nii_paths, _ = _write_synthetic_megre(tmp_path)
    echo1 = np.asanyarray(nib.load(str(nii_paths[0])).dataobj)[:, :, 1]
    mask = otsu_mask(echo1)
    assert mask.dtype == bool
    assert 0 < mask.sum() < mask.size  # separa, non prende tutto


def test_heldout_cv_small_nrmse_on_exact_model(tmp_path):
    nii_paths, json_paths = _write_synthetic_megre(tmp_path)
    megre = load_megre(nii_paths, json_paths)

    def ls_fit(te, signals):
        return fit_image(te, signals)

    out = heldout_echo_cv(megre.signals, megre.te_ms, ls_fit, megre.mask)
    assert len(out["per_fold"]) == len(TE_MS)
    # Il modello e' esatto (dati senza rumore): il residuo deve essere piccolo.
    assert out["mean"] < 0.02


def test_nrmse_zero_when_identical():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.ones_like(a, dtype=bool)
    assert nrmse(a, a, mask) == 0.0


def test_pinn_with_mask_ignores_garbage_background():
    """Su dato reale il background (primo eco ~0) produce normalizzazioni esplosive
    che, addestrando su tutta la slice, collassano la rete a un T2 costante alto.
    Allenando SOLO entro la mask la PINN recupera il T2* corretto del tessuto."""
    from pinn_qmri.train import TrainConfig, train_pinn

    h, w = 16, 16
    te = np.array([7.5, 15.0, 22.5, 30.0])
    k = len(te)
    s0 = np.zeros((h, w))
    t2 = np.full((h, w), 50.0)
    brain = np.zeros((h, w), dtype=bool)
    brain[:, : w // 2] = True
    s0[brain] = 100.0
    signals = s0[..., None] * np.exp(-te / np.maximum(t2, 1e-6)[..., None])  # (H,W,K)

    # Background: primo eco quasi-zero + echi successivi rumorosi -> rapporti esplosivi.
    rng = np.random.default_rng(0)
    bg = ~brain
    signals[bg, 0] = 1e-3
    signals[bg, 1:] = rng.uniform(0.5, 1.5, size=(int(bg.sum()), k - 1))

    _, _, t2_pred, _ = train_pinn(signals, te, TrainConfig(epochs=300, seed=0), mask=brain)
    assert 40.0 < float(np.median(t2_pred[brain])) < 62.0
