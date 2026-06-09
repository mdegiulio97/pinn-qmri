"""T2* mapping su dati multi-echo GRE REALI (OpenNeuro ds007116) — PINN vs LS.

Senza ground truth, la validazione e' una held-out echo cross-validation
(leave-one-echo-out): si stima (S0, T2*) da K-1 echi e si predice l'eco escluso,
misurando l'NRMSE entro la brain mask. Si confrontano PINN e least-squares.

Produce:
- results/real_metrics.json     (NRMSE held-out per metodo/fold, statistiche T2*)
- results/real_t2star_maps.png  (eco-1, mappa T2* LS, mappa T2* PINN)
- results/real_heldout_nrmse.png (NRMSE per fold: PINN vs LS)

Uso:
    uv run python scripts/download_real.py
    uv run python scripts/evaluate_real.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pinn_qmri.baseline import fit_pixel  # noqa: E402
from pinn_qmri.model import get_device  # noqa: E402
from pinn_qmri.real_data import heldout_echo_cv, load_megre  # noqa: E402
from pinn_qmri.train import TrainConfig, train_pinn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "real"
RESULTS_DIR = ROOT / "results"
SUBJECT = "sub-19861"
SESSION = "ses-1"


def _paths() -> tuple[list[Path], list[Path]]:
    nii, js = [], []
    for echo in range(1, 5):
        stem = f"{SUBJECT}_{SESSION}_echo-{echo}_part-mag_MEGRE"
        nii.append(DATA_DIR / f"{stem}.nii.gz")
        js.append(DATA_DIR / f"{stem}.json")
    return nii, js


def _ls_fit_masked(te: np.ndarray, signals: np.ndarray, mask: np.ndarray):
    """Least-squares per-pixel solo entro la mask (background lasciato a zero)."""
    h, w, _ = signals.shape
    s0 = np.zeros((h, w))
    t2 = np.zeros((h, w))
    for i, j in np.argwhere(mask):
        s0[i, j], t2[i, j] = fit_pixel(te, signals[i, j])
    return s0, t2


def _pinn_fit(te: np.ndarray, signals: np.ndarray, epochs: int, device, mask: np.ndarray):
    _, s0, t2, _ = train_pinn(
        signals, te, TrainConfig(epochs=epochs, seed=0), device=device, mask=mask
    )
    return s0, t2


def _stats(t2_map: np.ndarray, mask: np.ndarray) -> dict:
    vals = t2_map[mask]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return {
        "median_ms": float(np.median(vals)),
        "iqr_ms": [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--t2star-display-max", type=float, default=80.0)
    args = ap.parse_args()

    nii, js = _paths()
    missing = [p for p in nii + js if not p.exists()]
    if missing:
        raise SystemExit(
            "Dati reali assenti. Esegui prima:  uv run python scripts/download_real.py"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    megre = load_megre(nii, js)
    mask = megre.mask
    print(f"Slice {megre.slice_index}, volume {megre.volume_shape}, "
          f"TE={megre.te_ms.tolist()} ms, voxel in mask={int(mask.sum())}")

    # Held-out echo cross-validation (validazione senza ground truth).
    ls_fn = lambda te, sig: _ls_fit_masked(te, sig, mask)  # noqa: E731
    pinn_fn = lambda te, sig: _pinn_fit(te, sig, args.epochs, device, mask)  # noqa: E731
    print("Held-out echo CV: least-squares...")
    cv_ls = heldout_echo_cv(megre.signals, megre.te_ms, ls_fn, mask)
    print("Held-out echo CV: PINN...")
    cv_pinn = heldout_echo_cv(megre.signals, megre.te_ms, pinn_fn, mask)

    # Mappe T2* dal fit con tutti gli echi.
    s0_ls, t2_ls = _ls_fit_masked(megre.te_ms, megre.signals, mask)
    s0_pinn, t2_pinn = _pinn_fit(megre.te_ms, megre.signals, args.epochs, device, mask)

    metrics = {
        "dataset": {
            "accession": "ds007116",
            "name": "Penn LEAD",
            "license": "CC0",
            "subject": SUBJECT,
            "session": SESSION,
            "sequence": "MEGRE (magnitude)",
            "field_strength_T": 3,
            "te_ms": megre.te_ms.tolist(),
            "parameter": "T2star",
        },
        "slice_index": megre.slice_index,
        "n_mask_voxels": int(mask.sum()),
        "heldout_echo_nrmse": {
            "ls": cv_ls,
            "pinn": cv_pinn,
        },
        "t2star_stats_ms": {
            "ls": _stats(t2_ls, mask),
            "pinn": _stats(t2_pinn, mask),
        },
    }
    out_json = RESULTS_DIR / "real_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"\nNRMSE held-out (media): LS={cv_ls['mean']:.4f}  PINN={cv_pinn['mean']:.4f}")
    print(f"T2* mediano (ms): LS={metrics['t2star_stats_ms']['ls']['median_ms']:.1f}  "
          f"PINN={metrics['t2star_stats_ms']['pinn']['median_ms']:.1f}")

    # --- Figura 1: mappe ---
    vmax = args.t2star_display_max
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(megre.signals[:, :, 0].T, cmap="gray", origin="lower")
    axes[0].set_title(f"Eco 1 magnitude (TE={megre.te_ms[0]:.1f} ms)")
    for ax, t2, name in ((axes[1], t2_ls, "LS"), (axes[2], t2_pinn, "PINN")):
        disp = np.where(mask, np.clip(t2, 0, vmax), np.nan)
        im = ax.imshow(disp.T, cmap="viridis", origin="lower", vmin=0, vmax=vmax)
        ax.set_title(f"Mappa T2* — {name}")
        fig.colorbar(im, ax=ax, fraction=0.046, label="T2* (ms)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("T2* mapping su MEGRE reale (OpenNeuro ds007116, CC0)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "real_t2star_maps.png", dpi=120)
    plt.close(fig)

    # --- Figura 2: NRMSE held-out per fold ---
    k = len(megre.te_ms)
    x = np.arange(k)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.bar(x - 0.2, cv_ls["per_fold"], width=0.4, label=f"LS (media {cv_ls['mean']:.3f})")
    ax2.bar(x + 0.2, cv_pinn["per_fold"], width=0.4, label=f"PINN (media {cv_pinn['mean']:.3f})")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"eco {i + 1}\n({megre.te_ms[i]:.1f} ms)" for i in range(k)])
    ax2.set_ylabel("NRMSE eco escluso")
    ax2.set_title("Held-out echo cross-validation (senza ground truth)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(RESULTS_DIR / "real_heldout_nrmse.png", dpi=120)
    plt.close(fig2)

    print(f"Salvati: {out_json.name}, real_t2star_maps.png, real_heldout_nrmse.png")


if __name__ == "__main__":
    main()
