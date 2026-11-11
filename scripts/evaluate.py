"""PINN vs least-squares comparison over several noise levels.

For each noise level:
- generate synthetic data with known ground truth,
- estimate the T2 map with the (unsupervised) PINN and with per-pixel LS fitting,
- compute MAE/RMSE/MAPE on T2 and the run times.

Outputs:
- results/metrics.json
- results/maps_comparison.png   (T2 maps: GT, PINN, LS at the middle noise level)
- results/mae_vs_noise.png      (T2 MAE vs sigma for PINN and LS)

Usage:
    uv run python scripts/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pinn_qmri.baseline import fit_image  # noqa: E402
from pinn_qmri.metrics import all_metrics  # noqa: E402
from pinn_qmri.model import get_device  # noqa: E402
from pinn_qmri.synth import generate  # noqa: E402
from pinn_qmri.train import TrainConfig, train_pinn  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run(
    height: int,
    width: int,
    num_echoes: int,
    noise_sigmas: list[float],
    epochs: int,
    seed: int,
) -> dict:
    device = get_device()
    results: dict = {
        "config": {
            "height": height,
            "width": width,
            "num_echoes": num_echoes,
            "epochs": epochs,
            "seed": seed,
            "device": str(device),
            "noise_sigmas": noise_sigmas,
        },
        "per_noise": [],
    }

    maps_for_plot = None  # (sigma, gt, pinn, ls) at the middle level
    mid_idx = len(noise_sigmas) // 2

    for idx, sigma in enumerate(noise_sigmas):
        data = generate(
            height=height,
            width=width,
            num_echoes=num_echoes,
            noise_sigma=sigma,
            seed=seed,
        )

        # --- PINN ---
        t0 = time.perf_counter()
        cfg = TrainConfig(epochs=epochs, seed=seed)
        _, _, t2_pinn, _ = train_pinn(data.signals, data.te, cfg, device=device)
        t_pinn = time.perf_counter() - t0

        # --- Least-squares ---
        t0 = time.perf_counter()
        _, t2_ls = fit_image(data.te, data.signals)
        t_ls = time.perf_counter() - t0

        m_pinn = all_metrics(t2_pinn, data.t2_true)
        m_ls = all_metrics(t2_ls, data.t2_true)

        entry = {
            "noise_sigma": sigma,
            "pinn": {**m_pinn, "time_s": t_pinn},
            "ls": {**m_ls, "time_s": t_ls},
        }
        results["per_noise"].append(entry)
        print(
            f"sigma={sigma:5.1f} | PINN MAE={m_pinn['mae']:6.2f} ms ({t_pinn:5.2f}s) "
            f"| LS MAE={m_ls['mae']:6.2f} ms ({t_ls:5.2f}s)"
        )

        if idx == mid_idx:
            maps_for_plot = (sigma, data.t2_true, t2_pinn, t2_ls)

    _plot_maps(maps_for_plot)
    _plot_mae_vs_noise(results)
    return results


def _plot_maps(maps) -> None:
    sigma, gt, pinn, ls = maps
    vmax = float(np.percentile(gt, 99))
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, img, title in zip(
        axes, [gt, pinn, ls], ["Ground truth T2", "PINN T2", "Least-squares T2"]
    ):
        im = ax.imshow(img, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ms")
    fig.suptitle(f"T2 maps at sigma={sigma}")
    fig.tight_layout()
    out = RESULTS_DIR / "maps_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[ok] figure saved: {out}")


def _plot_mae_vs_noise(results: dict) -> None:
    sig = [e["noise_sigma"] for e in results["per_noise"]]
    pinn = [e["pinn"]["mae"] for e in results["per_noise"]]
    ls = [e["ls"]["mae"] for e in results["per_noise"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sig, pinn, "o-", label="PINN")
    ax.plot(sig, ls, "s-", label="Least-squares")
    ax.set_xlabel("noise sigma")
    ax.set_ylabel("T2 MAE (ms)")
    ax.set_title("Noise robustness: T2 MAE vs sigma")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "mae_vs_noise.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[ok] figure saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PINN vs least-squares evaluation.")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--num-echoes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-sigmas",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 6.0],
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = run(
        height=args.height,
        width=args.width,
        num_echoes=args.num_echoes,
        noise_sigmas=args.noise_sigmas,
        epochs=args.epochs,
        seed=args.seed,
    )
    out = RESULTS_DIR / "metrics.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[ok] metrics saved: {out}")


if __name__ == "__main__":
    main()
