"""Genera e salva il dataset sintetico (idempotente).

Non scarica nulla dalla rete: i dati sono interamente generati dal forward model.
Il file viene salvato in data/synth_dataset.npz. Se esiste gia' ed e' coerente con
i parametri richiesti, NON viene rigenerato (idempotenza).

Uso:
    uv run python scripts/download_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pinn_qmri.synth import generate

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASET_PATH = DATA_DIR / "synth_dataset.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera il dataset sintetico T2 (idempotente).")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--num-echoes", type=int, default=8)
    parser.add_argument("--noise-sigma", type=float, default=2.0)
    parser.add_argument("--noise-type", type=str, default="gaussian", choices=["gaussian", "rician"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="rigenera anche se gia' presente")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_PATH.exists() and not args.force:
        with np.load(DATASET_PATH) as d:
            same = (
                d["signals"].shape[:2] == (args.height, args.width)
                and d["signals"].shape[2] == args.num_echoes
                and float(d["noise_sigma"]) == args.noise_sigma
                and str(d["noise_type"]) == args.noise_type
                and int(d["seed"]) == args.seed
            )
        if same:
            print(f"[ok] dataset gia' presente e coerente: {DATASET_PATH}")
            return
        print("[info] parametri diversi: rigenero il dataset.")

    data = generate(
        height=args.height,
        width=args.width,
        num_echoes=args.num_echoes,
        noise_sigma=args.noise_sigma,
        noise_type=args.noise_type,
        seed=args.seed,
    )
    np.savez_compressed(
        DATASET_PATH,
        te=data.te,
        signals=data.signals,
        s0_true=data.s0_true,
        t2_true=data.t2_true,
        noise_sigma=data.noise_sigma,
        noise_type=data.noise_type,
        seed=args.seed,
    )
    print(f"[ok] dataset salvato: {DATASET_PATH}")
    print(
        f"     shape segnali={data.signals.shape}, K={len(data.te)}, "
        f"rumore={data.noise_type} sigma={data.noise_sigma}"
    )


if __name__ == "__main__":
    main()
