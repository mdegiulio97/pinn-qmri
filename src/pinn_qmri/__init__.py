"""pinn_qmri — PINN per rilassometria T2 (qMRI) su dati sintetici.

Moduli principali:
- physics:  forward model S(TE) = S0 * exp(-TE/T2) e generazione segnali
- synth:    immagini sintetiche con mappe S0, T2 ground-truth + rumore
- model:    rete PINN (MLP per-pixel)
- train:    loop di training con physics loss (non supervisionato)
- baseline: fitting least-squares per-pixel (scipy)
- metrics:  MAE / RMSE / MAPE sulle mappe
"""

from . import baseline, metrics, model, physics, synth, train

__all__ = ["physics", "synth", "model", "train", "baseline", "metrics"]

__version__ = "0.1.0"
