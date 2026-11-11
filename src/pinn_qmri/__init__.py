"""pinn_qmri — PINN for T2/T2* relaxometry (qMRI) on synthetic and real data.

Main modules:
- physics:  forward model S(TE) = S0 * exp(-TE/T2) and signal generation
- synth:    synthetic images with ground-truth S0, T2 maps + noise
- model:    PINN network (per-pixel MLP)
- train:    training loop with physics loss (unsupervised)
- baseline: per-pixel least-squares fitting (scipy)
- metrics:  MAE / RMSE / MAPE on the maps
- real_data: real multi-echo GRE loading, brain mask, held-out echo cross-validation
"""

from . import baseline, metrics, model, physics, synth, train

__all__ = ["physics", "synth", "model", "train", "baseline", "metrics"]

__version__ = "0.1.0"
