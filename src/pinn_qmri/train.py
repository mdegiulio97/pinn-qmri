"""Unsupervised PINN training loop with a physics loss.

The main loss is physical consistency:
    L_phys = mean( || S0*exp(-TE/T2) - measured_signals ||^2 )
Optionally, a Total-Variation smoothness prior on the predicted maps is added
(useful at high noise).

Training is fully self-supervised: the ground truth is NEVER used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .model import PINN, get_device
from .physics import signal_torch


@dataclass
class TrainConfig:
    """Training hyperparameters.

    Attributes:
        epochs:     number of epochs (full batch over all pixels).
        lr:         Adam learning rate.
        hidden:     hidden width of the network.
        depth:      number of hidden layers.
        lambda_tv:  weight of the Total-Variation prior (0 = disabled).
        seed:       seed for reproducibility.
        verbose:    if True, print the loss periodically.
    """

    epochs: int = 400
    lr: float = 1e-3
    hidden: int = 64
    depth: int = 3
    lambda_tv: float = 0.0
    seed: int = 0
    verbose: bool = False


def _tv_loss(maps: torch.Tensor) -> torch.Tensor:
    """2D (anisotropic) Total Variation of a map (H, W)."""
    dh = torch.abs(maps[1:, :] - maps[:-1, :]).mean()
    dw = torch.abs(maps[:, 1:] - maps[:, :-1]).mean()
    return dh + dw


def train_pinn(
    signals: np.ndarray,
    te: np.ndarray,
    config: TrainConfig | None = None,
    device: torch.device | None = None,
    mask: np.ndarray | None = None,
) -> tuple[PINN, np.ndarray, np.ndarray, list[float]]:
    """Train the PINN on the signals and return the estimated maps.

    Args:
        signals: (H, W, K) noisy multi-echo images.
        te:      (K,) echo times (ms).
        config:  hyperparameters (TrainConfig).
        device:  torch device; if None, get_device() is used.
        mask:    optional boolean (H, W). If given, the network is trained ONLY
                 on the voxels inside the mask: this is required on real data, where
                 background voxels (first echo ~0) produce exploding normalisations
                 that otherwise collapse the network to a constant T2. With a mask,
                 the Total-Variation prior is disabled and the returned maps are set
                 to zero outside the mask. Default None = unchanged behaviour.

    Returns:
        (model, s0_pred (H,W), t2_pred (H,W), loss_history).
    """
    if config is None:
        config = TrainConfig()
    if device is None:
        device = get_device()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    h, w, k = signals.shape
    x_np = signals.reshape(-1, k).astype(np.float32)  # (N, K)

    # Per-pixel normalisation: the signal vector is divided by the amplitude of the
    # first echo. This makes the network input dimensionless and well conditioned,
    # the physics loss is computed on the normalised scale (S0_norm ~ 1) and T2,
    # being scale invariant, is predicted directly in milliseconds.
    # The scale factor brings S0 back to the original physical scale afterwards.
    scale = np.maximum(np.abs(x_np[:, :1]), 1e-6)  # (N,1) ~ first-echo amplitude
    x_norm = (x_np / scale).astype(np.float32)

    x = torch.from_numpy(x_norm).to(device)
    target_norm = torch.from_numpy(x_norm).to(device)  # target on the normalised scale
    scale_t = torch.from_numpy(scale.astype(np.float32)).to(device).squeeze(1)  # (N,)
    te_t = torch.from_numpy(te.astype(np.float32)).to(device)

    # On the normalised scale S0 is ~1; s0_scale=2 avoids forcing values that are too high.
    model = PINN(num_echoes=k, hidden=config.hidden, depth=config.depth, s0_scale=2.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    mse = nn.MSELoss()

    # Training subset: all voxels, or only those inside the mask.
    if mask is not None:
        idx = torch.from_numpy(np.flatnonzero(np.asarray(mask, dtype=bool).reshape(-1))).to(device)
        x_tr = x[idx]
        target_tr = target_norm[idx]
    else:
        x_tr = x
        target_tr = target_norm

    history: list[float] = []
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        s0_norm, t2 = model(x_tr)
        recon = signal_torch(te_t, s0_norm, t2)  # (N_tr, K) on the normalised scale
        loss = mse(recon, target_tr)

        # The Total-Variation prior needs the full (H,W) grid: it applies only
        # when training on the whole slice (mask is None).
        if config.lambda_tv > 0 and mask is None:
            t2_map = t2.reshape(h, w)
            loss = loss + config.lambda_tv * _tv_loss(t2_map)

        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
        if config.verbose and (epoch % 50 == 0 or epoch == config.epochs - 1):
            print(f"epoch {epoch:4d}  loss {loss.item():.4e}")

    model.eval()
    with torch.no_grad():
        s0_norm, t2 = model(x)
        s0 = (s0_norm * scale_t).cpu().numpy().reshape(h, w)
        t2_map = t2.cpu().numpy().reshape(h, w)

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        s0 = np.where(m, s0, 0.0)
        t2_map = np.where(m, t2_map, 0.0)

    return model, s0.astype(np.float64), t2_map.astype(np.float64), history
