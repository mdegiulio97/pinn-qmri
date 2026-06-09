"""Loop di training non supervisionato della PINN con physics loss.

La loss principale e' la consistenza fisica:
    L_phys = mean( || S0*exp(-TE/T2) - segnali_misurati ||^2 )
Opzionalmente si aggiunge un prior di smoothness Total-Variation sulle mappe
predette (utile a rumore elevato).

Il training e' interamente self-supervised: la ground-truth NON viene mai usata.
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
    """Iperparametri del training.

    Attributi:
        epochs:     numero di epoche (full-batch su tutti i pixel).
        lr:         learning rate Adam.
        hidden:     larghezza hidden della rete.
        depth:      numero di strati nascosti.
        lambda_tv:  peso del prior Total-Variation (0 = disattivato).
        seed:       seme per riproducibilita'.
        verbose:    se True stampa la loss periodicamente.
    """

    epochs: int = 400
    lr: float = 1e-3
    hidden: int = 64
    depth: int = 3
    lambda_tv: float = 0.0
    seed: int = 0
    verbose: bool = False


def _tv_loss(maps: torch.Tensor) -> torch.Tensor:
    """Total Variation 2D (anisotropa) di una mappa (H, W)."""
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
    """Addestra la PINN sui segnali e restituisce le mappe stimate.

    Args:
        signals: (H, W, K) immagini multi-eco rumorose.
        te:      (K,) tempi di eco (ms).
        config:  iperparametri (TrainConfig).
        device:  device torch; se None usa get_device().
        mask:    (H, W) booleana opzionale. Se fornita, la rete e' addestrata SOLO
                 sui voxel in mask: indispensabile sui dati reali, dove i voxel di
                 background (primo eco ~0) producono normalizzazioni esplosive che
                 altrimenti collassano la rete a un T2 costante. Con mask, il prior
                 Total-Variation e' disattivato e le mappe restituite sono azzerate
                 fuori dalla mask. Default None = comportamento invariato.

    Returns:
        (modello, s0_pred (H,W), t2_pred (H,W), storia_loss).
    """
    if config is None:
        config = TrainConfig()
    if device is None:
        device = get_device()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    h, w, k = signals.shape
    x_np = signals.reshape(-1, k).astype(np.float32)  # (N, K)

    # Normalizzazione per-pixel: si divide il vettore di segnali per l'ampiezza del
    # primo eco. Cosi' l'input alla rete e' adimensionale e ben condizionato, la
    # physics loss e' calcolata in scala normalizzata (S0_norm ~ 1) e T2 - essendo
    # invariante di scala - viene predetto direttamente in millisecondi.
    # Il fattore di scala riporta S0 alla scala fisica originale a posteriori.
    scale = np.maximum(np.abs(x_np[:, :1]), 1e-6)  # (N,1) ~ ampiezza primo eco
    x_norm = (x_np / scale).astype(np.float32)

    x = torch.from_numpy(x_norm).to(device)
    target_norm = torch.from_numpy(x_norm).to(device)  # target in scala normalizzata
    scale_t = torch.from_numpy(scale.astype(np.float32)).to(device).squeeze(1)  # (N,)
    te_t = torch.from_numpy(te.astype(np.float32)).to(device)

    # In scala normalizzata S0 e' ~1; uso s0_scale=2 per non forzare valori troppo alti.
    model = PINN(num_echoes=k, hidden=config.hidden, depth=config.depth, s0_scale=2.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    mse = nn.MSELoss()

    # Sottoinsieme di training: tutti i voxel, oppure solo quelli in mask.
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
        recon = signal_torch(te_t, s0_norm, t2)  # (N_tr, K) in scala normalizzata
        loss = mse(recon, target_tr)

        # Il prior Total-Variation richiede la griglia (H,W) completa: applicabile
        # solo quando si addestra su tutta la slice (mask is None).
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
