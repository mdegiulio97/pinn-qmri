"""Rete PINN per la stima per-pixel di (S0, T2) dai segnali multi-eco.

La rete e' un semplice MLP che mappa il vettore dei K segnali di un pixel ->
2 valori positivi (S0, T2). La positivita' e' garantita da una Softplus in uscita.

L'aspetto "physics-informed" NON sta nell'architettura ma nella loss
(vedi train.py): la rete non vede mai la ground-truth, viene addestrata solo
imponendo la coerenza con il forward model S(TE)=S0*exp(-TE/T2).
"""

from __future__ import annotations

import torch
from torch import nn


def get_device() -> torch.device:
    """Device auto-detect: cuda se disponibile, altrimenti cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PINN(nn.Module):
    """MLP per-pixel: input (K segnali) -> output (S0, T2), entrambi positivi.

    Args:
        num_echoes: numero K di echi in ingresso.
        hidden:     larghezza degli strati nascosti.
        depth:      numero di strati nascosti.
        s0_scale:   scala iniziale di S0 (per ben condizionare l'output).
        t2_scale:   scala iniziale di T2 in ms.
    """

    def __init__(
        self,
        num_echoes: int,
        hidden: int = 64,
        depth: int = 3,
        s0_scale: float = 100.0,
        t2_scale: float = 80.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(num_echoes, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()
        self.register_buffer("scale", torch.tensor([s0_scale, t2_scale], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predice (S0, T2) positivi.

        Args:
            x: (N, K) segnali normalizzati (o grezzi) per N pixel.

        Returns:
            (s0, t2): due tensori (N,) strettamente positivi.
        """
        raw = self.net(x)
        pos = self.softplus(raw) * self.scale
        s0 = pos[:, 0]
        t2 = pos[:, 1]
        return s0, t2
