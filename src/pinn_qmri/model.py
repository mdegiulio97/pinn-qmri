"""PINN for per-pixel estimation of (S0, T2) from multi-echo signals.

The network is a simple MLP that maps the vector of K signals of a pixel ->
2 positive values (S0, T2). Positivity is enforced by a Softplus at the output.

The "physics-informed" part is NOT in the architecture but in the loss
(see train.py): the network never sees the ground truth, it is trained only
by enforcing consistency with the forward model S(TE)=S0*exp(-TE/T2).
"""

from __future__ import annotations

import torch
from torch import nn


def get_device() -> torch.device:
    """Device auto-detection: cuda if available, otherwise cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PINN(nn.Module):
    """Per-pixel MLP: input (K signals) -> output (S0, T2), both positive.

    Args:
        num_echoes: number K of input echoes.
        hidden:     width of the hidden layers.
        depth:      number of hidden layers.
        s0_scale:   initial scale of S0 (to keep the output well conditioned).
        t2_scale:   initial scale of T2 in ms.
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
        """Predict positive (S0, T2).

        Args:
            x: (N, K) normalised (or raw) signals for N pixels.

        Returns:
            (s0, t2): two strictly positive (N,) tensors.
        """
        raw = self.net(x)
        pos = self.softplus(raw) * self.scale
        s0 = pos[:, 0]
        t2 = pos[:, 1]
        return s0, t2
