"""Forward model fisico della rilassometria T2 e generazione dei segnali.

Il segnale multi-eco in una sequenza T2 segue un decadimento mono-esponenziale:

    S(TE) = S0 * exp(-TE / T2)

dove:
- TE  : tempo di eco (ms)
- S0  : ampiezza del segnale a TE -> 0 (densita' protonica pesata)
- T2  : tempo di rilassamento trasversale (ms)

Questo modulo fornisce il forward model sia in NumPy (per la generazione dei dati
e per la baseline least-squares) sia in PyTorch (per la physics loss della PINN).
"""

from __future__ import annotations

import numpy as np

try:  # torch e' opzionale per le sole funzioni numpy, ma sempre presente nel progetto
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def signal_numpy(te: np.ndarray, s0: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Forward model NumPy: S(TE) = S0 * exp(-TE/T2).

    Args:
        te:  tempi di eco, shape (K,) in ms.
        s0:  ampiezza, scalare o array broadcast-abile con t2.
        t2:  tempo di rilassamento (ms), stessa forma di s0.

    Returns:
        Segnali con shape (..., K), dove ... e' la forma di s0/t2.
    """
    te = np.asarray(te, dtype=np.float64)
    s0 = np.asarray(s0, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64)
    # Evita divisioni per zero: clip di T2 a un minimo positivo.
    t2_safe = np.clip(t2, 1e-6, None)
    # broadcast: (...,1) * exp(-(K,)/(...,1)) -> (...,K)
    return s0[..., None] * np.exp(-te / t2_safe[..., None])


def signal_torch(te: "torch.Tensor", s0: "torch.Tensor", t2: "torch.Tensor") -> "torch.Tensor":
    """Forward model PyTorch (differenziabile) per la physics loss.

    Args:
        te:  tensore (K,) dei tempi di eco (ms).
        s0:  tensore (N,) ampiezze predette.
        t2:  tensore (N,) T2 predetti (ms).

    Returns:
        Tensore (N, K) dei segnali ricostruiti.
    """
    t2_safe = torch.clamp(t2, min=1e-6)
    return s0[:, None] * torch.exp(-te[None, :] / t2_safe[:, None])


def default_te(num_echoes: int = 8, te_min: float = 10.0, te_max: float = 80.0) -> np.ndarray:
    """Tempi di eco equispaziati di default (ms).

    Args:
        num_echoes: numero di echi K.
        te_min:     primo TE (ms).
        te_max:     ultimo TE (ms).
    """
    return np.linspace(te_min, te_max, num_echoes, dtype=np.float64)
