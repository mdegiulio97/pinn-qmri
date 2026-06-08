"""Demo Streamlit: PINN per rilassometria T2 vs least-squares.

Genera dati sintetici, mostra le immagini multi-eco e, su richiesta dell'utente
(pulsante), addestra la PINN e la confronta con il fitting least-squares.

Per restare leggera al primo render l'app NON addestra nulla finche' l'utente non
preme il pulsante; i parametri di default usano un'immagine piccola e poche epoche.

Avvio:
    uv run streamlit run app/main.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Permette di importare il package quando l'app gira via `streamlit run`.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_qmri.baseline import fit_image  # noqa: E402
from pinn_qmri.metrics import all_metrics  # noqa: E402
from pinn_qmri.model import get_device  # noqa: E402
from pinn_qmri.synth import generate  # noqa: E402
from pinn_qmri.train import TrainConfig, train_pinn  # noqa: E402

st.set_page_config(page_title="PINN qMRI — rilassometria T2", layout="wide")
st.title("PINN qMRI — stima della mappa T2")
st.caption(
    "Physics-Informed Neural Network non supervisionata per la rilassometria T2, "
    "confrontata con il fitting least-squares per-pixel. Dati interamente sintetici."
)

with st.sidebar:
    st.header("Parametri")
    size = st.slider("Dimensione immagine (px)", 16, 64, 32, step=8)
    num_echoes = st.slider("Numero di echi K", 4, 12, 8)
    noise_sigma = st.slider("Sigma rumore", 0.0, 10.0, 3.0, step=0.5)
    epochs = st.slider("Epoche PINN", 50, 600, 200, step=50)
    run_btn = st.button("Genera e stima", type="primary")

st.markdown(
    "Il segnale segue **S(TE) = S0 · exp(-TE/T2)**. La PINN impara (S0, T2) "
    "minimizzando solo la coerenza con questo modello fisico, senza vedere la "
    "ground truth."
)

if not run_btn:
    st.info("Imposta i parametri nella barra laterale e premi **Genera e stima**.")
    st.stop()

with st.spinner("Generazione dati sintetici..."):
    data = generate(height=size, width=size, num_echoes=num_echoes, noise_sigma=noise_sigma, seed=0)

st.subheader("Immagini multi-eco (rumorose)")
ncol = min(num_echoes, 6)
fig_e, axes = plt.subplots(1, ncol, figsize=(2.0 * ncol, 2.2))
if ncol == 1:
    axes = [axes]
for i in range(ncol):
    axes[i].imshow(data.signals[:, :, i], cmap="gray")
    axes[i].set_title(f"TE={data.te[i]:.0f} ms")
    axes[i].axis("off")
st.pyplot(fig_e)

device = get_device()
with st.spinner(f"Addestramento PINN su {device}..."):
    t0 = time.perf_counter()
    cfg = TrainConfig(epochs=epochs, seed=0)
    _, _, t2_pinn, history = train_pinn(data.signals, data.te, cfg, device=device)
    t_pinn = time.perf_counter() - t0

with st.spinner("Fitting least-squares per-pixel..."):
    t0 = time.perf_counter()
    _, t2_ls = fit_image(data.te, data.signals)
    t_ls = time.perf_counter() - t0

m_pinn = all_metrics(t2_pinn, data.t2_true)
m_ls = all_metrics(t2_ls, data.t2_true)

st.subheader("Mappe T2 stimate")
vmax = float(np.percentile(data.t2_true, 99))
fig_m, axm = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, img, title in zip(
    axm, [data.t2_true, t2_pinn, t2_ls], ["Ground truth", "PINN", "Least-squares"]
):
    im = ax.imshow(img, cmap="viridis", vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig_m.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ms")
st.pyplot(fig_m)

c1, c2 = st.columns(2)
c1.metric("PINN — MAE T2 (ms)", f"{m_pinn['mae']:.2f}", help=f"tempo {t_pinn:.2f}s")
c2.metric("Least-squares — MAE T2 (ms)", f"{m_ls['mae']:.2f}", help=f"tempo {t_ls:.2f}s")

st.caption(
    f"Device: {device}. PINN {t_pinn:.2f}s (tutti i pixel in batch) vs "
    f"LS {t_ls:.2f}s (loop per-pixel). "
    f"RMSE: PINN {m_pinn['rmse']:.2f} / LS {m_ls['rmse']:.2f} ms."
)
