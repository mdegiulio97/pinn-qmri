"""Streamlit demo: PINN for T2 relaxometry vs least squares.

Generates synthetic data, shows the multi-echo images and, on user request
(button), trains the PINN and compares it with least-squares fitting.

To keep the first render light, the app does NOT train anything until the user
presses the button; the default parameters use a small image and few epochs.

Run:
    uv run streamlit run app/main.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Makes the package importable when the app runs via `streamlit run`.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_qmri.baseline import fit_image  # noqa: E402
from pinn_qmri.metrics import all_metrics  # noqa: E402
from pinn_qmri.model import get_device  # noqa: E402
from pinn_qmri.synth import generate  # noqa: E402
from pinn_qmri.train import TrainConfig, train_pinn  # noqa: E402

st.set_page_config(page_title="PINN qMRI — T2 relaxometry", layout="wide")
st.title("PINN qMRI — T2 map estimation")
st.caption(
    "Unsupervised physics-informed neural network for T2 relaxometry, "
    "compared with per-pixel least-squares fitting. Fully synthetic data."
)

with st.sidebar:
    st.header("Parameters")
    size = st.slider("Image size (px)", 16, 64, 32, step=8)
    num_echoes = st.slider("Number of echoes K", 4, 12, 8)
    noise_sigma = st.slider("Noise sigma", 0.0, 10.0, 3.0, step=0.5)
    epochs = st.slider("PINN epochs", 50, 600, 200, step=50)
    run_btn = st.button("Generate and estimate", type="primary")

st.markdown(
    "The signal follows **S(TE) = S0 · exp(-TE/T2)**. The PINN learns (S0, T2) "
    "by minimising only the consistency with this physical model, without seeing "
    "the ground truth."
)

if not run_btn:
    st.info("Set the parameters in the sidebar and press **Generate and estimate**.")
    st.stop()

with st.spinner("Generating synthetic data..."):
    data = generate(height=size, width=size, num_echoes=num_echoes, noise_sigma=noise_sigma, seed=0)

st.subheader("Multi-echo images (noisy)")
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
with st.spinner(f"Training the PINN on {device}..."):
    t0 = time.perf_counter()
    cfg = TrainConfig(epochs=epochs, seed=0)
    _, _, t2_pinn, history = train_pinn(data.signals, data.te, cfg, device=device)
    t_pinn = time.perf_counter() - t0

with st.spinner("Per-pixel least-squares fitting..."):
    t0 = time.perf_counter()
    _, t2_ls = fit_image(data.te, data.signals)
    t_ls = time.perf_counter() - t0

m_pinn = all_metrics(t2_pinn, data.t2_true)
m_ls = all_metrics(t2_ls, data.t2_true)

st.subheader("Estimated T2 maps")
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
c1.metric("PINN — T2 MAE (ms)", f"{m_pinn['mae']:.2f}", help=f"time {t_pinn:.2f}s")
c2.metric("Least-squares — T2 MAE (ms)", f"{m_ls['mae']:.2f}", help=f"time {t_ls:.2f}s")

st.caption(
    f"Device: {device}. PINN {t_pinn:.2f}s (all pixels in one batch) vs "
    f"LS {t_ls:.2f}s (per-pixel loop). "
    f"RMSE: PINN {m_pinn['rmse']:.2f} / LS {m_ls['rmse']:.2f} ms."
)
