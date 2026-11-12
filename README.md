# pinn-qmri

[![CI](https://github.com/mdegiulio97/pinn-qmri/actions/workflows/ci.yml/badge.svg)](https://github.com/mdegiulio97/pinn-qmri/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**PINN for T2 relaxometry (qMRI)**

A small research project that estimates quantitative MRI parameters
(**T2** relaxometry) with a **physics-informed neural network (PINN)** trained
**without supervision**, and compares it with the classical baseline of per-pixel
**nonlinear least-squares fitting**.

## The idea in brief

The multi-echo signal in T2 relaxometry follows a mono-exponential decay:

```
S(TE) = S0 * exp(-TE / T2)
```

Starting from `K` images acquired at different echo times (`TE`), we want to
reconstruct, for every pixel, the maps `S0` (amplitude) and `T2` (relaxation time).

- **PINN** (`src/pinn_qmri/model.py`, `train.py`): a network `f_θ` maps the vector
  of the `K` signals of a pixel → `(S0, T2)`. It uses no labels: it is trained by
  minimising a **physical consistency loss**
  `|| S0·exp(-TE/T2) − measured_signals ||²`, in which the physical *forward model*
  is **embedded in the loss** (hence *physics-informed*). A spatial *smoothness*
  prior (Total-Variation regularisation on the maps) can be added.
- **Baseline** (`src/pinn_qmri/baseline.py`): per-pixel `scipy.optimize.curve_fit`
  on the same exponential model.

The main track uses **synthetic data with ground truth**: we generate spatially
varying `(S0, T2)` maps (`src/pinn_qmri/synth.py`), simulate the `K` images at
different `TE` (`physics.py`) and add Gaussian or Rician noise. This makes it
possible to measure the actual error (MAE) against the known truth and the
robustness to noise. A second track applies the same inversion to real MRI data
(see [Real-data track](#real-data-track-openneuro-ds007116-cc0)).

## Structure

```
pinn-qmri/
├── app/main.py                 # Streamlit demo (generate → train PINN → compare with LS)
├── src/pinn_qmri/
│   ├── physics.py              # forward model S(TE)=S0*exp(-TE/T2) + signal generation
│   ├── synth.py                # synthetic images with S0,T2 maps + noise (gauss/rician)
│   ├── model.py                # PINN network (per-pixel MLP)
│   ├── train.py                # training loop with physics loss (+ optional smoothness)
│   ├── baseline.py             # per-pixel least-squares fitting
│   ├── metrics.py              # MAE, RMSE, MAPE on the maps
│   └── real_data.py            # real multi-echo GRE loading, brain mask, held-out echo CV
├── scripts/download_data.py    # generates/saves the synthetic dataset (idempotent)
├── scripts/evaluate.py         # PINN vs LS over several noise levels → results/
├── scripts/download_real.py    # downloads the real OpenNeuro subject (idempotent)
├── scripts/evaluate_real.py    # PINN vs LS on real data + held-out echo CV → results/
├── report/qmri_real_data.md    # short report on the real-data track
├── tests/                      # pytest (offline, fast, cpu)
├── notebooks/01_validation.ipynb
├── results/                    # PNG figures + metrics JSON (committed)
└── data/                       # synthetic and downloaded data (gitignored)
```

## Installation and usage

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync --extra dev                       # create the venv and install dependencies
uv run ruff check .                        # lint
uv run pytest -q                           # tests
uv run python scripts/download_data.py     # generate the synthetic dataset in data/
uv run python scripts/evaluate.py          # PINN vs LS comparison → results/
```

`scripts/evaluate.py` writes `results/metrics.json` and the figures
`results/maps_comparison.png` and `results/mae_vs_noise.png`.

## Physics-informed approach

The network never sees the ground-truth `T2`. The only learning signal is
**consistency with the physical model**: given the predicted parameters `(S0, T2)`,
the signal `Ŝ(TE)=S0·exp(-TE/T2)` is reconstructed and its distance from the
measured signals is minimised. This is the same idea as the PINNs of Raissi et al.
(2019), applied here to an algebraic forward model (not a differential equation):
the physics of exponential decay is the constraint that regularises the inversion.

A *smoothness* prior (Total Variation, weight `lambda_tv`) can be added to
stabilise the estimate at high noise.

## Results

The numbers below are produced by `scripts/evaluate.py` on a 64×64 image,
8 echoes (TE = 10..80 ms), fixed seed. T2 in milliseconds. Times on CPU.

<!-- RESULTS_TABLE_START -->
Configuration: 64×64 image, K=8 echoes (TE = 10..80 ms), 500 PINN epochs,
seed=0, device **CPU**. Actual numbers from one run of `scripts/evaluate.py`.

| noise sigma | T2 MAE PINN (ms) | T2 MAE LS (ms) | RMSE PINN (ms) | RMSE LS (ms) | time PINN (s) | time LS (s) |
|---|---|---|---|---|---|---|
| 1.0 | 2.51 | **1.31** | 5.56 | **1.86** | 3.7 | 4.2 |
| 3.0 | 4.59 | **3.96** | 6.88 | **5.70** | 1.9 | 4.3 |
| 6.0 | **8.02** | 8.18 | **10.84** | 12.46 | 2.0 | 4.6 |
<!-- RESULTS_TABLE_END -->

Reading the numbers:
- At **low noise**, per-pixel least-squares fitting is more accurate (it is the
  optimal reference when the model is exact and the noise is low): MAE 1.31 vs
  2.51 ms.
- As the noise increases, the two methods get closer, and at **high sigma** the
  PINN becomes competitive or better in RMSE (10.84 vs 12.46 ms at sigma=6),
  benefiting from the implicit regularisation given by the network shared across
  pixels.
- Run times are of the same order on CPU; the PINN processes all pixels in a
  batch in a single forward pass, while LS runs an independent optimisation per
  pixel (the LS cost depends strongly on the number of iterations to converge).

Interpretation: the PINN produces a T2 map with an error comparable to (and, thanks
to the regularisation, more robust to noise than) per-pixel least-squares fitting,
processing all pixels in a single batched forward pass.

Selecting the best method at a given noise level is done in code:
`best_model({"pinn": m_pinn, "ls": m_ls}, key="rmse")` returns the name of the
model with the lowest RMSE, while `rank_models(...)` returns the full ranking
(useful for automatic summaries and regression checks on the results).

## Real-data track (OpenNeuro ds007116, CC0)

Besides the synthetic track, the project applies the same inversion to **real
multi-echo GRE data** (OpenNeuro **ds007116** "Penn LEAD", CC0 license; 4 echoes,
TE = 7.5/15/22.5/30 ms, 3T). Here there is **no ground truth**: validation is a
**held-out echo cross-validation** (`(S0, T2*)` is estimated from `K−1` echoes and the
left-out echo is predicted; NRMSE within the brain mask). In a GRE sequence the
parameter is **T2\***.

```bash
uv run python scripts/download_real.py      # ~7 MB from S3 (data gitignored)
uv run python scripts/evaluate_real.py      # PINN vs LS + held-out CV → results/
```

Actual numbers from `results/real_metrics.json` (slice 46, 9706 voxels in the mask):

| Method | Median T2\* (ms) | Held-out NRMSE (mean) |
|---|---|---|
| PINN | **50.4** (IQR 39–59) | **0.075** |
| Least-squares | 51.8 (IQR 38–63) | 0.100 |

Both give physiological T2\* values at 3T; the PINN, thanks to the implicit
regularisation of the shared network, **generalises better to the left-out echo**
(0.075 vs 0.100). Full report with references:
[`report/qmri_real_data.md`](report/qmri_real_data.md).

> Methodological note (documented debugging): when the synthetic PINN was applied to
> the real data, T2\* collapsed to ~600 ms because background voxels (first echo ≈ 0)
> produced exploding normalisations that dominated the loss. Fix: train the PINN
> **only within the brain mask** (`train_pinn(..., mask=...)`), with a regression test.

## Methodological notes and limitations

- Mono-exponential model: in real tissue T2 can be multi-component.
- Synthetic track: no real acquisition artefacts (motion, B1, etc.).
- The PINN here is a **per-pixel regressor** constrained by physics; it does not
  yet exploit deep spatial correlations (beyond the optional TV prior).

## References

- M. Raissi, P. Perdikaris, G.E. Karniadakis, *Physics-informed neural networks:
  A deep learning framework for solving forward and inverse problems involving
  nonlinear partial differential equations*, Journal of Computational Physics 378
  (2019) 686–707. DOI: [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

## Data and licenses

- **Synthetic data**: generated by the code (`src/pinn_qmri/synth.py`,
  `scripts/download_data.py`); nothing is downloaded for the synthetic track.
- **Real data**: multi-echo GRE brain MRI from OpenNeuro
  [ds007116](https://openneuro.org/datasets/ds007116) ("Penn LEAD"), released under
  **CC0**. The files are downloaded by `scripts/download_real.py` into `data/real/`
  and are **not committed** to this repository.
- Only derived results (figures and metrics JSON in `results/`) are committed.

## License

The code is released under the MIT license (see `LICENSE`). Third-party datasets
remain subject to their own licenses (see [Data and licenses](#data-and-licenses)).
