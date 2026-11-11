# T2\* mapping on real multi-echo GRE data: PINN vs least squares, validated without ground truth

Short report on the *real-data track* of `pinn-qmri`. It extends the synthetic
demonstration (with ground truth) to real in-vivo images, where **no ground truth exists**,
and asks the central question of applied research: *how do you validate a qMRI inversion on
real data?*

## Data

OpenNeuro **ds007116** ("Penn LEAD"), **CC0** license [4]. Anatomical **MEGRE** (multi-echo
gradient echo) sequence, subject `sub-19861/ses-1`, 4-echo magnitude
(**TE = 7.5 / 15 / 22.5 / 30 ms**, 3T). One representative axial slice is used, with a brain
mask from Otsu thresholding. The raw data are not redistributed: they are downloaded from S3
with `scripts/download_real.py`. The parameter estimated from a GRE sequence is **T2\***
(not T2), with the related confounds [1, 2].

## Method

Same mono-exponential forward model as the synthetic track, `S(TE) = S0 · exp(-TE/T2*)`:

- **PINN** (unsupervised): a per-voxel MLP maps the signal vector → `(S0, T2*)`,
  trained by minimising only the physical consistency with the forward model [3].
- Per-voxel **least squares** (`scipy.optimize.curve_fit`): classical reference.

**Validation without ground truth: held-out echo cross-validation.** For each echo `j`,
`(S0, T2*)` is estimated from the other `K−1` echoes and the left-out echo is predicted; the
residual (NRMSE within the mask) measures the predictive ability of the estimated physical
model, without any known truth.

## Results

Actual numbers from `results/real_metrics.json` (slice 46, 9706 voxels in the mask).

**T2\* estimate** (median, IQR in ms):

| Method | Median T2\* | IQR |
|---|---|---|
| PINN | **50.4** | 39.0 – 58.5 |
| Least-squares | 51.8 | 38.4 – 63.2 |

Both values are **physiologically plausible** for grey/white matter at 3T; the PINN shows a
slightly narrower IQR, consistent with the implicit regularisation of the network shared
across voxels.

**Held-out echo cross-validation** (NRMSE of the left-out echo, lower is better):

| Fold (left-out echo) | LS | PINN |
|---|---|---|
| echo 1 (TE 7.5 ms) | 0.204 | **0.106** |
| echo 2 (TE 15 ms) | 0.057 | 0.058 |
| echo 3 (TE 22.5 ms) | 0.051 | 0.053 |
| echo 4 (TE 30 ms) | 0.089 | **0.085** |
| **mean** | 0.100 | **0.075** |

The PINN **generalises better** to the left-out echo (mean NRMSE 0.075 vs 0.100), mainly on
the first echo (high signal): per-voxel fitting optimises each pixel in isolation and
extrapolates worse, while the shared network learns a more robust representation.

## Methodological note: a real bug, diagnosed

When the PINN tuned on synthetic data was applied directly, the estimated T2\* collapsed to a
constant value of ~600 ms (held-out NRMSE ≈ 1.0). **Cause identified** (systematic
debugging): the **background** voxels (≈49% of the slice) have a first echo ≈ 0; normalising
by the first echo produced exploding ratios (~10⁵) that dominated the loss and collapsed the
shared network to a constant T2\*. The synthetic track did not show this because it has no
background. **Fix**: train the PINN **only on the voxels inside the mask**
(`train_pinn(..., mask=...)`), with a dedicated regression test. After the fix, median T2\*
is 50.4 ms and the held-out NRMSE is competitive.

## Limitations

- **Single subject, single slice**: a demonstration, not a study.
- **T2\* (not T2)**: B0 field inhomogeneity confounds are not corrected (no phase
  information, magnitude only) [1, 2].
- **Mono-exponential model**: the real decay can be multi-component.
- **Only 4 echoes at short TE (7.5–30 ms)**: minimal leave-one-out; the inversion is
  intrinsically ill-conditioned in this regime.
- **Not a clinical tool**: a research demonstration on public CC0 data.

## References

1. Yablonskiy DA, Sukstanskii AL, Luo J, Wang X. *Voxel spread function method for correction of
   magnetic field inhomogeneity effects in quantitative gradient-echo-based MRI.* Magn Reson Med.
   2013. DOI: [10.1002/mrm.24585](https://doi.org/10.1002/mrm.24585)
2. *T2\* quantification using multi-echo gradient echo sequences: a comparative study of different
   readout gradients.* Sci Rep. 2023. DOI:
   [10.1038/s41598-023-28265-0](https://doi.org/10.1038/s41598-023-28265-0)
3. Raissi M, Perdikaris P, Karniadakis GE. *Physics-informed neural networks.* J Comput Phys.
   2019;378:686–707. DOI:
   [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)
4. OpenNeuro dataset **ds007116** "Penn LEAD", CC0 license.
   [https://openneuro.org/datasets/ds007116](https://openneuro.org/datasets/ds007116)
