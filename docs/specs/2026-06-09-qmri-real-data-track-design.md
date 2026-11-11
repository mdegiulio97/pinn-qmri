# Spec — "Real data" track for pinn-qmri (T2* mapping on real multi-echo GRE)

Date: 2026-06-09
Repo: `pinn-qmri` (extension; the existing synthetic track stays unchanged)

## Context and goal

`pinn-qmri` demonstrates the **T2** relaxometry inversion (PINN vs least squares) on
**synthetic data with ground truth**. To turn it into an applied-research project with a
clinical perspective, we add a **real-data track**: apply the same mono-exponential
inversion to **real multi-echo GRE images** and validate it **without ground truth**.

Narrative: *validated on synthetic ground truth → applied to real data with an honest
GT-free validation (held-out echo cross-validation)*.

## Dataset (fixed and verified)

- **OpenNeuro ds007116** — "Penn LEAD". **CC0** license (verified in
  `dataset_description.json`).
- Anatomical **MEGRE** (multi-echo gradient echo) sequence, **4 echoes**, magnitude+phase,
  3D `.nii.gz`.
- **EchoTime = 7.5, 15.0, 22.5, 30.0 ms**; 3T; TR 35 ms (from verified BIDS sidecars).
- Reference subject: `sub-19861`, `ses-1`, files `*_part-mag_MEGRE.nii.gz`
  (we use **magnitude** only; phase is needed for QSM, out of scope).
- **Public S3** access verified (HTTP 200, ~1.67 MB/echo → ~7 MB total):
  `https://s3.amazonaws.com/openneuro.org/ds007116/sub-19861/ses-1/anat/<file>`
- **Raw data gitignored**; only derived maps/figures + `real_metrics.json` are committed.
- Physical model: `S(TE) = S0 · exp(-TE / T2*)` (mono-exponential magnitude). In a GRE
  setting the parameter is **T2\*** (not T2): documented explicitly, with its confounds.

## Architecture (new components; existing code reused without forks)

Direct reuse of `physics.py`, `model.py`, `train.py` (PINN), `baseline.py` (least squares),
`metrics.py`. The forward model is identical; only the T2→T2* meaning changes in the docs.

1. **`src/pinn_qmri/real_data.py`** — I/O and preparation of the real data.
   - `load_megre(paths: list[Path], te_ms: list[float]) -> MEGRE` where `MEGRE` is a small
     dataclass `{signals: np.ndarray [n_vox, K], te_ms: np.ndarray [K], mask: np.ndarray, shape, slice_index}`.
   - `read_te_from_sidecars(json_paths) -> list[float]` — reads `EchoTime` (s→ms) from the
     BIDS JSON files. **No hardcoded TE.**
   - `select_slice(volume_echoes, axis=2, strategy="max-signal") -> idx` — picks a
     representative axial slice (the one with maximum energy within the mask).
   - `brain_mask(echo1, method="otsu") -> bool array` — simple mask via Otsu thresholding of
     the first echo (no heavy dependency; optionally a provided mask is used if present).
   - New dependency: **nibabel** (NIfTI reading). Added to `pyproject.toml`.

2. **`scripts/download_real.py`** — idempotent fetch from S3.
   - Downloads the 4 magnitude files (+ their JSON) of `sub-19861/ses-1` into `data/real/`
     (gitignored) via `urllib`/`requests`, only if not already present. Accession+version and
     URL documented at the top. Prints the CC0 license and the dataset citation.

3. **`scripts/evaluate_real.py`** — evaluation on the real data.
   - Loads via `real_data.load_megre`, takes one slice, applies the mask.
   - Runs **PINN** and **least squares** to estimate `(S0, T2*)` per voxel.
   - **Held-out echo cross-validation** (leave-one-echo-out): for each echo `j`, fit on
     `K-1` echoes, predict echo `j`, **NRMSE** on the voxels in the mask; mean over the 4
     folds, for both methods. This is the main GT-free metric.
   - **Physical plausibility**: median/IQR T2\* within the mask, compared with the expected
     ranges at 3T (reported, not asserted).
   - Output: `results/real_metrics.json` (NRMSE per method+fold, T2\* stats, dataset info) and
     figures `results/real_t2star_maps.png` (PINN vs LS vs echo), `results/real_heldout_nrmse.png`.

4. **App (optional, gated)** — if `data/real/` exists, a section of `app/main.py` shows the
   real maps. Low priority; not blocking.

## GT-free validation: held-out echo cross-validation

Without ground truth we cannot measure the error on T2*. Instead we measure the **predictive
ability of the estimated physical model**: if `(S0, T2*)` estimated from `K-1` echoes predicts
the left-out echo well, the inversion is consistent.

- For each `j ∈ {1..K}`: estimate `(S0,T2*)` from echoes `{1..K}\{j}`; predict `Ŝ(TE_j)`;
  compute `NRMSE_j = RMSE(Ŝ(TE_j), S(TE_j)) / mean(S(TE_j))` on the voxels in the mask.
- Report the mean and per-fold values for **PINN** and **LS**. A fair and honest comparison.
- Known and stated limitation: with **4 echoes** leave-one-out is minimal; this is a demo,
  not a study.

## Tests (offline, deterministic, no network)

`tests/test_real_data.py`:
- **Fixture**: writes to `tmp_path` a small synthetic multi-echo NIfTI (e.g. 8×8×3, 4 echoes)
  generated from known `(S0,T2*)` with `physics.py`, + JSON sidecars with `EchoTime`.
- Tests: `read_te_from_sidecars` returns the correct TEs; `load_megre` returns consistent
  shapes and a non-empty mask; the held-out CV function on data **with known GT** gives a
  small NRMSE (the model is exact) → checks that the pipeline works.
- No downloads in the tests (the network is used only in `scripts/download_real.py`).

## Short report with references (`report/qmri_real_data.md`)

- Problem: qMRI inversion on real data without GT; why an alternative validation is needed.
- Method: PINN vs LS + held-out echo CV; mono-exponential magnitude model.
- Results: NRMSE table (from `real_metrics.json`) + T2\* maps.
- Limitations: single subject/slice; mono- vs multi-component; **T2\*** and confounds
  (B0 inhomogeneity, no phase correction); only 4 echoes.
- References with **DOI/PMID** (nothing made up): Raissi 2019 (PINN, already in the repo);
  a reference for T2\*/R2\* mapping from multi-echo GRE; dataset ds007116 with
  accession+version. Every reference verified before use.

## Governance / honesty

- **CC0** dataset, cited with accession and version; **raw data not committed**.
- Research demo, **not** for clinical use; explicitly **T2\*** (not T2).
- Consistent with the project's citation discipline (DOI/PMID for every reference).

## Out of scope (YAGNI)

Multi-component T2; QSM/phase processing; whole volume (one slice is used); atlas
registration; multi-subject analysis; app integration beyond the gated visualisation.

## Completion criteria

1. `uv run pytest -q` green (including the new offline tests).
2. `uv run ruff check .` clean.
3. `uv run python scripts/download_real.py` downloads ~7 MB into `data/real/` (gitignored).
4. `uv run python scripts/evaluate_real.py` produces `results/real_metrics.json` + 2 figures
   with actual numbers (NRMSE PINN vs LS, plausible T2\*).
5. README updated with a "real data" section + short report with references.
6. Commit and push to `mdegiulio97/pinn-qmri`.
