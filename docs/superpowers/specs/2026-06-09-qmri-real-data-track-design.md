# Spec — Track "dati reali" per pinn-qmri (T2* mapping su multi-echo GRE reale)

Data: 2026-06-09
Repo: `pinn-qmri` (estensione, il track sintetico esistente resta invariato)

## Contesto e obiettivo

`pinn-qmri` dimostra l'inversione di rilassometria **T2** (PINN vs least-squares) su dati
**sintetici con ground truth**. Per elevarlo a progetto "flagship" di ricerca applicata con
prospettiva clinica, aggiungiamo un **track su dati reali**: applicare la stessa inversione
mono-esponenziale a **immagini multi-echo GRE reali** e validarla **senza ground truth**.

Narrativa: *validato su ground truth sintetica → applicato al dato reale con una validazione
GT-free onesta (held-out echo cross-validation)*.

## Dataset (bloccato e verificato)

- **OpenNeuro ds007116** — "Penn LEAD". Licenza **CC0** (verificata in
  `dataset_description.json`).
- Sequenza **MEGRE** (multi-echo gradient echo) anatomica, **4 echi**, magnitude+phase,
  `.nii.gz` 3D.
- **EchoTime = 7.5, 15.0, 22.5, 30.0 ms**; 3T; TR 35 ms (da sidecar BIDS verificati).
- Soggetto di riferimento: `sub-19861`, `ses-1`, file `*_part-mag_MEGRE.nii.gz`
  (usiamo solo **magnitude**; la fase serve a QSM, fuori scope).
- Accesso **S3 pubblico** verificato (HTTP 200, ~1.67 MB/echo → ~7 MB totali):
  `https://s3.amazonaws.com/openneuro.org/ds007116/sub-19861/ses-1/anat/<file>`
- **Dati grezzi gitignorati**; si committano solo mappe/figure derivate + `real_metrics.json`.
- Modello fisico: `S(TE) = S0 · exp(-TE / T2*)` (mono-esponenziale magnitude). In contesto GRE
  il parametro è **T2\*** (non T2): documentato esplicitamente, con i suoi confondenti.

## Architettura (nuovi componenti; codice esistente riusato senza fork)

Riuso diretto di `physics.py`, `model.py`, `train.py` (PINN), `baseline.py` (least-squares),
`metrics.py`. Il forward model è identico; cambia solo la semantica T2→T2* nella doc.

1. **`src/pinn_qmri/real_data.py`** — I/O e preparazione del dato reale.
   - `load_megre(paths: list[Path], te_ms: list[float]) -> MEGRE` dove `MEGRE` è un piccolo
     dataclass `{signals: np.ndarray [n_vox, K], te_ms: np.ndarray [K], mask: np.ndarray, shape, slice_index}`.
   - `read_te_from_sidecars(json_paths) -> list[float]` — legge `EchoTime` (s→ms) dai JSON
     BIDS. **Niente TE hardcoded.**
   - `select_slice(volume_echoes, axis=2, strategy="max-signal") -> idx` — sceglie una slice
     assiale rappresentativa (quella a massima energia entro la mask).
   - `brain_mask(echo1, method="otsu") -> bool array` — mask semplice via soglia Otsu sul
     primo eco (nessuna dipendenza pesante; opzionale uso di una mask fornita se presente).
   - Dipendenza nuova: **nibabel** (lettura NIfTI). Aggiunta a `pyproject.toml`.

2. **`scripts/download_real.py`** — fetch idempotente da S3.
   - Scarica i 4 file magnitude (+ relativi JSON) di `sub-19861/ses-1` in `data/real/`
     (gitignorato) via `urllib`/`requests`, solo se non già presenti. Accession+versione e URL
     documentati in testa. Stampa licenza CC0 e citazione del dataset.

3. **`scripts/evaluate_real.py`** — valutazione sul dato reale.
   - Carica via `real_data.load_megre`, prende una slice, applica la mask.
   - Esegue **PINN** e **least-squares** per stimare `(S0, T2*)` per-voxel.
   - **Held-out echo cross-validation** (leave-one-echo-out): per ogni eco `j`, fit su
     `K-1` echi, predizione dell'eco `j`, **NRMSE** sui voxel in mask; media sui 4 fold, per
     entrambi i metodi. È la metrica GT-free principale.
   - **Plausibilità fisica**: T2\* mediano/IQR entro la mask, confronto con range attesi a 3T
     (riportato, non asserito).
   - Output: `results/real_metrics.json` (NRMSE per metodo+fold, T2\* stats, info dataset) e
     figure `results/real_t2star_maps.png` (PINN vs LS vs eco), `results/real_heldout_nrmse.png`.

4. **App (opzionale, gated)** — se `data/real/` esiste, una sezione di `app/main.py` mostra le
   mappe reali. Bassa priorità; non blocca il flagship.

## Validazione GT-free: held-out echo cross-validation

Senza ground truth non possiamo misurare l'errore su T2*. Misuriamo invece la **capacità
predittiva del modello fisico stimato**: se `(S0, T2*)` stimati da `K-1` echi predicono bene
l'eco escluso, l'inversione è coerente.

- Per ogni `j ∈ {1..K}`: stima `(S0,T2*)` da echi `{1..K}\{j}`; predice `Ŝ(TE_j)`; calcola
  `NRMSE_j = RMSE(Ŝ(TE_j), S(TE_j)) / mean(S(TE_j))` sui voxel in mask.
- Riporta media e per-fold per **PINN** e **LS**. Confronto equo e onesto.
- Limite noto e dichiarato: con **4 echi** il leave-one-out è minimale; è una demo, non uno
  studio.

## Test (offline, deterministici, nessuna rete)

`tests/test_real_data.py`:
- **Fixture**: scrive in `tmp_path` un piccolo NIfTI multi-echo sintetico (es. 8×8×3, 4 echi)
  generato da `(S0,T2*)` noti con `physics.py`, + JSON sidecar con `EchoTime`.
- Test: `read_te_from_sidecars` ritorna i TE corretti; `load_megre` ritorna shape coerenti e
  mask non vuota; la funzione di held-out CV su dato **con GT nota** dà NRMSE piccolo (il
  modello è esatto) → verifica che la pipeline funzioni.
- Nessun download nei test (la rete è solo in `scripts/download_real.py`).

## Mini-report citato (`report/qmri_real_data.md`)

- Problema: inversione qMRI su dato reale senza GT; perché serve una validazione alternativa.
- Metodo: PINN vs LS + held-out echo CV; modello mono-esponenziale magnitude.
- Risultati: tabella NRMSE (da `real_metrics.json`) + mappe T2\*.
- Limiti: singolo soggetto/slice; mono- vs multi-componente; **T2\*** e confondenti
  (disomogeneità B0, niente correzione di fase); solo 4 echi.
- Citazioni con **DOI/PMID** (no fabbricazioni): Raissi 2019 (PINN, già nel repo);
  riferimento per T2\*/R2\* mapping da multi-echo GRE; dataset ds007116 con accession+versione.
  Ogni citazione verificata prima dell'uso.

## Governance / onestà

- Dataset **CC0**, citato con accession e versione; **dati grezzi non committati**.
- Demo di ricerca, **non** uso clinico; esplicito che è **T2\*** (non T2).
- Coerente con la citation discipline del progetto (DOI/PMID per ogni riferimento).

## Fuori scope (YAGNI)

T2 multi-componente; QSM/elaborazione di fase; volume intero (si usa una slice); registrazione
ad atlante; analisi multi-soggetto; integrazione app oltre la visualizzazione gated.

## Criteri di completamento

1. `uv run pytest -q` verde (inclusi i nuovi test, offline).
2. `uv run ruff check .` pulito.
3. `uv run python scripts/download_real.py` scarica ~7 MB in `data/real/` (gitignorato).
4. `uv run python scripts/evaluate_real.py` produce `results/real_metrics.json` + 2 figure con
   numeri reali (NRMSE PINN vs LS, T2\* plausibili).
5. README aggiornato con sezione "dati reali" + mini-report citato.
6. Commit e push su `mdegiulio97/pinn-qmri`, senza riferimenti a Claude.
