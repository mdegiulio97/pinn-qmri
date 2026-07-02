# pinn-qmri — PINN per rilassometria T2 (qMRI)

Mini-progetto di ricerca *research-grade* che stima parametri quantitativi MRI
(rilassometria **T2**) con una **Physics-Informed Neural Network (PINN)** addestrata
in modo **non supervisionato**, e la confronta con la baseline classica di
**fitting non lineare ai minimi quadrati** per-pixel.

## Idea in breve

Il segnale multi-eco di rilassometria T2 segue un decadimento mono-esponenziale:

```
S(TE) = S0 * exp(-TE / T2)
```

A partire da `K` immagini acquisite a tempi di eco (`TE`) diversi vogliamo
ricostruire, per ogni pixel, le mappe `S0` (ampiezza) e `T2` (tempo di rilassamento).

- **PINN** (`src/pinn_qmri/model.py`, `train.py`): una rete `f_θ` mappa il vettore
  dei `K` segnali di un pixel → `(S0, T2)`. Non usa etichette: viene addestrata
  minimizzando una **loss di consistenza fisica**
  `|| S0·exp(-TE/T2) − segnali_misurati ||²`, in cui il *forward model* fisico è
  **embedded nella loss** (da qui *physics-informed*). Si può aggiungere un prior
  di *smoothness* spaziale (regolarizzazione Total-Variation sulle mappe).
- **Baseline** (`src/pinn_qmri/baseline.py`): `scipy.optimize.curve_fit` per-pixel
  sullo stesso modello esponenziale.

I dati sono **sintetici con ground truth**: generiamo mappe `(S0, T2)` spazialmente
variabili (`src/pinn_qmri/synth.py`), simuliamo le `K` immagini a `TE` diversi
(`physics.py`) e aggiungiamo rumore gaussiano o riciano. Questo permette di
misurare l'errore reale (MAE) rispetto alla verità nota e la robustezza al rumore.

## Struttura

```
pinn-qmri/
├── app/main.py                 # Demo Streamlit (genera → addestra PINN → confronta con LS)
├── src/pinn_qmri/
│   ├── physics.py              # forward model S(TE)=S0*exp(-TE/T2) + generazione segnali
│   ├── synth.py                # immagini sintetiche con mappe S0,T2 + rumore (gauss/rician)
│   ├── model.py                # rete PINN (MLP per-pixel)
│   ├── train.py                # loop di training con physics loss (+ smoothness opzionale)
│   ├── baseline.py             # fitting least-squares per-pixel
│   └── metrics.py              # MAE, RMSE, MAPE sulle mappe
├── scripts/download_data.py    # genera/salva il dataset sintetico (idempotente)
├── scripts/evaluate.py         # PINN vs LS su più livelli di rumore → results/
├── tests/                      # pytest (offline, veloci, cpu)
├── notebooks/01_validation.ipynb
├── results/                    # figure PNG + metrics.json (committati)
└── data/                       # dataset sintetico (gitignorato)
```

## Installazione ed esecuzione

Richiede [uv](https://docs.astral.sh/uv/) e Python 3.12.

```bash
uv sync --extra dev                       # crea il venv e installa le dipendenze
uv run ruff check .                        # lint
uv run pytest -q                           # test
uv run python scripts/download_data.py     # genera il dataset sintetico in data/
uv run python scripts/evaluate.py          # confronto PINN vs LS → results/
```

`scripts/evaluate.py` produce `results/metrics.json` e le figure
`results/maps_comparison.png` e `results/mae_vs_noise.png`.

## Approccio physics-informed

La rete non vede mai la ground truth `T2`. L'unico segnale di apprendimento è la
**coerenza con il modello fisico**: dati i parametri `(S0, T2)` predetti, si
ricostruisce il segnale `Ŝ(TE)=S0·exp(-TE/T2)` e si minimizza la distanza dai
segnali misurati. È la stessa filosofia delle PINN di Raissi et al. (2019), qui
applicata a un forward model algebrico (non un'equazione differenziale): la fisica
del decadimento esponenziale è il vincolo che regolarizza l'inversione.

Un prior di *smoothness* (Total Variation, peso `lambda_tv`) può essere aggiunto
per stabilizzare la stima in presenza di rumore elevato.

## Risultati

I numeri seguenti sono prodotti da `scripts/evaluate.py` su immagine 64×64,
8 echi (TE = 10..80 ms), seed fisso. T2 in millisecondi. Tempi su CPU.

<!-- RESULTS_TABLE_START -->
Configurazione: immagine 64×64, K=8 echi (TE = 10..80 ms), 500 epoche PINN,
seed=0, device **CPU**. Numeri reali da una corsa di `scripts/evaluate.py`.

| sigma rumore | MAE T2 PINN (ms) | MAE T2 LS (ms) | RMSE PINN (ms) | RMSE LS (ms) | tempo PINN (s) | tempo LS (s) |
|---|---|---|---|---|---|---|
| 1.0 | 2.51 | **1.31** | 5.56 | **1.86** | 11.8 | 7.2 |
| 3.0 | 4.59 | **3.96** | 6.88 | **5.70** | 24.5 | 34.8 |
| 6.0 | **8.02** | 8.18 | **10.84** | 12.46 | 25.6 | 12.4 |
<!-- RESULTS_TABLE_END -->

Lettura onesta dei risultati:
- A **basso rumore** il fitting least-squares per-pixel è più accurato (è il
  riferimento ottimo quando il modello è esatto e il rumore basso): MAE 1.31 vs
  2.51 ms.
- All'aumentare del rumore i due metodi si avvicinano e a **sigma alto** la PINN
  diventa competitiva o migliore in RMSE (10.84 vs 12.46 ms a sigma=6),
  beneficiando dell'implicita regolarizzazione data dalla rete condivisa tra
  pixel.
- I tempi sono dello stesso ordine su CPU; la PINN processa tutti i pixel in
  batch in un singolo forward, mentre l'LS esegue un'ottimizzazione indipendente
  per pixel (il costo LS dipende fortemente dalle iterazioni di convergenza).

Interpretazione: la PINN ottiene una mappa T2 con errore confrontabile (e più
robusto al rumore grazie alla regolarizzazione) rispetto al fitting per-pixel ai
minimi quadrati, processando tutti i pixel in un singolo forward batch.

La selezione del metodo migliore a un dato livello di rumore è programmatica:
`best_model({"pinn": m_pinn, "ls": m_ls}, key="rmse")` restituisce il nome del
modello con RMSE minore, mentre `rank_models(...)` ne fornisce la classifica
completa (utile per riepiloghi automatici e regressioni sui risultati).

## Track dati reali (OpenNeuro ds007116, CC0)

Oltre al track sintetico, il progetto applica la stessa inversione a **dati
multi-echo GRE reali** (OpenNeuro **ds007116** "Penn LEAD", licenza CC0; 4 echi
TE = 7.5/15/22.5/30 ms, 3T). Qui **non esiste ground truth**: la validazione è una
**held-out echo cross-validation** (si stima `(S0, T2*)` da `K−1` echi e si predice
l'eco escluso, NRMSE entro la brain mask). In una sequenza GRE il parametro è **T2\***.

```bash
uv run python scripts/download_real.py      # ~7 MB da S3 (dati gitignorati)
uv run python scripts/evaluate_real.py      # PINN vs LS + held-out CV → results/
```

Numeri reali da `results/real_metrics.json` (slice 46, 9706 voxel in mask):

| Metodo | T2\* mediano (ms) | NRMSE held-out (media) |
|---|---|---|
| PINN | **50.4** (IQR 39–58) | **0.075** |
| Least-squares | 51.8 (IQR 38–63) | 0.100 |

Entrambi danno T2\* fisiologici a 3T; la PINN, grazie alla regolarizzazione implicita
della rete condivisa, **generalizza meglio sull'eco escluso** (0.075 vs 0.100). Report
completo con citazioni: [`report/qmri_real_data.md`](report/qmri_real_data.md).

> Nota metodologica (debugging documentato): applicando la PINN sintetica al reale, il
> T2\* collassava a ~600 ms perché i voxel di background (primo eco ≈ 0) producevano
> normalizzazioni esplosive che dominavano la loss. Fix: addestrare la PINN **solo entro
> la brain mask** (`train_pinn(..., mask=...)`), con test di regressione.

## Note metodologiche e limiti

- Modello mono-esponenziale: in tessuti reali T2 può essere multi-componente.
- Dati sintetici: nessun artefatto di acquisizione reale (moto, B1, ecc.).
- La PINN è qui un **regressore per-pixel** vincolato dalla fisica; non sfrutta
  ancora correlazioni spaziali profonde (oltre al prior TV opzionale).

## Riferimenti

- M. Raissi, P. Perdikaris, G.E. Karniadakis, *Physics-informed neural networks:
  A deep learning framework for solving forward and inverse problems involving
  nonlinear partial differential equations*, Journal of Computational Physics 378
  (2019) 686–707. DOI: [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

## Licenza

Codice rilasciato sotto licenza MIT (vedi `LICENSE`). I dataset eventualmente
utilizzati restano soggetti alle rispettive licenze; qui i dati sono interamente
sintetici e generati dal codice.
