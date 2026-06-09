# T2\* mapping su dati multi-echo GRE reali: PINN vs least-squares, validato senza ground truth

Mini-report del *track dati reali* di `pinn-qmri`. Estende la dimostrazione sintetica
(con ground truth) a immagini cliniche reali, dove la ground truth **non esiste**, ponendo
la domanda centrale della ricerca applicata: *come si valida un'inversione qMRI sul dato reale?*

## Dati

OpenNeuro **ds007116** ("Penn LEAD"), licenza **CC0** [4]. Sequenza **MEGRE** (multi-echo
gradient echo) anatomica, soggetto `sub-19861/ses-1`, magnitude a 4 echi
(**TE = 7.5 / 15 / 22.5 / 30 ms**, 3T). Si usa una slice assiale rappresentativa con brain mask
via soglia di Otsu. I dati grezzi non sono ridistribuiti: si scaricano da S3 con
`scripts/download_real.py`. Il parametro stimato in una sequenza GRE è **T2\*** (non T2), con i
relativi confondenti [1, 2].

## Metodo

Stesso forward model mono-esponenziale del track sintetico, `S(TE) = S0 · exp(-TE/T2*)`:

- **PINN** (non supervisionata): un MLP per-voxel mappa il vettore dei segnali → `(S0, T2*)`,
  addestrato minimizzando la sola consistenza fisica col forward model [3].
- **Least-squares** per-voxel (`scipy.optimize.curve_fit`): riferimento classico.

**Validazione senza ground truth — held-out echo cross-validation.** Per ogni eco `j` si stima
`(S0, T2*)` dagli altri `K−1` echi e si predice l'eco escluso; il residuo (NRMSE entro la mask)
misura la capacità predittiva del modello fisico stimato, senza alcuna verità nota.

## Risultati

Numeri reali da `results/real_metrics.json` (slice 46, 9706 voxel in mask).

**Stima di T2\*** (mediana, IQR in ms):

| Metodo | T2\* mediano | IQR |
|---|---|---|
| PINN | **50.4** | 39.0 – 58.5 |
| Least-squares | 51.8 | 38.4 – 63.2 |

Entrambi i valori sono **fisiologicamente plausibili** per la sostanza grigia/bianca a 3T; la
PINN mostra un IQR leggermente più stretto, coerente con la regolarizzazione implicita della
rete condivisa tra voxel.

**Held-out echo cross-validation** (NRMSE dell'eco escluso, più basso è meglio):

| Fold (eco escluso) | LS | PINN |
|---|---|---|
| eco 1 (TE 7.5 ms) | 0.204 | **0.106** |
| eco 2 (TE 15 ms) | 0.057 | 0.058 |
| eco 3 (TE 22.5 ms) | 0.051 | 0.053 |
| eco 4 (TE 30 ms) | 0.089 | **0.085** |
| **media** | 0.100 | **0.075** |

La PINN **generalizza meglio** all'eco escluso (NRMSE medio 0.075 vs 0.100), soprattutto sul
primo eco (ad alto segnale): il fitting per-voxel ottimizza ogni pixel in isolamento ed
estrapola peggio, mentre la rete condivisa apprende una rappresentazione più robusta.

## Nota metodologica: un bug reale, diagnosticato

Applicando direttamente la PINN tarata sul sintetico, il T2\* stimato collassava a un valore
costante ~600 ms (NRMSE held-out ≈ 1.0). **Causa identificata** (debugging sistematico): i voxel
di **background** (≈49% della slice) hanno primo eco ≈ 0; la normalizzazione per il primo eco
produceva rapporti esplosivi (~10⁵) che dominavano la loss e collassavano la rete condivisa a un
T2\* costante. Il track sintetico non lo mostrava perché privo di background. **Fix**: addestrare
la PINN **solo sui voxel in mask** (`train_pinn(..., mask=...)`), con test di regressione dedicato.
Dopo il fix, T2\* mediano 50.4 ms e NRMSE held-out competitivo.

## Limiti

- **Singolo soggetto, singola slice**: dimostrazione, non uno studio.
- **T2\* (non T2)**: confondenti di disomogeneità di campo B0 non corretti (no informazione di
  fase, solo magnitude) [1, 2].
- **Modello mono-esponenziale**: il decadimento reale può essere multi-componente.
- **Solo 4 echi a TE corti (7.5–30 ms)**: held-out leave-one-out minimale; l'inversione è
  intrinsecamente mal-condizionata in questo regime.
- **Non è uno strumento clinico**: dimostrazione di ricerca su dati pubblici CC0.

## Riferimenti

1. Yablonskiy DA, Sukstanskii AL, Luo J, Wang X. *Voxel spread function method for correction of
   magnetic field inhomogeneity effects in quantitative gradient-echo-based MRI.* Magn Reson Med.
   2013. DOI: [10.1002/mrm.24585](https://doi.org/10.1002/mrm.24585)
2. *T2\* quantification using multi-echo gradient echo sequences: a comparative study of different
   readout gradients.* Sci Rep. 2023. DOI:
   [10.1038/s41598-023-28265-0](https://doi.org/10.1038/s41598-023-28265-0)
3. Raissi M, Perdikaris P, Karniadakis GE. *Physics-informed neural networks.* J Comput Phys.
   2019;378:686–707. DOI:
   [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)
4. OpenNeuro dataset **ds007116** "Penn LEAD", licenza CC0.
   [https://openneuro.org/datasets/ds007116](https://openneuro.org/datasets/ds007116)
