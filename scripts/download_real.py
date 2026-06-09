"""Scarica un soggetto multi-echo GRE reale da OpenNeuro (ds007116, CC0).

Dataset: OpenNeuro ds007116 "Penn LEAD" (licenza CC0). Sequenza MEGRE anatomica,
4 echi (TE = 7.5/15/22.5/30 ms), 3T. Si scaricano SOLO i 4 file magnitude (+ i
sidecar JSON) del soggetto di riferimento, dal bucket S3 pubblico. Idempotente:
i file gia' presenti non vengono ri-scaricati. I dati grezzi NON sono committati
(cartella data/ gitignorata).

Esegui:  uv run python scripts/download_real.py
"""

from __future__ import annotations

from pathlib import Path

import requests

ACCESSION = "ds007116"
SUBJECT = "sub-19861"
SESSION = "ses-1"
BASE = f"https://s3.amazonaws.com/openneuro.org/{ACCESSION}/{SUBJECT}/{SESSION}/anat"
DEST = Path(__file__).resolve().parents[1] / "data" / "real"


def _files() -> list[str]:
    names: list[str] = []
    for echo in range(1, 5):
        stem = f"{SUBJECT}_{SESSION}_echo-{echo}_part-mag_MEGRE"
        names.append(f"{stem}.nii.gz")
        names.append(f"{stem}.json")
    return names


def _download(url: str, dest: Path) -> None:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: OpenNeuro {ACCESSION} (Penn LEAD) — licenza CC0")
    print(f"Soggetto: {SUBJECT}/{SESSION}, sequenza MEGRE (solo magnitude)\n")
    for name in _files():
        dest = DEST / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [skip] {name}")
            continue
        url = f"{BASE}/{name}"
        print(f"  [get ] {name}")
        _download(url, dest)
    print(f"\nScaricati in: {DEST}")
    print("Citazione: dataset OpenNeuro ds007116 (Penn LEAD), CC0. "
          "Verificare l'attribuzione richiesta sulla pagina del dataset.")


if __name__ == "__main__":
    main()
