"""Download one real multi-echo GRE subject from OpenNeuro (ds007116, CC0).

Dataset: OpenNeuro ds007116 "Penn LEAD" (CC0 license). Anatomical MEGRE sequence,
4 echoes (TE = 7.5/15/22.5/30 ms), 3T. ONLY the 4 magnitude files (+ the JSON
sidecars) of the reference subject are downloaded, from the public S3 bucket.
Idempotent: files already present are not downloaded again. The raw data are NOT
committed (the data/ folder is gitignored).

Run:  uv run python scripts/download_real.py
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
    print(f"Dataset: OpenNeuro {ACCESSION} (Penn LEAD) — CC0 license")
    print(f"Subject: {SUBJECT}/{SESSION}, MEGRE sequence (magnitude only)\n")
    for name in _files():
        dest = DEST / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [skip] {name}")
            continue
        url = f"{BASE}/{name}"
        print(f"  [get ] {name}")
        _download(url, dest)
    print(f"\nDownloaded to: {DEST}")
    print("Citation: OpenNeuro dataset ds007116 (Penn LEAD), CC0. "
          "Check the attribution requested on the dataset page.")


if __name__ == "__main__":
    main()
