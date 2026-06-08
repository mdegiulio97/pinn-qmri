"""Smoke test dell'app Streamlit: il primo render gira senza eccezioni.

L'app non addestra nulla finche' non si preme il pulsante, quindi il primo render
e' leggero: verifichiamo solo che non sollevi eccezioni e mostri il titolo.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "main.py"


def test_app_initial_render():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("PINN qMRI" in t.value for t in at.title)
