"""Smoke test of the Streamlit app: the first render runs without exceptions.

The app does not train anything until the button is pressed, so the first render
is light: we only check that it raises no exception and shows the title.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "main.py"


def test_app_initial_render():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("PINN qMRI" in t.value for t in at.title)
