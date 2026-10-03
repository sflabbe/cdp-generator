from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("mode", ["Strain rate", "Temperature"])
def test_app_calculates_and_preserves_state(mode):
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "cdp_generator/web/app.py"), default_timeout=30
    ).run()
    assert not app.exception
    assert len(app.number_input) == 4
    assert app.number_input[0].value == 28.0
    app.selectbox[0].select(mode)
    app.button[0].click().run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Curves", "Properties", "Raw data", "Export"]
    result = app.session_state["last_result"]
    assert result.mode == ("temperature" if mode == "Temperature" else "strain_rate")
    app.number_input[0].set_value(35.0).run()
    assert app.session_state["last_result"].inputs == result.inputs
    app.number_input[0].set_value(-1.0)
    app.button[0].click().run()
    assert not app.exception
    assert app.error
    assert app.session_state["last_result"].inputs == result.inputs


def test_app_handles_bad_rate_text():
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "cdp_generator/web/app.py"), default_timeout=30
    ).run()
    app.text_input[0].set_value("oops")
    app.button[0].click().run()
    assert not app.exception
    assert app.error
