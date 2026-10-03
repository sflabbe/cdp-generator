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
    f_cm_input = next(widget for widget in app.number_input if widget.label == "f_cm [MPa]")
    assert f_cm_input.value == 28.0
    app.selectbox[0].select(mode)
    next(button for button in app.button if button.label == "Calculate").click().run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Curves", "Properties", "Raw data", "Abaqus CDP", "Export"
    ]
    result = app.session_state["last_result"]
    assert result.mode == ("temperature" if mode == "Temperature" else "strain_rate")
    next(
        widget for widget in app.number_input if widget.label == "f_cm [MPa]"
    ).set_value(35.0).run()
    assert app.session_state["last_result"].inputs == result.inputs
    next(
        widget for widget in app.number_input if widget.label == "f_cm [MPa]"
    ).set_value(-1.0)
    next(button for button in app.button if button.label == "Calculate").click().run()
    assert not app.exception
    assert app.error
    assert app.session_state["last_result"].inputs == result.inputs


def test_app_handles_bad_rate_text():
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "cdp_generator/web/app.py"), default_timeout=30
    ).run()
    app.text_input[0].set_value("oops")
    next(button for button in app.button if button.label == "Calculate").click().run()
    assert not app.exception
    assert app.error
