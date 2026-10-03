from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "cdp_generator/web/app.py")


def _start(workflow: str) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.radio(key="workflow").set_value(workflow).run()
    assert not app.exception
    return app


def _click(app: AppTest, label: str) -> None:
    next(button for button in app.button if button.label == label).click().run()
    assert not app.exception


def test_ui_fib_composition_and_cdpm2_plots_with_lchar():
    app = _start("Authority-aware material / CDPM2")
    app.selectbox(key="authority_profile").set_value("ec2_2004").run()
    app.selectbox(key="authority_class_ec2_2004").set_value("C30/37")
    app.checkbox(key="authority_fib_gf_if_missing").set_value(True)
    _click(app, "Build material / Assess CDPM2")
    result = app.session_state["authority_result"]
    assert result.readiness["state"] == "READY"
    assert result.fracture_energy_composition is not None
    assert len(result.curves) == 1
    assert len(app.get("plotly_chart")) == 1
    assert any("Secondary fib MC2010" in info.value for info in app.info)

    app.number_input(key="backend_lchar").set_value(12.5)
    _click(app, "Build backend payload")
    result = app.session_state["authority_result"]
    assert result.backend is not None
    assert len(result.curves) == 2
    assert len(app.get("plotly_chart")) == 2


def test_ui_high_age_fib_composition_removes_only_gf_blocker_then_e_override_ready():
    app = _start("Authority-aware material / CDPM2")
    app.selectbox(key="authority_profile").set_value("ec2_2023").run()
    app.selectbox(key="authority_class_ec2_2023").set_value("C30/37")
    app.number_input(key="authority_ec2_2023_reference_age_days").set_value(56.0)
    app.checkbox(key="authority_fib_gf_if_missing").set_value(True)
    _click(app, "Build material / Assess CDPM2")
    result = app.session_state["authority_result"]
    assert result.readiness["state"] == "UNRESOLVED_PHYSICAL_INPUT"
    assert result.readiness["blockers"] == ["E_initial unresolved; E_secant fallback forbidden"]

    app.checkbox(key="override_enable_E").set_value(True)
    app.number_input(key="override_value_E").set_value(41000.0)
    _click(app, "Build material / Assess CDPM2")
    assert app.session_state["authority_result"].readiness["state"] == "READY"


def test_ui_legacy_abaqus_default_powerlaw_and_error_preserves_prior_result():
    app = _start("Legacy curves")
    _click(app, "Calculate")
    assert "Abaqus CDP" in [tab.label for tab in app.tabs]
    _click(app, "Build Abaqus material")
    result = app.session_state["abaqus_legacy_result"]
    assert result.validation["passed"]
    assert result.inputs["tension_law"] == "bilinear"
    assert len(result.curves) == 4
    labels = [button.label for button in app.get("download_button")]
    assert "Download Abaqus provenance JSON" in labels
    assert "Download Abaqus .inp material card" in labels

    app.selectbox(key="abaqus_tension_law").set_value("Power law")
    _click(app, "Build Abaqus material")
    power = app.session_state["abaqus_legacy_result"]
    assert power.inputs["tension_law"] == "power_law"
    assert power.tension_stiffening != result.tension_stiffening

    app.number_input(key="abaqus_ref_length").set_value(0.0)
    _click(app, "Build Abaqus material")
    assert app.error
    assert app.session_state["abaqus_legacy_result"].to_json() == power.to_json()
