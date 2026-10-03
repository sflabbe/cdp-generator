"""WEB-M5 / ABAQUS-Q1 Streamlit AppTests for dependent Abaqus exports."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "cdp_generator/web/app.py")


def _legacy(mode: str = "Strain rate", rates: str | None = None) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.radio(key="workflow").set_value("Legacy curves").run()
    next(s for s in app.selectbox if s.label == "Analysis mode").set_value(mode)
    if rates is not None:
        next(t for t in app.text_input if t.label.startswith("Strain rates")).set_value(rates)
    _click(app, "Calculate")
    return app


def _click(app: AppTest, label: str) -> None:
    next(button for button in app.button if button.label == label).click().run()
    assert not app.exception


def _dependent(app: AppTest, mode_label: str, policy: str = "Omit damage") -> None:
    app.selectbox(key="abaqus_export_mode").set_value(mode_label).run()
    assert not app.exception
    app.selectbox(key="abaqus_dep_damage_policy").set_value(policy)
    _click(app, "Build dependent Abaqus material")


def _downloads(app: AppTest) -> list[str]:
    return [button.label for button in app.get("download_button")]


def _dataframe_columns(app: AppTest) -> list[list[str]]:
    return [list(df.value.columns) for df in app.dataframe]


def test_a5_r1_rate_dependent_export_omit():
    app = _legacy()
    modes = app.selectbox(key="abaqus_export_mode").options
    assert modes == ["Static reference", "Strain-rate dependent"]
    _dependent(app, "Strain-rate dependent")
    result = app.session_state["abaqus_dependent_result"]
    assert result.mode == "strain_rate"
    assert result.inputs["strain_rates"] == [0.0, 2.0, 30.0, 100.0]
    assert result.damage_policy["policy"] == "omit"
    assert any(
        "Abaqus tension crack-opening rate [mm/s]" in cols for cols in _dataframe_columns(app)
    )
    assert len(app.get("plotly_chart")) >= 2
    labels = _downloads(app)
    assert "Download dependent Abaqus JSON" in labels
    assert "Download dependent Abaqus .inp material card" in labels
    assert not any("Run Abaqus" in b.label for b in app.button)
    assert any("scripts/qualify_abaqus.py" in i.value for i in app.info)


def test_a5_r2_reference_damage_default_rates_rejected_prior_retained():
    app = _legacy()
    _dependent(app, "Strain-rate dependent")
    prior = app.session_state["abaqus_dependent_result"].to_json()
    app.selectbox(key="abaqus_dep_damage_policy").set_value(
        "Reuse reference damage if Abaqus-valid"
    )
    _click(app, "Build dependent Abaqus material")
    assert app.error
    assert "value=30.0" in app.error[0].value
    assert "compression_plastic_strain_monotonic" in app.error[0].value
    assert app.session_state["abaqus_dependent_result"].to_json() == prior


def test_a5_r2_reference_damage_compatible_rates_pass():
    app = _legacy(rates="0, 2")
    _dependent(app, "Strain-rate dependent", "Reuse reference damage if Abaqus-valid")
    result = app.session_state["abaqus_dependent_result"]
    assert result.damage_policy["policy"] == "reference_damage"
    assert result.compression_damage is not None
    groups = [c.group for c in result.curves]
    assert groups.count("abaqus_dependent_reference_compression_damage") == 1
    assert groups.count("abaqus_dependent_reference_tension_damage") == 1


def test_a5_t1_temperature_export_omit():
    app = _legacy("Temperature")
    assert app.selectbox(key="abaqus_export_mode").options == [
        "Static reference",
        "Temperature dependent",
    ]
    _dependent(app, "Temperature dependent")
    result = app.session_state["abaqus_dependent_result"]
    assert result.mode == "temperature" and result.elastic["temperature_dependent"]
    assert any(
        cols == ["temperature [°C]", "E(T) [MPa]", "nu [-]"] for cols in _dataframe_columns(app)
    )
    assert any("constant-nu" in c.value for c in app.caption)
    assert len(app.get("plotly_chart")) >= 3  # compression, tension, E(T)
    assert "Download dependent Abaqus .inp material card" in _downloads(app)


def test_a5_t2_temperature_reference_damage_rejected_and_invalid_input_keeps_result():
    app = _legacy("Temperature")
    _dependent(app, "Temperature dependent")
    prior = app.session_state["abaqus_dependent_result"].to_json()
    app.selectbox(key="abaqus_dep_damage_policy").set_value(
        "Reuse reference damage if Abaqus-valid"
    )
    _click(app, "Build dependent Abaqus material")
    assert app.error and "branch=compression" in app.error[0].value
    assert app.session_state["abaqus_dependent_result"].to_json() == prior

    app.selectbox(key="abaqus_dep_damage_policy").set_value("Omit damage")
    app.number_input(key="abaqus_dep_ref_length").set_value(0.0)
    _click(app, "Build dependent Abaqus material")
    assert app.error
    assert app.session_state["abaqus_dependent_result"].to_json() == prior


def test_static_mode_still_m4_and_unsorted_rates_controlled():
    app = _legacy(rates="100, 0")
    _click(app, "Build Abaqus material")
    assert app.session_state["abaqus_legacy_result"].validation["passed"]
    _dependent(app, "Strain-rate dependent")
    assert app.error and "strictly increasing" in app.error[0].value
    assert "abaqus_dependent_result" not in app.session_state
