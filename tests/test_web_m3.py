from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "cdp_generator/web/app.py")


def button(app, label):
    next(b for b in app.button if b.label == label).click().run()
    assert not app.exception


def start(workflow):
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.radio(key="workflow").set_value(workflow).run()
    assert not app.exception
    return app


def small_steel(app):
    app.text_input(key="steel_rates").set_value("0.001, 1")
    app.text_input(key="steel_temperatures").set_value("20, 400")
    button(app, "Calculate steel")
    return app.session_state["steel_result"]


def add(app, family, label):
    app.text_input(key=f"compare_label_{family}").set_value(label)
    button(app, "Add current result to Compare")
    assert not app.error


def test_steel_preset_exports_overrides_and_invalid_inputs():
    app = start("Steel Johnson-Cook")
    result = small_steel(app)
    assert result.material["grade"] == "B500C"
    assert result.data_status == "approximate_preset"
    assert any("approximate" in w.value for w in app.warning)
    assert any("Experimental template" in w.value for w in app.warning)
    assert [t.label for t in app.tabs] == [
        "Curves",
        "Material",
        "Johnson-Cook",
        "Raw data",
        "Export",
    ]
    assert len(app.get("plotly_chart")) == 2
    assert len(app.get("download_button")) == 3
    app.checkbox(key="steel_enable_fy").set_value(True)
    app.number_input(key="steel_override_fy").set_value(550)
    button(app, "Calculate steel")
    result = app.session_state["steel_result"]
    assert result.material["fy"] == 550
    app.text_input(key="steel_rates").set_value("bad")
    button(app, "Calculate steel")
    assert app.error
    assert app.session_state["steel_result"].to_json() == result.to_json()
    app.text_input(key="steel_rates").set_value(".001")
    app.checkbox(key="steel_enable_E").set_value(True)
    button(app, "Calculate steel")
    assert app.error
    assert app.session_state["steel_result"].to_json() == result.to_json()


def test_custom_domain_error_and_state_across_workflows():
    app = start("Steel Johnson-Cook")
    preset = small_steel(app)
    app.selectbox(key="steel_source").set_value("Custom material").run()
    custom = small_steel(app)
    assert custom.data_status == "user_provided"
    assert custom.material["grade"] == "MySteel"
    app.number_input(key="steel_custom_T_melt").set_value(10)
    button(app, "Calculate steel")
    assert app.error
    assert app.session_state["steel_result"].to_json() == custom.to_json()
    app.radio(key="workflow").set_value("Legacy curves").run()
    button(app, "Calculate")
    legacy = app.session_state["last_result"]
    app.radio(key="workflow").set_value("Authority-aware material / CDPM2").run()
    button(app, "Build material / Assess CDPM2")
    authority = app.session_state["authority_result"]
    app.radio(key="workflow").set_value("Steel Johnson-Cook").run()
    assert not app.exception
    assert app.session_state["steel_result"].to_json() == custom.to_json()
    assert app.session_state["last_result"].to_json() == legacy.to_json()
    assert app.session_state["authority_result"].to_json() == authority.to_json()
    assert preset.data_status == "approximate_preset"


def test_steel_active_calibration_engineering_and_custom_ratio():
    app = start("Steel Johnson-Cook")
    app.checkbox(key="steel_neutral").set_value(False)
    app.number_input(key="steel_C").set_value(0.02)
    app.number_input(key="steel_m").set_value(1.0)
    app.checkbox(key="steel_explicit_n").set_value(True)
    app.number_input(key="steel_n").set_value(0.25)
    app.selectbox(key="steel_output").set_value("engineering")
    result = small_steel(app)
    assert result.johnson_cook_parameters["C"] == 0.02
    assert result.johnson_cook_parameters["m"] == 1
    assert result.johnson_cook_parameters["n"] == 0.25
    assert result.curves[0].x_quantity == "Engineering strain"
    app.selectbox(key="steel_source").set_value("Custom material").run()
    app.radio(key="steel_strength").set_value("fu/fy ratio").run()
    app.number_input(key="steel_custom_fu_fy_ratio").set_value(1.3)
    result = small_steel(app)
    assert result.material["fu"] == 650


def test_legacy_comparison_management():
    app = start("Legacy curves")
    button(app, "Calculate")
    add(app, "legacy_concrete", "C28")
    app.number_input[0].set_value(35)
    button(app, "Calculate")
    add(app, "legacy_concrete", "C35")
    app.radio(key="workflow").set_value("Compare").run()
    assert not app.exception
    assert len(app.session_state["comparison_cases"]) == 2
    assert len(app.get("plotly_chart")) == 1
    current = app.selectbox(key="compare_manage_case").value
    app.text_input(key=f"compare_rename_{current.case_id}").set_value("renamed")
    button(app, "Rename case")
    assert "renamed" in [c.label for c in app.session_state["comparison_cases"]]
    button(app, "Remove case")
    assert len(app.session_state["comparison_cases"]) == 1
    button(app, "Clear family")
    assert app.session_state["comparison_cases"] == []
    assert any("at least two" in i.value for i in app.info)


def test_authority_ready_blocked_comparison():
    app = start("Authority-aware material / CDPM2")
    button(app, "Build material / Assess CDPM2")
    add(app, "authority_concrete", "fib")
    app.selectbox(key="authority_profile").set_value("ec2_2004").run()
    button(app, "Build material / Assess CDPM2")
    add(app, "authority_concrete", "EC2 blocked")
    app.radio(key="workflow").set_value("Compare").run()
    app.selectbox(key="compare_family").set_value("Authority-aware concrete / CDPM2").run()
    assert not app.exception
    assert len(app.dataframe) == 2
    readiness = app.dataframe[1].value
    assert set(readiness["State"]) == {"READY", "COMPOSITION_REQUIRED"}
    assert len(app.get("plotly_chart")) == 0


def test_steel_comparison_family_clear_and_incompatible_outputs():
    app = start("Legacy curves")
    button(app, "Calculate")
    add(app, "legacy_concrete", "legacy")
    app.radio(key="workflow").set_value("Steel Johnson-Cook").run()
    small_steel(app)
    add(app, "steel_johnson_cook", "steel A")
    app.checkbox(key="steel_enable_fy").set_value(True)
    app.number_input(key="steel_override_fy").set_value(550)
    button(app, "Calculate steel")
    add(app, "steel_johnson_cook", "steel B")
    app.radio(key="workflow").set_value("Compare").run()
    app.selectbox(key="compare_family").set_value("Steel Johnson-Cook").run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 1
    assert len(app.dataframe[0].value) == 26
    button(app, "Clear family")
    assert len(app.session_state["comparison_cases"]) == 1
    assert app.session_state["comparison_cases"][0].family == "legacy_concrete"
