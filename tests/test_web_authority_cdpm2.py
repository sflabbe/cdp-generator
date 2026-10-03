from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "cdp_generator/web/app.py")


def modern(profile="fib_mc2010", cls="C30"):
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.radio(key="workflow").set_value("Authority-aware material / CDPM2").run()
    app.selectbox(key="authority_profile").set_value(profile).run()
    app.selectbox(key=f"authority_class_{profile}").set_value(cls)
    return app


def submit(app):
    next(b for b in app.button if b.label == "Build material / Assess CDPM2").click().run()
    assert not app.exception
    return app.session_state["authority_result"]


def override(app, name, value):
    app.checkbox(key=f"override_enable_{name}").set_value(True)
    app.number_input(key=f"override_value_{name}").set_value(value)


def test_fib_semantics_backend_exports_and_result_state():
    app = modern()
    result = submit(app)
    assert result.readiness["state"] == "READY"
    assert result.backend is None
    assert len(result.semantic_parameters["values"]) == 20
    assert [t.label for t in app.tabs] == [
        "Physical properties",
        "Provenance",
        "CDPM2",
        "Backend",
        "Raw / Export",
    ]
    assert any(len(table.value) == 20 for table in app.dataframe)
    app.number_input(key="backend_lchar").set_value(12.5)
    next(b for b in app.button if b.label == "Build backend payload").click().run()
    assert not app.exception
    result = app.session_state["authority_result"]
    assert result.backend["characteristic_length"] == 12.5
    assert any(len(table.value) == 24 for table in app.dataframe)
    assert len(app.get("download_button")) == 4
    app.radio(key="workflow").set_value("Legacy curves").run()
    next(b for b in app.button if b.label == "Calculate").click().run()
    assert not app.exception
    assert "last_result" in app.session_state
    app.radio(key="workflow").set_value("Authority-aware material / CDPM2").run()
    assert app.session_state["authority_result"].to_dict() == result.to_dict()
    app.number_input(key="backend_lchar").set_value(0.0)
    next(b for b in app.button if b.label == "Build backend payload").click().run()
    assert not app.exception
    assert app.error
    assert app.session_state["authority_result"].to_dict() == result.to_dict()


def test_ec2_2004_blocked_then_explicit_gft():
    app = modern("ec2_2004", "C30/37")
    result = submit(app)
    assert result.readiness["state"] == "COMPOSITION_REQUIRED"
    assert [warning.value for warning in app.warning] == ["G_Ft composition required"]
    assert result.semantic_parameters is None
    assert not any(len(table.value) == 20 for table in app.dataframe)
    assert len(app.get("download_button")) == 2
    override(app, "G_Ft", 0.15)
    result = submit(app)
    assert result.readiness["state"] == "READY"


def test_ec2_2023_high_age_then_explicit_e_and_gft():
    app = modern("ec2_2023", "C30/37")
    app.number_input(key="authority_ec2_2023_reference_age_days").set_value(56.0)
    result = submit(app)
    assert result.readiness["state"] == "UNRESOLVED_PHYSICAL_INPUT"
    assert len(app.warning) == 2
    assert "E_initial unresolved; E_secant fallback forbidden" in [w.value for w in app.warning]
    assert "G_Ft composition required" in [w.value for w in app.warning]
    override(app, "E", 41000.0)
    override(app, "G_Ft", 0.15)
    result = submit(app)
    assert result.readiness["state"] == "READY"
    # Changed form controls do not rewrite the result or request until submitted.
    app.number_input(key="authority_ec2_2023_reference_age_days").set_value(28.0).run()
    assert app.session_state["authority_result"].to_dict() == result.to_dict()


def test_bad_advanced_override_preserves_last_success():
    app = modern()
    result = submit(app)
    app.text_area(key="advanced_overrides").set_value('{"unknown":1}')
    next(b for b in app.button if b.label == "Build material / Assess CDPM2").click().run()
    assert not app.exception
    assert app.error
    assert app.session_state["authority_result"].to_dict() == result.to_dict()


def test_enabled_blank_override_is_controlled():
    app = modern()
    app.checkbox(key="override_enable_G_Ft").set_value(True)
    next(b for b in app.button if b.label == "Build material / Assess CDPM2").click().run()
    assert not app.exception
    assert app.error
    assert "authority_result" not in app.session_state
