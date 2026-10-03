"""Explicit steel analysis using the existing scientific application service."""

from dataclasses import asdict

import streamlit as st

from ..application import (
    SteelAnalysisRequest,
    SteelInputError,
    available_steel_grades,
    available_steel_standards,
    parse_float_series,
    run_steel_analysis,
    steel_result_abaqus_text,
    steel_result_excel_bytes,
)
from ..application.steel_services import ABAQUS_DISCLAIMER, PRESET_DISCLAIMER
from ..visualization.steel_plotly import build_steel_figures
from .shared import add_to_compare


def render_steel() -> None:
    st.title("Steel Johnson-Cook")
    source = st.sidebar.selectbox(
        "Material source", ["Built-in approximate preset", "Custom material"], key="steel_source"
    )
    custom = source == "Custom material"
    if custom:
        st.caption("User-provided material input; not independently verified.")
        standard = "Custom"
        grade = ""
    else:
        st.warning(PRESET_DISCLAIMER)
        st.caption("Verify standard, jurisdiction, manufacturer and project data.")
        standard = st.sidebar.selectbox(
            "Standard", available_steel_standards(), key="steel_standard"
        )
        grades = available_steel_grades(standard)
        grade = st.sidebar.selectbox(
            "Grade",
            grades,
            index=grades.index("B500C") if "B500C" in grades else 0,
            key=f"steel_grade_{standard}",
        )
    strength = (
        st.sidebar.radio("Strength input", ["fu", "fu/fy ratio"], key="steel_strength")
        if custom
        else "fu"
    )
    with st.sidebar.form("steel_analysis"):
        overrides: dict[str, float | None] = {}
        if custom:
            grade = st.text_input("Material name", "MySteel", key="steel_name")
            for name, unit, default in [
                ("fy", "MPa", 500.0),
                ("Agt", "%", 10.0),
                ("E", "MPa", 200000.0),
                ("nu", "-", 0.3),
                ("T_room", "°C", 20.0),
                ("T_melt", "°C", 1500.0),
            ]:
                overrides[name] = st.number_input(
                    f"{name} [{unit}]", value=default, key=f"steel_custom_{name}"
                )
            name = "fu" if strength == "fu" else "fu_fy_ratio"
            overrides[name] = st.number_input(
                "fu [MPa]" if name == "fu" else "fu/fy [-]",
                value=620.0 if name == "fu" else 1.24,
                key=f"steel_custom_{name}",
            )
        else:
            with st.expander("Explicit preset overrides"):
                for name, unit in [
                    ("fy", "MPa"),
                    ("fu", "MPa"),
                    ("Agt", "%"),
                    ("E", "MPa"),
                    ("nu", "-"),
                ]:
                    enabled = st.checkbox(f"Override {name}", key=f"steel_enable_{name}")
                    value = st.number_input(
                        f"{name} override [{unit}]", value=None, key=f"steel_override_{name}"
                    )
                    if enabled:
                        overrides[name] = value
        with st.expander("Calibration"):
            neutral = st.checkbox("Rate/temperature neutral", value=True, key="steel_neutral")
            st.caption(
                "Neutral uses C=0 and m=0. The C/m inputs below apply only when neutral is disabled."
            )
            C = st.number_input("C [-]", value=0.0, key="steel_C")
            m = st.number_input("m [-]", value=0.0, key="steel_m")
            explicit_n = st.checkbox("Explicit hardening exponent n", key="steel_explicit_n")
            n = st.number_input("n [-]", value=None, key="steel_n")
            st.caption(
                "Automatic n is a heuristic of the current model, not verified experimental calibration."
            )
            reference = st.number_input(
                "Reference strain rate epsdot0 [1/s]",
                value=1e-3,
                format="%.6f",
                key="steel_epsdot0",
            )
        rates = st.text_input(
            "Strain rates [1/s]", "1e-4, 1e-3, 1e-2, 1, 10, 100", key="steel_rates"
        )
        temperatures = st.text_input(
            "Temperatures [°C]", "20, 200, 400, 600, 800", key="steel_temperatures"
        )
        eps_max = st.number_input("Maximum strain [-]", value=0.20, key="steel_eps_max")
        points = st.number_input("Number of points", value=100, step=1, key="steel_points")
        kind = st.selectbox("Output", ["true", "engineering"], key="steel_output")
        submitted = st.form_submit_button("Calculate steel")
    if submitted:
        try:
            if any(v is None for v in overrides.values()) or (explicit_n and n is None):
                raise SteelInputError("Enter a value for every enabled override/explicit n.")
            request = SteelAnalysisRequest(
                source="custom" if custom else "preset",
                standard=standard,
                grade=grade,
                material_overrides={
                    name: value for name, value in overrides.items() if value is not None
                },
                assume_rate_temp_neutral=neutral,
                n_default=n if explicit_n else None,
                C_default=C,
                m_default=m,
                epsdot0=reference,
                strain_rates=parse_float_series(rates),
                temperatures=parse_float_series(temperatures),
                eps_max=eps_max,
                n_points=points,
                output_kind=kind,
            )
            result = run_steel_analysis(request)
        except SteelInputError as exc:
            st.error(str(exc))
        else:
            st.session_state["steel_result"] = result
    if "steel_result" not in st.session_state:
        st.info("Enter material and analysis inputs, then Calculate steel.")
        return
    result = st.session_state["steel_result"]
    st.write(
        "Displayed result",
        {
            "standard": result.material["standard"],
            "grade": result.material["grade"],
            "data_status": result.data_status,
            "analysis": result.analysis,
            "overrides": result.metadata["material_overrides"],
        },
    )
    st.warning(result.disclaimer)
    add_to_compare(
        result, "steel_johnson_cook", f"{result.material['standard']} {result.material['grade']}"
    )
    curves, material, jc, raw, exports = st.tabs(
        ["Curves", "Material", "Johnson-Cook", "Raw data", "Export"]
    )
    with curves:
        for group, figure in build_steel_figures(result).items():
            st.plotly_chart(figure, width="stretch", key=f"steel_plot_{group}")
    with material:
        st.json(result.material)
    with jc:
        units = {
            "A": "MPa",
            "B": "MPa",
            "n": "-",
            "C": "-",
            "m": "-",
            "epsdot0": "1/s",
            "T_room": "°C",
            "T_melt": "°C",
        }
        st.dataframe(
            [
                {"Parameter": name, "Value": value, "Unit": units[name]}
                for name, value in result.johnson_cook_parameters.items()
            ],
            hide_index=True,
        )
        st.json(result.calibration)
    with raw:
        selected = st.selectbox(
            "Steel curve",
            result.curves,
            format_func=lambda c: f"{c.group} — {c.label}",
            key="steel_raw_curve",
        )
        st.dataframe(
            {
                f"{selected.x_quantity} [{selected.x_unit}]": selected.x,
                f"{selected.y_quantity} [{selected.y_unit}]": selected.y,
            }
        )
        st.json(asdict(selected)["metadata"])
    with exports:
        st.download_button(
            "Download steel JSON", result.to_json(), "Steel-JC-Result.json", "application/json"
        )
        st.download_button(
            "Download steel Excel",
            steel_result_excel_bytes(result),
            "Steel-JC-Results.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.warning(ABAQUS_DISCLAIMER)
        st.download_button(
            "Download experimental ABAQUS",
            steel_result_abaqus_text(result),
            "steel_abaqus_material.inp",
            "text/plain",
        )
