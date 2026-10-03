"""Legacy concrete presentation."""

from dataclasses import asdict

import streamlit as st

from cdp_generator.application import (
    AbaqusLegacyInputError,
    AbaqusLegacyMaterialRequest,
    AbaqusLegacyValidationError,
    LegacyConcreteAnalysisRequest,
    abaqus_legacy_material_text,
    result_excel_bytes,
    run_abaqus_legacy_material,
    run_legacy_concrete_analysis,
    temperature_cases,
)
from cdp_generator.application.requests import parse_strain_rates
from cdp_generator.visualization.abaqus_plotly import build_abaqus_figures
from cdp_generator.visualization.plotly import build_figures
from cdp_generator.web.shared import add_to_compare


def render_legacy() -> None:
    st.title("CDP Generator — legacy concrete")
    st.caption(
        "Legacy CDP curves. Verified physical materials and CDPM2 conversion have a separate "
        "workflow."
    )
    with st.sidebar.form("analysis"):
        mode_label = st.selectbox("Analysis mode", ["Strain rate", "Temperature"])
        f_cm = st.number_input("f_cm [MPa]", value=28.0)
        e_c1 = st.number_input("e_c1 [-]", value=0.0022, format="%.6f", step=0.0001)
        e_clim = st.number_input("e_clim [-]", value=0.0035, format="%.6f", step=0.0001)
        l_ch = st.number_input("l_ch [mm]", value=1.0)
        rates = st.text_input("Strain rates [1/s] (used in strain-rate mode)", "0, 2, 30, 100")
        st.caption("Temperature cases [°C]: " + ", ".join(f"{v:g}" for v in temperature_cases()))
        submitted = st.form_submit_button("Calculate")
    if submitted:
        try:
            request = LegacyConcreteAnalysisRequest(
                mode="strain_rate" if mode_label == "Strain rate" else "temperature",
                f_cm=f_cm,
                e_c1=e_c1,
                e_clim=e_clim,
                l_ch=l_ch,
                strain_rates=parse_strain_rates(rates) if mode_label == "Strain rate" else (),
            )
            bundle = run_legacy_concrete_analysis(request)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state["last_result"] = bundle
            st.session_state.pop("abaqus_legacy_result", None)
    if "last_result" not in st.session_state:
        st.info("Enter inputs and press Calculate.")
        return
    result = st.session_state["last_result"]
    st.write("Displayed result inputs", result.inputs)
    add_to_compare(result, "legacy_concrete", f"Legacy f_cm={result.inputs['f_cm']:g}")
    for warning in result.warnings:
        st.caption(warning)
    curves, properties, raw, abaqus_tab, exports = st.tabs(
        ["Curves", "Properties", "Raw data", "Abaqus CDP", "Export"]
    )
    with curves:
        for name, figure in build_figures(result).items():
            st.plotly_chart(figure, width="stretch", key=name)
    with properties:
        st.dataframe([asdict(p) for p in result.properties], hide_index=True)
    with raw:
        selected = st.selectbox(
            "Curve", result.curves, format_func=lambda c: f"{c.group} — {c.label}"
        )
        st.dataframe(
            {
                f"{selected.x_quantity} [{selected.x_unit}]": selected.x,
                f"{selected.y_quantity} [{selected.y_unit}]": selected.y,
            }
        )
    with abaqus_tab:
        st.subheader("Abaqus CDP — full legacy static-reference material")
        st.warning(
            "Legacy compatibility calibration. Verify against project calibration, experiments "
            "and the Abaqus version used."
        )
        st.caption(
            "Abaqus legacy material export currently uses the static reference calibration "
            "(strain rate 0, room/reference state). Existing rate/temperature plots remain "
            "analysis visualizations and are not exported as dependent Abaqus damage tables in M4."
        )
        st.caption(
            "Abaqus REF LENGTH is a backend damage-conversion reference length. It is not the "
            "legacy crack-band l_ch and is not a physical material property."
        )
        with st.form("legacy_abaqus_material"):
            law_label = st.selectbox(
                "Tension law", ["Bilinear", "Power law"], key="abaqus_tension_law"
            )
            ref_length = st.number_input(
                "Abaqus damage conversion REF LENGTH [mm]",
                value=1.0,
                min_value=0.0,
                key="abaqus_ref_length",
            )
            override_eccentricity = st.checkbox(
                "Override Abaqus eccentricity", key="abaqus_override_eccentricity"
            )
            eccentricity = st.number_input(
                "Eccentricity",
                value=0.1,
                disabled=not override_eccentricity,
                key="abaqus_eccentricity",
            )
            override_viscosity = st.checkbox(
                "Override Abaqus viscosity", key="abaqus_override_viscosity"
            )
            viscosity = st.number_input(
                "Viscosity",
                value=0.0,
                disabled=not override_viscosity,
                key="abaqus_viscosity",
            )
            include_tension_damage = st.checkbox(
                "Include tension damage", value=True, key="abaqus_include_tension_damage"
            )
            include_compression_damage = st.checkbox(
                "Include compression damage", value=True, key="abaqus_include_compression_damage"
            )
            build_abaqus = st.form_submit_button("Build Abaqus material")
        if build_abaqus:
            try:
                abaqus_request = AbaqusLegacyMaterialRequest(
                    f_cm=float(result.inputs["f_cm"]),
                    e_c1=float(result.inputs["e_c1"]),
                    e_clim=float(result.inputs["e_clim"]),
                    l_ch=float(result.inputs["l_ch"]),
                    tension_law="bilinear" if law_label == "Bilinear" else "power_law",
                    damage_conversion_reference_length_mm=float(ref_length),
                    eccentricity=float(eccentricity) if override_eccentricity else None,
                    viscosity=float(viscosity) if override_viscosity else None,
                    include_tension_damage=include_tension_damage,
                    include_compression_damage=include_compression_damage,
                )
                abaqus_result = run_abaqus_legacy_material(abaqus_request)
            except (AbaqusLegacyInputError, AbaqusLegacyValidationError) as exc:
                st.error(str(exc))
            else:
                st.session_state["abaqus_legacy_result"] = abaqus_result
        if "abaqus_legacy_result" in st.session_state:
            abaqus_result = st.session_state["abaqus_legacy_result"]
            st.success("Abaqus backend validation: PASS")
            scalar_rows = []
            scalar_values = {
                "E": abaqus_result.elastic["E_mpa"],
                "nu": abaqus_result.elastic["nu"],
                "dilation_angle": abaqus_result.plasticity["dilation_angle_deg"],
                "eccentricity": abaqus_result.plasticity["eccentricity"],
                "fbfc": abaqus_result.plasticity["fbfc"],
                "Kc": abaqus_result.plasticity["Kc"],
                "viscosity": abaqus_result.plasticity["viscosity"],
            }
            for name, value in scalar_values.items():
                category = "elastic" if name in {"E", "nu"} else "plasticity"
                provenance = abaqus_result.provenance[category][name]
                scalar_rows.append(
                    {
                        "Parameter": name,
                        "Value": value,
                        "Source kind": provenance["source_kind"],
                        "Source": provenance["source_id"],
                    }
                )
            st.dataframe(scalar_rows, hide_index=True)
            st.json(abaqus_result.validation)
            figures = build_abaqus_figures(abaqus_result)
            for curve in abaqus_result.curves:
                st.plotly_chart(
                    figures[curve.group], width="stretch", key=f"legacy_{curve.group}"
                )
                st.dataframe(
                    {
                        f"{curve.x_quantity} [{curve.x_unit}]": curve.x,
                        f"{curve.y_quantity} [{curve.y_unit}]": curve.y,
                    },
                    hide_index=True,
                )
            st.download_button(
                "Download Abaqus provenance JSON",
                abaqus_result.to_json(),
                "Concrete-Legacy-CDP.json",
                "application/json",
            )
            st.download_button(
                "Download Abaqus .inp material card",
                abaqus_legacy_material_text(abaqus_result),
                "Concrete-Legacy-CDP.inp",
                "text/plain",
            )
    with exports:
        st.download_button(
            "Download JSON", result.to_json(), "CDP-Results.json", "application/json"
        )
        st.download_button(
            "Download Excel",
            result_excel_bytes(result),
            "CDP-Results.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
