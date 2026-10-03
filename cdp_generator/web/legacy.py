"""Legacy concrete presentation."""

from dataclasses import asdict

import streamlit as st

from cdp_generator.application import (
    AbaqusLegacyDependentRequest,
    AbaqusLegacyDependentValidationError,
    AbaqusLegacyInputError,
    AbaqusLegacyMaterialRequest,
    AbaqusLegacyValidationError,
    LegacyConcreteAnalysisRequest,
    ResultBundle,
    abaqus_legacy_dependent_material_text,
    abaqus_legacy_material_text,
    rate_mapping_audit_rows,
    result_excel_bytes,
    run_abaqus_legacy_dependent_material,
    run_abaqus_legacy_material,
    run_legacy_concrete_analysis,
    temperature_cases,
    temperature_elastic_audit_rows,
)
from cdp_generator.application.requests import parse_strain_rates
from cdp_generator.visualization.abaqus_plotly import (
    build_abaqus_dependent_figures,
    build_abaqus_figures,
)
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
            st.session_state.pop("abaqus_dependent_result", None)
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
        export_modes = ["Static reference"] + (
            ["Strain-rate dependent"] if result.mode == "strain_rate" else ["Temperature dependent"]
        )
        export_mode = st.selectbox("Abaqus export mode", export_modes, key="abaqus_export_mode")
        st.info(
            "Abaqus solver qualification is an external optional gate. "
            "Run: uv run python scripts/qualify_abaqus.py"
        )
        if export_mode == "Static reference":
            _render_static_abaqus(result)
        else:
            _render_dependent_abaqus(result)
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


def _render_static_abaqus(result: ResultBundle) -> None:
    st.subheader("Abaqus CDP — full legacy static-reference material")
    st.warning(
        "Legacy compatibility calibration. Verify against project calibration, experiments "
        "and the Abaqus version used."
    )
    st.caption(
        "Abaqus legacy material export currently uses the static reference calibration "
        "(strain rate 0, room/reference state). Existing rate/temperature plots remain "
        "analysis visualizations here; rate/temperature dependent hardening and stiffening "
        "tables are built through the Abaqus export mode selector above."
    )
    st.caption(
        "Abaqus REF LENGTH is a backend damage-conversion reference length. It is not the "
        "legacy crack-band l_ch and is not a physical material property."
    )
    with st.form("legacy_abaqus_material"):
        law_label = st.selectbox("Tension law", ["Bilinear", "Power law"], key="abaqus_tension_law")
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
            st.plotly_chart(figures[curve.group], width="stretch", key=f"legacy_{curve.group}")
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


DAMAGE_POLICY_LABELS = {
    "Omit damage": "omit",
    "Reuse reference damage if Abaqus-valid": "reference_damage",
}


def _render_dependent_abaqus(result: ResultBundle) -> None:
    is_rate = result.mode == "strain_rate"
    st.subheader(
        "Abaqus CDP — legacy "
        + ("strain-rate dependent" if is_rate else "temperature dependent")
        + " material"
    )
    st.warning(
        "Legacy compatibility calibration. Dependent tables adapt the existing legacy curve "
        "families exactly; legacy rate/temperature laws are not asserted as normative laws."
    )
    if is_rate:
        st.caption(
            "Compression: the legacy curve-family control rate is used as the Abaqus "
            "compression-hardening rate coordinate (legacy rate-axis mapping, not reconstructed "
            "from a loading history). Tension: crack-opening rate = strain rate * l_ch "
            "(existing legacy mapping). Abaqus damage has no rate column."
        )
    else:
        st.caption(
            "E(T) is exported with the legacy secant modulus used by the inelastic-strain "
            "construction; nu is held constant (legacy constant-nu assumption). Hardening and "
            "stiffening depend on temperature; damage is omitted or reference-only."
        )
    with st.form("legacy_abaqus_dependent"):
        law_label = st.selectbox(
            "Tension law", ["Bilinear", "Power law"], key="abaqus_dep_tension_law"
        )
        policy_label = st.selectbox(
            "Damage policy", list(DAMAGE_POLICY_LABELS), key="abaqus_dep_damage_policy"
        )
        ref_length = st.number_input(
            "Abaqus damage conversion REF LENGTH [mm]",
            value=1.0,
            min_value=0.0,
            key="abaqus_dep_ref_length",
        )
        override_eccentricity = st.checkbox(
            "Override Abaqus eccentricity", key="abaqus_dep_override_eccentricity"
        )
        eccentricity = st.number_input(
            "Eccentricity",
            value=0.1,
            disabled=not override_eccentricity,
            key="abaqus_dep_eccentricity",
        )
        override_viscosity = st.checkbox(
            "Override Abaqus viscosity", key="abaqus_dep_override_viscosity"
        )
        viscosity = st.number_input(
            "Viscosity",
            value=0.0,
            disabled=not override_viscosity,
            key="abaqus_dep_viscosity",
        )
        build = st.form_submit_button("Build dependent Abaqus material")
    if build:
        try:
            request = AbaqusLegacyDependentRequest(
                mode=result.mode,
                f_cm=float(result.inputs["f_cm"]),
                e_c1=float(result.inputs["e_c1"]),
                e_clim=float(result.inputs["e_clim"]),
                l_ch=float(result.inputs["l_ch"]),
                strain_rates=tuple(float(v) for v in result.inputs["strain_rates"])
                if is_rate
                else (),
                tension_law="bilinear" if law_label == "Bilinear" else "power_law",
                damage_policy=DAMAGE_POLICY_LABELS[policy_label],
                damage_conversion_reference_length_mm=float(ref_length),
                eccentricity=float(eccentricity) if override_eccentricity else None,
                viscosity=float(viscosity) if override_viscosity else None,
            )
            dependent = run_abaqus_legacy_dependent_material(request)
        except AbaqusLegacyDependentValidationError as exc:
            st.error(str(exc))
            st.dataframe(list(exc.failures), hide_index=True)
        except (AbaqusLegacyInputError, AbaqusLegacyValidationError) as exc:
            st.error(str(exc))
        else:
            st.session_state["abaqus_dependent_result"] = dependent
    stored = st.session_state.get("abaqus_dependent_result")
    if stored is None or stored.mode != result.mode:
        return
    st.success(
        f"Abaqus dependent backend validation: PASS — {stored.validation['families_checked']} "
        f"families, damage policy {stored.damage_policy['policy']}"
    )
    st.caption(stored.damage_policy["notes"])
    st.markdown("**Dependency audit**")
    if is_rate:
        st.dataframe(rate_mapping_audit_rows(stored), hide_index=True)
    else:
        st.dataframe(temperature_elastic_audit_rows(stored), hide_index=True)
        st.caption("nu [-] is constant: legacy constant-nu assumption (no nu(T) law exists).")
    for name, figure in build_abaqus_dependent_figures(stored).items():
        st.plotly_chart(figure, width="stretch", key=f"legacy_dep_{name}")
    with st.expander("Validation per family"):
        st.json(stored.validation)
    for warning in stored.warnings:
        st.caption(warning)
    suffix = "Rate" if is_rate else "Temperature"
    st.download_button(
        "Download dependent Abaqus JSON",
        stored.to_json(),
        f"Concrete-Legacy-CDP-{suffix}.json",
        "application/json",
    )
    st.download_button(
        "Download dependent Abaqus .inp material card",
        abaqus_legacy_dependent_material_text(stored),
        f"Concrete-Legacy-CDP-{suffix}.inp",
        "text/plain",
    )
