"""Authority-aware Streamlit presentation; uses only application APIs."""

import json

import streamlit as st

from cdp_generator.application import (
    AuthorityConcreteRequest,
    AuthorityInputError,
    Cdpm2ConversionRequest,
    available_concrete_classes,
    available_physical_profiles,
    cdpm2_override_fields,
    cdpm2_parameter_units,
    parse_cdpm2_overrides,
    physical_profile_label,
    profile_parameter_specs,
    run_cdpm2_conversion,
)
from cdp_generator.application.authority_requests import JSONScalar
from cdp_generator.web.shared import add_to_compare


def _json_download(label: str, data: object, filename: str) -> None:
    st.download_button(
        label,
        json.dumps(data, sort_keys=True, indent=2, allow_nan=False),
        filename,
        "application/json",
    )


def render_authority() -> None:
    st.title("Authority-aware physical material / CDPM2")
    st.caption(
        "Physical authority → Grassl 2013 constitutive calibration → legacy24 compatibility payload. This workflow generates no EC2/fib stress-strain curves."
    )
    profile = st.sidebar.selectbox(
        "Physical authority/profile",
        available_physical_profiles(),
        format_func=physical_profile_label,
        key="authority_profile",
    )
    with st.sidebar.form("authority_assessment"):
        concrete_class = st.selectbox(
            "Concrete class", available_concrete_classes(profile), key=f"authority_class_{profile}"
        )
        profile_parameters: dict[str, JSONScalar] = {}
        for spec in profile_parameter_specs(profile):
            key = f"authority_{profile}_{spec.id}"
            if spec.kind == "choice":
                profile_parameters[spec.id] = st.selectbox(
                    spec.label, spec.options, index=spec.options.index(str(spec.default)), key=key
                )
            else:
                profile_parameters[spec.id] = st.number_input(
                    spec.label,
                    value=float(spec.default),
                    min_value=spec.minimum,
                    max_value=spec.maximum,
                    disabled=not spec.editable,
                    key=key,
                )
        st.caption(
            "Explicit constitutive overrides. Values apply only when their checkbox is enabled."
        )
        common: dict[str, float] = {}
        missing: list[str] = []
        units = cdpm2_parameter_units()
        for name in ("G_Ft", "E"):
            enabled = st.checkbox(f"Enable {name} override", key=f"override_enable_{name}")
            value = st.number_input(
                f"{name} [{units[name]}]", value=None, key=f"override_value_{name}"
            )
            if enabled:
                if value is None:
                    missing.append(name)
                else:
                    common[name] = float(value)
        with st.expander("Advanced CDPM2 overrides"):
            st.caption("Allowed fields: " + ", ".join(cdpm2_override_fields()))
            advanced = st.text_area("Override JSON object", value="{}", key="advanced_overrides")
        submitted = st.form_submit_button("Build material / Assess CDPM2")
    if submitted:
        try:
            if missing:
                raise AuthorityInputError(
                    "Enter a value for enabled override(s): " + ", ".join(missing)
                )
            request = Cdpm2ConversionRequest(
                AuthorityConcreteRequest(profile, concrete_class, profile_parameters),
                parse_cdpm2_overrides(advanced, common),
            )
            result = run_cdpm2_conversion(request)
        except AuthorityInputError as exc:
            st.error(str(exc))
        else:
            st.session_state["authority_request"] = request
            st.session_state["authority_result"] = result
    if "authority_result" not in st.session_state:
        st.info("Select a physical profile/class and build the material to assess CDPM2.")
        return
    result = st.session_state["authority_result"]
    material = result.material
    st.write(
        "Displayed material",
        {"profile": material.physical_profile, "class": material.concrete_class},
    )
    st.write("Requested profile parameters", material.requested_profile_parameters)
    st.write("Effective profile parameters", material.effective_profile_parameters)
    st.write("Displayed constitutive overrides", result.configuration["overrides"])
    physical, provenance, cdpm2, backend, exports = st.tabs(
        ["Physical properties", "Provenance", "CDPM2", "Backend", "Raw / Export"]
    )
    with physical:
        st.dataframe(
            [
                {
                    "Property": p.name,
                    "Value": "—" if p.value is None else str(p.value),
                    "Unit": p.unit,
                    "Resolution": p.resolution,
                }
                for p in material.physical_properties
            ],
            hide_index=True,
        )
    with provenance:
        st.dataframe(
            [
                {"Property": p.name, "Resolution": p.resolution, **(p.provenance or {})}
                for p in material.physical_properties
            ],
            hide_index=True,
        )
        selected = st.selectbox(
            "Property provenance",
            material.physical_properties,
            format_func=lambda p: p.name,
            key="authority_provenance",
        )
        st.json(
            selected.provenance
            if selected.provenance is not None
            else {"resolution": selected.resolution, "provenance": None}
        )
    with cdpm2:
        st.subheader("CDPM2 readiness: " + result.readiness["state"])
        for blocker in result.readiness["blockers"]:
            st.warning(blocker)
        if result.semantic_parameters is not None:
            semantic = result.semantic_parameters
            st.write("Constitutive model", semantic["model_id"])
            st.write("Calibration", semantic["calibration_id"])
            st.dataframe(
                [
                    {
                        "Parameter": name,
                        "Value": str(value),
                        "Unit": units[name],
                        **semantic["provenance"][name],
                    }
                    for name, value in semantic["values"].items()
                ],
                hide_index=True,
            )
    with backend:
        st.caption(
            "Backend compatibility representation. LCHAR [mm] is runtime/mesh context, separate from the 20 semantic parameters and cm(1:24)."
        )
        if result.semantic_parameters is None:
            st.info("Resolve CDPM2 before providing runtime context.")
        else:
            with st.form("backend_context"):
                lchar = st.number_input(
                    "Characteristic length LCHAR [mm]", value=None, key="backend_lchar"
                )
                build = st.form_submit_button("Build backend payload")
            if build:
                try:
                    if lchar is None:
                        raise AuthorityInputError("Enter LCHAR [mm].")
                    previous = st.session_state["authority_request"]
                    request = Cdpm2ConversionRequest(
                        previous.material, previous.overrides, float(lchar)
                    )
                    updated = run_cdpm2_conversion(request)
                except AuthorityInputError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["authority_request"] = request
                    st.session_state["authority_result"] = updated
                    st.rerun()
            if result.backend is not None:
                st.dataframe(
                    [
                        {"Slot": f"cm({i})", "Name": name, "Value": value}
                        for i, (name, value) in enumerate(
                            zip(result.backend_slot_names, result.backend["cm"], strict=True), 1
                        )
                    ],
                    hide_index=True,
                )
                st.write("Displayed LCHAR [mm]", result.backend["characteristic_length"])
    add_to_compare(
        result, "authority_concrete", f"{material.physical_profile} {material.concrete_class}"
    )
    with exports:
        st.json(result.to_dict())
        _json_download(
            "Download Material JSON", material.material_definition, "Concrete-Material.json"
        )
        st.download_button(
            "Download M2 result JSON", result.to_json(), "CDP-M2-Result.json", "application/json"
        )
        if result.semantic_parameters is not None:
            _json_download(
                "Download CDPM2 semantic JSON", result.semantic_parameters, "CDPM2-Semantic.json"
            )
        if result.backend is not None:
            _json_download("Download legacy24 backend JSON", result.backend, "CDPM2-legacy24.json")
