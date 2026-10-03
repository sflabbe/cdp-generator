"""Run with: streamlit run cdp_generator/web/app.py."""

from dataclasses import asdict

import streamlit as st

from cdp_generator.application import (
    LegacyConcreteAnalysisRequest,
    result_excel_bytes,
    run_legacy_concrete_analysis,
    temperature_cases,
)
from cdp_generator.application.requests import parse_strain_rates
from cdp_generator.visualization.plotly import build_figures


def render_legacy() -> None:
    st.title("CDP Generator — legacy concrete")
    st.caption(
        "Legacy CDP curves. Verified physical materials and CDPM2 conversion have a separate workflow."
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
    if "last_result" not in st.session_state:
        st.info("Enter inputs and press Calculate.")
        return
    result = st.session_state["last_result"]
    st.write("Displayed result inputs", result.inputs)
    for warning in result.warnings:
        st.caption(warning)
    curves, properties, raw, exports = st.tabs(["Curves", "Properties", "Raw data", "Export"])
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


def main() -> None:
    st.set_page_config(page_title="CDP Generator", layout="wide")
    workflow = st.sidebar.radio(
        "Workflow", ["Legacy curves", "Authority-aware material / CDPM2"], key="workflow"
    )
    if workflow == "Legacy curves":
        render_legacy()
    else:
        from cdp_generator.web.authority import render_authority

        render_authority()


if __name__ == "__main__":
    main()
