"""Shared session comparison action; snapshots remain independent of later analyses."""

import streamlit as st

from ..application import (
    Cdpm2ConversionResult,
    ComparisonInputError,
    ResultBundle,
    SteelAnalysisResult,
    add_comparison_case,
    make_comparison_case,
)


def add_to_compare(
    result: ResultBundle | Cdpm2ConversionResult | SteelAnalysisResult,
    family: str,
    default_label: str,
) -> None:
    with st.sidebar.expander("Add to Compare"):
        with st.form(f"compare_add_{family}"):
            label = st.text_input("Case label", value=default_label, key=f"compare_label_{family}")
            add = st.form_submit_button("Add current result to Compare")
        if add:
            try:
                cases = add_comparison_case(
                    st.session_state.get("comparison_cases", []),
                    make_comparison_case(result, label),
                )
            except ComparisonInputError as exc:
                st.error(str(exc))
            else:
                st.session_state["comparison_cases"] = cases
                st.success(f"Added {label} to Compare.")
