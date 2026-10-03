"""Descriptive session comparison UI; no ranking or scientific calculations."""

import streamlit as st

from ..application import (
    ComparisonInputError,
    authority_comparison_tables,
    compatible_curve_groups,
    rename_comparison_case,
    steel_comparison_table,
)
from ..visualization.comparison_plotly import comparison_figure


def render_comparison() -> None:
    st.title("Compare")
    st.caption(
        "Descriptive comparison within one workflow family. Cases live in this session; no rankings."
    )
    families = {
        "Legacy concrete curves": "legacy_concrete",
        "Authority-aware concrete / CDPM2": "authority_concrete",
        "Steel Johnson-Cook": "steel_johnson_cook",
    }
    selected = st.selectbox("Comparison family", tuple(families), key="compare_family")
    family = families[selected]
    all_cases = st.session_state.get("comparison_cases", [])
    cases = [c for c in all_cases if c.family == family]
    st.write(f"{len(cases)} cases in this family (maximum 4)")
    if cases:
        with st.expander("Manage cases"):
            current = st.selectbox(
                "Case", cases, format_func=lambda c: c.label, key="compare_manage_case"
            )
            label = st.text_input(
                "New label", current.label, key=f"compare_rename_{current.case_id}"
            )
            if st.button("Rename case"):
                try:
                    st.session_state["comparison_cases"] = rename_comparison_case(
                        all_cases, current.case_id, label
                    )
                except ComparisonInputError as exc:
                    st.error(str(exc))
                else:
                    st.rerun()
            if st.button("Remove case"):
                st.session_state["comparison_cases"] = [
                    c for c in all_cases if c.case_id != current.case_id
                ]
                st.rerun()
            if st.button("Clear family"):
                st.session_state["comparison_cases"] = [c for c in all_cases if c.family != family]
                st.rerun()
    if len(cases) < 2:
        st.info("Add at least two cases from the corresponding workflow.")
        return
    chosen = st.multiselect(
        "Cases to compare",
        cases,
        default=cases,
        format_func=lambda c: c.label,
        key=f"compare_selection_{family}",
    )
    if len(chosen) < 2:
        st.info("Select at least two cases.")
        return
    if family == "authority_concrete":
        for title, rows in authority_comparison_tables(chosen).items():
            st.subheader(title)
            if rows:
                st.dataframe(
                    [
                        {key: "—" if value is None else value for key, value in row.items()}
                        for row in rows
                    ],
                    hide_index=True,
                )
            else:
                st.info("At least two READY cases are required for semantic CDPM2 comparison.")
    else:
        groups = compatible_curve_groups(chosen)
        if groups:
            group = st.selectbox(
                "Common compatible curve group", groups, key=f"compare_group_{family}"
            )
            st.plotly_chart(
                comparison_figure(chosen, group), width="stretch", key=f"compare_plot_{family}"
            )
        else:
            st.info(
                "No common compatible curve group. Select cases with matching quantity/unit semantics."
            )
        if family == "steel_johnson_cook":
            st.subheader("Material and Johnson-Cook parameters")
            st.dataframe(steel_comparison_table(chosen), hide_index=True)
