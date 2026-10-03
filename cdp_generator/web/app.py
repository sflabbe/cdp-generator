"""Run with: streamlit run cdp_generator/web/app.py."""

import streamlit as st

from cdp_generator.web.authority import render_authority
from cdp_generator.web.comparison import render_comparison
from cdp_generator.web.legacy import render_legacy
from cdp_generator.web.steel import render_steel


def main() -> None:
    st.set_page_config(page_title="CDP Generator", layout="wide")
    workflows = {
        "Legacy curves": render_legacy,
        "Authority-aware material / CDPM2": render_authority,
        "Steel Johnson-Cook": render_steel,
        "Compare": render_comparison,
    }
    workflow = st.sidebar.radio("Workflow", tuple(workflows), key="workflow")
    workflows[workflow]()


if __name__ == "__main__":
    main()
