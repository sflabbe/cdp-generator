"""Plotly views of canonical CDPM2 softening curves; no constitutive formulas here."""

import plotly.graph_objects as go

from ..application.results import CurveSeries


def cdpm2_curve_figure(curve: CurveSeries) -> go.Figure:
    """Render exactly one canonical CDPM2 curve."""

    figure = go.Figure(
        [go.Scatter(x=curve.x, y=curve.y, name=curve.label, mode="lines+markers")]
    )
    figure.update_layout(
        title=curve.label,
        xaxis_title=f"{curve.x_quantity} [{curve.x_unit}]",
        yaxis_title=f"{curve.y_quantity} [{curve.y_unit}]",
        hovermode="closest",
    )
    return figure


def build_cdpm2_figures(curves: list[CurveSeries]) -> dict[str, go.Figure]:
    """Build one figure per canonical CDPM2 curve group."""

    return {curve.group: cdpm2_curve_figure(curve) for curve in curves}
