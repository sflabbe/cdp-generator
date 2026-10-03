"""Plotly views of canonical Abaqus export curves."""

import plotly.graph_objects as go

from ..application.abaqus_legacy import AbaqusLegacyMaterialResult
from ..application.results import CurveSeries


def abaqus_curve_figure(curve: CurveSeries) -> go.Figure:
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


def build_abaqus_figures(result: AbaqusLegacyMaterialResult) -> dict[str, go.Figure]:
    return {curve.group: abaqus_curve_figure(curve) for curve in result.curves}
