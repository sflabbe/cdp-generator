"""Plotly views of canonical Abaqus export curves."""

import plotly.graph_objects as go

from ..application.abaqus_dependent import AbaqusLegacyDependentResult
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


def build_abaqus_dependent_figures(result: AbaqusLegacyDependentResult) -> dict[str, go.Figure]:
    """One overlay per exported table: one trace per exported family, nothing synthesized."""

    figures: dict[str, go.Figure] = {}
    for group in dict.fromkeys(curve.group for curve in result.curves):
        curves = [curve for curve in result.curves if curve.group == group]
        figure = go.Figure(
            [go.Scatter(x=c.x, y=c.y, name=c.label, mode="lines+markers") for c in curves]
        )
        first = curves[0]
        figure.update_layout(
            title=group,
            xaxis_title=f"{first.x_quantity} [{first.x_unit}]",
            yaxis_title=f"{first.y_quantity} [{first.y_unit}]",
            hovermode="closest",
        )
        figures[group] = figure
    return figures
