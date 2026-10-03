"""Pure Plotly presentation of stored Johnson-Cook curves."""

import plotly.graph_objects as go

from ..application.steel_results import SteelAnalysisResult


def build_steel_figures(result: SteelAnalysisResult) -> dict[str, go.Figure]:
    figures = {}
    for group in dict.fromkeys(c.group for c in result.curves):
        curves = [c for c in result.curves if c.group == group]
        first = curves[0]
        figure = go.Figure(
            [go.Scatter(x=c.x, y=c.y, name=c.label, meta=c.metadata, mode="lines") for c in curves]
        )
        figure.update_layout(
            title=group,
            xaxis_title=f"{first.x_quantity} [{first.x_unit}]",
            yaxis_title=f"{first.y_quantity} [{first.y_unit}]",
            hovermode="closest",
        )
        figures[group] = figure
    return figures
